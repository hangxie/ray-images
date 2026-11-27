"""serve_tts's hooks into qwen-tts internals, against tiny random models of the installed qwen-tts on CPU.

These are the checks AGENTS.md asks to rerun when upgrading qwen-tts or transformers: the talker hooks, and
decode_tail / Streamer matching the full-clip decode.
"""

import numpy as np
import pytest
import torch
from conftest import CODE_GROUPS, CODEBOOK, EOS, frames, talker_inputs, talker_kwargs

import serve_tts


def generate(model, **kwargs):
    with torch.no_grad():  # as generate_voice_clone runs it
        return model.talker.generate(**talker_inputs(), **talker_kwargs(**kwargs))


class Recorder:
    """A Streamer stand-in that records the frames the talker hands over."""

    def __init__(self):
        self.frames, self.flushes = [], 0

    def frame(self, codes):
        self.frames.append(codes.clone())

    def flush(self):
        self.flushes += 1


@pytest.fixture
def hooked(tiny_tts_model):
    """The tiny model with serve_tts's hooks installed, and the slot holding the running job."""
    slot = {"job": serve_tts.Job()}
    serve_tts.hook_talker(tiny_tts_model, lambda: slot["job"])
    return tiny_tts_model, slot


def test_hooks_leave_greedy_codes_unchanged(tiny_tts_model):
    reference = generate(tiny_tts_model)
    serve_tts.hook_talker(tiny_tts_model, serve_tts.Job)
    result = generate(tiny_tts_model)
    assert torch.equal(result.sequences, reference.sequences)
    assert torch.equal(frames(result), frames(reference))


def test_token_limited_without_end_of_speech(hooked):
    model, slot = hooked
    result = generate(model, max_new_tokens=5)
    assert not (result.sequences == EOS).any()
    assert slot["job"].token_limited


def test_client_gone_stops_generation(hooked):
    model, slot = hooked
    slot["job"].stop.set()
    with pytest.raises(serve_tts.ClientGone):
        generate(model)


def test_stream_receives_every_frame(tiny_tts_model):
    reference = frames(generate(tiny_tts_model))
    job = serve_tts.Job()
    job.stream = Recorder()
    serve_tts.hook_talker(tiny_tts_model, lambda: job)
    with pytest.raises(serve_tts.StreamFinished):
        generate(tiny_tts_model)
    assert job.stream.flushes == 1
    # The last sampled first code has no frame yet when generation stops at max_new_tokens
    assert torch.equal(torch.stack(job.stream.frames), reference)
    assert all(f.shape == (CODE_GROUPS,) for f in job.stream.frames)


def eos_favoured(model, bias):
    """Make end-of-speech the most likely first code at every step, by `bias` logits."""
    head = model.talker.codec_head
    biased = torch.nn.Linear(head.in_features, head.out_features, bias=True)
    with torch.no_grad():
        biased.weight.copy_(head.weight)
        biased.bias.zero_()
        biased.bias[EOS] = bias
    model.talker.codec_head = biased


def stop_step(result):
    hits = (result.sequences[0] == EOS).nonzero()
    return int(hits[0]) if hits.numel() else None


def test_likely_end_of_speech_is_taken(tiny_tts_model):
    # Sampled at temperature 1.5 with end-of-speech at about a third of the mass, a plain draw often misses it
    eos_favoured(tiny_tts_model, 3.0)
    misses = 0
    for seed in range(8):
        torch.manual_seed(seed)
        misses += stop_step(generate(tiny_tts_model, do_sample=True, temperature=1.5, top_k=0)) != 2
    assert misses > 0, "end-of-speech is too likely for this test to show anything"

    serve_tts.hook_talker(tiny_tts_model, serve_tts.Job)
    for seed in range(8):
        torch.manual_seed(seed)
        # min_new_tokens=2 masks end-of-speech for two steps; the hook takes it on the first step it can
        assert stop_step(generate(tiny_tts_model, do_sample=True, temperature=1.5, top_k=0)) == 2


def test_prepare_hook_sees_talker_kwargs(tiny_tts_model):
    seen = []

    def prepare(kwargs):
        seen.append(kwargs)
        return {**kwargs, "max_new_tokens": 3}

    serve_tts.hook_talker(tiny_tts_model, serve_tts.Job, prepare)
    result = generate(tiny_tts_model, max_new_tokens=10)
    assert seen[0]["max_new_tokens"] == 10
    assert "trailing_text_hidden" in seen[0]
    assert result.sequences.shape[1] == 3


@pytest.mark.parametrize("new", [1, 12, 30])
def test_decode_tail_matches_full_decode(tiny_decoder, new):
    codes = torch.randint(0, CODEBOOK, (40, CODE_GROUPS), generator=torch.Generator().manual_seed(new))
    with torch.inference_mode():
        full = tiny_decoder(codes.T[None])[0, 0]
        tail = serve_tts.decode_tail(tiny_decoder, codes, new, serve_tts.STREAM_VOCODER_MARGIN)
        cut = serve_tts.decode_tail(tiny_decoder, codes, new, 0)
    samples = new * int(tiny_decoder.total_upsample)
    assert tail.shape == (samples,)
    assert full.abs().max() > 0.01  # the decoder makes sound, so the comparison means something
    torch.testing.assert_close(tail, full[-samples:], atol=1e-6, rtol=0)
    assert (cut - full[-samples:]).abs().max() > 1e-3  # without the margin the vocoder lacks context


def test_decode_tail_clamps_negative_codes(tiny_decoder):
    codes = torch.randint(0, CODEBOOK, (10, CODE_GROUPS), generator=torch.Generator().manual_seed(0))
    padded = codes.clone()
    padded[0] = -1
    clamped = codes.clone()
    clamped[0] = 0
    with torch.inference_mode():
        torch.testing.assert_close(
            serve_tts.decode_tail(tiny_decoder, padded, 4, 4), serve_tts.decode_tail(tiny_decoder, clamped, 4, 4)
        )


def streamed(tiny_decoder, ref, generated, context, gain_db):
    events: list[dict] = []
    streamer = serve_tts.Streamer(tiny_decoder, ref, 24000, context, gain_db, events.append)
    for frame in generated:
        streamer.frame(frame)
    streamer.flush()
    pcm = np.frombuffer(b"".join(e["pcm"] for e in events), dtype="<i2")
    return streamer, events, pcm


def test_stream_matches_full_clip_decode(tiny_decoder):
    """Streamed pieces, concatenated, are the full-clip decode of reference + speech, minus the reference."""
    generator = torch.Generator().manual_seed(0)
    ref = torch.randint(0, CODEBOOK, (20, CODE_GROUPS), generator=generator)
    generated = torch.randint(0, CODEBOOK, (40, CODE_GROUPS), generator=generator)
    gain_db = 6.0
    streamer, events, pcm = streamed(tiny_decoder, ref, generated, 300, gain_db)

    assert [len(e["pcm"]) // 2 for e in events] == [n * int(tiny_decoder.total_upsample) for n in (12, 12, 12, 4)]
    with torch.inference_mode():
        full = tiny_decoder(torch.cat([ref, generated]).T[None])[0, 0].numpy()
    expected = full[ref.shape[0] * int(tiny_decoder.total_upsample) :] * 10 ** (gain_db / 20)
    expected_pcm = np.round(np.clip(expected, -serve_tts.PEAK_CEILING, serve_tts.PEAK_CEILING) * 32767)
    assert pcm.size == expected_pcm.size
    assert np.abs(pcm - expected_pcm).max() <= 1
    assert streamer.samples == pcm.size
    assert streamer.clipped == int(np.count_nonzero(np.abs(expected) > serve_tts.PEAK_CEILING))
    assert streamer.first is not None


def test_stream_flush_without_new_frames_sends_nothing(tiny_decoder):
    ref = torch.zeros((3, CODE_GROUPS), dtype=torch.long)
    events: list[dict] = []
    streamer = serve_tts.Streamer(tiny_decoder, ref, 24000, 300, 0.0, events.append)
    streamer.flush()
    assert events == []
    for frame in torch.zeros((serve_tts.STREAM_FRAMES, CODE_GROUPS), dtype=torch.long):
        streamer.frame(frame)
    assert len(events) == 1  # a full piece is sent without waiting for flush
    streamer.flush()
    assert len(events) == 1


def test_stream_rejects_non_finite_audio(tiny_decoder):
    with torch.no_grad():
        next(tiny_decoder.decoder[-1].parameters()).fill_(float("nan"))
    streamer = serve_tts.Streamer(tiny_decoder, torch.zeros((3, CODE_GROUPS), dtype=torch.long), 24000, 300, 0.0, print)
    streamer.frame(torch.zeros(CODE_GROUPS, dtype=torch.long))
    with pytest.raises(ValueError, match="non-finite"):
        streamer.flush()
