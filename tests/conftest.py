"""Shared fixtures: tiny random qwen-tts models on CPU (the real architecture, few hidden units), CUDA stand-ins
and fake replicas."""

import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

# Talker vocabulary as released: codes, then 1024 suppressed specials but end-of-speech. Codes < CODEBOOK serve as
# code-predictor and decoder tokens too.
CODEBOOK = 32
VOCAB = CODEBOOK + 1024
EOS = CODEBOOK + 10
HIDDEN = 32
CODE_GROUPS = 4


def talker_kwargs(max_new_tokens=12, do_sample=False, **overrides):
    """The kwargs Qwen3TTSForConditionalGeneration.generate passes to the talker's generate."""
    return {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": 2,
        "do_sample": do_sample,
        "top_k": 50,
        "top_p": 1.0,
        "temperature": 0.9,
        "subtalker_dosample": False,
        "subtalker_top_k": 50,
        "subtalker_top_p": 1.0,
        "subtalker_temperature": 0.9,
        "eos_token_id": EOS,
        "repetition_penalty": 1.05,
        "suppress_tokens": [i for i in range(VOCAB - 1024, VOCAB) if i != EOS],
        "output_hidden_states": True,
        "return_dict_in_generate": True,
        **overrides,
    }


def talker_inputs(prompt=6, text=3, seed=1):
    """A talker prompt: input embeddings, mask, the trailing text hiddens, and the pad embedding."""
    generator = torch.Generator().manual_seed(seed)
    return {
        "inputs_embeds": torch.randn(1, prompt, HIDDEN, generator=generator),
        "attention_mask": torch.ones(1, prompt, dtype=torch.long),
        "trailing_text_hidden": torch.randn(1, text, HIDDEN, generator=generator),
        "tts_pad_embed": torch.randn(1, 1, HIDDEN, generator=generator),
    }


def frames(result):
    """Codec frames (first code + the code predictor's) from a talker generate result, as generate_voice_clone reads."""
    return torch.stack([h[-1][0] for h in result.hidden_states if h[-1] is not None])


def hiddens(result):
    """The talker's last hidden state per step, which generate_voice_clone also returns."""
    return torch.cat([h[0][-1][:, -1:] for h in result.hidden_states], dim=1)


@pytest.fixture
def tiny_tts_model():
    """A Qwen3TTSForConditionalGeneration stand-in: what hook_talker and cuda_graphs.install read from the model."""
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration

    torch.manual_seed(0)
    config = Qwen3TTSTalkerConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_code_groups=CODE_GROUPS,
        text_hidden_size=HIDDEN,
        text_vocab_size=64,
        codec_eos_token_id=EOS,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "interleaved": True},
        code_predictor_config={
            "vocab_size": CODEBOOK,
            "hidden_size": HIDDEN,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "num_code_groups": CODE_GROUPS,
        },
    )
    talker = Qwen3TTSTalkerForConditionalGeneration(config).eval()
    with torch.no_grad():
        for param in talker.parameters():
            param.normal_(0, 0.1)  # the default init attends almost uniformly, which hides attention-mask mistakes
    return SimpleNamespace(talker=talker, config=SimpleNamespace(talker_config=config))


@pytest.fixture
def tiny_decoder():
    """A speech-tokenizer decoder with qwen-tts's upsampling rates and random weights large enough to make sound."""
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2DecoderConfig
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Decoder

    torch.manual_seed(0)
    config = Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=CODEBOOK,
        hidden_size=16,
        latent_dim=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        num_hidden_layers=2,
        num_quantizers=CODE_GROUPS,
        decoder_dim=32,
        codebook_dim=16,
    )
    decoder = Qwen3TTSTokenizerV2Decoder(config).eval()
    with torch.no_grad():
        for param in decoder.parameters():
            param.normal_(0, 0.1)  # the default init decodes silence, which would make every comparison pass
    return decoder


@pytest.fixture
def no_cuda(monkeypatch):
    """Let replica code that synchronizes and reads CUDA memory stats run on CPU."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 1.5e9)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a, **k: None)


def tone(seconds=1.0, rate=24000, freq=220.0, amplitude=0.1):
    t = np.arange(round(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def wav_bytes(samples, rate, subtype="PCM_16"):
    out = io.BytesIO()
    sf.write(out, samples, rate, format="WAV", subtype=subtype)
    return out.getvalue()


def serve_client(module, instance):
    """A TestClient for a Serve app's FastAPI ingress, with `instance` as the replica its routes are bound to."""
    from fastapi.testclient import TestClient

    servable = next(
        r.dependant.dependencies[0].call
        for r in module.api.routes
        if getattr(r, "dependant", None) and r.dependant.dependencies
    )
    module.api.dependency_overrides[servable] = lambda: instance
    return TestClient(module.api)
