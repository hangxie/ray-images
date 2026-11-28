"""cuda_graphs matches qwen-tts's eager path: on CPU the graph bodies run eagerly, so greedy codes must match."""

import pytest
import torch
from conftest import EOS, frames, hiddens, talker_inputs, talker_kwargs
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

import cuda_graphs
import serve_tts

GREEDY_SUBTALKER = {
    "subtalker_dosample": False,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
    "subtalker_temperature": 0.9,
}


def generate(model, prepare=None, inputs=None, **kwargs):
    kwargs = {**(inputs or talker_inputs()), **talker_kwargs(**kwargs)}
    if prepare is not None:
        kwargs = prepare(kwargs)
    with torch.no_grad():  # as generate_voice_clone runs it
        return model.talker.generate(**kwargs)


class Counting:
    """Wraps a Graph to count how often generation goes through it."""

    def __init__(self, graph):
        self.graph, self.calls = graph, 0

    def __call__(self):
        self.calls += 1
        return self.graph()


def install(model, capacity=64, defaults=None):
    prepare = cuda_graphs.install(model, capacity, defaults or GREEDY_SUBTALKER)
    predictor = model.talker.code_predictor.generate.__self__
    decoder = model.talker.model.forward.__self__
    predictor.graph, decoder.graph = Counting(predictor.graph), Counting(decoder.graph)
    return prepare, predictor.graph, decoder.graph


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_greedy_codes_match_eager(tiny_tts_model, seed):
    inputs = talker_inputs(prompt=4 + seed, seed=seed)
    reference = generate(tiny_tts_model, inputs=inputs, max_new_tokens=16)
    prepare, predictor, decoder = install(tiny_tts_model)
    result = generate(tiny_tts_model, prepare, inputs=inputs, max_new_tokens=16)
    assert torch.equal(result.sequences, reference.sequences)
    assert torch.equal(frames(result), frames(reference))
    torch.testing.assert_close(hiddens(result), hiddens(reference), atol=1e-5, rtol=1e-4)
    steps = frames(result).shape[0]
    assert predictor.calls == steps  # one code-predictor graph replay per frame
    assert decoder.calls == steps  # one talker graph replay per decode step; prefill stays eager


def test_graph_codes_match_through_serve_hooks(tiny_tts_model):
    """The production wiring: hook_talker with cuda_graphs' prepare, then the stream hook on the graph path."""
    reference = frames(generate(tiny_tts_model, max_new_tokens=16))
    prepare = cuda_graphs.install(tiny_tts_model, 64, GREEDY_SUBTALKER)
    job = serve_tts.Job()
    received = []
    job.stream = type("S", (), {"frame": lambda self, c: received.append(c.clone()), "flush": lambda self: None})()
    serve_tts.hook_talker(tiny_tts_model, lambda: job, prepare)
    with pytest.raises(serve_tts.StreamFinished):
        generate(tiny_tts_model, max_new_tokens=16)
    assert torch.equal(torch.stack(received), reference)


def test_graph_is_reused_across_requests(tiny_tts_model):
    first, second = talker_inputs(seed=1), talker_inputs(prompt=9, seed=2)
    references = [generate(tiny_tts_model, inputs=i) for i in (first, second, first)]
    prepare, _, _ = install(tiny_tts_model)
    results = [generate(tiny_tts_model, prepare, inputs=i) for i in (first, second, first)]
    for result, reference in zip(results, references, strict=True):
        assert torch.equal(result.sequences, reference.sequences)  # the static cache is reset between requests
        torch.testing.assert_close(hiddens(result), hiddens(reference), atol=1e-5, rtol=1e-4)


def test_prepare_caps_tokens_to_cache(tiny_tts_model):
    prepare, _, _ = install(tiny_tts_model, capacity=10)
    kwargs = prepare({**talker_inputs(prompt=6), **talker_kwargs(max_new_tokens=100)})
    assert kwargs["max_new_tokens"] == 4
    assert isinstance(kwargs["past_key_values"], cuda_graphs.GraphCache)
    assert kwargs["disable_compile"] is True
    result = generate(tiny_tts_model, prepare, inputs=talker_inputs(prompt=6), max_new_tokens=100)
    assert result.sequences.shape[1] == 4


def test_long_prompt_falls_back_to_eager(tiny_tts_model):
    inputs = talker_inputs(prompt=12)
    reference = generate(tiny_tts_model, inputs=inputs)
    prepare, predictor, decoder = install(tiny_tts_model, capacity=12)
    kwargs = {**inputs, **talker_kwargs()}
    assert prepare(kwargs) is kwargs
    result = generate(tiny_tts_model, prepare, inputs=inputs)
    assert torch.equal(result.sequences, reference.sequences)
    assert decoder.calls == 0
    assert predictor.calls == frames(result).shape[0]  # the code predictor graph does not depend on the cache


def test_other_subtalker_options_fall_back_to_eager(tiny_tts_model):
    reference = generate(tiny_tts_model, subtalker_top_k=10)
    prepare, predictor, decoder = install(tiny_tts_model)
    result = generate(tiny_tts_model, prepare, subtalker_top_k=10)
    assert torch.equal(result.sequences, reference.sequences)
    assert predictor.calls == 0
    assert decoder.calls == frames(result).shape[0]


def test_graph_cache_is_not_compileable(tiny_tts_model):
    cache = cuda_graphs.GraphCache(config=tiny_tts_model.talker.config, max_cache_len=8)
    assert not cache.is_compileable


def test_graph_runs_body_eagerly_on_cpu():
    calls = []

    def body():
        calls.append(1)
        return "out"

    graph = cuda_graphs.Graph(body, torch.device("cpu"))
    assert graph() == "out"
    assert graph() == "out"
    assert len(calls) == 2
    assert graph.graph is None


def hf_filtered(logits, top_k, top_p, temperature):
    """HF's warpers in generate's order, as the eager code predictor applies them."""
    scores = TemperatureLogitsWarper(temperature)(None, logits.clone())
    if top_k > 0:
        scores = TopKLogitsWarper(top_k)(None, scores)
    if top_p < 1.0:
        scores = TopPLogitsWarper(top_p)(None, scores)
    return scores


def test_sample_greedy_is_argmax():
    logits = torch.randn(16, 32, generator=torch.Generator().manual_seed(0))
    assert torch.equal(cuda_graphs.sample(logits, False, 50, 1.0, 0.9), logits.argmax(-1))
    assert torch.equal(cuda_graphs.sample(logits, True, 1, 1.0, 0.9), logits.argmax(-1))


@pytest.mark.parametrize(
    ("top_k", "top_p", "temperature"), [(0, 1.0, 1.0), (5, 1.0, 0.7), (0, 0.6, 1.3), (8, 0.8, 0.9)]
)
def test_sample_matches_hf_warpers(top_k, top_p, temperature):
    """Same support as HF's warpers, and draws follow their distribution (Gumbel-max is exact)."""
    torch.manual_seed(0)
    row = torch.randn(32) * 2
    expected = hf_filtered(row[None], top_k, top_p, temperature)[0].softmax(-1)
    draws = 40_000
    tokens = cuda_graphs.sample(row.expand(draws, -1), True, top_k, top_p, temperature)
    observed = torch.bincount(tokens, minlength=32).float() / draws
    assert set(tokens.unique().tolist()) <= set(expected.nonzero().flatten().tolist())
    assert (observed - expected).abs().max() < 0.015


def test_sample_handles_bfloat16_logits():
    logits = torch.randn(4, 32).to(torch.bfloat16)
    tokens = cuda_graphs.sample(logits, True, 5, 0.9, 0.8)
    assert tokens.dtype == torch.long and tokens.shape == (4,)


def test_talker_eos_id_is_in_suppressed_range():
    """The fixture's vocabulary mirrors the released one: end-of-speech sits among the suppressed specials."""
    suppressed = talker_kwargs()["suppress_tokens"]
    assert EOS not in suppressed
    assert min(suppressed) < EOS < max(suppressed)
