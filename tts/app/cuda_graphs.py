"""CUDA-graph decoding for qwen-tts's talker and code predictor.

Each audio frame takes one talker step plus 15 code-predictor steps, all tiny batch-1 forward passes, so eager HF
generate is bound by Python and kernel-launch overhead rather than the GPU. This module replays them as CUDA graphs:

- The code predictor's whole per-frame decode (2-token prefill, 14 steps, sampling) is one graph. It replaces
  `code_predictor.generate`, whose result the talker only reads `.sequences` from.
- The talker's decode step (the 28-layer `talker.model` forward for one token) is one graph over a fixed-size KV
  cache. HF generate still drives the talker loop, so logits processors (repetition penalty, the end-of-speech hook)
  and stopping criteria (client disconnect) run once per frame outside the graph. Prefill stays eager.

On CPU (tests) the same bodies run eagerly without capture. Imported only inside the replica.
"""

import time
from types import SimpleNamespace

import torch
from transformers import StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast


class GraphCache(StaticCache):
    """StaticCache that generate treats as non-compileable.

    A compileable cache makes generate auto-compile the forward and pass a 4D attention mask, but the talker derives
    its rotary positions from the 2D mask at prefill.
    """

    @property
    def is_compileable(self):
        return False


def sample(logits, do_sample, top_k, top_p, temperature):
    """Sample one token per row like HF's temperature → top-k → top-p warpers, with only graph-safe ops.

    Gumbel-max replaces torch.multinomial: argmax(logits + Gumbel noise) is an exact categorical draw.
    """
    logits = logits.float()
    if not do_sample:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[..., -1:]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if top_p < 1.0:
        ordered, order = torch.sort(logits, descending=True)
        probs = ordered.softmax(dim=-1)
        drop = probs.cumsum(dim=-1) - probs > top_p  # always keeps the most likely token
        logits = logits.scatter(-1, order, ordered.masked_fill(drop, -float("inf")))
    noise = -torch.log(-torch.log(torch.rand_like(logits).clamp_(1e-20, 1.0)))
    return (logits + noise).argmax(dim=-1)


class Graph:
    """Runs `body` eagerly on CPU, or captures it once on CUDA and replays it."""

    def __init__(self, body, device):
        self.body, self.device, self.graph = body, device, None

    def __call__(self):
        if self.device.type != "cuda":
            return self.body()
        if self.graph is None:
            stream = torch.cuda.Stream(self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):  # warm-up runs are required before capture; they write identical values
                for _ in range(2):
                    self.body()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
                self.body()
        self.graph.replay()


class CodePredictorGraph:
    """The code predictor's per-frame decode as one graph, standing in for `code_predictor.generate`."""

    def __init__(self, code_predictor, do_sample, top_k, top_p, temperature):
        cp = self.cp = code_predictor
        self.steps = cp.config.num_code_groups - 1  # tokens per frame: one from the prefill, the rest decoded
        self.params = (bool(do_sample), int(top_k), float(top_p), float(temperature))
        param = next(cp.parameters())
        device, length = param.device, self.steps + 1  # 2 prefill positions + steps - 1 decode positions
        self.cache = GraphCache(config=cp.config, max_cache_len=length)
        self.inputs = torch.zeros((1, 2, cp.model.codec_embedding[0].embedding_dim), dtype=param.dtype, device=device)
        self.tokens = torch.zeros((1, self.steps), dtype=torch.long, device=device)
        positions = torch.arange(length, device=device)
        # Masks and positions for every step are fixed, so they are built once: query i attends keys 0..i.
        self.positions = [positions[:2]] + [positions[t + 1 : t + 2] for t in range(1, self.steps)]
        self.masks = [{"full_attention": (positions[None, :] <= p[:, None])[None, None]} for p in self.positions]
        self.graph = Graph(self._decode, device)
        self.original = cp.generate

    def _decode(self):
        cp = self.cp
        token = None
        for t in range(self.steps):
            embeds = self.inputs if t == 0 else cp.model.codec_embedding[t - 1](token[:, None])
            hidden = cp.model(
                inputs_embeds=cp.small_to_mtp_projection(embeds), attention_mask=self.masks[t], past_key_values=self.cache,
                cache_position=self.positions[t], use_cache=True,
            ).last_hidden_state
            token = sample(cp.lm_head[t](hidden[:, -1]), *self.params)
            self.tokens[:, t].copy_(token)

    def generate(self, inputs_embeds=None, max_new_tokens=None, do_sample=None, top_k=None, top_p=None,
                 temperature=None, **kwargs):
        try:
            params = (bool(do_sample), int(top_k), float(top_p), float(temperature))
        except (TypeError, ValueError):
            params = None
        if inputs_embeds is None or inputs_embeds.shape != self.inputs.shape or max_new_tokens != self.steps or params != self.params:
            return self.original(inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                 top_k=top_k, top_p=top_p, temperature=temperature, **kwargs)
        self.inputs.copy_(inputs_embeds)
        self.graph()
        return SimpleNamespace(sequences=self.tokens.clone())


class TalkerGraph:
    """The talker's one-token decode step as a graph over a fixed-size KV cache, standing in for `talker.model`."""

    def __init__(self, talker, capacity):
        self.model, self.capacity = talker.model, capacity
        param = next(talker.parameters())
        device = param.device
        self.cache = GraphCache(config=talker.config, max_cache_len=capacity)
        self.inputs = torch.zeros((1, 1, talker.config.hidden_size), dtype=param.dtype, device=device)
        self.position_ids = torch.zeros((3, 1, 1), dtype=torch.long, device=device)
        self.cache_position = torch.zeros(1, dtype=torch.long, device=device)
        self.slots = torch.arange(capacity, device=device)
        self.mask = torch.zeros((1, 1, 1, capacity), dtype=torch.bool, device=device)
        self.hidden = None
        self.graph = Graph(self._decode, device)
        self.original = self.model.forward

    def _decode(self):
        self.hidden = self.original(
            inputs_embeds=self.inputs, attention_mask=self.mask, position_ids=self.position_ids,
            past_key_values=self.cache, cache_position=self.cache_position, use_cache=True,
            output_hidden_states=False,
        ).last_hidden_state

    def forward(self, inputs_embeds=None, past_key_values=None, position_ids=None, cache_position=None,
                output_hidden_states=None, **kwargs):
        if past_key_values is not self.cache or inputs_embeds is None or inputs_embeds.shape[1] != 1:
            return self.original(inputs_embeds=inputs_embeds, past_key_values=past_key_values, position_ids=position_ids,
                                 cache_position=cache_position, output_hidden_states=output_hidden_states, **kwargs)
        self.inputs.copy_(inputs_embeds)
        self.position_ids.copy_(position_ids)
        self.cache_position.copy_(cache_position)
        torch.le(self.slots, self.cache_position, out=self.mask.view(-1))  # attend to filled slots only
        self.graph()
        hidden = self.hidden.clone()  # the graph overwrites its output on the next replay
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=self.cache,
                                       hidden_states=(hidden,) if output_hidden_states else None)


def install(model, capacity, defaults):
    """Route qwen-tts generation through the graphs; returns a hook for the talker's generate kwargs.

    `defaults` is the model's generate_config: the code-predictor graph is built for its subtalker sampling options,
    the ones generate_voice_clone passes when the request does not override them.
    """
    talker = model.talker
    predictor = CodePredictorGraph(
        talker.code_predictor, defaults.get("subtalker_dosample", True), defaults.get("subtalker_top_k", 50),
        defaults.get("subtalker_top_p", 1.0), defaults.get("subtalker_temperature", 0.9),
    )
    decoder = TalkerGraph(talker, capacity)
    talker.code_predictor.generate = predictor.generate
    talker.model.forward = decoder.forward

    def prepare(kwargs):
        """Point one talker generate call at the static cache, capping max_new_tokens to fit it."""
        prompt = kwargs["inputs_embeds"].shape[1]
        if kwargs["inputs_embeds"].shape[0] != 1 or prompt >= capacity:
            return kwargs  # batch > 1 or a prompt that cannot fit: eager DynamicCache path
        decoder.cache.reset()
        return {**kwargs, "past_key_values": decoder.cache, "disable_compile": True,
                "max_new_tokens": min(kwargs["max_new_tokens"], capacity - prompt)}

    return prepare


def warm_up(tts):
    """Capture both graphs with a throwaway synthesis from a synthetic reference; returns the seconds it took."""
    import numpy as np

    started = time.perf_counter()
    sample_rate = 24000
    tone = (0.1 * np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate)).astype(np.float32)
    tts.generate_voice_clone(text="Warm-up.", language="English", ref_audio=(tone, sample_rate), ref_text="Hello.",
                             max_new_tokens=8)
    torch.cuda.synchronize()
    return time.perf_counter() - started
