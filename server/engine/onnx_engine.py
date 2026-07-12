"""ONNX Runtime engine: exported IndicXlit encoder/decoder + external beam search.

CTranslate2 has a native fairseq converter and a C++ beam search; ONNX has
neither, so this engine carries the cost CT2 avoided: a Python beam search over
two ONNX graphs. It exists to benchmark ONNX Runtime against CTranslate2 on CPU.
Load `encoder.onnx`/`decoder.onnx` for fp32 or `*.int8.onnx` for INT8.

Note: the decoder has no KV cache (fairseq incremental decoding does not export
cleanly), so it re-runs the full prefix each step. Latency here is therefore an
upper bound; a production ONNX path would add caching or use ORT's BeamSearch op.
"""

import json
import os
from typing import List

import numpy as np
import onnxruntime as ort

from server.engine.base import TransliterationEngine, validate_beam


class ONNXEngine(TransliterationEngine):
    """IndicXlit via ONNX Runtime with an external batched beam search."""

    name = "onnx"

    def __init__(self, model_dir: str = "models/indicxlit/onnx",
                 precision: str = "int8", lang: str = "hi", beam_width: int = 5,
                 topk: int = 5, intra_threads: int = 1, max_len: int = 30) -> None:
        validate_beam(beam_width, topk)
        self.beam_width = beam_width
        self.max_len = max_len

        with open(f"{model_dir}/vocab.json", encoding="utf-8") as f:
            v = json.load(f)
        self.src2id = v["src_token2id"]
        self.id2tgt = v["tgt_id2token"]
        self.eos, self.pad, self.unk = v["eos"], v["pad"], v["unk"]
        self.src_eos, self.src_unk = v["src_eos"], v["src_unk"]
        self.lang_tag = v["lang_tag"]
        # Decoder is a fixed-length graph; prefixes are right-padded to this.
        self.dec_len = v.get("max_len", 32)
        self.max_len = min(max_len, self.dec_len - 1)

        suffix = ".int8" if precision == "int8" else ""
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        opts.inter_op_num_threads = 1
        self.enc = ort.InferenceSession(f"{model_dir}/encoder{suffix}.onnx", opts,
                                        providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(f"{model_dir}/decoder{suffix}.onnx", opts,
                                        providers=["CPUExecutionProvider"])

    def _encode(self, word: str) -> np.ndarray:
        ids = [self.src2id.get(self.lang_tag, self.src_unk)]
        ids += [self.src2id.get(c, self.src_unk) for c in word.lower()]
        ids.append(self.src_eos)
        return np.array([ids], dtype=np.int64)

    def transliterate(self, word: str, topk: int = 5) -> List[str]:
        src = self._encode(word)
        enc_out = self.enc.run(None, {"src_tokens": src})[0]  # [S, 1, C]

        beams = [([self.eos], 0.0)]
        finished: List = []
        for _ in range(self.max_len):
            active = [(toks, sc) for toks, sc in beams
                      if not (toks[-1] == self.eos and len(toks) > 1)]
            finished += [(toks, sc) for toks, sc in beams
                         if toks[-1] == self.eos and len(toks) > 1]
            if not active:
                break
            cur_len = len(active[0][0])
            prev = np.full((len(active), self.dec_len), self.pad, dtype=np.int64)
            for b, (toks, _) in enumerate(active):
                prev[b, :len(toks)] = toks
            enc_tiled = np.repeat(enc_out, len(active), axis=1)  # [S, n, C]
            logits = self.dec.run(None, {"prev_output_tokens": prev,
                                         "encoder_out": enc_tiled})[0]
            last = logits[:, cur_len - 1, :]  # logit at the true last position
            logp = last - _logsumexp(last, axis=-1, keepdims=True)
            cand = []
            for b, (toks, sc) in enumerate(active):
                idx = np.argpartition(-logp[b], self.beam_width)[:self.beam_width]
                for i in idx:
                    cand.append((toks + [int(i)], sc + float(logp[b, i])))
            cand.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
            beams = cand[:self.beam_width]

        finished += beams
        finished.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        out: List[str] = []
        for toks, _ in finished:
            s = "".join(self.id2tgt[i] for i in toks if i not in (self.eos, self.pad))
            if s and s not in out:
                out.append(s)
            if len(out) >= topk:
                break
        return out


def _logsumexp(x: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=keepdims))
