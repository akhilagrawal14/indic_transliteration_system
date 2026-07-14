"""Analytic FLOP budget for one IndicXlit transliteration request (report section 4.1f).

Why this exists: sections 2.3 and 4.1 claim the GPU loses because "the arithmetic is
negligible and per-call overhead dominates". That is a checkable claim, so this script
counts the arithmetic instead of asserting it.

Model shape is read from the fairseq checkpoint, so the numbers cannot drift from the
model actually being served. FLOPs = 2 x MACs (one multiply + one add).

Decoding is modelled the way CTranslate2 actually runs it:
  - KV cache: the prefix is not recomputed each step, so self-attention at step t only
    scores the new query against t cached keys.
  - Cross-attention K/V are projected ONCE from the encoder output and reused by every
    step and every beam.
  - Beam search multiplies decoder work by the beam width (B hypotheses per step).

Usage:
    python scripts/flops.py                      # defaults: S=12, T=11, beam=5
    python scripts/flops.py --src-len 12 --out-len 11 --beam 5
"""

import argparse

import torch

CKPT = "models/indicxlit/fairseq_original/indicxlit.pt"


def load_shape(path: str):
    """Read architecture dims straight from the checkpoint (no hardcoding)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg") or {}
    src = cfg.get("model") if isinstance(cfg, dict) and cfg.get("model") is not None \
        else ckpt.get("args")
    get = (lambda k: src.get(k)) if isinstance(src, dict) else (lambda k: getattr(src, k))
    tgt_vocab = ckpt["model"]["decoder.output_projection.weight"].shape[0]
    params = sum(v.numel() for v in ckpt["model"].values() if hasattr(v, "numel"))
    return {
        "d": get("encoder_embed_dim"),
        "ffn": get("encoder_ffn_embed_dim"),
        "layers": get("encoder_layers"),          # encoder and decoder depth match here
        "tgt_vocab": tgt_vocab,
        "params": params,
    }


def encoder_macs(s: int, d: int, ffn: int, layers: int) -> int:
    """Bidirectional encoder over `s` tokens: QKVO projections, attention, FFN."""
    per_layer = 4 * s * d * d + 2 * s * s * d + 2 * s * d * ffn
    return layers * per_layer


def decoder_macs(s: int, t: int, beam: int, d: int, ffn: int, layers: int,
                 tgt_vocab: int) -> int:
    """Autoregressive decoder: `t` sequential steps x `beam` hypotheses, KV-cached."""
    # One-time: cross-attention K,V projected from the encoder output (shared).
    total = layers * 2 * s * d * d
    for step in range(1, t + 1):
        per_layer = (
            4 * d * d          # self-attn Q,K,V,O for the single new token
            + 2 * step * d     # self-attn scores vs `step` cached keys (QK^T + AV)
            + 2 * d * d        # cross-attn Q and O projections
            + 2 * s * d        # cross-attn scores vs cached encoder K,V
            + 2 * d * ffn      # FFN
        )
        total += beam * (layers * per_layer + d * tgt_vocab)  # + output projection
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--src-len", type=int, default=12,
                   help="source tokens: ~10 chars + __hi__ tag + </s>")
    p.add_argument("--out-len", type=int, default=11, help="decode steps")
    p.add_argument("--beam", type=int, default=5)
    args = p.parse_args()

    m = load_shape(args.ckpt)
    d, ffn, layers, vocab = m["d"], m["ffn"], m["layers"], m["tgt_vocab"]
    s, t, b = args.src_len, args.out_len, args.beam

    enc = encoder_macs(s, d, ffn, layers)
    dec = decoder_macs(s, t, b, d, ffn, layers, vocab)
    greedy = enc + decoder_macs(s, t, 1, d, ffn, layers, vocab)
    total = enc + dec
    fl = lambda macs: 2 * macs  # noqa: E731

    print(f"params      : {m['params']:,} (~{m['params']/1e6:.2f}M)")
    print(f"architecture: {layers}+{layers} layers, d_model {d}, FFN {ffn}, "
          f"target vocab {vocab}")
    print(f"request     : S={s} tokens, T={t} steps, beam={b}\n")
    print(f"encoder      : {fl(enc)/1e6:8.2f} MFLOPs  ({100*enc/total:.0f}%)")
    print(f"decoder      : {fl(dec)/1e6:8.2f} MFLOPs  ({100*dec/total:.0f}%)")
    print(f"TOTAL/request: {fl(total)/1e6:8.2f} MFLOPs  ({fl(total)/1e9:.4f} GFLOPs)")
    print(f"greedy(beam=1): {fl(greedy)/1e6:7.2f} MFLOPs\n")

    # Tie the arithmetic back to the measured latencies (report section 4.1).
    cpu_ms, gpu_ms, l4_int8_tops = 7.39, 12.33, 121e12
    print(f"measured CPU p50 {cpu_ms} ms -> effective "
          f"{fl(total)/1e9/(cpu_ms/1e3):.1f} GFLOP/s on one core")
    print(f"same math on an L4 at peak ({l4_int8_tops/1e12:.0f} TOPS INT8): "
          f"{fl(total)/l4_int8_tops*1e6:.1f} us")
    print(f"measured L4 p50 {gpu_ms} ms -> "
          f">{100*(1 - (fl(total)/l4_int8_tops)/(gpu_ms/1e3)):.2f}% of GPU time is "
          f"launch/transfer overhead, not arithmetic")


if __name__ == "__main__":
    main()
