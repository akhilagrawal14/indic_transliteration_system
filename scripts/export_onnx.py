"""Export IndicXlit (fairseq) to ONNX and quantize to INT8 for CPU serving.

CTranslate2 handled the fairseq multilingual transformer natively. ONNX has no
such converter, so we export the encoder and decoder as separate graphs and drive
an external beam search (server/engine/onnx_engine.py). This exists to benchmark
ONNX Runtime against CTranslate2 on CPU and to test whether a different runtime's
INT8 changes quality.

Produces, under models/indicxlit/onnx/:
  encoder.onnx / decoder.onnx                 (fp32)
  encoder.int8.onnx / decoder.int8.onnx       (dynamic INT8 PTQ)
  vocab.json                                  (token maps + special ids)

Usage:
    python scripts/export_onnx.py
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.compat import stub_urduhack  # noqa: E402

stub_urduhack()
from ai4bharat.transliteration import XlitEngine  # noqa: E402

OUT_DIR = "models/indicxlit/onnx"
# Fixed decoder length. fairseq's causal future-mask bakes to the traced length,
# so we export at a constant T and right-pad the prefix at run time (padding sits
# in the future of the last real token, so causal masking makes it a no-op).
MAX_LEN = 32


class EncoderWrapper(torch.nn.Module):
    """src_tokens [B,S], src_lengths [B] -> encoder_out [S,B,C]."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, src_tokens, src_lengths):
        enc = self.encoder(src_tokens, src_lengths=src_lengths)
        return enc["encoder_out"][0]


class DecoderWrapper(torch.nn.Module):
    """prev_output_tokens [B,T], encoder_out [S,B,C] -> logits [B,T,V].

    Rebuilds the minimal encoder_out dict fairseq's decoder expects, with an
    empty padding mask (single-word inference never pads).
    """

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, prev_output_tokens, encoder_out):
        enc = {
            "encoder_out": [encoder_out],
            "encoder_padding_mask": [],
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
        }
        return self.decoder(prev_output_tokens, encoder_out=enc)[0]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    engine = XlitEngine("hi", beam_width=5, rescore=False)
    t = engine.transliterator
    model = t.models[0]
    model.to("cpu")
    model.eval()
    sd, td = t.src_dict, t.tgt_dict

    # --- Save vocab / token maps so serving needs no fairseq ---
    def src_line(word: str) -> str:
        return engine.pre_process([word], src_lang="en", tgt_lang="hi")[0]

    vocab = {
        "src_token2id": {sd[i]: i for i in range(len(sd))},
        "tgt_id2token": [td[i] for i in range(len(td))],
        "eos": td.eos(), "pad": td.pad(), "unk": td.unk(),
        "src_eos": sd.eos(), "src_unk": sd.unk(),
        "lang_tag": "__hi__",
    }
    with open(f"{OUT_DIR}/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    # --- Example inputs ---
    ids = sd.encode_line(src_line("mera"), add_if_not_exist=False,
                         append_eos=True).long()
    src_tokens = ids.unsqueeze(0)
    src_lengths = torch.tensor([ids.numel()])

    enc_wrap = EncoderWrapper(model).eval()
    dec_wrap = DecoderWrapper(model).eval()
    # Decoder is traced at a fixed length MAX_LEN (padded prefix).
    prev = torch.full((1, MAX_LEN), td.pad(), dtype=torch.long)
    prev[0, 0] = td.eos()
    with torch.no_grad():
        encoder_out = enc_wrap(src_tokens, src_lengths)
        _ = dec_wrap(prev, encoder_out)

    # --- Export encoder ---
    torch.onnx.export(
        enc_wrap, (src_tokens, src_lengths), f"{OUT_DIR}/encoder.onnx",
        input_names=["src_tokens", "src_lengths"], output_names=["encoder_out"],
        dynamic_axes={"src_tokens": {0: "B", 1: "S"},
                      "src_lengths": {0: "B"},
                      "encoder_out": {0: "S", 1: "B"}},
        opset_version=14,
    )
    # --- Export decoder (fixed T = MAX_LEN, only B and S dynamic) ---
    torch.onnx.export(
        dec_wrap, (prev, encoder_out), f"{OUT_DIR}/decoder.onnx",
        input_names=["prev_output_tokens", "encoder_out"], output_names=["logits"],
        dynamic_axes={"prev_output_tokens": {0: "B"},
                      "encoder_out": {0: "S", 1: "B"},
                      "logits": {0: "B"}},
        opset_version=14,
    )
    vocab["max_len"] = MAX_LEN
    with open(f"{OUT_DIR}/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"Exported fp32 encoder/decoder to {OUT_DIR}/ (decoder T={MAX_LEN})")

    # --- Dynamic INT8 quantization (weights int8, activations dynamic) ---
    from onnxruntime.quantization import quantize_dynamic, QuantType
    for name in ("encoder", "decoder"):
        quantize_dynamic(f"{OUT_DIR}/{name}.onnx", f"{OUT_DIR}/{name}.int8.onnx",
                         weight_type=QuantType.QInt8)
    print("Wrote dynamic INT8 encoder/decoder")

    for f in sorted(os.listdir(OUT_DIR)):
        mb = os.path.getsize(f"{OUT_DIR}/{f}") / (1024 * 1024)
        print(f"  {f:24s} {mb:6.1f} MB")


if __name__ == "__main__":
    main()
