"""Numerical tests of the generative masked-expression call on a tiny random model (T80-1 round 2, item 6).

Upstream's generative layer (bowang-lab/scGPT `flash_layers.py`, `FlashscGPTMHA`) needs flash-attn on CUDA, so it cannot
be imported here; these tests compare `_generate` with an independent two-stream reference written from that code's
structure instead: known positions (`<cls>` + unmasked genes) attend only to known positions, and each generated gene
attends to the known positions plus itself. This is a reimplementation check, not upstream numerical parity.
Skipped when torch is absent (CI installs no torch).
"""
# ruff: noqa: E501
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import scgpt_single_cell_pipeline.pipeline as P  # noqa: E402

E = 16
CONFIG = {"embsize": E, "d_hid": 32, "nhead": 2, "nlayers": 2, "vocab_size": 40, "dropout": 0.0}
TOKENS = [37, 3, 5, 8, 11, 13, 17, 19, 23]  # <cls> first
VALUES = [0.0, 4.0, 17.0, 33.0, 1.0, 49.0, 25.0, 12.0, 40.0]
MASKED = {2, 5, 7}


def _pipe(seed: int = 0):
    torch.manual_seed(seed)
    vocab = P.GeneVocabulary(tokens={f"G{i}": i for i in range(37)}, pad_id=38, cls_id=37, eoc_id=39)
    pipe = P.ScGPTPipeline(lambda rows: [], vocab, "cpu")
    pipe.model = P.build_model(CONFIG, vocab).eval()
    pipe.pretrain_heads = {
        "flag_encoder.weight": torch.randn(2, E),
        "mvc_decoder.W.weight": torch.randn(E, E) / 4,
        "mvc_decoder.gene2query.bias": torch.randn(E),
        "mvc_decoder.gene2query.weight": torch.randn(E, E) / 4,
    }
    return pipe


def _reference(pipe, tokens, values, masked, cls_value):
    """Two-stream forward written independently of `_generate`'s single masked sequence."""
    m, h = pipe.model, pipe.pretrain_heads
    known = [0] + [p for p in range(1, len(tokens)) if p not in masked]
    gen = sorted(masked)
    with torch.no_grad():
        ids_k = torch.tensor([[tokens[p] for p in known]])
        vals_k = torch.tensor([[cls_value] + [values[p] for p in known[1:]]], dtype=torch.float32)
        ids_g = torch.tensor([[tokens[p] for p in gen]])
        gene_g = m.embed_genes(ids_g)
        k = m.embed_genes(ids_k) + m.embed_values(vals_k)
        g = gene_g + h["flag_encoder.weight"][1]
        for layer in m.transformer.layers:
            ka = layer.self_attn(k, k, k, need_weights=False)[0]
            ga = torch.cat([layer.self_attn(g[:, j : j + 1], torch.cat([k, g[:, j : j + 1]], 1), torch.cat([k, g[:, j : j + 1]], 1), need_weights=False)[0] for j in range(g.shape[1])], 1)

            def block(x, a, layer=layer):
                x = layer.norm1(x + a)
                return layer.norm2(x + layer.linear2(torch.relu(layer.linear1(x))))

            k, g = block(k, ka), block(g, ga)
        expr = m.expr_decoder["fc"](g).squeeze(-1)[0]
        q = torch.sigmoid(gene_g @ h["mvc_decoder.gene2query.weight"].T + h["mvc_decoder.gene2query.bias"])
        mvc = (q @ h["mvc_decoder.W.weight"].T)[0] @ k[0, 0]
    return expr.tolist(), mvc.tolist()


@pytest.mark.parametrize("cls_value", [-2.0, 0.0])
def test_generate_matches_the_two_stream_reference(cls_value) -> None:
    pipe = _pipe()
    expr, mvc = pipe._generate(TOKENS, VALUES, MASKED, cls_value)
    ref_expr, ref_mvc = _reference(pipe, TOKENS, VALUES, MASKED, cls_value)
    assert len(expr) == len(mvc) == len(MASKED)  # one output per masked position, ascending order
    assert expr == pytest.approx(ref_expr, abs=1e-5) and mvc == pytest.approx(ref_mvc, abs=1e-5)


def test_attention_partition_and_inputs() -> None:
    pipe = _pipe()
    base_e, base_m = pipe._generate(TOKENS, VALUES, MASKED, -2.0)
    # generated -> other generated is blocked and known -> generated is blocked: changing generated gene 2's token
    # changes only its own outputs (the other generated genes and the <cls> cell embedding never see it).
    swapped = list(TOKENS)
    swapped[2] = 29
    e, mv = pipe._generate(swapped, VALUES, MASKED, -2.0)
    assert e[0] != pytest.approx(base_e[0], abs=1e-6) and e[1:] == pytest.approx(base_e[1:], abs=1e-6)
    assert mv[1:] == pytest.approx(base_m[1:], abs=1e-6)
    # a generated gene's own value is never read
    v2 = list(VALUES)
    v2[5] = 0.0
    assert pipe._generate(TOKENS, v2, MASKED, -2.0) == pytest.approx((base_e, base_m), abs=1e-7)
    # generated -> known is allowed: a known gene's value changes every generated output
    v3 = list(VALUES)
    v3[1] = 45.0
    e3, m3 = pipe._generate(TOKENS, v3, MASKED, -2.0)
    assert all(abs(a - b) > 1e-7 for a, b in zip(e3, base_e, strict=True)) and all(abs(a - b) > 1e-7 for a, b in zip(m3, base_m, strict=True))
    # the <cls> value is an input
    assert pipe._generate(TOKENS, VALUES, MASKED, 0.0)[0] != pytest.approx(base_e, abs=1e-6)
    # the flag: row 1 is used, row 0 is not
    pipe.pretrain_heads["flag_encoder.weight"][0] += 1.0
    assert pipe._generate(TOKENS, VALUES, MASKED, -2.0) == pytest.approx((base_e, base_m), abs=1e-7)
    pipe.pretrain_heads["flag_encoder.weight"][1] += 1.0
    assert pipe._generate(TOKENS, VALUES, MASKED, -2.0)[0] != pytest.approx(base_e, abs=1e-6)
