"""T80-1 reproduction: pipeline MLM(-1) probe vs upstream generative-pretraining convention, same cells.

Usage: python -I t80_1_repro.py <repo_dir> <work_dir> <out_json>
CPU only (CUDA_VISIBLE_DEVICES=-1). Read-only on the repo clone.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

repo, work, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(repo / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scgpt_single_cell_pipeline import pipeline as P  # noqa: E402
from scgpt_single_cell_pipeline import samples as S  # noqa: E402

torch.manual_seed(0)
assert not torch.cuda.is_available()
pipe = P.ScGPTPipeline.from_pretrained(device="cpu", weights_dir=work / "weights" / "scgpt")
model = pipe.model
voc = pipe.vocabulary

orig = torch.load(str(work / "weights" / "orig.bin"), map_location="cpu", weights_only=True)
extra = {k: v for k, v in orig.items() if k.startswith(("flag_encoder", "mvc_decoder", "cls_decoder"))}
flag_w = orig["flag_encoder.weight"]  # (2, 512)
# Sanity: shared tensors equal (decoder + gene embedding)
sd = model.state_dict()
same_dec = all(
    torch.equal(sd[f"expr_decoder.fc.{i}.{p}"], orig[f"decoder.fc.{i}.{p}"]) for i in (0, 2, 4) for p in ("weight", "bias")
)
same_emb = torch.equal(sd["gene_encoder.embedding.weight"], orig["encoder.embedding.weight"])


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def layer_forward(x, attn_mask):
    """Run the 12 post-norm layers with an explicit (L, L) bool mask (True = blocked)."""
    return model.transformer(x, mask=attn_mask)


def embed_inputs(ids, vals):
    gene = model.gene_encoder["enc_norm"](model.gene_encoder["embedding"](ids))
    v = torch.clamp(vals.unsqueeze(-1), max=512)
    v = torch.relu(model.value_encoder["linear1"](v))
    v = model.value_encoder["norm"](model.value_encoder["linear2"](v))
    return gene, v


def generative_predict(tokens, values, masked, cls_value):
    """Upstream generative convention (scGPT pretrain.py @4068d67, training_tasks='both'):
    pcpt = <cls> + known genes with value embedding; gen = masked genes, gene emb + flag_encoder(1), no value;
    pcpt queries see pcpt keys; each gen query sees all pcpt keys + itself. Decoder on gen outputs."""
    pcpt_idx = [0] + [p for p in range(1, len(tokens)) if p not in masked]
    gen_idx = sorted(masked)
    ids_p = torch.tensor([tokens[p] for p in pcpt_idx]).unsqueeze(0)
    vals_p = torch.tensor([cls_value] + [values[p] for p in pcpt_idx[1:]], dtype=torch.float32).unsqueeze(0)
    ids_g = torch.tensor([tokens[p] for p in gen_idx]).unsqueeze(0)
    gp, vp = embed_inputs(ids_p, vals_p)
    gg = model.gene_encoder["enc_norm"](model.gene_encoder["embedding"](ids_g))
    x = torch.cat([gp + vp, gg + flag_w[1]], dim=1)
    n_p, n_g = len(pcpt_idx), len(gen_idx)
    L = n_p + n_g
    m = torch.ones(L, L, dtype=torch.bool)
    m[:n_p, :n_p] = False  # pcpt <-> pcpt
    m[n_p:, :n_p] = False  # gen -> pcpt
    ar = torch.arange(n_p, L)
    m[ar, ar] = False  # gen -> self
    h = layer_forward(x, m)
    pred = model.expr_decoder["fc"](h[:, n_p:]).squeeze(-1)[0]
    # MVC / GEP (inner product): gene query from gene token emb, cell emb = pcpt <cls> output
    cell = h[:, 0]
    q = torch.sigmoid(
        torch.nn.functional.linear(gg, extra["mvc_decoder.gene2query.weight"], extra["mvc_decoder.gene2query.bias"])
    )
    wq = torch.nn.functional.linear(q, extra["mvc_decoder.W.weight"])
    mvc = torch.bmm(wq, cell.unsqueeze(2)).squeeze(-1)[0]
    return pred.tolist(), mvc.tolist()


def mlm_predict(tokens, values, masked, mask_value, cls_value):
    vals = [cls_value] + [mask_value if p in masked else values[p] for p in range(1, len(tokens))]
    ids = torch.tensor(tokens).unsqueeze(0)
    g, v = embed_inputs(ids, torch.tensor(vals, dtype=torch.float32).unsqueeze(0))
    h = model.transformer(g + v)
    return model.expr_decoder["fc"](h).squeeze(-1)[0].tolist()


def run(cells, label, mask_fraction=0.15, seed=42):
    res = {"n_cells": len(cells)}
    with torch.no_grad():
        pm = pipe.predict_masked(cells, mask_fraction=mask_fraction, seed=seed)
    res["pipeline_predict_masked_mean_pearson"] = float(np.nanmean([r["pearson"] for r in pm["cells"]]))
    res["pipeline_mean_predicted_bin"] = float(np.mean([r["mean_predicted_bin"] for r in pm["cells"]]))
    # Re-create the same masks as predict_masked
    rng = random.Random(seed)
    enc = [P.bin_expression(c, voc) for c in cells]
    masks = []
    for e in enc:
        pos = list(range(1, len(e["tokens"])))
        k = max(1, int(round(len(pos) * mask_fraction)))
        masks.append(set(rng.sample(pos, k)))
    acc = {k: [] for k in ("mlm_m1_cls0", "mlm_m1_clsm2", "gen_cls0", "gen_clsm2", "gen_mvc_clsm2", "gene_prior")}
    pred_ranges = {k: [] for k in acc}
    all_true, all_gen = [], []
    with torch.no_grad():
        for e, msk in zip(enc, masks):
            t, v = e["tokens"], e["values"]
            pos = sorted(msk)
            truth = [v[p] for p in pos]
            a = mlm_predict(t, v, msk, -1.0, 0.0)
            acc["mlm_m1_cls0"].append(pearson(truth, [a[p] for p in pos]))
            pred_ranges["mlm_m1_cls0"] += [a[p] for p in pos]
            b = mlm_predict(t, v, msk, -1.0, -2.0)
            acc["mlm_m1_clsm2"].append(pearson(truth, [b[p] for p in pos]))
            pred_ranges["mlm_m1_clsm2"] += [b[p] for p in pos]
            g0, _ = generative_predict(t, v, msk, 0.0)
            acc["gen_cls0"].append(pearson(truth, g0))
            pred_ranges["gen_cls0"] += g0
            g2, mv = generative_predict(t, v, msk, -2.0)
            acc["gen_clsm2"].append(pearson(truth, g2))
            acc["gen_mvc_clsm2"].append(pearson(truth, mv))
            pred_ranges["gen_clsm2"] += g2
            pred_ranges["gen_mvc_clsm2"] += mv
            all_true += truth
            all_gen += g2
    out = {k: float(np.nanmean(x)) for k, x in acc.items() if x}
    res.update({f"{k}_mean_pearson": val for k, val in out.items()})
    res["pred_band_p5_p95"] = {
        k: [round(float(np.percentile(x, 5)), 2), round(float(np.percentile(x, 95)), 2)] for k, x in pred_ranges.items() if x
    }
    res["gen_clsm2_mae_bins"] = float(np.mean(np.abs(np.array(all_true) - np.array(all_gen))))
    res["chance_mae_mean_bin"] = float(np.mean(np.abs(np.array(all_true) - np.mean(all_true))))
    return res


def pbmc_cells(n=32, seed=7):
    d = work / "pbmc" / "filtered_gene_bc_matrices" / "hg19"
    genes = [line.split("\t")[1].strip() for line in (d / "genes.tsv").read_text().splitlines()]
    rows = [ln for ln in (d / "matrix.mtx").read_text().splitlines() if not ln.startswith("%")]
    ng, nc, _ = map(int, rows[0].split())
    cols: dict[int, dict[str, float]] = {}
    for ln in rows[1:]:
        gi, ci, val = ln.split()
        cols.setdefault(int(ci) - 1, {})
        sym = genes[int(gi) - 1]
        cols[int(ci) - 1][sym] = cols[int(ci) - 1].get(sym, 0.0) + float(val)
    rng = random.Random(seed)
    pick = rng.sample(sorted(cols), n)
    return [cols[i] for i in pick]


report = {
    "torch": torch.__version__,
    "dtype": str(next(model.parameters()).dtype),
    "orig_bin_extra_keys": sorted(extra) + ["flag_encoder.weight"],
    "orig_decoder_equals_safetensors": same_dec,
    "orig_gene_embedding_equals_safetensors": same_emb,
}
pb = pbmc_cells()
report["pbmc3k_32"] = run(pb, "pbmc")
report["pbmc3k_32_mask40"] = run(pb, "pbmc", mask_fraction=0.4)
recs = S.generate_sample_dataset(voc)
syn = [dict(r["counts"]) for r in recs[:32]]
report["synthetic_32"] = run(syn, "syn")
out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
