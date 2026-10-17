"""T80-1 follow-up: vocab parity with upstream census vocab, and lineage conditioning on real PBMC cells.

Usage: python -I t80_1_lineage.py <repo_dir> <work_dir> <out_json>
"""
from __future__ import annotations

import json
import random
import runpy
import sys
from pathlib import Path

repo, work, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.argv = [sys.argv[0], str(repo), str(work), str(work / "_unused.json")]
# Reuse helpers without re-running the main experiment: exec definitions only.
src = (Path(__file__).parent / "t80_1_repro.py").read_text(encoding="utf-8").split("report = {")[0]
g: dict = {"__name__": "t80_helpers", "__file__": str(Path(__file__).parent / "t80_1_repro.py")}
exec(compile(src, "t80_1_repro.py", "exec"), g)
P, S, torch, np, voc, model = g["P"], g["S"], g["torch"], g["np"], g["voc"], g["model"]
embed_inputs, flag_w, extra = g["embed_inputs"], g["flag_w"], g["extra"]

# --- vocab parity -----------------------------------------------------------------------------
tdc = json.loads((work / "weights" / "scgpt" / "vocab.json").read_text(encoding="utf-8"))
census = json.loads((work / "up4068" / "default_census_vocab.json").read_text(encoding="utf-8"))
common = set(tdc) & set(census)
vocab = {
    "tdc_entries": len(tdc),
    "census_entries": len(census),
    "common": len(common),
    "same_id_for_common": sum(1 for k in common if tdc[k] == census[k]),
    "only_tdc": sorted(set(tdc) - set(census))[:10],
    "only_census": sorted(set(census) - set(tdc))[:10],
}

# --- real PBMC lineage probe ------------------------------------------------------------------
d = work / "pbmc" / "filtered_gene_bc_matrices" / "hg19"
genes = [ln.split("\t")[1].strip() for ln in (d / "genes.tsv").read_text().splitlines()]
rows = [ln for ln in (d / "matrix.mtx").read_text().splitlines() if not ln.startswith("%")]
cols: dict[int, dict[str, float]] = {}
for ln in rows[1:]:
    gi, ci, val = ln.split()
    c = cols.setdefault(int(ci) - 1, {})
    s = genes[int(gi) - 1]
    c[s] = c.get(s, 0.0) + float(val)
T = list(P.__dict__.get("T_CELL_PROGRAMME", S.T_CELL_PROGRAMME))
B = list(S.B_CELL_PROGRAMME)
tcells = [c for c in cols.values() if c.get("CD3D", 0) > 1 and c.get("CD3E", 0) > 0 and c.get("MS4A1", 0) == 0 and c.get("CD79A", 0) == 0]
bcells = [c for c in cols.values() if c.get("MS4A1", 0) > 0 and c.get("CD79A", 0) > 0 and c.get("CD3D", 0) == 0 and c.get("CD3E", 0) == 0]
rng = random.Random(11)
tcells, bcells = rng.sample(tcells, 12), rng.sample(bcells, 12)
query = [s for s in T + B if voc.resolve(s) is not None]


def gen_query(cell, qgenes):
    """Generative convention: drop the queried genes from the known set, query them as gen tokens."""
    known = {k: v for k, v in cell.items() if k not in set(qgenes)}
    e = P.bin_expression(known, voc)
    ids_p = torch.tensor(e["tokens"]).unsqueeze(0)
    vals_p = torch.tensor([0.0] + e["values"][1:], dtype=torch.float32).unsqueeze(0)
    ids_g = torch.tensor([voc.resolve(q) for q in qgenes]).unsqueeze(0)
    gp, vp = embed_inputs(ids_p, vals_p)
    gg = model.gene_encoder["enc_norm"](model.gene_encoder["embedding"](ids_g))
    x = torch.cat([gp + vp, gg + flag_w[1]], dim=1)
    n_p, n_g = ids_p.shape[1], ids_g.shape[1]
    L = n_p + n_g
    m = torch.ones(L, L, dtype=torch.bool)
    m[:n_p, :n_p] = False
    m[n_p:, :n_p] = False
    ar = torch.arange(n_p, L)
    m[ar, ar] = False
    h = model.transformer(x, mask=m)
    pred = model.expr_decoder["fc"](h[:, n_p:]).squeeze(-1)[0]
    q = torch.sigmoid(torch.nn.functional.linear(gg, extra["mvc_decoder.gene2query.weight"], extra["mvc_decoder.gene2query.bias"]))
    mvc = torch.bmm(torch.nn.functional.linear(q, extra["mvc_decoder.W.weight"]), h[:, 0].unsqueeze(2)).squeeze(-1)[0]
    return pred.tolist(), mvc.tolist()


lineage = {}
with torch.no_grad():
    for lab, cells in (("T cells", tcells), ("B cells", bcells)):
        for mname, mg in (("T markers", [q for q in query if q in T]), ("B markers", [q for q in query if q in B])):
            ex, mv = [], []
            for c in cells:
                a, b = gen_query(c, mg)
                ex.append(float(np.mean(a)))
                mv.append(float(np.mean(b)))
            present = [c for c in cells if any(c.get(s, 0) > 0 for s in mg)]
            pm = (g["pipe"].predict_masked_genes(present, mg) if present
                  else {"predicted_mean_bin": None, "true_mean_bin": None})
            pm = {**pm, "n_cells_with_markers": len(present)}
            lineage[f"{lab} / {mname}"] = {
                "gen_expr_decoder_mean_bin": round(float(np.mean(ex)), 2),
                "gen_mvc_mean_bin": round(float(np.mean(mv)), 2),
                "pipeline_mlm_predicted_mean_bin": pm["predicted_mean_bin"],
                "pipeline_mlm_true_mean_bin": pm["true_mean_bin"], "pipeline_mlm_n_cells": pm["n_cells_with_markers"],
            }
res = {"vocab": vocab, "n_t": len(tcells), "n_b": len(bcells), "lineage_real_pbmc": lineage}
out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res, indent=2))
