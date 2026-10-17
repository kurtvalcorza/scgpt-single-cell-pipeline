"""Round 2: the review's 32 real pbmc3k cells through the NEW API, with the gene-prior baseline, the constant baseline and
the shuffled-context control, over 5 mask seeds.

Usage: python -I t80_r2_pbmc.py <repo_dir> <weights_dir/scgpt> <hg19_dir> <out_json>
Cell selection copies review/scripts/t80_1_repro.py::pbmc_cells (seed 7, 32 cells). The gene prior is the mean bin
over 200 OTHER pbmc3k cells (disjoint from the 32), standing in for a training split. CPU only.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

repo, weights, hg19, out = (Path(a) for a in sys.argv[1:5])
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "tools"))

import torch  # noqa: E402

import tutorial_stages as T  # noqa: E402
from scgpt_single_cell_pipeline import pipeline as P  # noqa: E402

assert not torch.cuda.is_available()
genes = [line.split("\t")[1].strip() for line in (hg19 / "genes.tsv").read_text().splitlines()]
rows = [ln for ln in (hg19 / "matrix.mtx").read_text().splitlines() if not ln.startswith("%")]
cols: dict[int, dict[str, float]] = {}
for ln in rows[1:]:
    gi, ci, val = ln.split()
    cell = cols.setdefault(int(ci) - 1, {})
    sym = genes[int(gi) - 1]
    cell[sym] = cell.get(sym, 0.0) + float(val)
rng = random.Random(7)
pick = rng.sample(sorted(cols), 32)
cells = [cols[i] for i in pick]
others = [i for i in sorted(cols) if i not in set(pick)]
prior_cells = [cols[i] for i in random.Random(11).sample(others, 200)]

pipe = P.ScGPTPipeline.from_pretrained(device="cpu", weights_dir=weights)
prior = P.gene_bin_prior(prior_cells, pipe.vocabulary)
per_seed = []
for seed in range(42, 47):
    m = pipe.predict_masked(cells, mask_fraction=0.15, seed=seed)
    g = pipe.predict_masked_generative(cells, mask_fraction=0.15, seed=seed)
    c = pipe.masked_probe_controls(cells, mask_fraction=0.15, seed=seed, prior=prior)
    per_seed.append(T.probe_columns(m, g, c))
default = {k: T.mean_scores(v) for k, v in per_seed[0].items()}
g0 = pipe.predict_masked_generative(cells, mask_fraction=0.15, seed=42, cls_value=0.0)
report = {
    "torch": torch.__version__,
    "cells": 32,
    "default_seed_42": default,
    "generative_cls0_seed_42": T.mean_scores([x["expression_decoder"] for x in g0["cells"]]),
    "mvc_cls0_seed_42": T.mean_scores([x["mvc_head"] for x in g0["cells"]]),
    "seeds_42_46": T.summarise_probe(per_seed),
}
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
