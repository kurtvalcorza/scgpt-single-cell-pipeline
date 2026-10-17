"""T80-1 fix verification: the new pipeline API vs the review's reproduction, same cells and masks.

Usage: python -I t80_1_verify.py <repo_dir> <weights_dir/scgpt> <out_json>
CPU only (CUDA_VISIBLE_DEVICES=-1). Reviewer values (review/work/t80_1_results.json, synthetic_32):
MLM 0.087, generative <cls> 0 0.081, generative <cls> -2 0.065, MVC (<cls> -2) 0.151.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

repo, weights, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(repo / "src"))

import torch  # noqa: E402

from scgpt_single_cell_pipeline import pipeline as P  # noqa: E402
from scgpt_single_cell_pipeline import samples as S  # noqa: E402

assert not torch.cuda.is_available()
t0 = time.perf_counter()
pipe = P.ScGPTPipeline.from_pretrained(device="cpu", weights_dir=weights)
load_s = time.perf_counter() - t0


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def readings(cells, seed=42, frac=0.15):
    mlm = pipe.predict_masked(cells, mask_fraction=frac, seed=seed)
    out = {"mlm": mean([c["pearson"] for c in mlm["cells"]])}
    for cls_value in (0.0, -2.0):
        g = pipe.predict_masked_generative(cells, mask_fraction=frac, seed=seed, cls_value=cls_value)
        out[f"generative_cls{cls_value:g}"] = mean([c["expression_decoder"]["pearson"] for c in g["cells"]])
        out[f"mvc_cls{cls_value:g}"] = mean([c["mvc_head"]["pearson"] for c in g["cells"]])
        out[f"generative_cls{cls_value:g}_band"] = [
            min(c["expression_decoder"]["mean_predicted_bin"] for c in g["cells"]),
            max(c["expression_decoder"]["mean_predicted_bin"] for c in g["cells"]),
        ]
    return out


voc = pipe.vocabulary
recs = S.generate_sample_dataset(voc)
report = {
    "torch": torch.__version__,
    "load_seconds": round(load_s, 2),
    "heads_loaded": sorted(pipe.pretrain_heads or {}),
    "reviewer_synthetic_32": readings([dict(r["counts"]) for r in recs[:32]]),
}
out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
