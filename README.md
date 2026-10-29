# scGPT Single-Cell Pipeline

DIMER-oriented pipeline for **scGPT** (`tdc/scGPT`, the whole-human encoder repackaged in safetensors by Therapeutics Data Commons), pinned to an immutable Hugging Face revision. The repository exposes `<cls>` cell embeddings from gene-expression counts, two probes of the masked-expression pretraining objective, a labelled-dataset contract with explicit ceilings, a zero-training nearest-centroid classifier on the frozen embeddings, bounded fine-tuning of a cell-state classification head, held-out metrics with trivial baselines, and a safetensors adapter artifact that is digest-verified before it is loaded.

## Upstream alignment

- Model: `tdc/scGPT`
- Revision: `acf749f35bf5c0b00633838f02588272ed0d9911`
- Upstream weight license: MIT
- Upstream task: single-cell foundation modelling (masked-expression pretraining on 33 million human cells; Cui et al., Nature Methods 2024); this repository uses the encoder for cell embeddings and cell-state classification
- Runtime: **torch + safetensors only** — the encoder is re-implemented in this repository (see below); no transformers, no PyTDC, no flash-attention, no remote code
- Repository adaptation: **E2E** (bounded fine-tuning of a linear head plus the last *n* encoder layers, with a portable safetensors adapter)

## Three things to know before you start

**The architecture is re-implemented here.** The Hub repository ships weights and a config but no model code (`model_type: "scgpt"`, no `auto_map`), and the packager's loader lives in PyTDC with transformers and flash-attention as dependencies. `build_model()` rebuilds the scGPT encoder on `torch.nn.TransformerEncoder` plus the gene-token encoder, the continuous value encoder and the expression decoder, named so the pinned safetensors loads with `strict=True`, following the **original** `bowang-lab/scGPT` `model.py` (commit `cebd6fae655b9c585a4807daa3ac31bb764f06b4`) — including the value-encoder ReLU and clamp the PyTDC port omits. Functional evidence for the wiring (not proof of correctness): the 159 packaged tensors are bit-identical to the original `.bin`, and the frozen `<cls>` embeddings separate real B, T and monocyte cells from a public PBMC dataset at 72/72 leave-one-out. Numerical parity with the upstream `scgpt` package was not measured.

**The gene vocabulary is not in the Hub repository.** The packager fetches it from Harvard Dataverse at runtime; here it is a manifest entry (`vocab.json`, datafile `10809431`, SHA-256 `ee2b2c90…`), committed in `weights/scgpt/` and fetched by the standalone notebook from that persistent id, with an immutable raw-GitHub URL for the exact committed bytes as an availability fallback. The downloaded bytes are verified like every other file regardless of source. Symbols are HGNC gene symbols (`GAPDH`), matched case-insensitively; Ensembl ids are unknown and are dropped and **reported**, never mapped onto a real gene (the packager's tokenizer maps unknowns to id 0, which is the gene A1BG).

**This repository doesn't validate imputation; the masked-expression probe is descriptive.** Masked-expression prediction is a descriptive probe, read three ways on the same masks: `predict_masked` (upstream's fine-tuning MLM convention, masked value -1), and `predict_masked_generative`, which follows the generative convention in the cited upstream pretraining script; the released checkpoint's exact training configuration is unverified (masked genes enter as gene embedding plus the `flag_encoder` flag, no value; the checkpoint's `flag_encoder` and MVC head are compatible with it) and reads them with both the expression decoder and the restored MVC head. It is a reimplementation without upstream numerical parity: upstream's generative layer needs flash-attn on CUDA, so a two-stream reference in `tests/test_generative_numerics.py` checks the attention partition instead. On 32 real PBMC cells (10x Genomics `pbmc3k`, 15 % masked, seed 42; CPU runs of this API in `docs/reviews/2026-10-17-tutorial-review/`, not re-run by the notebook) the mean per-cell Pearson was 0.31 (MLM call), 0.37 (generative call, `<cls>` value -2; 0.39 with `<cls>` value 0) and 0.55 (MVC head), but a per-gene prior that ignores the cell scored 0.58 and the MVC head kept 0.48 with another cell's context, so these are reconstruction associations, not proof of contextual recovery. On 32 of the tutorial's synthetic profiles the same readings were 0.087, 0.065 and 0.151; these synthetic profiles differ substantially from measured cells; the low correlations here don't diagnose a broken decoder, and the cause hasn't been isolated. This tutorial doesn't validate imputation; on these sampled cells and masks the expression decoder's bin MAE didn't clearly beat a constant baseline that predicts every masked gene as the mean bin of the cell's unmasked genes (11.81 MLM and 12.00 generative against 11.82 on the PBMC cells). This profile is embeddings and classification, with a descriptive masked-expression probe. The two pretraining heads the TDC packaging dropped (`flag_encoder`, `mvc_decoder`) are a fifth manifest entry, `pretrain_heads.safetensors`, extracted once from the original checkpoint at the pinned revision by `tools/derive_pretrain_heads.py` (pinned size and SHA-256, a static pre-check of the pickle opcodes that is not a security audit, then `torch.load(weights_only=True)`); the pipeline never loads the pickle. See `docs/WEIGHTS.md`.

## Quick start

```python
from scgpt_single_cell_pipeline import ScGPTPipeline, generate_sample_dataset, split_dataset

pipe = ScGPTPipeline.from_pretrained()               # verifies the 4-file snapshot under weights/ first
records = generate_sample_dataset(pipe.vocabulary)   # 64 synthetic cells built from real T/B marker programmes
emb = pipe.embed([records[0]["counts"]])
print(emb["dimension"], len(emb["embeddings"][0]))   # 512 512

splits = split_dataset(records)
print(pipe.nearest_centroid_evaluate(splits["train"], splits["test"])["accuracy"])   # no training at all
pipe.adapt(splits["train"], splits["validation"])    # bounded AdamW fine-tuning (head + last layer)
print(pipe.evaluate(splits["test"])["accuracy"])
print(pipe.classify([records[0]["counts"]])["predictions"][0]["label"])
```

`embed()`, `predict_masked()` and `classify()` take 1..32 cells (`MAX_CELLS_PER_CALL`), each a `{gene symbol: count}` mapping of finite non-negative numbers — raw or normalised, because binning is rank-based within the cell — with at least 10 detected genes (`MIN_DETECTED_GENES`) that resolve to the vocabulary; a cell with more than 1,535 detected genes (`MAX_GENES_PER_CELL`, the 1,536-position input minus `<cls>`) keeps the most expressed and reports the truncation. Encoding mirrors `scgpt/preprocess.py`: normalise to 10,000 counts, log1p, quantile-bin within the cell into 51 levels with the training-time random tie spread (seeded, so reproducible), `<cls>` first with value 0. `classify()` requires a prior `adapt()` or `from_artifact()`. Datasets are `{id, counts, label}` records: at least 8 rows and 3 per class, at most 2,000 rows and 20 classes, unique ids.

## Weights layout

```
weights/scgpt/   config.json  model.safetensors  vocab.json  README.md  dimer-base-manifest.json
```

`from_pretrained()` calls `stage_missing_files()` then `verify_snapshot()` (byte size + SHA-256 of every manifest entry, refusing a manifest without the config, the weights or the vocabulary; staging fetches only absent entries — Hub files only at the pinned revision, the vocabulary from its Dataverse file id with an immutable exact-byte mirror fallback — and only with `allow_download=True`), asserts the pinned config's architecture fields against the package constants, validates the vocabulary (60,697 entries, ids 0..60696, `<pad>`/`<cls>`/`<eoc>` present), builds the encoder and loads the safetensors with `strict=True`. The upstream `scgpt_gh_repo_original_model.bin` is deliberately absent from the manifest and is never loaded. `.safetensors` files are git-ignored; `vocab.json` is committed. See `docs/WEIGHTS.md`.

## Sample data

`generate_sample_dataset(vocabulary)` builds 64 cells (32 `t-like`, 32 `b-like`) from two real PBMC lineage programmes — 24 T-cell markers (CD3D, CD3E, IL7R, TRAC, …) and 24 B-cell markers (CD79A, MS4A1, CD19, IGHM, …) — plus 300 background genes drawn from the pinned vocabulary. Every cell expresses the same 348 genes and is scaled to 20,000 total counts, so library size and gene detection carry no signal by construction; the classes differ only in where the two programmes sit in each cell's ranking. It is a generator rule with real marker genes in it, not measured cells.

## Adapter artifacts

`save_artifact(dir)` writes `adapter.safetensors` (the trained tensors — the head and the unfrozen encoder layers; about 6 MB with the default one layer, 6 KB head-only) and a `manifest.json` recording the artifact format, the exact base model id and revision, the class order, the tensor names, the file size and SHA-256, and the full adaptation configuration. `ScGPTPipeline.from_artifact(dir)` re-verifies the base snapshot, then checks the artifact manifest, the base identity and every digest **before** deserialising, and refuses any tensor the classifier architecture does not have.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: injected backends, toy vocabularies and temporary manifests, never the weights; the real vocabulary is checked when it is present.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/scgpt-single-cell-pipeline/blob/main/tutorials/scgpt_single_cell_colab.ipynb)

`tutorials/scgpt_single_cell_colab.ipynb` is declared `E2E` and `GUIDED` and is **standalone** (DIMER Notebook Specification 2.2 §4): it is generated by `tools/build_notebook.py` (/3) from `tools/notebook_template.py` and carries the 4 package modules, the stage runner `tools/tutorial_stages.py`, a hash-locked requirements file and the snapshot manifest, so the exported `.ipynb` keeps working without a clone, repository install or repository import. Nothing is installed into the notebook kernel: a pinned `uv` builds an isolated environment from the lock and every stage runs there in its own process, so `Run all` needs no restart. Do not edit it by hand; change the package, the runner or the template and regenerate (`python tools/build_notebook.py`; `--check` is enforced by the validator and CI). Its default path stages and digest-verifies the 4-file snapshot (the vocabulary from Dataverse, or its immutable exact-byte mirror when Dataverse is unavailable), rebuilds the encoder and loads the weights strictly, generates and validates the 64-cell marker-programme dataset, splits it 36/12/16 stratified by class, shows how a cell becomes (gene, bin) tokens, extracts `<cls>` embeddings and reports whether they are reproducible and unaffected by the input's gene order (a preprocessing check), runs a descriptive masked-expression probe with a gene-prior baseline and a shuffled-context control and reports the negative result, measures the zero-training nearest-centroid baseline plus majority-class and library-size baselines, runs a 4-epoch bounded fine-tuning of the head and last encoder layer, exports the safetensors adapter, evaluates accuracy/macro-F1/AUROC on the held-out split with the adapter rebuilt in a fresh process, and classifies six freshly generated cells with a reload-parity check. BYOD (genes-as-columns CSV, JSON or JSONL, by `BYOD_PATH` or Colab upload) is optional and gated off by default. See `tutorials/README.md`.

## Release status

**Candidate — one-pass hosted run recorded, awaiting maintainer promotion.** The current blob `be0b6b87` (commit `711a9e6`, the notebook with the Section 7 probe prose requoted) completed one pass with 0 errors on a fresh Colab Tesla T4 on 2026-10-17 (Colab account 2; Colab CLI 0.7.4 sequential execution, 14/14 code cells, 138.9 s, `cuda:0`). REL13: hosted re-run after the requote, this blob; every probe, baseline, test and reload figure equals the earlier 2026-10-17 run of blob `90b41684` (commit `5d95ece`): test accuracy / macro-F1 / AUROC 1.0 on 16 cells, tied with the zero-training centroid rule (`saturated`), 6/6 new cells, reload parity exact; masked-expression probe mean Pearson, default seed: MLM 0.0045, generative 0.0393, MVC head 0.0926. REL13: the earlier-blob figures that follow were not re-run on the current blob. The regenerated notebook (isolated environment, no restart): blob `07755aa2ddf8` (commit `8d35011`) completed one pass with no restart (`restarted: false`) and 0 errors on a fresh Colab Tesla T4 on 2026-10-10 (Colab CLI 0.7.4 sequential execution, 14/14 code cells, 143.7 s, `cuda:0`; test accuracy / macro-F1 / AUROC 1.0 on 16 cells, tied with the zero-training centroid rule (`saturated`), 6/6 new cells, reload parity exact). The previous notebook (commit `0f66318` / blob `31e26691`) completed all 14 code cells in a fresh Kaggle Tesla T4 container only after an interpreter restart (`restarted: true`), so that run is a functional record, not clean-runtime qualification; it fetched and verified the complete snapshot, including the documented immutable vocabulary fallback after Harvard Dataverse returned HTTP 504. See `docs/release-verification.md`. `Release-grade` requires a recorded one-pass Run all of the exact current blob and an explicit maintainer decision. The fleet inventory's nominated first contract for this row — multi-batch integration — is not implemented; see `MODEL_CARD.md` (*DIMER deployment notes*).

The current blob `9d521522` (code commit `d788479`) shows a refused BYOD input as exactly one `Refused: <file>: <reason>` line, with no traceback, and stops Run all at that cell (previously the stage printed the refusal and Run all carried on); genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20). Its hosted Colab T4 runs (the default path and a scripted refusal, `USE_BYOD=True;BYOD_PATH='/etc/hostname'`) are pending, and the hosted runs of earlier blobs, including `be0b6b87`, are history for it (REL14).

## Licensing

- Upstream weights: MIT (`tdc/scGPT`, packaging the MIT-licensed `bowang-lab/scGPT` whole-human checkpoint), staged unmodified from the pinned revision.
- Gene vocabulary: TDC `scgpt_vocab` (Harvard Dataverse datafile 10809431), distributed by PyTDC under MIT; committed here unmodified.
- This repository's code and documentation: Apache-2.0 (`LICENSE`).
- The upstream licence governs your use of the weights, including commercial use and redistribution; this repository grants no rights beyond it.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
