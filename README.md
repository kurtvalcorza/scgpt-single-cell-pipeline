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

**The architecture is re-implemented here.** The Hub repository ships weights and a config but no model code (`model_type: "scgpt"`, no `auto_map`), and the packager's loader lives in PyTDC with transformers and flash-attention as dependencies. `build_model()` rebuilds the scGPT encoder on `torch.nn.TransformerEncoder` plus the gene-token encoder, the continuous value encoder and the expression decoder, named so the pinned safetensors loads with `strict=True`, following the **original** `bowang-lab/scGPT` `model.py` (commit `cebd6fae655b9c585a4807daa3ac31bb764f06b4`) — including the value-encoder ReLU and clamp the PyTDC port omits. Evidence that it is wired right: the 159 packaged tensors are bit-identical to the original `.bin`, and the frozen `<cls>` embeddings separate real B, T and monocyte cells from a public PBMC dataset at 72/72 leave-one-out. Numerical parity with the upstream `scgpt` package was not measured.

**The gene vocabulary is not in the Hub repository.** The packager fetches it from Harvard Dataverse at runtime; here it is a fourth manifest entry (`vocab.json`, datafile `10809431`, SHA-256 `ee2b2c90…`), committed in `weights/scgpt/` and re-fetched from that persistent id by the standalone notebook, and verified like every other file. Symbols are HGNC gene symbols (`GAPDH`), matched case-insensitively; Ensembl ids are unknown and are dropped and **reported**, never mapped onto a real gene (the packager's tokenizer maps unknowns to id 0, which is the gene A1BG).

**The masked-expression decoder is not a usable imputer as shipped.** `predict_masked` and `predict_masked_genes` run the pretraining objective so you can measure it; on 32 real PBMC cells the decoder's predictions sat in a band around bins 26–32 whatever the context, and masking a lineage's marker genes in a cell of that lineage versus the other lineage changed nothing. The encoder is fine — see the embedding evidence above — so this pipeline exposes embeddings and classification and records the decoder's behaviour rather than claiming imputation.

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

`from_pretrained()` calls `stage_missing_files()` then `verify_snapshot()` (byte size + SHA-256 of every manifest entry, refusing a manifest without the config, the weights or the vocabulary; staging fetches only absent entries — Hub files only at the pinned revision, the vocabulary only from its Dataverse file id — and only with `allow_download=True`), asserts the pinned config's architecture fields against the package constants, validates the vocabulary (60,697 entries, ids 0..60696, `<pad>`/`<cls>`/`<eoc>` present), builds the encoder and loads the safetensors with `strict=True`. The upstream `scgpt_gh_repo_original_model.bin` is deliberately absent from the manifest and is never loaded. `.safetensors` files are git-ignored; `vocab.json` is committed. See `docs/WEIGHTS.md`.

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

`tutorials/scgpt_single_cell_colab.ipynb` is declared `E2E` and is **standalone** (DIMER Notebook Specification 2.0 §4): it is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and embeds the 3 package modules (`pipeline.py`, `samples.py`, `metrics.py`) verbatim in dependency order, the pinned model identity, the snapshot manifest and the exact runtime pins, so the exported `.ipynb` keeps working without this repository being reachable — no clone, no repository install, no repository import on its primary path. Do not edit it by hand; change the package or the template and regenerate (`python tools/build_notebook.py`; `--check` is enforced by the validator and CI). Its default path stages and digest-verifies the 4-file snapshot (the vocabulary from Dataverse), rebuilds the encoder and loads the weights strictly, generates and validates the 64-cell marker-programme dataset, splits it 36/12/16 stratified by class, shows how a cell becomes (gene, bin) tokens, extracts `<cls>` embeddings and asserts they are reproducible and gene-order invariant, probes the masked-expression objective and reports the negative result, measures the zero-training nearest-centroid baseline plus majority-class and library-size baselines, runs a 4-epoch bounded fine-tuning of the head and last encoder layer, evaluates accuracy/macro-F1/AUROC on the held-out split, classifies six freshly generated cells, exports the safetensors adapter and verifies reload parity. BYOD (genes-as-columns CSV, JSON or JSONL) is optional and gated off by default. See `tutorials/README.md`.

## Release status

**Candidate.** Static/unit checks — including the standalone generator parity checks (`tools/build_notebook.py --check`, `tests/test_notebook_parity.py`) — do not constitute clean-runtime notebook evidence. One local CPU pre-flight execution of the committed notebook is recorded in `docs/release-verification.md`; complete the supported clean-runtime procedure in that document against the exact release revision before calling the notebook release-grade. The fleet inventory's nominated first contract for this row — multi-batch integration — is not implemented; see `MODEL_CARD.md` (*DIMER deployment notes*).

## Licensing

- Upstream weights: MIT (`tdc/scGPT`, packaging the MIT-licensed `bowang-lab/scGPT` whole-human checkpoint), staged unmodified from the pinned revision.
- Gene vocabulary: TDC `scgpt_vocab` (Harvard Dataverse datafile 10809431), distributed by PyTDC under MIT; committed here unmodified.
- This repository's code and documentation: Apache-2.0 (`LICENSE`).
- The upstream licence governs your use of the weights, including commercial use and redistribution; this repository grants no rights beyond it.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
