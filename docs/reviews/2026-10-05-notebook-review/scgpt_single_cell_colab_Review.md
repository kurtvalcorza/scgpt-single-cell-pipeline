# scGPT Single-Cell E2E Notebook — Review

**Verdict: Needs revision** (two Majors, two Minors)  
**Review date:** 5 October 2026  
**Repository:** `kurtvalcorza/scgpt-single-cell-pipeline`  
**Notebook:** `tutorials/scgpt_single_cell_colab.ipynb`  
**Reviewed commit:** `6c35b4a` (`main`)  
**Notebook Git blob:** `31e266916153146c97b85f81a527937999a7a708`, generated from `67043c7`. This is the blob executed in the recorded Kaggle T4 run of 2026-09-18 (commit `0f66318`). At the reviewed commit, `tools/build_notebook.py --check` and `tools/validate_release_assets.py` both exit 0.  
**Finding prefix:** `SCG`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2, `ml-worker` `origin/main` at `b9fdd1f`.

## Executive assessment

This is the most scientifically honest of the older E2E notebooks reviewed so far.

- **Model and data provenance.**
  - It rebuilds the scGPT encoder on plain torch modules and loads the TDC safetensors with `strict=True`, so no remote code runs.
  - It pins the gene vocabulary by size and SHA-256. The vocabulary comes from Harvard Dataverse, with an exact-byte mirror fallback; the recorded Kaggle run used that fallback after a 504.
- **Data design.**
  - It generates a 64-cell dataset from real T- and B-lymphocyte marker programmes.
  - All cells share the same gene set and library size, so neither depth nor detection carries signal.
  - It explains binning, and shows that binning does not change when counts are scaled.
  - It checks that embeddings are reproducible and do not depend on gene order.
- **Honest negative result.** It probes the masked-expression decoder and reports the negative result openly: Pearson 0.0045, with no lineage conditioning.
- **Evaluation.**
  - It sets the zero-training nearest-centroid rule beside majority-class and library-size baselines, all fitted on training data only.
  - It says plainly that if the zero-training baseline is already perfect, "fine-tuning has nothing to add on this data".
  - It evaluates on an independent test split, and the validation split is used for monitoring only.
- **Artifact.** It exports a manifest-bound adapter and reloads it with exact parity.

Two problems remain from the generation-2 template:

1. **`Run all` needs a manual restart (SCG-M1).** The recorded Kaggle run completed "after one expected interpreter restart" following the in-kernel pip install.
2. **The guided layer is largely absent (SCG-M2).** The notebook is declared `GUIDED`. It has accurate "Look for" notes in Section 4 and good interpretation prose. It has no audience statement, how-to-use, roadmap, glossary, predictions, checkpoints, troubleshooting or conclusion template, and no cell is labelled or collapsed as infrastructure.

There are also two Minors:

- literal `{{id, counts, label}}` template braces in the learner text (SCG-m1);
- a BYOD branch that only works in Colab, with an unguarded upload, and that reuses test-split cells as its "new" data (SCG-m2).

| Measure (Kaggle T4, blob `31e26691`, 2026-09-18) | Value |
|---|---|
| Code cells | 14/14 after one restart; 193.1 s |
| Snapshot | 4 files verified; vocabulary from the mirror (Dataverse HTTP 504) |
| Split | 36 / 12 / 16 (stratified, seed 42) |
| Test metrics | adapted: accuracy, macro-F1 and AUROC all 1.0. Baselines: nearest-centroid (zero training) 1.0; majority 0.5 / 0.3333; library size 0.625 / 0.619 / AUROC 0.7188 |
| Masked-expression probe | Pearson 0.0045; no lineage conditioning (documented negative) |
| New cells / reload | 6/6; reload parity 0.0 |

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` |
| Declared spec | DIMER Notebook Specification **2.0** |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites: "what a single-cell expression profile is, what library size and quantile binning mean, and how accuracy, macro-F1 and AUROC differ" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12. CPU is enough"; CUDA used when present |
| Promised outcomes | pinned install; verified snapshot with vocabulary from a second source; 64 validated cells and a stratified split; tokenisation shown; embeddings checked for reproducibility and gene-order invariance; masked-expression probe; three baselines; bounded fine-tune; held-out accuracy, macro-F1 and AUROC; new-cell inference; adapter export and reload parity; provenance; BYOD (CSV, JSON or JSONL) through the same cells |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py` |
| Release status | `Candidate — clean-runtime qualified, promotion pending` (`STATUS.md`) |

### Evidence actually obtained

- **Source inspection:**
  - all 31 cells (14 code; cells 5, 7 and 9 are the carried `metrics.py`, `pipeline.py` and `samples.py`);
  - `samples.py`: `generate_sample_dataset`, `validate_dataset`, `split_dataset`, `load_byod_dataset`;
  - `metrics.py`: the baselines and AUROC;
  - `STATUS.md` and `docs/release-verification.md`.
- **Documented execution evidence:**
  - the Kaggle T4 qualification row in `docs/release-verification.md` (commit `0f66318`, blob `31e26691`), which records one expected restart;
  - a local Windows pre-flight row.
  - Neither executed notebook is committed. The evidence is the table rows only.
  - There is no Colab run and no BYOD run.
- **Direct execution (this review), torch-free** (Python 3.12, standard library and `numpy`):
  - Fetched the pinned vocabulary from the documented mirror: 1,317,639 bytes, SHA-256 `ee2b2c90…` (it matches the manifest).
  - **P3:** rebuilt the default 64-cell dataset and the 36/12/16 split; measured library size per class; recomputed the majority and library-size baselines with the carried `metrics.py`, plus a permutation reference for the test AUROC of total counts.
  - **P4:** ran `load_byod_dataset` and `validate_dataset` on 7 constructed inputs, and checked the cell's upload branch outside Colab.
  - Scripts and results are in `scgpt_single_cell_colab_Review_Probes.zip`.
- **Not executed here:** the encoder. The Hub is unreachable from this container and the locked torch is the multi-GB CUDA build. Model numbers come from the recorded runs.
- **Learner observation:** none.

## 2. Separate judgments

- **Technical correctness: very good.**
  - The strict-loaded re-implementation is correct, and the vocabulary is digest-checked from two sources.
  - Unknown gene symbols are dropped and reported, never mapped to id 0, which would be A1BG.
  - Baselines are fitted on training data only. The validation split is used for monitoring only. Reload parity is exact.
  - P3 reproduced the library-size baseline exactly (accuracy 0.625, macro-F1 0.619, AUROC 0.7188).
  - The one defect is the installation pattern, which forces a restart (SCG-M1).
- **Promise fulfilment.**
  - Every promised stage runs.
  - The adaptation cannot improve on the zero-training baseline (1.0), and the notebook says so explicitly. That is honest, though it leaves the fine-tune with nothing to demonstrate (SCG-S1).
  - BYOD falls short of "the same … cells" outside Colab, and for new-data inference (SCG-m2).
- **Learner experience.**
  - Strong expository prose: the vocabulary trap, binning ties, the embedding geometry, the decoder negative, and the advice to read the library-size baseline first.
  - The active-learning layer is missing (SCG-M2), and template braces leak into the opening cells (SCG-m1).
- **Spec conformance.**
  - Unresolved MUSTs: RUN1, RUN10, ENV6, REL2 (SCG-M1); SRC3 (SCG-m1).
  - SHOULD deviations: GDL1–GDL14, UX8 (SCG-M2); DAT16, UX10, INF2 (SCG-m2).
  - The notebook declares spec 2.0.

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass Run all | cell 3: in-kernel `pip install` plus stale-module guard | Kaggle: 14/14 "after one expected interpreter restart" | Section 1 prose presents the restart as designed | **Not met** (SCG-M1) |
| Verified snapshot plus second-source vocabulary | cell 11 | 4/4 verified; mirror used on Dataverse 504 (P3 fetched the same bytes) | clear | Met |
| 64 cells, validation, stratified split | cell 13 | 64; 348 detected and 348 encodable genes; 36/12/16 (P3 identical) | "Look for" note | Met |
| Tokenisation, scale invariance, refusals | cell 15 | four rejections and one truncation shown | clear | Met |
| Embeddings: reproducible, gene-order invariant | cell 17 | exact assertions; centred cosine geometry | "representation, not prediction" | Met |
| Masked-expression probe, honest negative | cell 19 | Pearson 0.0045; no lineage conditioning | explained | Met |
| Three baselines, training-only fits | cell 21 | centroid 1.0; majority 0.5; library size 0.625 / AUROC 0.72 | "sits at chance by construction" (P3: AUROC p ≈ 0.16 under permutation, so consistent with chance) | Met (SCG-S2) |
| Bounded fine-tune; held-out evaluation | cells 23, 25 | 1,579,010 params in 6.95 s; test 1.0 / 1.0 / 1.0 | "if already perfect, fine-tuning has nothing to add" | Met (SCG-S1) |
| New-cell inference, export, reload parity | cell 27 | 6/6; parity 0.0 | not calibrated; argmax rule | Met (BYOD: SCG-m2) |
| Provenance | cell 29 | `result.json` | — | Met |
| BYOD through the same cells | cell 13 | Colab upload only; P4: 2 accepted, 5 refused with named rules | contract stated | Partly met (SCG-m2) |
| Guided learning layer | — | absent apart from "Look for" notes | — | **Not met** (SCG-M2) |

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 31 cells | The prose is accurate and unusually candid: the decoder negative, the "nothing to add" warning and the vocabulary trap. There are no predictions, checkpoints, glossary or roadmap (SCG-M2). The opening cell and the Prerequisites show `{{id, counts, label}}` and `{{gene symbol: count}}` with doubled braces (SCG-m1). |
| **Clean default** | Documented (Kaggle T4, reviewed blob) + direct (P3: data and baselines) | 14/14 after one restart (SCG-M1). P3 reproduced the split and both trivial baselines exactly. |
| **Active learning** | Source | "Optional experiments" set `TRAINABLE_LAYERS` to 0 or 12. `adapt` copies the verified encoder each time, so a re-run compares cleanly, but there is no prediction or explanation scaffold and no side-by-side output (SCG-M2). |
| **Reuse and recovery** | Direct (P4) + source | See the BYOD results below. |

**BYOD results (P4).**

Accepted:

- a genes-as-columns CSV;
- a JSONL file.

Refused, each with a named rule:

- Ensembl-id columns: "0 gene(s) in the scGPT vocabulary … are these human gene symbols?";
- a single class;
- a ragged CSV row: "line 3 has 43 fields, header has 42";
- duplicate ids;
- a `.txt` file.

Two problems are inferred from source:

- The cell's upload branch imports `google.colab` unconditionally and has no path field, so on Kaggle or Jupyter `USE_BYOD = True` fails with `ModuleNotFoundError`.
- A cancelled upload raises a bare `StopIteration`.

## 5. Findings

### Major

#### SCG-M1 — `Run all` needs a manual restart after the install cell, and the qualification record counts the restarted run

- **Cell/section:** cell 3 and the Section 1 prose; the generator's install block; `docs/release-verification.md` and `STATUS.md`.
- **Observed issue:**
  - The cell runs `pip install` for four pins into the running kernel (`torch` 2.14.0, `numpy` 2.5.3, …).
  - It raises `Restart the runtime, then rerun from the top.` whenever a loaded distribution changed.
  - On stock Colab and Kaggle images, which preload older `numpy` and `torch`, this fires on the first pass.
  - The qualification row records "14/14 code cells after one expected restart" as PASS.
- **Consequence:** a learner who selects **Run all** meets an error in the first code cell. RUN1, RUN10 and ENV6 forbid this.
- **Evidence:** documented in the Kaggle qualification row ("one expected interpreter restart after dependency installation"); source.
- **Recommended correction:**
  - Adopt the fleet's isolated **uv** environment pattern. Install nothing into the kernel. Use a pinned `uv` wheel checked by size and SHA-256, `uv venv --managed-python --python 3.12.12`, and a hash-locked `requirements.txt` with `--require-hashes --only-binary :all:`, and run every stage in a subprocess.
  - Reference implementations on `main`: the bioclip2 capstone, and the four standalone notebooks built on 2026-10-04/05 (MediaPipe, NAFNet, RAFT-Stereo, Swin2SR x4).
  - Re-qualify with a one-pass hosted Run all, and correct the release record.
- **Acceptance check:**
  - A fresh Colab or Kaggle runtime completes every code cell in one **Run all** with no restart.
  - The run is recorded with the blob and `restarted: false`.
  - `grep -n "Restart the runtime" tutorials/scgpt_single_cell_colab.ipynb` returns nothing.
- **Spec:** RUN1, RUN10, ENV6, REL2.

#### SCG-M2 — Declared `GUIDED`, but the active-learning and orientation layer is absent

- **Cell/section:** opening cells 0–1, every section boundary, the carried cells 5, 7 and 9, the end of the notebook. Generator: `tools/notebook_template.py`.
- **Observed issue:**
  - **Orientation is missing:** there is no intended-learner statement, no **How to use this notebook**, no roadmap and no Input → Model → Output table.
  - **No glossary**, although the notebook relies on terms such as `<cls>` token, quantile binning, vocabulary, nearest-centroid, macro-F1, AUROC and adapter.
  - **No active-learning prompts:** no prediction before the baselines, the fine-tune or the decoder probe, and no checkpoint with a sample answer. The "Optional experiments" line is not a Predict → Change → Run → Observe → Explain activity, and it produces no side-by-side output.
  - **No troubleshooting section:** nothing on a Dataverse outage (which happened in the qualification run), digest mismatch, the restart, or BYOD errors.
  - **No conclusion template.**
  - **No infrastructure labelling:** the 1,680 carried lines in cells 5, 7 and 9 are not labelled **Infrastructure** or collapsed (`cellView` is absent on every code cell).
  - What is present, and good: "Look for" notes in Sections 4–6 and strong interpretation prose.
- **Consequence:** a self-paced learner reads excellent explanations, but is never asked to commit to an expectation or check their understanding. They also have to scroll past three carried modules before any learning activity.
- **Evidence:** source inspection; no `cellView` metadata on any cell.
- **Recommended correction:**
  - Add the GDL layer in the template, following NOTEBOOK_SPEC 2.2: audience, how-to-use, roadmap, task contract, glossary, predictions before Sections 7, 8 and 10, and collapsed checkpoints.
  - Turn "Optional experiments" into a structured activity on `TRAINABLE_LAYERS` that prints default and changed results side by side.
  - Add troubleshooting (including the Dataverse fallback) and a conclusion scaffold.
  - Title the install and carrier cells `# @title Infrastructure: …` with `cellView: form`.
- **Acceptance check:** each of GDL1–GDL14 maps to a named cell, and the install and carrier cells carry `cellView: form` with an Infrastructure title.
- **Spec:** GDL1–GDL14, UX8.

### Minor

#### SCG-m1 — Literal doubled braces from the template leak into the learner text

- **Cell/section:** cell 0 (the **Bring Your Own Data** paragraph: "a JSON array or a JSONL file of `{{id, counts, label}}` records") and cell 1 (the *Data contract*: "records are `{{id, counts, label}}`, with `counts` a `{{gene symbol: count}}` mapping"). The source is `tools/notebook_template.py`, lines 73 and 122.
- **Observed issue:** these template strings escape `{` as `{{` for `str.format`. The two markdown cells are emitted without formatting, so the learner sees doubled braces in the schema statement itself.
- **Consequence:** the first statement of the data contract shows a schema that does not match the JSON the learner must supply.
- **Evidence:** `grep -c "{{" tutorials/scgpt_single_cell_colab.ipynb` gives 2 (the cell-0 and cell-1 lines).
- **Recommended correction:** format those strings, or write single braces. Add a validator check that rejects `{{` and `}}` in markdown cells.
- **Acceptance check:** no markdown cell contains `{{` or `}}`.
- **Spec:** SRC3, DAT12.

#### SCG-m2 — BYOD works only in Colab, its upload is unguarded, and its "new data" are test-split cells

- **Cell/section:** cell 13 (`if USE_BYOD: from google.colab import files … next(iter(uploaded.items()))`) and cell 27 (`new_records = test_records[:6]` in BYOD mode).
- **Observed issue:**
  - **No path option.** The branch has no path field. On Kaggle or any Jupyter runtime, `USE_BYOD = True` fails with `ModuleNotFoundError: No module named 'google'`. The opening cell names "Google Colab or Jupyter" as supported.
  - **Unguarded upload.** A cancelled dialog raises a bare `StopIteration`, and a multi-file upload silently uses the first file.
  - **Reused test cells.** In BYOD mode, "inference on new cells" classifies the first six **test-split** cells, which were already scored in Section 10, and labels them "first six BYOD test-split cells". The sample path uses freshly generated cells. The lineage probe is also skipped in BYOD mode, without saying why.
- **Consequence:**
  - BYOD is unusable outside Colab.
  - A BYOD user gets no inference on cells that played no part in evaluation.
- **Evidence:**
  - Direct (P4): the import outside Colab fails as `ModuleNotFoundError: No module named 'google'`, and the cell has no `BYOD_PATH` field.
  - Source: the upload branch and the `test_records[:6]` reuse.
- **Recommended correction:**
  - Add `BYOD_PATH`, read before any dialog.
  - Require exactly one uploaded file, with a message.
  - In BYOD mode, hold back a small inference set from the user's data before the split, or accept a second unlabelled file for inference.
- **Acceptance check:**
  - A BYOD run on Kaggle with `BYOD_PATH` set completes.
  - A cancelled or multi-file upload stops with a message.
  - BYOD inference runs on cells outside train, validation and test.
- **Spec:** DAT16, DAT19, INF2, UX10.

### Suggestions

- **SCG-S1:** The default task is saturated: the zero-training nearest-centroid rule already scores 1.0, so the fine-tune cannot demonstrate a gain. The notebook says so honestly. A harder default would give learners a measurable adaptation effect. Options include overlapping programmes, noisier cells, a third class, or the real PBMC subset already used at build time.
- **SCG-S2:** "The library-size baseline sits at chance by construction" holds statistically. P3 found library sizes of 19,986–20,012 after rounding; the test AUROC of 0.72 has a two-sided permutation p ≈ 0.16 on n = 16. A learner who sees 0.625 accuracy and 0.72 AUROC may still read it as signal, so add one sentence noting that rounding leaves ±12 counts and that 16 cells cannot distinguish 0.72 from 0.5.
- **SCG-S3:** Commit the Kaggle executed notebooks and `run_summary.json` beside the release record, as the SAM2 repository does. The current evidence is table rows only.
- **SCG-S4:** Declare `notebook_spec` 2.2 once the restart and guided-layer fixes land.

## 6. Readiness

**Needs revision.**

- **Open Majors:** SCG-M1 and SCG-M2.
- **Gates remaining after the fixes:**
  - one-pass hosted Run all of the regenerated blob;
  - the REL12 BYOD journey with a path-based input on Kaggle or Jupyter and the upload dialog on Colab: one compatible file and one incompatible file, recorded in `docs/release-verification.md`.

## 7. Verified versus inferred

- **Verified by direct execution (torch-free):**
  - the vocabulary digest from the documented mirror;
  - the default dataset, its split and the per-class library sizes;
  - the majority and library-size baselines, which match the record exactly;
  - the permutation reference for the library-size AUROC;
  - the BYOD loader's acceptance and refusal matrix;
  - the upload import failing outside Colab.
- **Verified from documented evidence:**
  - the restart;
  - every encoder number (Kaggle T4 row);
  - the Dataverse 504 fallback.
- **Inferred from source:**
  - the bare `StopIteration` on a cancelled upload;
  - the BYOD "new data" reuse;
  - the clean restart-from-base behaviour of `adapt`.
- **Most likely to be wrong:** SCG-M2's severity. The notebook's prose is far richer than SAM2's, and a maintainer could treat the missing predictions and checkpoints as a Minor gap. It is rated Major for consistency with the other generation-2 reviews, because GUIDED is declared and GDL1–GDL4, GDL6, GDL7 and GDL9–GDL14 are all unmet.
