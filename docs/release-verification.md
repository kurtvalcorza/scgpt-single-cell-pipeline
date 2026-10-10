# Release verification

`tutorials/scgpt_single_cell_colab.ipynb` (`E2E`, `GUIDED`, **standalone** carrier, generator /3) is a **release
candidate** until the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON
validation, code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 2.2. This file is the durable release-gate
record. The executions recorded below ran the previous (/2, in-kernel install) notebook; they do not carry over to the
regenerated notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or execution
  counts; no unresolved placeholder markers (including doubled template braces in markdown); every code cell is preceded
  by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the spec version (`2.2`) and the
  standalone carrier; `metadata.dimer` declares the profile, mode `GUIDED`, `standalone: true` and `generated_from`
  (repository, generating commit, package paths and SHA-256, carried-file digests, generator `build_notebook.py/3.0`);
- the standalone carrier and isolated environment (ST1–ST8, PAR1–PAR3, RUN1, RUN10, ENV6): one carrier cell whose
  carried files equal the repository files (`src/scgpt_single_cell_pipeline/{__init__,metrics,pipeline,samples}.py`, the
  stage runner, `tutorials/requirements-colab.lock.txt`, the 4-file snapshot manifest, `LICENSE`) with matching digests;
  the lock pins every `pyproject.toml` runtime pin with hashes; a pinned `uv` builds a managed-CPython environment with
  `--require-hashes`, reused per lock digest; no in-kernel install and no restart instruction; the four Infrastructure
  cells are titled and collapsed; every learner cell runs a stage; the notebook byte-identical to
  `tools/build_notebook.py` output;
- the stage-runner markers (staging and verification, the sample and BYOD loaders, the true-minimum split check, the
  BYOD hold-back, binning with a scale-invariance verdict, embeddings with order and batch verdicts, the masked probes,
  the three baselines, the bounded fine-tune and adapter export, the fresh-process evaluation with deltas, the
  fresh-process reload parity against the adapting process, the provenance record), the form-parameter defaults, no
  quality `assert`, and the forbidden patterns (credential-in-URL, any clone or repository import on the primary path, a
  mutable revision, model-library use in the notebook's own cells, `trust_remote_code=True`, `pickle.load`, `torch.load(`,
  `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite (no weights and
no model library), including `tests/test_notebook_review_fixes.py`, which execs the notebook's own cells with stand-ins and
runs the model-free `data` and `tokens` stages with the committed vocabulary. These are source and unit checks, **not**
execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU or T4 runtime; any kernel Python — the stages run on the isolated environment's CPython 3.12.12 | The runtime the tutorial is written for; a clean one-pass **Run all** here is promotion evidence |
| Kaggle kernel or equivalent fresh container | Fresh CPU or GPU Linux x86_64 container; the committed notebook executed verbatim with **no repository checkout** | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor, stand-ins | Builder pre-flight; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a fresh runtime with **no repository checkout**, an empty Hugging Face cache and
   no `weights/scgpt/` directory;
3. run it with one **Run all** and no restart, form parameters at their defaults (`USE_BYOD = False`, `VAL_FRACTION = 0.2`,
   `TEST_FRACTION = 0.25`, `SEED = 42`, `INFERENCE_CELLS = 6`, `EPOCHS = 4`, `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 8`,
   `TRAINABLE_LAYERS = 1`, `RUN_ACTIVITY = False`), then re-run the export cell (Section 11) once;
4. verify: the carried-file verification and the isolated environment's versions (CPython 3.12.12, `torch` 2.14.0,
   `safetensors` 0.8.0, `numpy` 2.5.3) and that `transformers` is not installed; the 4-file snapshot staged and verified,
   with the vocabulary source printed (Dataverse, or the mirror); 64 cells, 348 detected and encodable genes, library
   sizes 19,986–20,012, digest `ce627233…`, splits 36 / 12 / 16 and six new cells; `scale_invariance: 'PASS'`, four
   refusals and one truncation (1,585 → 1,535); three embedding `'PASS'` verdicts; the masked probe's weak correlation
   and the lineage rows; the baselines (nearest centroid 1.0, majority 0.5 / 0.3333, library size 0.625 / 0.619 / AUROC
   0.7188 on the sample); `pipe.adapt` reporting 1,579,010 trainable parameters and a four-epoch history; the
   evaluation computed by the adapter in a fresh process (`same_as_adapting_process: True`, `interpretation:
   saturated…`); six predictions and `reload_parity: 'PASS'`; the result JSON with source, model identity, licence,
   vocabulary source and runtime;
5. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, device), the model identifier and
   immutable revision, whether the model cache and the weights directory were clean, `restarted: false`, outcome,
   produced outputs and the observed metrics (as observations, not a benchmark) in the tables below;
6. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `scgpt_single_cell_colab.ipynb` | `d192ed2` / `6b2676db` | 2026-09-18 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `scgpt_single_cell_colab.ipynb` | `0f66318` / `31e26691` | 2026-09-18 | Kaggle fresh GPU container (`gcr.io/kaggle-gpu-images/python@sha256:37c64f7…`, CPython 3.12.13, Tesla T4), strict serial executor v3 | **Completed with a restart (`restarted: true`) — not one-pass evidence.** Exact commit and fetched Git blob verified; empty Hub cache and no pre-staged snapshot; one interpreter restart after the in-kernel dependency installation, then all 14 code cells completed. A restart-dependent run does not meet RUN1/RUN10 (review SCG-M1), so it does not qualify that blob, and it does not carry over to the regenerated notebook. |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/scgpt_single_cell_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/scgpt_single_cell_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-18 | `d192ed2` / `6b2676db` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cpu`) | Default sample path (validate → split → binning + reject probes → embed + invariance checks → masked-expression probes → centroid/majority/library baselines → adapt → evaluate → classify → export → reload); weights and vocabulary pre-staged, so `stage_missing_files` fetched 0 of 4 entries and `verify_snapshot` verified all 4 | 16.1 s | **PASSED** — 14/14 code cells; nearest-centroid 1.0 with no training; adaptation of 1,579,010 params in 7.6 s; test accuracy/macro-F1/AUROC 1.0 (n=16) against majority 0.5/0.3333 and library-size 0.625/0.619; masked probe Pearson 0.0149 and no lineage conditioning (the recorded negative); 6/6 new cells; reload parity 0.0. Pre-flight; hosted clean-runtime run still required |
| 2026-09-18 | `0f66318` / `31e26691` | Kaggle fresh GPU container, CPython 3.12.13, Tesla T4, `torch 2.14.0+cu130`, CUDA 13.0; exact fetched blob verified; empty Hub cache | Default standalone path from an empty snapshot (install → expected restart → fetch and digest-verify → validate → split → binning + reject probes → embed + invariance checks → masked-expression probes → centroid/majority/library baselines → adapt → evaluate → classify → export → reload). The Harvard Dataverse request returned HTTP 504, so the documented immutable exact-byte mirror supplied `vocab.json`; its declared 1,317,639 bytes and SHA-256 were verified with the other three files. | 193.1 s | **PASSED** — 14/14 code cells after one expected restart; all 4 snapshot entries verified (205 MB staged); nearest-centroid accuracy 1.0; adaptation of 1,579,010 params in 6.95 s; test accuracy/macro-F1/AUROC 1.0 (n=16) against majority 0.5/0.3333 and library-size 0.625/0.619/0.7188; masked probe Pearson 0.0045 and the same documented weak lineage conditioning; 6/6 new cells; seven preserved artifacts with SHA-256 digests; adapter reload parity 0.0. Restart-dependent (`restarted: true`): a functional record of the previous notebook, not one-pass clean-runtime evidence. |

## Current status

**Candidate — verification pending.** The regenerated notebook (generator /3, isolated environment, no in-kernel install
and no restart) passes all static checks, including generator parity (`--check` OK). No hosted one-pass **Run all** of
its blob is recorded yet. The 2026-09-18 Kaggle Tesla T4 run of the previous blob `0f66318` / `31e26691` completed all
14 code cells only after an interpreter restart (`restarted: true`), so it is a functional record, not clean-runtime
qualification (RUN1/RUN10, review SCG-M1). The repository stays **Candidate** until a hosted one-pass Run all of the
exact current blob is recorded here with `restarted: false` and the maintainer explicitly approves promotion.
