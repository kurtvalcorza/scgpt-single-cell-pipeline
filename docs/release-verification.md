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
| 2026-10-10 (11:59:35 UTC start) | `8d35011f8a1aec7355f0effa425d0fa900749ed2` / `07755aa2ddf8c6a1fee7ea38964276b59b41f99d` (carried revision `90a8bcd103bb`, generator `build_notebook.py/3.0`) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-scgpt-8d35011-a079`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, stage subprocesses in an isolated CPython 3.12.12 environment (43 locked packages, setup 62 s, not reused), `torch 2.14.0`, `cuda:0`, float32 | Default synthetic marker-programme path, `RUN_ACTIVITY` off | 143.7 s | **PASSED** — one pass, `restarted: false`, 0 errors; 14/14 code cells in order; 4 snapshot files digest-verified at `acf749f35bf5` plus the Dataverse vocabulary; 64 cells (32/32); ×37 scale invariance, gene-order invariance and batch checks PASS; masked-expression probe mean Pearson 0.0045; baselines: centroid 1.0 / 1.0 / 1.0, majority 0.5 / 0.3333, library-size 0.625 / 0.619 / 0.7188; adaptation 4 epochs, loss 0.481 → 0.035, 9.1 s; test (n = 16) accuracy / macro-F1 / AUROC 1.0, delta vs centroid 0.0, interpretation `saturated`; 6/6 new cells; reload parity PASS (labels equal, max score difference 0.0). Evidence in `docs/execution-evidence/2026-10-10-8d35011/`: executed notebook SHA-256 `d734036f86cebe0ee6d3cd1c22d7e1d7a3c1bede304e3f718ef39afd58500aad`, `run_summary.json` `05954b334f8a40302268faf039e34afc48ca66f95e4c663ad47bd71af5f1abd5`, `exec.log` `3b667bd1c571c86b367eaf79ba28903c6cc370b0a09be024cf5c62730bab0106`. Not exercised: the optional activity, BYOD, a browser Run all |
| 2026-10-17 (12:14:10 UTC start) | `5d95ece434d3f54079962632b3958c0c07e3c2fe` / `90b4168492a9d26785861e84d69e3323ebc995be` (carried revision `3444392904430c948ac2ae6f62a4acbebcc1f290`, generator `build_notebook.py/3.0`, NOTEBOOK_SPEC 2.5) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM on Colab account 2 (session `suite-scgpt-5d95ece-a079`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, stage subprocesses in an isolated CPython 3.12.12 environment (43 locked packages, setup 56 s, not reused), `torch 2.14.0`, `cuda:0`, float32 | Default synthetic marker-programme path, `RUN_ACTIVITY` off | 125.2 s | **PASSED** (REL13: hosted Colab T4, this blob) — 0 errors; 14/14 code cells in order; source identity with the committed blob OK (44/44 cells); 5 snapshot files digest-verified at `acf749f35bf5` plus the Dataverse vocabulary; 64 cells (32/32); ×37 scale invariance, gene-order invariance and batch checks PASS; masked-expression probe (8 cells, default seed 42) mean Pearson MLM 0.0045 / generative 0.0393 / MVC head 0.0926 (MAE 12.611 / 12.534 / 12.757 bins), gene prior -0.0213, constant MAE 12.609; over seeds 42-46 MLM 0.0345 [0.0018, 0.0978], generative 0.0274 [-0.0238, 0.0658], MVC 0.1354 [0.0917, 0.2032], gene prior 0.0135 [-0.0213, 0.0692]; baselines: centroid 1.0 / 1.0 / 1.0, majority 0.5 / 0.3333, library-size 0.625 / 0.619 / 0.7188; adaptation 4 epochs, loss 0.481 → 0.035, 8.6 s; test (n = 16) accuracy / macro-F1 / AUROC 1.0, delta vs centroid 0.0, interpretation `saturated`; 6/6 new cells; reload parity PASS (labels equal, max score difference 0.0). Evidence in `docs/execution-evidence/2026-10-17-5d95ece/`: executed notebook SHA-256 `d18f93e8fd9133744a1933ecfbdd63fdf0197b39ad329f7ef766ad9ee4ad6270`, `run_summary.json` `7d7b8f27a4e6b83919e9074fcf3d056f60c9652468e6f78bb63806b1e789fe33`, `exec.log` `dfcec6f265504fc7fea560ccd3bc8c9ca6fb54e92c3696c46475f599a8c1b8b3`. Not exercised: the optional activity, BYOD, a browser Run all |
| 2026-10-17 (15:58:44 UTC start) | `711a9e6b526c575a4aefcbe7b5517207051a9558` / `be0b6b878568689b117368775950007166af43c8` (carried revision `d82bdd7b2b457f84fd288a066440149579f1dd6c`, generator `build_notebook.py/3.0`, NOTEBOOK_SPEC 2.5) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM on Colab account 2 (session `suite-scgpt-711a9e6-a079`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, stage subprocesses in an isolated CPython 3.12.12 environment (43 locked packages, setup 71 s, not reused), `torch 2.14.0`, `cuda:0`, float32 | Default synthetic marker-programme path, `RUN_ACTIVITY` off | 138.9 s | **PASSED** (REL13: hosted re-run after the requote, this blob) — 0 errors; 14/14 code cells in order; source identity with the committed blob OK (44/44 cells, param toggles only). Every printed figure equals the 12:14:10 UTC run of `5d95ece` / `90b41684` above (normalised line diff of the two output notebooks: only the carried revision, environment setup 71 s vs 56 s and adaptation time 10.2 s vs 8.6 s differ): probe mean Pearson MLM 0.0045 / generative 0.0393 / MVC head 0.0926 (MAE 12.611 / 12.534 / 12.757), gene prior -0.0213, constant MAE 12.609, five-seed means 0.0345 / 0.0274 / 0.1354 / 0.0135 with the same ranges; baselines centroid 1.0 / 1.0 / 1.0, majority 0.5 / 0.3333, library-size 0.625 / 0.619 / 0.7188; adaptation loss 0.481 → 0.035; test (n = 16) accuracy / macro-F1 / AUROC 1.0, `saturated`; 6/6 new cells; reload parity PASS (max score difference 0.0). Evidence in `docs/execution-evidence/2026-10-17/`: `scgpt_single_cell_colab_711a9e6_colab-cli-t4_output.ipynb` SHA-256 `c42b598eedbf1bf290520a13f9fa2856ae1838aefcd1212d941e0ed68d72eeb5`, `..._run_summary.json` `8e90d650563c7cd38a5237fac95f74bb079f21faace7950884f9110258ad5628`, `..._exec.log` `213353c2a657b566d1779dc040d85cae08b7f5456922443b9346576acf445203`. Not exercised: the optional activity, BYOD, a browser Run all |

## Local CPU run of the three-reading masked-expression probe — 2026-10-17 (not clean-runtime evidence)

Branch `fix/scgpt-tutorial-review-2026-10-17` (tutorial review row 80, finding T80-1) adds the generative masked-expression
call and the restored MVC head. Before any hosted run of the new blob, its probe stage was run locally: Windows, CPython 3.12,
`torch 2.14.1+cpu`, float32, `CUDA_VISIBLE_DEVICES=-1`, the stage runner's `data`, `tokens`, `embed` and `probe` stages on the
default sample, all five manifest entries digest-verified. This is a source-level CPU check, not a notebook execution.

| Reading (8 validation cells, 15 % masked, same masks) | Mean Pearson (CPU) | Mean absolute error (bins) |
|---|---|---|
| Pipeline MLM call (masked value -1) | 0.0149 (the recorded Colab CLI T4 run at `8d35011`: 0.0045) | 12.601 |
| Generative call, expression decoder (`<cls>` value -2) | 0.0346 | 12.546 |
| Generative call, MVC head | 0.1055 | 12.712 |

The MLM call's 0.0149 is identical on CPU at the base revision `4875d9b`, which rules out this source change as the cause of
the T4/CPU gap; the gap is an observed cross-environment difference (hardware, torch 2.14.0 vs 2.14.1, kernels) whose cause
was not isolated; the lineage rows of the MLM call match the T4 record to three decimals (28.858 / 27.038 / 28.368 / 32.022). On the review's
32-cell synthetic comparison the new API reproduces the review's CPU numbers: 0.0867 (MLM), 0.0809 (generative, `<cls>` 0),
0.0648 (generative, `<cls>` -2), 0.1509 (MVC head, `<cls>` -2). The notebook quotes these numbers as local CPU results until a
hosted run of the new blob is recorded.

Round 2 (Codex peer review) added, on the same masks, a per-gene prior (mean training-split bin of each masked gene), a
constant baseline (mean bin of the cell's unmasked genes; MAE only) and a shuffled-context control (the masked genes placed
into the next cell's context), plus 5 mask seeds (42-46). Default-seed numbers above are unchanged. Notebook sample, CPU,
default seed: gene prior 0.0025 (MAE 12.744), constant MAE 12.611, MLM / generative / MVC with another cell's context
-0.0142 / -0.0123 / 0.0117. Over 5 seeds, mean [min, max] of the mean Pearson: MLM 0.040 [0.007, 0.101], generative
0.031 [-0.030, 0.071], MVC 0.136 [0.099, 0.205], gene prior 0.020 [-0.030, 0.065]. Real `pbmc3k` (the review's 32 cells,
this API, prior from 200 other pbmc3k cells): MLM 0.314, generative 0.3712, MVC 0.551,
gene prior 0.5756, MVC with another cell's context 0.4782; MAE MLM 11.814, generative 12.0, constant 11.815.
Full records: `docs/reviews/2026-10-17-tutorial-review/t80_r2_pbmc.json` and `run3_probe.log`. All descriptive; CPU only.

## Hosted Colab T4 run of blob `90b41684` versus the local CPU probe — 2026-10-17

REL13: hosted Colab T4 figures from the output notebook of the 2026-10-17 run above (Colab account 2, commit `5d95ece`,
blob `90b41684`), compared with the local CPU figures of the previous section. Same 8 validation cells, same masks.

| Reading (default seed 42) | Mean Pearson, T4 | Mean Pearson, CPU | MAE (bins), T4 | MAE (bins), CPU |
|---|---|---|---|---|
| Pipeline MLM call (masked value -1) | 0.0045 | 0.0149 | 12.611 | 12.601 |
| Generative call, expression decoder | 0.0393 | 0.0346 | 12.534 | 12.546 |
| Generative call, MVC head | 0.0926 | 0.1055 | 12.757 | 12.712 |
| Gene prior | -0.0213 | 0.0025 | 12.776 | 12.744 |
| Constant (MAE only) | — | — | 12.609 | 12.611 |
| MLM / generative / MVC, another cell's context | -0.0272 / -0.0107 / -0.0053 | -0.0142 / -0.0123 / 0.0117 | 12.657 / 12.588 / 13.032 | not recorded |

| Over 5 mask seeds 42-46, mean [min, max] | T4 | CPU |
|---|---|---|
| MLM | 0.0345 [0.0018, 0.0978] | 0.040 [0.007, 0.101] |
| Generative | 0.0274 [-0.0238, 0.0658] | 0.031 [-0.030, 0.071] |
| MVC head | 0.1354 [0.0917, 0.2032] | 0.136 [0.099, 0.205] |
| Gene prior | 0.0135 [-0.0213, 0.0692] | 0.020 [-0.030, 0.065] |

The T4 MLM value 0.0045 equals the 2026-10-10 T4 record at `8d35011`, and the T4 lineage rows of the MLM call
(28.858 / 27.038 / 28.368 / 32.022) equal the CPU section's values, so the MLM gap is the same cross-environment
difference recorded above. The generative and MVC default-seed values differ from CPU by 0.005 and 0.013, and every
T4 five-seed mean lies inside the CPU five-seed range (and the reverse); the ordering MVC > MLM, generative is the same
on both. All readings are descriptive; the heads are not ranked. The notebook's Section 7 prose was later requoted
from these T4 figures (REL13: this run); the CPU figures stay disclosed above.

REL13: hosted re-run after the requote. The 2026-10-17 run of the requoted blob `be0b6b87` (commit `711a9e6`,
Colab account 2) printed every T4 figure in the two tables above unchanged, including the per-cell, lineage and
shuffled-context rows; see Recorded executions.

## Current status

**Candidate — one-pass hosted run of the current blob recorded.** The current blob `be0b6b87` (commit `711a9e6`, the notebook with the Section 7 probe prose requoted) completed one pass with 0 errors on a fresh Colab Tesla T4 on 2026-10-17 (Colab account 2; Colab CLI 0.7.4 sequential execution, 14/14 code cells, 138.9 s, `cuda:0`). REL13: hosted re-run after the requote, this blob; every probe, baseline, test and reload figure equals the earlier 2026-10-17 run of blob `90b41684` (commit `5d95ece`): test accuracy / macro-F1 / AUROC 1.0 on 16 cells, tied with the zero-training centroid rule (`saturated`), 6/6 new cells, reload parity exact; masked-expression probe mean Pearson, default seed: MLM 0.0045, generative 0.0393, MVC head 0.0926; see Recorded executions. REL13: the paragraph that follows describes the earlier blob `07755aa2ddf8` and older, not re-run on the current blob. The repository stays **Candidate** until the maintainer explicitly approves promotion.

The current blob `9d521522` (code commit `d788479`) shows a refused BYOD input as exactly one `Refused: <file>: <reason>` line, with no traceback, and stops Run all at that cell (previously the stage printed the refusal and Run all carried on); genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20). Its hosted Colab T4 runs (the default path and a scripted refusal, `USE_BYOD=True;BYOD_PATH='/etc/hostname'`) are pending, and the hosted runs of earlier blobs, including `be0b6b87`, are history for it (REL14).

The regenerated notebook (generator /3, isolated environment, no in-kernel install
and no restart) passes all static checks, including generator parity (`--check` OK). Its blob `07755aa2ddf8` (commit `8d35011`) completed one pass with no restart (`restarted: false`) and 0 errors on a fresh Colab Tesla T4 on 2026-10-10 (Colab CLI 0.7.4 sequential execution, 14/14 code cells, 143.7 s, `cuda:0`; test accuracy / macro-F1 / AUROC 1.0 on 16 cells, tied with the zero-training centroid rule (`saturated`), 6/6 new cells, reload parity exact); see Recorded executions. The 2026-09-18 Kaggle Tesla T4 run of the previous blob `0f66318` / `31e26691` completed all
14 code cells only after an interpreter restart (`restarted: true`), so it is a functional record, not clean-runtime
qualification (RUN1/RUN10, review SCG-M1). The repository stays **Candidate** until the maintainer explicitly approves promotion on the strength of that `restarted: false` record.
