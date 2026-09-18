# Release verification

`tutorials/scgpt_single_cell_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the inline
  `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the upstream
  scGPT commit the re-implementation follows is the one allowed second hash);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `ScGPTPipeline.from_pretrained(weights_dir=...)`, `validate_dataset` with the vocabulary, `split_dataset`,
  `write_dataset_csv`, `bin_expression` with its scale-invariance assertion, `validate_inputs` with the truncation
  probe, `pipe.embed` with its same-batch and gene-order assertions, `pipe.predict_masked`,
  `pipe.predict_masked_genes`, `pipe.nearest_centroid_evaluate`, `majority_baseline`, `library_size_baseline`,
  `pipe.adapt` with its explicit hyperparameters, `pipe.evaluate` on both the validation and the test split with
  the centroid and majority deltas, `pipe.classify`, `pipe.save_artifact`, `ScGPTPipeline.from_artifact` and the
  reload-parity assertion), the six expected `outputs/` paths, the learner-facing statements (scores are not
  calibrated probabilities, validation is monitoring only, embeddings are representations, no positional encoding,
  the Dataverse vocabulary, the decoder is not a usable imputer, split by donor/plate/batch) and the gated-off BYOD
  default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary
  path, a mutable `revision='main'`, direct `huggingface_hub` / `safetensors` / `urllib` use or calls into
  `pipe.model` / `pipe._batch` **outside the carried module cells**, `trust_remote_code=True`, `pickle.load`,
  `torch.load(` without `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected backends, toy vocabularies and temporary
manifests, no weights and no model library; the real vocabulary is checked when present). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/scgpt/` (the standalone path writes the manifest itself and stages every listed file — three
   from the Hub, the vocabulary from Dataverse — so the directory may not be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `VAL_FRACTION = 0.2`, `TEST_FRACTION = 0.25`, `SEED = 42`, `EPOCHS = 4`,
   `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 8`, `TRAINABLE_LAYERS = 1`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `safetensors==0.8.0`, `numpy==2.5.3`, `huggingface-hub==1.32.0` — and
   that `transformers` is **not** imported anywhere;
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `ScGPTPipeline`, `build_model`, `verify_snapshot`,
     `stage_missing_files`, `load_gene_vocabulary`, `bin_expression`, `validate_inputs`, `validate_dataset`,
     `split_dataset`, `generate_sample_dataset`, `select_sample_genes`, `write_dataset_csv`, `load_byod_dataset`,
     `classification_metrics`, `majority_baseline`, `library_size_baseline`) with no import of the repository
     package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(..., allow_download=True)`
     reporting the 4 entries fetched — `config.json`, `model.safetensors`, `README.md` from `tdc/scGPT` at the
     immutable revision and `vocab.json` from Dataverse datafile 10809431 — and `verify_snapshot` reporting 4
     verified files before the model loads;
   - the load report showing the re-implemented encoder loaded strictly (no warnings) on the chosen device;
   - the two marker programmes printed, then the dataset manifest with 64 records, classes `['b-like', 't-like']`,
     32 each, 348 detected and 348 encodable genes per cell, library sizes within 0.1 % of 20,000, the ceilings and
     the digest `ce627233…`, and the splits 36 / 12 / 16;
   - the encoding cell showing `<cls>` first, bins in 1..50, the scale-invariance assertion passing, four refusals
     (empty, two genes, negative count, no known symbols) and one truncation (1,585 → 1,535 kept, 50 truncated);
   - `pipe.embed` reporting 512-dimensional vectors, the same batch twice identical, a gene-order difference of
     exactly 0.0, a single-versus-batch difference below `1e-4`, and writing `outputs/scgpt_single_cell_embeddings.csv`;
   - the masked-expression probe reporting weak correlation and the lineage probe reporting similar predicted bins
     whichever lineage's markers are masked (a confident *positive* result here would be a change to investigate);
   - the zero-training nearest-centroid baseline on the test split (on the sample: accuracy 1.0, n = 16), the
     majority baseline (0.5 on the balanced split) and the library-size baseline near chance;
   - `pipe.adapt` reporting 1,579,010 trainable of 50,805,251 parameters and a four-epoch history with per-epoch
     validation metrics;
   - `pipe.evaluate` reporting validation and test accuracy, macro-F1, AUROC and per-class rows, and writing
     `outputs/scgpt_single_cell_evaluation_report.json` with the three baselines and the deltas;
   - `pipe.classify` on six freshly generated cells writing `outputs/scgpt_single_cell_predictions.csv` with
     per-class scores;
   - `pipe.save_artifact` writing `outputs/scgpt_single_cell_adapter/{adapter.safetensors,manifest.json}` (14
     tensors, about 6 MB), and `ScGPTPipeline.from_artifact` reloading it with identical labels and a maximum
     absolute score difference below `1e-5` (the cell asserts both);
   - `outputs/scgpt_single_cell_result.json` written with `NOTEBOOK_SOURCE`, the model identity, revision and
     licence, the vocabulary's source and digest, `remote_code_executed: false`, the dataset manifest, the
     evaluation report with the masked-expression probe, the predictions, the artifact manifest and the runtime
     versions;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, device), the model identifier and
   immutable revision, whether the model cache and the weights directory were clean, outcome, produced outputs, the
   observed metrics (as observations, not a benchmark) and any warning or applicable `SHOULD` deviation in the
   tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `scgpt_single_cell_colab.ipynb` | __LOCAL_ROW__ | | | |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/scgpt_single_cell_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/scgpt_single_cell_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| __LOCAL_EXEC__ | | | | | |

## Current status

The notebook source is complete and passes all static checks, including the generator parity checks (`--check` OK).
The repository stays at **Candidate** until a Colab or fresh-container run of the exact release revision is recorded
above.
