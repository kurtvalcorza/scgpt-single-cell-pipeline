"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.2 §4 standalone, §25.13 isolated environment).

The generator writes the infrastructure cells (runtime check, carrier, isolated install + stage runner, snapshot
staging) from repository files; this template holds the learner-facing prose, the guided layer and the learner cells.
Every learner cell calls ``run_stage(...)``: the carried ``tools/tutorial_stages.py`` runs one stage per process in an
isolated, hash-locked environment, so nothing is installed into the notebook kernel and no restart is needed.

The workflow: the pinned scGPT snapshot (weights, config and the Dataverse-sourced vocabulary) is digest-verified, a
synthetic marker-programme dataset built from real gene symbols is validated and split, cells are binned the way scGPT
expects, cell embeddings and the masked-expression objective are probed, a zero-training nearest-centroid baseline and
two trivial baselines are measured, a bounded AdamW fine-tuning runs, and the adapter is exported and rebuilt in fresh
processes for the held-out evaluation and the reload-parity check.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "scgpt-single-cell-pipeline"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/scgpt_single_cell_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-tdc%2FscGPT-ffcc4d?style=flat",
        "https://huggingface.co/tdc/scGPT",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-bowang--lab%2FscGPT-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/bowang-lab/scGPT",
    ),
    ("Paper", "https://img.shields.io/badge/Nat%20Methods-10.1038%2Fs41592--024--02201--0-b31b1b.svg", "https://doi.org/10.1038/s41592-024-02201-0"),
]

UV = {
    "version": "0.12.15",
    "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "bytes": 20081404,
    "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
}

TEMPLATE = {
    "package": "scgpt_single_cell_pipeline",
    "repo_name": REPO,
    "weights_key": "scgpt",
    "modules": ["__init__.py", "metrics.py", "pipeline.py", "samples.py"],
    "entry_module": "pipeline.py",
    "lock": "tutorials/requirements-colab.lock.txt",
    "managed_python": "3.12.12",
    "uv": UV,
    "disk_gib": {"weights": 0.25, "environment": 8.0},
    "runtime_modules": ["torch", "safetensors", "numpy"],
    "install_flags": ["--only-binary", ":all:"],
    "extra_hosts": "Harvard Dataverse (`dataverse.harvard.edu`) for the 1.3 MB gene vocabulary, which the Hub repository does not ship, with an immutable raw-GitHub mirror (`raw.githubusercontent.com`) of the same digest-pinned bytes used only when Dataverse is unavailable",
    "stem": "scgpt_single_cell",
    "notebook_name": "scgpt_single_cell_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "stage_runner": "tools/tutorial_stages.py",
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime builds an isolated Python environment from the carried "
        "hash-locked requirements (torch, safetensors, numpy, huggingface-hub — no transformers, no PyTDC) without touching "
        "the notebook kernel's own packages, stages and digest-verifies the pinned scGPT snapshot (4 files, ~205 MB: config, "
        "weights and README from the Hub, the gene vocabulary from its persistent Harvard Dataverse file id with an immutable "
        "exact-byte mirror fallback), rebuilds the encoder on plain torch modules and loads the weights strictly, generates a "
        "deterministic 64-cell dataset in code from two real PBMC marker programmes (no download), validates the cells and "
        "splits them into stratified train/validation/test sets, shows how a cell becomes (gene, bin) tokens, computes `<cls>` "
        "cell embeddings and checks they are reproducible and gene-order invariant, probes the masked-expression objective, "
        "measures a zero-training nearest-centroid baseline on the frozen embeddings plus majority-class and library-size "
        "baselines, runs a bounded AdamW fine-tuning of a classification head and the last encoder layer, exports the adapter, "
        "rebuilds it in a fresh process to evaluate accuracy, macro-F1 and AUROC on the held-out test split, classifies six "
        "freshly generated cells with a reload-parity check against the adapting process, and writes a provenance record. No "
        "repository clone, DIMER worker or service, credential, upload dialog, configuration edit or runtime restart is "
        "required (NOTEBOOK_SPEC 2.2 §5). After the one-time environment build, the model stages take about a minute on CPU."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and either set `BYOD_PATH` to a file in this "
        "runtime (any runtime) or leave it empty to open the upload dialog (Colab only), then re-run from that cell. Supply "
        "labelled cells as a genes-as-columns CSV (`id`, one column per gene symbol, `label`), a JSON array or a JSONL file of "
        "{id, counts, label} records. Before the split, up to `INFERENCE_CELLS` of your cells (default 6) are held back for "
        "Section 11, so the new-cell inference runs on cells that took no part in training, monitoring or evaluation. Your "
        "cells pass through the same validation, stratified split, baselines, adaptation, held-out evaluation, export and "
        "reload-parity stages as the synthetic sample; every refusal names the file and the rule. The schema, the gene-symbol "
        "convention and the ceilings are stated in the Prerequisites and in Section 4. Uploaded files stay inside this "
        "runtime. BYOD is optional and never part of the default path."
    ),
    "title": "scGPT — DIMER E2E cell-state classification tutorial (standalone)",
    "badges": BADGES,
    "capability": "single-cell expression embeddings, masked-expression prediction and bounded cell-state classification fine-tuning",
    "intro": (
        "scGPT is a single-cell foundation model: a 12-layer transformer encoder pretrained with a masked-expression objective "
        "on 33 million human cells (Cui et al., Nature Methods 2024). A cell enters the model as a set of **(gene token, "
        "expression bin)** pairs — every detected gene's expression is quantile-binned within the cell into 51 levels — with a "
        "`<cls>` token whose output is the cell embedding. There is no positional encoding, so the order in which genes are "
        "listed does not matter, which this notebook checks.\n\n"
        "The checkpoint here is the *whole-human* encoder repackaged in safetensors by Therapeutics Data Commons. The Hub "
        "repository ships no model code, so this pipeline rebuilds the encoder on plain torch modules whose parameter names "
        "match the checkpoint exactly, and the gene vocabulary — which is not in the Hub repository — is fetched from its "
        "persistent Dataverse file id or its immutable exact-byte mirror and digest-verified like everything else. Section 3 "
        "stages it; Section 7 probes what the pretrained decoder can and cannot do.\n\n"
        "The tutorial dataset is synthetic but built from **real lineage programmes**: 24 canonical T-lymphocyte marker genes "
        "and 24 B-lymphocyte marker genes, plus 300 background genes from the vocabulary. A `t-like` cell places the T "
        "programme high and the B programme low; a `b-like` cell does the reverse; every cell is scaled to the same library "
        "size and expresses the same 348 genes, so total counts and gene detection carry no signal by construction.\n\n"
        "**One thing to know before you start: the default task is easy on purpose.** The two programmes are swapped wholesale, "
        "so the frozen embedding already separates the classes with no training at all — the zero-training baseline in "
        "Section 8 scored 1.0 in the recorded run. The fine-tuning in Section 9 therefore cannot show a gain on this data; "
        "what it demonstrates is the adaptation contract (bounded training, an independent test split, an exported adapter "
        "that reloads with identical predictions). Bring your own harder labels (BYOD) to see whether adaptation helps."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Explain** how scGPT turns an expression profile into (gene, bin) tokens and why binning makes it scale-invariant (Section 5).\n"
        "2. **Verify** that `<cls>` embeddings are reproducible and gene-order invariant, and **describe** their shared component (Section 6).\n"
        "3. **Interpret** an honest negative result: the shipped masked-expression decoder does not use lineage context (Section 7).\n"
        "4. **Compare** a fine-tuned head with a zero-training nearest-centroid baseline and two trivial baselines, and say when fine-tuning has nothing to add (Sections 8, 10).\n"
        "5. **Evaluate** accuracy, macro-F1 and AUROC on an independent test split, computed by the exported adapter in a fresh process (Section 10).\n"
        "6. **Verify** that the exported adapter reproduces the adapting process's predictions, and **predict**, run and **explain** the effect of `TRAINABLE_LAYERS` in an optional activity (Sections 11, 13)."
    ),
    "exclusions": (
        "multi-batch integration and batch-correction metrics, perturbation-response prediction, gene-network inference, the "
        "published scGPT benchmarks, the organ-specific scGPT checkpoints, reading AnnData `.h5ad` files directly, and any claim "
        "that a synthetic profile is a real cell. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab (CPU is enough; a T4 GPU is used automatically when present), Kaggle or a Linux Jupyter kernel. The kernel's own Python version does not matter: the notebook installs nothing into it, and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (`torch` 2.14.0 with its CUDA libraries, `safetensors` 0.8.0, `numpy` 2.5.3, `huggingface-hub` 1.32.0). About 0.25 GB of disk is needed for the snapshot and about 8 GB for the isolated environment.",
        "- **Knowledge:** what a single-cell expression profile is, and how accuracy, macro-F1 and AUROC differ. The glossary below explains library size, quantile binning and the other terms the notebook uses.",
        "- **No remote code, no transformers:** the encoder is re-implemented on `torch.nn.TransformerEncoder` and three small blocks; the safetensors state dict loads with `strict=True`. Nothing from the Hub or Dataverse is executed.",
        "- **Data contract:** records are `{{id, counts, label}}`, with `counts` a `{{gene symbol: count}}` mapping of non-negative numbers (raw or normalised — binning is rank-based within the cell); at least 10 detected genes per cell and at least 10 that resolve to a human gene symbol in the scGPT vocabulary (`GAPDH`, not `ENSG00000111640`); unique ids; 2..20 classes and at most 2,000 records. **The true minimum is set by the training split:** after the validation and test cuts it must keep at least 8 cells and 3 per class, which with the default fractions means at least 7 cells per class for two classes (5 per class for three or more); Section 4 refuses a smaller file with that number, and holds back up to 6 further cells for Section 11 only while the split stays valid (10 per class for two classes to hold back all six). Cells with more than 1,535 detected genes keep the most expressed and report the truncation. BYOD accepts a genes-as-columns CSV, a JSON array or JSONL; export an AnnData `.h5ad` to CSV first.",
        "- **Validation is structural, not biological:** nothing checks that a profile is a real cell, that counts share a scale across cells, or that symbols follow one nomenclature.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — a patient-derived expression matrix is exactly that. The default path uploads nothing.",
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can run cells in a hosted notebook and read short Python, who know what "
                "a single-cell expression profile is, and who want to see a single-cell foundation model used end to end: how a "
                "cell becomes tokens, what its embeddings do and do not carry, how to judge a fine-tune against baselines that "
                "need no training, and how to package an adapter that reloads with identical predictions. No experience with "
                "transformers is assumed; the glossary below explains every term.\n\n"
                "**Running it.** Choose *Runtime → Run all*. The default path needs no edit, no upload, no account, no token and "
                "no runtime restart. Section 2 builds an isolated environment, which takes the longest. You can also run one cell "
                "at a time with *Shift + Enter*.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell "
                "calls `run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated "
                "environment's Python, streams what it prints, and stops the notebook with the stage's own error message if it "
                "fails. Stages hand results to each other only through files; the fine-tuned head reaches Sections 10–12 only "
                "through the exported adapter.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–13) are the workflow. *Infrastructure cells* (Sections 1–3) "
                "are collapsed and titled **Infrastructure**; you may run them without studying their implementation.\n\n"
                "**Form controls.** `USE_BYOD`, `BYOD_PATH`, `VAL_FRACTION`, `TEST_FRACTION`, `SEED` and `INFERENCE_CELLS` "
                "(Section 4); `EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE` and `TRAINABLE_LAYERS` (Section 9); `RUN_ACTIVITY` and "
                "`ACTIVITY_TRAINABLE_LAYERS` (Section 13). Leave them at their defaults for the first run.\n\n"
                "**Section tags.** **[Concept]** — what the model does and why. **[Evaluation practice]** — how the evidence is "
                "produced and how to read it. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before Sections 7, 8 and 10 a **Predict before running** prompt asks you to commit to "
                "an expectation; **What to notice** follows each stage; a collapsed **Check your reasoning** answer follows each "
                "checkpoint. The sample answers quote the recorded Kaggle T4 run of this workflow; your numbers should match "
                "them closely on the default path."
            ),
            (
                "## The task: Input → Model → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Validate and split** | 64 labelled cells (`t-like`, `b-like`) as {{gene: count}} maps | `validate_dataset`, stratified `split_dataset` | 36 / 12 / 16 cells; a dataset manifest |\n"
                "| **Tokenise** | one cell | `bin_expression`: normalise, log1p, quantile-bin into 51 levels | `<cls>` + (gene token, bin) pairs |\n"
                "| **Represent** | tokens | the frozen scGPT encoder | a 512-number `<cls>` embedding per cell |\n"
                "| **Baseline** | training and test embeddings, counts | nearest centroid (no training), majority class, library size | test accuracy, macro-F1, AUROC |\n"
                "| **Adapt** | training cells | AdamW on a linear head + the last encoder layer | an adapter (`adapter.safetensors` + `manifest.json`) |\n"
                "| **Evaluate and reuse** | test cells, new cells | the adapter rebuilt in a fresh process | held-out metrics; labels with uncalibrated scores; reload parity |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1–3 | [Engineering] | runtime, carried code, isolated environment, verified snapshot | versions, digests, the vocabulary source |\n"
                "| 4. Cells, validation, split | [Evaluation practice] | sample or BYOD cells, dataset manifest, split | counts per class and split |\n"
                "| 5. Tokens | [Concept] | binning, scale invariance, input refusals | tokens, bins, refusal messages |\n"
                "| 6. Embeddings | [Concept] | `<cls>` vectors, reproducibility and order checks, geometry | three verdicts, cosines |\n"
                "| 7. Masked-expression probe | [Concept] | the pretraining objective, tested honestly | Pearson, lineage probe |\n"
                "| 8. Baselines | [Evaluation practice] | three predictors with no gradient step | the floor every result is read against |\n"
                "| 9. Fine-tuning | [Concept] | bounded AdamW; the adapter is exported | loss and validation history |\n"
                "| 10. Held-out evaluation | [Evaluation practice] | the adapter, rebuilt fresh, scores the test split | metrics, deltas, interpretation |\n"
                "| 11. New cells and reload | [Engineering] | inference and reload parity across processes | predictions, `reload_parity` |\n"
                "| 12. Provenance | [Engineering] | the result record | `result.json` |\n"
                "| 13. Optional activity | [Concept] | `TRAINABLE_LAYERS` changed, printed beside the default | a three-row comparison |\n"
                "| Troubleshooting | [Engineering] | every failure and what to do | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | what was and was not shown | your conclusion |"
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **Expression profile / counts** | For one cell, how many RNA molecules of each gene were measured. |\n"
                "| **Library size** | The total count of a cell; it reflects sequencing depth as much as biology. |\n"
                "| **Gene vocabulary** | The 60,697 human gene symbols scGPT knows, each mapped to a token id; unknown symbols are dropped and reported. |\n"
                "| **Quantile binning** | Within one cell, ranking the detected genes' expression and cutting it into 51 levels; zeros stay 0. |\n"
                "| **`<cls>` token / cell embedding** | A special first token; the encoder's output at that position is the 512-number summary of the cell. |\n"
                "| **Masked-expression objective** | The pretraining task: hide some genes' bins and predict them from the rest of the cell. |\n"
                "| **Nearest-centroid baseline** | Average the training embeddings per class, assign each test cell to the closest average — no training. |\n"
                "| **Macro-F1** | The unweighted mean of per-class F1; it exposes a model that ignores a class. |\n"
                "| **AUROC** | How well a score ranks positives above negatives, independent of the decision threshold; 0.5 is chance. |\n"
                "| **Adapter** | The trained head and last encoder layer, saved as safetensors with a manifest; it is applied on top of the pinned base. |\n"
                "| **Reload parity** | A fresh process rebuilt from the adapter gives the same labels and scores as the process that trained it. |\n"
                "| **Hash-locked environment / stage** | The isolated Python environment every stage runs in; one workflow step run as its own process. |\n\n"
                "</details>"
            ),
        ],
        "infrastructure": {
            "weights": (
                "**What to notice (Section 3):** `fetched` lists `model.safetensors` and `vocab.json` on a first run. The "
                "vocabulary comes from Harvard Dataverse; if Dataverse is down (it returned HTTP 504 during the recorded run), "
                "the stage prints `vocabulary_source` with the mirror URL and accepts the mirror's bytes only because they match "
                "the pinned SHA-256. `verified_files: 4`."
            ),
        },
    },
    "cells": [
        {
            "md": (
                "## 4. Sample cells, validation and split · [Evaluation practice]\n\n"
                "The default dataset is generated in code with a fixed seed from the pinned vocabulary: 32 `t-like` and 32 "
                "`b-like` cells, each expressing the same 348 real gene symbols — 24 T-cell markers, 24 B-cell markers, 300 "
                "background genes — and each scaled to 20,000 total counts before rounding. `validate_dataset` checks the "
                "schema, every count, class coverage and, given the vocabulary, how many genes per cell scGPT can encode, "
                "before any model runs. `split_dataset` shuffles within each class and cuts 20 % validation / 25 % test "
                "(independent cells assumed: real cells from one donor or batch belong in the same split).\n\n"
                "**Bring your own cells.** Set `USE_BYOD = True` and `BYOD_PATH` to a `.csv`, `.json` or `.jsonl` file in this "
                "runtime; with `BYOD_PATH` empty the upload dialog opens in Colab, and elsewhere the cell stops naming "
                "`BYOD_PATH`. Before the split, up to `INFERENCE_CELLS` of your cells are held back for Section 11. A wrong "
                "header, a ragged row, a non-numeric count, Ensembl ids instead of symbols, duplicate ids, a single class or "
                "too few cells for a valid training split stop this cell with the file name and the rule."
            ),
            "code": (
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "VAL_FRACTION = 0.2  # @param {{type:\"number\"}}\n"
                "TEST_FRACTION = 0.25  # @param {{type:\"number\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n"
                "INFERENCE_CELLS = 6  # @param {{type:\"integer\"}}\n\n"
                "def upload_one(what, field):\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError(f'{{field}} is empty, and the upload dialog exists only in Google Colab: set {{field}} to {{what}} in this runtime.') from None\n"
                "    uploaded = files.upload()\n"
                "    if not uploaded:\n"
                "        raise RuntimeError(f'The upload was cancelled or empty: no file was received. Run this cell again and choose {{what}}, or set {{field}}.')\n"
                "    if len(uploaded) != 1:\n"
                "        raise ValueError(f'Upload exactly one file ({{what}}); got {{sorted(uploaded)}}.')\n"
                "    upload_name, payload = next(iter(uploaded.items()))\n"
                "    path = ROOT / 'inputs' / Path(upload_name).name\n"
                "    path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    path.write_bytes(payload)\n"
                "    return str(path)\n\n"
                "byod_path = ''\n"
                "if USE_BYOD:\n"
                "    byod_path = BYOD_PATH or upload_one('one labelled .csv, .json or .jsonl file of cells', 'BYOD_PATH')\n"
                "run_stage('data', use_byod=USE_BYOD, byod_path=byod_path, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, seed=SEED, inference_cells=INFERENCE_CELLS)"
            ),
        },
        {
            "md": (
                "**What to notice:** 64 cells, classes `['b-like', 't-like']`, 348 detected and 348 encodable genes in every "
                "cell, library sizes within a fraction of a percent of 20,000, splits 36 / 12 / 16, six new cells, and "
                "`outputs/{stem}_dataset.csv` in the genes-as-columns shape BYOD expects. The programmes are printed so you can "
                "see they are real markers; the cells are still a generator rule, not biology."
            ),
        },
        {
            "md": (
                "## 5. How a cell becomes tokens · [Concept]\n\n"
                "`bin_expression` is the input encoding scGPT was trained on: detected genes are normalised to 10,000 total "
                "counts, log1p-transformed and **quantile-binned within the cell** into 51 levels (bin 0 is reserved for zeros, "
                "which are never tokenised), then listed as (gene token, bin) pairs after a `<cls>` token with value 0. Because "
                "binning is rank-based, scaling a cell's counts changes nothing — the stage feeds one cell twice, once multiplied "
                "by 37, and prints a `scale_invariance` verdict. Sparse count data has many ties (every gene counted once shares "
                "one value); upstream spreads tied values uniformly at random across the bins they straddle, and the checkpoint "
                "was trained on that spread, so this package does the same with a seeded generator: reproducible, and "
                "distributed like training.\n\n"
                "The ceilings follow from the checkpoint: `MAX_GENES_PER_CELL = 1535` (1,536 positions minus `<cls>`; a cell "
                "with more detected genes keeps the most expressed and reports the truncation), `MAX_CELLS_PER_CALL = 32`, and "
                "at least `MIN_DETECTED_GENES = 10`. Unknown symbols are dropped and **reported**, never mapped onto a real "
                "gene — the packager's own tokenizer maps unknowns to id 0, which is the gene A1BG. The stage shows four "
                "refusals and one truncation, and writes the input manifest to `outputs/{stem}_input_manifest.json`."
            ),
            "code": "run_stage('tokens')",
        },
        {
            "md": (
                "**What to notice:** `first_token_is_cls: True`, the top genes with high bins (the cell's own programme), "
                "`scale_invariance: 'PASS'`, the four `rejected` messages (each names its rule), and `truncation_probe` reporting "
                "`genes_truncated` for a cell with more than 1,535 detected genes.\n\n"
                "**Checkpoint:** a cell measured at twice the sequencing depth has every count doubled. Does scGPT see a "
                "different cell?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "No. Binning ranks genes within the cell, and doubling every count leaves every rank — and so every bin — "
                "unchanged; the ×37 probe shows exactly that. Depth can still matter indirectly, because a deeper cell detects "
                "more genes, and detected genes are what get tokenised. That is why Section 8 checks a library-size baseline.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 6. Cell embeddings (representation, not prediction) · [Concept]\n\n"
                "`pipe.embed` runs the verified encoder and returns the `<cls>` output per cell: 512 numbers. Embeddings are "
                "representations — they carry no label and no metric of their own; a downstream labelled task is what gives "
                "them meaning. The stage embeds eight validation cells, writes them to `outputs/{stem}_embeddings.csv`, and "
                "prints a verdict for three properties: the same batch twice gives identical vectors; **shuffling the order of "
                "a cell's genes changes nothing** — tokens are ranked before encoding and scGPT has no positional encoding, so "
                "the difference must be exactly zero; and embedding one cell alone versus inside a batch differs only by float "
                "accumulation order (bounded at 1e-4, observed around 1e-6).\n\n"
                "It also prints the geometry a reader should know about: raw `<cls>` vectors of unrelated cells have cosine "
                "similarity around 0.95 (a large shared component), so within- and between-class cosines are shown after "
                "centring on the batch mean. That centring is what Section 8's zero-training baseline relies on."
            ),
            "code": "run_stage('embed')",
        },
        {
            "md": (
                "**What to notice:** `dimension: 512`, three `'PASS'` verdicts, `gene_order_max_abs_diff: 0.0`, a raw cosine "
                "near 0.95, and centred cosines that are high within a class and negative between classes."
            ),
        },
        {
            "md": (
                "## 7. The masked-expression objective, probed honestly · [Concept]\n\n"
                "scGPT was pretrained to predict the bins of masked genes from the rest of the cell, and the checkpoint carries "
                "that decoder. `pipe.predict_masked` masks a seeded 15 % of each cell's gene positions with the training-time "
                "mask value (−1) and reports, per cell, the mean absolute error in bins and the Pearson correlation between "
                "predicted and true bins. A second probe asks the question a biologist would: mask the T-cell markers in a "
                "`t-like` cell and in a `b-like` cell — does the decoder predict them higher where the rest of the cell says "
                "\"T lymphocyte\"? (With BYOD the lineage probe is skipped, and the stage says why: your labels need not be "
                "those lineages.)\n\n"
                "**Predict before running:** the encoder learned lineage structure from 33 million cells. Will its decoder "
                "predict masked T-cell markers higher in `t-like` cells than in `b-like` cells?"
            ),
            "code": "run_stage('probe')",
        },
        {
            "md": (
                "**What to notice:** the mean Pearson (0.0045 in the recorded run) and the lineage rows, whose predicted bins "
                "sit in a narrow band whatever the context.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "No. On this checkpoint the decoder's predictions sit around bins 28–32 whatever the context, so the "
                "masked-prediction correlation is near zero (0.0045 recorded) and the lineage probe shows no conditioning. That "
                "is a property of the shipped decoder, not of the encoder: Section 8 shows the *frozen embeddings* separate the "
                "lineages with no training at all, and the same encoder separated real B, T and monocyte cells from a public "
                "PBMC dataset at build time (recorded in the model card). A DIMER profile should therefore expose embeddings and "
                "classification, not imputation.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 8. Baselines on the test split, including one with no training at all · [Evaluation practice]\n\n"
                "Three predictors set the floor before any gradient step. `nearest_centroid_evaluate` is the frozen embedding's "
                "own separability: centre the `<cls>` vectors on the training mean, fit one centroid per class on the **training "
                "split only**, and assign each test cell to the nearest centroid by cosine. It is the zero-training number every "
                "adapted result must be read against — if it is already perfect, fine-tuning has nothing to add on this data. "
                "`majority_baseline` predicts the most frequent training class (0.5 on a balanced split). "
                "`library_size_baseline` thresholds on total counts, fitted on the training split; the sample scales every cell "
                "to the same total, so it should sit at chance.\n\n"
                "**Predict before running:** which of the three will score highest on the 16 test cells, and roughly what will "
                "the library-size baseline score?"
            ),
            "code": "run_stage('baselines')",
        },
        {
            "md": (
                "**What to notice:** nearest centroid `accuracy` 1.0 (recorded), majority 0.5 / macro-F1 0.3333, library size "
                "0.625 / 0.619 / AUROC 0.7188.\n\n"
                "**Checkpoint:** the library-size baseline scored 0.625 accuracy and 0.72 AUROC on a dataset built so that "
                "library size carries no signal. Is that signal?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "No. Rounding leaves library sizes between 19,986 and 20,012 — ±12 counts around 20,000 — and with 16 test cells "
                "an AUROC of 0.72 is not distinguishable from 0.5: the review's permutation test gave a two-sided p ≈ 0.16. On "
                "your own data, read this baseline first: if total counts separate your labels, the labels track sequencing "
                "depth, and any model will exploit that for the wrong reason.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 9. Bounded fine-tuning · [Concept]\n\n"
                "`pipe.adapt` copies the verified encoder, puts a newly initialised linear head over its `<cls>` output, freezes "
                "every parameter except the head and the last `TRAINABLE_LAYERS` encoder layers, and runs AdamW with the "
                "hyperparameters below: tutorial values chosen for a few tens of seconds of CPU, not production settings. "
                "Validation metrics are computed after each epoch for **monitoring only**; the final epoch's weights are kept, "
                "so no selection happens on the validation split. Training loss going down is optimisation evidence, not "
                "task-quality evidence — Section 10 is where quality is measured.\n\n"
                "At the end the stage exports the trained tensors as `outputs/{stem}_adapter/` (`adapter.safetensors` + "
                "`manifest.json` with the base model id and revision, class order, tensor names, sizes and SHA-256). It also "
                "records its own scores for the test split and the new cells, which the fresh reloads in Sections 10 and 11 "
                "must reproduce."
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_LAYERS = 1  # @param {{type:\"integer\"}}\n"
                "run_stage('adapt', epochs=EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_layers=TRAINABLE_LAYERS)"
            ),
        },
        {
            "md": (
                "**What to notice:** `trainable_parameters` (1,579,010 of the total in the recorded run: the head plus one "
                "encoder layer), the time (about 7 s on a T4), and the loss falling across four epochs while validation "
                "accuracy stays at 1.0."
            ),
        },
        {
            "md": (
                "## 10. Held-out evaluation · [Evaluation practice]\n\n"
                "This stage rebuilds the classifier **from the exported adapter in a fresh process** — the files a user would "
                "receive, not the trained object — and classifies every validation and test cell. It reports `accuracy`, "
                "`macro_f1`, per-class precision/recall/F1 with support, and `auroc`. The **test split** was never used for "
                "training or monitoring, so its numbers are the independent evidence. These are tutorial metrics on a synthetic "
                "16-cell split: one holdout, no dispersion estimate. The report — with all three baselines, the deltas, an "
                "interpretation line and the difference from the adapting process's own scores (it should be 0) — is written to "
                "`outputs/{stem}_evaluation_report.json`.\n\n"
                "**Predict before running:** given Section 8, what will `delta_vs_nearest_centroid` be on the default data?"
            ),
            "code": "run_stage('evaluate')",
        },
        {
            "md": (
                "**What to notice:** test `accuracy`, `macro_f1` and `auroc` (1.0 / 1.0 / 1.0 recorded), "
                "`same_as_adapting_process: True`, and the `interpretation` line.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "Zero. The zero-training centroid rule was already perfect, so the adapted head can at best tie it, and the "
                "interpretation line says `saturated`. A perfect score here says the adaptation contract works — training is "
                "bounded, the test split is independent, the adapter reproduces the trained model — not that fine-tuning helped. "
                "On a harder task (your own labels) the delta against the centroid rule is the number to read.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 11. Inference on new cells and fresh reload · [Engineering]\n\n"
                "A fresh process rebuilds the classifier from the adapter again and classifies the six new cells: freshly "
                "generated with a different seed on the default path, or the cells held back before the split with BYOD. "
                "`classify` returns, per cell, the argmax `label`, its `score` and the full `scores` dictionary in class order. "
                "The scores are softmax outputs of a head trained on a few dozen cells — **not calibrated probabilities**; the "
                "only decision rule is argmax. `ScGPTPipeline.from_artifact` re-verifies the base snapshot, checks the adapter "
                "manifest and digests **before** deserialising, rebuilds the classifier and overlays the tensors.\n\n"
                "The stage then compares these predictions with the ones the adapting process recorded in Section 9: identical "
                "labels and scores within `1e-5`, or it stops with *Do not ship this artifact*. Predictions are written to "
                "`outputs/{stem}_predictions.csv`. The stage only reads the adapter, so re-running it alone is safe."
            ),
            "code": "run_stage('export')",
        },
        {
            "md": (
                "**What to notice:** six predictions (6/6 matching their generated labels in the recorded run), the adapter "
                "manifest (`requires_remote_code: False`, the base model and its revision), and `reload_parity: 'PASS'` with "
                "`max_abs_score_diff` 0.0.\n\n"
                "**Checkpoint:** Section 10 already used the adapter. What does this parity check add?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Section 10 shows the adapter's quality; it does not show that the adapter *is* the model that was trained. This "
                "check compares two processes — the one that trained the head and a fresh one built only from files — on the "
                "same cells. Equal labels and scores mean the export captured every trained tensor and the base model it "
                "assumed; a mismatch would mean a user receives a different model from the one evaluated.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 12. Result export and provenance · [Engineering]\n\n"
                "The last output, `outputs/{stem}_result.json`, gathers what a reader needs to interpret the files above: the "
                "notebook source revision, the model id, immutable revision and licence, the vocabulary's source (and whether "
                "the mirror was used) and digest, the fact that no remote code was executed, the dataset source and digest, the "
                "embedding checks, the evaluation report, the new-cell predictions, the adapter manifest, the reload-parity "
                "result, and the isolated environment's versions and device. No credential is involved anywhere in this "
                "notebook, so none can leak into it."
            ),
            "code": "run_stage('result')",
        },
        {
            "md": (
                "## 13. Optional activity: how much of the encoder should you train? · [Concept]\n\n"
                "**Predict → Change → Run → Observe → Explain.** **Predict:** with `ACTIVITY_TRAINABLE_LAYERS = 0` (the head alone, "
                "on embeddings the frozen encoder computes once) instead of 1, will test accuracy fall, and will training be "
                "faster? What about 12 (the whole encoder)? **Change:** tick `RUN_ACTIVITY` and set `ACTIVITY_TRAINABLE_LAYERS`. "
                "**Run** this cell. **Observe** the three rows — zero-training centroid, the default fine-tune, your run — and "
                "the activity's loss history. **Explain** the difference. The activity trains a fresh copy of the base model and "
                "writes only to `outputs/activity/`; the canonical outputs are untouched."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "ACTIVITY_TRAINABLE_LAYERS = 0  # @param {{type:\"integer\"}}\n"
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity', trainable_layers=ACTIVITY_TRAINABLE_LAYERS)\n"
                "else:\n"
                "    print('Optional activity skipped: tick RUN_ACTIVITY to run it. The canonical outputs are complete.')"
            ),
        },
        {
            "md": (
                "**What to notice (if you ran it):** the `trainable_parameters` and `seconds` columns, and whether the activity's "
                "loss falls as fast as the default's.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "No hosted run of the activity has been recorded yet, so compare your own rows. With 0 layers only the linear "
                "head trains, so it is faster; the raw `<cls>` vectors share a large common component (Section 6), so a bare "
                "linear head usually needs more steps to separate the classes than the centred centroid rule, and with four "
                "epochs its loss falls more slowly. With 12 layers training takes much longer and cannot beat a centroid rule "
                "that is already perfect. On a task where the centroid rule is not perfect, this is the experiment that tells "
                "you how much of the encoder is worth adapting.\n\n"
                "</details>"
            ),
        },
    ],
    "closing": (
        "## Troubleshooting · [Engineering]\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| `This notebook needs a Linux x86_64 runtime` | macOS, Windows or ARM kernel | Use Colab, Kaggle or a Linux x86_64 Jupyter kernel. |\n"
        "| `Not enough free disk` | a used runtime | Start a fresh runtime (*Runtime → Disconnect and delete runtime*). |\n"
        "| `uv … wheel size/hash mismatch` or a `--require-hashes` error | a corrupted or substituted download | Re-run Section 2; never remove a pin or a hash. |\n"
        "| `could not download vocab.json from Harvard Dataverse or its immutable mirror` | both vocabulary hosts unreachable | Retry later; the stage already falls back to the mirror when Dataverse alone fails. |\n"
        "| `… sha256 … != manifest …` or `size … != manifest` | a file that is not the pinned one | Delete `weights/scgpt/` and re-run Section 3; never edit the manifest. |\n"
        "| `pinned config.json disagrees with the package constants` | a modified config | Delete `weights/scgpt/` and re-run Section 3. |\n"
        "| `Stage '…' failed …: … is missing: run the stage that writes it` | a cell run out of order | Run the notebook in order from Section 4 (or *Run all*). |\n"
        "| `BYOD_PATH is empty, and the upload dialog exists only in Google Colab` | BYOD outside Colab with no path | Set `BYOD_PATH` to your file. |\n"
        "| `The upload was cancelled or empty` / `Upload exactly one file` | a cancelled or multi-file upload | Run the cell again and choose one file. |\n"
        "| `BYOD must be a genes-as-columns .csv, a .json array or a .jsonl file` | another format (e.g. `.h5ad`, `.txt`) | Export the matrix to CSV with gene symbols as columns. |\n"
        "| `<file>: CSV must start with an 'id' column, end with a 'label' column` / `line N has M fields` | a malformed CSV | Fix the header or the ragged row. |\n"
        "| `<file>: record[i] (…) has 0 gene(s) in the scGPT vocabulary … (are these human gene symbols?)` | Ensembl ids or another species | Map the columns to HGNC symbols (`GAPDH`, not `ENSG…`). |\n"
        "| `<file>: record[i] duplicates id …` / `classification needs at least 2 classes` / `… classes exceeds the ceiling of 20` / a class with too few records | the dataset contract | Fix the records; see Prerequisites. |\n"
        "| `<file>: the training split would hold N cells … supply at least K cells per class` | a BYOD file too small for the split | Add cells (or lower `VAL_FRACTION` / `TEST_FRACTION`); K is the true minimum for your class count. |\n"
        "| `only N of 6 cells could be held back while keeping a valid training split` | a small BYOD dataset | Section 11 classifies the N held-back cells (with none, it reports parity on test cells, labelled as such); add cells for a full new-cell check. |\n"
        "| An embedding check prints `'FAIL'` | a changed encoder or runtime | Do not trust later sections; report the versions printed in Section 2. |\n"
        "| `The reloaded adapter does not reproduce the adapting process's predictions` | a broken export or a different base | Do not ship the adapter; re-run from Section 9. |\n\n"
        "## Interpretation and limits\n\n"
        "The frozen encoder already separates cells whose T-cell and B-cell marker programmes are swapped — the zero-training "
        "nearest-centroid rule scores as well as the fine-tuned head — and the bounded adaptation preserves that on an "
        "independent split, computed by the exported adapter in a fresh process. That is the claim: scGPT's `<cls>` embedding "
        "carries lineage structure it learned from 33 million real cells, it reads that structure out of nothing but a binned "
        "expression ranking, and the adaptation contract works on top of it. The default task is saturated, so it cannot show "
        "that adaptation *helps*. The masked-expression decoder, by contrast, is not a usable imputer as shipped, and the "
        "notebook says so with numbers rather than skipping the question.\n\n"
        "The test split has 16 synthetic cells, the metrics come from one seeded holdout with no dispersion estimate, and the "
        "classes are a generator rule with real marker genes in it rather than measured cells. So a perfect score says the "
        "contract works, not that scGPT annotates cell types at any published accuracy, integrates batches, or predicts "
        "perturbations — none of which this repository exercises.\n\n"
        "Three things to carry to real data. **Symbols:** the vocabulary is HGNC symbols, case-insensitively matched; Ensembl ids "
        "are unknown and dropped, and the input manifest tells you how many genes survived — read it before you trust an "
        "embedding. **Splits:** cells from one donor, plate or batch belong together; split by that grouping, not at random. "
        "**Depth:** read the library-size baseline first — if total counts predict your labels, so will any model, for the wrong "
        "reason.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model **and a vocabulary from a second source**, rebuild the encoder without any "
        "upstream code, validate the demonstrated dataset contract, execute bounded fine-tuning, evaluate the exported adapter "
        "against a zero-training baseline and two trivial ones on an independent split, reproduce its predictions in a fresh "
        "process, and emit the shown machine-readable artifacts — without the repository being reachable. It does **not** "
        "establish benchmark superiority, production fitness, or biological validity.\n\n"
        "## Conclusion · [Evaluation practice]\n\n"
        "Write three sentences: what the embedding checks and the masked-expression probe showed about the pretrained model; "
        "what the baselines imply about the fine-tuned head's perfect test score; and what you would need before using an "
        "adapter like this on real cells.\n\n"
        "<details>\n<summary>Sample conclusion (open after writing yours)</summary>\n\n"
        "The `<cls>` embeddings were reproducible and exactly gene-order invariant, and the frozen encoder alone separated the "
        "lineages (nearest centroid 1.0), while the shipped decoder ignored lineage context (Pearson 0.0045). Because the "
        "zero-training rule was already perfect and library size sat at chance (0.625, AUROC 0.72 on 16 cells, p ≈ 0.16), the "
        "fine-tuned head's 1.0 shows that the adaptation contract works — including an adapter that reloads with identical "
        "predictions — not that fine-tuning added anything. Before using such an adapter on real cells I would need real, "
        "donor-grouped splits, a task where the centroid baseline is not perfect, several seeds for a dispersion estimate, and a "
        "check that library size does not predict the labels.\n\n"
        "</details>\n\n"
        "**Next experiments:** run the activity with 0 and with 12 trainable layers; bring your own labelled cells through BYOD "
        "and read the nearest-centroid and library-size baselines before the adapted number; change `SEED` and see how much "
        "the library-size baseline moves.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/scgpt-single-cell-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/scgpt-single-cell-pipeline/blob/main/MODEL_CARD.md\n"
        "- Upstream model (safetensors packaging): https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/bowang-lab/scGPT\n"
        "- Cui, H., Wang, C., Maan, H., Pang, K., Luo, F., Duan, N., Wang, B. (2024). scGPT: toward building a foundation model "
        "for single-cell multi-omics using generative AI. Nature Methods 21, 1470–1480. https://doi.org/10.1038/s41592-024-02201-0\n"
        "- Velez-Arce, A., et al. (2024). Signals in the Cells: Multimodal and Contextualized Machine Learning Foundations for "
        "Therapeutics. NeurIPS 2024 Workshop on AI for New Drug Modalities (the TDC packaging)."
    ),
}
