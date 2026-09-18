"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E cell-state workflow: the pinned scGPT snapshot (weights, config and
the Dataverse-sourced vocabulary) is digest-verified, a synthetic marker-programme dataset built
from real gene symbols is validated and split, cells are binned the way scGPT expects, cell
embeddings and the masked-expression objective are probed, a zero-training nearest-centroid
baseline and two trivial baselines are measured, a bounded AdamW fine-tuning runs in the kernel,
and the adapter is exported and reloaded.
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

TEMPLATE = {
    "package": "scgpt_single_cell_pipeline",
    "repo_name": REPO,
    "stem": "scgpt_single_cell",
    "notebook_name": "scgpt_single_cell_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies (torch, safetensors, numpy, "
        "huggingface-hub — no transformers, no PyTDC), stages and digest-verifies the pinned scGPT snapshot (4 files, "
        "~205 MB: config, weights and README from the Hub, the gene vocabulary from its persistent Harvard Dataverse file "
        "id), rebuilds the encoder on plain torch modules and loads the weights strictly, generates a deterministic "
        "64-cell dataset in code from two real PBMC marker programmes (no download), validates the cells and the dataset "
        "contract, splits them into stratified train/validation/test sets, shows how a cell becomes (gene, bin) tokens, "
        "computes `<cls>` cell embeddings and checks they are reproducible and gene-order invariant, probes the "
        "masked-expression objective, measures a zero-training nearest-centroid baseline on the frozen embeddings plus "
        "majority-class and library-size baselines, runs a bounded AdamW fine-tuning of a classification head and the last "
        "encoder layer, evaluates accuracy, macro-F1 and AUROC on the held-out test split, classifies six freshly generated "
        "cells, exports the adapter as safetensors with a manifest, and reloads that artifact into a fresh pipeline to "
        "verify prediction parity. The default path needs no repository clone, no DIMER worker or service, no credential, "
        "no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes about a minute of "
        "model time after the download."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled cells as a genes-as-columns CSV (`id`, one column per gene symbol, `label`), a JSON array or a JSONL file of "
        "`{{id, counts, label}}` records. They pass through the same validation, stratified split, baselines, adaptation, "
        "held-out evaluation, inference, artifact export and reload-parity cells as the synthetic sample. The expected schema, "
        "the gene-symbol convention and the ceilings are stated in the Prerequisites and in Section 4, and uploaded files stay "
        "inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "ScGPTPipeline",
    "weights_key": "scgpt",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "safetensors", "numpy"],
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
        "persistent Dataverse file id and digest-verified like everything else. Section 3 loads it; Section 7 probes what the "
        "pretrained decoder can and cannot do.\n\n"
        "The tutorial dataset is synthetic but built from **real lineage programmes**: 24 canonical T-lymphocyte marker genes "
        "and 24 B-lymphocyte marker genes, plus 300 background genes from the vocabulary. A `t-like` cell places the T "
        "programme high and the B programme low; a `b-like` cell does the reverse; every cell is scaled to the same library "
        "size and expresses the same 348 genes, so total counts and gene detection carry no signal by construction."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, dataset and metrics modules; stage and digest-verify an "
        "immutable snapshot whose vocabulary comes from a second, non-Hub source; read how expression counts become (gene, "
        "bin) tokens and why ties are spread; validate cells and split a labelled dataset without leakage; extract `<cls>` "
        "cell embeddings and confirm they are reproducible and gene-order invariant; probe the masked-expression objective and "
        "read an honest negative result; measure a zero-training nearest-centroid baseline on the frozen embeddings and two "
        "trivial baselines; run a bounded fine-tuning with explicit hyperparameters; evaluate accuracy, macro-F1 and AUROC on "
        "an independent test split; classify new cells; and export a safetensors adapter that reloads against the pinned "
        "base with verified parity."
    ),
    "exclusions": (
        "multi-batch integration and batch-correction metrics, perturbation-response prediction, gene-network inference, "
        "the published scGPT benchmarks, the organ-specific scGPT checkpoints, reading AnnData `.h5ad` files directly, and "
        "any claim that a synthetic profile is a real cell. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). CPU is enough — a cell embeds in about 40 ms and the default fine-tuning takes about 20 s — and CUDA is used automatically when present.",
        "- **Knowledge:** what a single-cell expression profile is, what library size and quantile binning mean, and how accuracy, macro-F1 and AUROC differ.",
        "- **No remote code, no transformers:** the encoder is re-implemented on `torch.nn.TransformerEncoder` and three small blocks; the safetensors state dict loads with `strict=True`. Nothing from the Hub or Dataverse is executed.",
        "- **Data contract:** records are `{{id, counts, label}}`, with `counts` a `{{gene symbol: count}}` mapping of non-negative numbers (raw or normalised — binning is rank-based within the cell); at least 10 detected genes per cell and at least 10 that resolve to a human gene symbol in the scGPT vocabulary (`GAPDH`, not `ENSG00000111640`); unique ids; at least 8 records and 3 per class, 2..20 classes. Cells with more than 1,535 detected genes keep the most expressed and report the truncation. BYOD accepts a genes-as-columns CSV, JSON array or JSONL; export an AnnData `.h5ad` to CSV first.",
        "- **Validation is structural, not biological:** nothing checks that a profile is a real cell, that counts share a scale across cells, or that symbols follow one nomenclature.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — a patient-derived expression matrix is exactly that. The default path uploads nothing.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample cells, validation and split\n\n"
                "The default dataset is generated in code with a fixed seed from the pinned vocabulary: 32 `t-like` and 32 "
                "`b-like` cells, each expressing the same 348 real gene symbols — 24 T-cell markers, 24 B-cell markers, 300 "
                "background genes — and each scaled to 20,000 total counts before rounding. `validate_dataset` checks the "
                "schema, every count, class coverage and, given the vocabulary, how many genes per cell scGPT can encode, "
                "before any model runs. `split_dataset` shuffles within each class and cuts 20 % validation / 25 % test.\n\n"
                "Look for: 64 cells, classes `['b-like', 't-like']`, 348 detected and 348 encodable genes in every cell, "
                "library sizes within a fraction of a percent of 20,000, splits 36/12/16, and a written "
                "`outputs/{stem}_sample_dataset.csv` in the genes-as-columns shape BYOD expects. The programmes are printed so "
                "you can see they are real markers; the cells are still a generator rule, not biology."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "VAL_FRACTION = 0.2  # @param {{type:\"number\"}}\n"
                "TEST_FRACTION = 0.25  # @param {{type:\"number\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "vocabulary = pipe.vocabulary\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    records = generate_sample_dataset(vocabulary)\n"
                "    data_source = f'synthetic marker-programme dataset (seed {{SAMPLE_SEED}}, {{SAMPLE_SIZE}} cells, real gene symbols)'\n"
                "    programmes = select_sample_genes(vocabulary)\n"
                "    print({{'t_cell_programme': programmes['t_cell']}})\n"
                "    print({{'b_cell_programme': programmes['b_cell']}})\n"
                "    print({{'background_genes': len(programmes['background']), 'first_five': programmes['background'][:5]}})\n\n"
                "dataset_manifest = validate_dataset(records, vocabulary=vocabulary)\n"
                "CLASSES = dataset_manifest['classes']\n"
                "splits = split_dataset(records, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, seed=SEED)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_csv(records, 'outputs/{stem}_sample_dataset.csv')\n\n"
                "print({{'data_source': data_source, 'n_records': dataset_manifest['n_records'], 'classes': CLASSES, 'class_counts': dataset_manifest['class_counts']}})\n"
                "print({{'detected_genes': dataset_manifest['detected_genes'], 'encodable_genes': dataset_manifest['encodable_genes'], 'library_size': dataset_manifest['library_size']}})\n"
                "print({{'ceilings': dataset_manifest['ceilings'], 'digest': dataset_manifest['digest'][:16] + '...'}})\n"
                "print({{'train': len(train_records), 'validation': len(val_records), 'test': len(test_records)}})"
            ),
        },
        {
            "md": (
                "## 5. How a cell becomes tokens\n\n"
                "`bin_expression` is the input encoding scGPT was trained on: detected genes are normalised to 10,000 total "
                "counts, log1p-transformed and **quantile-binned within the cell** into 51 levels (bin 0 is reserved for zeros, "
                "which are never tokenised), then listed as (gene token, bin) pairs after a `<cls>` token with value 0. Because "
                "binning is rank-based, scaling a cell's counts changes nothing — the cell below is fed twice, once multiplied "
                "by 37. Sparse count data has many ties (every gene counted once shares one value); upstream spreads tied "
                "values uniformly at random across the bins they straddle, and the checkpoint was trained on that spread, so "
                "this package does the same with a seeded generator: reproducible, and distributed like training.\n\n"
                "The ceilings follow from the checkpoint: `MAX_GENES_PER_CELL = 1535` (1,536 positions minus `<cls>`; a cell "
                "with more detected genes keeps the most expressed and reports the truncation), `MAX_CELLS_PER_CALL = 32`, and "
                "at least `MIN_DETECTED_GENES = 10`. Unknown symbols are dropped and **reported**, never mapped onto a real "
                "gene — the packager's own tokenizer maps unknowns to id 0, which is the gene A1BG. The cell shows four "
                "rejections and one truncation."
            ),
            "code": (
                "example = test_records[0]\n"
                "encoded = bin_expression(example['counts'], vocabulary)\n"
                "print({{'id': example['id'], 'label': example['label'], 'tokens': len(encoded['tokens']), 'first_token_is_cls': encoded['tokens'][0] == vocabulary.cls_id, 'n_bins': encoded['n_bins']}})\n"
                "print({{'top_genes': encoded['gene_symbols'][:10], 'top_bins': encoded['values'][1:11]}})\n"
                "print({{'bottom_genes': encoded['gene_symbols'][-5:], 'bottom_bins': encoded['values'][-5:]}})\n"
                "scaled = bin_expression({{g: v * 37 for g, v in example['counts'].items()}}, vocabulary)\n"
                "print({{'scale_invariant': scaled['values'] == encoded['values'] and scaled['tokens'] == encoded['tokens']}})\n"
                "assert scaled['values'] == encoded['values']\n\n"
                "print({{'max_genes_per_cell': MAX_GENES_PER_CELL, 'max_cells_per_call': MAX_CELLS_PER_CALL, 'min_detected_genes': MIN_DETECTED_GENES, 'vocab_size': VOCAB_SIZE, 'specials': {{'pad': vocabulary.pad_id, 'cls': vocabulary.cls_id, 'eoc': vocabulary.eoc_id}}}})\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n\n"
                "probes = {{\n"
                "    'empty cell': {{}},\n"
                "    'two genes': {{'GAPDH': 3, 'ACTB': 5}},\n"
                "    'negative count': {{**example['counts'], 'CD3D': -1}},\n"
                "    'no known symbols': {{f'NOTAGENE{{i}}': 1.0 for i in range(12)}},\n"
                "}}\n"
                "for name, cell in probes.items():\n"
                "    try:\n"
                "        validate_inputs([cell], vocabulary)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})\n\n"
                "wide = {{symbol: 1.0 + k for k, symbol in enumerate(vocabulary.gene_symbols[:MAX_GENES_PER_CELL + 50])}}\n"
                "wide_manifest = validate_inputs([wide], vocabulary)\n"
                "print({{'truncation_probe': wide_manifest['inputs'][0]}})\n\n"
                "input_manifest = validate_inputs([r['counts'] for r in test_records[:4]], vocabulary, names=[r['id'] for r in test_records[:4]])\n"
                "print({{'verdict': input_manifest['verdict'], 'n_cells': input_manifest['n_cells'], 'max_tokens_observed': input_manifest['max_tokens_observed'], 'requires_remote_code': input_manifest['requires_remote_code']}})"
            ),
        },
        {
            "md": (
                "## 6. Cell embeddings (representation, not prediction)\n\n"
                "`pipe.embed` runs the verified encoder and returns the `<cls>` output per cell: 512 numbers. Embeddings are "
                "representations — they carry no label and no metric of their own; a downstream labelled task is what gives "
                "them meaning (EVAL9). The cell embeds eight validation cells, writes them to `outputs/{stem}_embeddings.csv` "
                "(OUT4), and checks three properties: the same batch twice gives identical vectors; **shuffling the order of "
                "a cell's genes changes nothing** — tokens are ranked before encoding and scGPT has no positional encoding, so "
                "the assertion is exact; and embedding one cell alone versus inside a batch differs only by float accumulation "
                "order (bounded below at 1e-4, observed around 1e-6).\n\n"
                "It also prints the geometry a reader should know about: raw `<cls>` vectors of unrelated cells have cosine "
                "similarity around 0.95 (a large shared component), so within- and between-class cosines are shown after "
                "centring on the batch mean. That centring is what Section 8's zero-training baseline relies on."
            ),
            "code": (
                "import csv\n"
                "import math\n"
                "import random\n\n"
                "embed_records = val_records[:8]\n"
                "embedding_result = pipe.embed([r['counts'] for r in embed_records], names=[r['id'] for r in embed_records])\n"
                "vectors = embedding_result['embeddings']\n"
                "print({{'n_cells': embedding_result['n_cells'], 'dimension': embedding_result['dimension'], 'tokens': embedding_result['tokens'], 'pooling': embedding_result['pooling']}})\n\n"
                "repeat = pipe.embed([r['counts'] for r in embed_records], names=[r['id'] for r in embed_records])['embeddings']\n"
                "single = pipe.embed([embed_records[0]['counts']])['embeddings'][0]\n"
                "shuffled = list(embed_records[0]['counts'].items())\n"
                "random.Random(3).shuffle(shuffled)\n"
                "reordered = pipe.embed([dict(shuffled)])['embeddings'][0]\n"
                "order_diff = max(abs(a - b) for a, b in zip(reordered, single))\n"
                "cross_batch = max(abs(a - b) for a, b in zip(single, vectors[0]))\n"
                "print({{'same_batch_twice_identical': repeat == vectors, 'gene_order_max_abs_diff': order_diff, 'single_vs_batch_max_abs_diff': cross_batch, 'note': 'gene order in the input never matters (tokens are ranked before encoding, and the model has no positional encoding); batch composition changes float accumulation order at the 1e-6 level'}})\n"
                "assert repeat == vectors\n"
                "assert order_diff == 0.0\n"
                "assert cross_batch < 1e-4\n\n"
                "def cosine(a, b):\n"
                "    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))\n\n"
                "mean = [sum(v[d] for v in vectors) / len(vectors) for d in range(len(vectors[0]))]\n"
                "centred = [[x - m for x, m in zip(v, mean)] for v in vectors]\n"
                "raw_pairs, within, between = [], [], []\n"
                "for i in range(len(embed_records)):\n"
                "    for j in range(i + 1, len(embed_records)):\n"
                "        raw_pairs.append(cosine(vectors[i], vectors[j]))\n"
                "        (within if embed_records[i]['label'] == embed_records[j]['label'] else between).append(cosine(centred[i], centred[j]))\n"
                "print({{'raw_cosine_between_cells_mean': round(sum(raw_pairs) / len(raw_pairs), 4), 'centred_cosine_within_class': round(sum(within) / len(within), 4) if within else None, 'centred_cosine_between_classes': round(sum(between) / len(between), 4) if between else None, 'note': 'inspection only; embeddings are unlabelled representations'}})\n\n"
                "with open('outputs/{stem}_embeddings.csv', 'w', encoding='utf-8', newline='') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['id', 'label', 'tokens'] + [f'dim_{{k}}' for k in range(embedding_result['dimension'])])\n"
                "    for r, n_tokens, vec in zip(embed_records, embedding_result['tokens'], vectors):\n"
                "        writer.writerow([r['id'], r['label'], n_tokens] + [f'{{x:.6f}}' for x in vec])\n"
                "print('wrote outputs/{stem}_embeddings.csv')"
            ),
        },
        {
            "md": (
                "## 7. The masked-expression objective, probed honestly\n\n"
                "scGPT was pretrained to predict the bins of masked genes from the rest of the cell, and the checkpoint carries "
                "that decoder. `pipe.predict_masked` masks a seeded 15 % of each cell's gene positions with the training-time "
                "mask value (−1) and reports, per cell, the mean absolute error in bins and the Pearson correlation between "
                "predicted and true bins. A second probe asks the question a biologist would: mask the T-cell markers in a "
                "`t-like` cell and in a `b-like` cell — does the decoder predict them higher where the rest of the cell says "
                "\"T lymphocyte\"?\n\n"
                "Read the numbers as they come. On this checkpoint the decoder's predictions sit in a narrow band around bin "
                "28–32 whatever the context, so the masked-prediction correlation is weak and the lineage probe shows no "
                "conditioning. That is a property of the shipped decoder, not of the encoder: Section 8 shows the *frozen "
                "embeddings* separate the lineages perfectly with no training at all, and the same encoder separated real B, T "
                "and monocyte cells from a public PBMC dataset at build time (recorded in the model card). A DIMER profile "
                "should therefore expose embeddings and classification, not imputation."
            ),
            "code": (
                "masked = pipe.predict_masked([r['counts'] for r in val_records[:8]], names=[r['id'] for r in val_records[:8]], mask_fraction=0.15, seed=SEED)\n"
                "pearsons = [c['pearson'] for c in masked['cells'] if c['pearson'] is not None]\n"
                "maes = [c['mean_absolute_error_bins'] for c in masked['cells']]\n"
                "print({{'mask_fraction': masked['mask_fraction'], 'mask_value': masked['mask_value'], 'n_bins': masked['n_bins']}})\n"
                "print({{'mean_pearson': round(sum(pearsons) / len(pearsons), 4), 'mean_absolute_error_bins': round(sum(maes) / len(maes), 3), 'note': masked['note']}})\n"
                "for c in masked['cells'][:3]:\n"
                "    print(c)\n\n"
                "if not USE_BYOD:\n"
                "    lineage_probe = {{}}\n"
                "    for label in CLASSES:\n"
                "        cells_of = [r['counts'] for r in val_records if r['label'] == label][:4]\n"
                "        for marker_name, markers in (('T markers', T_CELL_PROGRAMME), ('B markers', B_CELL_PROGRAMME)):\n"
                "            probe = pipe.predict_masked_genes(cells_of, markers)\n"
                "            lineage_probe[f'{{label}} / masked {{marker_name}}'] = {{'predicted_mean_bin': probe['predicted_mean_bin'], 'true_mean_bin': probe['true_mean_bin']}}\n"
                "    for key, row in lineage_probe.items():\n"
                "        print({{key: row}})\n"
                "    print({{'reading': 'a decoder that used lineage context would predict T markers high in t-like cells and low in b-like cells; the shipped decoder predicts a similar bin either way'}})\n"
                "else:\n"
                "    lineage_probe = None"
            ),
        },
        {
            "md": (
                "## 8. Baselines on the test split, including one with no training at all\n\n"
                "Three predictors set the floor before any gradient step (EVAL10/EVAL11). `pipe.nearest_centroid_evaluate` is "
                "the frozen embedding's own separability: centre the `<cls>` vectors on the training mean, fit one centroid "
                "per class on the **training split only** (SPL8), and assign each test cell to the nearest centroid by cosine. "
                "It is the zero-training number every adapted result must be read against — if it is already perfect, "
                "fine-tuning has nothing to add on this data. `majority_baseline` predicts the most frequent training class "
                "(0.5 on a balanced split). `library_size_baseline` thresholds on total counts, fitted on the training split; "
                "the sample scales every cell to the same total, so it sits at chance by construction, which is the point.\n\n"
                "On your own data, read the library-size baseline first: if total counts already separate your labels, the "
                "labels track sequencing depth, not biology."
            ),
            "code": (
                "baseline_centroid = pipe.nearest_centroid_evaluate(train_records, test_records, classes=CLASSES)\n"
                "print({{k: baseline_centroid[k] for k in ('baseline', 'fitted_on', 'n', 'accuracy', 'macro_f1', 'auroc')}})\n"
                "baseline_majority = majority_baseline(train_records, test_records, CLASSES)\n"
                "print({{k: baseline_majority[k] for k in ('baseline', 'predicted_label', 'accuracy', 'macro_f1')}})\n"
                "baseline_library = library_size_baseline(train_records, test_records, CLASSES)\n"
                "print({{k: baseline_library[k] for k in ('baseline', 'rule', 'train_accuracy', 'accuracy', 'macro_f1', 'auroc')}})"
            ),
        },
        {
            "md": (
                "## 9. Bounded fine-tuning\n\n"
                "`pipe.adapt` copies the verified encoder, puts a newly initialised linear head over its `<cls>` output, "
                "freezes every parameter except the head and the last `TRAINABLE_LAYERS` encoder layers, and runs AdamW with "
                "the hyperparameters below (FT4/FT6): tutorial values chosen for a few tens of seconds of CPU, not production "
                "settings. Validation metrics are computed after each epoch for **monitoring only**; the final epoch's weights "
                "are kept, so no selection happens on the validation split (EVAL14). Training loss going down is optimisation "
                "evidence, not task-quality evidence (FT7) — Section 10 is where quality is measured.\n\n"
                "`TRAINABLE_LAYERS = 0` trains the head alone on `<cls>` embeddings the frozen encoder computes once — fast, and "
                "worth comparing: the raw embeddings share a large common component, so a bare linear head learns more slowly "
                "than the centred nearest-centroid rule above."
            ),
            "code": (
                "import time\n\n"
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_LAYERS = 1  # @param {{type:\"integer\"}}\n\n"
                "started = time.perf_counter()\n"
                "adapt_result = pipe.adapt(\n"
                "    train_records,\n"
                "    val_records,\n"
                "    classes=CLASSES,\n"
                "    epochs=EPOCHS,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    trainable_layers=TRAINABLE_LAYERS,\n"
                "    seed=SEED,\n"
                ")\n"
                "adapt_seconds = round(time.perf_counter() - started, 2)\n"
                "print({{'method': adapt_result['method'], 'trainable_parameters': adapt_result['trainable_parameters'], 'total_parameters': adapt_result['total_parameters'], 'precision': adapt_result['precision'], 'device': pipe.device, 'seconds': adapt_seconds}})\n"
                "for step in adapt_result['history']:\n"
                "    print(step)"
            ),
        },
        {
            "md": (
                "## 10. Held-out evaluation\n\n"
                "`pipe.evaluate` classifies every cell of a split and reports `accuracy`, `macro_f1` (the unweighted mean of "
                "per-class F1, which exposes a model that ignores a class), per-class precision/recall/F1 with support, and "
                "`auroc` (ranking quality of the positive-class score, independent of the argmax threshold). The **test split** "
                "was never used for training or monitoring, so its numbers are the independent evidence (SPL6/SPL7). These are "
                "tutorial metrics on a synthetic 16-cell split (EVAL6): one holdout, no dispersion estimate. The report, with "
                "all three baselines and the deltas against the zero-training centroid rule and the majority rule, is written "
                "to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "val_metrics = pipe.evaluate(val_records)\n"
                "test_metrics = pipe.evaluate(test_records)\n"
                "print({{'split': 'validation', **{{k: val_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}}}})\n"
                "print({{'split': 'test', **{{k: test_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}}}})\n"
                "for cls_name, row in test_metrics['per_class'].items():\n"
                "    print({{'class': cls_name, **row}})\n\n"
                "evaluation_report = {{\n"
                "    'task': 'cell-state classification (bounded fine-tuning of scGPT)',\n"
                "    'evidence': 'tutorial sample-sanity metrics on one stratified holdout of synthetic marker-programme cells; not a benchmark',\n"
                "    'estimation': 'single train/validation/test split, seed ' + str(SEED) + ', no dispersion estimate',\n"
                "    'data_source': data_source,\n"
                "    'dataset_digest': dataset_manifest['digest'],\n"
                "    'classes': CLASSES,\n"
                "    'splits': {{'train': len(train_records), 'validation': len(val_records), 'test': len(test_records)}},\n"
                "    'baselines': {{'nearest_centroid': baseline_centroid, 'majority': baseline_majority, 'library_size': baseline_library}},\n"
                "    'validation_metrics': val_metrics,\n"
                "    'test_metrics': test_metrics,\n"
                "    'delta_vs_nearest_centroid': {{k: round(test_metrics[k] - baseline_centroid[k], 4) for k in ('accuracy', 'macro_f1')}},\n"
                "    'delta_vs_majority': {{k: round(test_metrics[k] - baseline_majority[k], 4) for k in ('accuracy', 'macro_f1')}},\n"
                "    'masked_expression_probe': {{'mean_pearson': round(sum(pearsons) / len(pearsons), 4), 'mean_absolute_error_bins': round(sum(maes) / len(maes), 3), 'lineage_probe': lineage_probe}},\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k != 'trainable_parameter_names'}},\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "print({{'delta_vs_nearest_centroid': evaluation_report['delta_vs_nearest_centroid'], 'delta_vs_majority': evaluation_report['delta_vs_majority'], 'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 11. Inference on new cells, artifact export and fresh reload\n\n"
                "`pipe.classify` returns, per cell, the argmax `label`, its `score` and the full `scores` dictionary in class "
                "order. The scores are softmax outputs of a head trained on a few dozen cells — **not calibrated "
                "probabilities** (UNC2); the only decision rule is argmax (UNC3). The new cells are generated with a different "
                "seed so they were never seen in any split.\n\n"
                "`pipe.save_artifact` writes the trained tensors as `adapter.safetensors`, with a `manifest.json` recording the "
                "artifact format, the base model id and revision, the class order, the tensor names, the file size and SHA-256, "
                "and the adaptation configuration (OUT8). `ScGPTPipeline.from_artifact` re-verifies the base snapshot, checks "
                "the artifact manifest and digests **before** deserialising, rebuilds the classifier and overlays the tensors — "
                "a fresh object from files, not the in-memory model (VER2). The cell asserts identical labels and scores within "
                "`1e-5` (VER4)."
            ),
            "code": (
                "if USE_BYOD:\n"
                "    new_records = test_records[:6]\n"
                "    new_source = 'first six BYOD test-split cells'\n"
                "else:\n"
                "    new_records = generate_sample_dataset(vocabulary, seed=7, size=6)\n"
                "    new_source = 'freshly generated marker-programme cells (seed 7)'\n"
                "inference_result = pipe.classify([r['counts'] for r in new_records], names=[r['id'] for r in new_records])\n"
                "predictions = inference_result['predictions']\n"
                "print({{'new_source': new_source, 'decision_rule': inference_result['decision_rule']}})\n"
                "n_match = 0\n"
                "for p, r in zip(predictions, new_records):\n"
                "    n_match += p['label'] == r['label']\n"
                "    print({{'id': p['id'], 'predicted': p['label'], 'score': round(p['score'], 4), 'true_label': r['label']}})\n"
                "print({{'matches': n_match, 'of': len(new_records), 'note': 'sanity check on generated labels, not an evaluation'}})\n\n"
                "with open('outputs/{stem}_predictions.csv', 'w', encoding='utf-8', newline='') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['id', 'predicted_label', 'score'] + [f'score_{{c}}' for c in CLASSES])\n"
                "    for p in predictions:\n"
                "        writer.writerow([p['id'], p['label'], f\"{{p['score']:.6f}}\"] + [f\"{{p['scores'][c]:.6f}}\" for c in CLASSES])\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "pipe.save_artifact(artifact_dir, metadata={{'data_source': data_source, 'dataset_digest': dataset_manifest['digest'], 'test_metrics': {{k: test_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}}}})\n"
                "with open(artifact_dir / ARTIFACT_MANIFEST_NAME, encoding='utf-8') as f:\n"
                "    artifact_manifest = json.load(f)\n"
                "print({{'format': artifact_manifest['format'], 'base_model': artifact_manifest['base_model'], 'requires_remote_code': artifact_manifest['requires_remote_code'], 'n_tensors': len(artifact_manifest['tensors']), 'files': artifact_manifest['files']}})\n\n"
                "reloaded_pipe = ScGPTPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR)\n"
                "reloaded_result = reloaded_pipe.classify([r['counts'] for r in new_records], names=[r['id'] for r in new_records])\n"
                "max_score_diff = 0.0\n"
                "for before, after in zip(predictions, reloaded_result['predictions']):\n"
                "    assert before['id'] == after['id'] and before['label'] == after['label'], f'reload parity failure on {{before[\"id\"]}}'\n"
                "    max_score_diff = max(max_score_diff, abs(before['score'] - after['score']))\n"
                "assert max_score_diff < 1e-5, f'reload score drift {{max_score_diff}}'\n"
                "print({{'reload_parity': 'PASS', 'labels_equal': True, 'max_abs_score_diff': max_score_diff}})"
            ),
        },
        {
            "md": (
                "## 12. Result export and provenance\n\n"
                "The last output, `outputs/{stem}_result.json`, gathers what a reader needs to interpret the files above: the "
                "notebook source revision, the model id, immutable revision and licence, the vocabulary's source and digest, "
                "the fact that no remote code was executed, the dataset source and digest, the masked-expression probe, the "
                "adaptation configuration, baseline and held-out metrics, the new-cell predictions, the artifact manifest, the "
                "reload-parity result, and the runtime versions and device (OUT6/OUT7). No credential is involved anywhere in "
                "this notebook, so none can leak into it (OUT10)."
            ),
            "code": (
                "import platform\n\n"
                "vocab_entry = next(e for e in MANIFEST['files'] if e['path'] == VOCAB_NAME)\n"
                "result_payload = {{\n"
                "    'task': 'cell-state classification adaptation (scGPT)',\n"
                "    'pipeline_class': 'ScGPTPipeline',\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'remote_code_executed': False,\n"
                "    'vocabulary': {{'source': vocab_entry.get('source'), 'sha256': vocab_entry['sha256'], 'size': VOCAB_SIZE}},\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'data_source': data_source,\n"
                "    'dataset_manifest': dataset_manifest,\n"
                "    'embedding_summary': {{'n_cells': embedding_result['n_cells'], 'dimension': embedding_result['dimension'], 'pooling': embedding_result['pooling'], 'gene_order_max_abs_diff': order_diff, 'single_vs_batch_max_abs_diff': cross_batch}},\n"
                "    'evaluation_report': evaluation_report,\n"
                "    'inference': {{'new_source': new_source, 'decision_rule': inference_result['decision_rule'], 'predictions': predictions}},\n"
                "    'artifact_format': ARTIFACT_FORMAT,\n"
                "    'artifact_format_version': ARTIFACT_FORMAT_VERSION,\n"
                "    'artifact_manifest': artifact_manifest,\n"
                "    'reload_parity': {{'labels_equal': True, 'max_abs_score_diff': max_score_diff}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'safetensors': safetensors.__version__,\n"
                "        'numpy': numpy.__version__,\n"
                "        'device': pipe.device,\n"
                "        'precision': 'float32',\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen encoder already separates cells whose T-cell and B-cell marker programmes are swapped — the zero-training "
        "nearest-centroid rule scores as well as the fine-tuned head — and the bounded adaptation preserves that on an "
        "independent split. That is the claim: scGPT's `<cls>` embedding carries lineage structure it learned from 33 million "
        "real cells, it reads that structure out of nothing but a binned expression ranking, and the adaptation contract works "
        "on top of it. The masked-expression decoder, by contrast, is not a usable imputer as shipped, and the notebook says so "
        "with numbers rather than skipping the question.\n\n"
        "The test split has 16 synthetic cells, the metrics come from one seeded holdout with no dispersion estimate, and the "
        "classes are a generator rule with real marker genes in it rather than measured cells. So a perfect score says the "
        "contract works, not that scGPT annotates cell types at any published accuracy, integrates batches, or predicts "
        "perturbations — none of which this repository exercises.\n\n"
        "Three things to carry to real data. **Symbols:** the vocabulary is HGNC symbols, case-insensitively matched; Ensembl ids "
        "are unknown and dropped, and the input manifest tells you how many genes survived — read it before you trust an "
        "embedding. **Splits:** cells from one donor, plate or batch belong together; split by that grouping, not at random. "
        "**Depth:** read the library-size baseline first — if total counts predict your labels, so will any model, for the wrong "
        "reason.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model **and a vocabulary from a second source**, rebuild the encoder "
        "without any upstream code, validate the demonstrated dataset contract, execute bounded fine-tuning, evaluate against a "
        "zero-training baseline and two trivial ones on an independent split, and emit the shown machine-readable artifacts — "
        "without the repository being reachable. It does **not** establish benchmark superiority, production fitness, or "
        "biological validity.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_LAYERS = 0` to train the head alone on "
        "cached embeddings and watch it learn more slowly than the centred centroid rule; raise `TRAINABLE_LAYERS` to 12 to "
        "fine-tune the whole encoder and compare the time; or bring your own labelled cells through BYOD and read the "
        "nearest-centroid and library-size baselines before the adapted number.\n\n"
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
