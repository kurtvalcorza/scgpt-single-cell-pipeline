"""Deterministic in-code sample cells and the labelled-dataset contract for cell-state classification.

The tutorial dataset is synthetic but built from **real human gene symbols and real lineage
programmes**: two sets of 24 canonical PBMC marker genes — a T-lymphocyte programme (CD3D, CD3E,
IL7R, TRAC, …) and a B-lymphocyte programme (CD79A, MS4A1, CD19, IGHM, …) — plus 300 background
genes drawn from the pinned scGPT vocabulary. A `t-like` cell places the T programme high and the B
programme low; a `b-like` cell does the reverse. It is built so that the two trivial handles carry
no signal: every cell is scaled to the same library size (integer rounding leaves a spread of about
0.1 %) and every cell expresses the same 348 genes, so neither total counts nor the number of
detected genes can separate the classes. What differs is *where the two programmes sit in each
cell's expression ranking*, which is what scGPT's per-cell quantile binning exposes to the model.
This is sanity evidence for the adaptation contract, not biology (NOTEBOOK_SPEC 2.0 DAT8): the
profiles are a generator rule with marker genes in it, not measured cells, and the background is
random rather than a real transcriptome.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import MAX_GENES_PER_CELL, MIN_DETECTED_GENES, GeneVocabulary

DATASET_REPRESENTATION = "io.github.kurtvalcorza.dataset.single-cell.binned-expression-labels.v1"
SAMPLE_CLASSES: tuple[str, ...] = ("b-like", "t-like")
SAMPLE_SEED = 20260918
SAMPLE_SIZE = 64  # 32 per class
PROGRAMME_GENES = 24  # per programme
BACKGROUND_GENES = 300
# Canonical PBMC lineage markers (HGNC symbols). These are what the two classes are made of; the
# model has seen their co-expression in 33 million cells, which is the point of using real ones.
T_CELL_PROGRAMME: tuple[str, ...] = (
    "CD3D",
    "CD3E",
    "CD3G",
    "CD2",
    "IL7R",
    "LTB",
    "CD7",
    "TRAC",
    "CD247",
    "LCK",
    "CD5",
    "CD6",
    "TCF7",
    "LEF1",
    "CCR7",
    "IL32",
    "GZMK",
    "CD8A",
    "CD8B",
    "CD4",
    "CD27",
    "CD28",
    "ITK",
    "ZAP70",
)
B_CELL_PROGRAMME: tuple[str, ...] = (
    "CD79A",
    "CD79B",
    "MS4A1",
    "CD19",
    "CD22",
    "BANK1",
    "PAX5",
    "VPREB3",
    "TCL1A",
    "IGHM",
    "IGHD",
    "IGKC",
    "CD74",
    "HLA-DRA",
    "HLA-DRB1",
    "HLA-DQA1",
    "HLA-DQB1",
    "HLA-DPA1",
    "HLA-DPB1",
    "FCER2",
    "CR2",
    "BLK",
    "FCRL1",
    "TNFRSF13C",
)
LIBRARY_SIZE = 20_000  # every cell is scaled to this total, so library size cannot separate the classes
HIGH_FACTOR = 6.0  # programme genes well above the background in the class that expresses them
LOW_FACTOR = 0.2  # the same genes well below it in the other class
MIN_RECORDS = 8
MAX_RECORDS = 2_000
MAX_CLASSES = 20
MIN_RECORDS_PER_CLASS = 3
MAX_ID_CHARS = 64
MAX_LABEL_CHARS = 64
REQUIRED_COLUMNS = ("id", "counts", "label")
# Canonical-looking HGNC symbols only (no clone-derived names such as RP11-22E12.2 or CTB-53D8.3),
# so the sample reads like a real expression matrix.
_CANONICAL_SYMBOL = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")


def select_sample_genes(vocabulary: GeneVocabulary, seed: int = SAMPLE_SEED) -> dict[str, list[str]]:
    """The two marker programmes (checked against the vocabulary) plus a seeded background set."""
    missing = [g for g in (*T_CELL_PROGRAMME, *B_CELL_PROGRAMME) if vocabulary.resolve(g) is None]
    if missing:
        raise RuntimeError(f"marker genes absent from the pinned vocabulary: {missing}")
    rng = random.Random(seed)
    excluded = set(T_CELL_PROGRAMME) | set(B_CELL_PROGRAMME)
    candidates = sorted(
        s for s in vocabulary.gene_symbols if _CANONICAL_SYMBOL.match(s) and s not in excluded
    )
    if len(candidates) < BACKGROUND_GENES:
        raise RuntimeError(
            f"the pinned vocabulary holds {len(candidates)} canonical symbols; {BACKGROUND_GENES} needed"
        )
    return {
        "t_cell": list(T_CELL_PROGRAMME),
        "b_cell": list(B_CELL_PROGRAMME),
        "background": sorted(rng.sample(candidates, BACKGROUND_GENES)),
    }


def generate_sample_dataset(
    vocabulary: GeneVocabulary, seed: int = SAMPLE_SEED, size: int = SAMPLE_SIZE
) -> list[dict[str, Any]]:
    """`size` labelled cells (half `b-like`, half `t-like`), deterministic for a seed.

    Every cell expresses the same 348 genes and is scaled to `LIBRARY_SIZE` total counts before
    rounding. Background genes get a log-normal weight; the cell's own lineage programme is placed
    well above the background and the other lineage's well below it, so the classes differ in
    rank order — which is what per-cell quantile binning reads — rather than in library size or
    gene detection.
    """
    if size < 2 or size % 2:
        raise ValueError("size must be an even number >= 2 (one cell per class per pair)")
    genes = select_sample_genes(vocabulary, seed)
    rng = random.Random(seed + 1)
    records: list[dict[str, Any]] = []
    for i in range(size // 2):
        for label in SAMPLE_CLASSES:
            high = genes["b_cell"] if label == "b-like" else genes["t_cell"]
            low = genes["t_cell"] if label == "b-like" else genes["b_cell"]
            weights: dict[str, float] = {}
            for symbol in genes["background"]:
                weights[symbol] = rng.lognormvariate(0.0, 1.0)
            for symbol in high:
                weights[symbol] = HIGH_FACTOR * rng.lognormvariate(0.0, 0.3)
            for symbol in low:
                weights[symbol] = LOW_FACTOR * rng.lognormvariate(0.0, 0.3)
            scale = LIBRARY_SIZE / sum(weights.values())
            counts = {symbol: max(1, round(value * scale)) for symbol, value in weights.items()}
            records.append(
                {
                    "id": f"{label[0]}-cell-{i:03d}",
                    "counts": counts,
                    "label": label,
                }
            )
    return records


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """SHA-256 over the canonical (id, sorted counts, label) rows; recorded in provenance (OUT9)."""
    canon = json.dumps(
        [[r["id"], sorted(r["counts"].items()), r["label"]] for r in records], separators=(",", ":")
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    vocabulary: GeneVocabulary | None = None,
    classes: Sequence[str] | None = None,
    min_records: int = MIN_RECORDS,
    min_per_class: int = MIN_RECORDS_PER_CLASS,
) -> dict[str, Any]:
    """Check a labelled cell dataset against the contract; return its manifest.

    Every error names the record and the violated rule (VAL4/DAT19). When `vocabulary` is supplied
    the manifest also reports how many genes per cell resolve to an scGPT token, which is the
    number that decides whether a cell can be encoded at all.
    """
    if isinstance(records, str | bytes | Mapping) or not isinstance(records, Sequence):
        raise TypeError("records must be a list of {'id', 'counts', 'label'} mappings")
    if len(records) < min_records:
        raise ValueError(f"dataset has {len(records)} records; at least {min_records} are required")
    if len(records) > MAX_RECORDS:
        raise ValueError(f"dataset has {len(records)} records; ceiling is {MAX_RECORDS}")
    seen_ids: set[str] = set()
    counts_by_class: dict[str, int] = {}
    detected: list[int] = []
    encodable: list[int] = []
    libraries: list[float] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise TypeError(f"record[{i}] must be a mapping, got {type(rec).__name__}")
        missing = [c for c in REQUIRED_COLUMNS if c not in rec]
        if missing:
            raise ValueError(
                f"record[{i}] is missing required column(s) {missing}; required: {list(REQUIRED_COLUMNS)}"
            )
        rid = str(rec["id"]).strip()
        if not rid or len(rid) > MAX_ID_CHARS:
            raise ValueError(f"record[{i}] id must be 1..{MAX_ID_CHARS} characters")
        if rid in seen_ids:
            raise ValueError(f"record[{i}] duplicates id {rid!r}")
        seen_ids.add(rid)
        cell = rec["counts"]
        if not isinstance(cell, Mapping) or not cell:
            raise ValueError(f"record[{i}] ({rid}) counts must be a non-empty mapping of gene to count")
        n_detected = 0
        total = 0.0
        for gene, value in cell.items():
            if not isinstance(gene, str) or not gene.strip():
                raise TypeError(f"record[{i}] ({rid}) has a non-string gene key {gene!r}")
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"record[{i}] ({rid}) count for {gene!r} must be a number")
            if value < 0 or value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"record[{i}] ({rid}) count for {gene!r} must be finite and non-negative")
            if value > 0:
                n_detected += 1
                total += float(value)
        if n_detected < MIN_DETECTED_GENES:
            raise ValueError(
                f"record[{i}] ({rid}) has {n_detected} detected gene(s); at least {MIN_DETECTED_GENES} "
                "are required to bin an expression profile"
            )
        detected.append(n_detected)
        libraries.append(total)
        if vocabulary is not None:
            n_encodable = sum(
                1 for gene, value in cell.items() if value > 0 and vocabulary.resolve(gene) is not None
            )
            if n_encodable < MIN_DETECTED_GENES:
                raise ValueError(
                    f"record[{i}] ({rid}) has {n_encodable} gene(s) in the scGPT vocabulary; at least "
                    f"{MIN_DETECTED_GENES} are required (are these human gene symbols?)"
                )
            encodable.append(n_encodable)
        label = rec["label"]
        if not isinstance(label, str) or not label.strip() or len(label) > MAX_LABEL_CHARS:
            raise ValueError(
                f"record[{i}] ({rid}) label must be a non-empty string of at most {MAX_LABEL_CHARS} chars"
            )
        counts_by_class[label] = counts_by_class.get(label, 0) + 1
    if classes is None:
        class_list = sorted(counts_by_class)
    else:
        class_list = [str(c) for c in classes]
        unknown = sorted(set(counts_by_class) - set(class_list))
        if unknown:
            raise ValueError(f"labels {unknown} are not in the class list {class_list}")
    if len(class_list) < 2:
        raise ValueError(f"classification needs at least 2 classes, found {class_list}")
    if len(class_list) > MAX_CLASSES:
        raise ValueError(f"{len(class_list)} classes exceeds the ceiling of {MAX_CLASSES}")
    thin = [c for c in class_list if counts_by_class.get(c, 0) < min_per_class]
    if thin:
        raise ValueError(f"classes {thin} have fewer than {min_per_class} records each (class coverage rule)")
    manifest: dict[str, Any] = {
        "verdict": "accepted",
        "representation": DATASET_REPRESENTATION,
        "n_records": len(records),
        "classes": class_list,
        "class_counts": {c: counts_by_class.get(c, 0) for c in class_list},
        "detected_genes": {
            "min": min(detected),
            "max": max(detected),
            "mean": round(sum(detected) / len(detected), 1),
        },
        "library_size": {
            "min": min(libraries),
            "max": max(libraries),
            "mean": round(sum(libraries) / len(libraries), 1),
        },
        "ceilings": {
            "max_genes_per_cell": MAX_GENES_PER_CELL,
            "min_detected_genes": MIN_DETECTED_GENES,
            "max_records": MAX_RECORDS,
            "max_classes": MAX_CLASSES,
            "min_records": min_records,
            "min_records_per_class": min_per_class,
        },
        "digest": dataset_digest(records),
        "findings": [],
    }
    if encodable:
        manifest["encodable_genes"] = {
            "min": min(encodable),
            "max": max(encodable),
            "mean": round(sum(encodable) / len(encodable), 1),
        }
    return manifest


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.25,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Stratified random train/validation/test split (assumes independent cells, SPL3).

    Real single-cell data is rarely independent — cells from one donor, plate or batch belong
    together — so a deployment must split by that grouping instead. The tutorial's cells are
    generated independently, which is why a random split is honest here.
    """
    if not (0.0 < val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("val_fraction and test_fraction must be in (0, 1) and sum to less than 1")
    manifest = validate_dataset(records)
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, Any]]] = {c: [] for c in manifest["classes"]}
    for rec in records:
        by_class[rec["label"]].append(dict(rec))
    out: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for cls in manifest["classes"]:
        rows = by_class[cls]
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * val_fraction))
        n_test = max(1, round(len(rows) * test_fraction))
        if len(rows) - n_val - n_test < 1:
            raise ValueError(f"class {cls!r} has {len(rows)} records; too few to leave one per split")
        out["validation"].extend(rows[:n_val])
        out["test"].extend(rows[n_val : n_val + n_test])
        out["train"].extend(rows[n_val + n_test :])
    for part in out.values():
        rng.shuffle(part)
    return out


def load_byod_dataset(source: str | Path) -> list[dict[str, Any]]:
    """Read a user-supplied cell dataset (JSON array, JSONL, or a genes-as-columns CSV).

    CSV shape: first column `id`, last column `label`, every other column a gene symbol whose cells
    hold counts; zero counts are dropped (they are never tokenised). JSON/JSONL records are
    `{"id": ..., "counts": {gene: count, ...}, "label": ...}`. Nothing is renamed or rescaled
    (VAL7); the records are then validated with `validate_dataset`. An AnnData `.h5ad` file is not
    read here — export its `X` with `var_names` as columns to CSV first.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"BYOD dataset file not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"BYOD dataset file is empty: {path}")
    suffix = path.suffix.lower()
    records: list[dict[str, Any]] = []
    if suffix == ".csv":
        reader = csv.reader(text.splitlines())
        header = [h.strip() for h in next(reader, [])]
        if len(header) < 3 or header[0] != "id" or header[-1] != "label":
            raise ValueError(
                "CSV must start with an 'id' column, end with a 'label' column, and carry one gene per "
                f"column in between; got header {header[:3]}...{header[-1:]}"
            )
        genes = header[1:-1]
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"line {line_no} has {len(row)} fields, header has {len(header)}")
            counts: dict[str, float] = {}
            for gene, value in zip(genes, row[1:-1], strict=True):
                value = value.strip()
                if not value:
                    continue
                try:
                    number = float(value)
                except ValueError as exc:
                    raise ValueError(f"line {line_no}, gene {gene!r}: {value!r} is not a number") from exc
                if number > 0:
                    counts[gene] = number
            records.append({"id": row[0].strip(), "counts": counts, "label": row[-1].strip()})
    elif suffix == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no} is not valid JSON: {exc}") from exc
    elif suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"file is not valid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise TypeError("JSON dataset must be a top-level array of objects")
        records = data
    else:
        raise ValueError(f"unsupported BYOD file type {suffix!r}; use .csv, .json or .jsonl")
    validate_dataset(records)
    return [dict(r) for r in records]


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write records in the BYOD CSV shape (`id`, one column per gene, `label`) as a template."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    genes = sorted({gene for r in records for gene in r["counts"]})
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", *genes, "label"])
        for r in records:
            writer.writerow([r["id"], *(r["counts"].get(gene, 0) for gene in genes), r["label"]])
    return out
