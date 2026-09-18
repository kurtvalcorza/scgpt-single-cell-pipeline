"""scGPT (`tdc/scGPT`) DIMER pipeline: verified snapshot, single-cell expression embeddings,
masked-expression prediction, and bounded cell-state classification fine-tuning with a portable
adapter.

The checkpoint is the scGPT *whole-human* encoder (Cui et al., Nature Methods 2024; 33 million
cells) as repackaged in safetensors by Therapeutics Data Commons: 12 post-norm transformer layers of
width 512 with 8 heads, a 60,697-gene vocabulary, a gene-token encoder, a continuous value encoder,
and the masked-expression decoder. Three things about this row are stated rather than hidden:

* **The architecture is re-implemented here on plain torch modules.** The Hub repository ships no
  model code (`config.json` declares `model_type: "scgpt"` with no `auto_map`), and the packager's
  own loader lives in the PyTDC package with flash-attention and transformers as dependencies. This
  module rebuilds the encoder from `torch.nn.TransformerEncoder` and three small blocks whose
  parameter names match the checkpoint exactly, so the safetensors file loads with `strict=True`.
  It follows the **original** scGPT `model.py` (bowang-lab/scGPT @ `cebd6fae`): the value encoder
  clamps at 512 and applies a ReLU between its two linears, which the PyTDC port omits.
* **The gene vocabulary is not in the Hub repository.** The packager fetches it from Harvard
  Dataverse at runtime; this manifest lists it as a fourth entry with that persistent file id as its
  source and its SHA-256, so it is verified like every other file.
* **Inputs are binned, not raw.** scGPT reads expression as per-cell quantile bins (51 bins, zeros
  stay zero) after total-count normalisation and log1p; the packager's example feeds raw values,
  which is off-distribution. The binning here mirrors `scgpt/preprocess.py` deterministically.

Everything model-related is imported lazily so that snapshot verification and input validation run
(and can refuse) before `torch` is imported (fleet RTM-001). No transformers, no remote code.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_ID = "tdc/scGPT"
MODEL_REVISION = "acf749f35bf5c0b00633838f02588272ed0d9911"
MODEL_LICENSE = "mit"
MODEL_KEY = "scgpt"
ARTIFACT_FORMAT = "org.valcorza.scgpt-single-cell.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
VOCAB_NAME = "vocab.json"
# The vocabulary's persistent source (TDC `scgpt_vocab`, Harvard Dataverse datafile 10809431). It is
# the only manifest entry that is not a Hub file; `stage_missing_files` fetches it from here.
VOCAB_SOURCE_URL = "https://dataverse.harvard.edu/api/access/datafile/10809431"

# Architecture facts from the pinned config.json; asserted against the file at load time.
EMBSIZE = 512
NLAYERS = 12
NHEAD = 8
D_HID = 512
MAX_SEQ_LEN = 1536
VOCAB_SIZE = 60697
SPECIAL_TOKENS = ("<pad>", "<cls>", "<eoc>")
# scGPT input convention: per-cell quantile binning into N_BINS after normalisation, `<cls>` first
# with value 0, masked positions carry MASK_VALUE, and the value encoder clamps at VALUE_CLAMP.
N_BINS = 51
TARGET_SUM = 10_000.0
MASK_VALUE = -1.0
VALUE_CLAMP = 512
BINNING_SEED = 42  # seeds the tie-spreading in per-cell quantile binning (see bin_expression)
# Ceilings.
MAX_GENES_PER_CELL = MAX_SEQ_LEN - 1  # one position is the <cls> token
MAX_CELLS_PER_CALL = 32
MIN_DETECTED_GENES = 10  # fewer measured genes cannot be binned into a meaningful profile


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    listed = {entry["path"] for entry in manifest["files"]}
    for required in (CONFIG_NAME, WEIGHTS_NAME, VOCAB_NAME):
        if required not in listed:
            raise ValueError(f"manifest does not list {required}; refusing to proceed")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest.hexdigest()} != manifest {entry['sha256']}")
    return manifest


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the snapshot against its DIMER manifest (size + SHA-256 of every listed file,
    including the Dataverse-sourced vocabulary)."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    return _verify_manifest(root, MODEL_ID, MODEL_REVISION)


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file: Hub files at the pinned revision, the vocabulary from its
    persistent Dataverse file id. The caller's `verify_snapshot` checks the digest afterwards."""
    if relative_path == VOCAB_NAME:
        import urllib.request

        request = urllib.request.Request(
            VOCAB_SOURCE_URL, headers={"User-Agent": "scgpt-single-cell-pipeline"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        (root / VOCAB_NAME).write_bytes(data)
        return
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest entries that are absent locally (a fresh clone commits the manifest, the config
    and the vocabulary but git-ignores the safetensors file). `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


INPUT_SCHEMA: dict[str, Any] = {
    "input": "1..MAX_CELLS_PER_CALL cells, each a {gene symbol: count} mapping of raw or normalised counts",
    "cells": [1, MAX_CELLS_PER_CALL],
    "genes_per_cell": [MIN_DETECTED_GENES, MAX_GENES_PER_CELL],
    "validation": (
        "type, finiteness and non-negativity of counts, a minimum number of detected genes, and gene-symbol "
        "resolution against the pinned 60,697-symbol vocabulary. Nothing checks that a profile is a real "
        "cell, that counts are on a consistent scale across cells, or that symbols follow one nomenclature"
    ),
    "preprocessing": (
        "genes with zero counts or unknown symbols are dropped and reported; the rest are normalised to "
        f"{TARGET_SUM:g} total counts, log1p-transformed, quantile-binned per cell into {N_BINS} bins (zeros "
        "stay 0), truncated to the most expressed MAX_GENES_PER_CELL genes, and fed as (gene token, bin) "
        "pairs after a <cls> token with value 0; the cell embedding is the <cls> output (512 floats)"
    ),
}


@dataclass(frozen=True)
class GeneVocabulary:
    """Gene symbol -> token id, from the pinned vocab.json; special tokens resolved by name."""

    tokens: dict[str, int]
    pad_id: int
    cls_id: int
    eoc_id: int

    @property
    def gene_symbols(self) -> list[str]:
        return [s for s in self.tokens if s not in SPECIAL_TOKENS]

    def resolve(self, symbol: str) -> int | None:
        """Exact match first, then a case-insensitive upper-case match; None when unknown."""
        if symbol in self.tokens:
            return self.tokens[symbol]
        upper = symbol.upper()
        return self.tokens.get(upper) if upper not in SPECIAL_TOKENS else None


def load_gene_vocabulary(path: str | Path | None = None) -> GeneVocabulary:
    """Read the pinned vocab.json (60,697 entries: 60,694 gene symbols + <pad>, <cls>, <eoc>)."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    raw = json.loads((root / VOCAB_NAME).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or len(raw) != VOCAB_SIZE:
        raise ValueError(
            f"vocab.json must map {VOCAB_SIZE} symbols to ids, got {type(raw).__name__} of {len(raw)}"
        )
    missing = [t for t in SPECIAL_TOKENS if t not in raw]
    if missing:
        raise ValueError(f"vocab.json lacks special tokens {missing}")
    ids = sorted(int(v) for v in raw.values())
    if ids != list(range(VOCAB_SIZE)):
        raise ValueError("vocab.json ids must be exactly 0..VOCAB_SIZE-1 with no gaps or duplicates")
    tokens = {str(k): int(v) for k, v in raw.items()}
    return GeneVocabulary(tokens, tokens["<pad>"], tokens["<cls>"], tokens["<eoc>"])


def _check_cells(cells: Any, names: Any = None) -> tuple[list[dict[str, float]], list[str]]:
    """Raise TypeError/ValueError naming the first violated rule; return (cells, ids).

    ``embed``, ``predict_masked``, ``classify`` and ``validate_inputs`` all route through this
    function so their acceptance criteria cannot diverge.
    """
    if isinstance(cells, Mapping) or not isinstance(cells, Sequence) or isinstance(cells, str | bytes):
        raise TypeError("cells must be a list of {gene: count} mappings (one mapping per cell)")
    if not 1 <= len(cells) <= MAX_CELLS_PER_CALL:
        raise ValueError(f"cells must hold 1..{MAX_CELLS_PER_CALL} items, got {len(cells)}")
    checked: list[dict[str, float]] = []
    for i, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            raise TypeError(f"cells[{i}] must be a mapping of gene to count, got {type(cell).__name__}")
        if not cell:
            raise ValueError(f"cells[{i}] is empty")
        counts: dict[str, float] = {}
        for gene, value in cell.items():
            if not isinstance(gene, str) or not gene.strip():
                raise TypeError(f"cells[{i}] has a non-string gene key {gene!r}")
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"cells[{i}][{gene!r}] must be a number, got {type(value).__name__}")
            if value < 0 or value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"cells[{i}][{gene!r}] must be a finite non-negative count, got {value!r}")
            counts[gene] = float(value)
        detected = sum(1 for v in counts.values() if v > 0)
        if detected < MIN_DETECTED_GENES:
            raise ValueError(
                f"cells[{i}] has {detected} detected gene(s); at least {MIN_DETECTED_GENES} are required "
                "to bin an expression profile"
            )
        checked.append(counts)
    if names is None:
        ids = [f"cell-{i}" for i in range(len(checked))]
    else:
        if isinstance(names, str | bytes) or not isinstance(names, Sequence) or len(names) != len(checked):
            raise ValueError("names must be a list with exactly one id per cell")
        ids = [str(n) for n in names]
        if len(set(ids)) != len(ids):
            raise ValueError("names must be unique")
    return checked, ids


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """numpy's default (linear) quantile on an ascending list, without importing numpy."""
    if not sorted_values:
        raise ValueError("quantile of an empty sequence")
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def bin_expression(
    counts: Mapping[str, float],
    vocabulary: GeneVocabulary,
    *,
    n_bins: int = N_BINS,
    target_sum: float = TARGET_SUM,
    max_genes: int = MAX_GENES_PER_CELL,
    seed: int = BINNING_SEED,
) -> dict[str, Any]:
    """scGPT input encoding of one cell: tokens, per-cell quantile bins, and what was dropped.

    Mirrors `scgpt/preprocess.py`: detected genes are normalised to `target_sum` total counts and
    log1p-transformed, then the non-zero values are digitised against `n_bins - 1` quantile edges
    (bins 1..n_bins-1; zeros are never tokenised). Upstream's `_digitize(side="both")` spreads tied
    values **uniformly at random** between their left and right bin — sparse count data has many
    ties (every count-of-1 gene shares one value), and the checkpoint was trained on that spread —
    so this function does the same with a seeded generator: identical input and seed give identical
    bins, and the distribution matches upstream's. Genes absent from the vocabulary are dropped and
    reported rather than mapped to a real gene (the packager's tokenizer maps unknowns to id 0,
    which is the gene A1BG). Cells with more than `max_genes` detected genes keep the most
    expressed ones and report the truncation.
    """
    import random

    rng = random.Random(seed)
    unknown: list[str] = []
    resolved: dict[int, float] = {}
    symbols: dict[int, str] = {}
    for gene, value in counts.items():
        if value <= 0:
            continue
        token = vocabulary.resolve(gene)
        if token is None:
            unknown.append(gene)
            continue
        resolved[token] = resolved.get(token, 0.0) + value
        symbols.setdefault(token, gene)
    if not resolved:
        raise ValueError(
            "no gene in this cell could be encoded: none of the detected genes is in the scGPT vocabulary "
            "(check that gene identifiers are human gene symbols such as GAPDH, not Ensembl ids)"
        )
    library = sum(resolved.values())
    logged = {tok: math.log1p(v / library * target_sum) for tok, v in resolved.items()}
    ascending = sorted(logged.values())
    edges = [_quantile(ascending, k / (n_bins - 2)) for k in range(n_bins - 1)]

    def digitize(x: float) -> int:
        # numpy.digitize(x, edges): edges <= x (left) and edges < x (right); ties spread between them.
        left = sum(1 for e in edges if e <= x)
        right = sum(1 for e in edges if e < x)
        return int(math.ceil(rng.random() * (left - right) + right)) if left != right else left

    # Digitise in a fixed order (ascending token id) so the seeded spread is reproducible.
    binned = {tok: max(1, min(n_bins - 1, digitize(logged[tok]))) for tok in sorted(logged)}
    ranked = sorted(binned.items(), key=lambda item: (-item[1], -logged[item[0]], item[0]))
    kept = ranked[:max_genes]
    return {
        "tokens": [vocabulary.cls_id] + [tok for tok, _ in kept],
        "values": [0.0] + [float(b) for _, b in kept],
        "gene_symbols": [symbols[tok] for tok, _ in kept],
        "n_detected": sum(1 for v in counts.values() if v > 0),
        "n_encoded": len(resolved),
        "n_kept": len(kept),
        "n_truncated": max(0, len(ranked) - len(kept)),
        "unknown_genes": sorted(set(unknown)),
        "library_size": library,
        "n_bins": n_bins,
        "bin_edges": edges,
    }


def validate_inputs(
    cells: Sequence[Mapping[str, float]],
    vocabulary: GeneVocabulary,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: encode every cell and return the input manifest (schema, observations, verdict).

    Rejection is reported by raising exactly as ``embed``/``classify`` would.
    """
    checked, ids = _check_cells(cells, names)
    rows = []
    for cid, counts in zip(ids, checked, strict=True):
        encoded = bin_expression(counts, vocabulary)
        rows.append(
            {
                "id": cid,
                "detected_genes": encoded["n_detected"],
                "encoded_genes": encoded["n_encoded"],
                "tokens_kept": encoded["n_kept"],
                "genes_truncated": encoded["n_truncated"],
                "unknown_genes": len(encoded["unknown_genes"]),
                "library_size": encoded["library_size"],
            }
        )
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": rows,
        "n_cells": len(checked),
        "max_tokens_observed": 1 + max(row["tokens_kept"] for row in rows),
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "requires_remote_code": False,
    }


def _softmax(logits: Sequence[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


def _pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return None
    return cov / math.sqrt(va * vb)


def build_model(config: Mapping[str, Any], vocabulary: GeneVocabulary) -> Any:
    """The scGPT encoder on plain torch modules, named so the pinned checkpoint loads strictly.

    Follows bowang-lab/scGPT `model.py` (`TransformerModel` with `input_emb_style="continuous"`,
    `cell_emb_style="cls"`, post-norm `nn.TransformerEncoderLayer` with ReLU, `ContinuousValueEncoder`
    with clamp and ReLU, `ExprDecoder` with LeakyReLU). torch is imported here, not at module import.
    """
    import torch
    from torch import nn

    class ScGPTEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            embsize, d_hid = int(config["embsize"]), int(config["d_hid"])
            dropout = float(config.get("dropout", 0.0))
            self.gene_encoder = nn.ModuleDict(
                {
                    "embedding": nn.Embedding(
                        int(config["vocab_size"]), embsize, padding_idx=vocabulary.pad_id
                    ),
                    "enc_norm": nn.LayerNorm(embsize),
                }
            )
            self.value_encoder = nn.ModuleDict(
                {
                    "linear1": nn.Linear(1, embsize),
                    "linear2": nn.Linear(embsize, embsize),
                    "norm": nn.LayerNorm(embsize),
                    "dropout": nn.Dropout(dropout),
                }
            )
            layer = nn.TransformerEncoderLayer(
                d_model=embsize,
                nhead=int(config["nhead"]),
                dim_feedforward=d_hid,
                dropout=dropout,
                activation="relu",
                batch_first=True,
                norm_first=False,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=int(config["nlayers"]), enable_nested_tensor=False
            )
            self.expr_decoder = nn.ModuleDict(
                {
                    "fc": nn.Sequential(
                        nn.Linear(embsize, embsize),
                        nn.LeakyReLU(),
                        nn.Linear(embsize, embsize),
                        nn.LeakyReLU(),
                        nn.Linear(embsize, 1),
                    )
                }
            )
            self.value_clamp = VALUE_CLAMP

        def forward(self, input_ids: Any, values: Any, padding_mask: Any) -> dict[str, Any]:
            gene = self.gene_encoder["enc_norm"](self.gene_encoder["embedding"](input_ids))
            v = torch.clamp(values.unsqueeze(-1), max=self.value_clamp)
            v = torch.relu(self.value_encoder["linear1"](v))
            v = self.value_encoder["norm"](self.value_encoder["linear2"](v))
            v = self.value_encoder["dropout"](v)
            hidden = self.transformer(gene + v, src_key_padding_mask=padding_mask)
            return {"cell_emb": hidden[:, 0], "pred": self.expr_decoder["fc"](hidden).squeeze(-1)}

    return ScGPTEncoder()


@dataclass
class ScGPTPipeline:
    """scGPT pipeline: `embed` and `predict_masked` always; `classify` after `adapt` or `from_artifact`."""

    _embedder: Callable[[list[dict[str, float]]], list[list[float]]]
    vocabulary: GeneVocabulary
    device: str
    load_warnings: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    _classifier: Callable[[list[dict[str, float]]], list[list[float]]] | None = None
    model: Any = None
    config: dict[str, Any] = field(default_factory=dict)
    classifier_model: Any = None
    weights_dir: Path | None = None
    adaptation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> ScGPTPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError(f"no snapshot manifest at {root} and allow_download={allow_download}")
        # Stage and verify before importing model libraries (RTM-001).
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        config = json.loads((root / CONFIG_NAME).read_text(encoding="utf-8"))
        expected = {
            "embsize": (config.get("embsize"), EMBSIZE),
            "nlayers": (config.get("nlayers"), NLAYERS),
            "nhead": (config.get("nhead"), NHEAD),
            "d_hid": (config.get("d_hid"), D_HID),
            "max_seq_len": (config.get("max_seq_len"), MAX_SEQ_LEN),
            "vocab_size": (config.get("vocab_size"), VOCAB_SIZE),
            "input_emb_style": (config.get("input_emb_style"), "continuous"),
            "cell_emb_style": (config.get("cell_emb_style"), "cls"),
            "norm_scheme": (config.get("norm_scheme"), "post"),
            "explicit_zero_prob": (config.get("explicit_zero_prob"), False),
        }
        drift = {k: v for k, v in expected.items() if v[0] != v[1]}
        if drift:
            raise ValueError(f"pinned config.json disagrees with the package constants: {drift}")
        vocabulary = load_gene_vocabulary(root)

        import torch
        from safetensors.torch import load_file

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = build_model(config, vocabulary)
            state = load_file(str(root / WEIGHTS_NAME))
            model.load_state_dict(state, strict=True)
            del state
        model = model.to(resolved_device).eval()
        messages = [f"{w.category.__name__}: {w.message}" for w in caught]
        pipe = cls(
            cls._make_embedder(model, vocabulary, resolved_device), vocabulary, resolved_device, messages
        )
        pipe.model, pipe.config, pipe.weights_dir = model, dict(config), root
        return pipe

    # -- backends ---------------------------------------------------------------------------------

    @staticmethod
    def _batch(
        cells: list[dict[str, float]],
        vocabulary: GeneVocabulary,
        device: str,
        mask: Mapping[int, set[int]] | None = None,
    ) -> tuple[Any, Any, Any, list[dict[str, Any]]]:
        """Encode cells, pad to the longest, and return (input_ids, values, padding_mask, encodings).

        `mask` maps a cell index to the set of *positions* whose value is replaced by MASK_VALUE.
        """
        import torch

        encoded = [bin_expression(c, vocabulary) for c in cells]
        width = max(len(e["tokens"]) for e in encoded)
        ids = torch.full((len(encoded), width), vocabulary.pad_id, dtype=torch.long)
        vals = torch.zeros((len(encoded), width), dtype=torch.float32)
        pad = torch.ones((len(encoded), width), dtype=torch.bool)
        for i, e in enumerate(encoded):
            n = len(e["tokens"])
            ids[i, :n] = torch.tensor(e["tokens"], dtype=torch.long)
            row = list(e["values"])
            for pos in (mask or {}).get(i, set()):
                row[pos] = MASK_VALUE
            vals[i, :n] = torch.tensor(row, dtype=torch.float32)
            pad[i, :n] = False
        return ids.to(device), vals.to(device), pad.to(device), encoded

    @classmethod
    def _make_embedder(
        cls, model: Any, vocabulary: GeneVocabulary, device: str
    ) -> Callable[[list[dict[str, float]]], list[list[float]]]:
        import torch

        def embedder(cells: list[dict[str, float]]) -> list[list[float]]:
            ids, vals, pad, _ = cls._batch(cells, vocabulary, device)
            # no_grad, not inference_mode: tensors produced here must stay usable by a later
            # training epoch that shares this module.
            with torch.no_grad():
                out = model(ids, vals, pad)
            return out["cell_emb"].float().cpu().tolist()

        return embedder

    @classmethod
    def _make_classifier(
        cls, clf: Any, vocabulary: GeneVocabulary, device: str
    ) -> Callable[[list[dict[str, float]]], list[list[float]]]:
        import torch

        def classifier(cells: list[dict[str, float]]) -> list[list[float]]:
            ids, vals, pad, _ = cls._batch(cells, vocabulary, device)
            with torch.no_grad():
                logits = clf(ids, vals, pad)
            return logits.float().cpu().tolist()

        return classifier

    # -- public stages ----------------------------------------------------------------------------

    def embed(
        self, cells: Sequence[Mapping[str, float]], *, names: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """`<cls>` output of the encoder per cell (EMBSIZE floats each); order-invariant over genes."""
        checked, ids = _check_cells(cells, names)
        vectors = self._embedder(checked)
        if len(vectors) != len(checked) or any(len(v) != EMBSIZE for v in vectors):
            raise RuntimeError("backend returned embeddings of the wrong shape")
        encodings = [bin_expression(c, self.vocabulary) for c in checked]
        return {
            "ids": ids,
            "embeddings": [[float(x) for x in v] for v in vectors],
            "dimension": EMBSIZE,
            "pooling": "<cls> token output of the last encoder layer (cell_emb_style 'cls')",
            "unit": "one vector per cell; representations, not cell-state predictions",
            "tokens": [e["n_kept"] + 1 for e in encodings],
            "unknown_genes": [len(e["unknown_genes"]) for e in encodings],
            "n_cells": len(checked),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def predict_masked(
        self,
        cells: Sequence[Mapping[str, float]],
        *,
        names: Sequence[str] | None = None,
        mask_fraction: float = 0.15,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Masked-expression prediction — scGPT's pretraining objective, run as a functional check.

        A seeded fraction of each cell's gene positions receives MASK_VALUE; the decoder predicts a
        bin for every position, and the masked ones are compared with the true bins (mean absolute
        error and Pearson correlation per cell). A correct re-implementation of the encoder should
        predict masked bins far better than chance; a wrong one produces noise.
        """
        if self.model is None:
            raise RuntimeError("predict_masked requires a pipeline built by from_pretrained")
        if not 0.0 < float(mask_fraction) <= 0.5:
            raise ValueError("mask_fraction must be in (0, 0.5]")
        import random

        import torch

        checked, ids = _check_cells(cells, names)
        rng = random.Random(seed)
        pre = [bin_expression(c, self.vocabulary) for c in checked]
        mask: dict[int, set[int]] = {}
        for i, e in enumerate(pre):
            positions = list(range(1, len(e["tokens"])))  # never mask <cls>
            k = max(1, int(round(len(positions) * float(mask_fraction))))
            mask[i] = set(rng.sample(positions, k))
        input_ids, vals, pad, encoded = self._batch(checked, self.vocabulary, self.device, mask)
        with torch.no_grad():
            out = self.model(input_ids, vals, pad)
        pred = out["pred"].float().cpu().tolist()
        rows = []
        for i, (cid, e) in enumerate(zip(ids, encoded, strict=True)):
            positions = sorted(mask[i])
            truth = [e["values"][p] for p in positions]
            guess = [pred[i][p] for p in positions]
            mae = sum(abs(t - g) for t, g in zip(truth, guess, strict=True)) / len(positions)
            rows.append(
                {
                    "id": cid,
                    "masked_positions": len(positions),
                    "mean_absolute_error_bins": round(mae, 4),
                    "pearson": (None if (r := _pearson(truth, guess)) is None else round(r, 4)),
                    "mean_true_bin": round(sum(truth) / len(truth), 3),
                    "mean_predicted_bin": round(sum(guess) / len(guess), 3),
                }
            )
        return {
            "cells": rows,
            "mask_fraction": float(mask_fraction),
            "mask_value": MASK_VALUE,
            "n_bins": N_BINS,
            "note": (
                "functional check of the re-implemented encoder against scGPT's masked-expression objective; "
                "bins are per-cell quantiles, so chance-level prediction sits near the mean bin"
            ),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def predict_masked_genes(
        self,
        cells: Sequence[Mapping[str, float]],
        genes: Sequence[str],
        *,
        names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Mask the named genes in each cell and report the decoder's mean predicted bin against the
        true mean bin — the lineage-conditioning probe: does the decoder predict a programme's genes
        high where the rest of the cell expresses that programme?"""
        if self.model is None:
            raise RuntimeError("predict_masked_genes requires a pipeline built by from_pretrained")
        if isinstance(genes, str) or not genes:
            raise ValueError("genes must be a non-empty list of gene symbols")
        import torch

        checked, ids = _check_cells(cells, names)
        wanted = {str(g) for g in genes}
        pre = [bin_expression(c, self.vocabulary) for c in checked]
        mask: dict[int, set[int]] = {}
        for i, e in enumerate(pre):
            mask[i] = {k + 1 for k, symbol in enumerate(e["gene_symbols"]) if symbol in wanted}
            if not mask[i]:
                raise ValueError(f"{ids[i]}: none of the requested genes is detected in this cell")
        input_ids, vals, pad, encoded = self._batch(checked, self.vocabulary, self.device, mask)
        with torch.no_grad():
            pred = self.model(input_ids, vals, pad)["pred"].float().cpu().tolist()
        rows = []
        for i, (cid, e) in enumerate(zip(ids, encoded, strict=True)):
            positions = sorted(mask[i])
            rows.append(
                {
                    "id": cid,
                    "masked_genes": len(positions),
                    "predicted_mean_bin": round(sum(pred[i][p] for p in positions) / len(positions), 3),
                    "true_mean_bin": round(sum(e["values"][p] for p in positions) / len(positions), 3),
                }
            )
        n = len(rows)
        return {
            "cells": rows,
            "predicted_mean_bin": round(sum(r["predicted_mean_bin"] for r in rows) / n, 3),
            "true_mean_bin": round(sum(r["true_mean_bin"] for r in rows) / n, 3),
            "genes_requested": len(wanted),
            "mask_value": MASK_VALUE,
        }

    def nearest_centroid_evaluate(
        self,
        train_records: Sequence[Mapping[str, Any]],
        eval_records: Sequence[Mapping[str, Any]],
        *,
        classes: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """The frozen embedding's own separability, with no gradient training: centre the `<cls>`
        embeddings on the training mean, fit one centroid per class on the training split only
        (SPL8), and assign each evaluation cell to the nearest centroid by cosine.

        scGPT's raw `<cls>` vectors share a large common component (cosine ~0.95 between unrelated
        cells), so centring is what makes the geometry readable; the training mean is the only
        fitted state besides the centroids. This is the zero-training baseline the adapted head must
        be read against.
        """
        from .metrics import classification_metrics
        from .samples import validate_dataset

        manifest = validate_dataset(train_records, classes=classes)
        class_list = list(manifest["classes"])
        validate_dataset(eval_records, classes=class_list, min_records=2, min_per_class=1)

        def embed_all(records: Sequence[Mapping[str, Any]]) -> list[list[float]]:
            out: list[list[float]] = []
            for start in range(0, len(records), MAX_CELLS_PER_CALL):
                chunk = records[start : start + MAX_CELLS_PER_CALL]
                out.extend(
                    self.embed([r["counts"] for r in chunk], names=[r["id"] for r in chunk])["embeddings"]
                )
            return out

        train_emb = embed_all(train_records)
        mean = [sum(e[d] for e in train_emb) / len(train_emb) for d in range(EMBSIZE)]

        def centred(e: Sequence[float]) -> list[float]:
            return [x - m for x, m in zip(e, mean, strict=True)]

        def cosine(a: Sequence[float], b: Sequence[float]) -> float:
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(x * x for x in b)) or 1.0
            return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)

        centroids: dict[str, list[float]] = {}
        for c in class_list:
            rows = [centred(e) for e, r in zip(train_emb, train_records, strict=True) if r["label"] == c]
            centroids[c] = [sum(r[d] for r in rows) / len(rows) for d in range(EMBSIZE)]
        predicted: list[str] = []
        scores: list[list[float]] = []
        for e in embed_all(eval_records):
            sims = [cosine(centred(e), centroids[c]) for c in class_list]
            predicted.append(class_list[max(range(len(sims)), key=sims.__getitem__)])
            scores.append(_softmax([s * 10.0 for s in sims]))
        metrics = classification_metrics([r["label"] for r in eval_records], predicted, scores, class_list)
        return {
            "baseline": "nearest class centroid on centred frozen <cls> embeddings (no training)",
            "fitted_on": "training split only (mean and centroids)",
            **metrics,
        }

    def classify(
        self, cells: Sequence[Mapping[str, float]], *, names: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Class scores and argmax label per cell; requires a prior `adapt` or `from_artifact`."""
        if self._classifier is None or not self.classes:
            raise RuntimeError(
                "classify requires an adapted head: call adapt(...) or load from_artifact(...) first"
            )
        checked, ids = _check_cells(cells, names)
        logits = self._classifier(checked)
        predictions = []
        for cid, row in zip(ids, logits, strict=True):
            if len(row) != len(self.classes):
                raise RuntimeError("backend returned a logits row that does not match the class list")
            scores = _softmax(row)
            best = max(range(len(scores)), key=scores.__getitem__)
            predictions.append(
                {
                    "id": cid,
                    "label": self.classes[best],
                    "score": scores[best],
                    "scores": dict(zip(self.classes, scores, strict=True)),
                }
            )
        return {
            "predictions": predictions,
            "classes": list(self.classes),
            "decision_rule": (
                "argmax over softmax(logits); scores are softmax outputs, not calibrated probabilities"
            ),
            "n_cells": len(checked),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "adaptation": dict(self.adaptation),
        }

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Held-out cell-state classification metrics (see metrics.classification_metrics)."""
        from .metrics import classification_metrics
        from .samples import validate_dataset

        validate_dataset(records, classes=self.classes, min_records=2, min_per_class=1)
        predicted: list[str] = []
        scores: list[list[float]] = []
        for start in range(0, len(records), MAX_CELLS_PER_CALL):
            chunk = records[start : start + MAX_CELLS_PER_CALL]
            result = self.classify([r["counts"] for r in chunk], names=[r["id"] for r in chunk])
            for p in result["predictions"]:
                predicted.append(p["label"])
                scores.append([p["scores"][c] for c in self.classes])
        return classification_metrics([r["label"] for r in records], predicted, scores, self.classes)

    def _build_classifier(self, n_classes: int) -> Any:
        """A fresh copy of the encoder with a linear head on its `<cls>` output."""
        import copy

        import torch

        class CellClassifier(torch.nn.Module):
            def __init__(self, encoder: Any, n: int) -> None:
                super().__init__()
                self.encoder = encoder
                self.head = torch.nn.Linear(EMBSIZE, n)

            def forward(self, input_ids: Any, values: Any, padding_mask: Any) -> Any:
                return self.head(self.encoder(input_ids, values, padding_mask)["cell_emb"])

        return CellClassifier(copy.deepcopy(self.model), n_classes)

    def adapt(
        self,
        train_records: Sequence[Mapping[str, Any]],
        val_records: Sequence[Mapping[str, Any]] | None = None,
        *,
        classes: Sequence[str] | None = None,
        epochs: int = 4,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        trainable_layers: int = 1,
        weight_decay: float = 0.01,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Bounded gradient fine-tuning of a cell-state classification head on the verified base.

        Copies the encoder, puts a newly initialised linear head over its `<cls>` output, freezes
        everything except the head and the last `trainable_layers` transformer layers, and runs
        AdamW for `epochs` passes. With `trainable_layers=0` the frozen encoder's output is computed
        once and only the head trains. Validation records are monitored per epoch only; the final
        epoch's weights are kept (no selection).
        """
        if self.model is None or self.weights_dir is None:
            raise RuntimeError("adapt requires a pipeline built by from_pretrained (no loaded base model)")
        from .samples import validate_dataset

        if not 1 <= int(epochs) <= 50:
            raise ValueError("epochs must be in 1..50 (tutorial-scale adaptation)")
        if not 1 <= int(batch_size) <= MAX_CELLS_PER_CALL:
            raise ValueError(f"batch_size must be in 1..{MAX_CELLS_PER_CALL}")
        if not 0 <= int(trainable_layers) <= NLAYERS:
            raise ValueError(f"trainable_layers must be in 0..{NLAYERS} (the encoder has that many)")
        train_manifest = validate_dataset(train_records, classes=classes)
        class_list = list(train_manifest["classes"])
        if val_records is not None:
            validate_dataset(val_records, classes=class_list, min_records=2, min_per_class=1)

        import random

        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        clf = self._build_classifier(len(class_list)).to(self.device)
        for p in clf.parameters():
            p.requires_grad = False
        layers = clf.encoder.transformer.layers
        for layer in layers[len(layers) - int(trainable_layers) :] if trainable_layers else []:
            for p in layer.parameters():
                p.requires_grad = True
        for p in clf.head.parameters():
            p.requires_grad = True
        trainable = [n for n, p in clf.named_parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in clf.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in clf.parameters())
        optimizer = torch.optim.AdamW(
            [p for p in clf.parameters() if p.requires_grad], lr=learning_rate, weight_decay=weight_decay
        )
        label_index = {c: i for i, c in enumerate(class_list)}
        examples = [(dict(r["counts"]), label_index[r["label"]]) for r in train_records]
        self.classes = class_list
        self.classifier_model = clf
        self._classifier = self._make_classifier(clf, self.vocabulary, self.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        cached: list[Any] | None = None
        if not trainable_layers:
            clf.eval()
            cached = []
            with torch.no_grad():
                for start in range(0, len(examples), MAX_CELLS_PER_CALL):
                    ids, vals, pad, _ = self._batch(
                        [c for c, _ in examples[start : start + MAX_CELLS_PER_CALL]],
                        self.vocabulary,
                        self.device,
                    )
                    cached.extend(clf.encoder(ids, vals, pad)["cell_emb"])

        history: list[dict[str, Any]] = []
        for epoch in range(1, int(epochs) + 1):
            clf.train()
            order = list(range(len(examples)))
            random.shuffle(order)
            total_loss, n_batches = 0.0, 0
            for start in range(0, len(order), int(batch_size)):
                idx = order[start : start + int(batch_size)]
                labels = torch.tensor([examples[i][1] for i in idx], device=self.device)
                optimizer.zero_grad()
                if cached is not None:
                    logits = clf.head(torch.stack([cached[i] for i in idx]))
                else:
                    ids, vals, pad, _ = self._batch(
                        [examples[i][0] for i in idx], self.vocabulary, self.device
                    )
                    logits = clf(ids, vals, pad)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1
            clf.eval()
            entry: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": round(total_loss / max(1, n_batches), 6),
                "n_batches": n_batches,
            }
            if val_records:
                val = self.evaluate(val_records)
                entry["val_accuracy"] = val["accuracy"]
                entry["val_macro_f1"] = val["macro_f1"]
            history.append(entry)
        clf.eval()
        self.adaptation = {
            "method": "gradient fine-tuning (AdamW) of a linear head over the <cls> cell embedding"
            + (
                f" and the last {int(trainable_layers)} encoder layer(s)"
                if trainable_layers
                else " (frozen encoder; embeddings computed once)"
            ),
            "classes": class_list,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "weight_decay": float(weight_decay),
            "trainable_layers": int(trainable_layers),
            "seed": int(seed),
            "precision": "float32",
            "trainable_parameters": int(n_trainable),
            "total_parameters": int(n_total),
            "trainable_parameter_names": trainable,
            "train_records": len(train_records),
            "val_records": len(val_records) if val_records else 0,
            "selection": "final epoch kept; validation metrics are monitoring only",
            "history": history,
        }
        return dict(self.adaptation)

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Export the trainable tensors as safetensors plus a JSON manifest binding them to the base."""
        if self.classifier_model is None or not self.classes:
            raise RuntimeError("save_artifact requires an adapted head (call adapt first)")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adaptation.get("trainable_parameter_names", []))
        state = self.classifier_model.state_dict()
        tensors = {k: v.detach().cpu().contiguous() for k, v in state.items() if k in names}
        if not tensors:
            raise RuntimeError("no trainable tensors recorded; nothing to export")
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path))
        digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "license": MODEL_LICENSE},
            "requires_remote_code": False,
            "classes": list(self.classes),
            "files": [
                {"path": ARTIFACT_WEIGHTS_NAME, "bytes": weights_path.stat().st_size, "sha256": digest}
            ],
            "tensors": sorted(tensors),
            "serving_state_tensors": [],
            "adaptation": {k: v for k, v in self.adaptation.items() if k != "trainable_parameter_names"},
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Rebuild the classification head from an exported artifact (manifest verified before loading)."""
        if self.model is None or self.weights_dir is None:
            raise RuntimeError("load_artifact requires a pipeline built by from_pretrained")
        art = Path(artifact_dir)
        manifest_path = art / ARTIFACT_MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        base = manifest.get("base_model", {})
        if (base.get("model_id"), base.get("model_revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError(f"artifact was trained on {base}, this package pins {MODEL_ID}@{MODEL_REVISION}")
        classes = [str(c) for c in manifest.get("classes", [])]
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("artifact manifest must list at least two unique classes")
        for entry in manifest["files"]:
            fp = art / entry["path"]
            if not fp.is_file():
                raise FileNotFoundError(f"artifact file missing: {fp}")
            if fp.stat().st_size != entry["bytes"]:
                raise ValueError(f"{entry['path']}: size {fp.stat().st_size} != manifest {entry['bytes']}")
            if hashlib.sha256(fp.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"{entry['path']}: sha256 mismatch against the artifact manifest")
        from safetensors.torch import load_file

        clf = self._build_classifier(len(classes))
        tensors = load_file(str(art / ARTIFACT_WEIGHTS_NAME))
        if set(tensors) != set(manifest.get("tensors", [])):
            raise ValueError("artifact tensors do not match the names listed in its manifest")
        _missing, unexpected = clf.load_state_dict(tensors, strict=False)
        if unexpected:
            raise ValueError(
                f"artifact carries tensors the base architecture does not have: {sorted(unexpected)[:5]}"
            )
        clf = clf.to(self.device).eval()
        self.classes = classes
        self.classifier_model = clf
        self._classifier = self._make_classifier(clf, self.vocabulary, self.device)
        self.adaptation = {**manifest.get("adaptation", {}), "loaded_from_artifact": str(art)}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> ScGPTPipeline:
        """Verified base snapshot + exported adapter, ready for `classify`."""
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe
