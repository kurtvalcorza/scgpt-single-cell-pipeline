"""Offline tests for the public validation-stage helpers (input manifest and dataset manifest)."""

from __future__ import annotations

from scgpt_single_cell_pipeline import (
    INPUT_SCHEMA,
    MAX_CELLS_PER_CALL,
    MAX_GENES_PER_CELL,
    MIN_DETECTED_GENES,
    MODEL_ID,
    MODEL_REVISION,
    GeneVocabulary,
    validate_inputs,
)

_SYMS = [f"GENE{i}" for i in range(12)]
VOCAB = GeneVocabulary(
    tokens={**{s: i for i, s in enumerate(_SYMS)}, "<pad>": 12, "<cls>": 13, "<eoc>": 14},
    pad_id=12,
    cls_id=13,
    eoc_id=14,
)
CELL = {s: 5.0 + i for i, s in enumerate(_SYMS)}


def test_validate_inputs_returns_manifest_with_schema_and_identity():
    manifest = validate_inputs([CELL], VOCAB, names=["c1"])
    assert manifest["verdict"] == "accepted" and manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["cells"] == [1, MAX_CELLS_PER_CALL]
    assert manifest["schema"]["genes_per_cell"] == [MIN_DETECTED_GENES, MAX_GENES_PER_CELL]
    assert "51 bins" in manifest["schema"]["preprocessing"]
    row = manifest["inputs"][0]
    assert row["id"] == "c1" and row["detected_genes"] == 12 and row["tokens_kept"] == 12
    assert row["unknown_genes"] == 0 and row["genes_truncated"] == 0
    assert manifest["n_cells"] == 1 and manifest["max_tokens_observed"] == 13  # + <cls>
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert manifest["requires_remote_code"] is False


def test_validate_inputs_default_ids_and_case_insensitive_symbols():
    manifest = validate_inputs([{**CELL, "gene0": 3.0, "MYSTERY": 1.0}], VOCAB)
    assert [row["id"] for row in manifest["inputs"]] == ["cell-0"]
    # `gene0` resolves onto GENE0 (upper-cased) and merges with it; MYSTERY is unknown and reported.
    assert manifest["inputs"][0]["unknown_genes"] == 1
    assert manifest["inputs"][0]["encoded_genes"] == 12
