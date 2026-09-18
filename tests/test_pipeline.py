"""Offline pipeline tests: identity, snapshot verification/staging (Hub files and the Dataverse
vocabulary), vocabulary loading, expression binning, input checks, and the embed/classify/artifact
contracts with injected backends (no weights, no torch)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from scgpt_single_cell_pipeline import (
    CONFIG_NAME,
    D_HID,
    DEFAULT_WEIGHTS_DIR,
    EMBSIZE,
    MANIFEST_NAME,
    MAX_CELLS_PER_CALL,
    MAX_GENES_PER_CELL,
    MAX_SEQ_LEN,
    MIN_DETECTED_GENES,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    N_BINS,
    NHEAD,
    NLAYERS,
    SPECIAL_TOKENS,
    VOCAB_NAME,
    VOCAB_SIZE,
    WEIGHTS_NAME,
    GeneVocabulary,
    ScGPTPipeline,
    bin_expression,
    load_gene_vocabulary,
    stage_missing_files,
    validate_inputs,
    verify_snapshot,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "weights" / MODEL_KEY / MANIFEST_NAME

# A 20-symbol toy vocabulary with the three specials at the top, like the real file.
_SYMS = [f"GENE{i}" for i in range(1, 21)]
VOCAB = GeneVocabulary(
    tokens={**{s: i for i, s in enumerate(_SYMS)}, "<pad>": 20, "<cls>": 21, "<eoc>": 22},
    pad_id=20,
    cls_id=21,
    eoc_id=22,
)
CELL = {s: float(i + 1) for i, s in enumerate(_SYMS)}  # strictly increasing counts, no ties


def _wide(value: int = 5) -> dict[str, float]:
    """A cell with just enough detected genes to pass the MIN_DETECTED_GENES rule."""
    return dict.fromkeys(_SYMS[:MIN_DETECTED_GENES], float(value))


def _fake(classes=None, logits=None):
    pipe = ScGPTPipeline(lambda rows: [[float(len(r))] * EMBSIZE for r in rows], VOCAB, "cpu")
    if classes is not None:
        pipe.classes = list(classes)
        rows = logits or [[0.0, 1.0]] * MAX_CELLS_PER_CALL
        pipe._classifier = lambda cells: [rows[i] for i in range(len(cells))]
    return pipe


def test_identity_constants_and_manifest():
    assert re.fullmatch(r"[0-9a-f]{40}", MODEL_REVISION)
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert MAX_GENES_PER_CELL == MAX_SEQ_LEN - 1
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert manifest["modelId"] == MODEL_ID
        assert manifest["revision"] == MODEL_REVISION
        paths = {f["path"]: f for f in manifest["files"]}
        assert {CONFIG_NAME, WEIGHTS_NAME, VOCAB_NAME} <= set(paths)
        # The vocabulary is the one non-Hub entry and must say where it comes from.
        assert paths[VOCAB_NAME]["source"].startswith("https://dataverse.harvard.edu/api/access/datafile/")
    config = REPO / "weights" / MODEL_KEY / CONFIG_NAME
    if config.is_file():
        cfg = json.loads(config.read_text(encoding="utf-8"))
        assert (cfg["embsize"], cfg["nlayers"], cfg["nhead"], cfg["d_hid"]) == (
            EMBSIZE,
            NLAYERS,
            NHEAD,
            D_HID,
        )
        assert cfg["max_seq_len"] == MAX_SEQ_LEN and cfg["vocab_size"] == VOCAB_SIZE
        assert cfg["input_emb_style"] == "continuous" and cfg["cell_emb_style"] == "cls"
        assert "auto_map" not in cfg  # no remote code: the architecture is re-implemented here


def test_real_vocabulary_if_present():
    path = REPO / "weights" / MODEL_KEY / VOCAB_NAME
    if not path.is_file():
        pytest.skip("vocab.json not staged")
    vocab = load_gene_vocabulary(path.parent)
    assert len(vocab.tokens) == VOCAB_SIZE
    assert (vocab.pad_id, vocab.cls_id, vocab.eoc_id) == (60694, 60695, 60696)
    assert vocab.resolve("GAPDH") == 9833 and vocab.resolve("gapdh") == 9833
    assert vocab.resolve("NOT_A_GENE_XYZ") is None
    assert vocab.resolve("<pad>") == vocab.pad_id and vocab.resolve("<PAD>") is None


def _write_snapshot(
    tmp_path: Path, content: bytes, sha256: str, revision: str = MODEL_REVISION, include_vocab: bool = True
) -> Path:
    files = [{"path": CONFIG_NAME, "bytes": len(content), "sha256": sha256}]
    (tmp_path / CONFIG_NAME).write_bytes(content)
    (tmp_path / WEIGHTS_NAME).write_bytes(content)
    files.append({"path": WEIGHTS_NAME, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    if include_vocab:
        (tmp_path / VOCAB_NAME).write_bytes(content)
        files.append(
            {"path": VOCAB_NAME, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        )
    manifest = {"modelId": MODEL_ID, "revision": revision, "files": files}
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_verify_snapshot_accepts_and_rejects(tmp_path):
    content = b'{"embsize": 512}'
    good = hashlib.sha256(content).hexdigest()
    assert verify_snapshot(_write_snapshot(tmp_path, content, good))["revision"] == MODEL_REVISION
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(_write_snapshot(tmp_path, content, "0" * 64))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(_write_snapshot(tmp_path, content, good, revision="0" * 40))
    root = _write_snapshot(tmp_path, content, good)
    (root / CONFIG_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_snapshot(root)


def test_verify_snapshot_refuses_a_manifest_without_the_vocabulary(tmp_path):
    content = b"{}"
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest(), include_vocab=False)
    with pytest.raises(ValueError, match=VOCAB_NAME):
        verify_snapshot(root)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path):
    content = b"{}"
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest())
    (root / VOCAB_NAME).unlink()
    (root / WEIGHTS_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    fetched = []

    def downloader(rel, dst):
        fetched.append(rel)
        (dst / rel).write_bytes(content)

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == [WEIGHTS_NAME, VOCAB_NAME]
    assert fetched == [WEIGHTS_NAME, VOCAB_NAME]
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == []
    assert verify_snapshot(root)["revision"] == MODEL_REVISION


def test_stage_refuses_manifest_for_another_model(tmp_path):
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps({"modelId": "other/model", "revision": MODEL_REVISION, "files": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True)


# --- vocabulary loading ---------------------------------------------------------------------------


def test_load_gene_vocabulary_rejections(tmp_path):
    (tmp_path / VOCAB_NAME).write_text(json.dumps({"A": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"map {VOCAB_SIZE} symbols"):
        load_gene_vocabulary(tmp_path)
    full = {f"G{i}": i for i in range(VOCAB_SIZE)}
    (tmp_path / VOCAB_NAME).write_text(json.dumps(full), encoding="utf-8")
    with pytest.raises(ValueError, match="special tokens"):
        load_gene_vocabulary(tmp_path)
    full = {f"G{i}": i for i in range(VOCAB_SIZE - 3)}
    full.update({"<pad>": VOCAB_SIZE - 3, "<cls>": VOCAB_SIZE - 2, "<eoc>": VOCAB_SIZE - 2})  # duplicate id
    (tmp_path / VOCAB_NAME).write_text(json.dumps(full), encoding="utf-8")
    with pytest.raises(ValueError, match="no gaps or duplicates"):
        load_gene_vocabulary(tmp_path)


def test_special_tokens_are_declared():
    assert set(SPECIAL_TOKENS) == {"<pad>", "<cls>", "<eoc>"}
    assert set(VOCAB.gene_symbols) == set(_SYMS)


# --- expression binning ---------------------------------------------------------------------------


def test_bin_expression_layout_and_range():
    enc = bin_expression(CELL, VOCAB)
    assert enc["tokens"][0] == VOCAB.cls_id and enc["values"][0] == 0.0
    assert len(enc["tokens"]) == len(enc["values"]) == len(CELL) + 1
    assert all(1 <= v <= N_BINS - 1 for v in enc["values"][1:])
    assert enc["n_bins"] == N_BINS and len(enc["bin_edges"]) == N_BINS - 1
    # highest count -> highest bin, kept first; the ranking is by bin then normalised value
    assert enc["gene_symbols"][0] == "GENE20" and enc["values"][1] == N_BINS - 1
    assert enc["gene_symbols"][-1] == "GENE1" and enc["values"][-1] == 1.0


def test_bin_expression_is_scale_invariant_and_deterministic():
    a = bin_expression(CELL, VOCAB)
    b = bin_expression({g: v * 37.0 for g, v in CELL.items()}, VOCAB)
    assert a["values"] == b["values"] and a["tokens"] == b["tokens"]
    assert bin_expression(CELL, VOCAB) == a  # seeded tie-spreading is reproducible


def test_bin_expression_spreads_ties_within_their_quantile_block():
    tied = {**dict.fromkeys(_SYMS[:12], 1.0), **{s: float(10 + i) for i, s in enumerate(_SYMS[12:])}}
    enc = bin_expression(tied, VOCAB)
    ones = [v for s, v in zip(enc["gene_symbols"], enc["values"][1:], strict=True) if s in _SYMS[:12]]
    assert min(ones) >= 1 and max(ones) <= 30 and len(set(ones)) > 1  # spread, not one bin
    assert (
        bin_expression(tied, VOCAB, seed=7)["values"] != enc["values"]
    )  # a different seed spreads differently
    assert bin_expression(tied, VOCAB, seed=7) == bin_expression(tied, VOCAB, seed=7)


def test_bin_expression_reports_dropped_genes_and_truncation():
    cell = {**CELL, "NOT_IN_VOCAB": 5.0, "GENE1": 0.0}
    enc = bin_expression(cell, VOCAB)
    assert enc["unknown_genes"] == ["NOT_IN_VOCAB"]
    assert "GENE1" not in enc["gene_symbols"]  # zero counts are never tokenised
    assert enc["n_detected"] == 20 and enc["n_encoded"] == 19
    small = bin_expression(CELL, VOCAB, max_genes=5)
    assert small["n_kept"] == 5 and small["n_truncated"] == 15 and len(small["tokens"]) == 6
    assert small["gene_symbols"] == ["GENE20", "GENE19", "GENE18", "GENE17", "GENE16"]


def test_bin_expression_refuses_an_unencodable_cell():
    with pytest.raises(ValueError, match="none of the detected genes is in the scGPT vocabulary"):
        bin_expression({"NOPE1": 1.0, "NOPE2": 2.0}, VOCAB)


# --- input validation -----------------------------------------------------------------------------


def test_validate_inputs_manifest_and_rejections():
    manifest = validate_inputs([CELL, _wide()], VOCAB, names=["a", "b"])
    assert manifest["verdict"] == "accepted"
    assert manifest["requires_remote_code"] is False
    assert [row["id"] for row in manifest["inputs"]] == ["a", "b"]
    assert manifest["inputs"][0]["tokens_kept"] == 20 and manifest["max_tokens_observed"] == 21
    with pytest.raises(TypeError, match="list of"):
        validate_inputs(CELL, VOCAB)
    with pytest.raises(ValueError, match="is empty"):
        validate_inputs([{}], VOCAB)
    with pytest.raises(TypeError, match="non-string gene key"):
        validate_inputs([{1: 2.0}], VOCAB)
    with pytest.raises(TypeError, match="must be a number"):
        validate_inputs([{"GENE1": "3"}], VOCAB)
    with pytest.raises(ValueError, match="finite non-negative"):
        validate_inputs([{**_wide(), "GENE1": -1.0}], VOCAB)
    with pytest.raises(ValueError, match=f"at least {MIN_DETECTED_GENES}"):
        validate_inputs([dict.fromkeys(_SYMS[:5], 1.0)], VOCAB)
    with pytest.raises(ValueError, match=f"1..{MAX_CELLS_PER_CALL}"):
        validate_inputs([CELL] * (MAX_CELLS_PER_CALL + 1), VOCAB)
    with pytest.raises(ValueError, match="exactly one id per cell"):
        validate_inputs([CELL, CELL], VOCAB, names=["a"])
    with pytest.raises(ValueError, match="names must be unique"):
        validate_inputs([CELL, CELL], VOCAB, names=["a", "a"])


# --- embed / classify / artifact contracts --------------------------------------------------------


def test_embed_contract_with_injected_backend():
    out = _fake().embed([CELL, _wide()], names=["a", "b"])
    assert out["ids"] == ["a", "b"] and out["dimension"] == EMBSIZE
    assert len(out["embeddings"][0]) == EMBSIZE
    assert out["tokens"] == [21, MIN_DETECTED_GENES + 1]
    assert out["model_revision"] == MODEL_REVISION


def test_embed_rejects_backend_shape_drift():
    pipe = ScGPTPipeline(lambda rows: [[0.0] * 3 for _ in rows], VOCAB, "cpu")
    with pytest.raises(RuntimeError, match="wrong shape"):
        pipe.embed([CELL])


def test_predict_masked_requires_a_loaded_model():
    with pytest.raises(RuntimeError, match="from_pretrained"):
        _fake().predict_masked([CELL])


def test_classify_requires_adaptation():
    with pytest.raises(RuntimeError, match="adapt"):
        _fake().classify([CELL])


def test_classify_contract_preserves_class_order():
    pipe = _fake(classes=["b-like", "t-like"], logits=[[2.0, 0.0], [0.0, 2.0]])
    out = pipe.classify([CELL, _wide()], names=["a", "b"])
    assert [p["label"] for p in out["predictions"]] == ["b-like", "t-like"]
    assert list(out["predictions"][0]["scores"]) == ["b-like", "t-like"]
    assert out["predictions"][0]["scores"]["b-like"] == pytest.approx(0.8808, abs=1e-3)
    assert "not calibrated" in out["decision_rule"]


def test_classify_rejects_logits_that_do_not_match_classes():
    pipe = _fake(classes=["a", "b", "c"], logits=[[0.0, 1.0]] * 4)
    with pytest.raises(RuntimeError, match="does not match the class list"):
        pipe.classify([CELL])


def test_save_and_load_artifact_require_a_loaded_model(tmp_path):
    pipe = _fake()
    with pytest.raises(RuntimeError, match="adapt"):
        pipe.save_artifact(tmp_path / "adapter")
    with pytest.raises(RuntimeError, match="from_pretrained"):
        pipe.load_artifact(tmp_path / "adapter")
    with pytest.raises(RuntimeError, match="from_pretrained"):
        pipe.adapt([], [])
