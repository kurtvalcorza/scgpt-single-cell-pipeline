"""Offline tests for the cell dataset contract, the marker-programme sample, splits, metrics,
baselines, BYOD loaders and artifact manifests."""

from __future__ import annotations

import json

import pytest

from scgpt_single_cell_pipeline import (
    ARTIFACT_FORMAT,
    ARTIFACT_MANIFEST_NAME,
    B_CELL_PROGRAMME,
    BACKGROUND_GENES,
    DATASET_REPRESENTATION,
    LIBRARY_SIZE,
    MIN_DETECTED_GENES,
    MIN_RECORDS,
    MIN_RECORDS_PER_CLASS,
    MODEL_ID,
    MODEL_REVISION,
    PROGRAMME_GENES,
    SAMPLE_CLASSES,
    SAMPLE_SEED,
    SAMPLE_SIZE,
    T_CELL_PROGRAMME,
    GeneVocabulary,
    ScGPTPipeline,
    auroc,
    bin_expression,
    classification_metrics,
    dataset_digest,
    generate_sample_dataset,
    library_size_baseline,
    load_byod_dataset,
    majority_baseline,
    select_sample_genes,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)

# A stand-in vocabulary: the real marker symbols plus enough canonical-looking background symbols.
_BACKGROUND = [f"GENE{i:04d}" for i in range(BACKGROUND_GENES + 50)]
_ALL = [*T_CELL_PROGRAMME, *B_CELL_PROGRAMME, *_BACKGROUND]
VOCAB = GeneVocabulary(
    tokens={
        **{s: i for i, s in enumerate(_ALL)},
        "<pad>": len(_ALL),
        "<cls>": len(_ALL) + 1,
        "<eoc>": len(_ALL) + 2,
    },
    pad_id=len(_ALL),
    cls_id=len(_ALL) + 1,
    eoc_id=len(_ALL) + 2,
)


# --- sample dataset ---------------------------------------------------------------------------


def test_sample_genes_are_the_marker_programmes_plus_background():
    genes = select_sample_genes(VOCAB)
    assert genes["t_cell"] == list(T_CELL_PROGRAMME) and genes["b_cell"] == list(B_CELL_PROGRAMME)
    assert len(T_CELL_PROGRAMME) == len(B_CELL_PROGRAMME) == PROGRAMME_GENES
    assert not (set(T_CELL_PROGRAMME) & set(B_CELL_PROGRAMME))
    assert len(genes["background"]) == BACKGROUND_GENES
    assert not (set(genes["background"]) & (set(T_CELL_PROGRAMME) | set(B_CELL_PROGRAMME)))
    assert all(g.startswith("GENE") for g in genes["background"])  # canonical-symbol filter
    thin = GeneVocabulary(
        tokens={"CD3D": 0, "<pad>": 1, "<cls>": 2, "<eoc>": 3}, pad_id=1, cls_id=2, eoc_id=3
    )
    with pytest.raises(RuntimeError, match="marker genes absent"):
        select_sample_genes(thin)


def test_sample_dataset_is_deterministic_balanced_and_valid():
    a = generate_sample_dataset(VOCAB)
    assert a == generate_sample_dataset(VOCAB, seed=SAMPLE_SEED, size=SAMPLE_SIZE)
    assert generate_sample_dataset(VOCAB, seed=7) != a
    manifest = validate_dataset(a, vocabulary=VOCAB)
    assert manifest["verdict"] == "accepted"
    assert manifest["representation"] == DATASET_REPRESENTATION
    assert manifest["classes"] == list(SAMPLE_CLASSES)
    assert manifest["class_counts"] == {c: SAMPLE_SIZE // 2 for c in SAMPLE_CLASSES}
    assert manifest["digest"] == dataset_digest(a)
    assert manifest["encodable_genes"]["min"] == manifest["detected_genes"]["min"] == 348
    with pytest.raises(ValueError, match="even number"):
        generate_sample_dataset(VOCAB, size=7)


def test_sample_cells_share_library_size_and_genes_but_differ_in_programme_rank():
    records = generate_sample_dataset(VOCAB)
    totals = [sum(r["counts"].values()) for r in records]
    assert (max(totals) - min(totals)) / min(totals) < 0.01
    assert abs(sum(totals) / len(totals) - LIBRARY_SIZE) / LIBRARY_SIZE < 0.01
    first_b = next(r for r in records if r["label"] == "b-like")
    first_t = next(r for r in records if r["label"] == "t-like")
    assert set(first_b["counts"]) == set(first_t["counts"])  # same genes detected in both classes
    top_b = bin_expression(first_b["counts"], VOCAB)["gene_symbols"][:PROGRAMME_GENES]
    top_t = bin_expression(first_t["counts"], VOCAB)["gene_symbols"][:PROGRAMME_GENES]
    # The cell's own lineage programme is heavily over-represented at the top of the ranking (chance
    # would put ~1.7 of its 24 genes there; log-normal background genes share the top bins), and the
    # other lineage never appears there.
    assert len(set(top_b) & set(B_CELL_PROGRAMME)) >= 8 and not (set(top_b) & set(T_CELL_PROGRAMME))
    assert len(set(top_t) & set(T_CELL_PROGRAMME)) >= 8 and not (set(top_t) & set(B_CELL_PROGRAMME))


# --- dataset validation -----------------------------------------------------------------------


def _records(n_per_class=MIN_RECORDS, classes=("x", "y")):
    out = []
    for c_i, c in enumerate(classes):
        for i in range(n_per_class):
            counts = {f"GENE{j:04d}": float(j + i + c_i + 1) for j in range(MIN_DETECTED_GENES + 2)}
            out.append({"id": f"{c}-{i}", "counts": counts, "label": c})
    return out


def test_validate_dataset_rejections_are_actionable():
    good = _records()
    assert validate_dataset(good)["classes"] == ["x", "y"]
    with pytest.raises(TypeError, match="list of"):
        validate_dataset({"id": 1})
    with pytest.raises(ValueError, match=f"at least {MIN_RECORDS}"):
        validate_dataset(good[:2])
    bad = [dict(r) for r in good]
    del bad[0]["label"]
    with pytest.raises(ValueError, match=r"missing required column\(s\) \['label'\]"):
        validate_dataset(bad)
    bad = [dict(r) for r in good]
    bad[1]["id"] = bad[0]["id"]
    with pytest.raises(ValueError, match="duplicates id"):
        validate_dataset(bad)
    bad = [dict(r) for r in good]
    bad[2] = {**bad[2], "counts": {"GENE0000": 5}}
    with pytest.raises(ValueError, match=f"at least {MIN_DETECTED_GENES}"):
        validate_dataset(bad)
    bad = [dict(r) for r in good]
    bad[3] = {**bad[3], "counts": {**bad[3]["counts"], "GENE0000": -2}}
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_dataset(bad)
    one_class = [dict(r, label="x") for r in good]
    with pytest.raises(ValueError, match="at least 2 classes"):
        validate_dataset(one_class)
    thin = good + [{"id": "z-0", "counts": good[0]["counts"], "label": "z"}]
    with pytest.raises(ValueError, match=f"fewer than {MIN_RECORDS_PER_CLASS}"):
        validate_dataset(thin)
    with pytest.raises(ValueError, match="not in the class list"):
        validate_dataset(good, classes=["x"])
    assert validate_dataset(good[:1] + good[-1:], min_records=2, min_per_class=1)["n_records"] == 2


def test_validate_dataset_rejects_unencodable_cells_when_given_a_vocabulary():
    records = [
        {"id": f"r-{i}", "counts": {f"NOTAGENE{j}": 3.0 for j in range(MIN_DETECTED_GENES + 1)}, "label": c}
        for i, c in enumerate(["x"] * MIN_RECORDS + ["y"] * MIN_RECORDS)
    ]
    with pytest.raises(ValueError, match="in the scGPT vocabulary"):
        validate_dataset(records, vocabulary=VOCAB)


def test_split_is_stratified_disjoint_and_seeded():
    records = generate_sample_dataset(VOCAB)
    s1 = split_dataset(records, seed=42)
    assert s1 == split_dataset(records, seed=42)
    assert split_dataset(records, seed=1) != s1
    ids = [r["id"] for part in s1.values() for r in part]
    assert len(ids) == len(set(ids)) == len(records)
    for part in s1.values():
        labels = [r["label"] for r in part]
        assert labels.count("b-like") == labels.count("t-like")
    assert {k: len(v) for k, v in s1.items()} == {"train": 36, "validation": 12, "test": 16}
    with pytest.raises(ValueError, match="sum to less than 1"):
        split_dataset(records, val_fraction=0.6, test_fraction=0.5)


# --- metrics and baselines ---------------------------------------------------------------------


def test_auroc_handles_ties_and_degenerate_labels():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc([0, 1, 0, 1], [0.5] * 4) == 0.5
    assert auroc([1, 1], [0.1, 0.2]) is None


def test_classification_metrics_binary():
    y = ["b-like", "t-like", "t-like", "b-like"]
    p = ["b-like", "t-like", "b-like", "b-like"]
    scores = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.5, 0.5]]
    m = classification_metrics(y, p, scores, list(SAMPLE_CLASSES))
    assert m["n"] == 4 and m["accuracy"] == 0.75
    assert m["per_class"]["t-like"]["recall"] == 0.5
    assert "positive class 't-like'" in m["auroc_definition"]
    with pytest.raises(ValueError, match="outside the class list"):
        classification_metrics(y, ["other"] * 4, None, list(SAMPLE_CLASSES))


def test_baselines_fit_on_train_only_and_library_size_is_uninformative():
    records = generate_sample_dataset(VOCAB)
    splits = split_dataset(records, seed=42)
    classes = list(SAMPLE_CLASSES)
    maj = majority_baseline(splits["train"], splits["test"], classes)
    assert maj["baseline"] == "majority-class" and maj["accuracy"] == 0.5
    lib = library_size_baseline(splits["train"], splits["test"], classes)
    assert lib["baseline"] == "library-size threshold"
    # The classes share their library size by construction, so this baseline cannot separate them.
    assert lib["accuracy"] <= 0.7
    with pytest.raises(ValueError, match="binary"):
        library_size_baseline(splits["train"], splits["test"], ["a", "b", "c"])


def test_nearest_centroid_requires_a_backend_and_fits_on_train():
    records = generate_sample_dataset(VOCAB, size=16)
    splits = split_dataset(records, seed=1)

    def embedder(cells):  # a backend that encodes the B-programme rank directly
        out = []
        for c in cells:
            top = set(bin_expression(c, VOCAB)["gene_symbols"][:PROGRAMME_GENES])
            b = len(top & set(B_CELL_PROGRAMME)) / PROGRAMME_GENES
            out.append([b, 1.0 - b] + [0.0] * 510)
        return out

    pipe = ScGPTPipeline(embedder, VOCAB, "cpu")
    nc = pipe.nearest_centroid_evaluate(splits["train"], splits["test"])
    assert nc["accuracy"] == 1.0 and "no training" in nc["baseline"]
    assert nc["fitted_on"].startswith("training split only")


# --- BYOD loaders -----------------------------------------------------------------------------


def test_byod_csv_roundtrip_and_rejections(tmp_path):
    records = generate_sample_dataset(VOCAB, size=8)
    path = write_dataset_csv(records, tmp_path / "cells.csv")
    loaded = load_byod_dataset(path)
    assert [r["id"] for r in loaded] == [r["id"] for r in records]
    assert loaded[0]["counts"] == {g: float(v) for g, v in records[0]["counts"].items() if v > 0}
    (tmp_path / "bad.csv").write_text("cell,GENE0000,label\nc1,5,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must start with an 'id' column"):
        load_byod_dataset(tmp_path / "bad.csv")
    (tmp_path / "short.csv").write_text("id,GENE0000,label\nc1,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has 2 fields"):
        load_byod_dataset(tmp_path / "short.csv")
    (tmp_path / "nan.csv").write_text("id,GENE0000,label\nc1,abc,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is not a number"):
        load_byod_dataset(tmp_path / "nan.csv")
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_byod_dataset(tmp_path / "empty.csv")
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "nope.csv")
    tsv = tmp_path / "cells.tsv"
    tsv.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported BYOD file type"):
        load_byod_dataset(tsv)


def test_byod_json_and_jsonl(tmp_path):
    records = generate_sample_dataset(VOCAB, size=8)
    (tmp_path / "d.json").write_text(json.dumps(records), encoding="utf-8")
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "d.json") == records
    assert load_byod_dataset(tmp_path / "d.jsonl") == records
    (tmp_path / "obj.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(TypeError, match="top-level array"):
        load_byod_dataset(tmp_path / "obj.json")
    (tmp_path / "broken.jsonl").write_text('{"id": 1}\n{bad\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 is not valid JSON"):
        load_byod_dataset(tmp_path / "broken.jsonl")


# --- artifact manifest checks -------------------------------------------------------------------


def _fake_loaded(tmp_path):
    pipe = ScGPTPipeline(lambda rows: [[0.0] * 512 for _ in rows], VOCAB, "cpu")
    pipe.model = object()
    pipe.weights_dir = tmp_path
    return pipe


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path):
    pipe = _fake_loaded(tmp_path)
    art = tmp_path / "adapter"
    art.mkdir()
    base = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION}
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        pipe.load_artifact(art)
    (art / ARTIFACT_MANIFEST_NAME).write_text(json.dumps({"format": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(art)
    (art / ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps({"format": ARTIFACT_FORMAT, "base_model": {**base, "model_revision": "0" * 40}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="this package pins"):
        pipe.load_artifact(art)
    (art / ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps({"format": ARTIFACT_FORMAT, "base_model": base, "classes": ["only"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="at least two unique classes"):
        pipe.load_artifact(art)
    (art / ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "format": ARTIFACT_FORMAT,
                "base_model": base,
                "classes": list(SAMPLE_CLASSES),
                "files": [{"path": "adapter.safetensors", "bytes": 1, "sha256": "0" * 64}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="artifact file missing"):
        pipe.load_artifact(art)
    (art / "adapter.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        pipe.load_artifact(art)
