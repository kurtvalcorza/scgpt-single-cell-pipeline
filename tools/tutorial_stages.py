"""Stage runner for the standalone scGPT single-cell E2E tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated-environment pattern).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --outputs OUTPUTS --weights WEIGHTS --stage data --options '{...}'

Nothing is installed into the notebook kernel. Each stage is a separate process and starts from files only: the
verified snapshot under ``--weights``, the records of earlier stages under ``RUN_DIR/state`` (JSON), and the
learner-facing exports under ``--outputs``. The adapted head lives only in the exported adapter: the ``adapt`` stage
writes it, and the ``evaluate``, ``export`` and ``result`` stages rebuild the classifier from it in fresh processes, so
every held-out number is computed by the artifact a user would receive. On failure a stage writes
``RUN_DIR/state/<stage>.error.json``, which the notebook re-raises in the kernel.

Stages: weights → data → tokens → embed → probe → baselines → adapt → evaluate → export → result, plus the optional
``activity``. ``data`` and ``tokens`` import no model library, so CI exercises them directly. No stage asserts a quality
level: invariance and parity checks print a verdict; only a broken contract (a reload that does not reproduce the
adapting process's predictions) stops the notebook.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

STEM = "scgpt_single_cell"
PACKAGE = "scgpt_single_cell_pipeline"
NEW_CELLS = 6  # cells classified after training: freshly generated (sample) or held back before the split (BYOD)
NEW_CELLS_SEED = 7
RELOAD_TOLERANCE = 1e-5  # max |score difference| between the adapting process and a fresh reload of its adapter
BATCH_TOLERANCE = 1e-4  # single-cell vs in-batch embedding: float accumulation order only


class Run:
    """Paths of one run: carried sources and state under ``root``; learner-facing files under ``outputs``."""

    def __init__(self, root: Path, outputs: Path, weights: Path, options: dict[str, Any]) -> None:
        self.root = root
        self.out = outputs
        self.weights = weights
        self.options = options
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook in order from Section 3)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path


def package(root: Path):
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(PACKAGE)


def snapshot_dir(run: Run, P) -> Path:
    return run.weights / P.MODEL_KEY


def vocabulary(run: Run, P):
    path = snapshot_dir(run, P)
    if not (path / P.VOCAB_NAME).is_file():
        raise RuntimeError(f"the pinned vocabulary is not staged at {path}: run Section 3 first")
    return P.load_gene_vocabulary(path)


def base_pipeline(run: Run, P):
    return P.ScGPTPipeline.from_pretrained(weights_dir=snapshot_dir(run, P), allow_download=False)


def records(run: Run, needed_by: str) -> dict[str, Any]:
    return run.read_state("data.json", needed_by)


def counts_of(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    return [r["counts"] for r in rows]


def ids_of(rows: list[dict[str, Any]]) -> list[str]:
    return [r["id"] for r in rows]


def rounded(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(v, digits) for v in value]
    return value


def clear_outputs(run: Run) -> list[str]:
    removed = []
    for path in sorted(run.out.glob(f"{STEM}_*")):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return removed


# --------------------------------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------------------------------


def stage_weights(run: Run) -> None:
    import shutil

    P = package(run.root)
    snapshot = snapshot_dir(run, P)
    snapshot.mkdir(parents=True, exist_ok=True)
    carried = run.root / "weights" / P.MODEL_KEY / P.MANIFEST_NAME
    manifest = json.loads(carried.read_text(encoding="utf-8"))
    if (manifest["modelId"], manifest["revision"]) != (P.MODEL_ID, P.MODEL_REVISION):
        raise RuntimeError("the carried manifest does not name the identity carried by the package; regenerate the notebook")
    shutil.copyfile(carried, snapshot / P.MANIFEST_NAME)
    print({"model_id": P.MODEL_ID, "revision": P.MODEL_REVISION, "license": P.MODEL_LICENSE, "files": len(manifest["files"]), "total_bytes": manifest["totalBytes"]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fetched = P.stage_missing_files(snapshot, allow_download=True)
    mirror = any("mirror" in str(w.message) for w in caught)
    print({"weights_dir": str(snapshot), "fetched": fetched})
    if P.VOCAB_NAME in fetched:
        print({"vocabulary_source": P.VOCAB_MIRROR_URL if mirror else P.VOCAB_SOURCE_URL, "note": "Harvard Dataverse was unavailable; the immutable mirror's bytes are accepted only because they match the pinned digest" if mirror else "fetched from the persistent Harvard Dataverse file id"})
    verified = P.verify_snapshot(snapshot)
    vocab_entry = [e for e in verified["files"] if e["path"] == P.VOCAB_NAME]
    print({"verified_files": len(verified["files"]), "revision": verified["revision"], "weights": P.WEIGHTS_NAME, "vocabulary_sha256": vocab_entry[0]["sha256"][:16] + "..." if vocab_entry else None})
    run.write_state("weights.json", {"snapshot": str(snapshot), "vocabulary_from_mirror": mirror, "fetched": fetched})


# --------------------------------------------------------------------------------------------------
# data (SCG-m2)
# --------------------------------------------------------------------------------------------------


def min_cells_per_class(P, n_classes: int, val_fraction: float, test_fraction: float) -> int:
    """The true BYOD minimum (SCG-m2): the smallest balanced class size whose stratified split leaves a training split
    of at least MIN_RECORDS cells and MIN_RECORDS_PER_CLASS per class (what the baselines and `adapt` validate)."""
    for n in range(P.MIN_RECORDS_PER_CLASS, P.MAX_RECORDS + 1):
        train = n - max(1, round(n * val_fraction)) - max(1, round(n * test_fraction))
        if train >= P.MIN_RECORDS_PER_CLASS and train * n_classes >= P.MIN_RECORDS:
            return n
    return P.MAX_RECORDS


def checked_split(P, rows: list[dict[str, Any]], val_fraction: float, test_fraction: float, seed: int) -> dict[str, list[dict[str, Any]]]:
    """Split, then check the training split against the same contract the baselines and `adapt` apply, so a dataset that
    is too small stops here with the true minimum instead of in Section 8 or 9."""
    splits = P.split_dataset(rows, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed)
    classes = sorted({r["label"] for r in rows})
    try:
        P.validate_dataset(splits["train"], classes=classes)
    except ValueError as exc:
        need = min_cells_per_class(P, len(classes), val_fraction, test_fraction)
        counts = {c: sum(r["label"] == c for r in splits["train"]) for c in classes}
        raise ValueError(f"the training split would hold {len(splits['train'])} cells {counts} ({exc}); the baselines and the fine-tuning need at least {P.MIN_RECORDS} training cells and {P.MIN_RECORDS_PER_CLASS} per class, so with VAL_FRACTION={val_fraction} and TEST_FRACTION={test_fraction} and {len(classes)} classes supply at least {need} cells per class") from None
    return splits


def hold_back(P, rows: list[dict[str, Any]], wanted: int, seed: int, val_fraction: float, test_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Hold back up to ``wanted`` BYOD cells for inference before the split, round-robin over classes, only while the
    remaining cells still split into a valid training split (``checked_split``). Returns (kept, held, note); the note says
    how many were held and why not more."""
    by_class: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_class.setdefault(r["label"], []).append(r)
    rng = random.Random(seed)
    for members in by_class.values():
        rng.shuffle(members)
    held: list[dict[str, Any]] = []
    blocked: set[str] = set()
    while len(held) < wanted and len(blocked) < len(by_class):
        for label in sorted(by_class):
            if len(held) >= wanted or label in blocked:
                continue
            if len(by_class[label]) <= 1:
                blocked.add(label)
                continue
            candidate = by_class[label][-1]
            kept = [r for r in rows if r["id"] not in {h["id"] for h in held} | {candidate["id"]}]
            try:
                checked_split(P, kept, val_fraction, test_fraction, seed)
            except ValueError:
                blocked.add(label)
                continue
            held.append(by_class[label].pop())
    held_ids = {r["id"] for r in held}
    kept = [r for r in rows if r["id"] not in held_ids]
    if len(held) == wanted:
        note = f"{len(held)} cells held back before the split, round-robin over classes; they take no part in training, monitoring or evaluation"
    else:
        note = f"only {len(held)} of {wanted} cells could be held back while keeping a valid training split; add cells for a full new-cell check" + ("" if held else "; Section 11 then reports reload parity on test-split cells, labelled as such")
    return kept, held, note


def stage_data(run: Run) -> None:
    P = package(run.root)
    opts = run.options
    use_byod = bool(opts.get("use_byod", False))
    seed = int(opts.get("seed", 42))
    val_fraction = float(opts.get("val_fraction", 0.2))
    test_fraction = float(opts.get("test_fraction", 0.25))
    wanted = int(opts.get("inference_cells", NEW_CELLS))
    removed = clear_outputs(run)
    for name in ("data.json", "embed.json", "probe.json", "baselines.json", "adapt.json", "evaluate.json", "export.json"):
        (run.state / name).unlink(missing_ok=True)
    if removed:
        print({"removed_previous_outputs": removed})
    vocab = vocabulary(run, P)
    if use_byod:
        path = Path(opts.get("byod_path") or "")
        if not str(path) or not path.is_file():
            raise FileNotFoundError(f"BYOD_PATH {str(path)!r} is not a file in this runtime: upload the file or correct the path.")
        if path.suffix.lower() not in (".csv", ".json", ".jsonl"):
            raise ValueError(f"{path.name}: BYOD must be a genes-as-columns .csv, a .json array or a .jsonl file of {{id, counts, label}} records (export an AnnData .h5ad to CSV first).")
        try:
            rows = P.load_byod_dataset(path)
            P.validate_dataset(rows, vocabulary=vocab)
            checked_split(P, rows, val_fraction, test_fraction, seed)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path.name}: {exc}") from None
        rows, new_rows, held_note = hold_back(P, rows, wanted, seed, val_fraction, test_fraction)
        new_source = f"{len(new_rows)} BYOD cells held back before the split" if new_rows else "none held back"
        source = f"BYOD ({path.name})"
        programmes = None
    else:
        rows = P.generate_sample_dataset(vocab)
        # The generator numbers cells per class from 000, so the new cells get a prefix: their ids must not collide with the split's.
        new_rows = [{**r, "id": f"new-{r['id']}"} for r in P.generate_sample_dataset(vocab, seed=NEW_CELLS_SEED, size=NEW_CELLS)]
        new_source = f"freshly generated marker-programme cells (seed {NEW_CELLS_SEED}), never in any split"
        held_note = None
        source = f"synthetic marker-programme dataset (seed {P.SAMPLE_SEED}, {P.SAMPLE_SIZE} cells, real gene symbols)"
        programmes = P.select_sample_genes(vocab)
    try:
        manifest = P.validate_dataset(rows, vocabulary=vocab)
        splits = checked_split(P, rows, val_fraction, test_fraction, seed)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{source}: {exc}") from None
    P.write_dataset_csv(rows, run.out / f"{STEM}_dataset.csv")
    if programmes:
        print({"t_cell_programme": programmes["t_cell"]})
        print({"b_cell_programme": programmes["b_cell"]})
        print({"background_genes": len(programmes["background"]), "first_five": programmes["background"][:5]})
    print({"data_source": source, "n_records": manifest["n_records"], "classes": manifest["classes"], "class_counts": manifest["class_counts"]})
    print({"detected_genes": manifest["detected_genes"], "encodable_genes": manifest["encodable_genes"], "library_size": manifest["library_size"]})
    print({"ceilings": manifest["ceilings"], "digest": manifest["digest"][:16] + "..."})
    print({"train": len(splits["train"]), "validation": len(splits["validation"]), "test": len(splits["test"]), "new_cells": len(new_rows), "new_cells_source": new_source})
    if held_note:
        print({"held_back": held_note})
    run.write_state("data.json", {"use_byod": use_byod, "source": source, "seed": seed, "manifest": manifest, "classes": manifest["classes"], "train": splits["train"], "validation": splits["validation"], "test": splits["test"], "new": new_rows, "new_source": new_source, "held_back_note": held_note})
    print(f"wrote outputs/{STEM}_dataset.csv (genes-as-columns, the shape BYOD expects)")


# --------------------------------------------------------------------------------------------------
# tokens (no model)
# --------------------------------------------------------------------------------------------------


def stage_tokens(run: Run) -> None:
    P = package(run.root)
    data = records(run, "tokens")
    vocab = vocabulary(run, P)
    example = data["test"][0]
    encoded = P.bin_expression(example["counts"], vocab)
    print({"id": example["id"], "label": example["label"], "tokens": len(encoded["tokens"]), "first_token_is_cls": encoded["tokens"][0] == vocab.cls_id, "n_bins": encoded["n_bins"]})
    print({"top_genes": encoded["gene_symbols"][:10], "top_bins": encoded["values"][1:11]})
    print({"bottom_genes": encoded["gene_symbols"][-5:], "bottom_bins": encoded["values"][-5:]})
    scaled = P.bin_expression({g: v * 37 for g, v in example["counts"].items()}, vocab)
    invariant = scaled["values"] == encoded["values"] and scaled["tokens"] == encoded["tokens"]
    print({"scale_invariance": "PASS" if invariant else "FAIL", "note": "binning is rank-based within the cell, so multiplying every count by 37 must leave the tokens and bins unchanged"})
    print({"max_genes_per_cell": P.MAX_GENES_PER_CELL, "max_cells_per_call": P.MAX_CELLS_PER_CALL, "min_detected_genes": P.MIN_DETECTED_GENES, "vocab_size": P.VOCAB_SIZE, "specials": {"pad": vocab.pad_id, "cls": vocab.cls_id, "eoc": vocab.eoc_id}})
    print({"validation": P.INPUT_SCHEMA["validation"]})
    probes = {
        "empty cell": {},
        "two genes": {"GAPDH": 3, "ACTB": 5},
        "negative count": {**example["counts"], "CD3D": -1},
        "no known symbols": {f"NOTAGENE{i}": 1.0 for i in range(12)},
    }
    refusals = {}
    for name, cell in probes.items():
        try:
            P.validate_inputs([cell], vocab)
            refusals[name] = "accepted"
            print({"probe": name, "verdict": "accepted"})
        except (TypeError, ValueError) as exc:
            refusals[name] = str(exc)
            print({"probe": name, "rejected": str(exc)[:110]})
    wide = {symbol: 1.0 + k for k, symbol in enumerate(vocab.gene_symbols[: P.MAX_GENES_PER_CELL + 50])}
    wide_manifest = P.validate_inputs([wide], vocab)
    print({"truncation_probe": wide_manifest["inputs"][0]})
    input_manifest = P.validate_inputs(counts_of(data["test"][:4]), vocab, names=ids_of(data["test"][:4]))
    input_manifest["probes"] = refusals
    run.write_output(f"{STEM}_input_manifest.json", input_manifest)
    print({"verdict": input_manifest["verdict"], "n_cells": input_manifest["n_cells"], "max_tokens_observed": input_manifest["max_tokens_observed"], "requires_remote_code": input_manifest["requires_remote_code"], "written": f"outputs/{STEM}_input_manifest.json"})


# --------------------------------------------------------------------------------------------------
# model stages
# --------------------------------------------------------------------------------------------------


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def stage_embed(run: Run) -> None:
    P = package(run.root)
    data = records(run, "embed")
    pipe = base_pipeline(run, P)
    rows = data["validation"][:8]
    result = pipe.embed(counts_of(rows), names=ids_of(rows))
    vectors = result["embeddings"]
    print({"n_cells": result["n_cells"], "dimension": result["dimension"], "tokens": result["tokens"], "pooling": result["pooling"], "device": pipe.device})
    repeat = pipe.embed(counts_of(rows), names=ids_of(rows))["embeddings"]
    single = pipe.embed([rows[0]["counts"]])["embeddings"][0]
    shuffled = list(rows[0]["counts"].items())
    random.Random(3).shuffle(shuffled)
    reordered = pipe.embed([dict(shuffled)])["embeddings"][0]
    order_diff = max(abs(a - b) for a, b in zip(reordered, single, strict=True))
    cross_batch = max(abs(a - b) for a, b in zip(single, vectors[0], strict=True))
    checks = {
        "same_batch_twice_identical": "PASS" if repeat == vectors else "FAIL",
        "gene_order_invariance": "PASS" if order_diff == 0.0 else "FAIL",
        "single_vs_batch_within_1e-4": "PASS" if cross_batch < BATCH_TOLERANCE else "FAIL",
    }
    print({**checks, "gene_order_max_abs_diff": order_diff, "single_vs_batch_max_abs_diff": cross_batch, "note": "gene order never matters (tokens are ranked before encoding; no positional encoding); batch composition changes float accumulation order at the 1e-6 level"})
    if "FAIL" in checks.values():
        print("A check above failed: the embeddings are not behaving as the encoder's design implies. Read Troubleshooting before trusting later sections.")
    mean = [sum(v[d] for v in vectors) / len(vectors) for d in range(len(vectors[0]))]
    centred = [[x - m for x, m in zip(v, mean, strict=True)] for v in vectors]
    raw_pairs, within, between = [], [], []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            raw_pairs.append(cosine(vectors[i], vectors[j]))
            (within if rows[i]["label"] == rows[j]["label"] else between).append(cosine(centred[i], centred[j]))
    geometry = {"raw_cosine_between_cells_mean": round(sum(raw_pairs) / len(raw_pairs), 4), "centred_cosine_within_class": round(sum(within) / len(within), 4) if within else None, "centred_cosine_between_classes": round(sum(between) / len(between), 4) if between else None}
    print({**geometry, "note": "inspection only; embeddings are unlabelled representations"})
    path = run.out / f"{STEM}_embeddings.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "tokens"] + [f"dim_{k}" for k in range(result["dimension"])])
        for r, n_tokens, vec in zip(rows, result["tokens"], vectors, strict=True):
            writer.writerow([r["id"], r["label"], n_tokens] + [f"{x:.6f}" for x in vec])
    run.write_state("embed.json", {"n_cells": result["n_cells"], "dimension": result["dimension"], "pooling": result["pooling"], "checks": checks, "gene_order_max_abs_diff": order_diff, "single_vs_batch_max_abs_diff": cross_batch, "geometry": geometry})
    print(f"wrote outputs/{STEM}_embeddings.csv")


def stage_probe(run: Run) -> None:
    P = package(run.root)
    data = records(run, "probe")
    pipe = base_pipeline(run, P)
    rows = data["validation"][:8]
    masked = pipe.predict_masked(counts_of(rows), names=ids_of(rows), mask_fraction=0.15, seed=data["seed"])
    pearsons = [c["pearson"] for c in masked["cells"] if c["pearson"] is not None]
    maes = [c["mean_absolute_error_bins"] for c in masked["cells"]]
    summary = {"mean_pearson": round(sum(pearsons) / len(pearsons), 4) if pearsons else None, "mean_absolute_error_bins": round(sum(maes) / len(maes), 3)}
    print({"mask_fraction": masked["mask_fraction"], "mask_value": masked["mask_value"], "n_bins": masked["n_bins"]})
    print({**summary, "note": masked["note"]})
    for c in masked["cells"][:3]:
        print(c)
    lineage: dict[str, Any] | None = None
    if data["use_byod"]:
        print({"lineage_probe": "skipped", "reason": "it masks the tutorial's T- and B-cell marker programmes and compares t-like with b-like cells; your labels need not be those lineages, so the question it asks does not apply"})
    else:
        lineage = {}
        for label in data["classes"]:
            cells_of = [r["counts"] for r in data["validation"] if r["label"] == label][:4]
            for marker_name, markers in (("T markers", P.T_CELL_PROGRAMME), ("B markers", P.B_CELL_PROGRAMME)):
                probe = pipe.predict_masked_genes(cells_of, list(markers))
                lineage[f"{label} / masked {marker_name}"] = {"predicted_mean_bin": probe["predicted_mean_bin"], "true_mean_bin": probe["true_mean_bin"]}
        for key, row in lineage.items():
            print({key: row})
        print({"reading": "a decoder that used lineage context would predict T markers high in t-like cells and low in b-like cells; compare the predicted bins across the rows"})
    run.write_state("probe.json", {**summary, "lineage_probe": lineage})


def stage_baselines(run: Run) -> None:
    P = package(run.root)
    data = records(run, "baselines")
    pipe = base_pipeline(run, P)
    train, test, classes = data["train"], data["test"], data["classes"]
    centroid = pipe.nearest_centroid_evaluate(train, test, classes=classes)
    print({k: centroid[k] for k in ("baseline", "fitted_on", "n", "accuracy", "macro_f1", "auroc")})
    majority = P.majority_baseline(train, test, classes)
    print({k: majority[k] for k in ("baseline", "predicted_label", "accuracy", "macro_f1")})
    library = P.library_size_baseline(train, test, classes)
    print({k: library[k] for k in ("baseline", "rule", "train_accuracy", "accuracy", "macro_f1", "auroc")})
    run.write_state("baselines.json", {"nearest_centroid": centroid, "majority": majority, "library_size": library})


def adapt_settings(opts: dict[str, Any]) -> dict[str, Any]:
    return {"epochs": int(opts.get("epochs", 4)), "learning_rate": float(opts.get("learning_rate", 1e-4)), "batch_size": int(opts.get("batch_size", 8)), "trainable_layers": int(opts.get("trainable_layers", 1))}


def stage_adapt(run: Run) -> None:
    P = package(run.root)
    data = records(run, "adapt")
    settings = adapt_settings(run.options)
    pipe = base_pipeline(run, P)
    started = time.perf_counter()
    result = pipe.adapt(data["train"], data["validation"], classes=data["classes"], seed=data["seed"], **settings)
    seconds = round(time.perf_counter() - started, 2)
    print({"method": result["method"], "trainable_parameters": result["trainable_parameters"], "total_parameters": result["total_parameters"], "precision": result["precision"], "device": pipe.device, "seconds": seconds})
    for step in result["history"]:
        print(step)
    # The adapting process's own scores are the reference the fresh reloads in Sections 10 and 11 must reproduce.
    reference = {split: pipe.evaluate(data[split]) for split in ("validation", "test")}
    parity_rows = data["new"] or data["test"][:NEW_CELLS]
    reference_predictions = pipe.classify(counts_of(parity_rows), names=ids_of(parity_rows))["predictions"]
    artifact = run.out / f"{STEM}_adapter"
    pipe.save_artifact(artifact, metadata={"data_source": data["source"], "dataset_digest": data["manifest"]["digest"], "settings": settings})
    run.write_state("adapt.json", {"settings": settings, "seconds": seconds, "adaptation": {k: v for k, v in result.items() if k != "trainable_parameter_names"}, "reference_metrics": reference, "reference_predictions": reference_predictions, "parity_ids": ids_of(parity_rows), "artifact": str(artifact), "device": pipe.device})
    print({"adapter": f"outputs/{STEM}_adapter", "note": "the trained tensors are written now; Sections 10-12 rebuild the classifier from these files in fresh processes"})


def reloaded(run: Run, P):
    adapt = run.read_state("adapt.json", "reload")
    return P.ScGPTPipeline.from_artifact(Path(adapt["artifact"]), weights_dir=snapshot_dir(run, P)), adapt


def stage_evaluate(run: Run) -> None:
    P = package(run.root)
    data = records(run, "evaluate")
    baselines = run.read_state("baselines.json", "evaluate")
    probe = run.read_state("probe.json", "evaluate")
    pipe, adapt = reloaded(run, P)
    val_metrics = pipe.evaluate(data["validation"])
    test_metrics = pipe.evaluate(data["test"])
    print({"split": "validation", **{k: val_metrics[k] for k in ("n", "accuracy", "macro_f1", "auroc")}})
    print({"split": "test", **{k: test_metrics[k] for k in ("n", "accuracy", "macro_f1", "auroc")}})
    for cls_name, row in test_metrics["per_class"].items():
        print({"class": cls_name, **row})
    reference = adapt["reference_metrics"]["test"]
    drift = {k: abs((test_metrics[k] or 0.0) - (reference[k] or 0.0)) for k in ("accuracy", "macro_f1", "auroc") if test_metrics.get(k) is not None or reference.get(k) is not None}
    centroid, majority = baselines["nearest_centroid"], baselines["majority"]
    gain = round(test_metrics["accuracy"] - centroid["accuracy"], 4)
    if centroid["accuracy"] >= 1.0 and test_metrics["accuracy"] >= 1.0:
        interpretation = "saturated: the zero-training nearest-centroid rule is already perfect on this split, so the fine-tuned head cannot show a gain here; the result shows the adaptation contract works, not that adaptation helps"
    elif gain > 0:
        interpretation = f"the adapted head beats the zero-training centroid rule by {gain} accuracy on {test_metrics['n']} test cells (one holdout, no dispersion estimate: read it as a direction, not a size)"
    else:
        interpretation = f"the adapted head does not beat the zero-training centroid rule (difference {gain}); on this data the frozen embedding already carries what the head learned"
    report = {
        "task": "cell-state classification (bounded fine-tuning of scGPT)",
        "evidence": "tutorial sample-sanity metrics on one stratified holdout; not a benchmark" if not data["use_byod"] else "one stratified holdout of the supplied cells; not a benchmark",
        "verdict": "sample-sanity",
        "estimation": f"single train/validation/test split, seed {data['seed']}, no dispersion estimate",
        "data_source": data["source"],
        "dataset_digest": data["manifest"]["digest"],
        "classes": data["classes"],
        "splits": {"train": len(data["train"]), "validation": len(data["validation"]), "test": len(data["test"])},
        "baselines": baselines,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "computed_by": "a fresh process that rebuilt the classifier from the exported adapter",
        "max_abs_metric_difference_vs_adapting_process": rounded(max(drift.values()) if drift else 0.0, 8),
        "delta_vs_nearest_centroid": {k: round(test_metrics[k] - centroid[k], 4) for k in ("accuracy", "macro_f1")},
        "delta_vs_majority": {k: round(test_metrics[k] - majority[k], 4) for k in ("accuracy", "macro_f1")},
        "interpretation": interpretation,
        "masked_expression_probe": probe,
        "adaptation": adapt["adaptation"],
        "adaptation_seconds": adapt["seconds"],
    }
    run.write_output(f"{STEM}_evaluation_report.json", report)
    run.write_state("evaluate.json", {"test": {k: test_metrics[k] for k in ("n", "accuracy", "macro_f1", "auroc")}})
    print({"delta_vs_nearest_centroid": report["delta_vs_nearest_centroid"], "delta_vs_majority": report["delta_vs_majority"], "same_as_adapting_process": report["max_abs_metric_difference_vs_adapting_process"] == 0.0})
    print({"interpretation": interpretation, "report": f"outputs/{STEM}_evaluation_report.json"})


def stage_export(run: Run) -> None:
    P = package(run.root)
    data = records(run, "export")
    pipe, adapt = reloaded(run, P)
    pool = data["new"] or data["test"]
    by_id = {r["id"]: r for r in pool}
    missing = [i for i in adapt["parity_ids"] if i not in by_id]
    if missing:
        raise RuntimeError(f"cells {missing} recorded by Section 9 are not in the current dataset: Section 4 was re-run after Section 9; run Sections 9-11 again")
    rows = [by_id[i] for i in adapt["parity_ids"]]
    result = pipe.classify(counts_of(rows), names=ids_of(rows))
    predictions = result["predictions"]
    source = data["new_source"] if data["new"] else "test-split cells (no BYOD cells could be held back; reload parity only, not new-cell inference)"
    print({"cells": source, "decision_rule": result["decision_rule"]})
    matches = 0
    for p, r in zip(predictions, rows, strict=True):
        matches += p["label"] == r["label"]
        print({"id": p["id"], "predicted": p["label"], "score": round(p["score"], 4), "true_label": r["label"]})
    print({"matches": matches, "of": len(rows), "note": "a sanity check on held-out labels, not an evaluation"})
    classes = pipe.classes
    with (run.out / f"{STEM}_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "predicted_label", "score"] + [f"score_{c}" for c in classes])
        for p in predictions:
            writer.writerow([p["id"], p["label"], f"{p['score']:.6f}"] + [f"{p['scores'][c]:.6f}" for c in classes])
    manifest = json.loads((Path(adapt["artifact"]) / P.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
    print({"format": manifest["format"], "base_model": manifest["base_model"], "requires_remote_code": manifest["requires_remote_code"], "n_tensors": len(manifest["tensors"]), "files": manifest["files"]})
    reference = adapt["reference_predictions"]
    labels_equal = [p["label"] for p in predictions] == [p["label"] for p in reference] and [p["id"] for p in predictions] == [p["id"] for p in reference]
    max_diff = max(abs(a["scores"][c] - b["scores"][c]) for a, b in zip(predictions, reference, strict=True) for c in classes)
    parity = {"process": "fresh (separate from the adapting stage)", "labels_equal": labels_equal, "max_abs_score_diff": max_diff, "tolerance": RELOAD_TOLERANCE}
    run.write_state("export.json", {"cells": source, "decision_rule": result["decision_rule"], "predictions": predictions, "reload_parity": parity, "artifact_manifest": manifest})
    if not labels_equal or max_diff > RELOAD_TOLERANCE:
        raise RuntimeError(f"The reloaded adapter does not reproduce the adapting process's predictions (labels equal: {labels_equal}, max score difference {max_diff:.3g} > {RELOAD_TOLERANCE}). Do not ship this artifact.")
    print({"reload_parity": "PASS", **parity})


def stage_result(run: Run) -> None:
    import importlib.metadata
    import platform

    P = package(run.root)
    data = records(run, "result")
    weights = run.read_state("weights.json", "result")
    embed = run.read_state("embed.json", "result")
    export = run.read_state("export.json", "result")
    adapt = run.read_state("adapt.json", "result")
    manifest = json.loads((snapshot_dir(run, P) / P.MANIFEST_NAME).read_text(encoding="utf-8"))
    vocab_entry = ([e for e in manifest["files"] if e["path"] == P.VOCAB_NAME] or [{}])[0]
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    report = json.loads((run.out / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    import torch

    payload = {
        "task": "cell-state classification adaptation (scGPT)",
        "pipeline_class": "ScGPTPipeline",
        "model_id": P.MODEL_ID,
        "model_revision": P.MODEL_REVISION,
        "model_license": P.MODEL_LICENSE,
        "remote_code_executed": False,
        "vocabulary": {"source": vocab_entry.get("source"), "fetched_from_mirror": weights.get("vocabulary_from_mirror"), "sha256": vocab_entry.get("sha256"), "size": P.VOCAB_SIZE},
        "repository_revision": source.get("revision"),
        "notebook_source": source,
        "data_source": data["source"],
        "dataset_manifest": data["manifest"],
        "embedding_summary": embed,
        "evaluation_report": report,
        "inference": {"cells": export["cells"], "decision_rule": export["decision_rule"], "predictions": export["predictions"]},
        "artifact_format": P.ARTIFACT_FORMAT,
        "artifact_format_version": P.ARTIFACT_FORMAT_VERSION,
        "artifact_manifest": export["artifact_manifest"],
        "reload_parity": export["reload_parity"],
        "runtime": {"python": platform.python_version(), **{d: importlib.metadata.version(d) for d in ("torch", "safetensors", "numpy")}, "device": adapt["device"], "cuda": torch.cuda.is_available(), "precision": "float32"},
    }
    run.write_output(f"{STEM}_result.json", payload)
    print({"model_id": P.MODEL_ID, "revision": P.MODEL_REVISION[:12], "repository_revision": (source.get("revision") or "")[:12], "runtime": payload["runtime"]})
    print("outputs/:")
    for path in sorted(run.out.rglob("*")):
        if path.is_file():
            print(f"  - {path.relative_to(run.out.parent).as_posix()} ({path.stat().st_size / 1024:.1f} KB)")


def stage_activity(run: Run) -> None:
    """SCG-M2: re-run the bounded fine-tune with a different TRAINABLE_LAYERS and print it beside the default; writes only
    to outputs/activity/, never to the canonical outputs."""
    P = package(run.root)
    data = records(run, "activity")
    adapt = run.read_state("adapt.json", "activity")
    default = run.read_state("evaluate.json", "activity")["test"]
    baselines = run.read_state("baselines.json", "activity")
    settings = {**adapt["settings"], "trainable_layers": int(run.options.get("trainable_layers", 0))}
    before = sorted(p.name for p in run.out.glob(f"{STEM}_*"))
    pipe = base_pipeline(run, P)
    started = time.perf_counter()
    result = pipe.adapt(data["train"], data["validation"], classes=data["classes"], seed=data["seed"], **settings)
    seconds = round(time.perf_counter() - started, 2)
    test = pipe.evaluate(data["test"])
    rows = [
        {"run": "zero-training nearest centroid", "trainable_layers": "-", "trainable_parameters": 0, "seconds": "-", **{k: baselines["nearest_centroid"][k] for k in ("accuracy", "macro_f1", "auroc")}},
        {"run": "default (Section 9)", "trainable_layers": adapt["settings"]["trainable_layers"], "trainable_parameters": adapt["adaptation"]["trainable_parameters"], "seconds": adapt["seconds"], **{k: default[k] for k in ("accuracy", "macro_f1", "auroc")}},
        {"run": "activity", "trainable_layers": settings["trainable_layers"], "trainable_parameters": result["trainable_parameters"], "seconds": seconds, **{k: test[k] for k in ("accuracy", "macro_f1", "auroc")}},
    ]
    for row in rows:
        print(rounded(row))
    print({"activity_history": [rounded(h) for h in result["history"]]})
    run.write_output(f"activity/{STEM}_activity_trainable_layers_{settings['trainable_layers']}.json", {"settings": settings, "rows": rows, "history": result["history"]})
    if sorted(p.name for p in run.out.glob(f"{STEM}_*")) != before:
        raise RuntimeError("the activity changed the canonical outputs; it must write only to outputs/activity/")
    print(f"wrote outputs/activity/{STEM}_activity_trainable_layers_{settings['trainable_layers']}.json (the canonical outputs are unchanged)")


STAGES = {
    "weights": stage_weights,
    "data": stage_data,
    "tokens": stage_tokens,
    "embed": stage_embed,
    "probe": stage_probe,
    "baselines": stage_baselines,
    "adapt": stage_adapt,
    "evaluate": stage_evaluate,
    "export": stage_export,
    "result": stage_result,
    "activity": stage_activity,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    run = Run(args.root.resolve(), args.outputs.resolve(), args.weights.resolve(), json.loads(args.options))
    error_file = run.state / f"{args.stage}.error.json"
    error_file.unlink(missing_ok=True)
    try:
        STAGES[args.stage](run)
    except BaseException as exc:  # noqa: BLE001 -- every failure is reported to the kernel with its own message
        traceback.print_exc()
        error_file.write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
