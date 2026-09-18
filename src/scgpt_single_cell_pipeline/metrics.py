"""Classification metrics and trivial baselines for scGPT cell-state classification.

Pure Python (no scikit-learn): accuracy, macro-F1, per-class precision/recall/F1/support, and AUROC
(binary: positive class = the last entry of `classes`; multiclass: macro one-vs-rest), computed by
the Mann-Whitney rank statistic with average ranks for ties.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _prf(hits: int, n_pred: int, n_true: int) -> dict[str, float]:
    precision = hits / n_pred if n_pred else 0.0
    recall = hits / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def auroc(y_true: Sequence[int], scores: Sequence[float]) -> float | None:
    """Area under the ROC curve for binary 0/1 labels; None when only one class is present."""
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-based average rank of the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, y_true, strict=True) if y == 1)
    return round((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg), 4)


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    scores: Sequence[Sequence[float]] | None,
    classes: Sequence[str],
) -> dict[str, Any]:
    """Discrete and ranking metrics over one evaluation split (labels are class names).

    `scores[i][k]` is the score of class `classes[k]` for record i (softmax outputs from the
    pipeline; any monotone score works for AUROC). Class order is preserved exactly as given.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} labels vs {len(y_pred)} predictions")
    class_list = list(classes)
    unknown = sorted((set(y_true) | set(y_pred)) - set(class_list))
    if unknown:
        raise ValueError(f"labels outside the class list {class_list}: {unknown}")
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    per_class: dict[str, dict[str, Any]] = {}
    f1s: list[float] = []
    for c in class_list:
        hits = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == c and p == c)
        n_pred = sum(1 for p in y_pred if p == c)
        n_true = sum(1 for t in y_true if t == c)
        prf = _prf(hits, n_pred, n_true)
        per_class[c] = {**prf, "support": n_true, "predicted": n_pred}
        if n_true:
            f1s.append(prf["f1"])
    result: dict[str, Any] = {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "per_class": per_class,
        "classes": class_list,
        "decision_rule": "argmax over class scores",
        "auroc": None,
        "auroc_definition": None,
    }
    if scores is not None and n:
        if len(scores) != n or any(len(row) != len(class_list) for row in scores):
            raise ValueError("scores must be one row per record with one column per class")
        if len(class_list) == 2:
            pos = class_list[-1]
            result["auroc"] = auroc([1 if t == pos else 0 for t in y_true], [row[-1] for row in scores])
            result["auroc_definition"] = f"binary AUROC with positive class {pos!r} (last class in the list)"
        else:
            values = []
            for k, c in enumerate(class_list):
                a = auroc([1 if t == c else 0 for t in y_true], [row[k] for row in scores])
                if a is not None:
                    values.append(a)
            result["auroc"] = round(sum(values) / len(values), 4) if values else None
            result["auroc_definition"] = "macro-averaged one-vs-rest AUROC over classes present in the split"
    return result


def majority_baseline(
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    classes: Sequence[str],
) -> dict[str, Any]:
    """Predict the most frequent training class for every evaluation record (EVAL11)."""
    counts: dict[str, int] = {}
    for r in train_records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    majority = max(sorted(counts), key=counts.__getitem__)
    metrics = classification_metrics(
        [r["label"] for r in eval_records], [majority] * len(eval_records), None, classes
    )
    return {"baseline": "majority-class", "predicted_label": majority, **metrics}


def library_size_baseline(
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    classes: Sequence[str],
) -> dict[str, Any]:
    """Threshold on a cell's total counts, fitted on the training split only (binary tasks).

    Library size is the first thing that separates cells in a badly designed single-cell experiment,
    so it is the baseline worth ruling out. The threshold and the class direction are chosen to
    maximise training accuracy; the evaluation split is never touched during fitting (SPL8). On the
    tutorial sample every cell carries the same total counts by construction, so this baseline is
    expected to sit at chance -- which is the point: it shows the model is reading the binned
    expression profile, not how much RNA was captured.
    """
    class_list = list(classes)
    if len(class_list) != 2:
        raise ValueError("library_size_baseline is defined for binary tasks only")
    lo, hi = class_list

    def total(record: Mapping[str, Any]) -> float:
        return float(sum(record["counts"].values()))

    train_x = [total(r) for r in train_records]
    train_y = [r["label"] for r in train_records]
    best = (-1.0, 0.0, True)  # accuracy, threshold, high_is_hi
    for t in sorted(set(train_x)):
        for high_is_hi in (True, False):
            pred = [(hi if (x >= t) == high_is_hi else lo) for x in train_x]
            acc = sum(p == y for p, y in zip(pred, train_y, strict=True)) / len(train_y)
            if acc > best[0]:
                best = (acc, t, high_is_hi)
    _, threshold, high_is_hi = best
    eval_x = [total(r) for r in eval_records]
    eval_pred = [(hi if (x >= threshold) == high_is_hi else lo) for x in eval_x]
    span = max(eval_x) - min(eval_x) or 1.0
    normalised = [(x - min(eval_x)) / span for x in eval_x]
    scores = [[1.0 - x, x] if high_is_hi else [x, 1.0 - x] for x in normalised]
    metrics = classification_metrics([r["label"] for r in eval_records], eval_pred, scores, class_list)
    return {
        "baseline": "library-size threshold",
        "threshold": round(threshold, 2),
        "rule": f"predict {hi!r} when total counts {'>=' if high_is_hi else '<'} {threshold:.0f}",
        "train_accuracy": round(best[0], 4),
        **metrics,
    }
