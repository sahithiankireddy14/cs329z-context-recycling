from __future__ import annotations

from collections import Counter
from typing import Any

from .models import ClassificationResult, Label


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(
    predictions: list[ClassificationResult], labels: dict[str, Label]
) -> dict[str, Any]:
    matched = [prediction for prediction in predictions if prediction.candidate_id in labels]
    if not matched:
        raise ValueError("No prediction candidate IDs matched the human labels.")

    classes = list(Label)
    confusion = {
        truth.value: {prediction.value: 0 for prediction in classes} for truth in classes
    }
    correct = 0
    for prediction in matched:
        truth = labels[prediction.candidate_id]
        confusion[truth.value][prediction.label.value] += 1
        correct += prediction.label == truth

    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in classes:
        name = label.value
        tp = confusion[name][name]
        fp = sum(confusion[truth.value][name] for truth in classes if truth != label)
        fn = sum(confusion[name][other.value] for other in classes if other != label)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}

    total = len(matched)
    truth_counts = Counter(labels[item.candidate_id] for item in matched)
    prediction_counts = Counter(item.label for item in matched)
    observed = correct / total
    expected = sum(
        (truth_counts[label] / total) * (prediction_counts[label] / total) for label in classes
    )
    kappa = _safe_divide(observed - expected, 1 - expected)

    latencies = [item.latency_ms for item in matched if item.latency_ms is not None]
    return {
        "n": total,
        "accuracy": observed,
        "cohen_kappa": kappa,
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "input_tokens": sum(item.input_tokens or 0 for item in matched),
        "output_tokens": sum(item.output_tokens or 0 for item in matched),
        "cost_usd": sum(item.cost_usd or 0 for item in matched),
    }


def compare_predictions(
    left: list[ClassificationResult], right: list[ClassificationResult]
) -> dict[str, Any]:
    left_by_id = {item.candidate_id: item for item in left}
    right_by_id = {item.candidate_id: item for item in right}
    shared_ids = sorted(set(left_by_id) & set(right_by_id))
    if not shared_ids:
        raise ValueError("The prediction files have no candidate IDs in common.")
    labels = {candidate_id: left_by_id[candidate_id].label for candidate_id in shared_ids}
    matched_right = [right_by_id[candidate_id] for candidate_id in shared_ids]
    agreement = evaluate(matched_right, labels)

    def summary(items: list[ClassificationResult]) -> dict[str, Any]:
        selected = [item for item in items if item.candidate_id in shared_ids]
        counts = Counter(item.label.value for item in selected)
        latencies = [item.latency_ms for item in selected if item.latency_ms is not None]
        return {
            "classifier": selected[0].classifier,
            "models": sorted({item.model for item in selected if item.model}),
            "label_counts": dict(sorted(counts.items())),
            "mean_confidence": sum(item.confidence for item in selected) / len(selected),
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "input_tokens": sum(item.input_tokens or 0 for item in selected),
            "output_tokens": sum(item.output_tokens or 0 for item in selected),
            "cost_usd": sum(item.cost_usd or 0 for item in selected),
        }

    return {
        "n_shared": len(shared_ids),
        "agreement_rate": agreement["accuracy"],
        "cohen_kappa": agreement["cohen_kappa"],
        "confusion_matrix_left_rows_right_columns": agreement["confusion_matrix"],
        "left": summary(left),
        "right": summary(right),
        "limitation": "Agreement between automated classifiers is not accuracy without human labels.",
    }
