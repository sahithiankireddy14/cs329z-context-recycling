from context_recycling.evaluation import compare_predictions, evaluate
from context_recycling.models import ClassificationResult, Label


def prediction(candidate_id: str, label: Label) -> ClassificationResult:
    return ClassificationResult(candidate_id, label, 1.0, {label.value: 1.0}, "test")


def test_perfect_predictions_have_perfect_accuracy_and_kappa() -> None:
    predictions = [
        prediction("a", Label.REDUNDANT),
        prediction("b", Label.LEGITIMATE),
        prediction("c", Label.VERIFICATION),
    ]
    labels = {
        "a": Label.REDUNDANT,
        "b": Label.LEGITIMATE,
        "c": Label.VERIFICATION,
    }
    metrics = evaluate(predictions, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_compare_predictions_reports_agreement() -> None:
    left = [prediction("a", Label.REDUNDANT), prediction("b", Label.LEGITIMATE)]
    right = [prediction("a", Label.REDUNDANT), prediction("b", Label.REDUNDANT)]
    comparison = compare_predictions(left, right)
    assert comparison["n_shared"] == 2
    assert comparison["agreement_rate"] == 0.5
