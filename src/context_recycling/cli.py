from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import time
from pathlib import Path

from .classifiers import (
    OpenRouterJevClassifier,
    OpenRouterLLMClassifier,
)
from .detector import detect_candidates
from .evaluation import compare_predictions, evaluate
from .io import read_jsonl, write_json, write_jsonl
from .models import Candidate, ClassificationResult, Event, Label
from .sweagent import convert_directory


def _detect(args: argparse.Namespace) -> None:
    events = [Event.from_dict(row) for row in read_jsonl(args.events)]
    candidates = detect_candidates(events)
    write_jsonl(args.output, (candidate.to_dict() for candidate in candidates))
    print(f"Detected {len(candidates)} candidate re-reads -> {args.output}")


def _classify(args: argparse.Namespace) -> None:
    candidates = [Candidate.from_dict(row) for row in read_jsonl(args.candidates)]
    if args.classifier == "openrouter-jev":
        classifier = OpenRouterJevClassifier(model=args.model or "typesafe/jev-1.14")
    else:
        classifier = OpenRouterLLMClassifier(model=args.model or "mistralai/mistral-small-3.2-24b-instruct")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if args.resume and output.exists():
        completed = {row["candidate_id"] for row in read_jsonl(output)}
    mode = "a" if args.resume else "w"
    pending = [candidate for candidate in candidates if candidate.candidate_id not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        print(f"No pending candidates; predictions already complete -> {args.output}")
        return
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    def classify_with_retry(candidate: Candidate) -> ClassificationResult:
        for attempt in range(1, args.max_retries + 1):
            try:
                result = classifier.classify(candidate)
                if args.delay:
                    time.sleep(args.delay)
                return result
            except RuntimeError:
                if attempt == args.max_retries:
                    raise
                time.sleep(min(2 ** (attempt - 1), 30))
        raise RuntimeError("Classification retry loop exited unexpectedly.")

    with output.open(mode, encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(classify_with_retry, candidate): candidate
                for candidate in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                stream.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
                stream.flush()
                if index == 1 or index % 25 == 0 or index == len(pending):
                    print(
                        f"Classified {len(completed) + index}/{len(candidates)} "
                        f"candidate re-reads",
                        flush=True,
                    )
    print(f"Predictions -> {args.output}")


def _evaluate(args: argparse.Namespace) -> None:
    predictions = [ClassificationResult.from_dict(row) for row in read_jsonl(args.predictions)]
    labels = {row["candidate_id"]: Label(row["label"]) for row in read_jsonl(args.labels)}
    metrics = evaluate(predictions, labels)
    write_json(args.output, metrics)
    print(
        f"Evaluated {metrics['n']} candidates: accuracy={metrics['accuracy']:.3f}, "
        f"kappa={metrics['cohen_kappa']:.3f} -> {args.output}"
    )


def _convert_sweagent(args: argparse.Namespace) -> None:
    events = convert_directory(args.input_dir, results_path=args.results)
    write_jsonl(args.output, (event.to_dict() for event in events))
    trajectory_count = len({event.trajectory_id for event in events})
    print(f"Converted {len(events)} events from {trajectory_count} trajectories -> {args.output}")


def _compare(args: argparse.Namespace) -> None:
    left = [ClassificationResult.from_dict(row) for row in read_jsonl(args.left)]
    right = [ClassificationResult.from_dict(row) for row in read_jsonl(args.right)]
    comparison = compare_predictions(left, right)
    write_json(args.output, comparison)
    print(
        f"Compared {comparison['n_shared']} predictions: "
        f"agreement={comparison['agreement_rate']:.3f}, "
        f"kappa={comparison['cohen_kappa']:.3f} -> {args.output}"
    )


def _filter_predictions(args: argparse.Namespace) -> None:
    candidate_ids = {row["candidate_id"] for row in read_jsonl(args.candidates)}
    predictions = [
        row for row in read_jsonl(args.predictions) if row["candidate_id"] in candidate_ids
    ]
    write_jsonl(args.output, predictions)
    print(f"Retained {len(predictions)} predictions -> {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure re-reading in coding-agent traces.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Detect candidate re-reads.")
    detect.add_argument("--events", required=True)
    detect.add_argument("--output", required=True)
    detect.set_defaults(func=_detect)

    classify = subparsers.add_parser("classify", help="Classify detected candidates.")
    classify.add_argument("--candidates", required=True)
    classify.add_argument("--output", required=True)
    classify.add_argument(
        "--classifier",
        choices=("openrouter-jev", "openrouter-llm"),
        required=True,
    )
    classify.add_argument("--model")
    classify.add_argument("--resume", action="store_true")
    classify.add_argument("--limit", type=int)
    classify.add_argument("--max-retries", type=int, default=5)
    classify.add_argument("--delay", type=float, default=0.0)
    classify.add_argument("--workers", type=int, default=1)
    classify.set_defaults(func=_classify)

    evaluation = subparsers.add_parser("evaluate", help="Evaluate against human labels.")
    evaluation.add_argument("--predictions", required=True)
    evaluation.add_argument("--labels", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.set_defaults(func=_evaluate)

    convert = subparsers.add_parser(
        "convert-sweagent", help="Convert SWE-agent .traj files to normalized events."
    )
    convert.add_argument("--input-dir", required=True)
    convert.add_argument("--results")
    convert.add_argument("--output", required=True)
    convert.set_defaults(func=_convert_sweagent)

    comparison = subparsers.add_parser("compare", help="Compare two classifier outputs.")
    comparison.add_argument("--left", required=True)
    comparison.add_argument("--right", required=True)
    comparison.add_argument("--output", required=True)
    comparison.set_defaults(func=_compare)

    filtering = subparsers.add_parser(
        "filter-predictions", help="Keep predictions for the current candidate set."
    )
    filtering.add_argument("--candidates", required=True)
    filtering.add_argument("--predictions", required=True)
    filtering.add_argument("--output", required=True)
    filtering.set_defaults(func=_filter_predictions)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
