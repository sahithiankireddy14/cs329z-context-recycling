#!/usr/bin/env bash
set -euo pipefail
run_results_dir="${RUN_RESULTS_DIR:-results}"
mkdir -p "$run_results_dir"

if [ ! -f .env ]; then
  echo "Missing .env with OPENROUTER_API_KEY" >&2
  exit 1
fi

set -a
source .env
set +a

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY is empty" >&2
  exit 1
fi

PYTHONPATH=src python -m context_recycling.cli classify \
  --candidates results/sweagent_full_candidates_v4.jsonl \
  --output "$run_results_dir/sweagent_full_jev_predictions_v4.jsonl" \
  --classifier openrouter-jev \
  --model typesafe/jev-1.14 \
  --workers 8 \
  --resume \
  > "$run_results_dir/jev_full_run_v4.log" 2>&1 &
jev_process=$!

PYTHONPATH=src python -m context_recycling.cli classify \
  --candidates results/sweagent_full_candidates_v4.jsonl \
  --output "$run_results_dir/sweagent_full_llm_predictions_v4.jsonl" \
  --classifier openrouter-llm \
  --model mistralai/mistral-small-3.2-24b-instruct \
  --workers 8 \
  --resume \
  > "$run_results_dir/llm_full_run_v4.log" 2>&1 &
llm_process=$!

set +e
wait "$jev_process"
jev_exit=$?
wait "$llm_process"
llm_exit=$?
set -e

if [ "$jev_exit" -ne 0 ] || [ "$llm_exit" -ne 0 ]; then
  echo "Classifier failure: Jev=$jev_exit LLM=$llm_exit" >&2
  exit 1
fi

PYTHONPATH=src python -m context_recycling.cli compare \
  --left "$run_results_dir/sweagent_full_jev_predictions_v4.jsonl" \
  --right "$run_results_dir/sweagent_full_llm_predictions_v4.jsonl" \
  --output "$run_results_dir/jev_vs_llm_comparison_v4.json"
