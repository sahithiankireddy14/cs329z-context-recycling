#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
trap 'printf "%s\n" "$?" > results/package_v4.exitcode' EXIT
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
task_python="${TASK_PYTHON:-python}"
"$task_python" scripts/summarize_classifier_results.py
"$task_python" scripts/export_results_v4.py
"$task_python" -m context_recycling.cli compare \
  --left artifacts/v4/sweagent_full_jev_predictions_v4.jsonl \
  --right artifacts/v4/sweagent_full_llm_predictions_v4.jsonl \
  --output results/recomputed_comparison_v4.json
"$task_python" - <<'PY'
import hashlib
import json
from pathlib import Path
archive = Path('artifacts/v4')
checksums = json.loads((archive / 'checksums.json').read_text())
for name, expected in checksums.items():
    if hashlib.sha256((archive / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f'Archive checksum mismatch: {name}')
original = json.loads((archive / 'jev_vs_llm_comparison_v4.json').read_text())
recomputed = json.loads(Path('results/recomputed_comparison_v4.json').read_text())
if original != recomputed:
    raise ValueError('Archived predictions did not reproduce the saved comparison exactly.')
print('Archive checksums verified; saved comparison reproduced exactly.')
PY
"$task_python" scripts/write_results_readme.py
"$task_python" scripts/verify_results_v4.py
bash -n scripts/run_full_classifier_comparison.sh scripts/check_saved_v4_results.sh
printf '%s\n' 'Results archive and README generated successfully.'
