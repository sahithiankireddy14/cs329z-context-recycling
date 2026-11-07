#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
trap 'printf "%s\n" "$?" > results/reproduction_check_v4.exitcode' EXIT
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
task_python="${TASK_PYTHON:-python}"
"$task_python" -m pytest -q
"$task_python" scripts/verify_results_v4.py
"$task_python" - <<'PY'
import hashlib
import json
from pathlib import Path
manifest = json.loads(Path('artifacts/v4/classifier_run_v4_manifest.json').read_text())
candidate = Path('results/sweagent_full_candidates_v4.jsonl')
if hashlib.sha256(candidate.read_bytes()).hexdigest() != manifest['candidate_sha256']:
    raise ValueError('Regenerated candidates differ from the classified v4 dataset.')
PY
"$task_python" scripts/audit_trajectory_evidence.py
"$task_python" scripts/summarize_classifier_results.py
"$task_python" - <<'PY'
import json
from pathlib import Path
for name in ('results_v4.json', 'sweagent_full_evidence_v4_summary.json',
             'consensus_file_read_audit_v4.jsonl', 'consensus_redundant_candidates_v4.jsonl'):
    def load(path):
        text = path.read_text()
        return [json.loads(line) for line in text.splitlines()] if path.suffix == '.jsonl' else json.loads(text)
    if load(Path('results') / name) != load(Path('artifacts/v4') / name):
        raise ValueError(f'Regenerated output differs from archive: {name}')
PY
printf '%s\n' 'Saved v4 results successfully validated and regenerated.'
