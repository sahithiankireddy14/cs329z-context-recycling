"""Export the saved public-data experiment outputs for inclusion in GitHub."""
from pathlib import Path
import hashlib
import json
import shutil

destination = Path('artifacts/v4')
destination.mkdir(parents=True, exist_ok=True)
names = [
    'sweagent_full_jev_predictions_v4.jsonl',
    'sweagent_full_llm_predictions_v4.jsonl',
    'jev_vs_llm_comparison_v4.json',
    'consensus_file_read_audit_v4.jsonl',
    'consensus_redundant_candidates_v4.jsonl',
    'consensus_file_read_audit_v4_summary.json',
    'sweagent_full_evidence_v4_summary.json',
    'classifier_run_v4_manifest.json', 'results_v4.json',
]
checksums = {}
for name in names:
    target = destination / name
    shutil.copyfile(Path('results') / name, target)
    checksums[name] = hashlib.sha256(target.read_bytes()).hexdigest()
(destination / 'checksums.json').write_text(json.dumps(checksums, indent=2) + '\n')
print(f'Exported {len(names)} result files and SHA-256 checksums to {destination}')
