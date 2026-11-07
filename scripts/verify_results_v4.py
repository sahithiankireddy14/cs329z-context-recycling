"""Verify the report's v4 archive offline without modifying archived evidence."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from context_recycling.evaluation import compare_predictions
from context_recycling.models import ClassificationResult


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(archive=Path('artifacts/v4')):
    def read(name):
        text = (archive / name).read_text()
        if name.endswith('.jsonl'):
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        return json.loads(text)

    checksums = read('checksums.json')
    expected_files = {
        'sweagent_full_jev_predictions_v4.jsonl', 'sweagent_full_llm_predictions_v4.jsonl',
        'jev_vs_llm_comparison_v4.json', 'results_v4.json',
        'consensus_file_read_audit_v4.jsonl', 'consensus_redundant_candidates_v4.jsonl',
        'consensus_file_read_audit_v4_summary.json', 'sweagent_full_evidence_v4_summary.json',
        'classifier_run_v4_manifest.json',
    }
    require(set(checksums) == expected_files, 'Archive checksum inventory is incomplete.')
    for name, expected in checksums.items():
        require(hashlib.sha256((archive / name).read_bytes()).hexdigest() == expected,
                f'Archive checksum mismatch: {name}')

    left = read('sweagent_full_jev_predictions_v4.jsonl')
    right = read('sweagent_full_llm_predictions_v4.jsonl')
    ids = []
    for rows in (left, right):
        unique = {row['candidate_id'] for row in rows}
        require(len(rows) == len(unique) == 1545, 'Incomplete or duplicate predictions.')
        ids.append(unique)
    require(ids[0] == ids[1], 'Classifier candidate sets differ.')
    comparison = compare_predictions(
        [ClassificationResult.from_dict(row) for row in left],
        [ClassificationResult.from_dict(row) for row in right],
    )
    require(comparison == read('jev_vs_llm_comparison_v4.json'), 'Comparison differs.')
    result = read('results_v4.json')
    require(result['candidate_count'] == 1545 and result['comparison'] == comparison,
            'Consolidated results differ.')
    audit = read('consensus_file_read_audit_v4_summary.json')
    require(result['consensus_audit'] == audit, 'Consensus summaries differ.')
    consensus = read('consensus_redundant_candidates_v4.jsonl')
    agreed = {row['candidate_id'] for row in left if row['label'] == 'redundant'} & {
        row['candidate_id'] for row in right if row['label'] == 'redundant'}
    require(len(consensus) == len(agreed) == audit['consensus_redundant'] == 153
            and {row['candidate_id'] for row in consensus} == agreed, 'Consensus differs.')
    require(len({row['trajectory_id'] for row in consensus}) == audit['trajectories'] == 62,
            'Consensus trajectory count differs.')
    require(Counter(row['match_type'] for row in consensus) == audit['types']
            == {'file_view': 122, 'command': 25, 'search': 6}, 'Consensus types differ.')
    files = read('consensus_file_read_audit_v4.jsonl')
    require(len(files) == 122 and {row['candidate_id'] for row in files}
            == {row['candidate_id'] for row in consensus if row['match_type'] == 'file_view'},
            'File-view audit coverage differs.')
    require(Counter(row['group'] for row in files) == audit['file_view_groups'] == {
        'exact_action_thought_response_observation': 21,
        'same_observation_different_action_or_reasoning': 77,
        'unchanged_overlap_but_different_observation': 10,
        'changed_content_or_detected_edit': 14,
    }, 'Report Table 1 differs.')
    evidence = read('sweagent_full_evidence_v4_summary.json')
    require((evidence['trajectories'], evidence['events'], evidence['counts']['candidates'])
            == (496, 10330, 1545), 'Report dataset totals differ.')
    manifest = read('classifier_run_v4_manifest.json')
    require(manifest['status'] == 'complete'
            and manifest['verified_predictions_per_classifier'] == 1545
            and manifest['missing_predictions'] == manifest['duplicate_predictions'] == 0,
            'Run manifest is incomplete.')
    print('Verified all 9 archive checksums, 1,545 predictions per model, exact comparison,')
    print('and the report consensus dataset and Table 1 (153 cases; 122 file views).')


if __name__ == '__main__':
    verify()
