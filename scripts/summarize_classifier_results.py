"""Recompute v4 results and evidence groups from saved model predictions."""
import json
from collections import Counter
from pathlib import Path

from context_recycling.evaluation import compare_predictions
from context_recycling.io import write_json, write_jsonl
from context_recycling.models import ClassificationResult


def predictions(path):
    with open(path) as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    indexed = {row['candidate_id']: row for row in rows}
    if len(rows) != len(indexed):
        raise ValueError(f'Duplicate predictions in {path}')
    return indexed


def consensus(left, right):
    return {key for key in left.keys() & right.keys()
            if left[key]['label'] == right[key]['label'] == 'redundant'}


def main():
    left = predictions('results/sweagent_full_jev_predictions_v4.jsonl')
    right = predictions('results/sweagent_full_llm_predictions_v4.jsonl')
    agreed = consensus(left, right)
    expected, trajectories = set(), set()
    types, groups, adjacency = Counter(), Counter(), Counter()
    file_rows, strict_trajectories = [], set()
    consensus_rows = []
    strict_open_calls = 0
    with open('results/sweagent_full_candidates_v4.jsonl') as stream:
        for line in stream:
            candidate = json.loads(line)
            key = candidate['candidate_id']
            if key in expected:
                raise ValueError(f'Duplicate candidate: {key}')
            expected.add(key)
            evidence = candidate['evidence']
            delta = evidence.get('line_delta', {})
            history = evidence.get('history_line_delta', {})
            matches = evidence['pair_exact_match']
            if key not in agreed:
                continue
            types[candidate['match_type']] += 1
            trajectories.add(candidate['trajectory_id'])
            consensus_rows.append({
                'candidate_id': key, 'trajectory_id': candidate['trajectory_id'],
                'match_type': candidate['match_type'],
                'previous_step': candidate['previous']['step'],
                'current_step': candidate['current']['step'],
                'target': candidate['current']['target'],
                'jev_label': left[key]['label'], 'mistral_label': right[key]['label'],
                'jev_confidence': left[key]['confidence'],
                'mistral_confidence': right[key]['confidence'],
                'evidence_group': None,
            })
            if candidate['match_type'] != 'file_view':
                continue
            if candidate['content_unchanged'] is not True or candidate['target_modified']:
                group = 'changed_content_or_detected_edit'
            elif all(matches.get(name) is True for name in ('action', 'thought', 'response', 'observation')) and evidence['stated_reasoning_available']:
                group = 'exact_action_thought_response_observation'
                strict_trajectories.add(candidate['trajectory_id'])
                adjacency['adjacent' if candidate['distance_steps'] == 1 else 'nonadjacent'] += 1
                strict_open_calls += candidate['current']['metadata'].get('raw_action', '').strip().startswith('open ')
            elif matches['observation'] is True:
                group = 'same_observation_different_action_or_reasoning'
            else:
                group = 'unchanged_overlap_but_different_observation'
            groups[group] += 1
            consensus_rows[-1]['evidence_group'] = group
            file_rows.append({
                'candidate_id': key, 'trajectory_id': candidate['trajectory_id'],
                'group': group, 'distance_steps': candidate['distance_steps'],
                'target': candidate['current']['target'], 'pair_exact_match': matches,
                'changed_shared_lines': delta['changed_shared_lines'],
                'new_to_previous_window_lines': delta['new_to_previous_window_lines'],
                'successful_target_edit_steps': evidence['target_edit_steps']['succeeded'],
                'failed_target_edit_steps': evidence['target_edit_steps']['failed'],
            })
    for name, rows in [('Jev', left), ('Mistral', right)]:
        if set(rows) != expected:
            raise ValueError(f'{name}: missing {len(expected-set(rows))}, extra {len(set(rows)-expected)}')
    comparison = compare_predictions(
        [ClassificationResult.from_dict(row) for row in left.values()],
        [ClassificationResult.from_dict(row) for row in right.values()],
    )
    summary = {
        'version': 4, 'consensus_redundant': len(agreed), 'trajectories': len(trajectories),
        'types': dict(types), 'file_view_groups': dict(groups),
        'strict_match_trajectories': len(strict_trajectories),
        'strict_match_adjacency': dict(adjacency), 'strict_match_open_calls': strict_open_calls,
        'method': 'Mutually exclusive evidence groups; changed content or confirmed edit takes precedence. Exact reasoning equality is stated text equality, not proof of waste. These are not human gold labels.',
    }
    write_json('results/jev_vs_llm_comparison_v4.json', comparison)
    write_json('results/consensus_file_read_audit_v4_summary.json', summary)
    write_jsonl('results/consensus_file_read_audit_v4.jsonl', file_rows)
    write_jsonl('results/consensus_redundant_candidates_v4.jsonl', consensus_rows)
    consolidated = {'candidate_count': len(expected), 'comparison': comparison, 'consensus_audit': summary}
    write_json('results/results_v4.json', consolidated)
    print(json.dumps(consolidated, indent=2))


if __name__ == '__main__':
    main()
