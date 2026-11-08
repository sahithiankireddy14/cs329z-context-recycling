"""Validate full context coverage and summarize the version 4 evidence artifacts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', default='data/processed/sweagent_gpt4_full_v4.jsonl')
    parser.add_argument('--candidates', default='results/sweagent_full_candidates_v4.jsonl')
    parser.add_argument('--output', default='results/sweagent_full_evidence_v4_summary.json')
    args = parser.parse_args()
    source = defaultdict(list)
    with open(args.events) as stream:
        for line in stream:
            event = json.loads(line)
            source[event['trajectory_id']].append(event)
    for events in source.values():
        events.sort(key=lambda event: event['step'])
    counts, mutations, examples = Counter(), Counter(), {}
    counts.update(file_views=0, searches=0, commands=0)
    for events in source.values():
        for event in events:
            if event['tool'] in {'edit_file', 'create_file'}:
                mutations[event['metadata'].get('mutation_status', 'unknown')] += 1
    with open(args.candidates) as stream:
        for line in stream:
            candidate = json.loads(line)
            counts['candidates'] += 1
            context = (candidate['preceding'] + [candidate['previous']] +
                       candidate['intervening'] + [candidate['current']] + candidate['following'])
            if context != source[candidate['trajectory_id']]:
                raise ValueError(f"Incomplete or inconsistent context: {candidate['candidate_id']}")
            counts['complete_context_validated'] += 1
            evidence = candidate['evidence']
            if candidate['match_type'] == 'search':
                counts['searches'] += 1
            elif candidate['match_type'] == 'command':
                counts['commands'] += 1
            if candidate['match_type'] == 'file_view':
                counts['file_views'] += 1
                delta, history = evidence['line_delta'], evidence['history_line_delta']
                checks = {
                    'file_views_with_new_lines_vs_previous': delta['new_to_previous_window_lines'],
                    'file_views_with_new_lines_vs_history': history['not_observed_since_last_known_edit_lines'],
                    'file_views_with_changed_shared_lines': delta['changed_shared_lines'],
                    'file_views_all_lines_repeat_previous': delta['all_current_lines_repeat_previous'],
                    'file_views_with_failed_edit_between': evidence['target_edit_steps']['failed'],
                    'file_views_with_successful_edit_between': evidence['target_edit_steps']['succeeded'],
                }
                for name, value in checks.items():
                    counts[name] += bool(value)
            if candidate['candidate_id'] in {
                'sympy__sympy-16766:25:29', 'astropy__astropy-13033:5:9',
                'django__django-13343:12:15',
            }:
                examples[candidate['candidate_id']] = evidence
    summary = {
        'version': 4, 'trajectories': len(source),
        'events': sum(map(len, source.values())), 'counts': dict(counts),
        'mutation_outcomes': dict(mutations), 'examples': examples,
        'classification_status': 'Evidence audit only; classification completion is recorded in classifier_run_v4_manifest.json and validated separately.',
    }
    Path(args.output).write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'examples'}, indent=2))


if __name__ == '__main__':
    main()
