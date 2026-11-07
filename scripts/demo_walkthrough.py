"""Print real trace evidence or archived results for a live IDE walkthrough."""
import argparse
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('view', choices=['trace', 'results'])
args = parser.parse_args()

if args.view == 'trace':
    path = ROOT / 'data/raw/sweagent_gpt4_20240402/trajs/sympy__sympy-16766.traj'
    if not path.exists():
        raise SystemExit('Download the public trajectories first, following README.md.')
    steps = json.loads(path.read_text())['trajectory']
    print('Actual trajectory: sympy__sympy-16766\n')
    for index in range(25, 30):
        row = steps[index - 1]
        print(f"Step {index}: {row['action'].strip().splitlines()[0]}")
    previous, current = steps[24], steps[28]
    print('\nExact raw-field comparison for steps 25 and 29:')
    for key in ['action', 'thought', 'response', 'observation']:
        print(f"  {key:12} identical = {previous.get(key) == current.get(key)}")
    print('\nStated reasoning at BOTH reads:')
    print(textwrap.fill(previous.get('thought', '').strip(), width=90))
    print('\nThese checks establish repetition, not human-verified waste.')
else:
    saved = json.loads((ROOT / 'artifacts/v4/results_v4.json').read_text())
    comparison, audit = saved['comparison'], saved['consensus_audit']
    print('Archived full experiment: 496 trajectories / 10,330 events')
    print(f"Candidates: {saved['candidate_count']}\n")
    print(f"{'Label':16} {'Jev':>8} {'Mistral':>8}")
    for label in ['legitimate', 'verification', 'redundant']:
        print(f"{label:16} {comparison['left']['label_counts'][label]:8} {comparison['right']['label_counts'][label]:8}")
    print(f"\nAgreement: {comparison['agreement_rate']:.2%}; kappa: {comparison['cohen_kappa']:.3f}")
    print(f"Both label redundant: {audit['consensus_redundant']}")
    print('122 file views / 25 commands / 6 searches')
    print('\nEvidence groups within the 122 file views:')
    names = {
        'exact_action_thought_response_observation': 'Exact action, reasoning, and observation',
        'same_observation_different_action_or_reasoning': 'Same observation; different action or reasoning',
        'unchanged_overlap_but_different_observation': 'Matching overlap; newly revealed lines',
        'changed_content_or_detected_edit': 'Changed content or confirmed intervening edit',
    }
    for key, label in names.items():
        print(f"  {audit['file_view_groups'][key]:3}  {label}")
    print('\nConsensus labels and evidence strength are not measured accuracy.')
