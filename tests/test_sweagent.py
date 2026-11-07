import json

from context_recycling.sweagent import convert_trajectory


def test_converts_file_view_and_numbered_lines() -> None:
    payload = {
        "trajectory": [
            {
                "action": "open pkg/a.py\n",
                "observation": "[File: /repo/pkg/a.py (20 lines total)]\n5:alpha\n6:beta\n",
                "state": json.dumps({"open_file": "/repo/old.py", "working_dir": "/repo"}),
                "thought": "inspect",
            }
        ]
    }
    event = convert_trajectory(payload, "task", resolved=True)[0]
    assert event.tool == "view_file"
    assert event.target == "pkg/a.py"
    assert event.line_start == 5
    assert event.line_end == 6
    assert event.metadata["line_contents"] == {"5": "alpha", "6": "beta"}
    assert event.metadata["resolved"] is True


def test_converts_search_and_edit_targets() -> None:
    payload = {
        "trajectory": [
            {
                "action": 'search_file "needle"\n',
                "observation": "Found 1 match",
                "state": json.dumps({"open_file": "/repo/pkg/a.py", "working_dir": "/repo"}),
            },
            {
                "action": "edit 5:5\nreplacement\nend_of_edit\n",
                "observation": "[File: /repo/pkg/a.py (20 lines total)]\n5:replacement\n",
                "state": json.dumps({"open_file": "/repo/pkg/a.py", "working_dir": "/repo"}),
            },
        ]
    }
    search, edit = convert_trajectory(payload, "task")
    assert search.tool == "search"
    assert search.query == "needle"
    assert search.target == "pkg/a.py"
    assert edit.tool == "edit_file"
    assert edit.target == "pkg/a.py"


def test_comment_only_action_is_not_a_shell_command() -> None:
    payload = {
        "trajectory": [
            {
                "action": "# No command to run; end of session.\n",
                "observation": "",
                "state": json.dumps({"open_file": "/repo/pkg/a.py", "working_dir": "/repo"}),
            }
        ]
    }
    event = convert_trajectory(payload, "task")[0]
    assert event.tool == "control"
    assert event.command is None



def test_failed_edit_preserves_response_but_discards_proposed_lines():
    payload = {'trajectory': [{'action': 'edit 1:1\nnew\nend_of_edit',
        'response': 'try edit', 'thought': 'fix',
        'state': {'open_file': '/repo/a.py', 'working_dir': '/repo'},
        'observation': 'Your proposed edit has introduced new syntax error(s).\n'
                       '[File: /repo/a.py (1 lines total)]\n1:proposed\n'
                       'Your changes have NOT been applied.'}]}
    event = convert_trajectory(payload, 't')[0]
    assert event.metadata['mutation_status'] == 'failed'
    assert 'line_contents' not in event.metadata
    assert event.metadata['response'] == 'try edit'


def test_successful_edit_and_failed_create_are_distinguished():
    payload = {'trajectory': [
        {'action': 'edit 1:1\nnew\nend_of_edit',
         'state': {'open_file': '/repo/a.py', 'working_dir': '/repo'},
         'observation': '[File: /repo/a.py (1 lines total)]\n1:new\nFile updated. Please review.'},
        {'action': 'create a.py', 'state': {'working_dir': '/repo'},
         'observation': "Error: File 'a.py' already exists.\n[File: /repo/a.py (1 lines total)]\n1:new\n"}]}
    success, failure = convert_trajectory(payload, 't')
    assert success.metadata['mutation_status'] == 'succeeded'
    assert failure.metadata['mutation_status'] == 'failed'
    assert 'line_contents' not in failure.metadata


def test_shell_level_mutation_failures_are_not_successful_edits():
    payload = {'trajectory': [
        {'action': 'edit 1:1\nx\nend_of_edit',
         'observation': '/bin/bash: line 2: syntax error near unexpected token'},
        {'action': 'create missing/a.py',
         'observation': 'main: line 1: missing/a.py: No such file or directory\nFile missing/a.py not found'}]}
    assert [e.metadata['mutation_status'] for e in convert_trajectory(payload, 't')] == ['failed', 'failed']
