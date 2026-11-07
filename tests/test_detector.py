from context_recycling.detector import detect_candidates
from context_recycling.models import Event


def test_detects_latest_overlapping_file_view_and_tracks_edit() -> None:
    events = [
        Event("t", 1, "view_file", target="a.py", line_start=1, line_end=10, content="old"),
        Event("t", 2, "edit_file", target="a.py", metadata={"mutation_status": "succeeded"}),
        Event("t", 3, "view_file", target="a.py", line_start=5, line_end=12, content="new"),
    ]

    candidates = detect_candidates(events)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.candidate_id == "t:1:3"
    assert candidate.overlap_lines == 6
    assert candidate.target_modified is True
    assert candidate.content_unchanged is False
    assert candidate.following == ()


def test_does_not_match_disjoint_ranges() -> None:
    events = [
        Event("t", 1, "view_file", target="a.py", line_start=1, line_end=5),
        Event("t", 2, "view_file", target="a.py", line_start=6, line_end=10),
    ]
    assert detect_candidates(events) == []


def test_ignores_mechanical_scroll_boundary_overlap() -> None:
    events = [
        Event("t", 1, "view_file", target="a.py", line_start=1, line_end=100),
        Event("t", 2, "view_file", target="a.py", line_start=99, line_end=198),
    ]
    assert detect_candidates(events) == []


def test_detects_repeated_search() -> None:
    events = [
        Event("t", 1, "search", query="needle"),
        Event("t", 2, "think"),
        Event("t", 3, "rg", query="needle"),
    ]
    candidate = detect_candidates(events)[0]
    assert candidate.match_type == "search"
    assert candidate.distance_steps == 2


def test_compares_only_overlapping_numbered_lines() -> None:
    events = [
        Event(
            "t",
            1,
            "view_file",
            target="a.py",
            line_start=1,
            line_end=7,
            content="different wrappers",
            metadata={
                "line_contents": {
                    "1": "a",
                    "2": "b",
                    "3": "c",
                    "4": "d",
                    "5": "e",
                    "6": "f",
                    "7": "g",
                }
            },
        ),
        Event(
            "t",
            2,
            "view_file",
            target="a.py",
            line_start=3,
            line_end=9,
            content="another wrapper",
            metadata={
                "line_contents": {
                    "3": "c",
                    "4": "d",
                    "5": "e",
                    "6": "f",
                    "7": "g",
                    "8": "h",
                    "9": "i",
                }
            },
        ),
    ]
    candidate = detect_candidates(events)[0]
    assert candidate.content_unchanged is True



def view(step, lines, **metadata):
    return Event('t', step, 'view_file', target='a.py', line_start=min(lines),
                 line_end=max(lines), content=str(lines),
                 metadata={'line_contents': {str(k): v for k, v in lines.items()}, **metadata})


def test_partial_overlap_reports_new_lines_and_checks_all_history():
    events = [view(1, {k: str(k) for k in range(1, 11)}),
              view(2, {k: str(k) for k in range(6, 16)}),
              view(3, {k: str(k) for k in range(1, 11)})]
    pairs = detect_candidates(events)
    delta = pairs[0].evidence['line_delta']
    assert delta['new_to_previous_window_lines'] == list(range(11, 16))
    assert delta['all_current_lines_repeat_previous'] is False
    # The third view reveals lines absent from the nearest window, but those
    # lines were already observed in the earlier window.
    assert pairs[1].evidence['line_delta']['new_to_previous_window_lines'] == list(range(1, 6))
    assert pairs[1].evidence['history_line_delta']['not_observed_since_last_known_edit_lines'] == []


def test_failed_edit_is_not_a_change_and_does_not_pollute_history():
    events = [view(1, {k: 'original' for k in range(1, 7)}),
              Event('t', 2, 'edit_file', target='a.py', metadata={
                  'mutation_status': 'failed', 'line_contents': {'1': 'proposed'}}),
              view(3, {k: 'original' for k in range(1, 7)})]
    candidate = detect_candidates(events)[0]
    assert candidate.target_modified is False
    assert candidate.evidence['target_edit_steps']['failed'] == [2]
    assert candidate.evidence['history_line_delta']['changed_from_last_observation_lines'] == []


def test_successful_edit_invalidates_old_line_coordinates():
    events = [view(1, {k: 'old' for k in range(1, 7)}),
              Event('t', 2, 'edit_file', target='a.py', metadata={'mutation_status': 'succeeded'}),
              view(3, {k: 'new' for k in range(1, 7)})]
    candidate = detect_candidates(events)[0]
    assert candidate.target_modified is True
    assert candidate.evidence['line_delta']['changed_shared_lines'] == list(range(1, 7))
    assert candidate.evidence['history_line_delta']['history_invalidated_at_step'] == 2
    assert candidate.evidence['history_line_delta']['not_observed_since_last_known_edit_lines'] == list(range(1, 7))


def test_complete_context_and_exact_reasoning_survive_roundtrip():
    from context_recycling.models import Candidate
    events = [Event('t', 1, 'control'),
              view(2, {k: 'a' for k in range(1, 7)}, raw_action='goto 1', thought='inspect', response='inspect\ngoto 1'),
              Event('t', 3, 'run_command', command='python repro.py'),
              view(4, {k: 'a' for k in range(1, 7)}, raw_action='goto 1', thought='inspect', response='inspect\ngoto 1'),
              Event('t', 5, 'control'), Event('t', 6, 'edit_file', target='a.py')]
    candidate = Candidate.from_dict(detect_candidates(events)[0].to_dict())
    reconstructed = (*candidate.preceding, candidate.previous, *candidate.intervening,
                     candidate.current, *candidate.following)
    assert list(reconstructed) == events
    assert candidate.evidence['pair_exact_match'] == dict.fromkeys(['action', 'thought', 'response', 'observation'], True)
    assert candidate.evidence['later_same_target_edit_steps'] == [6]


def test_same_query_different_known_scope_does_not_match():
    events = [Event('t', 1, 'search', target='a.py', query='q'),
              Event('t', 2, 'search', target='b.py', query='q')]
    assert detect_candidates(events) == []


def test_numbered_command_output_does_not_hide_changed_test_result():
    events = [Event('t', 1, 'run_command', command='python repro.py',
                    content='1:same\nPASS', metadata={'line_contents': {'1': 'same'}}),
              Event('t', 2, 'run_command', command='python repro.py',
                    content='1:same\nFAIL', metadata={'line_contents': {'1': 'same'}})]
    assert detect_candidates(events)[0].content_unchanged is False


def test_scroll_continuation_requires_actual_progress_not_a_stuck_window():
    a = view(1, {k: str(k) for k in range(1, 101)}, raw_action='scroll_down')
    b = view(2, {k: str(k) for k in range(82, 182)}, raw_action='scroll_down')
    assert detect_candidates([a, b]) == []
    stuck = view(3, {k: str(k) for k in range(82, 182)}, raw_action='scroll_down')
    candidates = detect_candidates([a, b, stuck])
    assert len(candidates) == 1
    assert candidates[0].candidate_id == 't:2:3'
    assert candidates[0].evidence['line_delta']['all_current_lines_repeat_previous'] is True
