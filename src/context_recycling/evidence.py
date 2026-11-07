"""Retrospective trace evidence. These features do not assign usefulness labels."""
from __future__ import annotations

from .models import Event

READS = {"view_file", "read_file", "open_file"}
MUTATIONS = {"edit_file", "write_file", "apply_patch", "create_file"}


def _lines(event: Event) -> dict[str, str]:
    value = event.metadata.get("line_contents")
    return value if isinstance(value, dict) else {}


def _numbers(keys: set[str]) -> list[int]:
    return sorted(int(key) for key in keys)


def build_evidence(trajectory: list[Event], previous_index: int, current_index: int) -> dict:
    previous, current = trajectory[previous_index], trajectory[current_index]
    between = trajectory[previous_index + 1:current_index]
    after = trajectory[current_index + 1:]
    same_file = current.tool in READS and bool(current.target)
    a, b = _lines(previous), _lines(current)
    shared = set(a) & set(b)
    unchanged = {key for key in shared if a[key] == b[key]}
    changed = shared - unchanged
    added = set(b) - set(a)
    edits = [event for event in between if event.tool in MUTATIONS and
             event.target == current.target] if same_file else []
    edit_steps = {
        status: [event.step for event in edits
                 if event.metadata.get("mutation_status", "unknown") == status]
        for status in ("succeeded", "failed", "unknown")
    }
    # Keep only observations from the current known file revision. Line numbers
    # can move after edits; invalidate the old map before adding an applied edit's
    # returned window. Failed proposed edits must never update this map.
    seen: dict[str, str] = {}
    invalidated_at = None
    prior_views = []
    if same_file:
        for event in trajectory[:current_index]:
            if event.target != current.target:
                continue
            if event.tool in MUTATIONS:
                status = event.metadata.get("mutation_status", "unknown")
                if status == "failed":
                    continue
                seen.clear()
                invalidated_at = event.step
                if status == "succeeded":
                    seen.update(_lines(event))
            elif event.tool in READS:
                seen.update(_lines(event))
                prior_views.append(event.step)

    equality = {}
    for name, left, right in (
        ("action", previous.metadata.get("raw_action"), current.metadata.get("raw_action")),
        ("thought", previous.metadata.get("thought"), current.metadata.get("thought")),
        ("response", previous.metadata.get("response"), current.metadata.get("response")),
        ("observation", previous.content, current.content),
    ):
        equality[name] = left == right if left is not None and right is not None else None

    result = {
        "version": 4,
        "context_scope": "entire_recorded_trajectory_retrospective",
        "trajectory_steps": [event.step for event in trajectory],
        "prior_same_file_view_steps": prior_views,
        "pair_exact_match": equality,
        "content_unchanged_scope": "shared_numbered_lines" if shared else "entire_observation",
        "stated_reasoning_available": bool(previous.metadata.get("thought")) and
                                      bool(current.metadata.get("thought")),
        "target_edit_steps": edit_steps,
        "target_edit_tracking_available": same_file,
        "intervening_mutations": [
            {"step": event.step, "target": event.target,
             "status": event.metadata.get("mutation_status", "unknown"),
             "edit_range": event.metadata.get("edit_range")}
            for event in between if event.tool in MUTATIONS
        ],
        "intervening_shell_steps": [event.step for event in between
                                    if event.tool in {"shell", "bash", "exec", "run_command"}],
        "later_same_target_edit_steps": [event.step for event in after
            if same_file and event.target == current.target and event.tool in MUTATIONS],
        "same_intent_is_inferred": False,
        "limits": [
            "Matching reasoning text is stated-reasoning equality, not proof of intent or waste.",
            "Shell commands may change files; their effects are not fully tracked.",
        ],
    }
    if same_file:
        result["line_delta"] = {
            "available": bool(a) and bool(b),
            "previous_line_count": len(a), "current_line_count": len(b),
            "shared_lines": _numbers(shared),
            "unchanged_shared_lines": _numbers(unchanged),
            "changed_shared_lines": _numbers(changed),
            "new_to_previous_window_lines": _numbers(added),
            "current_coverage_by_previous": len(shared) / len(b) if b else None,
            "all_current_lines_repeat_previous": bool(b) and set(b) <= unchanged,
            "same_numbered_window": bool(a) and bool(b) and set(a) == set(b),
        }
        result["history_line_delta"] = {
            "available": bool(b),
            "history_invalidated_at_step": invalidated_at,
            "not_observed_since_last_known_edit_lines": _numbers(set(b) - set(seen)),
            "unchanged_from_last_observation_lines": _numbers({key for key in b
                if key in seen and b[key] == seen[key]}),
            "changed_from_last_observation_lines": _numbers({key for key in b
                if key in seen and b[key] != seen[key]}),
            "limits": "Compares recorded line coordinates, not version-control snapshots.",
        }
    else:
        result["target_edit_tracking_available"] = False
        result["limits"].append(
            "Search directory edits and command dependencies are not established by the open-file state."
        )
    return result
