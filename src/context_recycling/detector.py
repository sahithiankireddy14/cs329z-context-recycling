from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .models import Candidate, Event
from .evidence import build_evidence

FILE_READ_TOOLS = {"view_file", "read_file", "open_file"}
SEARCH_TOOLS = {"search", "grep", "ripgrep", "rg"}
SHELL_TOOLS = {"shell", "bash", "exec", "run_command"}
MUTATION_TOOLS = {"edit_file", "write_file", "apply_patch", "create_file"}
MIN_FILE_OVERLAP_LINES = 5


def _ranges_overlap(left: Event, right: Event) -> int | None:
    values = (left.line_start, left.line_end, right.line_start, right.line_end)
    if any(value is None for value in values):
        return None
    start = max(left.line_start, right.line_start)  # type: ignore[arg-type]
    end = min(left.line_end, right.line_end)  # type: ignore[arg-type]
    return max(0, end - start + 1)


def _match(previous: Event, current: Event) -> tuple[str, int | None] | None:
    if previous.tool in FILE_READ_TOOLS and current.tool in FILE_READ_TOOLS:
        if not previous.target or previous.target != current.target:
            return None
        overlap = _ranges_overlap(previous, current)
        # Windowed file viewers commonly repeat one or two boundary lines when
        # scrolling. That mechanical overlap is not an agent re-read.
        if overlap is not None and overlap < MIN_FILE_OVERLAP_LINES:
            return None
        return "file_view", overlap
    if previous.tool in SEARCH_TOOLS and current.tool in SEARCH_TOOLS:
        # A repeated query in a different file/directory is a new search, not
        # a reread of the same search result. Unknown scopes may still match.
        same_scope = (
            not previous.target
            or not current.target
            or previous.target == current.target
        )
        if previous.query and previous.query == current.query and same_scope:
            return "search", None
    if previous.tool in SHELL_TOOLS and current.tool in SHELL_TOOLS:
        if previous.command and previous.command == current.command:
            return "command", None
    return None


def _target_modified(target: str | None, events: Iterable[Event]) -> bool:
    if target is None:
        return False
    return any(event.tool in MUTATION_TOOLS and event.target == target
               and event.metadata.get("mutation_status") == "succeeded" for event in events)


def _content_unchanged(previous: Event, current: Event) -> bool | None:
    previous_lines = previous.metadata.get("line_contents")
    current_lines = current.metadata.get("line_contents")
    if (previous.tool in FILE_READ_TOOLS and current.tool in FILE_READ_TOOLS
            and isinstance(previous_lines, dict) and isinstance(current_lines, dict)):
        shared = set(previous_lines) & set(current_lines)
        if shared:
            return all(previous_lines[line] == current_lines[line] for line in shared)
    if previous.content is not None and current.content is not None:
        return previous.content == current.content
    return None


def _raw_action(event: Event) -> str:
    """Return the tool action recorded by the source trajectory, if available."""
    lines = str(event.metadata.get("raw_action", "")).strip().splitlines()
    return lines[0] if lines else ""


def _is_scroll_continuation(previous: Event, current: Event) -> bool:
    """Same-direction scrolls that advance are one navigation sequence.

    SWE-agent's viewer returns overlapping windows when scrolling. The overlap can
    be dozens of lines, so a fixed overlap threshold alone mistakes ordinary
    traversal for a reread. Restrict this rule to consecutive same-direction
    scroll actions on the same file that reveal further lines. A scroll stuck on
    the same window remains eligible, as do explicit reopen/goto actions.
    """
    previous_action = _raw_action(previous)
    current_action = _raw_action(current)
    advances = (
        current_action == "scroll_down"
        and previous.line_end is not None and current.line_end is not None
        and current.line_end > previous.line_end
    ) or (
        current_action == "scroll_up"
        and previous.line_start is not None and current.line_start is not None
        and current.line_start < previous.line_start
    )
    return (
        previous.tool in FILE_READ_TOOLS
        and current.tool in FILE_READ_TOOLS
        and previous.target == current.target
        and previous_action in {"scroll_down", "scroll_up"}
        and current_action == previous_action
        and previous.step + 1 == current.step
        and advances
    )


def detect_candidates(events: Iterable[Event]) -> list[Candidate]:
    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        grouped[event.trajectory_id].append(event)

    candidates: list[Candidate] = []
    for trajectory_id, trajectory in grouped.items():
        trajectory.sort(key=lambda event: event.step)
        for current_index, current in enumerate(trajectory):
            # Adjacent same-direction scrolls are ordinary file traversal. Do not
            # let the overlapping viewer windows create a candidate reread.
            if current_index > 0 and _is_scroll_continuation(
                trajectory[current_index - 1], current
            ):
                continue
            for previous_index in range(current_index - 1, -1, -1):
                previous = trajectory[previous_index]
                match = _match(previous, current)
                if match is None:
                    continue
                match_type, overlap = match
                intervening = tuple(trajectory[previous_index + 1 : current_index])
                following = tuple(trajectory[current_index + 1 :])
                content_unchanged = _content_unchanged(previous, current)
                candidates.append(
                    Candidate(
                        candidate_id=f"{trajectory_id}:{previous.step}:{current.step}",
                        trajectory_id=trajectory_id,
                        previous=previous,
                        current=current,
                        intervening=intervening,
                        following=following,
                        match_type=match_type,
                        content_unchanged=content_unchanged,
                        target_modified=(current.tool in FILE_READ_TOOLS and
                                         _target_modified(current.target, intervening)),
                        distance_steps=current.step - previous.step,
                        overlap_lines=overlap,
                        preceding=tuple(trajectory[:previous_index]),
                        evidence=build_evidence(trajectory, previous_index, current_index),
                    )
                )
                # Compare with the most recent matching retrieval only. This prevents
                # one current event from producing many nested duplicate candidates.
                break
    return candidates
