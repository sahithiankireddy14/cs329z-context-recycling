from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from .models import Event

FILE_VIEW_ACTIONS = {"open", "goto", "scroll_down", "scroll_up"}
SEARCH_ACTIONS = {"search_file", "search_dir", "find_file"}
CONTROL_ACTIONS = {"submit", "exit_cost", "exit_context", "exit_error"}

FILE_HEADER = re.compile(r"^\[File: (?P<path>.+?) \(\d+ lines total\)\]", re.MULTILINE)
NUMBERED_LINE = re.compile(r"^(?P<number>\d+):(?P<content>.*)$", re.MULTILINE)


def _parse_state(raw_state: Any) -> dict[str, Any]:
    if isinstance(raw_state, dict):
        return raw_state
    if not raw_state:
        return {}
    try:
        return json.loads(raw_state)
    except (json.JSONDecodeError, TypeError):
        return {"raw_state": raw_state}


def _relative_target(path: str | None, working_dir: str | None) -> str | None:
    if not path or path == "n/a":
        return None
    if working_dir and path.startswith(working_dir.rstrip("/") + "/"):
        return path[len(working_dir.rstrip("/")) + 1 :]
    return path.lstrip("/")


def _observation_file(observation: str) -> str | None:
    match = FILE_HEADER.search(observation)
    return match.group("path") if match else None


def _numbered_lines(observation: str) -> dict[str, str]:
    return {
        match.group("number"): match.group("content")
        for match in NUMBERED_LINE.finditer(observation)
    }


def _action_parts(action: str) -> list[str]:
    first_line = action.splitlines()[0] if action else ""
    try:
        return shlex.split(first_line)
    except ValueError:
        return first_line.split()


def _outcome_sets(results_path: str | Path | None) -> tuple[set[str], set[str]]:
    if results_path is None:
        return set(), set()
    with Path(results_path).open(encoding="utf-8") as stream:
        results = json.load(stream)
    return set(results.get("resolved", [])), set(results.get("generated", []))


def convert_trajectory(
    payload: dict[str, Any],
    trajectory_id: str,
    *,
    resolved: bool | None = None,
) -> list[Event]:
    events: list[Event] = []
    for index, step in enumerate(payload.get("trajectory", []), start=1):
        action = str(step.get("action", "")).strip()
        observation = str(step.get("observation", ""))
        parts = _action_parts(action)
        if not parts:
            continue
        verb = parts[0]
        state = _parse_state(step.get("state"))
        working_dir = state.get("working_dir")
        observation_path = _observation_file(observation)
        state_path = state.get("open_file")
        target = _relative_target(observation_path or state_path, working_dir)
        line_contents = _numbered_lines(observation)
        line_numbers = [int(number) for number in line_contents]

        metadata: dict[str, Any] = {
            "source": "swe-bench/experiments:verified/20240402_sweagent_gpt4",
            "raw_action": action,
            "thought": step.get("thought"),
            "response": step.get("response"),
            "state": state,
            "resolved": resolved,
        }
        if line_contents:
            metadata["line_contents"] = line_contents

        event_kwargs: dict[str, Any] = {
            "trajectory_id": trajectory_id,
            "step": index,
            "tool": "control",
            "target": target,
            "content": observation,
            "metadata": metadata,
        }

        if verb in FILE_VIEW_ACTIONS:
            # An attempted open/goto is not necessarily a retrieval. SWE-agent
            # records failures such as "File not found" and out-of-range goto
            # errors in the same action channel. Only observations with a real
            # file header count as file-view events; retain failures as controls
            # so they remain visible in the surrounding trace.
            if observation_path:
                event_kwargs.update(
                    tool="view_file",
                    line_start=min(line_numbers) if line_numbers else None,
                    line_end=max(line_numbers) if line_numbers else None,
                )
        elif verb in SEARCH_ACTIONS:
            query = parts[1] if len(parts) > 1 else None
            explicit_scope = parts[2] if len(parts) > 2 else None
            if explicit_scope:
                target = _relative_target(explicit_scope, working_dir)
            event_kwargs.update(tool="search", target=target, query=query)
            metadata["search_action"] = verb
        elif verb == "edit":
            event_kwargs["tool"] = "edit_file"
            # Failed edit observations can contain both proposed and original
            # file windows. Neither is a newly applied edit.
            failed = any(marker in observation for marker in (
                "Your changes have NOT been applied", "Your proposed edit has introduced",
            )) or observation.startswith(("Error:", "Usage:")) or (
                observation.startswith(("/bin/bash:", "bash:")) and "syntax error" in observation
            )
            status = "failed" if failed else (
                "succeeded" if "\nFile updated." in observation else "unknown"
            )
            metadata["mutation_status"] = status
            if status != "succeeded":
                metadata.pop("line_contents", None)
            if len(parts) > 1:
                metadata["edit_range"] = parts[1]
        elif verb == "create":
            create_path = parts[1] if len(parts) > 1 else observation_path
            event_kwargs.update(
                tool="create_file",
                target=_relative_target(observation_path or create_path, working_dir),
            )
            metadata["mutation_status"] = (
                "failed" if observation.startswith(("Error:", "Usage:")) or (
                    not observation_path and "No such file or directory" in observation
                ) else
                "succeeded" if observation_path else "unknown"
            )
            if metadata["mutation_status"] != "succeeded":
                metadata.pop("line_contents", None)
        elif verb in CONTROL_ACTIONS or action.lstrip().startswith("#"):
            event_kwargs["tool"] = "control"
        else:
            event_kwargs.update(tool="run_command", command=action)

        events.append(Event(**event_kwargs))
    return events


def convert_directory(
    input_dir: str | Path,
    *,
    results_path: str | Path | None = None,
) -> list[Event]:
    resolved_ids, generated_ids = _outcome_sets(results_path)
    events: list[Event] = []
    paths = sorted(Path(input_dir).glob("*.traj"))
    if not paths:
        raise ValueError(f"No .traj files found in {input_dir}")
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        trajectory_id = path.stem
        resolved = trajectory_id in resolved_ids if trajectory_id in generated_ids else None
        events.extend(convert_trajectory(payload, trajectory_id, resolved=resolved))
    return events
