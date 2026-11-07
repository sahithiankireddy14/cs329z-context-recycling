from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Label(StrEnum):
    LEGITIMATE = "legitimate"
    VERIFICATION = "verification"
    REDUNDANT = "redundant"


@dataclass(frozen=True)
class Event:
    trajectory_id: str
    step: int
    tool: str
    target: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    content: str | None = None
    query: str | None = None
    command: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        known = {
            "trajectory_id",
            "step",
            "tool",
            "target",
            "line_start",
            "line_end",
            "content",
            "query",
            "command",
            "metadata",
        }
        metadata = dict(data.get("metadata", {}))
        metadata.update({key: value for key, value in data.items() if key not in known})
        return cls(
            trajectory_id=str(data["trajectory_id"]),
            step=int(data["step"]),
            tool=str(data["tool"]),
            target=data.get("target"),
            line_start=data.get("line_start"),
            line_end=data.get("line_end"),
            content=data.get("content"),
            query=data.get("query"),
            command=data.get("command"),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    trajectory_id: str
    previous: Event
    current: Event
    intervening: tuple[Event, ...]
    following: tuple[Event, ...]
    match_type: str
    content_unchanged: bool | None
    target_modified: bool
    distance_steps: int
    overlap_lines: int | None = None
    preceding: tuple[Event, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candidate:
        return cls(
            candidate_id=data["candidate_id"],
            trajectory_id=data["trajectory_id"],
            previous=Event.from_dict(data["previous"]),
            current=Event.from_dict(data["current"]),
            intervening=tuple(Event.from_dict(item) for item in data["intervening"]),
            following=tuple(Event.from_dict(item) for item in data.get("following", [])),
            match_type=data["match_type"],
            content_unchanged=data.get("content_unchanged"),
            target_modified=bool(data["target_modified"]),
            distance_steps=int(data["distance_steps"]),
            overlap_lines=data.get("overlap_lines"),
            preceding=tuple(Event.from_dict(item) for item in data.get("preceding", [])),
            evidence=dict(data.get("evidence", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["intervening"] = [event.to_dict() for event in self.intervening]
        data["following"] = [event.to_dict() for event in self.following]
        data["preceding"] = [event.to_dict() for event in self.preceding]
        return data


@dataclass(frozen=True)
class ClassificationResult:
    candidate_id: str
    label: Label
    confidence: float
    probabilities: dict[str, float]
    classifier: str
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    rationale: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationResult:
        return cls(
            candidate_id=data["candidate_id"],
            label=Label(data["label"]),
            confidence=float(data["confidence"]),
            probabilities={key: float(value) for key, value in data["probabilities"].items()},
            classifier=data["classifier"],
            model=data.get("model"),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            latency_ms=data.get("latency_ms"),
            cost_usd=data.get("cost_usd"),
            rationale=data.get("rationale"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["label"] = self.label.value
        return data
