from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from .models import Candidate, ClassificationResult, Label


LABEL_CRITERIA = {
    "legitimate": (
        "The retrieval supplies useful new information or changed content, or the trace "
        "supports a new purpose that makes the earlier observation insufficient."
    ),
    "verification": (
        "The information is unchanged, but the agent immediately uses this retrieval "
        "to prepare an edit of the same target."
    ),
    "redundant": (
        "The retrieval repeats already available unchanged information and the trace "
        "supports no useful new purpose or need to retrieve it again."
    ),
}
CLASSIFICATION_GUIDANCE = (
    "Classify the usefulness of the current retrieval using the supplied trace evidence. "
    "For file views, content_unchanged refers only to shared numbered lines; inspect "
    "trajectory_evidence.line_delta and history_line_delta for new and changed lines. "
    "Partial overlap or ordinary navigation revealing useful new lines is not sufficient "
    "evidence of redundancy. Changed retrieved lines generally make the earlier read stale. "
    "Distinguish successful edits, failed edits, and unknown outcomes. An attempted edit "
    "alone does not establish a content change, and an edit outside the viewed region "
    "does not necessarily justify repeating the view. Consider thought/response text "
    "alongside actions and observations; identical wording is evidence of stated reasoning "
    "equality, not proof of unnecessary retrieval. The absence of an immediate edit alone "
    "does not prove redundancy. Use preceding/intervening events to assess context and "
    "following events as retrospective evidence of use; a distant later edit is not an "
    "immediate verification action. Missing or truncated observations and untracked shell "
    "effects are uncertainty, not proof that the environment stayed unchanged. "
)


class Classifier(Protocol):
    def classify(self, candidate: Candidate) -> ClassificationResult: ...


class JevClassifier:
    endpoint = "https://api.typesafe.ai/v1/systemone"
    classifier_name = "jev-direct"

    def __init__(self, api_key: str | None = None, model: str = "jev-latest", timeout: float = 30.0):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("Set TYPESAFE_API_KEY before using the Jev classifier.")
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _compact_event(event: Any, *, content_limit: int) -> dict[str, Any]:
        content = event.content
        if content is not None and len(content) > content_limit:
            content = content[:content_limit] + "\n...[truncated]"
        return {
            "step": event.step,
            "tool": event.tool,
            "target": event.target,
            "line_start": event.line_start,
            "line_end": event.line_end,
            "query": event.query,
            "command": event.command,
            "action": event.metadata.get("raw_action"),
            "content": content,
            "thought": event.metadata.get("thought"),
            "response": event.metadata.get("response"),
            "pre_action_state": event.metadata.get("state"),
            "mutation_status": event.metadata.get("mutation_status"),
        }

    @staticmethod
    def _state(candidate: Candidate) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "match_type": candidate.match_type,
            "preceding_events": [
                JevClassifier._compact_event(event, content_limit=2_000)
                for event in candidate.preceding
            ],
            "previous_retrieval": JevClassifier._compact_event(
                candidate.previous, content_limit=16_000
            ),
            "intervening_events": [
                JevClassifier._compact_event(event, content_limit=2_000)
                for event in candidate.intervening
            ],
            "current_retrieval": JevClassifier._compact_event(
                candidate.current, content_limit=16_000
            ),
            "following_events": [
                JevClassifier._compact_event(event, content_limit=2_000)
                for event in candidate.following
            ],
            "derived_features": {
                "content_unchanged": candidate.content_unchanged,
                "target_modified": candidate.target_modified,
                "distance_steps": candidate.distance_steps,
                "overlap_lines": candidate.overlap_lines,
            },
            "trajectory_evidence": candidate.evidence,
        }

    def build_payload(self, candidate: Candidate) -> dict[str, Any]:
        return {
            "model": self.model,
            "state": self._state(candidate),
            "questions": {
                "reread_label": {
                    "type": "choice",
                    "instructions": CLASSIFICATION_GUIDANCE,
                    "criteria": LABEL_CRITERIA,
                }
            },
        }

    def classify(self, candidate: Candidate) -> ClassificationResult:
        body = json.dumps(self.build_payload(candidate)).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Jev returned HTTP {error.code}: {detail}") from error
        latency_ms = (time.perf_counter() - started) * 1000

        answer = payload["answers"]["reread_label"]
        if answer.get("type") != "choice":
            raise RuntimeError(f"Unexpected Jev answer type: {answer.get('type')!r}")
        label = Label(answer["choice"])
        usage = payload.get("usage", {})
        return ClassificationResult(
            candidate_id=candidate.candidate_id,
            label=label,
            confidence=float(answer["confidence"]),
            probabilities={key: float(value) for key, value in answer["probabilities"].items()},
            classifier=self.classifier_name,
            model=payload.get("model", self.model),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            latency_ms=latency_ms,
            cost_usd=usage.get("cost"),
        )


class OpenRouterJevClassifier(JevClassifier):
    endpoint = "https://openrouter.ai/api/alpha/decisions"
    classifier_name = "jev-openrouter"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "typesafe/jev-1.14",
        timeout: float = 30.0,
    ):
        self.api_key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("Set OPENROUTER_API_KEY before using Jev through OpenRouter.")
        self.model = model
        self.timeout = timeout


class OpenRouterLLMClassifier(JevClassifier):
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    classifier_name = "llm-openrouter"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mistralai/mistral-small-3.2-24b-instruct",
        timeout: float = 60.0,
    ):
        self.api_key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("Set OPENROUTER_API_KEY before using an OpenRouter LLM.")
        self.model = model
        self.timeout = timeout

    def build_payload(self, candidate: Candidate) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ["legitimate", "verification", "redundant"],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "probabilities": {
                    "type": "object",
                    "properties": {
                        "legitimate": {"type": "number", "minimum": 0, "maximum": 1},
                        "verification": {"type": "number", "minimum": 0, "maximum": 1},
                        "redundant": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["legitimate", "verification", "redundant"],
                    "additionalProperties": False,
                },
                "rationale": {"type": "string"},
            },
            "required": ["label", "confidence", "probabilities", "rationale"],
            "additionalProperties": False,
        }
        return {
            "model": self.model,
            "temperature": 0,
            "max_completion_tokens": 600,
            "provider": {"require_parameters": True},
            "messages": [
                {
                    "role": "system",
                    "content": CLASSIFICATION_GUIDANCE + " ".join(
                        f"{label}: {criterion}" for label, criterion in LABEL_CRITERIA.items()
                    ) + " Return the required JSON only.",
                },
                {"role": "user", "content": json.dumps(self._state(candidate), sort_keys=True)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "reread_classification",
                    "strict": True,
                    "schema": schema,
                },
            },
        }

    def classify(self, candidate: Candidate) -> ClassificationResult:
        body = json.dumps(self.build_payload(candidate)).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter returned HTTP {error.code}: {detail}") from error
        latency_ms = (time.perf_counter() - started) * 1000

        content = payload["choices"][0]["message"]["content"]
        try:
            answer = json.loads(content)
        except (json.JSONDecodeError, TypeError) as error:
            raise RuntimeError("OpenRouter LLM returned malformed structured JSON.") from error
        label = Label(answer["label"])
        probabilities = {
            key: float(value) for key, value in answer["probabilities"].items()
        }
        total_probability = sum(probabilities.values())
        if total_probability <= 0:
            raise RuntimeError("OpenRouter LLM returned zero total probability.")
        probabilities = {
            key: value / total_probability for key, value in probabilities.items()
        }
        usage = payload.get("usage", {})
        return ClassificationResult(
            candidate_id=candidate.candidate_id,
            label=label,
            confidence=float(answer["confidence"]),
            probabilities=probabilities,
            classifier=self.classifier_name,
            model=payload.get("model", self.model),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            cost_usd=usage.get("cost"),
            rationale=answer.get("rationale"),
        )
