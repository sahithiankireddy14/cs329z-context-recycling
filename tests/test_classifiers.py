import pytest

from context_recycling.classifiers import (
    JevClassifier,
    OpenRouterJevClassifier,
    OpenRouterLLMClassifier,
)
from context_recycling.models import Candidate, Event


def make_candidate(
    *, modified: bool = False, unchanged: bool | None = True, followed_by_edit: bool = False
) -> Candidate:
    previous = Event("t", 1, "view_file", target="a.py", content="old")
    current = Event("t", 3, "view_file", target="a.py", content="old")
    following = (Event("t", 4, "edit_file", target="a.py"),) if followed_by_edit else tuple()
    return Candidate(
        "t:1:3", "t", previous, current, tuple(), following, "file_view", unchanged, modified, 2
    )


def test_jev_payload_uses_three_choice_rubric() -> None:
    classifier = JevClassifier(api_key="test-key")
    payload = classifier.build_payload(make_candidate())
    question = payload["questions"]["reread_label"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"legitimate", "verification", "redundant"}


def test_jev_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        JevClassifier()


def test_openrouter_jev_uses_decisions_endpoint() -> None:
    classifier = OpenRouterJevClassifier(api_key="test-key")
    assert classifier.endpoint == "https://openrouter.ai/api/alpha/decisions"
    assert classifier.model == "typesafe/jev-1.14"


def test_openrouter_jev_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterJevClassifier()


def test_openrouter_llm_payload_has_strict_three_label_schema() -> None:
    classifier = OpenRouterLLMClassifier(api_key="test-key")
    payload = classifier.build_payload(make_candidate())
    label_schema = payload["response_format"]["json_schema"]["schema"]["properties"]["label"]
    assert label_schema["enum"] == ["legitimate", "verification", "redundant"]
    assert payload["temperature"] == 0



def test_classifier_payload_preserves_full_future_and_preceding_context():
    from dataclasses import replace
    base = make_candidate()
    candidate = replace(base, following=(Event('t', 4, 'control'),
        Event('t', 5, 'edit_file', target='a.py')),
        preceding=(Event('t', 0, 'control', metadata={'thought': 'setup'}),),
        evidence={'version': 4})
    state = JevClassifier._state(candidate)
    assert state['preceding_events'][0]['thought'] == 'setup'
    assert len(state['following_events']) == 2
    assert state['trajectory_evidence']['version'] == 4


def test_both_classifiers_use_updated_shared_rubric():
    from context_recycling.classifiers import CLASSIFICATION_GUIDANCE, LABEL_CRITERIA
    candidate = make_candidate()
    jev = JevClassifier(api_key='test').build_payload(candidate)['questions']['reread_label']
    llm = OpenRouterLLMClassifier(api_key='test').build_payload(candidate)['messages'][0]['content']
    assert jev['instructions'] == CLASSIFICATION_GUIDANCE
    assert jev['criteria'] == LABEL_CRITERIA
    assert CLASSIFICATION_GUIDANCE in llm
    for criterion in LABEL_CRITERIA.values():
        assert criterion in llm


def test_default_llm_matches_reported_mistral_model():
    classifier = OpenRouterLLMClassifier(api_key='test-key')
    assert classifier.model == 'mistralai/mistral-small-3.2-24b-instruct'


def test_cli_requires_one_of_the_reported_classifiers():
    from context_recycling.cli import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['classify', '--candidates', 'candidates.jsonl', '--output', 'predictions.jsonl'])
    for name in ('openrouter-jev', 'openrouter-llm'):
        args = parser.parse_args(['classify', '--candidates', 'candidates.jsonl', '--output', 'predictions.jsonl', '--classifier', name])
        assert args.classifier == name
