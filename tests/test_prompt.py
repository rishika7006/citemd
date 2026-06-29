from citemd.generate.prompt import (
    build_messages,
    format_contexts,
    parse_answer,
    resolve_citations,
    system_prompt,
)
from citemd.models import RetrievedChunk


def _ctx(n):
    return [
        RetrievedChunk(
            chunk_id=f"d#{i}", source_id=f"pmid:{i}", text=f"evidence {i}",
            title=f"Title {i}", section="Results",
        )
        for i in range(1, n + 1)
    ]


def test_format_contexts_numbers_from_one():
    text = format_contexts(_ctx(2))
    assert text.startswith("[1]")
    assert "[2]" in text
    assert "Title 1 / Results" in text


def test_build_messages_includes_options_and_force_clause():
    msgs = build_messages(
        "Q?", _ctx(1), options={"A": "yes", "B": "no"}, allow_abstain=False
    )
    assert msgs[0]["role"] == "system"
    assert "commit to the single most likely" in msgs[0]["content"]
    assert "A. yes" in msgs[1]["content"] and "B. no" in msgs[1]["content"]


def test_system_prompt_abstain_toggle():
    assert "MUST abstain" in system_prompt(allow_abstain=True)
    assert "MUST abstain" not in system_prompt(allow_abstain=False)


def test_parse_answer_valid_option():
    p = parse_answer(
        '{"answer": "B", "citations": [1, 2], "confidence": 0.8, "rationale": "x"}',
        option_keys=["A", "B", "C"],
    )
    assert p.option == "B"
    assert not p.abstained
    assert p.citation_markers == [1, 2]
    assert p.confidence == 0.8


def test_parse_answer_abstention_token():
    p = parse_answer('{"answer": "insufficient", "citations": []}', option_keys=["A", "B"])
    assert p.abstained
    assert p.option is None


def test_parse_answer_invalid_option_is_treated_as_abstention():
    p = parse_answer('{"answer": "Z"}', option_keys=["A", "B"])
    assert p.abstained
    assert p.option is None


def test_parse_answer_json_embedded_in_prose_and_fence():
    raw = 'Sure!\n```json\n{"answer": "a", "confidence": 1.5, "citations": [2]}\n```\nDone.'
    p = parse_answer(raw, option_keys=["A", "B"])
    assert p.option == "A"  # case-insensitive match
    assert p.confidence == 1.0  # clamped to [0, 1]


def test_parse_answer_unparseable_defaults_to_abstain():
    p = parse_answer("no json here", option_keys=["A", "B"])
    assert p.abstained


def test_resolve_citations_filters_and_dedups():
    ctx = _ctx(3)
    cits = resolve_citations([2, 2, 5, 0, 1], ctx)
    markers = [c.marker for c in cits]
    assert markers == [2, 1]  # dedup, drop out-of-range 5 and 0
    assert cits[0].source_id == "pmid:2"
