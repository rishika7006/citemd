from citemd.eval.mirage import parse_benchmark


def test_parse_benchmark_multiple_choice():
    raw = {
        "medqa": {
            "0": {
                "question": "Which drug is first-line for type 2 diabetes?",
                "options": {"A": "Metformin", "B": "Insulin", "C": "Aspirin", "D": "Warfarin"},
                "answer": "A",
            }
        }
    }
    parsed = parse_benchmark(raw)
    assert set(parsed) == {"medqa"}
    item = parsed["medqa"][0]
    assert item.dataset == "medqa"
    assert item.qid == "0"
    assert item.answer == "A"
    assert item.options["A"] == "Metformin"


def test_parse_benchmark_yes_no_and_answer_idx():
    raw = {
        "pubmedqa": {
            "q1": {"question": "Does X cause Y?", "answer": "yes"},
            "q2": {"query": "Does Z help?", "answer_idx": "no"},
        }
    }
    parsed = parse_benchmark(raw)
    items = {it.qid: it for it in parsed["pubmedqa"]}
    assert items["q1"].answer == "yes"
    assert items["q1"].question == "Does X cause Y?"
    # 'query' is accepted as an alias for 'question', 'answer_idx' for 'answer'.
    assert items["q2"].question == "Does Z help?"
    assert items["q2"].answer == "no"


def test_parse_benchmark_list_options_are_keyed():
    raw = {"bioasq": {"0": {"question": "q", "options": ["yes", "no"], "answer": "yes"}}}
    parsed = parse_benchmark(raw)
    assert parsed["bioasq"][0].options == {"A": "yes", "B": "no"}


def test_parse_benchmark_skips_non_dict_entries():
    raw = {"medqa": {"0": "not a dict", "1": {"question": "ok", "answer": "A"}}, "junk": 5}
    parsed = parse_benchmark(raw)
    assert [it.qid for it in parsed["medqa"]] == ["1"]
    assert "junk" not in parsed
