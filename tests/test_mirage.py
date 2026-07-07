from citemd.eval.mirage import QuestionItem, load_dataset, parse_benchmark


def _write_class_ordered(tmp_path):
    """Write a class-ordered dataset: 30 'A' then 20 'B' (mirrors MIRAGE's ordering)."""
    d = tmp_path / "mirage"
    d.mkdir(parents=True)
    def q(i, ans):
        return QuestionItem(dataset="t", qid=str(i), question="q", answer=ans)

    items = [q(i, "A") for i in range(30)] + [q(30 + i, "B") for i in range(20)]
    lines = "\n".join(it.model_dump_json() for it in items)
    (d / "t.jsonl").write_text(lines + "\n")


def test_limit_is_class_ordered_but_sample_is_representative(tmp_path):
    _write_class_ordered(tmp_path)
    # limit takes the first N in file order -> degenerate single class
    first10 = load_dataset("t", tmp_path, limit=10)
    assert {it.answer for it in first10} == {"A"}
    # sample draws across the whole file -> both classes appear
    sampled = load_dataset("t", tmp_path, sample=20, seed=0)
    assert len(sampled) == 20
    assert {it.answer for it in sampled} == {"A", "B"}


def test_sample_is_deterministic(tmp_path):
    _write_class_ordered(tmp_path)
    a = [it.qid for it in load_dataset("t", tmp_path, sample=15, seed=7)]
    b = [it.qid for it in load_dataset("t", tmp_path, sample=15, seed=7)]
    c = [it.qid for it in load_dataset("t", tmp_path, sample=15, seed=8)]
    assert a == b  # same seed -> same subset
    assert a != c  # different seed -> different subset


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


def test_parse_benchmark_captures_pmids():
    raw = {
        "pubmedqa": {
            "q1": {"question": "Q?", "answer": "yes", "PMID": [12377809]},
            "q2": {"question": "Q?", "answer": "no", "PMID": 26163474},  # bare, not a list
            "q3": {"question": "Q?", "answer": "maybe"},  # no PMID
        }
    }
    items = {it.qid: it for it in parse_benchmark(raw)["pubmedqa"]}
    assert items["q1"].pmids == ["12377809"]
    assert items["q2"].pmids == ["26163474"]
    assert items["q3"].pmids == []


def test_parse_benchmark_list_options_are_keyed():
    raw = {"bioasq": {"0": {"question": "q", "options": ["yes", "no"], "answer": "yes"}}}
    parsed = parse_benchmark(raw)
    assert parsed["bioasq"][0].options == {"A": "yes", "B": "no"}


def test_parse_benchmark_skips_non_dict_entries():
    raw = {"medqa": {"0": "not a dict", "1": {"question": "ok", "answer": "A"}}, "junk": 5}
    parsed = parse_benchmark(raw)
    assert [it.qid for it in parsed["medqa"]] == ["1"]
    assert "junk" not in parsed
