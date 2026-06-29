"""Grounded citation prompting and answer parsing.

The contract with the model is deliberately strict so the output is measurable:

  - The model sees the question and a numbered list of retrieved passages.
  - It must answer using only those passages and cite the passage numbers it relied on.
  - If the passages do not contain enough evidence, it must abstain rather than guess.
  - It replies with a single JSON object, which we parse robustly.

For MIRAGE the task is multiple choice, so the JSON ``answer`` is an option key (e.g. "B")
or the abstain sentinel. The same builder serves open-ended questions by passing no options,
in which case ``answer`` is free text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from citemd.models import Citation, RetrievedChunk

ABSTAIN_TOKENS = {"insufficient", "abstain", "unknown", "idk", "i don't know", "cannot answer"}

_SYSTEM_BASE = (
    "You are a clinical-evidence assistant. Answer the question using ONLY the numbered "
    "passages provided. Cite every passage you rely on by its number. This is a research "
    "and evaluation tool, not for clinical use.\n\n"
    "Reply with a single JSON object and nothing else, in this exact shape:\n"
    '{"answer": "<see below>", "citations": [<passage numbers>], '
    '"confidence": <number 0..1>, "rationale": "<one short sentence>"}\n'
    "- confidence is how well the passages support your answer (1 = fully supported)."
)

_ABSTAIN_CLAUSE = (
    "\n- If the passages do not contain enough evidence to answer, you MUST abstain instead "
    'of guessing: set "answer" to "insufficient" and "citations" to [].'
)

_FORCE_CLAUSE = (
    "\n- You must commit to the single most likely answer even if the evidence is weak; "
    "report your genuine (possibly low) confidence rather than abstaining."
)


def system_prompt(*, allow_abstain: bool = True) -> str:
    """The grounding system prompt, with or without the abstain option."""
    return _SYSTEM_BASE + (_ABSTAIN_CLAUSE if allow_abstain else _FORCE_CLAUSE)


# Backwards-friendly default used by callers that always allow abstention.
SYSTEM_PROMPT = system_prompt(allow_abstain=True)


def format_contexts(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered passage block for the prompt."""
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        loc = " / ".join(part for part in [c.title, c.section] if part)
        head = f"[{i}]" + (f" ({loc})" if loc else "")
        body = " ".join(c.text.split())
        lines.append(f"{head} {body}")
    return "\n\n".join(lines)


def build_messages(
    question: str,
    contexts: Sequence[RetrievedChunk],
    *,
    options: dict[str, str] | None = None,
    allow_abstain: bool = True,
) -> list[dict[str, str]]:
    """Build the [system, user] chat messages for a grounded, citing answer."""
    parts = [f"Question: {question.strip()}"]
    if options:
        opt_lines = "\n".join(f"{key}. {val}" for key, val in options.items())
        parts.append("Options:\n" + opt_lines)
        keys = ", ".join(options.keys())
        tail = ', or to "insufficient" to abstain' if allow_abstain else ""
        parts.append(f'Set "answer" to exactly one option key ({keys}){tail}.')
    parts.append("Passages:\n" + (format_contexts(contexts) or "(no passages retrieved)"))
    return [
        {"role": "system", "content": system_prompt(allow_abstain=allow_abstain)},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _extract_json_object(text: str) -> dict | None:
    """Return the first balanced top-level JSON object in ``text``, or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # malformed; try the next "{"
        start = text.find("{", start + 1)
    return None


class ParsedAnswer:
    """Lightweight container for a parsed model reply."""

    def __init__(
        self,
        *,
        option: str | None,
        text: str,
        abstained: bool,
        confidence: float,
        citation_markers: list[int],
        rationale: str,
    ):
        self.option = option
        self.text = text
        self.abstained = abstained
        self.confidence = confidence
        self.citation_markers = citation_markers
        self.rationale = rationale


def _coerce_confidence(value) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, conf))


def parse_answer(
    raw: str,
    *,
    option_keys: Sequence[str] | None = None,
) -> ParsedAnswer:
    """Parse a model reply into a structured answer.

    Tolerant of models that wrap JSON in prose or code fences. Falls back to treating the
    reply as an abstention when no answer can be recovered, which is the safe default for a
    trustworthiness-focused system.
    """
    obj = _extract_json_object(raw) or {}
    answer = obj.get("answer", "")
    answer_str = str(answer).strip()
    rationale = str(obj.get("rationale", "")).strip()
    confidence = _coerce_confidence(obj.get("confidence", 0.0))

    markers: list[int] = []
    for m in obj.get("citations", []) or []:
        try:
            markers.append(int(m))
        except (TypeError, ValueError):
            continue

    abstained = answer_str.lower() in ABSTAIN_TOKENS or answer_str == ""

    option: str | None = None
    if not abstained and option_keys:
        # Accept "B", "b", or "B. ..." and match case-insensitively to a known option key.
        head = answer_str.split(".")[0].split(")")[0].strip()
        for key in option_keys:
            if head.lower() == key.lower():
                option = key
                break
        if option is None:
            # Model returned something that is not a valid option key: treat as abstention
            # rather than scoring a malformed guess as a confident answer.
            abstained = True

    text = "" if option_keys else answer_str
    if abstained:
        confidence = min(confidence, 0.0) if option_keys else confidence
    return ParsedAnswer(
        option=option,
        text=text,
        abstained=abstained,
        confidence=confidence,
        citation_markers=markers,
        rationale=rationale,
    )


def resolve_citations(
    markers: Sequence[int],
    contexts: Sequence[RetrievedChunk],
) -> list[Citation]:
    """Map 1-based passage markers to Citation records, dropping out-of-range markers."""
    out: list[Citation] = []
    seen: set[int] = set()
    for m in markers:
        if m in seen or m < 1 or m > len(contexts):
            continue
        seen.add(m)
        c = contexts[m - 1]
        out.append(
            Citation(
                marker=m,
                chunk_id=c.chunk_id,
                source_id=c.source_id,
                title=c.title,
                section=c.section,
                page=c.page,
                uri=c.uri,
            )
        )
    return out


# Pre-compiled for any callers that want to find inline [n] markers in free text.
INLINE_MARKER_RE = re.compile(r"\[(\d+)\]")
