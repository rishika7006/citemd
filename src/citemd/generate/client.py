"""LLM client for generation.

CiteMD talks to an OpenAI-compatible chat-completions endpoint. By default that endpoint is
KVGate (the sibling inference gateway), which adds response caching, rate limiting, and
metrics in front of a hosted model; it can also point straight at a provider API as a
fallback by overriding ``llm_base_url`` / ``llm_api_key`` / ``llm_model``.

The client is defined behind a small ``LLMClient`` protocol so the pipeline depends only on
``complete(messages) -> text``. The real implementation (``OpenAICompatibleClient``) imports
the ``openai`` SDK lazily; tests and offline runs use ``FakeLLMClient`` and never touch the
network or spend money.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from citemd.config import get_settings

Message = dict[str, str]


@runtime_checkable
class LLMClient(Protocol):
    """Minimal generation interface the pipeline depends on."""

    model: str

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> str:
        """Return the assistant text for a chat-completions request."""
        ...


class OpenAICompatibleClient:
    """Calls an OpenAI-compatible ``/chat/completions`` endpoint (KVGate by default)."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        settings = get_settings()
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    def _require_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            raise ModuleNotFoundError(
                "Generation needs the 'openai' package. Install with: pip install 'citemd[llm]'"
            ) from exc
        # An empty api_key is allowed: some local/self-hosted KVGate deployments do not
        # require one. The SDK still needs a non-None value, so pass a placeholder.
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "not-needed",
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        return self._client

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> str:
        client = self._require_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class FakeLLMClient:
    """Deterministic in-memory client for tests and dry runs.

    Pass a ``responder`` that maps the message list to the assistant text. Records every call
    so tests can assert on the prompt that was built.
    """

    def __init__(
        self,
        responder: Callable[[Sequence[Message]], str] | str = "",
        *,
        model: str = "fake-model",
    ):
        self._responder = responder
        self.model = model
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> str:
        self.calls.append(list(messages))
        if callable(self._responder):
            return self._responder(messages)
        return self._responder
