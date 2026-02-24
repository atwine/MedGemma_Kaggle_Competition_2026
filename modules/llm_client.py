"""Local LLM client.

This project uses Ollama for local inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Optional, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class OllamaConfig:
    # Rationale: default to the locally pulled model tag to avoid mismatches.
    model: str = "aadide/medgemma-1.5-4b-it-Q4_K_S"
    # Rationale: allow setting the context window per request without changing server env.
    # Matches Ollama `num_ctx` parameter. [src]
    # - https://raw.githubusercontent.com/ollama/ollama/main/docs/modelfile.mdx#valid-parameters-and-values
    num_ctx: Optional[int] = None


class OllamaClient:
    def __init__(self, config: OllamaConfig | None = None) -> None:
        cfg = config or OllamaConfig()

        # Rationale: allow overriding via environment without changing code.
        model_override = os.getenv("OLLAMA_MODEL")
        self._model = model_override if model_override else cfg.model

        # Rationale: ollama-python allows a custom host via Client(host=...). [src]
        # - https://github.com/ollama/ollama-python#custom-client
        # Note: OLLAMA_HOST is sometimes set to a bind address like 0.0.0.0:11434;
        # clients cannot connect to 0.0.0.0, so we translate it to localhost. [src]
        # - https://github.com/ollama/ollama-python/issues/407
        raw_host = (os.getenv("OLLAMA_HOST") or "").strip()
        if raw_host:
            host = raw_host
            if "://" not in host:
                host = f"http://{host}"
            host = host.replace("http://0.0.0.0", "http://localhost").replace(
                "https://0.0.0.0", "http://localhost"
            )
            self._host = host
        else:
            self._host = "http://localhost:11434"

        # Rationale: surface the last failure reason to the UI for debugging.
        self._last_error: Optional[str] = None
        # Rationale: store num_ctx to pass via options when set.
        self._num_ctx: Optional[int] = cfg.num_ctx

    @property
    def model(self) -> str:
        return self._model

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def chat(
        self,
        messages: List[ChatMessage],
        *,
        format: Optional[Any] = None,
        options_override: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Call the local Ollama chat API.

        Uses the Ollama Python library interface documented on Ollama model pages.
        Example usage shown for `from ollama import chat ...`. [src]
        - https://ollama.com/MedAIBase/MedGemma1.5
        """

        try:
            from ollama import Client
        except Exception:
            # Rationale: if the Python client isn't installed, the app should still run
            # with deterministic fallback explanations.
            self._last_error = "Python package 'ollama' is not available in this environment"
            return None

        try:
            self._last_error = None
            # Rationale: use an explicit client so we can control the host used for
            # connectivity debugging. Allow up to 5 minutes for long-running
            # local model calls before treating them as timeouts.
            client = Client(host=self._host, timeout=600.0)
            # Rationale: keep outputs more deterministic for clinical UX.
            # Ollama supports passing generation parameters via `options` (e.g., temperature). [src]
            # - https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md
            chat_options: Dict[str, Any] = {
                "temperature": 0.2,
                "num_predict": 128,
            }
            if self._num_ctx is not None:
                chat_options["num_ctx"] = int(self._num_ctx)

            if options_override:
                # Rationale: allow per-call overrides (e.g., higher num_predict for long JSON).
                chat_options.update(options_override)

            # Rationale: Ollama supports structured outputs via the `format` parameter
            # (either `json` or a JSON schema). [src]
            # - https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md
            try:
                if format is not None:
                    response = client.chat(
                        model=self._model,
                        messages=messages,
                        options=chat_options,
                        format=format,
                    )
                else:
                    response = client.chat(
                        model=self._model,
                        messages=messages,
                        options=chat_options,
                    )
            except TypeError:
                # Rationale: older ollama-python client versions may not accept `format`.
                response = client.chat(
                    model=self._model,
                    messages=messages,
                    options=chat_options,
                )
            # Rationale: ollama-python supports both dict-style and object-style
            # access for responses. [src]
            # - https://raw.githubusercontent.com/ollama/ollama-python/main/README.md
            if response is None:
                self._last_error = "Ollama returned no response"
                return None

            if hasattr(response, "message") and getattr(response, "message") is not None:
                content = getattr(response, "message").content
                if not content:
                    self._last_error = "Ollama returned an empty message content"
                    return None
                # Rationale: with structured outputs (`format`), `content` may already be
                # a Python list/dict. Downstream callers (e.g. LLM screening) expect a
                # JSON string, so we serialize non‑string content here instead of using
                # str(content), which would not be valid JSON. [src]
                if not isinstance(content, str):
                    try:
                        return json.dumps(content, ensure_ascii=False)
                    except TypeError:
                        return str(content)
                return content

            if isinstance(response, dict):
                msg = response.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if not content:
                        self._last_error = "Ollama returned an empty message content"
                        return None
                    # See rationale above: ensure non‑string structured content is
                    # serialized as proper JSON so parsers can consume it safely.
                    if not isinstance(content, str):
                        try:
                            return json.dumps(content, ensure_ascii=False)
                        except TypeError:
                            return str(content)
                    return content

            self._last_error = f"Unrecognized ollama response type: {type(response)}"
            return None
        except Exception as e:
            # Rationale: local model may not be pulled/served yet; we degrade gracefully.
            self._last_error = f"Ollama call failed (host={self._host}): {type(e).__name__}: {e}"
            return None
