"""Local LLM client.

This project uses Ollama for local inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import List, Optional, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class OllamaConfig:
    # Rationale: default to the locally pulled model tag to avoid mismatches.
    model: str = "alibayram/medgemma:4b"


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

    @property
    def model(self) -> str:
        return self._model

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def chat(self, messages: List[ChatMessage]) -> Optional[str]:
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
            # connectivity debugging.
            client = Client(host=self._host)
            response = client.chat(model=self._model, messages=messages)
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
                return content

            if isinstance(response, dict):
                msg = response.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if not content:
                        self._last_error = "Ollama returned an empty message content"
                        return None
                    return str(content)

            self._last_error = f"Unrecognized ollama response type: {type(response)}"
            return None
        except Exception as e:
            # Rationale: local model may not be pulled/served yet; we degrade gracefully.
            self._last_error = f"Ollama call failed (host={self._host}): {type(e).__name__}: {e}"
            return None
