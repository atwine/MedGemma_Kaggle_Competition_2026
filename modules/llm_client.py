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

    @property
    def model(self) -> str:
        return self._model

    def chat(self, messages: List[ChatMessage]) -> Optional[str]:
        """Call the local Ollama chat API.

        Uses the Ollama Python library interface documented on Ollama model pages.
        Example usage shown for `from ollama import chat ...`. [src]
        - https://ollama.com/MedAIBase/MedGemma1.5
        """

        try:
            from ollama import chat as ollama_chat
        except Exception:
            # Rationale: if the Python client isn't installed, the app should still run
            # with deterministic fallback explanations.
            return None

        try:
            response = ollama_chat(model=self._model, messages=messages)
            # Rationale: ollama-python supports both dict-style and object-style
            # access for responses. [src]
            # - https://raw.githubusercontent.com/ollama/ollama-python/main/README.md
            if response is None:
                return None

            if hasattr(response, "message") and getattr(response, "message") is not None:
                return getattr(response, "message").content

            if isinstance(response, dict):
                msg = response.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    return str(content) if content else None

            return None
        except Exception:
            # Rationale: local model may not be pulled/served yet; we degrade gracefully.
            return None
