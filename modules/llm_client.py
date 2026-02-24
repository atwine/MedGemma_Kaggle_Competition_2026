"""Local LLM client.

Supports two backends:
  1. Ollama  — quantized models via the Ollama server.
  2. HuggingFace Transformers — FP16 models loaded directly on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# HuggingFace Transformers backend — loads model directly on GPU (FP16)
# [src] https://huggingface.co/google/medgemma-1.5-4b-it
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HuggingFaceConfig:
    # Rationale: default to the official MedGemma 1.5 4B IT model.
    model: str = "google/medgemma-1.5-4b-it"
    # Rationale: Tesla T4 does not natively support bfloat16; use float16
    # which gives identical precision for inference. [src]
    # - https://docs.nvidia.com/cuda/turing-tuning-guide/
    torch_dtype: str = "float16"
    # Rationale: max tokens to generate per call. Matches the num_predict
    # concept from Ollama but with a higher default since FP16 is faster.
    max_new_tokens: int = 2000
    # Rationale: lower temperature for more deterministic clinical output.
    temperature: float = 0.2
    # Rationale: repetition_penalty >1.0 discourages repeating tokens.
    # Equivalent to Ollama's repeat_penalty.
    repetition_penalty: float = 1.3


class HuggingFaceClient:
    """Drop-in replacement for OllamaClient using HuggingFace Transformers.

    Loads the model once at construction time and reuses it for all calls.
    Has the exact same chat() interface as OllamaClient so all downstream
    callers (llm_screening, explanation_generator, etc.) work unchanged.
    """

    def __init__(self, config: HuggingFaceConfig | None = None) -> None:
        cfg = config or HuggingFaceConfig()

        # Rationale: allow overriding via environment without changing code.
        model_override = os.getenv("HF_MODEL")
        self._model_id = model_override if model_override else cfg.model
        self._max_new_tokens = cfg.max_new_tokens
        self._temperature = cfg.temperature
        self._repetition_penalty = cfg.repetition_penalty
        self._last_error: Optional[str] = None

        # Rationale: lazy-load heavy imports so the module can still be
        # imported on machines without torch/transformers installed.
        self._hf_model = None
        self._hf_processor = None

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(cfg.torch_dtype, torch.float16)

            logger.info("Loading HuggingFace model %s (dtype=%s)...", self._model_id, cfg.torch_dtype)
            self._hf_processor = AutoProcessor.from_pretrained(self._model_id)
            self._hf_model = AutoModelForImageTextToText.from_pretrained(
                self._model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
            logger.info("HuggingFace model loaded successfully.")
        except Exception as e:
            self._last_error = f"Failed to load HuggingFace model '{self._model_id}': {type(e).__name__}: {e}"
            logger.error(self._last_error)

    @property
    def model(self) -> str:
        return self._model_id

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
        """Generate a response using HuggingFace Transformers.

        Accepts the same arguments as OllamaClient.chat() for compatibility.
        The `format` parameter is not used (Transformers doesn't have
        built-in structured output), but the caller can still pass it.
        Generation parameters can be overridden via `options_override`:
          - max_new_tokens / num_predict: max tokens to generate
          - temperature: sampling temperature
          - repetition_penalty / repeat_penalty: repetition penalty
          - do_sample: whether to sample (default True if temperature > 0)
        """
        if self._hf_model is None or self._hf_processor is None:
            # Rationale: model failed to load at init time; surface the error.
            if not self._last_error:
                self._last_error = "HuggingFace model is not loaded"
            return None

        try:
            import torch
        except ImportError:
            self._last_error = "PyTorch is not installed"
            return None

        try:
            self._last_error = None

            # Rationale: merge caller overrides into generation parameters.
            max_new_tokens = self._max_new_tokens
            temperature = self._temperature
            repetition_penalty = self._repetition_penalty

            if options_override:
                # Rationale: accept both Ollama-style and HF-style parameter names.
                max_new_tokens = options_override.get(
                    "max_new_tokens",
                    options_override.get("num_predict", max_new_tokens),
                )
                temperature = options_override.get("temperature", temperature)
                repetition_penalty = options_override.get(
                    "repetition_penalty",
                    options_override.get("repeat_penalty", repetition_penalty),
                )

            # Rationale: convert ChatMessage dicts to the format expected by
            # the HuggingFace processor's apply_chat_template. Text-only
            # messages use [{"type": "text", "text": "..."}] content format.
            # [src] https://huggingface.co/google/medgemma-1.5-4b-it
            hf_messages = []
            for msg in messages:
                hf_messages.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}],
                })

            inputs = self._hf_processor.apply_chat_template(
                hf_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._hf_model.device)

            input_len = inputs["input_ids"].shape[-1]

            # Rationale: build generation kwargs matching the caller's intent.
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens),
                "repetition_penalty": float(repetition_penalty),
            }
            if float(temperature) > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = float(temperature)
            else:
                gen_kwargs["do_sample"] = False

            with torch.inference_mode():
                generation = self._hf_model.generate(**inputs, **gen_kwargs)

            # Rationale: strip the input tokens to get only the generated output.
            generation = generation[0][input_len:]
            decoded = self._hf_processor.decode(generation, skip_special_tokens=True)

            if not decoded or not decoded.strip():
                self._last_error = "HuggingFace model returned empty output"
                return None

            return decoded

        except Exception as e:
            self._last_error = f"HuggingFace generation failed: {type(e).__name__}: {e}"
            logger.error(self._last_error, exc_info=True)
            return None
