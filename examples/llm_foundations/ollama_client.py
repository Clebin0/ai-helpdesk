"""Small, dependency-light HTTP client for the local Ollama API."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_MODEL = "qwen3.5:4b-q4_K_M"
DEFAULT_HOST = "http://127.0.0.1:11434"
MAX_RESPONSE_BYTES = 1_000_000
MAX_FORMAT_BYTES = 32_000
MAX_MESSAGES = 12
MAX_CONTENT_CHARS = 4_000


class OllamaError(RuntimeError):
    """Raised when the local Ollama request cannot be completed safely."""


@dataclass(frozen=True)
class OllamaClient:
    host: str = DEFAULT_HOST
    model: str = DEFAULT_MODEL
    timeout: float = 120.0

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("OLLAMA_HOST deve ser uma URL HTTP(S) válida.")

        parsed = urlsplit(self.host)
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("OLLAMA_HOST contém uma porta inválida.") from exc

        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("OLLAMA_HOST deve ser uma URL HTTP(S) válida.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "OLLAMA_HOST não pode conter credenciais, query ou fragmento."
            )
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("O nome do modelo não pode estar vazio.")
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("O timeout deve ser um número finito maior que zero.")

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.2,
        num_predict: int = 160,
        think: bool = False,
        format: str | dict[str, Any] | None = None,
    ) -> str:
        """Send a non-streaming chat request and return the assistant text.

        Set format="json" for JSON mode or pass a JSON Schema dictionary for
        constrained structured output. The caller must still validate the result.
        """
        validated = self._validate_messages(messages)

        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ValueError("temperature deve ser um número finito entre 0 e 2.")
        if (
            isinstance(num_predict, bool)
            or not isinstance(num_predict, int)
            or not 1 <= num_predict <= 4096
        ):
            raise ValueError("num_predict deve ser um inteiro entre 1 e 4096.")
        if not isinstance(think, bool):
            raise ValueError("think deve ser booleano.")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": validated,
            "stream": False,
            "think": think,
            "options": {
                "temperature": float(temperature),
                "num_predict": num_predict,
            },
        }

        if format is not None:
            if format != "json" and not isinstance(format, dict):
                raise ValueError("format deve ser 'json' ou um objeto JSON Schema.")
            if isinstance(format, dict):
                if not format:
                    raise ValueError("O JSON Schema não pode estar vazio.")
                try:
                    serialized_schema = json.dumps(
                        format, ensure_ascii=False, allow_nan=False
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError("O JSON Schema deve conter apenas valores JSON.") from exc
                if len(serialized_schema.encode("utf-8")) > MAX_FORMAT_BYTES:
                    raise ValueError("O JSON Schema excede o limite de 32 KB.")
            payload["format"] = format

        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        request = Request(
            f"{self.host.rstrip('/')}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raise OllamaError(f"Ollama respondeu com HTTP {exc.code}.") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaError(
                "Não foi possível acessar o Ollama local. Confira se ollama serve está ativo."
            ) from exc

        if len(raw) > MAX_RESPONSE_BYTES:
            raise OllamaError("A resposta do modelo excedeu o limite de 1 MB.")

        try:
            result: Any = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OllamaError("O Ollama retornou uma resposta JSON inválida.") from exc

        message = result.get("message") if isinstance(result, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise OllamaError(
                "A resposta do Ollama não contém texto em message.content."
            )
        return content.strip()

    @staticmethod
    def _validate_messages(
        messages: Sequence[dict[str, str]],
    ) -> list[dict[str, str]]:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise ValueError("messages deve ser uma sequência de mensagens.")
        if not messages or len(messages) > MAX_MESSAGES:
            raise ValueError(f"Envie entre 1 e {MAX_MESSAGES} mensagens.")

        validated: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"A mensagem {index + 1} deve ser um objeto.")

            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or role not in {
                "system",
                "user",
                "assistant",
            }:
                raise ValueError(f"Role inválida na mensagem {index + 1}.")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Conteúdo vazio ou inválido na mensagem {index + 1}.")
            if len(content) > MAX_CONTENT_CHARS:
                raise ValueError(
                    f"A mensagem {index + 1} excede {MAX_CONTENT_CHARS} caracteres."
                )
            validated.append({"role": role, "content": content.strip()})
        return validated
