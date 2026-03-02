from __future__ import annotations

from typing import Any, Callable

HistoryMessage = dict[str, str]
UsageStats = dict[str, int]
InputAttachment = dict[str, Any]


class BaseProviderHandler:
    def __init__(self, provider_id: str, label: str) -> None:
        normalized_id = provider_id.strip().lower()
        self.provider_id = normalized_id
        self.label = label.strip() or normalized_id

    def discover_models(self, timeout_seconds: int) -> list[str]:
        return []

    def supports_input_attachments(self) -> bool:
        return False

    def generate_text(
        self,
        prompt: str,
        model_name: str,
        history: list[HistoryMessage],
        attachments: list[InputAttachment] | None = None,
        *,
        max_output_tokens: int,
        timeout_seconds: int,
        strip_reasoning: Callable[[str], str],
    ) -> tuple[str, UsageStats]:
        raise NotImplementedError

    def generate_image(
        self,
        prompt: str,
        model_name: str,
        *,
        size: str,
        timeout_seconds: int,
    ) -> dict[str, str]:
        raise NotImplementedError

    def key_count(self) -> int:
        return 0
