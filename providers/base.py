from __future__ import annotations

import abc
from typing import Callable

from bot.schemas import HistoryMessage, InputAttachment, ProviderResponse, ImageResponse


class BaseProvider(abc.ABC):
    """Abstract base for all AI provider handlers."""

    def __init__(self, provider_id: str, label: str) -> None:
        self.provider_id = provider_id.strip().lower()
        self.label = (label.strip() or self.provider_id)

    @abc.abstractmethod
    async def discover_models(self, timeout: int) -> list[str]:
        ...

    @abc.abstractmethod
    async def generate_text(
        self,
        prompt: str,
        model: str,
        history: list[HistoryMessage],
        attachments: list[InputAttachment] | None = None,
        *,
        max_tokens: int,
        timeout: int,
        strip_reasoning: Callable[[str], str],
    ) -> ProviderResponse:
        ...

    @abc.abstractmethod
    async def generate_image(
        self,
        prompt: str,
        model: str,
        *,
        size: str,
        timeout: int,
    ) -> ImageResponse:
        ...

    def supports_attachments(self) -> bool:
        return False

    def supports_text(self) -> bool:
        return True

    def key_count(self) -> int:
        return 0
