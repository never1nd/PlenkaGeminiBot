from __future__ import annotations

import builtins
from typing import Optional

from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    role: str
    content: str


class InputAttachment(BaseModel):
    kind: str = "file"
    mime_type: str = "application/octet-stream"
    file_name: str = "attachment"
    bytes: Optional[builtins.bytes] = None
    text: str = ""


class UsageStats(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ProviderResponse(BaseModel):
    text: str
    usage: UsageStats = Field(default_factory=UsageStats)


class ImageResponse(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None
