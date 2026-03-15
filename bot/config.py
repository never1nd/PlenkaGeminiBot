from __future__ import annotations

import re
from pathlib import Path
from typing import FrozenSet

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── parsing helpers ─────────────────────────────────────────────────

def _parse_ints(raw: str) -> frozenset[int]:
    ids: set[int] = set()
    for p in re.split(r"[,\s;]+", str(raw or "")):
        p = p.strip()
        if not p:
            continue
        try:
            v = int(p)
            if v > 0:
                ids.add(v)
        except ValueError:
            pass
    return frozenset(ids)


def _parse_strs(raw: str) -> frozenset[str]:
    return frozenset(t.strip().lower() for t in str(raw or "").split(",") if t.strip())


# ── settings ────────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore",
    )

    # telegram
    telegram_bot_token: str = Field(alias="TELEGRAM_BOT_TOKEN")
    bot_username: str = Field(default="", alias="TELEGRAM_BOT_USERNAME")
    concurrent_updates: int = Field(default=8, alias="CONCURRENT_UPDATES")

    # owners & access — stored as str from env, parsed in validator
    owner_user_id: str = Field(default="", alias="OWNER_USER_ID")
    owner_user_ids: str = Field(default="", alias="OWNER_USER_IDS")
    parsed_owner_ids: FrozenSet[int] = Field(default_factory=frozenset)

    # group behaviour
    group_keyword_trigger: str = Field(default="кокс", alias="GROUP_KEYWORD_TRIGGER")
    group_keyword_require_question: bool = Field(
        default=False, alias="GROUP_KEYWORD_REQUIRE_QUESTION",
    )
    group_provider_allowlist: str = Field(default="nvidia,openrouter", alias="GROUP_PROVIDER_ALLOWLIST")
    parsed_group_providers: FrozenSet[str] = Field(default_factory=frozenset)
    image_provider_allowlist: str = Field(default="sosikibot,cloudflare", alias="IMAGE_PROVIDER_ALLOWLIST")
    parsed_image_providers: FrozenSet[str] = Field(default_factory=frozenset)
    group_photo_keyword: str = Field(default="коксфото", alias="GROUP_PHOTO_KEYWORD")
    group_fast_model_name: str = Field(default="gemini-flash-lite-latest", alias="GROUP_FAST_MODEL_NAME")
    group_fast_model_provider: str = Field(default="sosikibot", alias="GROUP_FAST_MODEL_PROVIDER")

    # default model rotation (round-robin to distribute API load)
    default_model_rotation: tuple[str, ...] = (
        "sosikibot:gemini-flash-lite-latest",
        "sosikibot:gpt-5.2",
        "sosikibot:gpt-5.2-chat-latest",
        "sosikibot:gpt-5.3-chat-latest",
        "sosikibot:gpt-5.4",
        "void:gpt-5.2",
        "void:gpt-5.2-chat-latest",
        "void:gpt-5.3-chat-latest",
        "void:gpt-5.4",
        "google:gemini-2.5-flash",
        "void:gemini-3-flash-preview",
        "void:gemini-3.1-flash-lite-preview",
    )
    default_rotation_provider: str = ""

    # reply instructions
    reply_instruction_default: str = (
        "Reply in Telegram Markdown. No HTML. Give a complete answer without filler. "
        "Ты - кокс, просто имя, как у любого человека. Отвечай как человек и похуист. "
        "Не используй лишний раз приветствий и лишних слов - отвечай четко и по месту"
        "Не начинай сообщение со слов 'Кокс' или 'Кокс, ', просто сразу к делу, а также не используй эм-дешы"
    )
    reply_instruction_short: str = (
        "Reply in Telegram Markdown. No HTML. Be direct and to the point without filler. "
        "Ты - кокс, просто имя, как у любого человека. Отвечай как человек и похуист. "
        "Не используй лишний раз приветствій и лишних слов - отвечай четко и по месту"
        "Не начинай сообщение со слов 'Кокс' или 'Кокс, ', просто сразу к делу, а также не используй эм-дешы"
    )

    # memory / history
    memory_context_messages: int = Field(default=20, alias="MEMORY_CONTEXT_MESSAGES")
    memory_max_messages: int = Field(default=10, alias="MEMORY_MAX_MESSAGES")
    memory_compact_every: int = Field(default=10, alias="MEMORY_COMPACT_EVERY")

    # providers
    provider_config_file: str = Field(
        default="providers.json", alias="CUSTOM_PROVIDER_CONFIG_FILE",
    )

    # timeouts
    regular_model_timeout: int = Field(default=20, alias="REGULAR_MODEL_TIMEOUT_SECONDS")
    reasoning_model_timeout: int = Field(default=60, alias="REASONING_MODEL_TIMEOUT_SECONDS")
    image_model_timeout: int = Field(default=50, alias="IMAGE_MODEL_TIMEOUT_SECONDS")
    fallback_attempt_timeout: int = Field(default=10, alias="FALLBACK_ATTEMPT_TIMEOUT_SECONDS")

    # generation
    max_output_tokens: int = Field(default=4096, alias="MAX_OUTPUT_TOKENS")
    inline_max_output_tokens: int = 250
    telegram_reply_chunk_chars: int = 4000

    # probing & capabilities
    model_capability_ttl: int = Field(default=300, alias="MODEL_CAPABILITY_TTL_SECONDS")
    model_probe_timeout: int = Field(default=10, alias="MODEL_PROBE_TIMEOUT_SECONDS")
    model_probe_workers: int = Field(default=8, alias="MODEL_PROBE_WORKERS")
    model_capability_probe_enabled: bool = Field(
        default=True, alias="MODEL_CAPABILITY_PROBE_ENABLED",
    )
    model_capability_recheck_enabled: bool = Field(
        default=True, alias="MODEL_CAPABILITY_RECHECK_ENABLED",
    )
    model_capability_recheck_interval: int = Field(
        default=300, alias="MODEL_CAPABILITY_RECHECK_INTERVAL_SECONDS",
    )
    model_capability_recheck_initial_delay: int = Field(
        default=45, alias="MODEL_CAPABILITY_RECHECK_INITIAL_DELAY_SECONDS",
    )
    model_capability_recheck_force_full: bool = Field(
        default=True, alias="MODEL_CAPABILITY_RECHECK_FORCE_FULL",
    )
    model_hide_unavailable: bool = Field(
        default=True, alias="MODEL_HIDE_UNAVAILABLE_MODELS",
    )

    # provider availability
    provider_unavailable_ttl: int = Field(
        default=15, alias="PROVIDER_UNAVAILABLE_TTL_SECONDS",
    )
    provider_available_ttl: int = Field(
        default=10, alias="PROVIDER_AVAILABLE_TTL_SECONDS",
    )
    provider_max_concurrency: int = Field(
        default=2, alias="PROVIDER_MAX_CONCURRENCY",
    )
    provider_availability_check_on_request: bool = Field(
        default=False, alias="PROVIDER_AVAILABILITY_CHECK_ON_REQUEST",
    )
    provider_availability_recheck_enabled: bool = Field(
        default=True, alias="PROVIDER_AVAILABILITY_RECHECK_ENABLED",
    )
    provider_availability_recheck_interval: int = Field(
        default=90, alias="PROVIDER_AVAILABILITY_RECHECK_INTERVAL_SECONDS",
    )
    provider_availability_recheck_initial_delay: int = Field(
        default=15, alias="PROVIDER_AVAILABILITY_RECHECK_INITIAL_DELAY_SECONDS",
    )
    non_reprobe_providers_str: str = Field(default="sosikibot,openrouter", alias="NON_REPROBE_PROVIDERS")
    parsed_non_reprobe: FrozenSet[str] = Field(default_factory=frozenset)

    # images
    image_generation_size: str = Field(default="1024x1024", alias="IMAGE_GENERATION_SIZE")
    default_image_model_name: str = Field(
        default="@cf/stabilityai/stable-diffusion-xl-base-1.0", alias="DEFAULT_IMAGE_MODEL_NAME",
    )
    default_image_model_provider: str = Field(
        default="cloudflare", alias="DEFAULT_IMAGE_MODEL_PROVIDER",
    )

    # attachments
    max_input_attachment_count: int = Field(default=3, alias="MAX_INPUT_ATTACHMENT_COUNT")
    max_input_attachment_bytes: int = Field(
        default=10 * 1024 * 1024, alias="MAX_INPUT_ATTACHMENT_BYTES",
    )
    max_input_text_attachment_bytes: int = Field(
        default=512 * 1024, alias="MAX_INPUT_TEXT_ATTACHMENT_BYTES",
    )
    max_input_text_attachment_chars: int = Field(
        default=20_000, alias="MAX_INPUT_TEXT_ATTACHMENT_CHARS",
    )

    # inline
    inline_allowed_model_bases: tuple[str, ...] = (
        "gemini-flash-lite-latest",
        "gemini-3.1-flash-lite",
        "gemini-3.0-flash",
        "gemini-2.5-flash",
    )
    inline_placeholder_text: str = "Generating response..."

    # paths
    allowlist_file: Path = Field(default=Path("allowlist.json"), alias="ALLOWLIST_FILE")
    users_db_file: Path = Field(default=Path("users.db"), alias="USERS_DB_FILE")
    model_prefs_file: Path = Field(default=Path("model_prefs.json"), alias="MODEL_PREFS_FILE")

    # misc
    sqlite_busy_timeout_ms: int = Field(default=5000, alias="SQLITE_BUSY_TIMEOUT_MS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @model_validator(mode="after")
    def parse_derived(self) -> Settings:
        # parse IDs
        combined_owners = ",".join(v for v in (self.owner_user_id, self.owner_user_ids) if v)
        self.parsed_owner_ids = _parse_ints(combined_owners)
        if not self.parsed_owner_ids:
            raise ValueError("OWNER_USER_ID or OWNER_USER_IDS must be set.")

        # parse string sets
        self.parsed_group_providers = _parse_strs(self.group_provider_allowlist)
        self.parsed_image_providers = _parse_strs(self.image_provider_allowlist)
        self.parsed_non_reprobe = _parse_strs(self.non_reprobe_providers_str)

        # normalize
        self.bot_username = self.bot_username.strip().lstrip("@")
        self.group_fast_model_provider = self.group_fast_model_provider.strip().lower()
        self.default_image_model_provider = self.default_image_model_provider.strip().lower()
        self.log_level = self.log_level.upper()

        # bounds
        self.provider_unavailable_ttl = max(15, self.provider_unavailable_ttl)
        self.provider_available_ttl = max(10, self.provider_available_ttl)
        self.max_input_attachment_bytes = max(64 * 1024, self.max_input_attachment_bytes)
        self.sqlite_busy_timeout_ms = max(1000, self.sqlite_busy_timeout_ms)
        if self.memory_max_messages <= 0:
            self.memory_max_messages = self.memory_context_messages
        self.memory_max_messages = max(1, self.memory_max_messages)
        if self.memory_compact_every <= 0:
            self.memory_compact_every = self.memory_max_messages
        self.memory_compact_every = max(1, self.memory_compact_every)

        return self

    # ── convenience properties ──────────────────────────────────────

    @property
    def owner_ids(self) -> FrozenSet[int]:
        return self.parsed_owner_ids

    @property
    def group_providers(self) -> FrozenSet[str]:
        return self.parsed_group_providers

    @property
    def image_providers(self) -> FrozenSet[str]:
        return self.parsed_image_providers

    @property
    def non_reprobe(self) -> FrozenSet[str]:
        return self.parsed_non_reprobe

    def is_owner(self, user_id: int) -> bool:
        return user_id in self.parsed_owner_ids



def load_settings() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]
    except Exception as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
