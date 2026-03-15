"""Config-driven provider factory.

Reads providers.json and creates the appropriate handler for each entry.
No more per-provider subclass files — everything is config-driven except
OpenRouter (custom discovery) and Google (SDK-based).
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from .base import BaseProvider
from .cloudflare import CloudflareProvider
from .krea import KreaProvider
from .google import GoogleProvider
from .openai_compat import OpenAICompatProvider, OpenRouterProvider

logger = logging.getLogger("bot")


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [s for s in (x.strip() for x in items) if s and s not in seen and not seen.add(s)]


def _split_keys(raw: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,\s;]+", str(raw or "")) if p.strip()]


def _create(
    *,
    pid: str,
    label: str,
    base_url: str,
    keys: list[str],
    fallback_models: list[str],
    discover: bool,
    models_path: str,
    chat_path: str,
    image_path: str,
    auth_header: str,
    auth_prefix: str,
    account_id: str = "",
    auth_email: str = "",
) -> BaseProvider:
    if pid == "google":
        return GoogleProvider(keys=keys, fallback_models=fallback_models, label=label)

    if pid == "cloudflare":
        return CloudflareProvider(
            provider_id=pid,
            label=label,
            account_id=account_id,
            keys=keys,
            fallback_models=fallback_models,
            auth_header=auth_header,
            auth_prefix=auth_prefix,
            auth_email=auth_email,
        )
    if pid == "krea":
        return KreaProvider(
            provider_id=pid,
            label=label,
            base_url=base_url,
            keys=keys,
            fallback_models=fallback_models,
            auth_header=auth_header,
            auth_prefix=auth_prefix,
        )

    cls = OpenRouterProvider if pid == "openrouter" else OpenAICompatProvider
    return cls(
        provider_id=pid,
        label=label,
        base_url=base_url,
        keys=keys,
        fallback_models=fallback_models,
        discover=discover,
        models_path=models_path,
        chat_path=chat_path,
        image_path=image_path,
        auth_header=auth_header,
        auth_prefix=auth_prefix,
    )


def load_providers(config_file: str) -> dict[str, BaseProvider]:
    """Load providers from a JSON config file."""
    path = Path(config_file)
    if not path.exists():
        logger.info("Provider config not found: %s", path)
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        logger.warning("Failed to parse provider config %s: %s", path, exc)
        return {}

    entries = data.get("providers", []) if isinstance(data, dict) else []
    if not isinstance(entries, list):
        logger.warning("Invalid 'providers' field in %s", path)
        return {}

    handlers: dict[str, BaseProvider] = {}

    for raw in entries:
        if not isinstance(raw, dict):
            continue

        pid = str(raw.get("id", "")).strip().lower()
        if not pid:
            continue

        label = str(raw.get("label", pid)).strip() or pid

        # normalize legacy ids
        if pid == "random":
            pid = "aithing"
            if label.lower().startswith("random"):
                label = "AIThing"

        if pid in handlers:
            logger.warning("Duplicate provider '%s' — skipping.", pid)
            continue

        # collect keys
        keys: list[str] = []
        api_keys = raw.get("api_keys", [])
        if isinstance(api_keys, list):
            keys.extend(str(k).strip() for k in api_keys if str(k).strip())
        single = str(raw.get("api_key", "")).strip()
        if single:
            keys.append(single)

        env_key = str(raw.get("api_key_env", "")).strip()
        if env_key:
            keys.extend(_split_keys(os.getenv(env_key, "")))

        env_keys = raw.get("api_keys_env", [])
        if isinstance(env_keys, list):
            for name in env_keys:
                name = str(name).strip()
                if name:
                    keys.extend(_split_keys(os.getenv(name, "")))
        elif isinstance(env_keys, str) and env_keys.strip():
            keys.extend(_split_keys(os.getenv(env_keys.strip(), "")))
        keys = _unique(keys)

        # base url
        base_url = str(raw.get("base_url", "")).strip()
        if not base_url:
            env_name = str(raw.get("base_url_env", "")).strip()
            if env_name:
                base_url = os.getenv(env_name, "").strip()

        if pid not in ("google", "cloudflare") and not base_url:
            logger.warning("Skipping provider '%s': no base_url.", pid)
            continue
        if not keys:
            logger.warning("Provider '%s' has no API keys.", pid)

        fallback = raw.get("fallback_models", [])
        fallback_models = _unique([str(x).strip() for x in fallback] if isinstance(fallback, list) else [])

        auth_header = str(raw.get("auth_header", "Authorization")).strip() or "Authorization"
        if "auth_prefix" in raw:
            rp = raw.get("auth_prefix")
            auth_prefix = str(rp) if rp is not None else ""
        else:
            auth_prefix = "Bearer "
        models_path = str(raw.get("models_path", "/models")).strip() or "/models"
        chat_path = str(raw.get("chat_path", "/chat/completions")).strip() or "/chat/completions"
        image_path = str(raw.get("image_path", "/images/generations")).strip() or "/images/generations"
        discover = bool(raw.get("discover_models", True))

        account_id = str(raw.get("account_id", "")).strip()
        auth_email = str(raw.get("auth_email", "")).strip()
        auth_email_env = str(raw.get("auth_email_env", "")).strip()
        if auth_email_env:
            auth_email = os.getenv(auth_email_env, "").strip() or auth_email

        handler = _create(
            pid=pid, label=label, base_url=base_url, keys=keys,
            fallback_models=fallback_models, discover=discover,
            models_path=models_path, chat_path=chat_path, image_path=image_path,
            auth_header=auth_header, auth_prefix=auth_prefix,
            account_id=account_id, auth_email=auth_email,
        )
        handlers[pid] = handler
        logger.info(
            "Loaded provider '%s' (%s): keys=%d fallback_models=%d",
            pid, handler.label, handler.key_count(), len(fallback_models),
        )

    return handlers
