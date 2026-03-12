from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .aithing import AIThingProviderHandler
from .base import BaseProviderHandler
from .google import GoogleGeminiProviderHandler
from .nvidia import NvidiaProviderHandler
from .openai_compat import OpenAICompatProviderHandler
from .openrouter import OpenRouterProviderHandler
from .utils import unique_keep_order
from .void import VoidProviderHandler

logger = logging.getLogger("bot")


def create_handler(
    *,
    provider_id: str,
    label: str,
    base_url: str,
    keys: list[str],
    fallback_models: list[str],
    discover_models: bool,
    models_path: str,
    chat_path: str,
    image_path: str,
    auth_header: str,
    auth_prefix: str,
) -> BaseProviderHandler:
    if provider_id == "google":
        return GoogleGeminiProviderHandler(
            api_keys=keys,
            fallback_models=fallback_models,
            label=label,
        )
    if provider_id == "nvidia":
        return NvidiaProviderHandler(
            label=label,
            base_url=base_url,
            api_keys=keys,
            fallback_models=fallback_models,
        )
    if provider_id == "void":
        return VoidProviderHandler(
            label=label,
            base_url=base_url,
            keys=keys,
            fallback_models=fallback_models,
            discover_models=discover_models,
            models_path=models_path,
            chat_path=chat_path,
            image_path=image_path,
            auth_header=auth_header,
            auth_prefix=auth_prefix,
        )
    if provider_id == "openrouter":
        return OpenRouterProviderHandler(
            label=label,
            base_url=base_url,
            keys=keys,
            fallback_models=fallback_models,
            discover_models=discover_models,
            models_path=models_path,
            chat_path=chat_path,
            image_path=image_path,
            auth_header=auth_header,
            auth_prefix=auth_prefix,
        )
    if provider_id in {"aithing", "random"}:
        return AIThingProviderHandler(
            label=label,
            base_url=base_url,
            keys=keys,
            fallback_models=fallback_models,
            discover_models=discover_models,
            models_path=models_path,
            chat_path=chat_path,
            image_path=image_path,
            auth_header=auth_header,
            auth_prefix=auth_prefix,
        )
    return OpenAICompatProviderHandler(
        provider_id=provider_id,
        label=label,
        base_url=base_url,
        keys=keys,
        fallback_models=fallback_models,
        discover_models=discover_models,
        models_path=models_path,
        chat_path=chat_path,
        image_path=image_path,
        auth_header=auth_header,
        auth_prefix=auth_prefix,
    )


def load_external_provider_handlers(config_file: str) -> dict[str, BaseProviderHandler]:
    config_path = Path(config_file)
    if not config_path.exists():
        logger.info("Custom provider config not found: %s", config_path)
        return {}

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        logger.warning("Failed to parse custom provider config %s: %s", config_path, exc)
        return {}

    providers = payload.get("providers", []) if isinstance(payload, dict) else []
    if not isinstance(providers, list):
        logger.warning("Custom provider config has invalid 'providers' field: %s", config_path)
        return {}

    handlers: dict[str, BaseProviderHandler] = {}
    for raw_provider in providers:
        if not isinstance(raw_provider, dict):
            continue

        provider_id = str(raw_provider.get("id", "")).strip().lower()
        if not provider_id:
            continue

        label = str(raw_provider.get("label", provider_id)).strip() or provider_id
        if provider_id == "random":
            provider_id = "aithing"
            if not label or label.lower().startswith("random"):
                label = "AIThing"

        if provider_id in handlers:
            logger.warning("Duplicate provider id '%s' in config; skipping later entry.", provider_id)
            continue

        collected_keys: list[str] = []

        inline_api_keys = raw_provider.get("api_keys", [])
        if isinstance(inline_api_keys, list):
            collected_keys.extend(str(key).strip() for key in inline_api_keys if str(key).strip())
        single_api_key = str(raw_provider.get("api_key", "")).strip()
        if single_api_key:
            collected_keys.append(single_api_key)

        base_url = str(raw_provider.get("base_url", "")).strip()
        if not base_url:
            env_name = str(raw_provider.get("base_url_env", "")).strip()
            if env_name:
                base_url = os.getenv(env_name, "").strip()

        fallback_models = raw_provider.get("fallback_models", [])
        fallback_model_names = [str(x).strip() for x in fallback_models] if isinstance(fallback_models, list) else []
        fallback_model_names = unique_keep_order([x for x in fallback_model_names if x])

        auth_header = str(raw_provider.get("auth_header", "Authorization")).strip() or "Authorization"
        if "auth_prefix" in raw_provider:
            raw_prefix = raw_provider.get("auth_prefix")
            auth_prefix = str(raw_prefix) if raw_prefix is not None else ""
        else:
            auth_prefix = "Bearer "
        models_path = str(raw_provider.get("models_path", "/models")).strip() or "/models"
        chat_path = str(raw_provider.get("chat_path", "/chat/completions")).strip() or "/chat/completions"
        image_path = str(raw_provider.get("image_path", "/images/generations")).strip() or "/images/generations"
        discover_models = bool(raw_provider.get("discover_models", True))
        keys = unique_keep_order(collected_keys)
        if provider_id not in {"google"} and not base_url:
            logger.warning("Skipping provider '%s': base_url is missing.", provider_id)
            continue
        if not keys:
            logger.warning("Provider '%s' has no keys loaded.", provider_id)

        handler = create_handler(
            provider_id=provider_id,
            label=label,
            base_url=base_url,
            keys=keys,
            fallback_models=fallback_model_names,
            discover_models=discover_models,
            models_path=models_path,
            chat_path=chat_path,
            image_path=image_path,
            auth_header=auth_header,
            auth_prefix=auth_prefix,
        )
        handlers[provider_id] = handler
        logger.info(
            "Loaded provider '%s' (%s): keys=%d fallback_models=%d",
            provider_id,
            handler.label,
            handler.key_count(),
            len(fallback_model_names),
        )

    return handlers
