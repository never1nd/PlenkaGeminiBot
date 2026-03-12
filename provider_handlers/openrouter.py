from __future__ import annotations

import logging

from .openai_compat import OpenAICompatProviderHandler
from .utils import unique_keep_order

logger = logging.getLogger("bot")


class OpenRouterProviderHandler(OpenAICompatProviderHandler):
    def __init__(
        self,
        *,
        label: str,
        base_url: str,
        keys: list[str],
        fallback_models: list[str],
        discover_models: bool = True,
        models_path: str = "/models",
        chat_path: str = "/chat/completions",
        image_path: str = "/images/generations",
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer ",
    ) -> None:
        super().__init__(
            provider_id="openrouter",
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

    def discover_models(self, timeout_seconds: int) -> list[str]:
        if not self.discover_models_enabled:
            return list(self.fallback_models)
        if not self.base_url or not self.keys:
            return list(self.fallback_models)

        try:
            session = self._get_http_session()
            url = f"{self.base_url}/{self.models_path.lstrip('/')}"
            resp = session.get(
                url,
                headers=self._build_headers(self.keys[0]),
                timeout=timeout_seconds,
            )
            if not resp.ok:
                logger.warning(
                    "Provider model discovery failed for %s: %s %s",
                    self.provider_id,
                    resp.status_code,
                    resp.text[:200],
                )
                return list(self.fallback_models)

            payload = resp.json()
            rows = payload.get("data", []) if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                return list(self.fallback_models)

            model_names: list[str] = []
            for item in rows:
                if not isinstance(item, dict):
                    continue
                pricing = item.get("pricing", {})
                if not self._is_free_pricing(pricing):
                    continue
                model_name = str(item.get("id", "")).strip()
                if model_name:
                    model_names.append(model_name)

            if "openrouter/free" not in model_names:
                model_names.insert(0, "openrouter/free")

            return unique_keep_order(model_names) or list(self.fallback_models)
        except Exception as exc:
            logger.warning("Provider model discovery failed for %s: %s", self.provider_id, exc)
            return list(self.fallback_models)

    @staticmethod
    def _is_free_pricing(pricing: object) -> bool:
        if not isinstance(pricing, dict):
            return False

        def is_zero(value: object) -> bool:
            try:
                return float(value) == 0.0
            except (TypeError, ValueError):
                return False

        return is_zero(pricing.get("prompt")) and is_zero(pricing.get("completion"))
