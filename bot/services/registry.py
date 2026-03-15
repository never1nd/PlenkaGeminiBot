from __future__ import annotations

import asyncio
import hashlib
import logging
import re

from bot.config import Settings
from providers import BaseProvider, load_providers

logger = logging.getLogger("bot")


def _key(pid: str, model: str) -> str:
    h = hashlib.sha1(f"{pid}:{model}".encode()).hexdigest()[:10]
    return f"cp_{pid}_{h}"


class ModelRegistry:
    """Discovers and indexes models from all configured providers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.handlers: dict[str, BaseProvider] = {}
        self.provider_labels: dict[str, str] = {}

        # model index: key = "provider:model_name"
        self.labels: dict[str, str] = {}
        self.model_provider: dict[str, str] = {}
        self.model_name: dict[str, str] = {}

        # ordering
        self.all_order: list[str] = []
        self.order: list[str] = []
        self.order_set: set[str] = set()
        self.provider_order: list[str] = []
        self.provider_model_keys: dict[str, list[str]] = {}
        self.default_key: str = ""

    async def build_catalog(self) -> None:
        """Load providers and discover all models."""
        self.handlers = load_providers(self.settings.provider_config_file)

        priority = [p for p in ("sosikibot", "nvidia", "openrouter") if p in self.handlers]
        priority.extend(sorted(p for p in self.handlers if p not in ("sosikibot", "nvidia", "openrouter")))

        # discover models in parallel
        async def _discover(pid: str) -> tuple[str, list[str]]:
            handler = self.handlers[pid]
            try:
                return pid, await handler.discover_models(self.settings.regular_model_timeout)
            except Exception as exc:
                logger.warning("Discovery failed for %s: %s", pid, exc)
                return pid, []

        results = dict(await asyncio.gather(*(_discover(p) for p in priority)))

        # build index
        for pid in priority:
            handler = self.handlers[pid]
            self.provider_labels[pid] = handler.label
            models = results.get(pid, [])
            if models:
                logger.info("Loaded %d models for '%s' (%s).", len(models), pid, handler.label)
            else:
                logger.warning("No models for '%s' (%s).", pid, handler.label)

            for name in models:
                name = str(name).strip()
                if not name:
                    continue
                mk = _key(pid, name)
                # handle collisions (rare)
                if mk in self.labels:
                    i = 2
                    while f"{mk}_{i}" in self.labels:
                        i += 1
                    mk = f"{mk}_{i}"

                display = name
                fn = getattr(handler, "get_model_display_name", None)
                if callable(fn):
                    try:
                        display = str(fn(name)).strip() or name
                    except Exception:
                        pass

                self.labels[mk] = display[:56] + "..." if len(display) > 56 else display
                self.model_provider[mk] = pid
                self.model_name[mk] = name
                self.all_order.append(mk)

        self.order = list(self.all_order)
        self.order_set = set(self.order)
        self._rebuild_index()
        self.default_key = self.order[0] if self.order else ""

    def _rebuild_index(self) -> None:
        mapping: dict[str, list[str]] = {pid: [] for pid in self.handlers}
        for mk in self.order:
            pid = self.model_provider.get(mk, "")
            mapping.setdefault(pid, []).append(mk)
        prio = {"nvidia": 0, "openrouter": 1, "sosikibot": 2}
        self.provider_order = sorted(mapping, key=lambda x: (prio.get(x, 2), x))
        self.provider_model_keys = mapping

    # ── lookups ─────────────────────────────────────────────────────

    def get_label(self, key: str) -> str:
        return self.labels.get(key, key)

    def get_full_name(self, key: str) -> str:
        return self.model_name.get(key, self.get_label(key))

    def get_provider_id(self, key: str) -> str:
        return self.model_provider.get(key, "unknown")

    def get_provider_label(self, pid: str) -> str:
        return self.provider_labels.get(pid, pid)

    def find_key_by_name(self, model_name: str, provider_id: str = "") -> str:
        raw = str(model_name or "").strip()
        if not raw:
            return ""
        low = raw.lower()
        norm = re.sub(r"[\s_]+", "-", low)
        providers = [provider_id] if provider_id else self.provider_order

        for pid in providers:
            for mk in self.provider_model_keys.get(pid, []):
                n = self.model_name.get(mk, "").strip().lower()
                if n == low or re.sub(r"[\s_]+", "-", n) == norm:
                    return mk
        return ""

    def find_provider_model_key(self, pid: str, model_name: str) -> str:
        low = model_name.strip().lower()
        for mk in self.provider_model_keys.get(pid, []):
            if self.model_name.get(mk, "").strip().lower() == low:
                return mk
        return ""

    def set_filtered_order(self, filtered: list[str]) -> None:
        if filtered:
            self.order = filtered
        else:
            self.order = []
            logger.error("Capability filter removed all models.")
        self.order_set = set(self.order)
        self._rebuild_index()
        self.default_key = self.order[0] if self.order else ""

    def is_reasoning_model(self, model_name: str) -> bool:
        low = model_name.lower()
        return any(h in low for h in ("thinking", "reason", "deepseek-r1", "qwq", "r1-distill"))

    def get_timeout(self, model_name: str) -> int:
        if self.is_reasoning_model(model_name):
            return self.settings.reasoning_model_timeout
        return self.settings.regular_model_timeout
