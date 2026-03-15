"""Generation service — retry loop and model dispatch."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from bot.config import Settings
from bot.model_utils import is_default_model_key, is_image_model_name
from bot.schemas import InputAttachment
from bot.services.probing import (
    CapabilityCache,
    ProviderAvailability,
    strip_reasoning,
)
from bot.services.registry import ModelRegistry
from providers.errors import ErrorKind, classify

logger = logging.getLogger("bot")


class GenerationService:
    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        capabilities: CapabilityCache,
        availability: ProviderAvailability,
    ) -> None:
        self._s = settings
        self._reg = registry
        self._cap = capabilities
        self._avail = availability
        self._rotation_counter = 0
        self._rotation_pool: list[str] = []  # built after catalog is ready
        self._rotation_pinned: list[str] = []
        self._rotation_preferred: list[str] = []
        self._rotation_fallback: list[str] = []
        self._provider_semaphores: dict[str, asyncio.Semaphore] = {}

    # ── candidate building ──────────────────────────────────────────

    def _candidates(self, primary: str) -> list[str]:
        pid = self._reg.model_provider.get(primary, "")
        name = self._reg.model_name.get(primary, "")
        seen: set[str] = set()
        out: list[str] = []

        def add(k: str) -> None:
            if k and k not in seen and k in self._reg.order_set:
                seen.add(k)
                out.append(k)

        add(primary)
        for mk in self._reg.provider_model_keys.get(pid, []):
            add(mk)
        if name:
            for p in self._reg.provider_order:
                if p != pid:
                    add(self._reg.find_provider_model_key(p, name))
        for mk in self._reg.order:
            add(mk)
        return out

    def _image_candidates(self, primary: str) -> list[str]:
        pid = self._reg.model_provider.get(primary, "")
        name = self._reg.model_name.get(primary, "")
        is_img = is_image_model_name(name)
        seen: set[str] = set()
        out: list[str] = []

        def add(k: str) -> None:
            if k and k not in seen and k in self._reg.order_set:
                seen.add(k)
                out.append(k)

        if is_img:
            add(primary)
        for mk in self._reg.provider_model_keys.get(pid, []):
            if is_image_model_name(self._reg.model_name.get(mk, "")):
                add(mk)
        if name and is_img:
            for p in self._reg.provider_order:
                if p != pid:
                    add(self._reg.find_provider_model_key(p, name))
        for mk in self._reg.order:
            if is_image_model_name(self._reg.model_name.get(mk, "")):
                add(mk)
        return out

    async def _filter(
        self,
        raw: list[str],
        *,
        providers: set[str] | None = None,
        need_attachments: bool = False,
        need_text: bool = False,
    ) -> list[str]:
        if providers:
            norm = {x.strip().lower() for x in providers if x.strip()}
            raw = [k for k in raw if self._reg.model_provider.get(k, "") in norm]
            if not raw:
                raw = [k for k in self._reg.order if self._reg.model_provider.get(k, "") in norm]
            if not raw:
                raise RuntimeError("No allowed providers configured.")

        # gather unique provider ids and check availability
        unique_pids = list(dict.fromkeys(
            self._reg.model_provider.get(k, "") for k in raw
            if self._reg.model_provider.get(k, "")
        ))
        if self._s.provider_availability_check_on_request:
            checks = await asyncio.gather(*(self._avail.check(p) for p in unique_pids))
            avail = dict(zip(unique_pids, checks))
        else:
            avail: dict[str, tuple[bool, str]] = {}
            for pid in unique_pids:
                cached = self._avail.get_cached(pid)
                if cached is None:
                    avail[pid] = (True, "")
                else:
                    avail[pid] = cached

        result: list[str] = []
        for k in raw:
            pid = self._reg.model_provider.get(k, "")
            name = self._reg.model_name.get(k, "")
            if not pid or not name:
                continue
            handler = self._reg.handlers.get(pid)
            if need_attachments and (not handler or not handler.supports_attachments()):
                continue
            if need_text and handler and not handler.supports_text():
                continue
            if not avail.get(pid, (False, ""))[0]:
                continue
            skip, _ = self._cap.should_skip(pid, name)
            if skip:
                continue
            result.append(k)

        return result

    # ── retry loop ──────────────────────────────────────────────────

    async def _retry(
        self,
        keys: list[str],
        call_fn,
        *,
        t0: float,
        mode: str,
        ignore_cooldown: bool = False,
    ) -> Any:
        tried: list[str] = []
        last_err: Exception | None = None

        for i, key in enumerate(keys):
            pid = self._reg.model_provider.get(key, "")
            name = self._reg.model_name.get(key, "")
            if not pid or not name:
                continue
            if not ignore_cooldown:
                cached = self._avail.get_cached(pid)
                if cached is not None and not cached[0]:
                    reason = cached[1]
                    logger.info("Skipping %s/%s due to provider cooldown: %s", pid, name, reason)
                    continue

            if mode == "image":
                timeout = self._s.image_model_timeout
            else:
                base_to = self._reg.get_timeout(name)
                elapsed = int(time.monotonic() - t0)
                remaining = max(5, base_to - elapsed)
                timeout = remaining if i == 0 else min(remaining, self._s.fallback_attempt_timeout)

            logger.info(
                "%s attempt %d/%d provider=%s model=%s timeout=%ss",
                mode.capitalize(), i + 1, len(keys), pid, name, timeout,
            )
            tried.append(f"{pid}:{name}")

            try:
                sem = self._provider_sem(pid)
                async with sem:
                    result = await call_fn(pid, name, timeout)
                self._cap.set(pid, name, ErrorKind.AVAILABLE)
                self._avail.set_cached(pid, True, "", self._s.provider_available_ttl)
                return result
            except Exception as exc:
                last_err = exc
                kind = classify(str(exc))

                if kind != ErrorKind.AVAILABLE:
                    self._cap.set(pid, name, kind, error_text=str(exc))
                if kind in (ErrorKind.QUOTA, ErrorKind.TRANSIENT, ErrorKind.AUTH):
                    self._avail.set_cached(pid, False, str(exc), self._s.provider_unavailable_ttl)

                logger.warning("%s attempt failed %s/%s: %s", mode.capitalize(), pid, name, exc)

                if not kind.is_retryable:
                    break

        desc = ", ".join(tried) or "none"
        if last_err:
            raise RuntimeError(f"{mode.capitalize()} failed after: {desc}. Error: {last_err}") from last_err
        raise RuntimeError(f"No {mode} models available.")

    # ── public API ──────────────────────────────────────────────────

    async def generate_text(
        self,
        prompt: str,
        model_key: str,
        history: list[dict[str, str]],
        attachments: list[InputAttachment] | None = None,
        *,
        provider_allowlist: set[str] | None = None,
    ) -> tuple[str, str, str, dict[str, int]]:
        is_default = (
            is_default_model_key(model_key)
            or model_key not in self._reg.order_set
            or model_key == self._reg.default_key
        )
        if is_default:
            raw = self._default_rotation_candidates()
            if not raw:
                raise RuntimeError("No default models available.")
        else:
            raw = self._candidates(model_key)

        has_att = bool(attachments)
        keys = await self._filter(
            raw, providers=provider_allowlist, need_attachments=has_att, need_text=True,
        )

        if not keys:
            if has_att:
                raise RuntimeError("No providers support the uploaded attachment types.")
            keys = list(raw)

        t0 = time.monotonic()

        async def call(pid: str, name: str, timeout: int):
            handler = self._reg.handlers.get(pid)
            if not handler:
                raise RuntimeError(f"Handler not loaded: {pid}")
            resp = await handler.generate_text(
                prompt, name, history, attachments,
                max_tokens=self._s.max_output_tokens,
                timeout=timeout,
                strip_reasoning=strip_reasoning,
            )
            if not resp.text:
                raise RuntimeError(f"{name} returned an empty response.")
            usage = resp.usage.model_dump(exclude_none=True) if resp.usage else {}
            return resp.text, name, pid, usage

        return await self._retry(keys, call, t0=t0, mode="text")

    async def generate_inline_text(self, prompt: str) -> tuple[str, str, str, dict[str, int]]:
        self._ensure_rotation_pool()
        rotation = self._default_rotation_candidates() if self._rotation_pool else []
        inline = self._inline_keys()
        if rotation:
            if inline:
                inline_set = set(inline)
                keys = [k for k in rotation if k in inline_set]
                if not keys:
                    keys = rotation
            else:
                keys = rotation
        else:
            keys = inline or list(self._reg.order)
        if not keys:
            raise RuntimeError("Inline mode unavailable (no models).")
        filtered = await self._filter(keys, need_text=True)
        if not filtered:
            filtered = list(keys)

        t0 = time.monotonic()

        async def call(pid: str, name: str, timeout: int):
            handler = self._reg.handlers.get(pid)
            if not handler:
                raise RuntimeError(f"Handler not loaded: {pid}")
            resp = await handler.generate_text(
                prompt, name, [], [],
                max_tokens=self._s.inline_max_output_tokens,
                timeout=timeout,
                strip_reasoning=strip_reasoning,
            )
            if not resp.text:
                raise RuntimeError(f"{name} returned an empty response.")
            usage = resp.usage.model_dump(exclude_none=True) if resp.usage else {}
            return resp.text, name, pid, usage

        return await self._retry(filtered, call, t0=t0, mode="inline", ignore_cooldown=True)

    async def generate_image(
        self,
        prompt: str,
        model_key: str,
        *,
        provider_allowlist: set[str] | None = None,
    ) -> tuple[dict[str, str], str, str]:
        chosen = model_key if model_key in self._reg.order_set else self._reg.default_key
        if not chosen:
            raise RuntimeError("No models available.")

        raw = self._image_candidates(chosen)

        if self._s.default_image_model_name:
            dk = self._resolve_image_model_key(
                self._s.default_image_model_name, self._s.default_image_model_provider,
            )
            if dk:
                logger.info("Default image model resolved: key=%s provider=%s model=%s",
                            dk, self._reg.model_provider.get(dk, "?"), self._reg.model_name.get(dk, "?"))
                raw = [dk] + [k for k in raw if k != dk]
            else:
                logger.warning("Default image model '%s' (provider=%s) not found in registry.",
                               self._s.default_image_model_name, self._s.default_image_model_provider)

        keys = await self._filter(raw, providers=provider_allowlist)
        if not keys:
            keys = list(raw)
        if not keys:
            raise RuntimeError("No image models available.")
        t0 = time.monotonic()

        async def call(pid: str, name: str, timeout: int):
            handler = self._reg.handlers.get(pid)
            if not handler:
                raise RuntimeError(f"Handler not loaded: {pid}")
            resp = await handler.generate_image(
                prompt, name, size=self._s.image_generation_size, timeout=timeout,
            )
            if not resp.url and not resp.b64_json:
                raise RuntimeError(f"{name} returned no image data.")
            data = {}
            if resp.url:
                data["url"] = resp.url
            if resp.b64_json:
                data["b64_json"] = resp.b64_json
            return data, name, pid

        return await self._retry(keys, call, t0=t0, mode="image")

    # ── probing ─────────────────────────────────────────────────────

    async def run_probe(self, *, force_full: bool = False) -> None:
        if not self._s.model_capability_probe_enabled and not force_full:
            return

        probe_keys = list(self._reg.all_order) if force_full else self._smart_keys()
        probe_keys = [k for k in probe_keys
                      if self._reg.model_provider.get(k, "") not in self._s.non_reprobe]

        if not force_full:
            now = int(time.time())
            probe_keys = [
                k for k in probe_keys
                if not (c := self._cap.get(
                    self._reg.model_provider.get(k, ""),
                    self._reg.model_name.get(k, ""),
                )) or int(c.get("expires_at", 0)) <= now
            ]

        if not probe_keys:
            return

        # check provider availability concurrently
        unique_pids = list(dict.fromkeys(
            self._reg.model_provider.get(k, "") for k in probe_keys
            if self._reg.model_provider.get(k, "")
        ))
        avail_results = await asyncio.gather(
            *(self._avail.check(p, force=force_full) for p in unique_pids)
        )
        prov_ok = dict(zip(unique_pids, avail_results))

        filtered: list[str] = []
        for k in probe_keys:
            pid = self._reg.model_provider.get(k, "")
            name = self._reg.model_name.get(k, "")
            if not pid or not name:
                continue
            if not prov_ok.get(pid, (False, ""))[0]:
                self._cap.set(pid, name, ErrorKind.TRANSIENT,
                              error_text=prov_ok.get(pid, (False, ""))[1])
                continue
            filtered.append(k)

        if not filtered:
            return

        logger.info("Probing %d model capabilities (workers=%d).", len(filtered), self._s.model_probe_workers)
        sem = asyncio.Semaphore(self._s.model_probe_workers)
        counters: dict[str, int] = {}
        probe_timeout = self._s.model_probe_timeout

        async def probe(key: str) -> str:
            async with sem:
                pid = self._reg.model_provider.get(key, "")
                name = self._reg.model_name.get(key, "")
                handler = self._reg.handlers.get(pid)
                if not handler:
                    return ErrorKind.UNKNOWN.value
                t0 = time.monotonic()
                try:
                    prov_sem = self._provider_sem(pid)
                    async with prov_sem:
                        if not handler.supports_text():
                            img = await asyncio.wait_for(
                                handler.generate_image(
                                    "probe image", name,
                                    size="512x512", timeout=probe_timeout,
                                ),
                                timeout=probe_timeout + 2,
                            )
                            kind = ErrorKind.AVAILABLE if (img.url or img.b64_json) else ErrorKind.UNKNOWN
                        else:
                            resp = await asyncio.wait_for(
                                handler.generate_text(
                                    "Reply with exactly: ok", name, [], [],
                                    max_tokens=8, timeout=probe_timeout,
                                    strip_reasoning=strip_reasoning,
                                ),
                                timeout=probe_timeout + 2,
                            )
                            kind = ErrorKind.AVAILABLE if resp.text.strip() else ErrorKind.UNKNOWN
                except Exception as exc:
                    kind = classify(str(exc)) if handler.supports_text() else ErrorKind.UNKNOWN
                    latency = (time.monotonic() - t0) * 1000
                    self._cap.set(pid, name, kind,
                                  error_text="" if kind == ErrorKind.AVAILABLE else str(exc),
                                  latency_ms=latency)
                    return kind.value
                latency = (time.monotonic() - t0) * 1000
                self._cap.set(pid, name, kind, latency_ms=latency)
                return kind.value

        results = await asyncio.gather(*(probe(k) for k in filtered), return_exceptions=True)
        for r in results:
            s = r if isinstance(r, str) else ErrorKind.UNKNOWN.value
            counters[s] = counters.get(s, 0) + 1

        summary = ", ".join(f"{k}={v}" for k, v in sorted(counters.items())) or "none"
        logger.info("Probe complete: %s", summary)

    def reorder_by_latency(self) -> None:
        """Sort catalog by latency, with curated fallback models first."""
        latencies = self._cap.get_all_latencies()
        if not latencies:
            logger.info("No latency data available, skipping reorder.")
            return

        def _lat(mk: str) -> float:
            pid = self._reg.model_provider.get(mk, "")
            name = self._reg.model_name.get(mk, "")
            lat = latencies.get(f"{pid}::{name}".lower())
            return lat if lat is not None else 999_999.0

        # collect curated fallback model keys
        fallback_keys: set[str] = set()
        for pid, handler in self._reg.handlers.items():
            for name in getattr(handler, "fallback_models", []) or []:
                mk = self._reg.find_provider_model_key(pid, str(name))
                if mk and mk in self._reg.order_set:
                    fallback_keys.add(mk)

        # fallback models first (sorted by latency), then everything else
        fb_sorted = sorted(
            [mk for mk in self._reg.order if mk in fallback_keys], key=_lat,
        )
        rest_sorted = sorted(
            [mk for mk in self._reg.order if mk not in fallback_keys], key=_lat,
        )
        reordered = fb_sorted + rest_sorted
        self._reg.set_filtered_order(reordered)

        # log top models
        top = reordered[:5]
        lines = []
        for mk in top:
            pid = self._reg.model_provider.get(mk, "")
            name = self._reg.model_name.get(mk, "")
            tag = " [fallback]" if mk in fallback_keys else ""
            lines.append(f"  {pid}/{name}: {_lat(mk):.0f}ms{tag}")
        logger.info("Models reordered by latency. Top 5:\n%s", "\n".join(lines))

    def apply_filter(self, *, force: bool = False, strict: bool = False) -> None:
        if not self._s.model_hide_unavailable and not force:
            return
        filtered: list[str] = []
        hidden = 0
        for mk in self._reg.all_order:
            pid = self._reg.model_provider.get(mk, "")
            name = self._reg.model_name.get(mk, "")
            if not pid or not name:
                continue
            skip, _ = self._cap.should_skip(pid, name)
            if skip:
                hidden += 1
            else:
                filtered.append(mk)
        if not filtered and strict:
            raise RuntimeError("No available models after probe.")
        self._reg.set_filtered_order(filtered)
        if hidden:
            logger.info("Capability filter hidden %d unavailable models.", hidden)

    async def reconcile_availability(self, *, force: bool = False) -> dict[str, int]:
        pids = sorted(
            p for p in self._reg.handlers if p not in self._s.non_reprobe
        )
        changed = False
        stats = {"available": 0, "unavailable": 0, "marked_transient": 0, "cleared": 0}

        # cache previous state before concurrent checks
        prev_states = {pid: self._avail.get_cached(pid) for pid in pids}

        # check all providers concurrently
        results = await asyncio.gather(
            *(self._avail.check(pid, force=force) for pid in pids)
        )

        for pid, (ok, reason) in zip(pids, results):
            prev = prev_states[pid]
            was_down = prev is not None and not prev[0]
            if ok:
                stats["available"] += 1
                if was_down:
                    n = self._cap.clear_transient(pid)
                    if n:
                        stats["cleared"] += n
                        changed = True
            else:
                stats["unavailable"] += 1
                for mk in self._reg.provider_model_keys.get(pid, []):
                    name = self._reg.model_name.get(mk, "")
                    if name:
                        self._cap.mark_transient(pid, name, reason)
                        stats["marked_transient"] += 1
                        changed = True

        if changed:
            self.apply_filter(force=True)
        return stats

    # ── helpers ──────────────────────────────────────────────────────

    def build_rotation_pool(self) -> None:
        """Resolve default_model_rotation names into registry keys."""
        pool: list[str] = []
        seen: set[str] = set()
        for raw in self._s.default_model_rotation:
            name = str(raw or "").strip()
            if not name:
                continue
            provider = self._s.default_rotation_provider
            if ":" in name:
                left, right = name.split(":", 1)
                provider = left.strip().lower()
                name = right.strip()

            key = self._reg.find_key_by_name(name, provider) if provider else self._reg.find_key_by_name(name)

            if not key and provider:
                target = name.strip().lower()
                for mk in self._reg.provider_model_keys.get(provider, []):
                    n = self._reg.model_name.get(mk, "").strip().lower()
                    if n == target or ("/" in n and n.endswith(f"/{target}")):
                        key = mk
                        break

            if key and key not in seen and key in self._reg.order_set:
                seen.add(key)
                pool.append(key)
            elif provider:
                logger.warning("Default rotation model not found: %s:%s", provider, name)
            else:
                logger.warning("Default rotation model not found: %s", name)
        preferred: list[str] = []
        fallback: list[str] = []
        for mk in pool:
            name = self._reg.model_name.get(mk, "").lower()
            if "gemini" in name or "qwen" in name:
                preferred.append(mk)
            else:
                fallback.append(mk)

        order_idx = {mk: i for i, mk in enumerate(pool)}
        provider_rank = {"sosikibot": 0, "nvidia": 1, "openrouter": 2}

        def _is_nvidia_qwen(mk: str) -> bool:
            return (
                self._reg.model_provider.get(mk, "") == "nvidia"
                and "qwen" in self._reg.model_name.get(mk, "").lower()
            )

        def _ranked(keys: list[str]) -> list[str]:
            return sorted(
                keys,
                key=lambda mk: (
                    provider_rank.get(self._reg.model_provider.get(mk, ""), 3),
                    order_idx.get(mk, 0),
                ),
            )

        self._rotation_pool = pool
        self._rotation_pinned = [mk for mk in preferred if _is_nvidia_qwen(mk)]
        preferred_rest = [mk for mk in preferred if mk not in self._rotation_pinned]
        self._rotation_preferred = _ranked(preferred_rest)
        self._rotation_fallback = _ranked(fallback)

        if pool:
            labels = [self._reg.get_label(k) for k in pool]
            logger.info("Default model rotation pool (%d models): %s", len(pool), ", ".join(labels))
            if fallback and preferred:
                logger.info(
                    "Auto mode priority: preferred=%d (gemini/qwen), fallback=%d (other)",
                    len(preferred), len(fallback),
                )
            if self._rotation_pinned:
                pinned_labels = [self._reg.get_label(k) for k in self._rotation_pinned]
                logger.info("Auto mode pinned models: %s", ", ".join(pinned_labels))
        else:
            logger.error("Default model rotation pool is empty. Check default_model_rotation settings.")

    def _default_rotation_candidates(self) -> list[str]:
        pinned = list(self._rotation_pinned)
        preferred = self._rotation_preferred
        fallback = self._rotation_fallback
        if not preferred:
            return pinned + list(fallback)
        idx = self._rotation_counter % len(preferred)
        self._rotation_counter += 1
        rotated = preferred[idx:] + preferred[:idx]
        return pinned + rotated + list(fallback)

    def get_group_fast_model_key(self, providers: set[str] | None = None) -> str:
        norm = {x.strip().lower() for x in providers if x.strip()} if providers else None

        if self._s.group_fast_model_name:
            key = self._reg.find_key_by_name(
                self._s.group_fast_model_name, self._s.group_fast_model_provider,
            )
            if key:
                pid = self._reg.model_provider.get(key, "")
                if not norm or pid in norm:
                    return key

        best_key, best_score = "", -10_000
        for mk in self._reg.order:
            pid = self._reg.model_provider.get(mk, "")
            name = self._reg.model_name.get(mk, "")
            if not pid or not name:
                continue
            if norm and pid not in norm:
                continue
            score = self._fast_score(name)
            if score > best_score:
                best_score = score
                best_key = mk
        return best_key

    def _fast_score(self, name: str) -> int:
        low = name.lower()
        score = 0
        if self._reg.is_reasoning_model(name):
            score -= 80
        for tok, w in (("flash", 80), ("lite", 60), ("mini", 50), ("nano", 40),
                       ("haiku", 35), ("turbo", 30), ("fast", 25)):
            if tok in low:
                score += w
        for tok, w in (("thinking", -80), ("reason", -70), ("r1", -60), ("pro", -25)):
            if tok in low:
                score += w
        return score

    def _inline_keys(self) -> list[str]:
        bases = tuple(
            str(b).strip().lower()
            for b in (self._s.inline_allowed_model_bases or ())
            if str(b).strip()
        )
        buckets: dict[str, list[str]] = {b: [] for b in bases}
        for mk in self._reg.order:
            name = self._reg.model_name.get(mk, "").strip().lower()
            if name.startswith("models/"):
                name = name.split("/", 1)[1]
            for b in bases:
                if name == b or name.startswith(f"{b}-"):
                    buckets[b].append(mk)
                    break
        result: list[str] = []
        seen: set[str] = set()
        for b in bases:
            for mk in buckets[b]:
                if mk not in seen and mk in self._reg.order_set:
                    seen.add(mk)
                    result.append(mk)

        # fallback: pick fastest models from any provider
        if not result:
            scored = sorted(
                self._reg.order,
                key=lambda mk: self._fast_score(self._reg.model_name.get(mk, "")),
                reverse=True,
            )
            for mk in scored[:5]:
                if mk not in seen:
                    seen.add(mk)
                    result.append(mk)
        return result

    def _ensure_rotation_pool(self) -> None:
        if self._rotation_pool or not self._s.default_model_rotation:
            return
        try:
            self.build_rotation_pool()
        except Exception as exc:
            logger.warning("Failed to build default rotation pool for inline: %s", exc)

    def _smart_keys(self) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for pid, handler in self._reg.handlers.items():
            if pid in self._s.non_reprobe:
                continue
            for name in getattr(handler, "fallback_models", []) or []:
                mk = self._reg.find_provider_model_key(pid, str(name))
                if mk and mk not in seen:
                    seen.add(mk)
                    result.append(mk)
        if self._reg.default_key and self._reg.default_key not in seen:
            result.append(self._reg.default_key)
        return result or [
            k for k in self._reg.all_order
            if self._reg.model_provider.get(k, "") not in self._s.non_reprobe
        ]

    def _provider_sem(self, pid: str) -> asyncio.Semaphore:
        key = pid or ""
        sem = self._provider_semaphores.get(key)
        if sem is None:
            limit = max(1, int(self._s.provider_max_concurrency))
            sem = asyncio.Semaphore(limit)
            self._provider_semaphores[key] = sem
        return sem

    def _resolve_image_model_key(self, model_name: str, provider_id: str = "") -> str:
        """Resolve an image model name to its registry key — more robust than find_key_by_name."""
        # 1) fast path: standard lookup
        dk = self._reg.find_key_by_name(model_name, provider_id)
        if dk:
            return dk

        # 2) brute-force: scan ALL keys for an exact model name match
        target = model_name.strip().lower()
        for mk in self._reg.all_order:
            name = self._reg.model_name.get(mk, "").strip().lower()
            pid = self._reg.model_provider.get(mk, "")
            if name == target:
                if not provider_id or pid == provider_id:
                    return mk

        # 3) substring match (e.g. "stable-diffusion-xl-base-1.0" in "@cf/stabilityai/stable-diffusion-xl-base-1.0")
        for mk in self._reg.all_order:
            name = self._reg.model_name.get(mk, "").strip().lower()
            pid = self._reg.model_provider.get(mk, "")
            if target in name or name in target:
                if not provider_id or pid == provider_id:
                    return mk

        return ""
