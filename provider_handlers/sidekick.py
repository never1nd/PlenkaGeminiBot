from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Callable

import requests
import websockets
from requests.adapters import HTTPAdapter

from .base import BaseProviderHandler, HistoryMessage, InputAttachment, UsageStats
from .utils import unique_keep_order

logger = logging.getLogger("bot")


class SidekickProviderHandler(BaseProviderHandler):
    def __init__(
        self,
        *,
        label: str,
        keys: list[str],
        person_id: str,
        fallback_models: list[str],
        discover_models: bool = True,
        models_url: str = "https://cube.tobit.cloud/chayns-ai-chatbot/nativeModelChatbot",
        thread_create_url: str = "https://cube.tobit.cloud/intercom-backend/v2/thread?forceCreate=true",
        ws_url: str = "wss://intercom.tobit.cloud/ws/socket.io/?EIO=4&transport=websocket",
        image_upload_url_template: str = "https://cube.tobit.cloud/image-service/v3/Images/{person_id}",
        origin: str = "https://sidekick.ki",
        referer: str = "https://sidekick.ki/",
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:147.0) Gecko/20100101 Firefox/147.0",
    ) -> None:
        super().__init__(provider_id="sidekick", label=label)
        self.keys = unique_keep_order(keys or [])
        self.person_id = str(person_id or "").strip()
        self.fallback_models = unique_keep_order(fallback_models or [])
        self.discover_models_enabled = bool(discover_models)
        self.models_url = str(models_url).strip()
        self.thread_create_url = str(thread_create_url).strip()
        self.ws_url = str(ws_url).strip()
        self.image_upload_url_template = str(image_upload_url_template).strip()
        self.origin = str(origin).strip()
        self.referer = str(referer).strip()
        self.user_agent = str(user_agent).strip() or "Mozilla/5.0"
        self.model_display_names: dict[str, str] = {}
        self._cursor = 0
        self._lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._session_local = threading.local()

    def key_count(self) -> int:
        return len(self.keys)

    def supports_input_attachments(self) -> bool:
        return True

    def get_model_display_name(self, model_name: str) -> str:
        normalized = str(model_name).strip()
        if not normalized:
            return ""
        display = str(self.model_display_names.get(normalized, normalized)).strip()
        return display or normalized

    def _rotated_keys(self) -> list[str]:
        if not self.keys:
            return []
        with self._lock:
            cursor = self._cursor % len(self.keys)
            self._cursor = (cursor + 1) % len(self.keys)
        return self.keys[cursor:] + self.keys[:cursor]

    def _build_headers(self, token: str, *, json_body: bool = True) -> dict[str, str]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Authorization": f"bearer {token}",
            "Origin": self.origin,
            "Referer": self.referer,
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _is_retryable_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        markers = (
            "timeout",
            "timed out",
            "temporar",
            "connection",
            "502",
            "503",
            "504",
            "429",
            "rate limit",
            "too many requests",
            "401",
            "403",
            "unauthorized",
            "permission denied",
        )
        return any(marker in text for marker in markers)

    def _get_http_session(self) -> requests.Session:
        session = getattr(self._session_local, "session", None)
        if isinstance(session, requests.Session):
            return session
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        self._session_local.session = session
        return session

    def get_http_session(self) -> requests.Session:
        return self._get_http_session()

    def discover_models(self, timeout_seconds: int) -> list[str]:
        self.model_display_names = {}
        if not self.discover_models_enabled:
            return list(self.fallback_models)
        if not self.models_url or not self.keys:
            return list(self.fallback_models)

        session = self._get_http_session()
        for token in self._rotated_keys():
            try:
                response = session.get(
                    self.models_url,
                    headers=self._build_headers(token),
                    timeout=timeout_seconds,
                )
                if not response.ok:
                    logger.warning(
                        "Sidekick model discovery failed: %s %s",
                        response.status_code,
                        response.text[:200],
                    )
                    continue
                payload = response.json()
                rows = payload if isinstance(payload, list) else payload.get("models", [])
                if not isinstance(rows, list):
                    continue
                model_names: list[str] = []
                model_display_names: dict[str, str] = {}
                for item in rows:
                    if not isinstance(item, dict):
                        continue
                    model_id = str(
                        item.get("personId")
                        or item.get("personID")
                        or item.get("id")
                        or ""
                    ).strip()
                    if model_id:
                        model_names.append(model_id)
                        show_name = str(item.get("showName") or item.get("name") or "").strip()
                        developer = str(item.get("developer") or item.get("provider") or "").strip()
                        display_name = show_name
                        if developer:
                            display_name = (
                                f"{display_name} ({developer})" if display_name else developer
                            )
                        if not display_name:
                            display_name = model_id
                        model_display_names[model_id] = display_name
                if model_names:
                    unique_models = unique_keep_order(model_names)
                    self.model_display_names = {
                        model_id: str(model_display_names.get(model_id, model_id)).strip() or model_id
                        for model_id in unique_models
                    }
                    return unique_models
            except Exception as exc:
                logger.warning("Sidekick model discovery request failed: %s", exc)
                continue
        return list(self.fallback_models)

    def _build_prompt_text(
        self,
        prompt: str,
        history: list[HistoryMessage],
        attachments: list[InputAttachment],
    ) -> str:
        if history:
            lines = ["Conversation history:"]
            for msg in history:
                role = str(msg.get("role", "user")).strip()
                content = str(msg.get("content", "")).strip()
                if not content:
                    continue
                if role not in {"user", "assistant"}:
                    role = "user"
                lines.append(f"{role}: {content}")
            lines.append(f"user: {prompt}")
            lines.append("assistant:")
            base = "\n".join(lines)
        else:
            base = prompt

        file_notes: list[str] = []
        for item in attachments:
            if not isinstance(item, dict):
                continue
            if str(item.get("kind", "")).strip().lower() != "file":
                continue
            text_payload = str(item.get("text", "")).strip()
            if not text_payload:
                continue
            file_name = str(item.get("file_name", "attachment")).strip() or "attachment"
            mime_type = str(item.get("mime_type", "application/octet-stream")).strip()
            file_notes.append(f"Attached file ({file_name}, {mime_type}):\n{text_payload}")
        if file_notes:
            return f"{base}\n\n" + "\n\n".join(file_notes)
        return base

    def _upload_image(self, token: str, item: InputAttachment, timeout_seconds: int) -> str:
        if not self.person_id:
            raise RuntimeError("Sidekick provider person_id is missing.")
        image_url = self.image_upload_url_template.format(person_id=self.person_id)
        raw_bytes = item.get("bytes", b"")
        if isinstance(raw_bytes, bytearray):
            raw_bytes = bytes(raw_bytes)
        if not isinstance(raw_bytes, bytes) or not raw_bytes:
            raise RuntimeError("Sidekick image attachment payload is empty.")
        file_name = str(item.get("file_name", "image.jpg")).strip() or "image.jpg"
        mime_type = str(item.get("mime_type", "image/jpeg")).strip() or "image/jpeg"
        files = {"File": (file_name, raw_bytes, mime_type)}
        headers = self._build_headers(token, json_body=False)
        response = self._get_http_session().post(image_url, headers=headers, files=files, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        base_domain = str(payload.get("baseDomain", "https://tsimg.cloud/")).strip()
        image_obj = payload.get("image", {}) if isinstance(payload, dict) else {}
        path = str(image_obj.get("path", "")).strip() if isinstance(image_obj, dict) else ""
        if not base_domain or not path:
            raise RuntimeError("Sidekick image upload returned invalid payload.")
        return f"{base_domain}{path}"

    async def _wait_for_reply(
        self,
        token: str,
        thread_id: str,
        ai_member_id: str,
        timeout_seconds: int,
    ) -> str:
        auth_payload = ["authenticate", {"accessToken": token, "threadTypeFilter": [8, 10, 15]}]
        join_payload = ["joinThread", {"threadId": thread_id}]
        joined = False
        answer_chunks: list[str] = []
        deadline = time.monotonic() + max(5, timeout_seconds)

        async with websockets.connect(
            self.ws_url,
            additional_headers={"User-Agent": self.user_agent},
            open_timeout=min(20, max(5, timeout_seconds)),
            close_timeout=5,
        ) as ws:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("timeout waiting for sidekick response")
                raw_msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
                if raw_msg == "2":
                    await ws.send("3")
                    continue
                if raw_msg.startswith("0"):
                    await ws.send("40")
                    continue
                if raw_msg.startswith("40"):
                    await ws.send(f"42{json.dumps(auth_payload, separators=(',', ':'))}")
                    continue
                if '42["authenticate"' in raw_msg and not joined:
                    await ws.send(f"42{json.dumps(join_payload, separators=(',', ':'))}")
                    joined = True
                    continue
                if not raw_msg.startswith("42"):
                    continue
                try:
                    payload = json.loads(raw_msg[2:])
                except Exception:
                    continue
                if not isinstance(payload, list) or len(payload) < 2:
                    continue
                event = str(payload[0]).strip()
                data = payload[1]

                if event == "typingMessage" and isinstance(data, dict):
                    member_id = str(data.get("memberId", "")).strip()
                    if ai_member_id and member_id and member_id != ai_member_id:
                        continue
                    type_id = int(data.get("typeId", 0) or 0)
                    if type_id != 1:
                        continue
                    chunk = str(data.get("messageChunk", "") or "")
                    if chunk:
                        answer_chunks.append(chunk)
                    if bool(data.get("isLastChunk", False)):
                        text = "".join(answer_chunks).strip()
                        if text:
                            return text
                    continue

                if event == "addMessage" and isinstance(data, dict):
                    message_obj = data.get("message", {})
                    if not isinstance(message_obj, dict):
                        continue
                    author = message_obj.get("author", {})
                    author_id = str(author.get("id", "")).strip() if isinstance(author, dict) else ""
                    if ai_member_id and author_id and author_id != ai_member_id:
                        continue
                    type_id = int(message_obj.get("typeId", 0) or 0)
                    text = str(message_obj.get("text", "") or "").strip()
                    if type_id == 1 and text:
                        return text

        raise RuntimeError("sidekick websocket ended without response")

    async def _generate_text_async(
        self,
        token: str,
        prompt: str,
        model_name: str,
        history: list[HistoryMessage],
        attachments: list[InputAttachment],
        timeout_seconds: int,
    ) -> str:
        if not self.person_id:
            raise RuntimeError("Sidekick provider requires person_id.")
        if not self.thread_create_url:
            raise RuntimeError("Sidekick thread_create_url is missing.")
        if not self.ws_url:
            raise RuntimeError("Sidekick ws_url is missing.")

        message_text = self._build_prompt_text(prompt, history, attachments)
        image_urls: list[str] = []
        for item in attachments:
            if not isinstance(item, dict):
                continue
            if str(item.get("kind", "")).strip().lower() != "image":
                continue
            image_urls.append(self._upload_image(token, item, timeout_seconds))

        message_payload: dict[str, object] = {"text": message_text}
        if image_urls:
            message_payload["images"] = [{"url": url} for url in image_urls]

        payload = {
            "members": [
                {"isAdmin": True, "personId": self.person_id},
                {"personId": model_name},
            ],
            "nerMode": "None",
            "priority": 0,
            "typeId": 8,
            "messages": [message_payload],
        }
        response = self._get_http_session().post(
            self.thread_create_url,
            json=payload,
            headers=self._build_headers(token),
            timeout=timeout_seconds,
        )
        if not response.ok:
            raise RuntimeError(f"{response.status_code} {response.text[:300]}")
        data = response.json()
        thread_id = str(data.get("id", "")).strip()
        if not thread_id:
            raise RuntimeError("Sidekick did not return thread id.")
        ai_member_id = ""
        members = data.get("members", []) if isinstance(data, dict) else []
        if isinstance(members, list):
            for member in members:
                if not isinstance(member, dict):
                    continue
                if bool(member.get("isAgent")):
                    ai_member_id = str(member.get("id", "")).strip()
                    if ai_member_id:
                        break
        return await self._wait_for_reply(token, thread_id, ai_member_id, timeout_seconds)

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
        _ = max_output_tokens
        # Sidekick backend is not concurrency-safe for parallel prompt requests.
        acquired = self._request_lock.acquire(blocking=False)
        if not acquired:
            logger.info("Sidekick request queued: waiting for active Sidekick prompt to finish.")
            self._request_lock.acquire()
        try:
            keys = self._rotated_keys()
            if not keys:
                raise RuntimeError("No API keys configured for provider: sidekick")
            if not self.person_id:
                raise RuntimeError("Provider 'sidekick' person_id is missing.")

            last_error = ""
            normalized_attachments = attachments or []
            for token in keys:
                try:
                    text = asyncio.run(
                        self._generate_text_async(
                            token,
                            prompt,
                            model_name,
                            history,
                            normalized_attachments,
                            timeout_seconds,
                        )
                    )
                    text = strip_reasoning(text)
                    if not text:
                        raise RuntimeError(f"{model_name} returned an empty response.")
                    return text, {}
                except Exception as exc:
                    last_error = str(exc)
                    if not self._is_retryable_error(exc):
                        break
                    continue
            raise RuntimeError(f"Provider 'sidekick' API error: {last_error}")
        finally:
            self._request_lock.release()
