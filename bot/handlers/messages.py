from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any

from telegram import InlineQueryResultArticle, InputTextMessageContent, Update
from telegram.ext import ContextTypes

from bot.handlers.deps import get_deps
from bot.handlers.guards import is_group
from bot.handlers.helpers import (
    edit_inline, inline_placeholder_keyboard,
    extract_attachments, search_models, send_text, display_model_label,
    format_user_label,
)
from bot.model_utils import DEFAULT_MODEL_KEY
from bot.formatting import trim_text

logger = logging.getLogger("bot")
_compacting: set[int] = set()



# ── message handler ────────────────────────────────────────────────

def _build_prompt(prompt: str, instruction: str) -> str:
    base = prompt.strip() or "Help the user."
    return f"{base}\n\n{instruction}" if instruction else base


def _build_summary_prompt(existing: str, messages: list[dict[str, str]]) -> str:
    def _is_noise(text: str) -> bool:
        s = text.strip()
        if not s:
            return True
        # Drop ultra-short spam like "s", "ss", "ok"
        alnum = "".join(ch for ch in s if ch.isalnum())
        if len(alnum) < 3:
            return True
        # Drop repeated single-character spam like "aaaaaa"
        if len(set(alnum.lower())) <= 2 and len(alnum) >= 6:
            return True
        return False

    lines: list[str] = []
    if existing:
        lines.append("Existing memory:")
        lines.append(existing.strip())
    lines.append("New conversation messages:")
    seen: set[str] = set()
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content or _is_noise(content):
            continue
        norm = f"{role}:{content}".lower()
        if norm in seen:
            continue
        seen.add(norm)
        lines.append(f"{role}: {content}")
    lines.append(
        "Task: Compact the memory. Do NOT analyze or interpret. "
        "Do NOT list every message. Ignore spam/short repeats. "
        "Only keep explicit facts, preferences, decisions, names, and tasks. "
        "If there is no durable info, return exactly: (no persistent memory). "
        "Do not add new information. Use short bullet points. "
        "Keep it under 1200 characters and in the same language as the conversation."
    )
    return "\n".join(lines)


async def _maybe_compact_history(d, chat_id: int) -> None:
    if chat_id in _compacting:
        return
    _compacting.add(chat_id)
    try:
        total = d.db.history_count(chat_id)
        every = int(d.settings.memory_compact_every or 0)
        if every <= 0 or total < every or (total % every) != 0:
            return
        history = d.db.all_history(chat_id)
        if not history:
            return
        existing = d.db.get_summary(chat_id)
        prompt = _build_summary_prompt(existing, history)
        summary, _, _, _ = await d.generation.generate_text(
            prompt, DEFAULT_MODEL_KEY, history=[], attachments=None,
        )
        summary = trim_text(summary, 1200).strip()

        def _fallback_compact(rows: list[dict[str, str]]) -> str:
            counts: dict[tuple[str, str], int] = {}
            order: list[tuple[str, str]] = []
            for msg in rows:
                role = str(msg.get("role", "user")).strip() or "user"
                content = " ".join(str(msg.get("content", "")).split()).strip()
                if not content:
                    continue
                key = (role, content)
                if key not in counts:
                    order.append(key)
                    counts[key] = 1
                else:
                    counts[key] += 1
            if not order:
                return ""
            lines: list[str] = []
            total_len = 0
            for role, content in order:
                count = counts.get((role, content), 1)
                suffix = f" (x{count})" if count > 1 else ""
                line = f"- {role}: {content}{suffix}"
                if total_len + len(line) + 1 > 1100:
                    break
                lines.append(line)
                total_len += len(line) + 1
            return "\n".join(lines).strip()

        if not summary or summary.lower() in ("(no persistent memory)", "(no persistent memory)."):
            summary = _fallback_compact(history) or "(no persistent memory)"

        d.db.set_summary(chat_id, summary)
        d.db.clear_history(chat_id, clear_summary=False)
        logger.info("Memory compacted for chat %s.", chat_id)
    except Exception as exc:
        logger.warning("Memory compaction failed for chat %s: %s", chat_id, exc)
    finally:
        _compacting.discard(chat_id)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return
    d = get_deps(context)
    s = d.settings
    user = update.effective_user
    chat = update.effective_chat

    prompt = str(message.text or message.caption or "").strip()
    group = is_group(chat)

    # group triggering
    if group:
        keyword = _keyword_trigger(s, prompt)
        reply_to_bot = _reply_to_bot(message, context)
        if not keyword and not reply_to_bot:
            return
    else:
        if not user or not d.allowlist.is_allowed(user.id):
            await message.reply_text("Access denied.")
            return

    # attachments
    attachments, notices = await extract_attachments(message, s)
    had_media = bool(message.photo or message.document)
    had_photo = bool(message.photo)
    if had_media and notices:
        logger.info("Attachment notices: %s", "; ".join(notices))
    if had_media and not attachments:
        reason = notices[0] if notices else "Unsupported attachment type."
        await message.reply_text(f"Could not use attachment: {reason}")
        return

    if not prompt:
        if attachments:
            prompt = "Analyze the attached content and help the user."
        else:
            if not group:
                await message.reply_text("Empty message.")
            return

    # model search mode
    if not group and context.user_data.get("model_search_waiting"):
        if attachments:
            await message.reply_text("Model search accepts text only.")
            return
        context.user_data["model_search_waiting"] = False
        context.user_data["model_search_query"] = prompt
        model_key = d.prefs.get(user.id, d.registry.default_key) if user else d.registry.default_key
        keys = search_models(d.registry, prompt)
        from bot.handlers.callbacks import search_keyboard
        await message.reply_text(
            f"Current model: {display_model_label(d.registry, model_key)}\nSearch: {prompt}\nMatches: {len(keys)}",
            reply_markup=search_keyboard(d.registry, model_key, prompt, 0),
        )
        return

    if not user or not chat:
        return

    # text generation
    try:
        model_key = d.prefs.get(user.id, d.registry.default_key)

        if group and s.group_fast_model_name:
            fast = d.generation.get_group_fast_model_key(s.group_providers or None)
            if fast:
                model_key = fast

        short = group
        instruction = s.reply_instruction_short if short else s.reply_instruction_default
        summary = d.db.get_summary(chat.id)
        user_label = format_user_label(user)
        group_context = "Context: Group chat\n" if group else ""
        base_prompt = f"{group_context}User: {user_label}\nUser message:\n{prompt}"
        if summary:
            base_prompt = (
                f"Conversation summary:\n{summary}\n\n"
                f"{group_context}User: {user_label}\nUser message:\n{prompt}"
            )
        full_prompt = _build_prompt(base_prompt, instruction)
        allowlist = s.group_providers if group else None

        # history
        history: list[dict[str, str]] = []
        if d.db.is_memory_enabled(chat.id, user.id):
            history = d.db.recent_history(chat.id, s.memory_context_messages)

        if group:
            reply = getattr(message, "reply_to_message", None)
            if reply:
                rt = str(getattr(reply, "text", "") or getattr(reply, "caption", "") or "").strip()
                if rt:
                    ru = getattr(reply, "from_user", None)
                    role = "assistant" if ru and getattr(ru, "is_bot", False) else "user"
                    history.append({"role": role, "content": rt})

        await message.chat.send_action(action="typing")
        answer, used_model, used_provider, usage = await d.generation.generate_text(
            full_prompt, model_key, history,
            attachments or None,
            provider_allowlist=allowlist,
        )

        logger.info(
            "Usage: user=%s chat=%s provider=%s model=%s tokens=%s",
            user.id, chat.id, used_provider, used_model, usage.get("total_tokens", "n/a"),
        )

        if d.db.is_memory_enabled(chat.id, user.id):
            if not had_photo:
                d.db.save_messages(chat.id, user.id, [("user", prompt), ("assistant", answer)])
                context.application.create_task(_maybe_compact_history(d, chat.id))

        await send_text(message, answer, s.telegram_reply_chunk_chars)
    except Exception as exc:
        logger.exception("Generation failed")
        await message.reply_text(f"Generation error: {exc}")


# ── inline query ────────────────────────────────────────────────────

async def inline_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    iq = update.inline_query
    if not iq:
        return
    d = get_deps(context)
    user = iq.from_user

    if not user or not d.allowlist.is_allowed(user.id):
        await iq.answer([], cache_time=1, is_personal=True)
        return

    prompt = str(iq.query or "").strip()
    seed = f"{user.id}:{prompt}:{time.time_ns()}"
    result_id = hashlib.sha1(seed.encode()).hexdigest()[:32]

    reply_markup = inline_placeholder_keyboard()

    result = InlineQueryResultArticle(
        id=result_id,
        title=prompt[:72] if prompt else "Generate response",
        description="Generate response",
        input_message_content=InputTextMessageContent(d.settings.inline_placeholder_text),
        reply_markup=reply_markup,
    )
    await iq.answer([result], cache_time=1, is_personal=True)


async def chosen_inline_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chosen = update.chosen_inline_result
    if not chosen or not chosen.inline_message_id:
        return
    d = get_deps(context)
    user = chosen.from_user

    if not user or not d.allowlist.is_allowed(user.id):
        await edit_inline(
            context.bot, chosen.inline_message_id, "Access denied.", reply_markup=None,
        )
        return

    inline_id = chosen.inline_message_id
    prompt = str(chosen.query or "").strip() or "Help the user."
    logger.info("Inline chosen: user=%s result_id=%s", user.id, getattr(chosen, "result_id", ""))

    async def _run() -> None:
        try:
            user_label = format_user_label(user)
            full_base = f"User: {user_label}\nUser message:\n{prompt}"
            full = _build_prompt(full_base, d.settings.reply_instruction_short)
            answer, used_model, used_provider, usage = await d.generation.generate_inline_text(full)
            logger.info(
                "Inline: user=%s provider=%s model=%s tokens=%s",
                user.id, used_provider, used_model, usage.get("total_tokens", "n/a"),
            )
            formatted = f'"{prompt}"\n\n---\n\n"{answer}"'
            await edit_inline(
                context.bot, inline_id, formatted,
                reply_markup=None,
            )
        except Exception as exc:
            logger.exception("Inline generation failed")
            await edit_inline(
                context.bot, inline_id, f"Error: {exc}",
                reply_markup=None,
            )

    context.application.create_task(_run())


# ── helpers ─────────────────────────────────────────────────────────

def _keyword_trigger(s: Any, prompt: str) -> bool:
    if not s.group_keyword_trigger:
        return False
    if s.group_keyword_trigger.casefold() not in prompt.casefold():
        return False
    if s.group_keyword_require_question and "?" not in prompt:
        return False
    return True


def _reply_to_bot(message: Any, context: ContextTypes.DEFAULT_TYPE) -> bool:
    reply = getattr(message, "reply_to_message", None)
    if not reply:
        return False
    ru = getattr(reply, "from_user", None)
    if not ru:
        return False
    bot_id = getattr(context.bot, "id", None)
    if bot_id is None:
        return bool(getattr(ru, "is_bot", False))
    return ru.id == bot_id


