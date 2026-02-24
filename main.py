import os
import logging
import base64
import json
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

import google.generativeai as genai

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DEFAULT_OWNER_USER_ID = 8082486311
DEFAULT_OWNER_USERNAME = "s0ryanix"

OWNER_USER_ID = int(os.getenv("OWNER_USER_ID", str(DEFAULT_OWNER_USER_ID)) or str(DEFAULT_OWNER_USER_ID))
OWNER_USERNAME = os.getenv("OWNER_USERNAME", DEFAULT_OWNER_USERNAME).lstrip("@").strip()

ALLOWED_USER_IDS = os.getenv("ALLOWED_USER_IDS", "").strip()
ALLOWLIST_FILE = os.getenv("ALLOWLIST_FILE", "allowlist.json").strip()
TEXT_MODELS = os.getenv("TEXT_MODELS", "gemini-2.5-pro,gemini-2.5-flash,gemini-2.0-flash").strip()
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-2.0-flash-preview-image-generation").strip()

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7") or "0.7")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024") or "1024")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
logger = logging.getLogger("bot")

if not TELEGRAM_BOT_TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN is required")
if not GEMINI_API_KEY:
    raise SystemExit("GEMINI_API_KEY is required")

allowlist_path = Path(ALLOWLIST_FILE)


def load_allowlist() -> set[int]:
    ids: set[int] = set()
    if ALLOWED_USER_IDS:
        for raw in ALLOWED_USER_IDS.split(","):
            raw = raw.strip()
            if raw:
                try:
                    ids.add(int(raw))
                except ValueError:
                    logger.warning("Skipping invalid user id in ALLOWED_USER_IDS: %s", raw)

    if allowlist_path.exists():
        try:
            data = json.loads(allowlist_path.read_text(encoding="utf-8"))
            for raw in data.get("user_ids", []):
                try:
                    ids.add(int(raw))
                except ValueError:
                    logger.warning("Skipping invalid user id in allowlist file: %s", raw)
        except Exception as exc:
            logger.warning("Failed to read allowlist file: %s", exc)

    if OWNER_USER_ID:
        ids.add(OWNER_USER_ID)

    return ids


def save_allowlist(ids: set[int]) -> None:
    data = {"user_ids": sorted(ids)}
    allowlist_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


allowed_user_ids = load_allowlist()

text_models: List[str] = [m.strip() for m in TEXT_MODELS.split(",") if m.strip()]

if not text_models:
    raise SystemExit("TEXT_MODELS is empty. Provide at least one model name.")

if not IMAGE_MODEL:
    raise SystemExit("IMAGE_MODEL is empty. Provide an image-capable model name.")

genai.configure(api_key=GEMINI_API_KEY)


def is_allowed(user_id: Optional[int]) -> bool:
    if user_id is None:
        return False
    return user_id in allowed_user_ids


def is_owner(user) -> bool:
    if not user:
        return False
    if OWNER_USER_ID and user.id == OWNER_USER_ID:
        return True
    if OWNER_USERNAME and user.username and user.username.lower() == OWNER_USERNAME.lower():
        return True
    return False


def parse_user_id(arg: str) -> Optional[int]:
    arg = arg.strip()
    if not arg:
        return None
    try:
        return int(arg)
    except ValueError:
        return None


def extract_text(response) -> str:
    if not response or not getattr(response, "candidates", None):
        return ""
    content = response.candidates[0].content
    if not content:
        return ""
    parts = getattr(content, "parts", []) or []
    texts = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


def extract_image_bytes(response) -> Optional[Tuple[bytes, str]]:
    if not response or not getattr(response, "candidates", None):
        return None
    content = response.candidates[0].content
    if not content:
        return None
    parts = getattr(content, "parts", []) or []
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            mime = getattr(inline, "mime_type", "image/png") or "image/png"
            data = base64.b64decode(inline.data)
            return data, mime
    return None


def generate_text_with_fallback(prompt: str) -> Tuple[str, str]:
    last_error = None
    for model_name in text_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                },
            )
            text = extract_text(response)
            if text:
                return text, model_name
        except Exception as exc:
            last_error = exc
            logger.warning("Model %s failed: %s", model_name, exc)
            continue

    raise RuntimeError(f"All models failed. Last error: {last_error}")


def generate_image(prompt: str) -> Tuple[bytes, str]:
    model = genai.GenerativeModel(IMAGE_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
    )
    image = extract_image_bytes(response)
    if image:
        return image

    text = extract_text(response)
    raise RuntimeError(f"Image model returned no image. Text: {text}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user.id if update.effective_user else None):
        await update.message.reply_text("Доступ запрещен. Попросите владельца добавить вас.")
        return
    await update.message.reply_text(
        "Готов. Отправьте текст или используйте /img <описание> для картинки."
    )


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user:
        await update.message.reply_text("Не удалось определить пользователя.")
        return
    await update.message.reply_text(f"Ваш user_id: {user.id}")


async def allow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_owner(user):
        await update.message.reply_text("Команда доступна только владельцу.")
        return
    if update.effective_chat and update.effective_chat.type != "private":
        await update.message.reply_text("Используйте команду в личных сообщениях с ботом.")
        return

    target_user = None
    if update.message and update.message.reply_to_message:
        target_user = update.message.reply_to_message.from_user

    if target_user:
        uid = target_user.id
    else:
        if not context.args:
            await update.message.reply_text(
                "Использование: /allow <user_id> или ответьте на сообщение пользователя."
            )
            return
        if context.args[0].startswith("@"):
            await update.message.reply_text(
                "Нельзя добавить по @username. Попросите пользователя написать боту, "
                "затем ответьте на его сообщение или используйте /whoami."
            )
            return
        uid = parse_user_id(context.args[0])
        if uid is None:
            await update.message.reply_text("Некорректный user_id.")
            return
    allowed_user_ids.add(uid)
    save_allowlist(allowed_user_ids)
    await update.message.reply_text(f"Добавлен user_id {uid}.")


async def deny(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_owner(user):
        await update.message.reply_text("Команда доступна только владельцу.")
        return
    if update.effective_chat and update.effective_chat.type != "private":
        await update.message.reply_text("Используйте команду в личных сообщениях с ботом.")
        return

    target_user = None
    if update.message and update.message.reply_to_message:
        target_user = update.message.reply_to_message.from_user

    if target_user:
        uid = target_user.id
    else:
        if not context.args:
            await update.message.reply_text(
                "Использование: /deny <user_id> или ответьте на сообщение пользователя."
            )
            return
        if context.args[0].startswith("@"):
            await update.message.reply_text(
                "Нельзя удалить по @username. Используйте user_id или ответьте на сообщение."
            )
            return
        uid = parse_user_id(context.args[0])
        if uid is None:
            await update.message.reply_text("Некорректный user_id.")
            return
    if uid == OWNER_USER_ID:
        await update.message.reply_text("Нельзя удалить владельца.")
        return
    allowed_user_ids.discard(uid)
    save_allowlist(allowed_user_ids)
    await update.message.reply_text(f"Удален user_id {uid}.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id if update.effective_user else None
    if not is_allowed(user_id):
        await update.message.reply_text("Доступ запрещен.")
        return
    prompt = update.message.text.strip() if update.message and update.message.text else ""
    if not prompt:
        await update.message.reply_text("Пустое сообщение.")
        return
    await update.message.chat.send_action(action="typing")
    try:
        answer, used_model = generate_text_with_fallback(prompt)
        await update.message.reply_text(f"{answer}\n\n[model: {used_model}]")
    except Exception as exc:
        logger.exception("Text generation failed")
        await update.message.reply_text(f"Ошибка генерации: {exc}")


async def handle_img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id if update.effective_user else None
    if not is_allowed(user_id):
        await update.message.reply_text("Доступ запрещен.")
        return
    if not context.args:
        await update.message.reply_text("Использование: /img <описание>")
        return
    prompt = " ".join(context.args).strip()
    if not prompt:
        await update.message.reply_text("Пустое описание.")
        return
    await update.message.chat.send_action(action="upload_photo")
    try:
        image_bytes, mime = generate_image(prompt)
        filename = "image.png" if mime == "image/png" else "image.jpg"
        await update.message.reply_photo(photo=image_bytes, filename=filename, caption="Готово.")
    except Exception as exc:
        logger.exception("Image generation failed")
        await update.message.reply_text(f"Ошибка генерации: {exc}")


def build_app():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("allow", allow))
    app.add_handler(CommandHandler("deny", deny))
    app.add_handler(CommandHandler("img", handle_img))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    return app


def main() -> None:
    app = build_app()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
