"""PlenkaGeminiBot — clean entry point."""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from pythonjsonlogger.json import JsonFormatter

load_dotenv()


def _setup_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.INFO)


async def _main() -> None:
    from bot.config import load_settings

    settings = load_settings()
    _setup_logging(settings.log_level)

    logger = logging.getLogger("bot")
    logger.info("Starting bot...")
    provider_config_path = Path(settings.provider_config_file).expanduser()
    if not provider_config_path.exists():
        logger.warning(
            "Provider config not found: %s. Waiting for it to appear...",
            provider_config_path,
        )
        while not provider_config_path.exists():
            await asyncio.sleep(1)

    # ── imports ─────────────────────────────────────────────────────
    from telegram.ext import (
        Application,
        CallbackQueryHandler,
        ChosenInlineResultHandler,
        CommandHandler,
        InlineQueryHandler,
        MessageHandler,
        filters,
    )
    from bot.database import AllowList, Database, ModelPrefs
    from bot.services.registry import ModelRegistry
    from bot.services.probing import CapabilityCache, ProviderAvailability
    from bot.services.generation import GenerationService
    from bot.handlers.deps import Deps
    from bot.handlers import (
        allow_user, clear_memory, deny_user, help_command,
        img_command, keys_status, list_allowed, memory_command, memory_show,
        model_menu, model_search, provider_menu, start, whoami,
        on_model_callback, on_model_page, on_model_provider,
        on_model_providers, on_search_page, on_search_prompt,
        on_inline_generate, on_inline_dm, on_inline_placeholder,
        chosen_inline_handler, handle_message, inline_query_handler,
    )

    # ── services ────────────────────────────────────────────────────
    registry = ModelRegistry(settings)
    await registry.build_catalog()

    db = Database(settings)
    allowlist = AllowList(db, settings)
    prefs = ModelPrefs(settings.model_prefs_file, registry.order_set)

    capabilities = CapabilityCache(settings)
    availability = ProviderAvailability(settings, registry)
    generation = GenerationService(settings, registry, capabilities, availability)

    deps = Deps(
        settings=settings,
        registry=registry,
        db=db,
        allowlist=allowlist,
        prefs=prefs,
        capabilities=capabilities,
        availability=availability,
        generation=generation,
    )

    # ── initial probe ───────────────────────────────────────────────
    if settings.model_capability_probe_enabled:
        try:
            await generation.run_probe(force_full=False)
            generation.apply_filter()
            generation.reorder_by_latency()
            generation.build_rotation_pool()
            prefs.update_valid_keys(registry.order_set)
        except Exception as exc:
            logger.warning("Initial probe failed: %s", exc)

    # ── telegram app ────────────────────────────────────────────────
    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .concurrent_updates(settings.concurrent_updates)
        .build()
    )
    app.bot_data["deps"] = deps

    # commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("keys", keys_status))
    app.add_handler(CommandHandler("list", list_allowed))
    app.add_handler(CommandHandler("allow", allow_user))
    app.add_handler(CommandHandler("deny", deny_user))
    app.add_handler(CommandHandler("clear", clear_memory))
    app.add_handler(CommandHandler("memory", memory_command))
    app.add_handler(CommandHandler("memoryshow", memory_show))
    app.add_handler(CommandHandler("api", model_menu))
    app.add_handler(CommandHandler("provider", provider_menu))
    app.add_handler(CommandHandler("modelsearch", model_search))
    app.add_handler(CommandHandler("img", img_command))

    # callbacks
    app.add_handler(CallbackQueryHandler(on_model_callback, pattern=r"^model:"))
    app.add_handler(CallbackQueryHandler(on_model_page, pattern=r"^modelpage:"))
    app.add_handler(CallbackQueryHandler(on_model_providers, pattern=r"^modelproviders:"))
    app.add_handler(CallbackQueryHandler(on_model_provider, pattern=r"^modelprovider:"))
    app.add_handler(CallbackQueryHandler(on_search_prompt, pattern=r"^modelsearchprompt$"))
    app.add_handler(CallbackQueryHandler(on_search_page, pattern=r"^modelsearchpage:"))
    app.add_handler(CallbackQueryHandler(on_inline_generate, pattern=r"^inlinegen:"))
    app.add_handler(CallbackQueryHandler(on_inline_dm, pattern=r"^inlinedm:"))
    app.add_handler(CallbackQueryHandler(on_inline_placeholder, pattern=r"^inlineplaceholder$"))

    # messages
    app.add_handler(MessageHandler(
        filters.TEXT | filters.PHOTO | filters.Document.ALL, handle_message,
    ))

    # inline
    app.add_handler(InlineQueryHandler(inline_query_handler))
    app.add_handler(ChosenInlineResultHandler(chosen_inline_handler))

    # ── background tasks ────────────────────────────────────────────
    bg_tasks: list[asyncio.Task] = []

    async def _capability_recheck() -> None:
        await asyncio.sleep(settings.model_capability_recheck_initial_delay)
        while True:
            try:
                await generation.run_probe(force_full=settings.model_capability_recheck_force_full)
                generation.apply_filter()
                generation.reorder_by_latency()
                prefs.update_valid_keys(registry.order_set)
            except Exception as exc:
                logger.warning("Capability recheck failed: %s", exc)
            await asyncio.sleep(settings.model_capability_recheck_interval)

    async def _availability_recheck() -> None:
        await asyncio.sleep(settings.provider_availability_recheck_initial_delay)
        while True:
            try:
                await generation.reconcile_availability(force=True)
                prefs.update_valid_keys(registry.order_set)
            except Exception as exc:
                logger.warning("Availability recheck failed: %s", exc)
            await asyncio.sleep(settings.provider_availability_recheck_interval)

    # ── run ─────────────────────────────────────────────────────────
    async with app:
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)

        if settings.model_capability_recheck_enabled:
            bg_tasks.append(asyncio.create_task(_capability_recheck()))
        if settings.provider_availability_recheck_enabled:
            bg_tasks.append(asyncio.create_task(_availability_recheck()))

        logger.info("Bot is running. Press Ctrl+C to stop.")
        try:
            await asyncio.Event().wait()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            for t in bg_tasks:
                t.cancel()
            await app.updater.stop()
            await app.stop()
            logger.info("Bot stopped.")


def main() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
