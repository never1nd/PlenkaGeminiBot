"""Typed dependency container — replaces the old bot_data dict service locator."""
from __future__ import annotations

from dataclasses import dataclass

from bot.config import Settings
from bot.database import AllowList, Database, ModelPrefs
from bot.services.generation import GenerationService
from bot.services.probing import CapabilityCache, ProviderAvailability
from bot.services.registry import ModelRegistry

from telegram.ext import ContextTypes


@dataclass
class Deps:
    settings: Settings
    registry: ModelRegistry
    db: Database
    allowlist: AllowList
    prefs: ModelPrefs
    capabilities: CapabilityCache
    availability: ProviderAvailability
    generation: GenerationService


def get_deps(context: ContextTypes.DEFAULT_TYPE) -> Deps:
    return context.bot_data["deps"]  # type: ignore[return-value]
