# PlenkaGeminiBot

ÐŸÑ€Ð¾ÑÑ‚Ð¾Ð¹ Telegram-Ð±Ð¾Ñ‚ Ð½Ð° Python Ñ Gemini API. ÐŸÐ¾Ð´Ð´ÐµÑ€Ð¶Ð¸Ð²Ð°ÐµÑ‚:
- Ð²Ñ‹Ð±Ð¾Ñ€ Ñ‚ÐµÐºÑÑ‚Ð¾Ð²Ð¾Ð¹ Ð¼Ð¾Ð´ÐµÐ»Ð¸ Ñ‡ÐµÑ€ÐµÐ· `/model`:
  - `Gemini 2.5 Flash`
  - `Gemini 3 Flash`
  - `Qwen Next 80B`
  - `Kimi K2 Thinking`
  - `MiniMax M2.1`
- Ð´Ð¾ÑÑ‚ÑƒÐ¿ Ñ‚Ð¾Ð»ÑŒÐºÐ¾ Ð´Ð»Ñ Ð²Ð»Ð°Ð´ÐµÐ»ÑŒÑ†Ð° Ð¸ Ð´Ð¾Ð±Ð°Ð²Ð»ÐµÐ½Ð½Ñ‹Ñ… Ð¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÐµÐ»ÐµÐ¹ (ÑÐ¿Ð¸ÑÐ¾Ðº Ñ…Ñ€Ð°Ð½Ð¸Ñ‚ÑÑ Ð² `users.db`)

## Ð‘Ñ‹ÑÑ‚Ñ€Ñ‹Ð¹ ÑÑ‚Ð°Ñ€Ñ‚

1. Ð¡Ð¾Ð·Ð´Ð°Ð¹Ñ‚Ðµ Ð²Ð¸Ñ€Ñ‚ÑƒÐ°Ð»ÑŒÐ½Ð¾Ðµ Ð¾ÐºÑ€ÑƒÐ¶ÐµÐ½Ð¸Ðµ Ð¸ ÑƒÑÑ‚Ð°Ð½Ð¾Ð²Ð¸Ñ‚Ðµ Ð·Ð°Ð²Ð¸ÑÐ¸Ð¼Ð¾ÑÑ‚Ð¸.
2. Ð¡ÐºÐ¾Ð¿Ð¸Ñ€ÑƒÐ¹Ñ‚Ðµ `.env.example` Ð² `.env` Ð¸ Ð·Ð°Ð¿Ð¾Ð»Ð½Ð¸Ñ‚Ðµ Ð·Ð½Ð°Ñ‡ÐµÐ½Ð¸Ñ.
3. Ð—Ð°Ð¿ÑƒÑÑ‚Ð¸Ñ‚Ðµ Ð±Ð¾Ñ‚Ð°.

ÐšÐ¾Ð¼Ð°Ð½Ð´Ñ‹:
- `/start` â€” Ð¿Ñ€Ð¸Ð²ÐµÑ‚ÑÑ‚Ð²Ð¸Ðµ
- `/whoami` â€” ÑƒÐ·Ð½Ð°Ñ‚ÑŒ ÑÐ²Ð¾Ð¹ `user_id`
- `/allow <user_id>` â€” Ð´Ð¾Ð±Ð°Ð²Ð¸Ñ‚ÑŒ Ð¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÐµÐ»Ñ (Ñ‚Ð¾Ð»ÑŒÐºÐ¾ Ð²Ð»Ð°Ð´ÐµÐ»ÐµÑ†). Ð¢Ð°ÐºÐ¶Ðµ Ð¼Ð¾Ð¶Ð½Ð¾ Ð¾Ñ‚Ð²ÐµÑ‚Ð¸Ñ‚ÑŒ Ð½Ð° ÑÐ¾Ð¾Ð±Ñ‰ÐµÐ½Ð¸Ðµ Ð¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÐµÐ»Ñ.
- `/deny <user_id>` â€” ÑƒÐ´Ð°Ð»Ð¸Ñ‚ÑŒ Ð¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÐµÐ»Ñ (Ñ‚Ð¾Ð»ÑŒÐºÐ¾ Ð²Ð»Ð°Ð´ÐµÐ»ÐµÑ†). Ð¢Ð°ÐºÐ¶Ðµ Ð¼Ð¾Ð¶Ð½Ð¾ Ð¾Ñ‚Ð²ÐµÑ‚Ð¸Ñ‚ÑŒ Ð½Ð° ÑÐ¾Ð¾Ð±Ñ‰ÐµÐ½Ð¸Ðµ Ð¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÐµÐ»Ñ.
- `/list` â€” ÑÐ¿Ð¸ÑÐ¾Ðº Ð¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÐµÐ»ÐµÐ¹ Ñ Ð´Ð¾ÑÑ‚ÑƒÐ¿Ð¾Ð¼ (Ñ‚Ð¾Ð»ÑŒÐºÐ¾ Ð²Ð»Ð°Ð´ÐµÐ»ÐµÑ†)
- `/model` â€” Ð¾Ñ‚ÐºÑ€Ñ‹Ñ‚ÑŒ Ð¼ÐµÐ½ÑŽ Ð²Ñ‹Ð±Ð¾Ñ€Ð° Ñ‚ÐµÐºÑÑ‚Ð¾Ð²Ð¾Ð¹ Ð¼Ð¾Ð´ÐµÐ»Ð¸
- `/memory on|off|status` â€” Ð²ÐºÐ»ÑŽÑ‡Ð¸Ñ‚ÑŒ/Ð²Ñ‹ÐºÐ»ÑŽÑ‡Ð¸Ñ‚ÑŒ Ð¿Ð°Ð¼ÑÑ‚ÑŒ Ð´Ð¸Ð°Ð»Ð¾Ð³Ð° Ð´Ð»Ñ Ñ‚ÐµÐºÑƒÑ‰ÐµÐ³Ð¾ Ñ‡Ð°Ñ‚Ð°
- `/clear` â€” Ð¾Ñ‡Ð¸ÑÑ‚Ð¸Ñ‚ÑŒ Ð¸ÑÑ‚Ð¾Ñ€Ð¸ÑŽ (Ð¿Ð°Ð¼ÑÑ‚ÑŒ) Ñ‚ÐµÐºÑƒÑ‰ÐµÐ³Ð¾ Ñ‡Ð°Ñ‚Ð°

## ÐÐ°ÑÑ‚Ñ€Ð¾Ð¹ÐºÐ¸

Ð’ÑÐµ Ð½Ð°ÑÑ‚Ñ€Ð¾Ð¹ÐºÐ¸ Ñ‡ÐµÑ€ÐµÐ· Ð¿ÐµÑ€ÐµÐ¼ÐµÐ½Ð½Ñ‹Ðµ Ð¾ÐºÑ€ÑƒÐ¶ÐµÐ½Ð¸Ñ, ÑÐ¼. `.env.example`.
Ð”Ð»Ñ Ð¿Ð°Ð¼ÑÑ‚Ð¸ Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·ÑƒÐµÑ‚ÑÑ SQLite (Ð¸ÑÑ‚Ð¾Ñ€Ð¸Ñ Ñ…Ñ€Ð°Ð½Ð¸Ñ‚ÑÑ Ð² `users.db`).
Ð”Ð»Ñ Ð´Ð¾ÑÑ‚ÑƒÐ¿Ð° Ð¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÐµÐ»ÐµÐ¹ Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·ÑƒÐµÑ‚ÑÑ Ð‘Ð” `users.db` Ð¸ Ñ€ÐµÐ·ÐµÑ€Ð²Ð½Ñ‹Ð¹ Ñ„Ð°Ð¹Ð» `allowlist.json`.

## Ð—Ð°Ð¿ÑƒÑÐº

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python main.py
```

## Custom Providers

Text providers are implemented as individual handlers in `provider_handlers/`.
All text providers are loaded from `providers.json`.

Each provider supports:
- `id`, `label`
- `base_url` or `base_url_env` (for OpenAI-compatible providers)
- `models_path`, `chat_path`
- `auth_header`, `auth_prefix`
- `api_keys` (inline list of tokens in `providers.json`)
- `fallback_models`

Top-level routing option in `providers.json`:
- `model_provider_priority`: general provider retry order list under `"*"` (or directly as a list).

Fallback behavior:
- If the selected model fails, the bot retries only the same model name on other providers.
- Provider retry order uses only the general priority order.
- Model names are strict. `gemini-3.1-pro-preview` and `gemini-3.1-pro-high` are treated as different models.

## Model Capability Probe

The bot now supports startup capability probing with cache.

- Cache file: `model_capabilities.json` (configurable via `MODEL_CAPABILITIES_FILE`)
- On startup, bot probes all discovered provider/model pairs before polling starts.
- Startup is blocked until probing and filtering are complete.
- Startup probe checks selected provider/model pairs and records status:
  - `available`
  - `quota_blocked`
  - `unsupported`
  - `auth_error`
  - `transient`
- Generation routing skips cached blocking pairs while cache is valid.
- Cache is also updated from real generation errors/successes at runtime.

Environment variables:
- `MODEL_CAPABILITY_PROBE_ENABLED` (default: `1`)
- `MODEL_PROBE_SCOPE` (`smart` or `all`, default: `smart`) for non-forced probes
- `MODEL_PROBE_MAX_MODELS` (default: `0`, no limit)
- `MODEL_CAPABILITY_TTL_SECONDS` (default: `21600`)
- `MODEL_PROBE_TIMEOUT_SECONDS` (default: `8`)
- `MODEL_PROBE_WORKERS` (default: `8`)
- `MODEL_HIDE_UNAVAILABLE_MODELS` (default: `1`)

