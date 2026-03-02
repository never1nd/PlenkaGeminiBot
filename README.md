# PlenkaGeminiBot

Telegram bot with multi-provider text generation and per-user API/model selection.

## Features

- Access control: owner plus allowlist stored in `users.db` (backup in `allowlist.json`)
- Per-user API/model selection with `/api` (alias: `/model`)
- Provider browser with `/provider`
- Model search with `/modelsearch <text>`
- Per-user chat memory controls: `/memory on|off|status` and `/clear`
- User attachments in text mode: send photo or file (with optional caption) to models/providers that support attachment input
  - If current model cannot process attachment, bot shows action buttons:
    - proceed without attachment
    - proceed with auto-model selection
    - choose another model
- Bot asks models to reply using Telegram Markdown-style formatting and renders formatted output for Telegram
- When auto-model is used for file attachments, reply includes `provider/model` label at the bottom
- Multi-provider fallback routing:
  - selected API/model and provider first
  - then other models from selected provider
  - then same model name on other providers
  - then all remaining available models
- Startup model capability probe with cache in `model_capabilities.json`

## Commands

- `/start`
- `/whoami`
- `/api`
- `/provider`
- `/model`
- `/modelsearch <text>`
- `/memory on|off|status` (per-user in current chat)
- `/clear`
- send photo/file in chat (text mode)
- `/allow <user_id>` (owner only)
- `/deny <user_id>` (owner only)
- `/list` (owner only)
- `/keys` (owner only)
- `/help` (owner only)

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# create .env manually
python main.py
```

Create `.env` manually and set required variables (at minimum `TELEGRAM_BOT_TOKEN` and `OWNER_USER_ID`).

## Provider Config (`providers.json`)

Each provider entry supports:

- `id`, `label`
- `base_url` or `base_url_env` (for OpenAI-compatible providers)
- `models_path`, `chat_path`
- `auth_header`, `auth_prefix`
- `api_keys` (or `api_key`)
- `discover_models`
- `fallback_models`

Sidekick-specific fields:

- `person_id` (required for `id: "sidekick"`)
- `models_url` (default: `https://cube.tobit.cloud/chayns-ai-chatbot/nativeModelChatbot`)
- `thread_create_url` (default: `https://cube.tobit.cloud/intercom-backend/v2/thread?forceCreate=true`)
- `ws_url` (default: `wss://intercom.tobit.cloud/ws/socket.io/?EIO=4&transport=websocket`)
- `image_upload_url_template` (default: `https://cube.tobit.cloud/image-service/v3/Images/{person_id}`)
- `origin`, `referer`, `user_agent` (optional headers override)

Top-level config:

- No top-level routing priority field is used; provider/model routing is automatic.

## Model Capability Probe

Cache file: `model_capabilities.json` (configurable by `MODEL_CAPABILITIES_FILE`).

Statuses:

- `available`
- `quota_blocked`
- `unsupported`
- `auth_error`
- `transient`
- `unknown`

Generation skips cached blocking statuses while cache is valid.

## Important Environment Variables

- `TELEGRAM_BOT_TOKEN`
- `OWNER_USER_ID`
- `ALLOWLIST_FILE` (default: `allowlist.json`)
- `USERS_DB_FILE` (default: `users.db`)
- `MODEL_PREFS_FILE` (default: `model_prefs.json`)
- `CUSTOM_PROVIDER_CONFIG_FILE` (default: `providers.json`)
- `MODEL_CAPABILITIES_FILE` (default: `model_capabilities.json`)
- `MEMORY_CONTEXT_MESSAGES` (default: `20`)
- `MAX_OUTPUT_TOKENS` (default: `1024`)
- `MAX_INPUT_ATTACHMENT_COUNT` (default: `3`)
- `MAX_INPUT_ATTACHMENT_BYTES` (default: `10485760`)
- `MAX_INPUT_TEXT_ATTACHMENT_BYTES` (default: `524288`)
- `MAX_INPUT_TEXT_ATTACHMENT_CHARS` (default: `20000`)
- `TELEGRAM_REPLY_CHUNK_CHARS` (default: `3200`, safe chunk size for long replies)
- `REGULAR_MODEL_TIMEOUT_SECONDS` (default: `90`)
- `REASONING_MODEL_TIMEOUT_SECONDS` (default: `180`)
- `FALLBACK_ATTEMPT_TIMEOUT_SECONDS` (default: `35`)
- `MODEL_CAPABILITY_PROBE_ENABLED` (default: `1`)
- `MODEL_PROBE_SCOPE` (`smart` or `all`, default: `smart`)
- `MODEL_PROBE_MAX_MODELS` (default: `0`, no limit)
- `MODEL_CAPABILITY_TTL_SECONDS` (default: `21600`)
- `MODEL_PROBE_TIMEOUT_MIN_SECONDS` (default: `30`)
- `MODEL_PROBE_TIMEOUT_SECONDS` (default: `0`, auto mode)
- `MODEL_PROBE_WORKERS` (default: `8`)
- `MODEL_HIDE_UNAVAILABLE_MODELS` (default: `1`)
- `NON_REPROBE_PROVIDERS` (comma-separated provider ids excluded from provider/model reprobes, default: `sidekick`)
- `LOG_LEVEL` (default: `INFO`)
