from __future__ import annotations

from .openai_compat import OpenAICompatProviderHandler


class NvidiaProviderHandler(OpenAICompatProviderHandler):
    def __init__(
        self,
        *,
        base_url: str,
        api_keys: list[str],
        fallback_models: list[str],
        label: str = "NVIDIA",
    ) -> None:
        super().__init__(
            provider_id="nvidia",
            label=label,
            base_url=base_url,
            keys=api_keys,
            fallback_models=fallback_models,
            discover_models=True,
            models_path="/models",
            chat_path="/chat/completions",
            auth_header="Authorization",
            auth_prefix="Bearer ",
        )
