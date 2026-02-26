from __future__ import annotations

from .openai_compat import OpenAICompatProviderHandler


class OrbitProviderHandler(OpenAICompatProviderHandler):
    def __init__(
        self,
        *,
        label: str,
        base_url: str,
        keys: list[str],
        fallback_models: list[str],
        discover_models: bool = True,
        models_path: str = "/models",
        chat_path: str = "/v1/chat/completions",
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer ",
    ) -> None:
        super().__init__(
            provider_id="orbit",
            label=label,
            base_url=base_url,
            keys=keys,
            fallback_models=fallback_models,
            discover_models=discover_models,
            models_path=models_path,
            chat_path=chat_path,
            auth_header=auth_header,
            auth_prefix=auth_prefix,
        )
