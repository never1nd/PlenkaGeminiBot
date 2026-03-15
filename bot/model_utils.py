from __future__ import annotations

IMAGE_MODEL_TOKENS = (
    "image", "imagen", "imagine", "midjourney", "flux",
    "recraft", "dall", "stable-diffusion", "sdxl", "kontext",
)

DEFAULT_MODEL_KEY = "__default__"
DEFAULT_MODEL_LABEL = "Default (rotation)"


def is_default_model_key(key: str) -> bool:
    return str(key) == DEFAULT_MODEL_KEY


def is_image_model_name(name: str) -> bool:
    low = name.lower()
    return any(token in low for token in IMAGE_MODEL_TOKENS)
