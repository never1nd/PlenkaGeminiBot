from __future__ import annotations

import html
import re


def _render_inline(text: str) -> str:
    """Convert markdown inline formatting to Telegram-compatible HTML."""
    raw = str(text or "")
    tokens: list[str] = []

    def _save_code(m: re.Match[str]) -> str:
        token = f"@@CODE{len(tokens)}@@"
        tokens.append(m.group(1))
        return token

    s = re.sub(r"`([^`\n]+)`", _save_code, raw)
    s = html.escape(s)

    # links
    s = re.sub(
        r"\[([^\]\n]{1,400})\]\((https?://[^\s)]+)\)",
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        s,
    )

    # inline styles
    for pattern, tag in (
        (r"\*\*([^\n*]+?)\*\*", "b"),
        (r"(?<!\w)\*([^\n*]+?)\*(?!\w)", "b"),
        (r"__([^\n_]+?)__", "u"),
        (r"(?<!\w)_([^\n_]+?)_(?!\w)", "i"),
        (r"~~([^\n~]+?)~~", "s"),
        (r"(?<!\w)~([^\n~]+?)~(?!\w)", "s"),
        (r"\|\|([^\n|]+?)\|\|", "tg-spoiler"),
    ):
        s = re.sub(pattern, lambda m, t=tag: f"<{t}>{m.group(1)}</{t}>", s)

    # restore inline code
    for i, code_text in enumerate(tokens):
        s = s.replace(f"@@CODE{i}@@", f"<code>{html.escape(code_text)}</code>")

    return s


def markdown_to_html(text: str) -> str:
    """Convert markdown (with fenced code blocks) to Telegram HTML."""
    if not text:
        return text
    parts: list[str] = []
    last = 0
    for m in re.finditer(r"```([a-zA-Z0-9_+\-#.]*)\n([\s\S]*?)```", text):
        before = text[last:m.start()]
        if before:
            parts.append(_render_inline(before))
        lang = (m.group(1) or "").strip()
        code = html.escape((m.group(2) or "").rstrip("\n"))
        if lang:
            parts.append(f'<pre><code class="language-{html.escape(lang)}">{code}</code></pre>')
        else:
            parts.append(f"<pre>{code}</pre>")
        last = m.end()
    tail = text[last:]
    if tail:
        parts.append(_render_inline(tail))
    return "".join(parts)


def split_message(text: str, limit: int) -> list[str]:
    """Split long text at natural boundaries."""
    s = str(text or "")
    limit = max(128, limit)
    if len(s) <= limit:
        return [s]
    chunks: list[str] = []
    while len(s) > limit:
        cut = -1
        for sep in ("\n\n", "\n", " "):
            pos = s.rfind(sep, 0, limit + 1)
            if pos > limit // 3:
                cut = pos + len(sep)
                break
        if cut <= 0:
            cut = limit
        chunk = s[:cut].strip() or s[:limit]
        chunks.append(chunk)
        s = s[cut:].lstrip()
    if s:
        chunks.append(s)
    return chunks or [text[:limit]]


def trim_text(text: str, limit: int = 4000) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit - 3].rstrip() + "..."


def redact_keys(text: str) -> str:
    """Mask API keys/tokens in error messages."""
    s = text
    s = re.sub(r"(sk-[a-zA-Z0-9_-]{8})[a-zA-Z0-9_-]+", r"\1...", s)
    s = re.sub(r"(nvapi-[a-zA-Z0-9_-]{8})[a-zA-Z0-9_-]+", r"\1...", s)
    s = re.sub(r"(AIzaSy[a-zA-Z0-9_-]{8})[a-zA-Z0-9_-]+", r"\1...", s)
    s = re.sub(r"(aih_[a-zA-Z0-9_-]{8})[a-zA-Z0-9_-]+", r"\1...", s)
    s = re.sub(r"(Bearer\s+)[^\s]{20,}", r"\1[REDACTED]", s)
    return s[:500] if len(s) > 500 else s
