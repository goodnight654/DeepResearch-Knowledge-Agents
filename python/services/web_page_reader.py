"""
Web page reader — fetch, clean, and chunk web pages for DeepResearch evidence.
"""

from __future__ import annotations

import asyncio
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

from config import settings


@dataclass
class WebPageChunk:
    url: str
    title: str
    content: str
    chunk_index: int
    retrieved_at: str
    metadata: dict[str, Any]


class _ReadableHTMLParser(HTMLParser):
    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "iframe"}
    BLOCK_TAGS = {"p", "div", "section", "article", "li", "br", "h1", "h2", "h3", "h4", "tr"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title = f"{self.title} {text}".strip()
        if self._skip_depth:
            return
        self._parts.append(text)

    def text(self) -> str:
        return " ".join(self._parts)


class WebPageReader:
    """Fetches web pages and converts them into compact text chunks."""

    def __init__(self) -> None:
        self.enabled = settings.web_page_fetch_enabled
        self.timeout = max(1, settings.web_page_timeout_seconds)
        self.max_chars = max(1000, settings.web_page_max_chars)
        self.chunks_per_result = max(1, settings.web_page_chunks_per_result)

    async def read(
        self,
        url: str,
        title: str = "",
        score: float = 1.0,
        max_chunks: int | None = None,
    ) -> list[WebPageChunk]:
        if not self.enabled or not url.startswith(("http://", "https://")):
            return []

        html = await self._fetch(url)
        if not html:
            return []

        page_title, text = self._extract_text(html)
        text = text[: self.max_chars]
        chunks = self._chunk_text(text, max_chunks or self.chunks_per_result)
        retrieved_at = datetime.now(timezone.utc).isoformat()
        resolved_title = title or page_title or url

        return [
            WebPageChunk(
                url=url,
                title=resolved_title,
                content=chunk,
                chunk_index=index,
                retrieved_at=retrieved_at,
                metadata={"score": score, "page_title": page_title},
            )
            for index, chunk in enumerate(chunks)
        ]

    async def _fetch(self, url: str) -> str:
        def _do_fetch() -> str:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; DeepResearchKnowledgeHub/1.0)"
                    )
                },
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return ""
                raw = response.read(self.max_chars * 4)
            return raw.decode("utf-8", errors="ignore")

        try:
            return await asyncio.to_thread(_do_fetch)
        except Exception:
            return ""

    @staticmethod
    def _extract_text(html: str) -> tuple[str, str]:
        parser = _ReadableHTMLParser()
        parser.feed(html)
        text = parser.text()
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"(\s*\n\s*)+", "\n", text)
        return parser.title.strip(), text

    @staticmethod
    def _chunk_text(text: str, max_chunks: int, chunk_size: int = 1600) -> list[str]:
        if not text:
            return []
        chunks: list[str] = []
        cursor = 0
        while cursor < len(text) and len(chunks) < max_chunks:
            end = min(cursor + chunk_size, len(text))
            boundary = text.rfind(". ", cursor, end)
            if boundary > cursor + 400:
                end = boundary + 1
            chunk = text[cursor:end].strip()
            if chunk:
                chunks.append(chunk)
            cursor = end
        return chunks
