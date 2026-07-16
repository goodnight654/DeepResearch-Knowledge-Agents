"""
Web page reader — fetch, clean, and chunk web pages for DeepResearch evidence.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import urllib.parse
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
    SKIP_TAGS = {
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "iframe",
        "nav",
        "footer",
        "form",
        "button",
    }
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
        self.block_private_networks = settings.web_page_block_private_networks

    async def read(
        self,
        url: str,
        title: str = "",
        score: float = 1.0,
        max_chunks: int | None = None,
    ) -> list[WebPageChunk]:
        if not self.enabled or not self._is_allowed_url(url, resolve_dns=False):
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
            if not self._is_allowed_url(url, resolve_dns=self.block_private_networks):
                return ""

            reader = self

            class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    if not reader._is_allowed_url(
                        urllib.parse.urljoin(req.full_url, newurl),
                        resolve_dns=reader.block_private_networks,
                    ):
                        raise ValueError("redirect target is not allowed")
                    return super().redirect_request(req, fp, code, msg, headers, newurl)

            request = urllib.request.Request(
                url,
                headers={"User-Agent": ("Mozilla/5.0 (compatible; DeepResearchKnowledgeHub/1.0)")},
            )
            opener = urllib.request.build_opener(_SafeRedirectHandler())
            with opener.open(request, timeout=self.timeout) as response:
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return ""
                raw = response.read(self.max_chars * 4)
                charset = response.headers.get_content_charset() or "utf-8"
            try:
                return raw.decode(charset, errors="replace")
            except LookupError:
                return raw.decode("utf-8", errors="replace")

        try:
            return await asyncio.to_thread(_do_fetch)
        except Exception:
            return ""

    def _is_allowed_url(self, url: str, *, resolve_dns: bool) -> bool:
        try:
            parsed = urllib.parse.urlsplit(url)
        except ValueError:
            return False
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        if parsed.username or parsed.password:
            return False
        if not self.block_private_networks:
            return True

        hostname = parsed.hostname.rstrip(".").casefold()
        if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
            return False
        try:
            direct_ip = ipaddress.ip_address(hostname)
        except ValueError:
            direct_ip = None
        if direct_ip is not None:
            return direct_ip.is_global
        if not resolve_dns:
            return True
        try:
            addresses = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        except OSError:
            return False
        resolved = {item[4][0].split("%", 1)[0] for item in addresses if item and item[4]}
        if not resolved:
            return False
        try:
            return all(ipaddress.ip_address(address).is_global for address in resolved)
        except ValueError:
            return False

    @staticmethod
    def _extract_text(html: str) -> tuple[str, str]:
        parser = _ReadableHTMLParser()
        parser.feed(html)
        text = parser.text()
        text = re.sub(r"[\t\r\f\v ]+", " ", text)
        text = re.sub(r" *(?:\n *)+", "\n", text).strip()
        return parser.title.strip(), text

    @staticmethod
    def _chunk_text(
        text: str,
        max_chunks: int,
        chunk_size: int = 1600,
        overlap: int = 120,
    ) -> list[str]:
        if not text or max_chunks <= 0 or chunk_size <= 0:
            return []
        overlap = min(max(overlap, 0), max(chunk_size - 1, 0))
        chunks: list[str] = []
        cursor = 0
        while cursor < len(text) and len(chunks) < max_chunks:
            end = min(cursor + chunk_size, len(text))
            if end < len(text):
                boundaries = [text.rfind(mark, cursor, end) for mark in ("。", "！", "？", ". ", "\n")]
                boundary = max(boundaries)
                if boundary > cursor + max(chunk_size // 3, 1):
                    end = boundary + 1
            chunk = text[cursor:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            next_cursor = end - overlap
            cursor = next_cursor if next_cursor > cursor else end
        return chunks
