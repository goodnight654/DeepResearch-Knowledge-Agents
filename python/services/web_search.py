"""
Web Search 服务 — DeepResearch 可选外部搜索能力

支持 provider:
  - disabled
  - tavily
  - serpapi
  - duckduckgo (instant answer)
"""

from __future__ import annotations

import asyncio
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from config import settings
from utils.json_utils import coerce_float


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    score: float
    provider: str


class _DuckDuckGoHTMLParser(HTMLParser):
    """Minimal parser for DuckDuckGo's no-JavaScript results page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        classes = set((attr_map.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self._current = {"title": "", "url": attr_map.get("href") or "", "snippet": ""}
            self.results.append(self._current)
            self._capture = "title"
        elif self._current is not None and "result__snippet" in classes:
            self._capture = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag in {"a", "div", "span"}:
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._current is None or self._capture is None:
            return
        value = " ".join(data.split())
        if value:
            current = self._current[self._capture]
            self._current[self._capture] = f"{current} {value}".strip()


class WebSearchService:
    """统一 Web 搜索接口。"""

    TAVILY_ENDPOINT = "https://api.tavily.com/search"
    SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
    DUCKDUCKGO_ENDPOINT = "https://api.duckduckgo.com/"
    DUCKDUCKGO_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"

    def __init__(self) -> None:
        self.provider = settings.web_search_provider.strip().lower()
        self.timeout = max(1, settings.web_search_timeout_seconds)

    @property
    def enabled(self) -> bool:
        return self.provider != "disabled"

    async def search(self, query: str, top_k: int | None = None) -> list[WebSearchResult]:
        if not self.enabled or not query.strip():
            return []

        size = max(1, min(int(top_k or settings.web_search_top_k), 20))

        if self.provider == "tavily":
            return await self._search_tavily(query, size)
        if self.provider == "serpapi":
            return await self._search_serpapi(query, size)
        if self.provider == "duckduckgo":
            return await self._search_duckduckgo(query, size)
        return []

    async def _search_tavily(self, query: str, top_k: int) -> list[WebSearchResult]:
        if not settings.tavily_api_key:
            return []
        payload = {
            "api_key": settings.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": top_k,
            "include_answer": False,
            "include_images": False,
        }
        data = await self._request_json(
            url=self.TAVILY_ENDPOINT,
            method="POST",
            payload=payload,
            headers={"Content-Type": "application/json"},
        )
        items = data.get("results", []) if isinstance(data, dict) else []
        results: list[WebSearchResult] = []
        for index, item in enumerate(items[:top_k]):
            if not isinstance(item, dict):
                continue
            results.append(
                WebSearchResult(
                    title=str(item.get("title", "") or ""),
                    url=str(item.get("url", "") or ""),
                    snippet=str(item.get("content", "") or ""),
                    score=min(max(coerce_float(item.get("score"), self._rank_score(index)), 0.0), 1.0),
                    provider="tavily",
                )
            )
        return results

    async def _search_serpapi(self, query: str, top_k: int) -> list[WebSearchResult]:
        if not settings.serpapi_api_key:
            return []
        params = {
            "q": query,
            "api_key": settings.serpapi_api_key,
            "engine": "google",
            "num": str(top_k),
        }
        data = await self._request_json(
            url=f"{self.SERPAPI_ENDPOINT}?{urllib.parse.urlencode(params)}",
            method="GET",
        )
        items = data.get("organic_results", []) if isinstance(data, dict) else []
        results: list[WebSearchResult] = []
        for index, item in enumerate(items[:top_k]):
            if not isinstance(item, dict):
                continue
            results.append(
                WebSearchResult(
                    title=str(item.get("title", "") or ""),
                    url=str(item.get("link", "") or ""),
                    snippet=str(item.get("snippet", "") or ""),
                    score=self._rank_score(index),
                    provider="serpapi",
                )
            )
        return results

    async def _search_duckduckgo(self, query: str, top_k: int) -> list[WebSearchResult]:
        html_results = await self._search_duckduckgo_html(query, top_k)
        if html_results:
            return html_results

        # The Instant Answer API is kept as a fallback for environments where
        # DuckDuckGo blocks its HTML endpoint.
        params = {
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
            "skip_disambig": "1",
        }
        data = await self._request_json(
            url=f"{self.DUCKDUCKGO_ENDPOINT}?{urllib.parse.urlencode(params)}",
            method="GET",
        )
        if not isinstance(data, dict):
            return []

        results: list[WebSearchResult] = []

        abstract_text = str(data.get("AbstractText", "") or "")
        abstract_url = str(data.get("AbstractURL", "") or "")
        abstract_source = str(data.get("Heading", "") or "DuckDuckGo")
        if abstract_text:
            results.append(
                WebSearchResult(
                    title=abstract_source,
                    url=abstract_url,
                    snippet=abstract_text,
                    score=1.0,
                    provider="duckduckgo",
                )
            )

        related = data.get("RelatedTopics", [])
        flat_topics = self._flatten_duck_topics(related)
        for index, topic in enumerate(flat_topics[: max(top_k - len(results), 0)]):
            title = topic.get("Text", "")
            url = topic.get("FirstURL", "")
            if not title:
                continue
            results.append(
                WebSearchResult(
                    title=str(title)[:120],
                    url=str(url),
                    snippet=str(title),
                    score=self._rank_score(index + 1),
                    provider="duckduckgo",
                )
            )

        return results[:top_k]

    async def _search_duckduckgo_html(self, query: str, top_k: int) -> list[WebSearchResult]:
        params = urllib.parse.urlencode({"q": query})
        html = await self._request_text(f"{self.DUCKDUCKGO_HTML_ENDPOINT}?{params}")
        if not html:
            return []
        parser = _DuckDuckGoHTMLParser()
        try:
            parser.feed(html)
        except Exception:
            return []

        results: list[WebSearchResult] = []
        for index, item in enumerate(parser.results):
            title = item.get("title", "").strip()
            url = self._unwrap_duck_url(item.get("url", ""))
            if not title or not url.startswith(("http://", "https://")):
                continue
            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=item.get("snippet", "").strip(),
                    score=self._rank_score(index),
                    provider="duckduckgo",
                )
            )
            if len(results) >= top_k:
                break
        return results

    async def _request_text(self, url: str) -> str:
        def _do_request() -> str:
            request = urllib.request.Request(
                url=url,
                method="GET",
                headers={"User-Agent": "Mozilla/5.0 (compatible; DeepResearchKnowledgeHub/1.0)"},
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(2_000_000)
                charset = response.headers.get_content_charset() or "utf-8"
            return raw.decode(charset, errors="replace")

        try:
            return await asyncio.to_thread(_do_request)
        except Exception:
            return ""

    async def _request_json(
        self,
        url: str,
        method: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | list[Any]:
        def _do_request() -> dict[str, Any] | list[Any]:
            req_headers = {
                "User-Agent": "DeepResearchKnowledgeHub/1.0",
                **(headers or {}),
            }
            body = None
            if payload is not None:
                body = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(url=url, data=body, headers=req_headers, method=method)
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="ignore")
            loaded = json.loads(raw)
            if isinstance(loaded, (dict, list)):
                return loaded
            return {}

        try:
            return await asyncio.to_thread(_do_request)
        except Exception:
            return {}

    @staticmethod
    def _flatten_duck_topics(raw_topics: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_topics, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw_topics:
            if not isinstance(item, dict):
                continue
            if "Topics" in item and isinstance(item.get("Topics"), list):
                out.extend(WebSearchService._flatten_duck_topics(item.get("Topics")))
            else:
                out.append(item)
        return out

    @staticmethod
    def _rank_score(index: int) -> float:
        return max(0.2, 1.0 - (index * 0.1))

    @staticmethod
    def _unwrap_duck_url(url: str) -> str:
        if url.startswith("//"):
            url = f"https:{url}"
        try:
            parsed = urllib.parse.urlsplit(url)
        except ValueError:
            return ""
        if parsed.hostname and parsed.hostname.endswith("duckduckgo.com"):
            target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
            if target:
                return urllib.parse.unquote(target)
        return url
