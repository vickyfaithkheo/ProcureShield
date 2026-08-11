from __future__ import annotations

from typing import List, Dict
import re
import requests
from bs4 import BeautifulSoup


def normalize_query(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:200]


def search_public_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Lightweight public web search using DuckDuckGo HTML.
    Returns a list of dicts with:
      - title
      - url
      - snippet
    """
    query = normalize_query(query)
    if not query:
        return []

    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    results: List[Dict[str, str]] = []

    try:
        resp = requests.post(url, data={"q": query}, headers=headers, timeout=20)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for card in soup.select(".result"):
            title_el = card.select_one(".result__title a.result__a")
            snippet_el = card.select_one(".result__snippet")

            if not title_el:
                continue

            title = title_el.get_text(" ", strip=True)
            href = title_el.get("href", "")
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""

            results.append(
                {
                    "title": title,
                    "url": href,
                    "snippet": snippet,
                }
            )

            if len(results) >= max_results:
                break

    except Exception as e:
        return [
            {
                "title": "Search error",
                "url": "",
                "snippet": str(e),
            }
        ]

    return results