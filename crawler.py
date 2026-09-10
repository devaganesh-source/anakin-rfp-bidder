"""Extract structured requirements from the local mock RFP page."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup, Tag


class RFPCrawler:
    """Fetch and parse the deliberately inconsistent local RFP document."""

    DEFAULT_URL = "http://localhost:8080/mock_rfp.html"

    def __init__(self, url: str = DEFAULT_URL) -> None:
        self.url = url
        self.html: str | None = None
        self.dom: BeautifulSoup | None = None

    async def fetch_html(self) -> str:
        """Fetch the configured RFP page once using an asynchronous client."""
        timeout = httpx.Timeout(15.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(self.url)
            response.raise_for_status()

        self.html = response.text
        return self.html

    def clean_dom(self, html: str | None = None) -> BeautifulSoup:
        """Remove non-content elements and retain a reusable parsed DOM."""
        source = html if html is not None else self.html
        if source is None:
            raise RuntimeError("Fetch HTML before cleaning the DOM.")

        soup = BeautifulSoup(source, "html.parser")

        for element in reversed(soup.find_all(["script", "style", "nav", "footer"])):
            element.decompose()

        # Legacy sites often use generic divs instead of semantic nav/footer tags.
        unwanted_marker = re.compile(
            r"(?:^|[-_\s])(cookie|cookie-banner|consent|gdpr|nav|navigation|navbar)(?:$|[-_\s])",
            re.IGNORECASE,
        )
        for element in reversed(soup.find_all(True)):
            element_id = str(element.get("id", ""))
            classes = element.get("class", [])
            class_names = " ".join(classes) if isinstance(classes, list) else str(classes)
            marker = f"{element_id} {class_names}".strip()
            if marker and unwanted_marker.search(marker):
                element.decompose()

        self.dom = soup
        return soup

    def parse_rfp(self, dom: BeautifulSoup | None = None) -> dict[str, str]:
        """Extract requirements using table, list, sibling, and inline-title heuristics."""
        soup = dom if dom is not None else self.dom
        if soup is None:
            raise RuntimeError("Clean the DOM before parsing the RFP.")

        requirements: dict[str, str] = {}

        def add_requirement(title: str, text: str) -> None:
            clean_title = self._normalise(title).rstrip(":.- ")
            clean_text = self._normalise(text).lstrip(":.- ")
            if not clean_title or not clean_text:
                return

            existing = requirements.get(clean_title)
            if existing is None:
                requirements[clean_title] = clean_text
            elif clean_text != existing and clean_text not in existing.split("\n"):
                requirements[clean_title] = f"{existing}\n{clean_text}"

        # Heuristic 1: rows containing a reference/title cell and a requirement cell.
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all(["th", "td"], recursive=False)
                if len(cells) != 2 or all(cell.name == "th" for cell in cells):
                    continue
                add_requirement(cells[0].get_text(" ", strip=True), cells[1].get_text(" ", strip=True))

        # Heuristic 3 is evaluated before header pairs so an inline title wins when
        # one paragraph could otherwise be attributed to its preceding header.
        strong_paragraphs: set[int] = set()
        for paragraph in soup.find_all("p"):
            strong = paragraph.find("strong")
            if strong is None:
                continue

            title = strong.get_text(" ", strip=True)
            paragraph_text = paragraph.get_text(" ", strip=True)
            requirement_text = self._text_after_title(paragraph_text, title)
            add_requirement(title, requirement_text)
            strong_paragraphs.add(id(paragraph))

        # Heuristic 2: a heading whose next element sibling is a paragraph.
        for heading in soup.find_all(["h2", "h3", "h4"]):
            sibling = heading.find_next_sibling()
            
            # Walk forward through non-Tag elements (like NavigableString whitespace) to find the actual element sibling
            while sibling is not None and not isinstance(sibling, Tag):
                sibling = sibling.next_sibling

            if not isinstance(sibling, Tag) or sibling.name != "p":
                continue
            if id(sibling) in strong_paragraphs:
                continue
            add_requirement(
                heading.get_text(" ", strip=True),
                sibling.get_text(" ", strip=True),
            )

        # Heuristic 4: emit every list item separately. Restrict each item's text
        # to content owned by that <li> so nested child items are not duplicated.
        list_item_number = 0
        for list_element in soup.find_all(["ul", "ol"]):
            for item in list_element.find_all("li", recursive=False):
                item_text = self._normalise(
                    " ".join(
                        str(text)
                        for text in item.find_all(string=True)
                        if text.find_parent("li") is item
                    )
                )
                if not item_text:
                    continue

                list_item_number += 1
                heading = item.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
                heading_text = (
                    heading.get_text(" ", strip=True)
                    if isinstance(heading, Tag)
                    else "List requirement"
                )
                add_requirement(f"{heading_text} - Item {list_item_number}", item_text)

        return requirements

    async def crawl(self) -> dict[str, str]:
        """Run the complete fetch, clean, and parse workflow."""
        await self.fetch_html()
        self.clean_dom()
        return self.parse_rfp()

    @staticmethod
    def _normalise(value: Any) -> str:
        """Collapse whitespace emitted by irregular portal markup."""
        return " ".join(str(value).split())

    @classmethod
    def _text_after_title(cls, paragraph_text: str, title: str) -> str:
        """Remove a leading strong-tag title from its containing paragraph."""
        clean_paragraph = cls._normalise(paragraph_text)
        clean_title = cls._normalise(title)
        if clean_paragraph.casefold().startswith(clean_title.casefold()):
            return clean_paragraph[len(clean_title) :]
        return clean_paragraph


async def main() -> None:
    """Crawl the mock RFP and print its requirements as readable JSON."""
    crawler = RFPCrawler()
    requirements = await crawler.crawl()
    print(json.dumps(requirements, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except httpx.HTTPError as exc:
        raise SystemExit(f"Could not fetch the mock RFP: {exc}") from exc
