"""Endpoint page parsing for :mod:`~fmp_docs_compiler`.

Purpose:
    Parse individual docs pages into structured fields suitable for later
    normalization and code generation.

Design:
    The parser uses fallback strategies in this order:

    1. heading and section extraction
    2. parameter table parsing
    3. definition-list parsing
    4. bullet and paragraph heuristics
    5. code-block example extraction

    This keeps the parser fairly robust when page markup differs slightly.

Attributes:
    PREMIUM_MARKERS:
        Case-insensitive phrases that indicate plan gating or premium upsell.

Examples:
    ::
        >>> parser = EndpointPageParser()
        >>> isinstance(parser, EndpointPageParser)
        True
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from bs4 import BeautifulSoup, Tag

from .models import (
    DiscoveredDocPage,
    ParsedEndpointPage,
    ParsedExampleDoc,
    ParsedParameterDoc,
)

PREMIUM_MARKERS: tuple[str, ...] = (
    "unlock premium",
    "upgrade to our premium",
    "premium financial insights",
    "exclusive access",
    "premium data",
)
_ENDPOINT_URL_RE = re.compile(
    r'https://financialmodelingprep\.com/(?:stable|api/v3)/[^\s<>"\']+', re.IGNORECASE
)
_PARAM_HINT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_\-]+)\s*(?:\((?P<type>[^)]+)\))?\s*[:-]\s*(?P<desc>.+)$"
)


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


class EndpointPageParser:
    """Parse individual docs pages into structured models."""

    def parse(self, page: DiscoveredDocPage, html: str) -> ParsedEndpointPage:
        soup = BeautifulSoup(html, "lxml")
        raw_text = _normalize_text(soup.get_text(" ", strip=True))
        title = self._parse_title(soup=soup, page=page)
        summary = self._parse_summary(soup=soup)
        about = self._parse_section(soup=soup, title_prefix="about")
        example_use_case = self._parse_section(
            soup=soup, title_prefix="example use case"
        )
        endpoint_url = self._extract_endpoint_url(raw_text=raw_text)
        parsed_parameters, parse_warnings = self._parse_parameters(soup=soup)
        examples = self._parse_examples(soup=soup, endpoint_url=endpoint_url)
        premium_signals = self._detect_premium_signals(raw_text=raw_text)
        return ParsedEndpointPage(
            docs_url=page.url,
            category=page.category,
            label=page.label,
            title=title,
            summary=summary,
            about=about,
            example_use_case=example_use_case,
            endpoint_url=endpoint_url,
            parsed_parameters=parsed_parameters,
            examples=examples,
            premium_signals=premium_signals,
            parse_warnings=parse_warnings,
            raw_text=raw_text,
            legacy=page.legacy,
        )

    def _parse_title(self, soup: BeautifulSoup, page: DiscoveredDocPage) -> str:
        heading = soup.find("h1")
        if isinstance(heading, Tag):
            text = _normalize_text(heading.get_text(" ", strip=True))
            if text:
                return text
        return page.label

    def _parse_summary(self, soup: BeautifulSoup) -> str | None:
        meta = soup.find("meta", attrs={"name": "description"})
        if isinstance(meta, Tag):
            content = _normalize_text(str(meta.get("content", "")))
            if content:
                return content
        for paragraph in soup.find_all("p"):
            text = _normalize_text(paragraph.get_text(" ", strip=True))
            if text and not text.lower().startswith("endpoint"):
                return text
        return None

    def _parse_section(self, soup: BeautifulSoup, title_prefix: str) -> str | None:
        for heading in soup.find_all(re.compile(r"^h[1-6]$")):
            heading_text = _normalize_text(heading.get_text(" ", strip=True)).lower()
            if not heading_text.startswith(title_prefix):
                continue
            parts: list[str] = []
            for sibling in heading.next_siblings:
                if (
                    isinstance(sibling, Tag)
                    and sibling.name
                    and re.match(r"^h[1-6]$", sibling.name)
                ):
                    break
                if isinstance(sibling, Tag):
                    text = _normalize_text(sibling.get_text(" ", strip=True))
                    if text:
                        parts.append(text)
            section_text = _normalize_text(" ".join(parts))
            if section_text:
                return section_text
        return None

    def _extract_endpoint_url(self, raw_text: str) -> str | None:
        match = _ENDPOINT_URL_RE.search(raw_text)
        return match.group(0) if match else None

    def _find_parameter_heading(self, soup: BeautifulSoup) -> Tag | None:
        for heading in soup.find_all(re.compile(r"^h[1-6]$")):
            if (
                "parameter"
                in _normalize_text(heading.get_text(" ", strip=True)).lower()
            ):
                return heading
        return None

    def _iter_section_tags(self, heading: Tag) -> Iterable[Tag]:
        for sibling in heading.next_siblings:
            if (
                isinstance(sibling, Tag)
                and sibling.name
                and re.match(r"^h[1-6]$", sibling.name)
            ):
                break
            if isinstance(sibling, Tag):
                yield sibling

    def _parse_parameters(
        self, soup: BeautifulSoup
    ) -> tuple[list[ParsedParameterDoc], list[str]]:
        heading = self._find_parameter_heading(soup)
        if heading is None:
            return [], []
        parameters: list[ParsedParameterDoc] = []
        warnings: list[str] = []
        parameters.extend(self._parse_table_parameters(heading))
        parameters.extend(self._parse_definition_list_parameters(heading))
        parameters.extend(self._parse_list_parameters(heading))
        deduped: dict[str, ParsedParameterDoc] = {}
        for parameter in parameters:
            deduped.setdefault(parameter.name, parameter)
        if not deduped:
            warnings.append("No structured parameters parsed from parameter section.")
        return list(deduped.values()), warnings

    def _parse_table_parameters(self, heading: Tag) -> list[ParsedParameterDoc]:
        parameters: list[ParsedParameterDoc] = []
        for tag in self._iter_section_tags(heading):
            if tag.name != "table":
                continue
            rows = tag.find_all("tr")
            if not rows:
                continue
            headers = [
                _normalize_text(cell.get_text(" ", strip=True)).lower()
                for cell in rows[0].find_all(["th", "td"])
            ]
            for row in rows[1:]:
                cells = [
                    _normalize_text(cell.get_text(" ", strip=True))
                    for cell in row.find_all(["td", "th"])
                ]
                if not cells:
                    continue
                data = dict(zip(headers, cells, strict=False))
                name = data.get("name") or data.get("parameter") or cells[0]
                parameters.append(
                    ParsedParameterDoc(
                        name=name,
                        type_hint=data.get("type") or None,
                        description=data.get("description")
                        or data.get("details")
                        or None,
                        required="required" in (data.get("required", "").lower()),
                        source="table",
                    )
                )
        return parameters

    def _parse_definition_list_parameters(
        self, heading: Tag
    ) -> list[ParsedParameterDoc]:
        parameters: list[ParsedParameterDoc] = []
        for tag in self._iter_section_tags(heading):
            if tag.name != "dl":
                continue
            current_name: str | None = None
            for child in tag.find_all(["dt", "dd"], recursive=False):
                if child.name == "dt":
                    current_name = _normalize_text(child.get_text(" ", strip=True))
                elif child.name == "dd" and current_name:
                    parameters.append(
                        ParsedParameterDoc(
                            name=current_name,
                            description=_normalize_text(child.get_text(" ", strip=True))
                            or None,
                            required="required"
                            in _normalize_text(child.get_text(" ", strip=True)).lower(),
                            source="definition_list",
                        )
                    )
                    current_name = None
        return parameters

    def _parse_list_parameters(self, heading: Tag) -> list[ParsedParameterDoc]:
        parameters: list[ParsedParameterDoc] = []
        for tag in self._iter_section_tags(heading):
            if tag.name not in {"ul", "ol"}:
                continue
            for item in tag.find_all("li", recursive=False):
                text = _normalize_text(item.get_text(" ", strip=True))
                if not text or "no required query parameters" in text.lower():
                    continue
                match = _PARAM_HINT_RE.match(text)
                if match:
                    parameters.append(
                        ParsedParameterDoc(
                            name=match.group("name"),
                            type_hint=match.group("type") or None,
                            description=match.group("desc"),
                            required="required" in match.group("desc").lower(),
                            source="list",
                        )
                    )
        return parameters

    def _parse_examples(
        self, soup: BeautifulSoup, endpoint_url: str | None
    ) -> list[ParsedExampleDoc]:
        examples: list[ParsedExampleDoc] = []
        if endpoint_url:
            examples.append(
                ParsedExampleDoc(
                    label="endpoint_url",
                    content_type="text/uri-list",
                    payload=endpoint_url,
                    source="text",
                )
            )
        for pre in soup.find_all(["pre", "code"]):
            text = _normalize_text(pre.get_text(" ", strip=True))
            if not text:
                continue
            try:
                payload = json.loads(text)
                content_type = "application/json"
            except json.JSONDecodeError:
                payload = text
                content_type = "text/plain"
            examples.append(
                ParsedExampleDoc(
                    label="code_example",
                    content_type=content_type,
                    payload=payload,
                    source="code",
                )
            )
        return examples

    def _detect_premium_signals(self, raw_text: str) -> list[str]:
        lowered = raw_text.lower()
        return [marker for marker in PREMIUM_MARKERS if marker in lowered]
