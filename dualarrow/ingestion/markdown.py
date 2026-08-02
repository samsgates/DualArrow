from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dualarrow.models import SourceDocument
from dualarrow.utils import sha256_text, stable_id

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)|\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


@dataclass(slots=True)
class MarkdownSection:
    heading_path: list[str]
    content: str
    line_start: int
    line_end: int
    level: int


@dataclass(slots=True)
class ParsedMarkdown:
    document: SourceDocument
    sections: list[MarkdownSection] = field(default_factory=list)
    links: list[str] = field(default_factory=list)


def _parse_datetime(value: Any) -> Any:
    return value


def parse_markdown(path: Path, root: Path) -> ParsedMarkdown:
    raw = path.read_text(encoding="utf-8")
    body = raw
    metadata: dict[str, Any] = {}
    line_offset = 0
    match = _FRONTMATTER_RE.match(raw)
    if match:
        loaded = yaml.safe_load(match.group(1)) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"YAML frontmatter must be a mapping: {path}")
        metadata = loaded
        body = raw[match.end():]
        line_offset = raw[: match.end()].count("\n")

    relative = path.relative_to(root).as_posix()
    document_id = str(metadata.get("id") or stable_id(relative, prefix="doc_"))
    title = str(metadata.get("title") or _first_title(body) or path.stem.replace("-", " ").title())
    authority = float(metadata.get("confidence", metadata.get("authority_score", 0.75)))
    source_meta = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    source_uri = str(source_meta.get("uri") or relative)
    version = str(metadata.get("version", "1"))

    document = SourceDocument(
        document_id=document_id,
        title=title,
        content=body.strip(),
        source_uri=source_uri,
        checksum=sha256_text(raw),
        version=version,
        authority=max(0.0, min(1.0, authority)),
        status=str(metadata.get("status", "approved")),
        effective_from=_parse_datetime(metadata.get("effective_from")),
        expires_at=_parse_datetime(metadata.get("expires_at")),
        metadata={**metadata, "relative_path": relative, "frontmatter_lines": line_offset},
    )
    sections = split_markdown_sections(body, line_offset=line_offset)
    links = extract_links(body)
    relations = metadata.get("relations", [])
    if isinstance(relations, list):
        for relation in relations:
            if isinstance(relation, dict) and relation.get("target"):
                links.append(str(relation["target"]))
    return ParsedMarkdown(document=document, sections=sections, links=sorted(set(links)))


def _first_title(body: str) -> str | None:
    for line in body.splitlines():
        match = _HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return None


def split_markdown_sections(body: str, line_offset: int = 0) -> list[MarkdownSection]:
    lines = body.splitlines()
    headings: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match:
            headings.append((index, len(match.group(1)), match.group(2).strip()))

    if not headings:
        return [
            MarkdownSection(
                heading_path=[],
                content=body.strip(),
                line_start=line_offset + 1,
                line_end=line_offset + max(1, len(lines)),
                level=0,
            )
        ] if body.strip() else []

    result: list[MarkdownSection] = []
    stack: list[tuple[int, str]] = []
    for position, (start, level, title) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        content = "\n".join(lines[start:end]).strip()
        if content:
            result.append(
                MarkdownSection(
                    heading_path=[item[1] for item in stack],
                    content=content,
                    line_start=line_offset + start + 1,
                    line_end=line_offset + max(start + 1, end),
                    level=level,
                )
            )
    return result


def extract_links(body: str) -> list[str]:
    links: list[str] = []
    for match in _LINK_RE.finditer(body):
        target = match.group(1) or match.group(2)
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        target = target.split("#", 1)[0].strip()
        if target:
            links.append(target)
    return links
