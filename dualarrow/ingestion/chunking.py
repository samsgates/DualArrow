from __future__ import annotations

import re
from collections.abc import Iterable

from dualarrow.ingestion.markdown import ParsedMarkdown
from dualarrow.models import KnowledgeUnit
from dualarrow.plugins import get_ingestion_view
from dualarrow.utils import sha256_text, stable_id, tokenize

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _unit(
    parsed: ParsedMarkdown,
    *,
    view_id: str,
    content: str,
    heading_path: list[str],
    line_start: int | None,
    line_end: int | None,
    parent_unit_id: str | None = None,
    suffix: str = "",
    metadata: dict | None = None,
) -> KnowledgeUnit:
    unit_id = stable_id(
        parsed.document.document_id,
        view_id,
        "/".join(heading_path),
        str(line_start),
        suffix,
        sha256_text(content),
        prefix="unit_",
    )
    return KnowledgeUnit(
        unit_id=unit_id,
        document_id=parsed.document.document_id,
        view_id=view_id,
        content=content.strip(),
        source_uri=parsed.document.source_uri,
        checksum=sha256_text(content),
        title=parsed.document.title,
        parent_unit_id=parent_unit_id,
        heading_path=heading_path,
        line_start=line_start,
        line_end=line_end,
        authority=parsed.document.authority,
        effective_from=parsed.document.effective_from,
        expires_at=parsed.document.expires_at,
        metadata={**parsed.document.metadata, **(metadata or {})},
    )


def full_document_units(parsed: ParsedMarkdown, view_id: str) -> list[KnowledgeUnit]:
    doc = parsed.document
    offset = int(doc.metadata.get("frontmatter_lines", 0))
    return [
        _unit(
            parsed,
            view_id=view_id,
            content=doc.content,
            heading_path=[doc.title],
            line_start=offset + 1,
            line_end=offset + max(1, len(doc.content.splitlines())),
        )
    ]


def heading_units(
    parsed: ParsedMarkdown,
    view_id: str,
    *,
    maximum_tokens: int = 1000,
) -> list[KnowledgeUnit]:
    units: list[KnowledgeUnit] = []
    for section in parsed.sections:
        chunks = split_by_tokens(section.content, maximum_tokens, overlap=80)
        for index, chunk in enumerate(chunks):
            units.append(
                _unit(
                    parsed,
                    view_id=view_id,
                    content=chunk,
                    heading_path=section.heading_path,
                    line_start=section.line_start,
                    line_end=section.line_end,
                    suffix=str(index),
                    metadata={"section_level": section.level, "chunk_index": index},
                )
            )
    return units


def recursive_units(
    parsed: ParsedMarkdown,
    view_id: str,
    *,
    chunk_size: int = 500,
    overlap: int = 75,
) -> list[KnowledgeUnit]:
    units: list[KnowledgeUnit] = []
    for section in parsed.sections:
        chunks = recursive_split(section.content, chunk_size, overlap)
        for index, chunk in enumerate(chunks):
            units.append(
                _unit(
                    parsed,
                    view_id=view_id,
                    content=chunk,
                    heading_path=section.heading_path,
                    line_start=section.line_start,
                    line_end=section.line_end,
                    suffix=str(index),
                    metadata={"chunk_index": index},
                )
            )
    return units


def parent_child_units(
    parsed: ParsedMarkdown,
    view_id: str,
    *,
    child_tokens: int = 180,
    parent_tokens: int = 900,
) -> list[KnowledgeUnit]:
    units: list[KnowledgeUnit] = []
    for section_index, section in enumerate(parsed.sections):
        parent_content = truncate_tokens(section.content, parent_tokens)
        parent = _unit(
            parsed,
            view_id=f"{view_id}:parent",
            content=parent_content,
            heading_path=section.heading_path,
            line_start=section.line_start,
            line_end=section.line_end,
            suffix=f"parent-{section_index}",
            metadata={"retrieval_role": "parent"},
        )
        units.append(parent)
        for child_index, child in enumerate(split_by_tokens(section.content, child_tokens, 30)):
            units.append(
                _unit(
                    parsed,
                    view_id=f"{view_id}:child",
                    content=child,
                    heading_path=section.heading_path,
                    line_start=section.line_start,
                    line_end=section.line_end,
                    parent_unit_id=parent.unit_id,
                    suffix=f"child-{section_index}-{child_index}",
                    metadata={"retrieval_role": "child", "chunk_index": child_index},
                )
            )
    return units


def sentence_window_units(
    parsed: ParsedMarkdown,
    view_id: str,
    *,
    previous_sentences: int = 2,
    next_sentences: int = 3,
) -> list[KnowledgeUnit]:
    units: list[KnowledgeUnit] = []
    for section in parsed.sections:
        sentences = [item.strip() for item in _SENTENCE_RE.split(section.content) if item.strip()]
        for index, sentence in enumerate(sentences):
            start = max(0, index - previous_sentences)
            end = min(len(sentences), index + next_sentences + 1)
            units.append(
                _unit(
                    parsed,
                    view_id=view_id,
                    content=" ".join(sentences[start:end]),
                    heading_path=section.heading_path,
                    line_start=section.line_start,
                    line_end=section.line_end,
                    suffix=str(index),
                    metadata={"focus_sentence": sentence, "sentence_index": index},
                )
            )
    return units


def proposition_units(parsed: ParsedMarkdown, view_id: str) -> list[KnowledgeUnit]:
    """Deterministic proposition baseline.

    Production deployments can replace this view builder with an LLM-backed extractor.
    The baseline preserves every sentence verbatim and therefore never invents facts.
    """
    units: list[KnowledgeUnit] = []
    for section in parsed.sections:
        sentences = [item.strip() for item in _SENTENCE_RE.split(section.content) if item.strip()]
        for index, sentence in enumerate(sentences):
            if len(tokenize(sentence)) < 4:
                continue
            units.append(
                _unit(
                    parsed,
                    view_id=view_id,
                    content=sentence,
                    heading_path=section.heading_path,
                    line_start=section.line_start,
                    line_end=section.line_end,
                    suffix=str(index),
                    metadata={"generated": False, "proposition_index": index},
                )
            )
    return units



def table_units(parsed: ParsedMarkdown, view_id: str) -> list[KnowledgeUnit]:
    units: list[KnowledgeUnit] = []
    separator = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
    for section in parsed.sections:
        lines = section.content.splitlines()
        index = 0
        table_index = 0
        while index + 1 < len(lines):
            if "|" not in lines[index] or not separator.match(lines[index + 1]):
                index += 1
                continue
            block = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                block.append(lines[index])
                index += 1
            headers = [cell.strip() for cell in block[0].strip(" |").split("|")]
            table_text = "\n".join(block)
            units.append(
                _unit(
                    parsed,
                    view_id=view_id,
                    content=table_text,
                    heading_path=section.heading_path,
                    line_start=section.line_start,
                    line_end=section.line_end,
                    suffix=f"table-{table_index}",
                    metadata={"content_type": "table", "headers": headers},
                )
            )
            for row_index, row in enumerate(block[2:]):
                values = [cell.strip() for cell in row.strip(" |").split("|")]
                if len(values) == len(headers):
                    row_text = "; ".join(f"{header}: {value}" for header, value in zip(headers, values, strict=True))
                    units.append(
                        _unit(
                            parsed,
                            view_id=f"{view_id}:row",
                            content=row_text,
                            heading_path=section.heading_path,
                            line_start=section.line_start,
                            line_end=section.line_end,
                            suffix=f"table-{table_index}-row-{row_index}",
                            metadata={"content_type": "table_row", "headers": headers},
                        )
                    )
            table_index += 1
    return units


def equation_units(parsed: ParsedMarkdown, view_id: str) -> list[KnowledgeUnit]:
    patterns = [
        re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
        re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
        re.compile(r"(?<!\$)\$([^\n$]{3,})\$(?!\$)"),
    ]
    units: list[KnowledgeUnit] = []
    seen: set[str] = set()
    for section in parsed.sections:
        for pattern in patterns:
            for index, match in enumerate(pattern.finditer(section.content)):
                equation = " ".join(match.group(1).split())
                if equation in seen:
                    continue
                seen.add(equation)
                units.append(
                    _unit(
                        parsed,
                        view_id=view_id,
                        content=equation,
                        heading_path=section.heading_path,
                        line_start=section.line_start,
                        line_end=section.line_end,
                        suffix=f"equation-{index}-{sha256_text(equation)[:8]}",
                        metadata={"content_type": "equation", "format": "latex"},
                    )
                )
    return units


def code_units(parsed: ParsedMarkdown, view_id: str) -> list[KnowledgeUnit]:
    pattern = re.compile(r"```([\w.+-]*)\n(.*?)```", re.DOTALL)
    units: list[KnowledgeUnit] = []
    for section in parsed.sections:
        for index, match in enumerate(pattern.finditer(section.content)):
            language = match.group(1) or "text"
            code = match.group(2).strip()
            units.append(
                _unit(
                    parsed,
                    view_id=view_id,
                    content=code,
                    heading_path=section.heading_path,
                    line_start=section.line_start,
                    line_end=section.line_end,
                    suffix=f"code-{index}",
                    metadata={"content_type": "code", "language": language},
                )
            )
    return units


def hierarchy_units(parsed: ParsedMarkdown, view_id: str, *, preview_tokens: int = 120) -> list[KnowledgeUnit]:
    return [
        _unit(
            parsed,
            view_id=view_id,
            content=f"Heading path: {' > '.join(section.heading_path)}\n{truncate_tokens(section.content, preview_tokens)}",
            heading_path=section.heading_path,
            line_start=section.line_start,
            line_end=section.line_end,
            suffix=f"hierarchy-{index}",
            metadata={"content_type": "hierarchy", "level": section.level},
        )
        for index, section in enumerate(parsed.sections)
    ]


def recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(tokenize(text)) <= chunk_size:
        return [text.strip()] if text.strip() else []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for paragraph in paragraphs:
        count = len(tokenize(paragraph))
        if count > chunk_size:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            chunks.extend(split_by_tokens(paragraph, chunk_size, overlap))
            continue
        if current and current_tokens + count > chunk_size:
            chunks.append("\n\n".join(current))
            overlap_text = truncate_tokens(" ".join(current), overlap, from_end=True)
            current = [overlap_text, paragraph] if overlap_text else [paragraph]
            current_tokens = len(tokenize(" ".join(current)))
        else:
            current.append(paragraph)
            current_tokens += count
    if current:
        chunks.append("\n\n".join(current))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def split_by_tokens(text: str, size: int, overlap: int = 0) -> list[str]:
    words = text.split()
    if not words:
        return []
    if size <= overlap:
        raise ValueError("chunk size must be larger than overlap")
    step = size - overlap
    return [" ".join(words[start:start + size]) for start in range(0, len(words), step)]


def truncate_tokens(text: str, maximum: int, *, from_end: bool = False) -> str:
    words = text.split()
    selected = words[-maximum:] if from_end else words[:maximum]
    return " ".join(selected)


def build_units(parsed: ParsedMarkdown, views: Iterable) -> list[KnowledgeUnit]:
    units: list[KnowledgeUnit] = []
    for view in views:
        if not view.enabled:
            continue
        options = view.options
        if view.type == "full_document":
            units.extend(full_document_units(parsed, view.id))
        elif view.type == "markdown_heading":
            units.extend(heading_units(parsed, view.id, **options))
        elif view.type == "recursive":
            units.extend(recursive_units(parsed, view.id, **options))
        elif view.type == "parent_child":
            units.extend(parent_child_units(parsed, view.id, **options))
        elif view.type == "sentence_window":
            units.extend(sentence_window_units(parsed, view.id, **options))
        elif view.type == "atomic_propositions":
            units.extend(proposition_units(parsed, view.id))
        elif view.type == "table":
            units.extend(table_units(parsed, view.id))
        elif view.type == "equation":
            units.extend(equation_units(parsed, view.id))
        elif view.type == "code":
            units.extend(code_units(parsed, view.id))
        elif view.type == "hierarchy":
            units.extend(hierarchy_units(parsed, view.id, **options))
        else:
            plugin = get_ingestion_view(view.type)
            if plugin is None:
                raise ValueError(f"Unsupported ingestion view: {view.type}")
            units.extend(plugin(parsed, view.id, options))
    unique: dict[str, KnowledgeUnit] = {unit.unit_id: unit for unit in units if unit.content.strip()}
    return list(unique.values())
