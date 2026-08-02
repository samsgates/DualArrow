from pathlib import Path

from dualarrow.config import ViewSpec
from dualarrow.ingestion.chunking import build_units
from dualarrow.ingestion.markdown import parse_markdown


def test_table_equation_and_code_views(tmp_path: Path):
    root = tmp_path / "knowledge"
    root.mkdir()
    path = root / "technical.md"
    path.write_text(
        """# Technical Note

## Values

| Part | Limit |
|---|---|
| BR-1 | 7.5 |

## Formula

The energy relation is $E = mc^2$.

## Example

```python
print("hello")
```
""",
        encoding="utf-8",
    )
    parsed = parse_markdown(path, root)
    units = build_units(
        parsed,
        [
            ViewSpec(id="tables", type="table"),
            ViewSpec(id="equations", type="equation"),
            ViewSpec(id="code", type="code"),
            ViewSpec(id="hierarchy", type="hierarchy"),
        ],
    )
    types = {unit.metadata.get("content_type") for unit in units}
    assert "table" in types
    assert "table_row" in types
    assert "equation" in types
    assert "code" in types
    assert "hierarchy" in types
