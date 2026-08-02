from dualarrow.ingestion.markdown import parse_markdown


def test_markdown_parsing(project):
    parsed = parse_markdown(project / "knowledge" / "machine.md", project / "knowledge")
    assert parsed.document.document_id == "machine.test"
    assert parsed.document.title == "Test Machine Manual"
    assert any(section.heading_path[-1] == "Approved part" for section in parsed.sections)
