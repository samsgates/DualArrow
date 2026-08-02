from pathlib import Path

from dualarrow.config import load_config
from dualarrow.pipeline import DualArrow


async def test_strict_plain_uses_ephemeral_index(project: Path):
    config_path = project / "dualarrow.yaml"
    text = config_path.read_text(encoding="utf-8").replace(
        "sidecar_path: ./knowledge/.dualarrow/test.db",
        "mode: strict_plain\n  sidecar_path: ./knowledge/.dualarrow/should-not-exist.db",
    )
    config_path.write_text(text, encoding="utf-8")
    engine = DualArrow.from_config(config_path)
    index_path = engine.store.canonical.path
    assert str(index_path).startswith("/tmp/")
    await engine.start()
    await engine.ingest()
    assert index_path.exists()
    await engine.close()
    assert not index_path.exists()
    assert not (project / "knowledge" / ".dualarrow" / "should-not-exist.db").exists()
