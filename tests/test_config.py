from dualarrow.config import load_config


def test_load_config_resolves_paths(project):
    config = load_config(project / "dualarrow.yaml")
    assert config.project.name == "Test"
    assert config.knowledge_store.root.is_absolute()
    assert config.knowledge_store.sidecar_path.is_absolute()
    assert "balanced" in config.retrieval_profiles
