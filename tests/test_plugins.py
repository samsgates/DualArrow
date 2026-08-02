from dualarrow.plugins import get_ingestion_view, register_ingestion_view


def test_runtime_plugin_registration():
    name = "test_custom_view"

    def builder(parsed, view_id, options):
        return []

    register_ingestion_view(name, builder)
    assert get_ingestion_view(name) is builder
