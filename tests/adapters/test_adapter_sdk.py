import pytest

from Medical_KG_rev.adapters import ExampleAdapter, run_adapter
from Medical_KG_rev.adapters.registry import AdapterRegistry
from Medical_KG_rev.adapters.yaml_parser import AdapterConfig, load_adapter_config


def test_example_adapter_run():
    adapter = ExampleAdapter()
    result = run_adapter(adapter)
    assert len(result.documents) == 1
    assert result.documents[0].source == "example"


def test_registry_register_and_create():
    local_registry = AdapterRegistry()
    local_registry.register(ExampleAdapter)
    instance = local_registry.create("ExampleAdapter")
    assert isinstance(instance, ExampleAdapter)

    with pytest.raises(ValueError):
        local_registry.register(ExampleAdapter)

    with pytest.raises(KeyError):
        local_registry.create("MissingAdapter")


def test_load_adapter_config(tmp_path):
    config_file = tmp_path / "adapter.yaml"
    config_file.write_text(
        """
name: example-config
source: example
base_url: https://example.com
request:
  method: GET
  path: /resource
mapping:
  id: id
"""
    )
    config = load_adapter_config(config_file)
    assert isinstance(config, AdapterConfig)
    assert config.source == "example"
    assert config.request.path == "/resource"

    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("{}")
    with pytest.raises(ValueError):
        load_adapter_config(empty_file)
