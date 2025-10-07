import pytest

import pytest

from Medical_KG_rev.adapters import AdapterDomain, AdapterPluginManager, AdapterRequest
from Medical_KG_rev.adapters.plugins.example import ExampleAdapterPlugin
from Medical_KG_rev.adapters.yaml_parser import AdapterConfig, load_adapter_config


def _request() -> AdapterRequest:
    return AdapterRequest(
        tenant_id="tenant",
        correlation_id="corr",
        domain=AdapterDomain.BIOMEDICAL,
    )


def test_example_plugin_run():
    manager = AdapterPluginManager()
    metadata = manager.register(ExampleAdapterPlugin())
    assert metadata.name == "example"
    response = manager.run("example", _request())
    assert response.items[0]["message"] == "hello world"
    assert manager.check_health("example") is True


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
