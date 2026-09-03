from pathlib import Path

import pytest
from pydantic import ValidationError

from edgeweaver.config import LoadedConfiguration, load_device_catalog


def test_phase1_configuration_loads_and_links_references(
    phase1_configuration: LoadedConfiguration,
) -> None:
    assert [device.id for device in phase1_configuration.devices.devices] == [
        "mobile",
        "gateway",
        "edge-server",
    ]
    assert len(phase1_configuration.network.links) == 2
    assert phase1_configuration.selected_device().id == "mobile"
    assert phase1_configuration.selected_model_profile().profile_kind == "synthetic_fixture"


def test_invalid_device_yaml_is_rejected(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid-devices.yaml"
    invalid_path.write_text(
        """devices:
  - id: mobile
    speed_multiplier: 0
    active_power_units: 1
    queue_capacity: 20
    supported_models: [light]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_device_catalog(invalid_path)
