"""YAML loading and cross-file validation for EdgeWeaver configuration."""

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import Field, model_validator

from edgeweaver.domain import (
    Device,
    DomainModel,
    ModelProfile,
    NetworkLink,
    ScenarioConfig,
    SimulationConfig,
)
from edgeweaver.scenarios import ResearchScenarioConfig


class DeviceCatalog(DomainModel):
    devices: list[Device] = Field(min_length=1)

    @model_validator(mode="after")
    def device_ids_are_unique(self) -> Self:
        ids = [device.id for device in self.devices]
        if len(ids) != len(set(ids)):
            raise ValueError("device IDs must be unique")
        return self


class NetworkCatalog(DomainModel):
    links: list[NetworkLink]

    @model_validator(mode="after")
    def link_ids_are_unique(self) -> Self:
        ids = [link.link_id for link in self.links]
        if len(ids) != len(set(ids)):
            raise ValueError("network link IDs must be unique")
        return self


class ModelProfileCatalog(DomainModel):
    profiles: list[ModelProfile] = Field(min_length=1)

    @model_validator(mode="after")
    def model_ids_are_unique(self) -> Self:
        ids = [profile.model_id for profile in self.profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("model profile IDs must be unique")
        return self


class LoadedConfiguration(DomainModel):
    devices: DeviceCatalog
    network: NetworkCatalog
    model_profiles: ModelProfileCatalog
    scenario: ScenarioConfig

    @model_validator(mode="after")
    def references_are_valid(self) -> Self:
        devices_by_id = {device.id: device for device in self.devices.devices}
        profiles_by_id = {profile.model_id: profile for profile in self.model_profiles.profiles}

        if self.scenario.device_id not in devices_by_id:
            raise ValueError(f"unknown scenario device: {self.scenario.device_id}")
        if self.scenario.model_id not in profiles_by_id:
            raise ValueError(f"unknown scenario model: {self.scenario.model_id}")

        selected_device = devices_by_id[self.scenario.device_id]
        if self.scenario.model_id not in selected_device.supported_models:
            raise ValueError(
                f"device {selected_device.id} does not support model {self.scenario.model_id}"
            )

        device_ids = set(devices_by_id)
        for request in self.scenario.requests:
            if request.source_device_id not in device_ids:
                raise ValueError(
                    f"request {request.request_id} has unknown source device "
                    f"{request.source_device_id}"
                )
        for link in self.network.links:
            if (
                link.source_device_id not in device_ids
                or link.destination_device_id not in device_ids
            ):
                raise ValueError(f"network link {link.link_id} references an unknown device")
        return self

    def selected_device(self) -> Device:
        return next(
            device for device in self.devices.devices if device.id == self.scenario.device_id
        )

    def selected_model_profile(self) -> ModelProfile:
        return next(
            profile
            for profile in self.model_profiles.profiles
            if profile.model_id == self.scenario.model_id
        )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a YAML mapping: {path}")
    return raw


def load_device_catalog(path: Path) -> DeviceCatalog:
    return DeviceCatalog.model_validate(_load_yaml_mapping(path))


def load_network_catalog(path: Path) -> NetworkCatalog:
    return NetworkCatalog.model_validate(_load_yaml_mapping(path))


def load_model_profile_catalog(path: Path) -> ModelProfileCatalog:
    return ModelProfileCatalog.model_validate(_load_yaml_mapping(path))


def load_scenario(path: Path) -> ScenarioConfig:
    return ScenarioConfig.model_validate(_load_yaml_mapping(path))


def load_simulation_config(path: Path) -> SimulationConfig:
    return SimulationConfig.model_validate(_load_yaml_mapping(path))


def load_research_scenario(path: Path) -> ResearchScenarioConfig:
    """Load one strict Phase 6 controlled-scenario definition."""

    return ResearchScenarioConfig.model_validate(_load_yaml_mapping(path))


def load_phase1_configuration(
    *,
    devices_path: Path,
    network_path: Path,
    model_profiles_path: Path,
    scenario_path: Path,
) -> LoadedConfiguration:
    """Load Phase 1 YAML files and validate their cross-file references."""

    return LoadedConfiguration(
        devices=load_device_catalog(devices_path),
        network=load_network_catalog(network_path),
        model_profiles=load_model_profile_catalog(model_profiles_path),
        scenario=load_scenario(scenario_path),
    )
