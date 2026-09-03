"""Shared model/device compatibility rule."""

from edgeweaver.domain import Device, ModelProfile


def is_device_model_compatible(device: Device, profile: ModelProfile) -> bool:
    role = profile.computational_role
    return role is not None and role.value in device.supported_models
