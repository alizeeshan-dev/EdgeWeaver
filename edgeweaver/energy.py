"""Normalized energy estimates; these values are not physical joules."""

from edgeweaver.domain import Device, EnergyBreakdown


def compute_energy_units(device: Device, inference_time_ms: float) -> float:
    if inference_time_ms < 0.0:
        raise ValueError("inference_time_ms must be non-negative")
    return device.active_power_units * inference_time_ms


def network_energy_units(transferred_bytes: int, energy_per_kilobyte: float) -> float:
    if transferred_bytes < 0 or energy_per_kilobyte < 0.0:
        raise ValueError("network energy inputs must be non-negative")
    return transferred_bytes / 1024.0 * energy_per_kilobyte


def estimate_energy(
    *,
    device: Device,
    inference_time_ms: float,
    transferred_bytes: int,
    network_energy_per_kilobyte: float,
) -> EnergyBreakdown:
    compute = compute_energy_units(device, inference_time_ms)
    network = network_energy_units(transferred_bytes, network_energy_per_kilobyte)
    return EnergyBreakdown(
        compute_energy_units=compute,
        network_energy_units=network,
        estimated_total_energy_units=compute + network,
    )
