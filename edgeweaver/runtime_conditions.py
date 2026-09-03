"""Deterministic time-bounded runtime conditions for one scenario run."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from edgeweaver.config import DeviceCatalog, NetworkCatalog
from edgeweaver.domain import EventType, JsonScalar, NetworkLink
from edgeweaver.scenarios import ResearchScenarioConfig


@dataclass(frozen=True)
class RuntimeConditionEvent:
    timestamp_ms: float
    event_type: EventType
    device_id: str | None = None
    details: dict[str, JsonScalar] | None = None


class RuntimeConditions:
    """Current conditions expose no future schedule to schedulers."""

    def reset(self) -> None:
        """Reset any per-run state. The built-in implementation is immutable."""

    def network_links_at(
        self,
        timestamp_ms: float,
        base_links: Sequence[NetworkLink],
    ) -> tuple[NetworkLink, ...]:
        return tuple(base_links)

    def service_time_multiplier(self, timestamp_ms: float, device_id: str) -> float:
        return 1.0

    def transition_events(self) -> tuple[RuntimeConditionEvent, ...]:
        return ()


class ScenarioRuntimeConditions(RuntimeConditions):
    """Apply one validated scenario's half-open slowdown windows at runtime."""

    def __init__(
        self,
        scenario: ResearchScenarioConfig,
        devices: DeviceCatalog,
        network: NetworkCatalog,
    ) -> None:
        self.scenario = scenario
        self._devices = {device.id: device for device in devices.devices}
        self._links = {link.link_id: link for link in network.links}
        if (
            scenario.network_slowdown is not None
            and scenario.network_slowdown.link_id not in self._links
        ):
            raise ValueError(
                f"scenario references unknown link {scenario.network_slowdown.link_id}"
            )
        if (
            scenario.device_slowdown is not None
            and scenario.device_slowdown.device_id not in self._devices
        ):
            raise ValueError(
                f"scenario references unknown device {scenario.device_slowdown.device_id}"
            )

    def network_links_at(
        self,
        timestamp_ms: float,
        base_links: Sequence[NetworkLink],
    ) -> tuple[NetworkLink, ...]:
        slowdown = self.scenario.network_slowdown
        if (
            slowdown is None
            or timestamp_ms < slowdown.start_time_ms
            or timestamp_ms >= slowdown.end_time_ms
        ):
            return tuple(base_links)
        return tuple(
            link.model_copy(
                update={
                    "base_latency_ms": (link.base_latency_ms * slowdown.base_latency_multiplier),
                    "bandwidth_mbps": link.bandwidth_mbps * slowdown.bandwidth_multiplier,
                }
            )
            if link.link_id == slowdown.link_id
            else link
            for link in base_links
        )

    def service_time_multiplier(self, timestamp_ms: float, device_id: str) -> float:
        slowdown = self.scenario.device_slowdown
        if (
            slowdown is not None
            and slowdown.device_id == device_id
            and slowdown.start_time_ms <= timestamp_ms < slowdown.end_time_ms
        ):
            return slowdown.service_time_multiplier
        return 1.0

    def transition_events(self) -> tuple[RuntimeConditionEvent, ...]:
        network_slowdown = self.scenario.network_slowdown
        if network_slowdown is not None:
            link = self._links[network_slowdown.link_id]
            details: dict[str, JsonScalar] = {
                "link_id": link.link_id,
                "original_base_latency_ms": link.base_latency_ms,
                "temporary_base_latency_ms": (
                    link.base_latency_ms * network_slowdown.base_latency_multiplier
                ),
                "original_bandwidth_mbps": link.bandwidth_mbps,
                "temporary_bandwidth_mbps": (
                    link.bandwidth_mbps * network_slowdown.bandwidth_multiplier
                ),
                "start_time_ms": network_slowdown.start_time_ms,
                "end_time_ms": network_slowdown.end_time_ms,
            }
            return (
                RuntimeConditionEvent(
                    timestamp_ms=network_slowdown.start_time_ms,
                    event_type=EventType.NETWORK_SLOWDOWN_STARTED,
                    details=details,
                ),
                RuntimeConditionEvent(
                    timestamp_ms=network_slowdown.end_time_ms,
                    event_type=EventType.NETWORK_SLOWDOWN_ENDED,
                    details=details,
                ),
            )

        device_slowdown = self.scenario.device_slowdown
        if device_slowdown is not None:
            device = self._devices[device_slowdown.device_id]
            details = {
                "original_service_time_multiplier": 1.0,
                "temporary_service_time_multiplier": (device_slowdown.service_time_multiplier),
                "configured_device_speed_multiplier": device.speed_multiplier,
                "start_time_ms": device_slowdown.start_time_ms,
                "end_time_ms": device_slowdown.end_time_ms,
            }
            return (
                RuntimeConditionEvent(
                    timestamp_ms=device_slowdown.start_time_ms,
                    event_type=EventType.DEVICE_SLOWDOWN_STARTED,
                    device_id=device.id,
                    details=details,
                ),
                RuntimeConditionEvent(
                    timestamp_ms=device_slowdown.end_time_ms,
                    event_type=EventType.DEVICE_SLOWDOWN_ENDED,
                    device_id=device.id,
                    details=details,
                ),
            )
        return ()
