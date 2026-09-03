"""Deterministic, deliberately simple network timing calculations."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from edgeweaver.domain import NetworkLink


@dataclass(frozen=True)
class NetworkTiming:
    upload_time_ms: float
    return_time_ms: float
    link_id: str | None


def transfer_time_ms(
    payload_size_bytes: int,
    link: NetworkLink,
    *,
    jitter_offset_ms: float = 0.0,
) -> float:
    """Calculate one-way base latency plus payload serialization time."""

    if payload_size_bytes < 0:
        raise ValueError("payload_size_bytes must be non-negative")
    if abs(jitter_offset_ms) > link.jitter_ms:
        raise ValueError("jitter offset exceeds configured link jitter")
    serialization_ms = payload_size_bytes * 8.0 / (link.bandwidth_mbps * 1_000.0)
    return max(0.0, link.base_latency_ms + jitter_offset_ms) + serialization_ms


def find_network_link(
    links: Sequence[NetworkLink],
    source_device_id: str,
    destination_device_id: str,
) -> NetworkLink:
    try:
        return next(
            link
            for link in links
            if link.source_device_id == source_device_id
            and link.destination_device_id == destination_device_id
        )
    except StopIteration as error:
        raise ValueError(
            f"no network link from {source_device_id} to {destination_device_id}"
        ) from error


def deterministic_jitter_ms(
    link: NetworkLink,
    *,
    random_seed: int,
    request_id: str,
    direction: Literal["upload", "return"],
) -> float:
    """Draw stable per-request jitter without depending on process execution order."""

    if link.jitter_ms == 0.0:
        return 0.0
    key = f"{random_seed}|{request_id}|{link.link_id}|{direction}".encode()
    stable_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], byteorder="big")
    return random.Random(stable_seed).uniform(-link.jitter_ms, link.jitter_ms)


def network_timing(
    *,
    source_device_id: str,
    destination_device_id: str,
    input_size_bytes: int,
    return_size_bytes: int,
    links: Sequence[NetworkLink],
    random_seed: int,
    request_id: str,
) -> NetworkTiming:
    """Return zero for local execution or both symmetric remote transfer legs."""

    if source_device_id == destination_device_id:
        return NetworkTiming(upload_time_ms=0.0, return_time_ms=0.0, link_id=None)
    link = find_network_link(links, source_device_id, destination_device_id)
    upload_jitter = deterministic_jitter_ms(
        link, random_seed=random_seed, request_id=request_id, direction="upload"
    )
    return_jitter = deterministic_jitter_ms(
        link, random_seed=random_seed, request_id=request_id, direction="return"
    )
    return NetworkTiming(
        upload_time_ms=transfer_time_ms(input_size_bytes, link, jitter_offset_ms=upload_jitter),
        return_time_ms=transfer_time_ms(return_size_bytes, link, jitter_offset_ms=return_jitter),
        link_id=link.link_id,
    )
