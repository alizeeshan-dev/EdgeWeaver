"""Explicit factory for Phase 4 baseline schedulers."""

from edgeweaver.scheduler import Scheduler
from edgeweaver.schedulers.edgeweaver import DEFAULT_EWMA_ALPHA, EdgeWeaverScheduler
from edgeweaver.schedulers.fastest_device import FastestDeviceScheduler
from edgeweaver.schedulers.min_completion import MinimumCompletionTimeScheduler
from edgeweaver.schedulers.round_robin import RoundRobinScheduler

BASELINE_SCHEDULER_NAMES = ("round_robin", "fastest_device", "min_completion")
SCHEDULER_NAMES = (*BASELINE_SCHEDULER_NAMES, "edgeweaver")


def create_scheduler(
    name: str,
    *,
    edgeweaver_alpha: float = DEFAULT_EWMA_ALPHA,
) -> Scheduler:
    if name == "round_robin":
        return RoundRobinScheduler()
    if name == "fastest_device":
        return FastestDeviceScheduler()
    if name == "min_completion":
        return MinimumCompletionTimeScheduler()
    if name == "edgeweaver":
        return EdgeWeaverScheduler(alpha=edgeweaver_alpha)
    raise ValueError(f"unknown scheduler {name!r}; expected one of {', '.join(SCHEDULER_NAMES)}")
