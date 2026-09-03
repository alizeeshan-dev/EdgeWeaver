"""Command-line entry point for the deterministic Phase 1 fixture."""

import argparse
from pathlib import Path

from edgeweaver.config import load_phase1_configuration
from edgeweaver.simulation import run_fixture_simulation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the complete event log and summary as JSON.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    configuration = load_phase1_configuration(
        devices_path=PROJECT_ROOT / "configs" / "devices.yaml",
        network_path=PROJECT_ROOT / "configs" / "network.yaml",
        model_profiles_path=PROJECT_ROOT / "artifacts" / "profiles" / "synthetic_fixture.yaml",
        scenario_path=PROJECT_ROOT / "configs" / "phase1_fixture.yaml",
    )
    result = run_fixture_simulation(configuration)

    print("EdgeWeaver Phase 1 deterministic fixture")
    print("request     wait_ms  completed_ms  latency_ms  deadline_met")
    for metric in result.summary.requests:
        print(
            f"{metric.request_id:<11} {metric.queue_wait_ms:>7.1f} "
            f"{metric.completion_time_ms:>13.1f} {metric.end_to_end_latency_ms:>11.1f} "
            f"{str(metric.deadline_met):>13}"
        )
    print(
        f"Summary: {result.summary.deadlines_met}/{result.summary.total_requests} deadlines met "
        f"({result.summary.deadline_satisfaction_rate:.0%}); {len(result.events)} events"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved full result to {args.output}")


if __name__ == "__main__":
    main()
