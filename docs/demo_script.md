# EdgeWeaver Demonstration Script

Target: a short manually recorded command-line and artifact walkthrough. Do not present simulated
values as hardware measurements.

1. **Show measured model profiles.** Open `artifacts/profiles/*.json` and
   `profiling_environment.json`. Point out logistic regression's 95.49% accuracy and 0.246 ms mean
   local latency, and explain that accuracy/F1/size/latency were measured on this one computer.

2. **Show simulated devices and links.** Open `configs/devices.yaml` and `configs/network.yaml`.
   Explain the three speed/power multipliers, compatibility lists, queue capacities, and simplified
   remote-link assumptions. State that these and normalized energy are simulated.

3. **Run Fastest Device on Bursty with seed 5.** Use:

   ```bash
   python scripts/run_simulation.py --scenario bursty --scheduler fastest_device --seed 5
   ```

   Show the saved trace and event/summary paths. Inspect `request-00080`: it uses the isolated-fast
   edge server and waits 0.110 ms. Be explicit that this is measurable queueing but not meaningful
   overload—the run still meets all deadlines—so H1 is not supported.

4. **Replay the identical burst trace with EdgeWeaver.** Use the trace emitted above:

   ```bash
   python scripts/run_simulation.py --scenario bursty --scheduler edgeweaver --seed 5 \
     --trace artifacts/workload_traces/bursty-seed-5.json
   ```

   Compare the trace ID/requests, decision reason, local assignment, and timing components. Explain
   that EdgeWeaver selects logistic regression for minimum feasible estimated energy, not because a
   deadline forced model switching.

5. **Show the real separating failure.** Open
   `experiments/phase7/results/failure_cases.json` and inspect Network Slowdown run
   `core-network_slowdown-fastest_device-seed-1`, request `request-00028`. Show 224.566 ms network
   time, the missed absolute deadline, and feasible mobile/gateway candidates.

6. **Compare aggregate outcomes.** Open the deadline, latency, estimated-energy, utilization, and
   model-selection figures in `experiments/phase7/figures/`. Highlight Network Slowdown means:
   EdgeWeaver/MCT 100%, Round Robin 93.66%, Fastest Device 79.90% deadline satisfaction.

7. **Show the ablations honestly.** Open `edgeweaver_ablations.png`. Full and no-switch EdgeWeaver
   are identical during Bursty because logistic regression is already most accurate and selected
   100%. Full and no-update EdgeWeaver are identical during Device Slowdown because both stay on
   mobile while the edge server slows. These hypotheses are inconclusive, not hidden failures.

8. **Show a simpler policy matching EdgeWeaver.** MCT and EdgeWeaver have identical aggregate
   assignments and outcomes throughout this configuration. Use this as the requested case where a
   simpler policy is equally effective.

9. **Finish with reproducibility.** Run:

   ```bash
   python scripts/run_experiments.py --validate-only
   python scripts/generate_research_outputs.py
   ```

   State that the matrix contains 80 core and 10 ablation runs over 20 paired traces. Close with the
   limitations: one physical profiling computer, simulated devices/network, normalized estimated
   energy, one ML domain, simplified system behavior, and no production-edge claim.
