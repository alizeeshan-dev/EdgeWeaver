export type PageId = "overview" | "run" | "inspector" | "compare" | "failures";

export interface RunPreset {
  scenario: Scenario["scenario_id"];
  scheduler: Scheduler["id"];
  seed: number;
  guided?: boolean;
}

export interface Device {
  id: string;
  speed_multiplier: number;
  active_power_units: number;
  queue_capacity: number;
  processing_capacity: number;
  supported_models: string[];
}

export interface NetworkLink {
  link_id: string;
  source_device_id: string;
  destination_device_id: string;
  base_latency_ms: number;
  bandwidth_mbps: number;
}

export interface ProjectPayload {
  name: string;
  subtitle: string;
  research_question: string;
  scope: string;
  devices: Device[];
  network_links: NetworkLink[];
  pipeline: string[];
  study: { core_runs: number; ablation_runs: number; paired_trace_groups: number };
  measurement_boundary: { measured: string[]; simulated: string[] };
}

export interface ModelProfile {
  model_id: string;
  display_name: string;
  computational_role: "light" | "balanced" | "heavy";
  accuracy: number;
  macro_f1: number;
  model_size_bytes: number;
  local_latency_ms_mean: number;
  local_latency_ms_p95: number;
}

export interface Scenario {
  scenario_id: "normal" | "bursty" | "network_slowdown" | "device_slowdown";
  description: string;
  simulation_duration_ms: number;
  arrival: { base_rate_per_second: number; bursts: unknown[] };
}

export interface Scheduler {
  id: "round_robin" | "fastest_device" | "min_completion" | "edgeweaver";
  name: string;
  description: string;
  accent: string;
}

export interface RunListItem {
  run_id: string;
  scenario_id: string;
  scheduler_name: string;
  seed: number;
  completed_requests: number;
  deadline_satisfaction_rate: number;
  source: "phase7" | "interactive";
}

export interface Candidate {
  device_id: string;
  model_id: string;
  model_accuracy: number;
  compatible: boolean;
  accuracy_eligible: boolean;
  network_available: boolean;
  eligible: boolean;
  rejection_reason: string | null;
  queue_admissible: boolean;
  predicted_upload_time_ms: number;
  predicted_queue_wait_ms: number;
  predicted_inference_time_ms: number;
  predicted_return_time_ms: number;
  scheduler_overhead_ms: number;
  predicted_completion_ms: number;
  predicted_energy_units: number;
  expected_to_meet_deadline: boolean;
}

export interface Assignment {
  request_id: string;
  device_id: string;
  model_id: string;
  predicted_completion_ms: number;
  predicted_energy_units: number;
  predicted_model_accuracy: number;
  expected_to_meet_deadline: boolean;
  decision_reason: string;
  candidates: Candidate[];
}

export interface TraceRequest {
  request_id: string;
  arrival_time_ms: number;
  deadline_ms: number;
  minimum_accuracy: number;
  input_size_bytes: number;
  source_device_id: string;
  true_label: number;
  true_label_name: string;
  feature_vector_id: number;
}

export interface Execution {
  request_id: string;
  status: "completed" | "rejected";
  assignment: Assignment;
  arrival_time_ms: number;
  absolute_deadline_ms: number;
  upload_time_ms: number;
  queue_wait_ms: number;
  inference_time_ms: number;
  return_time_ms: number;
  scheduler_overhead_ms: number;
  completion_time_ms: number | null;
  end_to_end_latency_ms: number | null;
  deadline_met: boolean;
  actual_prediction: number | null;
  actual_prediction_name: string;
  prediction_correct: boolean | null;
  energy: {
    compute_energy_units: number;
    network_energy_units: number;
    estimated_total_energy_units: number;
  };
  rejection_reason: string | null;
}

export interface ProfileUpdateEvent {
  timestamp_ms: number;
  details: {
    old_estimate_ms?: number;
    observed_inference_time_ms?: number;
    new_estimate_ms?: number;
    alpha?: number;
  };
}

export interface RequestBundle {
  request: TraceRequest;
  execution: Execution;
  profile_update: ProfileUpdateEvent | null;
}

export interface RunMetrics {
  run_id: string;
  scenario_id: string;
  scheduler_name: string;
  seed: number;
  simulation_duration_ms: number;
  total_requests: number;
  completed_requests: number;
  rejected_requests: number;
  deadline_satisfaction_rate: number;
  useful_goodput_requests_per_second: number;
  mean_end_to_end_latency_ms: number;
  p95_end_to_end_latency_ms: number;
  actual_prediction_accuracy: number;
  estimated_energy_per_completed_request_units: number;
  device_utilization: Array<{ device_id: string; utilization: number }>;
  model_selection_distribution: Array<{
    model_id: string;
    model_role: string;
    assigned_requests: number;
    percentage: number;
  }>;
}

export interface RunPayload {
  run: {
    run_id: string;
    scheduler_name: string;
    random_seed: number;
    request_count: number;
    completed_requests: number;
    rejected_requests: number;
  };
  summary: RunMetrics;
  trace: {
    format_version?: string;
    trace_id?: string;
    dataset_name?: string;
    dataset_split?: string;
    seed?: number;
    generation?: Record<string, unknown>;
  };
  requests: RequestBundle[];
}

export interface SimulationEvent {
  sequence: number;
  timestamp_ms: number;
  event_type: string;
  request_id: string | null;
  device_id: string | null;
  model_id: string | null;
  details: Record<string, unknown>;
}

export interface EventPayload {
  format_version: string;
  events: SimulationEvent[];
}

export interface AggregateRow {
  scenario_id: string;
  variant_id: string;
  scheduler_name: string;
  metric: string;
  dimension: string;
  unit: string;
  run_count: number;
  mean: number;
  sample_std: number;
  seed_values: Array<{ seed: number; value: number }>;
}

export interface ExperimentResults {
  rows: AggregateRow[];
  findings_markdown: string;
  figures: string[];
}

export interface FailureCase {
  case_id: string;
  title: string;
  observed: boolean;
  evidence: Record<string, unknown> | null;
  note: string;
}

export interface FailureCasesPayload {
  format_version: string;
  cases: FailureCase[];
}
