export function formatMs(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "—";
  if (value === 0) return "0 ms";
  return `${value.toFixed(value < 1 ? 3 : digits)} ms`;
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  return `${(value / 1024).toFixed(1)} KB`;
}

export function displayName(value: string): string {
  return value
    .replace(/-v\d+$/, "")
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function modelRole(modelId: string): string {
  if (modelId.startsWith("logistic")) return "light";
  if (modelId.startsWith("random")) return "balanced";
  if (modelId.startsWith("mlp")) return "heavy";
  return displayName(modelId);
}

export function compactRequestId(requestId: string): string {
  const digits = requestId.match(/\d+/)?.[0];
  return digits ? `R${Number(digits).toString().padStart(3, "0")}` : requestId;
}
