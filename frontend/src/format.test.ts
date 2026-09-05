import { describe, expect, it } from "vitest";

import { compactRequestId, displayName, formatBytes, formatMs, formatPercent, modelRole } from "./format";

describe("presentation formatters", () => {
  it("formats research units without changing their meaning", () => {
    expect(formatMs(0.6150155)).toBe("0.615 ms");
    expect(formatPercent(0.954869, 2)).toBe("95.49%");
    expect(formatBytes(2244)).toBe("2.2 KB");
  });

  it("uses stable human-facing identifiers", () => {
    expect(compactRequestId("request-00071")).toBe("R071");
    expect(displayName("network_slowdown")).toBe("Network Slowdown");
    expect(modelRole("logistic-regression-v1")).toBe("light");
  });
});
