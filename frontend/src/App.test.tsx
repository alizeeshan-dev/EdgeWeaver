import { afterEach, describe, expect, it } from "vitest";

import { routeFromHash } from "./App";

afterEach(() => {
  window.location.hash = "";
});

describe("routeFromHash", () => {
  it("restores an exact inspector run and request from a deep link", () => {
    window.location.hash = "/inspector?run=core-network_slowdown-fastest_device-seed-1&request=request-00028";
    expect(routeFromHash()).toEqual({
      page: "inspector",
      inspectorRunId: "core-network_slowdown-fastest_device-seed-1",
      inspectorRequestId: "request-00028",
    });
  });

  it("restores a guided same-trace run preset and rejects malformed presets", () => {
    window.location.hash = "/run?scenario=network_slowdown&scheduler=fastest_device&seed=1&guided=1";
    expect(routeFromHash()).toEqual({
      page: "run",
      runPreset: { scenario: "network_slowdown", scheduler: "fastest_device", seed: 1, guided: true },
    });

    window.location.hash = "/run?scenario=made_up&scheduler=edgeweaver&seed=-1";
    expect(routeFromHash()).toEqual({ page: "run" });
  });
});
