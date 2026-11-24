import { describe, expect, it } from "vitest";
import {
  ManualRunGate,
  manualRunGateFor,
} from "../src/services/manual-run-gate";

describe("ManualRunGate", () => {
  it("allows one run per date and releases it exactly once", () => {
    const gate = new ManualRunGate();

    const release = gate.tryBegin("2026-11-22");
    expect(release).toBeTypeOf("function");
    expect(gate.isActive("2026-11-22")).toBe(true);
    expect(gate.tryBegin("2026-11-22")).toBeUndefined();

    release?.();
    expect(gate.isActive("2026-11-22")).toBe(false);
    expect(gate.tryBegin("2026-11-22")).toBeTypeOf("function");

    release?.();
    expect(gate.isActive("2026-11-22")).toBe(true);
  });

  it("keeps different dates independent", () => {
    const gate = new ManualRunGate();

    const releaseToday = gate.tryBegin("2026-11-22");
    const releaseYesterday = gate.tryBegin("2026-11-21");

    expect(releaseToday).toBeTypeOf("function");
    expect(releaseYesterday).toBeTypeOf("function");
    expect(gate.tryBegin("2026-11-22")).toBeUndefined();
    expect(gate.tryBegin("2026-11-21")).toBeUndefined();
  });
});

describe("manualRunGateFor", () => {
  it("shares one gate across the surfaces of a plugin instance", () => {
    const plugin = {};

    const release = manualRunGateFor(plugin).tryBegin("2026-11-22");

    expect(release).toBeTypeOf("function");
    expect(manualRunGateFor(plugin).tryBegin("2026-11-22")).toBeUndefined();
  });

  it("does not leak an in-flight run into another plugin instance", () => {
    const first = {};
    const second = {};

    manualRunGateFor(first).tryBegin("2026-11-22");

    expect(manualRunGateFor(second).tryBegin("2026-11-22")).toBeTypeOf("function");
  });
});
