import { describe, expect, it, vi } from "vitest";
import { OperationRegistry } from "../src/services/operations";
import { RunCancellationService, throwIfCancelled } from "../src/services/cancellation";

describe("OperationRegistry", () => {
  it("owns unique active operations and idempotent finish", () => {
    const registry = new OperationRegistry();
    const first = registry.begin("detail-summary", "Detail", "2601.00001");
    const second = registry.begin("pdf-download", "PDF", "2601.00001");
    expect(first.id).not.toBe(second.id);
    expect(registry.snapshot()).toHaveLength(2);
    first.finish();
    first.finish();
    expect(registry.snapshot().map((item) => item.id)).toEqual([second.id]);
  });

  it("marks cancellation requested until unwind and isolates listeners", () => {
    const registry = new OperationRegistry();
    const safe = vi.fn();
    registry.subscribe(() => { throw new Error("listener failed"); });
    registry.subscribe(safe);
    const operation = registry.begin("pdf-download", "PDF");
    expect(registry.cancel(operation.id)).toBe(true);
    expect(operation.signal.aborted).toBe(true);
    expect(registry.snapshot()[0]?.cancellationRequested).toBe(true);
    expect(safe).toHaveBeenCalled();
    operation.finish();
    expect(registry.snapshot()).toEqual([]);
  });

  it("represents a scheduler batch as one user operation", () => {
    const registry = new OperationRegistry();
    const cancellation = new RunCancellationService(registry);
    const batch = cancellation.beginBatch("Pending dates");
    const first = cancellation.begin("2026-07-15", batch);
    cancellation.finish("2026-07-15");
    const second = cancellation.begin("2026-07-14", batch);
    expect(registry.snapshot()).toHaveLength(1);
    registry.cancelAll();
    expect(() => throwIfCancelled(first)).not.toThrow();
    expect(() => throwIfCancelled(second)).toThrow(/cancelled/);
    expect(batch.isCancellationRequested()).toBe(true);
    cancellation.finishBatch(batch);
    expect(registry.snapshot()).toEqual([]);
  });
});
