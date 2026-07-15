import { describe, it, expect } from "vitest";
import { RunLock } from "../src/services/run-lock";

describe("RunLock", () => {
  it("first acquire succeeds", () => {
    const lock = new RunLock();
    expect(lock.tryAcquire("2026-05-11")).toBe(true);
  });

  it("second acquire on same key fails", () => {
    const lock = new RunLock();
    expect(lock.tryAcquire("2026-05-11")).toBe(true);
    expect(lock.tryAcquire("2026-05-11")).toBe(false);
  });

  it("release allows re-acquire", () => {
    const lock = new RunLock();
    lock.tryAcquire("k");
    lock.release("k");
    expect(lock.tryAcquire("k")).toBe(true);
  });

  it("different keys are independent", () => {
    const lock = new RunLock();
    expect(lock.tryAcquire("a")).toBe(true);
    expect(lock.tryAcquire("b")).toBe(true);
  });

  it("withLock executes fn and releases on success", async () => {
    const lock = new RunLock();
    const result = await lock.withLock("k", async () => 42);
    expect(result).toBe(42);
    expect(lock.tryAcquire("k")).toBe(true);
  });

  it("withLock releases on error", async () => {
    const lock = new RunLock();
    await expect(
      lock.withLock("k", async () => {
        throw new Error("x");
      }),
    ).rejects.toThrow();
    expect(lock.tryAcquire("k")).toBe(true);
  });

  it("withLock returns undefined if locked", async () => {
    const lock = new RunLock();
    lock.tryAcquire("k");
    const r = await lock.withLock("k", async () => 1);
    expect(r).toBe(undefined);
  });

  it("isHeld reflects acquisition state", () => {
    const lock = new RunLock();
    expect(lock.isHeld("k")).toBe(false);
    lock.tryAcquire("k");
    expect(lock.isHeld("k")).toBe(true);
    lock.release("k");
    expect(lock.isHeld("k")).toBe(false);
  });
});
