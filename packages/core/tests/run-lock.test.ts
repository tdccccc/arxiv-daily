import { describe, it, expect, vi } from "vitest";
import { RunLock } from "../src/services/run-lock";

describe("RunLock", () => {
  it("does not enter work while its shared lock is busy", async () => {
    const acquire = vi.fn(async () => null);
    const lock = new RunLock(acquire);
    const work = vi.fn(async () => 42);
    expect(await lock.withLock("date", work)).toBeUndefined();
    expect(acquire).toHaveBeenCalledWith("date");
    expect(work).not.toHaveBeenCalled();
    expect(lock.isHeld("date")).toBe(false);
  });

  it("releases both shared and local ownership when work fails", async () => {
    const release = vi.fn(async () => {});
    const lock = new RunLock(async () => ({ release }));
    await expect(lock.withLock("date", async () => { throw new Error("work failed"); })).rejects.toThrow("work failed");
    expect(release).toHaveBeenCalledOnce();
    expect(lock.isHeld("date")).toBe(false);
  });

  it("releases shared ownership after successful work", async () => {
    const release = vi.fn(async () => {});
    const acquire = vi.fn(async () => ({ release }));
    const lock = new RunLock(acquire);
    expect(await lock.withLock("date", async () => 42)).toBe(42);
    expect(acquire).toHaveBeenCalledWith("date");
    expect(release).toHaveBeenCalledOnce();
    expect(lock.isHeld("date")).toBe(false);
  });

  it("clears local ownership when acquisition or release throws", async () => {
    const acquiring = new RunLock(async () => { throw new Error("acquire failed"); });
    const work = vi.fn(async () => 42);
    await expect(acquiring.withLock("date", work)).rejects.toThrow("acquire failed");
    expect(work).not.toHaveBeenCalled();
    expect(acquiring.isHeld("date")).toBe(false);

    const releasing = new RunLock(async () => ({
      release: async () => { throw new Error("release failed"); },
    }));
    await expect(releasing.withLock("date", work)).rejects.toThrow("release failed");
    expect(releasing.isHeld("date")).toBe(false);
  });

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
