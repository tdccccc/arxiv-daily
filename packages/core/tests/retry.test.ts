import { describe, it, expect, vi } from "vitest";
import { retry } from "../src/utils/retry";

describe("retry", () => {
  it("returns value on first success", async () => {
    const fn = vi.fn().mockResolvedValue("ok");
    const result = await retry(fn, { maxAttempts: 3, baseDelayMs: 1 });
    expect(result).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("retries on failure then succeeds", async () => {
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("boom"))
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValue("ok");
    const result = await retry(fn, { maxAttempts: 3, baseDelayMs: 1 });
    expect(result).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("throws after max attempts", async () => {
    const fn = vi.fn().mockRejectedValue(new Error("permanent"));
    await expect(retry(fn, { maxAttempts: 2, baseDelayMs: 1 })).rejects.toThrow("permanent");
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("respects shouldRetry predicate", async () => {
    const err = new Error("4xx");
    const fn = vi.fn().mockRejectedValue(err);
    await expect(
      retry(fn, {
        maxAttempts: 5,
        baseDelayMs: 1,
        shouldRetry: () => false,
      }),
    ).rejects.toThrow("4xx");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("calls onRetry with attempt and wait", async () => {
    const onRetry = vi.fn();
    const fn = vi.fn().mockRejectedValueOnce(new Error("x")).mockResolvedValue("ok");
    await retry(fn, { maxAttempts: 3, baseDelayMs: 1, onRetry });
    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(onRetry.mock.calls[0][1]).toBe(1);
  });

  it("does not start when the signal is already aborted", async () => {
    const controller = new AbortController();
    controller.abort("cancelled by test");
    const fn = vi.fn().mockResolvedValue("ok");
    await expect(
      retry(fn, {
        maxAttempts: 3,
        baseDelayMs: 1,
        signal: controller.signal,
      }),
    ).rejects.toThrow("cancelled by test");
    expect(fn).not.toHaveBeenCalled();
  });

  it("does not announce a retry when the attempt fails after cancellation", async () => {
    const controller = new AbortController();
    const onRetry = vi.fn();
    const fn = vi.fn(async () => {
      controller.abort("cancelled by test");
      throw new Error("request failed after abort");
    });

    await expect(
      retry(fn, {
        maxAttempts: 3,
        baseDelayMs: 1000,
        signal: controller.signal,
        onRetry,
      }),
    ).rejects.toThrow("cancelled by test");
    expect(fn).toHaveBeenCalledTimes(1);
    expect(onRetry).not.toHaveBeenCalled();
  });

  it("does not retry after cancellation during backoff", async () => {
    const controller = new AbortController();
    const fn = vi.fn().mockRejectedValue(new Error("boom"));
    await expect(
      retry(fn, {
        maxAttempts: 3,
        baseDelayMs: 1000,
        signal: controller.signal,
        onRetry: () => controller.abort("cancelled by test"),
      }),
    ).rejects.toThrow("cancelled by test");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("caps exponential backoff and applies jitter before sleeping", async () => {
    vi.spyOn(Math, "random").mockReturnValue(0);
    const controller = new AbortController();
    const waits: number[] = [];
    const fn = vi.fn().mockRejectedValue(new Error("boom"));

    await expect(
      retry(fn, {
        maxAttempts: 3,
        baseDelayMs: Number.MAX_VALUE,
        backoff: 10,
        signal: controller.signal,
        onRetry: (_err, _attempt, wait) => {
          waits.push(wait);
          controller.abort("stop after wait capture");
        },
      }),
    ).rejects.toThrow("stop after wait capture");

    expect(waits).toEqual([900_000]);
    expect(fn).toHaveBeenCalledTimes(1);
  });
});
