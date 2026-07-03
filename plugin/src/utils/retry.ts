import { isCancellationError, throwIfCancelled } from "../services/cancellation";

export interface RetryOptions {
  maxAttempts: number;
  baseDelayMs: number;
  backoff?: number;
  shouldRetry?: (err: unknown, attempt: number) => boolean;
  delayMs?: (err: unknown, attempt: number, defaultWaitMs: number) => number;
  onRetry?: (err: unknown, attempt: number, waitMs: number) => void;
  signal?: AbortSignal;
}

export async function retry<T>(fn: () => Promise<T>, opts: RetryOptions): Promise<T> {
  const backoff = opts.backoff ?? 2;
  let lastError: unknown;
  for (let attempt = 1; attempt <= opts.maxAttempts; attempt++) {
    throwIfCancelled(opts.signal);
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (isCancellationError(err)) throw err;
      if (attempt >= opts.maxAttempts) break;
      if (opts.shouldRetry && !opts.shouldRetry(err, attempt)) break;
      const defaultWait = opts.baseDelayMs * Math.pow(backoff, attempt - 1);
      const wait = Math.max(
        0,
        opts.delayMs?.(err, attempt, defaultWait) ?? defaultWait,
      );
      opts.onRetry?.(err, attempt, wait);
      await sleep(wait, opts.signal);
    }
  }
  throw lastError;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  throwIfCancelled(signal);
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(done, ms);
    const onAbort = () => {
      clearTimeout(timeout);
      cleanup();
      try {
        throwIfCancelled(signal);
      } catch (e) {
        reject(e);
      }
    };
    function done() {
      cleanup();
      resolve();
    }
    function cleanup() {
      signal?.removeEventListener("abort", onAbort);
    }
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) onAbort();
  });
}
