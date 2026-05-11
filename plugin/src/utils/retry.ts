export interface RetryOptions {
  maxAttempts: number;
  baseDelayMs: number;
  backoff?: number;
  shouldRetry?: (err: unknown, attempt: number) => boolean;
  onRetry?: (err: unknown, attempt: number, waitMs: number) => void;
}

export async function retry<T>(fn: () => Promise<T>, opts: RetryOptions): Promise<T> {
  const backoff = opts.backoff ?? 2;
  let lastError: unknown;
  for (let attempt = 1; attempt <= opts.maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (attempt >= opts.maxAttempts) break;
      if (opts.shouldRetry && !opts.shouldRetry(err, attempt)) break;
      const wait = opts.baseDelayMs * Math.pow(backoff, attempt - 1);
      opts.onRetry?.(err, attempt, wait);
      await new Promise((r) => setTimeout(r, wait));
    }
  }
  throw lastError;
}
