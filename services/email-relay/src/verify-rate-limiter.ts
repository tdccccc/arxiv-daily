import { hmacSha256Hex } from "./crypto";
import type { Env } from "./kv";

/**
 * Fixed-window hit counter for /v1/verify/start, one Durable Object per subject
 * (keyed hash of the IP or of the address). The whole check-and-increment is one
 * synchronous SQL section with no await inside it, so the object can never
 * interleave two requests between the read and the write. Workers KV has no
 * compare-and-swap, which is why the old KV counters were bypassable by
 * concurrent requests.
 */

export type VerifyRateLimitKind = "verify-ip" | "verify-email";

export interface VerifyRateLimitResult {
  ok: boolean;
  /** Hits counted in the current window, including this one when ok. */
  count: number;
  retryAfterSeconds: number;
}

interface HitBody {
  limit: number;
  windowSeconds: number;
}

interface WindowRow {
  startedAt: number;
  windowMs: number;
  count: number;
}

const MAX_WINDOW_SECONDS = 24 * 60 * 60;

export class VerifyRateLimiter {
  constructor(
    private readonly state: DurableObjectState,
    _env?: Env,
    private readonly now: () => number = () => Date.now(),
  ) {}

  async fetch(request: Request): Promise<Response> {
    if (request.method !== "POST") {
      return Response.json({ error: "method not allowed" }, { status: 405 });
    }
    let body: HitBody;
    try {
      body = parseHitBody(await request.json());
    } catch {
      return Response.json({ error: "invalid rate limit request" }, { status: 400 });
    }

    // No await between reading the row and writing it back: atomic by construction.
    const outcome = this.state.storage.transactionSync(() =>
      this.hit(body.limit, body.windowSeconds * 1000)
    );
    if (outcome.opened) {
      // One alarm per window deletes the row (and with it the object's storage) after
      // the window is over. Overwrite-on-new-window keeps storage bounded to a single
      // row even if this call fails or the alarm runs late.
      await this.state.storage.setAlarm(outcome.windowEnd);
    }
    return Response.json(outcome.result);
  }

  async alarm(): Promise<void> {
    const row = this.state.storage.transactionSync(() => {
      this.ensureSchema();
      return this.readRow();
    });
    if (!row || row.startedAt + row.windowMs <= this.now()) {
      await this.state.storage.deleteAll();
    } else {
      await this.state.storage.setAlarm(row.startedAt + row.windowMs);
    }
  }

  private ensureSchema(): void {
    this.state.storage.sql.exec(
      "CREATE TABLE IF NOT EXISTS verify_window (" +
        "id INTEGER PRIMARY KEY CHECK (id = 1), " +
        "started_at INTEGER NOT NULL, " +
        "window_ms INTEGER NOT NULL, " +
        "count INTEGER NOT NULL)",
    );
  }

  private readRow(): WindowRow | null {
    const rows = this.state.storage.sql
      .exec<{ started_at: number; window_ms: number; count: number }>(
        "SELECT started_at, window_ms, count FROM verify_window WHERE id = 1",
      )
      .toArray();
    const row = rows[0];
    return row
      ? { startedAt: row.started_at, windowMs: row.window_ms, count: row.count }
      : null;
  }

  private write(row: WindowRow): void {
    this.state.storage.sql.exec(
      "INSERT OR REPLACE INTO verify_window (id, started_at, window_ms, count) " +
        "VALUES (1, ?, ?, ?)",
      row.startedAt,
      row.windowMs,
      row.count,
    );
  }

  private hit(limit: number, windowMs: number): {
    result: VerifyRateLimitResult;
    windowEnd: number;
    opened: boolean;
  } {
    this.ensureSchema();
    const now = this.now();
    const row = this.readRow();
    if (!row || row.startedAt + row.windowMs <= now) {
      // First hit, or the previous window is over: overwrite the single row.
      this.write({ startedAt: now, windowMs, count: 1 });
      return {
        result: { ok: true, count: 1, retryAfterSeconds: Math.ceil(windowMs / 1000) },
        windowEnd: now + windowMs,
        opened: true,
      };
    }
    const windowEnd = row.startedAt + row.windowMs;
    const retryAfterSeconds = Math.max(1, Math.ceil((windowEnd - now) / 1000));
    if (row.count >= limit) {
      return {
        result: { ok: false, count: row.count, retryAfterSeconds },
        windowEnd,
        opened: false,
      };
    }
    this.write({ ...row, count: row.count + 1 });
    return {
      result: { ok: true, count: row.count + 1, retryAfterSeconds },
      windowEnd,
      opened: false,
    };
  }
}

function parseHitBody(raw: unknown): HitBody {
  if (!raw || typeof raw !== "object") throw new Error("body");
  const { limit, windowSeconds } = raw as Record<string, unknown>;
  if (
    typeof limit !== "number" || !Number.isInteger(limit) || limit < 1 ||
    typeof windowSeconds !== "number" || !Number.isInteger(windowSeconds) ||
    windowSeconds < 1 || windowSeconds > MAX_WINDOW_SECONDS
  ) {
    throw new Error("body");
  }
  return { limit, windowSeconds };
}

/**
 * Durable Object names show up in the Cloudflare dashboard, and the IPv4 space is small
 * enough to reverse a plain hash, so the subject is keyed with TOKEN_SECRET (HMAC-SHA256,
 * via the same crypto helper kv.ts uses for recipient identities).
 */
export async function verifyRateLimitName(
  kind: VerifyRateLimitKind,
  subject: string,
  secret: string,
): Promise<string> {
  const digest = await hmacSha256Hex(
    secret,
    `verify-rate-limit\u0000${kind}\u0000${subject}`,
  );
  return `${kind}:${digest}`;
}

/** Atomically counts one hit for `subject`; throws (fails closed) without the binding. */
export async function consumeVerifyRateLimit(
  env: Env,
  kind: VerifyRateLimitKind,
  subject: string,
  limit: number,
  windowSeconds: number,
): Promise<VerifyRateLimitResult> {
  const namespace = env.VERIFY_RATE_LIMIT;
  if (!namespace) {
    throw new Error("VERIFY_RATE_LIMIT Durable Object binding is not configured");
  }
  const name = await verifyRateLimitName(kind, subject, env.TOKEN_SECRET);
  const stub = namespace.get(namespace.idFromName(name));
  const response = await stub.fetch("https://verify-rate-limit/hit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ limit, windowSeconds }),
  });
  if (!response.ok) throw new Error("verify rate limiter unavailable");
  return (await response.json()) as VerifyRateLimitResult;
}
