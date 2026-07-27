import type { Env } from "./kv";
import { runDeliver, type DeliverBody } from "./deliver-logic";

/**
 * Durable Object: single-threaded mailbox per idempotency key (or per device).
 * Serializes /v1/deliver so concurrent requests for the same key cannot both
 * pass KV reserve-then-send (KV has no CAS).
 */
export class DeliverGate {
  constructor(
    private readonly state: DurableObjectState,
    private readonly env: Env,
  ) {}

  async fetch(request: Request): Promise<Response> {
    if (request.method !== "POST") {
      return Response.json({ error: "method not allowed" }, { status: 405 });
    }

    // DO execution is single-threaded per object; blockConcurrencyWhile
    // still helps if storage ops interleave with awaits.
    return this.state.blockConcurrencyWhile(async () => {
      let body: DeliverBody;
      try {
        body = (await request.json()) as DeliverBody;
      } catch {
        return Response.json({ error: "invalid JSON body" }, { status: 400 });
      }

      const outcome = await runDeliver({
        env: this.env,
        authorizationHeader: request.headers.get("Authorization"),
        idempotencyHeader: request.headers.get("Idempotency-Key"),
        body,
      });

      return Response.json(outcome.body, { status: outcome.status });
    });
  }
}
