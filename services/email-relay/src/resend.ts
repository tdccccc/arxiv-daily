import type { Env } from "./kv";

export class ResendProviderError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly ambiguous: boolean,
  ) {
    super(message);
    this.name = "ResendProviderError";
  }
}

export class ResendTransportError extends Error {
  constructor(message: string, options: ErrorOptions = {}) {
    super(message, options);
    this.name = "ResendTransportError";
  }
}

export async function sendResendEmail(
  env: Env,
  opts: {
    to: string;
    subject: string;
    html: string;
    text: string;
    idempotencyKey?: string;
  },
): Promise<void> {
  const from = env.FROM_NAME
    ? `${env.FROM_NAME.replace(/"/g, "")} <${env.FROM_EMAIL}>`
    : env.FROM_EMAIL;
  const idempotencyKey = opts.idempotencyKey?.trim();
  if (idempotencyKey && idempotencyKey.length > 128) {
    throw new Error("provider Idempotency-Key exceeds 128 characters");
  }

  let res: Response;
  try {
    res = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.RESEND_API_KEY}`,
        "Content-Type": "application/json",
        ...(idempotencyKey ? { "Idempotency-Key": idempotencyKey } : {}),
      },
      body: JSON.stringify({
        from,
        to: [opts.to],
        subject: opts.subject,
        html: opts.html,
        text: opts.text,
      }),
    });
  } catch (error) {
    throw new ResendTransportError("Resend transport outcome is unknown", {
      cause: error,
    });
  }

  let bodyText: string;
  try {
    bodyText = await res.text();
  } catch (error) {
    throw new ResendTransportError("Resend response outcome is unknown", {
      cause: error,
    });
  }
  if (!res.ok) {
    const ambiguous = ![400, 401, 403, 404, 422, 429].includes(res.status);
    // Provider bodies may contain recipient or payload fragments. Keep them out
    // of Worker responses and logs; status is sufficient for classification.
    throw new ResendProviderError(
      `Resend rejected the request (HTTP ${res.status})`,
      res.status,
      ambiguous,
    );
  }

  try {
    const parsed = JSON.parse(bodyText) as { id?: unknown };
    if (typeof parsed.id !== "string" || !parsed.id.trim()) {
      throw new Error("missing provider message id");
    }
    // The provider ID proves the success body shape, then is discarded. Relay
    // responses, ledger records, and logs expose only the content-free contract.
  } catch (error) {
    throw new ResendTransportError("Resend success response is invalid", {
      cause: error,
    });
  }
}
