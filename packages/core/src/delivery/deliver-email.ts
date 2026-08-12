import type { HttpClient, StorageAdapter } from "../core/adapters";
import type { Logger } from "../services/logger";
import {
  isCancellationError,
  throwIfCancelled,
} from "../services/cancellation";
import type { EmailSettings, OutputSettings } from "../settings/types";
import {
  claimAutomaticDelivery,
  finalizeAutomaticDelivery,
  markAutomaticDeliveryAttemptStarted,
  releaseAutomaticDeliveryBeforeAttempt,
  type DeliveryClaimHandle,
} from "./delivery-state";
import {
  HostedDeliveryError,
  hostedPayloadFromRendered,
  sendViaHosted,
} from "./hosted";
import {
  formatResendFrom,
  ResendSendError,
  sendViaResend,
  type SendViaResendOptions,
} from "./resend";
import {
  renderEmailHtml,
  renderEmailSubject,
  renderEmailText,
} from "./email-render";
import type {
  DailyDigest,
  DeliverEmailResult,
  EmailDeliveryChannel,
  EmailDeliveryReason,
} from "./types";
import {
  EMAIL_DELIVERY_CHANNEL,
  EMAIL_HOSTED_CHANNEL,
  OFFICIAL_DELIVERY_AVAILABLE,
} from "./types";

export const RESEND_API_KEY_ENV = "ARXIV_DAILY_RESEND_API_KEY";

/**
 * Resend's documented test/onboarding sender. Used when the user leaves From
 * empty (quick setup: To + API key only). Typically only delivers to the
 * Resend account inbox until a custom domain is verified.
 */
export const RESEND_QUICK_FROM_EMAIL = "onboarding@resend.dev";

export const RESEND_DEFAULT_FROM_NAME = "arXiv Daily";

export interface DeliverDailyEmailDeps {
  storage: StorageAdapter;
  http: HttpClient;
  output: OutputSettings;
  email: EmailSettings;
  /** Resolved API key (settings and/or env). */
  apiKey?: string;
  logger?: Pick<Logger, "info" | "warn" | "error" | "debug">;
  now?: () => Date;
  /** Force send even if already delivered (test-send / explicit resend). */
  force?: boolean;
  maxAttempts?: number;
  baseDelayMs?: number;
  sleep?: SendViaResendOptions["sleep"];
  signal?: AbortSignal;
}

export function resolveResendApiKey(
  email: Pick<EmailSettings, "apiKey"> | undefined,
  env: Record<string, string | undefined> = {},
): string {
  const fromEnv = env[RESEND_API_KEY_ENV]?.trim() ?? "";
  if (fromEnv) return fromEnv;
  return email?.apiKey?.trim() ?? "";
}

/** From address: custom when set, otherwise Resend quick-test sender. */
export function resolveResendFromEmail(
  email: Pick<EmailSettings, "fromEmail"> | undefined,
): string {
  const custom = email?.fromEmail?.trim() ?? "";
  return custom || RESEND_QUICK_FROM_EMAIL;
}

export function resolveResendFromName(
  email: Pick<EmailSettings, "fromName"> | undefined,
): string {
  const custom = email?.fromName?.trim() ?? "";
  return custom || RESEND_DEFAULT_FROM_NAME;
}

export function resolveEmailDeliveryMode(
  email: EmailSettings | undefined,
): "self" | "hosted" {
  return email?.mode === "hosted" ? "hosted" : "self";
}

/**
 * Credentials for a send (test or auto).
 * Self: To + Resend API key (From optional).
 * Hosted: To + token; service must be online (Beta).
 */
export function isEmailCredentialsReady(
  email: EmailSettings | undefined,
  apiKey?: string,
): { ok: true } | { ok: false; reason: EmailDeliveryReason } {
  if (!email?.to?.trim()) {
    return { ok: false, reason: "recipient_missing" };
  }
  const mode = resolveEmailDeliveryMode(email);
  if (mode === "hosted") {
    if (!OFFICIAL_DELIVERY_AVAILABLE) {
      return {
        ok: false,
        reason: "official_delivery_unavailable",
      };
    }
    if (!email.hostedToken?.trim()) {
      return {
        ok: false,
        reason: "verification_token_missing",
      };
    }
    return { ok: true };
  }
  if (!(apiKey ?? email.apiKey)?.trim()) {
    return { ok: false, reason: "resend_api_key_missing" };
  }
  return { ok: true };
}

/** Auto-send path: enabled flag + credentials for active mode. */
export function isEmailDeliveryConfigured(
  email: EmailSettings | undefined,
  apiKey?: string,
): { ok: true } | { ok: false; reason: EmailDeliveryReason } {
  if (!email?.enabled) {
    return { ok: false, reason: "email_delivery_disabled" };
  }
  return isEmailCredentialsReady(email, apiKey);
}

/**
 * Deliver a daily digest via Resend when enabled and not already delivered.
 * Never throws for provider/state failures — returns a result and logs.
 * Callers must not map failures onto pipeline run-state.
 */
export async function deliverDailyEmailIfEnabled(
  digest: DailyDigest,
  deps: DeliverDailyEmailDeps,
): Promise<DeliverEmailResult> {
  const email = deps.email;
  const apiKey = (deps.apiKey ?? email.apiKey ?? "").trim();
  const configured = deps.force
    ? isEmailCredentialsReady(email, apiKey)
    : isEmailDeliveryConfigured(email, apiKey);
  if (!configured.ok) return { kind: "disabled", reason: configured.reason };

  const recipient = email.to.trim();
  const now = deps.now ?? (() => new Date());
  const mode = resolveEmailDeliveryMode(email);
  const channel: EmailDeliveryChannel =
    mode === "hosted" ? EMAIL_HOSTED_CHANNEL : EMAIL_DELIVERY_CHANNEL;
  let subject: string;
  let html: string;
  let text: string;
  let idempotencyKey: string;
  try {
    subject = renderEmailSubject(digest);
    html = renderEmailHtml(digest);
    text = renderEmailText(digest);
    idempotencyKey = deps.force
      ? await testDeliveryIdempotencyKey()
      : await automaticDeliveryIdempotencyKey(digest.date, recipient);
  } catch {
    return { kind: "failed", reason: "email_render_failed", attempts: 0 };
  }

  let claim: DeliveryClaimHandle | undefined;
  if (!deps.force) {
    const claimed = await claimAutomaticDelivery(deps.storage, deps.output, {
      date: digest.date,
      recipient,
      channel,
      now: now(),
    });
    if (claimed.kind === "blocked") {
      deps.logger?.info(`email: skip ${digest.date} (${claimed.reason})`);
      return { kind: "skipped", reason: claimed.reason };
    }
    if (claimed.kind === "failed") {
      deps.logger?.error(`email: cannot claim ${digest.date}: ${claimed.reason}`);
      return { kind: "failed", reason: claimed.reason, attempts: 0 };
    }
    claim = claimed;
  }

  let providerInvoked = false;
  let claimNamespaceLost = false;
  let attemptMarked = false;
  const beforeProviderAttempt = async () => {
    throwIfCancelled(deps.signal);
    if (claim && !attemptMarked) {
      await markAutomaticDeliveryAttemptStarted(
        deps.storage,
        deps.output,
        claim,
        now(),
      );
      attemptMarked = true;
    }
    throwIfCancelled(deps.signal);
  };
  const onProviderInvocation = () => {
    try {
      if (claim && !claim.namespaceGuard) {
        throw new Error("delivery claim namespace guard is unavailable");
      }
      claim?.namespaceGuard?.assertCurrent();
    } catch (error) {
      claimNamespaceLost = true;
      throw error;
    }
    providerInvoked = true;
  };
  try {
    throwIfCancelled(deps.signal);

    let sent: { attempts: number };
    if (mode === "hosted") {
      const hostedReq = hostedPayloadFromRendered(
        recipient,
        digest,
        { subject, html, text },
        idempotencyKey,
      );
      sent = await sendViaHosted({
        http: deps.http,
        baseUrl: email.hostedBaseUrl,
        token: (email.hostedToken ?? "").replace(/\s+/g, ""),
        request: hostedReq,
        signal: deps.signal,
        beforeProviderAttempt,
        onProviderInvocation,
      });
    } else {
      const from = formatResendFrom(
        resolveResendFromEmail(email),
        resolveResendFromName(email),
      );
      sent = await sendViaResend({
        http: deps.http,
        apiKey,
        idempotencyKey,
        payload: { from, to: recipient, subject, html, text },
        maxAttempts: deps.maxAttempts,
        baseDelayMs: deps.baseDelayMs,
        sleep: deps.sleep,
        signal: deps.signal,
        beforeProviderAttempt,
        onProviderInvocation,
      });
    }

    if (claim) {
      try {
        await finalizeAutomaticDelivery(deps.storage, deps.output, {
          ...claim,
          outcome: "delivered",
          attempts: sent.attempts,
          now: now(),
        });
      } catch {
        deps.logger?.error("email: delivery_state_update_failed");
        return {
          kind: "delivered_unrecorded",
          reason: "delivery_state_update_failed",
          attempts: sent.attempts,
        };
      }
    }

    deps.logger?.info(
      `email: delivered date=${digest.date} channel=${channel}` +
        (deps.force ? " mode=test" : " mode=automatic"),
    );
    return {
      kind: "delivered",
      attempts: sent.attempts,
    };
  } catch (error) {
    if (!providerInvoked) {
      if (claimNamespaceLost) {
        deps.logger?.error("email: delivery_claim_storage_failed");
        return {
          kind: "ambiguous",
          reason: "delivery_claim_storage_failed",
          attempts: 0,
        };
      }
      const reason: EmailDeliveryReason = isCancellationError(error)
        ? "cancelled_before_provider_attempt"
        : "provider_not_invoked";
      if (claim) {
        try {
          await releaseAutomaticDeliveryBeforeAttempt(
            deps.storage,
            deps.output,
            claim,
            reason,
            now(),
          );
        } catch {
          deps.logger?.error("email: delivery_claim_storage_failed");
          return {
            kind: "ambiguous",
            reason: "delivery_claim_storage_failed",
            attempts: 0,
          };
        }
      }
      deps.logger?.warn(`email: ${reason} date=${digest.date}`);
      return { kind: "failed", reason, attempts: 0 };
    }

    const attempts = providerAttempts(error, deps.maxAttempts);
    const ambiguous = isCancellationError(error) || providerOutcomeAmbiguous(error);
    const reason: EmailDeliveryReason = ambiguous
      ? "provider_outcome_ambiguous"
      : "provider_definitive_rejection";
    if (claim) {
      try {
        await finalizeAutomaticDelivery(deps.storage, deps.output, {
          ...claim,
          outcome: ambiguous ? "ambiguous" : "failed",
          attempts,
          errorCode: reason,
          now: now(),
        });
      } catch {
        deps.logger?.error("email: delivery_state_update_failed");
        if (!ambiguous) {
          return {
            kind: "ambiguous",
            reason: "delivery_state_update_failed",
            attempts,
          };
        }
      }
    }

    deps.logger?.warn(`email: ${reason} date=${digest.date}`);
    return ambiguous
      ? { kind: "ambiguous", reason, attempts }
      : { kind: "failed", reason, attempts };
  } finally {
    await claim?.namespaceGuard?.release().catch(() => undefined);
  }
}

export async function automaticDeliveryIdempotencyKey(
  date: string,
  recipient: string,
): Promise<string> {
  return `arxiv-daily:auto:${await sha256Hex(
    `${date}\u0000${recipient.trim().toLowerCase()}`,
  )}`;
}

export async function testDeliveryIdempotencyKey(): Promise<string> {
  return `arxiv-daily:test:${crypto.randomUUID().replace(/-/g, "")}`;
}

function providerAttempts(error: unknown, maxAttempts?: number): number {
  if (error instanceof ResendSendError) return error.attempts;
  if (error instanceof HostedDeliveryError) return 1;
  return maxAttempts ?? 1;
}

function providerOutcomeAmbiguous(error: unknown): boolean {
  if (error instanceof ResendSendError) return error.ambiguous;
  if (error instanceof HostedDeliveryError) {
    if (
      error.status === undefined ||
      error.status === 408 ||
      error.status === 409 ||
      error.status >= 500
    ) {
      return true;
    }
    if ([400, 401, 403, 404, 422, 429].includes(error.status)) return false;
    return true;
  }
  return true;
}

async function sha256Hex(input: string): Promise<string> {
  const hash = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(input),
  );
  return Array.from(new Uint8Array(hash), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

/** Build a small sample digest for test-send without a pipeline run. */
export function sampleDailyDigest(input: {
  date: string;
  language?: DailyDigest["summaryLanguage"];
  categories?: string;
  dailyPath?: string;
  toName?: string;
}): DailyDigest {
  const language = input.language === "en" ? "en" : "zh";
  // Intentional TeX-like strings so Send test exercises emailProse math softening.
  const mathTitle =
    language === "en"
      ? "Math probe: $E=mc^2$, $N_{\\rm side}=2048$, $\\theta\\leq\\pi$"
      : "公式显示探测：$E=mc^2$，$N_{\\rm side}=2048$，$\\theta\\leq\\pi$";
  const fields =
    language === "en"
      ? {
          coreProblem:
            "Can mail clients read summaries with inline math such as $\\alpha\\approx 0.3$ and $\\Omega_m$?",
          keyMethod:
            "Deterministic emailProse (not a second LLM pass): strip math delimiters; simplify $\\frac{a}{b}$, $\\leq$, $\\simeq$.",
          mainResult:
            "Expect α≈0.3, Ω_m, (a)/(b), θ≤π — not raw $…$ or leftover command names.",
          whyRelevant:
            "Probes: $\\Delta\\chi^2$, $H_0\\simeq 70\\,\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}$, $\\sum_i x_i$.",
          limitations:
            "No MathJax in mail; complex layouts stay imperfect. Vault daily keeps full Markdown math.",
        }
      : {
          coreProblem:
            "邮件客户端能否读懂含公式的摘要，例如 $\\alpha\\approx 0.3$ 与 $\\Omega_m$？",
          keyMethod:
            "确定性 emailProse（不另调模型）：去掉数学定界符，简化 $\\frac{a}{b}$、$\\leq$、$\\simeq$ 等。",
          mainResult:
            "期望 α≈0.3、Ω_m、(a)/(b)、θ≤π，而不是残留 $…$ 或 mathrm/simeq 字样。",
          whyRelevant:
            "探测项：$\\Delta\\chi^2$、$H_0\\simeq 70\\,\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}$、$\\sum_i x_i$。",
          limitations:
            "邮件无 MathJax，复杂排版仍会打折；vault 日报仍保留完整科学 Markdown 公式。",
        };
  return {
    date: input.date,
    summaryLanguage: language,
    categories: input.categories ?? "astro-ph",
    dailyPath: input.dailyPath ?? `arxiv-daily/daily/${input.date}.md`,
    paperCount: 1,
    topics: [
      {
        tag: "sample",
        name: language === "en" ? "Sample topic (math probe)" : "示例主题（公式探测）",
        papers: [
          {
            id: "2601.00001",
            title: mathTitle,
            authors: input.toName?.trim() || "arXiv Daily",
            topicTag: "sample",
            // Intentionally long; email renderer must omit this block.
            sourceSections:
              language === "en"
                ? "Abstract, 1 Introduction, 2 Methods, 2.1 Setup, 3 Results, 4 Conclusions"
                : "摘要, 1 引言, 2 方法, 2.1 设置, 3 结果, 4 结论",
            absUrl: "https://arxiv.org/abs/2601.00001",
            pdfUrl: "https://arxiv.org/pdf/2601.00001",
            kind: "structured",
            fields: {
              coreProblem: fields.coreProblem,
              keyMethod: fields.keyMethod,
              mainResult: fields.mainResult,
              whyRelevant: fields.whyRelevant,
              limitations: fields.limitations,
            },
          },
        ],
      },
    ],
  };
}
