import type { HttpClient, StorageAdapter } from "../core/adapters";
import type { Logger } from "../services/logger";
import type { EmailSettings, OutputSettings } from "../settings/types";
import {
  loadDeliveryState,
  markDelivered,
  markFailed,
  saveDeliveryState,
  shouldSendEmail,
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
import type { DailyDigest, DeliverEmailResult, EmailDeliveryChannel } from "./types";
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
): { ok: true } | { ok: false; reason: string } {
  if (!email?.to?.trim()) {
    return { ok: false, reason: "email.to is empty" };
  }
  const mode = resolveEmailDeliveryMode(email);
  if (mode === "hosted") {
    if (!OFFICIAL_DELIVERY_AVAILABLE) {
      return {
        ok: false,
        reason:
          "Official delivery (Beta) is not online yet; switch to Send yourself or wait for the service",
      };
    }
    if (!email.hostedToken?.trim()) {
      return { ok: false, reason: "hosted delivery token is missing" };
    }
    return { ok: true };
  }
  if (!(apiKey ?? email.apiKey)?.trim()) {
    return { ok: false, reason: "Resend API key is missing" };
  }
  return { ok: true };
}

/** Auto-send path: enabled flag + credentials for active mode. */
export function isEmailDeliveryConfigured(
  email: EmailSettings | undefined,
  apiKey?: string,
): { ok: true } | { ok: false; reason: string } {
  if (!email?.enabled) {
    return { ok: false, reason: "email delivery disabled" };
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
  // force (test-send) only needs credentials; auto-send also needs enabled.
  const configured = deps.force
    ? isEmailCredentialsReady(email, apiKey)
    : isEmailDeliveryConfigured(email, apiKey);
  if (!configured.ok) {
    return { kind: "disabled", reason: configured.reason };
  }

  const recipient = email.to.trim();
  const now = deps.now ?? (() => new Date());

  try {
    let state = await loadDeliveryState(deps.storage, deps.output);
    if (!deps.force && !shouldSendEmail(state, digest.date, recipient)) {
      deps.logger?.info(
        `email: skip ${digest.date} → ${recipient} (already delivered)`,
      );
      return { kind: "skipped", reason: "already delivered" };
    }

    const subject = renderEmailSubject(digest);
    const html = renderEmailHtml(digest);
    const text = renderEmailText(digest);
    const mode = resolveEmailDeliveryMode(email);
    const channel: EmailDeliveryChannel =
      mode === "hosted" ? EMAIL_HOSTED_CHANNEL : EMAIL_DELIVERY_CHANNEL;

    try {
      let sent: { providerMessageId?: string; attempts: number };
      if (mode === "hosted") {
        const hostedReq = hostedPayloadFromRendered(recipient, digest, {
          subject,
          html,
          text,
        });
        sent = await sendViaHosted({
          http: deps.http,
          token: email.hostedToken ?? "",
          request: hostedReq,
          signal: deps.signal,
        });
      } else {
        const from = formatResendFrom(
          resolveResendFromEmail(email),
          resolveResendFromName(email),
        );
        sent = await sendViaResend({
          http: deps.http,
          apiKey,
          payload: { from, to: recipient, subject, html, text },
          maxAttempts: deps.maxAttempts,
          baseDelayMs: deps.baseDelayMs,
          sleep: deps.sleep,
          signal: deps.signal,
        });
      }
      state = markDelivered(state, {
        date: digest.date,
        recipient,
        channel,
        attempts: sent.attempts,
        providerMessageId: sent.providerMessageId,
        now: now(),
      });
      await saveDeliveryState(deps.storage, deps.output, state, now());
      deps.logger?.info(
        `email: delivered ${digest.date} → ${recipient} via ${channel}` +
          (sent.providerMessageId ? ` id=${sent.providerMessageId}` : ""),
      );
      return {
        kind: "delivered",
        providerMessageId: sent.providerMessageId,
        attempts: sent.attempts,
      };
    } catch (error) {
      const attempts =
        error instanceof ResendSendError
          ? error.attempts
          : error instanceof HostedDeliveryError
            ? 1
            : deps.maxAttempts ?? 3;
      const reason =
        error instanceof Error ? error.message : String(error);
      state = markFailed(state, {
        date: digest.date,
        recipient,
        channel,
        attempts,
        lastError: reason,
        now: now(),
      });
      await saveDeliveryState(deps.storage, deps.output, state, now()).catch(
        (saveError) => {
          deps.logger?.error("email: failed to persist delivery failure state", saveError);
        },
      );
      deps.logger?.warn(`email: delivery failed for ${digest.date}: ${reason}`);
      return { kind: "failed", reason, attempts };
    }
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    deps.logger?.error(`email: unexpected delivery error for ${digest.date}`, error);
    return { kind: "failed", reason, attempts: 0 };
  }
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
  return {
    date: input.date,
    summaryLanguage: language,
    categories: input.categories ?? "astro-ph",
    dailyPath: input.dailyPath ?? `arxiv-daily/daily/${input.date}.md`,
    paperCount: 1,
    topics: [
      {
        tag: "sample",
        name: language === "en" ? "Sample topic" : "示例主题",
        papers: [
          {
            id: "2601.00001",
            title:
              language === "en"
                ? "arXiv Daily test email ($E=mc^2$)"
                : "arXiv Daily 测试邮件（$E=mc^2$）",
            authors: input.toName?.trim() || "arXiv Daily",
            topicTag: "sample",
            sourceSections: language === "en" ? "Abstract" : "摘要",
            absUrl: "https://arxiv.org/abs/2601.00001",
            pdfUrl: "https://arxiv.org/pdf/2601.00001",
            kind: "structured",
            fields: {
              coreProblem:
                language === "en"
                  ? "Verify Resend delivery path."
                  : "验证 Resend 投递通路。",
              keyMethod:
                language === "en"
                  ? "Send one sample HTML+text message."
                  : "发送一封示例 HTML+text 邮件。",
              mainResult:
                language === "en"
                  ? "Inbox receives a readable digest."
                  : "收件箱收到可读日报。",
              whyRelevant:
                language === "en"
                  ? "Confirms settings and API key."
                  : "确认设置与 API key。",
              limitations:
                language === "en"
                  ? "Not a real paper summary."
                  : "非真实论文摘要。",
            },
          },
        ],
      },
    ],
  };
}
