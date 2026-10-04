import type { HttpClient, StorageAdapter } from "../core/adapters";
import { deliverDailyEmailIfEnabled, resolveResendApiKey, sampleDailyDigest, type DeliverDailyEmailDeps } from "../delivery/deliver-email";
import { startHostedEmailVerification } from "../delivery/hosted";
import type { DeliverEmailResult } from "../delivery/types";
import { arxivCategories } from "../settings/categories";
import type { PluginSettings } from "../settings/types";
import { formatDate, todayInTz } from "../utils/time";

export interface SettingsTestEmailOptions {
  settings: PluginSettings;
  storage: StorageAdapter;
  http: HttpClient;
  logger?: DeliverDailyEmailDeps["logger"];
  date?: string;
  now?: () => Date;
  signal?: AbortSignal;
}

/** Explicit settings action: uses a sample digest and a fresh test-send key. */
export async function sendSettingsTestEmail(options: SettingsTestEmailOptions): Promise<DeliverEmailResult> {
  const { settings } = options;
  const date = options.date ?? formatDate(todayInTz((options.now ?? (() => new Date()))(), settings.arxiv.timezone));
  const digest = sampleDailyDigest({
    date,
    language: settings.output.summaryLanguage,
    categories: arxivCategories(settings.arxiv).join(", "),
    dailyPath: `${settings.output.dailyDir}/${date}.md`,
  });
  return deliverDailyEmailIfEnabled(digest, {
    storage: options.storage,
    http: options.http,
    output: settings.output,
    email: { ...settings.email, enabled: true },
    apiKey: resolveResendApiKey(settings.email),
    logger: options.logger,
    now: options.now,
    signal: options.signal,
    force: true,
  });
}

export interface SettingsEmailVerificationOptions {
  settings: Pick<PluginSettings, "email">;
  http: HttpClient;
  /** Host deployment policy: omitted uses the official endpoint. */
  baseUrl?: string;
  signal?: AbortSignal;
}

export async function requestSettingsEmailVerification(options: SettingsEmailVerificationOptions): Promise<void> {
  return startHostedEmailVerification({
    http: options.http,
    email: options.settings.email.to?.trim() ?? "",
    baseUrl: options.baseUrl,
    signal: options.signal,
  });
}
