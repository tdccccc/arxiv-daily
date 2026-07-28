import {
  deliverDailyEmailIfEnabled,
  isEmailCredentialsReady,
  isEmailDeliveryConfigured,
  resolveEmailDeliveryMode,
  resolveResendApiKey,
  sampleDailyDigest,
  startHostedEmailVerification,
  formatDate,
  todayInTz,
  arxivCategories,
} from "@arxiv-daily/core";
import type { HostAdapters } from "@arxiv-daily/core";
import type { CliRuntimeConfig } from "./config";
import type { CliIo } from "./main-types";

export async function emailStatus(
  config: CliRuntimeConfig,
  io: CliIo,
): Promise<number> {
  const email = config.settings.email;
  const mode = resolveEmailDeliveryMode(email);
  const apiKey = resolveResendApiKey(email, {});
  const creds = isEmailCredentialsReady(email, apiKey);
  const configured = isEmailDeliveryConfigured(email, apiKey);
  writeLine(io.stdout, `email.mode: ${mode}`);
  writeLine(io.stdout, `email.enabled: ${email.enabled}`);
  writeLine(io.stdout, `email.to: ${email.to || "(empty)"}`);
  writeLine(
    io.stdout,
    `credentials: ${creds.ok ? "ready" : `not ready (${creds.reason})`}`,
  );
  writeLine(
    io.stdout,
    `auto-send: ${configured.ok ? "would run on completed daily" : `off (${configured.reason})`}`,
  );
  return 0;
}

export async function emailTest(
  config: CliRuntimeConfig,
  host: HostAdapters,
  io: CliIo,
  dateArg?: string,
  now: () => Date = () => new Date(),
): Promise<number> {
  const date =
    dateArg ??
    formatDate(todayInTz(now(), config.settings.arxiv.timezone));
  const digest = sampleDailyDigest({
    date,
    language: config.settings.output.summaryLanguage,
    categories: arxivCategories(config.settings.arxiv).join(", "),
    dailyPath: `${config.settings.output.dailyDir}/${date}.md`,
  });
  const email = { ...config.settings.email, enabled: true };
  const result = await deliverDailyEmailIfEnabled(digest, {
    storage: host.storage,
    http: host.http,
    output: config.settings.output,
    email,
    apiKey: resolveResendApiKey(config.settings.email, {}),
    force: true,
  });
  if (result.kind === "delivered") {
    writeLine(
      io.stdout,
      `email test: delivered to ${email.to}` +
        (result.providerMessageId ? ` id=${result.providerMessageId}` : ""),
    );
    return 0;
  }
  writeLine(io.stderr, `email test: ${result.kind} (${result.reason})`);
  return 1;
}

export async function emailVerifyStart(
  config: CliRuntimeConfig,
  host: HostAdapters,
  io: CliIo,
): Promise<number> {
  const to = config.settings.email.to?.trim() ?? "";
  if (!to) {
    writeLine(io.stderr, "email.to is empty; set it in config.toml first");
    return 2;
  }
  try {
    await startHostedEmailVerification({
      http: host.http,
      email: to,
      // never pass user hosted_base_url from file per product decision
    });
    writeLine(
      io.stdout,
      `verification email requested for ${to}; open the link and paste the long code into email.hosted_token, set mode = "hosted"`,
    );
    return 0;
  } catch (e) {
    writeLine(io.stderr, `email verify-start failed: ${(e as Error).message}`);
    return 1;
  }
}

function writeLine(stream: { write(chunk: string): unknown }, line: string): void {
  stream.write(`${line}\n`);
}
