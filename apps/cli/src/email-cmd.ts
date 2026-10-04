import {
  AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE,
  isEmailCredentialsReady,
  isEmailDeliveryConfigured,
  resolveEmailDeliveryMode,
  resolveResendApiKey,
  sendSettingsTestEmail,
  requestSettingsEmailVerification,
  supportsAutomaticEmailDelivery,
} from "@arxiv-daily/core";
import type { HostAdapters } from "@arxiv-daily/core";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import type { CliRuntimeConfig } from "./config";
import type { CliIo } from "./main-types";

export async function emailStatus(
  config: CliRuntimeConfig,
  io: CliIo,
  automaticSupported = supportsAutomaticEmailDelivery(
    new NodeStorageAdapter(config.vaultRoot),
  ),
): Promise<number> {
  const email = config.settings.email;
  const mode = resolveEmailDeliveryMode(email);
  const apiKey = resolveResendApiKey(email, {});
  const creds = isEmailCredentialsReady(email, apiKey);
  const configured = isEmailDeliveryConfigured(email, apiKey);
  writeLine(io.stdout, `email.mode: ${mode}`);
  writeLine(io.stdout, `email.enabled: ${email.enabled}`);
  writeLine(io.stdout, `email.recipient: ${email.to ? "configured" : "empty"}`);
  writeLine(
    io.stdout,
    `credentials: ${creds.ok ? "ready" : `not ready (${creds.reason})`}`,
  );
  writeLine(
    io.stdout,
    `auto-send: ${
      !automaticSupported
        ? "unsupported on this host (protected delivery storage unavailable); test emails still send"
        : configured.ok
          ? "would run on completed daily"
          : `off (${configured.reason})`
    }`,
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
  const result = await sendSettingsTestEmail({
    settings: config.settings,
    storage: host.storage,
    http: host.http,
    date: dateArg,
    now,
  });
  if (
    result.kind === "delivered" ||
    result.kind === "delivered_unrecorded"
  ) {
    writeLine(
      io.stdout,
      "email test: delivered" +
        (result.kind === "delivered_unrecorded"
          ? ` (delivery record unavailable: ${result.reason})`
          : ""),
    );
    if (!supportsAutomaticEmailDelivery(host.storage)) {
      writeLine(io.stderr, `warning: ${AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE}`);
    }
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
    await requestSettingsEmailVerification({
      settings: config.settings,
      http: host.http,
      // never pass user hosted_base_url from file per product decision
    });
    writeLine(
      io.stdout,
      "verification email requested; open the link and paste the long code into email.hosted_token, set mode = \"hosted\"",
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
