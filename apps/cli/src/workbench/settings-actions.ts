import { LlmClient, Logger, redactText, type HttpClient } from "@arxiv-daily/core";
import { NodeHttpClient } from "@arxiv-daily/node-runtime";
import { loadCliConfig, type CliRuntimeConfig } from "../config";
import type { CliIo } from "../main-types";
import { authorizeCliLibrary, connectCliLibrary, revokeCliLibrary, inspectCliLibraryConnection, type CliLibraryConnectionInspection } from "../library-connection-cmd";
import { runCliLibrary, type CliLibraryOptions } from "../library-cmd";
import { buildCliRuntime } from "../runtime";
import { emailTest, emailVerifyStart } from "../email-cmd";

export interface SettingsActionOptions {
  http?: HttpClient;
  library?: CliLibraryOptions;
  runLibrary?: typeof runCliLibrary;
  buildRuntime?: (config: CliRuntimeConfig) => Promise<Pick<Awaited<ReturnType<typeof buildCliRuntime>>, "host" | "dispose">>;
}
export interface SettingsActionResult { config?: CliRuntimeConfig; models?: string[]; library?: CliLibraryConnectionInspection; message?: string }
export const inspectWorkbenchLibrary = inspectCliLibraryConnection;

/** Adapter only: processing, consent and delivery remain owned by CLI/core. */
export async function performSettingsAction(config: CliRuntimeConfig, body: unknown, io: CliIo, signal: AbortSignal, options: SettingsActionOptions = {}): Promise<SettingsActionResult> {
  const request = parseAction(body);
  signal.throwIfAborted();
  const secrets = [config.settings.llm.apiKey, config.settings.embedding.apiKey, config.settings.email.apiKey ?? "", config.settings.email.hostedToken ?? ""];
  const safe = (value: string) => redactText(value, { secrets });
  const safeIo: CliIo = { stdout: { write: value => io.stdout.write(safe(value)) }, stderr: { write: value => io.stderr.write(safe(value)) } };
  const baseHttp = options.http ?? new NodeHttpClient();
  const http: HttpClient = { request: input => { signal.throwIfAborted(); return baseHttp.request({ ...input, signal: input.signal ? AbortSignal.any([signal, input.signal]) : signal }); } };
  try {
    if (request.action === "models") {
      const logger = new Logger(config.settings.advanced.logLevel);
      const client = new LlmClient(config.settings.llm, logger, http);
      logger.setSensitiveValues(secrets);
      const models = await client.fetchModels();
      signal.throwIfAborted();
      return { models };
    }
    if (request.action === "library-connect" || request.action === "library-revoke") {
      const next = request.action === "library-connect" ? await connectCliLibrary(config, request.path!) : await revokeCliLibrary(config);
      return { config: next, library: inspectWorkbenchLibrary(next) };
    }
    if (request.action === "library-build") {
      let next = await loadCliConfig({ configPath: config.configPath });
      if (next.configRevision !== config.configRevision) throw new Error("Configuration changed; reload settings before building the library");
      const inspection = inspectWorkbenchLibrary(next);
      if (inspection.status.kind !== "authorized") {
        if (!request.fingerprint || request.fingerprint !== inspection.disclosure?.authorizationFingerprint) throw new Error("Library processing requires the current disclosed fingerprint and explicit authorization");
        next = await authorizeCliLibrary(next, request.fingerprint);
      }
      const runLibrary = options.runLibrary ?? runCliLibrary;
      for (const command of ["prepare", "scan", "index"]) {
        signal.throwIfAborted();
        if (await runLibrary(next, [command], safeIo, { ...options.library, ...(options.http ? { http } : {}), signal }) !== 0) throw new Error(`Library ${command} failed; see task output`);
      }
      signal.throwIfAborted();
      return { config: next, library: inspectWorkbenchLibrary(next), message: "文献库已建立" };
    }
    const runtime = await (options.buildRuntime ?? buildCliRuntime)(config);
    try {
      signal.throwIfAborted();
      const host = { ...runtime.host, http: options.http ? http : { request: (input: Parameters<HttpClient["request"]>[0]) => runtime.host.http.request({ ...input, signal: input.signal ? AbortSignal.any([signal, input.signal]) : signal }) } };
      const code = request.action === "email-test" ? await emailTest(config, host, safeIo) : await emailVerifyStart(config, host, safeIo);
      signal.throwIfAborted();
      if (code !== 0) throw new Error(request.action === "email-test" ? "测试邮件发送失败，请查看任务输出" : "验证邮件发送失败，请查看任务输出");
      return { message: request.action === "email-test" ? "测试邮件已发送" : "验证邮件已发送，请检查收件箱" };
    } finally { runtime.dispose?.(); }
  } catch (error) {
    throw new Error(safe(error instanceof Error ? error.message : String(error)));
  }
}

function parseAction(body: unknown): { action: string; path?: string; fingerprint?: string } {
  const invalid = () => Object.assign(new Error("Invalid settings action request"), { status: 400 });
  if (!body || typeof body !== "object" || Array.isArray(body)) throw invalid();
  const value = body as Record<string, unknown>;
  if (typeof value.action !== "string" || !["models", "library-connect", "library-revoke", "library-build", "email-test", "email-verify"].includes(value.action)) throw invalid();
  const fields = ["action", ...(value.action === "library-connect" ? ["path"] : value.action === "library-build" ? ["fingerprint"] : [])];
  if (Object.keys(value).some(key => !fields.includes(key))) throw invalid();
  if (value.action === "library-connect" && (typeof value.path !== "string" || !value.path.trim() || value.path.length > 4096)) throw invalid();
  if (value.fingerprint !== undefined && (typeof value.fingerprint !== "string" || !value.fingerprint || value.fingerprint.length > 512)) throw invalid();
  return value as { action: string; path?: string; fingerprint?: string };
}
