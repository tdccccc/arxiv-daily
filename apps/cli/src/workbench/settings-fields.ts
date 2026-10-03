import { DEFAULT_SETTINGS, isDetailSelectionProfile, validateScheduleConfig, validateLocalPdfParserSidecarConfig, type PluginSettings, type DetailSelectionProfile } from "@arxiv-daily/core";
import type { CliRuntimeConfig } from "../config";
import { WorkbenchError } from "./documents";

export interface ExtendedSettingsValues {
  reasoningEffort: "none" | "low" | "medium" | "high";
  detailProfile: DetailSelectionProfile;
  linkStyle: "wikilink" | "relative";
  schedule: PluginSettings["schedule"];
  embedding: { mode: "local" | "remote"; baseUrl: string; model: string; dimension: number; apiKeyConfigured: boolean };
  pdfParserSidecar: PluginSettings["pdfParserSidecar"];
  email: { enabled: boolean; mode: "self" | "hosted"; to: string; fromEmail: string; fromName: string; apiKeyConfigured: boolean; hostedTokenConfigured: boolean };
  logLevel: PluginSettings["advanced"]["logLevel"];
}
export function safeEndpoint(input: string): string {
  if (!input) return "";
  try { const url = new URL(input); url.username = ""; url.password = ""; url.search = ""; url.hash = ""; return url.toString(); } catch { return ""; }
}
export function extendedSettings(config: CliRuntimeConfig | null): ExtendedSettingsValues {
  const s = config?.settings ?? DEFAULT_SETTINGS;
  const effort = s.llm.reasoningEffort;
  return {
    reasoningEffort: !s.llm.thinkingMode ? "none" : effort === "low" || effort === "high" ? effort : "medium",
    detailProfile: s.detailSelection.profile, linkStyle: s.output.linkStyle ?? "wikilink",
    schedule: { ...(config?.workbenchSchedule ?? DEFAULT_SETTINGS.schedule), enabled: config?.workbenchSchedule?.enabled ?? false },
    embedding: { mode: s.embedding.mode, baseUrl: safeEndpoint(s.embedding.baseUrl), model: s.embedding.model, dimension: s.embedding.dimension, apiKeyConfigured: Boolean(s.embedding.apiKey.trim()) },
    pdfParserSidecar: { enabled: s.pdfParserSidecar.enabled, capabilitiesUrl: safeEndpoint(s.pdfParserSidecar.capabilitiesUrl), parseUrl: safeEndpoint(s.pdfParserSidecar.parseUrl) },
    email: { enabled: s.email.enabled, mode: s.email.mode, to: s.email.to, fromEmail: s.email.fromEmail, fromName: s.email.fromName ?? "", apiKeyConfigured: Boolean(s.email.apiKey?.trim()), hostedTokenConfigured: Boolean(s.email.hostedToken?.trim()) },
    logLevel: s.advanced.logLevel,
  };
}
/** Patch only supplied groups so clients from earlier versions preserve new settings. */
export function patchExtendedSettings(document: Record<string, unknown>, input: Record<string, unknown>): void {
  if (input.reasoningEffort !== undefined) {
    const effort = choice(input.reasoningEffort, ["none", "low", "medium", "high"], "Reasoning effort");
    document.llm = { ...table(document.llm), thinking_mode: effort !== "none", ...(effort === "none" ? {} : { reasoning_effort: effort }) };
  }
  if (input.detailProfile !== undefined) {
    if (!isDetailSelectionProfile(input.detailProfile)) bad("Automatic detail notes 选项无效。");
    document.detail_selection = { ...table(document.detail_selection), profile: input.detailProfile };
  }
  if (input.linkStyle !== undefined) document.output = { ...table(document.output), link_style: choice(input.linkStyle, ["wikilink", "relative"], "Link style") };
  if (input.logLevel !== undefined) document.advanced = { ...table(document.advanced), log_level: choice(input.logLevel, ["debug", "info", "warn", "error"], "Log level") };
  if (input.schedule !== undefined) {
    const v = object(input.schedule, "Output & schedule");
    const schedule = { enabled: boolean(v.enabled, "Enable"), runAtLocal: string(v.runAtLocal, "Run window Start"), runUntilLocal: string(v.runUntilLocal, "Run window End"), tickIntervalMin: integer(v.tickIntervalMin, "Check every (minutes)", 1440) };
    const validation = validateScheduleConfig({ ...DEFAULT_SETTINGS, schedule });
    if (!validation.ok) bad("Run window 必须是有效的同一天起止时间，开始不能晚于结束。");
    document.workbench_schedule = { ...table(document.workbench_schedule), enabled: schedule.enabled, run_at_local: schedule.runAtLocal, run_until_local: schedule.runUntilLocal, tick_interval_min: schedule.tickIntervalMin };
  }
  if (input.embedding !== undefined) {
    const v = object(input.embedding, "Embedding"), prior = table(document.embedding);
    const baseUrl = endpoint(v.baseUrl, "Embedding API base URL");
    document.embedding = { ...prior, mode: choice(v.mode, ["local", "remote"], "Embedding"), base_url: baseUrl, model: string(v.model, "Embedding model"), dimension: integer(v.dimension, "Embedding dimension", 1000000), api_key: secret(v.apiKey, prior.api_key) };
  }
  if (input.pdfParserSidecar !== undefined) {
    const v = object(input.pdfParserSidecar, "Better PDF parser");
    const sidecar = { enabled: boolean(v.enabled, "Better PDF parser"), capabilitiesUrl: string(v.capabilitiesUrl, "Sidecar capability URL"), parseUrl: string(v.parseUrl, "Sidecar parse URL") };
    // Validate even while disabled when endpoints are present: they must never target an external host.
    const validation = validateLocalPdfParserSidecarConfig({ ...DEFAULT_SETTINGS, pdfParserSidecar: { ...sidecar, enabled: true } });
    if (!validation.ok) bad("Sidecar URL 必须使用同源的本机 HTTP loopback 地址。");
    document.pdf_parser_sidecar = { ...table(document.pdf_parser_sidecar), enabled: sidecar.enabled, capabilities_url: sidecar.capabilitiesUrl, parse_url: sidecar.parseUrl };
  }
  if (input.email !== undefined) {
    const v = object(input.email, "Email delivery"), prior = table(document.email);
    const to = string(v.to, "Your email"), from = string(v.fromEmail, "From email");
    for (const address of [to, from]) if (address && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(address)) bad("请输入有效的邮件地址。");
    document.email = { ...prior, enabled: boolean(v.enabled, "Daily auto-send"), mode: choice(v.mode, ["self", "hosted"], "How to send"), to, from_email: from, from_name: string(v.fromName, "From name"), api_key: secret(v.apiKey, prior.api_key), hosted_token: secret(v.hostedToken, prior.hosted_token) };
  }
}
function object(value: unknown, label: string): Record<string, unknown> { if (!value || typeof value !== "object" || Array.isArray(value)) bad(`${label} 格式无效。`); return value as Record<string, unknown>; }
function table(value: unknown): Record<string, unknown> { return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {}; }
function bad(message: string): never { throw new WorkbenchError(400, message); }
function choice<T extends string>(value: unknown, choices: readonly T[], label: string): T { if (typeof value !== "string" || !choices.includes(value as T)) bad(`${label} 选项无效。`); return value as T; }
function boolean(value: unknown, label: string): boolean { if (typeof value !== "boolean") bad(`${label} 格式无效。`); return value; }
function integer(value: unknown, label: string, max: number): number { if (typeof value !== "number" || !Number.isInteger(value) || value < 1 || value > max) bad(`${label} 必须是 1–${max} 的整数。`); return value; }
function string(value: unknown, label: string): string { if (typeof value !== "string" || value.length > 20000 || /[\u0000-\u001f]/u.test(value)) bad(`${label} 格式无效。`); return value.trim(); }
function secret(value: unknown, prior: unknown): string { return value === undefined || value === "" ? typeof prior === "string" ? prior : "" : string(value, "密钥") || (typeof prior === "string" ? prior : ""); }
function endpoint(value: unknown, label: string): string {
  const text = string(value, label);
  if (!text) return text;
  try { const url = new URL(text); if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) bad(`${label} 必须为无凭证的 HTTP(S) 地址。`); } catch { bad(`${label} 必须为无凭证的 HTTP(S) 地址。`); }
  return text;
}
