import { DEFAULT_SETTINGS, type PluginSettings, type DetailSelectionProfile } from "@arxiv-daily/core";
import type { CliRuntimeConfig } from "../config";

export interface ExtendedSettingsValues {
  reasoningEffort: string;
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
    reasoningEffort: !s.llm.thinkingMode ? "none" : effort,
    detailProfile: s.detailSelection.profile, linkStyle: s.output.linkStyle ?? "wikilink",
    schedule: { ...(config?.workbenchSchedule ?? DEFAULT_SETTINGS.schedule), enabled: config?.workbenchSchedule?.enabled ?? false },
    embedding: { mode: s.embedding.mode, baseUrl: safeEndpoint(s.embedding.baseUrl), model: s.embedding.model, dimension: s.embedding.dimension, apiKeyConfigured: Boolean(s.embedding.apiKey.trim()) },
    pdfParserSidecar: { enabled: s.pdfParserSidecar.enabled, capabilitiesUrl: safeEndpoint(s.pdfParserSidecar.capabilitiesUrl), parseUrl: safeEndpoint(s.pdfParserSidecar.parseUrl) },
    email: { enabled: s.email.enabled, mode: s.email.mode, to: s.email.to, fromEmail: s.email.fromEmail, fromName: s.email.fromName ?? "", apiKeyConfigured: Boolean(s.email.apiKey?.trim()), hostedTokenConfigured: Boolean(s.email.hostedToken?.trim()) },
    logLevel: s.advanced.logLevel,
  };
}
