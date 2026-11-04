import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { parse as parseToml } from "smol-toml";
import {
  DEFAULT_SETTINGS,
  arxivCategories,
  isValidMaxDailyPapers,
  decodeLibraryConnection,
  normalizeCategoryList,
  normalizeTopic,
  sanitizeDetailSelection,
  isDetailSelectionProfile,
  MAX_GENERATION_CONCURRENCY,
  validateScheduleConfig,
  validateLocalPdfParserSidecarConfig,
  sha256Hex,
  validateVaultRelativeDirectory,
  vaultRelativeDirectoriesCollide,
} from "@arxiv-daily/core";
import type {
  LinkStyle,
  PersistedLibraryConnection,
  PluginSettings,
  SummaryLanguage,
  Topic,
} from "@arxiv-daily/core";
import { resolveCliConfigPath } from "./config-path";

export type { LinkStyle } from "@arxiv-daily/core";

/** CLI-only schedule intent for OS timer install (not plugin in-process tick). */
export interface CliScheduleIntent {
  enabled: boolean;
  /** First daily fire HH:MM (machine local for cron). */
  on: string;
  /** 0 = once at `on`; >0 = every N hours from on while <= until. */
  intervalHours: number;
  until: string;
  weekdaysOnly: boolean;
}

export const DEFAULT_CLI_SCHEDULE: CliScheduleIntent = {
  enabled: false,
  on: "09:30",
  intervalHours: 0,
  until: "18:00",
  weekdaysOnly: true,
};

export interface CliRuntimeConfig {
  settings: PluginSettings;
  vaultRoot: string;
  cacheDir: string;
  linkStyle: LinkStyle;
  configPath: string;
  scheduleIntent: CliScheduleIntent;
  /** Optional for injected configurations; independent of external cron intent. */
  workbenchSchedule?: PluginSettings["schedule"];
  libraryConnection?: PersistedLibraryConnection;
  libraryConnectionError?: string;
  /** Fingerprint of the exact loaded bytes, used to reject stale config writes. */
  configRevision?: string;
  /** Load-time adjustments worth telling the user about (e.g. a raised request delay). */
  configWarnings?: string[];
}

export interface LoadCliConfigOptions {
  /** Override fixed config path (tests only). */
  configPath?: string;
  env?: Record<string, string | undefined>;
  platform?: NodeJS.Platform;
  readText?: (path: string) => Promise<string>;
  homedir?: () => string;
}

export class CliConfigError extends Error {
  constructor(message: string, readonly cause?: unknown) {
    super(message);
    this.name = "CliConfigError";
  }
}

export async function loadCliConfig(
  opts: LoadCliConfigOptions = {},
): Promise<CliRuntimeConfig> {
  const env = opts.env ?? process.env;
  const platform = opts.platform ?? process.platform;
  const configPath = opts.configPath ?? resolveCliConfigPath(env, platform);
  const readText = opts.readText ?? ((p: string) => fs.readFile(p, "utf8"));

  let raw: string;
  try {
    raw = await readText(configPath);
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "ENOENT") {
      throw new CliConfigError(
        `CLI config not found: ${configPath}\nRun: arxiv-daily init`, e,
      );
    }
    throw new CliConfigError(`failed to read CLI config: ${configPath}`, e);
  }

  let parsed: Record<string, unknown>;
  try {
    const value = parseToml(raw);
    if (!isRecord(value)) {
      throw new Error("config root must be a table");
    }
    parsed = value;
  } catch (e) {
    throw new CliConfigError(`failed to parse CLI config: ${configPath}`, e);
  }

  const configWarnings: string[] = [];
  const settings = mapTomlToSettings(parsed, configWarnings);

  const home = opts.homedir?.() ?? os.homedir();
  const vaultRoot = resolveUserPath(
    requireString(parsed.vault_root, "vault_root"),
    home,
  );
  const cacheRaw =
    typeof parsed.cache_dir === "string" && parsed.cache_dir.trim()
      ? parsed.cache_dir.trim()
      : path.join(vaultRoot, ".cache", "arxiv-daily");
  const cacheDir = resolveUserPath(cacheRaw, home);

  settings.output.dailyDir = normalizeOutputDirectory(
    "daily_dir",
    settings.output.dailyDir,
  );
  settings.output.papersDir = normalizeOutputDirectory(
    "papers_dir",
    settings.output.papersDir,
  );
  if (
    vaultRelativeDirectoriesCollide(
      settings.output.dailyDir,
      settings.output.papersDir,
    )
  ) {
    throw new CliConfigError("daily_dir and papers_dir must be different");
  }

  const linkStyle = settings.output.linkStyle ?? "wikilink";
  settings.output.linkStyle = linkStyle;
  settings.output.summaryLanguage = normalizeSummaryLanguage(
    settings.output.summaryLanguage ?? "zh",
  );

  const libraryConnection = decodeLibraryConnection(parsed.library);
  const libraryConnectionError = parsed.library === undefined
    ? undefined
    : !libraryConnection
      ? "Optional library connection is invalid; reconnect the library to enable library-guided discovery."
      : isRecord(parsed.library) && parsed.library.authorization !== undefined && !libraryConnection.authorization
        ? "Library authorization is invalid; review its disclosure and authorize processing again."
        : undefined;

  return {
    settings,
    vaultRoot,
    cacheDir,
    linkStyle,
    configPath,
    scheduleIntent: mapScheduleIntent(parsed.schedule),
    workbenchSchedule: mapWorkbenchSchedule(parsed.workbench_schedule),
    configRevision: `sha256:${sha256Hex(raw)}`,
    ...(configWarnings.length ? { configWarnings } : {}),
    ...(libraryConnection ? { libraryConnection } : {}),
    ...(libraryConnectionError ? { libraryConnectionError } : {}),
  };
}

function detailSelectionPresetBalanced(): PluginSettings["detailSelection"] {
  return sanitizeDetailSelection({ profile: "balanced" });
}

function mapTomlToSettings(root: Record<string, unknown>, warnings: string[]): PluginSettings {
  const base = structuredClone(DEFAULT_SETTINGS);
  base.detailSelection = detailSelectionPresetBalanced();

  const llm = asTable(root.llm);
  if (llm) {
    base.llm.apiKey = stringField(llm, "llm", "api_key", base.llm.apiKey);
    base.llm.provider = stringField(llm, "llm", "provider", base.llm.provider);
    base.llm.baseUrl = stringField(llm, "llm", "base_url", base.llm.baseUrl);
    base.llm.model = stringField(llm, "llm", "model", base.llm.model);
    base.llm.thinkingMode = booleanField(llm, "llm", "thinking_mode", base.llm.thinkingMode);
    base.llm.reasoningEffort = stringField(llm, "llm", "reasoning_effort", base.llm.reasoningEffort);
  }

  const embedding = asTable(root.embedding);
  if (embedding) {
    const mode = stringField(embedding, "embedding", "mode", base.embedding.mode);
    base.embedding.mode = mode === "remote" ? "remote" : "local";
    base.embedding.provider = stringField(embedding, "embedding", "provider", base.embedding.provider);
    base.embedding.baseUrl = stringField(embedding, "embedding", "base_url", base.embedding.baseUrl);
    base.embedding.apiKey = stringField(embedding, "embedding", "api_key", base.embedding.apiKey);
    base.embedding.model = stringField(embedding, "embedding", "model", base.embedding.model);
    const dimension = embedding.dimension;
    if (dimension !== undefined) {
      if (typeof dimension !== "number" || !Number.isSafeInteger(dimension) || dimension <= 0) {
        throw typeMismatch("embedding", "dimension", "a positive integer", dimension);
      }
      base.embedding.dimension = dimension;
    }
  }

  const arxiv = asTable(root.arxiv);
  if (arxiv) {
    const categories = arrayField(arxiv, "arxiv", "categories");
    if (categories) {
      base.arxiv.categories = normalizeCategoryList(categories, []);
    }
    base.arxiv.timezone = stringField(arxiv, "arxiv", "timezone", base.arxiv.timezone);
    const topics = arrayField(arxiv, "arxiv", "topics");
    if (topics) {
      base.arxiv.topics = topics.map((item, index) => mapTopic(item, index));
    }
  }
  base.arxiv.category =
    arxivCategories(base.arxiv)[0] ?? base.arxiv.category;

  const output = asTable(root.output);
  if (output) {
    if (output.max_daily_papers !== undefined) {
      if (!isValidMaxDailyPapers(output.max_daily_papers)) {
        throw new CliConfigError("invalid output.max_daily_papers: expected a positive safe integer");
      }
      base.output.maxDailyPapers = output.max_daily_papers;
    }
    base.output.dailyDir = stringField(output, "output", "daily_dir", base.output.dailyDir);
    base.output.papersDir = stringField(
      output,
      "output",
      "papers_dir",
      base.output.papersDir,
    );
    const link = stringField(output, "output", "link_style", base.output.linkStyle ?? "wikilink");
    if (link !== "wikilink" && link !== "relative") {
      throw new CliConfigError(`invalid link_style: ${link}`);
    }
    base.output.linkStyle = link;
    base.output.summaryLanguage = normalizeSummaryLanguage(
      stringField(output, "output", "summary_language", base.output.summaryLanguage ?? "zh"),
    );
  }

  const email = asTable(root.email);
  if (email) {
    base.email.enabled = booleanField(email, "email", "enabled", base.email.enabled);
    const mode = stringField(email, "email", "mode", base.email.mode);
    if (mode !== "self" && mode !== "hosted") {
      throw new CliConfigError(`invalid email.mode: ${mode}`);
    }
    base.email.mode = mode;
    base.email.to = stringField(email, "email", "to", base.email.to);
    base.email.fromEmail = stringField(email, "email", "from_email", base.email.fromEmail);
    base.email.fromName = stringField(
      email,
      "email",
      "from_name",
      base.email.fromName ?? "arXiv Daily",
    );
    base.email.apiKey = stringField(email, "email", "api_key", base.email.apiKey ?? "");
    base.email.hostedToken = stringField(
      email,
      "email",
      "hosted_token",
      base.email.hostedToken ?? "",
    );
    // hosted_base_url intentionally ignored if present
  }

  const advanced = asTable(root.advanced);
  if (advanced) {
    const level = stringField(advanced, "advanced", "log_level", base.advanced.logLevel);
    if (
      level !== "debug" &&
      level !== "info" &&
      level !== "warn" &&
      level !== "error"
    ) {
      throw new CliConfigError(`invalid log_level: ${level}`);
    }
    base.advanced.logLevel = level;
    if (advanced.request_delay_ms !== undefined) {
      const value = advanced.request_delay_ms;
      if (
        typeof value !== "number" ||
        !Number.isFinite(value) ||
        value < 0
      ) {
        throw new CliConfigError(
          "invalid advanced.request_delay_ms: expected a non-negative finite number",
        );
      }
      const raised = Math.max(3_000, value);
      if (raised !== value) {
        warnings.push(
          `advanced.request_delay_ms=${value} is below the 3000 ms floor that keeps arXiv requests gentle; ${raised} is used instead`,
        );
      }
      base.advanced.requestDelayMs = raised;
    }
    base.advanced.cacheExpiryDays = numberField(advanced, "advanced", "cache_expiry_days", base.advanced.cacheExpiryDays);
    base.advanced.sectionCharLimit = numberField(advanced, "advanced", "section_char_limit", base.advanced.sectionCharLimit);
    base.advanced.paperCharLimit = numberField(advanced, "advanced", "paper_char_limit", base.advanced.paperCharLimit);
    base.advanced.dailyCharLimit = numberField(advanced, "advanced", "daily_char_limit", base.advanced.dailyCharLimit);
    if (advanced.generation_concurrency !== undefined) {
      const value = advanced.generation_concurrency;
      if (
        !Number.isSafeInteger(value) ||
        (value as number) < 1 ||
        (value as number) > MAX_GENERATION_CONCURRENCY
      ) {
        throw new CliConfigError(
          `invalid advanced.generation_concurrency: expected an integer from 1 to ${MAX_GENERATION_CONCURRENCY}`,
        );
      }
      base.advanced.generationConcurrency = value as number;
    }
  }

  const detail = asTable(root.detail_selection);
  if (detail) {
    if (!isDetailSelectionProfile(detail.profile)) throw new CliConfigError("Invalid detail_selection.profile");
    base.detailSelection = sanitizeDetailSelection({ profile: detail.profile, normalThreshold: detail.normal_threshold, exceptionalThreshold: detail.exceptional_threshold, softLimit: detail.soft_limit });
  }
  const sidecar = asTable(root.pdf_parser_sidecar);
  if (sidecar) {
    base.pdfParserSidecar.enabled = booleanField(sidecar, "pdf_parser_sidecar", "enabled", base.pdfParserSidecar.enabled);
    base.pdfParserSidecar.capabilitiesUrl = stringField(sidecar, "pdf_parser_sidecar", "capabilities_url", base.pdfParserSidecar.capabilitiesUrl);
    base.pdfParserSidecar.parseUrl = stringField(sidecar, "pdf_parser_sidecar", "parse_url", base.pdfParserSidecar.parseUrl);
    const validation = validateLocalPdfParserSidecarConfig(base);
    if (!validation.ok) throw new CliConfigError(validation.reasons.join("; "));
  }

  // Plugin schedule table ignored for CLI runtime; scheduleIntent is separate.
  base.schedule = { ...DEFAULT_SETTINGS.schedule, enabled: false };

  return base;
}

function mapTopic(raw: unknown, index: number): Topic {
  const scope = `arxiv.topics[${index}]`;
  if (!isRecord(raw)) {
    throw new CliConfigError(`${scope} must be a table`);
  }
  const name = stringField(raw, scope, "name", "");
  const tag = stringField(raw, scope, "tag", "");
  const description = stringField(raw, scope, "description", "");
  const detail = booleanField(raw, scope, "detail", true);
  const id = stringField(raw, scope, "id", "").trim() || `topic-${index + 1}`;
  // `detail` keeps the CLI's own default (true) rather than the shared one;
  // everything else goes through normalizeTopic so the plugin and the CLI
  // derive the description shadow the same way (ADR 0012).
  // Read-only loads must not invent fresh direction identities on every read.
  // `Array.isArray` is typed `(arg: any) => arg is any[]` in lib.es5, so an
  // explicit `unknown[]` annotation (rather than relying on the narrowed
  // type) keeps the elements honestly unknown instead of silently `any`.
  const rawDirections: unknown[] | undefined = arrayField(raw, scope, "directions");
  const directions = rawDirections
    ? rawDirections.map((direction, directionIndex) => isRecord(direction)
      ? { ...direction, id: typeof direction.id === "string" && direction.id.trim()
        ? direction.id : `cli-direction-${sha256Hex(`${id}:${directionIndex}:${String(direction.text ?? "")}`).slice(0, 16)}` }
      : direction)
    : description.trim() ? [{ id: `cli-direction-${sha256Hex(`${id}:0:${description.trim()}`).slice(0, 16)}`, text: description, origin: "migrated" }] : [];
  return normalizeTopic({ id, name, tag, description, directions, detail });
}

function mapWorkbenchSchedule(raw: unknown): PluginSettings["schedule"] {
  const schedule = { ...DEFAULT_SETTINGS.schedule, enabled: false };
  if (raw === undefined) return schedule;
  if (!isRecord(raw)) throw new CliConfigError("Invalid workbench_schedule table");
  schedule.enabled = booleanField(raw, "workbench_schedule", "enabled", schedule.enabled);
  schedule.runAtLocal = stringField(raw, "workbench_schedule", "run_at_local", schedule.runAtLocal);
  schedule.runUntilLocal = stringField(raw, "workbench_schedule", "run_until_local", schedule.runUntilLocal);
  if (raw.tick_interval_min !== undefined) {
    if (typeof raw.tick_interval_min !== "number" || !Number.isInteger(raw.tick_interval_min) || raw.tick_interval_min < 1 || raw.tick_interval_min > 1440) throw new CliConfigError("Invalid workbench_schedule.tick_interval_min (1..1440)");
    schedule.tickIntervalMin = raw.tick_interval_min;
  }
  const validation = validateScheduleConfig({ ...DEFAULT_SETTINGS, schedule });
  if (!validation.ok) throw new CliConfigError(validation.reasons.join("; "));
  return schedule;
}

function mapScheduleIntent(raw: unknown): CliScheduleIntent {
  const d = { ...DEFAULT_CLI_SCHEDULE };
  if (!isRecord(raw)) return d;
  d.enabled = booleanField(raw, "schedule", "enabled", d.enabled);
  d.on = stringField(raw, "schedule", "on", d.on);
  d.intervalHours = Math.max(0, Math.floor(numberField(raw, "schedule", "interval_hours", d.intervalHours)));
  d.until = stringField(raw, "schedule", "until", d.until);
  d.weekdaysOnly = booleanField(raw, "schedule", "weekdays_only", d.weekdaysOnly);
  if (!/^\d{1,2}:\d{2}$/.test(d.on)) {
    throw new CliConfigError(`invalid schedule.on: ${d.on} (use HH:MM)`);
  }
  if (!/^\d{1,2}:\d{2}$/.test(d.until)) {
    throw new CliConfigError(`invalid schedule.until: ${d.until} (use HH:MM)`);
  }
  scheduleFireSlots(d);
  return d;
}

function normalizeOutputDirectory(name: string, value: unknown): string {
  const result = validateVaultRelativeDirectory(value);
  if (!result.ok || !result.value) {
    throw new CliConfigError(`invalid ${name}: ${result.reason}`);
  }
  return result.value;
}

function normalizeSummaryLanguage(value: unknown): SummaryLanguage {
  if (value === "zh" || value === undefined) return "zh";
  if (value === "en") return "en";
  throw new CliConfigError(`invalid summary_language: ${String(value)}`);
}

function resolveUserPath(input: string, home: string): string {
  const trimmed = input.trim();
  if (!trimmed) throw new CliConfigError("path must not be empty");
  if (trimmed === "~") return home;
  if (trimmed.startsWith("~/") || trimmed.startsWith("~\\")) {
    return path.resolve(home, trimmed.slice(2));
  }
  return path.resolve(trimmed);
}

function requireString(value: unknown, key: string): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new CliConfigError(`missing required ${key}`);
  }
  return value.trim();
}

/** How a wrongly typed TOML value reads back in an error message. */
function describeTomlValue(value: unknown): string {
  if (typeof value === "string") {
    const text = value.length > 40 ? `${value.slice(0, 37)}…` : value;
    return `string ${JSON.stringify(text)}`;
  }
  if (typeof value === "number" || typeof value === "boolean") return `${typeof value} ${String(value)}`;
  if (Array.isArray(value)) return "array";
  return value && typeof value === "object" ? "table" : typeof value;
}

function typeMismatch(scope: string, key: string, expected: string, value: unknown): CliConfigError {
  return new CliConfigError(
    `invalid ${scope}.${key}: expected ${expected}, found ${describeTomlValue(value)}`,
  );
}

/**
 * `scope.key` as a string, or `fallback` when the key is absent. A present key
 * with the wrong type is an error, the same as a bad `max_daily_papers` — a
 * silently ignored value would quietly reset the feature to its default.
 */
function stringField(
  table: Record<string, unknown>,
  scope: string,
  key: string,
  fallback: string,
): string {
  const value = table[key];
  if (value === undefined) return fallback;
  if (typeof value !== "string") throw typeMismatch(scope, key, "a string", value);
  return value;
}

function booleanField(
  table: Record<string, unknown>,
  scope: string,
  key: string,
  fallback: boolean,
): boolean {
  const value = table[key];
  if (value === undefined) return fallback;
  if (typeof value !== "boolean") throw typeMismatch(scope, key, "a boolean", value);
  return value;
}

function numberField(
  table: Record<string, unknown>,
  scope: string,
  key: string,
  fallback: number,
): number {
  const value = table[key];
  if (value === undefined) return fallback;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw typeMismatch(scope, key, "a finite number", value);
  }
  return value;
}

/** `scope.key` as an array, or undefined when absent; anything else is an error. */
function arrayField(
  table: Record<string, unknown>,
  scope: string,
  key: string,
): unknown[] | undefined {
  const value = table[key];
  if (value === undefined) return undefined;
  if (!Array.isArray(value)) throw typeMismatch(scope, key, "an array", value);
  return value;
}

function asTable(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

/** Expand HH:MM schedule slots for cron generation. */
export function scheduleFireSlots(intent: CliScheduleIntent): string[] {
  const start = parseHm(intent.on);
  const end = parseHm(intent.until);
  if (intent.intervalHours <= 0) {
    return [formatHm(start.h, start.m)];
  }
  const slots: string[] = [];
  let minutes = start.h * 60 + start.m;
  const endMin = end.h * 60 + end.m;
  if (endMin < minutes) {
    throw new CliConfigError("schedule.until must not be earlier than schedule.on for a recurring schedule");
  }
  const step = intent.intervalHours * 60;
  while (minutes <= endMin) {
    const h = Math.floor(minutes / 60) % 24;
    const m = minutes % 60;
    slots.push(formatHm(h, m));
    minutes += step;
    if (slots.length > 48) break;
  }
  return slots;
}

function parseHm(value: string): { h: number; m: number } {
  const [hs, ms] = value.split(":");
  const h = Number(hs);
  const m = Number(ms);
  if (!Number.isInteger(h) || !Number.isInteger(m) || h < 0 || h > 23 || m < 0 || m > 59) {
    throw new CliConfigError(`invalid time ${value}`);
  }
  return { h, m };
}

function formatHm(h: number, m: number): string {
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}
