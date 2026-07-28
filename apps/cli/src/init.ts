import * as fs from "node:fs/promises";
import * as path from "node:path";
import * as readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import * as p from "@clack/prompts";
import {
  ARXIV_CATEGORIES,
  DEFAULT_SETTINGS,
  LlmClient,
  Logger,
  PROVIDER_PRESETS,
  startHostedEmailVerification,
} from "@arxiv-daily/core";
import { buildNodeHostAdapters } from "@arxiv-daily/node-runtime";
import { resolveCliConfigPath } from "./config-path";
import type { WritableTextStream } from "./main-types";

export interface InitOptions {
  env?: Record<string, string | undefined>;
  platform?: NodeJS.Platform;
  configPath?: string;
  /**
   * Non-TUI answers for tests (prompt string → reply).
   * When set, uses plain questions instead of @clack/prompts.
   */
  ask?: (prompt: string) => Promise<string>;
  writeFile?: (path: string, body: string) => Promise<void>;
  readFile?: (path: string) => Promise<string>;
  mkdir?: (path: string) => Promise<void>;
  stdout?: WritableTextStream;
  stderr?: WritableTextStream;
  isTTY?: boolean;
  startHostedVerify?: (email: string) => Promise<void>;
  /** Inject model list / connection test for unit tests. */
  fetchModels?: (input: {
    apiKey: string;
    baseUrl: string;
  }) => Promise<string[]>;
}

const COMMON_TIMEZONES = [
  "Asia/Shanghai",
  "Asia/Tokyo",
  "Asia/Singapore",
  "Asia/Kolkata",
  "Europe/London",
  "Europe/Berlin",
  "Europe/Paris",
  "America/New_York",
  "America/Los_Angeles",
  "America/Chicago",
  "UTC",
] as const;

type CancelToken = symbol;

function cancelled(value: unknown): value is CancelToken {
  return p.isCancel(value);
}

export async function runInit(opts: InitOptions = {}): Promise<number> {
  const env = opts.env ?? process.env;
  const platform = opts.platform ?? process.platform;
  const configPath = opts.configPath ?? resolveCliConfigPath(env, platform);
  const write =
    opts.writeFile ??
    ((filePath: string, body: string) => fs.writeFile(filePath, body, "utf8"));
  const read =
    opts.readFile ?? ((filePath: string) => fs.readFile(filePath, "utf8"));
  const mkdir =
    opts.mkdir ??
    ((dir: string) => fs.mkdir(dir, { recursive: true }).then(() => undefined));
  const stdout = opts.stdout ?? process.stdout;
  const stderr = opts.stderr ?? process.stderr;
  const useClack = !opts.ask;
  const tty =
    opts.isTTY ??
    (Boolean((input as NodeJS.ReadStream).isTTY) &&
      Boolean((output as NodeJS.WriteStream).isTTY));

  if (!tty && !opts.ask) {
    writeLine(stderr, "init requires an interactive terminal");
    return 2;
  }

  if (useClack) {
    p.intro("arXiv Daily CLI setup");
    p.note(
      [
        `Config will be written to:`,
        configPath,
        "",
        "Use ↑/↓ to move, Space to toggle multi-select, Enter to confirm.",
        "Ctrl+C cancels.",
      ].join("\n"),
      "About this wizard",
    );
  } else {
    writeLine(stdout, "arXiv Daily CLI setup");
    writeLine(stdout, `Config file: ${configPath}`);
  }

  let existing: string | null = null;
  try {
    existing = await read(configPath);
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
  }

  let mode: "write" | "merge" | "cancel" = "write";
  if (existing !== null) {
    const choice = await askSelect(opts, {
      message: `Config already exists at ${configPath}`,
      options: [
        { value: "o", label: "Overwrite", hint: "replace entire file" },
        { value: "m", label: "Keep existing", hint: "do not rewrite" },
        { value: "c", label: "Cancel" },
      ],
      initialValue: "o",
    });
    if (cancelled(choice) || choice === "c") {
      if (useClack) p.cancel("init cancelled");
      else writeLine(stdout, "init cancelled");
      return 0;
    }
    if (choice === "m") mode = "merge";
  }

  // --- Paths ---
  const vaultRootRaw = await askText(opts, {
    message: "Vault root path",
    placeholder: "/home/you/Notes",
    initialValue: "",
    validate: (v) => (v.trim() ? undefined : "Required — absolute path to your notes folder"),
  });
  if (cancelled(vaultRootRaw)) return cancelInit(useClack);
  const vaultRoot = vaultRootRaw.trim();

  // --- LLM: provider → URL → key → test/fetch models → model ---
  const providerKeys = Object.keys(PROVIDER_PRESETS);
  const providerPick = await askSelect(opts, {
    message: "LLM provider",
    options: providerKeys.map((key) => ({
      value: key,
      label: PROVIDER_PRESETS[key]!.name,
      hint: key,
    })),
    initialValue: DEFAULT_SETTINGS.llm.provider,
  });
  if (cancelled(providerPick)) return cancelInit(useClack);
  const providerId = providerPick;
  const preset = PROVIDER_PRESETS[providerId] ?? PROVIDER_PRESETS.custom!;
  const baseUrlDefault = preset.baseUrl || DEFAULT_SETTINGS.llm.baseUrl;

  const baseUrlRaw = await askText(opts, {
    message: "API base URL",
    initialValue: baseUrlDefault,
    placeholder: baseUrlDefault,
    validate: (v) => (v.trim() ? undefined : "Base URL is required"),
  });
  if (cancelled(baseUrlRaw)) return cancelInit(useClack);
  const baseUrl = baseUrlRaw.trim();

  const apiKeyRaw = await askPassword(opts, {
    message: "LLM API key (stored in config file; do not commit it)",
    validate: (v) => (v.trim() ? undefined : "API key is required"),
  });
  if (cancelled(apiKeyRaw)) return cancelInit(useClack);
  const apiKey = apiKeyRaw.trim();

  let remoteModels: string[] = [];
  let modelDefault =
    preset.models[0]?.value || DEFAULT_SETTINGS.llm.model;

  const tryFetch = await askConfirm(opts, {
    message: "Test connection and load model list from the provider?",
    initialValue: true,
  });
  if (cancelled(tryFetch)) return cancelInit(useClack);

  if (tryFetch) {
    const spin = useClack ? p.spinner() : null;
    spin?.start("Calling provider /models …");
    try {
      const models = opts.fetchModels
        ? await opts.fetchModels({ apiKey, baseUrl })
        : await defaultFetchModels(apiKey, baseUrl);
      remoteModels = models;
      spin?.stop(
        models.length > 0
          ? `Connected — ${models.length} model(s) listed`
          : "Connected — empty model list; you can type a model id",
      );
      if (!useClack) {
        writeLine(
          stdout,
          models.length > 0
            ? `Connected — ${models.length} model(s)`
            : "Connected — no models returned",
        );
      }
      if (models[0]) modelDefault = models[0];
    } catch (e) {
      const msg = (e as Error).message;
      spin?.stop(`Could not list models: ${msg}`);
      if (!useClack) writeLine(stderr, `Could not list models: ${msg}`);
      if (useClack) {
        p.log.warn("You can still enter a model id manually.");
      } else {
        writeLine(stdout, "You can still enter a model id manually.");
      }
    }
  }

  let model: string;
  if (remoteModels.length > 0) {
    const options = remoteModels.slice(0, 40).map((id) => ({
      value: id,
      label: id,
    }));
    if (remoteModels.length > 40) {
      options.push({
        value: "__other__",
        label: "Other… (type model id)",
      });
    }
    const picked = await askSelect(opts, {
      message: "Model",
      options,
      initialValue: options.some((o) => o.value === modelDefault)
        ? modelDefault
        : options[0]!.value,
    });
    if (cancelled(picked)) return cancelInit(useClack);
    if (picked === "__other__") {
      const typed = await askText(opts, {
        message: "Model id",
        initialValue: modelDefault,
        validate: (v) => (v.trim() ? undefined : "Required"),
      });
      if (cancelled(typed)) return cancelInit(useClack);
      model = typed.trim();
    } else {
      model = picked;
    }
  } else if (preset.models.length > 0 && providerId !== "custom") {
    const picked = await askSelect(opts, {
      message: "Model (preset list — connection test skipped or failed)",
      options: [
        ...preset.models.map((m) => ({
          value: m.value,
          label: m.label,
          hint: m.value,
        })),
        { value: "__other__", label: "Other… (type model id)" },
      ],
      initialValue: modelDefault,
    });
    if (cancelled(picked)) return cancelInit(useClack);
    if (picked === "__other__") {
      const typed = await askText(opts, {
        message: "Model id",
        initialValue: modelDefault,
        validate: (v) => (v.trim() ? undefined : "Required"),
      });
      if (cancelled(typed)) return cancelInit(useClack);
      model = typed.trim();
    } else {
      model = picked;
    }
  } else {
    const typed = await askText(opts, {
      message: "Model id",
      initialValue: modelDefault,
      validate: (v) => (v.trim() ? undefined : "Required"),
    });
    if (cancelled(typed)) return cancelInit(useClack);
    model = typed.trim();
  }

  // --- Email ---
  const emailChoice = await askSelect(opts, {
    message: "Email digests after a successful daily run?",
    options: [
      {
        value: "skip",
        label: "Skip for now",
        hint: "local files only",
      },
      {
        value: "self",
        label: "Send yourself",
        hint: "your Resend API key",
      },
      {
        value: "hosted",
        label: "Official delivery (Beta)",
        hint: "verify email; shared free quota",
      },
    ],
    initialValue: "skip",
  });
  if (cancelled(emailChoice)) return cancelInit(useClack);

  let emailEnabled = false;
  let emailMode: "self" | "hosted" = "self";
  let emailTo = "";
  let emailApiKey = "";
  let hostedToken = "";

  if (emailChoice === "self") {
    emailMode = "self";
    if (useClack) {
      p.note(
        "Create a key at https://resend.com\nWith From empty, you can usually only send TO your Resend account email.",
        "Send yourself",
      );
    }
    const to = await askText(opts, {
      message: "Your inbox (To)",
      validate: (v) => (v.trim() ? undefined : "Required"),
    });
    if (cancelled(to)) return cancelInit(useClack);
    emailTo = to.trim();
    const reKey = await askPassword(opts, {
      message: "Resend API key (re_…)",
      validate: (v) => (v.trim() ? undefined : "Required"),
    });
    if (cancelled(reKey)) return cancelInit(useClack);
    emailApiKey = reKey.trim();
    const enable = await askConfirm(opts, {
      message: "Enable daily auto-send now? (prefer email test first)",
      initialValue: false,
    });
    if (cancelled(enable)) return cancelInit(useClack);
    emailEnabled = enable;
  } else if (emailChoice === "hosted") {
    emailMode = "hosted";
    if (useClack) {
      p.note(
        [
          "We email you a link. Open it, copy the LONG code from the web page",
          "(not the short code in the link), then paste it here.",
          "Shared free Beta — a few messages per inbox per UTC day.",
        ].join("\n"),
        "Official delivery",
      );
    }
    const to = await askText(opts, {
      message: "Email to verify (To)",
      validate: (v) => (v.trim() ? undefined : "Required"),
    });
    if (cancelled(to)) return cancelInit(useClack);
    emailTo = to.trim();

    const sendNow = await askConfirm(opts, {
      message: "Send verification email now?",
      initialValue: true,
    });
    if (cancelled(sendNow)) return cancelInit(useClack);
    if (sendNow) {
      const spin = useClack ? p.spinner() : null;
      spin?.start("Sending verification email…");
      try {
        const start =
          opts.startHostedVerify ??
          (async (email: string) => {
            const host = buildNodeHostAdapters({ rootDir: process.cwd() });
            await startHostedEmailVerification({ http: host.http, email });
          });
        await start(emailTo);
        spin?.stop(`Sent to ${emailTo} — open the link, copy the long code`);
        if (!useClack) {
          writeLine(stdout, `Sent to ${emailTo}`);
        }
      } catch (e) {
        spin?.stop(`Send failed: ${(e as Error).message}`);
        if (!useClack) {
          writeLine(stderr, `Send failed: ${(e as Error).message}`);
        }
        if (useClack) {
          p.log.warn("Later: arxiv-daily email verify-start");
        }
      }
    }

    const tokenRaw = await askText(opts, {
      message: "Paste verification code (long token), or leave empty",
      placeholder: "leave empty to fill later",
    });
    if (cancelled(tokenRaw)) return cancelInit(useClack);
    hostedToken = tokenRaw.trim().replace(/\s+/g, "");
    if (hostedToken) {
      const enable = await askConfirm(opts, {
        message: "Enable daily auto-send now? (prefer email test first)",
        initialValue: false,
      });
      if (cancelled(enable)) return cancelInit(useClack);
      emailEnabled = enable;
    } else if (useClack) {
      p.log.info(
        "Set email.hosted_token in config after verify, then: arxiv-daily email test",
      );
    }
  }

  // --- Categories (multi-select TUI) ---
  const flatCats = ARXIV_CATEGORIES.flatMap((g) =>
    g.categories.map((c) => ({
      value: c.id,
      label: `${c.id}`,
      hint: `${g.label} · ${c.name}`,
    })),
  );
  const defaultCats = DEFAULT_SETTINGS.arxiv.categories.filter((id) =>
    flatCats.some((c) => c.value === id),
  );
  const catPick = await askMultiSelect(opts, {
    message: "arXiv categories (Space toggle, Enter confirm)",
    options: flatCats,
    initialValues:
      defaultCats.length > 0 ? defaultCats : [flatCats[0]!.value],
    required: true,
  });
  if (cancelled(catPick)) return cancelInit(useClack);
  const categories =
    catPick.length > 0 ? catPick : [...DEFAULT_SETTINGS.arxiv.categories];

  // --- Timezone ---
  const tzPick = await askSelect(opts, {
    message: "Timezone for “today”",
    options: [
      ...COMMON_TIMEZONES.map((tz) => ({ value: tz, label: tz })),
      { value: "__other__", label: "Other IANA name…" },
    ],
    initialValue: DEFAULT_SETTINGS.arxiv.timezone,
  });
  if (cancelled(tzPick)) return cancelInit(useClack);
  let timezone = tzPick;
  if (tzPick === "__other__") {
    const typed = await askText(opts, {
      message: "IANA timezone",
      initialValue: DEFAULT_SETTINGS.arxiv.timezone,
      validate: (v) => (v.trim() ? undefined : "Required"),
    });
    if (cancelled(typed)) return cancelInit(useClack);
    timezone = typed.trim();
  }

  // --- Language ---
  const langPick = await askSelect(opts, {
    message: "Summary language for reports",
    options: [
      { value: "zh", label: "Chinese (zh)" },
      { value: "en", label: "English (en)" },
    ],
    initialValue: "zh",
  });
  if (cancelled(langPick)) return cancelInit(useClack);
  const summaryLanguage = langPick === "en" ? "en" : "zh";

  // --- Topic ---
  if (useClack) {
    p.note(
      "Topics control filtering. You can add more later under [[arxiv.topics]] in the config.",
      "Research topic",
    );
  }
  const topicNameRaw = await askText(opts, {
    message: "Topic display name",
    initialValue: "My research (edit me)",
  });
  if (cancelled(topicNameRaw)) return cancelInit(useClack);
  const topicTagRaw = await askText(opts, {
    message: "Topic tag slug",
    initialValue: "my-research",
  });
  if (cancelled(topicTagRaw)) return cancelInit(useClack);
  const topicDescRaw = await askText(opts, {
    message: "Topic description (what papers belong here)",
    initialValue:
      "Describe in natural language what papers belong in this topic (problems, methods, objects; what to exclude).",
  });
  if (cancelled(topicDescRaw)) return cancelInit(useClack);

  const cacheDir = path.join(vaultRoot, ".cache", "arxiv-daily");
  const body = renderInitToml({
    vaultRoot,
    cacheDir,
    apiKey,
    baseUrl,
    model,
    provider: providerId,
    thinkingMode: preset.thinkingMode,
    categories,
    timezone,
    summaryLanguage,
    topic: {
      name: topicNameRaw.trim() || "My research (edit me)",
      tag: topicTagRaw.trim() || "my-research",
      description:
        topicDescRaw.trim() ||
        "Describe in natural language what papers belong in this topic.",
    },
    email: {
      enabled: emailEnabled,
      mode: emailMode,
      to: emailTo,
      apiKey: emailApiKey,
      hostedToken,
    },
  });

  let finalBody = body;
  if (mode === "merge" && existing) {
    finalBody = mergeTomlPreferExisting(existing, body);
  }

  await mkdir(path.dirname(configPath));
  await write(configPath, finalBody);

  const nextSteps = [
    `Wrote ${configPath}`,
    "",
    "Next:",
    "  arxiv-daily run --today",
    "  Optional: refine [[arxiv.topics]] in the config",
    "  Optional: arxiv-daily email test  then set email.enabled = true",
    "  Optional: [schedule] enabled = true → arxiv-daily schedule install",
    "",
    "schema_version = 1 is for future config migrations (leave it alone).",
  ].join("\n");

  if (useClack) {
    p.outro(nextSteps);
  } else {
    writeLine(stdout, nextSteps);
  }
  return 0;
}

function cancelInit(useClack: boolean): number {
  if (useClack) p.cancel("init cancelled");
  return 0;
}

async function defaultFetchModels(
  apiKey: string,
  baseUrl: string,
): Promise<string[]> {
  const logger = new Logger("warn");
  const host = buildNodeHostAdapters({ rootDir: process.cwd() });
  const client = new LlmClient(
    {
      ...DEFAULT_SETTINGS.llm,
      apiKey,
      baseUrl,
      model: DEFAULT_SETTINGS.llm.model,
    },
    logger,
    host.http,
  );
  return client.fetchModels();
}

// --- UI adapters (clack vs injected ask) ---

async function askText(
  opts: InitOptions,
  args: {
    message: string;
    placeholder?: string;
    initialValue?: string;
    validate?: (value: string) => string | undefined;
  },
): Promise<string | CancelToken> {
  if (opts.ask) {
    for (;;) {
      const v = await plainPrompt(
        opts,
        `${args.message}${args.initialValue ? ` [${args.initialValue}]` : ""}: `,
      );
      const value = v.trim() ? v : (args.initialValue ?? "");
      const err = args.validate?.(value);
      if (!err) return value;
      writeLine(opts.stderr ?? process.stderr, `  ${err}`);
    }
  }
  return p.text({
    message: args.message,
    placeholder: args.placeholder,
    initialValue: args.initialValue,
    validate: args.validate
      ? (value) => args.validate!(value ?? "")
      : undefined,
  });
}

async function askPassword(
  opts: InitOptions,
  args: {
    message: string;
    validate?: (value: string) => string | undefined;
  },
): Promise<string | CancelToken> {
  if (opts.ask) {
    for (;;) {
      const v = await plainPrompt(opts, `${args.message}: `);
      const err = args.validate?.(v);
      if (!err) return v;
      writeLine(opts.stderr ?? process.stderr, `  ${err}`);
    }
  }
  return p.password({
    message: args.message,
    validate: args.validate
      ? (value) => args.validate!(value ?? "")
      : undefined,
  });
}

async function askConfirm(
  opts: InitOptions,
  args: { message: string; initialValue?: boolean },
): Promise<boolean | CancelToken> {
  if (opts.ask) {
    const def = args.initialValue ? "Y/n" : "y/N";
    const v = (
      await plainPrompt(opts, `${args.message} [${def}]: `)
    )
      .trim()
      .toLowerCase();
    if (!v) return Boolean(args.initialValue);
    return v === "y" || v === "yes";
  }
  return p.confirm({
    message: args.message,
    initialValue: args.initialValue,
  });
}

async function askSelect(
  opts: InitOptions,
  args: {
    message: string;
    options: Array<{ value: string; label: string; hint?: string }>;
    initialValue?: string;
  },
): Promise<string | CancelToken> {
  if (opts.ask) {
    const lines = args.options
      .map((o, i) => `  ${i + 1}) ${o.label}${o.hint ? ` (${o.hint})` : ""}`)
      .join("\n");
    writeLine(opts.stdout ?? process.stdout, lines);
    const raw = (
      await plainPrompt(
        opts,
        `${args.message} [default ${args.initialValue ?? "1"}]: `,
      )
    ).trim();
    if (!raw) {
      return args.initialValue ?? args.options[0]!.value;
    }
    const n = Number(raw);
    if (Number.isInteger(n) && n >= 1 && n <= args.options.length) {
      return args.options[n - 1]!.value;
    }
    const byValue = args.options.find((o) => o.value === raw);
    if (byValue) return byValue.value;
    return args.initialValue ?? args.options[0]!.value;
  }
  return p.select({
    message: args.message,
    options: args.options,
    initialValue: args.initialValue,
  });
}

async function askMultiSelect(
  opts: InitOptions,
  args: {
    message: string;
    options: Array<{ value: string; label: string; hint?: string }>;
    initialValues?: string[];
    required?: boolean;
  },
): Promise<string[] | CancelToken> {
  if (opts.ask) {
    // Test / non-clack: accept comma numbers or ids
    const lines = args.options
      .slice(0, 30)
      .map((o, i) => `  ${i + 1}) ${o.label}${o.hint ? ` — ${o.hint}` : ""}`)
      .join("\n");
    writeLine(opts.stdout ?? process.stdout, lines);
    const raw = (
      await plainPrompt(
        opts,
        `${args.message} (numbers or ids) [${(args.initialValues ?? []).join(",") || "1"}]: `,
      )
    ).trim();
    if (!raw) return args.initialValues ?? [args.options[0]!.value];
    if (/[a-z]/i.test(raw) && !/^\d/.test(raw)) {
      return raw.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean);
    }
    const indexes = raw
      .split(/[,\s]+/)
      .map((s) => Number(s.trim()))
      .filter((n) => Number.isInteger(n) && n >= 1 && n <= args.options.length);
    const picked = indexes.map((i) => args.options[i - 1]!.value);
    return picked.length > 0
      ? [...new Set(picked)]
      : (args.initialValues ?? [args.options[0]!.value]);
  }
  return p.multiselect({
    message: args.message,
    options: args.options,
    initialValues: args.initialValues,
    required: args.required,
  });
}

async function plainPrompt(
  opts: InitOptions,
  message: string,
): Promise<string> {
  if (opts.ask) return opts.ask(message);
  const rl = readline.createInterface({ input, output });
  try {
    return await rl.question(message);
  } finally {
    rl.close();
  }
}

function writeLine(stream: WritableTextStream, line: string): void {
  stream.write(`${line}\n`);
}

export function renderInitToml(input: {
  vaultRoot: string;
  cacheDir: string;
  apiKey: string;
  baseUrl: string;
  model: string;
  provider: string;
  thinkingMode: boolean;
  categories: string[];
  timezone: string;
  summaryLanguage: string;
  topic: { name: string; tag: string; description: string };
  email: {
    enabled: boolean;
    mode: "self" | "hosted";
    to: string;
    apiKey: string;
    hostedToken: string;
  };
}): string {
  const cats = input.categories.map((c) => tomlString(c)).join(", ");
  return `# =============================================================================
# arXiv Daily — CLI config
# Path: ~/.config/arxiv-daily/config.toml  (or $XDG_CONFIG_HOME/...)
# May contain secrets — do not commit or share this file publicly.
#
# schema_version: integer for future config format migrations.
#   Current value is 1. Leave it unless release notes tell you to change it.
#
# Editing tips (or hand to an agent):
#   - Research focus: [arxiv].categories and [[arxiv.topics]]
#   - Do not invent service URLs; do not add hosted_base_url
#   - Keep key names; only replace secret values
# Run: arxiv-daily run --today
# =============================================================================

schema_version = 1

# Folder where daily/ and papers/ are written (absolute path recommended)
vault_root = ${tomlString(input.vaultRoot)}
# Fetch cache (safe to delete; will be recreated)
cache_dir = ${tomlString(input.cacheDir)}

[llm]
api_key = ${tomlString(input.apiKey)}
base_url = ${tomlString(input.baseUrl)}
model = ${tomlString(input.model)}
# Preset id (deepseek, openai, anthropic, zhipu, custom, ...)
provider = ${tomlString(input.provider)}
thinking_mode = ${input.thinkingMode}
reasoning_effort = ${tomlString(DEFAULT_SETTINGS.llm.reasoningEffort)}

[arxiv]
# arXiv category ids, e.g. ["astro-ph"], ["cs.LG", "cs.AI"]
categories = [${cats}]
# IANA timezone for "today" (e.g. Asia/Shanghai, America/New_York, UTC)
timezone = ${tomlString(input.timezone)}

# Duplicate the whole [[arxiv.topics]] block to add more topics.
# Clearer description → better filtering.
[[arxiv.topics]]
name = ${tomlString(input.topic.name)}
tag = ${tomlString(input.topic.tag)}
description = ${tomlString(input.topic.description)}
# true = this topic may get longer paper notes under papers/
detail = true

[output]
# "zh" or "en"
summary_language = ${tomlString(input.summaryLanguage)}
daily_dir = "arxiv-daily/daily"
papers_dir = "arxiv-daily/papers"
link_style = "wikilink"

[email]
# true = email a digest after a successful daily run (failures never fail the report)
enabled = ${input.email.enabled}
# "self" = your Resend API key; "hosted" = Official delivery Beta (verification token)
mode = ${tomlString(input.email.mode)}
to = ${tomlString(input.email.to)}
# Leave empty for Resend test sender (often only delivers to your Resend account email)
from_email = ""
from_name = "arXiv Daily"
# mode = "self"
api_key = ${tomlString(input.email.apiKey)}
# mode = "hosted" — long code from the verification web page
hosted_token = ${tomlString(input.email.hostedToken)}

[schedule]
# Defaults only; init does not ask. Set enabled = true then: arxiv-daily schedule install
enabled = false
# First daily fire (machine local time HH:MM)
on = "09:30"
# 0 = once per day at on; e.g. 4 = every 4 hours from on until until
interval_hours = 0
until = "18:00"
weekdays_only = true

[advanced]
# debug | info | warn | error
log_level = "info"
`;
}

function tomlString(value: string): string {
  return JSON.stringify(value);
}

function mergeTomlPreferExisting(existing: string, generated: string): string {
  if (/^\s*vault_root\s*=/m.test(existing)) {
    return `${existing.trimEnd()}\n\n# --- init merge: kept existing file; re-run with overwrite to replace ---\n`;
  }
  return generated;
}
