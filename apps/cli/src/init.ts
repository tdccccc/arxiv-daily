import * as fs from "node:fs/promises";
import * as os from "node:os";
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
  /** Non-TUI answers for tests. When set, uses plain questions (no clack). */
  ask?: (prompt: string) => Promise<string>;
  writeFile?: (path: string, body: string) => Promise<void>;
  readFile?: (path: string) => Promise<string>;
  mkdir?: (path: string) => Promise<void>;
  stdout?: WritableTextStream;
  stderr?: WritableTextStream;
  isTTY?: boolean;
  startHostedVerify?: (email: string) => Promise<void>;
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

const BACK = "__back__" as const;

type StepId =
  | "vault"
  | "provider"
  | "baseUrl"
  | "apiKey"
  | "fetchModels"
  | "model"
  | "email"
  | "categories"
  | "timezone"
  | "language"
  | "topic"
  | "write";

const STEP_ORDER: StepId[] = [
  "vault",
  "provider",
  "baseUrl",
  "apiKey",
  "fetchModels",
  "model",
  "email",
  "categories",
  "timezone",
  "language",
  "topic",
  "write",
];

interface WizardState {
  vaultRoot: string;
  providerId: string;
  baseUrl: string;
  apiKey: string;
  tryFetch: boolean;
  remoteModels: string[];
  model: string;
  emailChoice: "skip" | "self" | "hosted";
  emailEnabled: boolean;
  emailMode: "self" | "hosted";
  emailTo: string;
  emailApiKey: string;
  hostedToken: string;
  categories: string[];
  timezone: string;
  summaryLanguage: "zh" | "en";
  topicName: string;
  topicTag: string;
  topicDescription: string;
}

type Nav = "next" | "back" | "abort";

function cancelled(value: unknown): boolean {
  return p.isCancel(value);
}

function prevStep(id: StepId): StepId | null {
  const i = STEP_ORDER.indexOf(id);
  return i > 0 ? STEP_ORDER[i - 1]! : null;
}

function nextStep(id: StepId): StepId | null {
  const i = STEP_ORDER.indexOf(id);
  return i >= 0 && i < STEP_ORDER.length - 1 ? STEP_ORDER[i + 1]! : null;
}

/**
 * Key bindings (clack limitation):
 * - ↑/↓ move, Space multi-select, Enter confirm
 * - Explicit "← Back" option on select menus (after first step)
 * - Esc / Ctrl+C: go back one step (not exit), except on the first step → abort
 * - PageUp/PageDn are NOT handled by @clack/prompts for wizard navigation
 */
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
        `Config file: ${configPath}`,
        "",
        "Navigation:",
        "  ↑/↓     move   ·  Space  multi-select  ·  Enter  confirm",
        "  ← Back  menu item (when available)",
        "  Esc or Ctrl+C  go to previous step (exit only on first step)",
        "",
        "Note: PageUp/PageDown are not used — @clack/prompts does not support them.",
      ].join("\n"),
      "Keys",
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

  let fileMode: "write" | "merge" = "write";
  if (existing !== null) {
    const choice = await askSelect(opts, {
      message: `Config already exists`,
      options: [
        { value: "o", label: "Overwrite", hint: "replace entire file" },
        { value: "m", label: "Keep existing", hint: "do not rewrite" },
        { value: "c", label: "Cancel" },
      ],
      initialValue: "o",
      allowBack: false,
    });
    if (choice.nav !== "next" || choice.value === "c") {
      return finishCancel(useClack);
    }
    if (choice.value === "m") fileMode = "merge";
  }

  const defaultVault = path.join(os.homedir(), "arxiv-daily");
  const state: WizardState = {
    vaultRoot: defaultVault,
    providerId: DEFAULT_SETTINGS.llm.provider,
    baseUrl: DEFAULT_SETTINGS.llm.baseUrl,
    apiKey: "",
    tryFetch: true,
    remoteModels: [],
    model: DEFAULT_SETTINGS.llm.model,
    emailChoice: "skip",
    emailEnabled: false,
    emailMode: "self",
    emailTo: "",
    emailApiKey: "",
    hostedToken: "",
    categories: [...DEFAULT_SETTINGS.arxiv.categories],
    timezone: DEFAULT_SETTINGS.arxiv.timezone,
    summaryLanguage: "zh",
    topicName: "My research (edit me)",
    topicTag: "my-research",
    topicDescription:
      "Describe in natural language what papers belong in this topic (problems, methods, objects; what to exclude).",
  };

  let step: StepId = "vault";
  while (step !== "write") {
    const nav = await runStep(step, state, opts, useClack, stdout, stderr);
    if (nav === "abort") return finishCancel(useClack);
    if (nav === "back") {
      const prev = prevStep(step);
      if (!prev) return finishCancel(useClack);
      step = prev;
      continue;
    }
    const next = nextStep(step);
    if (!next) break;
    step = next;
  }

  // write step
  const preset =
    PROVIDER_PRESETS[state.providerId] ?? PROVIDER_PRESETS.custom!;
  const cacheDir = path.join(state.vaultRoot, ".cache", "arxiv-daily");
  const body = renderInitToml({
    vaultRoot: state.vaultRoot,
    cacheDir,
    apiKey: state.apiKey,
    baseUrl: state.baseUrl,
    model: state.model,
    provider: state.providerId,
    thinkingMode: preset.thinkingMode,
    categories: state.categories,
    timezone: state.timezone,
    summaryLanguage: state.summaryLanguage,
    topic: {
      name: state.topicName,
      tag: state.topicTag,
      description: state.topicDescription,
    },
    email: {
      enabled: state.emailEnabled,
      mode: state.emailMode,
      to: state.emailTo,
      apiKey: state.emailApiKey,
      hostedToken: state.hostedToken,
    },
  });

  let finalBody = body;
  if (fileMode === "merge" && existing) {
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

  if (useClack) p.outro(nextSteps);
  else writeLine(stdout, nextSteps);
  return 0;
}

async function runStep(
  step: StepId,
  state: WizardState,
  opts: InitOptions,
  useClack: boolean,
  stdout: WritableTextStream,
  stderr: WritableTextStream,
): Promise<Nav> {
  const canBack = step !== "vault";

  switch (step) {
    case "vault": {
      const homeDefault = path.join(os.homedir(), "arxiv-daily");
      const v = await askText(opts, {
        message: "Vault root path (where daily/ and papers/ are written)",
        placeholder: "~/arxiv-daily",
        initialValue: state.vaultRoot || homeDefault,
        validate: (s) =>
          s.trim() ? undefined : "Required — path to your notes folder",
        allowBack: false,
      });
      if (v.nav !== "next") return v.nav;
      state.vaultRoot = expandUserPath(v.value.trim());
      return "next";
    }
    case "provider": {
      const keys = Object.keys(PROVIDER_PRESETS);
      const pick = await askSelect(opts, {
        message: "LLM provider",
        options: keys.map((key) => ({
          value: key,
          label: PROVIDER_PRESETS[key]!.name,
          hint: key,
        })),
        initialValue: state.providerId,
        allowBack: canBack,
      });
      if (pick.nav !== "next") return pick.nav;
      state.providerId = pick.value;
      const preset = PROVIDER_PRESETS[state.providerId] ?? PROVIDER_PRESETS.custom!;
      // Use the preset URL as default (Custom → empty string on purpose).
      state.baseUrl = preset.baseUrl ?? "";
      state.remoteModels = [];
      if (preset.models[0]) {
        state.model = preset.models[0].value;
      }
      return "next";
    }
    case "baseUrl": {
      const preset =
        PROVIDER_PRESETS[state.providerId] ?? PROVIDER_PRESETS.custom!;
      // Custom preset has baseUrl ""; do not substitute DeepSeek default.
      const def = preset.baseUrl;
      const v = await askText(opts, {
        message:
          state.providerId === "custom"
            ? "API base URL (required for Custom, e.g. https://api.example.com/v1)"
            : "API base URL",
        initialValue: state.baseUrl,
        placeholder:
          def || "https://api.example.com/v1",
        validate: (s) => (s.trim() ? undefined : "Base URL is required"),
        allowBack: canBack,
      });
      if (v.nav !== "next") return v.nav;
      state.baseUrl = v.value.trim();
      return "next";
    }
    case "apiKey": {
      const v = await askPassword(opts, {
        message: "LLM API key (stored in config; do not commit)",
        validate: (s) => (s.trim() ? undefined : "API key is required"),
        allowBack: canBack,
      });
      if (v.nav !== "next") return v.nav;
      state.apiKey = v.value.trim();
      return "next";
    }
    case "fetchModels": {
      const conf = await askConfirm(opts, {
        message: "Test connection and load model list from the provider?",
        initialValue: state.tryFetch,
        allowBack: canBack,
      });
      if (conf.nav !== "next") return conf.nav;
      state.tryFetch = conf.value;
      state.remoteModels = [];
      if (!state.tryFetch) return "next";

      const spin = useClack ? p.spinner() : null;
      spin?.start("Calling provider /models …");
      try {
        const models = opts.fetchModels
          ? await opts.fetchModels({
              apiKey: state.apiKey,
              baseUrl: state.baseUrl,
            })
          : await defaultFetchModels(state.apiKey, state.baseUrl);
        state.remoteModels = models;
        spin?.stop(
          models.length > 0
            ? `Connected — ${models.length} model(s) listed`
            : "Connected — empty model list",
        );
        if (!useClack) {
          writeLine(
            stdout,
            models.length > 0
              ? `Connected — ${models.length} model(s)`
              : "Connected — no models returned",
          );
        }
        if (models[0]) state.model = models[0];
      } catch (e) {
        const msg = (e as Error).message;
        spin?.stop(`Could not list models: ${msg}`);
        if (!useClack) writeLine(stderr, `Could not list models: ${msg}`);
        if (useClack) p.log.warn("You can still pick or type a model id.");
        else writeLine(stdout, "You can still pick or type a model id.");
      }
      return "next";
    }
    case "model": {
      const preset =
        PROVIDER_PRESETS[state.providerId] ?? PROVIDER_PRESETS.custom!;
      let options: Array<{ value: string; label: string; hint?: string }> = [];
      if (state.remoteModels.length > 0) {
        options = state.remoteModels.slice(0, 40).map((id) => ({
          value: id,
          label: id,
        }));
        if (state.remoteModels.length > 40) {
          options.push({ value: "__other__", label: "Other… (type model id)" });
        }
      } else if (preset.models.length > 0 && state.providerId !== "custom") {
        options = [
          ...preset.models.map((m) => ({
            value: m.value,
            label: m.label,
            hint: m.value,
          })),
          { value: "__other__", label: "Other… (type model id)" },
        ];
      }
      if (options.length === 0) {
        const typed = await askText(opts, {
          message: "Model id",
          initialValue: state.model,
          validate: (s) => (s.trim() ? undefined : "Required"),
          allowBack: canBack,
        });
        if (typed.nav !== "next") return typed.nav;
        state.model = typed.value.trim();
        return "next";
      }
      const initial = options.some((o) => o.value === state.model)
        ? state.model
        : options[0]!.value;
      const picked = await askSelect(opts, {
        message: "Model",
        options,
        initialValue: initial,
        allowBack: canBack,
      });
      if (picked.nav !== "next") return picked.nav;
      if (picked.value === "__other__") {
        const typed = await askText(opts, {
          message: "Model id",
          initialValue: state.model,
          validate: (s) => (s.trim() ? undefined : "Required"),
          allowBack: true,
        });
        if (typed.nav !== "next") return typed.nav;
        state.model = typed.value.trim();
      } else {
        state.model = picked.value;
      }
      return "next";
    }
    case "email": {
      const choice = await askSelect(opts, {
        message: "Email digests after a successful daily run?",
        options: [
          { value: "skip", label: "Skip for now", hint: "local files only" },
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
        initialValue: state.emailChoice,
        allowBack: canBack,
      });
      if (choice.nav !== "next") return choice.nav;
      state.emailChoice = choice.value as "skip" | "self" | "hosted";
      state.emailEnabled = false;
      state.emailTo = "";
      state.emailApiKey = "";
      state.hostedToken = "";
      state.emailMode = "self";

      if (state.emailChoice === "skip") return "next";

      if (state.emailChoice === "self") {
        state.emailMode = "self";
        if (useClack) {
          p.note(
            "Create a key at https://resend.com\nWith From empty, usually only TO your Resend account email.",
            "Send yourself",
          );
        }
        const to = await askText(opts, {
          message: "Your inbox (To)",
          initialValue: state.emailTo,
          validate: (s) => (s.trim() ? undefined : "Required"),
          allowBack: true,
        });
        if (to.nav !== "next") return to.nav;
        state.emailTo = to.value.trim();
        const reKey = await askPassword(opts, {
          message: "Resend API key (re_…)",
          validate: (s) => (s.trim() ? undefined : "Required"),
          allowBack: true,
        });
        if (reKey.nav !== "next") return reKey.nav;
        state.emailApiKey = reKey.value.trim();
        const enable = await askConfirm(opts, {
          message: "Enable daily auto-send now? (prefer email test first)",
          initialValue: false,
          allowBack: true,
        });
        if (enable.nav !== "next") return enable.nav;
        state.emailEnabled = enable.value;
        return "next";
      }

      // hosted
      state.emailMode = "hosted";
      if (useClack) {
        p.note(
          [
            "We email a link. Open it, copy the LONG code from the web page",
            "(not the short code in the link), then paste it here.",
            "Shared free Beta — a few messages per inbox per UTC day.",
          ].join("\n"),
          "Official delivery",
        );
      }
      const to = await askText(opts, {
        message: "Email to verify (To)",
        initialValue: state.emailTo,
        validate: (s) => (s.trim() ? undefined : "Required"),
        allowBack: true,
      });
      if (to.nav !== "next") return to.nav;
      state.emailTo = to.value.trim();

      const sendNow = await askConfirm(opts, {
        message: "Send verification email now?",
        initialValue: true,
        allowBack: true,
      });
      if (sendNow.nav !== "next") return sendNow.nav;
      if (sendNow.value) {
        const spin = useClack ? p.spinner() : null;
        spin?.start("Sending verification email…");
        try {
          const start =
            opts.startHostedVerify ??
            (async (email: string) => {
              const host = buildNodeHostAdapters({ rootDir: process.cwd() });
              await startHostedEmailVerification({ http: host.http, email });
            });
          await start(state.emailTo);
          spin?.stop(`Sent to ${state.emailTo} — open the link, copy the long code`);
          if (!useClack) writeLine(stdout, `Sent to ${state.emailTo}`);
        } catch (e) {
          spin?.stop(`Send failed: ${(e as Error).message}`);
          if (!useClack) {
            writeLine(stderr, `Send failed: ${(e as Error).message}`);
          }
          if (useClack) p.log.warn("Later: arxiv-daily email verify-start");
        }
      }

      const tokenRaw = await askText(opts, {
        message: "Paste verification code (long token), or leave empty",
        placeholder: "leave empty to fill later",
        initialValue: state.hostedToken,
        allowBack: true,
      });
      if (tokenRaw.nav !== "next") return tokenRaw.nav;
      state.hostedToken = tokenRaw.value.trim().replace(/\s+/g, "");
      if (state.hostedToken) {
        const enable = await askConfirm(opts, {
          message: "Enable daily auto-send now? (prefer email test first)",
          initialValue: false,
          allowBack: true,
        });
        if (enable.nav !== "next") return enable.nav;
        state.emailEnabled = enable.value;
      } else if (useClack) {
        p.log.info(
          "Set email.hosted_token after verify, then: arxiv-daily email test",
        );
      }
      return "next";
    }
    case "categories": {
      const flatCats = ARXIV_CATEGORIES.flatMap((g) =>
        g.categories.map((c) => ({
          value: c.id,
          label: c.id,
          hint: `${g.label} · ${c.name}`,
        })),
      );
      const pick = await askMultiSelect(opts, {
        message:
          "arXiv categories (Space toggle, Enter confirm; select only ← Back to go back)",
        options: [
          ...flatCats,
          { value: BACK, label: "← Back", hint: "previous step" },
        ],
        initialValues: state.categories.filter((id) =>
          flatCats.some((c) => c.value === id),
        ),
        required: true,
      });
      if (pick.nav !== "next") return pick.nav;
      if (pick.value.length === 1 && pick.value[0] === BACK) return "back";
      const cats = pick.value.filter((id) => id !== BACK);
      state.categories =
        cats.length > 0 ? cats : [...DEFAULT_SETTINGS.arxiv.categories];
      return "next";
    }
    case "timezone": {
      const pick = await askSelect(opts, {
        message: "Timezone for “today”",
        options: [
          ...COMMON_TIMEZONES.map((tz) => ({ value: tz, label: tz })),
          { value: "__other__", label: "Other IANA name…" },
        ],
        initialValue: state.timezone,
        allowBack: canBack,
      });
      if (pick.nav !== "next") return pick.nav;
      if (pick.value === "__other__") {
        const typed = await askText(opts, {
          message: "IANA timezone",
          initialValue: state.timezone,
          validate: (s) => (s.trim() ? undefined : "Required"),
          allowBack: true,
        });
        if (typed.nav !== "next") return typed.nav;
        state.timezone = typed.value.trim();
      } else {
        state.timezone = pick.value;
      }
      return "next";
    }
    case "language": {
      const pick = await askSelect(opts, {
        message: "Summary language for reports",
        options: [
          { value: "zh", label: "Chinese (zh)" },
          { value: "en", label: "English (en)" },
        ],
        initialValue: state.summaryLanguage,
        allowBack: canBack,
      });
      if (pick.nav !== "next") return pick.nav;
      state.summaryLanguage = pick.value === "en" ? "en" : "zh";
      return "next";
    }
    case "topic": {
      if (useClack) {
        p.note(
          "Topics control filtering. Add more later under [[arxiv.topics]] in the config.",
          "Research topic",
        );
      }
      const name = await askText(opts, {
        message: "Topic display name",
        initialValue: state.topicName,
        allowBack: canBack,
      });
      if (name.nav !== "next") return name.nav;
      state.topicName = name.value.trim() || "My research (edit me)";
      const tag = await askText(opts, {
        message: "Topic tag slug",
        initialValue: state.topicTag,
        allowBack: true,
      });
      if (tag.nav !== "next") return tag.nav;
      state.topicTag = tag.value.trim() || "my-research";
      const desc = await askText(opts, {
        message: "Topic description (what papers belong here)",
        initialValue: state.topicDescription,
        allowBack: true,
      });
      if (desc.nav !== "next") return desc.nav;
      state.topicDescription =
        desc.value.trim() ||
        "Describe in natural language what papers belong in this topic.";
      return "next";
    }
    default:
      return "next";
  }
}


function expandUserPath(inputPath: string): string {
  const trimmed = inputPath.trim();
  if (!trimmed) return trimmed;
  if (trimmed === "~") return os.homedir();
  if (trimmed.startsWith("~/") || trimmed.startsWith("~\\")) {
    return path.resolve(os.homedir(), trimmed.slice(2));
  }
  return path.resolve(trimmed);
}

function finishCancel(useClack: boolean): number {
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

type AskResult<T> =
  | { nav: "next"; value: T }
  | { nav: "back" }
  | { nav: "abort" };

function withBack<T extends { value: string; label: string; hint?: string }>(
  options: T[],
  allowBack: boolean,
): T[] {
  if (!allowBack) return options;
  // Put Back last so numbered choices stay stable (1 = first real option).
  return [
    ...options,
    { value: BACK, label: "← Back", hint: "previous step" } as T,
  ];
}

/**
 * Map cancel (Esc/Ctrl+C) to back when allowBack, else abort.
 */
function mapCancel(allowBack: boolean): "back" | "abort" {
  return allowBack ? "back" : "abort";
}

async function askText(
  opts: InitOptions,
  args: {
    message: string;
    placeholder?: string;
    initialValue?: string;
    validate?: (value: string) => string | undefined;
    allowBack?: boolean;
  },
): Promise<AskResult<string>> {
  const allowBack = Boolean(args.allowBack);
  if (opts.ask) {
    for (;;) {
      const hint = allowBack ? " (or type back)" : "";
      const v = await plainPrompt(
        opts,
        `${args.message}${args.initialValue ? ` [${args.initialValue}]` : ""}${hint}: `,
      );
      if (allowBack && v.trim().toLowerCase() === "back") {
        return { nav: "back" };
      }
      const value = v.trim() ? v : (args.initialValue ?? "");
      const err = args.validate?.(value);
      if (!err) return { nav: "next", value };
      writeLine(opts.stderr ?? process.stderr, `  ${err}`);
    }
  }
  // clack text: cancel → back/abort; no native Back option on text fields
  // Prepend note when back is available
  if (allowBack) {
    p.log.message("Esc / Ctrl+C = previous step");
  }
  const result = await p.text({
    message: args.message,
    placeholder: args.placeholder,
    initialValue: args.initialValue,
    validate: args.validate
      ? (value) => args.validate!(value ?? "")
      : undefined,
  });
  if (cancelled(result)) return { nav: mapCancel(allowBack) };
  return { nav: "next", value: String(result) };
}

async function askPassword(
  opts: InitOptions,
  args: {
    message: string;
    validate?: (value: string) => string | undefined;
    allowBack?: boolean;
  },
): Promise<AskResult<string>> {
  const allowBack = Boolean(args.allowBack);
  if (opts.ask) {
    for (;;) {
      const hint = allowBack ? " (or type back)" : "";
      const v = await plainPrompt(opts, `${args.message}${hint}: `);
      if (allowBack && v.trim().toLowerCase() === "back") {
        return { nav: "back" };
      }
      const err = args.validate?.(v);
      if (!err) return { nav: "next", value: v };
      writeLine(opts.stderr ?? process.stderr, `  ${err}`);
    }
  }
  if (allowBack) p.log.message("Esc / Ctrl+C = previous step");
  const result = await p.password({
    message: args.message,
    validate: args.validate
      ? (value) => args.validate!(value ?? "")
      : undefined,
  });
  if (cancelled(result)) return { nav: mapCancel(allowBack) };
  return { nav: "next", value: String(result) };
}

async function askConfirm(
  opts: InitOptions,
  args: {
    message: string;
    initialValue?: boolean;
    allowBack?: boolean;
  },
): Promise<AskResult<boolean>> {
  const allowBack = Boolean(args.allowBack);
  if (opts.ask) {
    const def = args.initialValue ? "Y/n" : "y/N";
    const hint = allowBack ? " / back" : "";
    const v = (
      await plainPrompt(opts, `${args.message} [${def}${hint}]: `)
    )
      .trim()
      .toLowerCase();
    if (allowBack && v === "back") return { nav: "back" };
    if (!v) return { nav: "next", value: Boolean(args.initialValue) };
    return { nav: "next", value: v === "y" || v === "yes" };
  }
  // confirm doesn't support Back option; use select instead when allowBack
  if (allowBack) {
    const pick = await askSelect(opts, {
      message: args.message,
      options: [
        { value: "yes", label: "Yes" },
        { value: "no", label: "No" },
      ],
      initialValue: args.initialValue ? "yes" : "no",
      allowBack: true,
    });
    if (pick.nav !== "next") return pick;
    return { nav: "next", value: pick.value === "yes" };
  }
  const result = await p.confirm({
    message: args.message,
    initialValue: args.initialValue,
  });
  if (cancelled(result)) return { nav: "abort" };
  return { nav: "next", value: Boolean(result) };
}

async function askSelect(
  opts: InitOptions,
  args: {
    message: string;
    options: Array<{ value: string; label: string; hint?: string }>;
    initialValue?: string;
    allowBack?: boolean;
  },
): Promise<AskResult<string>> {
  const allowBack = Boolean(args.allowBack);
  const options = withBack(args.options, allowBack);
  if (opts.ask) {
    const lines = options
      .map((o, i) => `  ${i + 1}) ${o.label}${o.hint ? ` (${o.hint})` : ""}`)
      .join("\n");
    writeLine(opts.stdout ?? process.stdout, lines);
    const raw = (
      await plainPrompt(
        opts,
        `${args.message} [default ${args.initialValue ?? "1"}]: `,
      )
    ).trim();
    if (allowBack && raw.toLowerCase() === "back") return { nav: "back" };
    if (!raw) {
      return {
        nav: "next",
        value: args.initialValue ?? args.options[0]!.value,
      };
    }
    const n = Number(raw);
    if (Number.isInteger(n) && n >= 1 && n <= options.length) {
      const val = options[n - 1]!.value;
      if (val === BACK) return { nav: "back" };
      return { nav: "next", value: val };
    }
    if (raw === BACK || raw === "back") return { nav: "back" };
    const byValue = options.find((o) => o.value === raw);
    if (byValue) {
      if (byValue.value === BACK) return { nav: "back" };
      return { nav: "next", value: byValue.value };
    }
    return {
      nav: "next",
      value: args.initialValue ?? args.options[0]!.value,
    };
  }
  const result = await p.select({
    message: args.message,
    options,
    initialValue: args.initialValue,
  });
  if (cancelled(result)) return { nav: mapCancel(allowBack) };
  if (result === BACK) return { nav: "back" };
  return { nav: "next", value: String(result) };
}

async function askMultiSelect(
  opts: InitOptions,
  args: {
    message: string;
    options: Array<{ value: string; label: string; hint?: string }>;
    initialValues?: string[];
    required?: boolean;
  },
): Promise<AskResult<string[]>> {
  if (opts.ask) {
    const lines = args.options
      .slice(0, 40)
      .map((o, i) => `  ${i + 1}) ${o.label}${o.hint ? ` — ${o.hint}` : ""}`)
      .join("\n");
    writeLine(opts.stdout ?? process.stdout, lines);
    const raw = (
      await plainPrompt(
        opts,
        `${args.message} [${(args.initialValues ?? []).join(",") || "1"}]: `,
      )
    ).trim();
    if (raw.toLowerCase() === "back") return { nav: "back" };
    if (!raw) {
      return {
        nav: "next",
        value: args.initialValues ?? [args.options[0]!.value],
      };
    }
    if (/[a-z]/i.test(raw) && !/^\d/.test(raw)) {
      const ids = raw.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean);
      if (ids.includes("back") || ids.includes(BACK)) return { nav: "back" };
      return { nav: "next", value: ids };
    }
    const indexes = raw
      .split(/[,\s]+/)
      .map((s) => Number(s.trim()))
      .filter(
        (n) =>
          Number.isInteger(n) && n >= 1 && n <= args.options.length,
      );
    const picked = indexes.map((i) => args.options[i - 1]!.value);
    if (picked.length === 1 && picked[0] === BACK) return { nav: "back" };
    return {
      nav: "next",
      value:
        picked.length > 0
          ? [...new Set(picked.filter((x) => x !== BACK))]
          : (args.initialValues ?? [args.options[0]!.value]),
    };
  }
  const result = await p.multiselect({
    message: args.message,
    options: args.options,
    initialValues: args.initialValues,
    required: args.required,
  });
  if (cancelled(result)) return { nav: "back" };
  const arr = result as string[];
  if (arr.length === 1 && arr[0] === BACK) return { nav: "back" };
  return {
    nav: "next",
    value: arr.filter((x) => x !== BACK),
  };
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
