import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createFixtureEnvironment } from "./fixtures.mjs";
import { buildReport, writeReport, redact } from "./report.mjs";

import { collectBuildEvidence } from "../build-evidence.mjs";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const SUITES = ["workbench", "obsidian", "exploration"];
const HELP = `Usage: npm run test:acceptance -- [options]

  --suite workbench|obsidian|all   Choose fixed UI suites (default: all)
  --explore                      Also run the model API exploration
  --suite exploration            Run only model API exploration
  --host workbench|obsidian      Exploration target product (default: workbench)
  --model-config PATH            Read only [llm] from this CLI TOML file
  --task ID[,ID...]              Run only these exploration goal tasks (repeatable)
  --persona ID[,ID...]           Run only these exploration persona tasks (repeatable)
  --release                      Also run every persona for --host (unless --persona overrides it)
  --max-steps N                  Model decisions, 1-300 (default: 90; 220 with --release)
  --max-api-calls N              Model HTTP requests including retries (default: 2x the effective step budget)
  --max-duration-ms N            Exploration time budget (default: 900000; 1200000 with --release)
  --task-max-steps N             Default per-task decision budget (default: 20 task, 15 persona)
  --task-max-api-calls N         Default per-task API-call budget (default: 2x its step budget)
  --browser PATH                 Browser executable override
  --headed                       Show the workbench browser
  --skip-build                   Use an existing build (for local debugging)
  --output PATH                  New, empty artifact directory
  --list                         List available suites without launching apps
  --help                         Show this help

Fixed suites use isolated data and controlled external responses.
Exploration uses the existing workbench model config only when selected.
Exit status: 0 passed; 1 failed; 2 blocked or incomplete.
`;

const EXPLORATION_HOSTS = ["workbench", "obsidian"];

// Run-wide defaults when `--release` is given and the caller did not pass an explicit budget flag.
// Measured from the real-model release runs in docs/helm/2026-10-10-model-driven-release-exploration
// (2026-10-12, two runs per host after P4 chunk 2's fixes): obsidian used up to 106 model calls over
// ~536s, workbench up to 80 calls over ~359s. These defaults give roughly 2x headroom over the larger
// (obsidian) figure, while staying within the hard maxima (maxSteps <= 300, maxDurationMs <= 3600000),
// so a release run needs no manual --max-* flags. `--max-api-calls` has no release-specific default: it
// keeps falling back to 2x the effective step budget (440 under --release), which already has far more
// headroom than observed (max 106) and needs no separate constant.
const RELEASE_DEFAULT_MAX_STEPS = 220;
const RELEASE_DEFAULT_MAX_DURATION_MS = 1200000;

export function parseArgs(argv) {
  const options = { suites: ["workbench", "obsidian"], build: true, headless: true, host: "workbench" };
  let explicit = false, explore = false, explicitMaxSteps = false, explicitMaxDurationMs = false;
  for (let i = 0; i < argv.length; i++) {
    const [flag, ...rest] = argv[i].split("=");
    const take = () => {
      const value = rest.length ? rest.join("=") : argv[++i];
      if (!value || value.startsWith("--")) throw new TypeError(`Missing value for ${flag}`);
      return value;
    };
    const number = (min, max) => {
      const raw = take(), value = Number(raw);
      if (!/^\d+$/.test(raw) || !Number.isSafeInteger(value) || value < min || value > max) throw new TypeError(`${flag} must be an integer from ${min} to ${max}`);
      return value;
    };
    switch (flag) {
      case "--suite": case "--suites": {
        const values = take().split(",").flatMap(value => value === "all" ? ["workbench", "obsidian"] : [value]);
        if (values.some(value => !SUITES.includes(value))) throw new TypeError(`Unknown suite: ${values.join(",")}`);
        options.suites = [...new Set([...(explicit ? options.suites : []), ...values])]; explicit = true;
        break;
      }
      case "--explore": explore = true; break;
      case "--release": options.release = true; break;
      case "--host": {
        const value = take();
        if (!EXPLORATION_HOSTS.includes(value)) throw new TypeError(`Unknown exploration host: ${value}. Supported hosts: ${EXPLORATION_HOSTS.join(", ")}`);
        options.host = value;
        break;
      }
      case "--skip-build": options.build = false; break;
      case "--headed": options.headless = false; break;
      case "--list": options.list = true; break;
      case "--help": case "-h": options.help = true; break;
      case "--output": options.output = resolve(take()); break;
      case "--browser": options.executablePath = resolve(take()); break;
      case "--model-config": options.configPath = resolve(take()); break;
      case "--task": {
        const ids = take().split(",").map(value => value.trim()).filter(Boolean);
        if (!ids.length) throw new TypeError("--task needs at least one id");
        options.taskIds = [...(options.taskIds ?? []), ...ids];
        break;
      }
      case "--persona": {
        const ids = take().split(",").map(value => value.trim()).filter(Boolean);
        if (!ids.length) throw new TypeError("--persona needs at least one id");
        options.personaIds = [...(options.personaIds ?? []), ...ids];
        break;
      }
      case "--max-steps": options.maxSteps = number(1, 300); explicitMaxSteps = true; break;
      case "--max-api-calls": options.maxApiCalls = number(1, 1000); break;
      case "--max-duration-ms": options.maxDurationMs = number(1, 3600000); explicitMaxDurationMs = true; break;
      case "--task-max-steps": options.taskMaxSteps = number(1, 300); break;
      case "--task-max-api-calls": options.taskMaxApiCalls = number(1, 1000); break;
      default: throw new TypeError(`Unknown acceptance option: ${flag}`);
    }
    if (rest.length && ["--explore", "--release", "--skip-build", "--headed", "--list", "--help", "-h"].includes(flag)) throw new TypeError(`${flag} does not take a value`);
  }
  if (explore && !options.suites.includes("exploration")) options.suites.push("exploration");
  // --release means a full-library run (every goal task plus every persona); apply its own run-wide
  // budget defaults (see RELEASE_DEFAULT_* above) unless the caller passed an explicit --max-steps /
  // --max-duration-ms, regardless of whether --release appeared before or after those flags.
  if (!explicitMaxSteps) options.maxSteps = options.release ? RELEASE_DEFAULT_MAX_STEPS : 90;
  if (!explicitMaxDurationMs) options.maxDurationMs = options.release ? RELEASE_DEFAULT_MAX_DURATION_MS : 900000;
  return options;
}
export function npmBuildCommand(platform = process.platform, env = process.env) {
  if (env.npm_execpath) return { command: process.execPath, prefix: [env.npm_execpath] };
  if (platform === "win32") throw new Error("Run acceptance through npm run test:acceptance on Windows");
  return { command: "npm", prefix: [] };
}

export function workspacesToBuild(suites, host = "workbench") {
  const workspaces = [];
  if (suites.includes("workbench") || (suites.includes("exploration") && host === "workbench")) workspaces.push("apps/cli");
  if (suites.includes("obsidian") || (suites.includes("exploration") && host === "obsidian")) workspaces.push("obsidian-arxiv-daily");
  return workspaces;
}

async function buildProducts(suites, { artifactDir, signal, onProgress, host }) {
  const workspaces = workspacesToBuild(suites, host);
  for (const workspace of workspaces) {
    if (signal?.aborted) throw signal.reason ?? new Error("Acceptance cancelled before build");
    onProgress(`Build ${workspace}`);
    const log = [];
    await new Promise((resolveBuild, rejectBuild) => {
      const npm = npmBuildCommand();
      const child = spawn(npm.command, [...npm.prefix, "run", "build", "--workspace", workspace], {
        cwd: repoRoot, stdio: ["ignore", "pipe", "pipe"], detached: process.platform !== "win32", windowsHide: true,
      });
      let forceTimer;
      const stop = () => {
        try { if (process.platform !== "win32" && child.pid) process.kill(-child.pid, "SIGTERM"); else child.kill("SIGTERM"); }
        catch (error) { if (error.code !== "ESRCH") log.push(error.message); }
        forceTimer = setTimeout(() => {
          try { if (process.platform !== "win32" && child.pid) process.kill(-child.pid, "SIGKILL"); else child.kill("SIGKILL"); }
          catch { /* Already reclaimed. */ }
        }, 3000);
      };
      signal?.addEventListener("abort", stop, { once: true });
      const collect = data => { if (log.join("").length < 2 * 1024 * 1024) log.push(String(data)); };
      child.stdout.on("data", collect); child.stderr.on("data", collect);
      child.once("error", rejectBuild);
      child.once("close", async code => {
        clearTimeout(forceTimer); signal?.removeEventListener("abort", stop);
        try {
          const logPath = join(artifactDir, `build-${workspace.replaceAll("/", "-")}.log`);
          await writeFile(logPath, log.join(""));
          if (code !== 0 || signal?.aborted) rejectBuild(Object.assign(new Error(`Build ${workspace} ${signal?.aborted ? "cancelled" : `failed with exit ${code}`}; see build log`), { artifacts: [logPath] }));
          else resolveBuild();
        } catch (error) { rejectBuild(error); }
      });
    });
  }
}

const defaultRunners = {
  workbench: async options => (await import("./workbench.mjs")).runWorkbenchAcceptance(options),
  obsidian: async options => (await import("./obsidian.mjs")).runObsidianAcceptance(options),
  exploration: async options => (await import("./exploration.mjs")).runExploration(options),
};

function failedSuite(suite, error, status = "failed") {
  const message = redact(error?.message ?? String(error));
  return { suite, status, errors: [message], artifacts: error?.artifacts ?? [], scenarios: [{ id: `${suite}.setup`, title: `${suite} setup and execution`, status, durationMs: 0, error: message, assertions: [], steps: [], artifacts: [] }] };
}

/** One fixture per suite; a failure never suppresses other requested evidence. */
export async function runAcceptance(options, dependencies = {}) {
  const createFixture = dependencies.createFixture ?? createFixtureEnvironment;
  const build = dependencies.build ?? buildProducts;
  const runners = dependencies.runners ?? defaultRunners;
  const save = dependencies.writeReport ?? writeReport;
  const onProgress = dependencies.onProgress ?? (() => {});
  const selected = options.suites;
  if (!Array.isArray(selected) || !selected.length || selected.some(suite => !SUITES.includes(suite)) || new Set(selected).size !== selected.length) throw new TypeError("Select unique supported acceptance suites");
  let artifactDir = options.output;
  if (artifactDir) {
    let entries;
    try { entries = await readdir(artifactDir); } catch (error) { if (error.code !== "ENOENT") throw error; }
    if (entries?.length) throw new Error("Acceptance output must be a new or empty directory; refusing to mix old evidence");
    await mkdir(artifactDir, { recursive: true });
  } else {
    const base = join(repoRoot, "output/playwright/acceptance");
    await mkdir(base, { recursive: true });
    artifactDir = await mkdtemp(join(base, "run-"));
  }
  onProgress(`Acceptance artifacts: ${artifactDir}`);
  const results = [];
  const metadata = { startedAt: new Date().toISOString(), buildRequested: options.build !== false, node: process.version, artifactDir, contentQualityEvaluation: false };
  const persist = async () => {
    const report = { ...buildReport({ selectedSuites: selected, suites: results, metadata }), artifactDir };
    await save(artifactDir, report);
    return report;
  };
  await persist();
  if (options.signal?.aborted) return persist();
  if (options.build !== false) {
    try { await build(selected, { artifactDir, signal: options.signal, onProgress, host: options.host }); }
    catch (error) {
      if (!options.signal?.aborted) for (const suite of selected) results.push(failedSuite(suite, error));
      return persist();
    }
  }
  Object.assign(metadata, await collectBuildEvidence(repoRoot, workspacesToBuild(selected, options.host)));
  for (const warning of metadata.buildWarnings) onProgress(`WARNING: ${warning}`);
  for (const suite of selected) {
    if (options.signal?.aborted) break;
    let fixture, result;
    const directory = join(artifactDir, suite);
    await mkdir(directory, { recursive: true });
    onProgress(`Run ${suite}`);
    try {
      fixture = await createFixture({ configured: true });
      result = await runners[suite]({
        fixture, artifactDir: directory, signal: options.signal,
        executablePath: options.executablePath, headless: options.headless,
        ...(suite === "exploration" ? { host: options.host, configPath: options.configPath, maxSteps: options.maxSteps, maxDurationMs: options.maxDurationMs, maxApiCalls: options.maxApiCalls, taskIds: options.taskIds, personaIds: options.personaIds, taskMaxSteps: options.taskMaxSteps, taskMaxApiCalls: options.taskMaxApiCalls, release: options.release } : {}),
      });
      if (!result || result.suite !== suite) throw new Error(`Runner returned no matching evidence for ${suite}`);
    } catch (error) {
      const blocked = options.signal?.aborted || error?.code === "ACCEPTANCE_BLOCKED" || /BlockedError$/.test(error?.name ?? "");
      result = failedSuite(suite, error, blocked ? "blocked" : "failed");
    } finally {
      try {
        if (fixture?.server?.requests) {
          const requestsPath = join(directory, "fixture-requests.json");
          await writeFile(requestsPath, JSON.stringify(redact(fixture.server.requests), null, 2) + "\n");
          result.artifacts = [...(result.artifacts ?? []), requestsPath];
          const unexpected = fixture.server.requests.filter(request => ["unexpected", "fixture-error"].includes(request.kind));
          if (unexpected.length) {
            result.status = "failed";
            result.scenarios.push({ id: `${suite}.fixture-contract`, title: "Every external request matches the controlled protocol", status: "failed", durationMs: 0, error: redact(`Unexpected or malformed external requests reached the fixture: ${unexpected.slice(0, 10).map(request => `${request.method} ${request.url}`).join("; ")}; inspect request evidence`), assertions: [{ label: "No unexpected fixture requests", passed: false, actual: unexpected.length }], artifacts: [requestsPath] });
          }
        }
      } catch (error) {
        result ??= failedSuite(suite, error);
        result.status = "failed";
        result.scenarios.push({ id: `${suite}.evidence`, title: "Save external-request evidence", status: "failed", error: redact(error.message), assertions: [{ label: "Evidence was saved before fixture cleanup", passed: false }] });
      }
      try { await fixture?.dispose(); }
      catch (error) {
        result ??= failedSuite(suite, error);
        result.status = "failed";
        result.scenarios.push({ id: `${suite}.cleanup`, title: "Release the owned test environment", status: "failed", durationMs: 0, error: redact(error.message), assertions: [{ label: "Fixture cleanup completed", passed: false }] });
      }
    }
    results.push(result);
    onProgress(`${suite}: ${result.status}`);
    await persist();
  }
  metadata.completedAt = new Date().toISOString();
  return persist();
}

/**
 * `--list`'s own text, as a pure(ish) async function so a test can assert on it without spawning the
 * CLI. Task/persona ids come from each host's own task-library module (scripts/acceptance/
 * exploration-tasks/{obsidian,workbench}.mjs) — the single place chunks 2-5 add tasks/personas to — so
 * this listing never drifts out of sync with what `selectExplorationTasks` actually offers.
 */
export async function describeAcceptanceSuites() {
  const workbenchTasks = await import("./exploration-tasks/workbench.mjs");
  const obsidianTasks = await import("./exploration-tasks/obsidian.mjs");
  const personaLine = (host, ids) => `${host} ${ids.length ? ids.join(", ") : "(none yet)"}`;
  return `workbench    Real browser: settings, daily run, recovery, reading and persistence\nobsidian     Real Obsidian: settings, run/cancel/retry, reading and PDF navigation\nexploration  Opt-in model API: bounded UI tasks with independent evidence checks\n  hosts (--host):                workbench (default), obsidian\n  workbench tasks (--task):      ${workbenchTasks.GOAL_TASK_IDS.join(", ")}\n  obsidian tasks (--task):       ${obsidianTasks.GOAL_TASK_IDS.join(", ")}\n  personas (--persona):          ${personaLine("workbench", workbenchTasks.PERSONA_TASK_IDS)}; ${personaLine("obsidian", obsidianTasks.PERSONA_TASK_IDS)}\n  --release runs every persona for --host too (personas are otherwise opt-in only)\n`;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const controller = new AbortController();
  const interrupt = () => controller.abort(new Error("Acceptance interrupted; unfinished scenarios are not verified"));
  process.once("SIGINT", interrupt); process.once("SIGTERM", interrupt);
  try {
    const options = parseArgs(process.argv.slice(2));
    if (options.help) process.stdout.write(HELP);
    else if (options.list) {
      process.stdout.write(await describeAcceptanceSuites());
    } else {
      const report = await runAcceptance({ ...options, signal: controller.signal }, { onProgress: message => process.stdout.write(`${message}\n`) });
      process.stdout.write(`Acceptance ${report.status}: ${JSON.stringify(report.counts)}; needs review: ${report.reviewCount}\nReport: ${join(report.artifactDir, "report.html")}\n`);
      process.exitCode = controller.signal.aborted ? 130 : report.exitCode;
    }
  } catch (error) {
    process.stderr.write(`${redact(error.message)}\n`);
    process.exitCode = 2;
  } finally { process.removeListener("SIGINT", interrupt); process.removeListener("SIGTERM", interrupt); }
}
