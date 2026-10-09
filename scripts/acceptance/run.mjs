import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createFixtureEnvironment } from "./fixtures.mjs";
import { buildReport, writeReport, redact } from "./report.mjs";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const SUITES = ["workbench", "obsidian", "exploration"];
const HELP = `Usage: npm run test:acceptance -- [options]

  --suite workbench|obsidian|all   Choose fixed UI suites (default: all)
  --explore                      Also run the model API exploration
  --suite exploration            Run only model API exploration
  --model-config PATH            Read only [llm] from this CLI TOML file
  --max-steps N                  Model decisions, 1-300 (default: 50)
  --max-api-calls N              Model HTTP requests including retries
  --max-duration-ms N            Exploration time budget (default: 600000)
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

export function parseArgs(argv) {
  const options = { suites: ["workbench", "obsidian"], build: true, headless: true, maxSteps: 50, maxDurationMs: 600000 };
  let explicit = false, explore = false;
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
      case "--skip-build": options.build = false; break;
      case "--headed": options.headless = false; break;
      case "--list": options.list = true; break;
      case "--help": case "-h": options.help = true; break;
      case "--output": options.output = resolve(take()); break;
      case "--browser": options.executablePath = resolve(take()); break;
      case "--model-config": options.configPath = resolve(take()); break;
      case "--max-steps": options.maxSteps = number(1, 300); break;
      case "--max-api-calls": options.maxApiCalls = number(1, 1000); break;
      case "--max-duration-ms": options.maxDurationMs = number(1, 3600000); break;
      default: throw new TypeError(`Unknown acceptance option: ${flag}`);
    }
    if (rest.length && ["--explore", "--skip-build", "--headed", "--list", "--help", "-h"].includes(flag)) throw new TypeError(`${flag} does not take a value`);
  }
  if (explore && !options.suites.includes("exploration")) options.suites.push("exploration");
  return options;
}
async function buildProducts(suites, { artifactDir, signal, onProgress }) {
  const workspaces = [];
  if (suites.some(suite => suite === "workbench" || suite === "exploration")) workspaces.push("apps/cli");
  if (suites.includes("obsidian")) workspaces.push("obsidian-arxiv-daily");
  for (const workspace of workspaces) {
    if (signal?.aborted) throw signal.reason ?? new Error("Acceptance cancelled before build");
    onProgress(`Build ${workspace}`);
    const log = [];
    await new Promise((resolveBuild, rejectBuild) => {
      const child = spawn(process.platform === "win32" ? "npm.cmd" : "npm", ["run", "build", "--workspace", workspace], {
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
    try { await build(selected, { artifactDir, signal: options.signal, onProgress }); }
    catch (error) {
      if (!options.signal?.aborted) for (const suite of selected) results.push(failedSuite(suite, error));
      return persist();
    }
  }
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
        ...(suite === "exploration" ? { configPath: options.configPath, maxSteps: options.maxSteps, maxDurationMs: options.maxDurationMs, maxApiCalls: options.maxApiCalls } : {}),
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
            result.scenarios.push({ id: `${suite}.fixture-contract`, title: "Every external request matches the controlled protocol", status: "failed", durationMs: 0, error: "Unexpected or malformed external requests reached the fixture; inspect request evidence", assertions: [{ label: "No unexpected fixture requests", passed: false, actual: unexpected.length }], artifacts: [requestsPath] });
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

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const controller = new AbortController();
  const interrupt = () => controller.abort(new Error("Acceptance interrupted; unfinished scenarios are not verified"));
  process.once("SIGINT", interrupt); process.once("SIGTERM", interrupt);
  try {
    const options = parseArgs(process.argv.slice(2));
    if (options.help) process.stdout.write(HELP);
    else if (options.list) {
      process.stdout.write("workbench    Real browser: settings, daily run, recovery, reading and persistence\nobsidian     Real Obsidian: settings, run/cancel/retry, reading and PDF navigation\nexploration  Opt-in model API: bounded UI tasks with independent evidence checks\n");
    } else {
      const report = await runAcceptance({ ...options, signal: controller.signal }, { onProgress: message => process.stdout.write(`${message}\n`) });
      process.stdout.write(`Acceptance ${report.status}: ${JSON.stringify(report.counts)}\nReport: ${join(report.artifactDir, "report.html")}\n`);
      process.exitCode = controller.signal.aborted ? 130 : report.exitCode;
    }
  } catch (error) {
    process.stderr.write(`${redact(error.message)}\n`);
    process.exitCode = 2;
  } finally { process.removeListener("SIGINT", interrupt); process.removeListener("SIGTERM", interrupt); }
}
