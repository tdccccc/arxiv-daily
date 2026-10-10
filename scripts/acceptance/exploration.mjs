import { appendFile, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const moduleDirectory = path.dirname(fileURLToPath(import.meta.url));
const actionKeys = {
  click: ["ref"], fill: ["ref", "value"], select: ["ref", "value"],
  back: [], reload: [], wait: ["milliseconds"], scroll: ["ref", "direction", "pixels"],
  finding: ["summary", "evidence"], done: [], blocked: [],
};
const browserActions = new Set(["click", "fill", "select", "back", "reload", "wait", "scroll"]);

export class ExplorationActionError extends Error {
  constructor(message) { super(message); this.name = "ExplorationActionError"; }
}

/** Every element reference belongs to one observation. Selectors and code are never model inputs. */
export function parseExplorationAction(raw, observation) {
  if (typeof raw !== "string" || raw.length > 16000) throw new ExplorationActionError("Model reply must be a JSON object below 16000 characters");
  let value;
  try { value = JSON.parse(raw.trim()); }
  catch { throw new ExplorationActionError("Model reply is not valid JSON; return exactly one action object without Markdown fences"); }
  if (!value || Array.isArray(value) || typeof value !== "object" || !Object.hasOwn(actionKeys, value.type)) {
    throw new ExplorationActionError("Unknown action; only the documented browser operations are allowed");
  }
  const allowed = new Set(["type", "reason", ...actionKeys[value.type]]);
  if (Object.keys(value).some(key => !allowed.has(key))) throw new ExplorationActionError("Action contains unsupported fields");
  if (value.reason !== undefined && (typeof value.reason !== "string" || value.reason.length > 1000)) throw new ExplorationActionError("Action reason must be brief text");
  let control;
  if (["click", "fill", "select"].includes(value.type) || value.ref !== undefined) {
    control = observation?.controls?.find(item => item.ref === value.ref);
    if (!control) throw new ExplorationActionError("Invalid or stale ref; use a ref from the latest observation");
    if (control.restricted || control.sensitive) throw new ExplorationActionError(`This control is restricted: ${control.restricted || "secret"}`);
    if (control.disabled) throw new ExplorationActionError("This control is disabled");
  }
  if (value.type === "fill") {
    if (!["input", "textarea"].includes(control.tag) || control.readOnly || ["file", "hidden", "checkbox", "radio", "button", "submit", "reset"].includes(control.type)) {
      throw new ExplorationActionError("Fill requires an editable text, date, or number field");
    }
    if (typeof value.value !== "string" || value.value.length > 2000) throw new ExplorationActionError("Fill value must be text below 2000 characters");
  }
  if (value.type === "select" && (control.tag !== "select" || typeof value.value !== "string" || value.value.length > 256 ||
    (control.options && !control.options.some(option => option.value === value.value && !option.disabled)))) {
    throw new ExplorationActionError("Select requires an enabled option of the observed select control");
  }
  if (value.type === "wait" && (!Number.isInteger(value.milliseconds) || value.milliseconds < 1 || value.milliseconds > 2000)) {
    throw new ExplorationActionError("Wait is limited to 1–2000 milliseconds");
  }
  if (value.type === "scroll" && (!["up", "down"].includes(value.direction) || !Number.isInteger(value.pixels) || value.pixels < 1 || value.pixels > 1200)) {
    throw new ExplorationActionError("Scroll needs up/down and 1–1200 pixels");
  }
  if (value.type === "finding" && (typeof value.summary !== "string" || !value.summary.trim() || value.summary.length > 1000 || typeof value.evidence !== "string" || !value.evidence.trim() || value.evidence.length > 2000)) {
    throw new ExplorationActionError("A finding needs a brief summary and visible evidence");
  }
  if (value.type === "blocked" && !value.reason?.trim()) throw new ExplorationActionError("Blocked needs a reason");
  return value;
}

export function createExplorationRedactor(secrets = []) {
  const known = [...new Set(secrets.filter(value => typeof value === "string" && value.length > 0))].sort((a, b) => b.length - a.length);
  return value => {
    let text = String(value ?? "");
    for (const secret of known) text = text.split(secret).join("[REDACTED]");
    return text.replace(/\bBearer\s+[^\s,;]+/gi, "Bearer [REDACTED]")
      .replace(/\b(?:sk|rk|pk|api)[-_][A-Za-z0-9_-]{8,}\b/g, "[REDACTED]")
      .replace(/([?&](?:api[-_]?key|token|secret|password)=)[^&#\s]+/gi, "$1[REDACTED]");
  };
}

function sanitize(value, redact) {
  if (typeof value === "string") return redact(value);
  if (Array.isArray(value)) return value.map(item => sanitize(item, redact));
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key,
    /^(?:api[-_]?key|authorization|password|secret|access[-_]?token)$/i.test(key) ? "[REDACTED]" : sanitize(item, redact)]));
}

function limit(value, fallback, maximum, label) {
  const result = value ?? fallback;
  if (!Number.isInteger(result) || result < 1 || result > maximum) throw new Error(`${label} must be an integer between 1 and ${maximum}`);
  return result;
}

function cancellationError(signal) {
  return new Error(`Exploration cancelled: ${signal?.reason instanceof Error ? signal.reason.message : signal?.reason || "parent requested cancellation"}`);
}

async function withDeadline(operation, milliseconds, label, parentSignal) {
  if (parentSignal?.aborted) throw cancellationError(parentSignal);
  const controller = new AbortController();
  let timer;
  let rejectCancellation;
  const cancellation = new Promise((_, reject) => { rejectCancellation = reject; });
  const onAbort = () => { controller.abort(parentSignal.reason); rejectCancellation(cancellationError(parentSignal)); };
  parentSignal?.addEventListener("abort", onAbort, { once: true });
  try {
    return await Promise.race([
      Promise.resolve().then(() => operation(controller.signal)),
      cancellation,
      new Promise((_, reject) => { timer = setTimeout(() => {
        controller.abort(`${label} timeout`);
        const error = new Error(`${label} timeout after ${milliseconds}ms`);
        error.name = "ExplorationTimeoutError";
        reject(error);
      }, Math.max(1, milliseconds)); }),
    ]);
  } finally { clearTimeout(timer); parentSignal?.removeEventListener("abort", onAbort); }
}

function pendingScenario(task) {
  return { id: task.id, title: task.title, status: "not-run", durationMs: 0, steps: [], assertions: [], artifacts: [] };
}

/** A model chooses actions; only trusted task verifiers can establish a pass. */
export async function runExplorationLoop({ driver, model, tasks, artifactDir, maxSteps, maxDurationMs, maxInvalidActions, modelCallTimeoutMs, signal }) {
  const stepsLimit = limit(maxSteps, 50, 300, "maxSteps");
  const timeLimit = limit(maxDurationMs, 600000, 3600000, "maxDurationMs");
  const invalidLimit = limit(maxInvalidActions, 3, 10, "maxInvalidActions");
  const callTimeout = limit(modelCallTimeoutMs, 45000, 300000, "modelCallTimeoutMs");
  if (!Array.isArray(tasks) || tasks.some(task => !/^[a-z0-9-]+$/.test(task.id) || !task.title || !task.goal || typeof task.verify !== "function") || new Set(tasks.map(task => task.id)).size !== tasks.length) {
    throw new Error("Exploration tasks need unique ids, titles, goals, and trusted program verifiers");
  }
  const callModel = typeof model === "function" ? model : request => model.next(request);
  const redact = typeof model?.redact === "function" ? model.redact : createExplorationRedactor();
  const safe = value => sanitize(value, redact);
  const started = Date.now();
  const deadline = started + timeLimit;
  const scenarios = tasks.map(pendingScenario);
  const findings = [], errors = [], history = [];
  const metrics = { modelCalls: 0, actions: 0, elapsedMs: 0 };
  await mkdir(artifactDir, { recursive: true, mode: 0o700 });
  const eventsPath = path.join(artifactDir, "exploration-events.jsonl");
  const reportPath = path.join(artifactDir, "exploration.json");
  await writeFile(eventsPath, "", { mode: 0o600 });
  const artifacts = [eventsPath, reportPath];
  const record = event => appendFile(eventsPath, `${JSON.stringify(safe(event))}\n`);

  for (const [index, task] of tasks.entries()) {
    const scenario = scenarios[index], taskStart = Date.now();
    const actions = [], observations = [];
    let invalidActions = 0;
    try {
      if (signal?.aborted) throw cancellationError(signal);
      if (typeof task.prepare === "function") await withDeadline(childSignal => task.prepare({ driver, signal: childSignal }), Math.min(10000, deadline - Date.now()), "Task preparation", signal);
      while (true) {
        if (signal?.aborted) throw cancellationError(signal);
        if (Date.now() >= deadline) throw new Error("Exploration duration budget exhausted; unfinished tasks are not verified");
        const observation = safe(await withDeadline(() => driver.observe(), Math.min(10000, deadline - Date.now()), "UI observation", signal));
        observations.push(observation);
        if (observation.errors?.length) {
          scenario.status = "failed";
          scenario.error = observation.errors.join("; ");
          await record({ kind: "product-error", taskId: task.id, observation });
          break;
        }
        const verification = await withDeadline(childSignal => task.verify({ driver, observation, observations, actions, startedAt: taskStart, signal: childSignal }), Math.min(10000, deadline - Date.now()), "Task verification", signal);
        scenario.assertions = safe((verification?.checks ?? []).map(check => ({ label: String(check.label), passed: check.passed === true, ...(check.actual === undefined ? {} : { actual: check.actual }) })));
        const verified = verification?.passed === true && scenario.assertions.length > 0 && scenario.assertions.every(check => check.passed) && actions.some(action => action.type !== "wait");
        if (verified || verification?.failed === true) {
          scenario.status = verified ? "passed" : "failed";
          if (!verified) scenario.error = verification.error || "Program verification observed a product failure";
          const screenshot = await driver.capture?.(`${task.id}-result`);
          if (screenshot) scenario.artifacts.push(screenshot);
          await record({ kind: "verification", taskId: task.id, status: scenario.status, observation, assertions: scenario.assertions, screenshot });
          break;
        }
        if (metrics.modelCalls >= stepsLimit) throw new Error(`Exploration step budget exhausted (${stepsLimit}); unfinished tasks are not verified`);
        metrics.modelCalls += 1;
        const event = { step: metrics.modelCalls, taskId: task.id, observation };
        const request = {
          task: { id: task.id, title: task.title, goal: task.goal }, observation,
          checks: scenario.assertions,
          history: safe(history.slice(-8)),
          remaining: { steps: stepsLimit - metrics.modelCalls + 1, milliseconds: Math.max(0, deadline - Date.now()) },
        };
        let action;
        try {
          const reply = await withDeadline(childSignal => callModel({ ...request, signal: childSignal }), Math.min(callTimeout, deadline - Date.now()), "Model API call", signal);
          event.modelReply = typeof reply === "string" ? reply.slice(0, 16000) : "[Non-string model response]";
          try { action = parseExplorationAction(reply, observation); }
          catch (error) {
            invalidActions += 1;
            event.error = error.message;
          }
          if (action) {
            event.action = action;
            if (browserActions.has(action.type)) {
              try {
                const result = await withDeadline(childSignal => driver.execute(action, observation, { signal: childSignal }), Math.min(10000, deadline - Date.now()), "Browser action", signal);
                event.result = result ?? { ok: true };
                if (event.result.ok === false) throw new ExplorationActionError(event.result.error || "Browser action was refused");
                actions.push({ ...action, control: observation.controls?.find(control => control.ref === action.ref), step: metrics.modelCalls });
                metrics.actions += 1;
                invalidActions = 0;
              } catch (error) { invalidActions += 1; event.error = error.message; }
            } else if (action.type === "finding") {
              findings.push(safe({ taskId: task.id, step: event.step, status: "needs-review", summary: action.summary, evidence: action.evidence, observationId: observation.id }));
              invalidActions = 0;
            } else if (action.type === "done") {
              scenario.status = "blocked";
              scenario.error = "Model declared done without satisfying all program evidence checks (判据)";
            } else if (action.type === "blocked") {
              scenario.status = "blocked";
              scenario.error = `Model could not continue: ${action.reason}`;
            }
          }
          const screenshot = await driver.capture?.(`${task.id}-step-${event.step}`);
          if (screenshot) { event.screenshot = screenshot; scenario.artifacts.push(screenshot); }
        } catch (error) {
          event.error = error.message;
          scenario.status = "blocked";
          scenario.error = `Model or evidence collection unavailable: ${error.message}`;
        }
        scenario.steps.push(safe({ step: event.step, action: event.action ?? null, ...(event.error ? { error: event.error } : {}), observationId: observation.id }));
        await record(event);
        history.push(safe({ step: event.step, taskId: task.id, action: event.action, result: event.error ? { error: event.error } : event.result }));
        if (scenario.status !== "not-run") break;
        if (invalidActions >= invalidLimit) throw new Error(`Model produced ${invalidActions} invalid or unexecutable actions; exploration is blocked`);
      }
    } catch (error) {
      scenario.status = "blocked";
      scenario.error = redact(error.message);
    } finally { scenario.durationMs = Date.now() - taskStart; }
    if (scenario.error) { scenario.error = redact(scenario.error); errors.push(`${task.id}: ${scenario.error}`); }
    if (scenario.status !== "passed") break;
  }
  metrics.elapsedMs = Date.now() - started;
  for (const scenario of scenarios) scenario.findings = findings.filter(finding => finding.taskId === scenario.id);
  const result = safe({ suite: "exploration", status: scenarios.some(scenario => scenario.status === "failed") ? "failed" : scenarios.some(scenario => scenario.status === "blocked") ? "blocked" : scenarios.length && scenarios.every(scenario => scenario.status === "passed") ? "passed" : "not-run",
    scenarios, findings, artifacts, errors, metrics,
    coverage: { completed: scenarios.filter(scenario => scenario.status === "passed").map(scenario => scenario.id), incomplete: scenarios.filter(scenario => scenario.status !== "passed").map(scenario => scenario.id) },
    ...(model?.describe ? { model: model.describe } : {}),
    ...(typeof model?.metrics === "function" ? { modelUsage: model.metrics() } : {}),
  });
  await writeFile(reportPath, `${JSON.stringify(result, null, 2)}\n`, { mode: 0o600 });
  return result;
}

/** Compile a narrow TypeScript adapter without putting test code in the shipping CLI. */
export async function loadExplorationModel(options = {}) {
  const { build } = await import("esbuild");
  const directory = await mkdtemp(path.join(tmpdir(), "arxiv-exploration-model-"));
  try {
    const outfile = path.join(directory, "bridge.cjs");
    const cliRequire = createRequire(path.resolve(moduleDirectory, "../../apps/cli/package.json"));
    await build({ entryPoints: [path.join(moduleDirectory, "exploration-model.ts")], outfile, bundle: true, platform: "node", format: "cjs", target: "node20", loader: { ".md": "text" }, alias: { "smol-toml": cliRequire.resolve("smol-toml") }, logLevel: "silent" });
    const bridge = createRequire(import.meta.url)(outfile);
    return await bridge.createConfiguredExplorationModel(options);
  } finally { await rm(directory, { recursive: true, force: true }); }
}

export async function runExploration({ fixture, artifactDir, configPath, maxSteps, maxDurationMs, tasks, model, executablePath, headless, modelCallTimeoutMs, maxApiCalls, signal } = {}) {
  let session;
  let driver;
  let configured = model;
  let result;
  const directory = artifactDir ?? path.resolve("output/playwright/acceptance/exploration");
  try {
    if (signal?.aborted) throw cancellationError(signal);
    configured ??= await loadExplorationModel({ configPath, maxApiCalls: maxApiCalls ?? (maxSteps ?? 50) * 2 });
    const { createExplorationDriver, createDefaultExplorationTasks } = await import("./exploration-browser.mjs");
    const { startWorkbenchSession } = await import("./workbench-session.mjs");
    session = await startWorkbenchSession({ fixture, artifactDir: directory, executablePath, headless, signal });
    driver = await createExplorationDriver({ ...session, fixture, artifactDir: directory, redact: configured.redact, signal });
    const selected = tasks ?? await createDefaultExplorationTasks(fixture);
    result = await runExplorationLoop({ driver, model: configured, tasks: selected, artifactDir: directory, maxSteps, maxDurationMs, modelCallTimeoutMs, signal });
    result.artifacts.push(...(session.artifacts || []));
    return result;
  } catch (error) {
    const redact = typeof configured?.redact === "function" ? configured.redact : createExplorationRedactor();
    const reason = redact(error.message);
    await mkdir(directory, { recursive: true, mode: 0o700 });
    result = { suite: "exploration", status: "blocked", scenarios: [{ id: "exploration-setup", title: "Start isolated model exploration", status: "blocked", durationMs: 0, error: reason }, ...(tasks ?? []).map(pendingScenario)], errors: [reason], artifacts: [path.join(directory, "exploration.json"), ...(error.artifacts || [])] };
    await writeFile(result.artifacts[0], `${JSON.stringify(result, null, 2)}\n`, { mode: 0o600 });
    return result;
  } finally {
    const cleanupErrors = [];
    for (const cleanup of [() => driver?.dispose?.(), () => session?.stop?.()]) {
      try { await cleanup(); }
      catch (error) { cleanupErrors.push((configured?.redact || createExplorationRedactor())(error.message)); }
    }
    if (result) {
      if (cleanupErrors.length) {
        if (result.status !== "failed") result.status = "blocked";
        result.errors.push(...cleanupErrors);
        result.scenarios.push({ id: "exploration-cleanup", title: "Close the isolated browser and process", status: "blocked", durationMs: 0, error: cleanupErrors.join("; "), assertions: [], artifacts: [], steps: [] });
      }
      await writeFile(path.join(directory, "exploration.json"), `${JSON.stringify(result, null, 2)}\n`, { mode: 0o600 });
    }
  }
}
