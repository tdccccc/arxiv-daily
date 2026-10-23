import { appendFile, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const moduleDirectory = path.dirname(fileURLToPath(import.meta.url));
const actionKeys = {
  click: ["ref"], fill: ["ref", "value"], select: ["ref", "value"],
  back: [], reload: [], wait: ["milliseconds"], scroll: ["ref", "direction", "pixels"],
  press: ["key"],
  // P4 Chunk 2 Fix 2: a host-agnostic recovery action, usable from any state (never carries a ref, like
  // back/reload/wait) — see exploration-browser.mjs/exploration-obsidian.mjs's own `home` handling for
  // what "a safe, known starting view" means per host.
  home: [],
  finding: ["summary", "evidence"], done: [], blocked: [],
};
const browserActions = new Set(["click", "fill", "select", "back", "reload", "wait", "scroll", "press", "home"]);
/**
 * A host-agnostic keyboard press for the keyboard-only persona (P3): key presses go to whatever
 * element currently has focus, so this never carries a ref — a host's executor is responsible for
 * rejecting a press when focus has drifted outside the explorable scope. The allowlist stays small
 * and free of character-producing keys (so this cannot type arbitrary text) and free of destructive
 * shortcuts (no Ctrl/Alt combinations, no function keys).
 */
export const PRESS_KEYS = new Set(["Enter", "Escape", "Tab", "Shift+Tab", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space", "Home", "End", "PageUp", "PageDown"]);
/**
 * Keys that only move focus or cancel, never activating or changing the focused control. Hosts still
 * refuse them outside the explorable scope, but send them even when the focused control is restricted,
 * so a keyboard-only persona that lands on one can always Tab past it or Escape away.
 */
export const PRESS_NAVIGATION_KEYS = new Set(["Tab", "Shift+Tab", "Escape"]);
/** Persona tasks explore freely for findings only; they never carry a program verifier. */
const isPersonaTask = task => task?.kind === "persona";

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
  if (value.type === "select") {
    // P4 Chunk 2 Fix 3: a select's own options may be inline (`control.options`, every pre-Chunk-2
    // observation and every existing fake-driver test) or, now, a shared `control.optionsRef` resolved
    // against `observation.optionSets` (see collapseOptionSets below) — either shape is checked the same
    // way, so this never becomes a second, looser validation path for the deduplicated case.
    const options = control.options ?? observation?.optionSets?.[control.optionsRef];
    if (control.tag !== "select" || typeof value.value !== "string" || value.value.length > 256 ||
      (options && !options.some(option => option.value === value.value && !option.disabled))) {
      throw new ExplorationActionError("Select requires an enabled option of the observed select control");
    }
  }
  if (value.type === "wait" && (!Number.isInteger(value.milliseconds) || value.milliseconds < 1 || value.milliseconds > 2000)) {
    throw new ExplorationActionError("Wait is limited to 1–2000 milliseconds");
  }
  if (value.type === "scroll" && (!["up", "down"].includes(value.direction) || !Number.isInteger(value.pixels) || value.pixels < 1 || value.pixels > 1200)) {
    throw new ExplorationActionError("Scroll needs up/down and 1–1200 pixels");
  }
  if (value.type === "press" && !PRESS_KEYS.has(value.key)) throw new ExplorationActionError("Press is limited to a small keyboard-navigation allowlist");
  if (value.type === "finding" && (typeof value.summary !== "string" || !value.summary.trim() || value.summary.length > 1000 || typeof value.evidence !== "string" || !value.evidence.trim() || value.evidence.length > 2000)) {
    throw new ExplorationActionError("A finding needs a brief summary and visible evidence");
  }
  if (value.type === "blocked" && !value.reason?.trim()) throw new ExplorationActionError("Blocked needs a reason");
  return value;
}

/**
 * P4 Chunk 2 Fix 3: a settings page with many identical `<select>` option lists (e.g. ~41 arXiv-category
 * options repeated once per category row) was serializing one full inline copy per control, which in the
 * real P4 Chunk 1 run blew past exploration-model.ts's 100,000-character model-input budget. Both hosts'
 * drivers call this, in Node (after getting their in-page/CDP observation snapshot back as plain data, not
 * inside the page/renderer), to replace every `<select>` control's own inline `options` array with a short
 * `optionsRef` into one shared, observation-level map — one entry per DISTINCT option list (by exact
 * value/label/disabled identity), however many controls share it. `parseExplorationAction`'s select
 * validation (above) resolves either shape, so this is purely a payload-size optimization: the same
 * values are still visible to the model (once, under `observation.optionSets`, instead of once per
 * control) and the same enabled/disabled values are still enforced. Mutates `controls` in place and
 * returns the `optionSets` map to assign onto the observation.
 */
export function collapseOptionSets(controls) {
  const optionSets = {};
  const idBySignature = new Map();
  for (const control of controls) {
    if (control.tag !== "select" || !Array.isArray(control.options)) continue;
    const signature = JSON.stringify(control.options);
    let id = idBySignature.get(signature);
    if (id === undefined) {
      id = `o${idBySignature.size + 1}`;
      idBySignature.set(signature, id);
      optionSets[id] = control.options;
    }
    control.optionsRef = id;
    delete control.options;
  }
  return optionSets;
}

/** Example persona-task factory for a later task library (not wired into any default suite yet). */
export function createPersonaExplorationTask({ id, title, persona, maxSteps } = {}) {
  if (!id || !title || !persona) throw new Error("A persona task needs id, title, and a persona description");
  return {
    id, title, kind: "persona",
    goal: `扮演「${persona}」，在当前页面自由探索。遇到可疑交互用 finding 记录证据；探索充分后可用 done 结束，不代表通过。`,
    ...(maxSteps === undefined ? {} : { maxSteps }),
  };
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

/** An optional global default override (e.g. --task-max-steps); undefined means "use the kind's built-in default". */
function optionalPositiveInt(value, label, maximum) {
  if (value === undefined) return undefined;
  if (!Number.isInteger(value) || value < 1 || value > maximum) throw new Error(`${label} must be an integer between 1 and ${maximum}`);
  return value;
}

/** Cumulative model usage as of now; a model that is a plain function (tests) reports no usage. */
function usageSnapshot(model) {
  const metrics = typeof model?.metrics === "function" ? model.metrics() : undefined;
  return { apiCalls: metrics?.apiCalls ?? 0, inputTokens: metrics?.inputTokens ?? 0, outputTokens: metrics?.outputTokens ?? 0, totalTokens: metrics?.totalTokens ?? 0 };
}

function usageDelta(start, end) {
  return { apiCalls: end.apiCalls - start.apiCalls, inputTokens: end.inputTokens - start.inputTokens, outputTokens: end.outputTokens - start.outputTokens, totalTokens: end.totalTokens - start.totalTokens };
}

/**
 * Picks a subset of a task library by id, preserving the library's own order (not the selection order).
 * Omitting a selection (undefined) keeps the kind's default — every goal task, but no personas (none are
 * wired into a default run yet). An explicit empty array selects none of that kind. Unknown ids fail before
 * any browser or model is launched, listing the ids that do exist.
 */
export function selectExplorationTasks({ goalTasks = [], personaTasks = [], taskIds, personaIds } = {}) {
  const pick = (library, ids, label, defaultAll) => {
    if (ids === undefined) return defaultAll ? library : [];
    const known = new Map(library.map(item => [item.id, item]));
    const missing = ids.filter(id => !known.has(id));
    if (missing.length) {
      const available = library.map(item => item.id).join(", ") || "(none)";
      throw new Error(`Unknown ${label} id(s): ${missing.join(", ")}. Available ${label} ids: ${available}`);
    }
    const wanted = new Set(ids);
    return library.filter(item => wanted.has(item.id));
  };
  return [...pick(goalTasks, taskIds, "task", true), ...pick(personaTasks, personaIds, "persona", false)];
}

function cancellationError(signal) {
  const error = new Error(`Exploration cancelled: ${signal?.reason instanceof Error ? signal.reason.message : signal?.reason || "parent requested cancellation"}`);
  error.fatal = true;
  return error;
}

/** Run-wide budgets end the whole run; per-task trouble (model, reset, invalid actions) never does. */
function budgetExhausted(message) {
  const error = new Error(message);
  error.fatal = true;
  return error;
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

/**
 * P3 Chunk 7: the error shape an in-scope navigation (a link click, a reload, a product redirect) leaves
 * behind when it destroys the page/renderer's execution context mid-observe or mid-capture — Playwright's
 * own wording for this ("Execution context was destroyed, most likely because of a navigation") and the
 * nearby CDP phrasings a destroyed context can surface as. Out-of-scope navigation (leaving the workbench
 * origin/path, a popup) is caught earlier by each driver's own scope guard with a distinct, unmatched
 * message (e.g. exploration-browser.mjs's assertScope(): "Browser left the isolated workbench scope") and
 * is deliberately NOT matched here, so it keeps blocking immediately with no retry, exactly as before.
 */
function isTransientNavigationError(error) {
  return /execution context was destroyed|context was destroyed|cannot find context with specified id|target closed/i.test(error?.message || "");
}

/**
 * Retries `operation` (a driver's observe()/capture() call) across a transient, in-scope
 * navigation-destroyed context: waits briefly and tries again until it succeeds, a non-transient error
 * is thrown (rethrown immediately, no retry), or `signal` aborts — `signal` is the per-call deadline's
 * own AbortSignal (from withDeadline below), so the wait is bounded by that same per-call timeout instead
 * of retrying forever. `attempts` on the returned value lets the caller record a recovery event only when
 * more than one attempt was needed, so the common, no-navigation case stays silent.
 */
async function withNavigationRecovery(operation, { signal, intervalMs = 150 } = {}) {
  let attempts = 0;
  while (true) {
    attempts += 1;
    try { return { value: await operation(), attempts }; }
    catch (error) {
      if (!isTransientNavigationError(error) || signal?.aborted) throw error;
      await new Promise(resolve => setTimeout(resolve, intervalMs));
    }
  }
}

function pendingScenario(task) {
  return { id: task.id, title: task.title, status: "not-run", durationMs: 0, steps: [], assertions: [], artifacts: [] };
}

/**
 * A model chooses actions; only trusted task verifiers can establish a pass. Persona (`kind: "persona"`)
 * tasks carry no verifier and never pass — they end as "explored" once the model sends `done` or once
 * their own per-task step cap is reached, keeping only the findings they raised along the way.
 */
export async function runExplorationLoop({ driver, model, tasks, artifactDir, maxSteps, maxDurationMs, maxInvalidActions, modelCallTimeoutMs, taskMaxSteps, taskMaxApiCalls, signal }) {
  const stepsLimit = limit(maxSteps, 90, 300, "maxSteps");
  const timeLimit = limit(maxDurationMs, 900000, 3600000, "maxDurationMs");
  const invalidLimit = limit(maxInvalidActions, 3, 10, "maxInvalidActions");
  const callTimeout = limit(modelCallTimeoutMs, 45000, 300000, "modelCallTimeoutMs");
  // Global fallback defaults for every task's own budget; a task's own maxSteps/maxApiCalls still wins.
  const taskStepsDefault = optionalPositiveInt(taskMaxSteps, "taskMaxSteps", stepsLimit);
  const taskApiCallsDefault = optionalPositiveInt(taskMaxApiCalls, "taskMaxApiCalls", 1000);
  if (!Array.isArray(tasks) || tasks.some(task => !/^[a-z0-9-]+$/.test(task.id) || !task.title || !task.goal ||
    (isPersonaTask(task) ? task.verify !== undefined : typeof task.verify !== "function") ||
    (task.maxSteps !== undefined && (!Number.isInteger(task.maxSteps) || task.maxSteps < 1)) ||
    (task.maxApiCalls !== undefined && (!Number.isInteger(task.maxApiCalls) || task.maxApiCalls < 1))) ||
    new Set(tasks.map(task => task.id)).size !== tasks.length) {
    throw new Error("Exploration tasks need unique ids, titles, goals, and a trusted program verifier; persona tasks must omit verify; optional maxSteps/maxApiCalls must be positive integers");
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
    let consecutiveModelTimeouts = 0;
    let runEnded = false;
    // Every task gets its own step/API-call budget so it cannot consume the whole run: a goal task
    // defaults to 20 steps (roomy over the ~6-7 steps a default task takes today), a persona to 15
    // (chunk 2's original cap, kept); either is overridable per task or via the global --task-max-* flags.
    const taskStepsLimit = Math.max(1, Math.min(stepsLimit, Number.isInteger(task.maxSteps) ? task.maxSteps : taskStepsDefault ?? (isPersonaTask(task) ? 15 : 20)));
    const taskApiCallsLimit = Math.max(1, Number.isInteger(task.maxApiCalls) ? task.maxApiCalls : taskApiCallsDefault ?? taskStepsLimit * 2);
    const usageStart = usageSnapshot(model);
    let taskSteps = 0;
    try {
      if (signal?.aborted) throw cancellationError(signal);
      if (Date.now() >= deadline) throw budgetExhausted("Exploration duration budget exhausted; unfinished tasks are not verified");
      if (typeof driver.reset === "function") {
        try { await withDeadline(childSignal => driver.reset({ signal: childSignal }), Math.min(10000, deadline - Date.now()), "Host reset", signal); }
        catch (error) { throw error.fatal ? error : new Error(`Host reset failed: ${error.message}`); }
      }
      // Task preparation gets a longer per-call ceiling than the other host calls below (observation,
      // verification, a single browser/CDP action, host reset): those are per-step and meant to catch a
      // hang quickly, but preparation may run a real product operation once per task — e.g. the Obsidian
      // star task generating a daily report through the plugin's own command, measured on a real host at
      // ~12s even with the fixture's zero request delay. It still obeys the run-wide duration budget.
      if (typeof task.prepare === "function") await withDeadline(childSignal => task.prepare({ driver, signal: childSignal }), Math.min(60000, deadline - Date.now()), "Task preparation", signal);
      while (true) {
        if (signal?.aborted) throw cancellationError(signal);
        if (Date.now() >= deadline) throw budgetExhausted("Exploration duration budget exhausted; unfinished tasks are not verified");
        // P3 Chunk 7: an in-scope navigation (a link click, a reload) that destroys the execution
        // context mid-observe must not immediately block the whole task — withNavigationRecovery waits
        // (bounded by this same per-call deadline) and retries; an out-of-scope navigation still throws
        // its own distinct, unmatched error (e.g. assertScope()'s "left the isolated ... scope") on the
        // first attempt with no retry, so that guard keeps blocking exactly as before.
        const observationOutcome = await withDeadline(
          childSignal => withNavigationRecovery(() => driver.observe(), { signal: childSignal }),
          Math.min(10000, deadline - Date.now()), "UI observation", signal,
        );
        if (observationOutcome.attempts > 1) await record({ kind: "navigation-recovered", taskId: task.id, context: "observe", attempts: observationOutcome.attempts - 1 });
        const observation = safe(observationOutcome.value);
        observations.push(observation);
        if (observation.errors?.length) {
          scenario.status = "failed";
          scenario.error = observation.errors.join("; ");
          await record({ kind: "product-error", taskId: task.id, observation });
          break;
        }
        if (typeof task.verify === "function") {
          const verification = await withDeadline(childSignal => task.verify({ driver, observation, observations, actions, startedAt: taskStart, signal: childSignal }), Math.min(10000, deadline - Date.now()), "Task verification", signal);
          scenario.assertions = safe((verification?.checks ?? []).map(check => ({ label: String(check.label), passed: check.passed === true, ...(check.actual === undefined ? {} : { actual: check.actual }) })));
          const verified = verification?.passed === true && scenario.assertions.length > 0 && scenario.assertions.every(check => check.passed) && actions.some(action => action.type !== "wait");
          if (verified || verification?.failed === true) {
            scenario.status = verified ? "passed" : "failed";
            if (!verified) scenario.error = verification.error || "Program verification observed a product failure";
            let screenshot, resultCaptureAttempts = 1;
            if (typeof driver.capture === "function") {
              try {
                const captureOutcome = await withDeadline(
                  childSignal => withNavigationRecovery(() => driver.capture(`${task.id}-result`), { signal: childSignal }),
                  Math.min(3000, deadline - Date.now()), "Result screenshot capture", signal, // shorter than the other per-call ceilings: pure evidence, never worth a long wait
                );
                screenshot = captureOutcome.value; resultCaptureAttempts = captureOutcome.attempts;
              } catch (error) {
                // A verdict already decided by the program verifier is never overturned by a missing
                // screenshot; the failure is still recorded as step evidence (see the record() below).
                await record({ kind: "capture-unavailable", taskId: task.id, context: "verification-result", error: error.message });
              }
            }
            if (screenshot) scenario.artifacts.push(screenshot);
            await record({ kind: "verification", taskId: task.id, status: scenario.status, observation, assertions: scenario.assertions, screenshot, ...(resultCaptureAttempts > 1 ? { navigationRecovered: resultCaptureAttempts - 1 } : {}) });
            break;
          }
        }
        if (taskSteps >= taskStepsLimit) {
          if (isPersonaTask(task)) {
            scenario.status = "explored";
            await record({ kind: "persona-explored", taskId: task.id, steps: taskSteps });
          } else {
            scenario.status = "blocked";
            scenario.error = `Task budget exhausted (steps, ${taskStepsLimit}); this task is blocked, other tasks continue`;
            await record({ kind: "task-budget-exhausted", taskId: task.id, reason: "steps", limit: taskStepsLimit });
          }
          break;
        }
        if (metrics.modelCalls >= stepsLimit) throw budgetExhausted(`Exploration step budget exhausted (${stepsLimit}); unfinished tasks are not verified`);
        metrics.modelCalls += 1;
        taskSteps += 1;
        const event = { step: metrics.modelCalls, taskId: task.id, observation };
        const apiCallsUsedSoFar = usageSnapshot(model).apiCalls - usageStart.apiCalls;
        const request = {
          task: { id: task.id, title: task.title, goal: task.goal, ...(isPersonaTask(task) ? { kind: "persona" } : {}) }, observation,
          checks: scenario.assertions,
          history: safe(history.slice(-8)),
          remaining: {
            steps: stepsLimit - metrics.modelCalls + 1, milliseconds: Math.max(0, deadline - Date.now()),
            taskSteps: Math.max(0, taskStepsLimit - taskSteps + 1), taskApiCalls: Math.max(0, taskApiCallsLimit - apiCallsUsedSoFar),
          },
        };
        let action;
        try {
          const reply = await withDeadline(childSignal => callModel({ ...request, signal: childSignal }), Math.min(callTimeout, deadline - Date.now()), "Model API call", signal);
          consecutiveModelTimeouts = 0;
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
              if (isPersonaTask(task)) scenario.status = "explored";
              else {
                scenario.status = "blocked";
                scenario.error = "Model declared done without satisfying all program evidence checks (判据)";
              }
            } else if (action.type === "blocked") {
              scenario.status = "blocked";
              scenario.error = `Model could not continue: ${action.reason}`;
            }
          }
          // P3 Chunk 7: a step's own screenshot is evidence only — it never overturns an action (or a
          // model-declared finding/done/blocked) that this step already recorded into "blocked" just
          // because the capture itself failed. An in-scope navigation that destroyed the execution
          // context mid-capture (e.g. the action just taken was a click that navigated, or `press`
          // landed on a link) is retried, bounded by this call's own deadline, via the same
          // withNavigationRecovery used for observe(); a capture that is still unavailable afterwards
          // (transient-but-exhausted, or any other capture error) is recorded on the event and the step
          // simply has no screenshot — the outer catch below never sees it.
          if (typeof driver.capture === "function") {
            try {
              const captureOutcome = await withDeadline(
                childSignal => withNavigationRecovery(() => driver.capture(`${task.id}-step-${event.step}`), { signal: childSignal }),
                Math.min(3000, deadline - Date.now()), "Step screenshot capture", signal, // shorter than the other per-call ceilings: pure evidence, never worth a long wait
              );
              if (captureOutcome.attempts > 1) event.navigationRecovered = captureOutcome.attempts - 1;
              const screenshot = captureOutcome.value;
              if (screenshot) {
                event.screenshot = screenshot; scenario.artifacts.push(screenshot);
                // P3 Chunk 5: a finding raised this same step gets this step's own screenshot attached (never
                // a later or earlier one — capture() always runs once per step, right after whichever action
                // this step took), so report.mjs's "待复核" section can link straight to the evidence instead
                // of just the step number.
                const lastFinding = findings.at(-1);
                if (lastFinding && lastFinding.taskId === task.id && lastFinding.step === event.step && lastFinding.screenshot === undefined) lastFinding.screenshot = screenshot;
              }
            } catch (error) { event.captureError = error.message; }
          }
        } catch (error) {
          event.error = error.message;
          if (error.name === "ExplorationBudgetError") {
            // The model adapter's own run-wide API-call cap is exhausted; this is a run-wide budget,
            // not a per-task one, so it ends the whole run like the step/duration budgets do.
            scenario.status = "blocked";
            scenario.error = "Exploration model API-call budget exhausted; the run cannot continue";
            scenario.steps.push(safe({ step: event.step, action: null, error: event.error, observationId: observation.id }));
            await record(event);
            throw budgetExhausted(scenario.error);
          }
          if (error.name === "ExplorationTimeoutError" && consecutiveModelTimeouts === 0
            && !signal?.aborted && Date.now() < deadline) {
            // No browser action happened. Re-observe and retry the decision once;
            // the normal step/API/duration guards below and above still apply.
            consecutiveModelTimeouts += 1;
            event.retry = "model-timeout";
          } else {
            scenario.status = "blocked";
            scenario.error = `Model or evidence collection unavailable: ${error.message}`;
          }
        }
        if (scenario.status === "not-run") {
          const apiCallsUsed = usageSnapshot(model).apiCalls - usageStart.apiCalls;
          if (apiCallsUsed >= taskApiCallsLimit) {
            if (isPersonaTask(task)) scenario.status = "explored";
            else {
              scenario.status = "blocked";
              scenario.error = `Task budget exhausted (model API calls, ${taskApiCallsLimit}); this task is blocked, other tasks continue`;
            }
          }
        }
        scenario.steps.push(safe({ step: event.step, action: event.action ?? null, ...(event.error ? { error: event.error } : {}),
          ...(event.navigationRecovered ? { navigationRecovered: event.navigationRecovered } : {}), ...(event.captureError ? { captureError: event.captureError } : {}),
          observationId: observation.id }));
        await record(event);
        history.push(safe({ step: event.step, taskId: task.id, action: event.action, result: event.error ? { error: event.error } : event.result }));
        if (scenario.status !== "not-run") break;
        if (invalidActions >= invalidLimit) throw new Error(`Model produced ${invalidActions} invalid or unexecutable actions; exploration is blocked`);
      }
    } catch (error) {
      scenario.status = "blocked";
      scenario.error = redact(error.message);
      runEnded = Boolean(error.fatal);
    } finally {
      scenario.durationMs = Date.now() - taskStart;
      scenario.usage = { modelCalls: taskSteps, durationMs: scenario.durationMs, ...usageDelta(usageStart, usageSnapshot(model)) };
    }
    if (scenario.error) { scenario.error = redact(scenario.error); errors.push(`${task.id}: ${scenario.error}`); }
    if (runEnded) break;
  }
  metrics.elapsedMs = Date.now() - started;
  for (const scenario of scenarios) scenario.findings = findings.filter(finding => finding.taskId === scenario.id);
  // Usage is attributed per task as deltas of the model's own cumulative counters, so the run totals
  // below always equal the sum of each scenario's own usage, including tasks that never ran (0 usage).
  metrics.usage = scenarios.reduce((total, scenario) => {
    const usage = scenario.usage ?? { modelCalls: 0, apiCalls: 0, inputTokens: 0, outputTokens: 0, totalTokens: 0, durationMs: 0 };
    for (const key of Object.keys(total)) total[key] += usage[key] ?? 0;
    return total;
  }, { modelCalls: 0, apiCalls: 0, inputTokens: 0, outputTokens: 0, totalTokens: 0, durationMs: 0 });
  // A persona's "explored" is pass-equivalent for the run's own status: it never represents a verified
  // goal, but it also never blocks or fails the run on its own. A run of only explored personas still
  // reports "passed" (exit 0); callers distinguish that case by the absence of any "passed" scenario.
  const result = safe({ suite: "exploration", status: scenarios.some(scenario => scenario.status === "failed") ? "failed" : scenarios.some(scenario => scenario.status === "blocked") ? "blocked" : scenarios.length && scenarios.every(scenario => scenario.status === "passed" || scenario.status === "explored") ? "passed" : "not-run",
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

const EXPLORATION_HOSTS = new Set(["workbench", "obsidian"]);

/**
 * `dependencies` is a test seam only (mirrors `runAcceptance(options, dependencies)` in run.mjs): ordinary
 * callers (run.mjs, real exploration) never pass it, and every host's real module is still the default.
 * It lets host → driver/session/task-library selection be red-green tested with fakes, without launching a
 * real browser or Obsidian for every wiring assertion.
 */
export async function runExploration({ fixture, artifactDir, configPath, maxSteps, maxDurationMs, tasks, goalTasks, personaTasks, taskIds, personaIds, release, taskMaxSteps, taskMaxApiCalls, model, executablePath, headless, modelCallTimeoutMs, maxApiCalls, host = "workbench", signal } = {}, dependencies = {}) {
  let session;
  let driver;
  let configured = model;
  let result;
  const directory = artifactDir ?? path.resolve("output/playwright/acceptance/exploration");
  try {
    if (signal?.aborted) throw cancellationError(signal);
    // Unknown host fails before any import, model, browser, or Obsidian launch — same fail-fast shape as
    // an unknown --task/--persona id below.
    if (!EXPLORATION_HOSTS.has(host)) throw new Error(`Unknown exploration host: ${host}. Supported hosts: ${[...EXPLORATION_HOSTS].join(", ")}`);
    const obsidian = host === "obsidian";
    let createDriver, startSession, resolvedGoalTasks, resolvedPersonaTasks;
    // Task library (what to do, how a program tells it passed) lives per-host in exploration-tasks/
    // {obsidian,workbench}.mjs, separate from the driver (how the host is observed and driven) — see
    // those files' own header comments. `dependencies.taskLibrary` is the test seam (mirrors every other
    // `dependencies.*` override here), letting a fake library be injected without touching either real file.
    if (obsidian) {
      const mod = await import("./exploration-obsidian.mjs");
      createDriver = dependencies.createObsidianExplorationDriver ?? mod.createObsidianExplorationDriver;
      startSession = dependencies.startObsidianExplorationSession ?? mod.startObsidianExplorationSession;
      const taskLibrary = dependencies.taskLibrary ?? await import("./exploration-tasks/obsidian.mjs");
      resolvedGoalTasks = goalTasks ?? await taskLibrary.goalTasks();
      resolvedPersonaTasks = personaTasks ?? await taskLibrary.personaTasks();
    } else {
      const mod = await import("./exploration-browser.mjs");
      createDriver = dependencies.createExplorationDriver ?? mod.createExplorationDriver;
      startSession = dependencies.startWorkbenchSession ?? (await import("./workbench-session.mjs")).startWorkbenchSession;
      const taskLibrary = dependencies.taskLibrary ?? await import("./exploration-tasks/workbench.mjs");
      resolvedGoalTasks = goalTasks ?? await taskLibrary.goalTasks(fixture);
      resolvedPersonaTasks = personaTasks ?? await taskLibrary.personaTasks();
    }
    // --release means "run the full library for a release": every goal task (already the default) plus
    // every persona (normally opt-in only, via --persona). An explicit --persona still wins over --release
    // for personas (e.g. `--release --persona keyboard-only` runs only that one persona, not all three);
    // --release only fills in the "no --persona given" default.
    const effectivePersonaIds = personaIds ?? (release ? resolvedPersonaTasks.map(task => task.id) : undefined);
    // Resolve and validate the task/persona selection before touching a model, a browser, or Obsidian, so an
    // unknown --task/--persona id fails fast instead of launching anything first.
    const selected = tasks ?? selectExplorationTasks({ goalTasks: resolvedGoalTasks, personaTasks: resolvedPersonaTasks, taskIds, personaIds: effectivePersonaIds });
    configured ??= await loadExplorationModel({ configPath, maxApiCalls: maxApiCalls ?? (maxSteps ?? 90) * 2 });
    if (obsidian) {
      session = await startSession({ fixture, artifactDir: directory, signal });
      driver = await createDriver({ session: session.session, fixture, artifactDir: directory, redact: configured.redact, signal });
    } else {
      // P4 Chunk 1: before the workbench server starts, make sure its fixture already has a
      // prepared pdfjs runtime (so a model's `library-build` never spawns `npm ci` against the real
      // registry) and force npm offline in its environment as a fail-fast backstop. The obsidian
      // host and the fixed workbench.mjs suite never call `library-build`, so only this branch opts
      // in. `dependencies.libraryRuntimeCache` is the test seam: the real module reaches the network
      // only on a persistent-cache miss (see library-runtime-cache.mjs for the full contract).
      const runtimeCache = dependencies.libraryRuntimeCache ?? (await import("./library-runtime-cache.mjs"));
      await runtimeCache.seedWorkbenchLibraryRuntime(fixture, { signal });
      const workbenchFixture = { ...fixture, env: runtimeCache.npmOfflineEnv(fixture.env) };
      session = await startSession({ fixture: workbenchFixture, artifactDir: directory, executablePath, headless, signal });
      driver = await createDriver({ ...session, fixture: workbenchFixture, artifactDir: directory, redact: configured.redact, signal });
    }
    result = await runExplorationLoop({ driver, model: configured, tasks: selected, artifactDir: directory, maxSteps, maxDurationMs, modelCallTimeoutMs, taskMaxSteps, taskMaxApiCalls, signal });
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
