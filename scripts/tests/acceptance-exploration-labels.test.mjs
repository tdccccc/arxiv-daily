/**
 * P2/P6 (docs/helm/2026-10-13-exploration-safety-and-coverage/phases/02-ui-label-check.md,
 * 06-label-check-by-source.md): an ordinary, free fixed test — no model, no app launch — that fails as
 * soon as a product UI label a goal task depends on (named in its goal text, or matched by its own
 * verify()) disappears from the exact product source file it names. Today this is only discovered by a
 * blocked, paid real-model run.
 *
 * P2 originally searched a whole host's product tree, case-insensitively, for each label as a plain
 * substring. That missed a rename: if the same text survives *anywhere else* in the tree (a stale
 * sibling control, a comment, a longer string that merely contains it), the check still passes. P6
 * tightens this: every `uiLabels` entry is now `uiLabel(text, source)` (scripts/acceptance/exploration-
 * tasks/checks.mjs), naming the repo-relative product source file(s) (or `uiLabelException(text,
 * reason)` for the handful that cannot be checked this way — see the matching rule below for why), and
 * the text must appear in every one of its own named files as a *complete*, case-sensitive string
 * literal — not merely as a substring of a longer one, and not only in a comment.
 *
 * Matching rule (kept deliberately simple, not a real parser): `text` counts as present in a source file
 * if it appears as the entire contents between a matching pair of quotes — `"text"`, `'text'`, or
 * `` `text` `` — or as a complete *static* segment of a larger template literal, i.e. immediately preceded
 * by a backtick or a `}` (the end of a `${...}` interpolation) and immediately followed by a backtick or
 * `${` (the start of one). Anything less — the text only inside a // or /* comment, or only as part of a
 * longer quoted string — does not count.
 *
 * Loads both task libraries the same way acceptance-exploration-host.test.mjs does: the Obsidian
 * library's `goalTasks()` needs no fixture at all; the workbench library's `goalTasks(fixture)` needs
 * the same minimal on-disk fixture (a config.toml plus an empty vault directory) that test file already
 * builds for its own "ids are unchanged" test.
 */
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import * as obsidianTaskLibrary from "../acceptance/exploration-tasks/obsidian.mjs";
import * as workbenchTaskLibrary from "../acceptance/exploration-tasks/workbench.mjs";

const REPO_ROOT = fileURLToPath(new URL("../../", import.meta.url));
// Each host's own UI, plus the shared core that supplies some of its visible text (row labels, field names).
const OBSIDIAN_SOURCE_PREFIX = ["plugin/src/", "packages/core/src/"];
const WORKBENCH_SOURCE_PREFIX = ["apps/cli/src/workbench/", "packages/core/src/"];

/**
 * Does `text` appear in `sourceText` as a complete string literal, per this file's own matching rule
 * (see the top-of-file comment)? Plain substring checks cover the `"..."`/`'...'`/`` `...` `` cases;
 * the loop below additionally covers a complete static segment of a larger template literal (e.g.
 * `` `${LABEL}…` `` in a task file is not itself what we search product source for, but a product file
 * building its own menu item as `` `${base}…` `` would need this).
 */
export function hasCompleteLiteral(sourceText, text) {
  if (!text) return false;
  if (sourceText.includes(`"${text}"`)) return true;
  if (sourceText.includes(`'${text}'`)) return true;
  if (sourceText.includes(`\`${text}\``)) return true;
  let from = 0;
  for (;;) {
    const index = sourceText.indexOf(text, from);
    if (index === -1) return false;
    const before = sourceText[index - 1];
    const after = sourceText.slice(index + text.length, index + text.length + 2);
    if ((before === "`" || before === "}") && (after[0] === "`" || after.startsWith("${"))) return true;
    from = index + 1;
  }
}

/** Normalizes a `uiLabel()` entry's `source` (a single path or an array of paths) to an array. */
function sourcePaths(entry) {
  return Array.isArray(entry.source) ? entry.source : [entry.source];
}

/**
 * Reads every distinct source path any task declares, once, into a path -> content map (`null` when the
 * path does not exist — a declared source that is itself a typo, which must fail just like a missing
 * label). Scoped to real paths under `root`'s repo-relative prefix so a mistakenly cross-host source
 * (e.g. an Obsidian label pointing at a workbench file) still resolves to real disk content rather than
 * silently reading nothing — the host-prefix check itself lives in `findMissingLabels` below.
 */
async function loadSources(tasks) {
  const paths = new Set();
  for (const task of tasks) {
    for (const entry of task.uiLabels ?? []) {
      if (entry.exception !== undefined) continue;
      for (const sourcePath of sourcePaths(entry)) paths.add(sourcePath);
    }
  }
  const sources = new Map();
  for (const sourcePath of paths) {
    try { sources.set(sourcePath, await readFile(path.join(REPO_ROOT, sourcePath), "utf8")); }
    catch { sources.set(sourcePath, null); }
  }
  return sources;
}

/**
 * The check this whole file exists to run, factored out so a test can inject a fake `sources` map and
 * assert the failure message shape without needing a real stale rename or touching disk. `sources` is a
 * Map from repo-relative path to file content, or `null` for a path that does not exist on disk (a typo
 * in a declared `source`). `sourcePrefix` lists the host's own product-root prefixes (e.g. "plugin/src/"); a
 * declared source outside them is reported as a mistake rather than silently skipped.
 *
 * Each failing (task, label, source file) triple is reported separately — a label naming two source
 * files only passes once every one of them contains the complete literal; each one that doesn't produces
 * its own entry, so the failure message can point at exactly which file still needs the rename finished.
 */
export function findMissingLabels(host, tasks, sources, sourcePrefix) {
  const missing = [];
  for (const task of tasks) {
    for (const entry of task.uiLabels ?? []) {
      if (entry.exception !== undefined) continue;
      for (const sourcePath of sourcePaths(entry)) {
        const prefixes = [sourcePrefix ?? []].flat();
        if (prefixes.length && !prefixes.some(prefix => sourcePath.startsWith(prefix))) {
          missing.push({ host, taskId: task.id, label: entry.text, source: sourcePath, reason: `source is outside ${prefixes.join(", ")}` });
          continue;
        }
        const content = sources.get(sourcePath);
        if (content === undefined || content === null) {
          missing.push({ host, taskId: task.id, label: entry.text, source: sourcePath, reason: "source file does not exist" });
        } else if (!hasCompleteLiteral(content, entry.text)) {
          missing.push({ host, taskId: task.id, label: entry.text, source: sourcePath, reason: "not present as a complete string literal" });
        }
      }
    }
  }
  return missing;
}

/** One line per failure, always naming the host, the task id, the label and the source file — the four
 * things the phase outcome requires a failure to point at. */
export function formatMissingLabels(missing) {
  return missing
    .map(({ host, taskId, label, source, reason }) => `${host} task "${taskId}" depends on the UI label "${label}" in ${source}, which fails: ${reason}`)
    .join("\n");
}

/** Every `uiLabelException` entry across every goal task of both hosts, for the report this phase's
 * verification asks a human to review (each with its own reason). */
function collectExceptions(host, tasks) {
  const exceptions = [];
  for (const task of tasks) {
    for (const entry of task.uiLabels ?? []) {
      if (entry.exception !== undefined) exceptions.push({ host, taskId: task.id, label: entry.text, reason: entry.exception });
    }
  }
  return exceptions;
}

async function workbenchFixture() {
  const root = await mkdtemp(path.join(tmpdir(), "exploration-labels-workbench-"));
  const vaultRoot = path.join(root, "vault");
  const configPath = path.join(root, "config.toml");
  await mkdir(vaultRoot, { recursive: true });
  await writeFile(configPath, `vault_root = ${JSON.stringify(vaultRoot)}\n[output]\nmax_daily_papers = 2\n`);
  return { root, vaultRoot, configPath };
}

// ---------------------------------------------------------------------------
// Every goal task declares a non-empty uiLabels array of well-shaped entries
// (either a checkable { text, source } or an { text, exception } to skip).
// ---------------------------------------------------------------------------

function assertWellShapedUiLabels(task) {
  assert.ok(Array.isArray(task.uiLabels) && task.uiLabels.length > 0, `${task.id}: uiLabels must be a non-empty array`);
  for (const entry of task.uiLabels) {
    assert.ok(entry && typeof entry === "object", `${task.id}: every uiLabels entry must be a uiLabel()/uiLabelException() object, not a plain string`);
    assert.ok(typeof entry.text === "string" && entry.text.length > 0, `${task.id}: every uiLabels entry needs a non-empty text`);
    if (entry.exception !== undefined) {
      assert.ok(typeof entry.exception === "string" && entry.exception.length > 0, `${task.id}: "${entry.text}"'s exception needs a non-empty reason`);
    } else {
      const paths = sourcePaths(entry);
      assert.ok(paths.length > 0 && paths.every(value => typeof value === "string" && value.length > 0), `${task.id}: "${entry.text}" needs a non-empty source path or array of paths`);
    }
  }
}

test("every Obsidian goal task declares a non-empty uiLabels array of well-shaped entries", async () => {
  const tasks = await obsidianTaskLibrary.goalTasks();
  assert.equal(tasks.length, obsidianTaskLibrary.GOAL_TASK_IDS.length);
  for (const task of tasks) assertWellShapedUiLabels(task);
});

test("every workbench goal task declares a non-empty uiLabels array of well-shaped entries", async () => {
  const fixture = await workbenchFixture();
  try {
    const tasks = await workbenchTaskLibrary.goalTasks(fixture);
    assert.equal(tasks.length, workbenchTaskLibrary.GOAL_TASK_IDS.length);
    for (const task of tasks) assertWellShapedUiLabels(task);
  } finally { await rm(fixture.root, { recursive: true, force: true }); }
});

// ---------------------------------------------------------------------------
// Every declared label is still present, as a complete, case-sensitive string
// literal, in every source file it names (all 21 Obsidian + 6 workbench goal
// tasks) — against the real product tree, not an injected fixture.
// ---------------------------------------------------------------------------

test("every Obsidian goal task's declared UI labels are complete literals in the source files they name", async () => {
  const tasks = await obsidianTaskLibrary.goalTasks();
  const sources = await loadSources(tasks);
  const missing = findMissingLabels("obsidian", tasks, sources, OBSIDIAN_SOURCE_PREFIX);
  assert.deepEqual(missing, [], formatMissingLabels(missing));
});

test("every workbench goal task's declared UI labels are complete literals in the source files they name", async () => {
  const fixture = await workbenchFixture();
  try {
    const tasks = await workbenchTaskLibrary.goalTasks(fixture);
    const sources = await loadSources(tasks);
    const missing = findMissingLabels("workbench", tasks, sources, WORKBENCH_SOURCE_PREFIX);
    assert.deepEqual(missing, [], formatMissingLabels(missing));
  } finally { await rm(fixture.root, { recursive: true, force: true }); }
});

test("every uiLabelException across both hosts carries a human-reviewable reason, and there are only a handful", async () => {
  const fixture = await workbenchFixture();
  try {
    const obsidianTasks = await obsidianTaskLibrary.goalTasks();
    const workbenchTasks = await workbenchTaskLibrary.goalTasks(fixture);
    const exceptions = [...collectExceptions("obsidian", obsidianTasks), ...collectExceptions("workbench", workbenchTasks)];
    for (const exception of exceptions) assert.ok(exception.reason.length > 10, `${exception.host} "${exception.label}": exception reason is too short to be reviewable`);
    // Abort trigger (phase doc): more than a handful of exceptions means the matching rule itself needs
    // reconsidering, not more exceptions piling up.
    assert.ok(exceptions.length <= 4, `expected at most 4 uiLabelException entries, found ${exceptions.length}: ${exceptions.map(e => e.label).join(", ")}`);
  } finally { await rm(fixture.root, { recursive: true, force: true }); }
});

// ---------------------------------------------------------------------------
// findMissingLabels/formatMissingLabels/hasCompleteLiteral themselves: each
// of the phase's own Red scenarios, run on injected source text rather than
// a real product rename.
// ---------------------------------------------------------------------------

test("Red: a label renamed in only one of two source files that used to share it is still reported missing (Run History)", () => {
  // Mirrors the real find that motivated P6 (journal.md, 2026-10-13 "tighten the label check after
  // P3"): plugin/src/dashboard/hub-modal.ts renamed "Run History" to something else while
  // plugin/src/commands.ts still has the differently-cased "Run history" — the old whole-tree,
  // case-insensitive P2 check wrongly passed this. Sourcing the label at hub-modal.ts alone now fails.
  const fakeTasks = [{ id: "auth-failure-retry", uiLabels: [{ text: "Run History", source: "plugin/src/dashboard/hub-modal.ts" }] }];
  const sources = new Map([
    ["plugin/src/dashboard/hub-modal.ts", 'this.createPanel(tabs, body, "history", "Logs");'],
    ["plugin/src/commands.ts", 'contentEl.createEl("h2", { text: "Run history" });'],
  ]);
  const missing = findMissingLabels("obsidian", fakeTasks, sources, OBSIDIAN_SOURCE_PREFIX);
  assert.deepEqual(missing, [{ host: "obsidian", taskId: "auth-failure-retry", label: "Run History", source: "plugin/src/dashboard/hub-modal.ts", reason: "not present as a complete string literal" }]);
  const message = formatMissingLabels(missing);
  assert.match(message, /obsidian/);
  assert.match(message, /auth-failure-retry/);
  assert.match(message, /Run History/);
  assert.match(message, /hub-modal\.ts/);
});

test("Red: a label surviving only in a comment is still reported missing", () => {
  const fakeTasks = [{ id: "daily-paper-limit", uiLabels: [{ text: "Daily paper limit", source: "plugin/src/settings/tab.ts" }] }];
  const sources = new Map([["plugin/src/settings/tab.ts", "// TODO: restore the Daily paper limit control removed in the last refactor"]]);
  const missing = findMissingLabels("obsidian", fakeTasks, sources, OBSIDIAN_SOURCE_PREFIX);
  assert.deepEqual(missing, [{ host: "obsidian", taskId: "daily-paper-limit", label: "Daily paper limit", source: "plugin/src/settings/tab.ts", reason: "not present as a complete string literal" }]);
});

test("Red: a label surviving only inside a longer string is still reported missing", () => {
  const fakeTasks = [{ id: "run-for-date", uiLabels: [{ text: "Run for date", source: "plugin/src/dashboard/view.ts" }] }];
  const sources = new Map([["plugin/src/dashboard/view.ts", 'this.addDeferredMenuItem(menu, "Run for date and summarize", "calendar", () => {});']]);
  const missing = findMissingLabels("obsidian", fakeTasks, sources, OBSIDIAN_SOURCE_PREFIX);
  assert.deepEqual(missing, [{ host: "obsidian", taskId: "run-for-date", label: "Run for date", source: "plugin/src/dashboard/view.ts", reason: "not present as a complete string literal" }]);
});

test("Green: a complete double-quoted literal passes", () => {
  const fakeTasks = [{ id: "ok-task", uiLabels: [{ text: "Run History", source: "plugin/src/a.ts" }, { text: "Cancel active tasks", source: "plugin/src/b.ts" }] }];
  const sources = new Map([
    ["plugin/src/a.ts", 'this.createPanel(tabs, body, "history", "Run History");'],
    ["plugin/src/b.ts", 'menu.addItem("Cancel active tasks");'],
  ]);
  assert.deepEqual(findMissingLabels("obsidian", fakeTasks, sources, OBSIDIAN_SOURCE_PREFIX), []);
});

test("Green: a complete single-quoted or template-literal-without-interpolation literal passes", () => {
  const fakeTasks = [{ id: "ok-task", uiLabels: [{ text: "放弃保存并关闭", source: "plugin/src/a.ts" }, { text: "Retry", source: "plugin/src/b.ts" }] }];
  const sources = new Map([
    ["plugin/src/a.ts", "button('confirm-discard-close','放弃保存并关闭')"],
    ["plugin/src/b.ts", "h(`Retry`)"],
  ]);
  assert.deepEqual(findMissingLabels("obsidian", fakeTasks, sources, OBSIDIAN_SOURCE_PREFIX), []);
});

test("Green: a complete static segment of a larger template literal (between backtick/${…} boundaries) passes", () => {
  const fakeTasks = [{ id: "ok-task", uiLabels: [{ text: "Force run for date…", source: "plugin/src/a.ts" }] }];
  const sources = new Map([["plugin/src/a.ts", 'this.addDeferredMenuItem(menu, `${base}…`, "rotate-cw", go); // where base ends in "Force run for date"\nconst literalForm = `Force run for date…`;']]);
  assert.deepEqual(findMissingLabels("obsidian", fakeTasks, sources, OBSIDIAN_SOURCE_PREFIX), []);
});

test("case-sensitive: a label that only changed case is reported missing (unlike the pre-P6 check)", () => {
  const fakeTasks = [{ id: "case-task", uiLabels: [{ text: "Reading Dashboard", source: "plugin/src/a.ts" }] }];
  const sources = new Map([["plugin/src/a.ts", 'getDisplayText() { return "Reading dashboard"; }']]);
  const missing = findMissingLabels("obsidian", fakeTasks, sources, OBSIDIAN_SOURCE_PREFIX);
  assert.deepEqual(missing, [{ host: "obsidian", taskId: "case-task", label: "Reading Dashboard", source: "plugin/src/a.ts", reason: "not present as a complete string literal" }]);
});

test("a uiLabelException entry is never checked against source, even when its source would otherwise be missing", () => {
  const fakeTasks = [{ id: "ok-task", uiLabels: [{ text: "Retry preparation", exception: "runtime-built from packages/core; no literal in plugin/src/**" }] }];
  assert.deepEqual(findMissingLabels("obsidian", fakeTasks, new Map(), OBSIDIAN_SOURCE_PREFIX), []);
});

test("a typo'd source path (one that does not exist on disk) is reported missing, not silently skipped", () => {
  const fakeTasks = [{ id: "ok-task", uiLabels: [{ text: "Run History", source: "plugin/src/dashboard/hud-modal.ts" }] }];
  const sources = new Map([["plugin/src/dashboard/hud-modal.ts", null]]);
  const missing = findMissingLabels("obsidian", fakeTasks, sources, OBSIDIAN_SOURCE_PREFIX);
  assert.deepEqual(missing, [{ host: "obsidian", taskId: "ok-task", label: "Run History", source: "plugin/src/dashboard/hud-modal.ts", reason: "source file does not exist" }]);
});

test("a source path outside the host's own product root is reported missing, not silently accepted", () => {
  const fakeTasks = [{ id: "ok-task", uiLabels: [{ text: "Retry preparation", source: "apps/cli/src/workbench/web/settings.ts" }] }];
  const sources = new Map([["apps/cli/src/workbench/web/settings.ts", 'primary: { label: "Retry preparation", disabled: false }']]);
  const missing = findMissingLabels("obsidian", fakeTasks, sources, OBSIDIAN_SOURCE_PREFIX);
  assert.deepEqual(missing, [{ host: "obsidian", taskId: "ok-task", label: "Retry preparation", source: "apps/cli/src/workbench/web/settings.ts", reason: `source is outside ${OBSIDIAN_SOURCE_PREFIX.join(", ")}` }]);
});
