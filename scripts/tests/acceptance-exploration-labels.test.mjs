/**
 * P2 (docs/helm/2026-10-13-exploration-safety-and-coverage/phases/02-ui-label-check.md): an ordinary,
 * free fixed test — no model, no app launch — that fails as soon as a product UI label a goal task
 * depends on (named in its goal text, or matched by its own verify()) disappears from that host's
 * product source. Today this is only discovered by a blocked, paid real-model run.
 *
 * Loads both task libraries the same way acceptance-exploration-host.test.mjs does: the Obsidian
 * library's `goalTasks()` needs no fixture at all; the workbench library's `goalTasks(fixture)` needs
 * the same minimal on-disk fixture (a config.toml plus an empty vault directory) that test file already
 * builds for its own "ids are unchanged" test.
 */
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import * as obsidianTaskLibrary from "../acceptance/exploration-tasks/obsidian.mjs";
import * as workbenchTaskLibrary from "../acceptance/exploration-tasks/workbench.mjs";

const OBSIDIAN_PRODUCT_ROOT = fileURLToPath(new URL("../../plugin/src", import.meta.url));
const WORKBENCH_PRODUCT_ROOT = fileURLToPath(new URL("../../apps/cli/src/workbench", import.meta.url));

/** Recursively concatenates every file under `root` (skipping node_modules/dist build output) into one
 * big string, so a label's existence can be checked with a single substring search — the same "plain
 * substring search over source files" the phase's own assumptions call precise enough for this purpose. */
async function collectSourceText(root) {
  const chunks = [];
  async function walk(dir) {
    let entries;
    try { entries = await readdir(dir, { withFileTypes: true }); }
    catch { return; }
    for (const entry of entries) {
      if (entry.name === "node_modules" || entry.name === "dist") continue;
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) await walk(full);
      else chunks.push(await readFile(full, "utf8").catch(() => ""));
    }
  }
  await walk(root);
  return chunks.join("\n");
}

/**
 * The check this whole file exists to run, factored out so a test can inject a fake missing label and
 * assert the failure message shape without needing a real stale rename. Case-insensitive, matching how
 * most of the goal tasks' own verify() functions already match a control's accessible name (e.g. the
 * pre-existing `/^cancel active tasks$/i`) — a label that only changed case is not the rename this check
 * is meant to catch.
 */
export function findMissingLabels(host, tasks, sourceText) {
  const haystack = sourceText.toLowerCase();
  const missing = [];
  for (const task of tasks) {
    for (const label of task.uiLabels ?? []) {
      if (!haystack.includes(label.toLowerCase())) missing.push({ host, taskId: task.id, label });
    }
  }
  return missing;
}

/** One line per missing label, always naming the host, the task id and the label — the three things the
 * phase outcome requires a failure to point at. */
export function formatMissingLabels(missing) {
  return missing
    .map(({ host, taskId, label }) => `${host} task "${taskId}" depends on the UI label "${label}", which no longer appears in ${host}'s product source`)
    .join("\n");
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
// Every goal task declares a non-empty uiLabels array of non-empty strings.
// ---------------------------------------------------------------------------

test("every Obsidian goal task declares a non-empty uiLabels array of non-empty strings", async () => {
  const tasks = await obsidianTaskLibrary.goalTasks();
  assert.equal(tasks.length, obsidianTaskLibrary.GOAL_TASK_IDS.length);
  for (const task of tasks) {
    assert.ok(Array.isArray(task.uiLabels) && task.uiLabels.length > 0, `${task.id}: uiLabels must be a non-empty array`);
    for (const label of task.uiLabels) assert.ok(typeof label === "string" && label.length > 0, `${task.id}: every uiLabels entry must be a non-empty string`);
  }
});

test("every workbench goal task declares a non-empty uiLabels array of non-empty strings", async () => {
  const fixture = await workbenchFixture();
  try {
    const tasks = await workbenchTaskLibrary.goalTasks(fixture);
    assert.equal(tasks.length, workbenchTaskLibrary.GOAL_TASK_IDS.length);
    for (const task of tasks) {
      assert.ok(Array.isArray(task.uiLabels) && task.uiLabels.length > 0, `${task.id}: uiLabels must be a non-empty array`);
      for (const label of task.uiLabels) assert.ok(typeof label === "string" && label.length > 0, `${task.id}: every uiLabels entry must be a non-empty string`);
    }
  } finally { await rm(fixture.root, { recursive: true, force: true }); }
});

// ---------------------------------------------------------------------------
// Every declared label is still literally present (case-insensitively) in
// that host's own product source.
// ---------------------------------------------------------------------------

test("every Obsidian goal task's declared UI labels are still literally present in plugin/src/**", async () => {
  const tasks = await obsidianTaskLibrary.goalTasks();
  const sourceText = await collectSourceText(OBSIDIAN_PRODUCT_ROOT);
  const missing = findMissingLabels("obsidian", tasks, sourceText);
  assert.deepEqual(missing, [], formatMissingLabels(missing));
});

test("every workbench goal task's declared UI labels are still literally present in apps/cli/src/workbench/**", async () => {
  const fixture = await workbenchFixture();
  try {
    const tasks = await workbenchTaskLibrary.goalTasks(fixture);
    const sourceText = await collectSourceText(WORKBENCH_PRODUCT_ROOT);
    const missing = findMissingLabels("workbench", tasks, sourceText);
    assert.deepEqual(missing, [], formatMissingLabels(missing));
  } finally { await rm(fixture.root, { recursive: true, force: true }); }
});

// ---------------------------------------------------------------------------
// findMissingLabels/formatMissingLabels themselves: an injected missing
// label is reported, naming the host, the task id and the label — the
// failure-path demonstration the phase's Red/Green plan asks for, run
// without needing a real product rename.
// ---------------------------------------------------------------------------

test("findMissingLabels reports an injected missing label naming the host, task id and label; formatMissingLabels' message includes all three", () => {
  const fakeTasks = [
    { id: "real-task", uiLabels: ["Daily paper limit"] },
    { id: "renamed-task", uiLabels: ["This Label Was Renamed Away"] },
  ];
  const sourceText = 'createEl("h2", { text: "Daily paper limit" })'; // only the first label exists
  const missing = findMissingLabels("obsidian", fakeTasks, sourceText);
  assert.deepEqual(missing, [{ host: "obsidian", taskId: "renamed-task", label: "This Label Was Renamed Away" }]);
  const message = formatMissingLabels(missing);
  assert.match(message, /obsidian/);
  assert.match(message, /renamed-task/);
  assert.match(message, /This Label Was Renamed Away/);
});

test("findMissingLabels is case-insensitive, matching how goal-task verifiers already match control names", () => {
  const fakeTasks = [{ id: "case-task", uiLabels: ["Reading Dashboard"] }];
  const missing = findMissingLabels("obsidian", fakeTasks, 'getDisplayText() { return "Reading dashboard"; }');
  assert.deepEqual(missing, []);
});

test("findMissingLabels passes (no missing labels) when every declared label is present", () => {
  const fakeTasks = [{ id: "ok-task", uiLabels: ["Run History", "Cancel active tasks"] }];
  const sourceText = 'createPanel(tabs, body, "history", "Run History"); ... "Cancel active tasks"';
  assert.deepEqual(findMissingLabels("obsidian", fakeTasks, sourceText), []);
});
