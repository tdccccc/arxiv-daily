/**
 * Shared by every host's goal-task `verify()` (exploration-tasks/{obsidian,workbench}.mjs). Both host
 * files used to carry their own identical copy of this pair (`check`/`result` in exploration-browser.mjs,
 * `checkResult`/`taskResult` in exploration-obsidian.mjs); canonicalized here when the task definitions
 * moved out into their own per-host modules (P3 Chunk 1 of the task-library phase).
 */
export const checkResult = (label, passed, actual) => ({ label, passed: Boolean(passed), ...(actual === undefined ? {} : { actual }) });
export const taskResult = checks => ({ passed: checks.every(value => value.passed), checks });

/**
 * P2 (docs/helm/2026-10-13-exploration-safety-and-coverage): builds the case-insensitive, whole-string
 * RegExp most goal-task verifiers already use to match a control's accessible name (e.g. the previous
 * ad-hoc `/^cancel active tasks$/i`). Takes the same declared `uiLabels` string a task exports, so the
 * verifier and the declared label can never drift apart — the label is typed once, not copy-pasted into
 * a second regex literal.
 */
export const exactLabelPattern = label => new RegExp(`^${label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`, "i");
