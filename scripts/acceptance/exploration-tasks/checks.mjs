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

/**
 * P6 (docs/helm/2026-10-13-exploration-safety-and-coverage/phases/06-label-check-by-source.md): wraps a
 * declared `uiLabels` entry with the repo-relative product source file(s) (relative to the repo root,
 * e.g. "plugin/src/dashboard/view.ts") the label's exact text must appear in as a complete,
 * case-sensitive string literal — see scripts/tests/acceptance-exploration-labels.test.mjs's own
 * matching rule. `source` is a single path or an array of paths when the task's own flow depends on the
 * same text surviving in more than one control/file (e.g. a confirmation dialog echoing a row's button).
 */
export const uiLabel = (text, source) => ({ text, source });

/**
 * A `uiLabels` entry that cannot be checked against product source this way (built at runtime from a
 * table outside this host's own source tree, or present only as a substring of a longer literal with no
 * standalone literal to anchor on) — the label-check test reports these with their reason instead of
 * checking them.
 */
export const uiLabelException = (text, reason) => ({ text, exception: reason });
