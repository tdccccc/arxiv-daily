/**
 * Shared by every host's goal-task `verify()` (exploration-tasks/{obsidian,workbench}.mjs). Both host
 * files used to carry their own identical copy of this pair (`check`/`result` in exploration-browser.mjs,
 * `checkResult`/`taskResult` in exploration-obsidian.mjs); canonicalized here when the task definitions
 * moved out into their own per-host modules (P3 Chunk 1 of the task-library phase).
 */
export const checkResult = (label, passed, actual) => ({ label, passed: Boolean(passed), ...(actual === undefined ? {} : { actual }) });
export const taskResult = checks => ({ passed: checks.every(value => value.passed), checks });
