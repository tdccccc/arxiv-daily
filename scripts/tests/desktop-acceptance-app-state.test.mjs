import assert from "node:assert/strict";
import test from "node:test";
import {
  APP_USABILITY_EXPRESSION,
  AppErrorStateError,
  blockersFromError,
  judgeAppUsability,
  withAppUsable,
} from "../desktop-acceptance/app-state.mjs";

/**
 * A renderer showing a vault window: Obsidian's own object graph is built, the
 * workspace is mounted, and the settings entry point exists.
 */
const usable = {
  url: "app://obsidian.md/index.html",
  app: true,
  workspace: true,
  settings: true,
  workspaceContainer: true,
  leaves: 3,
  buttons: [],
  text: "plugin_test Welcome",
};

/**
 * What the renderer looked like on 2026-08-31: Obsidian's own error page, with
 * the vault never opened. The ENOSPC wording is *data* here, not a criterion —
 * see the second error state below.
 */
const watchQuotaErrorPage = {
  url: "app://obsidian.md/index.html",
  app: false,
  workspace: false,
  settings: false,
  workspaceContainer: false,
  leaves: 0,
  buttons: ["Reload app", "Open another vault"],
  text: "ENOSPC: System limit for number of file watchers reached, watch '/home/tiandc/Desktop/plugin_test/'",
};

/** A different environment failure behind the same page. */
const missingVaultErrorPage = {
  ...watchQuotaErrorPage,
  text: "Failed to load vault: ENOENT: no such file or directory",
};

test("a mounted vault window is judged usable", () => {
  const verdict = judgeAppUsability(usable);
  assert.equal(verdict.ok, true);
});

test("an application sitting on its error page is not usable", () => {
  const verdict = judgeAppUsability(watchQuotaErrorPage);
  assert.equal(verdict.ok, false);
  // The reason names the capability that is absent…
  assert.match(verdict.reason, /workspace/i);
  // …and quotes what the page says, so the operator sees the real cause.
  assert.match(verdict.reason, /System limit for number of file watchers/);
  assert.match(verdict.reason, /Reload app/);
});

test("the criterion is the missing vault window, not any particular error wording", () => {
  // Same shape, unrelated failure text: the verdict must not depend on ENOSPC.
  const verdict = judgeAppUsability(missingVaultErrorPage);
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /no such file or directory/);
  const source = judgeAppUsability.toString() + APP_USABILITY_EXPRESSION;
  assert.equal(/ENOSPC/.test(source), false, "the check must not match on a specific error message");
  assert.equal(/file watchers/i.test(source), false);
  assert.equal(/Reload app/.test(source), false);
});

test("a half-built renderer — object graph up, workspace never mounted — is not usable", () => {
  const verdict = judgeAppUsability({ ...usable, workspaceContainer: false, leaves: 0 });
  assert.equal(verdict.ok, false);
});

test("the usability expression reads capabilities and reports the page text as data", () => {
  assert.match(APP_USABILITY_EXPRESSION, /workspace/);
  assert.match(APP_USABILITY_EXPRESSION, /setting/);
  assert.match(APP_USABILITY_EXPRESSION, /JSON.stringify/);
});

/** A renderer that answers with each state in turn. */
function renderer(states) {
  const queue = [...states];
  return async () => JSON.stringify(queue.length > 1 ? queue.shift() : queue[0]);
}

test("an application already in an error state stops the walk before a single assertion is produced", async () => {
  let ran = false;
  const sessions = [];
  let blocked = null;
  try {
    sessions.push(
      await withAppUsable({
        evaluate: renderer([watchQuotaErrorPage]),
        run: async () => {
          ran = true;
          return { results: [{ name: "library-settings-group-order", passed: true, detail: "…" }] };
        },
      }),
    );
  } catch (error) {
    blocked = error;
  }
  assert.equal(ran, false, "the walk must not start against an error page");
  assert.ok(blocked instanceof AppErrorStateError);
  assert.deepEqual(sessions, []);
  assert.equal(sessions.flatMap((s) => s.results).filter((r) => r.passed).length, 0);
});

test("an application that collapses mid-walk yields no PASS, only a blocker", async () => {
  const sessions = [];
  let blocked = null;
  try {
    sessions.push(
      await withAppUsable({
        evaluate: renderer([usable, watchQuotaErrorPage]),
        run: async () => ({
          // Exactly the shape of the false green: a full sheet of green.
          results: Array.from({ length: 17 }, (_, i) => ({ name: `assertion-${i}`, passed: true, detail: "…" })),
        }),
      }),
    );
  } catch (error) {
    blocked = error;
  }
  assert.ok(blocked instanceof AppErrorStateError, `expected a blocker, got ${blocked}`);
  // The core of this hardening: results computed against a broken application
  // never reach the caller, so they can never be reported as passes.
  assert.deepEqual(sessions, []);
  assert.equal(sessions.flatMap((s) => s.results ?? []).filter((r) => r.passed).length, 0);
  assert.equal(/PASS/.test(blocked.message), false);
});

test("a healthy application before and after passes its results straight through", async () => {
  const outcome = await withAppUsable({
    evaluate: renderer([usable, usable]),
    run: async () => ({ results: [{ name: "a", passed: true, detail: "…" }] }),
  });
  assert.equal(outcome.results.length, 1);
});

test("the error state is a blocker with a remedy, not a failed assertion", async () => {
  const error = await withAppUsable({
    evaluate: renderer([watchQuotaErrorPage]),
    run: async () => ({}),
  }).catch((thrown) => thrown);
  const blockers = blockersFromError(error);
  assert.ok(Array.isArray(blockers) && blockers.length === 1);
  assert.ok(blockers[0].message.length > 0);
  assert.ok(blockers[0].remedy.length > 0);
  // A plain failure is not a blocker: the two exit codes must stay distinct.
  assert.equal(blockersFromError(new Error("a scenario failed")), null);
});

test("the phase is named, so a blocker says whether the walk had already run", async () => {
  const before = await withAppUsable({
    evaluate: renderer([watchQuotaErrorPage]),
    run: async () => ({}),
  }).catch((error) => error);
  const after = await withAppUsable({
    evaluate: renderer([usable, watchQuotaErrorPage]),
    run: async () => ({}),
  }).catch((error) => error);
  assert.match(before.message, /before/i);
  assert.match(after.message, /after|during/i);
});
