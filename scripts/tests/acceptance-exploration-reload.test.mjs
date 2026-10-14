import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import vm from "node:vm";
import { createObsidianDriver } from "../acceptance/obsidian-ui.mjs";

/**
 * P3 Chunk 6 (docs/helm/2026-10-13-exploration-safety-and-coverage/phases/03-obsidian-library-tasks.md):
 * two overlapping `reloadPlugin()` calls on the same session create two live plugin instances (confirmed
 * on the real host, /tmp/reload-race: concurrent reloads reproduced "Attempting to register an existing
 * view type" 37-38 times in 20 iterations; sequential awaited reloads were clean). This is exactly the
 * shape `scripts/acceptance/exploration.mjs`'s `withDeadline()` produces on a timeout: it stops *waiting*
 * on an in-flight reset()/prepare() (whose `reloadPlugin()` keeps running for real) while the next task's
 * own reset() starts a second `reloadPlugin()` in the same renderer.
 *
 * These tests drive `createObsidianDriver`'s real `reloadPlugin()` against a fake `app.plugins` that
 * records call order/timing and models two things Obsidian's own bundled app.js does for real (read
 * directly from the installed obsidian.asar, see obsidian-ui.mjs's comment above `doReload`):
 *   - `enablePlugin(id)` sets `app.plugins.loadingPluginId = id` for the whole duration of the awaited
 *     onload(), clearing it back to null on both success and failure.
 *   - that applies equally to a caller-triggered enable and to Obsidian's own startup auto-enable.
 *
 * No real CDP/Electron is involved: `session.evaluate` runs the exact expression strings
 * `createObsidianDriver` builds inside a fresh `vm` context exposing a fake `app`, awaiting whatever
 * comes back exactly like `desktop-acceptance/cdp.mjs`'s real `evaluate()` does with `awaitPromise: true`.
 */
function buildFakeSession({ onloadDelayMs = 20, disableDelayMs = 5 } = {}) {
  const events = [];
  const now = () => Date.now();
  const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
  let loadingPluginId = null;
  const httpClient = { request: async () => ({}) };
  const app = {
    setting: {
      close: () => { events.push({ type: "close-settings", t: now() }); },
    },
    plugins: {
      plugins: { "arxiv-daily": { getHttpClient: () => httpClient } },
      get loadingPluginId() { return loadingPluginId; },
      async disablePlugin(id) {
        events.push({ type: "disable", id, t: now() });
        await pause(disableDelayMs);
      },
      async enablePlugin(id) {
        loadingPluginId = id;
        events.push({ type: "enable-start", id, t: now() });
        try {
          await pause(onloadDelayMs);
        } finally {
          loadingPluginId = null;
        }
        events.push({ type: "enable-end", id, t: now() });
      },
    },
  };
  const context = vm.createContext({ app });
  const evaluate = async expression => {
    const value = vm.runInContext(expression, context);
    return await value;
  };
  // createObsidianDriver's construction unconditionally wires up a screenshot writer (unused by these
  // tests) through `session.client`, so it needs a stub `send()` -- see
  // acceptance-exploration-obsidian.test.mjs's own `buildCaptureSession` for the same shape.
  const client = { send: async () => ({}) };
  return { events, app, session: { evaluate, client } };
}

async function withTempArtifactDir(run) {
  const root = await mkdtemp(path.join(tmpdir(), "exploration-reload-test-"));
  try { return await run(root); }
  finally { await rm(root, { recursive: true, force: true }); }
}

function cycles(events) {
  // Groups the flat event log back into whole disable→enable-start→enable-end cycles, in the order
  // they actually happened, so a test can assert cycles never interleave (every cycle's own events are
  // contiguous and ordered) without hard-coding how many there are.
  const result = [];
  let current = null;
  for (const event of events) {
    if (event.type === "disable") {
      if (current) result.push(current);
      current = { disable: event };
    } else if (event.type === "enable-start" && current) {
      current.enableStart = event;
    } else if (event.type === "enable-end" && current) {
      current.enableEnd = event;
    }
  }
  if (current) result.push(current);
  return result;
}

test("driver.reloadPlugin() serializes two calls issued without awaiting the first (the withDeadline-abandoned shape): the second cycle's disable never starts before the first cycle's enable has finished", async () => {
  await withTempArtifactDir(async root => {
    const { events, session } = buildFakeSession({ onloadDelayMs: 30, disableDelayMs: 5 });
    const fixture = { vaultRoot: path.join(root, "vault"), server: { url: "http://127.0.0.1:1" } };
    const driver = await createObsidianDriver({ session, fixture, artifactDir: path.join(root, "artifacts") });

    // Exactly the shape of the real bug: nobody awaits the first call before the second is issued.
    await Promise.all([driver.reloadPlugin(), driver.reloadPlugin()]);

    const [first, second] = cycles(events);
    assert.ok(first?.disable && first.enableStart && first.enableEnd, "first reload cycle must run to completion");
    assert.ok(second?.disable && second.enableStart && second.enableEnd, "second reload cycle must run to completion");
    assert.ok(
      second.disable.t >= first.enableEnd.t,
      `second cycle's disable (t=${second.disable.t}) must not start before the first cycle's enable finished (t=${first.enableEnd.t}) -- otherwise both reloads have a plugin instance registering views concurrently`,
    );
  });
});

test("driver.reloadPlugin() still serializes when the first call's caller never awaits it at all (withDeadline gives up, the next task starts its own reset())", async () => {
  await withTempArtifactDir(async root => {
    const { events, session } = buildFakeSession({ onloadDelayMs: 30, disableDelayMs: 5 });
    const fixture = { vaultRoot: path.join(root, "vault"), server: { url: "http://127.0.0.1:1" } };
    const driver = await createObsidianDriver({ session, fixture, artifactDir: path.join(root, "artifacts") });

    driver.reloadPlugin(); // abandoned, exactly like withDeadline's timeout path -- not awaited, not even referenced.
    await driver.reloadPlugin(); // the "next task's reset()" call.

    const [first, second] = cycles(events);
    assert.ok(first?.enableEnd && second?.enableEnd, "both the abandoned and the real caller's reload must run to completion");
    assert.ok(
      second.disable.t >= first.enableEnd.t,
      "the next task's reload must not start its own disable until the abandoned reload's enable has finished",
    );
  });
});

test("driver.reloadPlugin() waits for a plugin load already in flight (e.g. Obsidian's own startup auto-enable) before issuing its own disable/enable", async () => {
  await withTempArtifactDir(async root => {
    const { events, app, session } = buildFakeSession({ onloadDelayMs: 10, disableDelayMs: 5 });
    const fixture = { vaultRoot: path.join(root, "vault"), server: { url: "http://127.0.0.1:1" } };
    const driver = await createObsidianDriver({ session, fixture, artifactDir: path.join(root, "artifacts") });

    // Model Obsidian's own startup enable of the plugin, already in flight when reset() first runs: no
    // "disable"/"enable-start" event is recorded for it, exactly like the harness never calls
    // disablePlugin/enablePlugin for Obsidian's own auto-enable -- only `loadingPluginId` is observable.
    let startupCleared = false;
    const startupClearAt = Date.now() + 40;
    const originalDescriptor = Object.getOwnPropertyDescriptor(app.plugins, "loadingPluginId");
    Object.defineProperty(app.plugins, "loadingPluginId", {
      configurable: true,
      get() {
        if (!startupCleared && Date.now() >= startupClearAt) startupCleared = true;
        return startupCleared ? null : "arxiv-daily";
      },
    });

    const before = Date.now();
    await driver.reloadPlugin();

    assert.ok(startupCleared, "the simulated startup load must have been observed as cleared");
    const [first] = cycles(events);
    assert.ok(first?.disable, "reloadPlugin() must eventually issue its own disable once the startup load clears");
    assert.ok(
      first.disable.t >= startupClearAt,
      `reloadPlugin()'s own disable (t=${first.disable.t}) must not start before the in-flight startup load cleared (t=${startupClearAt})`,
    );
    assert.ok(Date.now() - before >= 40, "reloadPlugin() must actually have waited, not raced ahead");
    Object.defineProperty(app.plugins, "loadingPluginId", originalDescriptor);
  });
});
