import assert from "node:assert/strict";
import test from "node:test";
import {
  acceptVaultTrust,
  waitForPluginLoaded,
  waitForPluginReady,
} from "../desktop-acceptance/trust.mjs";

/** Stub evaluator: maps an expression substring to a canned result or thrower. */
function stubEvaluate(script) {
  const calls = [];
  const evaluate = async (expression) => {
    calls.push(expression);
    for (const [needle, produce] of script) {
      if (expression.includes(needle)) {
        return typeof produce === "function" ? produce(calls.length) : produce;
      }
    }
    return undefined;
  };
  evaluate.calls = calls;
  return evaluate;
}

test("the trust dialog is accepted when Obsidian shows it", async () => {
  const evaluate = stubEvaluate([["Trust author", "clicked"]]);
  const result = await acceptVaultTrust({ evaluate });
  assert.equal(result.accepted, true);
  assert.ok(evaluate.calls.some((call) => call.includes("Trust author")));
});

test("a run with no trust dialog is a no-op rather than a failure", async () => {
  const evaluate = stubEvaluate([["Trust author", "absent"]]);
  const result = await acceptVaultTrust({ evaluate });
  assert.equal(result.accepted, false);
});

test("waitForPluginLoaded returns the running plugin version", async () => {
  const evaluate = stubEvaluate([["manifest", "0.4.3"]]);
  const version = await waitForPluginLoaded({
    evaluate,
    pluginId: "arxiv-daily",
    sleep: async () => {},
  });
  assert.equal(version, "0.4.3");
});

test("waitForPluginLoaded polls instead of sleeping a fixed amount", async () => {
  let slept = 0;
  const evaluate = stubEvaluate([["manifest", (call) => (call < 3 ? null : "0.4.3")]]);
  const version = await waitForPluginLoaded({
    evaluate,
    pluginId: "arxiv-daily",
    sleep: async () => {
      slept += 1;
    },
  });
  assert.equal(version, "0.4.3");
  assert.equal(slept, 2);
});

test("waitForPluginLoaded fails with a diagnosable message when the plugin never loads", async () => {
  const evaluate = stubEvaluate([["manifest", null]]);
  await assert.rejects(
    () =>
      waitForPluginLoaded({
        evaluate,
        pluginId: "arxiv-daily",
        attempts: 3,
        sleep: async () => {},
      }),
    /arxiv-daily[\s\S]*(never loaded|3 attempts)/i,
  );
});

test("the plugin query names the requested plugin id", async () => {
  const evaluate = stubEvaluate([["manifest", "1.0.0"]]);
  await waitForPluginLoaded({ evaluate, pluginId: "some-other-plugin", sleep: async () => {} });
  assert.ok(evaluate.calls.some((call) => call.includes("some-other-plugin")));
});

test("the plugin query tolerates a renderer where app does not exist yet", async () => {
  const evaluate = stubEvaluate([["manifest", "0.4.3"]]);
  await waitForPluginLoaded({ evaluate, pluginId: "arxiv-daily", sleep: async () => {} });
  const query = evaluate.calls.find((call) => call.includes("manifest"));
  assert.match(query, /typeof app/);
});

test("waitForPluginReady accepts a trust dialog that only appears on a later poll", async () => {
  let poll = 0;
  const evaluate = async (expression) => {
    if (expression.includes("Trust author")) return poll >= 2 ? "clicked" : "absent";
    if (expression.includes("manifest")) return poll++ >= 3 ? "0.4.3" : null;
    return undefined;
  };
  const result = await waitForPluginReady({
    evaluate,
    pluginId: "arxiv-daily",
    sleep: async () => {},
  });
  assert.equal(result.version, "0.4.3");
  assert.equal(result.trustPromptAccepted, true);
});

test("waitForPluginReady reports when no trust dialog was ever needed", async () => {
  const evaluate = async (expression) => {
    if (expression.includes("Trust author")) return "absent";
    if (expression.includes("manifest")) return "0.4.3";
    return undefined;
  };
  const result = await waitForPluginReady({
    evaluate,
    pluginId: "arxiv-daily",
    sleep: async () => {},
  });
  assert.equal(result.trustPromptAccepted, false);
});

test("waitForPluginReady surfaces a renderer that never produces the plugin", async () => {
  const evaluate = async () => null;
  await assert.rejects(
    () =>
      waitForPluginReady({
        evaluate,
        pluginId: "arxiv-daily",
        attempts: 3,
        sleep: async () => {},
      }),
    /arxiv-daily/,
  );
});

test("waitForPluginReady reports diagnostics as complete when the plugin loads after trust", async () => {
  let accepted = false;
  const evaluate = async (expression) => {
    if (expression.includes("Trust author")) {
      accepted = true;
      return "clicked";
    }
    if (expression.includes("manifest")) return accepted ? "0.4.3" : null;
    return undefined;
  };
  const result = await waitForPluginReady({
    evaluate,
    pluginId: "arxiv-daily",
    sleep: async () => {},
  });
  assert.equal(result.loadedBeforeAttach, false);
});

test("waitForPluginReady flags a plugin that was already running when we attached", async () => {
  // A vault whose trust was already granted loads plugins before the harness
  // can enable diagnostics, so startup output would be missed.
  const evaluate = async (expression) => (expression.includes("manifest") ? "0.4.3" : "absent");
  const result = await waitForPluginReady({
    evaluate,
    pluginId: "arxiv-daily",
    sleep: async () => {},
  });
  assert.equal(result.loadedBeforeAttach, true);
});
