import assert from "node:assert/strict";
import test from "node:test";
import { createDiagnostics } from "../desktop-acceptance/diagnostics.mjs";

/** Fake CDP client that lets a test push protocol events. */
function fakeClient() {
  const subscribers = new Map();
  return {
    enabled: [],
    on(method, handler) {
      if (!subscribers.has(method)) subscribers.set(method, []);
      subscribers.get(method).push(handler);
    },
    async send(method) {
      this.enabled.push(method);
      return {};
    },
    emit(method, params) {
      for (const handler of subscribers.get(method) ?? []) handler(params);
    },
  };
}

async function started() {
  const client = fakeClient();
  const diagnostics = await createDiagnostics(client);
  return { client, diagnostics };
}

test("starting diagnostics enables the runtime domain", async () => {
  const { client } = await started();
  assert.ok(client.enabled.includes("Runtime.enable"));
});

test("console errors are collected with their text", async () => {
  const { client, diagnostics } = await started();
  client.emit("Runtime.consoleAPICalled", {
    type: "error",
    args: [{ value: "plugin failed to load" }],
  });
  assert.equal(diagnostics.errors().length, 1);
  assert.match(diagnostics.errors()[0].text, /plugin failed to load/);
});

test("uncaught exceptions are collected as page errors", async () => {
  const { client, diagnostics } = await started();
  client.emit("Runtime.exceptionThrown", {
    exceptionDetails: { exception: { description: "TypeError: x is not a function" } },
  });
  assert.equal(diagnostics.errors().length, 1);
  assert.equal(diagnostics.errors()[0].source, "pageerror");
});

test("warnings and logs are recorded but are not errors", async () => {
  const { client, diagnostics } = await started();
  client.emit("Runtime.consoleAPICalled", { type: "warning", args: [{ value: "deprecated" }] });
  client.emit("Runtime.consoleAPICalled", { type: "log", args: [{ value: "hello" }] });
  assert.equal(diagnostics.errors().length, 0);
  assert.equal(diagnostics.warnings().length, 1);
  assert.equal(diagnostics.entries().length, 2);
});

test("non-string console arguments are rendered rather than dropped", async () => {
  const { client, diagnostics } = await started();
  client.emit("Runtime.consoleAPICalled", {
    type: "error",
    args: [{ description: "Error: boom" }, { type: "object", className: "Object" }],
  });
  const { text } = diagnostics.errors()[0];
  assert.match(text, /Error: boom/);
  assert.ok(text.length > "Error: boom".length);
});

test("assertNoErrors passes on a clean run", async () => {
  const { diagnostics } = await started();
  diagnostics.assertNoErrors();
});

test("assertNoErrors names every error it found", async () => {
  const { client, diagnostics } = await started();
  client.emit("Runtime.consoleAPICalled", { type: "error", args: [{ value: "first failure" }] });
  client.emit("Runtime.exceptionThrown", {
    exceptionDetails: { exception: { description: "second failure" } },
  });
  assert.throws(() => diagnostics.assertNoErrors(), /first failure[\s\S]*second failure/);
});

test("ignored patterns keep a known-benign message out of the error list", async () => {
  const client = fakeClient();
  const diagnostics = await createDiagnostics(client, { ignore: [/Autofill\.enable/] });
  client.emit("Runtime.consoleAPICalled", {
    type: "error",
    args: [{ value: "Request Autofill.enable failed" }],
  });
  assert.equal(diagnostics.errors().length, 0);
  assert.equal(diagnostics.ignored().length, 1);
});

test("entries keep arrival order so a failure can be read as a sequence", async () => {
  const { client, diagnostics } = await started();
  client.emit("Runtime.consoleAPICalled", { type: "log", args: [{ value: "one" }] });
  client.emit("Runtime.consoleAPICalled", { type: "error", args: [{ value: "two" }] });
  client.emit("Runtime.consoleAPICalled", { type: "log", args: [{ value: "three" }] });
  assert.deepEqual(
    diagnostics.entries().map((entry) => entry.text),
    ["one", "two", "three"],
  );
});
