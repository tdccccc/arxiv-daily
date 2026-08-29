import assert from "node:assert/strict";
import test from "node:test";
import { createCdpClient, evaluate, selectVaultTarget } from "../desktop-acceptance/cdp.mjs";

const vaultTarget = {
  type: "page",
  url: "app://obsidian.md/index.html",
  title: "plugin_test - Obsidian",
  webSocketDebuggerUrl: "ws://127.0.0.1:9333/devtools/page/ABC",
};
const starterTarget = {
  type: "page",
  url: "app://obsidian.md/starter.html",
  title: "Obsidian",
  webSocketDebuggerUrl: "ws://127.0.0.1:9333/devtools/page/DEF",
};

test("selectVaultTarget picks the open vault window", () => {
  const target = selectVaultTarget([{ type: "worker", url: "" }, vaultTarget]);
  assert.equal(target.webSocketDebuggerUrl, vaultTarget.webSocketDebuggerUrl);
});

test("selectVaultTarget refuses the starter window rather than testing the wrong page", () => {
  assert.throws(() => selectVaultTarget([starterTarget]), /vault/i);
});

test("selectVaultTarget reports when no page target exists at all", () => {
  assert.throws(() => selectVaultTarget([{ type: "worker", url: "" }]), /no .*page|target/i);
});

/** Fake CDP socket: records frames sent and lets a test push replies and events. */
function fakeSocket() {
  const listeners = new Map();
  const socket = {
    sent: [],
    closed: false,
    addEventListener(type, handler) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(handler);
    },
    send(data) {
      socket.sent.push(JSON.parse(data));
    },
    close() {
      socket.closed = true;
      socket.emit("close", {});
    },
    emit(type, event) {
      for (const handler of listeners.get(type) ?? []) handler(event);
    },
    open() {
      socket.emit("open", {});
    },
    deliver(payload) {
      socket.emit("message", { data: JSON.stringify(payload) });
    },
  };
  return socket;
}

async function connectedClient() {
  const socket = fakeSocket();
  const client = createCdpClient({ url: vaultTarget.webSocketDebuggerUrl, createSocket: () => socket });
  const ready = client.ready();
  socket.open();
  await ready;
  return { socket, client };
}

test("a request is paired with its response by id", async () => {
  const { socket, client } = await connectedClient();
  const pending = client.send("Runtime.evaluate", { expression: "1+1" });
  const frame = socket.sent.at(-1);
  assert.equal(frame.method, "Runtime.evaluate");
  socket.deliver({ id: frame.id, result: { value: 2 } });
  assert.deepEqual(await pending, { value: 2 });
});

test("concurrent requests resolve independently and in any order", async () => {
  const { socket, client } = await connectedClient();
  const first = client.send("A");
  const second = client.send("B");
  const [frameA, frameB] = socket.sent.slice(-2);
  socket.deliver({ id: frameB.id, result: { which: "B" } });
  socket.deliver({ id: frameA.id, result: { which: "A" } });
  assert.deepEqual(await first, { which: "A" });
  assert.deepEqual(await second, { which: "B" });
});

test("a protocol error rejects with its message rather than resolving undefined", async () => {
  const { socket, client } = await connectedClient();
  const pending = client.send("Bogus.method");
  socket.deliver({ id: socket.sent.at(-1).id, error: { code: -32601, message: "'Bogus.method' wasn't found" } });
  await assert.rejects(pending, /wasn't found/);
});

test("events are dispatched to subscribers", async () => {
  const { socket, client } = await connectedClient();
  const seen = [];
  client.on("Runtime.consoleAPICalled", (params) => seen.push(params));
  socket.deliver({ method: "Runtime.consoleAPICalled", params: { type: "error" } });
  assert.deepEqual(seen, [{ type: "error" }]);
});

test("closing the connection rejects everything still in flight", async () => {
  const { socket, client } = await connectedClient();
  const pending = client.send("Runtime.evaluate", { expression: "never answered" });
  socket.close();
  await assert.rejects(pending, /closed/i);
});

test("a socket error surfaces instead of hanging forever", async () => {
  const socket = fakeSocket();
  const client = createCdpClient({ url: "ws://127.0.0.1:1/x", createSocket: () => socket });
  const ready = client.ready();
  socket.emit("error", { message: "ECONNREFUSED" });
  await assert.rejects(ready, /ECONNREFUSED|connect/i);
});

test("evaluate returns the value by value", async () => {
  const { socket, client } = await connectedClient();
  const pending = evaluate(client, "app.vault.getName()");
  const frame = socket.sent.at(-1);
  assert.equal(frame.params.returnByValue, true);
  assert.equal(frame.params.awaitPromise, true);
  socket.deliver({ id: frame.id, result: { result: { value: "plugin_test" } } });
  assert.equal(await pending, "plugin_test");
});

test("evaluate turns a thrown expression into a rejected promise carrying the description", async () => {
  const { socket, client } = await connectedClient();
  const pending = evaluate(client, "boom()");
  socket.deliver({
    id: socket.sent.at(-1).id,
    result: {
      result: { type: "object" },
      exceptionDetails: { text: "Uncaught", exception: { description: "ReferenceError: boom is not defined" } },
    },
  });
  await assert.rejects(pending, /boom is not defined/);
});

test("evaluate does not silently swallow an undefined result", async () => {
  const { socket, client } = await connectedClient();
  const pending = evaluate(client, "void 0");
  socket.deliver({ id: socket.sent.at(-1).id, result: { result: { type: "undefined" } } });
  assert.equal(await pending, undefined);
});
