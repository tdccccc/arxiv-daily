import assert from "node:assert/strict";
import test from "node:test";
import { createRequestLog } from "../desktop-acceptance/network.mjs";
import { installSettingsFixture, legacySettingsFixture } from "../desktop-acceptance/settings-fixture.mjs";

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

test("starting the request log enables the network domain", async () => {
  const client = fakeClient();
  await createRequestLog(client);
  assert.ok(client.enabled.includes("Network.enable"));
});

test("requests are recorded in order with their url", async () => {
  const client = fakeClient();
  const log = await createRequestLog(client);
  client.emit("Network.requestWillBeSent", { request: { url: "http://127.0.0.1:8765/capabilities" } });
  client.emit("Network.requestWillBeSent", { request: { url: "app://obsidian.md/app.js" } });
  assert.deepEqual(log.urls(), ["http://127.0.0.1:8765/capabilities", "app://obsidian.md/app.js"]);
});

test("the request log can report only requests that left the app scheme", async () => {
  const client = fakeClient();
  const log = await createRequestLog(client);
  client.emit("Network.requestWillBeSent", { request: { url: "app://obsidian.md/app.js" } });
  client.emit("Network.requestWillBeSent", { request: { url: "http://127.0.0.1:8765/parse" } });
  assert.deepEqual(log.networkUrls(), ["http://127.0.0.1:8765/parse"]);
});

test("inline data URIs are not counted as requests that left the process", async () => {
  const client = fakeClient();
  const log = await createRequestLog(client);
  client.emit("Network.requestWillBeSent", { request: { url: "data:image/svg+xml,%3Csvg/%3E" } });
  client.emit("Network.requestWillBeSent", { request: { url: "blob:app://obsidian.md/abc" } });
  client.emit("Network.requestWillBeSent", { request: { url: "http://127.0.0.1:5001/v1/parse" } });
  assert.deepEqual(log.networkUrls(), ["http://127.0.0.1:5001/v1/parse"]);
});

test("a malformed event does not break the log", async () => {
  const client = fakeClient();
  const log = await createRequestLog(client);
  client.emit("Network.requestWillBeSent", {});
  assert.deepEqual(log.urls(), []);
});

test("the legacy fixture has no sidecar section so migration has something to do", () => {
  const fixture = legacySettingsFixture();
  assert.equal("pdfParserSidecar" in fixture.settings, false);
  assert.ok(Object.keys(fixture.settings).length > 0);
});

test("installing the fixture writes it to the plugin settings store", async () => {
  const written = new Map();
  const fs = {
    async mkdir() {},
    async writeFile(path, data) {
      written.set(path, data);
    },
  };
  await installSettingsFixture({
    vaultPath: "/vault",
    pluginId: "arxiv-daily",
    fs,
    data: legacySettingsFixture(),
  });
  const [path, data] = [...written.entries()][0];
  assert.equal(path, "/vault/.obsidian/plugins/arxiv-daily/data.json");
  assert.equal("pdfParserSidecar" in JSON.parse(data).settings, false);
});

test("installing the fixture refuses a relative vault path", async () => {
  await assert.rejects(
    () => installSettingsFixture({ vaultPath: "vault", pluginId: "x", fs: {}, data: {} }),
    /absolute/i,
  );
});
