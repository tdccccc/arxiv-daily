import assert from "node:assert/strict";
import test from "node:test";
import { startModelsListener, startProbeListener } from "../desktop-acceptance/probe-listener.mjs";

test("the listener reports loopback urls a sidecar client can be pointed at", async () => {
  const listener = await startProbeListener();
  try {
    assert.match(listener.capabilitiesUrl, /^http:\/\/127\.0\.0\.1:\d+\/v1\/capabilities$/);
    assert.match(listener.parseUrl, /^http:\/\/127\.0\.0\.1:\d+\/v1\/parse$/);
  } finally {
    await listener.close();
  }
});

test("a request is recorded with its method and path", async () => {
  const listener = await startProbeListener();
  try {
    await fetch(listener.capabilitiesUrl).catch(() => {});
    assert.equal(listener.requests().length, 1);
    assert.equal(listener.requests()[0].path, "/v1/capabilities");
    assert.equal(listener.requests()[0].method, "GET");
  } finally {
    await listener.close();
  }
});

test("the listener fails requests by default so a probe cannot succeed", async () => {
  const listener = await startProbeListener();
  try {
    const response = await fetch(listener.capabilitiesUrl);
    assert.equal(response.ok, false);
  } finally {
    await listener.close();
  }
});

test("nothing is recorded when nobody calls", async () => {
  const listener = await startProbeListener();
  try {
    assert.deepEqual(listener.requests(), []);
  } finally {
    await listener.close();
  }
});

test("closing twice is safe", async () => {
  const listener = await startProbeListener();
  await listener.close();
  await listener.close();
});

test("each listener takes its own port so runs do not collide", async () => {
  const [a, b] = [await startProbeListener(), await startProbeListener()];
  try {
    assert.notEqual(a.capabilitiesUrl, b.capabilitiesUrl);
  } finally {
    await a.close();
    await b.close();
  }
});

test("the models listener answers any request with an OpenAI-compatible model list", async () => {
  const listener = await startModelsListener({ models: ["a", "b"] });
  try {
    const response = await fetch(`${listener.origin}/v1/models`);
    assert.equal(response.ok, true);
    assert.deepEqual(await response.json(), { data: [{ id: "a" }, { id: "b" }] });
    assert.equal(listener.requests().length, 1);
    assert.equal(listener.requests()[0].path, "/v1/models");
  } finally {
    await listener.close();
  }
});

test("the models listener defaults to two distinguishable stub models", async () => {
  const listener = await startModelsListener();
  try {
    assert.deepEqual(listener.models, ["stub-model-a", "stub-model-b"]);
  } finally {
    await listener.close();
  }
});
