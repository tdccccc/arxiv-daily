import assert from "node:assert/strict";
import test from "node:test";
import {
  checkPublishedManifest,
  DEFAULT_REPO,
  evaluatePublishedManifest,
  fetchReleaseByTag,
} from "../check-published-manifest.mjs";

function jsonResponse(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 404 ? "Not Found" : "Error",
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

test("evaluatePublishedManifest passes for a published, non-draft release", () => {
  const result = evaluatePublishedManifest("0.5.0", { kind: "found", draft: false });
  assert.equal(result.ok, true);
  assert.match(result.message, /Published GitHub release 0\.5\.0 found/);
});

test("evaluatePublishedManifest fails with a helpful message when the release is missing", () => {
  const result = evaluatePublishedManifest("0.5.0", { kind: "missing" });
  assert.equal(result.ok, false);
  assert.match(result.message, /no GitHub release with tag "0\.5\.0" exists yet/);
  assert.match(result.message, /release\/0\.5\.0 branch/);
  assert.match(result.message, /delisted/);
});

test("evaluatePublishedManifest fails when the matching release is still a draft", () => {
  const result = evaluatePublishedManifest("0.5.0", { kind: "found", draft: true });
  assert.equal(result.ok, false);
  assert.match(result.message, /exists only as a draft/);
  assert.match(result.message, /Publish the release/);
});

test("evaluatePublishedManifest fails clearly on an API/network error instead of passing silently", () => {
  const result = evaluatePublishedManifest("0.5.0", { kind: "error", message: "ECONNRESET" });
  assert.equal(result.ok, false);
  assert.match(result.message, /verifying its GitHub release failed: ECONNRESET/);
  assert.match(result.message, /re-run/i);
});

test("evaluatePublishedManifest fails on an unrecognized result instead of defaulting to pass", () => {
  const result = evaluatePublishedManifest("0.5.0", undefined);
  assert.equal(result.ok, false);
});

test("fetchReleaseByTag reports found with draft status from a successful lookup", async () => {
  let requestedUrl;
  let requestedHeaders;
  const fetchImpl = async (url, options) => {
    requestedUrl = url;
    requestedHeaders = options.headers;
    return jsonResponse(200, { draft: false, html_url: "https://github.com/tdccccc/arxiv-daily/releases/tag/0.5.0" });
  };
  const result = await fetchReleaseByTag("tdccccc/arxiv-daily", "0.5.0", { fetchImpl, token: "secret" });
  assert.deepEqual(result, {
    kind: "found",
    draft: false,
    htmlUrl: "https://github.com/tdccccc/arxiv-daily/releases/tag/0.5.0",
  });
  assert.equal(requestedUrl, "https://api.github.com/repos/tdccccc/arxiv-daily/releases/tags/0.5.0");
  assert.equal(requestedHeaders.Authorization, "Bearer secret");
});

test("fetchReleaseByTag reports missing on a 404", async () => {
  const fetchImpl = async () => jsonResponse(404, { message: "Not Found" });
  const result = await fetchReleaseByTag("tdccccc/arxiv-daily", "0.5.0", { fetchImpl });
  assert.deepEqual(result, { kind: "missing" });
});

test("fetchReleaseByTag reports an error on non-404 HTTP failures", async () => {
  const fetchImpl = async () => jsonResponse(500, { message: "boom" });
  const result = await fetchReleaseByTag("tdccccc/arxiv-daily", "0.5.0", { fetchImpl });
  assert.equal(result.kind, "error");
  assert.match(result.message, /500/);
});

test("fetchReleaseByTag reports an error when the request itself throws", async () => {
  const fetchImpl = async () => {
    throw new Error("network unreachable");
  };
  const result = await fetchReleaseByTag("tdccccc/arxiv-daily", "0.5.0", { fetchImpl });
  assert.equal(result.kind, "error");
  assert.match(result.message, /network unreachable/);
});

test("fetchReleaseByTag omits Authorization when no token is provided", async () => {
  let requestedHeaders;
  const fetchImpl = async (url, options) => {
    requestedHeaders = options.headers;
    return jsonResponse(200, { draft: false });
  };
  await fetchReleaseByTag("tdccccc/arxiv-daily", "0.5.0", { fetchImpl });
  assert.equal(requestedHeaders.Authorization, undefined);
});

test("checkPublishedManifest reads the root manifest version and defaults the repo", async () => {
  let requestedUrl;
  const fetchImpl = async (url) => {
    requestedUrl = url;
    return jsonResponse(200, { draft: false });
  };
  const result = await checkPublishedManifest({ fetchImpl, env: {} });
  assert.equal(result.ok, true);
  assert.ok(requestedUrl.startsWith(`https://api.github.com/repos/${DEFAULT_REPO}/releases/tags/`));
});

test("checkPublishedManifest honors GITHUB_REPOSITORY and GH_TOKEN/GITHUB_TOKEN overrides", async () => {
  let requestedUrl;
  let requestedHeaders;
  const fetchImpl = async (url, options) => {
    requestedUrl = url;
    requestedHeaders = options.headers;
    return jsonResponse(200, { draft: false });
  };
  await checkPublishedManifest({
    fetchImpl,
    env: { GITHUB_REPOSITORY: "example/fork", GH_TOKEN: "tok" },
  });
  assert.ok(requestedUrl.startsWith("https://api.github.com/repos/example/fork/releases/tags/"));
  assert.equal(requestedHeaders.Authorization, "Bearer tok");
});
