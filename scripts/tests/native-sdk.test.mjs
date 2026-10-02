import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { sha256 } from "../native-build.mjs";
import { sdkDownloadPlan, downloadSdkFile, prepareNativeSdk } from "../native-sdk.mjs";

test("SDK paths are limited to an exact Node version and supported native architecture", () => {
  const plan = sdkDownloadPlan({ version: "22.17.0", platform: "win32", arch: "arm64" });
  assert.deepEqual(plan.files, ["node-v22.17.0-headers.tar.gz", "win-arm64/node.lib"]);
  assert.equal(plan.base, "https://nodejs.org/download/release/v22.17.0/");
  assert.throws(() => sdkDownloadPlan({ version: "../latest", platform: "win32", arch: "x64" }), /version/i);
  assert.throws(() => sdkDownloadPlan({ version: "22.17.0", platform: "win32", arch: "../../escape" }), /architecture/i);
});

test("SDK downloads reject corruption and off-origin redirects before extraction", async () => {
  const bytes = Buffer.from("SDK fixture");
  const fetchImpl = async (_url, options) => {
    assert.equal(options.redirect, "error");
    return new Response(bytes);
  };
  assert.deepEqual(await downloadSdkFile("https://nodejs.org/fixture", sha256(bytes), fetchImpl), bytes);
  await assert.rejects(downloadSdkFile("https://nodejs.org/fixture", "0".repeat(64), fetchImpl), /digest|checksum/i);
  await assert.rejects(downloadSdkFile("https://other.invalid/fixture", sha256(bytes), fetchImpl), /origin|source/i);
});

test("prepares verified headers and the matching Windows import library without executing SDK code", async () => {
  const directory = await mkdtemp(join(tmpdir(), "arxiv-sdk-fixture-"));
  const archive = Buffer.from("archive fixture; extraction is injected");
  const library = Buffer.from("import library fixture");
  let extractions = 0;
  const fetchImpl = async url => {
    if (url.endsWith("SHASUMS256.txt")) return new Response(`${sha256(archive)}  node-v22.17.0-headers.tar.gz\n${sha256(library)}  win-arm64/node.lib\n`);
    return new Response(url.endsWith("node.lib") ? library : archive);
  };
  try {
    const result = await prepareNativeSdk({
      version: "22.17.0", platform: "win32", arch: "arm64", directory, fetchImpl,
      extractArchive: async (_archive, target) => {
        extractions += 1;
        await mkdir(join(target, "node-v22.17.0/include/node"), { recursive: true });
        await writeFile(join(target, "node-v22.17.0/include/node/node_api.h"), "header fixture");
      },
    });
    assert.equal(extractions, 1);
    assert.equal(await readFile(join(result, "include/node/node_api.h"), "utf8"), "header fixture");
    assert.deepEqual(await readFile(join(result, "win-arm64/node.lib")), library);
    const receipt = JSON.parse(await readFile(join(result, "sdk-receipt.json"), "utf8"));
    assert.equal(receipt.files["win-arm64/node.lib"], sha256(library));
  } finally { await rm(directory, { recursive: true, force: true }); }
});
