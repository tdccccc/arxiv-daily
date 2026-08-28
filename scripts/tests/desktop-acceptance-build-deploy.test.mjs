import assert from "node:assert/strict";
import test from "node:test";
import {
  assertVersionUnderTest,
  deployBuildUnderTest,
  deployedArtifactPaths,
} from "../desktop-acceptance/build-deploy.mjs";

const vaultPath = "/home/tester/Desktop/plugin_test";
const pluginId = "arxiv-daily";
const sourceDir = "/repo/plugin";
const pluginDir = `${vaultPath}/.obsidian/plugins/${pluginId}`;
const historicalBackup = `${pluginDir}/main.js.bak-20260807`;

function fakeFs(initial = {}) {
  const files = new Map(Object.entries(initial).map(([k, v]) => [k, Buffer.from(v)]));
  return {
    files,
    calls: [],
    async readFile(path) {
      this.calls.push(["readFile", path]);
      if (!files.has(path)) {
        const error = new Error(`ENOENT: ${path}`);
        error.code = "ENOENT";
        throw error;
      }
      return Buffer.from(files.get(path));
    },
    async writeFile(path, data) {
      this.calls.push(["writeFile", path]);
      files.set(path, Buffer.from(data));
    },
    async mkdir(path) {
      this.calls.push(["mkdir", path]);
    },
  };
}

const goodSource = () => ({
  [`${sourceDir}/main.js`]: "/* built bundle */",
  [`${sourceDir}/manifest.json`]: JSON.stringify({ id: pluginId, version: "0.4.6" }),
});

test("deployedArtifactPaths names only the two release artifacts", () => {
  assert.deepEqual(deployedArtifactPaths(vaultPath, { pluginId }), [
    `${pluginDir}/main.js`,
    `${pluginDir}/manifest.json`,
  ]);
});

test("deploy copies the branch build into the vault and reports its version", async () => {
  const fs = fakeFs({ ...goodSource(), [`${pluginDir}/main.js`]: "old 0.4.5 bundle" });
  const result = await deployBuildUnderTest({ vaultPath, pluginId, sourceDir, fs });
  assert.equal(result.version, "0.4.6");
  assert.equal(fs.files.get(`${pluginDir}/main.js`).toString(), "/* built bundle */");
  assert.deepEqual(result.deployed, deployedArtifactPaths(vaultPath, { pluginId }));
});

test("deploy never reads or writes historical build backups", async () => {
  const fs = fakeFs({ ...goodSource(), [historicalBackup]: "0.3.3 bundle" });
  await deployBuildUnderTest({ vaultPath, pluginId, sourceDir, fs });
  assert.equal(fs.calls.some(([, path]) => path === historicalBackup), false);
  assert.equal(fs.files.get(historicalBackup).toString(), "0.3.3 bundle");
});

test("deploy fails clearly when the branch was never built", async () => {
  const fs = fakeFs({ [`${sourceDir}/manifest.json`]: JSON.stringify({ version: "0.4.6" }) });
  await assert.rejects(
    () => deployBuildUnderTest({ vaultPath, pluginId, sourceDir, fs }),
    /main\.js/,
  );
});

test("deploy refuses a manifest without a usable version", async () => {
  for (const manifest of ['{"id":"x"}', '{"version":""}', '{"version":123}', "not json"]) {
    const fs = fakeFs({
      [`${sourceDir}/main.js`]: "bundle",
      [`${sourceDir}/manifest.json`]: manifest,
    });
    await assert.rejects(
      () => deployBuildUnderTest({ vaultPath, pluginId, sourceDir, fs }),
      /manifest/i,
    );
  }
});

test("deploy writes nothing when the source is incomplete", async () => {
  const fs = fakeFs({ [`${sourceDir}/main.js`]: "bundle" });
  await assert.rejects(() => deployBuildUnderTest({ vaultPath, pluginId, sourceDir, fs }));
  assert.equal(fs.calls.some(([call]) => call === "writeFile"), false);
});

test("assertVersionUnderTest accepts the deployed version", () => {
  assert.equal(assertVersionUnderTest({ expected: "0.4.6", reported: "0.4.6" }), "0.4.6");
});

test("assertVersionUnderTest names both versions on a mismatch", () => {
  assert.throws(
    () => assertVersionUnderTest({ expected: "0.4.6", reported: "0.4.5" }),
    /0\.4\.6[\s\S]*0\.4\.5|0\.4\.5[\s\S]*0\.4\.6/,
  );
});

test("assertVersionUnderTest refuses a missing reported version", () => {
  for (const reported of [undefined, null, "", 0]) {
    assert.throws(() => assertVersionUnderTest({ expected: "0.4.6", reported }), /version/i);
  }
});
