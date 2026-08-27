import assert from "node:assert/strict";
import test from "node:test";
import { createVaultStateGuard, vaultStatePaths } from "../desktop-acceptance/vault-state.mjs";

const vaultPath = "/home/tester/Desktop/plugin_test";
const pluginId = "arxiv-daily";
const settingsPath = `${vaultPath}/.obsidian/plugins/${pluginId}/data.json`;
const workspacePath = `${vaultPath}/.obsidian/workspace.json`;

/** In-memory stand-in for the byte-level file operations the guard needs. */
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
    async rm(path) {
      this.calls.push(["rm", path]);
      files.delete(path);
    },
  };
}

test("vaultStatePaths protects the plugin settings store and the workspace", () => {
  assert.deepEqual(vaultStatePaths(vaultPath, { pluginId }), [settingsPath, workspacePath]);
});

test("capture then restore returns byte-identical content", async () => {
  const fs = fakeFs({ [settingsPath]: '{"a":1}', [workspacePath]: '{"w":1}' });
  const guard = createVaultStateGuard({ vaultPath, pluginId, fs });
  await guard.capture();
  await fs.writeFile(settingsPath, '{"a":999,"corrupted":true}');
  await fs.writeFile(workspacePath, "{}");
  await guard.restore();
  assert.equal(fs.files.get(settingsPath).toString(), '{"a":1}');
  assert.equal(fs.files.get(workspacePath).toString(), '{"w":1}');
});

test("restore removes a file that did not exist at capture time", async () => {
  const fs = fakeFs({ [workspacePath]: "{}" });
  const guard = createVaultStateGuard({ vaultPath, pluginId, fs });
  await guard.capture();
  await fs.writeFile(settingsPath, '{"created":"by the run"}');
  await guard.restore();
  assert.equal(fs.files.has(settingsPath), false);
});

test("restore is idempotent", async () => {
  const fs = fakeFs({ [settingsPath]: '{"a":1}', [workspacePath]: '{"w":1}' });
  const guard = createVaultStateGuard({ vaultPath, pluginId, fs });
  await guard.capture();
  await fs.writeFile(settingsPath, "changed");
  await guard.restore();
  await guard.restore();
  assert.equal(fs.files.get(settingsPath).toString(), '{"a":1}');
});

test("restore without capture is refused rather than silently deleting", async () => {
  const fs = fakeFs({ [settingsPath]: '{"a":1}' });
  const guard = createVaultStateGuard({ vaultPath, pluginId, fs });
  await assert.rejects(() => guard.restore(), /captured/i);
  assert.equal(fs.files.get(settingsPath).toString(), '{"a":1}');
});

test("protect restores after the body succeeds and returns its value", async () => {
  const fs = fakeFs({ [settingsPath]: '{"a":1}', [workspacePath]: '{"w":1}' });
  const guard = createVaultStateGuard({ vaultPath, pluginId, fs });
  const result = await guard.protect(async () => {
    await fs.writeFile(settingsPath, "mutated by the run");
    return "body result";
  });
  assert.equal(result, "body result");
  assert.equal(fs.files.get(settingsPath).toString(), '{"a":1}');
});

test("protect restores after the body throws and rethrows the original error", async () => {
  const fs = fakeFs({ [settingsPath]: '{"a":1}', [workspacePath]: '{"w":1}' });
  const guard = createVaultStateGuard({ vaultPath, pluginId, fs });
  const boom = new Error("body exploded");
  await assert.rejects(
    () =>
      guard.protect(async () => {
        await fs.writeFile(settingsPath, "mutated by the run");
        throw boom;
      }),
    (error) => error === boom,
  );
  assert.equal(fs.files.get(settingsPath).toString(), '{"a":1}');
});

test("a restore failure does not mask the body's original error", async () => {
  const fs = fakeFs({ [settingsPath]: '{"a":1}', [workspacePath]: '{"w":1}' });
  fs.writeFile = async () => {
    throw new Error("disk gone");
  };
  const guard = createVaultStateGuard({ vaultPath, pluginId, fs });
  const boom = new Error("body exploded");
  await assert.rejects(
    () => guard.protect(() => Promise.reject(boom)),
    (error) => error === boom,
  );
});

test("a restore failure surfaces when the body succeeded", async () => {
  const fs = fakeFs({ [settingsPath]: '{"a":1}', [workspacePath]: '{"w":1}' });
  const guard = createVaultStateGuard({ vaultPath, pluginId, fs });
  await guard.capture();
  fs.writeFile = async () => {
    throw new Error("disk gone");
  };
  await assert.rejects(() => guard.restore(), /disk gone/);
});

/** Minimal stand-in for the signal surface the guard subscribes to. */
function fakeProcess() {
  const listeners = new Map();
  return {
    listeners,
    exitCode: undefined,
    on(signal, handler) {
      listeners.set(signal, handler);
      return this;
    },
    off(signal, handler) {
      if (listeners.get(signal) === handler) listeners.delete(signal);
      return this;
    },
    exit(code) {
      this.exitCode = code;
    },
    async raise(signal) {
      await listeners.get(signal)?.();
    },
  };
}

test("an interrupting signal restores the vault before exiting", async () => {
  const fs = fakeFs({ [settingsPath]: '{"a":1}', [workspacePath]: '{"w":1}' });
  const proc = fakeProcess();
  const guard = createVaultStateGuard({ vaultPath, pluginId, fs, process: proc });
  await guard.capture();
  guard.installSignalRestore();
  await fs.writeFile(settingsPath, "mutated by the run");
  await proc.raise("SIGINT");
  assert.equal(fs.files.get(settingsPath).toString(), '{"a":1}');
  assert.notEqual(proc.exitCode, undefined);
});

test("signal restore covers both SIGINT and SIGTERM", async () => {
  const fs = fakeFs({ [workspacePath]: "{}" });
  const proc = fakeProcess();
  const guard = createVaultStateGuard({ vaultPath, pluginId, fs, process: proc });
  await guard.capture();
  guard.installSignalRestore();
  assert.deepEqual([...proc.listeners.keys()].sort(), ["SIGINT", "SIGTERM"]);
});

test("signal handlers are removed once the guard is released", async () => {
  const fs = fakeFs({ [workspacePath]: "{}" });
  const proc = fakeProcess();
  const guard = createVaultStateGuard({ vaultPath, pluginId, fs, process: proc });
  await guard.capture();
  guard.installSignalRestore();
  guard.removeSignalRestore();
  assert.equal(proc.listeners.size, 0);
});

test("protect installs and removes its own signal handlers", async () => {
  const fs = fakeFs({ [workspacePath]: "{}" });
  const proc = fakeProcess();
  const guard = createVaultStateGuard({ vaultPath, pluginId, fs, process: proc });
  let duringBody = 0;
  await guard.protect(async () => {
    duringBody = proc.listeners.size;
  });
  assert.equal(duringBody, 2);
  assert.equal(proc.listeners.size, 0);
});

test("capture refuses a relative vault path", () => {
  assert.throws(
    () => createVaultStateGuard({ vaultPath: "Desktop/plugin_test", pluginId, fs: fakeFs() }),
    /absolute/i,
  );
});

test("the guard never touches historical build backups next to the settings store", async () => {
  const backup = `${vaultPath}/.obsidian/plugins/${pluginId}/main.js.bak-20260807`;
  const fs = fakeFs({ [settingsPath]: '{"a":1}', [workspacePath]: "{}", [backup]: "old build" });
  const guard = createVaultStateGuard({ vaultPath, pluginId, fs });
  await guard.protect(async () => {});
  assert.equal(fs.calls.some(([, path]) => path === backup), false);
  assert.equal(fs.files.get(backup).toString(), "old build");
});

test("the guard also protects artifacts the harness itself deploys", async () => {
  const mainJs = `${vaultPath}/.obsidian/plugins/${pluginId}/main.js`;
  const fs = fakeFs({ [settingsPath]: "{}", [workspacePath]: "{}", [mainJs]: "vault's own 0.4.5 build" });
  const guard = createVaultStateGuard({
    vaultPath,
    pluginId,
    fs,
    additionalPaths: [mainJs],
  });
  await guard.protect(async () => {
    await fs.writeFile(mainJs, "branch build under test");
  });
  assert.equal(fs.files.get(mainJs).toString(), "vault's own 0.4.5 build");
});

test("additionalPaths must be absolute and inside the vault", () => {
  for (const bad of ["relative/main.js", "/elsewhere/main.js"]) {
    assert.throws(
      () => createVaultStateGuard({ vaultPath, pluginId, fs: fakeFs(), additionalPaths: [bad] }),
      /absolute|inside the vault/i,
    );
  }
});
