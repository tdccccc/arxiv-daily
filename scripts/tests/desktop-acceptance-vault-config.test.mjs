import assert from "node:assert/strict";
import test from "node:test";
import {
  assertIsolatedConfigHome,
  composeVaultConfig,
} from "../desktop-acceptance/vault-config.mjs";

const vaultPath = "/home/tester/Desktop/plugin_test";
const timestamp = 1787835391946;

test("composeVaultConfig mounts exactly the requested vault", () => {
  const config = composeVaultConfig({ vaultPath, timestamp });
  const entries = Object.entries(config.vaults);
  assert.equal(entries.length, 1);
  const [, entry] = entries[0];
  assert.deepEqual(entry, { path: vaultPath, ts: timestamp, open: true });
});

test("composeVaultConfig derives a stable id for the same vault", () => {
  const first = composeVaultConfig({ vaultPath, timestamp });
  const second = composeVaultConfig({ vaultPath, timestamp: timestamp + 5000 });
  assert.deepEqual(Object.keys(first.vaults), Object.keys(second.vaults));
});

test("composeVaultConfig gives different vaults different ids", () => {
  const other = composeVaultConfig({ vaultPath: "/home/tester/Desktop/other", timestamp });
  const mine = composeVaultConfig({ vaultPath, timestamp });
  assert.notDeepEqual(Object.keys(other.vaults), Object.keys(mine.vaults));
});

test("composeVaultConfig normalizes a trailing separator to one identity", () => {
  const bare = composeVaultConfig({ vaultPath, timestamp });
  const trailing = composeVaultConfig({ vaultPath: `${vaultPath}/`, timestamp });
  assert.deepEqual(Object.keys(trailing.vaults), Object.keys(bare.vaults));
  assert.equal(Object.values(trailing.vaults)[0].path, vaultPath);
});

test("composeVaultConfig rejects a relative vault path", () => {
  assert.throws(
    () => composeVaultConfig({ vaultPath: "Desktop/plugin_test", timestamp }),
    /absolute/i,
  );
});

test("composeVaultConfig rejects a missing or non-string vault path", () => {
  for (const bad of [undefined, null, "", 42, {}]) {
    assert.throws(() => composeVaultConfig({ vaultPath: bad, timestamp }), /vault path/i);
  }
});

test("composeVaultConfig rejects a non-finite timestamp", () => {
  for (const bad of [undefined, Number.NaN, "now"]) {
    assert.throws(() => composeVaultConfig({ vaultPath, timestamp: bad }), /timestamp/i);
  }
});

const realConfigHome = "/home/tester/.config";

test("assertIsolatedConfigHome accepts a disposable directory", () => {
  assert.equal(
    assertIsolatedConfigHome("/tmp/obsidian-harness.abc123/config", { realConfigHome }),
    "/tmp/obsidian-harness.abc123/config",
  );
});

test("assertIsolatedConfigHome rejects the real config home", () => {
  assert.throws(
    () => assertIsolatedConfigHome(realConfigHome, { realConfigHome }),
    /real Obsidian configuration/i,
  );
});

test("assertIsolatedConfigHome rejects a directory inside the real config home", () => {
  assert.throws(
    () => assertIsolatedConfigHome(`${realConfigHome}/obsidian`, { realConfigHome }),
    /real Obsidian configuration/i,
  );
});

test("assertIsolatedConfigHome rejects a directory containing the real config home", () => {
  assert.throws(
    () => assertIsolatedConfigHome("/home/tester", { realConfigHome }),
    /real Obsidian configuration/i,
  );
});

test("assertIsolatedConfigHome is not fooled by a trailing separator or dot segments", () => {
  for (const spoof of [`${realConfigHome}/`, `${realConfigHome}/obsidian/..`, "/home/tester/.config/./obsidian"]) {
    assert.throws(
      () => assertIsolatedConfigHome(spoof, { realConfigHome }),
      /real Obsidian configuration/i,
    );
  }
});

test("assertIsolatedConfigHome rejects a relative directory", () => {
  assert.throws(
    () => assertIsolatedConfigHome("tmp/config", { realConfigHome }),
    /absolute/i,
  );
});
