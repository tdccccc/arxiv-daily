import assert from "node:assert/strict";
import test from "node:test";
import {
  connectedLibraryFixture,
  installSettingsFixture,
  legacySettingsFixture,
  readRootIdentity,
  resolveLibraryRoot,
} from "../desktop-acceptance/settings-fixture.mjs";

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

test("the connected-library fixture is selected but ungranted, with a disclosable endpoint", () => {
  const fixture = connectedLibraryFixture({ libraryRoot: "/vault/library", rootIdentity: "66:1234" });
  assert.equal(fixture.libraryConnection.selectedRoot, "/vault/library");
  assert.equal("authorization" in fixture.libraryConnection, false);
  assert.equal(fixture.settings.embedding.mode, "local");
  // Without a configured embedding endpoint there is nothing to disclose, and
  // the consent dialog the scenario waits for would never open.
  assert.ok(fixture.settings.embedding.baseUrl.length > 0);
});

test("the connected-library fixture refuses an invented root identity", () => {
  assert.throws(
    () => connectedLibraryFixture({ libraryRoot: "/vault/library", rootIdentity: "not-an-inode" }),
    /dev:ino/,
  );
  assert.throws(
    () => connectedLibraryFixture({ libraryRoot: "library", rootIdentity: "66:1" }),
    /absolute/,
  );
});

test("the root identity is the folder's device and inode, as the plugin re-checks it", async () => {
  const identity = await readRootIdentity("/vault/library", {
    stat: async () => ({ isDirectory: () => true, dev: 66, ino: 1234 }),
  });
  assert.equal(identity, "66:1234");
  await assert.rejects(
    () => readRootIdentity("/vault/file", { stat: async () => ({ isDirectory: () => false }) }),
    /not a directory/,
  );
});

test("the library root is chosen from folders inside the vault that hold PDFs", async () => {
  const fs = {
    readdir: async (dir, options) => {
      if (options?.withFileTypes) {
        return [
          { name: ".obsidian", isDirectory: () => true },
          { name: "notes", isDirectory: () => true },
          { name: "papers", isDirectory: () => true },
        ];
      }
      return dir.endsWith("papers") ? ["a.pdf"] : ["note.md"];
    },
  };
  assert.equal(await resolveLibraryRoot({ vaultPath: "/vault", fs }), "/vault/papers");
});

test("a vault with no folder of PDFs is reported rather than silently pointed somewhere", async () => {
  const fs = {
    readdir: async (_dir, options) => (options?.withFileTypes ? [] : []),
  };
  await assert.rejects(() => resolveLibraryRoot({ vaultPath: "/vault", fs }), /no folder of PDFs/);
});
