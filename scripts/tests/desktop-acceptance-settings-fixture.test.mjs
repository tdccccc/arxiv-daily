import assert from "node:assert/strict";
import test from "node:test";
import { installSettingsFixture, legacySettingsFixture } from "../desktop-acceptance/settings-fixture.mjs";

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
