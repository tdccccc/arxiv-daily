import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import { buildWorkbenchAssets } from "../../apps/cli/build-workbench.mjs";
import { root } from "../release-utils.mjs";

test("built math stylesheet resolves every font to bundled bytes", async () => {
  const { assets, notices } = await buildWorkbenchAssets(path.join(root, "apps/cli"));
  const fonts = [...assets["katex.css"].body.matchAll(/url\(["']?(fonts\/[^)'"\s]+)/g)].map(match => match[1]);
  assert.ok(fonts.length > 0);
  for (const name of fonts) {
    assert.ok(assets[name], `Missing bundled font ${name}`);
    assert.equal(assets[name].encoding, "base64");
    const bytes = Buffer.from(assets[name].body, "base64");
    assert.ok(bytes.length > 1000, `${name} must contain real font data`);
    if (name.endsWith(".woff2")) assert.equal(bytes.subarray(0, 4).toString(), "wOF2");
  }
  assert.match(notices, /katex 0\.19\.0/);
});
