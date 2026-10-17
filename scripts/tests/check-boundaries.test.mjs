import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path, { dirname, join } from "node:path";
import test from "node:test";
import { PLUGIN_NODE_BUILTIN_ALLOWLIST, checkBoundaries, displayPath, isPluginProductionFile } from "../check-boundaries.mjs";
import { root as repositoryRoot } from "../release-utils.mjs";

/** A minimal tree containing every directory the checker walks. */
async function tree(files = {}) {
  const dir = await mkdtemp(join(tmpdir(), "boundaries-"));
  for (const layer of ["packages/core", "packages/node-runtime", "apps/cli", "plugin"]) {
    await mkdir(join(dir, layer), { recursive: true });
  }
  for (const [name, body] of Object.entries(files)) {
    await mkdir(dirname(join(dir, name)), { recursive: true });
    await writeFile(join(dir, name), body);
  }
  return { dir, cleanup: () => rm(dir, { recursive: true, force: true }) };
}

async function errorsFor(files) {
  const fixture = await tree(files);
  try { return await checkBoundaries(fixture.dir); } finally { await fixture.cleanup(); }
}

const NODE_FS = 'import fs from "node:fs";\nexport const x = fs;\n';

test("the repository itself satisfies the boundaries", async () => {
  assert.deepEqual(await checkBoundaries(repositoryRoot), []);
  const cli = execFileSync(process.execPath, [join(repositoryRoot, "scripts/check-boundaries.mjs")], { encoding: "utf8" });
  assert.match(cli, /Workspace boundaries OK/);
});

test("a Node builtin in plugin/main.ts is reported", async () => {
  assert.deepEqual(await errorsFor({ "plugin/main.ts": NODE_FS }), ["plugin/main.ts: plugin imports Node builtin node:fs"]);
});

test("a Node builtin in plugin/src is still reported", async () => {
  assert.deepEqual(await errorsFor({ "plugin/src/a.ts": NODE_FS }), ["plugin/src/a.ts: plugin imports Node builtin node:fs"]);
});

test("a new root-level plugin file cannot import a Node builtin", async () => {
  assert.deepEqual(await errorsFor({ "plugin/helper.ts": 'import os from "os";\nexport const h = os;\n' }), ["plugin/helper.ts: plugin imports Node builtin os"]);
});

test("the allowlisted desktop storage file may import only its listed builtins", async () => {
  assert.deepEqual(await errorsFor({
    "plugin/node-fs-exclusive.ts": 'import * as fs from "node:fs/promises";\nimport * as path from "node:path";\nimport { existsSync } from "node:fs";\nexport const a = [fs, path, existsSync];\n',
  }), []);
  assert.deepEqual(await errorsFor({
    "plugin/node-fs-exclusive.ts": 'import cp from "node:child_process";\nexport const a = cp;\n',
  }), ["plugin/node-fs-exclusive.ts: plugin imports Node builtin node:child_process"]);
  assert.ok(PLUGIN_NODE_BUILTIN_ALLOWLIST.has("plugin/node-fs-exclusive.ts"));
});

test("the allowlist is path-specific, not name-specific", async () => {
  assert.deepEqual(await errorsFor({ "plugin/src/node-fs-exclusive.ts": NODE_FS }), ["plugin/src/node-fs-exclusive.ts: plugin imports Node builtin node:fs"]);
});

test("plugin tests and tool configs may use Node builtins", async () => {
  assert.deepEqual(await errorsFor({
    "plugin/tests/a.test.ts": NODE_FS,
    "plugin/esbuild.config.mjs": NODE_FS,
    "plugin/vitest.config.mts": NODE_FS,
  }), []);
});

test("core rules still apply under packages/core/src only", async () => {
  assert.deepEqual((await errorsFor({ "packages/core/src/a.ts": NODE_FS })).sort(), [
    "packages/core/src/a.ts: core imports Node builtin node:fs",
    "packages/core/src/a.ts: core third-party dependency is not allowlisted: node:fs",
  ]);
  assert.deepEqual(await errorsFor({ "packages/core/tests/a.test.ts": NODE_FS }), []);
  assert.deepEqual(await errorsFor({ "packages/core/src/b.ts": "export const p = process.env;\n" }), ["packages/core/src/b.ts: core uses forbidden Node global"]);
});

test("Windows-style paths are normalised before the path rules run", () => {
  assert.equal(displayPath("C:\\repo", "C:\\repo\\plugin\\main.ts", path.win32), "plugin/main.ts");
  assert.equal(displayPath("C:\\repo", "C:\\repo\\packages\\core\\src\\a.ts", path.win32), "packages/core/src/a.ts");
  assert.equal(isPluginProductionFile(displayPath("C:\\repo", "C:\\repo\\plugin\\main.ts", path.win32)), true);
  assert.equal(isPluginProductionFile(displayPath("C:\\repo", "C:\\repo\\plugin\\tests\\a.test.ts", path.win32)), false);
});

test("isPluginProductionFile classifies plugin paths", () => {
  for (const file of ["plugin/main.ts", "plugin/src/a/b.ts", "plugin/node-fs-exclusive.ts"]) assert.equal(isPluginProductionFile(file), true, file);
  for (const file of ["plugin/tests/a.ts", "plugin/esbuild.config.mjs", "plugin/vitest.config.mts", "apps/cli/src/a.ts", "plugin/types.d.ts"]) assert.equal(isPluginProductionFile(file), false, file);
});
