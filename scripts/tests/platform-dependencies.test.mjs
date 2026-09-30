import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const { packages } = JSON.parse(readFileSync(new URL("../../package-lock.json", import.meta.url), "utf8"));

test("the workspace lockfile retains every build-tool platform binary", () => {
  let checked = 0;
  for (const [path, entry] of Object.entries(packages)) {
    for (const [name, version] of Object.entries(entry.optionalDependencies ?? {})) {
      if (!/^@(rollup\/rollup-|esbuild\/|rolldown\/binding-)/.test(name)) continue;
      let scope = path;
      let binary;
      while (!binary) {
        binary = packages[`${scope ? `${scope}/` : ""}node_modules/${name}`];
        if (!scope) break;
        const parent = scope.lastIndexOf("/node_modules/");
        scope = parent < 0 ? "" : scope.slice(0, parent);
      }
      assert.ok(binary, `${path} is missing optional platform binary ${name}`);
      assert.equal(binary.version, version, `${path}: ${name} must match its build tool`);
      assert.ok(binary.integrity, `${name} must have a verified archive`);
      checked++;
    }
  }
  assert.ok(checked > 0, "must check the installed build-tool platform families");
});
