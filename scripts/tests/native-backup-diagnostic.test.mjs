import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { test } from "node:test";
import { parse } from "yaml";

const require = createRequire(import.meta.url);
const { runBackupDiagnostics } = require("../../packages/node-runtime/native/tests/backup-diagnostic.cjs");

function fakeBinding(fault) {
  const roots = [];
  return {
    roots,
    binding: {
      openDirectory(root, relative) {
        roots.push(root);
        const parent = path.join(root, relative);
        fs.mkdirSync(parent, { mode: 0o700 });
        return {
          createFile(name) {
            const filename = path.join(parent, name);
            const fd = fs.openSync(filename, "wx+", 0o600);
            return {
              writeText(text) {
                const value = fault === `write-${name}` ? Buffer.alloc(text.length) : Buffer.from(text);
                fs.writeSync(fd, value, 0, value.length, 0);
                fs.ftruncateSync(fd, value.length);
                fs.fsyncSync(fd);
              },
              close() {
                if (fault === `close-${name}`) fs.writeSync(fd, Buffer.alloc(3), 0, 3, 0);
                fs.closeSync(fd);
              },
            };
          },
          link(from, to) {
            if (fs.existsSync(path.join(parent, to))) return false;
            if (fault === "copied-backup") fs.copyFileSync(path.join(parent, from), path.join(parent, to));
            else fs.linkSync(path.join(parent, from), path.join(parent, to));
            return true;
          },
          rename(from, to) {
            fs.renameSync(path.join(parent, from), path.join(parent, to));
            if (fault === "replaced-backup") {
              fs.renameSync(path.join(parent, "primary.bak"), path.join(parent, "kept-old-inode"));
              fs.writeFileSync(path.join(parent, "primary.bak"), "old");
            }
          },
          sync() {},
          close() {},
        };
      },
    },
  };
}

test("reports bounded successful control modes and cleans only its own fixtures", () => {
  const { binding, roots } = fakeBinding();
  const unrelated = fs.mkdtempSync(path.join(os.tmpdir(), "arxiv-backup-unrelated-"));
  try {
    fs.writeFileSync(path.join(unrelated, "keep"), "untouched");
    const result = runBackupDiagnostics({ binding, iterations: 2 });
    assert.equal(result.ok, true);
    assert.deepEqual(result.modes.map(({ name }) => name), process.platform === "win32"
      ? ["native-final-only", "native-staged"]
      : ["native-final-only", "native-staged", "node-positioned-write", "node-without-truncate"]);
    for (const mode of result.modes) {
      assert.equal(mode.completed, 2);
      assert.equal(mode.attempted, 2);
      assert.equal(mode.failure, null);
    }
    assert.equal(result.platform, process.platform);
    assert.equal(result.arch, process.arch);
    assert.equal(result.node, process.versions.node);
    assert.equal(result.os.version, os.version());
    assert.equal(typeof result.filesystem.type, "string");
    assert.ok(result.nativeBuild && typeof result.nativeBuild === "object");
    assert.equal(new Set(roots).size, 4);
    assert.ok(roots.every((root) => !fs.existsSync(root)));
    assert.equal(fs.readFileSync(path.join(unrelated, "keep"), "utf8"), "untouched");
  } finally { fs.rmSync(unrelated, { recursive: true, force: true }); }
});

for (const fault of ["write-primary.json", "close-primary.json", "write-next.tmp"]) {
  test(`reports the first ${fault} corruption without retrying the failed mode`, () => {
    const { binding, roots } = fakeBinding(fault);
    const result = runBackupDiagnostics({ binding, iterations: 2 });
    assert.equal(result.ok, false);
    const staged = result.modes.find(({ name }) => name === "native-staged");
    assert.equal(staged.completed, 0);
    assert.equal(staged.attempted, 1);
    assert.equal(staged.failure.stage, fault);
    assert.equal(staged.failure.iteration, 0);
    const filename = fault.endsWith("next.tmp") ? "next.tmp" : "primary.json";
    const record = staged.failure.snapshots.at(-1).files[filename];
    assert.equal(record.hex, "000000");
    for (const field of ["size", "dev", "ino", "nlink"]) assert.equal(typeof record[field], "string");
    assert.equal(result.modes[0].failure.stage, "directory-sync");
    for (const control of result.modes.filter(({ name }) => name.startsWith("node-"))) {
      assert.equal(control.completed, 2);
      assert.equal(control.failure, null);
    }
    assert.ok(roots.every((root) => !fs.existsSync(root)));
  });
}

for (const [fault, stage] of [["copied-backup", "first-link"], ["replaced-backup", "rename"]]) {
  test(`detects ${fault} even when backup text still reads old`, () => {
    const result = runBackupDiagnostics({ binding: fakeBinding(fault).binding, iterations: 1 });
    const failure = result.modes.find(({ name }) => name === "native-staged").failure;
    assert.equal(result.ok, false);
    assert.equal(failure.stage, stage);
    assert.equal(failure.snapshots.at(-1).files["primary.bak"].hex, "6f6c64");
  });
}

test("rejects invalid or unbounded iteration requests before making fixtures", () => {
  const { binding, roots } = fakeBinding();
  for (const iterations of [0, -1, 1.5, 101, Infinity]) {
    assert.throws(() => runBackupDiagnostics({ binding, iterations }), /iterations/);
  }
  assert.deepEqual(roots, []);
});

test("runs backup diagnostics after a failed native test and always archives its report", () => {
  const workflow = parse(fs.readFileSync(new URL("../../.github/workflows/native-storage.yml", import.meta.url), "utf8"));
  const steps = workflow.jobs.build.steps;
  const build = steps.find((step) => step.name === "Build native support");
  assert.equal(build.id, "native_build");
  const baseIndex = steps.findIndex((step) => step.name === "Test native filesystem capability");
  const diagnostic = steps[baseIndex + 1];
  assert.equal(diagnostic.name, "Diagnose native hard-link backup integrity");
  assert.match(diagnostic.if, /!cancelled\(\).*steps\.native_build\.outcome\s*==\s*'success'/);
  assert.match(diagnostic.run, /backup-diagnostic\.cjs/);
  assert.match(diagnostic.run, /backup-diagnostic\.json/);
  assert.notEqual(diagnostic["continue-on-error"], true);
  const evidence = steps.find((step) => step.name === "Upload native verification evidence");
  assert.equal(evidence.if, "always()");
  assert.match(evidence.with.path, /backup-diagnostic\.json/);
});
