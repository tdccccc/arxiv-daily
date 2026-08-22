import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import {
  checkBundle,
  checkManifest,
  checkRepoFiles,
  OBSIDIAN_BUNDLE_BUDGET_BYTES,
  OBSIDIAN_DESCRIPTION_LIMIT,
  root,
} from "../check-obsidian-submission.mjs";

const validManifest = {
  id: "arxiv-daily",
  name: "arXiv Daily",
  version: "0.4.1",
  minAppVersion: "1.4.0",
  description: "Daily arXiv paper tracker with LLM-filtered summaries.",
  isDesktopOnly: true,
};

test("checkManifest accepts a valid manifest", () => {
  assert.deepEqual(checkManifest(validManifest), []);
});

test("checkManifest rejects an id containing obsidian", () => {
  assert.ok(
    checkManifest({ ...validManifest, id: "obsidian-arxiv-daily" }).some(
      (i) => i.includes("must not contain"),
    ),
  );
});

test("checkManifest rejects non-semver and prerelease versions", () => {
  for (const version of ["v0.4.1", "0.4", "0.4.1-beta", ""]) {
    assert.ok(
      checkManifest({ ...validManifest, version }).some((i) =>
        i.includes("stable semver"),
      ),
    );
  }
});

test("checkManifest rejects a missing or malformed minAppVersion", () => {
  assert.ok(checkManifest({ ...validManifest, minAppVersion: "latest" }).length > 0);
  assert.ok(checkManifest({ ...validManifest, minAppVersion: undefined }).length > 0);
});

test("checkManifest enforces description length, trailing period, and no emoji", () => {
  const tooLong = `${"a".repeat(OBSIDIAN_DESCRIPTION_LIMIT + 1)}.`;
  assert.ok(checkManifest({ ...validManifest, description: tooLong }).length > 0);
  assert.ok(checkManifest({ ...validManifest, description: "no period" }).length > 0);
  assert.ok(checkManifest({ ...validManifest, description: "Has emoji \u{1F600}." }).length > 0);
});

test("checkManifest requires a boolean isDesktopOnly", () => {
  assert.ok(checkManifest({ ...validManifest, isDesktopOnly: "true" }).length > 0);
  assert.ok(checkManifest({ ...validManifest, isDesktopOnly: undefined }).length > 0);
});

test("checkManifest rejects a malformed fundingUrl when present", () => {
  assert.deepEqual(checkManifest({ ...validManifest, fundingUrl: "https://sponsor.example" }), []);
  assert.ok(checkManifest({ ...validManifest, fundingUrl: "not a url" }).length > 0);
});

test("checkManifest rejects non-object input", () => {
  assert.ok(checkManifest(null).length > 0);
  assert.ok(checkManifest("[]").length > 0);
});

test("checkBundle accepts a clean bundle under the size limit", () => {
  const clean = 'console.log("plugin ok");';
  assert.deepEqual(checkBundle(clean), []);
});

test("checkBundle flags eval, Function constructors, and Electron requires", () => {
  for (const forbidden of [
    "eval(payload)",
    "new Function('return 1')",
    "Function('return 1')",
    "require('electron')",
    'require("electron")',
    "window.require('x')",
  ]) {
    assert.ok(checkBundle(forbidden).length > 0, `should flag ${forbidden}`);
  }
});

test("checkBundle flags unsafe DOM injection and global app access", () => {
  for (const forbidden of ["innerHTML", "outerHTML", "insertAdjacentHTML", "window.app"]) {
    assert.ok(checkBundle(forbidden).length > 0, `should flag ${forbidden}`);
  }
});

test("checkBundle flags an oversized bundle", () => {
  const big = "x".repeat(OBSIDIAN_BUNDLE_BUDGET_BYTES + 1);
  assert.ok(checkBundle(big).some((i) => i.includes("exceeds")));
});

test("checkRepoFiles requires README and LICENSE", () => {
  assert.deepEqual(checkRepoFiles(["README.md", "LICENSE"]), []);
  assert.ok(checkRepoFiles(["README.md"]).some((i) => i.includes("LICENSE")));
  assert.ok(checkRepoFiles([]).length === 2);
});

test("current built manifest passes the manifest checks", () => {
  const manifest = JSON.parse(readFileSync(join(root, "plugin/manifest.json"), "utf8"));
  assert.deepEqual(checkManifest(manifest), []);
});
