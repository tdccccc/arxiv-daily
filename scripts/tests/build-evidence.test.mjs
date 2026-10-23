import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, writeFile, utimes, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { collectBuildEvidence } from "../build-evidence.mjs";

test("fingerprints existing bytes and warns about a plugin bundle older than source", async t => {
  const root = await mkdtemp(join(tmpdir(), "build-evidence-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, "plugin/src"), { recursive: true });
  await writeFile(join(root, "plugin/main.js"), "old bundle");
  await utimes(join(root, "plugin/main.js"), new Date(0), new Date(0));
  await writeFile(join(root, "plugin/src/new.ts"), "new source");
  const evidence = await collectBuildEvidence(root, ["obsidian-arxiv-daily"]);
  assert.equal(evidence.buildArtifacts[0].sha256, createHash("sha256").update("old bundle").digest("hex"));
  assert.equal(evidence.buildArtifacts[0].mtimeMs, 0);
  assert.equal(evidence.buildArtifacts[1].missing, true);
  assert.match(evidence.buildWarnings.join("\n"), /plugin\/main.js is older/);
});
