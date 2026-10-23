import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import { join } from "node:path";
import { RELEASE_ASSETS } from "./release-assets.mjs";

export async function artifactFingerprint(file, bytes, storage = fs) {
  const info = await storage.stat?.(file);
  return { path: file, sha256: createHash("sha256").update(bytes).digest("hex"), mtimeMs: info?.mtimeMs ?? null };
}

async function newestSource(directory) {
  let entries;
  try { entries = await fs.readdir(directory, { withFileTypes: true }); }
  catch (error) { if (error.code === "ENOENT") return 0; throw error; }
  const times = await Promise.all(entries.map(async entry => entry.isDirectory()
    ? newestSource(join(directory, entry.name))
    : (await fs.stat(join(directory, entry.name))).mtimeMs));
  return Math.max(0, ...times);
}

/** Identify existing outputs even with --skip-build; missing artifacts are explicit evidence. */
export async function collectBuildEvidence(root, workspaces) {
  const files = [];
  if (workspaces.includes("apps/cli")) files.push("apps/cli/dist/arxiv-daily-cli.cjs");
  if (workspaces.includes("obsidian-arxiv-daily")) files.push(...RELEASE_ASSETS.map(name => `plugin/${name}`));
  const buildArtifacts = await Promise.all(files.map(async file => {
    try { return { ...await artifactFingerprint(join(root, file), await fs.readFile(join(root, file))), path: file }; }
    catch (error) { if (error.code === "ENOENT") return { path: file, missing: true }; throw error; }
  }));
  const buildWarnings = [];
  for (const [output, sources] of [
    ["plugin/main.js", ["plugin/src", "packages/core/src"]],
    ["apps/cli/dist/arxiv-daily-cli.cjs", ["apps/cli/src", "packages/core/src", "packages/node-runtime/src"]],
  ]) {
    const artifact = buildArtifacts.find(item => item.path === output);
    if (!artifact || artifact.missing) continue;
    let latest = Math.max(...await Promise.all(sources.map(source => newestSource(join(root, source)))));
    if (output === "plugin/main.js") {
      try { latest = Math.max(latest, (await fs.stat(join(root, "plugin/main.ts"))).mtimeMs); }
      catch (error) { if (error.code !== "ENOENT") throw error; }
    }
    if (artifact.mtimeMs < latest) buildWarnings.push(`${output} is older than source files; rebuild before accepting release evidence`);
  }
  return { buildArtifacts, buildWarnings };
}
