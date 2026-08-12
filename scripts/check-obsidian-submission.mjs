#!/usr/bin/env node
// Pre-publish check mirroring Obsidian's community-directory submission
// requirements against the built plugin assets. Read-only; run after build.
//
// Sources for the checks:
// - https://docs.obsidian.md/plugins/releasing/submit-plugin (directory submission)
// - Submission requirements for plugins (community directory)
// - Plugin guidelines (review recommendations marked as required)
// - Historical obsidian-releases verification (1 MB bundle limit)

import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const OBSIDIAN_BUNDLE_LIMIT_BYTES = 1024 * 1024;
export const OBSIDIAN_DESCRIPTION_LIMIT = 250;
export const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const EMOJI_RE = /[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}]/u;
const SEMVER_RE = /^\d+\.\d+\.\d+$/;
const URL_RE = /^https?:\/\/\S+$/;

export function checkManifest(manifest) {
  const issues = [];
  if (typeof manifest !== "object" || manifest === null || Array.isArray(manifest)) {
    return ["manifest.json must be a JSON object"];
  }
  const { id, name, version, minAppVersion, description, isDesktopOnly, fundingUrl } = manifest;
  if (typeof id !== "string" || id.trim() === "") {
    issues.push("manifest id must be a non-empty string");
  } else if (/obsidian/i.test(id)) {
    issues.push("manifest id must not contain \"obsidian\"");
  }
  if (typeof name !== "string" || name.trim() === "") {
    issues.push("manifest name must be a non-empty string");
  }
  if (typeof version !== "string" || !SEMVER_RE.test(version)) {
    issues.push(`manifest version must be stable semver x.y.z, got ${JSON.stringify(version)}`);
  }
  if (typeof minAppVersion !== "string" || !SEMVER_RE.test(minAppVersion)) {
    issues.push(`manifest minAppVersion must be x.y.z, got ${JSON.stringify(minAppVersion)}`);
  }
  if (typeof description !== "string") {
    issues.push("manifest description must be a string");
  } else {
    if (description.length > OBSIDIAN_DESCRIPTION_LIMIT) {
      issues.push(`manifest description exceeds ${OBSIDIAN_DESCRIPTION_LIMIT} characters (${description.length})`);
    }
    if (!description.endsWith(".")) {
      issues.push("manifest description must end with a period");
    }
    if (EMOJI_RE.test(description)) {
      issues.push("manifest description must avoid emoji or special characters");
    }
  }
  if (typeof isDesktopOnly !== "boolean") {
    issues.push("manifest isDesktopOnly must be a boolean (Node/Electron APIs require true)");
  }
  if (fundingUrl !== undefined && (typeof fundingUrl !== "string" || !URL_RE.test(fundingUrl))) {
    issues.push("manifest fundingUrl, if present, must be an http(s) URL");
  }
  return issues;
}

export function checkBundle(bundleText) {
  const issues = [];
  const bytes = Buffer.byteLength(bundleText, "utf8");
  if (bytes > OBSIDIAN_BUNDLE_LIMIT_BYTES) {
    issues.push(`main.js exceeds ${OBSIDIAN_BUNDLE_LIMIT_BYTES} bytes (${bytes})`);
  }
  for (const forbidden of [
    "eval(",
    "new Function",
    "Function(",
    "require('electron')",
    'require("electron")',
    "window.require",
    "process.getBuiltinModule",
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "window.app",
  ]) {
    if (bundleText.includes(forbidden)) {
      issues.push(`main.js contains forbidden pattern ${JSON.stringify(forbidden)}`);
    }
  }
  return issues;
}

export function checkRepoFiles(files) {
  const issues = [];
  for (const required of ["README.md", "LICENSE"]) {
    if (!files.includes(required)) {
      issues.push(`${required} must exist in the repository root for directory submission`);
    }
  }
  return issues;
}

export async function main() {
  const all = [];
  let manifest;
  try {
    manifest = JSON.parse(readFileSync(join(root, "plugin/manifest.json"), "utf8"));
  } catch (error) {
    all.push(`FAIL manifest — cannot read or parse plugin/manifest.json: ${error.message}`);
    manifest = null;
  }
  if (manifest !== null) {
    for (const issue of checkManifest(manifest)) all.push(`FAIL manifest — ${issue}`);
  }

  const bundlePath = join(root, "plugin/main.js");
  if (!existsSync(bundlePath)) {
    all.push("FAIL bundle — plugin/main.js missing; run the build first");
  } else {
    for (const issue of checkBundle(readFileSync(bundlePath, "utf8"))) {
      all.push(`FAIL bundle — ${issue}`);
    }
  }

  const repoFiles = ["README.md", "LICENSE"].filter((f) => existsSync(join(root, f)));
  for (const issue of checkRepoFiles(repoFiles)) all.push(`FAIL repo — ${issue}`);

  if (all.length === 0) {
    console.log("Obsidian submission check: PASS");
    process.exit(0);
  }
  for (const line of all) console.error(line);
  process.exit(1);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
