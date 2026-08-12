import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

export const root = resolve(import.meta.dirname, "..");

export const packageFiles = [
  "package.json",
  "plugin/package.json",
  "packages/core/package.json",
  "packages/node-runtime/package.json",
  "apps/cli/package.json",
];
export const manifestFiles = [
  "manifest.json",
  "plugin/manifest.json",
];

// SemVer 2.0.0, including the leading-zero rules for core and prerelease numbers.
export const SEMVER_PATTERN = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/;

export function validateSemVer(value) {
  if (typeof value !== "string" || !SEMVER_PATTERN.test(value)) {
    throw new Error(`Invalid SemVer 2.0.0 version: ${JSON.stringify(value)}`);
  }
  return value;
}

export async function readJson(relativePath) {
  return JSON.parse(await readFile(resolve(root, relativePath), "utf8"));
}

export async function readPakoNotice() {
  const notices = await readFile(resolve(root, "THIRD_PARTY_NOTICES.md"), "utf8");
  const match = notices.match(/## pako 2\.2\.0[\s\S]*?```text\n([\s\S]*?)\n```/);
  if (!match) throw new Error("THIRD_PARTY_NOTICES.md is missing the locked pako 2.2.0 notice");
  return match[1];
}

export function noticeBanner(notice) {
  return `/*!\nTHIRD-PARTY NOTICE: pako 2.2.0\n\n${notice}\n*/`;
}
