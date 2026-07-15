import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const expected = process.argv[2];
if (!expected) {
  console.error("Usage: node scripts/check-release-version.mjs VERSION");
  process.exit(2);
}

const manifests = [
  "package.json",
  "manifest.json",
  "plugin/package.json",
  "plugin/manifest.json",
  "packages/core/package.json",
  "packages/node-runtime/package.json",
  "apps/cli/package.json",
];
const parsed = new Map();
const errors = [];
for (const file of manifests) {
  const value = JSON.parse(await readFile(resolve(root, file), "utf8"));
  parsed.set(file, value);
  if (value.version !== expected) errors.push(`${file} version ${value.version} does not match ${expected}`);
}
for (const file of ["versions.json", "plugin/versions.json"]) {
  const value = JSON.parse(await readFile(resolve(root, file), "utf8"));
  if (!value[expected]) errors.push(`${file} does not contain ${expected}`);
}
for (const [file, value] of parsed) {
  for (const field of ["dependencies", "devDependencies", "peerDependencies", "optionalDependencies"]) {
    for (const [name, range] of Object.entries(value[field] ?? {})) {
      if (name.startsWith("@arxiv-daily/") && range !== expected) {
        errors.push(`${file} ${field}.${name} is ${range}, expected ${expected}`);
      }
    }
  }
}
if (errors.length) {
  console.error(errors.join("\n"));
  process.exit(1);
}
console.log(`Release versions OK: ${expected}`);
