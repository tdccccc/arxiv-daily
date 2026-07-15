import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const version = process.argv[2];
if (!version || !/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version)) {
  console.error("Usage: node scripts/sync-release-version.mjs VERSION");
  process.exit(2);
}

const packageFiles = [
  "package.json",
  "plugin/package.json",
  "packages/core/package.json",
  "packages/node-runtime/package.json",
  "apps/cli/package.json",
];
const manifestFiles = ["manifest.json", "plugin/manifest.json"];

for (const file of [...packageFiles, ...manifestFiles]) {
  const path = resolve(root, file);
  const value = JSON.parse(await readFile(path, "utf8"));
  value.version = version;
  for (const field of [
    "dependencies",
    "devDependencies",
    "peerDependencies",
    "optionalDependencies",
  ]) {
    for (const name of Object.keys(value[field] ?? {})) {
      if (name.startsWith("@arxiv-daily/")) value[field][name] = version;
    }
  }
  await writeJson(path, value);
}

const minAppVersion = JSON.parse(
  await readFile(resolve(root, "plugin/manifest.json"), "utf8"),
).minAppVersion;
for (const file of ["versions.json", "plugin/versions.json"]) {
  const path = resolve(root, file);
  const value = JSON.parse(await readFile(path, "utf8"));
  value[version] = minAppVersion;
  await writeJson(path, value);
}

const lockPath = resolve(root, "package-lock.json");
const lock = JSON.parse(await readFile(lockPath, "utf8"));
lock.version = version;
for (const [workspacePath, value] of Object.entries(lock.packages ?? {})) {
  if (
    workspacePath === "" ||
    workspacePath === "plugin" ||
    workspacePath.startsWith("apps/") ||
    workspacePath.startsWith("packages/")
  ) {
    value.version = version;
  }
  for (const field of ["dependencies", "devDependencies", "optionalDependencies"]) {
    for (const name of Object.keys(value[field] ?? {})) {
      if (name.startsWith("@arxiv-daily/")) value[field][name] = version;
    }
  }
}
await writeJson(lockPath, lock);

console.log(`Release versions synchronized: ${version}`);

async function writeJson(path, value) {
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`);
}
