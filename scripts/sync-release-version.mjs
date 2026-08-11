import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  manifestFiles,
  packageFiles,
  root,
  validateSemVer,
} from "./release-utils.mjs";

const version = process.argv[2];
try {
  validateSemVer(version);
} catch (error) {
  console.error("Usage: node scripts/sync-release-version.mjs VERSION");
  console.error(error.message);
  process.exit(2);
}

for (const file of [...packageFiles, ...manifestFiles]) {
  const path = resolve(root, file);
  const value = JSON.parse(await readFile(path, "utf8"));
  value.version = version;
  for (const field of ["dependencies", "devDependencies", "peerDependencies", "optionalDependencies"]) {
    for (const name of Object.keys(value[field] ?? {})) {
      if (name.startsWith("@arxiv-daily/")) value[field][name] = version;
    }
  }
  await writeJson(path, value);
}

const minAppVersion = JSON.parse(await readFile(resolve(root, "plugin/manifest.json"), "utf8")).minAppVersion;
const versionsPath = resolve(root, "versions.json");
const versions = JSON.parse(await readFile(versionsPath, "utf8"));
versions[version] = minAppVersion;
await writeJson(versionsPath, versions);
await writeJson(resolve(root, "plugin/versions.json"), versions);

const lockPath = resolve(root, "package-lock.json");
const lock = JSON.parse(await readFile(lockPath, "utf8"));
lock.version = version;
const workspacePaths = new Set(["", "apps/cli", "packages/core", "packages/node-runtime", "plugin"]);
for (const [workspacePath, value] of Object.entries(lock.packages ?? {})) {
  if (workspacePaths.has(workspacePath)) value.version = version;
  for (const field of ["dependencies", "devDependencies", "optionalDependencies"]) {
    for (const name of Object.keys(value[field] ?? {})) {
      if (name.startsWith("@arxiv-daily/")) value[field][name] = version;
    }
  }
}
await writeJson(lockPath, lock);

console.log(`Release versions synchronized: ${version}`);
console.log(`Run "npm run check:release-version -- ${version}" and review the diff before committing.`);

async function writeJson(path, value) {
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`);
}
