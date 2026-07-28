import { access, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { readJson, root, validateSemVer } from "./release-utils.mjs";

const expected = process.argv[2];
try {
  validateSemVer(expected);
} catch (error) {
  console.error("Usage: node scripts/check-release-version.mjs VERSION");
  console.error(error.message);
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
const parsed = new Map();
const errors = [];

for (const file of [...packageFiles, ...manifestFiles]) {
  const value = await readJson(file);
  parsed.set(file, value);
  if (value.version !== expected) errors.push(`${file} version ${value.version} does not match ${expected}`);
}

const rootPackage = parsed.get("package.json");
if (rootPackage.name !== "arxiv-daily-workspace" || rootPackage.private !== true) {
  errors.push("package.json must identify the private arxiv-daily-workspace");
}
if (JSON.stringify(rootPackage.workspaces) !== JSON.stringify(["packages/*", "apps/*", "plugin"])) {
  errors.push("package.json workspaces must be packages/*, apps/*, plugin in canonical order");
}
const publishablePackages = new Set(["apps/cli/package.json"]);
for (const file of packageFiles) {
  const value = parsed.get(file);
  if (publishablePackages.has(file)) {
    if (value.private === true) {
      errors.push(`${file} must be publishable (private must not be true)`);
    }
  } else if (value.private !== true) {
    errors.push(`${file} must be private`);
  }
  if (value.license !== "MIT") errors.push(`${file} license must be MIT`);
  if (value.engines?.node !== ">=20.11.0") errors.push(`${file} engines.node must be >=20.11.0`);
  if (value.repository?.type !== "git" || value.repository?.url !== "git+https://github.com/tdccccc/arxiv-daily.git") {
    errors.push(`${file} repository metadata is not canonical`);
  }
  for (const field of ["dependencies", "devDependencies", "peerDependencies", "optionalDependencies"]) {
    for (const [name, range] of Object.entries(value[field] ?? {})) {
      if (name.startsWith("@arxiv-daily/") && range !== expected) {
        errors.push(`${file} ${field}.${name} is ${range}, expected ${expected}`);
      }
    }
  }
}

const cliPackage = parsed.get("apps/cli/package.json");
if (cliPackage.name !== "arxiv-daily") {
  errors.push('apps/cli/package.json name must be "arxiv-daily"');
}
if (cliPackage.bin?.["arxiv-daily"] !== "./dist/arxiv-daily-cli.cjs") {
  errors.push("apps/cli/package.json bin.arxiv-daily must point at ./dist/arxiv-daily-cli.cjs");
}
const cliFiles = cliPackage.files ?? [];
if (!cliFiles.includes("dist/arxiv-daily-cli.cjs")) {
  errors.push("apps/cli/package.json files must include dist/arxiv-daily-cli.cjs");
}

const rootManifest = parsed.get("manifest.json");
const pluginManifest = parsed.get("plugin/manifest.json");
if (JSON.stringify(rootManifest) !== JSON.stringify(pluginManifest)) {
  errors.push("manifest.json and plugin/manifest.json must be identical");
}

const maps = [];
for (const file of ["versions.json", "plugin/versions.json"]) {
  const value = await readJson(file);
  maps.push(value);
  const keys = Object.keys(value);
  for (const key of keys) {
    try { validateSemVer(key); } catch { errors.push(`${file} contains invalid SemVer key ${key}`); }
    if (typeof value[key] !== "string" || !/^\d+\.\d+\.\d+$/.test(value[key])) {
      errors.push(`${file}.${key} has invalid minimum Obsidian version ${value[key]}`);
    }
  }
  if (keys.at(-1) !== expected) {
    errors.push(`${file} latest entry ${keys.at(-1)} does not match release ${expected}`);
  }
  if (value[expected] !== pluginManifest.minAppVersion) {
    errors.push(`${file}.${expected} must equal manifest minAppVersion ${pluginManifest.minAppVersion}`);
  }
}
if (JSON.stringify(maps[0]) !== JSON.stringify(maps[1])) {
  errors.push("versions.json and plugin/versions.json must be identical and ordered alike");
}

for (const file of ["plugin/package-lock.json", "apps/cli/package-lock.json", "packages/core/package-lock.json", "packages/node-runtime/package-lock.json"]) {
  try {
    await access(resolve(root, file));
    errors.push(`${file} must not exist; package-lock.json is authoritative`);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
}

const lock = await readJson("package-lock.json");
if (lock.name !== rootPackage.name) errors.push(`package-lock.json name ${lock.name} does not match package.json`);
if (lock.version !== expected) errors.push(`package-lock.json version ${lock.version} does not match ${expected}`);
if (lock.lockfileVersion !== 3) errors.push("package-lock.json lockfileVersion must be 3");
const workspacePaths = new Set(["", "apps/cli", "packages/core", "packages/node-runtime", "plugin"]);
for (const workspacePath of workspacePaths) {
  const value = lock.packages?.[workspacePath];
  if (!value) {
    errors.push(`package-lock.json is missing workspace ${workspacePath || "<root>"}`);
    continue;
  }
  if (value.version !== expected) {
    errors.push(`package-lock.json packages.${workspacePath || "<root>"}.version ${value.version} does not match ${expected}`);
  }
  const manifest = parsed.get(workspacePath ? `${workspacePath}/package.json` : "package.json");
  if (value.name !== manifest.name) errors.push(`package-lock.json ${workspacePath || "<root>"} name does not match its package.json`);
  for (const field of ["dependencies", "devDependencies", "optionalDependencies"]) {
    for (const [name, range] of Object.entries(value[field] ?? {})) {
      if (name.startsWith("@arxiv-daily/") && range !== expected) {
        errors.push(`package-lock.json ${workspacePath || "<root>"}.${field}.${name} is ${range}, expected ${expected}`);
      }
    }
  }
}
for (const path of Object.keys(lock.packages ?? {})) {
  if (/^(apps|packages)\//.test(path) && !path.startsWith("node_modules/") && !workspacePaths.has(path)) {
    errors.push(`package-lock.json contains unexpected workspace package ${path}`);
  }
}
const pako = lock.packages?.["node_modules/pako"];
if (pako?.version !== "2.2.0" || pako.resolved !== "https://registry.npmjs.org/pako/-/pako-2.2.0.tgz" || !pako.integrity) {
  errors.push("package-lock.json must lock pako 2.2.0 with canonical registry URL and integrity");
}
const notices = await readFile(resolve(root, "THIRD_PARTY_NOTICES.md"), "utf8");
if (!notices.includes("## pako 2.2.0")) errors.push("THIRD_PARTY_NOTICES.md must describe locked pako 2.2.0");

if (errors.length) {
  console.error(errors.join("\n"));
  process.exit(1);
}
console.log(`Release versions and metadata OK: ${expected}`);
