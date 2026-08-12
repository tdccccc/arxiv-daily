#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { parse as parseYaml } from "yaml";
import { root as repositoryRoot } from "./release-utils.mjs";

const CANONICAL_SCAN_ROOTS = ["packages", "apps", "plugin", "services", "extensions"];
const CANONICAL_ROOT_WORKSPACES = ["packages/*", "apps/*", "plugin"];
const CANONICAL_RELEASE_CONTRACT = "scripts/release-utils.mjs";
const CANONICAL_RELEASE_MANIFESTS = ["manifest.json", "plugin/manifest.json"];
const CANONICAL_GOVERNANCE_PATHS = [
  "product-units.json",
  "scripts/check-product-units.mjs",
  "scripts/tests/product-units.test.mjs",
  "scripts/tests/independent-product-workflows.test.mjs",
];
const REQUIRED_CLASSIFICATIONS = new Map([
  ["root-release-group", "root-release-group"],
  ["email-relay", "independent-service"],
  ["vscode-companion", "independent-extension"],
]);
const CANONICAL_ROOT_WORKFLOWS = [
  ".github/workflows/lint.yml",
  ".github/workflows/release.yml",
  ".github/workflows/publish-cli.yml",
];
const IGNORED_DIRECTORY_NAMES = new Set([
  ".git",
  ".cache",
  "coverage",
  "dist",
  "node_modules",
]);

function normalizeRelativePath(relativePath) {
  return relativePath.split(path.sep).join("/").replace(/^\.\//, "");
}

function allDeclaredFiles(inventory) {
  const files = new Set(["package.json", ...inventory.governancePaths]);
  for (const unit of inventory.units) {
    for (const relativePath of [
      ...unit.manifests,
      ...unit.lockfiles,
      ...(unit.dependencies ?? []),
      ...(unit.compatibilityInputs ?? []),
      ...(unit.versionPolicy.releaseTools ?? []),
      ...(unit.versionPolicy.releaseContract
        ? [unit.versionPolicy.releaseContract]
        : []),
    ]) {
      files.add(relativePath);
    }
    for (const workflow of unit.workflows) files.add(workflow.path);
  }
  return [...files].sort();
}

async function scanPackageManifests(absoluteRoot, relativeRoot, result) {
  let entries;
  try {
    entries = await readdir(absoluteRoot, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") return;
    throw error;
  }

  for (const entry of entries) {
    const relativePath = normalizeRelativePath(path.join(relativeRoot, entry.name));
    const absolutePath = path.join(absoluteRoot, entry.name);
    if (entry.isDirectory()) {
      if (!IGNORED_DIRECTORY_NAMES.has(entry.name)) {
        await scanPackageManifests(absolutePath, relativePath, result);
      }
    } else if (entry.isFile() && entry.name === "package.json") {
      result.push(relativePath);
    }
  }
}

async function readSnapshotFile(root, relativePath) {
  try {
    return { exists: true, text: await readFile(path.join(root, relativePath), "utf8") };
  } catch (error) {
    if (error.code === "ENOENT") return { exists: false, text: null };
    throw error;
  }
}

export async function loadProductUnitSnapshot(root, inventory) {
  const manifestPaths = [];
  const rootManifest = await readSnapshotFile(root, "package.json");
  if (rootManifest.exists) manifestPaths.push("package.json");
  for (const scanRoot of inventory.scanRoots ?? []) {
    await scanPackageManifests(path.join(root, scanRoot), scanRoot, manifestPaths);
  }

  const files = {};
  for (const relativePath of allDeclaredFiles(inventory)) {
    files[relativePath] = await readSnapshotFile(root, relativePath);
  }

  const releaseContracts = {};
  for (const unit of inventory.units ?? []) {
    const relativePath = unit.versionPolicy?.releaseContract;
    const file = relativePath && files[relativePath];
    if (!file?.exists || releaseContracts[relativePath]) continue;
    try {
      const digest = createHash("sha256").update(file.text).digest("hex");
      const moduleUrl = `${pathToFileURL(path.join(root, relativePath)).href}?snapshot=${digest}`;
      const contract = await import(moduleUrl);
      releaseContracts[relativePath] = {
        packageFiles: contract.packageFiles,
        manifestFiles: contract.manifestFiles,
      };
    } catch (error) {
      releaseContracts[relativePath] = { error: error.message };
    }
  }

  return {
    manifestPaths: [...new Set(manifestPaths)].sort(),
    files,
    releaseContracts,
  };
}

function parseJsonFile(snapshot, relativePath, errors) {
  const file = snapshot.files[relativePath];
  if (!file?.exists) return undefined;
  try {
    return JSON.parse(file.text);
  } catch (error) {
    errors.push(`${relativePath} is not valid JSON: ${error.message}`);
    return undefined;
  }
}

function parseWorkflowFile(snapshot, relativePath, errors) {
  const file = snapshot.files[relativePath];
  if (!file?.exists) return undefined;
  try {
    return parseYaml(file.text, { schema: "core", uniqueKeys: true });
  } catch (error) {
    errors.push(`${relativePath} is not valid unique-key YAML: ${error.message}`);
    return undefined;
  }
}

function equalArrays(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function workflowPaths(workflow, event) {
  const value = workflow?.on?.[event];
  return value && typeof value === "object" && !Array.isArray(value)
    ? value.paths
    : undefined;
}

function workspacePathForManifest(manifestPath) {
  return manifestPath === "package.json"
    ? ""
    : manifestPath.replace(/\/package\.json$/, "");
}

function validateInventorySchema(inventory, errors) {
  if (inventory?.schemaVersion !== 1) errors.push("inventory schemaVersion must be 1");
  if (!equalArrays(inventory?.scanRoots, CANONICAL_SCAN_ROOTS)) {
    errors.push(`scanRoots must exactly match ${CANONICAL_SCAN_ROOTS.join(", ")}`);
  }
  if (!equalArrays(inventory?.governancePaths, CANONICAL_GOVERNANCE_PATHS)) {
    errors.push(
      `inventory governancePaths must exactly match ${CANONICAL_GOVERNANCE_PATHS.join(", ")}`,
    );
  }
  if (!Array.isArray(inventory?.units) || inventory.units.length === 0) {
    errors.push("inventory units must be a non-empty array");
    return;
  }

  const ids = new Set();
  const roots = new Set();
  const manifests = new Set();
  for (const unit of inventory.units) {
    if (typeof unit.id !== "string" || !unit.id) errors.push("every unit must have an id");
    if (ids.has(unit.id)) errors.push(`duplicate unit id ${unit.id}`);
    ids.add(unit.id);
    if (!Array.isArray(unit.roots) || unit.roots.length === 0) {
      errors.push(`${unit.id} must declare product roots`);
    }
    if (!Array.isArray(unit.manifests) || unit.manifests.length === 0) {
      errors.push(`${unit.id} must declare manifests`);
    }
    if (!Array.isArray(unit.lockfiles) || unit.lockfiles.length !== 1) {
      errors.push(`${unit.id} must declare exactly one authoritative lockfile`);
    }
    if (!Array.isArray(unit.workflows) || unit.workflows.length === 0) {
      errors.push(`${unit.id} must declare at least one workflow`);
    }
    for (const productRoot of unit.roots ?? []) {
      if (roots.has(productRoot)) errors.push(`product root ${productRoot} is classified more than once`);
      roots.add(productRoot);
    }
    for (const manifest of unit.manifests ?? []) {
      if (manifests.has(manifest)) errors.push(`manifest ${manifest} is classified more than once`);
      manifests.add(manifest);
    }
  }
  for (const [id, classification] of REQUIRED_CLASSIFICATIONS) {
    const unit = inventory.units.find((candidate) => candidate.id === id);
    if (!unit || unit.classification !== classification) {
      errors.push(`${id} must be classified as ${classification}`);
    }
  }
}

function validateClassifiedFiles(inventory, snapshot, errors) {
  for (const relativePath of allDeclaredFiles(inventory)) {
    if (!snapshot.files[relativePath]?.exists) {
      errors.push(`missing classified file ${relativePath}`);
    }
  }

  const classifiedManifests = new Set(
    inventory.units.flatMap((unit) => unit.manifests ?? []),
  );
  for (const manifestPath of snapshot.manifestPaths) {
    if (!classifiedManifests.has(manifestPath)) {
      errors.push(`unclassified manifest ${manifestPath}`);
    }
  }
}

function validateRootReleaseGroup(inventory, snapshot, errors) {
  const rootUnits = inventory.units.filter(
    (unit) => unit.classification === "root-release-group",
  );
  if (rootUnits.length !== 1) {
    errors.push("inventory must contain exactly one root release group");
    return;
  }
  const unit = rootUnits[0];
  if (!equalArrays(unit.roots, [".", ...CANONICAL_ROOT_WORKSPACES])) {
    errors.push("root release group roots must be root, packages/*, apps/*, plugin");
  }
  if (!equalArrays(unit.workspaces, CANONICAL_ROOT_WORKSPACES)) {
    errors.push(`root inventory workspaces must exactly match ${CANONICAL_ROOT_WORKSPACES.join(", ")}`);
  }
  if (
    unit.versionPolicy?.kind !== "synchronized" ||
    unit.versionPolicy.authority !== "package.json" ||
    unit.versionPolicy.releaseContract !== CANONICAL_RELEASE_CONTRACT
  ) {
    errors.push(
      `root release group must use synchronized package.json version policy with ${CANONICAL_RELEASE_CONTRACT}`,
    );
  }
  if (!equalArrays(unit.lockfiles, ["package-lock.json"])) {
    errors.push("root release group must use only package-lock.json");
  }
  if (!equalArrays(unit.workflows.map(({ path: workflowPath }) => workflowPath), CANONICAL_ROOT_WORKFLOWS)) {
    errors.push("root release group must declare verification, GitHub release, and CLI publish workflows");
  }

  const rootManifest = parseJsonFile(snapshot, "package.json", errors);
  const lock = parseJsonFile(snapshot, "package-lock.json", errors);
  if (!rootManifest || !lock) return;
  if (!equalArrays(rootManifest.workspaces, CANONICAL_ROOT_WORKSPACES)) {
    errors.push(`root workspaces must exactly match ${CANONICAL_ROOT_WORKSPACES.join(", ")}`);
  }
  if (lock.name !== rootManifest.name || lock.version !== rootManifest.version) {
    errors.push("root package-lock name/version must match package.json");
  }

  for (const manifestPath of unit.manifests) {
    const manifest = parseJsonFile(snapshot, manifestPath, errors);
    if (!manifest) continue;
    if (manifest.version !== rootManifest.version) {
      errors.push(`${manifestPath} version must match synchronized root version ${rootManifest.version}`);
    }
    const workspacePath = workspacePathForManifest(manifestPath);
    const lockedPackage = lock.packages?.[workspacePath];
    if (!lockedPackage) {
      errors.push(`package-lock.json is missing root release package ${workspacePath || "<root>"}`);
    } else if (
      lockedPackage.name !== manifest.name ||
      lockedPackage.version !== manifest.version
    ) {
      errors.push(`package-lock.json ${workspacePath || "<root>"} name/version must match its manifest`);
    }
  }

  const releaseContractPath = unit.versionPolicy?.releaseContract;
  const releaseContract = snapshot.releaseContracts?.[releaseContractPath];
  if (releaseContract?.error) {
    errors.push(`${releaseContractPath} cannot be loaded: ${releaseContract.error}`);
  } else {
    const packageFiles = releaseContract?.packageFiles;
    const manifestFiles = releaseContract?.manifestFiles;
    if (!Array.isArray(packageFiles)) {
      errors.push(`${releaseContractPath} must export packageFiles as an array`);
    }
    if (!Array.isArray(manifestFiles)) {
      errors.push(`${releaseContractPath} must export manifestFiles as an array`);
    }
    if (Array.isArray(packageFiles) && Array.isArray(manifestFiles)) {
      const releaseFiles = [...packageFiles, ...manifestFiles];
      const stringReleaseFiles = releaseFiles.filter(
        (relativePath) => typeof relativePath === "string",
      );
      if (stringReleaseFiles.length !== releaseFiles.length) {
        errors.push(`${releaseContractPath} release files must contain only string paths`);
      }
      if (new Set(releaseFiles).size !== releaseFiles.length) {
        errors.push(`${releaseContractPath} release files must be unique paths`);
      }

      const packageFileSet = new Set(
        packageFiles.filter((relativePath) => typeof relativePath === "string"),
      );
      const manifestFileSet = new Set(
        manifestFiles.filter((relativePath) => typeof relativePath === "string"),
      );
      for (const manifestPath of unit.manifests) {
        if (!packageFileSet.has(manifestPath)) {
          errors.push(`${releaseContractPath} does not synchronize ${manifestPath}`);
        }
      }
      for (const manifestPath of CANONICAL_RELEASE_MANIFESTS) {
        if (!manifestFileSet.has(manifestPath)) {
          errors.push(`${releaseContractPath} does not synchronize ${manifestPath}`);
        }
      }

      const independentManifests = new Set(
        inventory.units
          .filter((candidate) => candidate.classification !== "root-release-group")
          .flatMap((candidate) => candidate.manifests),
      );
      for (const manifestPath of stringReleaseFiles) {
        if (independentManifests.has(manifestPath)) {
          errors.push(
            `${releaseContractPath} must not synchronize independent manifest ${manifestPath}`,
          );
        } else if (
          !unit.manifests.includes(manifestPath) &&
          !CANONICAL_RELEASE_MANIFESTS.includes(manifestPath)
        ) {
          errors.push(`${releaseContractPath} must not synchronize non-root manifest ${manifestPath}`);
        }
      }
      for (const manifestPath of packageFileSet) {
        if (CANONICAL_RELEASE_MANIFESTS.includes(manifestPath)) {
          errors.push(`${releaseContractPath} packageFiles must contain only package manifests`);
        }
      }
      for (const manifestPath of manifestFileSet) {
        if (unit.manifests.includes(manifestPath)) {
          errors.push(`${releaseContractPath} manifestFiles must contain only plugin manifests`);
        }
      }
    }
  }

  for (const independentUnit of inventory.units.filter(
    (candidate) => candidate.classification !== "root-release-group",
  )) {
    for (const independentRoot of independentUnit.roots) {
      if (rootManifest.workspaces.includes(independentRoot)) {
        errors.push(`${independentRoot} must not belong to root workspaces`);
      }
      if (lock.packages?.[independentRoot]) {
        errors.push(`${independentRoot} must not belong to the root package-lock version group`);
      }
    }
  }
}

function validateIndependentUnits(inventory, snapshot, errors) {
  for (const unit of inventory.units.filter(
    (candidate) => candidate.classification !== "root-release-group",
  )) {
    if (!new Set(["independent-service", "independent-extension"]).has(unit.classification)) {
      errors.push(`${unit.id} has unsupported classification ${unit.classification}`);
      continue;
    }
    if (
      unit.versionPolicy?.kind !== "independent" ||
      unit.versionPolicy.authority !== unit.manifests[0] ||
      unit.versionPolicy.lockfile !== unit.lockfiles[0]
    ) {
      errors.push(`${unit.id} must use independent version policy with its own manifest and lockfile`);
    }
    if ((unit.workspaces ?? []).length !== 0) {
      errors.push(`${unit.id} must not declare root workspaces`);
    }
    if (unit.manifests.length !== 1 || unit.roots.length !== 1) {
      errors.push(`${unit.id} must have one independent product root and manifest`);
    }
    const expectedPrefix =
      unit.classification === "independent-service" ? "services/" : "extensions/";
    if (!unit.roots[0]?.startsWith(expectedPrefix)) {
      errors.push(`${unit.id} classification does not match product root ${unit.roots[0]}`);
    }

    const manifest = parseJsonFile(snapshot, unit.manifests[0], errors);
    const lock = parseJsonFile(snapshot, unit.lockfiles[0], errors);
    if (!manifest || !lock) continue;
    const lockedRoot = lock.packages?.[""];
    if (
      lock.name !== manifest.name ||
      lock.version !== manifest.version ||
      lockedRoot?.name !== manifest.name ||
      lockedRoot?.version !== manifest.version
    ) {
      errors.push(`${unit.id} own lockfile version/name must match its manifest`);
    }
  }
}

function validateWorkflowCoverage(inventory, snapshot, errors) {
  for (const unit of inventory.units) {
    for (const workflowPolicy of unit.workflows) {
      const workflow = parseWorkflowFile(snapshot, workflowPolicy.path, errors);
      if (!workflow || workflowPolicy.pathPolicy !== "listed-paths") continue;
      const requiredPaths = workflowPolicy.requiredPaths ?? [];
      const policyInputs = [
        ...inventory.governancePaths,
        ...(unit.dependencies ?? []),
        ...(unit.compatibilityInputs ?? []),
        workflowPolicy.path,
      ];
      for (const requiredPath of policyInputs) {
        if (!requiredPaths.includes(requiredPath)) {
          errors.push(`${workflowPolicy.path} policy is missing required path ${requiredPath}`);
        }
      }
      for (const event of ["push", "pull_request"]) {
        const paths = workflowPaths(workflow, event);
        if (!equalArrays(paths, requiredPaths)) {
          const missing = requiredPaths.find((requiredPath) => !paths?.includes(requiredPath));
          errors.push(
            missing
              ? `${workflowPolicy.path} ${event} is missing required path ${missing}`
              : `${workflowPolicy.path} ${event} paths must exactly match inventory policy`,
          );
        }
      }
    }
  }
}

export function validateProductUnitSnapshot(inventory, snapshot) {
  const errors = [];
  validateInventorySchema(inventory, errors);
  if (!Array.isArray(inventory?.units)) return errors;
  validateClassifiedFiles(inventory, snapshot, errors);
  validateRootReleaseGroup(inventory, snapshot, errors);
  validateIndependentUnits(inventory, snapshot, errors);
  validateWorkflowCoverage(inventory, snapshot, errors);
  return errors;
}

export async function checkProductUnits(root = repositoryRoot) {
  const inventory = JSON.parse(
    await readFile(path.join(root, "product-units.json"), "utf8"),
  );
  const snapshot = await loadProductUnitSnapshot(root, inventory);
  return validateProductUnitSnapshot(inventory, snapshot);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const errors = await checkProductUnits();
  if (errors.length > 0) {
    console.error(errors.join("\n"));
    process.exitCode = 1;
  } else {
    console.log("Product unit inventory OK");
  }
}
