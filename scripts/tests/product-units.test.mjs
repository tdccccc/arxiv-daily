import assert from "node:assert/strict";
import {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  unlink,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  loadProductUnitSnapshot,
  validateProductUnitSnapshot,
} from "../check-product-units.mjs";
import { root } from "../release-utils.mjs";

const inventoryPath = `${root}/product-units.json`;

async function currentInventory() {
  return JSON.parse(await readFile(inventoryPath, "utf8"));
}

async function assertRepositoryValid(repositoryRoot, inventory) {
  const snapshot = await loadProductUnitSnapshot(repositoryRoot, inventory);
  const errors = validateProductUnitSnapshot(inventory, snapshot);
  assert.deepEqual(errors, []);
  return snapshot;
}

async function copyFixtureFile(fixtureRoot, relativePath) {
  const destination = path.join(fixtureRoot, relativePath);
  await mkdir(path.dirname(destination), { recursive: true });
  await copyFile(path.join(root, relativePath), destination);
}

async function createFixture(t) {
  const inventory = await currentInventory();
  const fixtureRoot = await mkdtemp(path.join(os.tmpdir(), "arxiv-daily-products-"));
  t.after(() => rm(fixtureRoot, { recursive: true, force: true }));

  const files = new Set([
    "product-units.json",
    "package.json",
    "scripts/release-utils.mjs",
  ]);
  for (const unit of inventory.units) {
    for (const relativePath of [
      ...unit.manifests,
      ...unit.lockfiles,
      ...unit.workflows.map(({ path: workflowPath }) => workflowPath),
      ...(unit.dependencies ?? []),
      ...(unit.compatibilityInputs ?? []),
      ...(unit.versionPolicy.releaseTools ?? []),
    ]) {
      files.add(relativePath);
    }
  }
  for (const relativePath of inventory.governancePaths) files.add(relativePath);
  for (const relativePath of files) await copyFixtureFile(fixtureRoot, relativePath);

  return { fixtureRoot, inventory };
}

function findUnit(inventory, id) {
  const unit = inventory.units.find((candidate) => candidate.id === id);
  assert(unit, `missing inventory unit ${id}`);
  return unit;
}

async function mutateReleaseContractSource(fixtureRoot, exportName, mutate) {
  for (const relativePath of [
    "scripts/release-utils.mjs",
    "scripts/sync-release-version.mjs",
    "scripts/check-release-version.mjs",
  ]) {
    const absolutePath = path.join(fixtureRoot, relativePath);
    const source = await readFile(absolutePath, "utf8");
    if (!new RegExp(`\\b${exportName}\\s*=\\s*\\[`).test(source)) continue;
    const mutated = mutate(source);
    assert.notEqual(
      mutated,
      source,
      `${relativePath} mutation must change the ${exportName} contract`,
    );
    await writeFile(absolutePath, mutated);
    return relativePath;
  }
  assert.fail(`missing release ${exportName} contract`);
}

test("current repository product units are closed and internally consistent", async () => {
  const inventory = await currentInventory();
  await assertRepositoryValid(root, inventory);
});

test("unknown product roots fail closed without becoming inventory fixtures", async (t) => {
  const { fixtureRoot, inventory } = await createFixture(t);
  const unknownManifest = path.join(
    fixtureRoot,
    "services/new-worker/package.json",
  );
  await mkdir(path.dirname(unknownManifest), { recursive: true });
  await writeFile(
    unknownManifest,
    '{"name":"new-worker","version":"1.0.0","private":true}\n',
  );

  const snapshot = await loadProductUnitSnapshot(fixtureRoot, inventory);
  assert.match(
    validateProductUnitSnapshot(inventory, snapshot).join("\n"),
    /unclassified manifest services\/new-worker\/package\.json/,
  );
});

test("nested service and extension manifests are never silently ignored", async (t) => {
  const { fixtureRoot, inventory } = await createFixture(t);
  for (const relativePath of [
    "services/email-relay/tools/diagnostics/package.json",
    "extensions/vscode-arxiv-daily/tools/generator/package.json",
  ]) {
    const manifestPath = path.join(fixtureRoot, relativePath);
    await mkdir(path.dirname(manifestPath), { recursive: true });
    await writeFile(
      manifestPath,
      '{"name":"nested-product","version":"1.0.0","private":true}\n',
    );
  }

  const snapshot = await loadProductUnitSnapshot(fixtureRoot, inventory);
  const result = validateProductUnitSnapshot(inventory, snapshot).join("\n");
  assert.match(result, /unclassified manifest services\/email-relay\/tools\/diagnostics\/package\.json/);
  assert.match(result, /unclassified manifest extensions\/vscode-arxiv-daily\/tools\/generator\/package\.json/);
});

test("classified manifests, locks, and workflows must exist", async (t) => {
  const { fixtureRoot, inventory } = await createFixture(t);
  for (const relativePath of [
    "services/email-relay/package-lock.json",
    "extensions/vscode-arxiv-daily/package.json",
    ".github/workflows/vscode-companion.yml",
  ]) {
    await unlink(path.join(fixtureRoot, relativePath));
    const snapshot = await loadProductUnitSnapshot(fixtureRoot, inventory);
    assert.match(
      validateProductUnitSnapshot(inventory, snapshot).join("\n"),
      new RegExp(`missing classified file ${relativePath.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`),
    );
    await copyFixtureFile(fixtureRoot, relativePath);
  }
});

test("workspace and independent version boundaries cannot drift", async (t) => {
  const { fixtureRoot, inventory } = await createFixture(t);
  const rootManifestPath = path.join(fixtureRoot, "package.json");
  const rootManifest = JSON.parse(await readFile(rootManifestPath, "utf8"));
  rootManifest.workspaces.push("services/*");
  await writeFile(rootManifestPath, `${JSON.stringify(rootManifest, null, 2)}\n`);

  let snapshot = await loadProductUnitSnapshot(fixtureRoot, inventory);
  assert.match(
    validateProductUnitSnapshot(inventory, snapshot).join("\n"),
    /root workspaces must exactly match/,
  );

  await copyFixtureFile(fixtureRoot, "package.json");
  const relayManifestPath = path.join(
    fixtureRoot,
    "services/email-relay/package.json",
  );
  const relayManifest = JSON.parse(await readFile(relayManifestPath, "utf8"));
  relayManifest.version = "9.9.9";
  await writeFile(relayManifestPath, `${JSON.stringify(relayManifest, null, 2)}\n`);
  snapshot = await loadProductUnitSnapshot(fixtureRoot, inventory);
  assert.match(
    validateProductUnitSnapshot(inventory, snapshot).join("\n"),
    /email-relay.*own lockfile version/i,
  );

  const wrongVersionPolicy = structuredClone(inventory);
  findUnit(wrongVersionPolicy, "email-relay").versionPolicy.kind = "synchronized";
  assert.match(
    validateProductUnitSnapshot(wrongVersionPolicy, snapshot).join("\n"),
    /email-relay.*must use independent version policy/i,
  );

  const wrongClassification = structuredClone(inventory);
  findUnit(wrongClassification, "email-relay").classification =
    "root-release-group";
  assert.match(
    validateProductUnitSnapshot(wrongClassification, snapshot).join("\n"),
    /email-relay must be classified as independent-service/i,
  );
});

test("release package contract rejects an independent manifest regardless of quote style", async (t) => {
  const { fixtureRoot, inventory } = await createFixture(t);
  const contractPath = await mutateReleaseContractSource(fixtureRoot, "packageFiles", (source) =>
    source.replace(
      /(\bpackageFiles\s*=\s*\[)/,
      "$1\n  'services/email-relay/package.json',",
    ),
  );

  const snapshot = await loadProductUnitSnapshot(fixtureRoot, inventory);
  assert.match(
    validateProductUnitSnapshot(inventory, snapshot).join("\n"),
    new RegExp(`${contractPath} must not synchronize independent manifest services/email-relay/package\\.json`),
  );
});

test("release manifest contract rejects an independent manifest", async (t) => {
  const { fixtureRoot, inventory } = await createFixture(t);
  const contractPath = await mutateReleaseContractSource(
    fixtureRoot,
    "manifestFiles",
    (source) =>
      source.replace(
        /(\bmanifestFiles\s*=\s*\[)/,
        '$1\n  "extensions/vscode-arxiv-daily/package.json",',
      ),
  );

  const snapshot = await loadProductUnitSnapshot(fixtureRoot, inventory);
  assert.match(
    validateProductUnitSnapshot(inventory, snapshot).join("\n"),
    new RegExp(
      `${contractPath} must not synchronize independent manifest extensions/vscode-arxiv-daily/package\\.json`,
    ),
  );
});

test("release package contract cannot replace a root manifest with a comment", async (t) => {
  const { fixtureRoot, inventory } = await createFixture(t);
  const contractPath = await mutateReleaseContractSource(fixtureRoot, "packageFiles", (source) =>
    source.replace(
      /^(\s*)["']plugin\/package\.json["'],/m,
      '$1// "plugin/package.json",',
    ),
  );

  const snapshot = await loadProductUnitSnapshot(fixtureRoot, inventory);
  assert.match(
    validateProductUnitSnapshot(inventory, snapshot).join("\n"),
    new RegExp(`${contractPath} does not synchronize plugin/package\\.json`),
  );
});

test("release package contract accepts equivalent single-quoted formatting", async (t) => {
  const { fixtureRoot, inventory } = await createFixture(t);
  await mutateReleaseContractSource(fixtureRoot, "packageFiles", (source) =>
    source.replace(/"([^"]*package\.json)"/g, "'$1'"),
  );

  await assertRepositoryValid(fixtureRoot, inventory);
});

test("workflow path policies cover compatibility and governance inputs", async (t) => {
  const { fixtureRoot, inventory } = await createFixture(t);
  for (const [workflowPath, requiredPath] of [
    [
      ".github/workflows/email-relay.yml",
      "packages/core/src/delivery/hosted.ts",
    ],
    [
      ".github/workflows/vscode-companion.yml",
      "contracts/companion-cli-commands.json",
    ],
    [".github/workflows/email-relay.yml", "scripts/check-product-units.mjs"],
  ]) {
    const absoluteWorkflowPath = path.join(fixtureRoot, workflowPath);
    const source = await readFile(absoluteWorkflowPath, "utf8");
    await writeFile(
      absoluteWorkflowPath,
      source.replace(`      - ${requiredPath}\n`, ""),
    );
    const snapshot = await loadProductUnitSnapshot(fixtureRoot, inventory);
    assert.match(
      validateProductUnitSnapshot(inventory, snapshot).join("\n"),
      new RegExp(`missing required path ${requiredPath.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`),
    );
    await copyFixtureFile(fixtureRoot, workflowPath);
  }
});
