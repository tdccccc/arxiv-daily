import { spawnSync } from "node:child_process";
import { readdirSync } from "node:fs";
import { relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const DEFAULT_BATCH_SIZE = 8;
const coreDir = resolve(import.meta.dirname, "../packages/core");
const coreConfig = resolve(coreDir, "vitest.config.mts");
const defaultVitestCli = fileURLToPath(import.meta.resolve("vitest/vitest.mjs"));

function comparePaths(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function walkTestFiles(directory, baseDirectory, files) {
  const entries = readdirSync(directory, { withFileTypes: true })
    .sort((left, right) => comparePaths(left.name, right.name));

  for (const entry of entries) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) {
      walkTestFiles(path, baseDirectory, files);
    } else if (entry.isFile() && entry.name.endsWith(".test.ts")) {
      files.push(relative(baseDirectory, path).split(sep).join("/"));
    }
  }
}

export function discoverCoreTestFiles(directory = coreDir) {
  const files = [];
  walkTestFiles(resolve(directory, "tests"), directory, files);
  return files.sort(comparePaths);
}

export function partitionTestFiles(files, batchSize = DEFAULT_BATCH_SIZE) {
  if (!Number.isSafeInteger(batchSize) || batchSize < 1) {
    throw new Error(`Core test batch size must be a positive integer; received ${batchSize}`);
  }

  const batches = [];
  for (let index = 0; index < files.length; index += batchSize) {
    batches.push(files.slice(index, index + batchSize));
  }
  return batches;
}

export function assertBatchCoverage(discoveredFiles, batches, batchSize = DEFAULT_BATCH_SIZE) {
  if (!Number.isSafeInteger(batchSize) || batchSize < 1) {
    throw new Error(`Core test batch size must be a positive integer; received ${batchSize}`);
  }
  if (batches.some((batch) => batch.length < 1 || batch.length > batchSize)) {
    throw new Error(`Core test batch size must stay between 1 and ${batchSize}`);
  }

  const plannedFiles = batches.flat();
  const discoveredCounts = new Map();
  const plannedCounts = new Map();
  for (const file of discoveredFiles) {
    discoveredCounts.set(file, (discoveredCounts.get(file) ?? 0) + 1);
  }
  for (const file of plannedFiles) {
    plannedCounts.set(file, (plannedCounts.get(file) ?? 0) + 1);
  }

  const allFiles = new Set([...discoveredCounts.keys(), ...plannedCounts.keys()]);
  if (
    plannedFiles.length !== discoveredFiles.length
    || [...allFiles].some((file) => discoveredCounts.get(file) !== 1 || plannedCounts.get(file) !== 1)
  ) {
    throw new Error("Every discovered Core test file must be planned exactly once");
  }

  if (plannedFiles.some((file, index) => file !== discoveredFiles[index])) {
    throw new Error("Core test batches must preserve deterministic order");
  }
}

function runVitest(argv, { spawn, vitestCli }) {
  const result = spawn(
    process.execPath,
    [vitestCli, "run", "--config", coreConfig, ...argv],
    { cwd: coreDir, stdio: "inherit" },
  );
  return result.status ?? 1;
}

export function runCoreTests(argv, options = {}) {
  const spawn = options.spawn ?? spawnSync;
  const vitestCli = options.vitestCli ?? defaultVitestCli;

  if (argv.length > 0) {
    return runVitest(argv, { spawn, vitestCli });
  }

  const discoveredFiles = Array.isArray(options.discoveredFiles)
    ? options.discoveredFiles
    : (options.discoveredFiles ?? discoverCoreTestFiles)(coreDir);
  if (discoveredFiles.length === 0) {
    throw new Error("No Core test files were discovered under tests/**/*.test.ts");
  }

  const batchSize = options.batchSize ?? DEFAULT_BATCH_SIZE;
  const batches = partitionTestFiles(discoveredFiles, batchSize);
  assertBatchCoverage(discoveredFiles, batches, batchSize);

  let firstFailure = 0;
  for (const batch of batches) {
    const status = runVitest(["--maxWorkers=1", ...batch], { spawn, vitestCli });
    if (firstFailure === 0 && status !== 0) firstFailure = status;
  }
  return firstFailure;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    process.exitCode = runCoreTests(process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  }
}
