import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";

const root = resolve(import.meta.dirname, "..");
const cli = resolve(root, "apps/cli/dist/arxiv-daily-cli.cjs");
const pluginCli = resolve(root, "plugin/arxiv-daily-cli.cjs");
const pythonShim = resolve(root, "arxiv_daily.py");
const pluginBundle = resolve(root, "plugin/main.js");

for (const command of [
  [process.execPath, cli, "--help"],
  [process.execPath, pluginCli, "--help"],
  ["python3", pythonShim, "--help"],
]) {
  const result = spawnSync(command[0], command.slice(1), { cwd: root, encoding: "utf8" });
  if (result.status !== 0 || !result.stdout.includes("Usage:")) {
    fail(`${command.join(" ")} help failed`, result);
  }
}

const temp = await mkdtemp(resolve(tmpdir(), "arxiv-daily-smoke-"));
try {
  const badConfig = resolve(temp, "bad.json");
  await writeFile(badConfig, "{not-json}\n");
  const result = spawnSync(process.execPath, [cli, "run", "--date", "2026-07-15", "--config", badConfig], {
    cwd: temp,
    encoding: "utf8",
  });
  if (result.status !== 2 || !result.stderr.includes("failed to parse CLI config")) {
    fail("CLI did not cross help and return exit 2 for controlled config error", result);
  }
} finally {
  await rm(temp, { recursive: true, force: true });
}

const bundle = await readFile(pluginBundle, "utf8");
for (const forbidden of ["@arxiv-daily/", "linkedom", "canvas", "node:"]) {
  if (bundle.includes(forbidden)) throw new Error(`plugin/main.js contains forbidden text: ${forbidden}`);
}
const runtimeRequires = new Set(
  Array.from(bundle.matchAll(/require\(["']([^"']+)["']\)/g), (match) => match[1]),
);
for (const specifier of runtimeRequires) {
  if (specifier !== "obsidian" && specifier !== "electron") {
    throw new Error(`plugin/main.js contains unexpected runtime require: ${specifier}`);
  }
}
console.log("Build smoke OK");

function fail(message, result) {
  throw new Error(`${message}\nstatus=${result.status}\nstdout=${result.stdout}\nstderr=${result.stderr}`);
}
