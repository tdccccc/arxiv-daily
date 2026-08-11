import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";

function defaultNpmExecutable() {
  return process.env.npm_execpath || (process.platform === "win32" ? "npm.cmd" : "npm");
}

function spawnNpm(args, spawn, executable) {
  const command = executable.endsWith(".js") || executable.endsWith(".cjs")
    ? process.execPath
    : executable;
  const commandArgs = command === process.execPath ? [executable, ...args] : args;
  const result = spawn(command, commandArgs, { stdio: "inherit" });
  return result.status ?? 1;
}

export function runRootTests(argv, options = {}) {
  const spawn = options.spawn ?? spawnSync;
  const executable = options.npmExecutable ?? defaultNpmExecutable();
  if (argv.length === 0) {
    return spawnNpm(["run", "test:workspaces"], spawn, executable);
  }
  return spawnNpm([
    "run",
    "test",
    "--workspace",
    "@arxiv-daily/core",
    "--",
    ...argv,
  ], spawn, executable);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    process.exitCode = runRootTests(process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  }
}
