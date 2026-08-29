#!/usr/bin/env node

import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";

const root = resolve(import.meta.dirname, "..");
const version = JSON.parse(await readFile(resolve(root, "package.json"), "utf8")).version;
const temp = await mkdtemp(resolve(tmpdir(), "arxiv-daily-install-smoke-"));

try {
  const pack = spawnSync(
    "npm",
    ["pack", "--workspace", "apps/cli", "--pack-destination", temp],
    { cwd: root, encoding: "utf8" },
  );
  if (pack.status !== 0) fail("npm pack failed", pack);
  const archiveName = pack.stdout.trim().split(/\r?\n/).at(-1);
  if (!archiveName || !archiveName.endsWith(".tgz")) {
    throw new Error(`npm pack did not report an archive: ${pack.stdout}`);
  }

  const installRoot = resolve(temp, "install");
  const install = spawnSync(
    "npm",
    ["install", "--prefix", installRoot, resolve(temp, archiveName)],
    { cwd: root, encoding: "utf8" },
  );
  if (install.status !== 0) fail("npm install of packed CLI failed", install);

  const binary = resolve(installRoot, "node_modules/.bin/arxiv-daily");
  const help = spawnSync(binary, ["--help"], {
    cwd: installRoot,
    encoding: "utf8",
  });
  if (help.status !== 0 || !help.stdout.includes("Usage:") || !help.stdout.includes(`Version: ${version}`)) {
    fail("installed CLI help smoke failed", help);
  }
  console.log("CLI package install smoke OK");
} finally {
  await rm(temp, { recursive: true, force: true });
}

function fail(message, result) {
  throw new Error(
    `${message}\nstatus=${result.status}\nstdout=${result.stdout}\nstderr=${result.stderr}`,
  );
}
