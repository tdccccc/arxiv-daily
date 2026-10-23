import { execFileSync, spawnSync } from "node:child_process";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { root } from "./release-utils.mjs";

export function trackedLockfiles(repoRoot = root) {
  return execFileSync("git", ["ls-files", "-z", "*package-lock.json"], { cwd: repoRoot, encoding: "utf8" })
    .split("\0").filter(file => path.posix.basename(file) === "package-lock.json").sort();
}

/** Every committed lockfile is audited, including runtimes installed outside the workspace. */
export function auditLockfiles({ repoRoot = root, lockfiles = trackedLockfiles(repoRoot), run = spawnSync, log = console.log } = {}) {
  if (lockfiles.length === 0) throw new Error("No tracked lockfiles found for dependency audit");
  const failures = [];
  for (const lockfile of lockfiles) {
    const directory = path.dirname(lockfile);
    log(`Audit ${lockfile}`);
    let result;
    for (let attempt = 1; attempt <= 3; attempt++) {
      result = run("npm", ["audit", "--audit-level=moderate", "--package-lock-only", "--prefix", path.resolve(repoRoot, directory)], { cwd: repoRoot, encoding: "utf8" });
      const registryError = /audit endpoint returned an error|Invalid package tree/.test(`${result.stdout || ""}\n${result.stderr || ""}`);
      if (!registryError || attempt === 3) break;
      log(`Retry registry audit failure for ${lockfile} (${attempt}/3)`);
    }
    if (result.stdout) log(result.stdout);
    if (result.stderr) log(result.stderr);
    if (result.error || result.status !== 0) failures.push(lockfile);
  }
  if (failures.length) throw new Error(`Dependency audit failed: ${failures.join(", ")}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try { auditLockfiles(); }
  catch (error) { console.error(error.message); process.exitCode = 1; }
}
