import { execFile } from "node:child_process";
import { watch as watchSync } from "node:fs";
import fsPromises from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

/**
 * How many watches the probe asks the kernel for.
 *
 * This is a floor, not a prediction: Obsidian takes one watch per directory in
 * the vault plus its own configuration, so the real appetite is larger. An
 * environment that cannot hand out even this many right now cannot open a vault
 * at all, which is the state this blocker exists to catch.
 */
export const WATCH_HEADROOM = 128;

/** Where the kernel publishes the ceiling — context for the remedy, not the verdict. */
const WATCH_LIMIT_PATH = "/proc/sys/fs/inotify/max_user_watches";

async function readWatchLimit(fs = fsPromises) {
  try {
    const raw = await fs.readFile(WATCH_LIMIT_PATH, "utf8");
    const value = Number(raw.trim());
    return Number.isFinite(value) ? value : null;
  } catch {
    return null;
  }
}

/**
 * Ask the kernel for real file watches and see whether it hands them over.
 *
 * The ceiling in `/proc/sys/fs/inotify/max_user_watches` is not the answer to
 * the question that matters. It says how many watches this user may hold in
 * total, not how many are still free — a machine with a 524288 ceiling that has
 * already spent all of it reads exactly like an idle one. Obsidian then starts,
 * fails to watch the vault, and shows its own error page where the workspace
 * should be; every assertion below that point is describing a page nobody
 * meant to test.
 *
 * So the probe establishes watches instead of reading a number, on one
 * throwaway directory per watch: inotify returns the *same* descriptor for a
 * path it already watches, so watching one directory `n` times would consume a
 * single slot and prove nothing. Every watch is released again, because a probe
 * that kept them would degrade the very headroom it reports.
 *
 * `ok: false` is returned only for a refusal that actually means "no watches
 * left" (`ENOSPC`). Anything else — a read-only temp directory, a filesystem
 * with no inotify at all — comes back `ok: true, measured: false`: the harness
 * would rather say it could not measure than invent a blocker it cannot stand
 * behind.
 */
export async function probeFileWatchCapacity({
  headroom = WATCH_HEADROOM,
  fs = fsPromises,
  tmpDir = os.tmpdir(),
  watch = watchSync,
  readLimit = () => readWatchLimit(fs),
} = {}) {
  const limit = await readLimit();
  let base;
  try {
    base = await fs.mkdtemp(path.join(tmpDir, "obsidian-acceptance-watch-"));
  } catch (error) {
    return { ok: true, measured: false, reason: `no writable temporary directory: ${error.message}`, limit };
  }

  const watchers = [];
  let refusal = null;
  try {
    for (let index = 0; index < headroom; index += 1) {
      const dir = path.join(base, `w${index}`);
      await fs.mkdir(dir, { recursive: true });
      try {
        watchers.push(watch(dir));
      } catch (error) {
        refusal = error;
        break;
      }
    }
  } finally {
    for (const watcher of watchers) {
      try {
        watcher.close();
      } catch {
        // A watcher that cannot be closed is already gone.
      }
    }
    await fs.rm(base, { recursive: true, force: true }).catch(() => undefined);
  }

  if (!refusal) {
    return { ok: true, measured: true, established: watchers.length, requested: headroom, limit };
  }
  if (refusal.code !== "ENOSPC") {
    return {
      ok: true,
      measured: false,
      established: watchers.length,
      requested: headroom,
      limit,
      reason: `${refusal.code ?? "the kernel"}: ${refusal.message}`,
    };
  }
  return {
    ok: false,
    measured: true,
    code: "ENOSPC",
    message: refusal.message,
    established: watchers.length,
    requested: headroom,
    limit,
  };
}

async function defaultWhich(command) {
  try {
    const { stdout } = await execFileAsync("which", [command]);
    return stdout.trim() || null;
  } catch {
    return null;
  }
}

async function defaultCountPdfs(vaultPath, fs) {
  const seen = [];
  const walk = async (dir, depth) => {
    if (depth > 3 || seen.length > 0) return;
    let entries;
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (seen.length > 0) return;
      if (entry.name === ".obsidian") continue;
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) await walk(full, depth + 1);
      else if (entry.name.toLowerCase().endsWith(".pdf")) seen.push(full);
    }
  };
  await walk(vaultPath, 0);
  return seen.length;
}

const exists = async (fs, target) => {
  try {
    await fs.access(target);
    return true;
  } catch {
    return false;
  }
};

/**
 * Separate "this environment cannot run the acceptance" from "the acceptance
 * failed". Every blocker is collected so one run tells the operator everything
 * that needs fixing, and each carries the action that fixes it.
 */
export async function preflight(
  { vaultPath, obsidianPath, sourceDir, virtualDisplay = true },
  { fs = fsPromises, which = defaultWhich, countPdfs, probeWatchCapacity = probeFileWatchCapacity } = {},
) {
  const blockers = [];
  const add = (message, remedy) => blockers.push({ message, remedy });

  if (!vaultPath) {
    add("no test vault was given", "set OBSIDIAN_TEST_VAULT to a disposable vault path");
  } else if (!(await exists(fs, vaultPath))) {
    add(`the test vault does not exist: ${vaultPath}`, "create it, or point OBSIDIAN_TEST_VAULT elsewhere");
  } else if (!(await exists(fs, path.join(vaultPath, ".obsidian")))) {
    add(
      `${vaultPath} is not an Obsidian vault: it has no .obsidian directory`,
      "open the folder in Obsidian once so it becomes a vault",
    );
  }

  if (!(await exists(fs, obsidianPath))) {
    add(`Obsidian was not found at ${obsidianPath}`, "install Obsidian or set OBSIDIAN_BINARY to its path");
  }

  for (const artifact of ["main.js", "manifest.json", "styles.css"]) {
    if (!(await exists(fs, path.join(sourceDir, artifact)))) {
      add(
        `the branch build is missing: ${path.join(sourceDir, artifact)} does not exist`,
        "run npm run build --workspace obsidian-arxiv-daily",
      );
      break;
    }
  }

  if (virtualDisplay && !(await which("xvfb-run"))) {
    add(
      "xvfb-run is not available, so Obsidian cannot run on a virtual display",
      "install xvfb, or pass virtualDisplay: false to use the real display",
    );
  }

  // Watches are what Obsidian needs before it can open the vault at all. When
  // the quota is spent it does not crash: it renders its own error page, and a
  // walk over that page produces evidence about nothing. That is an environment
  // that cannot run the acceptance, so it belongs here and not in a scenario.
  const watches = await probeWatchCapacity({ fs });
  if (watches && watches.ok === false) {
    const ceiling = Number.isFinite(watches.limit) ? watches.limit : "unknown";
    const raised = Number.isFinite(watches.limit) ? Math.max(watches.limit * 2, 524288) : 524288;
    add(
      `the kernel is out of file watches: it granted ${watches.established} of ${watches.requested} requested `
        + `(${watches.code}), with fs.inotify.max_user_watches = ${ceiling} already spoken for. `
        + `Obsidian would open on its own error page instead of the vault`,
      `free watches by closing whatever is holding them, or raise the ceiling: `
        + `sudo sysctl -w fs.inotify.max_user_watches=${raised} for this boot, and `
        + `echo fs.inotify.max_user_watches=${raised} | sudo tee /etc/sysctl.d/60-inotify.conf to persist it`,
    );
  }

  if (vaultPath) {
    const count = countPdfs ? await countPdfs(vaultPath) : await defaultCountPdfs(vaultPath, fs);
    if (count === 0) {
      add(
        `the vault contains no PDF, so page location cannot be exercised: ${vaultPath}`,
        "put at least one PDF in the vault",
      );
    }
  }

  return { ok: blockers.length === 0, blockers };
}

export function describeBlockers(blockers) {
  return blockers.map((blocker) => `  - ${blocker.message}\n    → ${blocker.remedy}`).join("\n");
}
