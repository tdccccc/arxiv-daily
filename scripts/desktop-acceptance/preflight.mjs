import { execFile } from "node:child_process";
import fsPromises from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

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
  { fs = fsPromises, which = defaultWhich, countPdfs } = {},
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
