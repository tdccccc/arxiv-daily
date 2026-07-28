import * as fs from "node:fs/promises";
import * as path from "node:path";
import JSZip from "jszip";
import type { CliRuntimeConfig } from "./config";
import type { CliIo } from "./main-types";

const MANIFEST_NAME = "arxiv-daily-export.json";
const LOGICAL = ["daily", "papers", ".index"] as const;

export async function dataExport(
  config: CliRuntimeConfig,
  io: CliIo,
  outPath: string,
): Promise<number> {
  const zip = new JSZip();
  const files: Array<{ logical: string; abs: string; mtimeMs: number }> = [];

  const rootMap: Record<(typeof LOGICAL)[number], string> = {
    daily: path.join(config.vaultRoot, config.settings.output.dailyDir),
    papers: path.join(config.vaultRoot, config.settings.output.papersDir),
    ".index": path.join(
      config.vaultRoot,
      path.dirname(config.settings.output.dailyDir),
      ".index",
    ),
  };
  // Prefer sibling .index under arxiv-daily/
  const indexCandidate = path.join(
    config.vaultRoot,
    "arxiv-daily",
    ".index",
  );
  if (await exists(indexCandidate)) {
    rootMap[".index"] = indexCandidate;
  } else {
    const alt = path.join(
      config.vaultRoot,
      path.dirname(config.settings.output.dailyDir),
      ".index",
    );
    rootMap[".index"] = alt;
  }

  for (const logical of LOGICAL) {
    const absRoot = rootMap[logical];
    if (!(await exists(absRoot))) continue;
    await walkFiles(absRoot, async (abs, rel) => {
      const stat = await fs.stat(abs);
      const logicalPath = `${logical}/${rel.split(path.sep).join("/")}`;
      files.push({ logical: logicalPath, abs, mtimeMs: stat.mtimeMs });
      const data = await fs.readFile(abs);
      const entry = zip.file(logicalPath, data, {
        date: new Date(stat.mtimeMs),
      });
      void entry;
    });
  }

  const manifest = {
    formatVersion: 1,
    exportedAt: new Date().toISOString(),
    contents: ["daily", "papers", "index"],
    fileCount: files.length,
  };
  zip.file(MANIFEST_NAME, JSON.stringify(manifest, null, 2));

  const buf = await zip.generateAsync({
    type: "nodebuffer",
    compression: "DEFLATE",
  });
  await fs.mkdir(path.dirname(path.resolve(outPath)), { recursive: true });
  await fs.writeFile(outPath, buf);
  writeLine(io.stdout, `exported ${files.length} files to ${outPath}`);
  return 0;
}

export async function dataImport(
  config: CliRuntimeConfig,
  io: CliIo,
  zipPath: string,
  opts: { yes?: boolean; isTTY?: boolean; confirm?: () => Promise<boolean> },
): Promise<number> {
  const data = await fs.readFile(zipPath);
  const zip = await JSZip.loadAsync(data);
  const manifestFile = zip.file(MANIFEST_NAME);
  if (!manifestFile) {
    writeLine(io.stderr, `missing ${MANIFEST_NAME} in zip`);
    return 2;
  }
  const manifest = JSON.parse(await manifestFile.async("string")) as {
    formatVersion?: number;
  };
  if (manifest.formatVersion !== 1) {
    writeLine(
      io.stderr,
      `unsupported formatVersion: ${String(manifest.formatVersion)}`,
    );
    return 2;
  }

  const targetRoots: Record<string, string> = {
    daily: path.join(config.vaultRoot, config.settings.output.dailyDir),
    papers: path.join(config.vaultRoot, config.settings.output.papersDir),
    ".index": path.join(config.vaultRoot, "arxiv-daily", ".index"),
  };

  type PlanItem = {
    logical: string;
    target: string;
    action: "create" | "keep-target" | "overwrite-from-zip";
    zipMtime: number;
    targetMtime?: number;
  };
  const plan: PlanItem[] = [];

  for (const [name, entry] of Object.entries(zip.files)) {
    if (entry.dir || name === MANIFEST_NAME) continue;
    const slash = name.indexOf("/");
    if (slash < 0) continue;
    const top = name.slice(0, slash);
    const rest = name.slice(slash + 1);
    if (!LOGICAL.includes(top as (typeof LOGICAL)[number])) continue;
    const targetRoot = targetRoots[top];
    if (!targetRoot) continue;
    const target = path.join(targetRoot, ...rest.split("/"));
    const zipDate = entry.date ? entry.date.getTime() : 0;
    if (!(await exists(target))) {
      plan.push({
        logical: name,
        target,
        action: "create",
        zipMtime: zipDate,
      });
      continue;
    }
    const st = await fs.stat(target);
    if (zipDate > st.mtimeMs) {
      plan.push({
        logical: name,
        target,
        action: "overwrite-from-zip",
        zipMtime: zipDate,
        targetMtime: st.mtimeMs,
      });
    } else {
      plan.push({
        logical: name,
        target,
        action: "keep-target",
        zipMtime: zipDate,
        targetMtime: st.mtimeMs,
      });
    }
  }

  const writes = plan.filter((p) => p.action !== "keep-target");
  writeLine(io.stdout, `import plan: ${plan.length} entries, ${writes.length} writes`);
  for (const item of plan.slice(0, 50)) {
    writeLine(io.stdout, `  ${item.action}: ${item.logical}`);
  }
  if (plan.length > 50) writeLine(io.stdout, `  ... ${plan.length - 50} more`);

  const tty = opts.isTTY ?? Boolean(process.stdin.isTTY);
  if (!opts.yes) {
    if (!tty) {
      writeLine(
        io.stderr,
        "non-interactive import requires --yes to apply; showing plan only",
      );
      return 0;
    }
    const ok = opts.confirm
      ? await opts.confirm()
      : await defaultConfirm(io);
    if (!ok) {
      writeLine(io.stdout, "import cancelled");
      return 0;
    }
  }

  for (const item of writes) {
    const entry = zip.file(item.logical);
    if (!entry) continue;
    const buf = await entry.async("nodebuffer");
    await fs.mkdir(path.dirname(item.target), { recursive: true });
    await fs.writeFile(item.target, buf);
    if (item.zipMtime > 0) {
      const d = new Date(item.zipMtime);
      await fs.utimes(item.target, d, d).catch(() => undefined);
    }
  }
  writeLine(io.stdout, `import applied: ${writes.length} file(s) written`);
  return 0;
}

async function defaultConfirm(io: CliIo): Promise<boolean> {
  writeLine(io.stdout, "Apply import? [y/N]");
  try {
    const readline = await import("node:readline/promises");
    const { stdin, stdout } = await import("node:process");
    const rl = readline.createInterface({ input: stdin, output: stdout });
    try {
      const ans = (await rl.question("")).trim().toLowerCase();
      return ans === "y" || ans === "yes";
    } finally {
      rl.close();
    }
  } catch {
    writeLine(io.stderr, "could not read confirmation; use --yes");
    return false;
  }
}

async function walkFiles(
  root: string,
  visit: (abs: string, rel: string) => Promise<void>,
  relBase = "",
): Promise<void> {
  const entries = await fs.readdir(root, { withFileTypes: true });
  for (const ent of entries) {
    const abs = path.join(root, ent.name);
    const rel = relBase ? path.join(relBase, ent.name) : ent.name;
    if (ent.isDirectory()) await walkFiles(abs, visit, rel);
    else if (ent.isFile()) await visit(abs, rel);
  }
}

async function exists(p: string): Promise<boolean> {
  try {
    await fs.stat(p);
    return true;
  } catch {
    return false;
  }
}

function writeLine(stream: { write(chunk: string): unknown }, line: string): void {
  stream.write(`${line}\n`);
}
