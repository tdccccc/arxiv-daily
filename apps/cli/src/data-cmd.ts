import * as fs from "node:fs/promises";
import * as path from "node:path";
import { randomUUID } from "node:crypto";
import JSZip from "jszip";
import {
  derivePaperInboxPaths,
  validateVaultRelativeDirectory,
} from "@arxiv-daily/core";
import type { CliRuntimeConfig } from "./config";
import type { CliIo } from "./main-types";

const MANIFEST_NAME = "arxiv-daily-export.json";
const LOGICAL = ["daily", "papers", ".index"] as const;
const LEGACY_INDEX_DIR = "arxiv-daily/.index";
const EXPORTER_VERSION = "0.3.5";
type Logical = (typeof LOGICAL)[number];
type OutputLayout = {
  dailyDir: string;
  papersDir: string;
  indexDir: string;
  exporterVersion?: string;
};

export async function dataExport(
  config: CliRuntimeConfig,
  io: CliIo,
  outPath: string,
): Promise<number> {
  const zip = new JSZip();
  const files: Array<{ logical: string; abs: string; mtimeMs: number }> = [];
  const rootMap = dataRoots(config);
  const vaultReal = await validateVaultRoot(config.vaultRoot);

  for (const logical of LOGICAL) {
    const absRoot = rootMap[logical];
    if (!(await pathExists(absRoot))) continue;
    await assertSafePath(config.vaultRoot, vaultReal, absRoot, true);
    await walkFiles(absRoot, async (abs, rel) => {
      await assertSafePath(config.vaultRoot, vaultReal, abs, true);
      const stat = await fs.stat(abs);
      const logicalPath = `${logical}/${rel.split(path.sep).join("/")}`;
      files.push({ logical: logicalPath, abs, mtimeMs: stat.mtimeMs });
      zip.file(logicalPath, await fs.readFile(abs), {
        date: new Date(stat.mtimeMs),
      });
    });
  }

  const layout = outputLayout(config);
  zip.file(
    MANIFEST_NAME,
    JSON.stringify(
      {
        formatVersion: 1,
        exportedAt: new Date().toISOString(),
        contents: ["daily", "papers", "index"],
        fileCount: files.length,
        outputLayout: { ...layout, exporterVersion: EXPORTER_VERSION },
      },
      null,
      2,
    ),
  );
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
  const zip = await JSZip.loadAsync(await fs.readFile(zipPath));
  const manifestFile = zip.file(MANIFEST_NAME);
  if (!manifestFile) {
    writeLine(io.stderr, `missing ${MANIFEST_NAME} in zip`);
    return 2;
  }
  const manifest = JSON.parse(await manifestFile.async("string")) as {
    formatVersion?: number;
    outputLayout?: OutputLayout;
  };
  if (manifest.formatVersion !== 1) {
    writeLine(
      io.stderr,
      `unsupported formatVersion: ${String(manifest.formatVersion)}`,
    );
    return 2;
  }
  let sourceLayout: OutputLayout | undefined;
  if (manifest.outputLayout !== undefined) {
    const canonicalLayout = canonicalOutputLayout(manifest.outputLayout);
    if (!canonicalLayout) {
      writeLine(io.stderr, "invalid outputLayout in export manifest");
      return 2;
    }
    sourceLayout = canonicalLayout;
  }

  const targetLayout = outputLayout(config);
  const targetRoots = dataRoots(config);
  const vaultReal = await validateVaultRoot(config.vaultRoot);
  for (const root of Object.values(targetRoots)) {
    await assertSafePath(config.vaultRoot, vaultReal, root, false);
  }
  const skipIndex = sourceLayout
    ? !outputLayoutsEqual(sourceLayout, targetLayout)
    : targetLayout.indexDir !== LEGACY_INDEX_DIR;
  let warnedSkippedIndex = false;

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
    const archiveName = entry.unsafeOriginalName ?? name;
    const slash = archiveName.indexOf("/");
    if (slash < 0) continue;
    const top = archiveName.slice(0, slash);
    const rest = archiveName.slice(slash + 1);
    if (!LOGICAL.includes(top as Logical)) continue;
    if (top === ".index" && skipIndex) {
      if (!warnedSkippedIndex) {
        const warning = sourceLayout
          ? `warning: archive outputLayout does not match the active output layout; skipping .index entries (${sourceLayout.indexDir} -> ${targetLayout.indexDir})`
          : `warning: legacy archive has no outputLayout; skipping .index entries because active index is ${targetLayout.indexDir}`;
        writeLine(io.stderr, warning);
        warnedSkippedIndex = true;
      }
      continue;
    }
    const target = resolveArchiveTarget(
      targetRoots[top as Logical],
      rest,
      archiveName,
    );
    await assertSafePath(config.vaultRoot, vaultReal, target, false);
    const zipDate = entry.date ? entry.date.getTime() : 0;
    if (!(await pathExists(target))) {
      plan.push({
        logical: name,
        target,
        action: "create",
        zipMtime: zipDate,
      });
      continue;
    }
    const st = await requireRegularImportTarget(target);
    plan.push({
      logical: name,
      target,
      action: zipDate > st.mtimeMs ? "overwrite-from-zip" : "keep-target",
      zipMtime: zipDate,
      targetMtime: st.mtimeMs,
    });
  }

  const writes = plan.filter((p) => p.action !== "keep-target");
  writeLine(
    io.stdout,
    `import plan: ${plan.length} entries, ${writes.length} writes`,
  );
  for (const item of plan.slice(0, 50)) {
    writeLine(io.stdout, `  ${item.action}: ${item.logical}`);
  }
  if (plan.length > 50) {
    writeLine(io.stdout, `  ... ${plan.length - 50} more`);
  }
  const tty = opts.isTTY ?? Boolean(process.stdin.isTTY);
  if (!opts.yes) {
    if (!tty) {
      writeLine(
        io.stderr,
        "non-interactive import requires --yes to apply; showing plan only",
      );
      return 0;
    }
    const ok = opts.confirm ? await opts.confirm() : await defaultConfirm(io);
    if (!ok) {
      writeLine(io.stdout, "import cancelled");
      return 0;
    }
  }

  for (const item of writes) {
    const entry = zip.file(item.logical);
    if (!entry) continue;
    // Revalidate immediately before mkdir and again before write. Node's path APIs
    // cannot make this race-free against a concurrent hostile filesystem actor.
    await assertSafePath(config.vaultRoot, vaultReal, item.target, false);
    await fs.mkdir(path.dirname(item.target), { recursive: true });
    await assertSafePath(config.vaultRoot, vaultReal, item.target, false);
    await replaceImportedFile(
      config.vaultRoot,
      vaultReal,
      item.target,
      await entry.async("nodebuffer"),
    );
    if (item.zipMtime > 0) {
      const d = new Date(item.zipMtime);
      await fs.utimes(item.target, d, d).catch(() => undefined);
    }
  }
  writeLine(io.stdout, `import applied: ${writes.length} file(s) written`);
  return 0;
}

function outputLayout(config: CliRuntimeConfig): OutputLayout {
  const { indexDir } = derivePaperInboxPaths(config.settings.output);
  const layout = canonicalOutputLayout({
    dailyDir: config.settings.output.dailyDir,
    papersDir: config.settings.output.papersDir,
    indexDir,
  });
  if (!layout) {
    throw new Error("invalid active output layout");
  }
  return layout;
}

function canonicalOutputLayout(value: unknown): OutputLayout | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const candidate = value as Record<string, unknown>;
  const keys = Object.keys(candidate).sort();
  const allowed = ["dailyDir", "exporterVersion", "indexDir", "papersDir"];
  if (keys.some((key) => !allowed.includes(key))) return null;
  if (
    typeof candidate.dailyDir !== "string" ||
    typeof candidate.papersDir !== "string" ||
    typeof candidate.indexDir !== "string" ||
    (candidate.exporterVersion !== undefined &&
      typeof candidate.exporterVersion !== "string")
  ) {
    return null;
  }

  const daily = validateVaultRelativeDirectory(candidate.dailyDir);
  const papers = validateVaultRelativeDirectory(candidate.papersDir);
  const index = validateVaultRelativeDirectory(candidate.indexDir);
  if (!daily.ok || !daily.value || candidate.dailyDir !== daily.value) return null;
  if (!papers.ok || !papers.value || candidate.papersDir !== papers.value) return null;
  if (!index.ok || !index.value || candidate.indexDir !== index.value) return null;

  let derivedIndex: string;
  try {
    derivedIndex = derivePaperInboxPaths({
      dailyDir: daily.value,
      papersDir: papers.value,
    }).indexDir;
  } catch {
    return null;
  }
  if (index.value !== derivedIndex) return null;
  return {
    dailyDir: daily.value,
    papersDir: papers.value,
    indexDir: derivedIndex,
    ...(candidate.exporterVersion === undefined
      ? {}
      : { exporterVersion: candidate.exporterVersion }),
  };
}

function outputLayoutsEqual(a: OutputLayout, b: OutputLayout): boolean {
  return (
    a.dailyDir === b.dailyDir &&
    a.papersDir === b.papersDir &&
    a.indexDir === b.indexDir
  );
}

function dataRoots(config: CliRuntimeConfig): Record<Logical, string> {
  const layout = outputLayout(config);
  return {
    daily: path.resolve(config.vaultRoot, layout.dailyDir),
    papers: path.resolve(config.vaultRoot, layout.papersDir),
    ".index": path.resolve(config.vaultRoot, layout.indexDir),
  };
}

function resolveArchiveTarget(
  targetRoot: string,
  rest: string,
  archiveName: string,
): string {
  if (
    !rest ||
    rest.includes("\\") ||
    path.posix.isAbsolute(rest) ||
    path.win32.isAbsolute(rest)
  ) {
    throw new Error(`unsafe archive path: ${archiveName}`);
  }
  const parts = rest.split("/");
  if (parts.some((part) => !part || part === "." || part === "..")) {
    throw new Error(`unsafe archive path: ${archiveName}`);
  }
  const root = path.resolve(targetRoot);
  const target = path.resolve(root, ...parts);
  const relative = path.relative(root, target);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`unsafe archive path: ${archiveName}`);
  }
  return target;
}

async function validateVaultRoot(vaultRoot: string): Promise<string> {
  const root = path.resolve(vaultRoot);
  const stat = await fs.lstat(root);
  if (stat.isSymbolicLink()) {
    throw new Error(`unsafe symlink path: ${root}`);
  }
  if (!stat.isDirectory()) {
    throw new Error(`vault root is not a directory: ${root}`);
  }
  return fs.realpath(root);
}

async function assertSafePath(
  vaultRoot: string,
  vaultReal: string,
  target: string,
  requireExisting: boolean,
): Promise<void> {
  const root = path.resolve(vaultRoot);
  const resolved = path.resolve(target);
  const relative = path.relative(root, resolved);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`path escapes vault: ${target}`);
  }
  const parts = relative ? relative.split(path.sep) : [];
  let current = root;
  for (let i = 0; i < parts.length; i += 1) {
    current = path.join(current, parts[i]!);
    try {
      const stat = await fs.lstat(current);
      if (stat.isSymbolicLink()) {
        throw new Error(`unsafe symlink path: ${current}`);
      }
      const real = await fs.realpath(current);
      const realRelative = path.relative(vaultReal, real);
      if (realRelative.startsWith("..") || path.isAbsolute(realRelative)) {
        throw new Error(`path escapes vault: ${current}`);
      }
    } catch (error) {
      if (
        (error as NodeJS.ErrnoException).code === "ENOENT" &&
        (!requireExisting || i < parts.length)
      ) {
        break;
      }
      throw error;
    }
  }
  if (requireExisting && !(await pathExists(resolved))) {
    throw new Error(`missing export path: ${resolved}`);
  }
}

async function replaceImportedFile(
  vaultRoot: string,
  vaultReal: string,
  target: string,
  content: Buffer,
): Promise<void> {
  const targetDir = path.dirname(target);
  const tmp = path.join(targetDir, `.arxiv-daily-import-${randomUUID()}.tmp`);

  try {
    const handle = await fs.open(tmp, "wx", 0o600);
    try {
      await handle.writeFile(content);
      await handle.sync();
    } finally {
      await handle.close();
    }
    await fs.chmod(tmp, 0o600);

    await assertSafePath(vaultRoot, vaultReal, target, false);
    if (await pathExists(target)) await requireRegularImportTarget(target);
    // A same-directory rename atomically replaces the directory entry with the
    // temporary inode, so an existing hardlink's external inode is untouched.
    await fs.rename(tmp, target);
  } finally {
    await fs.rm(tmp, { force: true });
  }
}

async function requireRegularImportTarget(target: string) {
  const stat = await fs.lstat(target);
  if (!stat.isFile()) {
    throw new Error(`import target is not a regular file: ${target}`);
  }
  return stat;
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
  for (const ent of await fs.readdir(root, { withFileTypes: true })) {
    const abs = path.join(root, ent.name);
    const rel = relBase ? path.join(relBase, ent.name) : ent.name;
    if (ent.isSymbolicLink()) {
      throw new Error(`unsafe symlink path: ${abs}`);
    }
    if (ent.isDirectory()) {
      await walkFiles(abs, visit, rel);
    } else if (ent.isFile()) {
      await visit(abs, rel);
    }
  }
}

async function pathExists(p: string): Promise<boolean> {
  try {
    await fs.lstat(p);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

function writeLine(
  stream: { write(chunk: string): unknown },
  line: string,
): void {
  stream.write(`${line}\n`);
}
