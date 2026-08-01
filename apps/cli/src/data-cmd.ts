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
export const DATA_IMPORT_LIMITS = Object.freeze({
  archiveBytes: 512 * 1024 * 1024,
  entryCount: 10_000,
  entryUncompressedBytes: 64 * 1024 * 1024,
  totalUncompressedBytes: 512 * 1024 * 1024,
  compressionRatio: 200,
});
export type DataImportLimits = typeof DATA_IMPORT_LIMITS;
export interface DataImportTestHooks {
  beforeRename?: (
    kind: "backup" | "promote" | "rollback",
    target: string,
    index: number,
  ) => void | Promise<void>;
}
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
  opts: {
    yes?: boolean;
    isTTY?: boolean;
    confirm?: () => Promise<boolean>;
    /** Test seams; production callers use DATA_IMPORT_LIMITS and no hooks. */
    limits?: Partial<DataImportLimits>;
    testHooks?: DataImportTestHooks;
  },
): Promise<number> {
  const limits = { ...DATA_IMPORT_LIMITS, ...opts.limits };
  const archiveStat = await fs.stat(zipPath);
  if (!archiveStat.isFile()) throw new Error(`import archive is not a regular file: ${zipPath}`);
  if (archiveStat.size > limits.archiveBytes) {
    throw new Error(
      `import archive exceeds compressed size limit (${archiveStat.size} > ${limits.archiveBytes})`,
    );
  }
  const archive = await fs.readFile(zipPath);
  const rawZip = validateRawZip(archive, limits);
  const zip = await JSZip.loadAsync(archive);
  assertJsZipMatchesRaw(zip, rawZip);
  const manifestFile = zip.file(MANIFEST_NAME);
  if (!manifestFile) {
    writeLine(io.stderr, `missing ${MANIFEST_NAME} in zip`);
    return 2;
  }
  const emitted = { total: 0 };
  const manifestRaw = rawZip.byJsZipName.get(MANIFEST_NAME);
  if (!manifestRaw) throw new Error(`verified central directory missing ${MANIFEST_NAME}`);
  const manifest = JSON.parse(
    (await readZipEntryBounded(
      manifestFile,
      manifestRaw.compressedSize,
      manifestRaw.crc32,
      limits,
      emitted,
    )).toString("utf8"),
  ) as {
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
    compressedSize: number;
    expectedCrc32: number;
    targetMtime?: number;
  };
  const plan: PlanItem[] = [];
  for (const [name, entry] of Object.entries(zip.files)) {
    if (entry.dir || name === MANIFEST_NAME) continue;
    const archiveName = entry.unsafeOriginalName ?? name;
    const rawEntry = rawZip.byJsZipName.get(name);
    if (!rawEntry) throw new Error(`JSZip entry missing from verified central directory: ${name}`);
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
        compressedSize: rawEntry.compressedSize,
        expectedCrc32: rawEntry.crc32,
      });
      continue;
    }
    const st = await requireRegularImportTarget(target);
    plan.push({
      logical: name,
      target,
      action: zipDate > st.mtimeMs ? "overwrite-from-zip" : "keep-target",
      zipMtime: zipDate,
      compressedSize: rawEntry.compressedSize,
      expectedCrc32: rawEntry.crc32,
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

  const staged: StagedImport[] = [];
  try {
    for (const item of writes) {
      const entry = zip.file(item.logical);
      if (!entry) throw new Error(`missing planned ZIP entry: ${item.logical}`);
      await assertSafePath(config.vaultRoot, vaultReal, item.target, false);
      staged.push(await stageImportedFileFromZip(
        config.vaultRoot,
        vaultReal,
        item.target,
        entry,
        item.compressedSize,
        item.expectedCrc32,
        limits,
        emitted,
      ));
    }
    await promoteImportTransaction(
      config.vaultRoot,
      vaultReal,
      staged,
      opts.testHooks,
    );
    for (let index = 0; index < staged.length; index += 1) {
      const item = writes[index]!;
      if (item.zipMtime > 0) {
        const d = new Date(item.zipMtime);
        await fs.utimes(item.target, d, d).catch(() => undefined);
      }
    }
  } finally {
    for (const item of staged.reverse()) await cleanupStagedImport(item);
  }
  writeLine(io.stdout, `import applied: ${writes.length} file(s) written`);
  return 0;
}

type VerifiedRawZipEntry = {
  rawName: string;
  jsZipName: string;
  compressedSize: number;
  crc32: number;
};
type VerifiedRawZip = {
  entries: VerifiedRawZipEntry[];
  byJsZipName: Map<string, VerifiedRawZipEntry>;
};

function validateRawZip(archive: Buffer, limits: DataImportLimits): VerifiedRawZip {
  const eocd = findEocd(archive);
  const disk = archive.readUInt16LE(eocd + 4);
  const centralDisk = archive.readUInt16LE(eocd + 6);
  const diskEntries = archive.readUInt16LE(eocd + 8);
  const totalEntries = archive.readUInt16LE(eocd + 10);
  const centralSize = archive.readUInt32LE(eocd + 12);
  const centralOffset = archive.readUInt32LE(eocd + 16);
  const commentLength = archive.readUInt16LE(eocd + 20);
  if (
    disk !== 0 || centralDisk !== 0 || diskEntries !== totalEntries ||
    totalEntries === 0xffff || centralSize === 0xffffffff || centralOffset === 0xffffffff
  ) throw new Error("unsupported ZIP64 or multi-disk archive");
  if (eocd + 22 + commentLength !== archive.length) {
    throw new Error("invalid ZIP end-of-central-directory boundary");
  }
  if (totalEntries > limits.entryCount) {
    throw new Error(`import archive exceeds entry count limit (${totalEntries} > ${limits.entryCount})`);
  }
  if (centralOffset + centralSize !== eocd || centralOffset > archive.length) {
    throw new Error("invalid ZIP central directory offset or size");
  }

  const entries: VerifiedRawZipEntry[] = [];
  const rawNames = new Set<string>();
  const normalizedNames = new Set<string>();
  const localRanges: Array<{ start: number; end: number }> = [];
  let cursor = centralOffset;
  for (let index = 0; index < totalEntries; index += 1) {
    if (cursor + 46 > eocd || archive.readUInt32LE(cursor) !== 0x02014b50) {
      throw new Error("invalid ZIP central directory header");
    }
    const flags = archive.readUInt16LE(cursor + 8);
    const method = archive.readUInt16LE(cursor + 10);
    const compressedSize = archive.readUInt32LE(cursor + 20);
    const uncompressedSize = archive.readUInt32LE(cursor + 24);
    const nameLength = archive.readUInt16LE(cursor + 28);
    const extraLength = archive.readUInt16LE(cursor + 30);
    const commentLen = archive.readUInt16LE(cursor + 32);
    const diskStart = archive.readUInt16LE(cursor + 34);
    const localOffset = archive.readUInt32LE(cursor + 42);
    const end = cursor + 46 + nameLength + extraLength + commentLen;
    if (
      end > eocd || diskStart !== 0 || compressedSize === 0xffffffff ||
      uncompressedSize === 0xffffffff || localOffset === 0xffffffff ||
      (method !== 0 && method !== 8) || (flags & ~0x0800) !== 0
    ) throw new Error("unsupported or invalid ZIP central directory entry");
    const nameBytes = archive.subarray(cursor + 46, cursor + 46 + nameLength);
    const extras = archive.subarray(cursor + 46 + nameLength, cursor + 46 + nameLength + extraLength);
    const rawName = decodeZipName(nameBytes, extras, flags);
    if (
      rawName.split("/").some((part) => part === "..") ||
      rawName.includes("\\")
    ) throw new Error(`unsafe archive path: ${rawName}`);
    const jsZipName = jsZipNormalizePath(rawName);
    if (!rawName || rawName.includes("\0")) throw new Error("invalid empty or NUL ZIP entry name");
    if (rawNames.has(rawName)) throw new Error(`duplicate ZIP entry name: ${rawName}`);
    if (normalizedNames.has(jsZipName)) {
      throw new Error(`ZIP path normalization collision: ${rawName}`);
    }
    localRanges.push(
      validateLocalHeader(
        archive,
        localOffset,
        centralOffset,
        nameBytes,
        method,
        flags,
        compressedSize,
        uncompressedSize,
        archive.readUInt32LE(cursor + 16),
      ),
    );
    rawNames.add(rawName);
    normalizedNames.add(jsZipName);
    entries.push({
      rawName,
      jsZipName,
      compressedSize,
      crc32: archive.readUInt32LE(cursor + 16),
    });
    cursor = end;
  }
  if (cursor !== eocd) throw new Error("ZIP central directory record count mismatch");
  localRanges.sort((a, b) => a.start - b.start);
  for (let index = 1; index < localRanges.length; index += 1) {
    if (localRanges[index]!.start < localRanges[index - 1]!.end) {
      throw new Error("overlapping ZIP local file ranges");
    }
  }
  return { entries, byJsZipName: new Map(entries.map((entry) => [entry.jsZipName, entry])) };
}

function findEocd(archive: Buffer): number {
  const minimum = Math.max(0, archive.length - 22 - 0xffff);
  for (let offset = archive.length - 22; offset >= minimum; offset -= 1) {
    if (archive.readUInt32LE(offset) === 0x06054b50) return offset;
  }
  throw new Error("missing ZIP end-of-central-directory record");
}

function validateLocalHeader(
  archive: Buffer,
  offset: number,
  centralOffset: number,
  centralName: Buffer,
  method: number,
  flags: number,
  compressedSize: number,
  uncompressedSize: number,
  crc: number,
): { start: number; end: number } {
  if (offset + 30 > archive.length || archive.readUInt32LE(offset) !== 0x04034b50) {
    throw new Error("invalid ZIP local header offset");
  }
  const localFlags = archive.readUInt16LE(offset + 6);
  const localMethod = archive.readUInt16LE(offset + 8);
  const localCrc = archive.readUInt32LE(offset + 14);
  const localCompressed = archive.readUInt32LE(offset + 18);
  const localUncompressed = archive.readUInt32LE(offset + 22);
  const nameLength = archive.readUInt16LE(offset + 26);
  const extraLength = archive.readUInt16LE(offset + 28);
  const dataStart = offset + 30 + nameLength + extraLength;
  if (dataStart > centralOffset) throw new Error("invalid ZIP local extra field boundary");
  visitZipExtras(
    archive.subarray(offset + 30 + nameLength, dataStart),
    () => undefined,
  );
  if (
    localFlags !== flags || localMethod !== method || localCrc !== crc ||
    localCompressed !== compressedSize || localUncompressed !== uncompressedSize ||
    dataStart + compressedSize > centralOffset ||
    !archive.subarray(offset + 30, offset + 30 + nameLength).equals(centralName)
  ) throw new Error("ZIP local and central headers do not match");
  return { start: offset, end: dataStart + compressedSize };
}

function decodeZipName(name: Buffer, extras: Buffer, flags: number): string {
  let unicodePath: string | undefined;
  visitZipExtras(extras, (id, value) => {
    if (id === 0x7075) {
      if (
        unicodePath !== undefined || value.length < 5 || value[0] !== 1 ||
        value.readUInt32LE(1) !== crc32(name)
      ) throw new Error("invalid ZIP Unicode path extra field");
      unicodePath = decodeUtf8Strict(value.subarray(5));
    }
  });
  if ((flags & 0x0800) !== 0) {
    if (unicodePath !== undefined) throw new Error("ambiguous ZIP Unicode path encoding");
    return decodeUtf8Strict(name);
  }
  return unicodePath ?? decodeUtf8Strict(name);
}

function visitZipExtras(
  extras: Buffer,
  visit: (id: number, value: Buffer) => void,
): void {
  let cursor = 0;
  while (cursor < extras.length) {
    if (cursor + 4 > extras.length) throw new Error("invalid ZIP extra field boundary");
    const id = extras.readUInt16LE(cursor);
    const size = extras.readUInt16LE(cursor + 2);
    const end = cursor + 4 + size;
    if (end > extras.length) throw new Error("invalid ZIP extra field boundary");
    if (id === 0x0001) throw new Error("unsupported ZIP64 extra field");
    visit(id, extras.subarray(cursor + 4, end));
    cursor = end;
  }
}

function decodeUtf8Strict(value: Buffer): string {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(value);
  } catch {
    throw new Error("invalid UTF-8 ZIP entry name");
  }
}

function jsZipNormalizePath(value: string): string {
  const result: string[] = [];
  const parts = value.split("/");
  for (let index = 0; index < parts.length; index += 1) {
    const part = parts[index]!;
    if (part === "." || (part === "" && index !== 0 && index !== parts.length - 1)) continue;
    if (part === "..") result.pop();
    else result.push(part);
  }
  return result.join("/");
}

function assertJsZipMatchesRaw(zip: JSZip, raw: VerifiedRawZip): void {
  const names = Object.keys(zip.files);
  if (names.length !== raw.entries.length) {
    throw new Error("JSZip entry map does not match verified central directory");
  }
  for (const entry of raw.entries) {
    if (!zip.files[entry.jsZipName]) {
      throw new Error(`JSZip entry missing after verified parse: ${entry.rawName}`);
    }
  }
}

function crc32(value: Buffer): number {
  return (crc32Update(0xffffffff, value) ^ 0xffffffff) >>> 0;
}

function crc32Update(state: number, value: Buffer): number {
  let crc = state;
  for (const byte of value) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
    }
  }
  return crc >>> 0;
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

type StagedImport = {
  target: string;
  tmp: string;
  createdDirs: string[];
  promoted: boolean;
  rollbackPath?: string;
};

async function stageImportedFileFromZip(
  vaultRoot: string,
  vaultReal: string,
  target: string,
  entry: JSZip.JSZipObject,
  compressedSize: number,
  expectedCrc32: number,
  limits: DataImportLimits,
  emitted: { total: number },
): Promise<StagedImport> {
  const targetDir = path.dirname(target);
  const createdDirs = await mkdirTracked(targetDir, path.resolve(vaultRoot));
  const tmp = path.join(targetDir, `.arxiv-daily-import-${randomUUID()}.tmp`);
  const staged = { target, tmp, createdDirs, promoted: false };
  try {
    await assertSafePath(vaultRoot, vaultReal, target, false);
    const handle = await fs.open(tmp, "wx", 0o600);
    try {
      await streamZipEntry(
        entry,
        compressedSize,
        expectedCrc32,
        limits,
        emitted,
        async (chunk) => {
        let offset = 0;
        while (offset < chunk.length) {
          const { bytesWritten } = await handle.write(
            chunk,
            offset,
            chunk.length - offset,
          );
          if (bytesWritten <= 0) throw new Error("failed to make progress writing import temp file");
          offset += bytesWritten;
        }
      });
      await handle.sync();
    } finally {
      await handle.close();
    }
    await fs.chmod(tmp, 0o600);
    return staged;
  } catch (error) {
    await cleanupStagedImport(staged);
    throw error;
  }
}

async function promoteImportTransaction(
  vaultRoot: string,
  vaultReal: string,
  staged: StagedImport[],
  hooks?: DataImportTestHooks,
): Promise<void> {
  try {
    // Move every existing target aside before promoting any staged inode.
    for (let index = 0; index < staged.length; index += 1) {
      const item = staged[index]!;
      await assertSafePath(vaultRoot, vaultReal, item.target, false);
      if (await pathExists(item.target)) {
        await requireRegularImportTarget(item.target);
        item.rollbackPath = path.join(
          path.dirname(item.target),
          `.arxiv-daily-rollback-${randomUUID()}.bak`,
        );
        await hooks?.beforeRename?.("backup", item.target, index);
        await fs.rename(item.target, item.rollbackPath);
      }
    }
    for (let index = 0; index < staged.length; index += 1) {
      const item = staged[index]!;
      await assertSafePath(vaultRoot, vaultReal, item.target, false);
      await hooks?.beforeRename?.("promote", item.target, index);
      await fs.rename(item.tmp, item.target);
      item.promoted = true;
    }
  } catch (cause) {
    const rollbackErrors: Error[] = [];
    for (let index = staged.length - 1; index >= 0; index -= 1) {
      const item = staged[index]!;
      try {
        if (item.promoted && await pathExists(item.target)) {
          await requireRegularImportTarget(item.target);
          await fs.rm(item.target);
          item.promoted = false;
        }
        if (item.rollbackPath && await pathExists(item.rollbackPath)) {
          await hooks?.beforeRename?.("rollback", item.target, index);
          await assertSafePath(vaultRoot, vaultReal, item.target, false);
          await fs.rename(item.rollbackPath, item.target);
          item.rollbackPath = undefined;
        }
      } catch (error) {
        rollbackErrors.push(error as Error);
      }
    }
    if (rollbackErrors.length > 0) {
      throw new AggregateError(
        [cause, ...rollbackErrors],
        `data import promotion failed and rollback was incomplete; recover preserved .arxiv-daily-rollback-* backups manually`,
        { cause },
      );
    }
    throw cause;
  }

  for (const item of staged) {
    if (item.rollbackPath) {
      await fs.rm(item.rollbackPath, { force: true });
      item.rollbackPath = undefined;
    }
  }
}

async function cleanupStagedImport(staged: StagedImport): Promise<void> {
  await fs.rm(staged.tmp, { force: true });
  // A rollback backup is deliberately never deleted here. If rollback failed,
  // it is the recoverable old target and the AggregateError names the pattern.
  if (!staged.promoted && !staged.rollbackPath) {
    for (const dir of staged.createdDirs) await fs.rmdir(dir).catch(() => undefined);
  }
}

async function mkdirTracked(targetDir: string, vaultRoot: string): Promise<string[]> {
  const missing: string[] = [];
  let current = targetDir;
  while (current !== vaultRoot && !(await pathExists(current))) {
    missing.push(current);
    const parent = path.dirname(current);
    if (parent === current) throw new Error(`path escapes vault: ${targetDir}`);
    current = parent;
  }
  await fs.mkdir(targetDir, { recursive: true });
  return missing;
}

async function readZipEntryBounded(
  entry: JSZip.JSZipObject,
  compressedSize: number,
  expectedCrc32: number,
  limits: DataImportLimits,
  emitted: { total: number },
): Promise<Buffer> {
  const chunks: Buffer[] = [];
  await streamZipEntry(
    entry,
    compressedSize,
    expectedCrc32,
    limits,
    emitted,
    async (chunk) => {
    chunks.push(Buffer.from(chunk));
  });
  return Buffer.concat(chunks);
}

async function streamZipEntry(
  entry: JSZip.JSZipObject,
  compressedSize: number,
  expectedCrc32: number,
  limits: DataImportLimits,
  emitted: { total: number },
  write: (chunk: Buffer) => Promise<void>,
): Promise<void> {
  type PausableStream = NodeJS.ReadableStream & {
    pause(): void;
    resume(): void;
    destroy?(error?: Error): void;
  };
  const stream = entry.nodeStream("nodebuffer") as PausableStream;
  let entryBytes = 0;
  let crcState = 0xffffffff;
  await new Promise<void>((resolve, reject) => {
    let settled = false;
    let ended = false;
    let pending = Promise.resolve();
    const fail = (error: unknown) => {
      if (settled) return;
      settled = true;
      stream.destroy?.(error as Error);
      reject(error);
    };
    const finish = () => {
      if (settled) return;
      const actualCrc32 = (crcState ^ 0xffffffff) >>> 0;
      if (actualCrc32 !== expectedCrc32) {
        fail(new Error(`import archive entry CRC32 mismatch: ${entry.name}`));
        return;
      }
      settled = true;
      resolve();
    };
    stream.on("data", (value: unknown) => {
      if (settled) return;
      stream.pause();
      const chunk = Buffer.isBuffer(value) ? value : Buffer.from(value as ArrayBuffer);
      entryBytes += chunk.length;
      emitted.total += chunk.length;
      crcState = crc32Update(crcState, chunk);
      let violation: Error | undefined;
      if (entryBytes > limits.entryUncompressedBytes) {
        violation = new Error(`import archive entry exceeds uncompressed size limit: ${entry.name}`);
      } else if (emitted.total > limits.totalUncompressedBytes) {
        violation = new Error("import archive exceeds total uncompressed size limit");
      } else if (
        entryBytes > 0 &&
        (compressedSize === 0 || entryBytes / compressedSize > limits.compressionRatio)
      ) {
        violation = new Error(`import archive entry exceeds compression ratio limit: ${entry.name}`);
      }
      if (violation) {
        fail(violation);
        return;
      }
      pending = pending.then(() => write(chunk));
      pending.then(
        () => {
          if (settled) return;
          if (ended) {
            finish();
          } else {
            stream.resume();
          }
        },
        fail,
      );
    });
    stream.on("end", () => {
      ended = true;
      pending.then(finish, fail);
    });
    stream.on("error", fail);
  });
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
