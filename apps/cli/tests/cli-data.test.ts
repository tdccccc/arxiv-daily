import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import JSZip from "jszip";
import { afterEach, describe, expect, it } from "vitest";
import {
  DailyFilterCheckpointStore,
  DailySummaryCheckpointStore,
  DEFAULT_SETTINGS,
  derivePaperInboxPaths,
  prepareDailyFilterCheckpoint,
  type DailyFilterCheckpointCompatibilityInput,
  type DailyPaperResult,
  type DailySummaryCheckpointCompatibilityInput,
} from "@arxiv-daily/core";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import type { CliRuntimeConfig } from "../src/config";
import { DEFAULT_CLI_SCHEDULE } from "../src/config";
import {
  DATA_IMPORT_LIMITS,
  dataExport,
  dataImport,
} from "../src/data-cmd";

const reportDate = "2026-08-01";
const compatibility: DailySummaryCheckpointCompatibilityInput = {
  paper: {
    id: "2608.00001",
    title: "Checkpoint paper",
    authors: "A. Author",
    abstract: "Abstract.",
    abstractConclusion: "## Abstract\nAbstract.\n\n## Conclusion\nConclusion.",
    fullSections: "## Methods\nMethod.",
  },
  summaryLanguage: "zh",
  llm: {
    provider: "custom",
    baseUrl: "https://example.test/v1",
    model: "model-a",
    thinkingMode: false,
    reasoningEffort: "medium",
  },
  temperature: 0,
};
const filterCompatibility: DailyFilterCheckpointCompatibilityInput = {
  papers: [compatibility.paper],
  arxivSettings: {
    ...DEFAULT_SETTINGS.arxiv,
    categories: ["astro-ph"],
    topics: [
      { id: "topic-id", name: "Topic", tag: "topic", description: "Topic", detail: false },
    ],
  },
  llm: compatibility.llm,
};
const preparedFilterCompatibility = prepareDailyFilterCheckpoint(filterCompatibility);
const filterResult = [
  { id: compatibility.paper.id, category: "topic" },
];
const result: DailyPaperResult = {
  kind: "structured",
  summary: {
    id: "2608.00001",
    coreProblem: "Problem",
    keyMethod: "Method",
    mainResult: "Result",
    whyRelevant: "Relevant",
    limitations: "Limitations",
  },
};

const tempRoots: string[] = [];

afterEach(async () => {
  await Promise.all(tempRoots.splice(0).map((root) => fs.rm(root, { recursive: true, force: true })));
});

function captureIo() {
  const stdout: string[] = [];
  const stderr: string[] = [];
  return {
    stdout,
    stderr,
    io: {
      stdout: { write: (chunk: string) => stdout.push(String(chunk)) },
      stderr: { write: (chunk: string) => stderr.push(String(chunk)) },
    },
  };
}

function config(vaultRoot: string, output = DEFAULT_SETTINGS.output): CliRuntimeConfig {
  return {
    settings: {
      ...DEFAULT_SETTINGS,
      output: { ...output },
    },
    vaultRoot,
    cacheDir: path.join(vaultRoot, ".cache"),
    linkStyle: "wikilink",
    configPath: path.join(vaultRoot, "config.toml"),
    scheduleIntent: { ...DEFAULT_CLI_SCHEDULE },
  };
}

async function tempDir(name: string): Promise<string> {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), `arxiv-daily-${name}-`));
  tempRoots.push(root);
  return root;
}

function eocdOffset(archive: Buffer): number {
  for (let offset = archive.length - 22; offset >= 0; offset -= 1) {
    if (archive.readUInt32LE(offset) === 0x06054b50) return offset;
  }
  throw new Error("missing EOCD");
}

function forgeZipSizes(
  archive: Buffer,
  entryName: string,
  sizes: { compressed?: number; uncompressed?: number },
): Buffer {
  const forged = Buffer.from(archive);
  for (let offset = 0; offset <= forged.length - 46; offset += 1) {
    if (forged.readUInt32LE(offset) !== 0x02014b50) continue;
    const nameLength = forged.readUInt16LE(offset + 28);
    const name = forged.subarray(offset + 46, offset + 46 + nameLength).toString("utf8");
    if (name !== entryName) continue;
    const localOffset = forged.readUInt32LE(offset + 42);
    if (sizes.compressed !== undefined) {
      forged.writeUInt32LE(sizes.compressed, offset + 20);
      forged.writeUInt32LE(sizes.compressed, localOffset + 18);
    }
    if (sizes.uncompressed !== undefined) {
      forged.writeUInt32LE(sizes.uncompressed, offset + 24);
      forged.writeUInt32LE(sizes.uncompressed, localOffset + 22);
    }
    return forged;
  }
  throw new Error(`missing central directory entry: ${entryName}`);
}

function forgeZipCrc(archive: Buffer, entryName: string, crc: number): Buffer {
  const forged = Buffer.from(archive);
  for (let offset = 0; offset <= forged.length - 46; offset += 1) {
    if (forged.readUInt32LE(offset) !== 0x02014b50) continue;
    const length = forged.readUInt16LE(offset + 28);
    if (forged.subarray(offset + 46, offset + 46 + length).toString("utf8") !== entryName) continue;
    forged.writeUInt32LE(crc, offset + 16);
    forged.writeUInt32LE(crc, forged.readUInt32LE(offset + 42) + 14);
    return forged;
  }
  throw new Error(`missing central entry: ${entryName}`);
}

function renameRawZipEntry(archive: Buffer, from: string, to: string): Buffer {
  if (Buffer.byteLength(from) !== Buffer.byteLength(to)) throw new Error("ZIP rename must preserve bytes");
  const forged = Buffer.from(archive);
  let central = -1;
  for (let offset = 0; offset <= forged.length - 46; offset += 1) {
    if (forged.readUInt32LE(offset) !== 0x02014b50) continue;
    const length = forged.readUInt16LE(offset + 28);
    if (forged.subarray(offset + 46, offset + 46 + length).toString("utf8") === from) {
      central = offset;
      break;
    }
  }
  if (central < 0) throw new Error(`missing central entry: ${from}`);
  const local = forged.readUInt32LE(central + 42);
  forged.write(to, central + 46, "utf8");
  forged.write(to, local + 30, "utf8");
  return forged;
}

async function exportAndImportCheckpoint(output = DEFAULT_SETTINGS.output) {
  const sourceRoot = await tempDir("export");
  const destinationRoot = await tempDir("import");
  const archivePath = path.join(await tempDir("archive"), "vault.zip");
  const sourceConfig = config(sourceRoot, output);
  const destinationConfig = config(destinationRoot, output);
  const sourceStorage = new NodeStorageAdapter(sourceRoot);
  const sourceFilterStore = new DailyFilterCheckpointStore(
    sourceStorage,
    sourceConfig.settings.output,
  );
  const sourceSummaryStore = new DailySummaryCheckpointStore(
    sourceStorage,
    sourceConfig.settings.output,
  );
  await sourceFilterStore.save(reportDate, preparedFilterCompatibility, filterResult);
  await sourceSummaryStore.upsert(reportDate, compatibility, result);

  const exportIo = captureIo();
  await expect(dataExport(sourceConfig, exportIo.io, archivePath)).resolves.toBe(0);
  const archive = await JSZip.loadAsync(await fs.readFile(archivePath));

  const importIo = captureIo();
  await expect(
    dataImport(destinationConfig, importIo.io, archivePath, { yes: true }),
  ).resolves.toBe(0);
  const destinationStorage = new NodeStorageAdapter(destinationRoot);
  const restoredFilterStore = new DailyFilterCheckpointStore(
    destinationStorage,
    destinationConfig.settings.output,
  );
  const restoredSummaryStore = new DailySummaryCheckpointStore(
    destinationStorage,
    destinationConfig.settings.output,
  );
  await expect(restoredFilterStore.lookupReusable(reportDate, preparedFilterCompatibility))
    .resolves.toEqual(filterResult);
  await expect(restoredSummaryStore.lookupReusable(reportDate, compatibility))
    .resolves.toEqual(result);

  return { archive, sourceRoot, destinationRoot };
}

describe("CLI data portability", () => {
  it("round-trips reusable filter and summary checkpoints under the default active .index root", async () => {
    const { archive } = await exportAndImportCheckpoint();
    expect(archive.file(`.index/filter-checkpoints/${reportDate}.json`)).not.toBeNull();
    expect(archive.file(`.index/daily-summary-checkpoints/${reportDate}.json`)).not.toBeNull();
  });

  it("round-trips a checkpoint under a custom output root and ignores stale default data", async () => {
    const output = {
      dailyDir: "research/reports/daily",
      papersDir: "research/reports/papers",
    };
    const sourceRoot = await tempDir("custom-export");
    const destinationRoot = await tempDir("custom-import");
    const archivePath = path.join(await tempDir("custom-archive"), "vault.zip");
    const sourceConfig = config(sourceRoot, output);
    const destinationConfig = config(destinationRoot, output);
    const activeIndex = derivePaperInboxPaths(output).indexDir;
    const staleRelative = "arxiv-daily/.index/daily-summary-checkpoints/stale.json";
    await fs.mkdir(path.join(sourceRoot, path.dirname(staleRelative)), { recursive: true });
    await fs.writeFile(path.join(sourceRoot, staleRelative), "stale-default");
    await fs.mkdir(path.join(destinationRoot, path.dirname(staleRelative)), { recursive: true });
    await fs.writeFile(path.join(destinationRoot, staleRelative), "keep-stale-default");

    const sourceStorage = new NodeStorageAdapter(sourceRoot);
    await new DailyFilterCheckpointStore(sourceStorage, output)
      .save(reportDate, preparedFilterCompatibility, filterResult);
    await new DailySummaryCheckpointStore(sourceStorage, output)
      .upsert(reportDate, compatibility, result);
    await dataExport(sourceConfig, captureIo().io, archivePath);
    const archive = await JSZip.loadAsync(await fs.readFile(archivePath));
    expect(archive.file(`.index/filter-checkpoints/${reportDate}.json`)).not.toBeNull();
    expect(archive.file(`.index/daily-summary-checkpoints/${reportDate}.json`)).not.toBeNull();
    const manifest = JSON.parse(await archive.file("arxiv-daily-export.json")!.async("string"));
    expect(manifest).toMatchObject({
      formatVersion: 1,
      outputLayout: {
        dailyDir: output.dailyDir,
        papersDir: output.papersDir,
        indexDir: activeIndex,
        exporterVersion: expect.any(String),
      },
    });
    expect(
      Object.values(archive.files).some((entry) => entry.name.includes("stale.json")),
    ).toBe(false);

    await dataImport(destinationConfig, captureIo().io, archivePath, { yes: true });
    const destinationStorage = new NodeStorageAdapter(destinationRoot);
    await expect(new DailyFilterCheckpointStore(destinationStorage, output)
      .lookupReusable(reportDate, preparedFilterCompatibility)).resolves.toEqual(filterResult);
    await expect(new DailySummaryCheckpointStore(destinationStorage, output)
      .lookupReusable(reportDate, compatibility)).resolves.toEqual(result);
    await expect(fs.readFile(path.join(destinationRoot, staleRelative), "utf8"))
      .resolves.toBe("keep-stale-default");
    await expect(
      fs.stat(path.join(destinationRoot, activeIndex, "daily-summary-checkpoints", `${reportDate}.json`)),
    ).resolves.toBeDefined();
  });

  it("rejects a compressed archive larger than the explicit limit before parsing", async () => {
    const vaultRoot = await tempDir("oversized-archive-vault");
    const archivePath = path.join(await tempDir("oversized-archive"), "data.zip");
    await fs.writeFile(archivePath, "not-a-zip");
    await fs.truncate(archivePath, DATA_IMPORT_LIMITS.archiveBytes + 1);

    await expect(dataImport(config(vaultRoot), captureIo().io, archivePath, { yes: true }))
      .rejects.toThrow(/compressed size limit/);
    await expect(fs.readdir(vaultRoot)).resolves.toEqual([]);
  });

  it("rejects excessive entry counts before expanding any entry", async () => {
    const vaultRoot = await tempDir("entry-count-vault");
    const archivePath = path.join(await tempDir("entry-count-archive"), "data.zip");
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({ formatVersion: 1 }));
    for (let index = 0; index < DATA_IMPORT_LIMITS.entryCount; index += 1) {
      zip.file(`daily/${index}.md`, "");
    }
    await fs.writeFile(archivePath, await zip.generateAsync({ type: "nodebuffer" }));

    await expect(dataImport(config(vaultRoot), captureIo().io, archivePath, { yes: true }))
      .rejects.toThrow(/entry count limit/);
    await expect(fs.readdir(vaultRoot)).resolves.toEqual([]);
  });

  it("rejects a forged EOCD record count before JSZip parsing", async () => {
    const vaultRoot = await tempDir("forged-count-vault");
    const archivePath = path.join(await tempDir("forged-count-archive"), "data.zip");
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({ formatVersion: 1 }));
    const archive = await zip.generateAsync({ type: "nodebuffer" });
    const forged = Buffer.from(archive);
    const eocd = eocdOffset(forged);
    forged.writeUInt16LE(DATA_IMPORT_LIMITS.entryCount + 1, eocd + 8);
    forged.writeUInt16LE(DATA_IMPORT_LIMITS.entryCount + 1, eocd + 10);
    await fs.writeFile(archivePath, forged);

    await expect(dataImport(config(vaultRoot), captureIo().io, archivePath, { yes: true }))
      .rejects.toThrow(/entry count limit/);
    await expect(fs.readdir(vaultRoot)).resolves.toEqual([]);
  });

  it("rejects duplicate raw central names before JSZip deduplication", async () => {
    const vaultRoot = await tempDir("duplicate-vault");
    const archivePath = path.join(await tempDir("duplicate-archive"), "data.zip");
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({ formatVersion: 1 }));
    zip.file("daily/a.md", "a");
    zip.file("daily/b.md", "b");
    const archive = renameRawZipEntry(
      await zip.generateAsync({ type: "nodebuffer" }),
      "daily/b.md",
      "daily/a.md",
    );
    await fs.writeFile(archivePath, archive);

    await expect(dataImport(config(vaultRoot), captureIo().io, archivePath, { yes: true }))
      .rejects.toThrow(/duplicate ZIP entry name/);
    await expect(fs.readdir(vaultRoot)).resolves.toEqual([]);
  });

  it("rejects JSZip-equivalent path normalization collisions", async () => {
    const vaultRoot = await tempDir("normalize-vault");
    const archivePath = path.join(await tempDir("normalize-archive"), "data.zip");
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({ formatVersion: 1 }));
    zip.file("daily/a.md", "a");
    zip.file("daily/./a.md", "b");
    await fs.writeFile(archivePath, await zip.generateAsync({ type: "nodebuffer" }));

    await expect(dataImport(config(vaultRoot), captureIo().io, archivePath, { yes: true }))
      .rejects.toThrow(/path normalization collision/);
    await expect(fs.readdir(vaultRoot)).resolves.toEqual([]);
  });

  it("rejects ZIP64 sentinels before JSZip parsing", async () => {
    const vaultRoot = await tempDir("zip64-vault");
    const archivePath = path.join(await tempDir("zip64-archive"), "data.zip");
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({ formatVersion: 1 }));
    const archive = await zip.generateAsync({ type: "nodebuffer" });
    const forged = Buffer.from(archive);
    forged.writeUInt32LE(0xffffffff, eocdOffset(forged) + 16);
    await fs.writeFile(archivePath, forged);

    await expect(dataImport(config(vaultRoot), captureIo().io, archivePath, { yes: true }))
      .rejects.toThrow(/ZIP64 or multi-disk/);
    await expect(fs.readdir(vaultRoot)).resolves.toEqual([]);
  });

  it("rejects mismatched central-directory bounds before JSZip parsing", async () => {
    const vaultRoot = await tempDir("central-mismatch-vault");
    const archivePath = path.join(await tempDir("central-mismatch-archive"), "data.zip");
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({ formatVersion: 1 }));
    const archive = await zip.generateAsync({ type: "nodebuffer" });
    const forged = Buffer.from(archive);
    const eocd = eocdOffset(forged);
    forged.writeUInt32LE(forged.readUInt32LE(eocd + 12) + 1, eocd + 12);
    await fs.writeFile(archivePath, forged);

    await expect(dataImport(config(vaultRoot), captureIo().io, archivePath, { yes: true }))
      .rejects.toThrow(/central directory offset or size/);
    await expect(fs.readdir(vaultRoot)).resolves.toEqual([]);
  });

  it.each(["STORE", "DEFLATE"] as const)(
    "rejects a streamed %s entry whose validated central CRC is wrong before promotion",
    async (compression) => {
      const vaultRoot = await tempDir(`crc-${compression}-vault`);
      const archivePath = path.join(await tempDir(`crc-${compression}-archive`), "data.zip");
      const zip = new JSZip();
      zip.file("arxiv-daily-export.json", JSON.stringify({ formatVersion: 1 }));
      zip.file("daily/report.md", "crc-protected-content");
      let archive = await zip.generateAsync({ type: "nodebuffer", compression });
      archive = forgeZipCrc(archive, "daily/report.md", 0x12345678);
      await fs.writeFile(archivePath, archive);

      await expect(dataImport(config(vaultRoot), captureIo().io, archivePath, { yes: true }))
        .rejects.toThrow(/CRC32 mismatch/);
      await expect(fs.readdir(vaultRoot)).resolves.toEqual([]);
    },
  );

  it("enforces actual streamed entry bytes despite forged-small declared metadata", async () => {
    const vaultRoot = await tempDir("entry-size-vault");
    const archivePath = path.join(await tempDir("entry-size-archive"), "data.zip");
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({ formatVersion: 1 }));
    zip.file("daily/report.md", "x".repeat(4096));
    let archive = await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" });
    archive = forgeZipSizes(archive, "daily/report.md", { uncompressed: 8 });
    await fs.writeFile(archivePath, archive);

    await expect(dataImport(config(vaultRoot), captureIo().io, archivePath, {
      yes: true,
      limits: { entryUncompressedBytes: 1024, compressionRatio: 10_000 },
    })).rejects.toThrow(/entry exceeds uncompressed size limit/);
    await expect(fs.readdir(vaultRoot)).resolves.toEqual([]);
  });

  it("aborts a streamed compression bomb and removes its temp and directories", async () => {
    const vaultRoot = await tempDir("ratio-vault");
    const archivePath = path.join(await tempDir("ratio-archive"), "data.zip");
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({ formatVersion: 1 }));
    zip.file("daily/report.md", "0".repeat(64 * 1024));
    await fs.writeFile(
      archivePath,
      await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }),
    );

    await expect(dataImport(config(vaultRoot), captureIo().io, archivePath, {
      yes: true,
      limits: { compressionRatio: 2 },
    })).rejects.toThrow(/compression ratio limit/);
    await expect(fs.readdir(vaultRoot)).resolves.toEqual([]);
  });

  it("enforces cumulative actual streamed bytes across entries", async () => {
    const vaultRoot = await tempDir("total-size-vault");
    const archivePath = path.join(await tempDir("total-size-archive"), "data.zip");
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({ formatVersion: 1 }));
    zip.file("daily/a.md", "a".repeat(700));
    zip.file("daily/b.md", "b".repeat(700));
    await fs.writeFile(
      archivePath,
      await zip.generateAsync({ type: "nodebuffer", compression: "STORE" }),
    );

    await expect(dataImport(config(vaultRoot), captureIo().io, archivePath, {
      yes: true,
      limits: {
        entryUncompressedBytes: 1024,
        totalUncompressedBytes: 1200,
        compressionRatio: 10_000,
      },
    })).rejects.toThrow(/total uncompressed size limit/);
    await expect(fs.readdir(vaultRoot)).resolves.toEqual([]);
  });

  it("replaces an existing hardlink without modifying its external inode", async () => {
    const vaultRoot = await tempDir("hardlink-vault");
    const outsideRoot = await tempDir("hardlink-outside");
    const archivePath = path.join(await tempDir("hardlink-archive"), "data.zip");
    const target = path.join(vaultRoot, "arxiv-daily/daily/report.md");
    const outside = path.join(outsideRoot, "outside.md");
    await fs.mkdir(path.dirname(target), { recursive: true });
    await fs.writeFile(outside, "outside-original");
    await fs.link(outside, target);
    const old = new Date("2020-01-01T00:00:00.000Z");
    await fs.utimes(target, old, old);
    const before = await fs.stat(outside);

    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({
      formatVersion: 1,
      outputLayout: {
        dailyDir: "arxiv-daily/daily",
        papersDir: "arxiv-daily/papers",
        indexDir: "arxiv-daily/.index",
      },
    }));
    zip.file("daily/report.md", "imported-new-content", { date: new Date() });
    await fs.writeFile(archivePath, await zip.generateAsync({ type: "nodebuffer" }));

    await dataImport(config(vaultRoot), captureIo().io, archivePath, { yes: true });

    expect(await fs.readFile(outside, "utf8")).toBe("outside-original");
    expect(await fs.readFile(target, "utf8")).toBe("imported-new-content");
    const afterOutside = await fs.stat(outside);
    const afterTarget = await fs.stat(target);
    expect(afterOutside.ino).toBe(before.ino);
    expect(afterTarget.ino).not.toBe(afterOutside.ino);
    expect(afterTarget.mode & 0o777).toBe(0o600);
  });

  it("rolls back earlier created and overwritten files when a later promotion fails", async () => {
    const vaultRoot = await tempDir("promotion-rollback-vault");
    const outsideRoot = await tempDir("promotion-rollback-outside");
    const archivePath = path.join(await tempDir("promotion-rollback-archive"), "data.zip");
    const overwritten = path.join(vaultRoot, "arxiv-daily/daily/a.md");
    const created = path.join(vaultRoot, "arxiv-daily/daily/b.md");
    const later = path.join(vaultRoot, "arxiv-daily/daily/c.md");
    const outside = path.join(outsideRoot, "outside.md");
    await fs.mkdir(path.dirname(overwritten), { recursive: true });
    await fs.writeFile(outside, "old-hardlink-content");
    await fs.link(outside, overwritten);
    const old = new Date("2020-01-01T00:00:00.000Z");
    await fs.utimes(overwritten, old, old);
    const originalInode = (await fs.stat(outside)).ino;
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({
      formatVersion: 1,
      outputLayout: {
        dailyDir: "arxiv-daily/daily",
        papersDir: "arxiv-daily/papers",
        indexDir: "arxiv-daily/.index",
      },
    }));
    zip.file("daily/a.md", "new-a", { date: new Date() });
    zip.file("daily/b.md", "new-b", { date: new Date() });
    zip.file("daily/c.md", "new-c", { date: new Date() });
    await fs.writeFile(archivePath, await zip.generateAsync({ type: "nodebuffer" }));

    await expect(dataImport(config(vaultRoot), captureIo().io, archivePath, {
      yes: true,
      testHooks: {
        beforeRename: (kind, _target, index) => {
          if (kind === "promote" && index === 2) throw new Error("injected third promotion failure");
        },
      },
    })).rejects.toThrow(/injected third promotion failure/);

    expect(await fs.readFile(overwritten, "utf8")).toBe("old-hardlink-content");
    expect(await fs.readFile(outside, "utf8")).toBe("old-hardlink-content");
    expect((await fs.stat(overwritten)).ino).toBe(originalInode);
    expect((await fs.stat(outside)).ino).toBe(originalInode);
    await expect(fs.stat(created)).rejects.toMatchObject({ code: "ENOENT" });
    await expect(fs.stat(later)).rejects.toMatchObject({ code: "ENOENT" });
    const names = await fs.readdir(path.dirname(overwritten));
    expect(names).toEqual(["a.md"]);
    expect(names.some((name) => name.includes(".arxiv-daily-import-") ||
      name.includes(".arxiv-daily-rollback-"))).toBe(false);
  });

  it("reports incomplete rollback and preserves its recoverable backup", async () => {
    const vaultRoot = await tempDir("rollback-failure-vault");
    const archivePath = path.join(await tempDir("rollback-failure-archive"), "data.zip");
    const target = path.join(vaultRoot, "arxiv-daily/daily/a.md");
    await fs.mkdir(path.dirname(target), { recursive: true });
    await fs.writeFile(target, "old-a");
    const old = new Date("2020-01-01T00:00:00.000Z");
    await fs.utimes(target, old, old);
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({
      formatVersion: 1,
      outputLayout: {
        dailyDir: "arxiv-daily/daily",
        papersDir: "arxiv-daily/papers",
        indexDir: "arxiv-daily/.index",
      },
    }));
    zip.file("daily/a.md", "new-a", { date: new Date() });
    zip.file("daily/b.md", "new-b", { date: new Date() });
    await fs.writeFile(archivePath, await zip.generateAsync({ type: "nodebuffer" }));

    await expect(dataImport(config(vaultRoot), captureIo().io, archivePath, {
      yes: true,
      testHooks: {
        beforeRename: (kind, _target, index) => {
          if (kind === "promote" && index === 1) throw new Error("injected promotion failure");
          if (kind === "rollback" && index === 0) throw new Error("injected rollback failure");
        },
      },
    })).rejects.toMatchObject({
      name: "AggregateError",
      message: expect.stringContaining("rollback was incomplete"),
    });

    await expect(fs.stat(target)).rejects.toMatchObject({ code: "ENOENT" });
    const names = await fs.readdir(path.dirname(target));
    const backups = names.filter((name) => name.includes(".arxiv-daily-rollback-"));
    expect(backups).toHaveLength(1);
    expect(await fs.readFile(path.join(path.dirname(target), backups[0]!), "utf8"))
      .toBe("old-a");
    expect(names.some((name) => name.includes(".arxiv-daily-import-"))).toBe(false);
  });

  it("rejects a directory where an archive file would be imported", async () => {
    const vaultRoot = await tempDir("directory-conflict-vault");
    const archivePath = path.join(await tempDir("directory-conflict-archive"), "data.zip");
    const target = path.join(vaultRoot, "arxiv-daily/daily/project");
    await fs.mkdir(target, { recursive: true });
    await fs.writeFile(path.join(target, "keep.md"), "keep");

    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({
      formatVersion: 1,
      outputLayout: {
        dailyDir: "arxiv-daily/daily",
        papersDir: "arxiv-daily/papers",
        indexDir: "arxiv-daily/.index",
      },
    }));
    zip.file("daily/project", "must-not-replace-directory", { date: new Date() });
    await fs.writeFile(archivePath, await zip.generateAsync({ type: "nodebuffer" }));

    await expect(dataImport(config(vaultRoot), captureIo().io, archivePath, { yes: true }))
      .rejects.toThrow(/not a regular file/);
    await expect(fs.readFile(path.join(target, "keep.md"), "utf8")).resolves.toBe("keep");
    await expect(fs.readdir(path.dirname(target))).resolves.toEqual(["project"]);
  });

  it.each([
    {
      name: "default to custom",
      source: {
        dailyDir: "arxiv-daily/daily",
        papersDir: "arxiv-daily/papers",
        indexDir: "arxiv-daily/.index",
      },
      target: { dailyDir: "custom/daily", papersDir: "custom/papers" },
    },
    {
      name: "custom A to custom B",
      source: {
        dailyDir: "source/daily",
        papersDir: "source/papers",
        indexDir: "source/.index",
      },
      target: { dailyDir: "target/daily", papersDir: "target/papers" },
    },
  ])("imports daily files but skips .index when new archive layout changes: $name", async ({ source, target }) => {
    const vaultRoot = await tempDir("layout-mismatch");
    const archivePath = path.join(await tempDir("layout-mismatch-archive"), "data.zip");
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({
      formatVersion: 1,
      outputLayout: source,
    }));
    zip.file("daily/report.md", "report");
    zip.file(".index/checkpoint.json", "checkpoint");
    await fs.writeFile(archivePath, await zip.generateAsync({ type: "nodebuffer" }));

    const io = captureIo();
    await expect(dataImport(config(vaultRoot, target), io.io, archivePath, { yes: true }))
      .resolves.toBe(0);

    await expect(fs.readFile(path.join(vaultRoot, target.dailyDir, "report.md"), "utf8"))
      .resolves.toBe("report");
    expect(io.stderr.join("")).toContain("outputLayout does not match");
    await expect(fs.stat(path.join(
      vaultRoot,
      derivePaperInboxPaths(target).indexDir,
      "checkpoint.json",
    ))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it.each([
    ["forged index", {
      dailyDir: "arxiv-daily/daily",
      papersDir: "arxiv-daily/papers",
      indexDir: "forged/.index",
    }],
    ["noncanonical daily", {
      dailyDir: " arxiv-daily/daily ",
      papersDir: "arxiv-daily/papers",
      indexDir: "arxiv-daily/.index",
    }],
    ["malformed field", {
      dailyDir: "arxiv-daily/daily",
      papersDir: 42,
      indexDir: "arxiv-daily/.index",
    }],
  ])("rejects an invalid new manifest outputLayout: %s", async (_name, layout) => {
    const vaultRoot = await tempDir("invalid-layout");
    const archivePath = path.join(await tempDir("invalid-layout-archive"), "data.zip");
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({
      formatVersion: 1,
      outputLayout: layout,
    }));
    zip.file("daily/report.md", "report");
    await fs.writeFile(archivePath, await zip.generateAsync({ type: "nodebuffer" }));
    const io = captureIo();

    await expect(dataImport(config(vaultRoot), io.io, archivePath, { yes: true }))
      .resolves.toBe(2);
    expect(io.stderr.join("")).toContain("invalid outputLayout");
    await expect(fs.readdir(vaultRoot)).resolves.toEqual([]);
  });

  it("legacy v1 keeps .index compatibility at the default target", async () => {
    const vaultRoot = await tempDir("legacy-default");
    const archivePath = path.join(await tempDir("legacy-default-archive"), "legacy.zip");
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({ formatVersion: 1 }));
    zip.file(".index/daily-summary-checkpoints/legacy.json", "legacy");
    await fs.writeFile(archivePath, await zip.generateAsync({ type: "nodebuffer" }));

    const io = captureIo();
    await dataImport(config(vaultRoot), io.io, archivePath, { yes: true });
    await expect(fs.readFile(path.join(vaultRoot, "arxiv-daily/.index/daily-summary-checkpoints/legacy.json"), "utf8"))
      .resolves.toBe("legacy");
    expect(io.stderr.join("")).not.toContain("skipping .index");
  });

  it("legacy v1 skips .index instead of silently relocating it to a custom target", async () => {
    const vaultRoot = await tempDir("legacy-custom");
    const archivePath = path.join(await tempDir("legacy-custom-archive"), "legacy.zip");
    const output = { dailyDir: "custom/daily", papersDir: "custom/papers" };
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({ formatVersion: 1 }));
    zip.file(".index/daily-summary-checkpoints/legacy.json", "legacy");
    zip.file("daily/report.md", "report");
    await fs.writeFile(archivePath, await zip.generateAsync({ type: "nodebuffer" }));

    const io = captureIo();
    await dataImport(config(vaultRoot, output), io.io, archivePath, { yes: true });
    expect(io.stderr.join("")).toContain("legacy archive has no outputLayout; skipping .index");
    await expect(fs.readFile(path.join(vaultRoot, "custom/daily/report.md"), "utf8")).resolves.toBe("report");
    await expect(fs.stat(path.join(vaultRoot, derivePaperInboxPaths(output).indexDir, "daily-summary-checkpoints/legacy.json")))
      .rejects.toMatchObject({ code: "ENOENT" });
  });

  it("export rejects symlink roots and nested symlink components", async () => {
    const realVault = await tempDir("export-real");
    const parent = await tempDir("export-links");
    const rootLink = path.join(parent, "vault-link");
    await fs.symlink(realVault, rootLink, "dir");
    await expect(dataExport(config(rootLink), captureIo().io, path.join(parent, "root.zip")))
      .rejects.toThrow(/unsafe symlink path/);

    const nestedRoot = await tempDir("export-nested");
    const outside = await tempDir("export-outside");
    await fs.mkdir(path.join(nestedRoot, "arxiv-daily"), { recursive: true });
    await fs.symlink(outside, path.join(nestedRoot, "arxiv-daily", "daily"), "dir");
    await expect(dataExport(config(nestedRoot), captureIo().io, path.join(parent, "nested.zip")))
      .rejects.toThrow(/unsafe symlink path/);
  });

  it.each(["root", "intermediate", "target"])("import rejects a %s symlink", async (kind) => {
    const realVault = await tempDir(`import-${kind}-real`);
    const parent = await tempDir(`import-${kind}-parent`);
    const outside = await tempDir(`import-${kind}-outside`);
    const archivePath = path.join(parent, "archive.zip");
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({
      formatVersion: 1,
      outputLayout: { dailyDir: "arxiv-daily/daily", papersDir: "arxiv-daily/papers", indexDir: "arxiv-daily/.index" },
    }));
    zip.file("daily/nested/report.md", "owned");
    await fs.writeFile(archivePath, await zip.generateAsync({ type: "nodebuffer" }));
    let vaultRoot = realVault;
    if (kind === "root") {
      vaultRoot = path.join(parent, "vault-link");
      await fs.symlink(realVault, vaultRoot, "dir");
    } else if (kind === "intermediate") {
      await fs.mkdir(path.join(realVault, "arxiv-daily"), { recursive: true });
      await fs.symlink(outside, path.join(realVault, "arxiv-daily", "daily"), "dir");
    } else {
      await fs.mkdir(path.join(realVault, "arxiv-daily/daily/nested"), { recursive: true });
      await fs.writeFile(path.join(outside, "report.md"), "outside");
      await fs.symlink(path.join(outside, "report.md"), path.join(realVault, "arxiv-daily/daily/nested/report.md"));
    }
    await expect(dataImport(config(vaultRoot), captureIo().io, archivePath, { yes: true }))
      .rejects.toThrow(/unsafe symlink path/);
  });

  it.each([
    ["traversal", ".index/../../escaped.txt"],
    ["POSIX absolute rest", ".index//tmp/escaped.txt"],
    ["Windows absolute rest", ".index/C:/escaped.txt"],
  ])("rejects a malicious %s archive path without writing outside the vault", async (_case, maliciousPath) => {
    const vaultRoot = await tempDir("malicious-import");
    const archivePath = path.join(await tempDir("malicious-archive"), "malicious.zip");
    const zip = new JSZip();
    zip.file("arxiv-daily-export.json", JSON.stringify({ formatVersion: 1 }));
    zip.file(maliciousPath, "owned");
    await fs.writeFile(archivePath, await zip.generateAsync({ type: "nodebuffer" }));

    await expect(
      dataImport(config(vaultRoot), captureIo().io, archivePath, { yes: true }),
    ).rejects.toThrow(/unsafe archive path/);
    await expect(fs.readdir(vaultRoot)).resolves.toEqual([]);
  });
});
