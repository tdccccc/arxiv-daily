import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import JSZip from "jszip";
import { afterEach, describe, expect, it } from "vitest";
import {
  DailySummaryCheckpointStore,
  DEFAULT_SETTINGS,
  derivePaperInboxPaths,
  type DailyPaperResult,
  type DailySummaryCheckpointCompatibilityInput,
} from "@arxiv-daily/core";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import type { CliRuntimeConfig } from "../src/config";
import { DEFAULT_CLI_SCHEDULE } from "../src/config";
import { dataExport, dataImport } from "../src/data-cmd";

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

async function exportAndImportCheckpoint(output = DEFAULT_SETTINGS.output) {
  const sourceRoot = await tempDir("export");
  const destinationRoot = await tempDir("import");
  const archivePath = path.join(await tempDir("archive"), "vault.zip");
  const sourceConfig = config(sourceRoot, output);
  const destinationConfig = config(destinationRoot, output);
  const sourceStore = new DailySummaryCheckpointStore(
    new NodeStorageAdapter(sourceRoot),
    sourceConfig.settings.output,
  );
  await sourceStore.upsert(reportDate, compatibility, result);

  const exportIo = captureIo();
  await expect(dataExport(sourceConfig, exportIo.io, archivePath)).resolves.toBe(0);
  const archive = await JSZip.loadAsync(await fs.readFile(archivePath));

  const importIo = captureIo();
  await expect(
    dataImport(destinationConfig, importIo.io, archivePath, { yes: true }),
  ).resolves.toBe(0);
  const restoredStore = new DailySummaryCheckpointStore(
    new NodeStorageAdapter(destinationRoot),
    destinationConfig.settings.output,
  );
  await expect(restoredStore.lookupReusable(reportDate, compatibility)).resolves.toEqual(result);

  return { archive, sourceRoot, destinationRoot };
}

describe("CLI data portability", () => {
  it("round-trips a nested checkpoint under the default active .index root", async () => {
    const { archive } = await exportAndImportCheckpoint();
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

    const sourceStore = new DailySummaryCheckpointStore(
      new NodeStorageAdapter(sourceRoot),
      output,
    );
    await sourceStore.upsert(reportDate, compatibility, result);
    await dataExport(sourceConfig, captureIo().io, archivePath);
    const archive = await JSZip.loadAsync(await fs.readFile(archivePath));
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
    const restoredStore = new DailySummaryCheckpointStore(
      new NodeStorageAdapter(destinationRoot),
      output,
    );
    await expect(restoredStore.lookupReusable(reportDate, compatibility)).resolves.toEqual(result);
    await expect(fs.readFile(path.join(destinationRoot, staleRelative), "utf8"))
      .resolves.toBe("keep-stale-default");
    await expect(
      fs.stat(path.join(destinationRoot, activeIndex, "daily-summary-checkpoints", `${reportDate}.json`)),
    ).resolves.toBeDefined();
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
