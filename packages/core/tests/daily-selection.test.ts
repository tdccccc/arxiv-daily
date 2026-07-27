import { describe, expect, it } from "vitest";
import {
  applySelectionsToIndex,
  DailySelectionSyncService,
  parseDailySelections,
} from "../src/services/daily-selection";
import { PaperIndexStore, type PaperInbox, type PaperIndexEntry } from "../src/services/paper-index";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { StorageAdapter } from "../src/core/adapters";

function entry(
  id: string,
  status: PaperIndexEntry["status"] = "inbox",
  priority: PaperIndexEntry["priority"] = "normal",
): PaperIndexEntry {
  return {
    paperKey: `arxiv:${id}`,
    source: "arxiv",
    externalId: id,
    arxivId: id,
    title: `Paper ${id}`,
    authors: ["A"],
    published: "2026-06-12",
    updated: "2026-06-12",
    category: "astro-ph",
    topics: ["ml-astro"],
    primaryTopic: "ml-astro",
    detail: false,
    status,
    priority,
    seenDates: ["2026-06-12"],
    dailyReports: ["arxiv-daily/daily/2026-06-12.md"],
    paperPath: null,
    arxivUrl: `https://arxiv.org/abs/${id}`,
    pdfUrl: `https://arxiv.org/pdf/${id}`,
    pdfPath: "",
    zoteroKey: "",
    zoteroUri: "",
    citationKey: "",
    projects: [],
  };
}

function index(entries: PaperIndexEntry[]): PaperInbox {
  return {
    schemaVersion: 4,
    updatedAt: "2026-06-12T00:00:00.000Z",
    papers: Object.fromEntries(entries.map((e) => [e.paperKey, e])),
  };
}

function storageFromVault(vault: any): StorageAdapter {
  return {
    normalizePath(path: string) {
      return path.replace(/\\/g, "/");
    },
    readText: vault.adapter.read,
    writeText: vault.adapter.write,
    exists: vault.adapter.exists,
    mkdir: vault.adapter.mkdir,
    rename: vault.adapter.rename,
    remove: vault.adapter.remove,
  };
}

describe("daily selection", () => {
  it("parses checked watch and highlight controls", () => {
    const selections = parseDailySelections(
      [
        "- [x] 关注 <!-- arxiv-daily:2606.12345:watch -->",
        "- [ ] 重点 <!-- arxiv-daily:2606.12345:highlight -->",
        "- [x] 关注 <!-- arxiv-daily:2606.54321:watch -->",
        "- [x] 重点 <!-- arxiv-daily:2606.54321:highlight -->",
      ].join("\n"),
    );
    expect(selections).toEqual([
      { arxivId: "2606.12345", watch: true, highlight: false },
      { arxivId: "2606.54321", watch: true, highlight: true },
    ]);
  });

  it("keeps standalone legacy controls compatible and ignores inline fake marker prose", () => {
    const selections = parseDailySelections([
      "- [x] Watch <!-- arxiv-daily:2606.10001:watch -->",
      "* [X] Highlight <!-- arxiv-daily:2606.10002:highlight -->",
      "- [x] Watch <!-- arxiv-daily:2606.10003:selection:watch -->",
      "- **Research problem**: prose <!-- arxiv-daily:2606.19999:watch -->",
      "prefix - [x] Watch <!-- arxiv-daily:2606.19998:watch --> suffix",
    ].join("\n"));

    expect(selections).toEqual([
      { arxivId: "2606.10001", watch: true, highlight: false },
      { arxivId: "2606.10002", watch: false, highlight: true },
      { arxivId: "2606.10003", watch: true, highlight: false },
    ]);
  });

  it("accepts historical marker whitespace, both namespaces, and CRLF while retaining strict checkbox lines", () => {
    const markdown = [
      "- [x] Watch <!--arxiv-daily:2606.11001:watch-->",
      "* [X] Highlight <!--  arxiv-daily:2606.11002:highlight  -->",
      "- [ ] Watch <!--\tarxiv-daily:2606.11003:selection:watch\t-->",
      "- [x] Highlight <!-- arxiv-daily:2606.11003:selection:highlight -->",
      "- [x] Watch <!-- arxiv-daily:2606.11004:watch -->",
      "prefix - [x] Watch <!--  arxiv-daily:2606.11998:watch  -->",
      "- **Research problem**: fake <!-- arxiv-daily:2606.11999:highlight -->",
    ].join("\r\n");

    expect(parseDailySelections(markdown)).toEqual([
      { arxivId: "2606.11001", watch: true, highlight: false },
      { arxivId: "2606.11002", watch: false, highlight: true },
      { arxivId: "2606.11003", watch: false, highlight: true },
      { arxivId: "2606.11004", watch: true, highlight: false },
    ]);
  });

  it("applies selections as to_read priorities and can clear plugin to_read state", () => {
    const data = index([
      entry("2606.12345"),
      entry("2606.54321"),
      entry("2606.99999", "to_read", "high"),
      entry("2606.88888", "saved", "high"),
      entry("2606.77777", "read", "normal"),
      entry("2606.66666", "ignored", "normal"),
    ]);
    const result = applySelectionsToIndex(data, [
      { arxivId: "2606.12345", watch: true, highlight: false },
      { arxivId: "2606.54321", watch: true, highlight: true },
      { arxivId: "2606.99999", watch: false, highlight: false },
      { arxivId: "2606.88888", watch: false, highlight: false },
      { arxivId: "2606.77777", watch: true, highlight: false },
      { arxivId: "2606.66666", watch: true, highlight: true },
    ]);
    expect(result.changed).toBe(3);
    expect(data.papers["arxiv:2606.12345"].status).toBe("to_read");
    expect(data.papers["arxiv:2606.12345"].priority).toBe("normal");
    expect(data.papers["arxiv:2606.54321"].status).toBe("to_read");
    expect(data.papers["arxiv:2606.54321"].priority).toBe("high");
    expect(data.papers["arxiv:2606.99999"].status).toBe("inbox");
    expect(data.papers["arxiv:2606.99999"].priority).toBe("normal");
    expect(data.papers["arxiv:2606.88888"].status).toBe("saved");
    expect(data.papers["arxiv:2606.88888"].priority).toBe("high");
    expect(data.papers["arxiv:2606.77777"].status).toBe("read");
    expect(data.papers["arxiv:2606.77777"].priority).toBe("normal");
    expect(data.papers["arxiv:2606.66666"].status).toBe("ignored");
    expect(data.papers["arxiv:2606.66666"].priority).toBe("normal");
  });

  it("syncs a daily file into papers.json", async () => {
    const files: Record<string, string> = {
      "arxiv-daily/daily/2026-06-12.md": [
        "- [x] 关注 <!-- arxiv-daily:2606.12345:watch -->",
        "- [x] 重点 <!-- arxiv-daily:2606.12345:highlight -->",
      ].join("\n"),
    };
    const dirs = new Set<string>();
    const vault = {
      adapter: {
        async read(path: string) {
          return files[path];
        },
        async write(path: string, content: string) {
          files[path] = content;
        },
        async exists(path: string) {
          return path in files || dirs.has(path);
        },
        async mkdir(path: string) {
          dirs.add(path);
        },
        async rename(from: string, to: string) {
          files[to] = files[from];
          delete files[from];
        },
        async remove(path: string) {
          delete files[path];
          dirs.delete(path);
        },
      },
    };
    const store = new PaperIndexStore(
      storageFromVault(vault),
      DEFAULT_SETTINGS.output,
      () => new Date("2026-06-12T00:00:00.000Z"),
    );
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "A paper",
      authors: "A",
      date: "2026-06-12",
      arxivCategory: "astro-ph",
      primaryTopic: "ml-astro",
      detail: false,
    });
    const sync = new DailySelectionSyncService({
      storage: storageFromVault(vault),
      getOutput: () => DEFAULT_SETTINGS.output,
      buildPaperIndex: () => store,
      logger: new Logger("error"),
      debounceMs: 1,
    });
    const result = await sync.syncPath("arxiv-daily/daily/2026-06-12.md");
    expect(result?.changed).toBe(1);
    const saved = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    expect(saved.papers["arxiv:2606.12345"].status).toBe("to_read");
    expect(saved.papers["arxiv:2606.12345"].priority).toBe("high");
  });

  it("syncs recent daily files on startup without clearing saved papers", async () => {
    const files: Record<string, string> = {
      "arxiv-daily/daily/2026-06-11.md": [
        "- [x] 关注 <!-- arxiv-daily:2606.12345:watch -->",
        "- [ ] 重点 <!-- arxiv-daily:2606.12345:highlight -->",
        "- [x] 关注 <!-- arxiv-daily:2606.77777:watch -->",
        "- [x] 重点 <!-- arxiv-daily:2606.77777:highlight -->",
      ].join("\n"),
      "arxiv-daily/daily/2026-06-12.md": [
        "- [ ] 关注 <!-- arxiv-daily:2606.12345:watch -->",
        "- [ ] 重点 <!-- arxiv-daily:2606.12345:highlight -->",
        "- [x] 关注 <!-- arxiv-daily:2606.54321:watch -->",
        "- [x] 重点 <!-- arxiv-daily:2606.54321:highlight -->",
      ].join("\n"),
    };
    const dirs = new Set<string>();
    const vault = {
      adapter: {
        async read(path: string) {
          return files[path];
        },
        async write(path: string, content: string) {
          files[path] = content;
        },
        async exists(path: string) {
          return path in files || dirs.has(path);
        },
        async mkdir(path: string) {
          dirs.add(path);
        },
        async rename(from: string, to: string) {
          files[to] = files[from];
          delete files[from];
        },
        async remove(path: string) {
          delete files[path];
          dirs.delete(path);
        },
      },
    };
    const store = new PaperIndexStore(
      storageFromVault(vault),
      DEFAULT_SETTINGS.output,
      () => new Date("2026-06-12T00:00:00.000Z"),
    );
    await store.upsertManyFromDailyPapers([
      {
        arxivId: "2606.12345",
        title: "A paper",
        authors: "A",
        date: "2026-06-11",
        arxivCategory: "astro-ph",
        primaryTopic: "ml-astro",
        detail: false,
      },
      {
        arxivId: "2606.54321",
        title: "Another paper",
        authors: "B",
        date: "2026-06-12",
        arxivCategory: "astro-ph",
        primaryTopic: "ml-astro",
        detail: false,
      },
      {
        arxivId: "2606.77777",
        title: "Saved paper",
        authors: "C",
        date: "2026-06-11",
        arxivCategory: "astro-ph",
        primaryTopic: "ml-astro",
        detail: false,
      },
    ]);
    await store.setStatus("2606.77777", "saved");
    await store.setPriority("2606.77777", "high");

    const sync = new DailySelectionSyncService({
      storage: storageFromVault(vault),
      getOutput: () => DEFAULT_SETTINGS.output,
      getLookbackDays: () => 2,
      getTimezone: () => "Asia/Shanghai",
      now: () => new Date("2026-06-12T12:00:00+08:00"),
      buildPaperIndex: () => store,
      logger: new Logger("error"),
      debounceMs: 1,
    });

    const result = await sync.syncRecentDailyFiles();

    expect(result.scanned).toBe(2);
    expect(result.paths).toEqual([
      "arxiv-daily/daily/2026-06-11.md",
      "arxiv-daily/daily/2026-06-12.md",
    ]);
    expect(result.changed).toBe(2);
    const saved = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    expect(saved.papers["arxiv:2606.12345"].status).toBe("to_read");
    expect(saved.papers["arxiv:2606.12345"].priority).toBe("normal");
    expect(saved.papers["arxiv:2606.54321"].status).toBe("to_read");
    expect(saved.papers["arxiv:2606.54321"].priority).toBe("high");
    expect(saved.papers["arxiv:2606.77777"].status).toBe("saved");
    expect(saved.papers["arxiv:2606.77777"].priority).toBe("high");

    const second = await sync.syncRecentDailyFiles();

    expect(second.changed).toBe(0);
    const savedAgain = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    expect(savedAgain.papers["arxiv:2606.77777"].status).toBe("saved");
    expect(savedAgain.papers["arxiv:2606.77777"].priority).toBe("high");
  });

  it("startup sync is a no-op without recent daily files or an index", async () => {
    const files: Record<string, string> = {};
    const dirs = new Set<string>();
    const vault = {
      adapter: {
        async read(path: string) {
          return files[path];
        },
        async write(path: string, content: string) {
          files[path] = content;
        },
        async exists(path: string) {
          return path in files || dirs.has(path);
        },
        async mkdir(path: string) {
          dirs.add(path);
        },
        async rename(from: string, to: string) {
          files[to] = files[from];
          delete files[from];
        },
        async remove(path: string) {
          delete files[path];
          dirs.delete(path);
        },
      },
    };
    const store = new PaperIndexStore(
      storageFromVault(vault),
      DEFAULT_SETTINGS.output,
      () => new Date("2026-06-12T00:00:00.000Z"),
    );
    const sync = new DailySelectionSyncService({
      storage: storageFromVault(vault),
      getOutput: () => DEFAULT_SETTINGS.output,
      getLookbackDays: () => 2,
      getTimezone: () => "Asia/Shanghai",
      now: () => new Date("2026-06-12T12:00:00+08:00"),
      buildPaperIndex: () => store,
      logger: new Logger("error"),
      debounceMs: 1,
    });

    const result = await sync.syncRecentDailyFiles();

    expect(result).toEqual({
      found: 0,
      changed: 0,
      missing: [],
      scanned: 0,
      paths: [],
    });
    expect(files["arxiv-daily/.index/papers.json"]).toBeUndefined();
  });
});
