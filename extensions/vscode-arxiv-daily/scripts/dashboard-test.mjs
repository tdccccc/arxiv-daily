import { createRequire } from "node:module";
import assert from "node:assert/strict";

const require = createRequire(import.meta.url);
const { buildDashboardState, matchesTab } = require("../src/dashboard-model.js");
const {
  renderDashboardHtml,
  resourceTargetForEntry,
  updatePaperStatus,
} = require("../src/dashboard.js");

// Schema-5 shape written by core's PaperIndexStore: entries are keyed
// "arxiv:<id>" and carry paperKey/source/externalId.
const index = {
  schemaVersion: 5,
  updatedAt: "2026-06-13T00:00:00.000Z",
  papers: {
    "arxiv:2606.00001": entry({
      arxivId: "2606.00001",
      title: "Photometric Redshift Calibration",
      status: "to_read",
      priority: "normal",
      primaryTopic: "photo-z",
      summary: { coreProblem: "calibrating photo-z catalogs" },
    }),
    "arxiv:2606.00002": entry({
      arxivId: "2606.00002",
      title: "Important Lens Finding",
      status: "to_read",
      priority: "high",
      primaryTopic: "lensing",
    }),
    "arxiv:2606.00003": entry({
      arxivId: "2606.00003",
      title: "Finished Survey Paper",
      status: "saved",
      priority: "low",
      primaryTopic: "survey",
      citationKey: "",
      zoteroKey: "",
    }),
  },
};

const watch = buildDashboardState(index, { tab: "watch" });
assert.deepEqual(watch.rows.map((row) => row.arxivId), ["2606.00001"]);
assert.deepEqual(watch.allRows.map((row) => row.arxivId), [
  "2606.00002",
  "2606.00001",
  "2606.00003",
]);
assert.equal(watch.tabCounts.highlight, 1);
assert.equal(watch.tabCounts.saved, 1);
assert.equal(matchesTab(index.papers["arxiv:2606.00002"], "highlight"), true);

const search = buildDashboardState(index, {
  tab: "all",
  search: "photo catalogs",
});
assert.deepEqual(search.rows.map((row) => row.arxivId), ["2606.00001"]);

assert.deepEqual(watch.allRows.map((row) => row.key), [
  "arxiv:2606.00002",
  "arxiv:2606.00001",
  "arxiv:2606.00003",
]);

// Status updates resolve by map key, by bare arXiv id, and by prefixed id.
const storage = memoryStorage(index);
await updatePaperStatus(storage, "arxiv:2606.00001", "reading", () => new Date("2026-06-13T12:00:00.000Z"));
let saved = JSON.parse(await storage.readText("arxiv-daily/.index/papers.json"));
assert.equal(saved.papers["arxiv:2606.00001"].status, "reading");
assert.equal(saved.papers["arxiv:2606.00001"].paperKey, "arxiv:2606.00001");
assert.equal(saved.schemaVersion, 5);
assert.equal(saved.updatedAt, "2026-06-13T12:00:00.000Z");
assert.equal(saved.papers["arxiv:2606.00002"].status, "to_read");
await updatePaperStatus(storage, "2606.00002", "saved");
saved = JSON.parse(await storage.readText("arxiv-daily/.index/papers.json"));
assert.equal(saved.papers["arxiv:2606.00002"].status, "saved");
assert.equal(storage.atomicWrites, 2, "papers.json must be replaced atomically");
await assert.rejects(updatePaperStatus(storage, "2606.99999", "read"), /not in papers\.json/);
await assert.rejects(updatePaperStatus(storage, "__proto__", "read"), /not in papers\.json/);
await assert.rejects(updatePaperStatus(storage, "2606.00001", "bogus"), /invalid status/);

// Pre-schema-4 files keyed by the bare id still work.
const legacyStorage = memoryStorage({
  schemaVersion: 3,
  updatedAt: "2026-06-13T00:00:00.000Z",
  papers: { "2606.00001": { ...entry({ arxivId: "2606.00001" }), paperKey: undefined } },
});
await updatePaperStatus(legacyStorage, "2606.00001", "read");
assert.equal(
  JSON.parse(await legacyStorage.readText("arxiv-daily/.index/papers.json")).papers["2606.00001"].status,
  "read",
);

// A format the extension does not know is never rewritten.
const futureStorage = memoryStorage({ ...index, schemaVersion: 99 });
const futureBefore = await futureStorage.readText("arxiv-daily/.index/papers.json");
await assert.rejects(updatePaperStatus(futureStorage, "2606.00001", "read"), /schemaVersion 99 is not supported/);
assert.equal(await futureStorage.readText("arxiv-daily/.index/papers.json"), futureBefore);
assert.equal(futureStorage.atomicWrites, 0);

assert.equal(
  resourceTargetForEntry(index.papers["arxiv:2606.00001"], "note").value,
  "arxiv-daily/papers/2606.00001.md",
);
assert.equal(
  resourceTargetForEntry(index.papers["arxiv:2606.00001"], "pdf").value,
  "https://arxiv.org/pdf/2606.00001",
);
assert.equal(
  resourceTargetForEntry(index.papers["arxiv:2606.00001"], "arxiv").value,
  "https://arxiv.org/abs/2606.00001",
);

// papers.json may come from a cloned repository: it must not pick the URI
// scheme or host handed to openExternal.
const hostile = entry({
  arxivId: "2606.00004",
  arxivUrl: "vscode://file/etc/passwd",
  pdfUrl: "https://evil.example/paper.pdf",
});
assert.equal(resourceTargetForEntry(hostile, "arxiv").value, "https://arxiv.org/abs/2606.00004");
assert.equal(resourceTargetForEntry(hostile, "pdf").value, "https://arxiv.org/pdf/2606.00004");
const hostileNoId = entry({
  arxivId: "not-an-id",
  arxivUrl: "file:///etc/passwd",
  pdfUrl: "http://arxiv.org/pdf/x",
});
assert.equal(resourceTargetForEntry(hostileNoId, "arxiv"), null);
assert.equal(resourceTargetForEntry(hostileNoId, "pdf"), null);

const html = renderDashboardHtml({
  nonce: "test",
  workspaceName: "vault",
  state: buildDashboardState({
    ...index,
    papers: {
      "arxiv:2606.99999": entry({
        arxivId: "2606.99999",
        title: "</script><script>alert(1)</script>",
      }),
    },
  }),
});
assert(html.includes('script nonce="test"'));
assert(!html.includes("</script><script>alert(1)</script>"));

// The page script must stay syntactically valid and accept state updates by message.
const script = /<script nonce="test">([\s\S]*)<\/script>/.exec(html)?.[1];
assert(script, "dashboard page script should be present");
assert.doesNotThrow(() => new Function("acquireVsCodeApi", "document", "window", script));
assert(script.includes('addEventListener("message"'));

console.log("arXiv Daily VS Code Dashboard OK");

function entry(overrides = {}) {
  const arxivId = overrides.arxivId ?? "2606.00000";
  return {
    paperKey: `arxiv:${arxivId}`,
    source: "arxiv",
    externalId: arxivId,
    arxivId,
    title: overrides.title ?? "Example Paper",
    authors: overrides.authors ?? ["Jane Doe"],
    published: overrides.published ?? "2026-06-13",
    updated: overrides.updated ?? "2026-06-13",
    category: overrides.category ?? "astro-ph.CO",
    categories: overrides.categories ?? ["astro-ph.CO"],
    summary: overrides.summary ?? {},
    topics: overrides.topics ?? [overrides.primaryTopic ?? "photo-z"],
    primaryTopic: overrides.primaryTopic ?? "photo-z",
    detail: overrides.detail ?? false,
    status: overrides.status ?? "to_read",
    priority: overrides.priority ?? "normal",
    seenDates: overrides.seenDates ?? ["2026-06-13"],
    discoveryProvenanceByReport: {},
    dailyReports: overrides.dailyReports ?? ["arxiv-daily/daily/2026-06-13.md"],
    paperPath: overrides.paperPath ?? `arxiv-daily/papers/${arxivId}.md`,
    arxivUrl: overrides.arxivUrl ?? `https://arxiv.org/abs/${arxivId}`,
    pdfUrl: overrides.pdfUrl ?? `https://arxiv.org/pdf/${arxivId}`,
    pdfPath: overrides.pdfPath ?? "",
    zoteroKey: overrides.zoteroKey ?? "Doe2026",
    zoteroUri: overrides.zoteroUri ?? "",
    citationKey: overrides.citationKey ?? "Doe2026",
    projects: overrides.projects ?? [],
  };
}

function memoryStorage(initialIndex) {
  let content = JSON.stringify(initialIndex, null, 2);
  return {
    atomicWrites: 0,
    normalizePath(path) {
      return path.replace(/\\/g, "/");
    },
    async exists(path) {
      return path === "arxiv-daily/.index/papers.json";
    },
    async readText() {
      return content;
    },
    async writeText() {
      throw new Error("papers.json must be written through writeTextAtomic");
    },
    async writeTextAtomic(_path, next) {
      this.atomicWrites += 1;
      content = next;
    },
  };
}
