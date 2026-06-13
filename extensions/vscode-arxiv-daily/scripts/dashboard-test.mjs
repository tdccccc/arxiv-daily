import { createRequire } from "node:module";
import assert from "node:assert/strict";

const require = createRequire(import.meta.url);
const { buildDashboardState, matchesTab } = require("../src/dashboard-model.js");
const {
  renderDashboardHtml,
  resourceTargetForEntry,
  updatePaperStatus,
} = require("../src/dashboard.js");

const index = {
  schemaVersion: 2,
  updatedAt: "2026-06-13T00:00:00.000Z",
  papers: {
    "2606.00001": entry({
      arxivId: "2606.00001",
      title: "Photometric Redshift Calibration",
      status: "to_read",
      priority: "normal",
      primaryTopic: "photo-z",
      summary: { coreProblem: "calibrating photo-z catalogs" },
    }),
    "2606.00002": entry({
      arxivId: "2606.00002",
      title: "Important Lens Finding",
      status: "to_read",
      priority: "high",
      primaryTopic: "lensing",
    }),
    "2606.00003": entry({
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
assert.equal(matchesTab(index.papers["2606.00002"], "highlight"), true);

const search = buildDashboardState(index, {
  tab: "all",
  search: "photo catalogs",
});
assert.deepEqual(search.rows.map((row) => row.arxivId), ["2606.00001"]);

const storage = memoryStorage(index);
await updatePaperStatus(storage, "2606.00001", "reading", () => new Date("2026-06-13T12:00:00.000Z"));
const saved = JSON.parse(await storage.readText("arxiv-daily/.index/papers.json"));
assert.equal(saved.papers["2606.00001"].status, "reading");
assert.equal(saved.papers["2606.00001"].updated, "2026-06-13");
assert.equal(saved.updatedAt, "2026-06-13T12:00:00.000Z");

assert.equal(
  resourceTargetForEntry(index.papers["2606.00001"], "note").value,
  "arxiv-daily/papers/2606.00001.md",
);
assert.equal(
  resourceTargetForEntry(index.papers["2606.00001"], "pdf").value,
  "https://arxiv.org/pdf/2606.00001",
);

const html = renderDashboardHtml({
  nonce: "test",
  workspaceName: "vault",
  state: buildDashboardState({
    ...index,
    papers: {
      "2606.99999": entry({
        arxivId: "2606.99999",
        title: "</script><script>alert(1)</script>",
      }),
    },
  }),
});
assert(html.includes('script nonce="test"'));
assert(!html.includes("</script><script>alert(1)</script>"));

console.log("arXiv Daily VS Code Dashboard OK");

function entry(overrides = {}) {
  const arxivId = overrides.arxivId ?? "2606.00000";
  return {
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
    normalizePath(path) {
      return path.replace(/\\/g, "/");
    },
    async exists(path) {
      return path === "arxiv-daily/.index/papers.json";
    },
    async readText() {
      return content;
    },
    async writeText(_path, next) {
      content = next;
    },
  };
}
