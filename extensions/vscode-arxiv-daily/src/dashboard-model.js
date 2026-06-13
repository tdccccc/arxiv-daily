const DASHBOARD_TABS = [
  { id: "watch", label: "Watch" },
  { id: "highlight", label: "Highlight" },
  { id: "reading", label: "Reading" },
  { id: "saved", label: "Saved" },
  { id: "read", label: "Read" },
  { id: "all", label: "All" },
  { id: "ignored", label: "Ignored" },
];

const PAPER_STATUSES = ["inbox", "to_read", "reading", "read", "saved", "ignored"];
const PAPER_PRIORITIES = ["high", "normal", "low"];
const STATUS_ORDER = {
  to_read: 0,
  reading: 1,
  saved: 2,
  inbox: 3,
  read: 4,
  ignored: 5,
};
const PRIORITY_ORDER = {
  high: 0,
  normal: 1,
  low: 2,
};

function buildDashboardState(index, query = {}) {
  const entries = Object.entries(index?.papers ?? {}).map(([arxivId, entry]) =>
    normalizeEntry(arxivId, entry),
  );
  const allRows = entries.map(toRow).sort(compareRows);
  const rows = entries
    .filter((entry) => matchesQuery(entry, query))
    .map(toRow)
    .sort(compareRows);
  return {
    query: {
      tab: query.tab || "watch",
      search: query.search || "",
      status: query.status || "",
      priority: query.priority || "",
    },
    allRows,
    rows,
    stats: buildStats(rows.map((row) => row.entry)),
    tabCounts: buildTabCounts(entries),
    tabs: DASHBOARD_TABS,
    statuses: PAPER_STATUSES,
    priorities: PAPER_PRIORITIES,
  };
}

function matchesQuery(entry, query = {}) {
  if (!matchesTab(entry, query.tab || "watch")) return false;
  if (query.status && entry.status !== query.status) return false;
  if (query.priority && entry.priority !== query.priority) return false;
  return matchesSearch(entry, query.search || "");
}

function matchesTab(entry, tab) {
  switch (tab) {
    case "watch":
      return entry.status === "to_read" && entry.priority !== "high";
    case "highlight":
      return entry.status !== "ignored" && entry.priority === "high";
    case "reading":
      return entry.status === "reading";
    case "saved":
      return entry.status === "saved";
    case "read":
      return entry.status === "read";
    case "all":
      return entry.status !== "ignored";
    case "ignored":
      return entry.status === "ignored";
    default:
      return false;
  }
}

function buildStats(entries) {
  const stats = {
    total: entries.length,
    saved: 0,
    missingCitation: 0,
    missingZotero: 0,
    statusCounts: emptyCounts(PAPER_STATUSES),
    priorityCounts: emptyCounts(PAPER_PRIORITIES),
  };
  for (const entry of entries) {
    stats.statusCounts[entry.status] += 1;
    stats.priorityCounts[entry.priority] += 1;
    if (entry.status === "saved") {
      stats.saved += 1;
      if (!entry.citationKey.trim()) stats.missingCitation += 1;
      if (!entry.zoteroKey.trim() && !entry.zoteroUri.trim()) stats.missingZotero += 1;
    }
  }
  return stats;
}

function buildTabCounts(entries) {
  const counts = {};
  for (const tab of DASHBOARD_TABS) {
    counts[tab.id] = entries.filter((entry) => matchesTab(entry, tab.id)).length;
  }
  return counts;
}

function toRow(entry) {
  return {
    arxivId: entry.arxivId,
    title: entry.title,
    authors: entry.authors.join(", "),
    topic: entry.primaryTopic || entry.topics[0] || "(none)",
    firstSeen: [...entry.seenDates].sort()[0] || entry.published || "",
    hasNote: Boolean(entry.paperPath),
    entry,
  };
}

function compareRows(a, b) {
  const priority = PRIORITY_ORDER[a.entry.priority] - PRIORITY_ORDER[b.entry.priority];
  if (priority !== 0) return priority;
  const firstSeen = b.firstSeen.localeCompare(a.firstSeen);
  if (firstSeen !== 0) return firstSeen;
  const published = b.entry.published.localeCompare(a.entry.published);
  if (published !== 0) return published;
  return a.title.localeCompare(b.title);
}

function matchesSearch(entry, search) {
  const tokens = search.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return true;
  const haystack = searchableText(entry);
  return tokens.every((token) => haystack.includes(token));
}

function searchableText(entry) {
  return [
    entry.arxivId,
    entry.title,
    ...entry.authors,
    entry.primaryTopic,
    ...entry.topics,
    entry.category,
    ...entry.categories,
    entry.zoteroKey,
    entry.zoteroUri,
    entry.summary?.coreProblem,
    entry.summary?.keyMethod,
    entry.summary?.mainResult,
    entry.summary?.whyRelevant,
    entry.summary?.limitations,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function normalizeEntry(arxivId, entry) {
  return {
    arxivId: stringOr(entry?.arxivId, arxivId),
    title: stringOr(entry?.title, arxivId),
    authors: arrayOfStrings(entry?.authors),
    published: stringOr(entry?.published, ""),
    updated: stringOr(entry?.updated, ""),
    category: stringOr(entry?.category, ""),
    categories: arrayOfStrings(entry?.categories),
    summary: entry?.summary && typeof entry.summary === "object" ? entry.summary : {},
    topics: arrayOfStrings(entry?.topics),
    primaryTopic: stringOr(entry?.primaryTopic, ""),
    detail: Boolean(entry?.detail),
    status: PAPER_STATUSES.includes(entry?.status) ? entry.status : "inbox",
    priority: PAPER_PRIORITIES.includes(entry?.priority) ? entry.priority : "normal",
    seenDates: arrayOfStrings(entry?.seenDates),
    dailyReports: arrayOfStrings(entry?.dailyReports),
    paperPath: entry?.paperPath ? String(entry.paperPath) : "",
    arxivUrl: stringOr(entry?.arxivUrl, ""),
    pdfUrl: stringOr(entry?.pdfUrl, ""),
    pdfPath: stringOr(entry?.pdfPath, ""),
    zoteroKey: stringOr(entry?.zoteroKey, ""),
    zoteroUri: stringOr(entry?.zoteroUri, ""),
    citationKey: stringOr(entry?.citationKey, ""),
    projects: arrayOfStrings(entry?.projects),
  };
}

function emptyCounts(keys) {
  return Object.fromEntries(keys.map((key) => [key, 0]));
}

function arrayOfStrings(value) {
  return Array.isArray(value) ? value.map((item) => String(item)).filter(Boolean) : [];
}

function stringOr(value, fallback) {
  return typeof value === "string" ? value : fallback;
}

module.exports = {
  DASHBOARD_TABS,
  PAPER_PRIORITIES,
  PAPER_STATUSES,
  buildDashboardState,
  matchesQuery,
  matchesTab,
};
