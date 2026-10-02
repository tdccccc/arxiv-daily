import path from "node:path";
import { modernArxivResources, PaperIndexStore, PaperSearchIndex, queryDashboard, projectDashboardOccurrenceProvenance, projectDashboardOccurrenceNovelty, type PaperIndexEntry, type PaperPriority, type PaperStatus, type PaperSummary, type DashboardOccurrenceProvenance, type DashboardPersonalNovelty, type DashboardSortKey } from "@arxiv-daily/core";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import type { CliRuntimeConfig } from "../config";
import { WorkbenchError, type DocumentEntry, type WorkbenchDocuments } from "./documents";
import { inspectCalendar, type WorkbenchCalendarDay } from "./calendar";
import type { WorkbenchRun } from "./server";

export interface WorkbenchPaper {
  key: string; arxivId: string; title: string; authors: string[]; published: string;
  topics: string[]; category: string; status: PaperStatus; priority: PaperPriority; starred: boolean;
  abstract: string; summary: PaperSummary | null; detailPath: string | null;
  reports: Array<{ path: string; date: string; title: string; available: boolean }>;
  originalUrl: string | null; pdfUrl: string | null;
  provenance: DashboardOccurrenceProvenance | null; novelty: DashboardPersonalNovelty | null;
}
export type PaperScope = "all" | "inbox" | "to_read" | "read" | "starred";
export interface WorkbenchPaperList {
  papers: WorkbenchPaper[]; total: number; offset: number; limit: number; nextOffset: number | null;
  counts: Record<PaperScope, number>; libraryCount: number; topics: string[]; day: WorkbenchCalendarDay | null;
}

/** Host projection only: the shared index remains the owner of paper identity and marks. */
export class WorkbenchPapers {
  private readonly store: PaperIndexStore;
  constructor(private config: CliRuntimeConfig, private documents: WorkbenchDocuments) {
    this.store = new PaperIndexStore(new NodeStorageAdapter(config.vaultRoot), config.settings.output);
  }

  async list(params: URLSearchParams, now: Date, run: WorkbenchRun | null): Promise<WorkbenchPaperList> {
    const scope = params.get("scope") || "all", sort = params.get("sort") || (params.get("q") ? "relevance" : "published");
    const direction = params.get("direction") || (sort === "title" || sort === "priority" ? "asc" : "desc");
    const offset = Number(params.get("offset") ?? 0), limit = Number(params.get("limit") ?? 20), date = params.get("date") || "";
    if (!["all", "inbox", "to_read", "read", "starred"].includes(scope) || !["published", "title", "priority", "relevance"].includes(sort) || !["asc", "desc"].includes(direction) || !Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(limit) || limit < 1 || limit > 100 || (date && !validDate(date))) throw new WorkbenchError(400, "论文列表参数无效。");
    const [{ inbox }, catalog] = await Promise.all([this.store.inspect(), this.documents.list()]);
    const entries = Object.values(inbox.papers);
    const visible = entries.filter(entry => entry.status !== "ignored");
    const dailyDate = (report: string) => catalog.find(item => item.kind === "daily" && item.path === report)?.date || reportDate(report);
    const dated = date ? entries.filter(entry => entry.dailyReports.some(report => this.reportInScope(report) && dailyDate(report) === date)) : entries;
    const result = queryDashboard(dated, { tab: "all", search: params.get("q") || "", topics: params.get("topic") ? [params.get("topic")!] : [], sort: { key: sort as DashboardSortKey, direction: direction as "asc" | "desc" } }, { searchIndex: new PaperSearchIndex(dated), topics: this.config.settings.arxiv.topics });
    const counts = { all: result.rows.length, inbox: 0, to_read: 0, read: 0, starred: 0 };
    for (const { entry } of result.rows) {
      if (entry.status === "inbox" || entry.status === "to_read" || entry.status === "read") counts[entry.status]++;
      if (entry.priority === "high") counts.starred++;
    }
    const rows = result.rows.filter(({ entry }) => scope === "all" || (scope === "starred" ? entry.priority === "high" : entry.status === scope));
    const day = date ? (await inspectCalendar(this.config, { list: async () => catalog }, date.slice(0, 7), now, run)).cells.find(day => day?.date === date) ?? null : null;
    return { papers: rows.slice(offset, offset + limit).map(({ entry }) => this.project(entry, catalog, date)), total: rows.length, offset, limit, nextOffset: offset + limit < rows.length ? offset + limit : null, counts, libraryCount: visible.length, topics: [...new Set(visible.flatMap(entry => [entry.primaryTopic, ...entry.topics]).filter(Boolean))].sort(), day };
  }

  async paper(key: string): Promise<WorkbenchPaper> {
    const [{ inbox }, catalog] = await Promise.all([this.store.inspect(), this.documents.list()]);
    const entry = Object.hasOwn(inbox.papers, key) ? inbox.papers[key] : undefined;
    if (!entry) throw new WorkbenchError(404, "找不到这篇论文，索引可能已改变。");
    return this.project(entry, catalog);
  }

  async mark(body: Record<string, unknown>, beforeWrite?: () => Promise<void>): Promise<WorkbenchPaper> {
    const { key, action, value, expected } = body;
    const statuses = ["inbox", "to_read", "reading", "read", "saved", "ignored"];
    if (typeof key !== "string" || !key || typeof expected !== "string" || (action === "status" ? typeof value !== "string" || !["inbox", "to_read", "read"].includes(String(value)) || !statuses.includes(String(expected)) : action === "star" ? typeof value !== "boolean" || !["low", "normal", "high"].includes(String(expected)) : true)) throw new WorkbenchError(400, "论文标记参数无效。");
    await beforeWrite?.();
    await this.store.mutate(inbox => {
      const entry = Object.hasOwn(inbox.papers, key) ? inbox.papers[key] : undefined;
      if (!entry) throw new WorkbenchError(404, "找不到这篇论文。");
      const current = action === "status" ? entry.status : entry.priority;
      const desired = action === "status" ? value : value ? "high" : entry.priority === "high" ? "normal" : entry.priority;
      if (current === desired) return { result: undefined, changed: false };
      if (current !== expected) throw new WorkbenchError(409, "标记已在其他页面改变，已刷新当前状态，请重试。");
      if (action === "status") entry.status = desired as PaperStatus; else entry.priority = desired as PaperPriority;
      return { result: undefined, changed: true };
    });
    return this.paper(key);
  }

  private reportInScope(value: string): boolean {
    return value.startsWith(`${this.config.settings.output.dailyDir}/`) && !value.includes("\\") && !value.split("/").some(part => part.startsWith(".")) && /\.md$/i.test(value);
  }

  private project(entry: PaperIndexEntry, catalog: DocumentEntry[], date = ""): WorkbenchPaper {
    const reports = [...new Set(entry.dailyReports)].filter(value => this.reportInScope(value)).map(value => {
      const document = catalog.find(item => item.path === value && item.kind === "daily");
      return { path: value, date: document?.date || reportDate(value), title: document?.title || reportDate(value) || path.posix.basename(value), available: Boolean(document) };
    }).sort((a, b) => b.date.localeCompare(a.date));
    const occurrence = { ...entry, dailyReports: reports.filter(report => !date || report.date === date).map(report => report.path) };
    const resources = entry.source === "arxiv" ? modernArxivResources(entry.arxivId) : null;
    return { key: entry.paperKey, arxivId: entry.arxivId, title: entry.title, authors: entry.authors, published: entry.published, topics: [...new Set([entry.primaryTopic, ...entry.topics].filter(Boolean))], category: entry.category, status: entry.status, priority: entry.priority, starred: entry.priority === "high", abstract: entry.abstract || "", summary: entry.summary ?? null, detailPath: catalog.some(item => item.path === entry.paperPath && item.kind === "papers") ? entry.paperPath : null, reports, originalUrl: resources?.absUrl ?? safeHttp(entry.arxivUrl), pdfUrl: resources?.pdfUrl ?? safeHttp(entry.pdfUrl), provenance: projectDashboardOccurrenceProvenance(occurrence, this.config.settings.arxiv.topics).occurrenceProvenance ?? null, novelty: projectDashboardOccurrenceNovelty(occurrence).personalNovelty ?? null };
  }
}
function reportDate(value: string): string { return /^(\d{4}-\d{2}-\d{2})\.md$/.exec(path.posix.basename(value))?.[1] || ""; }
function validDate(value: string): boolean { const time = new Date(value); return /^\d{4}-\d{2}-\d{2}$/.test(value) && Number(value.slice(0, 4)) >= 100 && Number.isFinite(time.valueOf()) && time.toISOString().slice(0, 10) === value; }
function safeHttp(value: string): string | null { try { const url = new URL(value); return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password ? url.href : null; } catch { return null; } }
