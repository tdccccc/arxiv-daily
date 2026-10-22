import { describe, expect, it, vi } from "vitest";
import { DEFAULT_SETTINGS, OperationRegistry, PaperIndexStore, StateStore } from "@arxiv-daily/core";
import {
  ARXIV_DAILY_DASHBOARD_VIEW,
  registerDashboardView,
} from "../src/dashboard/view";

const output = { dailyDir: "arxiv/daily", papersDir: "arxiv/papers" };
const TODAY = "2026-10-21";

function makeStorage(initial: Record<string, string>) {
  const files = { ...initial };
  return {
    files,
    storage: {
      normalizePath: (p: string) => p.replace(/\\/g, "/"),
      readText: async (p: string) => {
        if (!(p in files)) throw new Error(`missing ${p}`);
        return files[p]!;
      },
      writeText: async (p: string, c: string) => { files[p] = c; },
      exists: async (p: string) => p in files,
      mkdir: async () => {},
      rename: async (a: string, b: string) => { files[b] = files[a]!; delete files[a]; },
      remove: async (p: string) => { delete files[p]; },
    },
  };
}

function indexEntry(id: string) {
  return {
    arxivId: id, source: "arxiv", title: "T", authors: ["A"], published: "2026-10-21", updated: "2026-10-21",
    category: "astro-ph.CO", categories: ["astro-ph.CO"], topics: [], primaryTopic: "", detail: false,
    status: "inbox", priority: "normal", seenDates: ["2026-10-21"],
    dailyReports: [], discoveryProvenanceByReport: {}, paperPath: null,
    arxivUrl: `https://arxiv.org/abs/${id}`, pdfUrl: `https://arxiv.org/pdf/${id}`, pdfPath: "",
    zoteroKey: "", zoteroUri: "", citationKey: "", projects: [],
  };
}

const papersJson = (ids: string[]) => JSON.stringify({
  schemaVersion: 3,
  updatedAt: "2026-10-21T00:00:00.000Z",
  papers: Object.fromEntries(ids.map((id) => [id, indexEntry(id)])),
});

/** Obsidian's scoped element creation, enough for the dashboard header. */
function installCreateEl(element: HTMLElement): HTMLElement {
  const create = function (
    this: HTMLElement,
    tag: string,
    options?: { cls?: string; text?: string },
  ): HTMLElement {
    const child = this.ownerDocument.createElement(tag);
    if (options?.cls) child.className = options.cls;
    if (options?.text !== undefined) child.textContent = options.text;
    installCreateEl(child);
    this.append(child);
    return child;
  };
  (element as any).createEl = create;
  (element as any).empty = function (this: HTMLElement) { this.replaceChildren(); };
  (element as any).createDiv = function (this: HTMLElement, options?: { cls?: string; text?: string }) {
    return create.call(this, "div", options);
  };
  return element;
}

async function setup() {
  const { files, storage } = makeStorage({
    "arxiv/.index/papers.json": papersJson(["2610.00001"]),
  });
  const durable = { runState: {} as Record<string, unknown> };
  const stateStore = new StateStore(
    async () => ({ runState: structuredClone(durable.runState) as never }),
    async ({ runState }) => { durable.runState = structuredClone(runState); },
  );
  await stateStore.load();
  const operations = new OperationRegistry();
  const vault = {
    getMarkdownFiles: () => Object.keys(files).filter((p) => p.endsWith(".md")).map((path) => ({ path })),
    adapter: { read: async (p: string) => files[p]! },
  };
  let create: ((leaf: unknown) => unknown) | undefined;
  const plugin = {
    app: { vault, workspace: {} },
    settings: { ...DEFAULT_SETTINGS, output },
    registerView: (type: string, creator: (leaf: unknown) => unknown) => {
      if (type === ARXIV_DAILY_DASHBOARD_VIEW) create = creator;
    },
    logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
    recentDates: {
      refreshWithin: async () => ({ completed: true, timedOut: false, snapshot: { refreshedAt: 0 } }),
      hasDate: () => false,
    },
    buildPaperIndex: () => new PaperIndexStore(storage as never, output),
    operations,
    stateStore,
  };
  registerDashboardView(plugin as never);
  const view = create!({}) as unknown as {
    entries: Array<{ arxivId: string }>;
    contentEl: HTMLElement;
    onOpen(): Promise<void>;
    onClose(): Promise<void>;
    reloadIndex(): Promise<void>;
    renderHeader(contentEl: HTMLElement): void;
    render: () => void;
    renderLoading: () => void;
    refreshCalendarDailyReports: () => Promise<void>;
  };
  installCreateEl(view.contentEl);
  // The full page is not under test; the header is, so draw only that.
  view.render = vi.fn(() => { view.contentEl.replaceChildren(); view.renderHeader(view.contentEl); });
  view.renderLoading = vi.fn();
  view.refreshCalendarDailyReports = async () => {};
  const reload = vi.spyOn(view, "reloadIndex");
  const statusLine = () => view.contentEl.querySelector(".arxiv-daily-dashboard__status-line")?.textContent;
  /** What a scheduler run leaves behind: a completed date and the papers it indexed. */
  const completeRun = async (ids: string[]) => {
    await stateStore.setRunning(TODAY);
    files["arxiv/.index/papers.json"] = papersJson(ids);
    await stateStore.setCompleted(TODAY, ids.length);
  };
  return { files, operations, stateStore, view, reload, statusLine, completeRun };
}

describe("an open dashboard while the scheduler runs on its own", () => {
  it("shows the new papers and the new last run once the daily run has ended", async () => {
    const { operations, view, reload, statusLine, completeRun } = await setup();
    const run = operations.begin("daily-run", "Daily run");
    await view.onOpen();
    expect(statusLine()).toBe("Running…");
    expect(view.entries.map((entry) => entry.arxivId)).toEqual(["2610.00001"]);
    expect(reload).toHaveBeenCalledTimes(1);

    await completeRun(["2610.00001", "2610.00002"]);
    run.finish();

    await vi.waitFor(() => expect(reload).toHaveBeenCalledTimes(2));
    await vi.waitFor(() => expect(view.entries.map((entry) => entry.arxivId)).toEqual(["2610.00001", "2610.00002"]));
    expect(statusLine()).toBe(`Last run: ${TODAY}`);
  });

  it("also refreshes a run that began after the dashboard was opened", async () => {
    const { operations, view, reload, statusLine, completeRun } = await setup();
    await view.onOpen();
    expect(statusLine()).toBe("Last run: never");

    const run = operations.begin("daily-run", "Daily run");
    expect(statusLine()).toBe("Running…");
    expect(reload).toHaveBeenCalledTimes(1);
    await completeRun(["2610.00001", "2610.00002"]);
    run.finish();

    await vi.waitFor(() => expect(reload).toHaveBeenCalledTimes(2));
    await vi.waitFor(() => expect(statusLine()).toBe(`Last run: ${TODAY}`));
  });

  it("leaves the page alone when a daily run ended without changing anything", async () => {
    const { operations, view, reload, statusLine } = await setup();
    await view.onOpen();

    // A timer tick on a weekend or after the day is done opens and closes a batch and runs nothing.
    const tick = operations.begin("daily-run", "Daily run");
    expect(statusLine()).toBe("Running…");
    tick.finish();

    expect(statusLine()).toBe("Last run: never");
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it("stops listening when the dashboard is closed", async () => {
    const { operations, view, reload, completeRun } = await setup();
    const subscribe = operations.subscribe.bind(operations);
    const unsubscribed = vi.fn();
    vi.spyOn(operations, "subscribe").mockImplementation((listener) => {
      const unsubscribe = subscribe(listener);
      return () => { unsubscribed(); unsubscribe(); };
    });
    await view.onOpen();
    expect(unsubscribed).not.toHaveBeenCalled();
    await view.onClose();
    expect(unsubscribed).toHaveBeenCalledTimes(1);

    const run = operations.begin("daily-run", "Daily run");
    await completeRun(["2610.00001", "2610.00002"]);
    run.finish();

    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it("only updates the status line when a task that is not a daily run ends", async () => {
    const { operations, view, reload, statusLine } = await setup();
    await view.onOpen();

    const download = operations.begin("pdf-download", "PDF download", "2610.00001");
    expect(statusLine()).toBe("Running…");
    download.finish();

    expect(statusLine()).toBe("Last run: never");
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(reload).toHaveBeenCalledTimes(1);
  });
});
