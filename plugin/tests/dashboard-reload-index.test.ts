import { describe, expect, it, vi } from "vitest";
import { DEFAULT_SETTINGS, PaperIndexStore } from "@arxiv-daily/core";
import {
  ARXIV_DAILY_DASHBOARD_VIEW,
  registerDashboardView,
} from "../src/dashboard/view";

const output = { dailyDir: "arxiv/daily", papersDir: "arxiv/papers" };

function makeStorage(initial: Record<string, string>) {
  const files = { ...initial };
  const dirs = new Set<string>();
  return {
    files,
    storage: {
      normalizePath: (p: string) => p.replace(/\\/g, "/"),
      readText: async (p: string) => {
        if (!(p in files)) throw new Error(`missing ${p}`);
        return files[p]!;
      },
      writeText: async (p: string, c: string) => { files[p] = c; },
      exists: async (p: string) => p in files || dirs.has(p),
      mkdir: async (p: string) => { dirs.add(p); },
      rename: async (a: string, b: string) => { files[b] = files[a]!; delete files[a]; },
      remove: async (p: string) => { delete files[p]; },
    },
  };
}

function indexEntry(id: string) {
  return {
    arxivId: id, source: "arxiv", title: "T", authors: ["A"], published: "2026-06-10", updated: "2026-06-10",
    category: "astro-ph.CO", categories: ["astro-ph.CO"], topics: [], primaryTopic: "", detail: false,
    status: "inbox", priority: "normal", seenDates: ["2026-06-10"],
    dailyReports: ["arxiv/daily/2026-06-10.md"], discoveryProvenanceByReport: {}, paperPath: null,
    arxivUrl: `https://arxiv.org/abs/${id}`, pdfUrl: `https://arxiv.org/pdf/${id}`, pdfPath: "",
    zoteroKey: "", zoteroUri: "", citationKey: "", projects: [],
  };
}

interface RecentDatesStub {
  snapshot: () => {
    status: "idle" | "ready" | "failed";
    dates: Set<string>;
    refreshedAt: number;
    error?: string;
  };
  hasDate: (date: string) => boolean;
  refreshWithin: (timeoutMs: number) => Promise<unknown>;
}

function setup(recentDates?: Partial<RecentDatesStub>) {
  const { files, storage } = makeStorage({
    "arxiv/daily/2026-06-10.md": "## Topic\n### T\n- **Authors**: A\n- **arXiv**: [2606.00001](https://arxiv.org/abs/2606.00001)\n",
    "arxiv/.index/papers.json": JSON.stringify({
      schemaVersion: 3, updatedAt: "2026-06-13T00:00:00.000Z", papers: { "2606.00001": indexEntry("2606.00001") },
    }),
  });
  const vault = {
    getMarkdownFiles: () => Object.keys(files).filter((p) => p.endsWith(".md")).map((path) => ({ path })),
    adapter: { read: async (p: string) => files[p]! },
  };
  let create: ((leaf: unknown) => unknown) | undefined;
  const emptySnapshot = { status: "idle" as const, dates: new Set<string>(), refreshedAt: 0 };
  const plugin = {
    app: { vault, workspace: {} },
    settings: { ...DEFAULT_SETTINGS, output },
    registerView: (type: string, creator: (leaf: unknown) => unknown) => {
      if (type === ARXIV_DAILY_DASHBOARD_VIEW) create = creator;
    },
    logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
    recentDates: {
      refreshWithin: async () => ({ completed: true, timedOut: false, snapshot: emptySnapshot }),
      hasDate: () => false,
      snapshot: () => emptySnapshot,
      ...recentDates,
    },
    buildPaperIndex: () => new PaperIndexStore(storage as never, output),
    operations: { snapshot: () => [], subscribe: vi.fn(() => () => {}) },
    stateStore: { snapshot: () => ({}) },
  };
  registerDashboardView(plugin as never);
  const view = create!({}) as unknown as {
    entries: Array<{ pdfPath: string; status: string }>;
    onOpen(): Promise<void>;
    reloadIndex(): Promise<void>;
    refreshFromVault(): Promise<void>;
    render: () => void;
    renderLoading: () => void;
    refreshCalendarDailyReports: () => Promise<void>;
  };
  view.render = vi.fn();
  view.renderLoading = vi.fn();
  view.refreshCalendarDailyReports = async () => {};
  return { plugin, view };
}

describe("dashboard reloadIndex with unchanged markdown files", () => {
  it("still picks up papers.json changes made without any markdown change", async () => {
    const { plugin, view } = setup();
    await view.reloadIndex();
    expect(view.entries).toHaveLength(1);
    expect(view.entries[0]!.pdfPath).toBe("");

    // What PdfService does after a download: papers.json changes, no .md file does.
    await plugin.buildPaperIndex().setPdfPath("2606.00001", "arxiv/pdfs/2606.00001.pdf");
    await plugin.buildPaperIndex().setStatus("2606.00001", "saved");
    await view.reloadIndex();

    expect(view.entries[0]!.pdfPath).toBe("arxiv/pdfs/2606.00001.pdf");
    expect(view.entries[0]!.status).toBe("saved");
  });

  it("skips the vault scan when the markdown path set is unchanged", async () => {
    const { plugin, view } = setup();
    await view.reloadIndex();
    const readSpy = vi.spyOn(plugin.app.vault.adapter, "read");
    await view.reloadIndex();
    expect(readSpy).not.toHaveBeenCalled();
    await view.refreshFromVault();
    expect(readSpy).toHaveBeenCalled();
  });
});

describe("dashboard recent-dates foreground refresh", () => {
  it("re-renders when the refresh completes after the first index render", async () => {
    let resolveRefresh!: (value: unknown) => void;
    let snapshot = {
      status: "idle" as const,
      dates: new Set<string>(),
      refreshedAt: 0,
    };
    const { view } = setup({
      snapshot: () => snapshot,
      refreshWithin: () => new Promise((resolve) => { resolveRefresh = resolve; }),
    });

    const opening = view.onOpen();
    await vi.waitFor(() => expect(view.render).toHaveBeenCalledTimes(1));

    snapshot = {
      status: "ready" as const,
      dates: new Set(["2026-06-10"]),
      refreshedAt: 1000,
    };
    resolveRefresh({ completed: true, timedOut: false, snapshot });
    await vi.waitFor(() => expect(view.render).toHaveBeenCalledTimes(2));
    await opening;
  });

  it("renders the timeout fallback notice immediately and clears it after the background refresh", async () => {
    let resolveBackground!: (value: unknown) => void;
    const empty = { status: "idle" as const, dates: new Set<string>(), refreshedAt: 0 };
    let current = empty;
    const { view } = setup({
      snapshot: () => current,
      refreshWithin: async () => ({
        completed: false,
        timedOut: true,
        snapshot: empty,
        refresh: new Promise((resolve) => { resolveBackground = resolve; }),
      }),
    });

    await view.onOpen();
    // The timeout notice rides on the first index render; painting before the
    // scan finishes would replace Loading with an empty-state frame.
    await vi.waitFor(() => expect(view.render).toHaveBeenCalledTimes(1));
    const notice = (view as unknown as { recentDatesNotice: string | null })
      .recentDatesNotice;
    expect(notice).toContain("still refreshing in the background");

    current = {
      status: "ready" as const,
      dates: new Set(["2026-06-10"]),
      refreshedAt: 5000,
    };
    resolveBackground(current);
    await vi.waitFor(() => expect(view.render).toHaveBeenCalledTimes(2));
    expect((view as unknown as { recentDatesNotice: string | null }).recentDatesNotice)
      .toBeNull();
  });

  it("clears the timeout notice even when the background snapshot is unchanged", async () => {
    let resolveBackground!: (value: unknown) => void;
    const empty = { status: "idle" as const, dates: new Set<string>(), refreshedAt: 0 };
    const { view } = setup({
      snapshot: () => empty,
      refreshWithin: async () => ({
        completed: false,
        timedOut: true,
        snapshot: empty,
        refresh: new Promise((resolve) => { resolveBackground = resolve; }),
      }),
    });

    await view.onOpen();
    await vi.waitFor(() => expect(view.render).toHaveBeenCalledTimes(1));
    expect((view as unknown as { recentDatesNotice: string | null }).recentDatesNotice)
      .toContain("still refreshing in the background");

    // Same dates and refreshedAt: only the notice changes, and the background
    // callback must compare against the timeout frame, not the pre-timeout one.
    resolveBackground(empty);
    await vi.waitFor(() => expect(view.render).toHaveBeenCalledTimes(2));
    expect((view as unknown as { recentDatesNotice: string | null }).recentDatesNotice)
      .toBeNull();
  });

  it("never repaints over Loading while the first index scan is running", () => {
    const { view } = setup();
    const internals = view as unknown as {
      indexReady: boolean;
      recentDatesNotice: string | null;
      renderRecentDatesIfChanged(before: string): void;
    };
    internals.indexReady = false;
    internals.recentDatesNotice = "partially refreshed";
    internals.renderRecentDatesIfChanged("stale");
    expect(view.render).not.toHaveBeenCalled();
  });

  it("does not redraw again when the completed refresh matches the rendered snapshot", async () => {
    const snapshot = {
      status: "ready" as const,
      dates: new Set(["2026-06-10"]),
      refreshedAt: 900,
    };
    const refreshWithin = vi.fn(async () => ({
      completed: true,
      timedOut: false,
      snapshot,
    }));
    const { view } = setup({ snapshot: () => snapshot, refreshWithin });

    await view.onOpen();
    await vi.waitFor(() => expect(refreshWithin).toHaveBeenCalled());
    await Promise.resolve();

    expect(view.render).toHaveBeenCalledTimes(1);
  });
});

describe("dashboard scheduler toggle", () => {
  it("reads the setting at click time and reports a failed save instead of dropping it", async () => {
    const { plugin, view } = setup();
    const toggle = view as unknown as { toggleScheduler(): Promise<void>; runDetached(p: Promise<unknown>, action: string): void };
    const state = plugin as unknown as {
      settings: { schedule: { enabled: boolean } };
      setScheduleEnabled?: (enabled: boolean) => Promise<boolean>;
      logger: { error: ReturnType<typeof vi.fn> };
    };
    state.settings = { ...state.settings, schedule: { ...state.settings.schedule, enabled: true } };
    const setScheduleEnabled = vi.fn(async (_enabled: boolean) => {
      throw new Error("disk full");
    });
    state.setScheduleEnabled = setScheduleEnabled;

    // The menu was built while the scheduler was off, but it is on by the time of the click.
    toggle.runDetached(toggle.toggleScheduler(), "change the scheduler state");
    await vi.waitFor(() => expect(state.logger.error).toHaveBeenCalled());

    expect(setScheduleEnabled).toHaveBeenCalledWith(false);
    expect(state.logger.error).toHaveBeenCalledWith(
      "dashboard: failed to change the scheduler state",
      expect.objectContaining({ message: "disk full" }),
    );
  });
});
