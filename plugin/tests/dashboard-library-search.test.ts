import { beforeEach, describe, expect, it, vi } from "vitest";
import { ARXIV_DAILY_DASHBOARD_VIEW, registerDashboardView } from "../src/dashboard/view";

const rendered: Array<{ kind: string; message?: string }> = [];
vi.mock("../src/dashboard/library-search-block", () => ({
  renderLibrarySearchBlock: (_container: unknown, state: { kind: string; message?: string }) => {
    rendered.push(state);
  },
}));

type View = {
  query: { tab: string; search?: string };
  libraryResultsEl: unknown;
  refreshLibrarySearch(): void;
  onClose(): Promise<void>;
  contentEl: unknown;
};

function setup(overrides: {
  connection?: "disconnected" | "authorized";
  lastRun?: { updatedAt: string; papers: number };
  preparationError?: string;
  search?: ReturnType<typeof vi.fn>;
} = {}) {
  let create: ((leaf: unknown) => unknown) | undefined;
  const search = overrides.search ?? vi.fn(async () => [
    { paperKey: "arxiv:2606.00001", title: "A", score: 0.5, scoreKind: "cosine", rankingScore: 0.5, rankingScoreKind: "cosine", hits: [] },
  ]);
  const plugin = {
    app: {},
    registerView: (type: string, creator: (leaf: unknown) => unknown) => {
      if (type === ARXIV_DAILY_DASHBOARD_VIEW) create = creator;
    },
    settings: { embedding: { mode: "local" } },
    getLibraryConnectionStatus: () => ({ kind: overrides.connection ?? "authorized" }),
    libraryIndexStatus: {
      snapshot: () => ({
        ...("lastRun" in overrides ? { lastRun: overrides.lastRun } : { lastRun: { updatedAt: "t1", papers: 3 } }),
        ...(overrides.preparationError ? { preparationError: overrides.preparationError } : {}),
      }),
    },
    searchPersonalLibraryFullText: search,
  };
  registerDashboardView(plugin as never);
  const view = create!({}) as unknown as View;
  const empty = vi.fn();
  view.libraryResultsEl = { empty };
  view.query = { tab: "all", search: "galaxy clusters" };
  return { view, search, empty, plugin };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("dashboard library search block", () => {
  beforeEach(() => { rendered.length = 0; });

  it("does not search or render an error when no library is connected", async () => {
    const { view, search, empty } = setup({ connection: "disconnected" });
    view.refreshLibrarySearch();
    await flush();
    expect(search).not.toHaveBeenCalled();
    expect(empty).toHaveBeenCalled();
    expect(rendered).toEqual([]);
  });

  it("shows a neutral hint instead of searching while the index is not prepared", async () => {
    for (const overrides of [{ lastRun: undefined }, { preparationError: "Retry preparation" }]) {
      rendered.length = 0;
      const { view, search } = setup(overrides);
      view.refreshLibrarySearch();
      await flush();
      expect(search).not.toHaveBeenCalled();
      expect(rendered.map((state) => state.kind)).toEqual(["hint"]);
    }
  });

  it("memoises the outcome so re-renders with an unchanged query do not re-run the search", async () => {
    const { view, search } = setup();
    view.refreshLibrarySearch();
    view.refreshLibrarySearch(); // re-render while the first search is still in flight
    await flush();
    view.refreshLibrarySearch(); // re-render after it settled (e.g. a star toggle)
    await flush();
    expect(search).toHaveBeenCalledTimes(1);
    expect(rendered.filter((state) => state.kind === "matches")).toHaveLength(2);
  });

  it("searches again when the query changes", async () => {
    const { view, search } = setup();
    view.refreshLibrarySearch();
    await flush();
    view.query = { tab: "all", search: "weak lensing" };
    view.refreshLibrarySearch();
    await flush();
    expect(search).toHaveBeenCalledTimes(2);
  });

  describe("cancellation", () => {
    /** A search that stays in flight until its signal aborts, then rejects like an aborted request. */
    function hangingSearch() {
      const signals: AbortSignal[] = [];
      const search = vi.fn((_query: string, options?: { signal?: AbortSignal }) => {
        const signal = options?.signal;
        if (signal) signals.push(signal);
        return new Promise<never>((_resolve, reject) => {
          signal?.addEventListener("abort", () => {
            const error = new Error("cancelled by user");
            error.name = "AbortError";
            reject(error);
          }, { once: true });
        });
      });
      return { search, signals };
    }

    it("aborts the in-flight search when a newer query supersedes it", async () => {
      const { search, signals } = hangingSearch();
      const { view } = setup({ search });
      view.refreshLibrarySearch();
      expect(signals).toHaveLength(1);
      expect(signals[0]!.aborted).toBe(false);

      view.query = { tab: "all", search: "weak lensing" };
      view.refreshLibrarySearch();
      await flush();

      expect(signals).toHaveLength(2);
      expect(signals[0]!.aborted).toBe(true);
      expect(signals[1]!.aborted).toBe(false);
      expect(rendered.some((state) => state.kind === "error")).toBe(false);
    });

    it("does not abort a shared search when the same query renders again", async () => {
      const { search, signals } = hangingSearch();
      const { view } = setup({ search });
      view.refreshLibrarySearch();
      view.refreshLibrarySearch();
      view.refreshLibrarySearch();
      await flush();
      expect(search).toHaveBeenCalledTimes(1);
      expect(signals[0]!.aborted).toBe(false);
    });

    it("neither caches nor shows an aborted search, and searches again for the same query", async () => {
      const { search, signals } = hangingSearch();
      const { view } = setup({ search });
      view.refreshLibrarySearch(); // "galaxy clusters"
      view.query = { tab: "all", search: "weak lensing" };
      view.refreshLibrarySearch(); // supersedes it
      await flush();
      expect(signals[0]!.aborted).toBe(true);

      view.query = { tab: "all", search: "galaxy clusters" };
      view.refreshLibrarySearch(); // back to the first query: nothing was cached
      await flush();
      expect(search).toHaveBeenCalledTimes(3);
      expect(signals[1]!.aborted).toBe(true);
      expect(signals[2]!.aborted).toBe(false);
      expect(rendered.map((state) => state.kind)).toEqual(["loading", "loading", "loading"]);
    });

    it("aborts the in-flight search when the query becomes too short to search", async () => {
      const { search, signals } = hangingSearch();
      const { view, empty } = setup({ search });
      view.refreshLibrarySearch();
      view.query = { tab: "all", search: "g" };
      view.refreshLibrarySearch();
      await flush();
      expect(signals[0]!.aborted).toBe(true);
      expect(empty).toHaveBeenCalled();
      expect(rendered.map((state) => state.kind)).toEqual(["loading"]);
    });

    it("aborts the in-flight search when the view closes", async () => {
      const { search, signals } = hangingSearch();
      const { view } = setup({ search });
      view.contentEl = { empty: vi.fn() };
      view.refreshLibrarySearch();
      await view.onClose();
      await flush();
      expect(signals[0]!.aborted).toBe(true);
      expect(rendered.map((state) => state.kind)).toEqual(["loading"]);
    });

    it("still caches a search that finished before it could be superseded", async () => {
      const { view, search } = setup();
      view.refreshLibrarySearch();
      await flush();
      view.query = { tab: "all", search: "weak lensing" };
      view.refreshLibrarySearch();
      await flush();
      view.query = { tab: "all", search: "weak lensing" };
      view.refreshLibrarySearch();
      await flush();
      expect(search).toHaveBeenCalledTimes(2);
    });
  });
});
