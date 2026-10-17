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
});
