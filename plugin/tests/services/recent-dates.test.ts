import { describe, expect, it, vi } from "vitest";
import { RecentDatesCache } from "../../src/services/recent-dates";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import type { PluginSettings } from "../../src/settings/types";

function recentHtml(date: string, id: string): string {
  return `
    <html>
      <body>
        <dl id="articles">
          <h3>${dateHeader(date)}</h3>
          <dt><a title="Abstract">arXiv:${id}</a></dt>
          <dd>
            <div class="list-title">Title: Example ${id}</div>
            <div class="list-authors"><a>Author</a></div>
          </dd>
        </dl>
      </body>
    </html>
  `;
}

function dateHeader(date: string): string {
  const [year, month, day] = date.split("-").map(Number);
  const monthName = new Date(Date.UTC(year, month - 1, day)).toLocaleDateString("en-US", {
    month: "short",
    timeZone: "UTC",
  });
  return `${day} ${monthName} ${year}`;
}

function settingsWithCategories(categories: string[]): PluginSettings {
  return {
    ...DEFAULT_SETTINGS,
    arxiv: {
      ...DEFAULT_SETTINGS.arxiv,
      category: categories[0],
      categories,
    },
  };
}

describe("RecentDatesCache", () => {
  it("treats a date as present when any configured category has it", async () => {
    const fetcher = {
      fetchRecent: vi.fn(async (category: string) =>
        category === "cs.CL"
          ? recentHtml("2026-06-22", "2606.00001")
          : recentHtml("2026-06-19", "2606.00002"),
      ),
    };
    const cache = new RecentDatesCache({
      getSettings: () => settingsWithCategories(["cs.CL", "astro-ph"]),
      buildFetcher: () => fetcher,
      logger: { debug: vi.fn(), warn: vi.fn() },
      now: () => new Date("2026-06-23T01:00:00Z"),
    });

    await cache.refresh();

    expect(cache.hasDate("2026-06-22")).toBe(true);
    expect(cache.hasDate("2026-06-19")).toBe(true);
    expect(cache.hasDate("2026-06-18")).toBe(false);
    expect(fetcher.fetchRecent).toHaveBeenCalledWith("cs.CL");
    expect(fetcher.fetchRecent).toHaveBeenCalledWith("astro-ph");
  });

  it("keeps successful category dates when another category fails", async () => {
    const logger = { debug: vi.fn(), warn: vi.fn() };
    const fetcher = {
      fetchRecent: vi.fn(async (category: string) => {
        if (category === "astro-ph") throw new Error("network down");
        return recentHtml("2026-06-22", "2606.00001");
      }),
    };
    const cache = new RecentDatesCache({
      getSettings: () => settingsWithCategories(["cs.CL", "astro-ph"]),
      buildFetcher: () => fetcher,
      logger,
      now: () => new Date("2026-06-23T01:00:00Z"),
    });

    const snapshot = await cache.refresh();

    expect(snapshot.status).toBe("ready");
    expect(cache.hasDate("2026-06-22")).toBe(true);
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("recent dates refresh failed for astro-ph"),
    );
  });

  it("clears confirmed dates when every configured category fails", async () => {
    const fetcher = {
      fetchRecent: vi.fn(async () => {
        throw new Error("arXiv unavailable");
      }),
    };
    const cache = new RecentDatesCache({
      getSettings: () => settingsWithCategories(["cs.CL"]),
      buildFetcher: () => fetcher,
      logger: { debug: vi.fn(), warn: vi.fn() },
      now: () => new Date("2026-06-23T01:00:00Z"),
    });

    const snapshot = await cache.refresh();

    expect(snapshot.status).toBe("failed");
    expect(snapshot.dates.size).toBe(0);
    expect(cache.hasDate("2026-06-22")).toBe(false);
  });

  it("returns immutable snapshots", async () => {
    const fetcher = {
      fetchRecent: vi.fn(async () => recentHtml("2026-06-22", "2606.00001")),
    };
    const cache = new RecentDatesCache({
      getSettings: () => settingsWithCategories(["cs.CL"]),
      buildFetcher: () => fetcher,
      logger: { debug: vi.fn(), warn: vi.fn() },
      now: () => new Date("2026-06-23T01:00:00Z"),
    });
    await cache.refresh();

    const snapshot = cache.snapshot();
    snapshot.dates.add("2026-01-01");

    expect(cache.hasDate("2026-01-01")).toBe(false);
  });
});
