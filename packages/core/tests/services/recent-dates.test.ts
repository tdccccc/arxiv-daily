import { markupParser } from "../markup-parser";
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
        markupParser,
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
        markupParser,
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
        markupParser,
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
        markupParser,
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

  it("reuses the ready snapshot for refreshes inside the TTL", async () => {
    let nowMs = Date.parse("2026-06-23T01:00:00Z");
    const fetcher = {
      fetchRecent: vi.fn(async () => recentHtml("2026-06-22", "2606.00001")),
    };
    const buildFetcher = vi.fn(() => fetcher);
    const cache = new RecentDatesCache({
        markupParser,
      getSettings: () => settingsWithCategories(["cs.CL"]),
      buildFetcher,
      logger: { debug: vi.fn(), warn: vi.fn() },
      now: () => new Date(nowMs),
    });

    const first = await cache.refresh();
    nowMs += 60_000;
    const second = await cache.refresh();

    expect(second.status).toBe("ready");
    expect(second.refreshedAt).toBe(first.refreshedAt);
    expect(second.dates.has("2026-06-22")).toBe(true);
    expect(buildFetcher).toHaveBeenCalledTimes(1);
    expect(fetcher.fetchRecent).toHaveBeenCalledTimes(1);
  });

  it("refreshes ready snapshots after the TTL expires", async () => {
    let nowMs = Date.parse("2026-06-23T01:00:00Z");
    const fetcher = {
      fetchRecent: vi
        .fn()
        .mockResolvedValueOnce(recentHtml("2026-06-22", "2606.00001"))
        .mockResolvedValueOnce(recentHtml("2026-06-23", "2606.00002")),
    };
    const buildFetcher = vi.fn(() => fetcher);
    const cache = new RecentDatesCache({
        markupParser,
      getSettings: () => settingsWithCategories(["cs.CL"]),
      buildFetcher,
      logger: { debug: vi.fn(), warn: vi.fn() },
      now: () => new Date(nowMs),
    });

    await cache.refresh();
    nowMs += 10 * 60_000 + 1;
    const snapshot = await cache.refresh();

    expect(snapshot.status).toBe("ready");
    expect(snapshot.dates.has("2026-06-23")).toBe(true);
    expect(buildFetcher).toHaveBeenCalledTimes(2);
    expect(fetcher.fetchRecent).toHaveBeenCalledTimes(2);
  });

  it("retries failed snapshots even inside the TTL", async () => {
    let nowMs = Date.parse("2026-06-23T01:00:00Z");
    const fetcher = {
      fetchRecent: vi
        .fn()
        .mockRejectedValueOnce(new Error("arXiv unavailable"))
        .mockResolvedValueOnce(recentHtml("2026-06-22", "2606.00001")),
    };
    const buildFetcher = vi.fn(() => fetcher);
    const cache = new RecentDatesCache({
        markupParser,
      getSettings: () => settingsWithCategories(["cs.CL"]),
      buildFetcher,
      logger: { debug: vi.fn(), warn: vi.fn() },
      now: () => new Date(nowMs),
    });

    const failed = await cache.refresh();
    nowMs += 60_000;
    const recovered = await cache.refresh();

    expect(failed.status).toBe("failed");
    expect(recovered.status).toBe("ready");
    expect(recovered.dates.has("2026-06-22")).toBe(true);
    expect(buildFetcher).toHaveBeenCalledTimes(2);
    expect(fetcher.fetchRecent).toHaveBeenCalledTimes(2);
  });

  it("returns the cached snapshot at the foreground deadline while the refresh continues", async () => {
    vi.useFakeTimers();
    try {
      let resolveFetch: (html: string) => void = () => {};
      const fetcher = {
        fetchRecent: vi
          .fn()
          .mockResolvedValueOnce(recentHtml("2026-06-21", "2606.00000"))
          .mockImplementationOnce(
            () =>
              new Promise<string>((resolve) => {
                resolveFetch = resolve;
              }),
          ),
      };
      const cache = new RecentDatesCache({
        markupParser,
        getSettings: () => settingsWithCategories(["cs.CL"]),
        buildFetcher: () => fetcher,
        logger: { debug: vi.fn(), warn: vi.fn() },
        now: () => new Date("2026-06-23T01:00:00Z"),
        ttlMs: 0,
      });
      await cache.refresh();
      expect(cache.hasDate("2026-06-21")).toBe(true);
      expect(cache.hasDate("2026-06-22")).toBe(false);

      const pending = cache.refreshWithin(1000);
      await vi.advanceTimersByTimeAsync(1000);

      const result = await pending;
      expect(result.completed).toBe(false);
      expect(result.timedOut).toBe(true);
      expect(result.snapshot.status).toBe("ready");
      expect(result.snapshot.dates.has("2026-06-21")).toBe(true);
      expect(result.snapshot.dates.has("2026-06-22")).toBe(false);

      resolveFetch(recentHtml("2026-06-22", "2606.00001"));
      await result.refresh;

      expect(cache.hasDate("2026-06-22")).toBe(true);
      expect(fetcher.fetchRecent).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it("lets a later cancellable caller stop waiting for an uncancellable first refresh", async () => {
    let resolveFetch: (html: string) => void = () => {};
    const fetcher = {
      fetchRecent: vi.fn(
        () => new Promise<string>((resolve) => { resolveFetch = resolve; }),
      ),
    };
    const cache = new RecentDatesCache({
      markupParser,
      getSettings: () => settingsWithCategories(["cs.CL"]),
      buildFetcher: () => fetcher,
      logger: { debug: vi.fn(), warn: vi.fn() },
    });
    const first = cache.refresh();
    const controller = new AbortController();
    const second = cache.refresh(controller.signal);

    controller.abort("second caller cancelled");
    await expect(second).rejects.toThrow("second caller cancelled");
    expect(fetcher.fetchRecent).toHaveBeenCalledTimes(1);

    resolveFetch(recentHtml("2026-06-22", "2606.00001"));
    await expect(first).resolves.toMatchObject({ status: "ready" });
  });

  it("does not cancel an unrelated later caller when the first caller cancels", async () => {
    let resolveFetch: (html: string) => void = () => {};
    const fetcher = {
      fetchRecent: vi.fn(
        () => new Promise<string>((resolve) => { resolveFetch = resolve; }),
      ),
    };
    const cache = new RecentDatesCache({
      markupParser,
      getSettings: () => settingsWithCategories(["cs.CL"]),
      buildFetcher: () => fetcher,
      logger: { debug: vi.fn(), warn: vi.fn() },
    });
    const controller = new AbortController();
    const first = cache.refresh(controller.signal);
    const second = cache.refresh();

    controller.abort("first caller cancelled");
    await expect(first).rejects.toThrow("first caller cancelled");
    expect(fetcher.fetchRecent).toHaveBeenCalledTimes(1);

    resolveFetch(recentHtml("2026-06-22", "2606.00001"));
    await expect(second).resolves.toMatchObject({ status: "ready" });
  });

  it("reuses an in-flight refresh for concurrent foreground waits", async () => {
    vi.useFakeTimers();
    try {
      let resolveFetch: (html: string) => void = () => {};
      const fetcher = {
        fetchRecent: vi.fn(
          () =>
            new Promise<string>((resolve) => {
              resolveFetch = resolve;
            }),
        ),
      };
      const cache = new RecentDatesCache({
        markupParser,
        getSettings: () => settingsWithCategories(["cs.CL"]),
        buildFetcher: () => fetcher,
        logger: { debug: vi.fn(), warn: vi.fn() },
        now: () => new Date("2026-06-23T01:00:00Z"),
      });

      const first = cache.refreshWithin(1000);
      const second = cache.refreshWithin(1000);
      await vi.advanceTimersByTimeAsync(1000);

      const [firstResult, secondResult] = await Promise.all([first, second]);
      expect(firstResult.timedOut).toBe(true);
      expect(secondResult.timedOut).toBe(true);
      expect(fetcher.fetchRecent).toHaveBeenCalledTimes(1);

      resolveFetch(recentHtml("2026-06-22", "2606.00001"));
      await Promise.all([firstResult.refresh, secondResult.refresh]);

      expect(cache.hasDate("2026-06-22")).toBe(true);
      expect(fetcher.fetchRecent).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });
});
