import { afterEach, expect, it, vi } from "vitest";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import { DEFAULT_CLI_SCHEDULE, type CliRuntimeConfig } from "../src/config";
import { latestAnnouncedDayFrom } from "../src/workbench/announced-day";

afterEach(() => { vi.unstubAllGlobals(); });

const config = (): CliRuntimeConfig => {
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.arxiv.category = "cs.AI"; settings.arxiv.categories = ["cs.AI"];
  return { settings, vaultRoot: "/vault", cacheDir: "/cache", configPath: "/config.toml", linkStyle: "wikilink", scheduleIntent: { ...DEFAULT_CLI_SCHEDULE } };
};

// The /recent listing names each announced day in a heading, as arXiv's page does.
function listing(dates: string[]): string {
  const heading = (date: string) => new Intl.DateTimeFormat("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }).format(new Date(`${date}T12:00:00Z`));
  return `<html><body>${dates.map((date, index) => `<dl id="articles"><h3>${heading(date)} (showing 1 of 1 entries)</h3><dt><a title="Abstract" href="/abs/2610.0000${index}">arXiv:2610.0000${index}</a></dt><dd><div class="list-title">Title: Paper ${index}</div></dd></dl>`).join("")}</body></html>`;
}

function stubArxiv(status: number, body = "") {
  const fetch = vi.fn(async (_url: string) => new Response(body, { status, headers: { "content-type": "text/html" } }));
  vi.stubGlobal("fetch", fetch);
  return fetch;
}

it("names the newest day arXiv has announced that is not after the one asked about", async () => {
  const fetch = stubArxiv(200, listing(["2026-10-09", "2026-10-08", "2026-10-07"]));
  // arXiv spaces its requests, so one question is asked: 2026-10-09 is later than this day and is left out.
  expect(await latestAnnouncedDayFrom(config)("2026-10-08", new AbortController().signal)).toBe("2026-10-08");
  expect(fetch.mock.calls[0]![0]).toContain("https://arxiv.org/list/cs.AI/recent");
});

it("answers null instead of failing when there is no configuration", async () => {
  const fetch = stubArxiv(200, listing(["2026-10-09"]));
  expect(await latestAnnouncedDayFrom(() => undefined)("2026-10-21", new AbortController().signal)).toBeNull();
  expect(fetch).not.toHaveBeenCalled();
});
