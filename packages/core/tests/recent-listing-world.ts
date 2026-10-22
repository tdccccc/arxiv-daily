import { vi } from "vitest";

/** One announcement day of a category's /recent listing, newest first in a world. */
export interface WorldDay {
  /** Announcement date, YYYY-MM-DD. */
  date: string;
  /** Entry ids in listing order. */
  ids: string[];
}

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "Mon, 11 May 2026", the way arXiv labels a day. */
export function listingDateLabel(date: string): string {
  const parsed = new Date(`${date}T00:00:00Z`);
  return `${WEEKDAYS[parsed.getUTCDay()]}, ${parsed.getUTCDate()} ${MONTHS[parsed.getUTCMonth()]} ${parsed.getUTCFullYear()}`;
}

/** `count` modern arXiv ids starting at `first`. */
export function listingIds(prefix: string, first: number, count: number): string[] {
  return Array.from({ length: count }, (_, i) => `${prefix}.${String(first + i).padStart(5, "0")}`);
}

/** The most entries arXiv puts on one page ("Showing up to 2000 entries per page"). */
export const LISTING_PAGE_LIMIT = 2000;

/**
 * A category's /recent listing as arXiv serves it: day links with the offset
 * where each day starts, "Total of N entries", and for the requested window one
 * `dl#articles` per day with a "(showing X of M entries )" header. A day cut by
 * the page boundary reads "showing first X of M".
 */
export function recentListingWorld(days: readonly WorldDay[], category = "cs.LG") {
  const starts: number[] = [];
  let total = 0;
  for (const day of days) {
    starts.push(total);
    total += day.ids.length;
  }
  const requests: Array<{ skip: number; show: number }> = [];

  const entry = (id: string, position: number) => `
    <dt>
      <a name='item${position}'>[${position}]</a>
      <a href ="/abs/${id}" title="Abstract" id="${id}">
        arXiv:${id}
      </a>
    </dt>
    <dd>
      <div class='meta'>
        <div class='list-title mathjax'><span class='descriptor'>Title:</span>
          Title ${id}
        </div>
        <div class='list-authors'><a href="https://arxiv.org/search/?query=Author&amp;searchtype=author">Author ${id}</a></div>
      </div>
    </dd>`;

  function page(skip = 0, show = LISTING_PAGE_LIMIT): string {
    const end = Math.min(skip + Math.min(show, LISTING_PAGE_LIMIT), total);
    const links = days
      .map((day, i) => `<li>
        <a href="/list/${category}/recent?skip=${starts[i]}&amp;show=${LISTING_PAGE_LIMIT}">
          ${listingDateLabel(day.date)}
        </a>
      </li>`)
      .join("");
    const sections = days.flatMap((day, i) => {
      const from = Math.max(skip, starts[i]!);
      const to = Math.min(end, starts[i]! + day.ids.length);
      if (from >= to) return [];
      const shown = day.ids.slice(from - starts[i]!, to - starts[i]!);
      const first = from === starts[i] && shown.length < day.ids.length ? "first " : "";
      return [`
      <dl id='articles'>
    <h3>${listingDateLabel(day.date)} (showing ${first}${shown.length} of ${day.ids.length} entries )</h3>
    ${shown.map((id, k) => entry(id, from + k + 1)).join("")}
</dl>`];
    });
    return `<html><body><div id='dlpage'>
  <h1>Machine Learning</h1>
  <h2>Authors and titles for recent submissions</h2>
  <ul>${links}</ul>
  <div class='paging'>Total of ${total} entries
  </div>
  <div class='morefewer'>Showing up to ${LISTING_PAGE_LIMIT} entries per page:
    <a href=/list/${category}/recent?skip=0&amp;show=1000 rel="nofollow">fewer</a> | <span>more</span> | <span>all</span>
  </div>
  ${sections.join("")}
</div></body></html>`;
  }

  /** Stands in for `ArxivFetcher.fetchRecent`, recording each request. */
  const fetchRecent = vi.fn(async (
    _category: string,
    _signal?: AbortSignal,
    options?: { skip?: number; show?: number },
  ) => {
    const skip = options?.skip ?? 0;
    const show = options?.show ?? LISTING_PAGE_LIMIT;
    requests.push({ skip, show });
    return page(skip, show);
  });

  return { page, fetchRecent, requests, starts, total };
}
