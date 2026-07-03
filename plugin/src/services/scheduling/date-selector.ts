import { daysBefore, formatDate, todayInTz } from "../../utils/time";

/** The YYYY-MM-DD for "today" in the configured timezone. Pure given `now`. */
export function todayDateString(tz: string, now: () => Date): string {
  return formatDate(todayInTz(now(), tz));
}

/**
 * `count` calendar days ending today (inclusive), newest first.
 * E.g. count=5 on 2026-05-11 -> [05-11, 05-10, 05-09, 05-08, 05-07].
 */
export function lookbackDateStrings(tz: string, count: number, now: () => Date): string[] {
  const todayObj = todayInTz(now(), tz);
  const out: string[] = [];
  for (let i = 0; i < count; i += 1) {
    out.push(formatDate(daysBefore(todayObj, i)));
  }
  return out;
}
