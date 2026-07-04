export function todayInTz(now: Date, tz: string): { y: number; m: number; d: number } {
  const fmt = new Intl.DateTimeFormat("en-CA", {
    timeZone: tz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  const parts = fmt.formatToParts(now);
  const get = (t: string) => Number(parts.find((p) => p.type === t)!.value);
  return { y: get("year"), m: get("month"), d: get("day") };
}

export function formatDate(d: { y: number; m: number; d: number }): string {
  const mm = String(d.m).padStart(2, "0");
  const dd = String(d.d).padStart(2, "0");
  return `${d.y}-${mm}-${dd}`;
}

export function parseHHMM(s: string): { hour: number; minute: number } {
  const m = /^(\d{2}):(\d{2})$/.exec(s);
  if (!m) throw new Error(`Invalid HH:MM: ${s}`);
  const hour = Number(m[1]);
  const minute = Number(m[2]);
  if (hour > 23 || minute > 59) throw new Error(`Invalid HH:MM: ${s}`);
  return { hour, minute };
}

export function minutesSinceMidnight(now: Date, tz: string): number {
  const fmt = new Intl.DateTimeFormat("en-GB", {
    timeZone: tz,
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  });
  const parts = fmt.formatToParts(now);
  const hour = Number(parts.find((p) => p.type === "hour")!.value);
  const minute = Number(parts.find((p) => p.type === "minute")!.value);
  return hour * 60 + minute;
}

export function isTimeWithinLocalWindow(
  now: Date,
  tz: string,
  startHHMM: string,
  endHHMM: string,
): boolean {
  const minutesNow = minutesSinceMidnight(now, tz);
  const start = minutesFromHHMM(startHHMM);
  const end = minutesFromHHMM(endHHMM);
  return isMinutesWithinWindow(minutesNow, start, end);
}

export function isMinutesWithinWindow(
  minutesNow: number,
  startMinutes: number,
  endMinutes: number,
): boolean {
  if (startMinutes > endMinutes) {
    return minutesNow >= startMinutes || minutesNow <= endMinutes;
  }
  return minutesNow >= startMinutes && minutesNow <= endMinutes;
}

export function daysBefore(
  date: { y: number; m: number; d: number },
  n: number,
): { y: number; m: number; d: number } {
  const utc = Date.UTC(date.y, date.m - 1, date.d) - n * 86_400_000;
  const dt = new Date(utc);
  return {
    y: dt.getUTCFullYear(),
    m: dt.getUTCMonth() + 1,
    d: dt.getUTCDate(),
  };
}

export function isWeekendInTz(now: Date, tz: string): boolean {
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: tz,
    weekday: "short",
  });
  const weekday = fmt.format(now);
  return weekday === "Sat" || weekday === "Sun";
}

export function isWeekendDate(date: { y: number; m: number; d: number }): boolean {
  const day = new Date(Date.UTC(date.y, date.m - 1, date.d)).getUTCDay();
  return day === 0 || day === 6;
}

export function minutesFromHHMM(value: string): number {
  const parsed = parseHHMM(value);
  return parsed.hour * 60 + parsed.minute;
}
