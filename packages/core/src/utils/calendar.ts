// Shared date arithmetic; host-specific generation eligibility stays in each host.
export function shiftMonth(month: string, delta: number): string {
  const [rawYear, rawMonthIndex] = month.split("-").map(Number);
  if (
    typeof rawYear !== "number" ||
    typeof rawMonthIndex !== "number" ||
    !Number.isFinite(rawYear) ||
    !Number.isFinite(rawMonthIndex)
  ) {
    return month;
  }
  const year = rawYear;
  const monthIndex = rawMonthIndex;
  const date = new Date(Date.UTC(year, monthIndex - 1 + delta, 1));
  return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}`;
}

export function calendarCells(month: string): Array<{ date: string | null }> {
  const [rawYear, rawMonthIndex] = month.split("-").map(Number);
  if (
    typeof rawYear !== "number" ||
    typeof rawMonthIndex !== "number" ||
    !Number.isFinite(rawYear) ||
    !Number.isFinite(rawMonthIndex)
  ) {
    return [];
  }
  const year = rawYear;
  const monthIndex = rawMonthIndex;
  const first = new Date(Date.UTC(year, monthIndex - 1, 1));
  const daysInMonth = new Date(Date.UTC(year, monthIndex, 0)).getUTCDate();
  const mondayOffset = (first.getUTCDay() + 6) % 7;
  const cells: Array<{ date: string | null }> = Array.from(
    { length: mondayOffset },
    () => ({ date: null }),
  );
  for (let day = 1; day <= daysInMonth; day++) {
    cells.push({
      date: `${month}-${String(day).padStart(2, "0")}`,
    });
  }
  while (cells.length % 7 !== 0) cells.push({ date: null });
  return cells;
}

export function parseCalendarDate(date: string): { y: number; m: number; d: number } | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(date);
  if (!match) return null;
  return {
    y: Number(match[1]),
    m: Number(match[2]),
    d: Number(match[3]),
  };
}
