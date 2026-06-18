import type { SummaryLanguage } from "./types";

export function normalizeSummaryLanguage(value: unknown): SummaryLanguage {
  return value === "en" ? "en" : "zh";
}

export function dailyHeader(
  language: SummaryLanguage | undefined,
  categories: string,
  dateStr: string,
): string {
  return normalizeSummaryLanguage(language) === "en"
    ? `# arXiv ${categories} Daily Digest ${dateStr}`
    : `# arXiv ${categories} 每日追踪 ${dateStr}`;
}

export function dailyCountLine(
  language: SummaryLanguage | undefined,
  nTotal: number,
  nDetail: number,
): string {
  if (normalizeSummaryLanguage(language) === "en") {
    return (
      `${nTotal} relevant ${plural(nTotal, "paper")}, ` +
      `including ${nDetail} with detail ${plural(nDetail, "note")}.`
    );
  }
  return `共 ${nTotal} 篇相关论文，其中 ${nDetail} 篇详细收录。`;
}

export function noCategoryPapersText(
  language: SummaryLanguage | undefined,
): string {
  return normalizeSummaryLanguage(language) === "en"
    ? "No relevant paper updates today."
    : "今日无相关论文更新。";
}

export function noDailyPapersText(
  language: SummaryLanguage | undefined,
): string {
  return normalizeSummaryLanguage(language) === "en"
    ? "No relevant papers found today."
    : "今日未发现相关论文。";
}

function plural(count: number, word: string): string {
  return count === 1 ? word : `${word}s`;
}
