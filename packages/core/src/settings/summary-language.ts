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
  omittedCount = 0,
): string {
  if (omittedCount > 0) return omittedPapersText(language, omittedCount);
  return normalizeSummaryLanguage(language) === "en"
    ? "No relevant paper updates today."
    : "今日无相关论文更新。";
}

/** Counts are validated at the assembly boundary before reaching display. */
export function omittedPapersText(
  language: SummaryLanguage | undefined,
  omittedCount: number,
  additional = false,
): string {
  if (omittedCount === 0) return "";
  if (normalizeSummaryLanguage(language) === "en") {
    return `${omittedCount} ${additional ? "additional " : ""}relevant ${plural(omittedCount, "paper")} `
      + `${omittedCount === 1 ? "was" : "were"} omitted because of the daily paper limit.`;
  }
  return `因每日总数上限，${additional ? "另有 " : ""}${omittedCount} 篇相关论文未展示。`;
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
