import { isWeekendDate } from '@arxiv-daily/core';

export const WEEKEND_REPORT_MESSAGE = '周末 arXiv 无更新，已跳过日报生成。';
/** Match the weekday policy already used by the shared automatic scheduler. */
export function isWeekendReportDate(date: string): boolean {
  const parts = /^(\d{4})-(\d{2})-(\d{2})$/.exec(date);
  return Boolean(parts && isWeekendDate({ y: Number(parts[1]), m: Number(parts[2]), d: Number(parts[3]) }));
}
/** Read-only compatibility for old runs: every category must report the known missing-new-bucket case. */
export function isLegacyWeekendAnnouncementGap(date: string, error: string | undefined): boolean {
  if (!isWeekendReportDate(date) || !error?.startsWith('all arXiv categories failed: ')) return false;
  const item = `date ${date} is newer than newest [\\w.-]+ /recent bucket \\d{4}-\\d{2}-\\d{2}; arXiv announce page may not be available yet`;
  return new RegExp(`^all arXiv categories failed: ${item}(?:; ${item})*$`).test(error);
}
