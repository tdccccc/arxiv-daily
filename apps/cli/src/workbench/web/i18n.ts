import { settingsMessages, readingMessages, workbenchMessages, DEFAULT_UI_APPEARANCE, type UiAppearancePreferences } from '@arxiv-daily/core';
export type UiLanguage = UiAppearancePreferences['language'];
let language: UiLanguage = DEFAULT_UI_APPEARANCE.language;
const catalog = new Map<string, readonly [string,string]>();
/** Consent line for the daily report's use of library paper text (ADR 0015); CLI-local wording. */
export const DAILY_NOVELTY_DISCLOSURE = "日报也会把检索到的文献库论文的标题和摘要发送给同一模型端点，用于对比每篇入选论文相对文献库的新颖性。";
const cliMessages: ReadonlyArray<readonly [string,string]> = [
 ["更新方向建议", "Update direction suggestions"],
 ["只会把 {0} 篇新增论文的标题和摘要，连同现有研究方向的文字发送给已授权的模型；其余文献库论文不会重新发送。新的候选方向会替换当前候选，请先保存需要保留的修改；之后仍需另行审核接受。",
  "Only the {0} new paper(s) will be sent (titles and abstracts, plus the text of your existing research directions) to the authorized model; other library papers are not sent again. The new suggestions replace the current candidates, so save any edits you want to keep first; you still review and accept them separately."],
 ["还没有可供对比的研究主题，将改为分析整个文献库的标题和摘要。新的候选方向会替换当前候选，请先保存需要保留的修改。",
  "There are no research topics to compare with yet, so the titles and abstracts of the whole library will be analysed instead. The new suggestions replace the current candidates, so save any edits you want to keep first."],
 ["本次新增 {0} 篇", "{0} added in this update"],
 [DAILY_NOVELTY_DISCLOSURE, "The daily report also sends the titles and abstracts of retrieved library papers to the same model endpoint, to compare each digest paper's novelty against your library."],
];
for (const pair of [...workbenchMessages,...settingsMessages,...readingMessages,...cliMessages]) { catalog.set(pair[0],pair); catalog.set(pair[1],pair); }
export function setUiLanguage(value: UiLanguage): void { language=value; }
export function getUiLanguage(): UiLanguage { return language; }
/** Only UI messages should enter this function; user-authored research stays untouched. */
export function t(source: string, ...args: Array<string|number>): string {
 const pair=catalog.get(source); const message=pair ? pair[language==='zh'?0:1] : source;
 return message.replace(/\{(\d+)\}/g,(_,i)=>String(args[Number(i)]??`{${i}}`));
}
