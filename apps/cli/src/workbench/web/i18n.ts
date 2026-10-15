import { settingsMessages, readingMessages, workbenchMessages, DEFAULT_UI_APPEARANCE, type UiAppearancePreferences } from '@arxiv-daily/core';
export type UiLanguage = UiAppearancePreferences['language'];
let language: UiLanguage = DEFAULT_UI_APPEARANCE.language;
const catalog = new Map<string, readonly [string,string]>();
/** Consent line for the daily report's use of library paper text (ADR 0015); CLI-local wording. */
export const DAILY_NOVELTY_DISCLOSURE = "日报也会把检索到的文献库论文的标题和摘要发送给同一模型端点，用于对比每篇入选论文相对文献库的新颖性。";
const cliMessages: ReadonlyArray<readonly [string,string]> = [
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
