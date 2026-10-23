import { settingsMessages, readingMessages, workbenchMessages, DEFAULT_UI_APPEARANCE, type UiAppearancePreferences } from '@arxiv-daily/core';
export type UiLanguage = UiAppearancePreferences['language'];
let language: UiLanguage = DEFAULT_UI_APPEARANCE.language;
const catalog = new Map<string, readonly [string,string]>();
const cliMessages: ReadonlyArray<readonly [string,string]> = [
 ["更新方向建议", "Update direction suggestions"],
 ["只会把 {0} 篇新增论文的标题和摘要，连同现有研究方向的文字发送给已授权的模型；其余文献库论文不会重新发送。新的候选方向会替换当前候选，请先保存需要保留的修改；之后仍需另行审核接受。",
  "Only the {0} new paper(s) will be sent (titles and abstracts, plus the text of your existing research directions) to the authorized model; other library papers are not sent again. The new suggestions replace the current candidates, so save any edits you want to keep first; you still review and accept them separately."],
 ["还没有可供对比的研究主题，将改为分析整个文献库的标题和摘要。新的候选方向会替换当前候选，请先保存需要保留的修改。",
  "There are no research topics to compare with yet, so the titles and abstracts of the whole library will be analysed instead. The new suggestions replace the current candidates, so save any edits you want to keep first."],
 ["本次新增 {0} 篇", "{0} added in this update"],
 ["其中 {0} 篇新论文已被你现有的方向覆盖", "New papers already covered by your existing directions: {0}"],
 ["正在生成…", "Generating…"],
 ["正在生成 {0} 的日报…", "Generating the report for {0}…"],
 ["今天的论文还没公告，第一份日报改用最近一次公告日 {0}。", "Today's papers are not announced yet, so the first report uses the latest announced day, {0}."],
 ["稍后再来检查，或在日历里选择已公告的日期。", "Check again later, or pick an announced day in the calendar."],
 ["该日没有公告新论文，请在日历里选择另一天。", "No new papers were announced that day. Pick another day in the calendar."],
 ["{0} 的论文还没公告，最近一次公告是 {1}。", "The papers for {0} are not announced yet. The latest announcement was {1}."],
 ["改为生成 {0} 的日报", "Generate the report for {0} instead"],
];
for (const pair of [...workbenchMessages,...settingsMessages,...readingMessages,...cliMessages]) { catalog.set(pair[0],pair); catalog.set(pair[1],pair); }
export function setUiLanguage(value: UiLanguage): void { language=value; }
export function getUiLanguage(): UiLanguage { return language; }
/** Only UI messages should enter this function; user-authored research stays untouched. */
export function t(source: string, ...args: Array<string|number>): string {
 const pair=catalog.get(source); const message=pair ? pair[language==='zh'?0:1] : source;
 return message.replace(/\{(\d+)\}/g,(_,i)=>String(args[Number(i)]??`{${i}}`));
}
