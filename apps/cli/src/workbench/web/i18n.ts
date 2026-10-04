import { settingsMessages, readingMessages, workbenchMessages, DEFAULT_UI_APPEARANCE, type UiAppearancePreferences } from '@arxiv-daily/core';
export type UiLanguage = UiAppearancePreferences['language'];
let language: UiLanguage = DEFAULT_UI_APPEARANCE.language;
const catalog = new Map<string, readonly [string,string]>();
for (const pair of [...workbenchMessages,...settingsMessages,...readingMessages]) { catalog.set(pair[0],pair); catalog.set(pair[1],pair); }
export function setUiLanguage(value: UiLanguage): void { language=value; }
export function getUiLanguage(): UiLanguage { return language; }
/** Only UI messages should enter this function; user-authored research stays untouched. */
export function t(source: string, ...args: Array<string|number>): string {
 const pair=catalog.get(source); const message=pair ? pair[language==='zh'?0:1] : source;
 return message.replace(/\{(\d+)\}/g,(_,i)=>String(args[Number(i)]??`{${i}}`));
}
