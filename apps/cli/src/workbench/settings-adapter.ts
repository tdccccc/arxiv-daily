import { DEFAULT_SETTINGS, getBusinessSetting, normalizeSettingsEdits, type PluginSettings } from '@arxiv-daily/core';
import type { CliRuntimeConfig } from '../config';

/** Only host wire names live here; defaults and editing rules come from core. */
const paths: Record<string, readonly [string,string]> = {
 'llm.apiKey':['llm','api_key'],'llm.provider':['llm','provider'],'llm.baseUrl':['llm','base_url'],'llm.model':['llm','model'],'llm.thinkingMode':['llm','thinking_mode'],'llm.reasoningEffort':['llm','reasoning_effort'],
 'arxiv.categories':['arxiv','categories'],'arxiv.topics':['arxiv','topics'],'arxiv.timezone':['arxiv','timezone'],
 'output.dailyDir':['output','daily_dir'],'output.papersDir':['output','papers_dir'],'output.linkStyle':['output','link_style'],'output.summaryLanguage':['output','summary_language'],
 'schedule.enabled':['workbench_schedule','enabled'],'schedule.runAtLocal':['workbench_schedule','run_at_local'],'schedule.runUntilLocal':['workbench_schedule','run_until_local'],'schedule.tickIntervalMin':['workbench_schedule','tick_interval_min'],
 'embedding.mode':['embedding','mode'],'embedding.baseUrl':['embedding','base_url'],'embedding.apiKey':['embedding','api_key'],'embedding.model':['embedding','model'],'embedding.dimension':['embedding','dimension'],
 'pdfParserSidecar.enabled':['pdf_parser_sidecar','enabled'],'pdfParserSidecar.capabilitiesUrl':['pdf_parser_sidecar','capabilities_url'],'pdfParserSidecar.parseUrl':['pdf_parser_sidecar','parse_url'],
 'email.enabled':['email','enabled'],'email.mode':['email','mode'],'email.to':['email','to'],'email.fromEmail':['email','from_email'],'email.fromName':['email','from_name'],'email.apiKey':['email','api_key'],'email.hostedToken':['email','hosted_token'],
 'advanced.logLevel':['advanced','log_level'],
 'detailSelection.profile':['detail_selection','profile'],'detailSelection.normalThreshold':['detail_selection','normal_threshold'],'detailSelection.exceptionalThreshold':['detail_selection','exceptional_threshold'],'detailSelection.softLimit':['detail_selection','soft_limit'],
};
const flat: Record<string,string> = { apiKey:'llm.apiKey',provider:'llm.provider',baseUrl:'llm.baseUrl',model:'llm.model',categories:'arxiv.categories',topics:'arxiv.topics',timezone:'arxiv.timezone',dailyDir:'output.dailyDir',papersDir:'output.papersDir',summaryLanguage:'output.summaryLanguage',linkStyle:'output.linkStyle',detailProfile:'detailSelection.profile',logLevel:'advanced.logLevel' };
const secrets = new Set(['llm.apiKey','embedding.apiKey','email.apiKey','email.hostedToken']);
export function patchWorkbenchBusinessSettings(document: Record<string,unknown>, input: Record<string,unknown>, previous: CliRuntimeConfig | null): void {
 const before = structuredClone(previous?.settings ?? DEFAULT_SETTINGS);
 before.schedule = { ...(previous?.workbenchSchedule ?? DEFAULT_SETTINGS.schedule) };
 const candidate = structuredClone(before), changed: string[] = [];
 function assign(key: string,value: unknown) {
  if(secrets.has(key) && (value===undefined||value===''))return;
  const [group,field]=key.split('.') as [keyof PluginSettings,string];
  const target=candidate[group] as unknown as Record<string,unknown>;
  if(JSON.stringify(target[field])!==JSON.stringify(value))changed.push(key);
  target[field]=value;
 }
 for(const [source,key] of Object.entries(flat))if(Object.hasOwn(input,source))assign(key,input[source]);
 if(Object.hasOwn(input,'reasoningEffort')) {
  const value=input.reasoningEffort;
  if(typeof value!=='string'||!Object.hasOwn(getBusinessSetting('reasoningEffort',{reasoningEffort:before.llm.reasoningEffort}).options!,value))throw new Error('Invalid reasoning effort');
  assign('llm.thinkingMode',value!=='none');if(value!=='none')assign('llm.reasoningEffort',value);
 }
 for(const group of ['schedule','embedding','pdfParserSidecar','email'])if(Object.hasOwn(input,group)) {
  const fields=input[group];
  if(!fields||typeof fields!=='object'||Array.isArray(fields))throw new Error(`Invalid ${group}`);
  for(const key of Object.keys(paths).filter(key=>key.startsWith(group+'.'))){const field=key.slice(group.length+1);if(Object.hasOwn(fields,field))assign(key,(fields as Record<string,unknown>)[field]);}
 }
 const normalized=normalizeSettingsEdits(candidate,changed);
 // Preserve unknown TOML fields and untouched secrets while persisting normalized values.
 for(const [key,[tableName,column]] of Object.entries(paths)) {
  const [group,field]=key.split('.') as [keyof PluginSettings,string];
  const value=(normalized[group] as unknown as Record<string,unknown>)[field];
  if(value===undefined||JSON.stringify(value)===JSON.stringify((before[group] as unknown as Record<string,unknown>)[field]))continue;
  const table=document[tableName];
  document[tableName]={...(table&&typeof table==='object'&&!Array.isArray(table)?table as Record<string,unknown>:{}),[column]:value};
 }
}
