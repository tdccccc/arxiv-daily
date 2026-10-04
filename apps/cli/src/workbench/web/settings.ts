import { t } from "./i18n";
import type { UiAppearancePreferences } from "@arxiv-daily/core";
import { modelCombobox, mountModelCombobox } from "./model-combobox";
import { mountSettingsNavigation } from "./settings-navigation";
import { settingsSetupGuide } from "./settings-setup";
import type { CliLibraryConnectionInspection } from "../../library-connection-cmd";
import type { WorkbenchSettings as SettingsSnapshot } from "../settings";
import { ARXIV_CATEGORIES, DEFAULT_SETTINGS, DEFAULT_UI_APPEARANCE, libraryRowPresentation, type Topic } from "@arxiv-daily/core";
import { descriptions } from './settings-copy';
export type { WorkbenchSettings as SettingsSnapshot } from "../settings";
const escape = (text: string) => text.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
const button = (action: string, label: string) => `<button type="button" data-settings="${action}">${escape(t(label))}</button>`;
const zones = { 'Asia/Shanghai': 'Shanghai (UTC+8)', 'Asia/Tokyo': 'Tokyo (UTC+9)', 'US/Eastern': 'US East (UTC-5)', 'US/Pacific': 'US West (UTC-8)', 'Europe/London': 'London (UTC+0)', 'Europe/Berlin': 'Berlin (UTC+1)', 'Europe/Moscow': 'Moscow (UTC+3)', 'Australia/Sydney': 'Sydney (UTC+10)', UTC: 'UTC' };
const controlLabels: Record<string,string> = {"vaultRoot": "Save root (absolute path on this device)", "baseUrl": "API base URL", "apiKey": "API key", "model": "Model", "reasoningEffort": "Reasoning effort", "detailProfile": "Automatic detail notes", "timezone": "Timezone", "timezoneCustom": "Or enter custom timezone", "dailyDir": "Daily reports folder", "papersDir": "Paper notes folder", "linkStyle": "Link style", "summaryLanguage": "Summary language", "topicName": "Name", "topicTag": "Tag", "topicDetail": "Detail report", "libraryPath": "PDF folder absolute path", "logLevel": "Log level", "appearance.theme": "Theme", "appearance.language": "Interface language", "schedule.enabled": "Enable \u00b7 Running", "schedule.runAtLocal": "Start", "schedule.runUntilLocal": "End", "schedule.tickIntervalMin": "Check every (minutes)", "embedding.mode": "Embedding", "embedding.baseUrl": "Embedding API base URL", "embedding.apiKey": "Embedding API key", "embedding.model": "Embedding model", "embedding.dimension": "Embedding dimension", "pdfParserSidecar.enabled": "Better PDF parser", "pdfParserSidecar.capabilitiesUrl": "Sidecar capability URL", "pdfParserSidecar.parseUrl": "Sidecar parse URL", "email.mode": "How to send", "email.to": "Your email", "email.apiKey": "Resend API key", "email.hostedToken": "Verification code", "email.fromEmail": "From email", "email.fromName": "From name", "email.enabled": "Daily auto-send"};
function controlLabel(name:string):string{return t(controlLabels[name]??name);}
function input(name: string, value: string | number, type = 'text', extra = ''): string { return `<input name="${name}" aria-label="${escape(controlLabel(name))}" type="${type}" value="${escape(String(value))}" ${extra.replace(/placeholder="([^"]*)"/g, (_match, value: string) => `placeholder="${escape(t(value))}"`)}>`; }
function select(name: string, value: string, options: Record<string, string>, key = name): string { return `<select name="${name}" aria-label="${escape(controlLabel(name))}" data-setting-key="${key}">${Object.entries(options).map(([id,label]) => `<option value="${id}" ${id === value ? 'selected' : ''}>${escape(t(label))}</option>`).join('')}</select>`; }
function toggle(name: string, value: boolean, key = name): string { return `<input name="${name}" aria-label="${escape(controlLabel(name))}" type="checkbox" role="switch" data-setting-key="${key}" ${value ? 'checked' : ''}>`; }
function secret(name: string, configured: boolean): string { return `<div class="settings-secret">${input(name,'','password','autocomplete="new-password"')}<button type="button" data-settings="show-secret" aria-label="${escape(t('Show')+' '+controlLabel(name))}">${escape(t('Show'))}</button><small>${escape(t(configured ? 'Saved. Leave blank to keep unchanged.' : 'Saved only on this device.'))}</small></div>`; }
function row(name: string, controls: string, desc = descriptions[name] ?? '', extra = ''): string { return `<div class="setting-item" data-setting-name="${escape(name)}" ${extra}><div class="setting-item-info"><div class="setting-item-name">${escape(t(name))}</div><div class="setting-item-description">${escape(t(desc))}</div></div><div class="setting-item-control">${controls}</div></div>`; }
function group(name: string, content: string): string { return `<section class="settings-section"><h3 data-settings-heading data-settings-key="${escape(name)}">${escape(t(name))}</h3>${content}</section>`; }
function categoryRow(value: string, index: number): string {
 const known = ARXIV_CATEGORIES.some(group => group.categories.some(c => c.id === value));
 return row(String(index + 1), `<select name="category" aria-label="${escape(t('Category {0}',index+1))}">${ARXIV_CATEGORIES.map(group => `<optgroup label="${escape(t(group.label))}">${group.categories.map(c => `<option value="${c.id}" ${c.id === value ? 'selected' : ''}>${escape(`${c.id} — ${t(c.name)}`)}</option>`).join('')}</optgroup>`).join('')}${!known ? `<option selected value="${escape(value)}">${escape(value)} — ${escape(t('custom'))}</option>` : ''}</select>${button('remove-category','Delete')}`, '', 'data-category-row');
}
function topicRow(topic: Topic, open = false): string { return `<details class="settings-topic" data-topic-id="${escape(topic.id)}" data-setting-name="${escape(topic.name.trim() || '(unnamed)')}" ${open ? 'open' : ''}><summary><span class="settings-topic-name">${escape(topic.name.trim() || t('(unnamed)'))}</span><span class="settings-topic-tag">${topic.tag ? escape('#'+topic.tag) : ''}</span><span class="settings-topic-star" title="${escape(t('Detail report enabled'))}">${topic.detail ? '★' : ''}</span></summary><div class="settings-topic-body"><label>${escape(t('Name'))}${input('topicName',topic.name)}</label><label>${escape(t('Tag'))}${input('topicTag',topic.tag)}</label><label>${escape(t('Description'))}<textarea name="topicDescription" rows="3">${escape(topic.description)}</textarea></label><label class="settings-detail-toggle">${escape(t('Detail report'))} <input name="topicDetail" type="checkbox" ${topic.detail ? 'checked' : ''}></label>${button('remove-topic','Delete')}</div></details>`; }
function times(value: string): Record<string,string> { const out: Record<string,string> = {}; for(let h=0;h<24;h++)for(let m=0;m<60;m+=15){const t=`${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}`;out[t]=t;} out[value]=value; return Object.fromEntries(Object.entries(out).sort()); }
export function settingsForm(snapshot: SettingsSnapshot, firstReportComplete = false, appearance: UiAppearancePreferences = DEFAULT_UI_APPEARANCE): string {
 const d=DEFAULT_SETTINGS, v=Object.assign({ reasoningEffort: d.llm.thinkingMode ? d.llm.reasoningEffort : 'none', detailProfile: d.detailSelection.profile, linkStyle:d.output.linkStyle, schedule:d.schedule, embedding:{...d.embedding,apiKeyConfigured:false}, pdfParserSidecar:d.pdfParserSidecar, email:{...d.email,apiKeyConfigured:false,hostedTokenConfigured:false}, logLevel:d.advanced.logLevel }, snapshot.values);
 const hiddenRemote = v.embedding.mode !== 'remote' ? 'hidden' : '', hiddenParser = !v.pdfParserSidecar.enabled ? 'hidden' : '';
 const emailHosted = v.email.mode === 'hosted';
 return `<form class="settings-form"><div class="settings-host-context"><p>${escape(t(snapshot.setupRequired ? '首次使用：选择保存目录，再配置下方 LLM 和研究主题。' : '设置保存后立即生效。'))}</p><label>${escape(t('保存根目录（本机绝对路径）'))}${input('vaultRoot',v.vaultRoot)}</label><p>${escape(t('模型 API 与 DSH / Claude Code 对话模型独立。密钥留空保留现有值。'))}</p><code>${escape(snapshot.configPath)}</code></div><div class="settings-setup-host">${settingsSetupGuide(snapshot,firstReportComplete)}</div>
 ${row(v.schedule.enabled?'Enable · Running':'Enable · Paused',toggle('schedule.enabled',v.schedule.enabled),'When on, daily reports run automatically on weekdays (weekends are skipped).')}
 ${group('LLM',row('API base URL',input('baseUrl',v.baseUrl,'url','placeholder="Provider URL"'))+row('API key',secret('apiKey',v.apiKeyConfigured))+row('Model',modelCombobox(v.model)+button('models','Get models')+'<span class="settings-model-status" role="status"></span>')+row('Reasoning effort',select('reasoningEffort',v.reasoningEffort,{none:'None',low:'Low',medium:'Medium',high:'High'})))}
 ${group('arXiv categories',`<div class="settings-categories">${v.categories.map(categoryRow).join('')}</div>${button('add-category','Add category')}`)}
 ${group('Research topics',`<div class="settings-topic-list">${v.topics.map(t=>topicRow(t)).join('')}</div>${button('add-topic','Add topic')}`)}
 ${row('Automatic detail notes',select('detailProfile',v.detailProfile,{conservative:'Fewer',balanced:'Recommended',broad:'More',...(v.detailProfile==='custom'?{custom:'Custom (current values)'}:{})},'detailSelection.profile'))}
 ${row('Timezone',select('timezone',v.timezone,zones)+input('timezoneCustom',Object.hasOwn(zones,v.timezone)?'':v.timezone,'text','placeholder="Or enter custom timezone"'))}
 ${group('Output & schedule',row('Daily reports folder',input('dailyDir',v.dailyDir))+row('Paper notes folder',input('papersDir',v.papersDir))+row('Link style',select('linkStyle',v.linkStyle,{wikilink:'Obsidian wikilink',relative:'Standard relative link'},'output.linkStyle'))+row('Summary language',select('summaryLanguage',v.summaryLanguage,{zh:'Chinese',en:'English'},'output.summaryLanguage'))+row('Run window',`<label>${escape(t('Start'))} ${select('schedule.runAtLocal',v.schedule.runAtLocal,times(v.schedule.runAtLocal))}</label><label>${escape(t('End'))} ${select('schedule.runUntilLocal',v.schedule.runUntilLocal,times(v.schedule.runUntilLocal))}</label>`)+row('Check every (minutes)',input('schedule.tickIntervalMin',v.schedule.tickIntervalMin)))}
 ${group('Personal library',row('Library','<div class="settings-library-controls">'+button('library-connect','Choose folder')+'</div>','Choose a folder of PDFs, then build a search index. Separate from daily reports.','data-library-row')+row('Embedding',select('embedding.mode',v.embedding.mode,{local:'Local (offline, default)',remote:'Remote (fast, full text leaves this device)'}),'Local embeds on this device. Switch to remote only if you have an embeddings API.')+row('Embedding API base URL',input('embedding.baseUrl',v.embedding.baseUrl,'url'),'OpenAI-compatible embeddings endpoint.',`data-remote ${hiddenRemote}`)+row('Embedding API key',secret('embedding.apiKey',v.embedding.apiKeyConfigured),'Saved only on this device.',`data-remote ${hiddenRemote}`)+row('Embedding model',input('embedding.model',v.embedding.model),'Model name sent to the endpoint.',`data-remote ${hiddenRemote}`)+row('Embedding dimension',input('embedding.dimension',v.embedding.dimension,'number','min="1" step="1"'),'Vector width of the remote model. Must match the model.',`data-remote ${hiddenRemote}`)+row('Better PDF parser',toggle('pdfParserSidecar.enabled',v.pdfParserSidecar.enabled))+row('Sidecar capability URL',input('pdfParserSidecar.capabilitiesUrl',v.pdfParserSidecar.capabilitiesUrl,'url'),descriptions['Sidecar capability URL'],`data-parser ${hiddenParser}`)+row('Sidecar parse URL',input('pdfParserSidecar.parseUrl',v.pdfParserSidecar.parseUrl,'url'),descriptions['Sidecar parse URL'],`data-parser ${hiddenParser}`))}
 ${group('Email delivery',`<div class="settings-email-guide"></div>`+row('How to send',select('email.mode',v.email.mode,{self:'Send yourself',hosted:'Official delivery (beta)'}),'Send yourself uses your own Resend account (no project quota). Official delivery (Beta) is a limited free option for light personal use.')+row('Your email',input('email.to',v.email.to,'email','placeholder="you@example.com"')+`<span data-email-hosted ${emailHosted?'':'hidden'}>${button('email-verify','Send verification')}</span>`,'Where digests are delivered. With From empty, use the email on your Resend account.')+row('Verification code',secret('email.hostedToken',v.email.hostedTokenConfigured)+button('email-test','Send test'),descriptions['Verification code'],`data-email-hosted ${emailHosted?'':'hidden'}`)+row('Resend API key',secret('email.apiKey',v.email.apiKeyConfigured)+button('email-test','Send test'),descriptions['Resend API key'],`data-email-self ${emailHosted?'hidden':''}`)+row('From email',input('email.fromEmail',v.email.fromEmail,'text','placeholder="Leave blank for simplest setup"'),descriptions['From email'],`data-email-self ${emailHosted?'hidden':''}`)+row('From name',input('email.fromName',v.email.fromName,'text','placeholder="arXiv Daily"'),descriptions['From name'],`data-email-self ${emailHosted?'hidden':''}`)+row('Daily auto-send',toggle('email.enabled',v.email.enabled),'When on, a digest is emailed after each successful daily report. Email problems do not stop report generation.'))}
 ${group('Appearance',row('Theme',select('appearance.theme',appearance.theme,{light:'Light',dark:'Dark',system:'System'}))+row('Interface language',select('appearance.language',appearance.language,{zh:'Chinese',en:'English'})))}
 ${group('Advanced',row('Log level',select('logLevel',v.logLevel,{debug:'Debug',info:'Info',warn:'Warn',error:'Error'},'advanced.logLevel')))}
 ${group('Help & feedback',row('Report a bug',`<a href="https://github.com/tdccccc/arxiv-daily/issues/new?body=-%20arXiv%20Daily%3A%20Workbench" target="_blank" rel="noopener noreferrer">${escape(t('Open'))}</a>`)+row('Request a feature',`<a href="https://github.com/tdccccc/arxiv-daily/issues/new" target="_blank" rel="noopener noreferrer">${escape(t('Open'))}</a>`)+row('Documentation',`<a href="https://github.com/tdccccc/arxiv-daily/blob/main/docs/getting-started.md" target="_blank" rel="noopener noreferrer">${escape(t('Open'))}</a>`)+row('Repository',`<a href="https://github.com/tdccccc/arxiv-daily" target="_blank" rel="noopener noreferrer">${escape(t('Open'))}</a>`))}
 <div class="settings-action-host"></div><p class="settings-action-status" role="status"></p><p class="form-error" role="alert" hidden></p><div class="dialog-footer">${button('close','取消')}<button type="submit" class="primary-button">${escape(t(snapshot.setupRequired?'保存并开始使用':'保存设置'))}</button></div></form>`;
}

export function bindSettings(form: HTMLFormElement, snapshot: SettingsSnapshot, request: <T>(url: string, body?: unknown) => Promise<T>, saved: () => Promise<void>, onRun?: (run: import('../server').WorkbenchRun) => void, firstReportComplete = false, options: { appearance?: UiAppearancePreferences; onAppearanceSaved?: (appearance: UiAppearancePreferences) => Promise<void> } = {}): void {
 mountSettingsNavigation(form);
 const modelControl=mountModelCombobox(form);
 let busy=false, revision=snapshot.revision, libraryCancelling=false, libraryRevisionPending=false;
 let currentSnapshot=snapshot;
 let lastCompletedRun: string | undefined;
 let library: (CliLibraryConnectionInspection & { run?: import('../server').WorkbenchRun }) | undefined;
 const get=(name:string) => (form.elements.namedItem(name) as HTMLInputElement)?.value.trim() ?? '';
 const checked=(name:string) => (form.elements.namedItem(name) as HTMLInputElement).checked;
 const find=<T extends HTMLElement=HTMLElement>(selector:string)=>form.querySelector<T>(selector)!;
 function values() {
  const topics=Array.from(form.querySelectorAll<HTMLElement>('.settings-topic')).map(row=>({id:row.dataset.topicId!,name:row.querySelector<HTMLInputElement>('[name="topicName"]')!.value.trim(),tag:row.querySelector<HTMLInputElement>('[name="topicTag"]')!.value.trim(),description:row.querySelector<HTMLTextAreaElement>('[name="topicDescription"]')!.value.trim(),detail:row.querySelector<HTMLInputElement>('[name="topicDetail"]')!.checked}));
  const key=(name:string,field='apiKey')=>get(name)?{[field]:get(name)}:{};
  return {vaultRoot:get('vaultRoot'),provider:snapshot.values.provider,baseUrl:get('baseUrl'),model:get('model'),...key('apiKey'),reasoningEffort:get('reasoningEffort'),categories:Array.from(form.querySelectorAll<HTMLElement>('[data-category-row]')).map(row=>row.querySelector<HTMLSelectElement>('select')!.value),topics,timezone:get('timezoneCustom')||get('timezone'),detailProfile:get('detailProfile'),dailyDir:get('dailyDir'),papersDir:get('papersDir'),linkStyle:get('linkStyle'),summaryLanguage:get('summaryLanguage'),schedule:{enabled:checked('schedule.enabled'),runAtLocal:get('schedule.runAtLocal'),runUntilLocal:get('schedule.runUntilLocal'),tickIntervalMin:Number(get('schedule.tickIntervalMin'))},embedding:{mode:get('embedding.mode'),baseUrl:get('embedding.baseUrl'),model:get('embedding.model'),dimension:Number(get('embedding.dimension')),...key('embedding.apiKey')},pdfParserSidecar:{enabled:checked('pdfParserSidecar.enabled'),capabilitiesUrl:get('pdfParserSidecar.capabilitiesUrl'),parseUrl:get('pdfParserSidecar.parseUrl')},email:{enabled:checked('email.enabled'),mode:get('email.mode'),to:get('email.to'),fromEmail:get('email.fromEmail'),fromName:get('email.fromName'),...key('email.apiKey'),...key('email.hostedToken','hostedToken')},logLevel:get('logLevel')};
 }
 function conditions() {
  for(const el of Array.from(form.querySelectorAll<HTMLElement>('[data-remote]')))el.hidden=get('embedding.mode')!=='remote';
  for(const el of Array.from(form.querySelectorAll<HTMLElement>('[data-parser]')))el.hidden=!checked('pdfParserSidecar.enabled');
  for(const el of Array.from(form.querySelectorAll<HTMLElement>('[data-email-hosted]')))el.hidden=get('email.mode')!=='hosted';
  for(const el of Array.from(form.querySelectorAll<HTMLElement>('[data-email-self]')))el.hidden=get('email.mode')==='hosted';
  const schedule=find('[data-setting-name^="Enable ·"]'); schedule.dataset.settingName=checked('schedule.enabled')?'Enable · Running':'Enable · Paused';schedule.querySelector('.setting-item-name')!.textContent=t(schedule.dataset.settingName);
  const hosted=get('email.mode')==='hosted';
  const desc=(name:string,text:string)=>{find(`[data-setting-name="${name}"] .setting-item-description`).textContent=t(text);};
  desc('Embedding',get('embedding.mode')==='remote'?'Remote sends full text to an embeddings API. Switching modes rebuilds the index.':'Local embeds on this device. Switch to remote only if you have an embeddings API.');
  desc('How to send',hosted?'Official delivery (Beta) is a shared free service with a small daily limit. Prefer Send yourself if you need many messages or reliable high volume.':'Send yourself uses your own Resend account (no project quota). Official delivery (Beta) is a limited free option for light personal use.');
  desc('Your email',hosted?'Where verification and daily digests are sent.':'Where digests are delivered. With From empty, use the email on your Resend account.');
  desc('Daily auto-send',hosted?'When on, a digest is emailed after each successful daily report. Official delivery may stop for the day if the shared limit is reached; report generation still continues.':'When on, a digest is emailed after each successful daily report. Email problems do not stop report generation.');
  find('.settings-email-guide').textContent=t(hosted?'Official delivery (Beta): enter your email, send verification, then paste the code from the verification page.':'Send yourself: enter your email and Resend API key, then send a test.');
  for(const el of Array.from(form.querySelectorAll<HTMLButtonElement>('[data-settings="remove-category"]')))el.hidden=form.querySelectorAll('[data-category-row]').length<=1;
 }
 function renderLibrary() {
  if(!library)return;
  const run=library.run;
  const row=libraryRowPresentation({status:library.status,embeddingMode:get('embedding.mode')==='remote'?'remote':'local',...(run?.status==='running'?{activity:{phase:run.label,cancelling:libraryCancelling}}:{})});
  let description=row.description;
  if(library.status.kind==='disconnected')description=t(row.description);
  else {
   const label=library.status.rootLabel;
   if(run?.status==='running')description=libraryCancelling?t('Stopping the index run for {0} — it finishes the step it is on first.',label):t('Indexing {0} — {1}. Nothing is saved until the run finishes, so cancelling discards it.',label,run.label);
   else if(get('embedding.mode')==='remote' && library.status.kind!=='authorized')description=t(library.status.kind==='authorization-invalidated'?'Selected: {0}. The embedding endpoint changed, so building the index asks you to confirm what full text leaves this device.':'Selected: {0}. Remote embedding sends full text off this device — building the index asks you to confirm first.',label);
   else description=t(get('embedding.mode')==='remote'?'Connected: {0}. Authorized for remote full-text embedding. Build the search index next.':'Selected: {0}. Local embedding stays on this device. Build the search index to search these PDFs.',label);
  }
  find('[data-library-row] .setting-item-description').textContent=description;
  find('.settings-library-controls').innerHTML=[['library-connect',row.chooseFolder],['library-build',row.primary],['library-cancel',row.cancel],['library-revoke',row.revoke]].map(([action,item])=>{if(!item||typeof item==='string')return '';return `<button type="button" data-settings="${action}" ${item.disabled?'disabled':''}>${escape(t(item.label))}</button>`;}).join('');
 }
 async function refreshLibrary() { library=await request<typeof library>('api/settings/library'); if(form.isConnected){renderLibrary();if(library?.run?.status==='running'){lockLibraryInputs(true);onRun?.(library.run);}} }
 function refreshVisibleSetupGuide() {
  const host=find('.settings-setup-host');
  if(!host.querySelector('.settings-setup'))return;
  const content=settingsSetupGuide(currentSnapshot,firstReportComplete);
  if(host.innerHTML!==content)host.innerHTML=content;
 }
 async function saveDraft(refreshGuide = false) {
  const secrets=Array.from(form.querySelectorAll<HTMLInputElement>('.settings-secret input')).map(el=>({el,value:el.value}));
  const result=await request<SettingsSnapshot>('api/settings',{revision,values:values()}); revision=result.revision;currentSnapshot=result;
  if(refreshGuide)refreshVisibleSetupGuide();
  for(const {el,value} of secrets)if(el.value===value){el.value='';delete el.dataset.revealed;el.type='password';el.closest('.settings-secret')!.querySelector('button')!.textContent=t('Show');el.closest('.settings-secret')!.querySelector('small')!.textContent=t('Saved. Leave blank to keep unchanged.');}
  return result;
 }
 async function task(operation:()=>Promise<void>, actionButton?: HTMLButtonElement) {
  if(busy)return;busy=true;find('[role="alert"]').hidden=true;
  const submit=find<HTMLButtonElement>('[type="submit"]');submit.disabled=true;
  const originalLabel=actionButton?.textContent ?? '';
  if(actionButton?.dataset.settings==='models')find('.settings-model-status').textContent=t('Fetching models…');
  if(actionButton){actionButton.disabled=true;if(actionButton.dataset.settings==='models')actionButton.textContent=t('Fetching…');else if(actionButton.dataset.settings?.startsWith('email-'))actionButton.textContent=t('Sending…');}
  try{await operation();}catch(error){if(form.isConnected){if(actionButton?.dataset.settings==='models')find('.settings-model-status').textContent=error instanceof Error?error.message:t('Could not load models.');find('[role="alert"]').hidden=false;find('[role="alert"]').textContent=error instanceof Error?error.message:t('操作失败，请重试。');}}
  finally{busy=false;submit.disabled=library?.run?.status==='running'||libraryRevisionPending;if(actionButton){actionButton.disabled=false;actionButton.textContent=originalLabel;}}
 }
 async function action(name:string,extra:Record<string,unknown>={}) {
  const result=await request<{settings?:SettingsSnapshot;library?:typeof library;models?:string[];run?:import('../server').WorkbenchRun}>('api/settings/action',{revision,action:name,...extra});
  if(!form.isConnected)return;
  if(result.settings)revision=result.settings.revision;
  if(result.library){library=result.library;renderLibrary();}
  if(result.models){
   modelControl.setOptions(result.models);
   find('.settings-model-status').textContent=result.models.length?t('Loaded {0} models. Choose from the list or type a model name.', result.models.length):t('No models returned; type a model name to continue.');
  }
  if(result.run){onRun?.(result.run);if(name==='library-build'&&library){library.run=result.run;renderLibrary();lockLibraryInputs(result.run.status==='running');}find('.settings-action-status').textContent=result.run.label;}
 }
 function lockLibraryInputs(locked: boolean) {
  for(const control of Array.from(form.querySelectorAll<HTMLInputElement|HTMLSelectElement|HTMLTextAreaElement|HTMLButtonElement>('input,select,textarea,button:not([data-settings="library-cancel"]):not([data-settings="close"]):not([data-settings-nav])')))control.disabled=locked;
  find<HTMLButtonElement>('[type="submit"]').disabled=locked || busy;if(!locked)renderLibrary();
 }
 form.addEventListener('workbench-run',event=>{
  const run=(event as CustomEvent<import('../server').WorkbenchRun|null>).detail;
  if(run?.status==='completed' && run.id!==lastCompletedRun){lastCompletedRun=run.id;void request<{recentRuns?:Array<{status:string}>}>('api/status').then(status=>{firstReportComplete=status.recentRuns?.some(entry=>entry.status==='completed')??firstReportComplete;if(form.isConnected)refreshVisibleSetupGuide();}).catch(()=>{});}
  if(library?.run && run?.id===library.run.id){
   library.run=run;if(run.status!=='running')libraryCancelling=false;renderLibrary();
   if(run.status!=='running'){libraryRevisionPending=true;void Promise.all([request<SettingsSnapshot>('api/settings'),refreshLibrary()]).then(([current])=>{revision=current.revision;}).catch(error=>{if(form.isConnected){find('[role="alert"]').hidden=false;find('[role="alert"]').textContent=error instanceof Error?error.message:t('Could not reload settings.');}}).finally(()=>{libraryRevisionPending=false;lockLibraryInputs(false);});}
  }
 });
 form.addEventListener('change',event=>{
  const target=event.target;
  if(target instanceof HTMLSelectElement && target.name==='timezone')find<HTMLInputElement>('[name="timezoneCustom"]').value='';
  if(target instanceof HTMLInputElement && target.name==='timezoneCustom' && target.value.trim()){
   const dropdown=find<HTMLSelectElement>('[name="timezone"]');const value=target.value.trim();
   if(!Array.from(dropdown.options).some(option=>option.value===value))dropdown.add(new Option(value,value));
   dropdown.value=value;
  }
  conditions();renderLibrary();updateTopic(event);
 });
 function updateTopic(event:Event) {
  const target=event.target;if(!(target instanceof HTMLInputElement))return;
  const topic=target.closest<HTMLElement>('.settings-topic');if(!topic)return;
  const name=topic.querySelector<HTMLInputElement>('[name="topicName"]')!.value.trim()||'(unnamed)';
  const tag=topic.querySelector<HTMLInputElement>('[name="topicTag"]')!.value.trim();
  topic.dataset.settingName=name;topic.querySelector('.settings-topic-name')!.textContent=topic.querySelector<HTMLInputElement>('[name="topicName"]')!.value.trim()||t('(unnamed)');
  topic.querySelector('.settings-topic-tag')!.textContent=tag?'#'+tag:'';
  topic.querySelector('.settings-topic-star')!.textContent=topic.querySelector<HTMLInputElement>('[name="topicDetail"]')!.checked?'★':'';
 }
 form.addEventListener('input',event=>{if(event.target instanceof HTMLInputElement)delete event.target.dataset.revealed;updateTopic(event);});
 form.addEventListener('click',event=>{
  const setup=event.target instanceof HTMLElement?event.target.closest<HTMLButtonElement>('[data-setup-action]'):null;
  if(setup){
   const name=setup.dataset.setupAction;
   if(name==='llm'||name==='arxiv'||name==='topics'){
    const heading=Array.from(form.querySelectorAll<HTMLElement>('[data-settings-heading]')).find(h=>h.dataset.settingsKey===({llm:'LLM',arxiv:'arXiv categories',topics:'Research topics'})[name]);
    heading?.scrollIntoView?.({block:'start',behavior:'smooth'});heading?.parentElement?.querySelector<HTMLElement>('input,select,button')?.focus();return;
   }
   void task(async()=>{
    if(name==='dashboard'){await saved();return;}
    if(name==='enable'){find<HTMLInputElement>('[name="schedule.enabled"]').checked=true;conditions();}
    await saveDraft(true);
    if(name==='generate'){const result=await request<{run:import('../server').WorkbenchRun}>('api/runs',{kind:'daily'});onRun?.(result.run);}
   },setup);return;
  }
  const target=event.target instanceof HTMLElement?event.target.closest<HTMLButtonElement>('[data-settings]'):null;if(!target)return;
  const name=target.dataset.settings!;
  if(name==='close'){form.closest('dialog')?.querySelector<HTMLButtonElement>('[data-action="close-dialog"]')?.click();return;}
  if(name==='show-secret'){
   const field=target.previousElementSibling as HTMLInputElement;
   if(field.type==='text'){field.type='password';if(field.dataset.revealed==='true'){field.value='';delete field.dataset.revealed;}target.textContent=t('Show');return;}
   if(field.value){field.type='text';target.textContent=t('Hide');return;}
   if(revision===null){field.closest('.settings-secret')!.querySelector('small')!.textContent=t('尚未保存密钥，请先输入。');return;}
   target.disabled=true;target.textContent=t('Loading…');const initial=field.value;
   void request<{value:string}>('api/settings/secret',{revision,field:field.name}).then(result=>{
    if(!form.isConnected||field.value!==initial)return;
    if(!result.value){field.closest('.settings-secret')!.querySelector('small')!.textContent=t('尚未保存密钥。');return;}
    field.value=result.value;field.type='text';field.dataset.revealed='true';
   }).catch(error=>{if(form.isConnected)field.closest('.settings-secret')!.querySelector('small')!.textContent=error instanceof Error?error.message:t('无法显示密钥。');}).finally(()=>{target.disabled=false;target.textContent=t(field.type==='text'?'Hide':'Show');});return;
  }
  if(busy)return;
  if(name==='add-topic'){find('.settings-topic-list').insertAdjacentHTML('beforeend',topicRow({id:crypto.randomUUID(),name:'',tag:'',description:'',detail:true},true));return;}
  if(name==='remove-topic'){target.closest('.settings-topic')!.remove();return;}
  if(name==='add-category'){const current=values().categories;const next=ARXIV_CATEGORIES.flatMap(g=>g.categories).find(c=>!current.includes(c.id));find('.settings-categories').insertAdjacentHTML('beforeend',categoryRow(next?.id??'cs.LG',current.length));conditions();return;}
  if(name==='remove-category'){if(form.querySelectorAll('[data-category-row]').length>1)target.closest('[data-category-row]')!.remove();Array.from(form.querySelectorAll<HTMLElement>('[data-category-row]')).forEach((row,i)=>{row.dataset.settingName=String(i+1);row.querySelector('.setting-item-name')!.textContent=String(i+1);});conditions();return;}
  if(name==='library-connect'){find('.settings-action-host').innerHTML=`<div class="settings-confirm" role="group" aria-label="${escape(t('Choose library folder'))}"><label>${escape(t('PDF folder absolute path'))} ${input('libraryPath','')}</label>${button('confirm-library-connect','Choose folder')}${button('cancel-action','Cancel')}</div>`;find<HTMLInputElement>('[name="libraryPath"]').focus();return;}
  if(name==='cancel-action'){find('.settings-action-host').innerHTML='';return;}
  void task(async()=>{
   if(name==='library-cancel'){if(library?.run){libraryCancelling=true;renderLibrary();try{const result=await request<{run:import('../server').WorkbenchRun}>('api/runs/cancel',{id:library.run.id});library.run=result.run;libraryCancelling=result.run.status==='running';onRun?.(result.run);}catch(error){libraryCancelling=false;throw error;}finally{renderLibrary();}}return;}
   await saveDraft();
   if(name==='confirm-library-connect'){await action('library-connect',{path:get('libraryPath')});find('.settings-action-host').innerHTML='';return;}
   if(name==='library-build'){
    await refreshLibrary();
    if(library?.status.kind!=='authorized' && library?.disclosure){find('.settings-action-host').innerHTML=`<div class="settings-confirm" role="group" aria-label="${escape(t('Library processing consent'))}"><h4>${escape(t('Confirm library processing'))}</h4><pre>${escape([t('Folder: {0}',library.disclosure.selectedRoot),t('Eligible files: {0}',library.disclosure.eligibleExtensions.join(', ')),t('Processing depth: {0}',t(library.disclosure.processingDepth)),t('Model endpoint: {0}',library.disclosure.endpoint),...(library.disclosure.embeddingEndpoint?[t('Embedding endpoint: {0}',library.disclosure.embeddingEndpoint)]:[])].join('\n'))}</pre>${button('confirm-library-build','Confirm and build index')}${button('cancel-action','Cancel')}</div>`;return;}
   }
   if(name==='confirm-library-build'){if(!library?.disclosure)throw new Error(t('请重新查看授权范围。'));await action('library-build',{fingerprint:library.disclosure.authorizationFingerprint});find('.settings-action-host').innerHTML='';return;}
   await action(name);
  },target);
 });
 form.addEventListener('submit',event=>{event.preventDefault();void task(async()=>{await saveDraft();const appearance: UiAppearancePreferences={theme:get('appearance.theme') as UiAppearancePreferences['theme'],language:get('appearance.language') as UiAppearancePreferences['language']};await request('api/preferences',{appearance});await options.onAppearanceSaved?.(appearance);if(form.isConnected)await saved();});});
 conditions();
 if(!snapshot.setupRequired)void refreshLibrary().catch(error=>{if(form.isConnected)find('[data-library-row] .setting-item-description').textContent=error instanceof Error?error.message:t('Unable to load library.');});
}
