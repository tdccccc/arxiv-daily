import { t, setUiLanguage } from "./i18n";
import type { UiAppearancePreferences } from "@arxiv-daily/core";
import { h, type DomAttrs } from "./dom";
import { modelCombobox, mountModelCombobox } from "./model-combobox";
import { mountSettingsNavigation } from "./settings-navigation";
import { settingsSetupGuide } from "./settings-setup";
import type { CliLibraryConnectionInspection } from "../../library-connection-cmd";
import type { WorkbenchSettings as SettingsSnapshot } from "../settings";
import { deriveTopicDescription, normalizeTopic, slugify, isCompletedDiscovery, ARXIV_CATEGORIES, getBusinessSettingsSections, getBusinessSetting, getTopicSettingField, runWindowTimeOptions, isBusinessSettingVisible, TIMEZONE_OPTIONS, type BusinessSetting, type BusinessSettingId, type BusinessSettingsContext, DEFAULT_SETTINGS, DEFAULT_UI_APPEARANCE, libraryRowPresentation, type Topic } from "@arxiv-daily/core";
import { descriptions } from './settings-copy';
export type { WorkbenchSettings as SettingsSnapshot } from "../settings";
const button = (action: string, label: string) => h("button", { type: "button", "data-settings": action }, t(label));
const zones = Object.fromEntries(TIMEZONE_OPTIONS.map(zone => [zone.value,zone.label]));
const controlFields: Record<string,BusinessSettingId> = {maxDailyPapers:'maxDailyPapers',baseUrl:'apiBaseUrl',apiKey:'apiKey',model:'model',reasoningEffort:'reasoningEffort',detailProfile:'detailProfile',timezone:'timezone',dailyDir:'dailyDir',papersDir:'papersDir',linkStyle:'linkStyle',summaryLanguage:'summaryLanguage',logLevel:'logLevel','schedule.enabled':'scheduleEnabled','schedule.tickIntervalMin':'tickInterval','embedding.mode':'embeddingMode','embedding.baseUrl':'embeddingBaseUrl','embedding.apiKey':'embeddingApiKey','embedding.model':'embeddingModel','embedding.dimension':'embeddingDimension','pdfParserSidecar.enabled':'sidecarEnabled','pdfParserSidecar.capabilitiesUrl':'sidecarCapabilitiesUrl','pdfParserSidecar.parseUrl':'sidecarParseUrl','email.mode':'emailMode','email.to':'emailTo','email.apiKey':'emailApiKey','email.hostedToken':'hostedToken','email.fromEmail':'fromEmail','email.fromName':'fromName','email.enabled':'emailEnabled'};
const controlLabels: Record<string,string> = {vaultRoot:'Save root (absolute path on this device)',timezoneCustom:'Or enter custom timezone',topicName:'Name',topicTag:'Tag',topicDetail:'Detail report',libraryPath:'PDF folder absolute path','appearance.theme':'Theme','appearance.language':'Interface language','schedule.runAtLocal':'Start','schedule.runUntilLocal':'End'};
function controlLabel(name:string):string{const topicKey=({topicName:'name',topicTag:'tag',topicDetail:'detail'} as const)[name];return t(topicKey?getTopicSettingField(topicKey).name:controlFields[name]?getBusinessSetting(controlFields[name]).name:controlLabels[name]??name);}
function input(name: string, value: string | number, type = 'text', extra: { autocomplete?: string; placeholder?: string } = {}): HTMLInputElement {
 return h("input", { name, "aria-label": controlLabel(name), type, value: String(value), ...(extra.autocomplete ? { autocomplete: extra.autocomplete } : {}), ...(extra.placeholder ? { placeholder: t(extra.placeholder) } : {}) });
}
function select(name: string, value: string, options: Record<string, string>, key = name): HTMLSelectElement {
 return h("select", { name, "aria-label": controlLabel(name), "data-setting-key": key }, ...Object.entries(options).map(([id,label]) => h("option", { value: id, selected: id === value }, t(label))));
}
function toggle(name: string, value: boolean, key = name): HTMLInputElement {
 return h("input", { name, "aria-label": controlLabel(name), type: "checkbox", role: "switch", "data-setting-key": key, checked: value });
}
function secret(name: string, configured: boolean): HTMLElement {
 return h("div", { class: "settings-secret" },
  input(name, '', 'password', { autocomplete: 'new-password' }),
  h("button", { type: "button", "data-settings": "show-secret", "aria-label": `${t('Show')} ${controlLabel(name)}` }, t('Show')),
  h("small", null, t(configured ? 'Saved. Leave blank to keep unchanged.' : 'Saved only on this device.')),
 );
}
function row(name: string, controls: Node | Node[], desc = descriptions[name] ?? '', extra: DomAttrs = {}): HTMLElement {
 return h("div", { class: "setting-item", "data-setting-name": name, ...extra },
  h("div", { class: "setting-item-info" }, h("div", { class: "setting-item-name" }, t(name)), h("div", { class: "setting-item-description" }, t(desc))),
  h("div", { class: "setting-item-control" }, ...(Array.isArray(controls) ? controls : [controls])),
 );
}
function group(name: string, content: Node | Node[]): HTMLElement {
 return h("section", { class: "settings-section" }, h("h3", { "data-settings-heading": true, "data-settings-key": name }, t(name)), ...(Array.isArray(content) ? content : [content]));
}
function categoryRow(value: string, index: number): HTMLElement {
 const known = ARXIV_CATEGORIES.some(group => group.categories.some(c => c.id === value));
 return row(String(index + 1), [
  h("select", { name: "category", "aria-label": t("Category {0}", index + 1) },
   ...ARXIV_CATEGORIES.map(group => h("optgroup", { label: t(group.label) }, ...group.categories.map(c => h("option", { value: c.id, selected: c.id === value }, `${c.id} — ${t(c.name)}`)))),
   ...(known ? [] : [h("option", { selected: true, value }, `${value} — ${t('custom')}`)]),
  ),
  button('remove-category', 'Delete'),
 ], '', { "data-category-row": true });
}
function directionRow(direction: Topic['directions'][number]): HTMLElement {
 return h("div", { "data-direction-id": direction.id, "data-direction-origin": direction.origin },
  h("textarea", { name: "topicDirection", rows: 2, "aria-label": t(getTopicSettingField('directions').name), placeholder: t(getTopicSettingField('directions').placeholder) }, direction.text),
  button('remove-direction', 'Remove direction'),
 );
}
function topicRow(raw: Topic, open = false): HTMLElement {
 const topic=normalizeTopic(raw);
 return h("details", { class: "settings-topic", "data-topic-id": topic.id, "data-topic-tag": topic.tag, "data-original-name": topic.name, "data-setting-name": topic.name.trim() || '(unnamed)', open },
  h("summary", null,
   h("span", { class: "settings-topic-name" }, topic.name.trim() || t('(unnamed)')),
   h("span", { class: "settings-topic-star", title: t('Detail report enabled') }, topic.detail ? '★' : ''),
  ),
  h("div", { class: "settings-topic-body" },
   h("label", null, t(getTopicSettingField('name').name), input('topicName', topic.name)),
   h("div", null,
    t(getTopicSettingField('directions').name),
    h("div", { class: "settings-topic-directions" }, ...topic.directions.map(directionRow)),
    button('add-direction', 'Add direction'),
   ),
   h("label", { class: "settings-detail-toggle" }, `${t(getTopicSettingField('detail').name)} `, h("input", { name: "topicDetail", type: "checkbox", checked: topic.detail })),
   button('remove-topic', 'Delete'),
  ),
 );
}
function times(value: string): Record<string,string> { return Object.fromEntries(runWindowTimeOptions(value).map(option=>[option.value,option.label])); }
export function settingsForm(snapshot: SettingsSnapshot, firstReportComplete = false, appearance: UiAppearancePreferences = DEFAULT_UI_APPEARANCE): HTMLFormElement {
 const d=DEFAULT_SETTINGS, v=Object.assign({ maxDailyPapers:d.output.maxDailyPapers, reasoningEffort: d.llm.thinkingMode ? d.llm.reasoningEffort : 'none', detailProfile: d.detailSelection.profile, linkStyle:d.output.linkStyle, schedule:d.schedule, embedding:{...d.embedding,apiKeyConfigured:false}, pdfParserSidecar:d.pdfParserSidecar, email:{...d.email,apiKeyConfigured:false,hostedTokenConfigured:false}, logLevel:d.advanced.logLevel }, snapshot.values);
 const context: Partial<BusinessSettingsContext> = {emailMode:v.email.mode,embeddingMode:v.embedding.mode,sidecarEnabled:v.pdfParserSidecar.enabled,detailProfile:v.detailProfile,scheduleEnabled:v.schedule.enabled,reasoningEffort:v.reasoningEffort};
 const emailHosted = isBusinessSettingVisible('hostedToken',context);
 function renderField(field: BusinessSetting): HTMLElement {
  const id=field.id, opt=field.options??{}; let controls: Node[];
  const bindings: Partial<Record<BusinessSettingId,{name:string;value:string|number|boolean}>> = {
   maxDailyPapers:{name:'maxDailyPapers',value:v.maxDailyPapers},scheduleEnabled:{name:'schedule.enabled',value:v.schedule.enabled},apiBaseUrl:{name:'baseUrl',value:v.baseUrl},apiKey:{name:'apiKey',value:v.apiKeyConfigured},reasoningEffort:{name:'reasoningEffort',value:v.reasoningEffort},detailProfile:{name:'detailProfile',value:v.detailProfile},dailyDir:{name:'dailyDir',value:v.dailyDir},papersDir:{name:'papersDir',value:v.papersDir},linkStyle:{name:'linkStyle',value:v.linkStyle},summaryLanguage:{name:'summaryLanguage',value:v.summaryLanguage},tickInterval:{name:'schedule.tickIntervalMin',value:v.schedule.tickIntervalMin},
   embeddingMode:{name:'embedding.mode',value:v.embedding.mode},embeddingBaseUrl:{name:'embedding.baseUrl',value:v.embedding.baseUrl},embeddingApiKey:{name:'embedding.apiKey',value:v.embedding.apiKeyConfigured},embeddingModel:{name:'embedding.model',value:v.embedding.model},embeddingDimension:{name:'embedding.dimension',value:v.embedding.dimension},sidecarEnabled:{name:'pdfParserSidecar.enabled',value:v.pdfParserSidecar.enabled},sidecarCapabilitiesUrl:{name:'pdfParserSidecar.capabilitiesUrl',value:v.pdfParserSidecar.capabilitiesUrl},sidecarParseUrl:{name:'pdfParserSidecar.parseUrl',value:v.pdfParserSidecar.parseUrl},
   emailMode:{name:'email.mode',value:v.email.mode},emailTo:{name:'email.to',value:v.email.to},hostedToken:{name:'email.hostedToken',value:v.email.hostedTokenConfigured},emailApiKey:{name:'email.apiKey',value:v.email.apiKeyConfigured},fromEmail:{name:'email.fromEmail',value:v.email.fromEmail},fromName:{name:'email.fromName',value:v.email.fromName},emailEnabled:{name:'email.enabled',value:v.email.enabled},logLevel:{name:'logLevel',value:v.logLevel}
  };
  const binding=bindings[id];
  if(binding) {
   let control: Node;
   if(field.control==='dropdown')control=select(binding.name,String(binding.value),opt,field.key);
   else if(field.control==='toggle')control=toggle(binding.name,Boolean(binding.value),field.key);
   else if(field.control==='secret')control=secret(binding.name,Boolean(binding.value));
   else if(field.control==='text'||field.control==='number')control=input(binding.name,binding.value as string|number,field.control==='number'?'number':'text');
   else throw new Error(`Unsupported control ${field.control} for ${id}`);
   controls=[control];
   if(id==='emailApiKey'||id==='hostedToken')controls.push(button('email-test','Send test'));
   if(id==='emailTo')controls.push(h("span", { "data-email-hosted": true, hidden: !emailHosted }, button('email-verify','Send verification')));
  } else {
   switch(id) {
    case 'model': controls=[modelCombobox(v.model), button('models','Get models'), h("span", { class: "settings-model-status", role: "status" })];break;
    case 'timezone': controls=[select('timezone',v.timezone,zones), input('timezoneCustom',Object.hasOwn(zones,v.timezone)?'':v.timezone,'text',{placeholder:'Or enter custom timezone'})];break;
    case 'runWindow': controls=[h("label", null, t('Start'), select('schedule.runAtLocal',v.schedule.runAtLocal,times(v.schedule.runAtLocal))), h("label", null, t('End'), select('schedule.runUntilLocal',v.schedule.runUntilLocal,times(v.schedule.runUntilLocal)))];break;
    case 'library': controls=[h("div", { class: "settings-library-controls" }, button('library-connect','Choose folder'))];break;
    default:throw new Error(`Missing workbench control for ${id}`);
   }
  }
  const extra: DomAttrs = {
   "data-business-setting": id,
   hidden: !(field.visible && !id.startsWith('sidecar')),
   ...(id==='library' ? { "data-library-row": true } : {}),
   ...(['emailApiKey','fromEmail','fromName'].includes(id) ? { "data-email-self": true } : {}),
   ...(id==='hostedToken' ? { "data-email-hosted": true } : {}),
  };
  return row(field.name,controls,field.description,extra);
 }
 const appearanceSection=()=>group('Appearance',[
  row('Theme',select('appearance.theme',appearance.theme,{light:'Light',dark:'Dark',system:'System'})),
  row('Interface language',select('appearance.language',appearance.language,{zh:'Chinese',en:'English'})),
 ]);
 const business=getBusinessSettingsSections(context,{includeHidden:true}).flatMap(section=>{
  if(section.type==='field')return [renderField(section.field)];
  if(section.type==='list')return [group(section.heading,section.id==='categories'
   ? [h("div", { class: "settings-categories" }, ...v.categories.map(categoryRow)), button('add-category',section.addItemName)]
   : [h("div", { class: "settings-topic-list" }, ...v.topics.map(topic=>topicRow(topic))), button('add-topic',section.addItemName)])];
  return [
   ...(section.id==='advanced' ? [appearanceSection()] : []),
   group(section.heading,[
    ...(section.id==='email' ? [h("div", { class: "settings-email-guide" })] : []),
    ...section.items.map(renderField),
   ]),
  ];
 });
 return h("form", { class: "settings-form" },
  h("div", { class: "settings-host-context" },
   h("p", null, t(snapshot.setupRequired ? '首次使用：选择保存目录，再配置下方 LLM 和研究主题。' : '修改会自动保存并立即生效。')),
   h("label", null, t('保存根目录（本机绝对路径）'), input('vaultRoot',v.vaultRoot)),
   h("p", null, t('模型 API 与 DSH / Claude Code 对话模型独立。密钥留空保留现有值。')),
   h("code", null, snapshot.configPath),
  ),
  h("div", { class: "settings-setup-host" }, settingsSetupGuide(snapshot,firstReportComplete)),
  ...business,
  group('Help & feedback',[
   row('Report a bug', h("a", { href: "https://github.com/tdccccc/arxiv-daily/issues/new?body=-%20arXiv%20Daily%3A%20Workbench", target: "_blank", rel: "noopener noreferrer" }, t('Open'))),
   row('Request a feature', h("a", { href: "https://github.com/tdccccc/arxiv-daily/issues/new", target: "_blank", rel: "noopener noreferrer" }, t('Open'))),
   row('Documentation', h("a", { href: "https://github.com/tdccccc/arxiv-daily/blob/main/docs/getting-started.md", target: "_blank", rel: "noopener noreferrer" }, t('Open'))),
   row('Repository', h("a", { href: "https://github.com/tdccccc/arxiv-daily", target: "_blank", rel: "noopener noreferrer" }, t('Open'))),
  ]),
  h("div", { class: "settings-action-host" }),
  h("p", { class: "settings-action-status", role: "status" }),
  h("p", { class: "form-error", role: "alert", hidden: true }),
  h("div", { class: "dialog-footer" },
   h("span", { class: "settings-save-status", role: "status" }),
   h("button", { type: "button", "data-settings": "retry-save", hidden: true }, t('重试保存')),
   h("button", { type: "button", "data-settings": "discard-close", hidden: true }, t('放弃未保存修改并关闭')),
  ),
 );
}

export function bindSettings(form: HTMLFormElement, snapshot: SettingsSnapshot, request: <T>(url: string, body?: unknown) => Promise<T>, saved: () => Promise<void>, onRun?: (run: import('../server').WorkbenchRun) => void, firstReportComplete = false, options: { appearance?: UiAppearancePreferences; onAppearanceSaved?: (appearance: UiAppearancePreferences, meta?: { closing: boolean }) => Promise<void> } = {}): { flush: () => Promise<boolean>; close: () => Promise<boolean>; dispose: () => void } {
 mountSettingsNavigation(form);
 let modelControl=mountModelCombobox(form);
 let modelOptions:string[]=[];
 let busy=false, revision=snapshot.revision, libraryCancelling=false, libraryRevisionPending=false;
 let currentSnapshot=snapshot;
 let lastCompletedRun: string | undefined;
 let library: (CliLibraryConnectionInspection & { run?: import('../server').WorkbenchRun }) | undefined;
 const get=(name:string) => (form.elements.namedItem(name) as HTMLInputElement)?.value.trim() ?? '';
 const checked=(name:string) => (form.elements.namedItem(name) as HTMLInputElement).checked;
 const find=<T extends HTMLElement=HTMLElement>(selector:string)=>form.querySelector<T>(selector)!;
 function values() {
  const topics=Array.from(form.querySelectorAll<HTMLElement>('.settings-topic')).map(row=>{
   const name=row.querySelector<HTMLInputElement>('[name="topicName"]')!.value.trim();
   const directions=Array.from(row.querySelectorAll<HTMLElement>('[data-direction-id]')).map(line=>({id:line.dataset.directionId!,origin:line.dataset.directionOrigin as Topic['directions'][number]['origin'],text:line.querySelector<HTMLTextAreaElement>('textarea')!.value.trim().replace(/\s*\n+\s*/g,' ')})).filter(direction=>direction.text);
   return {id:row.dataset.topicId!,name,tag:row.dataset.topicTag??'',description:deriveTopicDescription(directions),directions,detail:row.querySelector<HTMLInputElement>('[name="topicDetail"]')!.checked,originalName:row.dataset.originalName??''};
  });
  const tags=new Set(topics.map(topic=>topic.tag).filter(Boolean));
  for(const topic of topics){
   const oldBase=slugify(topic.originalName);
   if(!topic.tag || /^topic-\d+$/.test(topic.tag) || (topic.name!==topic.originalName && oldBase && (topic.tag===oldBase || new RegExp(`^${oldBase}-\\d+$`).test(topic.tag)))){
    tags.delete(topic.tag);const base=slugify(topic.name)||'topic';let tag=base;let suffix=2;while(tags.has(tag))tag=`${base}-${suffix++}`;topic.tag=tag;tags.add(tag);
   }
  }

  const key=(name:string,field='apiKey')=>get(name)&&(form.elements.namedItem(name) as HTMLInputElement)?.dataset.revealed!=='true'?{[field]:get(name)}:{};
  return {maxDailyPapers:Number(get('maxDailyPapers')),vaultRoot:get('vaultRoot'),provider:currentSnapshot.values.provider,baseUrl:get('baseUrl'),model:get('model'),...key('apiKey'),reasoningEffort:get('reasoningEffort'),categories:Array.from(form.querySelectorAll<HTMLElement>('[data-category-row]')).map(row=>row.querySelector<HTMLSelectElement>('select')!.value),topics:topics.map(({originalName,...topic})=>topic),timezone:get('timezoneCustom')||get('timezone'),detailProfile:get('detailProfile'),dailyDir:get('dailyDir'),papersDir:get('papersDir'),linkStyle:get('linkStyle'),summaryLanguage:get('summaryLanguage'),schedule:{enabled:checked('schedule.enabled'),runAtLocal:get('schedule.runAtLocal'),runUntilLocal:get('schedule.runUntilLocal'),tickIntervalMin:Number(get('schedule.tickIntervalMin'))},embedding:{mode:get('embedding.mode'),baseUrl:get('embedding.baseUrl'),model:get('embedding.model'),dimension:Number(get('embedding.dimension')),...key('embedding.apiKey')},pdfParserSidecar:{enabled:checked('pdfParserSidecar.enabled'),capabilitiesUrl:get('pdfParserSidecar.capabilitiesUrl'),parseUrl:get('pdfParserSidecar.parseUrl')},email:{enabled:checked('email.enabled'),mode:get('email.mode'),to:get('email.to'),fromEmail:get('email.fromEmail'),fromName:(form.elements.namedItem('email.fromName') as HTMLInputElement).value,...key('email.apiKey'),...key('email.hostedToken','hostedToken')},logLevel:get('logLevel')};
 }
 function conditions() {
  const context: Partial<BusinessSettingsContext>={emailMode:get('email.mode')==='hosted'?'hosted':'self',embeddingMode:get('embedding.mode')==='remote'?'remote':'local',sidecarEnabled:checked('pdfParserSidecar.enabled'),scheduleEnabled:checked('schedule.enabled')};
  for(const element of Array.from(form.querySelectorAll<HTMLElement>('[data-business-setting]'))) {
   const field=getBusinessSetting(element.dataset.businessSetting as BusinessSettingId,context);
   // As in Obsidian, title/abstract indexing has no structured sidecar consumer (ADR 0013).
   element.hidden=!field.visible||field.id.startsWith('sidecar');element.dataset.settingName=field.name;
   element.querySelector('.setting-item-name')!.textContent=t(field.name);
   if(field.id==='scheduleEnabled')element.querySelector('input')!.setAttribute('aria-label',t(field.name));
   element.querySelector('.setting-item-description')!.textContent=t(field.description);
  }
  const hosted=isBusinessSettingVisible('hostedToken',context);
  for(const element of Array.from(form.querySelectorAll<HTMLElement>('[data-email-hosted]:not([data-business-setting])')))element.hidden=!hosted;
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
   else if(get('embedding.mode')==='remote' && library.status.kind!=='authorized')description=t(library.status.kind==='authorization-invalidated'?'Selected: {0}. The embedding endpoint changed, so building the index asks you to confirm which titles and abstracts leave this device.':'Selected: {0}. Remote embedding sends titles and abstracts off this device — building the index asks you to confirm first.',label);
   else description=t(get('embedding.mode')==='remote'?'Connected: {0}. Authorized to embed titles and abstracts remotely. Build the search index next.':'Selected: {0}. Local embedding stays on this device. Build the search index to search these PDFs.',label);
  }
  find('[data-library-row] .setting-item-description').textContent=description;
  find('.settings-library-controls').replaceChildren(
   ...([['library-connect',row.chooseFolder],['library-build',row.primary],['library-cancel',row.cancel],['library-revoke',row.revoke]] as const)
    .flatMap(([action,item])=>!item||typeof item==='string'?[]:[h("button",{type:"button","data-settings":action,disabled:item.disabled},t(item.label))]),
  );
 }
 async function refreshLibrary() { library=await request<typeof library>('api/settings/library'); if(form.isConnected){renderLibrary();if(library?.run?.status==='running'){lockLibraryInputs(true);onRun?.(library.run);}} }
 function refreshVisibleSetupGuide() {
  const host=find('.settings-setup-host');
  if(!host.querySelector('.settings-setup'))return;
  const content=settingsSetupGuide(currentSnapshot,firstReportComplete);
  if(!content || !host.firstElementChild || !content.isEqualNode(host.firstElementChild))host.replaceChildren(...(content?[content]:[]));
 }
 let disposed=false, rootEditing=false, closing=false, rootBlocksClose=false;
 let timer: number|undefined;
 let draining: Promise<boolean>|undefined;
 let actionGate: Promise<void>|undefined;
 let appliedAppearance={...(options.appearance??DEFAULT_UI_APPEARANCE)};
 let lastValues=JSON.stringify(values());
 const appearanceValues=():UiAppearancePreferences=>({theme:get('appearance.theme') as UiAppearancePreferences['theme'],language:get('appearance.language') as UiAppearancePreferences['language']});
 function saveStatus(message:string,failed=false) {
  if(disposed||!form.isConnected)return;
  const status=find('.settings-save-status');
  status.textContent=t(message);
  // The save root message blocks every business save during first run; call it out in the
  // (always-visible, non-scrolling) footer so it doesn't read as just another transient status.
  status.classList.toggle('is-warning',message==='请先填写完整的保存根目录。');
  find<HTMLButtonElement>('[data-settings="retry-save"]').hidden=!failed;
  find<HTMLButtonElement>('[data-settings="discard-close"]').hidden=!failed;
 }
 function localizeForm(selectedAppearance=appearanceValues()) {
  const current=values(), topics=find('.settings-topic-list');
  const secretStates=Array.from(form.querySelectorAll<HTMLInputElement>('.settings-secret input')).map(el=>({name:el.name,value:el.value,type:el.type,revealed:el.dataset.revealed}));
  const scroll=find('.settings-content').scrollTop;
  const actionContent=find('.settings-action-host');
  const hadGuide=Boolean(find('.settings-setup-host').querySelector('.settings-setup'));
  const renderSnapshot:SettingsSnapshot={...currentSnapshot,values:{...currentSnapshot.values,...current,summaryLanguage:current.summaryLanguage as 'zh'|'en',reasoningEffort:current.reasoningEffort,detailProfile:current.detailProfile as SettingsSnapshot['values']['detailProfile'],linkStyle:current.linkStyle as SettingsSnapshot['values']['linkStyle'],logLevel:current.logLevel as SettingsSnapshot['values']['logLevel'],embedding:{...currentSnapshot.values.embedding,...current.embedding,mode:current.embedding.mode as 'local'|'remote'},email:{...currentSnapshot.values.email,...current.email,mode:current.email.mode as 'self'|'hosted'}}};
  const rendered=settingsForm(renderSnapshot,firstReportComplete,selectedAppearance);
  form.replaceChildren(...Array.from(rendered.children));
  find('.settings-topic-list').replaceWith(topics);find('.settings-action-host').replaceWith(actionContent);
  for(const node of Array.from(topics.querySelectorAll('button')))node.textContent=t(node.textContent??'');
  for(const node of Array.from(topics.querySelectorAll('.settings-topic-body > div')))if(node.firstChild?.nodeType===Node.TEXT_NODE)node.firstChild.textContent=t(node.firstChild.textContent?.trim()??'');
  for(const topic of Array.from(topics.querySelectorAll<HTMLElement>('.settings-topic'))){topic.querySelector('.settings-topic-star')!.setAttribute('title',t('Detail report enabled'));if(!topic.querySelector<HTMLInputElement>('[name="topicName"]')!.value.trim())topic.querySelector('.settings-topic-name')!.textContent=t('(unnamed)');}
  for(const label of Array.from(topics.querySelectorAll('label')))if(label.firstChild?.nodeType===Node.TEXT_NODE)label.firstChild.textContent=t(label.firstChild.textContent?.trim()??'');
  for(const control of Array.from(topics.querySelectorAll('input,textarea')))for(const attr of ['aria-label','placeholder'])if(control.hasAttribute(attr))control.setAttribute(attr,t(control.getAttribute(attr)!));
  if(!hadGuide)find('.settings-setup-host').replaceChildren();
  for(const state of secretStates){const el=form.elements.namedItem(state.name) as HTMLInputElement;el.value=state.value;el.type=state.type;if(state.revealed)el.dataset.revealed=state.revealed;el.closest('.settings-secret')!.querySelector('button')!.textContent=t(el.type==='text'?'Hide':'Show');}
  mountSettingsNavigation(form);modelControl=mountModelCombobox(form);if(modelOptions.length)modelControl.setOptions(modelOptions);
  conditions();renderLibrary();find('.settings-content').scrollTop=scroll;
 }
 function stripSecrets(value:ReturnType<typeof values>) {
  const clean=structuredClone(value) as ReturnType<typeof values>&{apiKey?:string};delete clean.apiKey;delete (clean.embedding as {apiKey?:string}).apiKey;delete (clean.email as {apiKey?:string}).apiKey;delete (clean.email as {hostedToken?:string}).hostedToken;return clean;
 }
 async function drain(force:boolean):Promise<boolean> {
  rootBlocksClose=false;
  try {
   while(!disposed&&form.isConnected){
    if(actionGate)await actionGate;
    const next=values(), signature=JSON.stringify(next);
    const nextAppearance=appearanceValues();
    const businessChanged=signature!==lastValues;
    const waitingForRoot=(rootEditing&&!force)||!next.vaultRoot||!(/^(?:\/|\\\\|[A-Za-z]:[\\/]|~(?:\/|$))/.test(next.vaultRoot));
    if(businessChanged&&!waitingForRoot){
     if(libraryRevisionPending){saveStatus('索引任务结束后将自动保存。');return false;}
     saveStatus('正在自动保存…');
     const secrets=Array.from(form.querySelectorAll<HTMLInputElement>('.settings-secret input')).map(el=>({el,value:el.value,revealed:el.dataset.revealed}));
     const result=await request<SettingsSnapshot>('api/settings',{revision,values:next});
     if(disposed||!form.isConnected)return false;
     revision=result.revision;currentSnapshot=result;
     lastValues=JSON.stringify(stripSecrets(next));
     for(const {el,value,revealed} of secrets)if(el.value===value&&!revealed&&value){el.value='';delete el.dataset.revealed;el.type='password';el.closest('.settings-secret')!.querySelector('button')!.textContent=t('Show');el.closest('.settings-secret')!.querySelector('small')!.textContent=t('Saved. Leave blank to keep unchanged.');}
     refreshVisibleSetupGuide();
     continue;
    }
    if(JSON.stringify(nextAppearance)!==JSON.stringify(appliedAppearance)){
     saveStatus('正在自动保存…');await request('api/preferences',{appearance:nextAppearance});
     if(disposed||!form.isConnected)return false;
     const selectedAppearance=appearanceValues();
     const changedLanguage=appliedAppearance.language!==nextAppearance.language;
     appliedAppearance=nextAppearance;setUiLanguage(nextAppearance.language);
     if(changedLanguage)localizeForm(selectedAppearance);
     // `closing` tells the caller whether this change was discovered mid-`close()`: in that case
     // the dialog is about to be torn down anyway, so it should defer any heavier reaction (like
     // relocalizing the main view) until the dialog is actually gone, instead of reacting here.
     await options.onAppearanceSaved?.(nextAppearance,{closing});
     continue;
    }
    if(waitingForRoot&&(businessChanged||currentSnapshot.setupRequired)){saveStatus('请先填写完整的保存根目录。');if(force){find('[role="alert"]').hidden=false;find('[role="alert"]').textContent=t('请先填写完整的保存根目录。');}rootBlocksClose=true;return false;}
    find('[role="alert"]').hidden=true;saveStatus('已自动保存');return true;
   }
   return false;
  }catch(error){if(!disposed&&form.isConnected){find('[role="alert"]').hidden=false;find('[role="alert"]').textContent=t(error instanceof Error?error.message:'自动保存失败，修改仍保留。');saveStatus('自动保存失败，修改仍保留。',true);}return false;}
 }
 async function flush(force=true):Promise<boolean> {
  if(timer){window.clearTimeout(timer);timer=undefined;}
  if(disposed)return false;
  if(draining){const success=await draining;if(!success)return false;return flush(force);}
  const pending=drain(force);draining=pending;
  try{return await pending;}finally{if(draining===pending)draining=undefined;}
 }
 function scheduleSave(immediate=false) {
  if(disposed)return;form.dataset.edited='true';if(timer)window.clearTimeout(timer);
  saveStatus('修改尚未保存…');
  if(immediate)void flush(false);else timer=window.setTimeout(()=>{timer=undefined;void flush(false);},500);
 }
 async function saveDraft(refreshGuide=false){if(!await flush())throw new Error(t('请先解决自动保存问题，再继续操作。'));if(refreshGuide)refreshVisibleSetupGuide();return currentSnapshot;}
 async function close():Promise<boolean>{
  if(closing)return false;closing=true;
  try{
   if(!await flush()){
    // First run only: an empty/invalid save root blocks every business save (there is nowhere to
    // write the config yet), which used to trap the dialog open forever (close had no other
    // trigger than this). Let it close anyway — the draft business edits are discarded (they
    // never reached the server) and the main view explains setup isn't finished with a way back
    // in. A genuine autosave failure (network, revision conflict, …) still blocks close so
    // nothing already-typed is lost silently.
    if(!currentSnapshot.setupRequired||!rootBlocksClose)return false;
   }
   await saved();return true;
  }finally{closing=false;}
 }
 async function task(operation:()=>Promise<void>, actionButton?: HTMLButtonElement) {
  if(busy)return;busy=true;find('[role="alert"]').hidden=true;

  const originalLabel=actionButton?.textContent ?? '';
  if(actionButton?.dataset.settings==='models')find('.settings-model-status').textContent=t('Fetching models…');
  if(actionButton){actionButton.disabled=true;if(actionButton.dataset.settings==='models')actionButton.textContent=t('Fetching…');else if(actionButton.dataset.settings?.startsWith('email-'))actionButton.textContent=t('Sending…');}
  try{await operation();}catch(error){if(form.isConnected){if(actionButton?.dataset.settings==='models')find('.settings-model-status').textContent=error instanceof Error?error.message:t('Could not load models.');find('[role="alert"]').hidden=false;find('[role="alert"]').textContent=error instanceof Error?error.message:t('操作失败，请重试。');}}
  finally{busy=false;if(actionButton){actionButton.disabled=false;actionButton.textContent=originalLabel;}}
 }
 async function action(name:string,extra:Record<string,unknown>={}) {
  let release!:()=>void;actionGate=new Promise<void>(resolve=>{release=resolve;});
  let result:{settings?:SettingsSnapshot;library?:typeof library;models?:string[];run?:import('../server').WorkbenchRun};
  try{result=await request('api/settings/action',{revision,action:name,...extra});if(result.settings){revision=result.settings.revision;currentSnapshot=result.settings;}}finally{release();actionGate=undefined;}
  if(!form.isConnected)return;
  if(result.settings)revision=result.settings.revision;
  if(result.library){library=result.library;renderLibrary();}
  if(result.models){
   modelOptions=result.models;modelControl.setOptions(result.models);
   find('.settings-model-status').textContent=result.models.length?t('Loaded {0} models. Choose from the list or type a model name.', result.models.length):t('No models returned; type a model name to continue.');
  }
  if(result.run){onRun?.(result.run);if(name==='library-build'&&library){library.run=result.run;renderLibrary();lockLibraryInputs(result.run.status==='running');}find('.settings-action-status').textContent=result.run.label;}
 }
 function lockLibraryInputs(locked: boolean) {
  // The dialog's × (close-dialog) button lives outside this form (see `showDialog` in app.ts), so
  // it's never touched here regardless — it keeps working to close the dialog while locked.
  for(const control of Array.from(form.querySelectorAll<HTMLInputElement|HTMLSelectElement|HTMLTextAreaElement|HTMLButtonElement>('input,select,textarea,button:not([data-settings="library-cancel"]):not([data-settings-nav])')))control.disabled=locked;
  if(!locked)renderLibrary();
 }
 form.addEventListener('workbench-run',event=>{
  const run=(event as CustomEvent<import('../server').WorkbenchRun|null>).detail;
  if(run?.status==='completed' && run.id!==lastCompletedRun){lastCompletedRun=run.id;void request<{recentRuns?:Array<{status:string;outcome?:string}>}>('api/status').then(status=>{firstReportComplete=status.recentRuns?.some(isCompletedDiscovery)??firstReportComplete;if(form.isConnected)refreshVisibleSetupGuide();}).catch(()=>{});}
  if(library?.run && run?.id===library.run.id){
   library.run=run;if(run.status!=='running')libraryCancelling=false;renderLibrary();
   if(run.status!=='running'){libraryRevisionPending=true;void Promise.all([request<SettingsSnapshot>('api/settings'),refreshLibrary()]).then(([current])=>{revision=current.revision;currentSnapshot=current;}).catch(error=>{if(form.isConnected){find('[role="alert"]').hidden=false;find('[role="alert"]').textContent=error instanceof Error?error.message:t('Could not reload settings.');}}).finally(()=>{libraryRevisionPending=false;lockLibraryInputs(false);void flush(false);});}
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
  if(target instanceof HTMLInputElement&&target.name==='vaultRoot')rootEditing=false;
  if(target instanceof HTMLElement&&target.getAttribute('name')!=='libraryPath')scheduleSave(true);
 });
 function updateTopic(event:Event) {
  const target=event.target;if(!(target instanceof HTMLInputElement))return;
  const topic=target.closest<HTMLElement>('.settings-topic');if(!topic)return;
  const name=topic.querySelector<HTMLInputElement>('[name="topicName"]')!.value.trim()||'(unnamed)';
  topic.dataset.settingName=name;topic.querySelector('.settings-topic-name')!.textContent=topic.querySelector<HTMLInputElement>('[name="topicName"]')!.value.trim()||t('(unnamed)');
  topic.querySelector('.settings-topic-star')!.textContent=topic.querySelector<HTMLInputElement>('[name="topicDetail"]')!.checked?'★':'';
 }
 form.addEventListener('input',event=>{if(event.target instanceof HTMLInputElement)delete event.target.dataset.revealed;updateTopic(event);const target=event.target as HTMLInputElement;if(target.name==='libraryPath')return;if(target.name==='vaultRoot'){form.dataset.edited='true';rootEditing=true;saveStatus('修改尚未保存…');return;}scheduleSave();});
 form.addEventListener('click',event=>{
  const setup=event.target instanceof HTMLElement?event.target.closest<HTMLButtonElement>('[data-setup-action]'):null;
  if(setup){
   const name=setup.dataset.setupAction;
   if(name==='llm'||name==='arxiv'||name==='topics'){
    const heading=Array.from(form.querySelectorAll<HTMLElement>('[data-settings-heading]')).find(h=>h.dataset.settingsKey===({llm:'LLM',arxiv:'arXiv categories',topics:'Research topics'})[name]);
    heading?.scrollIntoView?.({block:'start',behavior:'smooth'});heading?.parentElement?.querySelector<HTMLElement>('input,select,button')?.focus();return;
   }
   void task(async()=>{
    if(name==='dashboard'){await close();return;}
    if(name==='enable'){find<HTMLInputElement>('[name="schedule.enabled"]').checked=true;conditions();}
    await saveDraft(true);
    if(name==='generate'){const result=await request<{run:import('../server').WorkbenchRun}>('api/runs',{kind:'daily'});onRun?.(result.run);}
   },setup);return;
  }
  const target=event.target instanceof HTMLElement?event.target.closest<HTMLButtonElement>('[data-settings]'):null;if(!target)return;
  const name=target.dataset.settings!;
  if(name==='retry-save'){void flush();return;}
  if(name==='discard-close'){
   find('.settings-action-host').replaceChildren(
    h("div", { class: "settings-confirm", role: "group", "aria-label": t('放弃未保存修改并关闭') },
     h("p", null, t('尚未保存的修改将被放弃，已保存的设置保持不变。')),
     button('confirm-discard-close','放弃修改并关闭'),
     button('cancel-action','Cancel'),
    ),
   );
   return;
  }
  if(name==='confirm-discard-close'){if(timer){window.clearTimeout(timer);timer=undefined;}void (async()=>{if(draining)await draining;if(!disposed)await saved();})();return;}
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
  if(name==='add-topic'){find('.settings-topic-list').append(topicRow({id:crypto.randomUUID(),name:'',tag:'',description:'',directions:[],detail:true},true));return;}
  if(name==='add-direction'){target.closest('.settings-topic')!.querySelector('.settings-topic-directions')!.append(directionRow({id:crypto.randomUUID(),text:'',origin:'manual'}));return;}
  if(name==='remove-direction'){target.closest('[data-direction-id]')!.remove();scheduleSave(true);return;}
  if(name==='remove-topic'){target.closest('.settings-topic')!.remove();scheduleSave(true);return;}
  if(name==='add-category'){const current=values().categories;const next=ARXIV_CATEGORIES.flatMap(g=>g.categories).find(c=>!current.includes(c.id));find('.settings-categories').append(categoryRow(next?.id??'cs.LG',current.length));conditions();scheduleSave(true);return;}
  if(name==='remove-category'){if(form.querySelectorAll('[data-category-row]').length>1)target.closest('[data-category-row]')!.remove();Array.from(form.querySelectorAll<HTMLElement>('[data-category-row]')).forEach((row,i)=>{row.dataset.settingName=String(i+1);row.querySelector('.setting-item-name')!.textContent=String(i+1);});conditions();scheduleSave(true);return;}
  if(name==='library-connect'){
   find('.settings-action-host').replaceChildren(
    h("div", { class: "settings-confirm", role: "group", "aria-label": t('Choose library folder') },
     h("label", null, `${t('PDF folder absolute path')} `, input('libraryPath','')),
     button('confirm-library-connect','Choose folder'),
     button('cancel-action','Cancel'),
    ),
   );
   find<HTMLInputElement>('[name="libraryPath"]').focus();return;
  }
  if(name==='cancel-action'){find('.settings-action-host').replaceChildren();return;}
  void task(async()=>{
   if(name==='library-cancel'){if(library?.run){libraryCancelling=true;renderLibrary();try{const result=await request<{run:import('../server').WorkbenchRun}>('api/runs/cancel',{id:library.run.id});library.run=result.run;libraryCancelling=result.run.status==='running';onRun?.(result.run);}catch(error){libraryCancelling=false;throw error;}finally{renderLibrary();}}return;}
   await saveDraft();
   if(name==='confirm-library-connect'){await action('library-connect',{path:get('libraryPath')});find('.settings-action-host').replaceChildren();return;}
   if(name==='library-build'){
    await refreshLibrary();
    if(library?.status.kind!=='authorized' && library?.disclosure){
     const lines=[t('Folder: {0}',library.disclosure.selectedRoot),t('Eligible files: {0}',library.disclosure.eligibleExtensions.join(', ')),t('Processing depth: {0}',t(library.disclosure.processingDepth==='full-text'?'Titles and abstracts':library.disclosure.processingDepth)),t('Model endpoint: {0}',library.disclosure.endpoint),...(library.disclosure.embeddingEndpoint?[t('Embedding endpoint: {0}',library.disclosure.embeddingEndpoint)]:[])];
     find('.settings-action-host').replaceChildren(
      h("div", { class: "settings-confirm", role: "group", "aria-label": t('Library processing consent') },
       h("h4", null, t('Confirm library processing')),
       h("pre", null, lines.join('\n')),
       button('confirm-library-build','Confirm and build index'),
       button('cancel-action','Cancel'),
      ),
     );
     return;
    }
   }
   if(name==='confirm-library-build'){if(!library?.disclosure)throw new Error(t('请重新查看授权范围。'));await action('library-build',{fingerprint:library.disclosure.authorizationFingerprint});find('.settings-action-host').replaceChildren();return;}
   await action(name);
  },target);
 });
 form.addEventListener('submit',event=>{event.preventDefault();void close();});
 conditions();
 if(!snapshot.setupRequired)void refreshLibrary().catch(error=>{if(form.isConnected)find('[data-library-row] .setting-item-description').textContent=error instanceof Error?error.message:t('Unable to load library.');});
 return {flush:()=>flush(),close,dispose(){disposed=true;if(timer)window.clearTimeout(timer);}};
}
