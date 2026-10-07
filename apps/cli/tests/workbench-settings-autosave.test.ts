// @vitest-environment happy-dom
import { afterEach, expect, it, vi } from "vitest";
import { bindSettings, settingsForm, type SettingsSnapshot } from "../src/workbench/web/settings";
import { readWorkbenchSettings } from "../src/workbench/settings";
import { setUiLanguage } from "../src/workbench/web/i18n";
const dispose: Array<() => void> = [];
afterEach(() => { dispose.splice(0).forEach(fn => fn()); document.body.innerHTML=""; setUiLanguage("zh"); });
const settle=async()=>{await new Promise(resolve=>setTimeout(resolve,0));};
async function setup(first=false, custom?: (url:string,body?:unknown)=>Promise<unknown>) {
 let snapshot=await readWorkbenchSettings('/tmp/arxiv-settings-autosave-fixture/nonexistent.toml');
 snapshot={...snapshot,setupRequired:first,revision:first?null:'r1',values:{...snapshot.values,vaultRoot:first?'':'/notes',model:'old',apiKeyConfigured:true,schedule:{...snapshot.values.schedule,enabled:false}}};
 let sequence=1;
 const request=vi.fn(async<T>(url:string,body?:unknown):Promise<T>=>{
  if(custom){const result=await custom(url,body);if(result!==undefined)return result as T;}
  if(url==='api/settings'&&body){const payload=body as {values:Partial<SettingsSnapshot['values']>};snapshot={...snapshot,revision:`r${++sequence}`,setupRequired:false,values:{...snapshot.values,...payload.values}};return snapshot as T;}
  if(url==='api/settings/library')return {status:{kind:'disconnected'},disclosure:null} as T;
  if(url==='api/preferences')return {} as T;
  if(url==='api/settings/secret')return {value:'previous-secret'} as T;
  if(url==='api/settings/action')return {models:['selected-model']} as T;
  return snapshot as T;
 });
 document.body.innerHTML=settingsForm(snapshot);
 const form=document.querySelector('form')!; const closed=vi.fn().mockResolvedValue(undefined), appearance=vi.fn().mockResolvedValue(undefined);
 const binding=bindSettings(form,snapshot,request,closed,undefined,false,{appearance:{theme:'light',language:'zh'},onAppearanceSaved:appearance});
 if(binding)dispose.push(binding.dispose);
 await settle(); return {form,request,binding,closed,appearance};
}
const field=(form:HTMLFormElement,name:string)=>form.elements.namedItem(name) as HTMLInputElement;
function change(form:HTMLFormElement,name:string,value:string|boolean){const element=field(form,name);if(typeof value==='boolean')element.checked=value;else element.value=value;element.dispatchEvent(new Event('change',{bubbles:true}));}
const posts=(request:ReturnType<typeof vi.fn>)=>request.mock.calls.filter(([url,body])=>url==='api/settings'&&body);

it('persists edits without Save and leaves scheduling disabled merely by opening settings',async()=>{
 const {form,request}=await setup();expect(posts(request)).toHaveLength(0);expect(form.querySelector('[type="submit"]')).toBeNull();
 expect(field(form,'schedule.enabled').checked).toBe(false);
 change(form,'model','new-model');await vi.waitFor(()=>expect(posts(request)).toHaveLength(1));
 expect((posts(request)[0]![1] as any).values).toMatchObject({model:'new-model',schedule:{enabled:false}});
 expect(form.isConnected).toBe(true);expect(form.querySelector('.settings-save-status')?.textContent).toContain('已自动保存');
});
it('queues edits made while saving and sends each write with the returned revision',async()=>{
 let release!:(value:unknown)=>void;let captured:any;
 const {form,request}=await setup(false,async(url,body)=>{if(url==='api/settings'&&body&&!captured){captured=body;return new Promise(resolve=>{release=resolve;});}});
 change(form,'model','first');await settle();change(form,'summaryLanguage','en');await settle();expect(posts(request)).toHaveLength(1);
 const initial=await readWorkbenchSettings('/tmp/arxiv-settings-autosave-fixture/nonexistent.toml');
 release({...initial,revision:'after-first',setupRequired:false,values:{...initial.values,...captured.values}});
 await vi.waitFor(()=>expect(posts(request)).toHaveLength(2));
 expect((posts(request)[1]![1] as any)).toMatchObject({revision:'after-first',values:{model:'first',summaryLanguage:'en'}});
});
it('keeps failed edits open and retries before closing',async()=>{
 let fail=true;const {form,binding,closed}=await setup(false,async(url,body)=>{if(url==='api/settings'&&body&&fail)throw new Error('Write conflict');});
 change(form,'model','keep-me');await settle();expect(form.querySelector('[role="alert"]')?.textContent).toContain('Write conflict');
 expect(await binding.close()).toBe(false);expect(closed).not.toHaveBeenCalled();expect(field(form,'model').value).toBe('keep-me');
 fail=false;expect(await binding.close()).toBe(true);expect(closed).toHaveBeenCalledOnce();
});
it('never writes revealed secrets back and clears newly saved secrets without losing configured status',async()=>{
 const {form,request}=await setup();(form.querySelector('[data-settings="show-secret"]') as HTMLButtonElement).click();await settle();
 expect(field(form,'apiKey').value).toBe('previous-secret');change(form,'model','next');await settle();expect((posts(request)[0]![1] as any).values.apiKey).toBeUndefined();
 field(form,'apiKey').value='new-secret';field(form,'apiKey').dispatchEvent(new Event('input',{bubbles:true}));field(form,'apiKey').dispatchEvent(new Event('change',{bubbles:true}));await settle();
 expect((posts(request).at(-1)![1] as any).values.apiKey).toBe('new-secret');expect(field(form,'apiKey').value).toBe('');
 expect(field(form,'apiKey').closest('.settings-secret')?.textContent).toContain('已保存');
});
it('waits for a complete first-use root and flushes on close once chosen',async()=>{
 const {form,request,binding}=await setup(true);field(form,'vaultRoot').value='/chosen';
 expect(await binding.close()).toBe(true);expect(posts(request)).toHaveLength(1);expect((posts(request)[0]![1] as any).values.vaultRoot).toBe('/chosen');
});
it('closes during first-use even without a save root, discarding the unsaved draft',async()=>{
 const {form,binding,closed}=await setup(true);change(form,'model','draft');await settle();
 expect(await binding.close()).toBe(true);expect(closed).toHaveBeenCalledOnce();
});
it('flushes pending edits before an explicit model fetch and does not invoke tasks automatically',async()=>{
 const {form,request}=await setup();field(form,'model').value='draft-model';field(form,'model').dispatchEvent(new Event('input',{bubbles:true}));
 (form.querySelector('[data-settings="models"]') as HTMLButtonElement).click();await vi.waitFor(()=>expect(request.mock.calls.some(([url])=>url==='api/settings/action')).toBe(true));
 const calls=request.mock.calls.filter(([url])=>url==='api/settings'||url==='api/settings/action');expect(calls.map(call=>call[0])).toEqual(['api/settings','api/settings/action']);
});
it('automatically applies appearance while keeping unsaved first-use business inputs',async()=>{
 const {form,request,appearance}=await setup(true);field(form,'model').value='user-draft';field(form,'model').dispatchEvent(new Event('input',{bubbles:true}));
 change(form,'appearance.language','en');await vi.waitFor(()=>expect(appearance).toHaveBeenCalledWith({theme:'light',language:'en'},{closing:false}));
 expect(posts(request)).toHaveLength(0);expect(field(form,'model').value).toBe('user-draft');expect(form.textContent).toContain('Appearance');expect(form.isConnected).toBe(true);
});

it('serializes input and close behind an explicit action and preserves the action revision',async()=>{
 let release!:(value:unknown)=>void;
 const {form,request,binding,closed}=await setup(false,async(url)=>{if(url==='api/settings/action')return new Promise(resolve=>{release=resolve;});});
 (form.querySelector('[data-settings="models"]') as HTMLButtonElement).click();await vi.waitFor(()=>expect(release).toBeTypeOf('function'));
 change(form,'model','during-action');await settle();const closing=binding.close();await settle();expect(posts(request)).toHaveLength(0);expect(closed).not.toHaveBeenCalled();
 const initial=await readWorkbenchSettings('/tmp/arxiv-settings-autosave-fixture/nonexistent.toml');
 release({settings:{...initial,revision:'action-revision',setupRequired:false,values:{...initial.values,vaultRoot:'/notes',model:'old'}},models:['model']});
 expect(await closing).toBe(true);expect((posts(request)[0]![1] as any).revision).toBe('action-revision');expect((posts(request)[0]![1] as any).values.model).toBe('during-action');
});

it('keeps drafts, newly added empty directions and topic identities when language changes',async()=>{
 const {form}=await setup(true);(form.querySelector('[data-settings="add-topic"]') as HTMLButtonElement).click();
 const topic=form.querySelector<HTMLElement>('.settings-topic:last-child')!;
 (topic.querySelector('[name="topicName"]') as HTMLInputElement).value='Theme';
 (topic.querySelector('[data-settings="add-direction"]') as HTMLButtonElement).click();
 const id=topic.dataset.topicId, directionCount=topic.querySelectorAll('[data-direction-id]').length;
 change(form,'appearance.language','en');await vi.waitFor(()=>expect(form.querySelector('[data-settings-key="Appearance"]')?.textContent).toBe('Appearance'));
 const restored=form.querySelector<HTMLElement>(`[data-topic-id="${id}"]`)!;
 expect((restored.querySelector('[name="topicName"]') as HTMLInputElement).value).toBe('Theme');expect(restored.querySelectorAll('[data-direction-id]')).toHaveLength(directionCount);
});

it('preserves and queues appearance selections changed during an outstanding preference save',async()=>{
 let release!:(value:unknown)=>void;let first=true;
 const {form,request,appearance}=await setup(false,async(url)=>{if(url==='api/preferences'&&first){first=false;return new Promise(resolve=>{release=resolve;});}});
 change(form,'appearance.language','en');await vi.waitFor(()=>expect(release).toBeTypeOf('function'));
 change(form,'appearance.theme','dark');change(form,'appearance.language','zh');
 release({});await vi.waitFor(()=>expect(appearance).toHaveBeenLastCalledWith({theme:'dark',language:'zh'},{closing:false}));
 const writes=request.mock.calls.filter(([url])=>url==='api/preferences');expect(writes.map(([,body])=>body)).toEqual([{appearance:{theme:'light',language:'en'}},{appearance:{theme:'dark',language:'zh'}}]);
 expect(field(form,'appearance.theme').value).toBe('dark');expect(field(form,'appearance.language').value).toBe('zh');
});

it('offers an explicit discard escape after persistent revision conflicts without more writes',async()=>{
 const {form,request,closed}=await setup(false,async(url,body)=>{if(url==='api/settings'&&body)throw new Error('Persistent revision conflict');});
 change(form,'model','unsaved');await settle();
 (form.querySelector('[data-settings="retry-save"]') as HTMLButtonElement).click();await settle();expect(posts(request)).toHaveLength(2);
 (form.querySelector('[data-settings="discard-close"]') as HTMLButtonElement).click();
 expect(form.querySelector('.settings-action-host [role="group"]')).toBeTruthy();
 (form.querySelector('[data-settings="cancel-action"]') as HTMLButtonElement).click();expect(closed).not.toHaveBeenCalled();expect(field(form,'model').value).toBe('unsaved');
 (form.querySelector('[data-settings="discard-close"]') as HTMLButtonElement).click();
 (form.querySelector('[data-settings="confirm-discard-close"]') as HTMLButtonElement).click();await settle();
 expect(closed).toHaveBeenCalledOnce();expect(posts(request)).toHaveLength(2);
});

it('translates category controls and empty topic UI without changing selection or typed topics',async()=>{
 const {form}=await setup(true);(form.querySelector('[data-settings="add-topic"]') as HTMLButtonElement).click();
 const before=Array.from(form.querySelectorAll<HTMLSelectElement>('[name="category"]')).map(el=>el.value);
 change(form,'appearance.language','en');await vi.waitFor(()=>expect(form.querySelector('[data-settings-key="Appearance"]')?.textContent).toBe('Appearance'));
 expect(Array.from(form.querySelectorAll<HTMLSelectElement>('[name="category"]')).map(el=>el.value)).toEqual(before);
 expect(form.querySelector('[name="category"]')?.getAttribute('aria-label')).toBe('Category 1');
 expect(form.querySelector('[name="category"] optgroup')?.getAttribute('label')).not.toMatch(/[\u4e00-\u9fff]/);
 expect(form.querySelector('.settings-topic:last-child .settings-topic-name')?.textContent).toBe('(unnamed)');
 expect(form.querySelector('.settings-topic:last-child .settings-topic-body')?.textContent).not.toMatch(/[\u4e00-\u9fff]/);
});

it('debounces text edits but waits for a root-field change before saving a partial path',async()=>{
 const {form,request}=await setup();field(form,'model').value='debounced';field(form,'model').dispatchEvent(new Event('input',{bubbles:true}));
 expect(posts(request)).toHaveLength(0);await vi.waitFor(()=>expect(posts(request)).toHaveLength(1));
 field(form,'vaultRoot').value='/still-typing';field(form,'vaultRoot').dispatchEvent(new Event('input',{bubbles:true}));
 change(form,'summaryLanguage','en');await settle();expect(posts(request)).toHaveLength(1);
 field(form,'vaultRoot').value='/complete-root';field(form,'vaultRoot').dispatchEvent(new Event('change',{bubbles:true}));
 await vi.waitFor(()=>expect(posts(request)).toHaveLength(2));expect((posts(request)[1]![1] as any).values).toMatchObject({vaultRoot:'/complete-root',summaryLanguage:'en'});
});

it('lets the server validate a complete UNC save root',async()=>{
 const {form,binding,request}=await setup(true);field(form,'vaultRoot').value='\\\\server\\share';
 expect(await binding.close()).toBe(true);expect((posts(request)[0]![1] as any).values.vaultRoot).toBe('\\\\server\\share');
});
