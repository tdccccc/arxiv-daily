// @vitest-environment happy-dom
import { afterEach, expect, it, vi } from 'vitest';
import { mountWorkbench } from '../src/workbench/web/app';
import { setUiLanguage } from '../src/workbench/web/i18n';
import { readWorkbenchSettings } from '../src/workbench/settings';
let dispose=()=>{};
afterEach(()=>{dispose();document.body.innerHTML='';setUiLanguage('zh');vi.restoreAllMocks();});
it('loads saved interface appearance and switches language without changing summary language or user content',async()=>{
 setUiLanguage('zh');history.replaceState({},'', '/capability/');
 let appearance={theme:'light',language:'zh'};
 let snapshot=await readWorkbenchSettings('/tmp/arxiv-ui-language-test/nonexistent/config.toml');
 snapshot={...snapshot,setupRequired:false,revision:'r1',values:{...snapshot.values,vaultRoot:'/test/notes',model:'test-model',apiKeyConfigured:true,summaryLanguage:'zh'}};
 const json=(value:unknown)=>new Response(JSON.stringify(value),{status:200});
 const fetcher=async(url:RequestInfo|URL,init?:RequestInit)=>{
  const route=String(url).split('?')[0];
  if(route==='api/preferences'){if(init?.method==='POST'){const body=JSON.parse(String(init.body));if(body.appearance)appearance=body.appearance;}return json({sidebarWidth:null,sidebarCollapsed:false,appearance});}
  if(route==='api/settings'){if(init?.method==='POST')snapshot={...snapshot,values:{...snapshot.values,...JSON.parse(String(init.body)).values}};return json(snapshot);}
  if(route==='api/status')return json({llm:{ready:true},topics:[],categories:['cs.AI'],output:{summaryLanguage:'zh'},recentRuns:[]});
  if(route==='api/settings/library')return json({status:{kind:'disconnected'},disclosure:null});
  if(route==='api/calendar')return json({month:'2026-10',today:'2026-10-04',timezone:'UTC',previousMonth:'2026-09',nextMonth:'2026-11',cells:[]});
  if(route==='api/runs/current')return json({run:null});
  if(route==='api/papers')return json({papers:[],total:0,nextOffset:null,topics:[],counts:{},day:null});
  throw new Error(String(url));
 };
 const root=document.createElement('div');document.body.append(root);dispose=mountWorkbench(root,{fetch:fetcher});
 await expect.poll(()=>root.querySelector('.paper-workspace')!==null).toBe(true);
 expect(root.querySelector('[data-action=theme]')).toBeNull();
 root.querySelector<HTMLButtonElement>('[data-action=settings]')!.click();
 await expect.poll(()=>root.querySelector('[name="appearance.language"]')!==null).toBe(true);
 (root.querySelector('[name="appearance.language"]') as HTMLSelectElement).value='en';
 (root.querySelector('[name="appearance.theme"]') as HTMLSelectElement).value='dark';
 (root.querySelector('[name=model]') as HTMLInputElement).value='user-model-draft';
 root.querySelector('form')!.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));
 await expect.poll(()=>root.querySelector('[data-action=settings]')?.textContent).toBe('Settings');
 expect(root.dataset.theme).toBe('dark');expect(document.documentElement.lang).toBe('en');
 expect(root.textContent).not.toMatch(/[\p{Script=Han}]/u);
 expect(root.querySelector('.topic-filter')?.textContent).toContain('Topic');
 expect(snapshot.values.model).toBe('user-model-draft');expect(snapshot.values.summaryLanguage).toBe('zh');
 dispose();setUiLanguage('zh');dispose=mountWorkbench(root,{fetch:fetcher});
 await expect.poll(()=>root.querySelector('[data-action=settings]')?.textContent).toBe('Settings');
 expect(root.dataset.theme).toBe('dark');
});

it('follows system theme changes and removes the listener when disposed',async()=>{
 let listener:()=>void=()=>{};const remove=vi.fn();
 const media={matches:true,addEventListener:vi.fn((_event:string,fn:()=>void)=>{listener=fn;}),removeEventListener:remove};
 const original=window.matchMedia.bind(window);
 vi.spyOn(window,'matchMedia').mockImplementation(query=>query.includes('prefers-color-scheme')?media as unknown as MediaQueryList:original(query));
 const fetcher=async(url:RequestInfo|URL)=>new Response(JSON.stringify(String(url)==='api/status'?{llm:{ready:true},topics:[],recentRuns:[]}:String(url).startsWith('api/calendar')?{month:'2026-10',cells:[]}:String(url).startsWith('api/papers')?{papers:[],total:0,nextOffset:null,topics:[],counts:{},day:null}:String(url)==='api/runs/current'?{run:null}:{sidebarWidth:null,sidebarCollapsed:false}),{status:200});
 const root=document.createElement('div');document.body.append(root);dispose=mountWorkbench(root,{fetch:fetcher,appearance:{theme:'system',language:'en'}});
 expect(root.dataset.theme).toBe('dark');media.matches=false;listener();expect(root.dataset.theme).toBe('light');
 dispose();expect(remove).toHaveBeenCalledWith('change',listener);
});
