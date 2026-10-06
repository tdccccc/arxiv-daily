// @vitest-environment happy-dom
import { expect, it, vi } from 'vitest';
import { DEFAULT_SETTINGS, normalizeSettingsEdits } from '@arxiv-daily/core';
import { settingsForm, bindSettings } from '../src/workbench/web/settings';
import { readWorkbenchSettings } from '../src/workbench/settings';
import { patchWorkbenchBusinessSettings } from '../src/workbench/settings-adapter';
it('derives the description from all authoritative directions and rejects invalid daily limits', () => {
 const settings=structuredClone(DEFAULT_SETTINGS);
 settings.arxiv.topics=[{id:'t',name:'Topic',tag:'topic',description:'stale',detail:true,directions:[{id:'a',text:'First',origin:'library'},{id:'b',text:'Second',origin:'manual'}]}];
 const next=normalizeSettingsEdits(settings,['arxiv.topics']);
 expect(next.arxiv.topics[0]?.description).toBe('First');
 expect(next.arxiv.topics[0]?.directions).toEqual(settings.arxiv.topics[0]?.directions);
 settings.output.maxDailyPapers=0;
 expect(()=>normalizeSettingsEdits(settings,['output.maxDailyPapers'])).toThrow();
});
it('writes daily limit to the existing TOML output table',()=>{
 const document:Record<string,unknown>={output:{daily_dir:'daily'}};
 patchWorkbenchBusinessSettings(document,{maxDailyPapers:35},null);
 expect(document.output).toEqual({daily_dir:'daily',max_daily_papers:35});
});
it('keeps direction identities and origin when editing settings, and renders the daily limit', async()=>{
 const snapshot=await readWorkbenchSettings('/tmp/arxiv-daily-missing-roundtrip/config.toml');
 snapshot.values.vaultRoot='/notes';
 snapshot.values.topics=[{id:'t',name:'Topic',tag:'topic',description:'First',detail:true,directions:[{id:'a',text:'First',origin:'library'},{id:'b',text:'Second',origin:'manual'}]}];
 document.body.innerHTML=settingsForm(snapshot);
 const form=document.querySelector('form')!;
 const request=vi.fn(async()=>({revision:'next',values:snapshot.values}));
 bindSettings(form,snapshot,request as never,async()=>{});
 expect(form.querySelector('[name="topicTag"]')).toBeNull();
 const directions=form.querySelectorAll<HTMLTextAreaElement>('[name="topicDirection"]');
 expect(directions).toHaveLength(2);
 directions[1]!.value='Second edited';
 (form.querySelector('[name="maxDailyPapers"]') as HTMLInputElement).value='35';
 form.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));
 await vi.waitFor(()=>expect(request).toHaveBeenCalled());
 const payload=request.mock.calls[0] as unknown as [string,{values:{topics:typeof snapshot.values.topics;maxDailyPapers:number}}];
 expect(payload[1].values.topics[0]).toMatchObject({description:'First',directions:[{id:'a',text:'First',origin:'library'},{id:'b',text:'Second edited',origin:'manual'}]});
 expect(payload[1].values.maxDailyPapers).toBe(35);
 document.body.innerHTML='';
});
it('roundtrips multiple accepted directions and the cap through disk without replacing them for old clients', async()=>{
 const {mkdtemp,rm}=await import('node:fs/promises');const {tmpdir}=await import('node:os');const {join}=await import('node:path');
 const {saveWorkbenchSettings}=await import('../src/workbench/settings');
 const root=await mkdtemp(join(tmpdir(),'direction-settings-'));
 try{
  const file=join(root,'config.toml');const initial=await readWorkbenchSettings(file);
  const topic={id:'t',name:'Topic',tag:'custom-tag',description:'stale',detail:true,directions:[{id:'a',text:'First',origin:'library'},{id:'b',text:'Second',origin:'manual'}]};
  const saved=await saveWorkbenchSettings(file,{revision:null,values:{...initial.values,vaultRoot:join(root,'vault'),topics:[topic],maxDailyPapers:35}});
  const reread=await readWorkbenchSettings(file);
  expect(reread.values.maxDailyPapers).toBe(35);expect(reread.values.topics[0]).toEqual({...topic,description:'First'});
  const {directions,...legacy}=topic;
  await saveWorkbenchSettings(file,{revision:saved.configRevision,values:{topics:[{...legacy,name:'Renamed'}]}});
  const preserved=await readWorkbenchSettings(file);
  expect(preserved.values.topics[0]?.directions).toEqual(directions);
  const emptied=await saveWorkbenchSettings(file,{revision:preserved.revision,values:{topics:[{...topic,directions:[]}]}});
  expect(emptied.settings.arxiv.topics[0]).toMatchObject({directions:[],description:''});
 }finally{await rm(root,{recursive:true,force:true});}
});
it('translates direction and daily limit controls while preserving research text',async()=>{
 const {setUiLanguage}=await import('../src/workbench/web/i18n');
 const snapshot=await readWorkbenchSettings('/tmp/arxiv-daily-missing-roundtrip/config.toml');
 snapshot.values.topics=[{id:'t',name:'Topic',tag:'topic',description:'User research',detail:true,directions:[{id:'a',text:'User research',origin:'library'}]}];
 setUiLanguage('zh');
 try{
  document.body.innerHTML=settingsForm(snapshot);
  expect(document.querySelector('[name="maxDailyPapers"]')?.getAttribute('aria-label')).toBe('每日日报论文上限');
  expect(document.querySelector('[name="topicDirection"]')?.getAttribute('aria-label')).toBe('研究方向');
  expect(document.querySelector('[data-settings="add-direction"]')?.textContent).toBe('添加方向');
  expect((document.querySelector('[name="topicDirection"]') as HTMLTextAreaElement).value).toBe('User research');
 }finally{setUiLanguage('en');document.body.innerHTML='';}
});
