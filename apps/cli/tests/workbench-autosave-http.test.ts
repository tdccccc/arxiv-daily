// @vitest-environment happy-dom
import { expect, it, vi } from 'vitest';
import { request as httpRequest } from 'node:http';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startWorkbench } from '../src/workbench/server';
import { readWorkbenchSettings, saveWorkbenchSettings } from '../src/workbench/settings';
import { loadCliConfig } from '../src/config';
import { bindSettings, settingsForm } from '../src/workbench/web/settings';

it('persists automatic edits through HTTP into TOML and preserves disabled scheduling on reopen', async () => {
 const root=await mkdtemp(join(tmpdir(),'arxiv-autosave-http-'));
 const configPath=join(root,'config.toml');
 const empty=await readWorkbenchSettings(configPath);
 const config=await saveWorkbenchSettings(configPath,{revision:null,values:{...empty.values,vaultRoot:join(root,'notes'),apiKey:'initial-fixture-key',model:'fixture-model',baseUrl:'https://fixture.invalid/v1',topics:[{id:'t',name:'Research',tag:'research',detail:false,description:'Research directions',directions:[{id:'d',text:'Research directions',origin:'manual'}]}],schedule:{...empty.values.schedule,enabled:true}}});
 const app=await startWorkbench({config});
 const request=<T>(route:string,body?:unknown)=>new Promise<T>((resolve,reject)=>{
  const payload=body===undefined?undefined:JSON.stringify(body);
  const req=httpRequest(new URL(route,app.url),{method:payload===undefined?'GET':'POST',headers:payload===undefined?{}:{'Content-Type':'application/json'}},res=>{
   let text='';res.setEncoding('utf8');res.on('data',chunk=>{text+=chunk;});res.on('end',()=>{try{const value=JSON.parse(text);if(res.statusCode!==200)reject(new Error(value.error));else resolve(value as T);}catch(error){reject(error);}});
  });req.on('error',reject);req.end(payload);
 });
 const snapshot=await readWorkbenchSettings(configPath);
 document.body.innerHTML=settingsForm(snapshot);
 const form=document.querySelector('form')!;
 const binding=bindSettings(form,snapshot,request,async()=>{});
 try {
  const enabled=form.elements.namedItem('schedule.enabled') as HTMLInputElement;
  expect(enabled.checked).toBe(true);
  enabled.checked=false;enabled.dispatchEvent(new Event('change',{bubbles:true}));
  await vi.waitFor(async()=>expect((await loadCliConfig({configPath})).workbenchSchedule?.enabled).toBe(false));
  const model=form.elements.namedItem('model') as HTMLInputElement;
  model.value='user-chosen-model';model.dispatchEvent(new Event('change',{bubbles:true}));
  await vi.waitFor(async()=>expect((await loadCliConfig({configPath})).settings.llm.model).toBe('user-chosen-model'));
  const apiKey=form.elements.namedItem('apiKey') as HTMLInputElement;
  apiKey.value='fixture-only-secret';apiKey.dispatchEvent(new Event('change',{bubbles:true}));
  await vi.waitFor(async()=>expect((await loadCliConfig({configPath})).settings.llm.apiKey).toBe('fixture-only-secret'));
  expect((await loadCliConfig({configPath})).workbenchSchedule?.enabled).toBe(false);
  const reopened=await readWorkbenchSettings(configPath);
  document.body.innerHTML=settingsForm(reopened);
  expect((document.querySelector('[name="schedule.enabled"]') as HTMLInputElement).checked).toBe(false);
  expect((document.querySelector('[name="apiKey"]') as HTMLInputElement).value).toBe('');
  expect(reopened.values.apiKeyConfigured).toBe(true);
 } finally { binding?.dispose();document.body.innerHTML='';await app.close();await rm(root,{recursive:true,force:true}); }
});
