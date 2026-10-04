import { afterEach, expect, it, vi } from 'vitest';
import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SettingsChangeService } from '../../../plugin/src/settings/change-service';
import { readWorkbenchSettings, saveWorkbenchSettings } from '../src/workbench/settings';
const roots:string[]=[];afterEach(async()=>{for(const root of roots.splice(0))await rm(root,{recursive:true,force:true});});
async function fixture(){const root=await mkdtemp(join(tmpdir(),'shared-settings-hosts-'));roots.push(root);const file=join(root,'config.toml');const initial=await readWorkbenchSettings(file);const config=await saveWorkbenchSettings(file,{revision:null,values:{...initial.values,vaultRoot:join(root,'vault')}});const settings=structuredClone(config.settings);const persist=vi.fn(async()=>{});const plugin=new SettingsChangeService({settings,persistSettings:persist});return {file,config,settings,persist,plugin};}
it('applies identical model, topic and email draft edits through both real host adapters',async()=>{
 const {file,config,settings,plugin}=await fixture();
 const topics=[{id:'focus',name:'  Research  ',tag:'  focus ',description:'stale',directions:[{id:'first',text:'First line',origin:'manual'}],detail:true}];
 await plugin.change({changes:[{key:'llm.model',value:'  same-model  '},{key:'arxiv.topics',value:topics},{key:'email.to',value:' reader@ '},{key:'email.fromName',value:'  Lab sender  '}]});
 const next=await saveWorkbenchSettings(file,{revision:config.configRevision,values:{model:'  same-model  ',topics,email:{to:' reader@ ',fromName:'  Lab sender  '}}});
 expect(next.settings.llm).toEqual(settings.llm);expect(next.settings.arxiv.topics).toEqual(settings.arxiv.topics);expect(next.settings.email).toEqual(settings.email);
});
it('rejects the same invalid endpoint without persisting either host',async()=>{
 const {file,config,plugin,persist}=await fixture();const before=await readFile(file,'utf8');
 await expect(plugin.changeValue('llm.baseUrl','file:///private')).rejects.toThrow('llm.baseUrl');
 await expect(saveWorkbenchSettings(file,{revision:config.configRevision,values:{baseUrl:'file:///private'}})).rejects.toMatchObject({status:400});
 expect(persist).not.toHaveBeenCalled();expect(await readFile(file,'utf8')).toBe(before);
});
