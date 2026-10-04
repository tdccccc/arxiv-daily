import { afterEach, expect, it, vi } from "vitest";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { stringify, parse } from "smol-toml";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { loadCliConfig } from "../src/config";
import { createCliTopicSettings } from "../src/library-topic-settings";
const roots: string[] = [];
afterEach(async () => { vi.restoreAllMocks(); await Promise.all(roots.splice(0).map(root => fs.rm(root, {recursive:true,force:true}))); });
async function fixture(extra = {}) {
 const root = await fs.mkdtemp(path.join(os.tmpdir(), 'topic-settings-')); roots.push(root);
 const configPath = path.join(root,'config.toml');
 await fs.writeFile(configPath,stringify({schema_version:1,vault_root:root,llm:{api_key:'fixture'},arxiv:{topics:[{id:'manual',name:'Manual',tag:'manual',directions:[{id:'manual-direction',text:'Manual research',origin:'manual'}],detail:false}]},...extra}));
 return loadCliConfig({configPath});
}
const receipt = {scopeFingerprint:`sha256:${'a'.repeat(64)}`,proposalId:'proposal',topicTargets:{candidate:'manual'},processedCandidateIds:['direction']};
it('persists topics and receipts together, preserves configuration and publishes the revision',async()=>{
 const config=await fixture(); const before=config.configRevision; const port=createCliTopicSettings(config);
 await port.change(async current=>({topics:current.topics.map(topic=>({...topic,directions:[...topic.directions,{id:'accepted',text:'Accepted direction',origin:'library' as const}]})),acceptances:[receipt]}));
 expect(config.configRevision).not.toBe(before); await expect(port.assertCurrent()).resolves.toBeUndefined();
 expect((await port.read()).acceptances).toEqual([receipt]);
 expect((await loadCliConfig({configPath:config.configPath})).settings.arxiv.topics[0]?.directions.map(x=>x.text)).toContain('Accepted direction');
 expect(parse(await fs.readFile(config.configPath,'utf8')).llm).toMatchObject({api_key:'fixture'});
});
it('rejects stale configuration without overwriting external changes',async()=>{
 const config=await fixture(); const port=createCliTopicSettings(config); await fs.appendFile(config.configPath,'\n# changed\n');
 await expect(port.change(async x=>x)).rejects.toThrow(/changed/);
 expect(await fs.readFile(config.configPath,'utf8')).toContain('# changed');
});
it('does not publish settings or receipts when persistence fails',async()=>{
 const config=await fixture(); const before=await fs.readFile(config.configPath,'utf8'); const revision=config.configRevision;
 vi.spyOn(NodeStorageAdapter.prototype,'writeTextAtomic').mockRejectedValueOnce(new Error('disk full'));
 await expect(createCliTopicSettings(config).change(async x=>({...x,acceptances:[receipt]}))).rejects.toThrow('disk full');
 expect(config.configRevision).toBe(revision); expect(await fs.readFile(config.configPath,'utf8')).toBe(before);
});
it('refuses malformed acceptance receipts rather than losing idempotence',async()=>{
 const config=await fixture({library_proposal_acceptances:[{proposalId:'broken'}]});
 await expect(createCliTopicSettings(config).read()).rejects.toThrow(/receipt/i);
});
it('allows its exact atomic write while rejecting other concurrent bytes',async()=>{
 const config=await fixture(); const port=createCliTopicSettings(config);
 const original=NodeStorageAdapter.prototype.writeTextAtomic;
 vi.spyOn(NodeStorageAdapter.prototype,'writeTextAtomic').mockImplementationOnce(async function(this:NodeStorageAdapter,...args:Parameters<typeof original>){
   await original.apply(this,args); await port.assertCurrent();
 });
 await port.change(async current=>({...current,acceptances:[receipt]}));
 await port.assertCurrent();
 await fs.appendFile(config.configPath,'\n# external\n');
 await expect(port.assertCurrent()).rejects.toThrow(/changed/);
});
it('rejects an editor write that occurs while acceptance is computed',async()=>{
 const config=await fixture(); const port=createCliTopicSettings(config);
 await expect(port.change(async current=>{await fs.appendFile(config.configPath,'\n# editor\n');return {...current,acceptances:[receipt]};})).rejects.toThrow(/changed/);
 expect(parse(await fs.readFile(config.configPath,'utf8')).library_proposal_acceptances).toBeUndefined();
});
