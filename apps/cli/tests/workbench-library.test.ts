import { afterEach, expect, it, vi } from "vitest";
import { mkdtemp, mkdir, realpath, rm, stat, writeFile, rename, symlink, readFile, truncate } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { stringify } from "smol-toml";
import { authorizeLibraryConnection, createLibraryConnection, PersonalLibraryDirectionProposalStore, createPersonalLibraryCatalogInputManifest, createPersonalLibraryCatalogInputManifestFingerprint, createPersonalLibraryRepresentativeSetFingerprint, createPersonalLibraryGenerationContractFingerprint, type EmbeddingModel, type HttpClient } from "@arxiv-daily/core";
import { loadCliConfig } from "../src/config";
import { createCliLibraryContext, runCliLibrary } from "../src/library-cmd";
import { startWorkbench } from "../src/workbench/server";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { WorkbenchLibrary } from "../src/workbench/library";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, {recursive:true,force:true}))); });
async function fixture(remote = false, connected = true, authorized = false) {
  const root = await mkdtemp(join(tmpdir(), "workbench-library-")); roots.push(root);
  const source = join(root,"source"), vault = join(root,"vault"), configPath = join(root,"config.toml");
  await mkdir(source); await mkdir(vault);
  const info = await stat(source);
  let connection = createLibraryConnection(await realpath(source), `${info.dev}:${info.ino}`);
  if (authorized) connection = authorizeLibraryConnection(connection, {llmBaseUrl:"https://fixture.invalid/v1"});
  for(let n=1;n<=3;n++) await writeFile(join(source,`2610.0000${n}.pdf`),`%PDF-1.4\nPaper ${n}\n%%EOF`);
  await writeFile(configPath,stringify({schema_version:1,vault_root:vault,cache_dir:join(root,"cache"),
    llm:{provider:"openai",base_url:"https://fixture.invalid/v1",api_key:"secret",model:"test",thinking_mode:false},
    arxiv:{categories:["cs.AI"],topics:[]},embedding:{mode:remote?"remote":"local",base_url:"https://fixture.invalid/v1",api_key:"secret",model:"test",dimension:2},
    advanced:{log_level:"error",request_delay_ms:0},...(connected?{library:connection}:{})}));
  const config = await loadCliConfig({configPath});
  const embedding: EmbeddingModel = {modelId:"fixture",dimension:2,prefixPolicy:"none",embed:vi.fn(async texts=>texts.map(()=>new Float32Array([1,0])))};
  const http: HttpClient = {request:vi.fn(async req=>{
    if (new URL(req.url).pathname === "/v1/chat/completions") {
      const body = JSON.parse(String(req.body));
      const user = body.messages.find((message:{role:string})=>message.role==="user").content as string;
      const raw = /<paper_data>\n([\s\S]*)\n<\/paper_data>/.exec(user)![1]!;
      const parsed = user.startsWith("Library sample") ? {} : JSON.parse(raw);
      const answer = parsed.groups ? {topics:parsed.groups.map((group:{id:string,papers:{paperKey:string}[]})=>({suggestedName:"Generated topic",directions:[{text:"Generated agent research",discoveryCues:["evaluation"],groupIds:[group.id],representativePaperKeys:group.papers.slice(0,3).map(p=>p.paperKey)}]}))} : {papers:[{id:"sample-1",category:"preview",directions:["preview#1"],relevanceScore:90}]};
      return {status:200,headers:{},bodyText:`data: ${JSON.stringify({choices:[{delta:{content:JSON.stringify(answer)}}]})}\n\ndata: [DONE]\n\n`};
    }
    const ids = new URL(req.url).searchParams.get("id_list")!.split(",");
    return {status:200,headers:{},bodyText:`<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">${ids.map(id=>`<entry><id>http://arxiv.org/abs/${id}v1</id><title>Paper ${id}</title><author><name>Researcher</name></author><summary>${id.endsWith("3")?"vision":"agents"} research evidence</summary><published>2026-10-01T00:00:00Z</published><updated>2026-10-01T00:00:00Z</updated><arxiv:primary_category term="cs.AI"/><category term="cs.AI"/></entry>`).join("")}</feed>`};
  })};
  const options = {http,createEmbedding:vi.fn(async()=>embedding),createParser:vi.fn(async()=>({parser:{capabilities:["page-text"] as const,provenance:{id:"fixture",version:"1"},parse:async(bytes:ArrayBuffer)=>({mediaType:"application/pdf",blocks:[{kind:"page" as const,text:`${new TextDecoder().decode(bytes).includes("Paper 3")?"Vision":"Agents"} research methods\n\nAbstract\n${"Scientific research evidence and experimental results. ".repeat(15)}\n\n1 Introduction\nAdditional context`,locator:{page:1,block:0}}]})}}))};
  const invoke = async (command:string)=>{
    const errors:string[]=[];
    const code=await runCliLibrary(config,[command],{stdout:{write:()=>{}},stderr:{write:text=>errors.push(text)}},options);
    expect(code,errors.join("")).toBe(0);
  };
  return {root,source,configPath,config,options,embedding,invoke,service:new WorkbenchLibrary(config,options)};
}

it("returns an empty disconnected catalog without initializing any model",async()=>{
  const f=await fixture(false,false);
  expect(await f.service.catalog(new URLSearchParams())).toEqual({papers:[],total:0,offset:0,nextOffset:null,summary:null,connected:false});
  expect(f.options.createEmbedding).not.toHaveBeenCalled();
});
it("paginates and filters the actual catalog and searches the shared lexical index without loading embeddings",async()=>{
  const f=await fixture(); await f.invoke("scan"); await f.invoke("index");
  f.options.createEmbedding.mockClear(); vi.mocked(f.embedding.embed).mockClear();
  const page=await f.service.catalog(new URLSearchParams({limit:"2"}));
  expect(page).toMatchObject({total:3,offset:0,nextOffset:2,connected:true,summary:{papers:3}});
  expect(page.papers).toHaveLength(2); expect(page.papers.every(p=>p.pdfAvailable)).toBe(true);
  expect((await f.service.catalog(new URLSearchParams({offset:"2",limit:"2"}))).nextOffset).toBeNull();
  expect((await f.service.catalog(new URLSearchParams({q:"vision"}))).papers).toHaveLength(1);
  const results=await f.service.search({query:"agents",mode:"lexical",limit:1});
  expect(results.papers).toHaveLength(1); expect(results.papers[0]).toMatchObject({abstract:"agents research evidence",pdfAvailable:true});
  expect(f.options.createEmbedding).not.toHaveBeenCalled(); expect(f.embedding.embed).not.toHaveBeenCalled();
  expect((await f.service.search({query:"agents",mode:"hybrid"})).papers.length).toBeGreaterThan(0);
  expect(f.options.createEmbedding).toHaveBeenCalledOnce();
});
it("serves only catalog PDF paper keys and rejects replacement symlinks",async()=>{
  const f=await fixture(); await f.invoke("scan");
  expect(new TextDecoder().decode(await f.service.pdf("arxiv:2610.00001"))).toContain("%PDF");
  await expect(f.service.pdf("../source/2610.00001.pdf")).rejects.toThrow();
  await expect(f.service.pdf("2610.00001.pdf")).rejects.toThrow();
  await rm(join(f.source,"2610.00001.pdf")); await symlink(f.configPath,join(f.source,"2610.00001.pdf"));
  await expect(f.service.pdf("arxiv:2610.00001")).rejects.toThrow();
  expect((await f.service.catalog(new URLSearchParams())).papers.find(p=>p.paperKey==="arxiv:2610.00001")?.pdfAvailable).toBe(false);
});
it("rejects stale settings and changed library roots",async()=>{
  const f=await fixture(); await f.invoke("scan");
  await writeFile(f.configPath,(await readFile(f.configPath,"utf8"))+"\n#changed\n");
  await expect(f.service.catalog(new URLSearchParams())).rejects.toThrow(/changed|reload/i);
  const fresh = new WorkbenchLibrary(await loadCliConfig({configPath:f.configPath}),f.options);
  await rename(f.source,`${f.source}-old`); await mkdir(f.source);
  await expect(fresh.pdf("arxiv:2610.00001")).rejects.toThrow(/identity|connect/i);
});
it("rejects unauthorized remote retrieval before creating a client, but permits lexical retrieval",async()=>{
  const f=await fixture(true); await f.invoke("scan");
  await expect(f.service.search({query:"agents",mode:"hybrid"})).rejects.toThrow(/authoriz/i);
  expect(f.options.createEmbedding).not.toHaveBeenCalled();
  expect(await f.service.search({query:"agents",mode:"lexical"})).toEqual({papers:[],total:0});
  expect(f.options.createEmbedding).not.toHaveBeenCalled();
});
it("validates bounds and propagates cancellation",async()=>{
  const f=await fixture();
  await expect(f.service.catalog(new URLSearchParams({offset:"-1"}))).rejects.toThrow();
  await expect(f.service.search({query:"",mode:"lexical"})).rejects.toThrow();
  const controller=new AbortController(); controller.abort(new Error("cancelled"));
  await expect(f.service.search({query:"agents"},controller.signal)).rejects.toThrow(/cancelled/);
});
it("includes non-arXiv indexed PDFs in browsing, retrieval and bounded paper-key reads",async()=>{
  const f=await fixture();
  await writeFile(join(f.source,"local-research.pdf"),"%PDF-1.4\nLocal paper\n%%EOF");
  await f.invoke("scan"); await f.invoke("index");
  const page=await f.service.catalog(new URLSearchParams());
  expect(page.total).toBe(4);
  const local=page.papers.find(p=>p.paperKey.startsWith("file:"));
  expect(local).toMatchObject({source:"file",title:"Agents research methods",pdfAvailable:true});
  expect((await f.service.search({query:"agents"})).papers.some(p=>p.paperKey===local!.paperKey)).toBe(true);
  expect(new TextDecoder().decode(await f.service.pdf(local!.paperKey))).toContain("Local paper");
});
it("uses HTTP-safe validation errors for untrusted search bodies",async()=>{
  const f=await fixture();
  await expect(f.service.search(null)).rejects.toMatchObject({status:400});
  await expect(f.service.search({query:"agents",mode:"invalid"})).rejects.toMatchObject({status:400});
});

it("serves the actual HTTP catalog, lexical results and PDF with no runtime model",async()=>{
  const f=await fixture(); await f.invoke("scan"); await f.invoke("index");
  const app=await startWorkbench({config:f.config});
  try {
    const list=await fetch(new URL("api/library?limit=2",app.url));
    expect(list.status).toBe(200); expect(await list.json()).toMatchObject({total:3,nextOffset:2});
    const result=await fetch(new URL("api/library/search",app.url),{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({query:"agents",mode:"lexical"})});
    expect(result.status).toBe(200); expect((await result.json()).papers).toHaveLength(2);
    const pdf=await fetch(new URL("api/library/pdf?key=arxiv%3A2610.00001",app.url));
    expect(pdf.status).toBe(200); expect(pdf.headers.get("content-type")).toBe("application/pdf");
    expect(await pdf.text()).toContain("%PDF-1.4");
  } finally { await app.close(); }
});
it("bounds PDF sizes and cancels while waiting for the shared workflow lease",async()=>{
  const f=await fixture(); await f.invoke("scan");
  await truncate(join(f.source,"2610.00001.pdf"),26*1024*1024);
  await expect(f.service.pdf("arxiv:2610.00001")).rejects.toMatchObject({status:413});
  const storage=new NodeStorageAdapter(f.config.vaultRoot);
  const lease=await storage.acquireLock("personal-library-workflow",{wait:true});
  const controller=new AbortController();
  try {
    const pending=f.service.search({query:"agents"},controller.signal);
    controller.abort(new Error("cancelled during lease"));
    await expect(pending).rejects.toThrow(/cancelled/);
  } finally { await lease?.release(); }
});
it("rejects coercible nonnumeric limits and reports disconnected stale config as a conflict",async()=>{
  const f=await fixture(false,false);
  await expect(f.service.search({query:"agents",limit:true})).rejects.toMatchObject({status:400});
  await writeFile(f.configPath,(await readFile(f.configPath,"utf8"))+"\n#changed\n");
  await expect(f.service.catalog(new URLSearchParams())).rejects.toMatchObject({status:409});
});

async function reviewFixture(authorized = false) {
  const f=await fixture(false,true,authorized); await f.invoke("scan"); await f.invoke("index");
  const context=await createCliLibraryContext(f.config,f.options);
  const snapshot=await context.workflow.review(), catalog=snapshot.catalog;
  const input=createPersonalLibraryCatalogInputManifest(Object.values(catalog.papers));
  const store=new PersonalLibraryDirectionProposalStore(context.storage,f.config.settings.output,catalog.scopeFingerprint,catalog.identificationFingerprint);
  const proposal=await store.replace({schemaVersion:6,revision:0,proposalId:"proposal.1",scopeFingerprint:catalog.scopeFingerprint,identificationFingerprint:catalog.identificationFingerprint,
    catalogInputPapers:input,catalogInputFingerprint:createPersonalLibraryCatalogInputManifestFingerprint({scopeFingerprint:catalog.scopeFingerprint,identificationFingerprint:catalog.identificationFingerprint,catalogInputPapers:input}),
    generationContractFingerprint:createPersonalLibraryGenerationContractFingerprint("fixture"),generatedAt:"2026-10-01T00:00:00.000Z",
    topics:[{id:"topic.1",suggestedName:"Agents",directions:input.slice(0,2).map((representative,index)=>({id:`candidate.${index+1}`,text:`Research methods ${index+1}`,discoveryCues:["evaluation"],representatives:[representative],representativeSetFingerprint:createPersonalLibraryRepresentativeSetFingerprint([representative]),lineage:{candidateIds:[`candidate.${index+1}`]}}))}]},null);
  return {...f,store,proposal,versions:{configRevision:f.config.configRevision,expectedProposalRevision:proposal.revision}};
}
it("reviews real proposals and preserves partial acceptance receipts across retries",async()=>{
  const f=await reviewFixture();
  const snapshot=await f.service.review();
  expect(snapshot).toMatchObject({connected:true,configRevision:f.config.configRevision,proposal:{proposalId:"proposal.1"}});
  expect(snapshot.indexedPapers).toHaveLength(3);
  const accepted=await f.service.action({...f.versions,operation:"accept-topics",topicIds:["topic.1"],candidateIds:["candidate.1"]});
  expect(accepted.topics[0]!.directions).toHaveLength(1); expect(accepted.acceptances[0]!.processedCandidateIds).toEqual(["candidate.1"]);
  expect(accepted.configRevision).not.toBe(snapshot.configRevision);
  const retry=await f.service.action({configRevision:accepted.configRevision,expectedProposalRevision:accepted.proposal!.revision,operation:"accept-topics",topicIds:["topic.1"],candidateIds:["candidate.1"]});
  expect(retry.topics).toEqual(accepted.topics); expect(retry.acceptances).toEqual(accepted.acceptances);
  await expect(f.service.action({...f.versions,operation:"accept-topics",topicIds:["topic.1"]})).rejects.toMatchObject({status:409});
});
it("edits, renames, moves and removes candidates with strict revision checks",async()=>{
  const f=await reviewFixture();
  const action=async(operation:string,fields:Record<string,unknown>)=>{
    const current=await f.service.review();
    return f.service.action({operation,...fields,configRevision:current.configRevision,expectedProposalRevision:current.proposal!.revision});
  };
  const edited=await action("update-candidate",{candidateId:"candidate.1",patch:{text:"Edited direction"},representativePaperKeys:["arxiv:2610.00001"]});
  expect(edited.proposal!.topics[0]!.directions[0]!.text).toBe("Edited direction");
  await expect(f.service.action({...f.versions,operation:"rename-topic",topicId:"topic.1",suggestedName:"Stale"})).rejects.toMatchObject({status:409});
  expect((await action("rename-topic",{topicId:"topic.1",suggestedName:"Reviewed agents"})).proposal!.topics[0]!.suggestedName).toBe("Reviewed agents");
  expect((await action("move-direction",{candidateId:"candidate.1",targetTopicId:null,suggestedName:"New field"})).proposal!.topics).toHaveLength(2);
  expect((await action("remove-candidate",{candidateId:"candidate.2"})).proposal!.topics.flatMap(topic=>topic.directions)).toHaveLength(1);
});
it("rejects invalid review bodies and unauthorized model work without changing stores",async()=>{
  const f=await reviewFixture(); const before=await f.store.load();
  await expect(f.service.action({...f.versions,operation:"remove-candidate",candidateId:"candidate.1",extra:true})).rejects.toMatchObject({status:400});
  await expect(f.service.action({...f.versions,operation:"update-candidate",candidateId:"candidate.1",patch:{hidden:true}})).rejects.toMatchObject({status:400});
  await expect(f.service.propose(f.versions)).rejects.toMatchObject({status:409});
  await expect(f.service.preview({...f.versions,candidateId:"candidate.1"})).rejects.toMatchObject({status:409});
  expect(await f.store.load()).toEqual(before);
});
it("previews with authorization without changing proposal or settings and generates only on explicit request",async()=>{
  const f=await reviewFixture(true); const before=await f.store.load(), configBefore=await readFile(f.configPath,"utf8");
  const preview=await f.service.preview({...f.versions,candidateId:"candidate.1"});
  expect(preview.papers[0]).toMatchObject({matched:true,paperKey:"arxiv:2610.00001"});
  expect(await f.store.load()).toEqual(before); expect(await readFile(f.configPath,"utf8")).toBe(configBefore);
  const controller=new AbortController(); controller.abort(new Error("cancelled"));
  await expect(f.service.propose(f.versions,controller.signal)).rejects.toThrow(/cancelled/);
  expect(await f.store.load()).toEqual(before);
  const generated=await f.service.propose(f.versions);
  expect(generated.proposal!.revision).toBeGreaterThan(before!.revision);
});
it("runs proposal and preview HTTP jobs then persists accepted directions and rejects stale pages",async()=>{
  const f=await reviewFixture(true);
  const app=await startWorkbench({config:f.config,libraryOptions:f.options});
  const get=(route:string)=>fetch(new URL(route,app.url));
  const post=(route:string,body:unknown)=>fetch(new URL(route,app.url),{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const done=async()=>{
    let current:{status:string;output:string}|undefined;
    await vi.waitFor(async()=>{current=(await (await get("api/runs/current")).json()).run; expect(current?.status).not.toBe("running");},{timeout:4000});
    expect(current?.status,current?.output).toBe("completed");
  };
  try {
    const response=await post("api/library/propose",f.versions); expect(response.status).toBe(202); await done();
    const snapshot=await (await get("api/library/review")).json();
    expect(snapshot.proposal.revision).toBeGreaterThan(f.proposal.revision);
    const topic=snapshot.proposal.topics[0],candidate=topic.directions[0];
    const versions={configRevision:snapshot.configRevision,expectedProposalRevision:snapshot.proposal.revision};
    const settingsBefore=await readFile(f.configPath,"utf8");
    const preview=await post("api/library/preview",{...versions,candidateId:candidate.id}); expect(preview.status).toBe(202);
    const previewRun=(await preview.json()).run; await done();
    const result=await get(`api/library/preview?runId=${previewRun.id}`); expect(result.status).toBe(200);
    expect((await result.json()).preview.papers.length).toBeGreaterThan(0);
    expect(await readFile(f.configPath,"utf8")).toBe(settingsBefore);
    const accepted=await post("api/library/review",{...versions,operation:"accept-topics",topicIds:[topic.id],candidateIds:[candidate.id]}); expect(accepted.status).toBe(200);
    const saved=await accepted.json();
    expect((await loadCliConfig({configPath:f.configPath})).settings.arxiv.topics).toEqual(saved.topics);
    expect((await (await get("api/library/review")).json()).topics).toEqual(saved.topics);
    expect((await (await get("api/status")).json()).topics).toEqual(saved.topics.map(({name,tag,description,detail}:{name:string;tag:string;description:string;detail:boolean})=>({name,tag,description,detail})));
    await writeFile(f.configPath,(await readFile(f.configPath,"utf8"))+"\n# external change\n");
    expect((await post("api/library/review",{operation:"remove-candidate",candidateId:candidate.id,configRevision:saved.configRevision,expectedProposalRevision:saved.proposal.revision})).status).toBe(409);
  } finally { await app.close(); }
},15000);
it("rejects competing HTTP mutations and cancels in-flight model preview without saving",async()=>{
  const f=await reviewFixture(true), before=await f.store.load();
  let entered!:()=>void; const modelStarted=new Promise<void>(resolve=>{entered=resolve;});
  const request=f.options.http.request;
  f.options.http.request=async req=>{
    if (!req.url.endsWith("/chat/completions")) return request(req);
    entered();
    return new Promise((_resolve,reject)=>{
      if (req.signal?.aborted) reject(req.signal.reason);
      else req.signal?.addEventListener("abort",()=>reject(req.signal!.reason),{once:true});
    });
  };
  const app=await startWorkbench({config:f.config,libraryOptions:f.options});
  const post=(route:string,body:unknown)=>fetch(new URL(route,app.url),{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  try {
    const started=await post("api/library/preview",{...f.versions,candidateId:"candidate.1"});
    expect(started.status).toBe(202); const run=(await started.json()).run;
    await modelStarted;
    expect((await post("api/library/review",{...f.versions,operation:"remove-candidate",candidateId:"candidate.1"})).status).toBe(409);
    expect((await post("api/library/propose",f.versions)).status).toBe(409);
    expect((await post("api/runs/cancel",{id:run.id})).status).toBe(202);
    await vi.waitFor(async()=>{const body=await (await fetch(new URL("api/runs/current",app.url))).json();expect(body.run.status).toBe("cancelled");});
    expect(await f.store.load()).toEqual(before);
    expect((await loadCliConfig({configPath:f.configPath})).settings.arxiv.topics).toEqual([]);
  } finally { await app.close(); }
});
it("keeps catalog, review and PDF reading available while a library model job owns the mutation lease",async()=>{
  const f=await reviewFixture();
  const storage=new NodeStorageAdapter(f.config.vaultRoot);
  const lease=await storage.acquireLock("personal-library-workflow",{wait:true});
  const httpCalls=vi.mocked(f.options.http.request).mock.calls.length;
  const reader=new WorkbenchLibrary(f.config,{...f.options,signal:AbortSignal.timeout(700)});
  try {
    const [catalog,review,pdf]=await Promise.all([
      reader.catalog(new URLSearchParams()),reader.review(),reader.pdf("arxiv:2610.00001"),
    ]);
    expect(catalog.total).toBe(3); expect(review.proposal?.proposalId).toBe("proposal.1");
    expect(new TextDecoder().decode(pdf)).toContain("%PDF-1.4");
    expect(vi.mocked(f.options.http.request).mock.calls).toHaveLength(httpCalls);
    // Reading must not release or bypass the writer's exclusive lease.
    const controller=new AbortController();
    const mutation=f.service.action({...f.versions,operation:"remove-candidate",candidateId:"candidate.1"},controller.signal);
    const timer=setTimeout(()=>controller.abort(new Error("writer remains exclusive")),50);
    try { await expect(mutation).rejects.toThrow(/writer remains exclusive|cancelled/); }
    finally { clearTimeout(timer); }
    expect((await f.store.load())?.topics[0]?.directions).toHaveLength(2);
  } finally { await lease?.release(); }
});
