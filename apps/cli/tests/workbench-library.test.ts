import { afterEach, expect, it, vi } from "vitest";
import { mkdtemp, mkdir, realpath, rm, stat, writeFile, rename, symlink, readFile, truncate } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { stringify } from "smol-toml";
import { createLibraryConnection, type EmbeddingModel, type HttpClient } from "@arxiv-daily/core";
import { loadCliConfig } from "../src/config";
import { runCliLibrary } from "../src/library-cmd";
import { startWorkbench } from "../src/workbench/server";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { WorkbenchLibrary } from "../src/workbench/library";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, {recursive:true,force:true}))); });
async function fixture(remote = false, connected = true) {
  const root = await mkdtemp(join(tmpdir(), "workbench-library-")); roots.push(root);
  const source = join(root,"source"), vault = join(root,"vault"), configPath = join(root,"config.toml");
  await mkdir(source); await mkdir(vault);
  const info = await stat(source);
  const connection = createLibraryConnection(await realpath(source), `${info.dev}:${info.ino}`);
  for(let n=1;n<=3;n++) await writeFile(join(source,`2610.0000${n}.pdf`),`%PDF-1.4\nPaper ${n}\n%%EOF`);
  await writeFile(configPath,stringify({schema_version:1,vault_root:vault,cache_dir:join(root,"cache"),
    llm:{provider:"openai",base_url:"https://fixture.invalid/v1",api_key:"secret",model:"test"},
    arxiv:{categories:["cs.AI"],topics:[]},embedding:{mode:remote?"remote":"local",base_url:"https://fixture.invalid/v1",api_key:"secret",model:"test",dimension:2},
    advanced:{log_level:"error"},...(connected?{library:connection}:{})}));
  const config = await loadCliConfig({configPath});
  const embedding: EmbeddingModel = {modelId:"fixture",dimension:2,prefixPolicy:"none",embed:vi.fn(async texts=>texts.map(()=>new Float32Array([1,0])))};
  const http: HttpClient = {request:vi.fn(async req=>{
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
