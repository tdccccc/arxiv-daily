import { setUiLanguage, getUiLanguage } from "../src/workbench/web/i18n";
// @vitest-environment happy-dom
import { beforeEach, afterEach, expect, it, vi } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readWorkbenchSettings, saveWorkbenchSettings } from "../src/workbench/settings";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import { buildSettingDefinitions, type SettingDefinitionsHost } from "../../../plugin/src/settings/definitions";
import { mountWorkbench } from "../src/workbench/web/app";
const values = { vaultRoot: "/notes", baseUrl: "https://api.example/v1", provider: "openai", model: "test", apiKeyConfigured: true, categories: ["cs.AI"], timezone: "Asia/Shanghai", summaryLanguage: "zh", topics: [{ id: "existing-topic", name: "Models", tag: "models", description: "Inference", detail: true }], dailyDir: "daily", papersDir: "papers" };
const json = (v: unknown, status = 200) => new Response(JSON.stringify(v), { status });
let dispose = () => {};
beforeEach(()=>setUiLanguage("en"));
afterEach(() => { dispose(); document.body.innerHTML = ""; vi.restoreAllMocks(); });
function setup(first = false, fail = false, configPath?: string, override?: (path: string, init?: RequestInit) => Response | Promise<Response> | undefined) {
  history.replaceState({}, "", "/capability/");
  let configured = !first;
  const fetcher = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
    const path = String(url).split("?")[0];
    const custom = override?.(path!, init); if (custom) return custom;
    if (path === "api/status") return json(configured ? { configPath: "/config.toml", vaultRoot: "/notes", categories: ["cs.AI"], output: { summaryLanguage: "zh" }, llm: { ready: true, provider: "openai", model: "test", keyConfigured: true }, topics: [], emailEnabled: false } : { setupRequired: true });
    if (path === "api/settings") {
      if (init?.method === "POST") {
        if (fail) return json({ error: "配置已被其他窗口修改，请重新加载" }, 409);
        if (configPath) {
          try { await saveWorkbenchSettings(configPath, JSON.parse(String(init.body))); }
          catch (error) { return json({ error: String(error) }, 400); }
        }
        configured = true;
      }
      return json({ setupRequired: !configured, revision: configured ? "revision-1" : null, configPath: "/config.toml", values });
    }
    if (path === "api/settings/library") return json({ status: { kind: "disconnected" } });
    if (path === "api/preferences") return json({ sidebarWidth: null, sidebarCollapsed: false, appearance: {theme:"light",language:getUiLanguage()} });
    if (!configured) return json({ error: "请先设置", setupRequired: true }, 409);
    if (path === "api/runs/current") return json({ run: null });
    if (path === "api/calendar") return json({ month: "2026-10", today: "2026-10-03", timezone: "Asia/Shanghai", previousMonth: "2026-09", nextMonth: "2026-11", cells: [] });
    if (path === "api/papers") return json({ papers: [], total: 0, nextOffset: null, topics: [], counts: {}, day: null });
    throw new Error(`Unexpected ${path}`);
  });
  const root = document.createElement("div"); document.body.append(root);
  dispose = mountWorkbench(root, { fetch: fetcher });
  return { root, fetcher };
}
function input(root: HTMLElement, name: string, value: string) { root.querySelector<HTMLInputElement>(`[name="${name}"]`)!.value = value; }
function submit(root: HTMLElement) { root.querySelector(".settings-form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); }
it("opens first-run setup before data requests and refreshes the workspace after saving", async () => {
  const directory = await mkdtemp(join(tmpdir(), "workbench-settings-ui-"));
  const configPath = join(directory, "config.toml");
  const { root, fetcher } = setup(true, false, configPath);
  await vi.waitFor(() => expect(root.querySelector(".settings-form")).toBeTruthy());
  expect(root.querySelector("dialog")?.textContent).toMatch(/首次使用|First/);
  expect(fetcher.mock.calls.some(([u]) => /api\/(calendar|papers|runs)/.test(String(u)))).toBe(false);
  expect(root.querySelector<HTMLInputElement>('[name="apiKey"]')?.type).toBe("password");
  expect(root.querySelector<HTMLInputElement>('[name="apiKey"]')?.value).toBe("");
  expect(root.querySelector("dialog")?.textContent).toContain("independent");
  input(root, "vaultRoot", "/new-notes"); input(root, "apiKey", "new-secret");
  root.querySelector<HTMLButtonElement>('[data-settings="add-topic"]')!.click();
  expect(root.querySelectorAll(".settings-topic")).toHaveLength(2);
  const second = root.querySelectorAll<HTMLElement>(".settings-topic")[1]!;
  input(second, "topicName", "Vision"); input(second, "topicTag", "vision"); input(second, "topicDescription", "Video");
  input(root, "reasoningEffort", "high"); input(root, "detailProfile", "broad");
  input(root, "schedule.runAtLocal", "09:30"); input(root, "schedule.runUntilLocal", "18:00"); input(root, "schedule.tickIntervalMin", "15");
  input(root, "embedding.mode", "remote"); input(root, "embedding.baseUrl", "https://embedding.example/v1"); input(root, "embedding.model", "embed-test"); input(root, "embedding.dimension", "768"); input(root, "embedding.apiKey", "embedding-secret");
  input(root, "email.to", "reader@example.com"); input(root, "email.apiKey", "mail-secret"); input(root, "logLevel", "warn");
  submit(root);
  await vi.waitFor(() => expect(root.querySelector(".paper-workspace"), root.querySelector(".form-error")?.textContent ?? "").toBeTruthy());
  expect(root.querySelector("dialog")).toBeNull();
  const body = JSON.parse(String(fetcher.mock.calls.find(([, i]) => i?.method === "POST")![1]!.body));
  expect(body.revision).toBeNull(); expect(body.values.vaultRoot).toBe("/new-notes"); expect(body.values.apiKey).toBe("new-secret");
  expect(body.values.topics).toHaveLength(2);
  expect(body.values.topics[0].id).toBe("existing-topic");
  expect(body.values.topics[1].id).toEqual(expect.any(String));
  expect(new Set(body.values.topics.map((t: { id: string }) => t.id)).size).toBe(2);
  const persisted=(await readWorkbenchSettings(configPath)).values;
  expect(persisted.topics).toEqual(body.values.topics);
  expect(persisted).toMatchObject({reasoningEffort:"high",detailProfile:"broad",schedule:{runAtLocal:"09:30",tickIntervalMin:15},embedding:{mode:"remote",model:"embed-test",dimension:768,apiKeyConfigured:true},email:{to:"reader@example.com",apiKeyConfigured:true},logLevel:"warn"});
  await rm(directory, { recursive: true, force: true });
  expect(root.querySelector<HTMLElement>(".connection-banner")!.hidden).toBe(true);
});
it("keeps edits and a blank unchanged key on save conflicts", async () => {
  const { root, fetcher } = setup(false, true);
  await vi.waitFor(() => expect(root.querySelector(".paper-workspace")).toBeTruthy());
  root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
  await vi.waitFor(() => expect(root.querySelector(".settings-form")).toBeTruthy());
  input(root, "model", "edited-model"); submit(root);
  await vi.waitFor(() => expect(root.querySelector(".settings-form [role=alert]")?.textContent).toContain("其他窗口"));
  expect(root.querySelector<HTMLInputElement>('[name="model"]')!.value).toBe("edited-model");
  expect(root.querySelector<HTMLButtonElement>('.settings-form [type="submit"]')!.disabled).toBe(false);
  const body = JSON.parse(String(fetcher.mock.calls.find(([, i]) => i?.method === "POST")![1]!.body));
  expect(body.revision).toBe("revision-1"); expect(body.values.apiKey).toBeUndefined();
});
it.each([{remote:false,hosted:false,sidecar:false},{remote:true,hosted:true,sidecar:true}])("matches Obsidian groups, row order and controls for %o", async ({remote,hosted,sidecar}) => {
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.embedding.mode = remote ? 'remote' : 'local'; settings.email.mode = hosted ? 'hosted' : 'self'; settings.pdfParserSidecar.enabled=sidecar;
  const { root } = setup(false,false,undefined,(path)=>path==='api/settings'?json({setupRequired:false,revision:'r1',configPath:'/config.toml',values:{...values,embedding:{...settings.embedding,apiKeyConfigured:true},email:{...settings.email,apiKeyConfigured:true,hostedTokenConfigured:true},pdfParserSidecar:settings.pdfParserSidecar}}):undefined);
  await vi.waitFor(() => expect(root.querySelector(".paper-workspace")).toBeTruthy());
  root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
  await vi.waitFor(() => expect(root.querySelector(".settings-form")).toBeTruthy());
  settings.arxiv.topics = values.topics; settings.arxiv.categories = values.categories;
  const host = new Proxy({ plugin: { settings }, showSetupGuide: false }, { get(target, key) { if (key in target) return target[key as keyof typeof target]; if (String(key).startsWith("render")) return () => {}; return undefined; } }) as unknown as SettingDefinitionsHost;
  const definitions = buildSettingDefinitions(host);
  const expectedNames: string[] = [], expectedGroups: string[] = [];
  for (const definition of definitions) {
    const item = definition as { heading?: string; name?: string; items?: { name?: string; control?: { type: string; key: string; options?: Record<string, string> } }[]; control?: { type: string; key: string; options?: Record<string, string> } };
    if (item.heading) expectedGroups.push(item.heading);
    for (const row of item.items ?? [item]) {
      if (row.name) expectedNames.push(row.name);
      if (row.control?.options) {
        const select = root.querySelector<HTMLSelectElement>(`[data-setting-key="${row.control.key}"]`);
        expect(select, row.name).toBeTruthy();
        expect(Array.from(select!.options).map(o => [o.value, o.textContent])).toEqual(Object.entries(row.control.options));
      }
    }
  }
  expect(Array.from(root.querySelectorAll<HTMLElement>('[data-settings-heading]')).filter(e=>e.dataset.settingsKey!=='Appearance').map(e => e.textContent)).toEqual(expectedGroups);
  expect(Array.from(root.querySelectorAll<HTMLElement>('[data-setting-name]')).filter(e=>!e.closest('[hidden]')&&!['Theme','Interface language'].includes(e.dataset.settingName??'')).map(e => e.dataset.settingName)).toEqual(expectedNames);
  expect(root.querySelector('input[name="model"][role="combobox"]')?.getAttribute('aria-controls')).toBeTruthy();
  expect(root.querySelector('select[name="reasoningEffort"]')).toBeTruthy();
  expect(root.querySelector('select[name="schedule.runAtLocal"]')).toBeTruthy();
  expect(root.querySelector('input[name="schedule.enabled"]')?.getAttribute('type')).toBe('checkbox');
  expect(root.querySelector('input[name="timezoneCustom"]')).toBeTruthy();
  expect(root.querySelector('details.settings-topic')).toBeTruthy();
});

it("discloses library processing and submits the exact fingerprint only after confirmation", async () => {
 const disclosure = { selectedRoot: "/pdfs", eligibleExtensions: [".pdf"], processingDepth: "full-text", endpoint: "https://api.example/v1/chat/completions", authorizationFingerprint: "sha256:consent" };
 const {root,fetcher}=setup(false,false,undefined,(path)=>path==='api/settings/library'?json({status:{kind:'authorization-required',rootLabel:'/pdfs'},disclosure}):path==='api/settings/action'?json({run:{id:'build',status:'running',label:'Building library'}}):undefined);
 await vi.waitFor(()=>expect(root.querySelector('.paper-workspace')).toBeTruthy());
 root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('[data-settings="library-build"]')).toBeTruthy());
 root.querySelector<HTMLButtonElement>('[data-settings="library-build"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('[aria-label="Library processing consent"]')).toBeTruthy());
 expect(fetcher.mock.calls.filter(([url])=>String(url)==='api/settings/action')).toHaveLength(0);
 expect(root.querySelector('[aria-label="Library processing consent"]')!.textContent).toContain('/pdfs');
 root.querySelector<HTMLButtonElement>('[data-settings="confirm-library-build"]')!.click();
 await vi.waitFor(()=>expect(fetcher.mock.calls.filter(([url])=>String(url)==='api/settings/action')).toHaveLength(1));
 const action=fetcher.mock.calls.find(([url])=>String(url)==='api/settings/action')!;
 expect(JSON.parse(String(action[1]!.body))).toMatchObject({revision:'revision-1',action:'library-build',fingerprint:disclosure.authorizationFingerprint});
});

it("loads models without changing the draft and sends email only from its explicit action", async () => {
 const {root,fetcher}=setup(false,false,undefined,(path,init)=> {
  if(path!=='api/settings/action')return;
  const body=JSON.parse(String(init?.body));
  return json(body.action==='models'?{models:['listed-model']}:{run:{id:'mail',status:'running',label:'Sending test'}});
 });
 await vi.waitFor(()=>expect(root.querySelector('.paper-workspace')).toBeTruthy());
 root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('.settings-form')).toBeTruthy());
 input(root,'model','unlisted-draft'); input(root,'apiKey','new-key');
 root.querySelector<HTMLButtonElement>('[data-settings="models"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('[role=option]')?.textContent).toBe('listed-model'));
 expect(root.querySelector<HTMLInputElement>('[name="model"]')!.value).toBe('unlisted-draft');
 expect(root.querySelector<HTMLInputElement>('[name="apiKey"]')!.value).toBe('');
 const actions=()=>fetcher.mock.calls.filter(([url])=>String(url)==='api/settings/action').map(([,init])=>JSON.parse(String(init!.body)));
 expect(actions().map(a=>a.action)).toEqual(['models']);
 expect(fetcher.mock.calls.some(([url,init])=>String(url)==='api/preferences'&&init?.method==='POST')).toBe(false);
 root.querySelector<HTMLButtonElement>('[data-email-self] [data-settings="email-test"]')!.click();
 await vi.waitFor(()=>expect(actions().map(a=>a.action)).toEqual(['models','email-test']));
 expect(actions().every(a=>a.revision==='revision-1')).toBe(true);
});
it("uses the newly selected timezone and mirrors topic tag and detail indicators", async () => {
 const {root,fetcher}=setup(false,false,undefined,(path,init)=>path==='api/settings'&&!init?.method?json({setupRequired:false,revision:'r1',configPath:'/config.toml',values:{...values,timezone:'Pacific/Auckland'}}):undefined);
 await vi.waitFor(()=>expect(root.querySelector('.paper-workspace')).toBeTruthy());
 root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('.settings-form')).toBeTruthy());
 expect(root.querySelector('[name="categoryCustom"]')).toBeNull();
 const zone=root.querySelector<HTMLSelectElement>('[name="timezone"]')!;
 zone.value='UTC'; zone.dispatchEvent(new Event('change',{bubbles:true}));
 expect(root.querySelector<HTMLInputElement>('[name="timezoneCustom"]')!.value).toBe('');
 expect(root.querySelector('.settings-topic summary')!.textContent).toContain('#models');
 expect(root.querySelector('.settings-topic summary')!.textContent).toContain('★');
 input(root,'topicTag','new-tag'); root.querySelector('[name="topicTag"]')!.dispatchEvent(new Event('input',{bubbles:true}));
 expect(root.querySelector('.settings-topic summary')!.textContent).toContain('#new-tag');
 submit(root);
 await vi.waitFor(()=>expect(fetcher.mock.calls.some(([,init])=>init?.method==='POST')).toBe(true));
 expect(JSON.parse(String(fetcher.mock.calls.find(([,init])=>init?.method==='POST')![1]!.body)).values.timezone).toBe('UTC');
});
it("shows busy labels while fetching models and sending email", async () => {
 let release!: (r:Response)=>void;
 const {root}=setup(false,false,undefined,(path)=>path==='api/settings/action'?new Promise<Response>(resolve=>{release=resolve;}):undefined);
 await vi.waitFor(()=>expect(root.querySelector('.paper-workspace')).toBeTruthy());
 root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('.settings-form')).toBeTruthy());
 const models=root.querySelector<HTMLButtonElement>('[data-settings="models"]')!;models.click();
 await vi.waitFor(()=>expect(release).toBeTypeOf('function'));
 expect(models.textContent).toBe('Fetching…');expect(models.disabled).toBe(true);
 release(json({models:['test']}));await vi.waitFor(()=>expect(models.textContent).toBe('Get models'));
 const previousRelease=release;const mail=root.querySelector<HTMLButtonElement>('[data-email-self] [data-settings="email-test"]')!;mail.click();
 await vi.waitFor(()=>expect(mail.textContent).toBe('Sending…'));expect(mail.disabled).toBe(true);await vi.waitFor(()=>expect(release).not.toBe(previousRelease));
 release(json({run:{id:'mail',status:'completed',label:'Sent'}}));await vi.waitFor(()=>expect(mail.textContent).toBe('Send test'));
});
it("matches the exact quarter-hour run-window options and retains a custom current minute", async () => {
 const {root}=setup(false,false,undefined,path=>path==='api/settings'?json({setupRequired:false,revision:'r1',configPath:'/config.toml',values:{...values,schedule:{...DEFAULT_SETTINGS.schedule,runAtLocal:'08:07'}}}):undefined);
 await vi.waitFor(()=>expect(root.querySelector('.paper-workspace')).toBeTruthy());root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('.settings-form')).toBeTruthy());
 const expected=Array.from({length:96},(_,i)=>`${String(Math.floor(i/4)).padStart(2,'0')}:${String((i%4)*15).padStart(2,'0')}`);expected.push('08:07');expected.sort();
 expect(Array.from(root.querySelector<HTMLSelectElement>('[name="schedule.runAtLocal"]')!.options).map(o=>o.value)).toEqual(expected);
});
it("reloads the revision after library authorization completes and unlocks editing", async () => {
 let finished=false;let configReads=0;
 const run={id:'library-job',label:'Building library',status:'running',output:'',exitCode:null,startedAt:'2026-10-03',finishedAt:null};
 const {root,fetcher}=setup(false,false,undefined,(path,init)=>{
  if(path==='api/settings'&&!init?.method){configReads++;return json({setupRequired:false,revision:finished?'after-authorization':'revision-1',configPath:'/config.toml',values});}
  if(path==='api/settings/library')return json({status:{kind:'authorized',rootLabel:'/pdfs',grantedAt:'2026-10-03'},disclosure:null});
  if(path==='api/settings/action'){finished=true;return json({run});}
  if(path==='api/runs/current')return json({run:finished?{...run,status:'completed'}:null});
 });
 await vi.waitFor(()=>expect(root.querySelector('.paper-workspace')).toBeTruthy());root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('[data-settings="library-build"]')).toBeTruthy());
 input(root,'model','draft-kept');root.querySelector<HTMLButtonElement>('[data-settings="library-build"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector<HTMLInputElement>('[name="model"]')!.disabled).toBe(true));
 expect(root.querySelector<HTMLButtonElement>('[data-settings="models"]')!.disabled).toBe(true);
 await vi.waitFor(()=>expect(configReads).toBeGreaterThan(1),{timeout:3000});
 expect(root.querySelector<HTMLInputElement>('[name="model"]')!.value).toBe('draft-kept');
 expect(root.querySelector<HTMLInputElement>('[name="model"]')!.disabled).toBe(false);
 submit(root);
 await vi.waitFor(()=>expect(root.querySelector('.settings-form')).toBeNull());
 const saves=fetcher.mock.calls.filter(([url,init])=>String(url)==='api/settings'&&init?.method==='POST');
 expect(JSON.parse(String(saves.at(-1)![1]!.body)).revision).toBe('after-authorization');
});
it("runs the first-report guide action explicitly after saving the draft", async () => {
 const {root,fetcher}=setup(false,false,undefined,path=>path==='api/runs'?json({run:{id:'first',status:'running',label:'First report'}}):undefined);
 await vi.waitFor(()=>expect(root.querySelector('.paper-workspace')).toBeTruthy());root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('[data-setup-action="generate"]')).toBeTruthy());
 expect(fetcher.mock.calls.some(([url])=>String(url)==='api/runs')).toBe(false);
 root.querySelector<HTMLButtonElement>('[data-setup-action="generate"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('.run-tray')?.textContent).toContain('First report'));
 const postPaths=fetcher.mock.calls.filter(([,init])=>init?.method==='POST').map(([url])=>String(url));
 expect(postPaths).toEqual(['api/settings','api/runs']);
});
it("restores an active library run when settings opens and prevents duplicate cancellation", async () => {
 let finishCancel!: (r:Response)=>void;
 const run={id:'existing-index',label:'Build index',status:'running',output:'',exitCode:null,startedAt:'2026-10-03',finishedAt:null};
 const {root}=setup(false,false,undefined,path=>path==='api/settings/library'?json({status:{kind:'authorized',rootLabel:'/pdfs'},disclosure:null,run}):path==='api/runs/cancel'?new Promise<Response>(resolve=>{finishCancel=resolve;}):undefined);
 await vi.waitFor(()=>expect(root.querySelector('.paper-workspace')).toBeTruthy());root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('[data-settings="library-cancel"]')).toBeTruthy());
 expect(root.querySelector<HTMLInputElement>('[name="model"]')!.disabled).toBe(true);
 expect(root.querySelector<HTMLButtonElement>('[data-settings="add-topic"]')!.disabled).toBe(true);
 root.querySelector<HTMLButtonElement>('[data-settings="library-cancel"]')!.click();
 await vi.waitFor(()=>expect(finishCancel).toBeTypeOf('function'));
 expect(root.querySelector<HTMLButtonElement>('[data-settings="library-cancel"]')!.textContent).toBe('Cancelling…');
 expect(root.querySelector<HTMLButtonElement>('[data-settings="library-cancel"]')!.disabled).toBe(true);
 finishCancel(json({run:{...run,status:'cancelled'}}));
});

it('reveals a saved key on demand and offers an explicit model selector after fetching', async()=>{
 const {root,fetcher}=setup(false,false,undefined,(path)=>path==='api/settings/secret'?json({value:'stored-secret'}):path==='api/settings/action'?json({models:['model-a','model-b']}):undefined);
 await vi.waitFor(()=>expect(root.querySelector('.paper-workspace')).toBeTruthy());root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('.settings-form')).toBeTruthy());
 const key=root.querySelector<HTMLInputElement>('[name="apiKey"]')!;const show=key.nextElementSibling as HTMLButtonElement;
 expect(fetcher.mock.calls.some(([u])=>String(u)==='api/settings/secret')).toBe(false);
 show.click();await vi.waitFor(()=>expect(key.value).toBe('stored-secret'));expect(key.type).toBe('text');
 show.click();expect(key.type).toBe('password');expect(key.value).toBe('');
 key.value='new-draft';show.click();expect(key.value).toBe('new-draft');expect(key.type).toBe('text');
 expect(fetcher.mock.calls.filter(([u])=>String(u)==='api/settings/secret')).toHaveLength(1);
 root.querySelector<HTMLButtonElement>('[data-settings="models"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector<HTMLInputElement>('[name=model]')?.getAttribute('aria-expanded')).toBe('true'));
 expect(root.querySelector('[data-setting-name="Model"] select')).toBeNull();
 expect(Array.from(root.querySelectorAll('[role=option]')).map(o=>o.textContent)).toEqual(['model-a','model-b']);
 root.querySelectorAll<HTMLButtonElement>('[role=option]')[1]!.click();
 expect(root.querySelector<HTMLInputElement>('[name="model"]')!.value).toBe('model-b');
 show.click();await vi.waitFor(()=>expect(key.value).toBe('stored-secret'));
});

it('does not rebuild Getting started or move the settings position when loading models',async()=>{
 const {root}=setup(false,false,undefined,path=>path==='api/settings/action'?json({models:['test','another-model']}):undefined);
 await vi.waitFor(()=>expect(root.querySelector('.paper-workspace')).toBeTruthy());root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('.settings-setup')).toBeTruthy());
 root.querySelector<HTMLElement>('.settings-setup')!.dataset.preserved='true';
 const scroll=root.querySelector<HTMLElement>('.settings-content')!;scroll.scrollTop=240;
 root.querySelector<HTMLButtonElement>('[data-settings="models"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('[role=option]')?.textContent).toBe('test'));
 expect(root.querySelector<HTMLElement>('.settings-setup')?.dataset.preserved).toBe('true');
 expect(scroll.scrollTop).toBe(240);
});

it('keeps a completed setup guide absent when Get models saves changed settings',async()=>{
 const completeValues={...values,schedule:{...DEFAULT_SETTINGS.schedule,enabled:true}};
 const {root}=setup(false,false,undefined,(path,init)=>{
  if(path==='api/status')return json({llm:{ready:true},recentRuns:[{status:'completed'}]});
  if(path==='api/settings')return json({setupRequired:false,revision:'r1',configPath:'/config.toml',values:init?.method==='POST'?{...completeValues,...JSON.parse(String(init.body)).values,apiKeyConfigured:true}:completeValues});
  if(path==='api/settings/action')return json({models:['test']});
 });
 await vi.waitFor(()=>expect(root.querySelector('.paper-workspace')).toBeTruthy());root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('.settings-form')).toBeTruthy());
 expect(root.querySelector('.settings-setup')).toBeNull();
 root.querySelector<HTMLInputElement>('[name="schedule.enabled"]')!.checked=false;
 root.querySelector<HTMLButtonElement>('[data-settings="models"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('[role=option]')?.textContent).toBe('test'));
 expect(root.querySelector('.settings-setup')).toBeNull();
});
it('offers interface appearance separately from the report language and preserves user text', async () => {
 setUiLanguage('zh');
 const {root,fetcher}=setup();
 await vi.waitFor(()=>expect(root.querySelector('.paper-workspace')).toBeTruthy());root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('.settings-form')).toBeTruthy());
 expect(root.querySelector('[data-settings-key="Appearance"]')?.textContent).toBe('外观');
 expect(root.querySelector('[data-setting-name="API key"] .setting-item-name')?.textContent).toBe('API 密钥');
 expect(root.querySelector<HTMLInputElement>('[name="topicName"]')!.value).toBe('Models');
 const theme=root.querySelector<HTMLSelectElement>('[name="appearance.theme"]')!;
 const language=root.querySelector<HTMLSelectElement>('[name="appearance.language"]')!;
 theme.value='dark';language.value='en';
 submit(root);
 await vi.waitFor(()=>expect(root.querySelector('.settings-form')).toBeNull());
 const product=fetcher.mock.calls.find(([url,init])=>String(url)==='api/settings'&&init?.method==='POST')!;
 const preferences=fetcher.mock.calls.find(([url,init])=>String(url)==='api/preferences'&&init?.method==='POST')!;
 expect(JSON.parse(String(product[1]!.body)).values.summaryLanguage).toBe('zh');
 expect(JSON.parse(String(preferences[1]!.body))).toEqual({appearance:{theme:'dark',language:'en'}});
});
it('keeps translated labels separate from user topics, model identifiers and submitted fields', async()=>{
 setUiLanguage('zh');
 const customValues={...values,model:'English',topics:[{id:'theme',name:'Theme',tag:'Theme',description:'Saved only on this device.',detail:true}]};
 const {root}=setup(false,false,undefined,(path,init)=>path==='api/settings'&&!init?.method?json({setupRequired:false,revision:'r1',configPath:'/config.toml',values:customValues}):undefined);
 await vi.waitFor(()=>expect(root.querySelector('.paper-workspace')).toBeTruthy());root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
 await vi.waitFor(()=>expect(root.querySelector('.settings-form')).toBeTruthy());
 expect(root.querySelector<HTMLInputElement>('[name="model"]')!.value).toBe('English');
 expect(root.querySelector<HTMLInputElement>('[name="topicName"]')!.value).toBe('Theme');
 expect(root.querySelector<HTMLTextAreaElement>('[name="topicDescription"]')!.value).toBe('Saved only on this device.');
 expect(root.querySelector('.settings-topic-name')!.textContent).toBe('Theme');
 expect(root.querySelector('[data-setting-name="Theme"] .setting-item-name')!.textContent).toBe('主题');
 expect(root.querySelector('[data-setting-name="Run window"] .setting-item-name')!.textContent).toBe('运行时间段');
 expect(root.querySelector<HTMLInputElement>('[name="email.apiKey"]')!.getAttribute('aria-label')).toBe('Resend API 密钥');
 expect(root.querySelector('[data-settings="models"]')!.textContent).toBe('获取模型');
 expect(root.querySelector('nav[aria-label="设置导航"]')).toBeTruthy();
});
