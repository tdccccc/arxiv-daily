// @vitest-environment happy-dom
import { afterEach, expect, it, vi } from "vitest";
import { mountLibraryReview, type LibraryReviewViewSnapshot } from "../src/workbench/web/library-review";
import { setUiLanguage } from "../src/workbench/web/i18n";
import type { WorkbenchRun } from "../src/workbench/server";
const evidence = (paperKey: string) => ({ paperKey, evidenceFingerprint: "sha256:evidence" });
const candidate = (id: string, keys = ["p1", "p2"]) => ({ id, text: `Direction ${id} $H_0$`, discoveryCues: ["galaxies"], representatives: keys.map(evidence), representativeSetFingerprint: "sha256:representatives", lineage: { candidateIds: [id] } });
const fixture = (): LibraryReviewViewSnapshot => ({
 configRevision: "config-1", connected: true, topics: [{ id: "existing", name: "Cosmology", tag: "cosmo", detail: false, description: "Changed direction", directions: [{ id: "old", text: "Changed direction", origin: "manual" }] }], acceptances: [{ proposalId: "proposal-1", scopeFingerprint: "scope", topicTargets: {}, processedCandidateIds: ["processed"] }],
 catalog: { schemaVersion: 1, revision: 1, scopeFingerprint: "scope", identificationFingerprint: "identity", updatedAt: "2026-10-05", lastScan: null, files: {}, papers: {} },
 indexedPapers: ["p1", "p2", "p3", "p4"].map(paperKey => ({ paperKey, title: `Paper ${paperKey}`, abstract: "abstract", categories: ["astro-ph.CO"] })),
 proposal: { schemaVersion: 6, revision: 4, proposalId: "proposal-1", scopeFingerprint: "scope", identificationFingerprint: "identity", catalogInputFingerprint: "input", catalogInputPapers: ["p1", "p2", "p3"].map(evidence), generationContractFingerprint: "contract", generatedAt: "2026-10-05T00:00:00Z", topics: [{ id: "topic", suggestedName: "Stars", directions: [candidate("strong"), candidate("thin", ["p1"]), candidate("processed")] }], coveredPaperKeys: ["p3"], coverageEvidence: [{ topicId: "existing", directionId: "old", directionText: "Original direction", paperKeys: ["p3"] }] }
});
const run: WorkbenchRun = { id: "run-1", label: "preview", date: null, status: "running", output: "", exitCode: null, startedAt: "2026-10-05T00:00:00Z", finishedAt: null };
const dispose: Array<() => void> = [];
afterEach(() => { dispose.splice(0).forEach(fn => fn()); document.body.innerHTML = ""; setUiLanguage("zh"); });
const settle = async () => { await new Promise(resolve => setTimeout(resolve, 0)); };
function setup(request = vi.fn().mockResolvedValue(fixture()), inspection: unknown = {status:{kind:"authorized"},disclosure:null}) {
 const root = document.createElement("div"); document.body.append(root); const onRun = vi.fn();
 const inspect = vi.fn().mockResolvedValue(inspection);
 const transport = <T>(url:string,body?:unknown):Promise<T> => url === "api/settings/library" ? inspect() : body === undefined ? request(url) : request(url,body);
 const view = mountLibraryReview(root, { request:transport, onSettings: vi.fn(), onLibrary: vi.fn(), onRun });
 dispose.push(view.dispose); return { root, request, view, onRun, inspect };
}
const click = (root: HTMLElement, action: string) => (root.querySelector(`[data-review="${action}"]`) as HTMLButtonElement).click();
const row = (root: HTMLElement, id = "strong") => root.querySelector<HTMLElement>(`[data-candidate="${id}"]`)!;
function edit(root: HTMLElement, field: string, value: string) { const input = root.querySelector<HTMLInputElement>(`[data-field="${field}"]`)!; input.value = value; input.dispatchEvent(new Event("input", { bubbles: true })); }

it("loads proposals without model calls, uses matching receipts and leaves thin evidence unselected", async () => {
 const {root, request} = setup(); await settle();
 expect(request).toHaveBeenCalledExactlyOnceWith("api/library/review");
 expect(row(root).querySelector<HTMLInputElement>('[data-select]')!.checked).toBe(true);
 expect(row(root,"thin").querySelector<HTMLInputElement>('[data-select]')!.checked).toBe(false);
 expect(row(root,"processed").querySelector<HTMLInputElement>('[data-select]')!.disabled).toBe(true);
 expect(root.textContent).toContain("证据较少"); expect(root.textContent).toContain("已处理");
 expect(root.querySelector(".katex")).toBeTruthy();
 expect(root.querySelector('a')?.getAttribute("href")).toBe("api/library/pdf?key=p1");
});

it("edits text, cues and representatives with both revision guards and preserves draft on failure", async () => {
 const request = vi.fn().mockResolvedValueOnce(fixture()).mockRejectedValueOnce(new Error("Proposal changed"));
 const {root} = setup(request); await settle();
 edit(row(root),"text","Edited direction"); edit(row(root),"cues","galaxies\nclusters");
 const reps = row(root).querySelector<HTMLSelectElement>('[data-field="representatives"]')!;
 for (const option of reps.options) option.selected = ["p2","p3"].includes(option.value);
 reps.dispatchEvent(new Event("change", {bubbles:true}));
 click(row(root),"save"); await settle();
 expect(request).toHaveBeenLastCalledWith("api/library/review", { operation:"update-candidate",configRevision:"config-1",expectedProposalRevision:4,candidateId:"strong",patch:{text:"Edited direction",discoveryCues:["galaxies","clusters"]},representativePaperKeys:["p2","p3"] });
 expect(root.querySelector('[role="alert"]')?.textContent).toContain("Proposal changed");
 expect(row(root).querySelector<HTMLTextAreaElement>('[data-field="text"]')!.value).toBe("Edited direction");
});

it("requires confirmation before accepting selected candidates and does not accept thin or processed ones implicitly", async () => {
 const {root,request} = setup(); await settle(); click(root,"accept");
 expect(request).toHaveBeenCalledTimes(1); expect(root.querySelector('[data-confirmation][role="group"]')).toBeTruthy();
 click(root,"confirm"); await settle();
 expect(request).toHaveBeenLastCalledWith("api/library/review", { operation:"accept-topics",configRevision:"config-1",expectedProposalRevision:4,topicIds:["topic"],candidateIds:["strong"] });
});

it("supports rename, moving to an existing topic and confirmed deletion", async () => {
 const {root,request} = setup(); await settle();
 edit(root,"topic-name","New topic name"); click(root,"rename"); await settle();
 expect(request).toHaveBeenLastCalledWith("api/library/review", {operation:"rename-topic",configRevision:"config-1",expectedProposalRevision:4,topicId:"topic",suggestedName:"New topic name"});
 const destination = row(root).querySelector<HTMLSelectElement>('[data-field="destination"]')!; destination.value="existing"; destination.dispatchEvent(new Event("change",{bubbles:true}));
 click(row(root),"move"); await settle();
 expect(request).toHaveBeenLastCalledWith("api/library/review", {operation:"move-direction",configRevision:"config-1",expectedProposalRevision:4,candidateId:"strong",targetTopicId:"existing",suggestedName:"Cosmology"});
 click(row(root),"remove"); const count=request.mock.calls.length; expect(root.querySelector('[data-confirmation][role="group"]')).toBeTruthy();
 click(root,"confirm"); await settle(); expect(request.mock.calls.length).toBe(count+1);
 expect(request).toHaveBeenLastCalledWith("api/library/review", {operation:"remove-candidate",configRevision:"config-1",expectedProposalRevision:4,candidateId:"strong"});
});

it("shows dated overview, changed coverage and papers outside the analysis", async () => {
 const {root} = setup(); await settle(); click(root,"overview");
 expect(root.querySelector("time")?.dateTime).toBe("2026-10-05T00:00:00Z");
 expect(root.textContent).toContain("方向已改变，需重新生成验证");
 expect(root.textContent).toContain("分析之后新增的论文"); expect(root.textContent).toContain("Paper p4");
});

it("starts preview in the shared run tray and fetches its result without discarding another candidate draft", async () => {
 const request=vi.fn().mockResolvedValueOnce(fixture()).mockResolvedValueOnce({run}).mockResolvedValueOnce({preview:{directionText:"Direction strong",categories:["astro-ph.CO"],missingCategories:["astro-ph.GA"],papers:[{paperKey:"p1",title:"Preview paper",abstract:"",categories:["astro-ph.GA"],matched:true,directionText:"Direction strong",categoryCoverage:"outside"}]}});
 const {root,onRun,view}=setup(request); await settle(); edit(row(root,"thin"),"text","Keep this edit"); click(row(root),"preview"); await settle();
 expect(request).toHaveBeenLastCalledWith("api/library/preview",{configRevision:"config-1",expectedProposalRevision:4,candidateId:"strong"}); expect(onRun).toHaveBeenCalledWith(run);
 await view.handleRun({...run,status:"completed"}); await settle();
 expect(root.textContent).toContain("Preview paper"); expect(root.textContent).toContain("astro-ph.GA");
 expect(row(root,"thin").querySelector<HTMLTextAreaElement>('[data-field="text"]')!.value).toBe("Keep this edit");
});

it("keeps editors open and preserves other drafts across a failed save and refresh", async () => {
 const request=vi.fn().mockResolvedValueOnce(fixture()).mockRejectedValueOnce(new Error("Direction proposal changed; reload the review and retry")).mockResolvedValue(fixture());
 const {root,view}=setup(request); await settle();
 row(root).querySelector<HTMLDetailsElement>("details")!.open=true; await settle();
 edit(row(root),"text","Draft survives"); click(row(root),"save"); await settle();
 expect(row(root).querySelector<HTMLDetailsElement>("details")!.open).toBe(true);
 await view.refresh();
 expect(row(root).querySelector<HTMLTextAreaElement>('[data-field="text"]')!.value).toBe("Draft survives");
});

it("does not accept unsaved topic names or regenerate over unsaved direction edits", async () => {
 const {root}=setup(); await settle();
 edit(root,"topic-name","Unsaved topic");
 expect((root.querySelector('[data-review="accept"]') as HTMLButtonElement).disabled).toBe(true);
 edit(row(root),"text","Unsaved direction");
 expect((root.querySelector('[data-review="propose"]') as HTMLButtonElement).disabled).toBe(true);
});

it("localizes the whole review surface and translates common stale-state failures", async () => {
 setUiLanguage("en"); const {root}=setup(); await settle();
 expect(root.textContent).toContain("Proposed directions");
 expect(root.textContent).toContain("Thin evidence");
 expect(root.textContent).toContain("Accept selected directions");
 expect(root.textContent).not.toMatch(/[\u4e00-\u9fff]/);
 setUiLanguage("zh"); const second=setup(vi.fn().mockResolvedValueOnce(fixture()).mockRejectedValueOnce(new Error("Direction proposal changed; reload the review and retry"))); await settle();
 click(row(second.root),"save"); await settle();
 expect(second.root.querySelector('[role="alert"]')?.textContent).toContain("候选方向已改变");
});

it("locks the reviewed selection while confirmation is visible", async () => {
 const {root,request}=setup(); await settle(); click(root,"accept");
 const thin = row(root,"thin").querySelector<HTMLInputElement>('[data-select]')!;
 expect(thin.disabled).toBe(true);
 thin.checked=true; thin.dispatchEvent(new Event("change",{bubbles:true}));
 click(root,"confirm"); await settle();
 expect(request.mock.calls.at(-1)?.[1].candidateIds).toEqual(["strong"]);
});

it("protects renamed topics on a replaced analysis and offers explicit discard and refresh", async () => {
 const newer=fixture(); newer.proposal!.proposalId="new-proposal";
 const request=vi.fn().mockResolvedValueOnce(fixture()).mockResolvedValue(newer);
 const {root,view}=setup(request); await settle(); edit(root,"topic-name","Unsaved topic"); await view.refresh();
 expect(root.textContent).toContain("当前修改仍保留");
 expect(root.querySelector<HTMLInputElement>('[data-field="topic-name"]')!.value).toBe("Unsaved topic");
 click(root,"discard-refresh"); click(root,"confirm"); await settle();
 expect(root.querySelector<HTMLInputElement>('[data-field="topic-name"]')!.value).toBe("Stars");
});

it("recognizes processed ancestor identities and resumed model runs", async () => {
 const initial=fixture(); initial.proposal!.topics[0]!.directions[0]!.lineage.candidateIds.push("processed");
 const request=vi.fn().mockResolvedValueOnce(initial).mockResolvedValueOnce({preview:{directionText:"Restored preview",categories:[],missingCategories:[],papers:[]}});
 const {root,view}=setup(request); await settle();
 expect(row(root).querySelector<HTMLInputElement>('[data-select]')!.disabled).toBe(true);
 await view.handleRun({...run,label:"预览研究方向"});
 await view.handleRun({...run,label:"预览研究方向",status:"completed"});
 expect(root.textContent).toContain("Restored preview");
 await view.handleRun({...run,label:"预览研究方向",status:"completed"});
 expect(request).toHaveBeenCalledTimes(2);
});


const disclosure = { selectedRoot:"/literature",eligibleExtensions:["pdf"],processingDepth:"full-text",endpoint:"https://model.example/v1",embeddingEndpoint:"https://embed.example/v1",authorizationFingerprint:"consent-fingerprint" };
it("discloses model processing before analysis and cancellation sends no authorization or model request", async () => {
 const {root,request,inspect}=setup(vi.fn().mockResolvedValue(fixture()),{status:{kind:"authorization-required"},disclosure}); await settle();
 click(root,"propose"); await settle();
 expect(inspect).toHaveBeenCalledOnce();
 const confirmation=root.querySelector("[data-confirmation]")!;
 expect(confirmation.textContent).toContain("/literature"); expect(confirmation.textContent).toContain("https://model.example/v1"); expect(confirmation.textContent).toContain("https://embed.example/v1");
 expect(confirmation.textContent).toContain("标题与摘要");
 click(root,"cancel"); expect(request).toHaveBeenCalledTimes(1);
});

it("authorizes only after confirmation and starts analysis with the newly saved config revision", async () => {
 const authorized={...fixture(),configRevision:"config-after-consent"};
 const request=vi.fn().mockResolvedValueOnce(fixture()).mockResolvedValueOnce(authorized).mockResolvedValueOnce({run});
 const {root,onRun}=setup(request,{status:{kind:"authorization-required"},disclosure}); await settle(); click(root,"propose"); await settle(); click(root,"confirm"); await settle();
 expect(request.mock.calls[1]).toEqual(["api/library/authorize",{configRevision:"config-1",fingerprint:"consent-fingerprint"}]);
 expect(request.mock.calls[2]).toEqual(["api/library/propose",{configRevision:"config-after-consent",expectedProposalRevision:4}]); expect(onRun).toHaveBeenCalledWith(run);
});

it("preserves the review and other drafts when authorization or model configuration is invalid", async () => {
 const request=vi.fn().mockResolvedValueOnce(fixture()).mockRejectedValueOnce(new Error("Library model endpoint is invalid; fix its settings before authorizing processing."));
 const {root}=setup(request,{status:{kind:"authorization-required"},disclosure}); await settle(); edit(row(root,"thin"),"text","Still here");
 click(row(root),"preview"); await settle(); click(root,"confirm"); await settle();
 expect(request).toHaveBeenCalledTimes(2); expect(root.querySelector('[role="alert"]')?.textContent).toContain("模型端点无效");
 expect(row(root,"thin").querySelector<HTMLTextAreaElement>('[data-field="text"]')!.value).toBe("Still here");
});

it("ignores stale snapshot requests and completion after disposal", async () => {
 let first!: (value: LibraryReviewViewSnapshot) => void;
 const request=vi.fn().mockImplementationOnce(() => new Promise(resolve => {first=resolve;})).mockResolvedValue(fixture());
 const {root,view}=setup(request); await view.refresh();
 const old=fixture(); old.proposal!.topics[0]!.suggestedName="Stale topic"; first(old); await settle();
 expect(root.textContent).toContain("Stars"); expect(root.textContent).not.toContain("Stale topic");
 let late!: (value: LibraryReviewViewSnapshot) => void;
 request.mockImplementationOnce(() => new Promise(resolve => {late=resolve;}));
 const pending=view.refresh(); view.dispose(); root.innerHTML="Another view"; late(fixture()); await pending;
 expect(root.innerHTML).toBe("Another view");
});

it("does not refresh for an old completed proposal run on initial mount", async () => {
 const {view,request}=setup(); await settle();
 await view.handleRun({...run,label:"生成候选方向",status:"completed",finishedAt:"2026-01-01T00:00:00Z"});
 await view.handleRun({...run,label:"生成候选方向",status:"completed",finishedAt:"2026-01-01T00:00:00Z"});
 expect(request).toHaveBeenCalledOnce();
});

it("offers connection settings instead of authorizing when disclosure is unavailable", async () => {
 const {root,request}=setup(vi.fn().mockResolvedValue(fixture()),{status:{kind:"invalid"},disclosure:null}); await settle();
 click(row(root),"preview"); await settle();
 expect(root.querySelector('[role="alert"]')?.textContent).toContain("修正文献库或模型配置");
 expect(root.querySelector("[data-confirmation]")).toBeNull(); expect(request).toHaveBeenCalledOnce();
 expect(root.querySelector('[data-review="settings"]')).toBeTruthy();
});

it("does not let an older preview failure overwrite the latest completed preview", async () => {
 let rejectOld!: (reason: Error) => void;
 const request=vi.fn().mockResolvedValueOnce(fixture()).mockImplementationOnce(() => new Promise((_resolve,reject) => {rejectOld=reject;})).mockResolvedValueOnce({preview:{directionText:"Latest preview",categories:[],missingCategories:[],papers:[]}});
 const {root,view}=setup(request); await settle();
 const old=view.handleRun({...run,id:"old",label:"预览研究方向",status:"completed"});
 await view.handleRun({...run,id:"new",label:"预览研究方向",status:"completed"});
 rejectOld(new Error("Old preview failed")); await old;
 expect(root.textContent).toContain("Latest preview"); expect(root.querySelector('[role="alert"]')).toBeNull();
});

it("keeps a restored preview when the initial proposal snapshot arrives afterwards", async () => {
 let resolveSnapshot!: (value: LibraryReviewViewSnapshot) => void;
 const request=vi.fn().mockImplementationOnce(() => new Promise(resolve => {resolveSnapshot=resolve;})).mockResolvedValueOnce({preview:{directionText:"Restored before snapshot",categories:[],missingCategories:[],papers:[]}});
 const {root,view}=setup(request);
 await view.handleRun({...run,label:"预览研究方向",status:"completed"});
 resolveSnapshot(fixture()); await settle();
 expect(root.querySelector(".review-preview")?.textContent).toContain("Restored before snapshot");
});
