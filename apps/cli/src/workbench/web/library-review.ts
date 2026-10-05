import {
  isThinEvidenceDirectionCandidate, matchingProposalAcceptance, resolveProposedTopicTarget,
  type LibraryReviewSnapshot, type LibraryPreviewPaper, type LibraryDirectionPreview,
  type PersonalLibraryDirectionCandidate, type PersonalLibraryProposedTopic, type LibraryAuthorizationDisclosure,
} from "@arxiv-daily/core";
import { t } from "./i18n";
import { escapeHtml as e, scientificInline } from "./papers";
import type { WorkbenchRun } from "../server";

export interface LibraryReviewViewSnapshot extends LibraryReviewSnapshot {
  configRevision: string;
  connected: boolean;
  indexedPapers: LibraryPreviewPaper[];
}
interface Options {
  request<T>(url: string, body?: unknown): Promise<T>;
  onSettings(): void;
  onLibrary(): void;
  onRun(run: WorkbenchRun): void;
}
interface Draft { text: string; cues: string; representatives: string[]; destination: string; name: string }
interface Confirmation { title: string; description: string; disclosure?: LibraryAuthorizationDisclosure; execute(): Promise<void> }

export function mountLibraryReview(root: HTMLElement, options: Options) {
  let snapshot: LibraryReviewViewSnapshot | null = null;
  let disposed = false;
  let sequence = 0;
  let busy = false;
  let loading = true;
  let error = "";
  let tab: "proposed" | "overview" = "proposed";
  let identity = "";
  let confirmation: Confirmation | null = null;
  let preview: LibraryDirectionPreview | null = null;
  const selected = new Set<string>();
  const drafts = new Map<string, Draft>();
  const names = new Map<string, string>();
  const collapsed = new Set<string>();
  const editors = new Set<string>();
  const jobs = new Map<string, "propose" | "preview">();
  const finishedJobs = new Set<string>();
  const mountedAt = Date.now();
  let previewSequence = 0;
  const text = (source: string, ...args: Array<string | number>) => e(t(source, ...args));
  const button = (action: string, source: string, disabled = false) => `<button type="button" class="quiet-button" data-review="${action}" ${disabled || busy ? "disabled" : ""}>${text(source)}</button>`;
  const receipt = () => snapshot?.proposal ? snapshot.acceptances.map(item => matchingProposalAcceptance(snapshot!.proposal!, item)).find(Boolean) : null;
  const processed = () => {
    const ids = new Set(receipt()?.processedCandidateIds ?? []);
    for (const candidate of snapshot?.proposal?.topics.flatMap(topic => topic.directions) ?? []) if (candidate.lineage.candidateIds.some(id => ids.has(id))) ids.add(candidate.id);
    return ids;
  };
  const candidateById = (id: string) => snapshot?.proposal?.topics.flatMap(topic => topic.directions).find(candidate => candidate.id === id);
  const original = (candidate: PersonalLibraryDirectionCandidate): Draft => ({ text: candidate.text, cues: candidate.discoveryCues.join("\n"), representatives: candidate.representatives.map(item => item.paperKey), destination: "", name: "" });
  const draft = (candidate: PersonalLibraryDirectionCandidate) => drafts.get(candidate.id) ?? original(candidate);
  function dirty(candidate: PersonalLibraryDirectionCandidate): boolean {
    const value = draft(candidate), initial = original(candidate);
    return value.text !== initial.text || value.cues !== initial.cues || value.representatives.join("\0") !== initial.representatives.join("\0");
  }
  const dirtyAny = () => snapshot?.proposal?.topics.some(topic => topic.directions.some(dirty) || (names.has(topic.id) && names.get(topic.id) !== topic.suggestedName)) ?? false;
  const guards = () => ({ configRevision: snapshot!.configRevision, expectedProposalRevision: snapshot!.proposal?.revision ?? null });
  function destination(topic: PersonalLibraryProposedTopic) {
    try { return { topic: resolveProposedTopicTarget(topic, snapshot!.topics, receipt()), error: "" }; }
    catch (reason) { return { topic: null, error: t(reason instanceof Error ? reason.message : String(reason)) }; }
  }
  function paperLink(key: string): string {
    const paper = snapshot?.catalog.papers[key] ?? snapshot?.indexedPapers.find(item => item.paperKey === key);
    return `<a href="api/library/pdf?key=${encodeURIComponent(key)}" target="_blank" rel="noopener noreferrer">${scientificInline(paper?.title ?? key)} ↗</a>`;
  }
  const paperList = (keys: string[]) => `<ul class="review-paper-links">${keys.map(key => `<li>${paperLink(key)}</li>`).join("")}</ul>`;
  function previewHtml(): string {
    if (!preview) return "";
    return `<section class="review-preview"><h2>${text("方向预览")}</h2><p>${scientificInline(preview.directionText)}</p><p class="review-help">${text("预览仅检查文献库样本，不修改订阅，也不预测未来日报数量。")}</p>${preview.missingCategories.length ? `<p>${text("匹配论文中的未订阅分类：")}${e(preview.missingCategories.join(", "))}</p>` : ""}<ul>${preview.papers.map(paper => `<li>${paperLink(paper.paperKey)} <span>${text(paper.matched ? "匹配" : "未匹配")} · ${text(paper.categoryCoverage === "inside" ? "已订阅分类" : paper.categoryCoverage === "outside" ? "未订阅分类" : "分类未知")}</span><div>${scientificInline(paper.title)}</div></li>`).join("")}</ul>${button("close-preview", "关闭预览")}</section>`;
  }
  function candidateHtml(candidate: PersonalLibraryDirectionCandidate): string {
    const value = draft(candidate), done = processed().has(candidate.id), thin = isThinEvidenceDirectionCandidate(candidate);
    const allowedKeys = snapshot!.proposal!.catalogInputPapers.map(item => item.paperKey);
    return `<article class="review-candidate" data-candidate="${e(candidate.id)}"><header><label><input type="checkbox" data-select="${e(candidate.id)}" ${selected.has(candidate.id) ? "checked" : ""} ${done || busy ? "disabled" : ""}> <span>${scientificInline(candidate.text)}</span></label>${done ? `<span class="review-badge">${text("已处理")}</span>` : thin ? `<span class="review-badge">${text("证据较少")}</span>` : ""}</header>${paperList(candidate.representatives.map(item => item.paperKey))}
      ${done ? `<p class="review-help">${text("已审核的方向请在研究主题设置中管理。")}</p>` : `<details class="review-editor" ${editors.has(candidate.id) ? "open" : ""}><summary>${text("编辑方向与代表论文")}</summary><fieldset ${busy ? "disabled" : ""}><label>${text("方向文本")}<textarea data-field="text" rows="2" maxlength="1000">${e(value.text)}</textarea></label><label>${text("发现线索（每行一条）")}<textarea data-field="cues" rows="3">${e(value.cues)}</textarea></label><label>${text("代表论文（选择 1–5 篇）")}<select multiple size="${Math.max(2, Math.min(5, allowedKeys.length))}" data-field="representatives">${allowedKeys.map(key => `<option value="${e(key)}" ${value.representatives.includes(key) ? "selected" : ""}>${e(snapshot!.catalog.papers[key]?.title ?? snapshot!.indexedPapers.find(paper => paper.paperKey === key)?.title ?? key)}</option>`).join("")}</select></label><div class="review-actions">${button("save", "保存方向")}${button("reset", "放弃此方向的修改")}</div><div class="review-move"><label>${text("移动到主题")}<select data-field="destination"><option value="">${text("新主题")}</option>${snapshot!.topics.map(topic => `<option value="${e(topic.id)}" ${value.destination === topic.id ? "selected" : ""}>${e(topic.name)}</option>`).join("")}</select></label><label>${text("新主题名称")}<input data-field="name" value="${e(value.name)}" maxlength="120"></label>${button("move", "移动方向")}</div></fieldset></details><div class="review-actions">${button("preview", "预览方向", dirty(candidate) || jobs.size > 0)}${button("remove", "删除候选")}<span class="review-dirty" ${dirty(candidate) ? "" : "hidden"}>${text("请先保存修改，再预览或接受。")}</span></div>`}</article>`;
  }
  function proposedHtml(): string {
    const proposal = snapshot!.proposal;
    if (!proposal) return `<div class="review-empty"><h2>${text("尚无候选方向")}</h2><p>${text("先建立文献库索引，再生成候选方向；审核接受后才参与每日发现。")}</p></div>`;
    if (!proposal.topics.length) return `<div class="review-empty"><p>${text("本次分析没有新增候选方向。请查看文献库概览。")}</p></div>`;
    return `<div class="review-selection">${button("select-all", "选择全部可审核方向")}${button("select-none", "取消选择")}<span>${text("已选择 {0} 个方向", selected.size)}</span>${button("accept", "接受所选方向", selected.size === 0 || dirtyAny())}</div>${proposal.topics.map(topic => {
      const target = destination(topic), remaining = topic.directions.filter(item => !processed().has(item.id));
      return `<details class="review-topic" data-topic="${e(topic.id)}" ${collapsed.has(topic.id) ? "" : "open"}><summary><strong>${e(topic.suggestedName)}</strong> <span>${text("{0} 个方向", topic.directions.length)}</span></summary><div class="review-topic-body"><div class="review-topic-controls"><label><input type="checkbox" data-topic-select="${e(topic.id)}" ${remaining.length && remaining.every(item => selected.has(item.id)) ? "checked" : ""} ${!remaining.length || busy ? "disabled" : ""}> ${text("选择此主题")}</label>${target.error ? `<p role="alert">${e(target.error)}</p>` : target.topic ? `<p>${text("将加入现有主题：")}${e(target.topic.name)}</p>` : `<label>${text("建议主题名称")}<input data-field="topic-name" value="${e(names.get(topic.id) ?? topic.suggestedName)}" maxlength="120" ${busy ? "disabled" : ""}></label>${button("rename", "保存主题名称")}`}</div>${topic.directions.map(candidateHtml).join("")}</div></details>`;
    }).join("")}`;
  }
  function overviewHtml(): string {
    const proposal = snapshot!.proposal;
    if (!proposal) return `<p class="review-empty">${text("尚无文献库分析")}</p>`;
    const analyzed = new Set(proposal.catalogInputPapers.map(item => item.paperKey));
    const represented = new Set(proposal.topics.flatMap(topic => topic.directions.flatMap(candidate => (candidate.clusterMembers?.length ? candidate.clusterMembers : candidate.representatives).map(item => item.paperKey))));
    const covered = new Set(proposal.coveredPaperKeys ?? []);
    const uncovered = [...analyzed].filter(key => !covered.has(key) && !represented.has(key));
    const added = snapshot!.indexedPapers.filter(paper => !analyzed.has(paper.paperKey)).map(paper => paper.paperKey);
    const coverage = (proposal.coverageEvidence ?? []).map(item => {
      const topic = snapshot!.topics.find(topic => topic.id === item.topicId);
      const current = topic?.directions.find(direction => direction.id === item.directionId)?.text === item.directionText;
      return `<li><strong>${e(topic?.name ?? t("主题已删除"))}</strong> · ${scientificInline(item.directionText)}<p>${text(current ? "当前方向已覆盖" : "方向已改变，需重新生成验证")}</p>${paperList(item.paperKeys)}</li>`;
    }).join("");
    return `<div class="review-overview"><p>${text("分析了 {0} 篇论文", analyzed.size)} · <time datetime="${e(proposal.generatedAt)}">${e(proposal.generatedAt)}</time></p><section><h2>${text("已有方向覆盖")}</h2>${coverage ? `<ul>${coverage}</ul>` : `<p>${text("暂无已确认的覆盖记录")}</p>`}${proposal.coverageEvidence === undefined && covered.size ? `<p>${text("历史覆盖尚未验证，请重新生成分析。")}</p>` : ""}</section><section><h2>${text("候选方向与代表论文")}</h2>${proposal.topics.map(topic => `<details open><summary>${e(topic.suggestedName)}</summary>${topic.directions.map(candidate => `<div><h3>${scientificInline(candidate.text)} ${processed().has(candidate.id) ? `<small>${text("已处理")}</small>` : ""}</h3>${paperList(candidate.representatives.map(item => item.paperKey))}</div>`).join("")}</details>`).join("")}</section><section><h2>${text("暂未归入方向的论文")}</h2>${paperList(uncovered)}${!uncovered.length ? `<p>${text("暂无")}</p>` : ""}</section><section><h2>${text("分析之后新增的论文")}</h2>${paperList(added)}${!added.length ? `<p>${text("暂无")}</p>` : ""}</section></div>`;
  }
  function render(): void {
    if (disposed) return;
    root.innerHTML = `<section class="library-review-workspace"><header class="review-heading"><div><h1>${text("方向审核")}</h1><p>${text("检查候选方向及其依据；接受后保存到普通研究主题。")}</p></div><div class="review-actions">${button("library", "返回文献库")}${button("settings", "连接与索引")}${button("refresh", "刷新")}</div></header><div class="review-feedback" aria-live="polite">${error ? `<div role="alert">${e(error)} ${button("refresh", "重新加载")}${dirtyAny() ? button("discard-refresh", "放弃修改并刷新") : ""}</div>` : ""}${busy ? `<p role="status">${text("正在保存…")}</p>` : ""}</div>
      ${loading ? `<p role="status">${text("正在加载方向审核…")}</p>` : !snapshot ? `<p>${text("方向审核加载失败")}</p>` : !snapshot.connected ? `<div class="review-empty"><h2>${text("尚未连接个人文献库")}</h2><p>${text("先在连接与索引中选择文献目录。")}</p></div>` : `<div class="review-topbar"><div role="tablist" aria-label="${text("文献库分析视图")}"><button role="tab" data-review="proposed" aria-selected="${tab === "proposed"}" tabindex="${tab === "proposed" ? 0 : -1}">${text("候选方向")}</button><button role="tab" data-review="overview" aria-selected="${tab === "overview"}" tabindex="${tab === "overview" ? 0 : -1}">${text("文献库概览")}</button></div>${button("propose", snapshot.proposal ? "重新生成候选" : "生成候选方向", jobs.size > 0 || dirtyAny())}</div><div role="tabpanel">${tab === "proposed" ? proposedHtml() : overviewHtml()}</div>${previewHtml()}`}
      ${confirmation ? `<section class="review-confirmation" role="group" data-confirmation aria-label="${e(confirmation.title)}"><h2>${e(confirmation.title)}</h2><p>${e(confirmation.description)}</p>${confirmation.disclosure ? disclosureHtml(confirmation.disclosure) : ""}<div class="review-actions">${button("confirm", "确认")}${button("cancel", "取消")}</div></section>` : ""}</section>`;
    for (const details of Array.from(root.querySelectorAll<HTMLDetailsElement>(".review-topic"))) details.addEventListener("toggle", () => { if (details.open) collapsed.delete(details.dataset.topic!); else collapsed.add(details.dataset.topic!); });
    for (const details of Array.from(root.querySelectorAll<HTMLDetailsElement>(".review-editor"))) details.addEventListener("toggle", () => { const id = details.closest<HTMLElement>("[data-candidate]")!.dataset.candidate!; if (details.open) editors.add(id); else editors.delete(id); });
    if (confirmation) for (const control of Array.from(root.querySelectorAll<HTMLInputElement | HTMLButtonElement | HTMLSelectElement | HTMLTextAreaElement>("input, button, select, textarea"))) if (!control.closest("[data-confirmation]")) control.disabled = true;
    if (confirmation) root.querySelector<HTMLButtonElement>('[data-review="cancel"]')?.focus();
  }
  function apply(value: LibraryReviewViewSnapshot): void {
    if (snapshot && snapshot.proposal?.revision !== value.proposal?.revision) { preview = null; ++previewSequence; }
    const key = value.proposal ? `${value.proposal.scopeFingerprint}/${value.proposal.proposalId}` : "";
    if (key !== identity) { selected.clear(); drafts.clear(); names.clear(); collapsed.clear(); editors.clear(); identity = key; }
    const wasMissing = !snapshot?.proposal || snapshot.proposal.proposalId !== value.proposal?.proposalId;
    snapshot = value;
    const done = processed();
    if (wasMissing) for (const candidate of value.proposal?.topics.flatMap(topic => topic.directions) ?? []) if (!done.has(candidate.id) && !isThinEvidenceDirectionCandidate(candidate)) selected.add(candidate.id);
    const ids = new Set(value.proposal?.topics.flatMap(topic => topic.directions.map(candidate => candidate.id)) ?? []);
    for (const id of selected) if (!ids.has(id) || done.has(id)) selected.delete(id);
    for (const id of drafts.keys()) if (!ids.has(id) || done.has(id)) drafts.delete(id);
  }
  async function refresh(): Promise<void> {
    if (disposed || busy) return;
    const token = ++sequence; loading = !snapshot; error = ""; render();
    try {
      const value = await options.request<LibraryReviewViewSnapshot>("api/library/review");
      if (disposed || token !== sequence) return;
      if (dirtyAny() && value.proposal?.proposalId !== snapshot?.proposal?.proposalId) error = t("候选分析已被替换。当前修改仍保留，请先复制需要保留的内容，再放弃修改并刷新。");
      else apply(value);
    } catch (reason) { if (!disposed && token === sequence) error = t(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (!disposed && token === sequence) { loading = false; render(); } }
  }
  async function mutate(operation: string, payload: Record<string, unknown>, clear?: () => void): Promise<void> {
    if (!snapshot || busy || disposed) return;
    const token = ++sequence; busy = true; error = ""; confirmation = null; render();
    try {
      const value = await options.request<LibraryReviewViewSnapshot>("api/library/review", { operation, ...guards(), ...payload });
      if (disposed || token !== sequence) return;
      clear?.(); apply(value);
    } catch (reason) { if (!disposed && token === sequence) error = t(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (!disposed && token === sequence) { busy = false; render(); } }
  }
  function disclosureHtml(disclosure: LibraryAuthorizationDisclosure): string {
    return `<div class="review-disclosure">${[
      [t("Folder: {0}", disclosure.selectedRoot)],
      [t("Eligible files: {0}", disclosure.eligibleExtensions.join(", "))],
      [t("Processing depth: {0}", t("Titles and abstracts"))],
      [t("Model endpoint: {0}", disclosure.endpoint)],
      ...(disclosure.embeddingEndpoint ? [[t("Embedding endpoint: {0}", disclosure.embeddingEndpoint)]] : []),
    ].map(([line]) => `<p>${e(line!)}</p>`).join("")}</div>`;
  }
  async function requestJob(kind: "propose" | "preview", candidateId?: string): Promise<void> {
    if (!snapshot || disposed || busy || jobs.size) return;
    busy = true; error = ""; render();
    try {
      const inspection = await options.request<{ status: { kind: string }; disclosure: LibraryAuthorizationDisclosure | null }>("api/settings/library");
      if (disposed) return;
      busy = false;
      if (inspection.status.kind !== "authorized") {
        if (!inspection.disclosure) { error = t("请先在连接与索引中修正文献库或模型配置，再进行分析。"); render(); return; }
        const disclosure = inspection.disclosure;
        confirmation = {
          title: t("确认授权并分析文献库"), description: t("允许将所列范围内的标题和摘要发送给这些模型端点，然后运行本次分析。候选方向仍需另行审核接受。"), disclosure,
          execute: async () => {
            busy = true; error = ""; render();
            try {
              const value = await options.request<LibraryReviewViewSnapshot>("api/library/authorize", { configRevision: snapshot!.configRevision, fingerprint: disclosure.authorizationFingerprint });
              if (disposed) return;
              apply(value); busy = false; await startJob(kind, candidateId);
            } catch (reason) { if (!disposed) error = t(reason instanceof Error ? reason.message : String(reason)); }
            finally { if (!disposed) { busy = false; render(); } }
          },
        }; render(); return;
      }
      if (kind === "propose") { confirmation = { title: t("生成候选方向"), description: t("将使用已授权的模型分析索引证据。重新生成会替换当前候选，请先保存需要保留的修改。"), execute: () => startJob("propose") }; render(); }
      else await startJob(kind, candidateId);
    } catch (reason) { if (!disposed) error = t(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (!disposed) { busy = false; render(); } }
  }
  async function startJob(kind: "propose" | "preview", candidateId?: string): Promise<void> {
    if (!snapshot || busy || disposed || jobs.size) return;
    busy = true; error = ""; confirmation = null; render();
    try {
      const result = await options.request<{ run: WorkbenchRun }>(`api/library/${kind}`, { ...guards(), ...(candidateId ? { candidateId } : {}) });
      if (disposed) return;
      jobs.set(result.run.id, kind); options.onRun(result.run);
      if (result.run.status !== "running") { busy = false; await handleRun(result.run); }
    } catch (reason) { if (!disposed) error = t(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (!disposed) { busy = false; render(); } }
  }
  async function handleRun(run: WorkbenchRun): Promise<void> {
    if (!run || disposed || finishedJobs.has(run.id)) return;
    let kind = jobs.get(run.id);
    if (!kind) {
      kind = run.label === "生成候选方向" ? "propose" : run.label === "预览研究方向" ? "preview" : undefined;
      if (!kind) return;
      if (run.status === "running") jobs.set(run.id, kind);
      // The initial snapshot already includes runs completed before this view opened.
      else if (kind === "propose" && (!run.finishedAt || Date.parse(run.finishedAt) <= mountedAt)) { finishedJobs.add(run.id); return; }
    }
    if (run.status === "running") { updateDirtyControls(); return; }
    finishedJobs.add(run.id);
    jobs.delete(run.id);
    if (run.status !== "completed") { error = t("任务未完成，请在运行面板查看详情。修改仍保留。"); render(); return; }
    if (kind === "propose") { await refresh(); return; }
    const previewToken = ++previewSequence;
    try {
      const result = await options.request<{ preview: LibraryDirectionPreview | null }>(`api/library/preview?runId=${encodeURIComponent(run.id)}`);
      if (disposed || previewToken !== previewSequence) return;
      preview = result.preview; if (!preview) error = t("预览结果已不可用，请重新运行预览。");
    } catch (reason) { if (!disposed && previewToken === previewSequence) error = t(reason instanceof Error ? reason.message : String(reason)); }
    if (!disposed && previewToken === previewSequence) render();
  }
  function onInput(event: Event): void {
    event.stopPropagation();
    if (confirmation || busy) return;
    const input = event.target as HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement;
    const id = input.closest<HTMLElement>("[data-candidate]")?.dataset.candidate;
    const topicId = input.closest<HTMLElement>("[data-topic]")?.dataset.topic;
    if (input.dataset.field === "topic-name" && topicId) { names.set(topicId, input.value); updateDirtyControls(); return; }
    if (!id) return;
    const candidate = candidateById(id); if (!candidate) return;
    const value = { ...draft(candidate) };
    if (input.dataset.field === "representatives") value.representatives = Array.from((input as HTMLSelectElement).selectedOptions).map(option => option.value);
    else if (input.dataset.field === "text") value.text = input.value;
    else if (input.dataset.field === "cues") value.cues = input.value;
    else if (input.dataset.field === "destination") value.destination = input.value;
    else if (input.dataset.field === "name") value.name = input.value;
    else return;
    drafts.set(id, value);
    const row = input.closest<HTMLElement>("[data-candidate]")!;
    const previewButton = row.querySelector<HTMLButtonElement>('[data-review="preview"]'); if (previewButton) previewButton.disabled = dirty(candidate) || jobs.size > 0;
    row.querySelector<HTMLElement>(".review-dirty")!.hidden = !dirty(candidate);
    updateDirtyControls();
  }
  function updateDirtyControls(): void {
    const accept = root.querySelector<HTMLButtonElement>('[data-review="accept"]'); if (accept) accept.disabled = busy || dirtyAny() || selected.size === 0;
    const propose = root.querySelector<HTMLButtonElement>('[data-review="propose"]'); if (propose) propose.disabled = busy || dirtyAny() || jobs.size > 0;
  }
  function onChange(event: Event): void {
    event.stopPropagation();
    if (confirmation || busy) return;
    onInput(event);
    const input = event.target as HTMLInputElement;
    if (input.dataset.select) { if (input.checked) selected.add(input.dataset.select); else selected.delete(input.dataset.select); render(); }
    if (input.dataset.topicSelect) {
      const topic = snapshot?.proposal?.topics.find(item => item.id === input.dataset.topicSelect);
      for (const candidate of topic?.directions ?? []) if (!processed().has(candidate.id)) {
        if (input.checked && !isThinEvidenceDirectionCandidate(candidate)) selected.add(candidate.id); else selected.delete(candidate.id);
      }
      render();
    }
  }
  function onClick(event: MouseEvent): void {
    event.stopPropagation();
    const control = (event.target as Element).closest<HTMLElement>("[data-review]");
    const action = control?.dataset.review;
    const id = control?.closest<HTMLElement>("[data-candidate]")?.dataset.candidate;
    const topicId = control?.closest<HTMLElement>("[data-topic]")?.dataset.topic;
    if (!action || (confirmation && action !== "cancel" && action !== "confirm")) return;
    if (action === "settings") options.onSettings();
    else if (action === "library") options.onLibrary();
    else if (action === "refresh") void refresh();
    else if (action === "discard-refresh") { confirmation = { title: t("放弃修改并刷新"), description: t("放弃尚未保存的方向与主题名称修改，重新读取最新候选。"), execute: async () => { drafts.clear(); names.clear(); await refresh(); } }; render(); }
    else if (action === "proposed" || action === "overview") { tab = action; render(); }
    else if (action === "cancel") { confirmation = null; render(); }
    else if (action === "confirm") { const pending = confirmation; confirmation = null; void pending?.execute(); }
    else if (action === "close-preview") { preview = null; render(); }
    else if (action === "select-all" || action === "select-none") {
      selected.clear(); if (action === "select-all") for (const candidate of snapshot?.proposal?.topics.flatMap(topic => topic.directions) ?? []) if (!processed().has(candidate.id) && !isThinEvidenceDirectionCandidate(candidate)) selected.add(candidate.id); render();
    } else if (action === "accept" && snapshot?.proposal && selected.size && !dirtyAny()) {
      const candidateIds = [...selected], topicIds = snapshot.proposal.topics.filter(topic => topic.directions.some(candidate => selected.has(candidate.id))).map(topic => topic.id);
      confirmation = { title: t("接受所选方向"), description: t("将 {0} 个方向加入研究主题，之后参与每日发现。", selected.size), execute: () => mutate("accept-topics", { topicIds, candidateIds }) }; render();
    } else if (action === "rename" && topicId) void mutate("rename-topic", { topicId, suggestedName: names.get(topicId) ?? snapshot!.proposal!.topics.find(topic => topic.id === topicId)!.suggestedName }, () => names.delete(topicId));
    else if (action === "propose" && !dirtyAny()) void requestJob("propose");
    else if (id) {
      const candidate = candidateById(id); if (!candidate || processed().has(id)) return;
      if (action === "save") { const value = draft(candidate); void mutate("update-candidate", { candidateId: id, patch: { text: value.text, discoveryCues: value.cues.split("\n").map(cue => cue.trim()).filter(Boolean) }, representativePaperKeys: value.representatives }, () => drafts.delete(id)); }
      else if (action === "reset") { drafts.delete(id); render(); }
      else if (action === "preview" && !dirty(candidate)) void requestJob("preview", id);
      else if (action === "move") { const value = draft(candidate), target = snapshot!.topics.find(topic => topic.id === value.destination); if (dirty(candidate)) { error = t("请先保存修改，再预览或接受。"); render(); return; } void mutate("move-direction", { candidateId: id, targetTopicId: value.destination || null, suggestedName: target?.name ?? value.name }); }
      else if (action === "remove") { confirmation = { title: t("删除候选"), description: t("仅删除这条候选方向，不删除论文或已接受的研究方向。"), execute: () => mutate("remove-candidate", { candidateId: id }) }; render(); }
    }
  }
  function onKey(event: KeyboardEvent): void {
    event.stopPropagation();
    if (event.key === "Escape" && confirmation) { confirmation = null; render(); return; }
    if (!(event.target as Element).closest('[role="tab"]') || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault(); tab = event.key === "Home" ? "proposed" : event.key === "End" ? "overview" : tab === "proposed" ? "overview" : "proposed"; render(); root.querySelector<HTMLElement>(`[data-review="${tab}"]`)?.focus();
  }
  const runEvent = (event: Event) => { void handleRun((event as CustomEvent<WorkbenchRun>).detail); };
  root.addEventListener("click", onClick); root.addEventListener("input", onInput); root.addEventListener("change", onChange); root.addEventListener("keydown", onKey); root.addEventListener("workbench-run", runEvent);
  void refresh();
  return { refresh, handleRun, dispose() { disposed = true; ++sequence; root.removeEventListener("click", onClick); root.removeEventListener("input", onInput); root.removeEventListener("change", onChange); root.removeEventListener("keydown", onKey); root.removeEventListener("workbench-run", runEvent); } };
}
