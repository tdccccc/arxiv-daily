import {
  isThinEvidenceDirectionCandidate, matchingProposalAcceptance, resolveProposedTopicTarget,
  type LibraryReviewSnapshot, type LibraryPreviewPaper, type LibraryDirectionPreview,
  type PersonalLibraryDirectionCandidate, type PersonalLibraryProposedTopic, type LibraryAuthorizationDisclosure,
} from "@arxiv-daily/core";
import { t } from "./i18n";
import { h, fragment } from "./dom";
import { scientificInline } from "./papers";
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
  const button = (action: string, source: string, disabled = false) => h("button", { type: "button", class: "quiet-button", "data-review": action, disabled: disabled || busy }, t(source));
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
  function paperLink(key: string): HTMLAnchorElement {
    const paper = snapshot?.catalog.papers[key] ?? snapshot?.indexedPapers.find(item => item.paperKey === key);
    return h("a", { href: `api/library/pdf?key=${encodeURIComponent(key)}`, target: "_blank", rel: "noopener noreferrer" }, scientificInline(paper?.title ?? key), " ↗");
  }
  const paperList = (keys: string[]) => h("ul", { class: "review-paper-links" }, ...keys.map(key => h("li", null, paperLink(key))));
  function previewSection(): HTMLElement | null {
    if (!preview) return null;
    return h("section", { class: "review-preview" },
      h("h2", null, t("方向预览")),
      h("p", null, scientificInline(preview.directionText)),
      h("p", { class: "review-help" }, t("预览仅检查文献库样本，不修改订阅，也不预测未来日报数量。")),
      preview.missingCategories.length ? h("p", null, `${t("匹配论文中的未订阅分类：")}${preview.missingCategories.join(", ")}`) : null,
      h("ul", null, ...preview.papers.map(paper => h("li", null,
        paperLink(paper.paperKey), " ",
        h("span", null, `${t(paper.matched ? "匹配" : "未匹配")} · ${t(paper.categoryCoverage === "inside" ? "已订阅分类" : paper.categoryCoverage === "outside" ? "未订阅分类" : "分类未知")}`),
        h("div", null, scientificInline(paper.title)),
      ))),
      button("close-preview", "关闭预览"),
    );
  }
  function candidateArticle(candidate: PersonalLibraryDirectionCandidate): HTMLElement {
    const value = draft(candidate), done = processed().has(candidate.id), thin = isThinEvidenceDirectionCandidate(candidate);
    const allowedKeys = snapshot!.proposal!.catalogInputPapers.map(item => item.paperKey);
    return h("article", { class: "review-candidate", "data-candidate": candidate.id },
      h("header", null,
        h("label", null,
          h("input", { type: "checkbox", "data-select": candidate.id, checked: selected.has(candidate.id), disabled: done || busy }),
          " ",
          h("span", null, scientificInline(candidate.text)),
        ),
        done ? h("span", { class: "review-badge" }, t("已处理")) : thin ? h("span", { class: "review-badge" }, t("证据较少")) : null,
      ),
      paperList(candidate.representatives.map(item => item.paperKey)),
      done
        ? h("p", { class: "review-help" }, t("已审核的方向请在研究主题设置中管理。"))
        : fragment(
            h("details", { class: "review-editor", open: editors.has(candidate.id) },
              h("summary", null, t("编辑方向与代表论文")),
              h("fieldset", { disabled: busy },
                h("label", null, t("方向文本"), h("textarea", { "data-field": "text", rows: 2, maxlength: 1000 }, value.text)),
                h("label", null, t("发现线索（每行一条）"), h("textarea", { "data-field": "cues", rows: 3 }, value.cues)),
                h("label", null, t("代表论文（选择 1–5 篇）"),
                  h("select", { multiple: true, size: Math.max(2, Math.min(5, allowedKeys.length)), "data-field": "representatives" },
                    ...allowedKeys.map(key => h("option", { value: key, selected: value.representatives.includes(key) }, snapshot!.catalog.papers[key]?.title ?? snapshot!.indexedPapers.find(paper => paper.paperKey === key)?.title ?? key)),
                  ),
                ),
                h("div", { class: "review-actions" }, button("save", "保存方向"), button("reset", "放弃此方向的修改")),
                h("div", { class: "review-move" },
                  h("label", null, t("移动到主题"),
                    h("select", { "data-field": "destination" },
                      h("option", { value: "" }, t("新主题")),
                      ...snapshot!.topics.map(topic => h("option", { value: topic.id, selected: value.destination === topic.id }, topic.name)),
                    ),
                  ),
                  h("label", null, t("新主题名称"), h("input", { "data-field": "name", value: value.name, maxlength: 120 })),
                  button("move", "移动方向"),
                ),
              ),
            ),
            h("div", { class: "review-actions" },
              button("preview", "预览方向", dirty(candidate) || jobs.size > 0),
              button("remove", "删除候选"),
              h("span", { class: "review-dirty", hidden: !dirty(candidate) }, t("请先保存修改，再预览或接受。")),
            ),
          ),
    );
  }
  function proposedView(): Node[] {
    const proposal = snapshot!.proposal;
    if (!proposal) return [h("div", { class: "review-empty" }, h("h2", null, t("尚无候选方向")), h("p", null, t("先建立文献库索引，再生成候选方向；审核接受后才参与每日发现。")))];
    if (!proposal.topics.length) return [h("div", { class: "review-empty" }, h("p", null, t("本次分析没有新增候选方向。请查看文献库概览。")))];
    return [
      h("div", { class: "review-selection" },
        button("select-all", "选择全部可审核方向"),
        button("select-none", "取消选择"),
        h("span", null, t("已选择 {0} 个方向", selected.size)),
        button("accept", "接受所选方向", selected.size === 0 || dirtyAny()),
      ),
      ...proposal.topics.map(topic => {
        const target = destination(topic), remaining = topic.directions.filter(item => !processed().has(item.id));
        return h("details", { class: "review-topic", "data-topic": topic.id, open: !collapsed.has(topic.id) },
          h("summary", null, h("strong", null, topic.suggestedName), " ", h("span", null, t("{0} 个方向", topic.directions.length))),
          h("div", { class: "review-topic-body" },
            h("div", { class: "review-topic-controls" },
              h("label", null,
                h("input", { type: "checkbox", "data-topic-select": topic.id, checked: remaining.length > 0 && remaining.every(item => selected.has(item.id)), disabled: !remaining.length || busy }),
                ` ${t("选择此主题")}`,
              ),
              target.error
                ? h("p", { role: "alert" }, target.error)
                : target.topic
                ? h("p", null, `${t("将加入现有主题：")}${target.topic.name}`)
                : fragment(
                    h("label", null, t("建议主题名称"), h("input", { "data-field": "topic-name", value: names.get(topic.id) ?? topic.suggestedName, maxlength: 120, disabled: busy })),
                    button("rename", "保存主题名称"),
                  ),
            ),
            ...topic.directions.map(candidateArticle),
          ),
        );
      }),
    ];
  }
  function overviewView(): HTMLElement {
    const proposal = snapshot!.proposal;
    if (!proposal) return h("p", { class: "review-empty" }, t("尚无文献库分析"));
    const analyzed = new Set(proposal.catalogInputPapers.map(item => item.paperKey));
    const represented = new Set(proposal.topics.flatMap(topic => topic.directions.flatMap(candidate => (candidate.clusterMembers?.length ? candidate.clusterMembers : candidate.representatives).map(item => item.paperKey))));
    const covered = new Set(proposal.coveredPaperKeys ?? []);
    const uncovered = [...analyzed].filter(key => !covered.has(key) && !represented.has(key));
    const added = snapshot!.indexedPapers.filter(paper => !analyzed.has(paper.paperKey)).map(paper => paper.paperKey);
    const coverage = (proposal.coverageEvidence ?? []).map(item => {
      const topic = snapshot!.topics.find(topic => topic.id === item.topicId);
      const current = topic?.directions.find(direction => direction.id === item.directionId)?.text === item.directionText;
      return h("li", null,
        h("strong", null, topic?.name ?? t("主题已删除")), " · ", scientificInline(item.directionText),
        h("p", null, t(current ? "当前方向已覆盖" : "方向已改变，需重新生成验证")),
        paperList(item.paperKeys),
      );
    });
    return h("div", { class: "review-overview" },
      h("p", null, t("分析了 {0} 篇论文", analyzed.size), " · ", h("time", { datetime: proposal.generatedAt }, proposal.generatedAt)),
      h("section", null,
        h("h2", null, t("已有方向覆盖")),
        coverage.length ? h("ul", null, ...coverage) : h("p", null, t("暂无已确认的覆盖记录")),
        proposal.coverageEvidence === undefined && covered.size ? h("p", null, t("历史覆盖尚未验证，请重新生成分析。")) : null,
      ),
      h("section", null,
        h("h2", null, t("候选方向与代表论文")),
        ...proposal.topics.map(topic => h("details", { open: true },
          h("summary", null, topic.suggestedName),
          ...topic.directions.map(candidate => h("div", null,
            h("h3", null, scientificInline(candidate.text), processed().has(candidate.id) ? h("small", null, t("已处理")) : null),
            paperList(candidate.representatives.map(item => item.paperKey)),
          )),
        )),
      ),
      h("section", null, h("h2", null, t("暂未归入方向的论文")), paperList(uncovered), !uncovered.length ? h("p", null, t("暂无")) : null),
      h("section", null, h("h2", null, t("分析之后新增的论文")), paperList(added), !added.length ? h("p", null, t("暂无")) : null),
    );
  }
  function disclosureSection(disclosure: LibraryAuthorizationDisclosure): HTMLElement {
    const lines = [
      t("Folder: {0}", disclosure.selectedRoot),
      t("Eligible files: {0}", disclosure.eligibleExtensions.join(", ")),
      t("Processing depth: {0}", t("Titles and abstracts")),
      t("Model endpoint: {0}", disclosure.endpoint),
      ...(disclosure.embeddingEndpoint ? [t("Embedding endpoint: {0}", disclosure.embeddingEndpoint)] : []),
    ];
    return h("div", { class: "review-disclosure" }, ...lines.map(line => h("p", null, line)));
  }
  function render(): void {
    if (disposed) return;
    const feedback = h("div", { class: "review-feedback", "aria-live": "polite" },
      error ? h("div", { role: "alert" }, error, " ", button("refresh", "重新加载"), dirtyAny() ? button("discard-refresh", "放弃修改并刷新") : null) : null,
      busy ? h("p", { role: "status" }, t("正在保存…")) : null,
    );
    const body = loading
      ? h("p", { role: "status" }, t("正在加载方向审核…"))
      : !snapshot
      ? h("p", null, t("方向审核加载失败"))
      : !snapshot.connected
      ? h("div", { class: "review-empty" }, h("h2", null, t("尚未连接个人文献库")), h("p", null, t("先在连接与索引中选择文献目录。")))
      : fragment(
          h("div", { class: "review-topbar" },
            h("div", { role: "tablist", "aria-label": t("文献库分析视图") },
              h("button", { role: "tab", "data-review": "proposed", "aria-selected": String(tab === "proposed"), tabindex: tab === "proposed" ? 0 : -1 }, t("候选方向")),
              h("button", { role: "tab", "data-review": "overview", "aria-selected": String(tab === "overview"), tabindex: tab === "overview" ? 0 : -1 }, t("文献库概览")),
            ),
            button("propose", snapshot.proposal ? "重新生成候选" : "生成候选方向", jobs.size > 0 || dirtyAny()),
          ),
          h("div", { role: "tabpanel" }, ...(tab === "proposed" ? proposedView() : [overviewView()])),
          previewSection(),
        );
    root.replaceChildren(
      h("section", { class: "library-review-workspace" },
        h("header", { class: "review-heading" },
          h("div", null, h("h1", null, t("方向审核")), h("p", null, t("检查候选方向及其依据；接受后保存到普通研究主题。"))),
          h("div", { class: "review-actions" }, button("library", "返回文献库"), button("settings", "连接与索引"), button("refresh", "刷新")),
        ),
        feedback,
        body,
        confirmation
          ? h("section", { class: "review-confirmation", role: "group", "data-confirmation": true, "aria-label": confirmation.title },
              h("h2", null, confirmation.title),
              h("p", null, confirmation.description),
              confirmation.disclosure ? disclosureSection(confirmation.disclosure) : null,
              h("div", { class: "review-actions" }, button("confirm", "确认"), button("cancel", "取消")),
            )
          : null,
      ),
    );
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
