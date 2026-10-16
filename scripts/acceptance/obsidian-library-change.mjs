/**
 * Real-Obsidian acceptance for the library change nudge (ADR 0015 sections 3-4) and for reading daily
 * reports that still carry withdrawn personal-novelty markers (ADR 0015 withdrawal).
 *
 * Run on its own, not as part of the default `obsidian` suite (which stays at its 8 scenarios):
 *
 *   node scripts/acceptance/obsidian-library-change.mjs --output /tmp/some/new-dir
 *
 * Isolation and fixed responses are the same as the other fixed suites: a disposable vault, the owned
 * loopback fixture for the business model, embeddings and arXiv metadata, and the real plugin/host. Papers
 * are real fixture PDFs on disk, scanned and indexed by the real pipeline. Clicks on the notice and the
 * review modal go through real mouse events, so a control that renders but cannot be hit fails.
 */
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { ACCEPTANCE_DATE, buildFixturePdf, createFixtureEnvironment } from "./fixtures.mjs";
import { runScenario, redact, buildReport, writeReport } from "./report.mjs";
import { createObsidianSettings } from "./obsidian.mjs";
import { createObsidianDriver, PLUGIN, poll } from "./obsidian-ui.mjs";
import { blockersFromError } from "../desktop-acceptance/app-state.mjs";
import { preflight } from "../desktop-acceptance/preflight.mjs";
import { runDesktopSession } from "../desktop-acceptance/session.mjs";
import { connectedLibraryFixture, installSettingsFixture, readRootIdentity } from "../desktop-acceptance/settings-fixture.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const LEGACY_REPORT = path.join(repoRoot, "packages/core/tests/fixtures/legacy-personal-novelty-report.md");
const LEGACY_DATE = "2026-06-10";
const NOTICES = 'Array.from(document.querySelectorAll(".notice")).map(item => item.textContent)';
const NUDGE_PATTERN = /new papers in your library aren't reflected in your direction suggestions yet\./;
const EMBEDDING_KEY = "fixture-embedding-key";

export const LIBRARY_CHANGE_SCENARIOS = [
  ["obsidian.library.change-nudge", "库新增论文后的提醒、复审按钮与只扫新增的更新"],
  ["obsidian.history.legacy-novelty", "含旧版个人新颖性标记的日报仍可在 Dashboard 读取"],
];

const BASELINE_EXTRA = { id: "9912.00099", title: "Repeated measurement protocols for controlled research", abstract: "This study repeats the same controlled measurement protocol to validate a reliable research measurement outcome." };
const FIRST_BATCH = [
  { id: "2611.20001", title: "Calibrating photometric pipelines for survey cosmology", abstract: "We calibrate photometric pipelines and quantify residual systematics in survey cosmology." },
  { id: "2611.20002", title: "Neural estimators for redshift distributions", abstract: "Neural estimators recover redshift distributions from broad-band photometry." },
  { id: "2611.20003", title: "Spectroscopic follow-up of faint galaxy samples", abstract: "Spectroscopic follow-up validates faint galaxy samples selected from imaging." },
  { id: "2611.20004", title: "Forward modelling of selection effects", abstract: "Forward modelling accounts for selection effects in galaxy surveys." },
];
const SECOND_BATCH = [
  { id: "2611.20005", title: "Stellar streams as probes of the halo", abstract: "Stellar streams constrain the shape of the galactic halo." },
  { id: "2611.20006", title: "Gravitational lensing mass maps", abstract: "Weak lensing mass maps reconstructed from shear catalogues." },
];

const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

export function createLibraryChangeSettings({ endpoint, fixtureOrigin, libraryRoot, rootIdentity }) {
  const base = createObsidianSettings({ endpoint });
  const library = connectedLibraryFixture({ libraryRoot, rootIdentity, embeddingMode: "remote" });
  return {
    ...base,
    settings: {
      ...base.settings,
      embedding: { ...library.settings.embedding, baseUrl: `${fixtureOrigin}/v1`, dimension: 3, apiKey: EMBEDDING_KEY },
    },
    libraryConnection: library.libraryConnection,
  };
}

async function writePapers(libraryRoot, papers) {
  await Promise.all(papers.map(paper => fs.writeFile(path.join(libraryRoot, `${paper.id}.pdf`), buildFixturePdf(paper))));
}

/** A real pointer click at the centre of the control, after proving that the control is what the pointer would hit. */
async function realClick(d, selector, { text } = {}) {
  const target = await poll(() => d.read(`(() => {
    const matches = Array.from(document.querySelectorAll(${JSON.stringify(selector)}))
      .filter(el => ${text === undefined ? "true" : `el.textContent.trim() === ${JSON.stringify(text)}`});
    const el = matches.find(candidate => candidate.getClientRects().length > 0);
    if (!el) return { ready: false, reason: "not found", count: matches.length };
    el.scrollIntoView({ block: "center" });
    const rect = el.getBoundingClientRect();
    const x = rect.x + rect.width / 2, y = rect.y + rect.height / 2;
    const hit = document.elementFromPoint(x, y);
    return { ready: !el.disabled && (hit === el || el.contains(hit)), disabled: el.disabled, x, y, hit: hit ? hit.tagName + "." + hit.className : null };
  })()`), value => value.ready, `a clickable ${selector}${text ? ` "${text}"` : ""}`, { timeoutMs: 8000, intervalMs: 100 });
  await d.session.client.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: target.x, y: target.y });
  await d.session.client.send("Input.dispatchMouseEvent", { type: "mousePressed", x: target.x, y: target.y, button: "left", clickCount: 1 });
  await d.session.client.send("Input.dispatchMouseEvent", { type: "mouseReleased", x: target.x, y: target.y, button: "left", clickCount: 1 });
}

async function shoot(d, name, selector, t) {
  await d.screenshots.capture(name, { selector });
  t.artifact(path.join(d.artifactDir, `${name}.png`));
}

const snapshot = d => d.read(`(() => { const s = ${PLUGIN}.getPersonalLibraryProfileSnapshot(); const p = s.proposal;
  return { newPaperKeys: s.newPaperKeys, proposalId: p?.proposalId ?? p?.id ?? null, revision: p?.revision ?? null,
    inputKeys: p ? p.catalogInputPapers.map(x => x.paperKey) : null, accountedKeys: p ? (p.accountedPaperKeys ?? null) : null,
    coveredKeys: p?.coveredPaperKeys ?? null, topics: p ? p.topics.map(topic => ({ name: topic.suggestedName ?? topic.name, targetTopicId: topic.targetTopicId ?? null, directions: topic.directions.map(dir => ({ text: dir.text, papers: (dir.representativePaperKeys ?? dir.paperKeys ?? []).length })) })) : null }; })()`);

// Only the "Index" command: no explicit scan. The index run itself notices that the folder changed and scans first.
async function indexByCommand(d) {
  await d.command("index-personal-library-fulltext");
  await d.wait(`${PLUGIN}.operations.snapshot()`, operations => operations.length === 0, "index to finish", { timeoutMs: 90000 });
}

async function changeNudge(d, t) {
  const { fixture } = d;
  const libraryRoot = fixture.libraryRoot;
  const model = () => fixture.server.requests.filter(request => request.prompt !== undefined && request.kind !== "embeddings");

  t.step("授权文献库处理，建立基线：扫描、索引并生成第一份方向建议（固定模型响应）");
  await d.evaluate(`(async () => { await ${PLUGIN}.authorizeLibraryProcessing(); return true; })()`);
  await d.evaluate(`(async () => { await ${PLUGIN}.indexPersonalLibraryFullText(); return true; })()`);
  await d.evaluate(`(async () => { await ${PLUGIN}.generatePersonalLibraryDirections(); return true; })()`);
  const baseline = await snapshot(d);
  t.check(baseline.inputKeys?.length === 3 && baseline.newPaperKeys.length === 0, "基线提案已存在，包含 3 篇论文，没有新增论文", baseline);
  const baselineKeys = new Set(baseline.accountedKeys ?? baseline.inputKeys);
  await d.openDashboard();

  t.step("往库里加入 4 篇新 PDF，只运行「索引」命令（不手动扫描；索引会先发现文件夹变了并先扫描）");
  fixture.server.addPapers(FIRST_BATCH);
  await writePapers(libraryRoot, FIRST_BATCH);
  await indexByCommand(d);
  const notices = await d.read(NOTICES);
  const nudges = notices.filter(text => NUDGE_PATTERN.test(text));
  const afterScan = await snapshot(d);
  t.check(afterScan.newPaperKeys.length === 4, "本地计数为 4 篇新增论文", afterScan.newPaperKeys);
  t.check(nudges.length === 1, "只出现一条提醒", notices);
  t.check(nudges[0].includes("4 new papers in your library aren't reflected in your direction suggestions yet."), "提醒文字与约定一致", nudges[0]);
  t.check(await d.read('Array.from(document.querySelectorAll(".notice button")).filter(b => b.textContent === "Update direction suggestions").length') === 1, "提醒带「Update direction suggestions」按钮");
  t.check(model().length === 1, "提醒与索引没有调用模型（模型调用仍只有基线那一次）", model().length);
  await shoot(d, "library-change-nudge-notice", ".notice-container", t);

  t.step("真实鼠标点击提醒上的按钮");
  await realClick(d, ".notice button", { text: "Update direction suggestions" });
  await d.wait('document.querySelectorAll(".arxiv-daily-interest-review__update-new").length', value => value === 1, "review modal with the update button", { timeoutMs: 10000 });
  const buttonText = await d.read('document.querySelector(".arxiv-daily-interest-review__update-new").textContent');
  t.check(buttonText === "Update direction suggestions (4 new papers)", "复审弹窗打开，主按钮显示 N", buttonText);
  const lineText = await d.read('document.querySelector(".arxiv-daily-interest-review__library-change")?.textContent ?? ""');
  t.check(lineText.includes("4 new papers in your library aren't reflected in your direction suggestions yet."), "弹窗里也显示同样的现状说明", lineText);
  t.check(await d.read('Array.from(document.querySelectorAll(".notice")).filter(n => /aren.t reflected/.test(n.textContent)).length') === 0, "点击后提醒已收起");
  await shoot(d, "library-change-review-modal", ".modal", t);

  t.step("点击更新按钮，确认替换未复核建议");
  const modelBefore = model().length;
  await realClick(d, ".arxiv-daily-interest-review__update-new");
  const confirmation = await d.wait('Array.from(document.querySelectorAll(".modal")).map(m => m.textContent)', value => value.some(text => /Update direction suggestions/.test(text) && /Cancel/.test(text) && /Update/.test(text) && !/Library overview/.test(text)), "confirmation dialog", { timeoutMs: 10000 }).catch(error => { throw error; });
  t.check(model().length === modelBefore, "确认之前没有调用模型", model().length - modelBefore);
  await shoot(d, "library-change-confirmation", ".modal", t);
  await fs.writeFile(path.join(d.artifactDir, "confirmation-text.json"), JSON.stringify(confirmation, null, 2));
  await realClick(d, ".modal button", { text: "Update" });
  await poll(() => snapshot(d), value => value.newPaperKeys.length === 0 && value.inputKeys?.length === 4, "the new-papers proposal to be saved", { timeoutMs: 60000, intervalMs: 300 });
  const updated = await snapshot(d);
  const newRequests = model().slice(modelBefore);
  t.check(newRequests.length === 1, "更新只调用了一次模型", newRequests.length);
  t.check(updated.inputKeys.every(key => !baselineKeys.has(key)) && updated.inputKeys.length === 4, "新提案只包含新增的 4 篇论文", updated.inputKeys);
  t.check(updated.accountedKeys && [...baselineKeys].every(key => updated.accountedKeys.includes(key)) && updated.accountedKeys.length === 7, "已计入集合带上了旧的 3 篇与新的 4 篇", updated.accountedKeys);
  const sent = newRequests[0].groupTitles ?? [];
  t.check(sent.length === 4 && FIRST_BATCH.every(paper => sent.some(title => title.includes(paper.title) || title.includes(paper.id))), "发给模型的只有这 4 篇新论文（来自请求记录）", sent);
  t.check(updated.coveredKeys?.length === 1 && updated.topics.length >= 1, "结果同时包含「已被现有方向覆盖」与新主题建议", updated);
  await d.wait('document.querySelectorAll(".arxiv-daily-interest-review__update-new").length', value => value === 0, "update button to leave a fully accounted library", { timeoutMs: 10000 });
  const afterUpdateText = await d.read('document.querySelector(".modal")?.textContent ?? ""');
  t.check(afterUpdateText.includes("New papers already covered by your existing directions: 1"), "更新后的弹窗显示「New papers already covered by your existing directions: 1」", afterUpdateText);
  t.check(await d.read('Array.from(document.querySelectorAll(".modal p")).filter(p => /already covered by your existing directions/.test(p.textContent) && !p.closest("details")).length') === 1, "这句话在主视图（不在折叠的 More options 里）");
  await shoot(d, "library-change-after-update", ".modal", t);
  await fs.writeFile(path.join(d.artifactDir, "after-update-modal-text.json"), JSON.stringify({
    modalText: await d.read('Array.from(document.querySelectorAll(".modal")).map(m => m.textContent)'),
    coverageEvidence: await d.read(`${PLUGIN}.getPersonalLibraryDirectionProposal()?.coverageEvidence ?? null`),
    topicsAndPapers: await d.read(`${PLUGIN}.getPersonalLibraryDirectionProposal()?.topics ?? null`),
  }, null, 2));
  await d.evaluate('(() => { document.querySelectorAll(".modal-close-button").forEach(button => button.click()); return true; })()');
  await d.wait('document.querySelectorAll(".modal").length', value => value === 0, "review modal to close", { timeoutMs: 8000 });

  t.step("文件夹没变再次点索引：没有新增论文，不应提醒");
  await indexByCommand(d);
  const quiet = await d.read(NOTICES);
  t.check(!quiet.some(text => NUDGE_PATTERN.test(text) || /isn't reflected/.test(text)), "再次索引没有提醒", quiet);

  t.step("再加 2 篇并只点索引：低于提醒条件，不提醒，但复审弹窗仍提供按钮");
  fixture.server.addPapers(SECOND_BATCH);
  await writePapers(libraryRoot, SECOND_BATCH);
  await indexByCommand(d);
  const two = await snapshot(d);
  const twoNotices = await d.read(NOTICES);
  t.check(two.newPaperKeys.length === 2, "本地计数为 2", two.newPaperKeys);
  t.check(!twoNotices.some(text => NUDGE_PATTERN.test(text)), "2 篇不弹提醒", twoNotices);
  await d.evaluate(`(() => { ${PLUGIN}.openPersonalLibraryDirectionReview(); return true; })()`);
  await d.wait('document.querySelectorAll(".arxiv-daily-interest-review__update-new").length', value => value === 1, "update button for two papers", { timeoutMs: 10000 });
  const twoText = await d.read('document.querySelector(".arxiv-daily-interest-review__update-new").textContent');
  t.check(twoText === "Update direction suggestions (2 new papers)", "N=2 时弹窗仍显示更新按钮", twoText);
  await shoot(d, "library-change-two-papers-modal", ".modal", t);
  await d.evaluate('(() => { document.querySelectorAll(".modal-close-button").forEach(button => button.click()); return true; })()');
}

async function legacyReports(d, t) {
  const { fixture } = d;
  const markdown = await fs.readFile(LEGACY_REPORT, "utf8");
  const target = path.join(fixture.vaultRoot, "arxiv-daily/daily", `${LEGACY_DATE}.md`);
  t.step("把含 v1/v2 个人新颖性标记与方向标记的旧日报放进 Vault 的日报目录");
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.writeFile(target, markdown);
  const diagnosticsStart = d.session.diagnostics.entries().length;
  await d.wait(`app.vault.getMarkdownFiles().some(file => file.path === "arxiv-daily/daily/${LEGACY_DATE}.md")`, value => value === true, "Obsidian to notice the new report file", { timeoutMs: 20000 });
  await d.openDashboard();
  await d.click('button[aria-label="Refresh dashboard"]');
  await d.wait('document.querySelectorAll(".arxiv-daily-dashboard__tab").length', value => value > 0, "dashboard to render after refresh", { timeoutMs: 15000 });
  await d.click('.arxiv-daily-dashboard__tab[data-tab="all"]');
  const rows = await d.wait('Array.from(document.querySelectorAll(".arxiv-daily-dashboard__table tbody tr")).map(row => row.textContent)', value => value.some(text => text.includes("2606.00001")) && value.some(text => text.includes("2606.00002")), "legacy report rows", { timeoutMs: 30000 });
  t.check(rows.some(text => text.includes("2606.00001")) && rows.some(text => text.includes("2606.00002")), "两篇旧日报论文都出现在 Dashboard 行里", rows);
  const rowHtml = await d.read('Array.from(document.querySelectorAll(".arxiv-daily-dashboard__table tbody tr")).filter(row => /2606\\.0000[12]/.test(row.textContent)).map(row => row.innerHTML)');
  await fs.writeFile(path.join(d.artifactDir, "legacy-rows.html.json"), JSON.stringify(rowHtml, null, 2));
  const pageText = await d.read('document.querySelector(".arxiv-daily-dashboard__table").textContent');
  t.check(!/novelty|新颖/i.test(pageText), "Dashboard 表格里没有新颖性内容", pageText);
  const index = await d.jsonFile("arxiv-daily/.index/papers.json");
  const entries = Object.values(index.papers).filter(entry => ["2606.00001", "2606.00002"].includes(entry.arxivId));
  t.check(entries.length === 2 && entries.every(entry => entry.dailyReports?.some(report => report.endsWith(`${LEGACY_DATE}.md`))), "论文索引记录了两篇旧日报论文及其日报路径", entries);
  t.check(entries.every(entry => !JSON.stringify(entry).toLowerCase().includes("novelty")), "索引条目里没有新颖性字段", entries);
  const chips = await d.read('Array.from(document.querySelectorAll(".arxiv-daily-dashboard__table tbody tr")).filter(row => /2606\\.0000[12]/.test(row.textContent)).map(row => ({ topic: row.querySelectorAll("td")[3]?.textContent ?? "", provenance: Array.from(row.querySelectorAll(".arxiv-daily-dashboard__provenance")).map(el => el.textContent) }))');
  await fs.writeFile(path.join(d.artifactDir, "legacy-provenance-chips.json"), JSON.stringify(chips, null, 2));
  t.check(chips.length === 2 && chips.every(row => row.topic === "photo-z" && row.provenance.some(text => /Manual topics: photo-z/.test(text)) && row.provenance.some(text => /Library directions/.test(text))), "两行都显示主题 photo-z 以及方向来源（手动主题与文献库方向）", chips);
  const bad = d.session.diagnostics.entries().slice(diagnosticsStart).filter(entry => /skipped invalid|novelty/i.test(entry.text ?? "") || ["error", "assert", "warning", "warn"].includes(entry.level) || entry.source === "pageerror");
  t.check(bad.length === 0, "控制台没有 skipped invalid、新颖性警告或错误", bad);
  const pluginLog = await d.read(`${PLUGIN}.logger.getBuffer()`);
  await fs.writeFile(path.join(d.artifactDir, "legacy-plugin-log.json"), JSON.stringify(pluginLog, null, 2));
  t.check(pluginLog.some(line => /dashboard: scanning/.test(line)), "插件日志里有这次历史同步的记录（日志确有被读取）", pluginLog.slice(-5));
  t.check(!pluginLog.some(line => /skipped invalid|novelty|\[WARN\]|\[ERROR\]/i.test(line)), "插件日志没有 skipped invalid、新颖性相关或任何 WARN/ERROR 行", pluginLog.filter(line => /skipped invalid|novelty|\[WARN\]|\[ERROR\]/i.test(line)));
  await shoot(d, "legacy-report-dashboard", ".workspace", t);
  t.step("打开旧日报阅读面板");
  await d.click('.arxiv-daily-dashboard__table tbody tr button[aria-label="Open daily report"]');
  const active = await d.wait('app.workspace.getActiveFile()?.path ?? null', value => value === `arxiv-daily/daily/${LEGACY_DATE}.md`, "legacy daily report view", { timeoutMs: 10000 });
  t.check(active === `arxiv-daily/daily/${LEGACY_DATE}.md`, "可以正常打开旧日报文件", active);
  await shoot(d, "legacy-report-reading", ".workspace", t);
  const logs = d.session.diagnostics.entries().slice(diagnosticsStart);
  await fs.writeFile(path.join(d.artifactDir, "legacy-console-entries.json"), JSON.stringify(redact(logs), null, 2));
}

export async function runLibraryChangeAcceptance({ fixture, artifactDir }, runtime = {}) {
  const sourceDir = path.join(repoRoot, "plugin");
  const obsidianPath = process.env.OBSIDIAN_BINARY ?? "/opt/Obsidian/obsidian";
  const environment = await (runtime.preflight ?? preflight)({ vaultPath: fixture.vaultRoot, obsidianPath, sourceDir });
  const stopped = status => ({ suite: "obsidian-library-change", status, errors: ["preflight blocked"], scenarios: [] });
  if (!environment.ok) return stopped("blocked");
  await fs.mkdir(artifactDir, { recursive: true });
  await writePapers(fixture.libraryRoot, [BASELINE_EXTRA]);
  fixture.server.addPapers([BASELINE_EXTRA]);
  const rootIdentity = await readRootIdentity(fixture.libraryRoot);
  const data = createLibraryChangeSettings({ endpoint: `${fixture.server.url}/v1`, fixtureOrigin: fixture.server.url, libraryRoot: fixture.libraryRoot, rootIdentity });
  try {
    return await (runtime.runDesktopSession ?? runDesktopSession)({
      vaultPath: fixture.vaultRoot, pluginId: "arxiv-daily", sourceDir, obsidianPath,
      beforeLaunch: ({ vaultPath, pluginId, fs: storage }) => installSettingsFixture({ vaultPath, pluginId, fs: storage, data }),
      body: async ({ session }) => {
        const d = await createObsidianDriver({ session, fixture, artifactDir });
        const scenarios = [];
        for (const [index, [id, title]] of LIBRARY_CHANGE_SCENARIOS.entries()) {
          scenarios.push(await runScenario({ id, title, artifactDir, captureFailure: () => d.screenshot(`${id.replaceAll(".", "-")}-failure`) },
            t => [changeNudge, legacyReports][index](d, t)));
        }
        const diagnosticsPath = path.join(artifactDir, "obsidian-diagnostics.json");
        const requestsPath = path.join(artifactDir, "obsidian-requests.json");
        await fs.writeFile(diagnosticsPath, JSON.stringify(redact({ complete: session.diagnosticsComplete, entries: session.diagnostics.entries() }), null, 2) + "\n");
        await fs.writeFile(requestsPath, JSON.stringify(redact(fixture.server.requests), null, 2) + "\n");
        return { suite: "obsidian-library-change", status: scenarios.every(s => s.status === "passed") ? "passed" : "failed", scenarios, artifacts: [diagnosticsPath, requestsPath], errors: [] };
      },
    });
  } catch (error) {
    const blockers = blockersFromError(error);
    return { suite: "obsidian-library-change", status: blockers ? "blocked" : "failed", errors: [error.stack ?? error.message], scenarios: [] };
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const flag = process.argv.indexOf("--output");
  const output = flag >= 0 ? path.resolve(process.argv[flag + 1]) : null;
  if (!output) { process.stderr.write("Usage: node scripts/acceptance/obsidian-library-change.mjs --output NEW_DIR\n"); process.exit(2); }
  const existing = await fs.readdir(output).catch(() => []);
  if (existing.length) { process.stderr.write("Output directory must be new or empty\n"); process.exit(2); }
  const fixture = await createFixtureEnvironment({ configured: true });
  let result;
  try {
    result = await runLibraryChangeAcceptance({ fixture, artifactDir: output });
    const unexpected = fixture.server.requests.filter(request => ["unexpected", "fixture-error"].includes(request.kind));
    if (unexpected.length) result.status = "failed";
    const report = buildReport({ selectedSuites: ["obsidian-library-change"], suites: [result], metadata: { startedAt: new Date().toISOString(), node: process.version, unexpectedFixtureRequests: unexpected.length } });
    await writeReport(output, report);
    process.stdout.write(`Library change acceptance ${report.status}: ${JSON.stringify(report.counts)}\nReport: ${path.join(output, "report.html")}\n`);
    for (const scenario of result.scenarios) if (scenario.status !== "passed") process.stdout.write(`${scenario.id}: ${scenario.error}\n`);
    process.exitCode = report.exitCode;
  } finally { await fixture.dispose(); }
}
