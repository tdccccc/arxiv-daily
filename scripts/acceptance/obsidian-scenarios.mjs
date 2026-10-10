import fs from "node:fs/promises";
import path from "node:path";
import { ACCEPTANCE_DATE, PAPERS } from "./fixtures.mjs";
import { runScenario, redact } from "./report.mjs";
import { OBSIDIAN_SCENARIOS, classifyObsidianDiagnostics } from "./obsidian.mjs";
import { createObsidianDriver, PLUGIN, poll, pollJsonFile } from "./obsidian-ui.mjs";
import { pdfPageLocationScenario } from "../desktop-acceptance/scenarios.mjs";

const SETTINGS = ".obsidian/plugins/arxiv-daily/data.json";
const STATE = "arxiv-daily/.index/run-state.json";
const HISTORY = "arxiv-daily/.index/run-history.jsonl";
const INDEX = "arxiv-daily/.index/papers.json";
const DAILY = `arxiv-daily/daily/${ACCEPTANCE_DATE}.md`;
const RUN_STATE = `${PLUGIN}.stateStore.get(${JSON.stringify(ACCEPTANCE_DATE)})`;
const NOTICES = 'Array.from(document.querySelectorAll(".notice")).map(item => item.textContent)';

const savedModel = async (d, model) => pollJsonFile(() => d.jsonFile(SETTINGS), data => data.settings.llm.model === model, "model setting on disk");
const savedKey = async (d, key) => pollJsonFile(() => d.jsonFile(SETTINGS), data => data.settings.llm.apiKey === key, "API key setting on disk");
const entryFor = (index, id) => Object.values(index.papers).find(entry => entry.arxivId === id);
const recordFile = async (d, source, name, t) => {
  const target = path.join(d.artifactDir, name);
  await fs.writeFile(target, await d.file(source));
  t.artifact(target);
};

async function settingsPersistence(d, t) {
  t.step("打开 Settings → arXiv Daily，检查未完成的新手引导");
  await d.openSettings();
  t.check(await d.read('Boolean(document.querySelector(".arxiv-daily-setup"))'), "首次使用引导可见");
  t.step("在真实输入框清空 API key，确认不完整配置阻止启动");
  await d.fill('input[aria-label="LLM API key"]', "");
  await savedKey(d, "");
  const requestCount = d.fixture.server.requests.length;
  await d.command("run-for-date");
  const notices = await d.wait(NOTICES, value => value.some(text => /cannot run/i.test(text)), "configuration refusal notice");
  t.check(notices.some(text => /cannot run/i.test(text)), "配置不完整时显示无法运行原因", notices);
  t.check(await d.read('document.querySelectorAll(".modal input[type=date]").length') === 0, "未绕过配置校验进入日期运行");
  t.check(d.fixture.server.requests.length === requestCount, "配置不完整时没有外部请求");

  t.step("填写测试 key，将模型改为 fixture-model-2，等待真实设置文件保存");
  await d.fill('input[aria-label="LLM API key"]', "acceptance-stub-key");
  await savedKey(d, "acceptance-stub-key");
  await d.fill('input[aria-label="Model"]', "fixture-model-2");
  const persisted = await savedModel(d, "fixture-model-2");
  t.check(persisted.settings.llm.model === "fixture-model-2", "新模型已写入 plugin data.json");
  await d.screenshot("obsidian-settings-saved", t);
  t.step("重新加载插件，再打开设置");
  await d.reloadPlugin();
  await d.openSettings();
  t.check(await d.read('document.querySelector("input[aria-label=Model]").value') === "fixture-model-2", "重新加载后界面恢复保存的模型");
  t.check(await d.read(`${PLUGIN}.settings.llm.model`) === "fixture-model-2", "重新加载后的运行配置一致");
  t.check(await d.read(`${PLUGIN}.settings.schedule.enabled`) === false, "测试环境仍未启动定时任务");
  await d.closeSettings();
}

async function cancelAndRepeat(d, t) {
  d.fixture.server.setMode({ llm: "hold" });
  const start = d.fixture.server.requests.length;
  try {
    t.step(`通过 Run for date 日期对话框启动 ${ACCEPTANCE_DATE}；测试服务暂挂模型请求`);
    await d.runDate(ACCEPTANCE_DATE);
    await poll(() => d.fixture.server.requests.slice(start), requests => requests.some(request => request.held), "the real model request to reach the hold gate");
    const running = await d.read(RUN_STATE);
    t.check(running.status === "running", "真实调度状态进入 running", running);
    const onDisk = await d.jsonFile(STATE);
    t.check(onDisk.runState[ACCEPTANCE_DATE].status === "running", "运行状态已落盘");
    await d.openDashboard();
    t.check(/Running/.test(await d.read('document.querySelector(".arxiv-daily-dashboard__status-line").textContent')), "工作台显示正在运行");
    t.check(/filter/.test(await d.read('document.querySelector(".arxiv-daily-progress")?.textContent ?? ""')), "真实流程显示筛选进度");
    await d.screenshot("obsidian-run-in-progress", t);

    t.step("在前一次任务运行中，再从日期对话框启动同一天");
    await d.runDate(ACCEPTANCE_DATE);
    const notices = await d.wait(NOTICES, value => value.some(text => /lock held|already running/i.test(text)), "duplicate run refusal");
    t.check(notices.some(text => /lock held|already running/i.test(text)), "重复启动有可见反馈", notices);
    const active = await d.wait(`${PLUGIN}.scheduler.activeRuns()`, value => value.length === 1, "one active date");
    t.check(active[0] === ACCEPTANCE_DATE, "同一日期仍只有一次业务运行", active);
    t.check(d.fixture.server.requests.slice(start).filter(request => request.held).length === 1, "重复操作没有发起第二次模型调用");

    t.step("执行 Cancel active tasks，观察任务结束与可重试状态");
    await d.command("cancel-current-run");
    await d.idle();
    const cancelled = await d.read(RUN_STATE);
    t.check(cancelled.status === "pending", "取消后日期回到可重试状态", cancelled);
    t.check((await d.jsonFile(STATE)).runState[ACCEPTANCE_DATE].status === "pending", "磁盘也已清除 running 状态");
    t.check(!(await d.exists(DAILY)), "取消不会留下半份权威日报");
    const history = (await d.file(HISTORY)).trim().split("\n").map(line => JSON.parse(line));
    t.check(history.some(record => record.resultKind === "cancelled"), "运行历史记下真实取消结果");
    await d.screenshot("obsidian-run-cancelled", t);
  } finally {
    d.fixture.server.releaseHeld();
    d.fixture.server.setMode({ llm: "normal" });
  }
}

async function authorizationError(d, t) {
  d.fixture.server.setMode({ llm: "unauthorized" });
  const start = d.fixture.server.requests.length;
  try {
    t.step("在取消后的同一日期重新运行，让测试模型端点返回 HTTP 401");
    await d.runDate(ACCEPTANCE_DATE);
    const state = await d.wait(RUN_STATE, value => value.status === "failed_permanent", "permanent authentication failure", { timeoutMs: 45000 });
    await d.idle();
    t.check(d.fixture.server.requests.slice(start).some(request => request.method === "POST" && request.kind === "model" && request.status === 401), "真实模型 HTTP 请求收到受控 401");
    t.check(state.status === "failed_permanent" && (state.error ?? "").includes("Controlled acceptance authentication failure"), "失败日期保留可诊断的认证错误", state);
    t.check((await d.jsonFile(STATE)).runState[ACCEPTANCE_DATE].status === "failed_permanent", "失败状态真实写盘");
    const notices = await d.wait(NOTICES, value => value.some(text => text.includes("Controlled acceptance authentication failure")), "authentication error feedback");
    t.check(notices.some(text => text.includes("Controlled acceptance authentication failure")), "界面提示本次认证失败", notices);
    t.check(!(await d.exists(DAILY)), "认证失败不生成空白成功日报");
    await d.screenshot("obsidian-run-authentication-error", t);
  } finally {
    d.fixture.server.setMode({ llm: "normal" });
  }
}

async function retryAndComplete(d, t) {
  const start = d.fixture.server.requests.length;
  t.step("将测试端点恢复正常，通过 Force run for date 重试已永久失败的固定日期");
  // The fixed date must remain reproducible after the rolling five-day retry window moves on.
  await d.runDate(ACCEPTANCE_DATE, { force: true });
  const completed = await d.wait(RUN_STATE, value => value.status === "completed", "daily report completion", { timeoutMs: 90000 });
  await d.idle();
  t.check(completed.status === "completed", "重新运行成功完成", completed);
  const persisted = await d.jsonFile(STATE);
  t.check(persisted.runState[ACCEPTANCE_DATE].status === "completed", "成功状态已经持久化");
  t.check(persisted.runState[ACCEPTANCE_DATE].papersWritten === PAPERS.length, "状态文件论文数量与夹具一致");
  const markdown = await d.file(DAILY);
  t.check(PAPERS.every(paper => markdown.includes(paper.id) && markdown.includes(paper.title)), "日报文件含两篇实际测试论文");
  const index = await d.jsonFile(INDEX);
  t.check(Object.keys(index.papers).length === PAPERS.length, "真实 paper index 没有缺项或重复项");
  t.check(PAPERS.every(paper => entryFor(index, paper.id)?.dailyReports.includes(DAILY)), "索引中的日报链接指向已写出的文件");
  const requests = d.fixture.server.requests.slice(start);
  t.check(requests.some(request => request.kind === "filter") && requests.filter(request => request.kind === "summary").length === PAPERS.length, "筛选及逐篇摘要均通过真实 pipeline 到达外部边界");
  t.check(requests.filter(request => request.model).every(request => request.model === "fixture-model-2"), "实际运行使用界面保存的新模型");
  const history = (await d.file(HISTORY)).trim().split("\n").map(line => JSON.parse(line));
  t.check(history.some(record => record.date === ACCEPTANCE_DATE && record.event === "completed" && record.trigger === "force"), "运行历史保存完成记录和重试入口");
  await d.openDashboard();
  await d.click('.arxiv-daily-dashboard__tab[data-tab="all"]');
  await d.wait('document.querySelectorAll(".arxiv-daily-dashboard__table tbody tr").length', value => value === PAPERS.length, "generated papers in dashboard");
  t.check(await d.read('document.querySelectorAll(".arxiv-daily-dashboard__table tbody tr").length') === PAPERS.length, "界面列出已生成的论文");
  await d.screenshot("obsidian-daily-completed", t);
  await recordFile(d, DAILY, "obsidian-generated-daily.md", t);
  await recordFile(d, STATE, "obsidian-completed-run-state.json", t);
  await recordFile(d, HISTORY, "obsidian-run-history.jsonl", t);
}

async function completedIdempotency(d, t) {
  const before = await d.file(DAILY);
  const start = d.fixture.server.requests.length;
  t.step("再次运行相同已完成日期");
  await d.runDate(ACCEPTANCE_DATE);
  await d.wait(NOTICES, value => value.some(text => /already done/i.test(text)), "already completed feedback");
  await d.idle();
  t.check(await d.file(DAILY) === before, "已有日报逐字保持不变");
  t.check(!d.fixture.server.requests.slice(start).some(request => request.model), "没有再次请求模型");
  t.check(Object.keys((await d.jsonFile(INDEX)).papers).length === PAPERS.length, "索引没有产生重复论文");
}

async function readingAndReload(d, t) {
  t.step("从 Dashboard 打开真实日报");
  await d.openDashboard();
  await d.click('.arxiv-daily-dashboard__tab[data-tab="all"]');
  await d.wait('document.querySelectorAll(".arxiv-daily-dashboard__table tbody tr").length', value => value === PAPERS.length, "dashboard paper rows");
  await d.click('.arxiv-daily-dashboard__table tbody tr button[aria-label="Open daily report"]');
  const activeFile = await d.wait('app.workspace.getActiveFile()?.path ?? null', value => value === DAILY, "daily Markdown view");
  t.check(activeFile === DAILY, "打开的是对应日期的日报文件");
  await d.screenshot("obsidian-reading-daily", t);

  t.step("返回 Dashboard，收藏第一篇论文，再从 Starred 列表查看");
  await d.openDashboard();
  await d.click('.arxiv-daily-dashboard__tab[data-tab="all"]');
  const star = `.arxiv-daily-dashboard__star[data-arxiv-id="${PAPERS[0].id}"]`;
  await d.click(star);
  const marked = await pollJsonFile(() => d.jsonFile(INDEX), value => entryFor(value, PAPERS[0].id)?.priority === "high", "starred priority on disk");
  t.check(entryFor(marked, PAPERS[0].id).priority === "high", "收藏标记真实写入索引");
  await d.click('.arxiv-daily-dashboard__tab[data-tab="starred"]');
  const titles = await d.wait('Array.from(document.querySelectorAll(".arxiv-daily-dashboard__table tbody tr")).map(row => row.textContent)', value => value.length === 1, "starred paper filter");
  t.check(titles[0].includes(PAPERS[0].id), "Starred 只显示收藏的论文", titles);
  await d.screenshot("obsidian-paper-starred", t);

  t.step("重新加载插件并重新打开 Dashboard");
  await d.reloadPlugin();
  await d.openDashboard();
  await d.click('.arxiv-daily-dashboard__tab[data-tab="starred"]');
  const restored = await d.wait('Array.from(document.querySelectorAll(".arxiv-daily-dashboard__table tbody tr")).map(row => row.textContent)', value => value.length === 1, "restored starred paper");
  t.check(restored[0].includes(PAPERS[0].id), "重新加载后收藏与过滤结果仍保留", restored);
  t.check((await d.read(RUN_STATE)).status === "completed", "重新加载后日报完成状态仍保留");
  t.check(await d.read(`${PLUGIN}.settings.llm.model`) === "fixture-model-2", "设置和阅读状态同时恢复");
  await recordFile(d, INDEX, "obsidian-paper-index.json", t);
  await d.screenshot("obsidian-reading-restored", t);
}

export async function runObsidianJourneys({ session, fixture, artifactDir }) {
  const d = await createObsidianDriver({ session, fixture, artifactDir });
  const scenarios = [], expectedErrors = new Set();
  const actions = [settingsPersistence, cancelAndRepeat, authorizationError, retryAndComplete, completedIdempotency, readingAndReload,
    async (driver, t) => {
      t.step("从自有文献库打开 PDF #page=4，读取真实 viewer 页码");
      const result = await pdfPageLocationScenario({ session: driver.session });
      t.check(result.passed, "PDF 页面定位准确", result.detail);
      await driver.screenshot("obsidian-pdf-page-four", t);
    },
  ];
  let previousFailed = false;
  for (const [index, [id, title]] of OBSIDIAN_SCENARIOS.entries()) {
    if (previousFailed && id !== "obsidian.pdf.location") {
      scenarios.push({ id, title, status: "blocked", durationMs: 0, assertions: [], steps: [], artifacts: [], error: "前置流程未通过，本项未执行" });
      continue;
    }
    const diagnosticStart = session.diagnostics.entries().length;
    const requestStart = fixture.server.requests.length;
    const scenario = await runScenario({
      id, title, artifactDir,
      captureFailure: () => d.screenshot(`${id.replaceAll(".", "-")}-failure`),
    }, async t => {
      await actions[index](d, t);
      const diagnostics = classifyObsidianDiagnostics(session.diagnostics.entries().slice(diagnosticStart), {
        authorizationFailureExpected: id === "obsidian.run.authorization-error",
        fixtureRequests: fixture.server.requests.slice(requestStart),
      });
      for (const error of diagnostics.expected) expectedErrors.add(error);
      t.check(diagnostics.unexpected.length === 0, "本场景没有意外 console error 或未捕获异常", diagnostics.unexpected);
    });
    scenarios.push(scenario);
    if (scenario.status !== "passed") {
      previousFailed = true;
      fixture.server.releaseHeld();
      await d.evaluate(`(() => { ${PLUGIN}.operations.cancelAll("acceptance cleanup"); return true; })()`).catch(() => undefined);
      await d.idle().catch(() => undefined);
    }
  }
  const errors = session.diagnostics.errors().filter(error => !expectedErrors.has(error));
  scenarios.push(await runScenario({ id: "obsidian.diagnostics", title: "宿主诊断完整且不存在意外错误", artifactDir }, async t => {
    t.check(session.diagnosticsComplete === true, "诊断在插件启动前已经开始收集");
    t.check(errors.length === 0, "全程没有遗漏的意外错误", errors);
    t.check(!fixture.server.requests.some(request => ["unexpected", "fixture-error"].includes(request.kind)), "所有业务外部请求均命中明确夹具", fixture.server.requests.filter(request => ["unexpected", "fixture-error"].includes(request.kind)));
  }));
  const diagnosticsPath = path.join(artifactDir, "obsidian-diagnostics.json");
  const requestsPath = path.join(artifactDir, "obsidian-requests.json");
  await fs.writeFile(diagnosticsPath, JSON.stringify(redact({ version: session.pluginVersion, complete: session.diagnosticsComplete, entries: session.diagnostics.entries(), expectedErrors: [...expectedErrors] }), null, 2) + "\n");
  await fs.writeFile(requestsPath, JSON.stringify(redact(fixture.server.requests), null, 2) + "\n");
  return { suite: "obsidian", status: scenarios.some(scenario => scenario.status === "failed") ? "failed" : scenarios.some(scenario => scenario.status !== "passed") ? "blocked" : "passed", scenarios, artifacts: [diagnosticsPath, requestsPath, d.interactionTracePath], errors };
}
