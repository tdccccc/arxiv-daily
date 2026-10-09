import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { ACCEPTANCE_DATE, PAPERS, createFixtureEnvironment } from "./fixtures.mjs";
import { BlockedError, buildReport, runScenario, writeReport } from "./report.mjs";
import { startWorkbenchSession } from "./workbench-session.mjs";
export { startWorkbenchSession } from "./workbench-session.mjs";

const require = createRequire(new URL("../../apps/cli/package.json", import.meta.url));
const { parse: parseToml } = require("smol-toml");
const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const CANCEL_DATE = "2026-10-02", FAILURE_DATE = "2026-10-05";
const DIRECTION = "Reliable research methods and controlled measurements";
const MODEL = "fixture-model-2";
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

export const WORKBENCH_SCENARIOS = [
  { id: "workbench.startup", title: "真实工作台启动且浏览不触发生成" },
  { id: "workbench.first-use", title: "首次使用可保存隔离目录草稿并退出" },
  { id: "workbench.settings", title: "设置自动保存、模型列表及密钥保留" },
  { id: "workbench.daily-generation", title: "生成日报与磁盘、索引一致且重复执行幂等" },
  { id: "workbench.reading", title: "阅读、收藏、筛选与返回导航" },
  { id: "workbench.paper-note", title: "单篇详细总结保存后就地可读" },
  { id: "workbench.cancel-retry", title: "重复启动与运行中保存防护、取消后重跑" },
  { id: "workbench.failure-retry", title: "401 失败清楚呈现，更正后可重试" },
  { id: "workbench.settings-conflict", title: "外部配置冲突保留草稿并可放弃重开" },
  { id: "workbench.library", title: "文献库连接与浏览、方向审核入口" },
  { id: "workbench.restart", title: "重启保留设置、阅读标记与界面偏好" },
  { id: "workbench.browser-errors", title: "所有实际场景没有未捕获的页面异常" },
];

async function exists(path) { try { await access(path); return true; } catch (error) { if (error.code === "ENOENT") return false; throw error; } }
const readJson = async path => JSON.parse(await readFile(path, "utf8"));
const readConfig = async fixture => parseToml(await readFile(fixture.configPath, "utf8"));

async function until(check, label, { timeout = 45000, signal } = {}) {
  const deadline = Date.now() + timeout;
  let latest;
  while (Date.now() < deadline) {
    signal?.throwIfAborted();
    try { latest = await check(); if (latest) return latest; }
    catch (error) { if (error.code !== "ENOENT") throw error; }
    await pause(80);
  }
  throw new Error("Timed out: " + label + (latest === undefined ? "" : " (last result: " + String(latest) + ")"));
}

async function api(session, route) {
  const response = await fetch(new URL(route, session.url), { signal: AbortSignal.timeout(10000) });
  if (!response.ok) throw new Error("Workbench read " + route + " failed: HTTP " + response.status);
  return response.json();
}

async function closeDialog(page) {
  const dialog = page.locator("dialog[open]");
  if (await dialog.count()) {
    await dialog.locator('[data-action="close-dialog"]').first().click();
    await dialog.waitFor({ state: "hidden" });
  }
}

async function openSettings(page) {
  await closeDialog(page);
  await page.locator('[data-action="settings"]').first().click();
  await page.locator(".settings-form").waitFor();
}

async function discardSettings(page) {
  await page.locator('[data-settings="discard-close"]').click();
  await page.locator('[data-settings="confirm-discard-close"]').click();
  await page.locator(".settings-form").waitFor({ state: "hidden" });
}

async function editField(page, name, value) {
  const field = page.locator('.settings-form [name="' + name + '"]').first();
  await field.fill(value);
  await field.press("Tab");
}

async function startDaily(session, date) {
  const page = session.page;
  await closeDialog(page);
  await page.locator('[data-action="generate"]').click();
  await page.locator("#run-date").fill(date);
  const response = page.waitForResponse(response => response.url().endsWith("/api/runs") && response.request().method() === "POST");
  await page.locator('.generation-form button[type="submit"]').click();
  const started = await response, body = await started.json();
  if (started.status() !== 202) throw new Error("Daily launch failed: " + JSON.stringify(body));
  await page.locator(".generation-form").waitFor({ state: "hidden" });
  return body.run;
}

export async function waitForWorkbenchRun(session, run, signal) {
  const result = await until(async () => {
    const current = (await api(session, "api/runs/current")).run;
    return current?.id === run.id && current.status !== "running" ? current : false;
  }, "run " + run.id + " finishes", { timeout: 90000, signal });
  const labels = { completed: "已完成", failed: "生成失败", cancelled: "已取消", skipped: "已跳过", pending: "等待公告发布" };
  const outcomes = { papers_written: "日报已保存", no_matches: "无匹配论文", no_updates: "当日无更新", awaiting_announcement: "等待公告发布" };
  const label = ["completed", "pending"].includes(result.status) && outcomes[result.outcome] || labels[result.status];
  await until(async () => {
    const state = session.page.locator(".run-tray .run-state");
    return await state.isVisible() && (await state.getAttribute("class") ?? "").split(/\s+/).includes(result.status)
      && (await state.innerText()).includes(label);
  }, "UI displays confirmed " + result.status + " / " + label, { timeout: 8000, signal });
  return result;
}

async function pathsFor(fixture) {
  const config = await readConfig(fixture);
  const output = config.output;
  return {
    daily: date => join(config.vault_root, output.daily_dir, date + ".md"),
    index: join(config.vault_root, dirname(output.daily_dir), ".index", "papers.json"),
    state: join(config.vault_root, dirname(output.daily_dir), ".index", "run-state.json"),
    history: join(config.vault_root, dirname(output.daily_dir), ".index", "run-history.jsonl"),
  };
}

async function ensureDaily(t, fixture, date) {
  const paths = await pathsFor(fixture);
  const markdown = await readFile(paths.daily(date), "utf8");
  const index = await readJson(paths.index), state = await readJson(paths.state);
  t.check(PAPERS.every(paper => markdown.includes(paper.id)), "日报文件包含实际选入的两篇论文");
  t.check(state.runState[date]?.status === "completed", "持久化运行状态为 completed", state.runState[date]);
  const expected = (await readConfig(fixture)).output.daily_dir + "/" + date + ".md";
  t.check(PAPERS.every(paper => index.papers["arxiv:" + paper.id]?.dailyReports.includes(expected)), "两篇索引条目都指向真实日报");
  return { paths, markdown, index, state };
}

/** Fixed real-user journeys. External protocols are controlled by the shared fixture. */
export async function runWorkbenchAcceptance({ fixture, artifactDir, signal, onScenario, headless = true, executablePath } = {}) {
  await mkdir(artifactDir, { recursive: true });
  const scenarios = [], sessions = [], artifacts = [], errors = [];
  let session, sequence = 0, firstMarkdown = "", starKey = "arxiv:" + PAPERS[0].id;
  const start = async () => {
    try {
      const value = await startWorkbenchSession({ fixture, artifactDir: join(artifactDir, "session-" + ++sequence), signal, headless, executablePath });
      sessions.push(value); artifacts.push(...value.artifacts); return value;
    } catch (error) {
      artifacts.push(...(error.artifacts ?? []));
      if (error.code === "WORKBENCH_BLOCKED") throw new BlockedError(error.message);
      throw error;
    }
  };
  const screenshot = async (t, id, active = session) => {
    const path = join(artifactDir, id + ".png");
    await active.page.screenshot({ path, fullPage: true });
    t.artifact(path);
  };
  const passed = id => scenarios.find(value => value.id === id)?.status === "passed";
  async function scenario(id, action, prerequisites = ["workbench.startup"]) {
    const spec = WORKBENCH_SCENARIOS.find(value => value.id === id);
    const missing = prerequisites.filter(id => !passed(id));
    if (signal?.aborted || missing.length) {
      const result = { ...spec, status: "not-run", durationMs: 0, steps: [], assertions: [], artifacts: [],
        error: signal?.aborted ? "Acceptance was cancelled" : "Prerequisite did not pass: " + missing.join(", ") };
      scenarios.push(result); onScenario?.(result); return result;
    }
    const result = await runScenario({
      ...spec, artifactDir,
      captureFailure: async path => {
        if (!session?.page || session.page.isClosed()) return;
        const domPath = join(artifactDir, id + ".dom.txt");
        await writeFile(domPath, await session.page.locator("body").innerText()); artifacts.push(domPath);
        await session.page.screenshot({ path, fullPage: true }); return path;
      },
    }, async t => {
      signal?.throwIfAborted();
      await action(t);
      signal?.throwIfAborted();
    });
    if (signal?.aborted && result.status !== "passed") result.status = "not-run";
    scenarios.push(result); onScenario?.(result);
    if (result.status === "failed" && session && !signal?.aborted) {
      // A failed case may leave a modal or live task. Restart only our session;
      // keep its files and failure evidence intact for diagnosis and later cases.
      fixture.server.releaseHeld();
      await session.stop();
      session = await start();
    }
    return result;
  }

  try {
    await scenario("workbench.startup", async t => {
      t.step("启动当前分支打包 CLI，打开带随机能力路径的工作台");
      const before = fixture.server.requests.length;
      session = await start();
      await session.page.locator(".reading-pane").waitFor();
      t.check((await api(session, "api/status")).vaultRoot === fixture.vaultRoot, "工作台使用本次临时 Vault");
      t.check(fixture.server.requests.length === before, "浏览工作台没有调用论文来源或模型");
      await screenshot(t, "startup");
    }, []);

    await scenario("workbench.first-use", async t => {
      const empty = await createFixtureEnvironment({ configured: false });
      let first;
      try {
        t.step("无配置启动，首先将保存目录改成本次临时 Vault");
        first = await startWorkbenchSession({ fixture: empty, artifactDir: join(artifactDir, "first-use"), signal, headless, executablePath });
        sessions.push(first); artifacts.push(...first.artifacts);
        await first.page.locator(".settings-form").waitFor();
        t.check((await api(first, "api/status")).setupRequired === true, "首次使用自动展示配置界面");
        await editField(first.page, "vaultRoot", empty.vaultRoot);
        await until(() => exists(empty.configPath), "first-use draft reaches TOML", { signal });
        await closeDialog(first.page);
        t.check((await readConfig(empty)).vault_root === empty.vaultRoot, "目录草稿保存到独立配置");
        await openSettings(first.page);
        t.check(await first.page.locator('[name="vaultRoot"]').inputValue() === empty.vaultRoot, "重新打开仍显示测试目录");
        t.check(empty.server.requests.length === 0, "首次设置不自动调用模型");
        await screenshot(t, "first-use", first);
      } catch (error) {
        artifacts.push(...(error.artifacts ?? []));
        if (first?.page && !first.page.isClosed()) {
          const path = join(artifactDir, "first-use-failure.png");
          await first.page.screenshot({ path, fullPage: true }); t.artifact(path);
        }
        if (error.code === "WORKBENCH_BLOCKED") throw new BlockedError(error.message);
        throw error;
      } finally { await first?.stop(); await empty.dispose(); }
    });

    await scenario("workbench.settings", async t => {
      const page = session.page;
      t.step("打开设置，修改模型与研究方向，等待实际自动保存");
      await openSettings(page);
      const key = page.locator('[name="apiKey"]');
      t.check(await key.inputValue() === "", "已保存的密钥默认不回填");
      await editField(page, "model", MODEL);
      await page.locator(".settings-topic > summary").first().click();
      await editField(page, "topicDirection", DIRECTION);
      await until(async () => {
        const value = await readConfig(fixture);
        return value.llm.model === MODEL && value.arxiv.topics[0].directions[0].text === DIRECTION;
      }, "model and direction autosave to TOML", { signal });
      t.check((await readConfig(fixture)).llm.api_key === "fixture-key", "空密钥字段保留原有密钥");
      t.step("显式获取模型并使用键盘从同一个模型输入框选择");
      await page.locator('[data-settings="models"]').click();
      await page.getByRole("option", { name: MODEL, exact: true }).waitFor();
      t.check(fixture.server.requests.some(request => request.kind === "models"), "模型列表由真实设置接口发起");
      const model = page.locator('[name="model"]');
      await model.fill("fixture-model-2");
      await model.press("ArrowDown");
      await model.press("Enter");
      await closeDialog(page);
      await page.reload();
      await openSettings(page);
      t.check(await page.locator('[name="model"]').inputValue() === MODEL, "刷新后仍使用已保存模型");
      await page.locator(".settings-topic > summary").first().click();
      t.check(await page.locator('[name="topicDirection"]').first().inputValue() === DIRECTION, "刷新后研究方向仍一致");
      t.check((await readConfig(fixture)).workbench_schedule.enabled === false, "仅打开设置没有启动自动任务");
      await screenshot(t, "settings");
      await closeDialog(page);
    });

    await scenario("workbench.daily-generation", async t => {
      t.step("通过生成对话框启动固定日期的真实日报流程");
      const before = fixture.server.requests.length;
      const run = await startDaily(session, ACCEPTANCE_DATE);
      const result = await waitForWorkbenchRun(session, run, signal);
      t.check(result.status === "completed", "界面任务确认 completed", result);
      const saved = await ensureDaily(t, fixture, ACCEPTANCE_DATE); firstMarkdown = saved.markdown;
      const requests = fixture.server.requests.slice(before);
      t.check(requests.some(request => request.kind === "filter" && request.model === MODEL && request.prompt.includes(DIRECTION)), "实际筛选请求使用界面保存的模型与方向");
      t.check(requests.filter(request => request.kind === "summary").length === PAPERS.length, "每篇实际调用摘要步骤");
      t.step("重复选择同一天，确认不再次请求模型且原始日报不变");
      const calls = fixture.server.requests.length;
      const duplicate = await waitForWorkbenchRun(session, await startDaily(session, ACCEPTANCE_DATE), signal);
      t.check(duplicate.status === "skipped", "已完成日期重复运行显示跳过");
      t.check(fixture.server.requests.length === calls, "重复运行未重新调用外部接口");
      t.check(await readFile(saved.paths.daily(ACCEPTANCE_DATE), "utf8") === firstMarkdown, "重复运行未重写日报");
      await screenshot(t, "daily-generation");
    }, ["workbench.settings"]);

    await scenario("workbench.reading", async t => {
      const page = session.page, paths = await pathsFor(fixture);
      t.step("筛选论文，打开概览，收藏并标为待读");
      await page.locator('[data-action="clear-date"]').click();
      const search = page.locator('.search-box input[type="search"]');
      await search.fill(PAPERS[0].id);
      await until(async () => await page.locator(".paper-row").count() === 1, "search narrows to one real paper", { signal });
      await page.locator('[data-paper="' + starKey + '"]').click();
      await page.locator(".paper-overview").waitFor();
      t.check((await page.locator(".document-title").innerText()).includes(PAPERS[0].title), "概览属于被点击的论文");
      await page.locator('[data-mark="star"]').click();
      await until(async () => (await readJson(paths.index)).papers[starKey].priority === "high", "star persists", { signal });
      await page.locator('[data-mark="status"]').selectOption("to_read");
      await until(async () => (await readJson(paths.index)).papers[starKey].status === "to_read", "reading status persists", { signal });
      t.check(await page.locator(".paper-overview").isVisible(), "搜索后阅读和修改标记不会意外返回列表");
      t.step("打开来源日报再后退，保持概览和列表筛选");
      await page.locator('.report-links [data-document]').first().click();
      await page.locator("article.markdown-body").waitFor();
      t.check((await page.locator("article.markdown-body").innerText()).includes(PAPERS[0].id), "阅读的是已保存的完整日报");
      await page.locator('[data-action="history-back"]').click();
      await page.locator(".paper-overview").waitFor();
      await page.locator('[data-action="back"]').click();
      await page.locator(".paper-workspace").waitFor();
      t.check(await search.inputValue() === PAPERS[0].id, "返回列表保留搜索输入");
      await search.fill("");
      await page.locator('[data-scope="starred"]').click();
      await until(async () => await page.locator(".paper-row").count() === 1, "favorite filter shows one paper", { signal });
      t.check(await page.locator('[data-paper="' + starKey + '"]').count() === 1, "收藏筛选展示实际已收藏论文");
      t.check(await readFile(paths.daily(ACCEPTANCE_DATE), "utf8") === firstMarkdown, "阅读与标记未改写日报");
      await screenshot(t, "reading");
      await page.locator('[data-scope="all"]').click();
    }, ["workbench.daily-generation"]);

    await scenario("workbench.paper-note", async t => {
      const page = session.page;
      t.step("从论文概览显式生成单篇详细总结");
      await page.locator('[data-paper="' + starKey + '"]').click();
      await page.locator('[data-action="generate-paper"]').click();
      const response = page.waitForResponse(response => response.url().endsWith("/api/runs") && response.request().method() === "POST");
      await page.locator('.generation-form button[type="submit"]').click();
      const started = await (await response).json();
      const result = await waitForWorkbenchRun(session, started.run, signal);
      t.check(result.status === "completed", "单篇任务完成", result);
      const paper = (await api(session, "api/paper?key=" + encodeURIComponent(starKey))).paper;
      t.check(typeof paper.detailPath === "string" && await exists(join(fixture.vaultRoot, paper.detailPath)), "论文总结文件实际存在");
      await page.locator(".document-links [data-document]").waitFor();
      t.check((await page.locator(".document-title").innerText()).includes(PAPERS[0].title), "完成后仍停留在同一论文概览");
      await page.locator(".document-links [data-document]").click();
      await page.locator("article.markdown-body").waitFor();
      t.check((await page.locator("article.markdown-body").innerText()).length > 300, "保存的详细总结可打开阅读");
      await screenshot(t, "paper-note");
      await page.locator('[data-action="back"]').click();
    }, ["workbench.reading"]);

    await scenario("workbench.cancel-retry", async t => {
      const page = session.page, paths = await pathsFor(fixture), before = fixture.server.requests.length;
      const configuration = await readFile(fixture.configPath, "utf8");
      fixture.server.setMode({ llm: "hold" });
      try {
        t.step("挂起外部模型响应，在真实任务运行中尝试再次生成");
        const run = await startDaily(session, CANCEL_DATE);
        await until(() => fixture.server.requests.slice(before).some(request => request.held), "model request is held", { signal });
        await page.locator('[data-action="generate"]').click();
        t.check(await page.locator('.generation-form button[type="submit"]').isDisabled(), "运行中再次打开生成对话框不能启动第二个任务");
        await closeDialog(page);
        t.step("运行中编辑设置，显示失败且不覆盖已生效配置");
        await openSettings(page);
        await editField(page, "model", "fixture-model");
        await page.locator('[data-settings="retry-save"]').waitFor();
        t.check(await readFile(fixture.configPath, "utf8") === configuration, "运行中的设置保存没有写入磁盘");
        t.check(await page.locator('[name="model"]').inputValue() === "fixture-model", "被拒绝的设置仍保留在草稿中");
        await discardSettings(page);
        t.step("点击取消任务，确认外部请求终止且没有半份日报");
        await page.locator('[data-action="cancel-run"]').click();
        const cancelled = await waitForWorkbenchRun(session, run, signal);
        t.check(cancelled.status === "cancelled", "取消由后台确认", cancelled);
        await until(() => fixture.server.requests.slice(before).some(request => request.held && request.cancelled), "held transport observes cancellation", { signal });
        t.check(!await exists(paths.daily(CANCEL_DATE)), "取消没有产生日报文件");
        const state = await readJson(paths.state);
        t.check(state.runState[CANCEL_DATE]?.status !== "running", "磁盘运行状态不再卡在 running");
      } finally { fixture.server.releaseHeld(); }
      t.step("恢复模型响应，再次生成同一日期");
      const recovered = await waitForWorkbenchRun(session, await startDaily(session, CANCEL_DATE), signal);
      t.check(recovered.status === "completed", "取消后同一日期可重新生成", recovered);
      await ensureDaily(t, fixture, CANCEL_DATE);
      await screenshot(t, "cancel-retry");
    }, ["workbench.settings"]);

    await scenario("workbench.failure-retry", async t => {
      const page = session.page, paths = await pathsFor(fixture);
      t.step("固定模型返回 401，确认失败状态和未提交文件");
      fixture.server.setMode({ llm: "unauthorized" });
      try {
        const result = await waitForWorkbenchRun(session, await startDaily(session, FAILURE_DATE), signal);
        t.check(result.status === "failed", "认证错误显示为失败", result);
        t.check(!await exists(paths.daily(FAILURE_DATE)), "失败没有生成残缺日报");
        t.check((await readJson(paths.state)).runState[FAILURE_DATE]?.status === "failed_permanent", "磁盘保留失败类型供恢复判断");
      } finally { fixture.server.setMode({ llm: "normal" }); }
      t.step("更正模型设置，再从界面重试同一日期");
      await openSettings(page);
      await editField(page, "apiKey", "fixture-repaired-key");
      await until(async () => (await readConfig(fixture)).llm.api_key === "fixture-repaired-key", "repaired key is saved", { signal });
      await closeDialog(page);
      const result = await waitForWorkbenchRun(session, await startDaily(session, FAILURE_DATE), signal);
      t.check(result.status === "completed", "修正配置后失败日期实际重跑而非假成功或跳过", result);
      await ensureDaily(t, fixture, FAILURE_DATE);
      await screenshot(t, "failure-retry");
    }, ["workbench.settings"]);

    await scenario("workbench.settings-conflict", async t => {
      const page = session.page;
      t.step("设置已打开时模拟另一编辑器修改测试 TOML");
      await openSettings(page);
      const changed = await readFile(fixture.configPath, "utf8") + "\n# acceptance external editor revision\n";
      await writeFile(fixture.configPath, changed);
      await editField(page, "model", "fixture-model");
      await page.locator('[data-settings="retry-save"]').waitFor();
      t.check(await readFile(fixture.configPath, "utf8") === changed, "旧表单没有覆盖外部修改");
      t.check(await page.locator('[name="model"]').inputValue() === "fixture-model", "发生冲突后用户输入仍保留");
      t.step("明确放弃草稿后重开，读取最新设置");
      await discardSettings(page);
      await openSettings(page);
      t.check(await page.locator('[name="model"]').inputValue() === MODEL, "重新打开读取真实配置而非失败草稿");
      // Save one actual edit to activate the latest revision in the running server.
      await editField(page, "model", "fixture-model");
      await until(async () => (await readConfig(fixture)).llm.model === "fixture-model", "new revision becomes active", { signal });
      await editField(page, "model", MODEL);
      await until(async () => (await readConfig(fixture)).llm.model === MODEL, "restore selected model through UI", { signal });
      await closeDialog(page);
      await screenshot(t, "settings-conflict");
    }, ["workbench.settings"]);

    await scenario("workbench.library", async t => {
      const page = session.page, before = fixture.server.requests.length;
      t.step("连接测试 PDF 目录，不自动接受处理授权或开始索引");
      await openSettings(page);
      await page.locator('[data-settings="library-connect"]').click();
      await page.locator('[name="libraryPath"]').fill(fixture.libraryRoot);
      await page.locator('[data-settings="confirm-library-connect"]').click();
      await until(async () => (await api(session, "api/settings/library")).status.kind !== "disconnected", "library connects", { signal });
      t.check((await readFile(fixture.configPath, "utf8")).includes(fixture.libraryRoot), "文献库目录持久化到测试配置");
      await closeDialog(page);
      await page.locator('[data-action="personal-library"]').click();
      await page.locator(".library-workspace").waitFor();
      t.check((await api(session, "api/library")).connected === true, "真实文献库浏览接口识别已连接目录");
      await page.locator('[data-library="review"]').click();
      await page.locator(".library-review-workspace").waitFor();
      t.check(await page.locator(".library-review-workspace").isVisible(), "方向审核入口可访问");
      t.check(fixture.server.requests.length === before, "连接与浏览入口没有隐式模型处理");
      await screenshot(t, "library");
      await page.locator('[data-action="history-back"]').click();
    }, ["workbench.settings"]);

    await scenario("workbench.restart", async t => {
      const page = session.page;
      t.step("修改外观和侧栏，等待偏好保存后停止服务");
      await page.goto(session.url);
      await openSettings(page);
      await page.locator('[name="appearance.theme"]').selectOption("dark");
      await until(async () => (await api(session, "api/preferences")).appearance.theme === "dark", "dark theme persists", { signal });
      await closeDialog(page);
      await page.locator(".sidebar-toggle").click();
      await until(async () => (await api(session, "api/preferences")).sidebarCollapsed === true, "collapsed sidebar persists", { signal });
      const oldUrl = session.url;
      await session.stop();
      t.step("用同一测试配置重新启动到新端口和新能力链接");
      session = await start();
      await session.page.locator(".paper-workspace").waitFor();
      t.check(session.url !== oldUrl, "重启获得独立工作台会话");
      await until(async () => await session.page.locator("#app").getAttribute("data-theme") === "dark", "theme is restored after restart", { signal });
      t.check(await session.page.locator("#app").getAttribute("data-theme") === "dark", "主题从磁盘恢复");
      await until(async () => await session.page.locator(".sidebar-toggle").getAttribute("aria-expanded") === "false", "sidebar is restored after restart", { signal });
      t.check(await session.page.locator(".sidebar-toggle").getAttribute("aria-expanded") === "false", "侧栏收起状态从磁盘恢复");
      await session.page.locator(".sidebar-toggle").click();
      await session.page.locator('[data-scope="starred"]').click();
      await until(async () => await session.page.locator('[data-paper="' + starKey + '"]').count() === 1, "favorite survives process restart", { signal });
      const paper = (await api(session, "api/paper?key=" + encodeURIComponent(starKey))).paper;
      t.check(paper.starred === true && paper.status === "to_read", "收藏与待读标记跨进程保留");
      await openSettings(session.page);
      t.check(await session.page.locator('[name="model"]').inputValue() === MODEL, "模型配置跨进程保留");
      await closeDialog(session.page);
      t.check((await readConfig(fixture)).workbench_schedule.enabled === false, "重启没有意外开启自动生成");
      await screenshot(t, "restart");
    }, ["workbench.reading", "workbench.settings"]);

    await scenario("workbench.browser-errors", async t => {
      const pageErrors = sessions.flatMap(value => value.diagnostics.pageErrors);
      t.check(pageErrors.length === 0, "实际浏览器未捕获的异常为零", pageErrors);
      const unexpected = fixture.server.requests.filter(request => ["unexpected", "fixture-error"].includes(request.kind));
      t.check(unexpected.length === 0, "业务流程没有未知外部请求或无法识别的模型任务", unexpected);
    });
  } catch (error) {
    errors.push(error.message);
    for (const spec of WORKBENCH_SCENARIOS) if (!scenarios.some(value => value.id === spec.id)) {
      scenarios.push({ ...spec, status: "not-run", durationMs: 0, steps: [], assertions: [], artifacts: [], error: "Suite interrupted: " + error.message });
    }
  } finally {
    fixture.server.releaseHeld();
    for (const value of sessions) {
      try { await value.stop(); } catch (error) { errors.push(error.message); }
    }
    const requestsPath = join(artifactDir, "workbench.external-requests.json");
    await writeFile(requestsPath, JSON.stringify(fixture.server.requests, null, 2) + "\n");
    artifacts.push(requestsPath);
  }
  const status = errors.length || scenarios.some(value => value.status === "failed") ? "failed"
    : scenarios.some(value => value.status !== "passed") ? "blocked" : "passed";
  return { suite: "workbench", status, scenarios, artifacts, ...(errors.length ? { errors } : {}) };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const artifactDir = resolve(process.argv[2] || join(root, "output/playwright/acceptance/workbench-" + Date.now()));
  const controller = new AbortController();
  const stop = () => controller.abort();
  process.once("SIGINT", stop); process.once("SIGTERM", stop);
  const fixture = await createFixtureEnvironment();
  try {
    const suite = await runWorkbenchAcceptance({ fixture, artifactDir, signal: controller.signal,
      onScenario: value => console.log(value.id + ": " + value.status + (value.error ? " — " + value.error : "")) });
    const report = buildReport({ suites: [suite], selectedSuites: ["workbench"] });
    await writeReport(artifactDir, report);
    console.log("Workbench evidence: " + artifactDir);
    process.exitCode = report.exitCode;
  } finally {
    await fixture.dispose();
    process.off("SIGINT", stop); process.off("SIGTERM", stop);
  }
}
