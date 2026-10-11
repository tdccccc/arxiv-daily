/**
 * The workbench goal-task library (P3 of docs/helm/2026-10-10-model-driven-release-exploration),
 * separated from exploration-browser.mjs (the driver: observation, Playwright execution, session
 * wiring) so "what the model must do, and how a program tells it passed" grows independently of "how
 * the host is observed and driven" — chunks 2-5 only touch this file (and obsidian.mjs) to add
 * tasks/personas.
 *
 * `assertFixture`/`settings` stay defined once in exploration-browser.mjs (the driver also uses them for
 * `inspect()`/`dailyEvidence()`) and are imported here, not duplicated.
 */
import { assertFixture, settings } from "../exploration-browser.mjs";
import { checkResult, taskResult } from "./checks.mjs";

/** Single source of truth for the default workbench goal-task ids, usable without a fixture (e.g. --list). */
export const GOAL_TASK_IDS = ["settings-persistence", "sidebar-persistence", "generate-and-read"];

/** No workbench personas yet (P3 Chunk 5); kept as an explicit empty array, not undefined, so `--list` and selection stay uniform across hosts. */
export const PERSONA_TASK_IDS = [];

/** The default workbench goal tasks (originally `createDefaultExplorationTasks`, P1). */
export async function goalTasks(fixture) {
  await assertFixture(fixture);
  const initial = await settings(fixture);
  const target = initial.output?.max_daily_papers === 13 ? 14 : 13;
  let collapsedAfterReload = false, collapseReloadStep = 0;
  return [
    {
      id: GOAL_TASK_IDS[0], title: "设置自动保存并刷新复查",
      goal: `打开设置，把每日论文数量上限 maxDailyPapers 改为 ${target}。移开焦点并等待保存完成，关闭设置，刷新页面，再打开设置，确认上限仍为 ${target}。不要修改路径、模型连接或密钥。`,
      async verify({ driver, observation, actions }) {
        const changed = actions.find(action => action.type === "fill" && (action.control?.field || action.control?.name) === "maxDailyPapers" && action.value === String(target));
        const refreshed = changed && actions.some(action => action.type === "reload" && action.step > changed.step);
        const state = await driver.inspect();
        return taskResult([
          checkResult("通过界面修改每日论文数量上限", changed),
          checkResult("修改后刷新了页面", refreshed),
          checkResult("隔离配置文件保存了新上限", state.maxDailyPapers === target, state.maxDailyPapers),
          checkResult("刷新后重新打开设置仍显示新上限", refreshed && observation.controls.some(control => control.field === "maxDailyPapers" && control.value === String(target))),
        ]);
      },
    },
    {
      id: GOAL_TASK_IDS[1], title: "侧栏折叠和展开状态持久化",
      goal: "关闭设置（如果仍打开）。收起侧栏，刷新页面确认侧栏仍收起；再展开侧栏，再次刷新，确认侧栏仍展开。等待偏好保存后再刷新。",
      async verify({ driver, observation, actions }) {
        const state = await driver.inspect();
        const last = actions.findLast(action => action.type === "reload");
        const lastToggle = actions.findLast(action => action.type === "click" && action.control?.purpose === "sidebar-toggle");
        const clicked = actions.some(action => action.type === "click" && action.control?.purpose === "sidebar-toggle");
        if (clicked && last && lastToggle && last.step > lastToggle.step && state.preferences.sidebarCollapsed === true && observation.layout.sidebarCollapsed) {
          collapsedAfterReload = true; collapseReloadStep = last.step;
        }
        const expandedAfterReload = collapsedAfterReload && last && lastToggle && last.step > lastToggle.step && last.step > collapseReloadStep && state.preferences.sidebarCollapsed === false && !observation.layout.sidebarCollapsed;
        return taskResult([checkResult("通过界面切换侧栏", clicked), checkResult("刷新后持久化收起状态与页面一致", collapsedAfterReload), checkResult("再展开并刷新后持久化状态与页面一致", expandedAfterReload)]);
      },
    },
    {
      id: GOAL_TASK_IDS[2], title: "生成固定日期日报并打开论文",
      goal: "通过生成按钮生成 2026-10-01 日报（选择指定日期，不要生成今天）。等待成功，再从列表打开 arXiv 2610.10001 论文，确认阅读内容出现。无需判断论文内容质量。",
      async verify({ driver, observation, actions, startedAt }) {
        const state = await driver.dailyEvidence("2026-10-01");
        const generated = actions.some(action => action.type === "click" && action.control?.dataAction === "generate");
        const run = state.run;
        const checks = [
          checkResult("通过生成入口启动任务", generated),
          checkResult("本任务的指定日期运行已完成", run?.status === "completed" && run.date === "2026-10-01" && Date.parse(run.startedAt) >= startedAt, run ? { date: run.date, status: run.status } : null),
          checkResult("真实日报文件包含夹具论文", state.report),
          checkResult("论文阅读视图已打开", observation.layout.view === "reading" && observation.text.includes("2610.10001")),
        ];
        if (run?.date === "2026-10-01" && Date.parse(run.startedAt) >= startedAt && (run.status === "failed" || (run.status === "completed" && !state.report))) {
          return { passed: false, failed: true, checks, error: run.status === "failed" ? "The requested fixture generation reported failure" : "Generation reported completion without the expected report file" };
        }
        return taskResult(checks);
      },
    },
  ];
}

/** No workbench personas yet (P3 Chunk 5). */
export function personaTasks() {
  return [];
}
