/**
 * The Obsidian goal-task library (P3 of docs/helm/2026-10-10-model-driven-release-exploration),
 * separated from exploration-obsidian.mjs (the driver: observation, CDP execution, session/host wiring)
 * so "what the model must do, and how a program tells it passed" grows independently of "how the host is
 * observed and driven" — chunks 2-5 only touch this file (and workbench.mjs) to add tasks/personas.
 *
 * `SETTINGS_FILE`/`INDEX_FILE`/`entryFor` stay defined once in exploration-obsidian.mjs (the driver also
 * uses them for `reset()`/`ensureDailyReport()`/`ensurePaperUnstarred()`) and are imported here, not
 * duplicated. Each task's `verify()` reads real, independent plugin state via `driver.jsonFile(...)` —
 * never the model's own claim of success — matching every other goal task in this project.
 */
import { ACCEPTANCE_DATE, PAPERS } from "../fixtures.mjs";
import { entryFor, INDEX_FILE, SETTINGS_FILE } from "../exploration-obsidian.mjs";
import { poll, pollJsonFile } from "../obsidian-ui.mjs";
import { checkResult, taskResult } from "./checks.mjs";

/** Single source of truth for the default Obsidian goal-task ids, usable without a fixture (e.g. --list). */
export const GOAL_TASK_IDS = ["daily-paper-limit", "star-paper-from-report"];

/** No Obsidian personas yet (P3 Chunk 5); kept as an explicit empty array, not undefined, so `--list` and selection stay uniform across hosts. */
export const PERSONA_TASK_IDS = [];

/**
 * The default Obsidian goal tasks (originally `createDefaultObsidianExplorationTasks`, P2 Chunk 4). Each
 * verifies real, independent state: the plugin's own settings store on disk (`data.json`) and its paper
 * index (`papers.json`) — never the model's own claim of success.
 */
export function goalTasks() {
  const target = 7;
  const paperId = PAPERS[0].id;
  // Memoized across repeated verify() calls within one task run: closing and reopening Obsidian's own
  // settings window is a real, visible side effect, so it must happen at most once per task, not once per
  // loop step while other checks are still pending.
  let reopenCheck;
  return [
    {
      id: GOAL_TASK_IDS[0], title: "修改每日论文数量上限",
      goal: `打开插件设置页，把 Daily paper limit 改为 ${target}。移开焦点，等待保存完成即可，不需要关闭设置；不要修改路径、模型连接、密钥或邮件相关设置。`,
      async verify({ driver, observation, actions }) {
        const filled = actions.some(action => action.type === "fill" && action.control?.name === "Daily paper limit" && action.value === String(target));
        if (!filled) return taskResult([checkResult(`通过设置页将 Daily paper limit 改为 ${target}`, false)]);
        const onDisk = await pollJsonFile(() => driver.jsonFile(SETTINGS_FILE), value => value?.settings?.output?.maxDailyPapers === target, "daily paper limit on disk", { timeoutMs: 5000 }).catch(() => null);
        const persisted = onDisk?.settings?.output?.maxDailyPapers === target;
        if (!persisted) {
          return taskResult([checkResult(`通过设置页将 Daily paper limit 改为 ${target}`, true), checkResult("隔离设置文件保存了新上限", false)]);
        }
        // Only the test program closes and reopens settings (the model has no exposed control for Obsidian's
        // own settings chrome — the plugin's observed surface is scoped to its own tab container), and only
        // once persistence is already confirmed, so an in-progress model action is never interrupted by this.
        // Reading the re-rendered value is polled, not read once: a real host briefly shows the previous
        // render while the declarative settings page re-paints after re-opening.
        reopenCheck ??= (async () => {
          await driver.closeThenReopenSettings();
          const value = await poll(
            () => driver.read(`Array.from(document.querySelectorAll("input")).find(el => el.getAttribute("aria-label") === "Daily paper limit")?.value`),
            value => value === String(target), "re-rendered daily paper limit", { timeoutMs: 3000 },
          ).catch(() => undefined);
          return value;
        })();
        const reopened = await reopenCheck;
        return taskResult([
          checkResult(`通过设置页将 Daily paper limit 改为 ${target}`, true),
          checkResult("隔离设置文件保存了新上限", persisted, onDisk.settings.output.maxDailyPapers),
          checkResult("重新打开设置后界面仍显示新上限", reopened === String(target), reopened),
        ]);
      },
    },
    {
      id: GOAL_TASK_IDS[1], title: "在阅读面板收藏一篇论文",
      goal: `打开阅读面板（Reading Dashboard）的 All 列表，找到 arXiv ${paperId} 对应行，点击它的 Star paper 按钮将其标记为收藏。`,
      async prepare({ driver }) {
        await driver.ensureDailyReport(ACCEPTANCE_DATE);
        await driver.ensurePaperUnstarred(paperId);
      },
      async verify({ driver, actions }) {
        const starred = actions.some(action => action.type === "click" && action.control?.paperKey === paperId && /star paper/i.test(action.control?.name || ""));
        if (!starred) return taskResult([checkResult(`点击 arXiv ${paperId} 的 Star paper 收藏按钮`, false)]);
        const onDisk = await pollJsonFile(() => driver.jsonFile(INDEX_FILE), value => entryFor(value, paperId)?.priority === "high", "starred priority on disk", { timeoutMs: 5000 }).catch(() => null);
        const marked = entryFor(onDisk, paperId)?.priority === "high";
        return taskResult([
          checkResult(`点击 arXiv ${paperId} 的 Star paper 收藏按钮`, true),
          checkResult("插件的收藏标记已写入论文索引", marked, onDisk ? entryFor(onDisk, paperId)?.priority : undefined),
        ]);
      },
    },
  ];
}

/** No Obsidian personas yet (P3 Chunk 5). */
export function personaTasks() {
  return [];
}
