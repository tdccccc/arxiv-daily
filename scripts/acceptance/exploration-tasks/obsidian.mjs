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
export const GOAL_TASK_IDS = [
  "daily-paper-limit", "star-paper-from-report",
  "add-topic-direction", "remove-topic-direction", "add-arxiv-category",
];

/** No Obsidian personas yet (P3 Chunk 5); kept as an explicit empty array, not undefined, so `--list` and selection stay uniform across hosts. */
export const PERSONA_TASK_IDS = [];

// Fixture topic/direction (scripts/acceptance/obsidian.mjs's createObsidianSettings): one topic named
// "Research" (id "acceptance-research", tag "research") holding one direction "Reliable research
// methods" (id "acceptance-reliable", origin "manual"). Exported so unit tests can build a fake driver's
// on-disk state without duplicating these fixture values; the P3 Chunk 2 tasks below read/mutate them
// through the same topic id and direction text, never through a hardcoded index.
export const FIXTURE_TOPIC_ID = "acceptance-research";
export const FIXTURE_TOPIC_NAME = "Research";
export const FIXTURE_DIRECTION_TEXT = "Reliable research methods";
/** The new direction `add-topic-direction` asks the model to add. */
export const NEW_DIRECTION_TEXT = "Scaling laws for sparse mixture-of-experts models";
/** The direction `remove-topic-direction`'s `prepare()` seeds (deterministically, not via the model) for the model to delete. */
export const SEEDED_DIRECTION_TEXT = "Temporary seeded direction for the remove-topic-direction task";
/** The category `add-arxiv-category` asks the model to add, chosen because it is neither the fixture's "cs.AI" nor the dropdown's own auto-picked default ("astro-ph", the first category alphabetically absent from the list) — so reaching it requires an explicit `select`, not just accepting whatever "Add category" defaults to. */
export const TARGET_CATEGORY_ID = "cs.LG";
export const TARGET_CATEGORY_LABEL = "Machine Learning";

/**
 * The default Obsidian goal tasks (originally `createDefaultObsidianExplorationTasks`, P2 Chunk 4; P3
 * Chunk 2 adds the topic/direction/category tasks below). Each verifies real, independent state: the
 * plugin's own settings store on disk (`data.json`) and its paper index (`papers.json`) — never the
 * model's own claim of success.
 */
export function goalTasks() {
  const target = 7;
  const paperId = PAPERS[0].id;
  // Memoized across repeated verify() calls within one task run: closing/reopening or reloading
  // Obsidian's own settings window is a real, visible side effect, so each kind must happen at most once
  // per task, not once per loop step while other checks are still pending.
  let reopenCheck, reloadCheck, addDirectionReopenCheck, removeDirectionReopenCheck, categoryReopenCheck;
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
        // P3 Chunk 2: re-opening the settings tab proves the *rendered value* survived, but not that the
        // setting itself is durable — a stale in-memory settings object could still render correctly from
        // its own stale copy. A full plugin reload (disablePlugin/enablePlugin) rebuilds every in-memory
        // settings object from data.json, so this is the stronger, independent proof the phase contract
        // asks for ("verify after the program reloads the plugin, not only re-opening settings").
        reloadCheck ??= (async () => {
          await driver.reloadThenReopenSettings();
          const value = await poll(
            () => driver.read(`Array.from(document.querySelectorAll("input")).find(el => el.getAttribute("aria-label") === "Daily paper limit")?.value`),
            value => value === String(target), "re-rendered daily paper limit after a full plugin reload", { timeoutMs: 3000 },
          ).catch(() => undefined);
          return value;
        })();
        const reloaded = await reloadCheck;
        return taskResult([
          checkResult(`通过设置页将 Daily paper limit 改为 ${target}`, true),
          checkResult("隔离设置文件保存了新上限", persisted, onDisk.settings.output.maxDailyPapers),
          checkResult("重新打开设置后界面仍显示新上限", reopened === String(target), reopened),
          checkResult("完整重载插件后设置页仍显示新上限", reloaded === String(target), reloaded),
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
    // ─── P3 Chunk 2: Research topic/direction editing ─────────────────────
    {
      id: "add-topic-direction", title: "新增研究方向",
      goal: `打开插件设置页，展开研究主题“${FIXTURE_TOPIC_NAME}”（点击该主题的标题行即可展开），点击 Add direction 新增一个方向，并将其内容填写为：“${NEW_DIRECTION_TEXT}”。不要修改已有的方向“${FIXTURE_DIRECTION_TEXT}”，也不要新增或删除任何主题。`,
      async verify({ driver }) {
        const onDisk = await pollJsonFile(() => driver.jsonFile(SETTINGS_FILE), value => {
          const topic = value?.settings?.arxiv?.topics?.find(item => item.id === FIXTURE_TOPIC_ID);
          return Boolean(topic?.directions?.some(direction => direction.text === NEW_DIRECTION_TEXT));
        }, "new direction saved on disk", { timeoutMs: 5000 }).catch(() => null);
        const topic = onDisk?.settings?.arxiv?.topics?.find(item => item.id === FIXTURE_TOPIC_ID);
        const added = Boolean(topic?.directions?.some(direction => direction.text === NEW_DIRECTION_TEXT));
        const kept = Boolean(topic?.directions?.some(direction => direction.text === FIXTURE_DIRECTION_TEXT));
        if (!added || !kept) {
          return taskResult([
            checkResult(`主题“${FIXTURE_TOPIC_NAME}”新增方向“${NEW_DIRECTION_TEXT}”`, added, topic?.directions?.map(direction => direction.text)),
            checkResult(`原有方向“${FIXTURE_DIRECTION_TEXT}”保持不变`, kept),
          ]);
        }
        addDirectionReopenCheck ??= (async () => {
          await driver.closeThenReopenSettings();
          return poll(
            () => driver.read(`Array.from(document.querySelectorAll(".arxiv-daily-settings__topic-direction-input")).map(el => el.value)`),
            value => Array.isArray(value) && value.includes(NEW_DIRECTION_TEXT), "re-rendered direction list", { timeoutMs: 5000 },
          ).catch(() => undefined);
        })();
        const reopened = await addDirectionReopenCheck;
        return taskResult([
          checkResult(`主题“${FIXTURE_TOPIC_NAME}”新增方向“${NEW_DIRECTION_TEXT}”`, true),
          checkResult(`原有方向“${FIXTURE_DIRECTION_TEXT}”保持不变`, true),
          checkResult("重新打开设置后方向列表包含新方向", Array.isArray(reopened) && reopened.includes(NEW_DIRECTION_TEXT), reopened),
        ]);
      },
    },
    {
      id: "remove-topic-direction", title: "删除研究方向",
      goal: `插件设置页中，研究主题“${FIXTURE_TOPIC_NAME}”现在有两个方向：“${FIXTURE_DIRECTION_TEXT}”和“${SEEDED_DIRECTION_TEXT}”。展开该主题，删除“${SEEDED_DIRECTION_TEXT}”这一条（点击它旁边的 × 按钮），保留“${FIXTURE_DIRECTION_TEXT}”不变。`,
      // Deterministically seeds the second direction by writing data.json directly and reloading the
      // plugin — never through the model's own actions — so the task starts from a known two-direction
      // state regardless of what any earlier task left behind (every task's reset() already restored the
      // one-direction fixture baseline before this prepare() runs).
      async prepare({ driver }) {
        await driver.applySettingsPatch(current => {
          const topic = current.settings?.arxiv?.topics?.find(item => item.id === FIXTURE_TOPIC_ID);
          if (!topic) throw new Error(`Fixture topic ${FIXTURE_TOPIC_ID} is missing; cannot seed a second direction`);
          if (!topic.directions.some(direction => direction.text === SEEDED_DIRECTION_TEXT)) {
            topic.directions.push({ id: "acceptance-seeded-direction", text: SEEDED_DIRECTION_TEXT, origin: "manual" });
          }
        });
        // applySettingsPatch's reload leaves no plugin surface open; every other task starts on the
        // dashboard after reset(), so this task does too, and the model navigates into settings itself
        // (via the dashboard's own "Settings" button), exactly like the other settings tasks above.
        await driver.openDashboard();
      },
      async verify({ driver }) {
        const onDisk = await pollJsonFile(() => driver.jsonFile(SETTINGS_FILE), value => {
          const topic = value?.settings?.arxiv?.topics?.find(item => item.id === FIXTURE_TOPIC_ID);
          return Array.isArray(topic?.directions) && !topic.directions.some(direction => direction.text === SEEDED_DIRECTION_TEXT);
        }, "seeded direction removed on disk", { timeoutMs: 5000 }).catch(() => null);
        const topic = onDisk?.settings?.arxiv?.topics?.find(item => item.id === FIXTURE_TOPIC_ID);
        const removed = Boolean(topic) && !topic.directions.some(direction => direction.text === SEEDED_DIRECTION_TEXT);
        const kept = Boolean(topic?.directions?.some(direction => direction.text === FIXTURE_DIRECTION_TEXT));
        if (!removed || !kept) {
          return taskResult([
            checkResult(`删除方向“${SEEDED_DIRECTION_TEXT}”`, removed, topic?.directions?.map(direction => direction.text)),
            checkResult(`保留方向“${FIXTURE_DIRECTION_TEXT}”`, kept),
          ]);
        }
        removeDirectionReopenCheck ??= (async () => {
          await driver.closeThenReopenSettings();
          return poll(
            () => driver.read(`Array.from(document.querySelectorAll(".arxiv-daily-settings__topic-direction-input")).map(el => el.value)`),
            value => Array.isArray(value) && value.includes(FIXTURE_DIRECTION_TEXT) && !value.includes(SEEDED_DIRECTION_TEXT),
            "re-rendered direction list after removal", { timeoutMs: 5000 },
          ).catch(() => undefined);
        })();
        const reopened = await removeDirectionReopenCheck;
        const reopenedOk = Array.isArray(reopened) && reopened.includes(FIXTURE_DIRECTION_TEXT) && !reopened.includes(SEEDED_DIRECTION_TEXT);
        return taskResult([
          checkResult(`删除方向“${SEEDED_DIRECTION_TEXT}”`, true),
          checkResult(`保留方向“${FIXTURE_DIRECTION_TEXT}”`, true),
          checkResult("重新打开设置后方向列表已更新", reopenedOk, reopened),
        ]);
      },
    },
    // ─── P3 Chunk 2: arXiv category editing ────────────────────────────────
    // Replaces the phase doc's planned "reorder directions" task: reading plugin/src/settings/tab.ts's
    // direction editor (appendDirection/renderDirections) shows each direction row has only an "Add
    // direction" button and a per-row "×" remove button — no up/down control and no drag-and-drop at all,
    // so there is no reorder control to drive (not even an undrivable drag one). Per the phase's abort
    // trigger, this is reported rather than faked, and replaced with the example the trigger itself
    // suggests: toggling (adding) an arXiv category, which is driven by a real native <select> dropdown
    // (renderCategoryRow, declarative-rows.ts) the `select` action already supports.
    {
      id: "add-arxiv-category", title: "新增 arXiv 分类",
      goal: `打开插件设置页，在 arXiv categories 区域点击 Add category 新增一行，并把新增的下拉框选为 ${TARGET_CATEGORY_ID}（${TARGET_CATEGORY_LABEL}）。不要移除或修改已有的 cs.AI。`,
      async verify({ driver }) {
        const onDisk = await pollJsonFile(() => driver.jsonFile(SETTINGS_FILE), value => {
          const categories = value?.settings?.arxiv?.categories;
          return Array.isArray(categories) && categories.includes(TARGET_CATEGORY_ID) && categories.includes("cs.AI");
        }, "new category saved on disk", { timeoutMs: 5000 }).catch(() => null);
        const categories = onDisk?.settings?.arxiv?.categories;
        const added = Array.isArray(categories) && categories.includes(TARGET_CATEGORY_ID);
        const kept = Array.isArray(categories) && categories.includes("cs.AI");
        if (!added || !kept) {
          return taskResult([
            checkResult(`新增 arXiv 分类 ${TARGET_CATEGORY_ID}`, added, categories),
            checkResult("保留已有分类 cs.AI", kept),
          ]);
        }
        categoryReopenCheck ??= (async () => {
          await driver.closeThenReopenSettings();
          // Not scoped by a CSS class: category dropdowns are rendered either by the <1.13 legacy
          // display() path (Obsidian's own DropdownComponent, no custom class on its <select>) or by the
          // 1.13+ declarative renderCategoryRow (declarative-rows.ts, class "arxiv-daily-settings__category-
          // select") depending on the installed Obsidian version — this test host runs 1.11.5, the legacy
          // path. Both paths build each select's options from the same ARXIV_CATEGORIES list
          // (addCategoryOptions), so matching "a <select> that has an option with this category's value" is
          // the one selector that works on either rendering path, the same heuristic the controlled model
          // itself uses to find the row (see /tmp/p3c2/controlled.mjs's addArxivCategoryStep).
          return poll(
            () => driver.read(`Array.from(document.querySelectorAll("select")).filter(el => Array.from(el.options).some(option => option.value === ${JSON.stringify(TARGET_CATEGORY_ID)})).map(el => el.value)`),
            value => Array.isArray(value) && value.includes(TARGET_CATEGORY_ID), "re-rendered category list", { timeoutMs: 5000 },
          ).catch(() => undefined);
        })();
        const reopened = await categoryReopenCheck;
        return taskResult([
          checkResult(`新增 arXiv 分类 ${TARGET_CATEGORY_ID}`, true),
          checkResult("保留已有分类 cs.AI", true),
          checkResult("重新打开设置后分类列表包含新分类", Array.isArray(reopened) && reopened.includes(TARGET_CATEGORY_ID), reopened),
        ]);
      },
    },
  ];
}

/** No Obsidian personas yet (P3 Chunk 5). */
export function personaTasks() {
  return [];
}
