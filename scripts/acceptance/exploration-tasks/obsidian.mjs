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
import { dailyReportFile, entryFor, HISTORY_FILE, INDEX_FILE, SETTINGS_FILE, STATE_FILE } from "../exploration-obsidian.mjs";
import { poll, pollJsonFile } from "../obsidian-ui.mjs";
import { checkResult, taskResult } from "./checks.mjs";

/** Single source of truth for the default Obsidian goal-task ids, usable without a fixture (e.g. --list). */
export const GOAL_TASK_IDS = [
  "daily-paper-limit", "star-paper-from-report",
  "add-topic-direction", "remove-topic-direction", "add-arxiv-category",
  "run-for-date", "cancel-then-rerun", "auth-failure-retry", "completed-date-idempotent",
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

// ─── P3 Chunk 3: Obsidian run tasks ──────────────────────────────────────────
// Each run task below uses its own date, distinct from ACCEPTANCE_DATE (already used by
// star-paper-from-report's prepare()) and from each other, so the vault content earlier tasks leave
// behind (reset() deliberately does not touch it) never interferes across tasks within one pass. Dates
// are drawn from fixtures.mjs's own hardcoded `DATES` list (the only dates the fixture's scraped arXiv
// "recent" listing actually serves real papers for); any other date would list zero papers and the run
// would "complete" with nothing to verify against. completed-date-idempotent deliberately reuses
// ACCEPTANCE_DATE instead of a fresh one: the task needs a date that is already completed, and
// ACCEPTANCE_DATE's report is exactly what star-paper-from-report's own prepare() already establishes
// deterministically (and this task's own prepare() re-asserts independently, never relying on task order).
export const RUN_FOR_DATE_DATE = "2026-10-02";
export const CANCEL_THEN_RERUN_DATE = "2026-10-05";
export const AUTH_FAILURE_RETRY_DATE = "2026-10-06";

/** Parses the real run-history.jsonl (append-only, never truncated by clearDateState) into records. */
async function readHistory(driver) {
  const text = await driver.file(HISTORY_FILE).catch(error => { if (error.code === "ENOENT") return ""; throw error; });
  return text.trim().split("\n").filter(Boolean).map(line => JSON.parse(line));
}

/**
 * Polls readHistory() until `accepts` sees a matching record, tolerating the brief window before a write
 * lands. `timeoutMs` defaults to 5000 — comfortably under the exploration loop's own ~10s per-verify()-call
 * budget (exploration.mjs's `withDeadline` around `task.verify(...)`), the same convention the existing
 * settings tasks' `pollJsonFile(..., { timeoutMs: 5000 })` calls already use: a run that genuinely takes
 * longer than one verify() call to finish (the real pipeline can take tens of seconds) is still observed
 * correctly because verify() is re-invoked, and re-polls from scratch, on every subsequent loop step.
 */
async function pollHistory(driver, accepts, label, options) {
  return poll(() => readHistory(driver), accepts, label, { timeoutMs: 5000, ...options });
}

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
  // P3 Chunk 3 memoized side effects: a held model request is released at most once (the first time the
  // model's cancel click is observed), and the fixture is restored to "normal" at most once (the first
  // time the model has actually reviewed the recorded failure) — never earlier than the action that
  // earns it, and never repeated on every polling pass of verify().
  let cancelReleaseDone = false, authFailureSeen = false, authFixtureRestored = false;
  let idempotentBaseline;
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
    // ─── P3 Chunk 3: run tasks ──────────────────────────────────────────────
    {
      id: "run-for-date", title: "按日期生成日报",
      goal: `点击仪表盘工具栏的“More arXiv Daily actions”按钮打开菜单，点击其中的“Run for date…”，在弹出的日期对话框里把 Date 填为 ${RUN_FOR_DATE_DATE}，再点击 Run 提交，等待生成完成。`,
      // Deterministically starts this date "not yet generated" (never through the model): clears any
      // run-state entry, daily report file and paper-index links a previous pass of this same task left
      // behind, so a second full-library pass in one session still begins from the same untouched state.
      async prepare({ driver }) {
        await driver.clearDateState(RUN_FOR_DATE_DATE);
        await driver.openDashboard();
      },
      async verify({ driver, actions }) {
        const submitted = actions.some(action => action.type === "fill" && action.control?.name === "Date" && action.value === RUN_FOR_DATE_DATE);
        if (!submitted) return taskResult([checkResult(`通过 Run for date 对话框提交 ${RUN_FOR_DATE_DATE}`, false)]);
        const state = await pollJsonFile(() => driver.jsonFile(STATE_FILE), value => value?.runState?.[RUN_FOR_DATE_DATE]?.status === "completed", "run-state completed", { timeoutMs: 5000 }).catch(() => null);
        const completed = state?.runState?.[RUN_FOR_DATE_DATE]?.status === "completed";
        if (!completed) {
          return taskResult([
            checkResult(`通过 Run for date 对话框提交 ${RUN_FOR_DATE_DATE}`, true),
            checkResult("运行状态标记该日期已完成", false, state?.runState?.[RUN_FOR_DATE_DATE]?.status),
          ]);
        }
        const reportPath = dailyReportFile(RUN_FOR_DATE_DATE);
        const content = await driver.file(reportPath).catch(() => "");
        const hasPapers = PAPERS.every(paper => content.includes(paper.id) && content.includes(paper.title));
        const index = await driver.jsonFile(INDEX_FILE).catch(() => null);
        const indexed = PAPERS.every(paper => entryFor(index, paper.id)?.dailyReports?.some(path => path.endsWith(`${RUN_FOR_DATE_DATE}.md`)));
        return taskResult([
          checkResult(`通过 Run for date 对话框提交 ${RUN_FOR_DATE_DATE}`, true),
          checkResult("运行状态标记该日期已完成", true),
          checkResult("日报文件写出固定夹具论文", hasPapers, content.slice(0, 200)),
          checkResult("论文索引记录了该日期的日报链接", indexed),
        ]);
      },
    },
    {
      id: "cancel-then-rerun", title: "运行中取消后重跑",
      goal: `点击“More arXiv Daily actions”打开菜单，点击“Run for date…”，日期填 ${CANCEL_THEN_RERUN_DATE} 并提交；测试环境会让这次运行暂停在处理中。再次打开菜单，点击“Cancel active tasks”取消这次运行。取消完成后，再次打开菜单点击“Run for date…”，同样填入 ${CANCEL_THEN_RERUN_DATE} 并提交，让这次真正跑完。`,
      async prepare({ driver }) {
        await driver.clearDateState(CANCEL_THEN_RERUN_DATE);
        // Program-only fixture control (mirrors obsidian-scenarios.mjs's cancelAndRepeat): holds the
        // model request open so there is a real window in which the model's cancel click matters,
        // instead of racing a run that could complete before the model ever reaches the menu.
        driver.fixture.server.setMode({ llm: "hold" });
        await driver.openDashboard();
      },
      async verify({ driver, actions }) {
        const firstSubmitted = actions.some(action => action.type === "fill" && action.control?.name === "Date" && action.value === CANCEL_THEN_RERUN_DATE);
        if (!firstSubmitted) return taskResult([checkResult(`通过 Run for date 提交 ${CANCEL_THEN_RERUN_DATE}`, false)]);
        const cancelClickIndex = actions.findIndex(action => action.type === "click" && /^cancel active tasks$/i.test((action.control?.name || "").trim()));
        const cancelClicked = cancelClickIndex !== -1;
        // Releasing the held request and restoring normal fixture mode is a program-only side effect
        // (never a UI action the model can reach), performed at most once, and only once the model has
        // actually clicked Cancel — never before, so the rerun's eventual success is contingent on that
        // click, not merely on the program's own timing.
        if (cancelClicked && !cancelReleaseDone) {
          cancelReleaseDone = true;
          driver.fixture.server.releaseHeld();
          driver.fixture.server.setMode({ llm: "normal" });
        }
        if (!cancelClicked) {
          return taskResult([
            checkResult(`通过 Run for date 提交 ${CANCEL_THEN_RERUN_DATE}`, true),
            checkResult("点击 Cancel active tasks 取消正在运行的任务", false),
          ]);
        }
        const cancelHistory = await pollHistory(driver, records => records.some(record => record.date === CANCEL_THEN_RERUN_DATE && record.resultKind === "cancelled"), "cancelled run-history record").catch(() => []);
        const cancelRecorded = cancelHistory.some(record => record.date === CANCEL_THEN_RERUN_DATE && record.resultKind === "cancelled");
        const rerunSubmitted = actions.slice(cancelClickIndex + 1).some(action => action.type === "fill" && action.control?.name === "Date" && action.value === CANCEL_THEN_RERUN_DATE);
        if (!cancelRecorded || !rerunSubmitted) {
          return taskResult([
            checkResult(`通过 Run for date 提交 ${CANCEL_THEN_RERUN_DATE}`, true),
            checkResult("点击 Cancel active tasks 取消正在运行的任务", true),
            checkResult("运行历史记录了这次取消", cancelRecorded, cancelHistory.filter(record => record.date === CANCEL_THEN_RERUN_DATE)),
            checkResult(`取消后再次通过 Run for date 提交 ${CANCEL_THEN_RERUN_DATE}`, rerunSubmitted),
          ]);
        }
        const state = await pollJsonFile(() => driver.jsonFile(STATE_FILE), value => value?.runState?.[CANCEL_THEN_RERUN_DATE]?.status === "completed", "run-state completed after rerun", { timeoutMs: 5000 }).catch(() => null);
        const completed = state?.runState?.[CANCEL_THEN_RERUN_DATE]?.status === "completed";
        const index = await driver.jsonFile(INDEX_FILE).catch(() => null);
        const noDuplicates = PAPERS.every(paper => (entryFor(index, paper.id)?.dailyReports ?? []).filter(path => path.endsWith(`${CANCEL_THEN_RERUN_DATE}.md`)).length === 1);
        return taskResult([
          checkResult(`通过 Run for date 提交 ${CANCEL_THEN_RERUN_DATE}`, true),
          checkResult("点击 Cancel active tasks 取消正在运行的任务", true),
          checkResult("运行历史记录了这次取消", true),
          checkResult(`取消后再次通过 Run for date 提交 ${CANCEL_THEN_RERUN_DATE}`, true),
          checkResult("最终状态为已完成（非重复）", completed, state?.runState?.[CANCEL_THEN_RERUN_DATE]?.status),
          checkResult("论文索引没有重复的日报条目", noDuplicates),
        ]);
      },
    },
    {
      id: "auth-failure-retry", title: "认证失败后重试",
      goal: `点击“More arXiv Daily actions”打开菜单，点击“Run for date…”，日期填 ${AUTH_FAILURE_RETRY_DATE} 并提交；这次运行会因认证失败而结束。打开菜单，点击“Show logs & history”，在弹出的面板里点击“Run History”标签查看这次失败的记录，然后关闭面板。确认失败原因后，再次打开菜单点击“Force run for date…”，同样填入 ${AUTH_FAILURE_RETRY_DATE} 并提交以重新运行。`,
      // Design: the API key field is restricted/sensitive and off-limits to the model (classify
      // ControlRestriction in exploration-obsidian.mjs), so "fix the credential" is not an action the
      // model can take. Instead the fixture itself is only restored to accepting requests once the model
      // has done the one allowed, observable thing that matters here — actually reviewing the recorded
      // failure (opening the Hub modal's own Run History tab) — not merely clicking Force run blindly.
      // Until that review happens, Force run would still hit the same 401 and fail again, so verify
      // cannot pass without the model's specific review action; this is checked below, not assumed.
      async prepare({ driver }) {
        await driver.clearDateState(AUTH_FAILURE_RETRY_DATE);
        driver.fixture.server.setMode({ llm: "unauthorized" });
        driver.expectAuthorizationFailure();
        await driver.openDashboard();
      },
      async verify({ driver, actions }) {
        const submitted = actions.some(action => action.type === "fill" && action.control?.name === "Date" && action.value === AUTH_FAILURE_RETRY_DATE);
        if (!submitted) return taskResult([checkResult(`通过 Run for date 提交 ${AUTH_FAILURE_RETRY_DATE}`, false)]);
        // Memoized once observed: by the time the model has force-retried, the run-state's CURRENT
        // status has already moved on to "completed", overwriting "failed_permanent" — re-polling the
        // live status at that point would wrongly conclude the failure never happened. Once this task
        // has genuinely seen the failure recorded on disk, that fact is remembered for the rest of the
        // task run instead of being re-derived from the (by then different) current status.
        if (!authFailureSeen) {
          const failedState = await pollJsonFile(() => driver.jsonFile(STATE_FILE), value => value?.runState?.[AUTH_FAILURE_RETRY_DATE]?.status === "failed_permanent", "run-state failed_permanent", { timeoutMs: 5000 }).catch(() => null);
          if (failedState?.runState?.[AUTH_FAILURE_RETRY_DATE]?.status === "failed_permanent") authFailureSeen = true;
        }
        if (!authFailureSeen) {
          return taskResult([
            checkResult(`通过 Run for date 提交 ${AUTH_FAILURE_RETRY_DATE}`, true),
            checkResult("运行状态记录了认证失败", false),
          ]);
        }
        const reviewedHistory = actions.some(action => action.type === "click" && /^run history$/i.test((action.control?.name || "").trim()));
        // Restoring fixture access is withheld until the failure is both real (confirmed above) and the
        // model has reviewed it; memoized so it only ever runs once per task.
        if (reviewedHistory && !authFixtureRestored) {
          authFixtureRestored = true;
          driver.fixture.server.setMode({ llm: "normal" });
        }
        if (!reviewedHistory) {
          return taskResult([
            checkResult(`通过 Run for date 提交 ${AUTH_FAILURE_RETRY_DATE}`, true),
            checkResult("运行状态记录了认证失败", true),
            checkResult("打开 Show logs & history 并查看 Run History 标签", false),
          ]);
        }
        const forceSubmitted = actions.some(action => action.type === "click" && /^force run for date…?$/i.test((action.control?.name || "").trim()));
        const retrySubmitted = actions.filter(action => action.type === "fill" && action.control?.name === "Date" && action.value === AUTH_FAILURE_RETRY_DATE).length >= 2;
        if (!forceSubmitted || !retrySubmitted) {
          return taskResult([
            checkResult(`通过 Run for date 提交 ${AUTH_FAILURE_RETRY_DATE}`, true),
            checkResult("运行状态记录了认证失败", true),
            checkResult("打开 Show logs & history 并查看 Run History 标签", true),
            checkResult(`通过 Force run for date 重新提交 ${AUTH_FAILURE_RETRY_DATE}`, forceSubmitted && retrySubmitted),
          ]);
        }
        const completedState = await pollJsonFile(() => driver.jsonFile(STATE_FILE), value => value?.runState?.[AUTH_FAILURE_RETRY_DATE]?.status === "completed", "run-state completed after force retry", { timeoutMs: 5000 }).catch(() => null);
        const completed = completedState?.runState?.[AUTH_FAILURE_RETRY_DATE]?.status === "completed";
        const history = await readHistory(driver).catch(() => []);
        const forceCompletedRecorded = history.some(record => record.date === AUTH_FAILURE_RETRY_DATE && record.event === "completed" && record.trigger === "force");
        const reportPath = dailyReportFile(AUTH_FAILURE_RETRY_DATE);
        const content = await driver.file(reportPath).catch(() => "");
        const hasPapers = PAPERS.every(paper => content.includes(paper.id));
        return taskResult([
          checkResult(`通过 Run for date 提交 ${AUTH_FAILURE_RETRY_DATE}`, true),
          checkResult("运行状态记录了认证失败", true),
          checkResult("打开 Show logs & history 并查看 Run History 标签", true),
          checkResult(`通过 Force run for date 重新提交 ${AUTH_FAILURE_RETRY_DATE}`, true),
          checkResult("重试后运行状态变为已完成", completed, completedState?.runState?.[AUTH_FAILURE_RETRY_DATE]?.status),
          checkResult("运行历史记录了这次强制重跑的完成", forceCompletedRecorded),
          checkResult("日报文件写出固定夹具论文", hasPapers, content.slice(0, 200)),
        ]);
      },
    },
    {
      id: "completed-date-idempotent", title: "对已完成日期重复运行不重复生成",
      goal: `该日期（${ACCEPTANCE_DATE}）已经生成过日报。点击“More arXiv Daily actions”打开菜单，点击“Run for date…”（不是 Force run），日期填 ${ACCEPTANCE_DATE} 并提交；预期它会被判定为已完成而跳过，不会重新生成。`,
      // Deterministically ensures a completed date exists (never via the model) and records the exact
      // baseline — report content and the fixture request count — prepare()'s own moment, independent
      // of whatever earlier task may or may not have already generated this same date.
      async prepare({ driver }) {
        await driver.ensureDailyReport(ACCEPTANCE_DATE);
        await driver.idle();
        idempotentBaseline = {
          content: await driver.file(dailyReportFile(ACCEPTANCE_DATE)),
          requestCount: driver.fixture.server.requests.length,
          historyCount: (await readHistory(driver).catch(() => [])).length,
        };
        await driver.openDashboard();
      },
      async verify({ driver, actions }) {
        const submitted = actions.some(action => action.type === "fill" && action.control?.name === "Date" && action.value === ACCEPTANCE_DATE);
        if (!submitted) return taskResult([checkResult(`通过 Run for date 提交 ${ACCEPTANCE_DATE}`, false)]);
        const history = await readHistory(driver).catch(() => []);
        const freshHistory = history.slice(idempotentBaseline.historyCount);
        const skipRecorded = freshHistory.some(record => record.date === ACCEPTANCE_DATE && record.event === "skipped" && record.resultKind === "skipped" && record.reason === "already done");
        if (!skipRecorded) {
          return taskResult([
            checkResult(`通过 Run for date 提交 ${ACCEPTANCE_DATE}`, true),
            checkResult("运行历史记录了“已完成，跳过”的结果", false, freshHistory.filter(record => record.date === ACCEPTANCE_DATE)),
          ]);
        }
        const content = await driver.file(dailyReportFile(ACCEPTANCE_DATE)).catch(() => "");
        const unchanged = content === idempotentBaseline.content;
        const noNewRequests = driver.fixture.server.requests.slice(idempotentBaseline.requestCount).every(request => !request.model);
        const index = await driver.jsonFile(INDEX_FILE).catch(() => null);
        const noDuplicates = PAPERS.every(paper => (entryFor(index, paper.id)?.dailyReports ?? []).filter(path => path.endsWith(`${ACCEPTANCE_DATE}.md`)).length === 1);
        return taskResult([
          checkResult(`通过 Run for date 提交 ${ACCEPTANCE_DATE}`, true),
          checkResult("运行历史记录了“已完成，跳过”的结果", true),
          checkResult("日报文件逐字保持不变", unchanged),
          checkResult("没有发起新的模型请求", noNewRequests, driver.fixture.server.requests.slice(idempotentBaseline.requestCount).filter(request => request.model)),
          checkResult("论文索引没有产生重复条目", noDuplicates),
        ]);
      },
    },
  ];
}

/** No Obsidian personas yet (P3 Chunk 5). */
export function personaTasks() {
  return [];
}
