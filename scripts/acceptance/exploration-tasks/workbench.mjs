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
import { readFile, rm } from "node:fs/promises";
import path from "node:path";
import { ACCEPTANCE_DATE, PAPERS } from "../fixtures.mjs";
import { assertFixture, settings, writeConfigPatch } from "../exploration-browser.mjs";
import { checkResult, exactLabelPattern, taskResult, uiLabel, uiLabelException } from "./checks.mjs";
import { PERSONA_TASK_IDS, personaTasks as sharedPersonaTasks } from "./personas.mjs";

/** Single source of truth for the default workbench goal-task ids, usable without a fixture (e.g. --list). */
export const GOAL_TASK_IDS = [
  "settings-persistence", "sidebar-persistence", "generate-and-read",
  // P3 Chunk 5: reading-journey and settings-conflict tasks.
  "star-filter-restore", "generate-paper-note", "external-config-conflict",
];

/** P3 Chunk 5: re-exports the shared persona id list (exploration-tasks/personas.mjs) so `--list`/selection keep reading it from each host's own module, unchanged from before chunk 5 wired it up. */
export { PERSONA_TASK_IDS };

// ─── P3 Chunk 5: reading-journey fixtures ────────────────────────────────────
// star-filter-restore stars PAPERS[1] (not PAPERS[0], generate-and-read's own target) so the two tasks
// never touch the same mark; both fixture papers appear in the "all" scope once a report exists, which is
// exactly what lets the "starred" scope's filtering be observed (one paper visible, the other excluded).
const STAR_FILTER_PAPER = PAPERS[1];
const STAR_FILTER_OTHER_PAPER = PAPERS[0];
const STAR_FILTER_KEY = `arxiv:${STAR_FILTER_PAPER.id}`;
const STAR_FILTER_OTHER_KEY = `arxiv:${STAR_FILTER_OTHER_PAPER.id}`;
// generate-paper-note reuses PAPERS[0] (generate-and-read's own target): that task only opens the paper's
// reading view, never generates a detail summary, so there is no real collision with this task's own
// precondition (no existing paper note) across a single library pass or repeated passes (prepare() below
// clears it deterministically every time, exactly like the Obsidian create-paper-note task does).
const NOTE_PAPER = PAPERS[0];
const NOTE_KEY = `arxiv:${NOTE_PAPER.id}`;
// external-config-conflict never touches vault data; only the CLI TOML's own [llm] model field, which no
// other workbench task reads or asserts on, so this task is independent of task order and of how many
// times the library has already run.
const EXTERNAL_CONFLICT_DRAFT_MODEL = "workbench-conflict-draft-model";
const EXTERNAL_CONFLICT_EXTERNAL_MODEL = "workbench-conflict-external-model";

// ─── P2/P6: declared UI labels ────────────────────────────────────────────────
// scripts/tests/acceptance-exploration-labels.test.mjs asserts every task's own `uiLabels` entry (every
// product UI label its goal text names or its verify() matches on) is still present, as a complete,
// case-sensitive string literal, in the specific product source file(s) named by its own `uiLabel(text,
// source)` wrapper (P6; P2 originally searched all of apps/cli/src/workbench/** case-insensitively).
// Every verify() that matches one of these reads the exact constant below, never a second,
// independently-typed copy of the same string, so the goal text, the verifier and the declared label
// cannot drift apart. `uiLabelException` marks the one entry below that cannot be checked this way.
const MAX_DAILY_PAPERS_FIELD_LABEL = "maxDailyPapers";
// The driver's own `purpose` tag (exploration-browser.mjs) is derived from this CSS class name, but the
// product (web/sidebar.ts) only ever writes it as part of a longer className literal
// (`"quiet-button sidebar-toggle"`), never as a standalone complete string literal — so a source-scoped
// exact-literal check cannot anchor on it (P6 exception below).
const SIDEBAR_TOGGLE_PURPOSE_LABEL = "sidebar-toggle";
const GENERATE_ACTION_LABEL = "generate";
const STAR_FILTER_TAB_LABEL = "收藏";
const BACK_TO_LIST_LABEL = "← 返回列表";
const GENERATE_DETAIL_SUMMARY_LABEL = "生成详细总结";
const START_GENERATION_LABEL = "开始生成";
const READ_DETAIL_SUMMARY_LABEL = "阅读详细总结";
const MODEL_FIELD_LABEL = "Model";
const DISCARD_UNSAVED_AND_CLOSE_LABEL = "放弃未保存修改并关闭";
const CONFIRM_DISCARD_AND_CLOSE_LABEL = "放弃修改并关闭";
const SETTINGS_BUTTON_LABEL = "设置";
const RETRY_SAVE_LABEL = "重试保存";

async function paperIndexPath(fixture) {
  const config = await settings(fixture);
  return path.join(fixture.vaultRoot, path.dirname(config.output.daily_dir), ".index", "papers.json");
}
async function readPaperIndex(fixture) {
  try { return JSON.parse(await readFile(await paperIndexPath(fixture), "utf8")); }
  catch (error) { if (error.code === "ENOENT") return null; throw error; }
}
async function paperNotePath(fixture, arxivId) {
  const config = await settings(fixture);
  return path.resolve(fixture.vaultRoot, config.output.papers_dir, `${arxivId}.md`);
}
/** Bounded like every other goal task's own polling (pollJsonFile's 5000ms default in the Obsidian task
 * library): long enough for a real but fast fixture write to land, short enough to stay under verify()'s
 * own ~10s per-call budget (exploration.mjs's withDeadline) so a slow task just tries again next loop step
 * instead of timing the whole call out. */
async function pollFile(read, accepts, { timeoutMs = 5000, intervalMs = 150 } = {}) {
  const deadline = Date.now() + timeoutMs;
  while (true) {
    let value;
    try { value = await read(); } catch { value = undefined; }
    if (accepts(value)) return value;
    if (Date.now() >= deadline) return value;
    await new Promise(resolve => setTimeout(resolve, intervalMs));
  }
}

/** The default workbench goal tasks (originally `createDefaultExplorationTasks`, P1). */
export async function goalTasks(fixture) {
  await assertFixture(fixture);
  const initial = await settings(fixture);
  const target = initial.output?.max_daily_papers === 13 ? 14 : 13;
  let collapsedAfterReload = false, collapseReloadStep = 0;
  // P3 Chunk 5: the external-config-conflict task's own one-shot side effect, mirroring the Obsidian run
  // tasks' memoized flags (e.g. cancelReleaseDone in exploration-tasks/obsidian.mjs) — the external write
  // must happen exactly once, the first time settings are observed loaded, never on every later poll.
  let externalConflictWritten = false;
  return [
    {
      id: GOAL_TASK_IDS[0], title: "设置自动保存并刷新复查",
      goal: `打开设置，把每日论文数量上限 ${MAX_DAILY_PAPERS_FIELD_LABEL} 改为 ${target}。移开焦点并等待保存完成，关闭设置，刷新页面，再打开设置，确认上限仍为 ${target}。不要修改路径、模型连接或密钥。`,
      uiLabels: [uiLabel(MAX_DAILY_PAPERS_FIELD_LABEL, "apps/cli/src/workbench/web/settings.ts")],
      async verify({ driver, observation, actions }) {
        const changed = actions.find(action => action.type === "fill" && (action.control?.field || action.control?.name) === MAX_DAILY_PAPERS_FIELD_LABEL && action.value === String(target));
        const refreshed = changed && actions.some(action => action.type === "reload" && action.step > changed.step);
        const state = await driver.inspect();
        return taskResult([
          checkResult("通过界面修改每日论文数量上限", changed),
          checkResult("修改后刷新了页面", refreshed),
          checkResult("隔离配置文件保存了新上限", state.maxDailyPapers === target, state.maxDailyPapers),
          checkResult("刷新后重新打开设置仍显示新上限", refreshed && observation.controls.some(control => control.field === MAX_DAILY_PAPERS_FIELD_LABEL && control.value === String(target))),
        ]);
      },
    },
    {
      id: GOAL_TASK_IDS[1], title: "侧栏折叠和展开状态持久化",
      goal: "关闭设置（如果仍打开）。收起侧栏，刷新页面确认侧栏仍收起；再展开侧栏，再次刷新，确认侧栏仍展开。等待偏好保存后再刷新。",
      uiLabels: [uiLabelException(SIDEBAR_TOGGLE_PURPOSE_LABEL, "test-only driver hook: only a substring of the longer className literal \"quiet-button sidebar-toggle\" in apps/cli/src/workbench/web/sidebar.ts, never a standalone literal or display text")],
      async verify({ driver, observation, actions }) {
        const state = await driver.inspect();
        const last = actions.findLast(action => action.type === "reload");
        const lastToggle = actions.findLast(action => action.type === "click" && action.control?.purpose === SIDEBAR_TOGGLE_PURPOSE_LABEL);
        const clicked = actions.some(action => action.type === "click" && action.control?.purpose === SIDEBAR_TOGGLE_PURPOSE_LABEL);
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
      uiLabels: [uiLabel(GENERATE_ACTION_LABEL, "apps/cli/src/workbench/web/app.ts")],
      async verify({ driver, observation, actions, startedAt }) {
        const state = await driver.dailyEvidence("2026-10-01");
        const generated = actions.some(action => action.type === "click" && action.control?.dataAction === GENERATE_ACTION_LABEL);
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
    // ─── P3 Chunk 5: reading-journey and settings-conflict tasks ────────────
    {
      id: GOAL_TASK_IDS[3], title: "收藏后切换筛选并恢复阅读",
      goal: `打开阅读面板。找到标题为“${STAR_FILTER_PAPER.title}”的论文，点击它的收藏按钮将其标记为收藏。点击顶部的“${STAR_FILTER_TAB_LABEL}”筛选，确认这篇论文仍显示在列表里。点击它的标题进入阅读概览，再点击“${BACK_TO_LIST_LABEL}”返回列表。确认返回后仍停留在“${STAR_FILTER_TAB_LABEL}”筛选，并且仍能看到这篇论文。`,
      uiLabels: [
        uiLabel(STAR_FILTER_TAB_LABEL, "apps/cli/src/workbench/web/papers.ts"),
        uiLabel(BACK_TO_LIST_LABEL, "apps/cli/src/workbench/web/papers.ts"),
      ],
      async prepare({ driver }) {
        await driver.ensureDailyReport(ACCEPTANCE_DATE);
        await driver.ensurePaperUnstarred(STAR_FILTER_KEY);
      },
      async verify({ observation, actions }) {
        const starName = control => (control?.name || "").includes(STAR_FILTER_TAB_LABEL) && (control?.name || "").includes(STAR_FILTER_PAPER.title);
        const starClickIndex = actions.findIndex(action => action.type === "click" && starName(action.control));
        if (starClickIndex === -1) return taskResult([checkResult(`点击“${STAR_FILTER_PAPER.title}”的收藏按钮`, false)]);
        // The scope filter button's own accessible name is just "收藏" plus a live count ("收藏12"), never
        // " · <title>" — distinct from the star button's aria-label by construction, not just by luck.
        const filterClickIndex = actions.findIndex((action, index) => index > starClickIndex && action.type === "click" && new RegExp(`^${STAR_FILTER_TAB_LABEL}[\\d—]*$`).test((action.control?.name || "").trim()));
        if (filterClickIndex === -1) return taskResult([checkResult(`点击“${STAR_FILTER_PAPER.title}”的收藏按钮`, true), checkResult("切换到顶部“收藏”筛选", false)]);
        const openClickIndex = actions.findIndex((action, index) => index > filterClickIndex && action.type === "click" && action.control?.paperKey === STAR_FILTER_KEY);
        if (openClickIndex === -1) {
          return taskResult([checkResult(`点击“${STAR_FILTER_PAPER.title}”的收藏按钮`, true), checkResult("切换到顶部“收藏”筛选", true), checkResult("点击该论文标题进入阅读概览", false)]);
        }
        const backClicked = actions.slice(openClickIndex + 1).some(action => action.type === "click" && (action.control?.name || "").trim() === BACK_TO_LIST_LABEL);
        if (!backClicked) {
          return taskResult([checkResult(`点击“${STAR_FILTER_PAPER.title}”的收藏按钮`, true), checkResult("切换到顶部“收藏”筛选", true), checkResult("点击该论文标题进入阅读概览", true), checkResult("点击“← 返回列表”返回", false)]);
        }
        const index = await pollFile(() => readPaperIndex(fixture), value => value?.papers?.[STAR_FILTER_KEY]?.priority === "high").catch(() => null);
        const marked = index?.papers?.[STAR_FILTER_KEY]?.priority === "high";
        const restored = observation.controls.some(control => control.paperKey === STAR_FILTER_KEY) && !observation.controls.some(control => control.paperKey === STAR_FILTER_OTHER_KEY);
        return taskResult([
          checkResult(`点击“${STAR_FILTER_PAPER.title}”的收藏按钮`, true),
          checkResult("切换到顶部“收藏”筛选", true),
          checkResult("点击该论文标题进入阅读概览", true),
          checkResult("点击“← 返回列表”返回", true),
          checkResult("独立论文索引记录了收藏标记", marked, index ? index.papers?.[STAR_FILTER_KEY] : undefined),
          checkResult("返回后仍停留在“收藏”筛选，且仍显示这篇论文、不显示未收藏的另一篇", restored),
        ]);
      },
    },
    {
      id: GOAL_TASK_IDS[4], title: "为单篇论文生成详细总结",
      goal: `打开阅读面板，点击标题为“${NOTE_PAPER.title}”的论文进入概览。点击“${GENERATE_DETAIL_SUMMARY_LABEL}”按钮，在弹出的对话框里直接点击“${START_GENERATION_LABEL}”，等待生成完成（概览里出现“${READ_DETAIL_SUMMARY_LABEL}”链接即代表完成）。`,
      uiLabels: [
        uiLabel(GENERATE_DETAIL_SUMMARY_LABEL, "apps/cli/src/workbench/web/papers.ts"),
        uiLabel(START_GENERATION_LABEL, "apps/cli/src/workbench/web/app.ts"),
        uiLabel(READ_DETAIL_SUMMARY_LABEL, "apps/cli/src/workbench/web/papers.ts"),
      ],
      async prepare({ driver }) {
        await driver.ensureDailyReport(ACCEPTANCE_DATE);
        await rm(await paperNotePath(fixture, NOTE_PAPER.id), { force: true });
      },
      async verify({ observation, actions }) {
        const openedOverview = actions.some(action => action.type === "click" && action.control?.paperKey === NOTE_KEY);
        if (!openedOverview) return taskResult([checkResult(`点击“${NOTE_PAPER.title}”进入概览`, false)]);
        const clickedGenerate = actions.some(action => action.type === "click" && (action.control?.name || "").trim() === GENERATE_DETAIL_SUMMARY_LABEL);
        if (!clickedGenerate) return taskResult([checkResult(`点击“${NOTE_PAPER.title}”进入概览`, true), checkResult("点击“生成详细总结”按钮", false)]);
        const submitted = actions.some(action => action.type === "click" && (action.control?.name || "").trim() === START_GENERATION_LABEL);
        if (!submitted) {
          return taskResult([checkResult(`点击“${NOTE_PAPER.title}”进入概览`, true), checkResult("点击“生成详细总结”按钮", true), checkResult("在弹出的对话框里点击“开始生成”", false)]);
        }
        const notePath = await paperNotePath(fixture, NOTE_PAPER.id);
        const content = await pollFile(() => readFile(notePath, "utf8"), value => typeof value === "string" && value.length > 0);
        const index = await pollFile(() => readPaperIndex(fixture), value => value?.papers?.[NOTE_KEY]?.detail === true);
        const entry = index?.papers?.[NOTE_KEY];
        const indexed = entry?.detail === true && typeof entry.paperPath === "string";
        const shownInUi = observation.controls.some(control => (control.name || "").trim() === READ_DETAIL_SUMMARY_LABEL);
        return taskResult([
          checkResult(`点击“${NOTE_PAPER.title}”进入概览`, true),
          checkResult("点击“生成详细总结”按钮", true),
          checkResult("在弹出的对话框里点击“开始生成”", true),
          checkResult("真实的论文总结文件已写出", Boolean(content) && content.length > 100 && content.includes(NOTE_PAPER.id), content?.slice(0, 200)),
          checkResult("独立论文索引记录了该总结路径", indexed, entry),
          checkResult("概览界面出现“阅读详细总结”链接", shownInUi),
        ]);
      },
    },
    {
      id: GOAL_TASK_IDS[5], title: "设置冲突后放弃草稿并重新打开",
      goal: `打开设置，把 ${MODEL_FIELD_LABEL} 字段改为“${EXTERNAL_CONFLICT_DRAFT_MODEL}”。测试环境会在设置加载完成后立刻在磁盘上修改配置文件，这会让自动保存因“配置已改变”失败，界面上会出现“${DISCARD_UNSAVED_AND_CLOSE_LABEL}”按钮。不要点击“${RETRY_SAVE_LABEL}”：点击“${DISCARD_UNSAVED_AND_CLOSE_LABEL}”，再点击随后出现的确认按钮“${CONFIRM_DISCARD_AND_CLOSE_LABEL}”，设置会关闭。关闭后重新点击“${SETTINGS_BUTTON_LABEL}”打开设置，确认 ${MODEL_FIELD_LABEL} 字段显示的是磁盘上的最新值，而不是你刚才填写的草稿。`,
      uiLabels: [
        uiLabel(MODEL_FIELD_LABEL, "apps/cli/src/workbench/web/model-combobox.ts"),
        uiLabel(DISCARD_UNSAVED_AND_CLOSE_LABEL, "apps/cli/src/workbench/web/settings.ts"),
        uiLabel(CONFIRM_DISCARD_AND_CLOSE_LABEL, "apps/cli/src/workbench/web/settings.ts"),
        uiLabel(SETTINGS_BUTTON_LABEL, "apps/cli/src/workbench/web/app.ts"),
        uiLabel(RETRY_SAVE_LABEL, "apps/cli/src/workbench/web/settings.ts"),
      ],
      async prepare() {
        externalConflictWritten = false;
      },
      async verify({ observation, actions }) {
        // Triggered by the settings form actually being rendered (the model field is visible), never by
        // merely observing a click on "设置": that guarantees the dialog's own GET /api/settings (which
        // captures the revision this task's conflict needs to be stale against) has already completed,
        // so this write always lands strictly after that read, never racing it.
        if (!externalConflictWritten && observation.controls.some(control => control.field === "model")) {
          externalConflictWritten = true;
          await writeConfigPatch(fixture, table => { table.llm = { ...table.llm, model: EXTERNAL_CONFLICT_EXTERNAL_MODEL }; });
        }
        const filledDraft = actions.some(action => action.type === "fill" && action.control?.field === "model" && action.value === EXTERNAL_CONFLICT_DRAFT_MODEL);
        if (!filledDraft) return taskResult([checkResult(`把 ${MODEL_FIELD_LABEL} 字段改为“${EXTERNAL_CONFLICT_DRAFT_MODEL}”`, false)]);
        const discardClicked = actions.some(action => action.type === "click" && (action.control?.name || "").trim() === DISCARD_UNSAVED_AND_CLOSE_LABEL);
        if (!discardClicked) {
          return taskResult([checkResult(`把 ${MODEL_FIELD_LABEL} 字段改为“${EXTERNAL_CONFLICT_DRAFT_MODEL}”`, true), checkResult("看到保存失败后点击“放弃未保存修改并关闭”", false)]);
        }
        const confirmIndex = actions.findIndex(action => action.type === "click" && (action.control?.name || "").trim() === CONFIRM_DISCARD_AND_CLOSE_LABEL);
        if (confirmIndex === -1) {
          return taskResult([checkResult(`把 ${MODEL_FIELD_LABEL} 字段改为“${EXTERNAL_CONFLICT_DRAFT_MODEL}”`, true), checkResult("看到保存失败后点击“放弃未保存修改并关闭”", true), checkResult("点击确认按钮“放弃修改并关闭”", false)]);
        }
        const reopened = actions.slice(confirmIndex + 1).some(action => action.type === "click" && (action.control?.name || "").trim() === SETTINGS_BUTTON_LABEL);
        if (!reopened) {
          return taskResult([checkResult(`把 ${MODEL_FIELD_LABEL} 字段改为“${EXTERNAL_CONFLICT_DRAFT_MODEL}”`, true), checkResult("看到保存失败后点击“放弃未保存修改并关闭”", true), checkResult("点击确认按钮“放弃修改并关闭”", true), checkResult("重新点击“设置”打开设置", false)]);
        }
        const config = await settings(fixture);
        const onDisk = config.llm?.model === EXTERNAL_CONFLICT_EXTERNAL_MODEL;
        const shownValue = observation.controls.find(control => control.field === "model")?.value;
        return taskResult([
          checkResult(`把 ${MODEL_FIELD_LABEL} 字段改为“${EXTERNAL_CONFLICT_DRAFT_MODEL}”`, true),
          checkResult("看到保存失败后点击“放弃未保存修改并关闭”", true),
          checkResult("点击确认按钮“放弃修改并关闭”", true),
          checkResult("重新点击“设置”打开设置", true),
          checkResult("磁盘上的配置仍是外部修改后的值，模型的草稿没有被保存", onDisk, config.llm?.model),
          checkResult("重新打开后界面显示的是外部修改后的值", shownValue === EXTERNAL_CONFLICT_EXTERNAL_MODEL, shownValue),
        ]);
      },
    },
  ];
}

/** P3 Chunk 5: the shared personas (exploration-tasks/personas.mjs), worded for the workbench surface. */
export function personaTasks() {
  return sharedPersonaTasks("workbench");
}
