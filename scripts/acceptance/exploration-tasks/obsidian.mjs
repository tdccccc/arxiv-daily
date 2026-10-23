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
import { dailyReportFile, entryFor, HISTORY_FILE, INDEX_FILE, paperNoteFile, SETTINGS_FILE, STATE_FILE } from "../exploration-obsidian.mjs";
import { poll, pollJsonFile } from "../obsidian-ui.mjs";
import { checkResult, exactLabelPattern, taskResult, uiLabel } from "./checks.mjs";
import { seedClusterableLibraryPaper, startLibraryFixture } from "./library.mjs";
import { PERSONA_TASK_IDS, personaTasks as sharedPersonaTasks } from "./personas.mjs";

/** Single source of truth for the default Obsidian goal-task ids, usable without a fixture (e.g. --list). */
export const GOAL_TASK_IDS = [
  "daily-paper-limit", "star-paper-from-report",
  "add-topic-direction", "remove-topic-direction", "add-arxiv-category",
  "run-for-date", "cancel-then-rerun", "auth-failure-retry", "completed-date-idempotent",
  // P3 Chunk 4: reading tasks.
  "dashboard-tabs-paging", "mark-to-read", "create-paper-note", "summarize-by-id", "open-today-report",
  // Exploration safety and coverage P3: personal literature library tasks.
  "library-revoke-consent", "library-remote-consent",
  // Exploration safety and coverage P3 Chunk 2: index build, library search and similar-paper tasks.
  "library-build-index", "library-search", "library-similar-papers",
  // Exploration safety and coverage P3 Chunk 3: Direction candidate review tasks.
  "library-accept-direction", "library-edit-candidate",
];

/** P3 Chunk 5: re-exports the shared persona id list (exploration-tasks/personas.mjs) so `--list`/selection keep reading it from each host's own module, unchanged from before chunk 5 wired it up. */
export { PERSONA_TASK_IDS };

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

// ─── P3 Chunk 4: Obsidian reading tasks ──────────────────────────────────────
// Shared by dashboard-tabs-paging (the fixture's other papers) and create-paper-note (the "other" fixture
// paper's pre-seeded note): a minimal but real Paper note file — frontmatter identity (`arxiv_id`) plus six
// H2 sections — matching what plugin/src/dashboard/detail-refs.ts's `isExpectedGeneratedDetailSummary`
// (packages/core's `classifyPaperNote`/`looksLikeDetailSummary`) actually requires to recognize a file as a
// real detail summary: an H1, and at least 3 of its known H2 headings (not merely an index flag — see the
// dashboard-tabs-paging comment below for why writing directly to papers.json was not sufficient).
const DETAIL_NOTE_SECTIONS = ["研究问题", "方法设计", "核心结果", "主要结论", "关键图表", "适用边界"];
const detailNoteMarkdown = (id, title) => `---\ntitle: "${title}"\nauthors: "A. Researcher"\narxiv_id: "${id}"\nprimary_topic: Pagination\ntags: []\n---\n\n# ${title}\n\n${DETAIL_NOTE_SECTIONS.map(section => `## ${section}\n\nSeeded fixture placeholder content for ${id}, long enough to be recognized as a real detail summary body rather than an empty stub, describing the same controlled research measurement in each section.\n`).join("\n")}`;

// dashboard-tabs-paging: the fixture only ever serves 2 unique papers (fixtures.mjs's PAPERS), far short
// of the "All" tab's default page size (20, plugin/src/dashboard/constants.ts PAGE_SIZE_OPTIONS — not
// settings-persisted, so there is no settings patch that could lower it). A first attempt seeded
// PAGINATION_PAPER_COUNT synthetic papers.json entries directly; a real-host run showed this does NOT
// work — the Dashboard rebuilds its paper list from scratch on every load by scanning real vault Markdown
// (packages/core/src/dashboard/history-sync.ts's `syncDashboardHistory`: `collectDailyCandidates` parses
// daily-report files, `collectPaperCandidates` parses papersDir Paper notes), and prunes any index entry
// that isn't backed by one of those — a synthetic entry pointing at nothing real simply vanishes on the
// next reload. prepare() instead writes one real daily-report-shaped Markdown file (the format
// `packages/core/src/pipeline/daily-summary-rendering.ts`'s `renderPaperHeader` actually produces: an `##`
// topic heading matching the fixture's own topic name, then one `###` paper heading per synthetic id with
// an `- **arXiv**: [id](https://arxiv.org/abs/id)` line, which is exactly what `extractArxivIds`'s first
// pattern matches) plus one real Paper note per synthetic id (`detailNoteMarkdown`, so every synthetic row
// already shows "Open detail summary" rather than "Summarize by arXiv ID" — keeping create-paper-note's own
// precondition, that exactly one row shows that action, true regardless of task order). Both are picked up
// by the normal vault scan after a plugin reload, the same path a real generated report/paper note takes.
// Titles are zero-padded ("01".."30") because every synthetic paper shares one report date, so the default
// sort (priority tie, then published desc, then **title ascending**) falls through to lexicographic title
// order — unpadded "10" would sort before "2". The fixture's own 2 real papers (published in 2026, later
// than the synthetic "2024-01-15" report date) always sort before all synthetic rows, so the last synthetic
// paper is still guaranteed past the first 20 rows regardless of how many real papers already exist.
export const PAGINATION_PAPER_COUNT = 30;
export const PAGINATION_REPORT_DATE = "2024-01-15";
const paginationPaperId = i => `9901.${String(i).padStart(5, "0")}`;
const paginationPaperTitle = i => `Synthetic pagination paper ${String(i).padStart(2, "0")}`;
export const PAGINATION_FIRST_PAPER_ID = paginationPaperId(1);
export const PAGINATION_LAST_PAPER_ID = paginationPaperId(PAGINATION_PAPER_COUNT);

// mark-to-read: the only mark the plugin's PaperStatus model offers that reads as "to read" is "to_read"
// (plugin/src/commands.ts PAPER_MARKS — "Watch" sets status "to_read"/priority "normal"); the one UI path
// to it (the "Set paper mark…" command's PaperMarkModal) was not wired into any dashboard control before
// this chunk — see the product change in plugin/src/dashboard/view.ts's showMoreMenu (new "Set paper
// mark…" item reusing the existing addCommandMenuItem helper and the command's own already-`.arxiv-daily-
// modal`-marked modal). Uses PAPERS[1], not PAPERS[0] (star-paper-from-report's paper), to stay independent
// of that task's own priority toggling. `status`/`priority` are user marks, preserved as-is by the vault
// scan above (unlike `detail`/`paperPath`, they are not recomputed from file presence — confirmed by
// reading packages/core/src/services/paper-index.ts's `upsertEntry`), so no reset is needed between passes.
export const MARK_TO_READ_ID = PAPERS[1].id;

// create-paper-note: the dashboard's per-row "Summarize by arXiv ID" action (sparkles icon, shown only
// when the row has no existing detail summary) already drives manualFetch.fetchAndSummarize for a named
// paper with no product change needed. But that action button carries no per-paper identifying DOM
// attribute (unlike the star toggle's `data-arxiv-id`) — every row with no detail summary shows the exact
// same label — so prepare() must leave exactly one such row in the table: CREATE_PAPER_NOTE_ID's own note
// is cleared, and the fixture's other paper is given a pre-seeded detail summary (so its row shows "Open
// detail summary" instead), independently of whatever order this task runs in across repeated passes.
export const CREATE_PAPER_NOTE_ID = PAPERS[0].id;
export const CREATE_PAPER_NOTE_OTHER_ID = PAPERS[1].id;

// summarize-by-id: a fresh id the fixture serves generically (fixtures.mjs's atom()/paperHtml() fall back
// to a synthetic title/abstract/content for any arXiv-shaped id outside PAPERS), deliberately not one of
// the two fixture papers already in the dashboard — the point of this task is the toolbar's "by ID" entry
// for a paper the model has not otherwise seen, as opposed to create-paper-note's already-listed row.
export const SUMMARIZE_BY_ID_TARGET = "2610.20001";

// open-today-report: ACCEPTANCE_DATE's report (already deterministically established by
// star-paper-from-report/completed-date-idempotent's own prepare(), and re-asserted independently here)
// rather than the wall-clock "today" — the fixture's "recent" arXiv listing only ever serves fixtures.mjs's
// own hardcoded DATES, so a run for the real current date would list zero papers. The calendar defaults to
// the latest report's month, which is the same October-2026 month as the real host's current month, so no
// month navigation is needed to reach ACCEPTANCE_DATE's cell.

// ─── P2/P6: declared UI labels ────────────────────────────────────────────────
// scripts/tests/acceptance-exploration-labels.test.mjs asserts every task's own `uiLabels` entry (every
// product UI label its goal text names or its verify() matches on) is still present, as a complete,
// case-sensitive string literal, in the specific product source file(s) named by its own `uiLabel(text,
// source)` wrapper (P6; P2 originally searched all of plugin/src/** case-insensitively) — so a rename
// shows up as a failing fixed test instead of only a blocked, paid real-model run. Shared the same way
// the fixture topic/direction constants above are: every verify() that matches one of these reads the
// exact constant below, never a second, independently-typed copy of the same string, so the goal text,
// the verifier and the declared label cannot drift apart.
const DAILY_PAPER_LIMIT_LABEL = "Daily paper limit";
const STAR_PAPER_LABEL = "Star paper";
const ALL_TAB_LABEL = "All";
// plugin/src/dashboard/view.ts's getDisplayText() renders this "Reading dashboard" (lowercase "d"); kept
// lowercase here too — P6's check is case-sensitive, unlike the regex-matched labels below which stay
// case-insensitive (matching a control's accessible name, not a source literal).
const READING_DASHBOARD_LABEL = "Reading dashboard";
const ADD_DIRECTION_LABEL = "Add direction";
const REMOVE_DIRECTION_BUTTON_LABEL = "×";
const ARXIV_CATEGORIES_SECTION_LABEL = "arXiv categories";
const ADD_CATEGORY_LABEL = "Add category";
const MORE_ACTIONS_LABEL = "More arXiv Daily actions";
const RUN_FOR_DATE_LABEL = "Run for date…";
const DATE_FIELD_LABEL = "Date";
const CANCEL_ACTIVE_TASKS_LABEL = "Cancel active tasks";
const SHOW_LOGS_AND_HISTORY_LABEL = "Show logs & history";
const RUN_HISTORY_LABEL = "Run History";
// The menu item is this plus an ellipsis ("Force run for date…"); the dialog's own submit button is the
// shorter FORCE_RUN_BUTTON_LABEL below. Kept without the ellipsis so it is still a literal substring of
// the menu item's label — the ellipsis itself is handled by the verifier's own regex, unchanged from
// before this chunk.
const FORCE_RUN_FOR_DATE_LABEL = "Force run for date";
const FORCE_RUN_BUTTON_LABEL = "Force run";
const SET_PAPER_MARK_LABEL = "Set paper mark…";
const PAPER_ID_OR_URL_LABEL = "Paper ID or URL";
const MARK_FIELD_LABEL = "Mark";
const WATCH_MARK_LABEL = "Watch";
const SET_MARK_LABEL = "Set mark";
const NEXT_PAGE_LABEL = "Next page";
const SUMMARIZE_BY_ARXIV_ID_LABEL = "Summarize by arXiv ID";
const SUMMARIZE_BY_ID_LABEL = "Summarize by ID";
const SUMMARIZE_LABEL = "Summarize";
const DAILY_REPORTS_LABEL = "Daily reports";
// The library row's Revoke button (core's libraryRowPresentation) and the revocation dialog's confirm button
// (plugin/src/library/modal.ts) share this text; both sources are checked.
const REVOKE_LABEL = "Revoke";
// The Embedding mode dropdown's name: a literal in the legacy settings tab and a field name in core's
// settings schema; both sources are checked. The consent confirm button's text does not depend on
// processing depth (plugin/src/library/modal.ts, libraryAuthorizationCopy).
const EMBEDDING_MODE_LABEL = "Embedding";
const SEND_TITLES_AND_ABSTRACTS_LABEL = "Send titles and abstracts";
// The library row's primary button when there is no completed index run or a preparation error; the
// text comes from core's libraryRowPresentation (packages/core/src/library/library-connection.ts).
const RETRY_PREPARATION_LABEL = "Retry preparation";
// The Dashboard's own search box (its accessible name is the <label> wrapping it — view.ts's
// createFilterField(filters, "Search", ...)), the async "Library matches" result block it feeds
// (library-search-block.ts's own heading text) and each result row's "Open PDF" action (same file,
// `const action = "Open PDF"`).
const SEARCH_FIELD_LABEL = "Search";
const LIBRARY_MATCHES_LABEL = "Library matches";
const OPEN_PDF_LABEL = "Open PDF";
// Each Dashboard row's own "Find similar papers" action (view.ts) and the SimilarPapersModal's "Library
// similar" tab button (similar-papers-modal.ts) — selected by default when the modal opens, but the
// goal still asks the model to click it once, for an observed action instead of relying on a default
// that could change later.
const FIND_SIMILAR_PAPERS_LABEL = "Find similar papers";
const LIBRARY_SIMILAR_TAB_LABEL = "Library similar";
// A term literally present in the fixture's own PAPERS[0] (fixtures.mjs: title "Reliable measurement
// with controlled experiments", abstract "...repeatable research measurements.") — both the title and
// the abstract contain this substring, which is exactly what the fixture's `/embeddings` handler keys
// its vector on (`String(value).includes("measurement") ? [1,0,0] : [0.8,0.6,0]`, fixtures.mjs), so a
// query embedding of this term scores a perfect cosine match against that same paper's own indexed
// chunks — guaranteeing at least one non-empty "Library matches" result without depending on any
// particular ranking/threshold behavior.
const LIBRARY_SEARCH_QUERY = "measurement";
// Exploration safety and coverage P3 Chunk 3 (Direction candidate review tasks): the "Personal library"
// section heading and the "Topics from library" row's own name (plugin/src/settings/tab.ts:1876,1884;
// `definitions.ts:226` for the 1.13+ declarative path, which names the row identically), and its
// "Review suggestions" button (tab.ts:767's `renderLibrarySuggestionsControls`) — opens the review
// dialog, auto-generating a proposal on first use (`openPersonalLibraryDirectionReview({
// generateIfMissing: true })`, tab.ts:774) when the index already has papers, exactly the precondition
// library-accept-direction's own prepare() establishes.
const PERSONAL_LIBRARY_SECTION_LABEL = "Personal library";
const TOPICS_FROM_LIBRARY_LABEL = "Topics from library";
const REVIEW_SUGGESTIONS_LABEL = "Review suggestions";
// The accept bar's own button (interest-profile-modal.ts:405), only enabled once at least one proposed
// direction is selected for a selected topic. Real-host evidence (2026-10-13): this button also carries
// a `title` attribute (its own tooltip text), and the shared Obsidian observer's `resolveName()`
// (exploration-obsidian.mjs) reads `title` before a <button>'s own text content — so the control's
// *observed* accessible name is the longer tooltip sentence below, never the visible "Add to research
// topics" label. The button is still fully observable and clickable; a model (or this task's own
// verify()) matching on the visible label alone would never find it. Both strings are declared and
// matched so a rename of either one is still caught by the label-literalness check.
const ADD_TO_RESEARCH_TOPICS_LABEL = "Add to research topics";
const ADD_TO_RESEARCH_TOPICS_TITLE = "Adds the selected topics with only their checked directions to your research topics. You can edit or remove them there afterwards.";
// interest-profile-modal.ts's `renameTopic`'s own "Rename topic" <details> disclosure and its "Topic
// name" field (both the aria-label and the wrapping <label>'s own text, lines 480-484) — library-edit-
// candidate's rename path. "Edit" is `createCardDetail`'s own per-candidate-card disclosure summary
// (line ~744) that must be expanded before its "Discard" button (the per-candidate card's own button,
// line 930) becomes reachable — library-edit-candidate's discard path.
const RENAME_TOPIC_LABEL = "Rename topic";
const TOPIC_NAME_FIELD_LABEL = "Topic name";
const CARD_EDIT_LABEL = "Edit";
const DISCARD_LABEL = "Discard";
/** The new topic name library-edit-candidate's rename path asks the model to type, verified byte-for-byte on disk. */
const LIBRARY_EDIT_NEW_TOPIC_NAME = "文献库候选主题（已重命名）";

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
  // Leftover-state audit (docs/helm/2026-10-13-exploration-safety-and-coverage open question 2): run-history.jsonl
  // is append-only and survives reset() and clearDateState(), so a verify() that searches the whole file can
  // count a cancellation or a forced completion that an earlier pass of this task recorded. Each task's
  // prepare() records how many history lines already existed, and verify() counts only the lines after that.
  let cancelHistoryBaseline = 0, authHistoryBaseline = 0;
  let idempotentBaseline;
  // Exploration safety and coverage P3 Chunk 2: library-build-index's own baseline — the fixture's
  // request count at the moment prepare() finishes connecting the library, so verify() can tell whether
  // any embeddings request happened *after* this task actually started, not merely that one exists
  // somewhere in the fixture's full history (e.g. left over from a different task in the same pass).
  let libraryBuildIndexBaseline;
  // Exploration safety and coverage P3 Chunk 3: library-edit-candidate's own baseline — the real
  // proposal's first topic/candidate identity, captured once by prepare()'s own real generation (never
  // the model's claim of which one it edited), so verify() can tell this exact topic's name or candidate
  // actually changed on disk, not merely that *some* proposal document changed.
  let libraryEditCandidateBaseline;
  return [
    {
      id: GOAL_TASK_IDS[0], title: "修改每日论文数量上限",
      goal: `打开插件设置页，把 ${DAILY_PAPER_LIMIT_LABEL} 改为 ${target}。移开焦点，等待保存完成即可，不需要关闭设置；不要修改路径、模型连接、密钥或邮件相关设置。`,
      uiLabels: [uiLabel(DAILY_PAPER_LIMIT_LABEL, "plugin/src/settings/declarative-rows.ts")],
      async verify({ driver, observation, actions }) {
        const filled = actions.some(action => action.type === "fill" && action.control?.name === DAILY_PAPER_LIMIT_LABEL && action.value === String(target));
        if (!filled) return taskResult([checkResult(`通过设置页将 ${DAILY_PAPER_LIMIT_LABEL} 改为 ${target}`, false)]);
        const onDisk = await pollJsonFile(() => driver.jsonFile(SETTINGS_FILE), value => value?.settings?.output?.maxDailyPapers === target, "daily paper limit on disk", { timeoutMs: 5000 }).catch(() => null);
        const persisted = onDisk?.settings?.output?.maxDailyPapers === target;
        if (!persisted) {
          return taskResult([checkResult(`通过设置页将 ${DAILY_PAPER_LIMIT_LABEL} 改为 ${target}`, true), checkResult("隔离设置文件保存了新上限", false)]);
        }
        // Only the test program closes and reopens settings (the model has no exposed control for Obsidian's
        // own settings chrome — the plugin's observed surface is scoped to its own tab container), and only
        // once persistence is already confirmed, so an in-progress model action is never interrupted by this.
        // Reading the re-rendered value is polled, not read once: a real host briefly shows the previous
        // render while the declarative settings page re-paints after re-opening.
        reopenCheck ??= (async () => {
          await driver.closeThenReopenSettings();
          const value = await poll(
            () => driver.read(`Array.from(document.querySelectorAll("input")).find(el => el.getAttribute("aria-label") === ${JSON.stringify(DAILY_PAPER_LIMIT_LABEL)})?.value`),
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
            () => driver.read(`Array.from(document.querySelectorAll("input")).find(el => el.getAttribute("aria-label") === ${JSON.stringify(DAILY_PAPER_LIMIT_LABEL)})?.value`),
            value => value === String(target), "re-rendered daily paper limit after a full plugin reload", { timeoutMs: 3000 },
          ).catch(() => undefined);
          return value;
        })();
        const reloaded = await reloadCheck;
        return taskResult([
          checkResult(`通过设置页将 ${DAILY_PAPER_LIMIT_LABEL} 改为 ${target}`, true),
          checkResult("隔离设置文件保存了新上限", persisted, onDisk.settings.output.maxDailyPapers),
          checkResult("重新打开设置后界面仍显示新上限", reopened === String(target), reopened),
          checkResult("完整重载插件后设置页仍显示新上限", reloaded === String(target), reloaded),
        ]);
      },
    },
    {
      id: GOAL_TASK_IDS[1], title: "在阅读面板收藏一篇论文",
      goal: `打开阅读面板（${READING_DASHBOARD_LABEL}）的 ${ALL_TAB_LABEL} 列表，找到 arXiv ${paperId} 对应行，点击它的 ${STAR_PAPER_LABEL} 按钮将其标记为收藏。`,
      uiLabels: [
        uiLabel(READING_DASHBOARD_LABEL, "plugin/src/dashboard/view.ts"),
        uiLabel(ALL_TAB_LABEL, "plugin/src/dashboard/constants.ts"),
        uiLabel(STAR_PAPER_LABEL, "plugin/src/dashboard/actions.ts"),
      ],
      async prepare({ driver }) {
        await driver.ensureDailyReport(ACCEPTANCE_DATE);
        await driver.ensurePaperUnstarred(paperId);
      },
      async verify({ driver, actions }) {
        const starred = actions.some(action => action.type === "click" && action.control?.paperKey === paperId && new RegExp(STAR_PAPER_LABEL, "i").test(action.control?.name || ""));
        if (!starred) return taskResult([checkResult(`点击 arXiv ${paperId} 的 ${STAR_PAPER_LABEL} 收藏按钮`, false)]);
        const onDisk = await pollJsonFile(() => driver.jsonFile(INDEX_FILE), value => entryFor(value, paperId)?.priority === "high", "starred priority on disk", { timeoutMs: 5000 }).catch(() => null);
        const marked = entryFor(onDisk, paperId)?.priority === "high";
        return taskResult([
          checkResult(`点击 arXiv ${paperId} 的 ${STAR_PAPER_LABEL} 收藏按钮`, true),
          checkResult("插件的收藏标记已写入论文索引", marked, onDisk ? entryFor(onDisk, paperId)?.priority : undefined),
        ]);
      },
    },
    // ─── P3 Chunk 2: Research topic/direction editing ─────────────────────
    {
      id: "add-topic-direction", title: "新增研究方向",
      goal: `打开插件设置页，展开研究主题“${FIXTURE_TOPIC_NAME}”（点击该主题的标题行即可展开），点击 ${ADD_DIRECTION_LABEL} 新增一个方向，并将其内容填写为：“${NEW_DIRECTION_TEXT}”。不要修改已有的方向“${FIXTURE_DIRECTION_TEXT}”，也不要新增或删除任何主题。`,
      uiLabels: [uiLabel(ADD_DIRECTION_LABEL, "plugin/src/settings/tab.ts")],
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
      goal: `插件设置页中，研究主题“${FIXTURE_TOPIC_NAME}”现在有两个方向：“${FIXTURE_DIRECTION_TEXT}”和“${SEEDED_DIRECTION_TEXT}”。展开该主题，删除“${SEEDED_DIRECTION_TEXT}”这一条（点击它旁边的 ${REMOVE_DIRECTION_BUTTON_LABEL} 按钮），保留“${FIXTURE_DIRECTION_TEXT}”不变。`,
      uiLabels: [uiLabel(REMOVE_DIRECTION_BUTTON_LABEL, "plugin/src/settings/tab.ts")],
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
      goal: `打开插件设置页，在 ${ARXIV_CATEGORIES_SECTION_LABEL} 区域点击 ${ADD_CATEGORY_LABEL} 新增一行，并把新增的下拉框选为 ${TARGET_CATEGORY_ID}（${TARGET_CATEGORY_LABEL}）。不要移除或修改已有的 cs.AI。`,
      uiLabels: [
        uiLabel(ARXIV_CATEGORIES_SECTION_LABEL, "plugin/src/settings/tab.ts"),
        uiLabel(ADD_CATEGORY_LABEL, "plugin/src/settings/tab.ts"),
      ],
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
      goal: `点击仪表盘工具栏的“${MORE_ACTIONS_LABEL}”按钮打开菜单，点击其中的“${RUN_FOR_DATE_LABEL}”，在弹出的日期对话框里把 ${DATE_FIELD_LABEL} 填为 ${RUN_FOR_DATE_DATE}，再点击 Run 提交，等待生成完成。`,
      uiLabels: [
        uiLabel(MORE_ACTIONS_LABEL, "plugin/src/dashboard/view.ts"),
        uiLabel(RUN_FOR_DATE_LABEL, "plugin/src/dashboard/view.ts"),
        uiLabel(DATE_FIELD_LABEL, "plugin/src/date-picker-modal.ts"),
        uiLabel("Run", "plugin/src/date-picker-modal.ts"),
      ],
      // Deterministically starts this date "not yet generated" (never through the model): clears any
      // run-state entry, daily report file and paper-index links a previous pass of this same task left
      // behind, so a second full-library pass in one session still begins from the same untouched state.
      async prepare({ driver }) {
        await driver.clearDateState(RUN_FOR_DATE_DATE);
        await driver.openDashboard();
      },
      async verify({ driver, actions }) {
        const submitted = actions.some(action => action.type === "fill" && action.control?.name === DATE_FIELD_LABEL && action.value === RUN_FOR_DATE_DATE);
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
      goal: `点击“${MORE_ACTIONS_LABEL}”打开菜单，点击“${RUN_FOR_DATE_LABEL}”，日期填 ${CANCEL_THEN_RERUN_DATE} 并提交；测试环境会让这次运行暂停在处理中。再次打开菜单，点击“${CANCEL_ACTIVE_TASKS_LABEL}”取消这次运行。取消完成后，再次打开菜单点击“${RUN_FOR_DATE_LABEL}”，同样填入 ${CANCEL_THEN_RERUN_DATE} 并提交，让这次真正跑完。`,
      uiLabels: [
        uiLabel(MORE_ACTIONS_LABEL, "plugin/src/dashboard/view.ts"),
        uiLabel(RUN_FOR_DATE_LABEL, "plugin/src/dashboard/view.ts"),
        uiLabel(DATE_FIELD_LABEL, "plugin/src/date-picker-modal.ts"),
        uiLabel(CANCEL_ACTIVE_TASKS_LABEL, "plugin/src/dashboard/view.ts"),
      ],
      async prepare({ driver }) {
        await driver.clearDateState(CANCEL_THEN_RERUN_DATE);
        cancelHistoryBaseline = (await readHistory(driver).catch(() => [])).length;
        // Program-only fixture control (mirrors obsidian-scenarios.mjs's cancelAndRepeat): holds the
        // model request open so there is a real window in which the model's cancel click matters,
        // instead of racing a run that could complete before the model ever reaches the menu.
        driver.fixture.server.setMode({ llm: "hold" });
        await driver.openDashboard();
      },
      async verify({ driver, actions }) {
        const firstSubmitted = actions.some(action => action.type === "fill" && action.control?.name === DATE_FIELD_LABEL && action.value === CANCEL_THEN_RERUN_DATE);
        if (!firstSubmitted) return taskResult([checkResult(`通过 Run for date 提交 ${CANCEL_THEN_RERUN_DATE}`, false)]);
        const cancelClickIndex = actions.findIndex(action => action.type === "click" && exactLabelPattern(CANCEL_ACTIVE_TASKS_LABEL).test((action.control?.name || "").trim()));
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
        const cancelHistory = await pollHistory(driver, records => records.slice(cancelHistoryBaseline).some(record => record.date === CANCEL_THEN_RERUN_DATE && record.resultKind === "cancelled"), "cancelled run-history record").then(records => records.slice(cancelHistoryBaseline)).catch(() => []);
        const cancelRecorded = cancelHistory.some(record => record.date === CANCEL_THEN_RERUN_DATE && record.resultKind === "cancelled");
        const rerunSubmitted = actions.slice(cancelClickIndex + 1).some(action => action.type === "fill" && action.control?.name === DATE_FIELD_LABEL && action.value === CANCEL_THEN_RERUN_DATE);
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
      goal: `点击“${MORE_ACTIONS_LABEL}”打开菜单，点击“${RUN_FOR_DATE_LABEL}”，日期填 ${AUTH_FAILURE_RETRY_DATE} 并提交；这次运行会因认证失败而结束。打开菜单，点击“${SHOW_LOGS_AND_HISTORY_LABEL}”，在弹出的面板里点击“${RUN_HISTORY_LABEL}”标签查看这次失败的记录，然后关闭面板。确认失败原因后，再次打开菜单点击“${FORCE_RUN_FOR_DATE_LABEL}…”，同样填入 ${AUTH_FAILURE_RETRY_DATE} 并提交以重新运行。`,
      uiLabels: [
        uiLabel(MORE_ACTIONS_LABEL, "plugin/src/dashboard/view.ts"),
        uiLabel(RUN_FOR_DATE_LABEL, "plugin/src/dashboard/view.ts"),
        uiLabel(DATE_FIELD_LABEL, "plugin/src/date-picker-modal.ts"),
        uiLabel(SHOW_LOGS_AND_HISTORY_LABEL, "plugin/src/dashboard/view.ts"),
        uiLabel(RUN_HISTORY_LABEL, "plugin/src/dashboard/hub-modal.ts"),
        uiLabel(`${FORCE_RUN_FOR_DATE_LABEL}…`, "plugin/src/dashboard/view.ts"),
      ],
      // Design: the API key field is restricted/sensitive and off-limits to the model (classify
      // ControlRestriction in exploration-obsidian.mjs), so "fix the credential" is not an action the
      // model can take. Instead the fixture itself is only restored to accepting requests once the model
      // has done the one allowed, observable thing that matters here — actually reviewing the recorded
      // failure (opening the Hub modal's own Run History tab) — not merely clicking Force run blindly.
      // Until that review happens, Force run would still hit the same 401 and fail again, so verify
      // cannot pass without the model's specific review action; this is checked below, not assumed.
      async prepare({ driver }) {
        await driver.clearDateState(AUTH_FAILURE_RETRY_DATE);
        authHistoryBaseline = (await readHistory(driver).catch(() => [])).length;
        driver.fixture.server.setMode({ llm: "unauthorized" });
        driver.expectAuthorizationFailure();
        await driver.openDashboard();
      },
      async verify({ driver, actions }) {
        const submitted = actions.some(action => action.type === "fill" && action.control?.name === DATE_FIELD_LABEL && action.value === AUTH_FAILURE_RETRY_DATE);
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
        const reviewedHistory = actions.some(action => action.type === "click" && exactLabelPattern(RUN_HISTORY_LABEL).test((action.control?.name || "").trim()));
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
        const forceSubmitted = actions.some(action => action.type === "click" && new RegExp(`^${FORCE_RUN_FOR_DATE_LABEL}…?$`, "i").test((action.control?.name || "").trim()));
        const retrySubmitted = actions.filter(action => action.type === "fill" && action.control?.name === DATE_FIELD_LABEL && action.value === AUTH_FAILURE_RETRY_DATE).length >= 2;
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
        const history = (await readHistory(driver).catch(() => [])).slice(authHistoryBaseline);
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
      goal: `该日期（${ACCEPTANCE_DATE}）已经生成过日报。点击“${MORE_ACTIONS_LABEL}”打开菜单，点击“${RUN_FOR_DATE_LABEL}”（不是 ${FORCE_RUN_BUTTON_LABEL}），日期填 ${ACCEPTANCE_DATE} 并提交；预期它会被判定为已完成而跳过，不会重新生成。`,
      uiLabels: [
        uiLabel(MORE_ACTIONS_LABEL, "plugin/src/dashboard/view.ts"),
        uiLabel(RUN_FOR_DATE_LABEL, "plugin/src/dashboard/view.ts"),
        uiLabel(DATE_FIELD_LABEL, "plugin/src/date-picker-modal.ts"),
        uiLabel(FORCE_RUN_BUTTON_LABEL, "plugin/src/dashboard/view.ts"),
      ],
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
        const submitted = actions.some(action => action.type === "fill" && action.control?.name === DATE_FIELD_LABEL && action.value === ACCEPTANCE_DATE);
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
    // ─── P3 Chunk 4: reading tasks ──────────────────────────────────────────
    {
      id: "dashboard-tabs-paging", title: "切换到 All 标签并翻到第二页",
      goal: `打开阅读面板，点击 ${ALL_TAB_LABEL} 标签，然后点击分页区域的 ${NEXT_PAGE_LABEL} 按钮，翻到第 2 页。`,
      uiLabels: [
        uiLabel(ALL_TAB_LABEL, "plugin/src/dashboard/constants.ts"),
        uiLabel(NEXT_PAGE_LABEL, "plugin/src/dashboard/view.ts"),
      ],
      // Writes one real daily-report-shaped Markdown file plus one real Paper note per synthetic id (never
      // through the model), so the normal vault scan (not a direct papers.json write — see this file's own
      // comment above for why that does not survive a reload) indexes enough papers that the existing
      // default page size (20) alone produces a second page.
      async prepare({ driver }) {
        const lines = [
          "---", `date: ${PAGINATION_REPORT_DATE}`, "tags: [arxiv, daily]", "---", "",
          `# arXiv cs.AI Daily Digest ${PAGINATION_REPORT_DATE}`,
          `${PAGINATION_PAPER_COUNT} relevant papers, including ${PAGINATION_PAPER_COUNT} with separate paper notes.`,
          "", `## ${FIXTURE_TOPIC_NAME}`,
        ];
        for (let i = 1; i <= PAGINATION_PAPER_COUNT; i++) {
          const id = paginationPaperId(i);
          lines.push("", `### ${paginationPaperTitle(i)}`, "", `- **Authors**: A. Researcher`, `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`);
          await driver.writeVaultFile(paperNoteFile(id), detailNoteMarkdown(id, paginationPaperTitle(i)));
        }
        await driver.writeVaultFile(dailyReportFile(PAGINATION_REPORT_DATE), lines.join("\n") + "\n");
        await driver.reloadPlugin();
        await driver.openDashboard();
      },
      async verify({ driver, actions }) {
        // Default active tab is "starred" and any tab click resets to page 0 (view.ts), so reaching "All,
        // page 2" is only possible through a real tab click followed by a real Next-page click — never a
        // default.
        const switchedToAll = actions.some(action => action.type === "click" && new RegExp(`^${ALL_TAB_LABEL}\\d*$`, "i").test((action.control?.name || "").trim()));
        if (!switchedToAll) return taskResult([checkResult("点击 All 标签", false)]);
        const clickedNext = actions.some(action => action.type === "click" && exactLabelPattern(NEXT_PAGE_LABEL).test((action.control?.name || "").trim()));
        if (!clickedNext) return taskResult([checkResult("点击 All 标签", true), checkResult("点击 Next page 翻到第 2 页", false)]);
        const activeTab = await poll(
          () => driver.read('document.querySelector(\'.arxiv-daily-dashboard__tab[data-tab="all"]\')?.getAttribute("aria-pressed")'),
          value => value === "true", "All 标签当前处于激活状态", { timeoutMs: 5000 },
        ).catch(() => undefined);
        const pageLabel = await poll(
          () => driver.read('document.querySelector(".arxiv-daily-dashboard__pagination-label")?.textContent ?? ""'),
          value => /^Page 2 \//.test(value || ""), "分页指示当前显示第 2 页", { timeoutMs: 5000 },
        ).catch(() => "");
        const tableText = await driver.read('document.querySelector(".arxiv-daily-dashboard__table")?.textContent ?? ""').catch(() => "");
        const onPageTwo = Boolean(tableText) && tableText.includes(PAGINATION_LAST_PAPER_ID) && !tableText.includes(PAGINATION_FIRST_PAPER_ID);
        return taskResult([
          checkResult("点击 All 标签", true),
          checkResult("点击 Next page 翻到第 2 页", true),
          checkResult("All 标签当前处于激活状态", activeTab === "true", activeTab),
          checkResult("分页指示当前显示第 2 页", /^Page 2 \//.test(pageLabel || ""), pageLabel),
          checkResult("当前页显示只在第二页出现的论文，且不显示第一页独有的论文", onPageTwo, tableText.slice(0, 200)),
        ]);
      },
    },
    {
      id: "mark-to-read", title: "将论文标记为待读",
      goal: `点击仪表盘工具栏的“${MORE_ACTIONS_LABEL}”按钮打开菜单，点击其中的“${SET_PAPER_MARK_LABEL}”，在弹出的对话框里把 ${PAPER_ID_OR_URL_LABEL} 填为 ${MARK_TO_READ_ID}，${MARK_FIELD_LABEL} 保持默认的 ${WATCH_MARK_LABEL}，点击 ${SET_MARK_LABEL} 提交。`,
      uiLabels: [
        uiLabel(MORE_ACTIONS_LABEL, "plugin/src/dashboard/view.ts"),
        uiLabel(SET_PAPER_MARK_LABEL, "plugin/src/dashboard/view.ts"),
        uiLabel(PAPER_ID_OR_URL_LABEL, "plugin/src/commands.ts"),
        uiLabel(MARK_FIELD_LABEL, "plugin/src/commands.ts"),
        uiLabel(WATCH_MARK_LABEL, "plugin/src/commands.ts"),
        uiLabel(SET_MARK_LABEL, "plugin/src/commands.ts"),
      ],
      // Leftover-state audit: papers.json is not reset between tasks, so an earlier pass of this task
      // leaves the target already marked to_read. Restore the untouched default first, so verify() below
      // can only see to_read if this pass's own submit set it.
      async prepare({ driver }) {
        await driver.ensureDailyReport(ACCEPTANCE_DATE);
        await driver.applyIndexPatch(current => {
          const target = entryFor(current, MARK_TO_READ_ID);
          if (target) { target.status = "inbox"; target.priority = "normal"; }
        });
        await driver.openDashboard();
      },
      async verify({ driver, actions }) {
        const filled = actions.some(action => action.type === "fill" && new RegExp(PAPER_ID_OR_URL_LABEL, "i").test(action.control?.name || "") && action.value === MARK_TO_READ_ID);
        if (!filled) return taskResult([checkResult(`在 Set paper mark 对话框中填写 ${MARK_TO_READ_ID}`, false)]);
        const submitted = actions.some(action => action.type === "click" && exactLabelPattern(SET_MARK_LABEL).test((action.control?.name || "").trim()));
        if (!submitted) {
          return taskResult([checkResult(`在 Set paper mark 对话框中填写 ${MARK_TO_READ_ID}`, true), checkResult("点击 Set mark 提交", false)]);
        }
        const onDisk = await pollJsonFile(() => driver.jsonFile(INDEX_FILE), value => entryFor(value, MARK_TO_READ_ID)?.status === "to_read", "to_read status on disk", { timeoutMs: 5000 }).catch(() => null);
        const entry = entryFor(onDisk, MARK_TO_READ_ID);
        const marked = entry?.status === "to_read";
        return taskResult([
          checkResult(`在 Set paper mark 对话框中填写 ${MARK_TO_READ_ID}`, true),
          checkResult("点击 Set mark 提交", true),
          checkResult("论文索引中该论文状态变为 to_read（待读）", marked, entry?.status),
        ]);
      },
    },
    {
      id: "create-paper-note", title: "为指定论文生成论文总结",
      goal: `打开阅读面板，切换到 ${ALL_TAB_LABEL} 标签，找到 arXiv ${CREATE_PAPER_NOTE_ID} 对应行，点击该行的“${SUMMARIZE_BY_ARXIV_ID_LABEL}”按钮，为这篇论文生成一份论文总结（Paper note）。`,
      uiLabels: [
        uiLabel(ALL_TAB_LABEL, "plugin/src/dashboard/constants.ts"),
        uiLabel(SUMMARIZE_BY_ARXIV_ID_LABEL, "plugin/src/dashboard/view.ts"),
      ],
      // Clears the target's own note and seeds the fixture's other paper with one, so exactly one row in
      // the table shows "Summarize by arXiv ID" (see CREATE_PAPER_NOTE_ID's own comment above for why this
      // is necessary — the action buttons carry no per-paper DOM attribute to disambiguate by).
      async prepare({ driver }) {
        await driver.ensureDailyReport(ACCEPTANCE_DATE);
        await driver.removeVaultFile(paperNoteFile(CREATE_PAPER_NOTE_ID));
        const otherPaper = PAPERS.find(paper => paper.id === CREATE_PAPER_NOTE_OTHER_ID);
        await driver.writeVaultFile(paperNoteFile(CREATE_PAPER_NOTE_OTHER_ID), detailNoteMarkdown(CREATE_PAPER_NOTE_OTHER_ID, otherPaper.title));
        await driver.applyIndexPatch(current => {
          const target = entryFor(current, CREATE_PAPER_NOTE_ID);
          if (target) { target.detail = false; target.paperPath = null; }
          const other = entryFor(current, CREATE_PAPER_NOTE_OTHER_ID);
          if (other) { other.detail = true; other.paperPath = paperNoteFile(CREATE_PAPER_NOTE_OTHER_ID); }
        });
        await driver.openDashboard();
      },
      async verify({ driver, actions }) {
        const clicked = actions.some(action => action.type === "click" && exactLabelPattern(SUMMARIZE_BY_ARXIV_ID_LABEL).test((action.control?.name || "").trim()));
        if (!clicked) return taskResult([checkResult(`点击 arXiv ${CREATE_PAPER_NOTE_ID} 行的 Summarize by arXiv ID 按钮`, false)]);
        const notePath = paperNoteFile(CREATE_PAPER_NOTE_ID);
        const content = await poll(() => driver.file(notePath).catch(() => null), value => value !== null, "论文总结文件已写出", { timeoutMs: 5000 }).catch(() => null);
        const sections = ["研究问题", "方法设计", "核心结果", "主要结论", "关键图表", "适用边界"];
        const hasStructure = Boolean(content) && sections.every(section => content.includes(section)) && content.includes(CREATE_PAPER_NOTE_ID);
        const index = await driver.jsonFile(INDEX_FILE).catch(() => null);
        const entry = entryFor(index, CREATE_PAPER_NOTE_ID);
        const indexed = entry?.detail === true && typeof entry.paperPath === "string" && entry.paperPath.endsWith(`${CREATE_PAPER_NOTE_ID}.md`);
        const activeFile = await poll(() => driver.read("app.workspace.getActiveFile()?.path ?? null"), value => value === notePath, "新生成的论文总结在 Obsidian 中被打开", { timeoutMs: 5000 }).catch(() => undefined);
        return taskResult([
          checkResult(`点击 arXiv ${CREATE_PAPER_NOTE_ID} 行的 Summarize by arXiv ID 按钮`, true),
          checkResult("论文总结文件写出固定夹具的结构化章节", hasStructure, content?.slice(0, 200)),
          checkResult("论文索引记录了该论文的总结路径", indexed, entry),
          checkResult("该论文总结在 Obsidian 中被打开", activeFile === notePath, activeFile),
        ]);
      },
    },
    {
      id: "summarize-by-id", title: "按 arXiv 编号生成论文总结",
      goal: `点击阅读面板工具栏的“${SUMMARIZE_BY_ID_LABEL}”按钮，在弹出的对话框里把 ${PAPER_ID_OR_URL_LABEL} 填为 ${SUMMARIZE_BY_ID_TARGET}，点击 ${SUMMARIZE_LABEL} 提交，等待生成完成。`,
      uiLabels: [
        uiLabel(SUMMARIZE_BY_ID_LABEL, "plugin/src/dashboard/view.ts"),
        uiLabel(PAPER_ID_OR_URL_LABEL, "plugin/src/commands.ts"),
        uiLabel(SUMMARIZE_LABEL, "plugin/src/commands.ts"),
      ],
      async prepare({ driver }) {
        await driver.removeVaultFile(paperNoteFile(SUMMARIZE_BY_ID_TARGET));
        await driver.applyIndexPatch(current => { delete current.papers[`arxiv:${SUMMARIZE_BY_ID_TARGET}`]; });
        await driver.openDashboard();
      },
      async verify({ driver, actions }) {
        const filled = actions.some(action => action.type === "fill" && new RegExp(PAPER_ID_OR_URL_LABEL, "i").test(action.control?.name || "") && action.value === SUMMARIZE_BY_ID_TARGET);
        if (!filled) return taskResult([checkResult(`在 Summarize by ID 对话框中填写 ${SUMMARIZE_BY_ID_TARGET}`, false)]);
        const submitted = actions.some(action => action.type === "click" && exactLabelPattern(SUMMARIZE_LABEL).test((action.control?.name || "").trim()));
        if (!submitted) {
          return taskResult([checkResult(`在 Summarize by ID 对话框中填写 ${SUMMARIZE_BY_ID_TARGET}`, true), checkResult("点击 Summarize 提交", false)]);
        }
        const notePath = paperNoteFile(SUMMARIZE_BY_ID_TARGET);
        const content = await poll(() => driver.file(notePath).catch(() => null), value => value !== null, "按编号生成的论文总结文件已写出", { timeoutMs: 5000 }).catch(() => null);
        const sections = ["研究问题", "方法设计", "核心结果", "主要结论", "关键图表", "适用边界"];
        const hasStructure = Boolean(content) && sections.every(section => content.includes(section)) && content.includes(SUMMARIZE_BY_ID_TARGET);
        const index = await driver.jsonFile(INDEX_FILE).catch(() => null);
        const entry = entryFor(index, SUMMARIZE_BY_ID_TARGET);
        const indexed = entry?.detail === true && typeof entry.paperPath === "string" && entry.paperPath.endsWith(`${SUMMARIZE_BY_ID_TARGET}.md`);
        return taskResult([
          checkResult(`在 Summarize by ID 对话框中填写 ${SUMMARIZE_BY_ID_TARGET}`, true),
          checkResult("点击 Summarize 提交", true),
          checkResult("按编号生成的论文总结文件写出固定夹具的结构化章节", hasStructure, content?.slice(0, 200)),
          checkResult("论文索引新增了该论文条目并记录总结路径", indexed, entry),
        ]);
      },
    },
    {
      id: "open-today-report", title: "从日历打开日报",
      goal: `打开阅读面板，在 ${DAILY_REPORTS_LABEL} 日历中找到 ${ACCEPTANCE_DATE} 对应的日期格（已生成日报，带有论文数角标），点击它打开当天的日报。`,
      uiLabels: [uiLabel(DAILY_REPORTS_LABEL, "plugin/src/dashboard/view.ts")],
      async prepare({ driver }) {
        await driver.ensureDailyReport(ACCEPTANCE_DATE);
        await driver.openDashboard();
      },
      // Opening a vault note leaves the plugin's own surface (resolveSurface() reports view:"none"
      // afterward); verify reads app.workspace state directly, as readingAndReload (obsidian-scenarios.mjs)
      // already does for the same signal, and the model need not act again once the file is open.
      async verify({ driver, actions }) {
        const clicked = actions.some(action => action.type === "click" && new RegExp(`^${ACCEPTANCE_DATE}:`).test((action.control?.name || "").trim()));
        if (!clicked) return taskResult([checkResult(`点击 ${ACCEPTANCE_DATE} 对应的日历日期格`, false)]);
        const activeFile = await poll(() => driver.read("app.workspace.getActiveFile()?.path ?? null"), value => value === dailyReportFile(ACCEPTANCE_DATE), "工作区打开了该日期的日报", { timeoutMs: 5000 }).catch(() => undefined);
        return taskResult([
          checkResult(`点击 ${ACCEPTANCE_DATE} 对应的日历日期格`, true),
          checkResult("打开的笔记正是该日期的日报文件", activeFile === dailyReportFile(ACCEPTANCE_DATE), activeFile),
        ]);
      },
    },
    // ─── Exploration safety and coverage P3: personal literature library tasks ───
    {
      id: "library-revoke-consent", title: "撤销个人文献库的处理授权",
      goal: `打开插件设置页，个人文献库目前已连接并已获得处理授权（远程 Embedding，已发送标题和摘要）。在“Personal library”区域的库连接行上点击 ${REVOKE_LABEL} 按钮；系统会弹出确认对话框，说明撤销后 Embedding 会改回本地且已有索引需要重建，请再次点击对话框中的 ${REVOKE_LABEL} 按钮确认撤销。不要点击 Cancel，也不要修改其他设置。`,
      uiLabels: [uiLabel(REVOKE_LABEL, ["plugin/src/library/modal.ts", "packages/core/src/library/library-connection.ts"])],
      // Starts already connected, remote and authorized (never through the model — startLibraryFixture
      // calls the plugin's own authorizeLibraryProcessing() over CDP, exactly like every other task's
      // prepare() seeds its own deterministic starting state directly on disk), so the only thing left
      // for the model to do is the one UI action this task actually tests: find and confirm Revoke.
      async prepare({ driver }) {
        await startLibraryFixture(driver, { embeddingMode: "remote", authorized: true });
      },
      async verify({ driver, actions }) {
        const revokeClicks = actions.filter(action => action.type === "click" && exactLabelPattern(REVOKE_LABEL).test((action.control?.name || "").trim())).length;
        if (revokeClicks === 0) return taskResult([checkResult(`点击库连接行上的 ${REVOKE_LABEL} 按钮，打开撤销确认对话框`, false)]);
        if (revokeClicks < 2) {
          return taskResult([
            checkResult(`点击库连接行上的 ${REVOKE_LABEL} 按钮，打开撤销确认对话框`, true),
            checkResult(`在确认对话框中再次点击 ${REVOKE_LABEL} 确认撤销`, false),
          ]);
        }
        const onDisk = await pollJsonFile(() => driver.jsonFile(SETTINGS_FILE), value => !value?.libraryConnection?.authorization, "library authorization revoked on disk", { timeoutMs: 5000 }).catch(() => null);
        const revoked = Boolean(onDisk) && !onDisk.libraryConnection?.authorization;
        const switchedToLocal = onDisk?.settings?.embedding?.mode === "local";
        return taskResult([
          checkResult(`点击库连接行上的 ${REVOKE_LABEL} 按钮，打开撤销确认对话框`, true),
          checkResult(`在确认对话框中再次点击 ${REVOKE_LABEL} 确认撤销`, true),
          checkResult("个人文献库的处理授权已从设置中撤销", revoked, onDisk?.libraryConnection),
          checkResult("Embedding 模式已随撤销改回 local", switchedToLocal, onDisk?.settings?.embedding?.mode),
        ]);
      },
    },
    {
      id: "library-remote-consent", title: "将 Embedding 切换为远程并确认处理授权披露",
      goal: `打开插件设置页，个人文献库目前已连接，但尚未获得处理授权，Embedding 模式为 Local（本地）。在“Personal library”区域找到 ${EMBEDDING_MODE_LABEL} 下拉选择框，将其切换为 Remote。切换后会弹出一个披露对话框，说明切换为远程后标题和摘要会发送到已配置的 Embedding 端点；请阅读该说明，然后点击对话框中的 “${SEND_TITLES_AND_ABSTRACTS_LABEL}” 按钮确认同意。不要点击 Cancel，也不要修改其他设置。`,
      uiLabels: [
        uiLabel(EMBEDDING_MODE_LABEL, ["plugin/src/settings/tab.ts", "packages/core/src/settings/schema.ts"]),
        uiLabel(SEND_TITLES_AND_ABSTRACTS_LABEL, "plugin/src/library/modal.ts"),
      ],
      // Starts connected but unauthorized, still in local Embedding mode (never remote/authorized
      // through the model — the one thing this task actually tests is the model finding and completing
      // the switch-then-consent flow itself). startLibraryFixture already points the embedding baseUrl
      // and dimension at the fixture server even while local, so the fixture's own /embeddings handler
      // answers the moment the model switches to remote — no separate setup step needed here.
      async prepare({ driver }) {
        await startLibraryFixture(driver, { embeddingMode: "local" });
      },
      async verify({ driver, actions }) {
        const switchedToRemote = actions.some(action => action.type === "select"
          && exactLabelPattern(EMBEDDING_MODE_LABEL).test((action.control?.name || "").trim())
          && action.value === "remote");
        if (!switchedToRemote) return taskResult([checkResult(`将 ${EMBEDDING_MODE_LABEL} 下拉框切换为 remote`, false)]);
        const confirmed = actions.some(action => action.type === "click" && exactLabelPattern(SEND_TITLES_AND_ABSTRACTS_LABEL).test((action.control?.name || "").trim()));
        if (!confirmed) {
          return taskResult([
            checkResult(`将 ${EMBEDDING_MODE_LABEL} 下拉框切换为 remote`, true),
            checkResult(`在披露对话框中点击 “${SEND_TITLES_AND_ABSTRACTS_LABEL}” 确认`, false),
          ]);
        }
        const onDisk = await pollJsonFile(
          () => driver.jsonFile(SETTINGS_FILE),
          value => value?.settings?.embedding?.mode === "remote" && Boolean(value?.libraryConnection?.authorization),
          "remote Embedding mode and library authorization on disk",
          { timeoutMs: 5000 },
        ).catch(() => null);
        const modeRemote = onDisk?.settings?.embedding?.mode === "remote";
        const authorized = Boolean(onDisk?.libraryConnection?.authorization);
        return taskResult([
          checkResult(`将 ${EMBEDDING_MODE_LABEL} 下拉框切换为 remote`, true),
          checkResult(`在披露对话框中点击 “${SEND_TITLES_AND_ABSTRACTS_LABEL}” 确认`, true),
          checkResult("Embedding 模式已保存为 remote", modeRemote, onDisk?.settings?.embedding?.mode),
          checkResult("个人文献库的处理授权已写入设置", authorized, onDisk?.libraryConnection?.authorization),
        ]);
      },
    },
    // ─── Exploration safety and coverage P3 Chunk 2: index build, library search and similar-paper
    // tasks (docs/helm/2026-10-13-exploration-safety-and-coverage/phases/03-obsidian-library-tasks.md).
    // ───────────────────────────────────────────────────────────────────────
    {
      id: "library-build-index", title: "构建个人文献库搜索索引",
      goal: `打开插件设置页，个人文献库目前已连接、已获得处理授权（远程 Embedding），但还没有搜索索引。在“Personal library”区域点击 ${RETRY_PREPARATION_LABEL} 按钮，开始构建索引，等待构建完成即可，不需要关闭设置。`,
      uiLabels: [uiLabel(RETRY_PREPARATION_LABEL, "packages/core/src/library/library-connection.ts")],
      // Starts connected, remote and already authorized (never through the model — see
      // library-revoke-consent's own prepare() comment), but deliberately with no index yet
      // (connectedLibraryFixture never sets lastRun): the one thing this task tests is the model finding
      // and clicking the row's own "Retry preparation" button and the real indexing it drives actually
      // finishing. Records the fixture's own request count at this exact moment so verify() can tell a
      // fresh embeddings request (driven by this task's own click) apart from one left over from an
      // earlier task in the same pass.
      async prepare({ driver }) {
        await startLibraryFixture(driver, { embeddingMode: "remote", authorized: true });
        libraryBuildIndexBaseline = { requestCount: driver.fixture.server.requests.length };
      },
      async verify({ driver, actions }) {
        const clicked = actions.some(action => action.type === "click" && exactLabelPattern(RETRY_PREPARATION_LABEL).test((action.control?.name || "").trim()));
        if (!clicked) return taskResult([checkResult(`点击 Personal library 区域的 ${RETRY_PREPARATION_LABEL} 按钮`, false)]);
        const indexFile = await poll(
          () => driver.findLibrarySearchIndexFile(),
          value => value !== null, "个人文献库搜索索引的 current.json 已生成", { timeoutMs: 5000 },
        ).catch(() => null);
        const baseline = libraryBuildIndexBaseline?.requestCount ?? 0;
        const embeddingsRequested = driver.fixture.server.requests.slice(baseline).some(request => request.kind === "embeddings");
        return taskResult([
          checkResult(`点击 Personal library 区域的 ${RETRY_PREPARATION_LABEL} 按钮`, true),
          checkResult("磁盘上生成了个人文献库搜索索引的 current.json", Boolean(indexFile), indexFile),
          checkResult("任务开始后夹具收到了 embeddings 请求", embeddingsRequested),
        ]);
      },
    },
    {
      id: "library-search", title: "在阅读面板搜索个人文献库",
      goal: `打开阅读面板，个人文献库已经连接、已获得授权，并且已经完成过一次索引构建。在 ${SEARCH_FIELD_LABEL} 输入框中填写“${LIBRARY_SEARCH_QUERY}”，等待下方出现“${LIBRARY_MATCHES_LABEL}”结果区域，再点击其中一条结果的 ${OPEN_PDF_LABEL} 按钮，打开对应的 PDF 文件。`,
      uiLabels: [
        uiLabel(SEARCH_FIELD_LABEL, "plugin/src/dashboard/view.ts"),
        uiLabel(LIBRARY_MATCHES_LABEL, "plugin/src/dashboard/library-search-block.ts"),
        uiLabel(OPEN_PDF_LABEL, "plugin/src/dashboard/library-search-block.ts"),
      ],
      // Builds the real index in prepare() by actually running the plugin's own
      // indexPersonalLibraryFullText() over CDP (driver.buildLibraryIndex(), exploration-obsidian.mjs) —
      // never a hand-written index file — so the model's search hits a real, populated knowledge base
      // exactly like a person who had already prepared their library once.
      async prepare({ driver }) {
        await startLibraryFixture(driver, { embeddingMode: "remote", authorized: true });
        await driver.buildLibraryIndex();
        await driver.openDashboard();
      },
      async verify({ driver, actions }) {
        const filled = actions.some(action => action.type === "fill" && exactLabelPattern(SEARCH_FIELD_LABEL).test((action.control?.name || "").trim()) && action.value === LIBRARY_SEARCH_QUERY);
        if (!filled) return taskResult([checkResult(`在 ${SEARCH_FIELD_LABEL} 输入框中搜索“${LIBRARY_SEARCH_QUERY}”`, false)]);
        const resultsCount = await poll(
          () => driver.read('document.querySelectorAll(".arxiv-daily-dashboard__library-list .arxiv-daily-dashboard__library-item").length'),
          value => Number(value) > 0, "Library matches 结果列表非空", { timeoutMs: 8000 },
        ).catch(() => 0);
        if (!resultsCount) {
          return taskResult([
            checkResult(`在 ${SEARCH_FIELD_LABEL} 输入框中搜索“${LIBRARY_SEARCH_QUERY}”`, true),
            checkResult(`${LIBRARY_MATCHES_LABEL} 区域显示至少一条结果`, false),
          ]);
        }
        const openClicked = actions.some(action => action.type === "click" && exactLabelPattern(OPEN_PDF_LABEL).test((action.control?.name || "").trim()));
        if (!openClicked) {
          return taskResult([
            checkResult(`在 ${SEARCH_FIELD_LABEL} 输入框中搜索“${LIBRARY_SEARCH_QUERY}”`, true),
            checkResult(`${LIBRARY_MATCHES_LABEL} 区域显示至少一条结果`, true),
            checkResult(`点击某条结果的 ${OPEN_PDF_LABEL} 按钮`, false),
          ]);
        }
        const activeFile = await poll(
          () => driver.read("app.workspace.getActiveFile()?.path ?? null"),
          value => typeof value === "string" && value.startsWith("library/") && value.endsWith(".pdf"),
          "工作区打开了文献库中的某个 PDF 文件", { timeoutMs: 5000 },
        ).catch(() => undefined);
        const opened = typeof activeFile === "string" && activeFile.startsWith("library/") && activeFile.endsWith(".pdf");
        return taskResult([
          checkResult(`在 ${SEARCH_FIELD_LABEL} 输入框中搜索“${LIBRARY_SEARCH_QUERY}”`, true),
          checkResult(`${LIBRARY_MATCHES_LABEL} 区域显示至少一条结果`, true),
          checkResult(`点击某条结果的 ${OPEN_PDF_LABEL} 按钮`, true),
          checkResult("工作区打开了文献库中的某个 PDF 文件", opened, activeFile),
        ]);
      },
    },
    {
      id: "library-similar-papers", title: "查看论文的文献库相似结果",
      goal: `打开阅读面板，点击 ${ALL_TAB_LABEL} 标签查看全部论文。找到其中任意一行，点击它的 ${FIND_SIMILAR_PAPERS_LABEL} 按钮打开 Similar papers 对话框。对话框默认就会显示“${LIBRARY_SIMILAR_TAB_LABEL}”标签的内容，请再点击一次该标签以确认，并等待其中的文献库匹配列表加载完成。`,
      uiLabels: [
        uiLabel(ALL_TAB_LABEL, "plugin/src/dashboard/constants.ts"),
        uiLabel(FIND_SIMILAR_PAPERS_LABEL, "plugin/src/dashboard/view.ts"),
        uiLabel(LIBRARY_SIMILAR_TAB_LABEL, "plugin/src/dashboard/similar-papers-modal.ts"),
      ],
      // Same real-index prepare() as library-search, plus a daily report so at least one row (with a
      // real title/abstract, which buildLibrarySimilarOption needs to enable the library tab at all) is
      // visible once the model switches to the All tab.
      async prepare({ driver }) {
        await startLibraryFixture(driver, { embeddingMode: "remote", authorized: true });
        await driver.buildLibraryIndex();
        await driver.ensureDailyReport(ACCEPTANCE_DATE);
        await driver.openDashboard();
      },
      async verify({ driver, actions }) {
        const switchedToAll = actions.some(action => action.type === "click" && new RegExp(`^${ALL_TAB_LABEL}\\d*$`, "i").test((action.control?.name || "").trim()));
        if (!switchedToAll) return taskResult([checkResult(`点击 ${ALL_TAB_LABEL} 标签`, false)]);
        const openedSimilar = actions.some(action => action.type === "click" && exactLabelPattern(FIND_SIMILAR_PAPERS_LABEL).test((action.control?.name || "").trim()));
        if (!openedSimilar) {
          return taskResult([
            checkResult(`点击 ${ALL_TAB_LABEL} 标签`, true),
            checkResult(`点击某一行的 ${FIND_SIMILAR_PAPERS_LABEL} 按钮`, false),
          ]);
        }
        const switchedToLibraryTab = actions.some(action => action.type === "click" && exactLabelPattern(LIBRARY_SIMILAR_TAB_LABEL).test((action.control?.name || "").trim()));
        if (!switchedToLibraryTab) {
          return taskResult([
            checkResult(`点击 ${ALL_TAB_LABEL} 标签`, true),
            checkResult(`点击某一行的 ${FIND_SIMILAR_PAPERS_LABEL} 按钮`, true),
            checkResult(`在对话框中点击“${LIBRARY_SIMILAR_TAB_LABEL}”标签`, false),
          ]);
        }
        const libraryListCount = await poll(
          () => driver.read('document.querySelectorAll(".arxiv-daily-similar-modal .arxiv-daily-dashboard__library-item").length'),
          value => Number(value) > 0, "Similar papers 对话框的文献库匹配列表非空", { timeoutMs: 8000 },
        ).catch(() => 0);
        return taskResult([
          checkResult(`点击 ${ALL_TAB_LABEL} 标签`, true),
          checkResult(`点击某一行的 ${FIND_SIMILAR_PAPERS_LABEL} 按钮`, true),
          checkResult(`在对话框中点击“${LIBRARY_SIMILAR_TAB_LABEL}”标签`, true),
          checkResult("Similar papers 对话框的文献库匹配列表非空", libraryListCount > 0, libraryListCount),
        ]);
      },
    },
    // ─── Exploration safety and coverage P3 Chunk 3: Direction candidate review tasks ───
    {
      id: "library-accept-direction", title: "采纳一个来自个人文献库的候选研究方向",
      goal: `打开插件设置页，找到 ${PERSONAL_LIBRARY_SECTION_LABEL} 区域下的 ${TOPICS_FROM_LIBRARY_LABEL} 这一行，点击 ${REVIEW_SUGGESTIONS_LABEL} 按钮打开建议审核对话框（个人文献库已连接、已授权且已完成索引，首次打开会自动生成建议，请等待生成完成）。在列出的候选研究方向中，确保其中一个方向的复选框处于勾选状态（如果还没有勾选，点击它勾选），然后点击 ${ADD_TO_RESEARCH_TOPICS_LABEL} 按钮（它可能显示为这段说明文字：“${ADD_TO_RESEARCH_TOPICS_TITLE}”），把它加入研究主题。`,
      uiLabels: [
        uiLabel(PERSONAL_LIBRARY_SECTION_LABEL, "plugin/src/settings/tab.ts"),
        uiLabel(TOPICS_FROM_LIBRARY_LABEL, ["plugin/src/settings/tab.ts", "plugin/src/settings/definitions.ts"]),
        uiLabel(REVIEW_SUGGESTIONS_LABEL, "plugin/src/settings/tab.ts"),
        uiLabel(ADD_TO_RESEARCH_TOPICS_LABEL, "plugin/src/library/interest-profile-modal.ts"),
        uiLabel(ADD_TO_RESEARCH_TOPICS_TITLE, "plugin/src/library/interest-profile-modal.ts"),
      ],
      // Starts connected, remote and already authorized with a real completed index (never through the
      // model — see library-build-index's own prepare() comment): "Review suggestions" is disabled
      // until the index has a completed run with papers, and the authorization scope startLibraryFixture
      // grants (llmBaseUrl + the remote embedding endpoint) already covers the generation disclosure too
      // (both read the plugin's own libraryAuthorizationScope(), main.ts:604-613), so the model never has
      // to drive a second consent dialog — the one thing this task actually tests is generating,
      // selecting and accepting a real proposed direction.
      async prepare({ driver }) {
        await startLibraryFixture(driver, { embeddingMode: "remote", authorized: true });
        // See seedClusterableLibraryPaper's own comment: the fixture's two library PDFs alone embed to
        // two different vectors and cluster into nothing (real-host evidence, 2026-10-13); this third
        // paper shares PAPERS[0]'s own vector so a real two-member cluster — and so a real proposal —
        // always exists for the model's own Review suggestions click to generate.
        await seedClusterableLibraryPaper(driver);
        await driver.buildLibraryIndex();
        await driver.openDashboard();
      },
      async verify({ driver, actions }) {
        const reviewOpened = actions.some(action => action.type === "click" && exactLabelPattern(REVIEW_SUGGESTIONS_LABEL).test((action.control?.name || "").trim()));
        if (!reviewOpened) return taskResult([checkResult(`点击 ${REVIEW_SUGGESTIONS_LABEL} 打开建议审核对话框`, false)]);
        const accepted = actions.some(action => action.type === "click" && (name => exactLabelPattern(ADD_TO_RESEARCH_TOPICS_LABEL).test(name) || exactLabelPattern(ADD_TO_RESEARCH_TOPICS_TITLE).test(name))((action.control?.name || "").trim()));
        if (!accepted) {
          return taskResult([
            checkResult(`点击 ${REVIEW_SUGGESTIONS_LABEL} 打开建议审核对话框`, true),
            checkResult(`点击 ${ADD_TO_RESEARCH_TOPICS_LABEL} 采纳选中的候选方向`, false),
          ]);
        }
        const onDisk = await pollJsonFile(() => driver.jsonFile(SETTINGS_FILE), value => {
          const addedDirection = (value?.settings?.arxiv?.topics ?? []).some(topic => topic.directions?.some(direction => direction.origin === "library"));
          return addedDirection && Array.isArray(value?.libraryProposalAcceptances) && value.libraryProposalAcceptances.length > 0;
        }, "library-origin direction and acceptance receipt on disk", { timeoutMs: 8000 }).catch(() => null);
        const addedDirection = (onDisk?.settings?.arxiv?.topics ?? []).some(topic => topic.directions?.some(direction => direction.origin === "library"));
        const hasReceipt = Array.isArray(onDisk?.libraryProposalAcceptances) && onDisk.libraryProposalAcceptances.length > 0;
        return taskResult([
          checkResult(`点击 ${REVIEW_SUGGESTIONS_LABEL} 打开建议审核对话框`, true),
          checkResult(`点击 ${ADD_TO_RESEARCH_TOPICS_LABEL} 采纳选中的候选方向`, true),
          checkResult("研究主题中新增了一个来源为 library 的方向", addedDirection, onDisk?.settings?.arxiv?.topics),
          checkResult("插件记录了本次建议采纳的回执（libraryProposalAcceptances）", hasReceipt, onDisk?.libraryProposalAcceptances),
        ]);
      },
    },
    {
      id: "library-edit-candidate", title: "编辑个人文献库的一个候选研究方向",
      // Real-host evidence (2026-10-13), two separate findings, both now fixed at the shared Obsidian
      // observer level (exploration-obsidian.mjs), not worked around here:
      //
      // 1. Every proposed topic renders collapsed (<details>, closed by default — interest-profile-
      //    modal.ts's own `expandedTopics` starts empty). `controlVisible()` did not check whether a
      //    node's ancestor was a closed <details>, so it wrongly reported every control inside a still-
      //    collapsed topic as present and clickable; a real-host run confirmed clicking any of them timed
      //    out hit-testing — safely refused, never a stray click, but wasted model steps. Fixed (P3
      //    Chunk 7): `controlVisible()` now excludes a closed `<details>`'s content, except its own
      //    `<summary>` — so the goal still has the model click the topic's own heading row (its
      //    `<summary>`) to expand it before anything inside becomes visible or reachable.
      //
      // 2. Clicking the candidate card's own "Discard" button opens a second `.arxiv-daily-modal`-marked
      //    confirmation dialog (chooseModal, services/modal.ts) stacked on top of the still-open review
      //    modal. `resolveSurface()` used to resolve the FIRST `.arxiv-daily-modal` in DOM order — the
      //    older review modal underneath, never the newer confirm dialog — so the confirm dialog's own
      //    "Discard"/"Cancel" buttons were never observed and the flow could never be completed. Fixed
      //    (P3 Chunk 7): `resolveSurface()` now resolves the LAST (topmost, most recently opened)
      //    `.arxiv-daily-modal`. A real-host run of the discard path below (expand the topic, expand the
      //    candidate card's own "Edit", click Discard, then click Discard again in the now-observable
      //    confirmation dialog) passes.
      //
      // The goal below asks for either outcome — rename or discard — matching verify()'s own two
      // branches; a model picks one, never both.
      goal: `打开插件设置页，找到 ${PERSONAL_LIBRARY_SECTION_LABEL} 区域下的 ${TOPICS_FROM_LIBRARY_LABEL} 这一行，点击 ${REVIEW_SUGGESTIONS_LABEL} 按钮打开建议审核对话框（已经生成好一份建议，不需要重新生成）。点击对话框里列出的第一个候选研究方向所在主题的标题行，将其展开。然后二选一完成以下一项：(1) 展开它的“${RENAME_TOPIC_LABEL}”，把“${TOPIC_NAME_FIELD_LABEL}”改为“${LIBRARY_EDIT_NEW_TOPIC_NAME}”并移开焦点确认；(2) 展开该候选方向卡片自己的“${CARD_EDIT_LABEL}”，点击其中的 ${DISCARD_LABEL} 按钮丢弃该候选方向，并在随后弹出的确认对话框中再次点击 ${DISCARD_LABEL} 以确认丢弃。不需要点击 ${ADD_TO_RESEARCH_TOPICS_LABEL}。`,
      uiLabels: [
        uiLabel(PERSONAL_LIBRARY_SECTION_LABEL, "plugin/src/settings/tab.ts"),
        uiLabel(TOPICS_FROM_LIBRARY_LABEL, ["plugin/src/settings/tab.ts", "plugin/src/settings/definitions.ts"]),
        uiLabel(REVIEW_SUGGESTIONS_LABEL, "plugin/src/settings/tab.ts"),
        uiLabel(RENAME_TOPIC_LABEL, "plugin/src/library/interest-profile-modal.ts"),
        uiLabel(TOPIC_NAME_FIELD_LABEL, "plugin/src/library/interest-profile-modal.ts"),
        uiLabel(CARD_EDIT_LABEL, "plugin/src/library/interest-profile-modal.ts"),
        uiLabel(DISCARD_LABEL, "plugin/src/library/interest-profile-modal.ts"),
        uiLabel(ADD_TO_RESEARCH_TOPICS_LABEL, "plugin/src/library/interest-profile-modal.ts"),
      ],
      // Starts connected, remote, authorized and indexed like library-accept-direction, but additionally
      // generates a real proposal in prepare() itself (driver.generatePersonalLibraryDirections(), the
      // plugin's own real clustering/generation pipeline — never a hand-written direction-proposal.json,
      // per the phase's own approach) so the model edits an already-existing candidate rather than also
      // having to generate one; captures that candidate's real identity so verify() can check the exact
      // record changed, not merely that *some* proposal document did.
      async prepare({ driver }) {
        await startLibraryFixture(driver, { embeddingMode: "remote", authorized: true });
        // See seedClusterableLibraryPaper's own comment (library-accept-direction's prepare()): the
        // fixture's two library PDFs alone cluster into nothing, so real generation here needs this
        // third, same-vector paper too.
        await seedClusterableLibraryPaper(driver);
        await driver.buildLibraryIndex();
        const generated = await driver.generatePersonalLibraryDirections();
        if (generated?.outcome === "nothing-new") throw new Error("library-edit-candidate: generation outcome nothing-new; no new papers produced a proposal");
        const proposal = generated?.outcome === "generated" ? generated.proposal : null;
        const topic = proposal?.topics?.[0];
        const candidate = topic?.directions?.[0];
        if (!topic || !candidate) {
          throw new Error(`library-edit-candidate: prepare()'s real generation did not produce any proposed topic/direction to edit (outcome: ${generated?.outcome ?? "missing"})`);
        }
        libraryEditCandidateBaseline = { topicId: topic.id, originalTopicName: topic.suggestedName, candidateId: candidate.id };
        await driver.openDashboard();
      },
      async verify({ driver, actions }) {
        const reviewOpened = actions.some(action => action.type === "click" && exactLabelPattern(REVIEW_SUGGESTIONS_LABEL).test((action.control?.name || "").trim()));
        if (!reviewOpened) return taskResult([checkResult(`点击 ${REVIEW_SUGGESTIONS_LABEL} 打开建议审核对话框`, false)]);
        const renamed = actions.some(action => action.type === "fill" && exactLabelPattern(TOPIC_NAME_FIELD_LABEL).test((action.control?.name || "").trim()) && action.value === LIBRARY_EDIT_NEW_TOPIC_NAME);
        const discarded = actions.some(action => action.type === "click" && exactLabelPattern(DISCARD_LABEL).test((action.control?.name || "").trim()));
        if (!renamed && !discarded) {
          return taskResult([
            checkResult(`点击 ${REVIEW_SUGGESTIONS_LABEL} 打开建议审核对话框`, true),
            checkResult(`重命名该主题为“${LIBRARY_EDIT_NEW_TOPIC_NAME}”，或点击该候选方向的 ${DISCARD_LABEL} 按钮丢弃它（二选一）`, false),
          ]);
        }
        if (renamed) {
          const proposal = await poll(
            () => driver.readLibraryDirectionProposal(),
            value => value?.topics?.find(item => item.id === libraryEditCandidateBaseline?.topicId)?.suggestedName === LIBRARY_EDIT_NEW_TOPIC_NAME,
            "direction-proposal.json 中该主题已重命名", { timeoutMs: 8000 },
          ).catch(() => null);
          const topic = proposal?.topics?.find(item => item.id === libraryEditCandidateBaseline?.topicId);
          const ok = topic?.suggestedName === LIBRARY_EDIT_NEW_TOPIC_NAME;
          return taskResult([
            checkResult(`点击 ${REVIEW_SUGGESTIONS_LABEL} 打开建议审核对话框`, true),
            checkResult(`填写“${TOPIC_NAME_FIELD_LABEL}”为“${LIBRARY_EDIT_NEW_TOPIC_NAME}”`, true),
            checkResult("direction-proposal.json 中该主题的名称已更新为新名称", ok, topic?.suggestedName),
          ]);
        }
        const proposal = await poll(
          () => driver.readLibraryDirectionProposal(),
          value => !(value?.topics ?? []).some(topic => topic.directions?.some(direction => direction.id === libraryEditCandidateBaseline?.candidateId)),
          "direction-proposal.json 中该候选方向已被丢弃", { timeoutMs: 8000 },
        ).catch(() => null);
        const stillPresent = (proposal?.topics ?? []).some(topic => topic.directions?.some(direction => direction.id === libraryEditCandidateBaseline?.candidateId));
        const ok = Boolean(proposal) && !stillPresent;
        return taskResult([
          checkResult(`点击 ${REVIEW_SUGGESTIONS_LABEL} 打开建议审核对话框`, true),
          checkResult(`点击该候选方向的 ${DISCARD_LABEL} 按钮`, true),
          checkResult("direction-proposal.json 中该候选方向已被丢弃", ok, proposal?.topics),
        ]);
      },
    },
  ];
}

/** P3 Chunk 5: the shared personas (exploration-tasks/personas.mjs), worded for the Obsidian surface. */
export function personaTasks() {
  return sharedPersonaTasks("obsidian");
}
