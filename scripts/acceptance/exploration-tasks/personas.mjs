/**
 * Shared persona (`kind: "persona"`, no verifier) definitions for both hosts (P3 Chunk 5 of
 * docs/helm/2026-10-10-model-driven-release-exploration). Ids, titles and per-task step budgets are the
 * same for workbench and Obsidian, defined once here, so neither host's own task-library file
 * (exploration-tasks/{workbench,obsidian}.mjs) duplicates that structure; only the Chinese brief's
 * host-specific wording (which surface the persona is exploring) differs, picked by `personaTasks(host)`.
 *
 * These are built as plain task objects (not through exploration.mjs's `createPersonaExplorationTask`,
 * whose own generic goal template is a short role label wrapped in "扮演「...」"): the phase contract asks
 * for a concrete, behavior-shaping brief per persona (and, for keyboard-only, a hard constraint on which
 * actions are allowed), which reads better as the task's own `goal` than nested inside that template.
 */
export const PERSONA_TASK_IDS = ["first-time-user", "impatient-clicker", "keyboard-only"];

const TITLES = {
  "first-time-user": "第一次使用的新用户自由探索",
  "impatient-clicker": "没耐心快速乱点的用户自由探索",
  "keyboard-only": "只用键盘操作的用户自由探索",
};

// Per-persona step budget (passed as the task's own maxSteps; exploration.mjs's runExplorationLoop
// defaults a persona to 15 if omitted). keyboard-only gets the smallest budget: every one of its steps is
// a single key press, so it covers much less ground per step than a persona that can click/fill/select
// directly, and a generous ceiling here would mostly just mean more idle polling once it is stuck.
const MAX_STEPS = { "first-time-user": 18, "impatient-clicker": 16, "keyboard-only": 12 };

const SURFACE = { workbench: "阅读工作台", obsidian: "本插件界面（在 Obsidian 中）" };

const BRIEFS = {
  "first-time-user": host => `你是第一次使用${SURFACE[host]}的新用户，没看过任何文档。请在现有数据基础上自己摸索，尝试完成这个目标：确认或完成必要的设置、生成（或找到）一份日报、再打开其中一篇论文读一读。每当你找不到入口、看不懂某个控件的含义，或者卡住不知道下一步该做什么，就用 finding 记录下来（summary 写清楚是什么问题，evidence 写清楚你当时看到的文字或状态）。探索得差不多了，无论有没有走完目标，都用 done 结束；这不代表任务通过。`,
  "impatient-clicker": host => `你是一个没耐心、手速很快的用户，在${SURFACE[host]}里操作时习惯：同一个按钮连续点好几次、不等上一步界面反应完就切换到别的标签或页面、来回快速切换筛选和视图。请按这种风格实际操作几轮，覆盖你能找到的几类不同控件。只要你观察到重复提交、界面状态和你做的操作不一致、点击后看起来没有反应等情况，就用 finding 记录（summary 是你观察到的问题，evidence 是具体看到的文字或状态）。探索得差不多了就用 done 结束；这不代表任务通过。`,
  "keyboard-only": host => `你是一个完全不用鼠标、只用键盘的用户，在${SURFACE[host]}里浏览。从现在开始你每一步只能发出 press 动作（允许的键：Tab、Shift+Tab、Enter、Escape、ArrowUp、ArrowDown、ArrowLeft、ArrowRight、Space、Home、End、PageUp、PageDown）或者 wait 动作，绝对不能使用 click、fill、select、scroll、back、reload。用 Tab / Shift+Tab 在界面上移动焦点，用 Enter / Space / 方向键尝试激活你找到的控件。每当你没法把焦点移动到一个看起来明显可见的控件上、焦点好像消失了、或者按下某个键之后画面没有任何反应，就用 finding 记录下来（summary 是问题，evidence 是你当时看到的状态）。探索得差不多了就用 done 结束；这不代表任务通过。`,
};

/** `host`: "workbench" | "obsidian". Order matches PERSONA_TASK_IDS, matching every other task library here. */
export function personaTasks(host) {
  if (!Object.hasOwn(SURFACE, host)) throw new Error(`personaTasks needs a known host, got: ${host}`);
  return PERSONA_TASK_IDS.map(id => ({ id, title: TITLES[id], kind: "persona", goal: BRIEFS[id](host), maxSteps: MAX_STEPS[id] }));
}
