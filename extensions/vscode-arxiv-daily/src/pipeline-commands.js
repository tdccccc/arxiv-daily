const DEFAULT_CLI_PATH = "arxiv-daily";
const MODERN_ARXIV_ID_RE = /^(\d{2})(\d{2})\.(\d{4,5})(?:v[1-9]\d*)?$/;
const ARXIV_URL_RE = /^https?:\/\/(?:www\.)?arxiv\.org\/(abs|pdf)\/(\d{4}\.\d{4,5}(?:v[1-9]\d*)?)(\.pdf)?(?:[?#].*)?$/;

async function runForToday(vscodeApi) {
  return await runCliTask(vscodeApi, ["run", "--today"]);
}

async function summarizeById(vscodeApi) {
  const input = await vscodeApi.window.showInputBox({
    title: "Summarize arXiv ID",
    prompt: "Enter a modern arXiv ID or an arxiv.org abs/PDF URL.",
    placeHolder: "2606.12345",
    ignoreFocusOut: true,
  });
  if (input === undefined) return false;
  const arxivId = normalizeArxivId(input);
  if (!arxivId) {
    void vscodeApi.window.showWarningMessage("arXiv Daily: invalid arXiv ID.");
    return false;
  }
  return await runCliTask(vscodeApi, ["run", "--id", arxivId]);
}

async function runCliTask(vscodeApi, args) {
  const execution = new vscodeApi.ProcessExecution(
    cliPathFromSettings(vscodeApi),
    args,
  );
  const task = new vscodeApi.Task(
    { type: "process" },
    vscodeApi.TaskScope.Workspace,
    "Run CLI",
    "arXiv Daily",
    execution,
  );

  let dispatchedExecution;
  let resolveCompletion;
  let bufferedEvents = [];
  const completion = new Promise((resolve) => {
    resolveCompletion = resolve;
  });
  const handleEndEvent = (kind, event) => {
    const completionEvent = { kind, event };
    if (!dispatchedExecution) {
      bufferedEvents.push(completionEvent);
      return;
    }
    if (event.execution === dispatchedExecution) {
      resolveCompletion(completionEvent);
    }
  };
  const processListener = vscodeApi.tasks.onDidEndTaskProcess((event) => {
    handleEndEvent("process", event);
  });
  let taskListener;

  try {
    taskListener = vscodeApi.tasks.onDidEndTask((event) => {
      handleEndEvent("task", event);
    });
    dispatchedExecution = await vscodeApi.tasks.executeTask(task);
    const matchingEarlyEvents = bufferedEvents.filter(
      ({ event }) => event.execution === dispatchedExecution,
    );
    const earlyCompletion =
      matchingEarlyEvents.find(({ kind }) => kind === "process") ??
      matchingEarlyEvents[0];
    bufferedEvents = [];
    const completed = earlyCompletion ?? await completion;
    if (completed.kind === "task") {
      throw new Error(
        "arXiv Daily CLI task ended without a process exit; launch may have failed or the task was cancelled.",
      );
    }
    if (completed.event.exitCode === 0) return true;
    if (completed.event.exitCode === undefined) {
      throw new Error(
        "arXiv Daily CLI task was cancelled before reporting an exit code.",
      );
    }
    throw new Error(
      `arXiv Daily CLI process exited with exit code ${completed.event.exitCode}.`,
    );
  } finally {
    processListener.dispose();
    taskListener?.dispose();
  }
}

function cliPathFromSettings(vscodeApi) {
  const configured = vscodeApi.workspace
    .getConfiguration("arxivDaily")
    .get("cliPath", DEFAULT_CLI_PATH);
  return String(configured || DEFAULT_CLI_PATH).trim() || DEFAULT_CLI_PATH;
}

function normalizeArxivId(input) {
  const candidate = String(input).trim();
  if (isValidModernArxivId(candidate)) {
    return candidate.replace(/v[1-9]\d*$/, "");
  }

  const urlMatch = ARXIV_URL_RE.exec(candidate);
  if (!urlMatch) return "";
  const [, pathKind, id, pdfSuffix] = urlMatch;
  if (pdfSuffix && pathKind !== "pdf") return "";
  return isValidModernArxivId(id)
    ? id.replace(/v[1-9]\d*$/, "")
    : "";
}

function isValidModernArxivId(candidate) {
  const match = MODERN_ARXIV_ID_RE.exec(candidate);
  if (!match) return false;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const sequence = match[3];
  if (month < 1 || month > 12 || Number(sequence) === 0) return false;

  const issue = year * 100 + month;
  if (issue < 704) return false;
  return issue <= 1412 ? sequence.length === 4 : sequence.length === 5;
}

module.exports = {
  normalizeArxivId,
  runForToday,
  summarizeById,
};
