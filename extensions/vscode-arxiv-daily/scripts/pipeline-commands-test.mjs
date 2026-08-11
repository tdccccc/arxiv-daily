import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import assert from "node:assert/strict";
import test from "node:test";

const require = createRequire(import.meta.url);
const {
  normalizeArxivId,
  runForToday,
  summarizeById,
} = require("../src/pipeline-commands.js");
const cliContract = JSON.parse(
  readFileSync(
    new URL("../../../contracts/companion-cli-commands.json", import.meta.url),
    "utf8",
  ),
);
const contractArgv = new Map(
  cliContract.commands.map(({ id, argv }) => [id, argv]),
);

assert.equal(cliContract.schemaVersion, 1);
assert.deepEqual([...contractArgv.keys()], ["runForToday", "summarizeById"]);

test("normalizes only historically valid modern IDs and literal arxiv.org URLs", () => {
  const accepted = new Map([
    ["0704.0001", "0704.0001"],
    ["1412.9999v1", "1412.9999"],
    ["1501.00001", "1501.00001"],
    ["2606.12345v20", "2606.12345"],
    ["https://arxiv.org/abs/2606.12345", "2606.12345"],
    ["http://www.arxiv.org/abs/2606.12345v3#section", "2606.12345"],
    ["https://arxiv.org/pdf/2606.12345v2?download=1", "2606.12345"],
    ["https://arxiv.org/pdf/2606.12345.pdf", "2606.12345"],
  ]);
  for (const [input, expected] of accepted) {
    assert.equal(normalizeArxivId(input), expected, input);
  }

  const rejected = [
    "",
    "arXiv:2606.12345",
    "0703.0001",
    "0700.0001",
    "0713.0001",
    "1412.10000",
    "1501.0001",
    "1501.00000",
    "0704.0000",
    "2606.12345v0",
    "2606.12345v01",
    "2606.123456",
    "2606.12345suffix",
    "prefix2606.12345",
    "read 2606.12345 please",
    "2606.12345;touch${IFS}/tmp/pwn",
    "https://evil.test/abs/2606.12345",
    "https://evil.arxiv.org/abs/2606.12345",
    "https://user@arxiv.org/abs/2606.12345",
    "https://@arxiv.org/abs/2606.12345",
    "https://arxiv.org:443/abs/2606.12345",
    "http://arxiv.org:80/abs/2606.12345",
    "https://arxiv%2eorg/abs/2606.12345",
    "https://arⅹiv.org/abs/2606.12345",
    "https://arxiv.org/html/2606.12345",
    "https://arxiv.org/abs/2606.12345.pdf",
    "https://arxiv.org/abs/2606.12345/extra",
    "https://arxiv.org/%61bs/2606.12345",
    "https://arxiv.org/abs/2606%2e12345",
  ];
  for (const input of rejected) {
    assert.equal(normalizeArxivId(input), "", input);
  }
});

test("Run for Today uses a process task and waits for exit 0 without a workspace", async () => {
  const cliPath = "arxiv-daily;touch${IFS}/tmp/pwn";
  const vscodeApi = createMockVscodeApi({ cliPath, hasWorkspace: false });

  const observed = observe(runForToday(vscodeApi));
  const taskExecution = await waitForTaskExecution(vscodeApi);

  assert.deepEqual(vscodeApi.lifecycle.slice(0, 3), [
    "listen:process",
    "listen:task",
    "execute",
  ]);
  assert.equal(observed.settled, false, "launch must not be reported as completion");
  assertTaskExecution(
    vscodeApi,
    taskExecution.task,
    cliPath,
    contractArgv.get("runForToday"),
  );
  assert.equal(taskExecution.task.scope, vscodeApi.TaskScope.Workspace);
  assert.equal(vscodeApi.shellExecutionCount, 0);
  assert.equal(vscodeApi.activeProcessListenerCount, 1);
  assert.equal(vscodeApi.activeTaskListenerCount, 1);

  vscodeApi.endTaskProcess(taskExecution, 0);
  vscodeApi.endTask(taskExecution);

  assert.equal(await observed.promise, true);
  assertNoTaskListeners(vscodeApi);
});

test("Summarize by ID dispatches canonical argv through ProcessExecution", async () => {
  const cliPath = "C:\\Program Files\\arxiv daily & tools\\arxiv-daily.exe";
  const vscodeApi = createMockVscodeApi({
    cliPath,
    hasWorkspace: false,
    inputValue: "https://arxiv.org/pdf/2606.12345v9.pdf?download=1",
  });

  const runPromise = summarizeById(vscodeApi);
  const taskExecution = await waitForTaskExecution(vscodeApi);
  assertTaskExecution(
    vscodeApi,
    taskExecution.task,
    cliPath,
    contractArgv.get("summarizeById"),
  );

  vscodeApi.endTaskProcess(taskExecution, 0);
  vscodeApi.endTask(taskExecution);

  assert.equal(await runPromise, true);
  assertNoTaskListeners(vscodeApi);
});

test("captures a process end that races ahead of executeTask resolution", async () => {
  const vscodeApi = createMockVscodeApi({
    earlyEvent: "process",
    earlyExitCode: 0,
  });

  assert.equal(await runForToday(vscodeApi), true);
  assert.equal(vscodeApi.taskExecutions.length, 1);
  assertNoTaskListeners(vscodeApi);
});

test("rejects a task-only end that races ahead of executeTask resolution", async () => {
  const vscodeApi = createMockVscodeApi({ earlyEvent: "task" });

  await assert.rejects(
    settleWithin(runForToday(vscodeApi)),
    /ended without a process exit/i,
  );
  assertNoTaskListeners(vscodeApi);
});

test("rejects a task-only end after executeTask resolution", async () => {
  const vscodeApi = createMockVscodeApi();
  const runPromise = runForToday(vscodeApi);
  const taskExecution = await waitForTaskExecution(vscodeApi);
  await nextTurn();

  vscodeApi.endTask(taskExecution);

  await assert.rejects(
    settleWithin(runPromise),
    /ended without a process exit/i,
  );
  assertNoTaskListeners(vscodeApi);
});

test("keeps process success when the normal task-end event follows", async () => {
  const vscodeApi = createMockVscodeApi();
  const runPromise = runForToday(vscodeApi);
  const taskExecution = await waitForTaskExecution(vscodeApi);

  vscodeApi.endTaskProcess(taskExecution, 0);
  vscodeApi.endTask(taskExecution);

  assert.equal(await runPromise, true);
  assertNoTaskListeners(vscodeApi);
});

test("ignores process and task end events for other task executions", async () => {
  const vscodeApi = createMockVscodeApi();
  const observed = observe(runForToday(vscodeApi));
  const taskExecution = await waitForTaskExecution(vscodeApi);
  const unrelatedTask = new vscodeApi.Task(
    { type: "process" },
    vscodeApi.TaskScope.Workspace,
    "Unrelated",
    "test",
    new vscodeApi.ProcessExecution("unrelated", ["--version"]),
  );
  const unrelatedExecution = new vscodeApi.TaskExecution(unrelatedTask);

  vscodeApi.endTaskProcess(unrelatedExecution, 0);
  vscodeApi.endTask(unrelatedExecution);
  vscodeApi.endTaskProcess(new vscodeApi.TaskExecution(taskExecution.task), 0);
  vscodeApi.endTask(new vscodeApi.TaskExecution(taskExecution.task));
  await nextTurn();

  assert.equal(observed.settled, false);
  assert.equal(vscodeApi.activeProcessListenerCount, 1);
  assert.equal(vscodeApi.activeTaskListenerCount, 1);

  vscodeApi.endTaskProcess(taskExecution, 0);
  vscodeApi.endTask(taskExecution);

  assert.equal(await observed.promise, true);
  assertNoTaskListeners(vscodeApi);
});

test("propagates executeTask rejection and disposes both listeners", async () => {
  const launchError = new Error("process launch rejected");
  const vscodeApi = createMockVscodeApi({ launchError });

  await assert.rejects(runForToday(vscodeApi), (error) => error === launchError);
  assert.deepEqual(vscodeApi.lifecycle, [
    "listen:process",
    "listen:task",
    "execute",
    "dispose:process",
    "dispose:task",
  ]);
  assertNoTaskListeners(vscodeApi);
});

test("rejects a nonzero process exit and disposes both listeners", async () => {
  const vscodeApi = createMockVscodeApi();
  const runPromise = runForToday(vscodeApi);
  const taskExecution = await waitForTaskExecution(vscodeApi);

  vscodeApi.endTaskProcess(taskExecution, 17);

  await assert.rejects(runPromise, /exit code 17/);
  assertNoTaskListeners(vscodeApi);
});

test("rejects cancellation without an exit code and disposes both listeners", async () => {
  const vscodeApi = createMockVscodeApi();
  const runPromise = runForToday(vscodeApi);
  const taskExecution = await waitForTaskExecution(vscodeApi);

  vscodeApi.endTaskProcess(taskExecution, undefined);

  await assert.rejects(runPromise, /cancelled|exit code/i);
  assertNoTaskListeners(vscodeApi);
});

test("input cancellation does not dispatch or register task listeners", async () => {
  const vscodeApi = createMockVscodeApi({ inputValue: undefined });

  assert.equal(await summarizeById(vscodeApi), false);
  assert.equal(vscodeApi.executedTasks.length, 0);
  assert.equal(vscodeApi.taskExecutions.length, 0);
  assert.equal(vscodeApi.processListenerRegistrations, 0);
  assert.equal(vscodeApi.taskListenerRegistrations, 0);
  assert.equal(vscodeApi.activeProcessListenerCount, 0);
  assert.equal(vscodeApi.activeTaskListenerCount, 0);
});

function assertTaskExecution(vscodeApi, task, executable, expectedArgv) {
  assert(task.execution instanceof vscodeApi.ProcessExecution);
  assert.equal(task.definition.type, "process");
  assert.equal(task.execution.process, executable);
  assert.deepEqual(task.execution.args, expectedArgv);
  assert(cliContract.commands.some(({ argv }) => sameArgv(argv, task.execution.args)));
  for (const { id, argv } of cliContract.removedCommands) {
    assert.equal(
      sameArgv(argv, task.execution.args),
      false,
      `removed CLI form emitted: ${id}`,
    );
  }
}

function sameArgv(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function assertNoTaskListeners(vscodeApi) {
  assert.equal(vscodeApi.activeProcessListenerCount, 0);
  assert.equal(vscodeApi.activeTaskListenerCount, 0);
  assert.equal(vscodeApi.processListenerDisposals, 1);
  assert.equal(vscodeApi.taskListenerDisposals, 1);
}

function createMockVscodeApi(options = {}) {
  const cliPath = options.cliPath ?? "arxiv-daily";
  const inputValue = Object.hasOwn(options, "inputValue")
    ? options.inputValue
    : "";
  const hasWorkspace = options.hasWorkspace ?? true;
  const executedTasks = [];
  const taskExecutions = [];
  const processListeners = new Set();
  const taskListeners = new Set();
  const lifecycle = [];
  let processListenerRegistrations = 0;
  let processListenerDisposals = 0;
  let taskListenerRegistrations = 0;
  let taskListenerDisposals = 0;
  let shellExecutionCount = 0;

  class ProcessExecution {
    constructor(process, args) {
      this.process = process;
      this.args = [...args];
    }
  }

  class ShellExecution {
    constructor(command, args) {
      shellExecutionCount += 1;
      this.command = command;
      this.args = [...args];
    }
  }

  class Task {
    constructor(definition, scope, name, source, execution) {
      this.definition = definition;
      this.scope = scope;
      this.name = name;
      this.source = source;
      this.execution = execution;
    }
  }

  class TaskExecution {
    constructor(task) {
      this.task = task;
    }
  }

  const vscodeApi = {
    ProcessExecution,
    ShellExecution,
    Task,
    TaskExecution,
    TaskScope: {
      Global: 1,
      Workspace: 2,
    },
    executedTasks,
    taskExecutions,
    lifecycle,
    tasks: {
      onDidEndTaskProcess(listener) {
        lifecycle.push("listen:process");
        processListenerRegistrations += 1;
        processListeners.add(listener);
        let disposed = false;
        return {
          dispose() {
            if (disposed) return;
            disposed = true;
            lifecycle.push("dispose:process");
            processListenerDisposals += 1;
            processListeners.delete(listener);
          },
        };
      },
      onDidEndTask(listener) {
        lifecycle.push("listen:task");
        taskListenerRegistrations += 1;
        taskListeners.add(listener);
        let disposed = false;
        return {
          dispose() {
            if (disposed) return;
            disposed = true;
            lifecycle.push("dispose:task");
            taskListenerDisposals += 1;
            taskListeners.delete(listener);
          },
        };
      },
      async executeTask(task) {
        lifecycle.push("execute");
        executedTasks.push(task);
        if (options.launchError) throw options.launchError;
        const taskExecution = new TaskExecution(task);
        taskExecutions.push(taskExecution);
        if (options.earlyEvent === "process") {
          vscodeApi.endTaskProcess(taskExecution, options.earlyExitCode);
        }
        if (options.earlyEvent === "task") {
          vscodeApi.endTask(taskExecution);
        }
        return taskExecution;
      },
    },
    workspace: {
      workspaceFolders: hasWorkspace
        ? [{ name: "vault", uri: uri("/workspace/vault") }]
        : undefined,
      getConfiguration(section) {
        assert.equal(section, "arxivDaily");
        return {
          get(key, fallback) {
            assert.equal(key, "cliPath");
            return cliPath || fallback;
          },
        };
      },
    },
    window: {
      async showInputBox() {
        return inputValue;
      },
      showWarningMessage() {},
    },
    endTaskProcess(execution, exitCode) {
      for (const listener of [...processListeners]) {
        listener({ execution, exitCode });
      }
    },
    endTask(execution) {
      for (const listener of [...taskListeners]) {
        listener({ execution });
      }
    },
  };

  Object.defineProperties(vscodeApi, {
    activeProcessListenerCount: {
      get: () => processListeners.size,
    },
    activeTaskListenerCount: {
      get: () => taskListeners.size,
    },
    processListenerRegistrations: {
      get: () => processListenerRegistrations,
    },
    processListenerDisposals: {
      get: () => processListenerDisposals,
    },
    taskListenerRegistrations: {
      get: () => taskListenerRegistrations,
    },
    taskListenerDisposals: {
      get: () => taskListenerDisposals,
    },
    shellExecutionCount: {
      get: () => shellExecutionCount,
    },
  });

  return vscodeApi;
}

function observe(promise) {
  const state = { settled: false };
  state.promise = promise.then(
    (value) => {
      state.settled = true;
      return value;
    },
    (error) => {
      state.settled = true;
      throw error;
    },
  );
  return state;
}

async function waitForTaskExecution(vscodeApi) {
  for (let attempt = 0; attempt < 10; attempt += 1) {
    if (vscodeApi.taskExecutions.length > 0) {
      return vscodeApi.taskExecutions[0];
    }
    await nextTurn();
  }
  assert.fail("task execution was not dispatched");
}

function nextTurn() {
  return new Promise((resolve) => setImmediate(resolve));
}

async function settleWithin(promise, timeoutMs = 100) {
  let timeout;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timeout = setTimeout(
          () => reject(new Error("task command did not settle")),
          timeoutMs,
        );
      }),
    ]);
  } finally {
    clearTimeout(timeout);
  }
}

function uri(path) {
  const normalized = `/${String(path).split("/").filter(Boolean).join("/")}`;
  return {
    path: normalized,
    fsPath: normalized,
  };
}
