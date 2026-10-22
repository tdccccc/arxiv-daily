import { expect, it, vi } from "vitest";
import { SchedulerService } from "../src/services/scheduler";
import { createStorageStateStore } from "../src/services/state-store";
import { RunLock } from "../src/services/run-lock";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { StorageAdapter } from "../src/core/adapters";

const DATE = "2026-10-01";
const PRIMARY = "arxiv-daily/.index/run-state.json";
const CONFLICT = "<<<<<<< HEAD\n{}\n=======\n{}\n>>>>>>> other\n";

function memoryStorage(initial: Record<string, string>) {
  const files = { ...initial };
  const dirs = new Set<string>();
  const storage = {
    normalizePath: (path: string) => path.replace(/\\/g, "/"),
    readText: async (path: string) => files[path]!,
    writeText: async (path: string, content: string) => { files[path] = content; },
    exists: async (path: string) => path in files || dirs.has(path),
    mkdir: async (path: string) => { dirs.add(path); },
    rename: async (from: string, to: string) => { files[to] = files[from]!; delete files[from]; },
    remove: async (path: string) => { delete files[path]; },
  } satisfies StorageAdapter;
  return { files, storage };
}

it("fails a run with a readable reason while run-state.json is damaged, and runs again once it is cleared", async () => {
  const { files, storage } = memoryStorage({ [PRIMARY]: CONFLICT });
  const store = createStorageStateStore(storage, DEFAULT_SETTINGS.output, { warn: vi.fn() });
  const run = vi.fn(async () => ({ kind: "completed" as const, papersWritten: 2 }));
  const scheduler = new SchedulerService({
    getSettings: () => DEFAULT_SETTINGS, store, lock: new RunLock(), runForDate: run,
    logger: new Logger("error"), now: () => new Date(`${DATE}T10:00:00Z`),
  });

  const failed = await scheduler.runForDateNow(DATE);

  expect(failed).toMatchObject({ kind: "failed_transient" });
  const reason = (failed as { reason: string }).reason;
  expect(reason).toContain(PRIMARY);
  expect(reason).toContain("Clear arXiv Daily run state");
  expect(reason).not.toMatch(/^Unexpected token/);
  expect(run).not.toHaveBeenCalled();
  expect(files[PRIMARY]).toBe(CONFLICT);

  await store.clearAll();

  expect(await scheduler.runForDateNow(DATE)).toMatchObject({ kind: "completed", papersWritten: 2 });
  expect(run).toHaveBeenCalledTimes(1);
  expect(JSON.parse(files[PRIMARY]).runState[DATE].status).toBe("completed");
  expect(Object.keys(files).filter(name => name.includes(".corrupt-"))).toHaveLength(1);
});
