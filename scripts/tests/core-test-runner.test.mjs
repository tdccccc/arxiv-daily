import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  assertBatchCoverage,
  discoverCoreTestFiles,
  partitionTestFiles,
  runCoreTests,
} from "../run-core-tests.mjs";

async function withCoreFixture(files, callback) {
  const coreDir = await mkdtemp(join(tmpdir(), "arxiv-daily-core-runner-"));
  try {
    for (const file of files) {
      const path = join(coreDir, file);
      await mkdir(join(path, ".."), { recursive: true });
      await writeFile(path, "// fixture\n");
    }
    await callback(coreDir);
  } finally {
    await rm(coreDir, { recursive: true, force: true });
  }
}

function recordingSpawn(statuses = []) {
  const calls = [];
  return {
    calls,
    spawn(command, args, options) {
      const status = statuses[calls.length] ?? 0;
      calls.push({ command, args, options });
      return { status };
    },
  };
}

function batchFiles(call) {
  const marker = call.args.indexOf("--maxWorkers=1");
  assert.notEqual(marker, -1, "default batches must bound each Vitest child to one worker");
  return call.args.slice(marker + 1);
}

test("discovers Core tests recursively in deterministic order and batches complete coverage", async () => {
  await withCoreFixture([
    "tests/z-last.test.ts",
    "tests/services/beta.test.ts",
    "tests/a-first.test.ts",
    "tests/services/alpha.test.ts",
    "tests/ignored.spec.ts",
    "tests/not-a-test.ts",
  ], async (coreDir) => {
    const discovered = discoverCoreTestFiles(coreDir);
    assert.deepEqual(discovered, [
      "tests/a-first.test.ts",
      "tests/services/alpha.test.ts",
      "tests/services/beta.test.ts",
      "tests/z-last.test.ts",
    ]);

    const batches = partitionTestFiles(discovered, 3);
    assert.deepEqual(batches, [
      ["tests/a-first.test.ts", "tests/services/alpha.test.ts", "tests/services/beta.test.ts"],
      ["tests/z-last.test.ts"],
    ]);
    assert.doesNotThrow(() => assertBatchCoverage(discovered, batches, 3));
  });
});

test("batch invariants reject duplicate, omitted, reordered, and oversized file plans", () => {
  const discovered = ["tests/a.test.ts", "tests/b.test.ts", "tests/c.test.ts"];
  assert.throws(
    () => assertBatchCoverage(discovered, [["tests/a.test.ts", "tests/b.test.ts"], ["tests/b.test.ts"]], 2),
    /exactly once/,
  );
  assert.throws(
    () => assertBatchCoverage(discovered, [["tests/a.test.ts"], ["tests/c.test.ts"]], 2),
    /exactly once/,
  );
  assert.throws(
    () => assertBatchCoverage(discovered, [["tests/b.test.ts"], ["tests/a.test.ts", "tests/c.test.ts"]], 2),
    /deterministic order/,
  );
  assert.throws(
    () => assertBatchCoverage(discovered, [["tests/a.test.ts", "tests/b.test.ts", "tests/c.test.ts"]], 2),
    /batch size/,
  );
  assert.throws(() => partitionTestFiles(discovered, 0), /positive integer/);
  assert.throws(() => partitionTestFiles(discovered, Number.NaN), /positive integer/);
});

test("an empty default discovery fails closed before spawning Vitest", () => {
  const recorder = recordingSpawn();

  assert.throws(
    () => runCoreTests([], {
      discoveredFiles: [],
      spawn: recorder.spawn,
      vitestCli: "/fixture/vitest.mjs",
    }),
    /No Core test files were discovered/,
  );
  assert.equal(recorder.calls.length, 0);
});

test("default Core run invokes every discovered file exactly once in bounded short-lived children", () => {
  const discovered = [
    "tests/a.test.ts",
    "tests/b.test.ts",
    "tests/c.test.ts",
    "tests/nested/d.test.ts",
    "tests/nested/e.test.ts",
  ];
  const recorder = recordingSpawn();

  const status = runCoreTests([], {
    batchSize: 2,
    discoveredFiles: discovered,
    spawn: recorder.spawn,
    vitestCli: "/fixture/vitest.mjs",
  });

  assert.equal(status, 0);
  assert.equal(recorder.calls.length, 3);
  assert.ok(recorder.calls.every(({ command }) => command === process.execPath));
  assert.ok(recorder.calls.every(({ options }) => options.stdio === "inherit"));
  assert.deepEqual(recorder.calls.flatMap(batchFiles), discovered);
  assert.equal(new Set(recorder.calls.flatMap(batchFiles)).size, discovered.length);
});

test("an explicit Core test target and all Vitest arguments are forwarded unchanged to one invocation", () => {
  const argv = [
    "tests/pipeline.test.ts",
    "--maxWorkers=1",
    "--reporter=verbose",
    "-t",
    "returns failed_transient",
  ];
  const recorder = recordingSpawn();

  const status = runCoreTests(argv, {
    discoveredFiles: () => {
      throw new Error("focused invocation must not discover or batch the full suite");
    },
    spawn: recorder.spawn,
    vitestCli: "/fixture/vitest.mjs",
  });

  assert.equal(status, 0);
  assert.equal(recorder.calls.length, 1);
  assert.deepEqual(recorder.calls[0].args.slice(4), argv);
  assert.equal(recorder.calls[0].args.includes("--maxWorkers=1"), true);
});

test("a failed batch produces a nonzero result without omitting later planned files", () => {
  const discovered = ["tests/a.test.ts", "tests/b.test.ts", "tests/c.test.ts"];
  const recorder = recordingSpawn([0, 7, 0]);

  const status = runCoreTests([], {
    batchSize: 1,
    discoveredFiles: discovered,
    spawn: recorder.spawn,
    vitestCli: "/fixture/vitest.mjs",
  });

  assert.equal(status, 7);
  assert.deepEqual(recorder.calls.flatMap(batchFiles), discovered);
});

test("a child spawn without an exit status fails closed", () => {
  const status = runCoreTests(["tests/pipeline.test.ts"], {
    spawn: () => ({ status: null, error: new Error("spawn failed") }),
    vitestCli: "/fixture/vitest.mjs",
  });

  assert.equal(status, 1);
});
