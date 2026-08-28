import assert from "node:assert/strict";
import test from "node:test";
import {
  assertProcessGroupTarget,
  reclaimProcessGroup,
  spawnInProcessGroup,
} from "../desktop-acceptance/process-group.mjs";

const ownProcessGroupId = 4242;

function fakeKill() {
  const sent = [];
  const kill = (target, signal) => {
    sent.push([target, signal]);
  };
  kill.sent = sent;
  return kill;
}

test("spawnInProcessGroup detaches so the child leads its own group", () => {
  let seen;
  const spawn = (command, args, options) => {
    seen = { command, args, options };
    return { pid: 5150, unref() {} };
  };
  const handle = spawnInProcessGroup(
    { command: "/opt/Obsidian/obsidian", args: ["--remote-debugging-port=0"] },
    { spawn },
  );
  assert.equal(seen.options.detached, true);
  assert.equal(handle.pgid, 5150);
  assert.equal(handle.pid, 5150);
});

test("spawnInProcessGroup passes the isolated environment through unchanged", () => {
  let seen;
  const spawn = (command, args, options) => {
    seen = options;
    return { pid: 1, unref() {} };
  };
  const env = { XDG_CONFIG_HOME: "/tmp/harness/config" };
  spawnInProcessGroup({ command: "x", args: [], env }, { spawn });
  assert.deepEqual(seen.env, env);
});

test("spawnInProcessGroup refuses a child that reported no pid", () => {
  const spawn = () => ({ pid: undefined, unref() {} });
  assert.throws(() => spawnInProcessGroup({ command: "x", args: [] }, { spawn }), /pid/i);
});

test("a process group target must be a positive integer", () => {
  for (const bad of [0, -1, 1.5, Number.NaN, "1234", null, undefined]) {
    assert.throws(() => assertProcessGroupTarget(bad, { ownProcessGroupId }), /process group/i);
  }
});

test("process group 0 is refused because it means the caller's own group", () => {
  assert.throws(() => assertProcessGroupTarget(0, { ownProcessGroupId }), /own|caller|0/i);
});

test("init's process group is refused", () => {
  assert.throws(() => assertProcessGroupTarget(1, { ownProcessGroupId }), /process group/i);
});

test("the harness refuses to signal its own process group", () => {
  assert.throws(
    () => assertProcessGroupTarget(ownProcessGroupId, { ownProcessGroupId }),
    /own process group/i,
  );
});

test("a name or command-line pattern is not a valid target", () => {
  for (const pattern of ["obsidian", "remote-debugging-port=9222", "/opt/Obsidian/obsidian"]) {
    assert.throws(() => assertProcessGroupTarget(pattern, { ownProcessGroupId }), /process group/i);
  }
});

test("reclaim signals the whole group, never a bare pid", async () => {
  const kill = fakeKill();
  await reclaimProcessGroup(
    { pgid: 5150, ownProcessGroupId },
    { kill, isAlive: () => false, sleep: async () => {} },
  );
  assert.equal(kill.sent.length, 1);
  assert.deepEqual(kill.sent[0], [-5150, "SIGTERM"]);
});

test("reclaim escalates to SIGKILL when the group outlives SIGTERM", async () => {
  const kill = fakeKill();
  const result = await reclaimProcessGroup(
    { pgid: 5150, ownProcessGroupId },
    { kill, isAlive: () => true, sleep: async () => {}, escalateAfterMs: 30, pollIntervalMs: 10 },
  );
  assert.deepEqual(kill.sent, [
    [-5150, "SIGTERM"],
    [-5150, "SIGKILL"],
  ]);
  assert.equal(result.escalated, true);
});

test("reclaim does not escalate against a group that already exited", async () => {
  const kill = fakeKill();
  let probes = 0;
  const result = await reclaimProcessGroup(
    { pgid: 5150, ownProcessGroupId },
    {
      kill,
      isAlive: () => {
        probes += 1;
        return probes < 2;
      },
      sleep: async () => {},
      escalateAfterMs: 100,
      pollIntervalMs: 10,
    },
  );
  assert.deepEqual(kill.sent, [[-5150, "SIGTERM"]]);
  assert.equal(result.escalated, false);
});

test("reclaim treats an already-gone group as success", async () => {
  const kill = () => {
    const error = new Error("no such process");
    error.code = "ESRCH";
    throw error;
  };
  const result = await reclaimProcessGroup(
    { pgid: 5150, ownProcessGroupId },
    { kill, isAlive: () => false, sleep: async () => {} },
  );
  assert.equal(result.alreadyGone, true);
});

test("reclaim surfaces a permission error instead of reporting success", async () => {
  const kill = () => {
    const error = new Error("operation not permitted");
    error.code = "EPERM";
    throw error;
  };
  await assert.rejects(
    () =>
      reclaimProcessGroup(
        { pgid: 5150, ownProcessGroupId },
        { kill, isAlive: () => false, sleep: async () => {} },
      ),
    /EPERM|not permitted/i,
  );
});

test("reclaim validates the target before sending anything", async () => {
  const kill = fakeKill();
  await assert.rejects(
    () =>
      reclaimProcessGroup(
        { pgid: ownProcessGroupId, ownProcessGroupId },
        { kill, isAlive: () => false, sleep: async () => {} },
      ),
    /own process group/i,
  );
  assert.equal(kill.sent.length, 0);
});
