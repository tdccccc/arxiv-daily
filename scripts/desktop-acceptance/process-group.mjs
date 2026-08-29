import { spawn as nodeSpawn } from "node:child_process";

/**
 * Reclaiming by process name or command-line pattern is unsafe here for two
 * separate reasons: the user keeps a real Obsidian session running, and a
 * pattern match against full command lines also matches the harness script that
 * carries the pattern in its own arguments. Only a numeric process group that
 * this harness created is an acceptable target.
 */
export function assertProcessGroupTarget(pgid, { ownProcessGroupId }) {
  if (typeof pgid !== "number" || !Number.isInteger(pgid) || pgid <= 1) {
    throw new TypeError(
      `process group must be an integer greater than 1, received ${JSON.stringify(pgid)}`,
    );
  }
  if (pgid === ownProcessGroupId) {
    throw new Error(`refusing to signal the harness's own process group ${pgid}`);
  }
  return pgid;
}

/**
 * Start a child that leads its own process group, so the whole Obsidian process
 * tree can later be reclaimed by group without naming anything.
 */
export function spawnInProcessGroup({ command, args, env, stdio }, { spawn = nodeSpawn } = {}) {
  const child = spawn(command, args, { detached: true, env, stdio });
  if (typeof child.pid !== "number") {
    throw new Error(`spawn returned no pid for ${command}`);
  }
  // detached: true puts the child in a new session, so its group id is its pid.
  return { child, pid: child.pid, pgid: child.pid };
}

function sendGroupSignal(kill, pgid, signal) {
  try {
    kill(-pgid, signal);
    return { delivered: true };
  } catch (error) {
    if (error?.code === "ESRCH") return { delivered: false };
    throw error;
  }
}

/**
 * Graduated reclamation: SIGTERM to the group, poll, then SIGKILL to the same
 * group if anything survived.
 */
export async function reclaimProcessGroup(
  { pgid, ownProcessGroupId },
  { kill, isAlive, sleep, escalateAfterMs = 5000, pollIntervalMs = 100 },
) {
  assertProcessGroupTarget(pgid, { ownProcessGroupId });

  const term = sendGroupSignal(kill, pgid, "SIGTERM");
  if (!term.delivered) return { escalated: false, alreadyGone: true };

  for (let waited = 0; waited < escalateAfterMs; waited += pollIntervalMs) {
    if (!(await isAlive(pgid))) return { escalated: false, alreadyGone: false };
    await sleep(pollIntervalMs);
  }
  if (!(await isAlive(pgid))) return { escalated: false, alreadyGone: false };

  const forced = sendGroupSignal(kill, pgid, "SIGKILL");
  return { escalated: true, alreadyGone: !forced.delivered };
}
