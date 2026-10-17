import { readFileSync } from "node:fs";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { clearTimer, setTimer, throwIfCancelled, type StorageLock, type StorageLockOptions } from "@arxiv-daily/core";

/**
 * What makes a PID refer to one particular process. A bare PID is recycled
 * quickly (always after a reboot), so a lock whose owner died would otherwise
 * look held by whatever unrelated process now has that number.
 */
type ProcessIdentity =
  | { kind: "linux"; bootId: string; startTime: string }
  | { kind: "boot-time"; host: string; bootTimeMs: number };

interface Owner {
  version: 1;
  generation: number;
  pid: number;
  token: string;
  /** Absent in records written by older versions; those keep PID-only liveness. */
  identity?: ProcessIdentity;
}

/** Wall-clock drift tolerated when comparing approximate boot times (non-Linux). */
const BOOT_TIME_TOLERANCE_MS = 5 * 60 * 1000;

export interface NodeFileLockOptions {
  lockRoot?: string;
  beforeOwnerPublish?: () => Promise<void>;
  /** Told when a lock is taken over because its recorded owner can no longer be that process. */
  onStaleOwner?: (message: string) => void;
}
interface Decision {
  version: 1;
  token: string;
  kind: "released" | "recovered";
}

/** Local PID namespace only; no time-based takeover of a live owner. */
export class NodeFileLock {
  constructor(
    private readonly root: string,
    private readonly options: NodeFileLockOptions = {},
  ) {}

  async acquire(key: string, options: StorageLockOptions = {}): Promise<StorageLock | null> {
    throwIfCancelled(options.signal);
    await fs.mkdir(this.root, { recursive: true });
    const canonicalRoot = await fs.realpath(this.root);
    const base = this.options.lockRoot ?? path.join(os.homedir(), ".arxiv-daily", "host-locks");
    await fs.mkdir(base, { recursive: true, mode: 0o700 });
    await requireDirectory(base);
    const rootIdentity = process.platform === "win32" ? canonicalRoot.toLowerCase() : canonicalRoot;
    const name = createHash("sha256").update(`${rootIdentity}\0${key}`).digest("hex");
    const dir = path.join(base, name);
    await fs.mkdir(dir, { mode: 0o700 }).catch((error: unknown) => {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    });
    const identity = await requireDirectory(dir);
    const assertCurrent = async () => {
      const current = await requireDirectory(dir);
      if (current.dev !== identity.dev || current.ino !== identity.ino) {
        throw new Error("shared lock namespace changed");
      }
    };
    const deadline = performance.now() + (options.timeoutMs ?? 30_000);
    for (;;) {
      throwIfCancelled(options.signal);
      await assertCurrent();
      let lease: StorageLock | null;
      try { lease = await this.tryAcquire(dir, assertCurrent); }
      catch (error) {
        // A concurrent holder can prune old generations after publishing a newer
        // one. Re-read the latest generation instead of treating this as empty.
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
        if (!options.wait) return null;
        if (performance.now() >= deadline) throw new Error("shared storage lock timed out");
        continue;
      }
      if (lease) {
        if (options.signal?.aborted) {
          await lease.release();
          throwIfCancelled(options.signal);
        }
        return lease;
      }
      if (!options.wait) return null;
      if (performance.now() >= deadline) throw new Error("shared storage lock timed out");
      await pause(options.signal);
    }
  }

  private async tryAcquire(dir: string, assertCurrent: () => Promise<void>): Promise<StorageLock | null> {
    const names = await fs.readdir(dir);
    const generations = names.flatMap((name) => {
      const match = /^g(\d+)\.owner\.json$/.exec(name);
      return match ? [Number(match[1])] : [];
    });
    const latest = generations.length ? Math.max(...generations) : -1;
    if (!Number.isSafeInteger(latest) || latest >= Number.MAX_SAFE_INTEGER) {
      throw new Error("shared lock generation is invalid");
    }
    if (latest >= 0) {
      const previous = await readOwner(dir, latest);
      let decision = await readDecision(dir, previous);
      if (!decision) {
        if (pidIsAlive(previous.pid)) {
          const staleReason = staleIdentityReason(previous);
          if (!staleReason) return null;
          (this.options.onStaleOwner ?? defaultStaleOwnerNotice)(
            `recovering stale lock: recorded owner pid ${previous.pid} is no longer that process (${staleReason})`,
          );
        }
        await assertCurrent();
        await publish(path.join(dir, `g${latest}.decision.json`), {
          version: 1, token: previous.token, kind: "recovered",
        } satisfies Decision);
        decision = await readDecision(dir, previous);
        if (!decision) throw new Error("shared lock recovery decision is missing");
      }
    }
    const identity = captureProcessIdentity(process.pid);
    const owner: Owner = {
      version: 1,
      generation: latest + 1,
      pid: process.pid,
      token: randomUUID(),
      ...(identity ? { identity } : {}),
    };
    await assertCurrent();
    await this.options.beforeOwnerPublish?.();
    if (!await publish(path.join(dir, `g${owner.generation}.owner.json`), owner)) return null;
    // Releases prune generations two behind them, so an acquirer that stalled
    // here can re-create an already pruned owner record. Pruning never removes
    // the newest generation, so a newer one still exists in that case: close
    // the stale claim and start over from the real latest generation.
    if (await hasNewerGeneration(dir, owner.generation)) {
      await publish(path.join(dir, `g${owner.generation}.decision.json`), {
        version: 1, token: owner.token, kind: "released",
      } satisfies Decision).catch(() => false);
      return this.tryAcquire(dir, assertCurrent);
    }
    let released = false;
    return {
      release: async () => {
        if (released) return;
        await assertCurrent();
        await publish(path.join(dir, `g${owner.generation}.decision.json`), {
          version: 1, token: owner.token, kind: "released",
        } satisfies Decision);
        const decision = await readDecision(dir, owner);
        if (decision?.kind !== "released") throw new Error("shared lock release lost ownership");
        released = true;
        // Keep the newest and previous generations; never reset numbering or
        // remove the record from which a new acquisition derives its identity.
        const stale = (await fs.readdir(dir).catch(() => [] as string[])).filter((name) => {
          const match = /^g(\d+)\.(?:owner|decision)\.json$/.exec(name);
          return match && Number(match[1]) < owner.generation - 1;
        });
        await Promise.all(stale.map((name) => fs.rm(path.join(dir, name), { force: true }).catch(() => undefined)));
      },
    };
  }
}

async function hasNewerGeneration(dir: string, generation: number): Promise<boolean> {
  return (await fs.readdir(dir)).some((name) => {
    const match = /^g(\d+)\.owner\.json$/.exec(name);
    return match !== null && Number(match[1]) > generation;
  });
}

async function requireDirectory(dir: string) {
  const stat = await fs.lstat(dir);
  if (stat.isSymbolicLink() || !stat.isDirectory()) throw new Error("unsafe shared lock directory");
  return stat;
}

function pidIsAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true; }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ESRCH") return false;
    // Permission failures and unknown liveness are not evidence of exit.
    return true;
  }
}

function defaultStaleOwnerNotice(message: string): void {
  console.warn(`[arxiv-daily] ${message}`);
}

function captureProcessIdentity(pid: number): ProcessIdentity | undefined {
  try {
    if (process.platform === "linux") {
      const bootId = readBootId();
      const startTime = readLinuxStartTime(pid);
      if (bootId && startTime) return { kind: "linux", bootId, startTime };
    }
    return {
      kind: "boot-time",
      host: os.hostname(),
      bootTimeMs: Math.round(Date.now() - os.uptime() * 1000),
    };
  } catch {
    return undefined;
  }
}

/** Why the recorded owner cannot be the live process that now has its PID; null when it may be. */
function staleIdentityReason(owner: Owner): string | null {
  const recorded = owner.identity;
  if (!recorded) return null;
  try {
    if (recorded.kind === "linux") {
      const bootId = readBootId();
      if (!bootId) return null;
      if (bootId !== recorded.bootId) return "the machine has rebooted since";
      const startTime = readLinuxStartTime(owner.pid);
      if (startTime === undefined) return null;
      return startTime !== recorded.startTime ? "the PID was reused by a newer process" : null;
    }
    // Without /proc a reused PID within one boot cannot be told apart, but a
    // reboot can. The host name is recorded for diagnosis only: it changes
    // with DHCP and renames, which must not break a lock held by a live run.
    const bootTimeMs = Date.now() - os.uptime() * 1000;
    return Math.abs(bootTimeMs - recorded.bootTimeMs) > BOOT_TIME_TOLERANCE_MS
      ? "the machine has rebooted since"
      : null;
  } catch {
    return null;
  }
}

function readBootId(): string | undefined {
  try {
    return readFileSync("/proc/sys/kernel/random/boot_id", "utf8").trim() || undefined;
  } catch {
    return undefined;
  }
}

/** Field 22 of /proc/<pid>/stat (clock ticks since boot at process start). */
function readLinuxStartTime(pid: number): string | undefined {
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, "utf8");
    // The command name is parenthesised and may itself contain spaces or ')'.
    const fields = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
    return fields[19] || undefined;
  } catch {
    return undefined;
  }
}

function isProcessIdentity(value: unknown): value is ProcessIdentity {
  if (!value || typeof value !== "object") return false;
  const identity = value as Record<string, unknown>;
  if (identity.kind === "linux") {
    return typeof identity.bootId === "string" && typeof identity.startTime === "string";
  }
  return identity.kind === "boot-time"
    && typeof identity.host === "string"
    && typeof identity.bootTimeMs === "number"
    && Number.isFinite(identity.bootTimeMs);
}

async function readOwner(dir: string, generation: number): Promise<Owner> {
  const ownerPath = path.join(dir, `g${generation}.owner.json`);
  if (!(await fs.lstat(ownerPath)).isFile()) throw new Error("unsafe shared lock record");
  const value = JSON.parse(await fs.readFile(ownerPath, "utf8")) as Owner;
  if (value?.version !== 1 || value.generation !== generation || !Number.isSafeInteger(value.pid) || value.pid < 1 || typeof value.token !== "string" || !value.token
    || (value.identity !== undefined && !isProcessIdentity(value.identity))) {
    throw new Error("shared lock ownership record is invalid");
  }
  return value;
}

async function readDecision(dir: string, owner: Owner): Promise<Decision | null> {
  let text: string;
  try { text = await fs.readFile(path.join(dir, `g${owner.generation}.decision.json`), "utf8"); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return null; throw error; }
  const value = JSON.parse(text) as Decision;
  if (value?.version !== 1 || value.token !== owner.token || !["released", "recovered"].includes(value.kind)) {
    throw new Error("shared lock decision record is invalid");
  }
  return value;
}

async function publish(target: string, value: Owner | Decision): Promise<boolean> {
  const tmp = `${target}.${process.pid}-${randomUUID()}.tmp`;
  try {
    const file = await fs.open(tmp, "wx", 0o600);
    try {
      await file.writeFile(JSON.stringify(value), "utf8");
      await file.sync();
    } finally { await file.close(); }
    try {
      await fs.link(tmp, target);
      return true;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "EEXIST") return false;
      throw error;
    }
  } finally { await fs.rm(tmp, { force: true }).catch(() => undefined); }
}

function pause(signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimer(() => { signal?.removeEventListener("abort", abort); resolve(); }, 15);
    const abort = () => {
      clearTimer(timer);
      signal?.removeEventListener("abort", abort);
      try {
        throwIfCancelled(signal);
      } catch (error) {
        reject(error instanceof Error ? error : new Error(String(error), { cause: error }));
      }
    };
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
  });
}
