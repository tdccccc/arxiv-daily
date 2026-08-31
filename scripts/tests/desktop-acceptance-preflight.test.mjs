import assert from "node:assert/strict";
import test from "node:test";
import {
  WATCH_HEADROOM,
  describeBlockers,
  preflight,
  probeFileWatchCapacity,
} from "../desktop-acceptance/preflight.mjs";

function env({ present = [], pdfCount = 1, watchCapacity } = {}) {
  return {
    fs: {
      async access(path) {
        if (!present.includes(path)) {
          const error = new Error(`ENOENT: ${path}`);
          error.code = "ENOENT";
          throw error;
        }
      },
    },
    countPdfs: async () => pdfCount,
    which: async (command) => (present.includes(command) ? `/usr/bin/${command}` : null),
    probeWatchCapacity:
      watchCapacity ?? (async () => ({ ok: true, established: WATCH_HEADROOM, requested: WATCH_HEADROOM })),
  };
}

/** What the kernel raises once `fs.inotify.max_user_watches` is fully spoken for. */
function enospc() {
  const error = new Error("ENOSPC: System limit for number of file watchers reached, watch '/tmp/x'");
  error.code = "ENOSPC";
  return error;
}

const ok = {
  vaultPath: "/vault",
  obsidianPath: "/opt/Obsidian/obsidian",
  sourceDir: "/repo/plugin",
};
const allPresent = [
  "/vault",
  "/vault/.obsidian",
  "/opt/Obsidian/obsidian",
  "/repo/plugin/main.js",
  "/repo/plugin/manifest.json",
  "/repo/plugin/styles.css",
  "xvfb-run",
];

test("a complete environment reports no blockers", async () => {
  const result = await preflight(ok, env({ present: allPresent }));
  assert.deepEqual(result.blockers, []);
  assert.equal(result.ok, true);
});

test("a missing vault is named", async () => {
  const result = await preflight(ok, env({ present: allPresent.filter((p) => p !== "/vault") }));
  assert.equal(result.ok, false);
  assert.ok(result.blockers.some((b) => /vault/i.test(b.message)));
});

test("a directory that is not a vault is distinguished from a missing one", async () => {
  const result = await preflight(
    ok,
    env({ present: allPresent.filter((p) => p !== "/vault/.obsidian") }),
  );
  assert.ok(result.blockers.some((b) => /\.obsidian|not an obsidian vault/i.test(b.message)));
});

test("a missing Obsidian binary is named with its path", async () => {
  const result = await preflight(
    ok,
    env({ present: allPresent.filter((p) => p !== "/opt/Obsidian/obsidian") }),
  );
  assert.ok(result.blockers.some((b) => b.message.includes("/opt/Obsidian/obsidian")));
});

test("an unbuilt branch is reported as needing a build, not as a missing file", async () => {
  const result = await preflight(
    ok,
    env({ present: allPresent.filter((p) => p !== "/repo/plugin/main.js") }),
  );
  assert.ok(result.blockers.some((b) => /build/i.test(b.message)));
});

test("a missing virtual display is named", async () => {
  const result = await preflight(ok, env({ present: allPresent.filter((p) => p !== "xvfb-run") }));
  assert.ok(result.blockers.some((b) => /xvfb/i.test(b.message)));
});

test("a vault with no PDF is reported because page location cannot be exercised", async () => {
  const result = await preflight(ok, env({ present: allPresent, pdfCount: 0 }));
  assert.ok(result.blockers.some((b) => /pdf/i.test(b.message)));
});

test("every blocker is collected, not just the first", async () => {
  const result = await preflight(ok, env({ present: [], pdfCount: 0 }));
  assert.ok(result.blockers.length >= 4);
});

test("each blocker carries an actionable remedy", async () => {
  const result = await preflight(ok, env({ present: [], pdfCount: 0 }));
  for (const blocker of result.blockers) {
    assert.ok(blocker.remedy && blocker.remedy.length > 0, `${blocker.message} has no remedy`);
  }
});

test("describeBlockers renders a readable, actionable report", () => {
  const text = describeBlockers([
    { message: "Obsidian was not found at /opt/Obsidian/obsidian", remedy: "install Obsidian" },
  ]);
  assert.match(text, /Obsidian was not found/);
  assert.match(text, /install Obsidian/);
});

test("a missing virtual display is not a blocker when the real display is used", async () => {
  const result = await preflight(
    { ...ok, virtualDisplay: false },
    env({ present: allPresent.filter((p) => p !== "xvfb-run") }),
  );
  assert.equal(result.blockers.some((b) => /xvfb/i.test(b.message)), false);
});

// ── file-watch capacity ─────────────────────────────────────────────────────
//
// A real ENOSPC needs the machine's watch quota to be exhausted, which needs
// root to arrange. These inject the probe's answer instead, so the blocker,
// its exit path and its remedy are pinned deterministically.

test("an exhausted file-watch quota blocks the run rather than letting Obsidian open on an error page", async () => {
  const result = await preflight(
    ok,
    env({
      present: allPresent,
      watchCapacity: async () => ({
        ok: false,
        code: "ENOSPC",
        message: "ENOSPC: System limit for number of file watchers reached, watch '/tmp/x'",
        established: 0,
        requested: 128,
        limit: 65536,
      }),
    }),
  );
  assert.equal(result.ok, false);
  const blocker = result.blockers.find((b) => /watch/i.test(b.message));
  assert.ok(blocker, `no file-watch blocker in ${JSON.stringify(result.blockers)}`);
  assert.match(blocker.message, /ENOSPC/);
  // The blocker has to say the quota is spent now, not merely quote the ceiling.
  assert.match(blocker.message, /0 of 128/);
  assert.match(blocker.remedy, /fs\.inotify\.max_user_watches/);
  assert.match(blocker.remedy, /sysctl/);
  assert.match(blocker.remedy, /\/etc\/sysctl\.d\//);
});

test("the file-watch blocker names the ceiling as context but is not decided by it", async () => {
  const result = await preflight(
    ok,
    env({
      present: allPresent,
      // A very high ceiling that is nonetheless fully spent: reading the limit
      // alone would have called this environment healthy.
      watchCapacity: async () => ({
        ok: false,
        code: "ENOSPC",
        message: "ENOSPC",
        established: 3,
        requested: 128,
        limit: 524288,
      }),
    }),
  );
  const blocker = result.blockers.find((b) => /watch/i.test(b.message));
  assert.ok(blocker);
  assert.match(blocker.message, /524288/);
  assert.match(blocker.message, /3 of 128/);
});

test("a healthy watch quota adds no blocker", async () => {
  const result = await preflight(ok, env({ present: allPresent }));
  assert.equal(result.blockers.some((b) => /watch/i.test(b.message)), false);
});

test("the probe establishes real watches, so it reports live headroom rather than a static ceiling", async () => {
  const opened = [];
  const closed = [];
  const outcome = await probeFileWatchCapacity({
    headroom: 5,
    watch: (dir) => {
      opened.push(dir);
      return { close: () => closed.push(dir) };
    },
    readLimit: async () => 65536,
  });
  assert.equal(outcome.ok, true);
  assert.equal(outcome.established, 5);
  // Distinct paths, because inotify hands back the same descriptor for a path
  // it already watches — probing one directory five times would consume one
  // slot and prove nothing about headroom.
  assert.equal(new Set(opened).size, 5);
  // The probe must give back every slot it took, or it degrades what it measures.
  assert.deepEqual(closed.sort(), opened.sort());
});

test("the probe reports how many watches it got before the kernel refused, and releases those", async () => {
  const closed = [];
  let established = 0;
  const outcome = await probeFileWatchCapacity({
    headroom: 128,
    watch: (dir) => {
      if (established >= 3) throw enospc();
      established += 1;
      return { close: () => closed.push(dir) };
    },
    readLimit: async () => 65536,
  });
  assert.equal(outcome.ok, false);
  assert.equal(outcome.code, "ENOSPC");
  assert.equal(outcome.established, 3);
  assert.equal(outcome.requested, 128);
  assert.equal(outcome.limit, 65536);
  assert.equal(closed.length, 3);
});

test("a probe failure that is not a quota refusal is not reported as an exhausted quota", async () => {
  const outcome = await probeFileWatchCapacity({
    headroom: 8,
    watch: () => {
      const error = new Error("EPERM: operation not permitted");
      error.code = "EPERM";
      throw error;
    },
    readLimit: async () => null,
  });
  // Unknown ground is not a verdict: the harness refuses to invent a blocker it
  // cannot substantiate, and says why it could not measure.
  assert.equal(outcome.ok, true);
  assert.equal(outcome.measured, false);
  assert.match(outcome.reason, /EPERM/);
});

test("the real probe, run against this machine, actually opens and releases watches", async () => {
  const outcome = await probeFileWatchCapacity({ headroom: 16 });
  assert.equal(outcome.ok, true);
  assert.equal(outcome.measured, true);
  assert.equal(outcome.established, 16);
});
