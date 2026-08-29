import assert from "node:assert/strict";
import test from "node:test";
import { describeBlockers, preflight } from "../desktop-acceptance/preflight.mjs";

function env({ present = [], pdfCount = 1 } = {}) {
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
  };
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
