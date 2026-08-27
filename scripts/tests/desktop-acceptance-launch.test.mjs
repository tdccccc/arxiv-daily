import assert from "node:assert/strict";
import test from "node:test";
import {
  buildIsolatedEnv,
  buildLaunchCommand,
  waitForCdp,
} from "../desktop-acceptance/launch.mjs";

const obsidianPath = "/opt/Obsidian/obsidian";

test("the launch command runs Obsidian under a virtual display by default", () => {
  const { command, args } = buildLaunchCommand({ obsidianPath, port: 9333 });
  assert.equal(command, "xvfb-run");
  assert.ok(args.includes(obsidianPath));
  assert.ok(args.includes("--remote-debugging-port=9333"));
});

test("the launch command can target the real display when asked", () => {
  const { command, args } = buildLaunchCommand({ obsidianPath, port: 9333, virtualDisplay: false });
  assert.equal(command, obsidianPath);
  assert.deepEqual(args, ["--remote-debugging-port=9333", "--no-sandbox"]);
});

test("the launch command refuses a port outside the ephemeral range", () => {
  for (const bad of [0, -1, 80, 65536, 1.5, "9222"]) {
    assert.throws(() => buildLaunchCommand({ obsidianPath, port: bad }), /port/i);
  }
});

test("the isolated environment redirects every XDG directory into the sandbox", () => {
  const env = buildIsolatedEnv({
    configHome: "/tmp/h/config",
    dataHome: "/tmp/h/data",
    cacheHome: "/tmp/h/cache",
    baseEnv: { PATH: "/usr/bin", XDG_CONFIG_HOME: "/home/tester/.config" },
  });
  assert.equal(env.XDG_CONFIG_HOME, "/tmp/h/config");
  assert.equal(env.XDG_DATA_HOME, "/tmp/h/data");
  assert.equal(env.XDG_CACHE_HOME, "/tmp/h/cache");
  assert.equal(env.PATH, "/usr/bin");
});

test("the isolated environment never inherits the real Obsidian config home", () => {
  const env = buildIsolatedEnv({
    configHome: "/tmp/h/config",
    dataHome: "/tmp/h/data",
    cacheHome: "/tmp/h/cache",
    baseEnv: { XDG_CONFIG_HOME: "/home/tester/.config" },
  });
  assert.notEqual(env.XDG_CONFIG_HOME, "/home/tester/.config");
});

test("waitForCdp returns the browser version once the endpoint answers", async () => {
  let attempts = 0;
  const fetchImpl = async () => {
    attempts += 1;
    if (attempts < 3) throw new Error("ECONNREFUSED");
    return { ok: true, json: async () => ({ Browser: "Chrome/142", webSocketDebuggerUrl: "ws://x" }) };
  };
  const version = await waitForCdp({
    port: 9333,
    fetch: fetchImpl,
    sleep: async () => {},
    attempts: 10,
  });
  assert.equal(version.Browser, "Chrome/142");
  assert.equal(attempts, 3);
});

test("waitForCdp gives up with a diagnosable error", async () => {
  await assert.rejects(
    () =>
      waitForCdp({
        port: 9333,
        fetch: async () => {
          throw new Error("ECONNREFUSED");
        },
        sleep: async () => {},
        attempts: 3,
      }),
    /9333[\s\S]*3 attempts|did not answer/i,
  );
});

test("waitForCdp only talks to loopback", async () => {
  const seen = [];
  await waitForCdp({
    port: 9333,
    fetch: async (url) => {
      seen.push(url);
      return { ok: true, json: async () => ({ Browser: "Chrome/142" }) };
    },
    sleep: async () => {},
    attempts: 1,
  });
  assert.equal(seen.length, 1);
  assert.match(seen[0], /^http:\/\/127\.0\.0\.1:9333\//);
});
