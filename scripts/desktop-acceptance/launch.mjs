import net from "node:net";

const EPHEMERAL_MIN = 1024;
const EPHEMERAL_MAX = 65535;

/**
 * Obsidian is launched under a virtual display by default so an acceptance run
 * never steals focus from, or draws over, the user's real desktop session.
 */
export function buildLaunchCommand({
  obsidianPath,
  port,
  virtualDisplay = true,
  screen = "1400x900x24",
}) {
  if (typeof port !== "number" || !Number.isInteger(port) || port < EPHEMERAL_MIN || port > EPHEMERAL_MAX) {
    throw new TypeError(`debugging port must be an integer in [${EPHEMERAL_MIN}, ${EPHEMERAL_MAX}], received ${JSON.stringify(port)}`);
  }
  const obsidianArgs = [`--remote-debugging-port=${port}`, "--no-sandbox"];
  if (!virtualDisplay) return { command: obsidianPath, args: obsidianArgs };
  return {
    command: "xvfb-run",
    args: ["-a", `--server-args=-screen 0 ${screen}`, obsidianPath, ...obsidianArgs],
  };
}

/**
 * Every XDG directory is redirected, so Obsidian cannot reach the user's real
 * vault list even if it consults a directory the harness did not anticipate.
 */
export function buildIsolatedEnv({ configHome, dataHome, cacheHome, baseEnv = process.env }) {
  return {
    ...baseEnv,
    XDG_CONFIG_HOME: configHome,
    XDG_DATA_HOME: dataHome,
    XDG_CACHE_HOME: cacheHome,
  };
}

/** Ask the OS for a free port instead of competing for a fixed one. */
export function pickFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

/** Poll the loopback CDP endpoint until Obsidian's renderer is reachable. */
export async function waitForCdp({
  port,
  fetch: fetchImpl = fetch,
  sleep = (ms) => new Promise((r) => setTimeout(r, ms)),
  attempts = 40,
  intervalMs = 500,
}) {
  const url = `http://127.0.0.1:${port}/json/version`;
  let lastError;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetchImpl(url);
      if (response.ok) return await response.json();
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await sleep(intervalMs);
  }
  throw new Error(
    `CDP on 127.0.0.1:${port} did not answer within ${attempts} attempts: ${lastError?.message ?? "unknown"}`,
  );
}
