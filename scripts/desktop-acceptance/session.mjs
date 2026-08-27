import fsPromises from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { assertVersionUnderTest, deployBuildUnderTest } from "./build-deploy.mjs";
import { createCdpClient, evaluate, selectVaultTarget } from "./cdp.mjs";
import { createDiagnostics } from "./diagnostics.mjs";
import { buildIsolatedEnv, buildLaunchCommand, pickFreePort, waitForCdp } from "./launch.mjs";
import { reclaimProcessGroup, spawnInProcessGroup } from "./process-group.mjs";
import { assertIsolatedConfigHome, composeVaultConfig } from "./vault-config.mjs";
import { waitForPluginReady } from "./trust.mjs";
import { createVaultStateGuard } from "./vault-state.mjs";

/**
 * Node exposes no getpgid binding, so read the harness's own process group from
 * procfs. It is only ever used to refuse signalling ourselves.
 */
export async function readOwnProcessGroupId(fs = fsPromises) {
  const stat = await fs.readFile("/proc/self/stat", "utf8");
  // The comm field may contain spaces and parentheses; everything after the
  // final ')' is positional: state, ppid, pgrp, ...
  const fields = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
  const pgrp = Number(fields[2]);
  if (!Number.isInteger(pgrp)) throw new Error(`could not read own process group from procfs`);
  return pgrp;
}

/**
 * Run `body` against a real, isolated Obsidian.
 *
 * Everything destructive is bracketed: the vault's mutable state and the
 * artifacts this harness deploys are captured up front and restored on every
 * exit path, and the Obsidian process tree is reclaimed by its own process
 * group rather than by name.
 */
export async function runDesktopSession({
  vaultPath,
  pluginId,
  sourceDir,
  obsidianPath = "/opt/Obsidian/obsidian",
  virtualDisplay = true,
  ignoreDiagnostics = [],
  body,
  fs = fsPromises,
  homeDir = os.homedir(),
  tmpDir = os.tmpdir(),
}) {
  const sandbox = await fs.mkdtemp(path.join(tmpDir, "obsidian-acceptance-"));
  const configHome = path.join(sandbox, "config");
  assertIsolatedConfigHome(configHome, { realConfigHome: path.join(homeDir, ".config") });

  // Only state that cannot be regenerated is restored. The deployed build is
  // not: `npm run build` reproduces it, and leaving the branch build in place
  // matches the ordinary plugin development loop and keeps a failed run open
  // for manual inspection.
  const guard = createVaultStateGuard({ vaultPath, pluginId, fs });

  try {
    return await guard.protect(async () => {
      const { version } = await deployBuildUnderTest({ vaultPath, pluginId, sourceDir, fs });

      const obsidianConfigDir = path.join(configHome, "obsidian");
      await fs.mkdir(obsidianConfigDir, { recursive: true });
      await fs.mkdir(path.join(sandbox, "data"), { recursive: true });
      await fs.mkdir(path.join(sandbox, "cache"), { recursive: true });
      await fs.writeFile(
        path.join(obsidianConfigDir, "obsidian.json"),
        JSON.stringify(composeVaultConfig({ vaultPath, timestamp: Date.now() })),
      );

      const port = await pickFreePort();
      const { command, args } = buildLaunchCommand({ obsidianPath, port, virtualDisplay });
      const env = buildIsolatedEnv({
        configHome,
        dataHome: path.join(sandbox, "data"),
        cacheHome: path.join(sandbox, "cache"),
      });

      const ownProcessGroupId = await readOwnProcessGroupId(fs);
      const handle = spawnInProcessGroup({ command, args, env, stdio: "ignore" });
      let client;
      try {
        await waitForCdp({ port });
        const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
        const target = selectVaultTarget(targets);

        client = createCdpClient({ url: target.webSocketDebuggerUrl });
        await client.ready();

        // Order matters: diagnostics are enabled before the trust prompt is
        // accepted, and community plugins do not load until it is, so the whole
        // plugin startup falls inside the collection window.
        const diagnostics = await createDiagnostics(client, { ignore: ignoreDiagnostics });
        const evaluateInRenderer = (expression) => evaluate(client, expression);

        const { version: pluginVersion, trustPromptAccepted, loadedBeforeAttach } =
          await waitForPluginReady({ evaluate: evaluateInRenderer, pluginId });
        assertVersionUnderTest({ expected: version, reported: pluginVersion });

        return await body({
          port,
          sandbox,
          expectedVersion: version,
          session: {
            evaluate: evaluateInRenderer,
            diagnostics,
            client,
            pluginVersion,
            trustPromptAccepted,
            diagnosticsComplete: !loadedBeforeAttach,
          },
        });
      } finally {
        client?.close();
        await reclaimProcessGroup(
          { pgid: handle.pgid, ownProcessGroupId },
          {
            kill: process.kill.bind(process),
            isAlive: (pgid) => {
              try {
                process.kill(-pgid, 0);
                return true;
              } catch {
                return false;
              }
            },
            sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
          },
        );
      }
    });
  } finally {
    await fs.rm(sandbox, { recursive: true, force: true });
  }
}
