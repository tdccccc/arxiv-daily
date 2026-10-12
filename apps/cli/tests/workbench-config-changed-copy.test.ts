import { afterEach, expect, it } from "vitest";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { readWorkbenchSettings, saveWorkbenchSettings } from "../src/workbench/settings";
import { startWorkbench } from "../src/workbench/server";
import { createWorkbenchConfigGuard, WORKBENCH_CONFIG_CHANGED_MESSAGE } from "../src/workbench/launch";

const cleanup: Array<() => Promise<unknown>> = [];
afterEach(async () => { for (const fn of cleanup.splice(0).reverse()) await fn(); });

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "arxiv-ui-config-guard-"));
  cleanup.push(() => rm(root, { recursive: true, force: true }));
  const env = { XDG_CONFIG_HOME: root };
  const configPath = join(root, "arxiv-daily/config.toml");
  const vaultRoot = join(root, "vault");
  await mkdir(vaultRoot, { recursive: true });
  const initial = await readWorkbenchSettings(configPath);
  const config = await saveWorkbenchSettings(configPath, { revision: null, values: { ...initial.values, vaultRoot } });
  return { root, env, configPath, config };
}

it("states which actions a stale config revision blocks, and that reading still works, instead of implying a full lockout", async () => {
  const { env, configPath, config } = await fixture();
  // Simulate the workbench having loaded `config` a while ago, then someone editing the CLI TOML
  // directly on disk (not through the workbench's own settings save) while the workbench stays open.
  await saveWorkbenchSettings(configPath, { revision: config.configRevision, values: { ...(await readWorkbenchSettings(configPath)).values, model: "externally-edited-model" } });

  const guard = createWorkbenchConfigGuard(() => config, env);
  await expect(guard()).rejects.toMatchObject({ status: 409, message: WORKBENCH_CONFIG_CHANGED_MESSAGE });

  // The exact copy must not claim a restart is the only way back, and must not claim every action is
  // blocked -- it must say which operations are affected (generation and marks) and that reading still
  // works, since that is what a reopen-and-save of Settings (without restarting the workbench) also fixes.
  expect(WORKBENCH_CONFIG_CHANGED_MESSAGE).not.toBe("配置已改变，请重启工作台后再操作。");
  expect(WORKBENCH_CONFIG_CHANGED_MESSAGE).toMatch(/生成|标记/);
  expect(WORKBENCH_CONFIG_CHANGED_MESSAGE).toMatch(/浏览|阅读/);
  expect(WORKBENCH_CONFIG_CHANGED_MESSAGE).toMatch(/设置并保存|重新保存/);

  const app = await startWorkbench({ config, configPath, beforeWrite: guard });
  cleanup.push(app.close);
  const get = (route: string) => fetch(new URL(route, app.url));
  const post = (route: string, body: unknown) => fetch(new URL(route, app.url), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

  // Marking a paper is one of the two write paths this guard actually blocks -- the HTTP response the
  // UI renders (in the connection banner) must carry the new, accurate copy.
  const mark = await post("api/paper/mark", { key: "arxiv:stale.00001", action: "star", value: true, expected: "normal" });
  expect(mark.status).toBe(409);
  expect((await mark.json()).error).toBe(WORKBENCH_CONFIG_CHANGED_MESSAGE);

  // Reading/browsing is not gated by this guard at all and must keep working while the banner is shown.
  expect((await get("api/papers")).status).toBe(200);
  expect((await get("api/status")).status).toBe(200);
});
