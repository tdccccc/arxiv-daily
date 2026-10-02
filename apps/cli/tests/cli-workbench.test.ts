import { expect, it, vi } from "vitest";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import { DEFAULT_CLI_SCHEDULE, type CliRuntimeConfig } from "../src/config";
import { runCli } from "../src/main";

function fixture() {
  const config: CliRuntimeConfig = { settings: structuredClone(DEFAULT_SETTINGS), vaultRoot: "/vault", cacheDir: "/cache", configPath: "/config/config.toml", linkStyle: "wikilink", scheduleIntent: { ...DEFAULT_CLI_SCHEDULE } };
  const text: string[] = [];
  const io = { stdout: { write: (value: string) => text.push(value) }, stderr: { write: (value: string) => text.push(value) } };
  return { config, io, text };
}

it("opens the workbench through the CLI without requiring model readiness or building a generation runtime", async () => {
  const { config, io } = fixture();
  const ui = vi.fn(async () => 0);
  const runtime = vi.fn();
  expect(await runCli({ argv: ["ui"], loadConfig: async () => config, io, ui, buildRuntime: runtime, env: { XDG_CONFIG_HOME: "/trial" } })).toBe(0);
  expect(ui).toHaveBeenCalledWith(config, expect.anything(), { port: 0, open: true, env: { XDG_CONFIG_HOME: "/trial" } });
  expect(runtime).not.toHaveBeenCalled();
});

it("accepts an explicit port and no-open for background / remote terminals", async () => {
  const { config, io } = fixture();
  const ui = vi.fn(async () => 0);
  expect(await runCli({ argv: ["ui", "--no-open", "--port", "8123"], loadConfig: async () => config, io, ui })).toBe(0);
  expect(ui).toHaveBeenCalledWith(config, expect.anything(), expect.objectContaining({ port: 8123, open: false }));
});

it("rejects invalid launch options before loading configuration", async () => {
  const { config, io, text } = fixture();
  const loadConfig = vi.fn(async () => config);
  for (const argv of [["ui", "--port", "-1"], ["ui", "--port", "70000"], ["ui", "--port"], ["ui", "--host", "0.0.0.0"], ["ui", "--unknown"], ["ui", "--port", "1.2"]]) {
    expect(await runCli({ argv, loadConfig, io })).toBe(2);
  }
  expect(loadConfig).not.toHaveBeenCalled();
  expect(text.join("")).toContain("ui");
});

it("waits for the workbench lifetime and redacts launch errors", async () => {
  const { config, io, text } = fixture();
  config.settings.llm.apiKey = "secret-in-launch-error";
  let stop!: () => void;
  const ui = vi.fn(() => new Promise<number>(resolve => { stop = () => resolve(0); }));
  const result = runCli({ argv: ["ui"], loadConfig: async () => config, io, ui });
  await vi.waitFor(() => expect(ui).toHaveBeenCalledOnce());
  stop();
  expect(await result).toBe(0);
  expect(await runCli({ argv: ["ui"], loadConfig: async () => config, io, ui: async () => { throw new Error("secret-in-launch-error could not listen"); } })).toBe(1);
  expect(text.join("")).toContain("could not listen");
  expect(text.join("")).not.toContain("secret-in-launch-error");
});

it("accepts one exact local embedding origin and rejects broad or malformed ancestors before config", async () => {
  const { config, io } = fixture(); const ui = vi.fn(async () => 0), loadConfig = vi.fn(async () => config);
  expect(await runCli({ argv: ["ui", "--no-open", "--frame-origin", "http://127.0.0.1:3080"], loadConfig, io, ui })).toBe(0);
  expect(ui).toHaveBeenCalledWith(config, expect.anything(), expect.objectContaining({ frameOrigin: "http://127.0.0.1:3080" }));
  loadConfig.mockClear();
  for (const value of ["*", "https://example.com", "http://localhost:3080/path", "http://localhost:3080?x=1", "http://user@localhost:3080", "http://localhost:3080#x", "http://127.0.0.2:3080"]) {
    expect(await runCli({ argv: ["ui", "--frame-origin", value], loadConfig, io, ui })).toBe(2);
  }
  expect(loadConfig).not.toHaveBeenCalled();
});

it("allows only the fixed DSH desktop application origin among custom schemes", async () => {
  const { config, io } = fixture(); const ui = vi.fn(async () => 0), loadConfig = vi.fn(async () => config);
  expect(await runCli({ argv: ["ui", "--frame-origin", "dsh-app://app"], loadConfig, io, ui })).toBe(0);
  expect(ui).toHaveBeenCalledWith(config, expect.anything(), expect.objectContaining({ frameOrigin: "dsh-app://app" }));
  loadConfig.mockClear();
  for (const origin of ["dsh-app://evil", "dsh-app://app/path", "file://", "null", "dsh-app:"]) expect(await runCli({ argv: ["ui", "--frame-origin", origin], loadConfig, io, ui })).toBe(2);
  expect(loadConfig).not.toHaveBeenCalled();
});
