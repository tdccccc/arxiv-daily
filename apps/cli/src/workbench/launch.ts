import { spawn } from "node:child_process";
import { clearTimer, setTimer } from "@arxiv-daily/core";
import { loadCliConfig, type CliRuntimeConfig } from "../config";
import { resolveCliConfigPath } from "../config-path";
import { isCliRunEvent, type CliIo } from "../main-types";
import { startWorkbench } from "./server";
import { workbenchAssets } from "./assets";
import { WorkbenchError } from "./documents";

export interface RunWorkbenchOptions {
  port: number;
  open: boolean;
  frameOrigin?: string;
  env: Record<string, string | undefined>;
}

export async function runWorkbench(config: CliRuntimeConfig | undefined, io: CliIo, options: RunWorkbenchOptions): Promise<number> {
  const executable = process.argv[1];
  if (!executable) throw new Error("The workbench must be launched from the built CLI");
  const beforeWrite = async () => {
    const current = await loadCliConfig({ env: options.env });
    if (!config || current.configPath !== config.configPath || current.configRevision !== config.configRevision) throw new WorkbenchError(409, "配置已改变，请重启工作台后再操作。");
  };
  const app = await startWorkbench({
    config, configPath: config?.configPath ?? resolveCliConfigPath(options.env), onConfigSaved: next => { config = next; }, port: options.port, assets: workbenchAssets(), beforeWrite, frameOrigin: options.frameOrigin,
    run: async (args, output, signal) => {
      await beforeWrite();
      if (signal.aborted) return 1;
      return new Promise<number>((resolve, reject) => {
        // Dispatch the same executable; do not reconstruct scheduler / manual-fetch behavior here.
        const child = spawn(process.execPath, [executable, ...args], { env: options.env, stdio: ["ignore", "pipe", "pipe", "ipc"], windowsHide: true });
        child.on("message", (message: unknown) => {
          if (!message || typeof message !== "object") return;
          const envelope = message as { type?: unknown; event?: unknown };
          if (envelope.type === "arxiv-daily/run-result" && isCliRunEvent(envelope.event)) output.onRunResult?.(envelope.event);
        });
        child.stdout!.setEncoding("utf8").on("data", (chunk: string) => output.stdout.write(chunk));
        child.stderr!.setEncoding("utf8").on("data", (chunk: string) => output.stderr.write(chunk));
        const cancel = () => { child.kill("SIGINT"); };
        signal.addEventListener("abort", cancel, { once: true });
        child.once("spawn", () => { if (signal.aborted) cancel(); });
        child.once("error", error => { signal.removeEventListener("abort", cancel); reject(error); });
        child.once("close", code => { signal.removeEventListener("abort", cancel); resolve(code ?? 1); });
      });
    },
  });
  let stopping = false;
  const stopped = new Promise<number>(resolve => {
    const stop = () => {
      if (stopping) return;
      stopping = true;
      io.stdout.write("Stopping workbench…\n");
      void app.close().then(() => resolve(0), error => { io.stderr.write(`${String(error)}\n`); resolve(1); }).finally(() => {
        process.off("SIGINT", stop);
        process.off("SIGTERM", stop);
      });
    };
    process.on("SIGINT", stop);
    process.on("SIGTERM", stop);
  });
  io.stdout.write(`Workbench: ${app.url}\nKeep this process running. Press Ctrl+C to stop.\n`);
  if (options.open && !await openBrowser(app.url)) io.stderr.write("Could not open a browser automatically. Open the Workbench URL above.\n");
  return stopped;
}

async function openBrowser(url: string): Promise<boolean> {
  const command = process.platform === "darwin" ? "open" : process.platform === "win32" ? "rundll32.exe" : "xdg-open";
  const args = process.platform === "win32" ? ["url.dll,FileProtocolHandler", url] : [url];
  return new Promise(resolve => {
    const child = spawn(command, args, { stdio: "ignore", detached: true, windowsHide: true });
    const timer = setTimer(() => { child.unref(); resolve(true); }, 4000);
    child.once("error", () => { clearTimer(timer); resolve(false); });
    child.once("exit", code => { clearTimer(timer); resolve(code === 0); });
  });
}
