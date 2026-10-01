import { randomBytes } from "node:crypto";
import { createReadStream } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { pipeline } from "node:stream/promises";
import { modernArxivResources, redactText } from "@arxiv-daily/core";
import type { CliRuntimeConfig } from "../config";
import type { CliIo } from "../main-types";
import { inspectProduct } from "../inspect-cmd";
import { WorkbenchDocuments, WorkbenchError } from "./documents";

export interface WorkbenchAsset { type: string; body: string; encoding?: "base64" }
export interface WorkbenchOptions {
  config: CliRuntimeConfig;
  port?: number;
  assets?: Record<string, WorkbenchAsset>;
  run?: (args: string[], io: CliIo, signal: AbortSignal) => Promise<number>;
}

export interface WorkbenchRun {
  id: string;
  label: string;
  status: "running" | "completed" | "failed" | "cancelled";
  output: string;
  exitCode: number | null;
  startedAt: string;
  finishedAt: string | null;
}

/** Ephemeral reading server. Durable state stays owned by the existing product. */
export async function startWorkbench(options: WorkbenchOptions) {
  const { config } = options;
  if (!Number.isInteger(options.port ?? 0) || (options.port ?? 0) < 0 || (options.port ?? 0) > 65535) throw new Error("Port must be 0..65535");
  const documents = new WorkbenchDocuments(config);
  const prefix = `/${randomBytes(24).toString("hex")}/`;
  const secrets = [config.settings.llm.apiKey, config.settings.embedding.apiKey, config.settings.email.apiKey, config.settings.email.hostedToken].filter((value): value is string => Boolean(value));
  const redact = (text: string) => redactText(text, { secrets });
  let origin = "";
  let run: WorkbenchRun | null = null;
  let controller: AbortController | null = null;
  let running: Promise<void> | null = null;
  const json = (res: ServerResponse, code: number, value: unknown) => {
    res.writeHead(code, { "Content-Type": "application/json; charset=utf-8" });
    res.end(redact(JSON.stringify(value)));
  };

  async function handle(req: IncomingMessage, res: ServerResponse) {
    res.setHeader("Cache-Control", "no-store");
    res.setHeader("Referrer-Policy", "no-referrer");
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' https: http: data:; font-src 'self'; connect-src 'self'; frame-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'");
    if (req.headers.host !== new URL(origin).host || (req.headers.origin && req.headers.origin !== origin) || req.headers["sec-fetch-site"] === "cross-site") throw new WorkbenchError(403, "此工作台只接受本机页面的请求。");
    const url = new URL(req.url || "/", origin);
    if (!url.pathname.startsWith(prefix)) throw new WorkbenchError(404, "请使用启动时显示的工作台链接。");
    const route = url.pathname.slice(prefix.length);
    const method = req.method || "GET";
    if (method === "GET" && route === "api/status") return json(res, 200, await inspectProduct(config));
    if (method === "GET" && route === "api/documents") {
      const kind = url.searchParams.get("kind") || "all";
      const offset = Number(url.searchParams.get("offset") ?? 0);
      const limit = Number(url.searchParams.get("limit") ?? 60);
      if (!["all", "daily", "papers"].includes(kind) || !Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(limit) || limit < 1 || limit > 100) throw new WorkbenchError(400, "列表参数无效。");
      const query = (url.searchParams.get("q") || "").trim().toLocaleLowerCase();
      const all = await documents.list();
      const found = all.filter(entry => (kind === "all" || entry.kind === kind) && (!query || `${entry.title} ${entry.authors} ${entry.arxivId} ${entry.date} ${entry.path}`.toLocaleLowerCase().includes(query)));
      return json(res, 200, { documents: found.slice(offset, offset + limit), total: found.length, offset, nextOffset: offset + limit < found.length ? offset + limit : null, counts: { daily: all.filter(entry => entry.kind === "daily").length, papers: all.filter(entry => entry.kind === "papers").length } });
    }
    if (method === "GET" && route === "api/document") return json(res, 200, await documents.document(url.searchParams.get("path") || ""));
    if (method === "GET" && route === "api/raw") {
      const source = await documents.raw(url.searchParams.get("path") || "");
      res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8" });
      return res.end(source);
    }
    if (method === "GET" && route === "api/asset") {
      const asset = await documents.asset(url.searchParams.get("path") || "");
      res.setHeader("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; sandbox");
      let start = 0, end = asset.size - 1;
      if (req.headers.range) {
        const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
        if (!range || (!range[1] && !range[2])) throw new WorkbenchError(416, "附件范围无效。");
        start = range[1] ? Number(range[1]) : Math.max(0, asset.size - Number(range[2]));
        end = range[1] && range[2] ? Math.min(Number(range[2]), end) : end;
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || start >= asset.size) {
          res.setHeader("Content-Range", `bytes */${asset.size}`);
          throw new WorkbenchError(416, "附件范围无效。");
        }
        res.setHeader("Content-Range", `bytes ${start}-${end}/${asset.size}`);
      }
      res.writeHead(req.headers.range ? 206 : 200, { "Content-Type": asset.type, "Content-Length": Math.max(0, end - start + 1), "Accept-Ranges": "bytes" });
      if (!asset.size) return res.end();
      await pipeline(createReadStream(asset.file, { start, end }), res);
      return;
    }
    if (method === "GET" && route === "api/runs/current") return json(res, 200, { run });
    if (method === "POST" && route === "api/runs") {
      const body = await readJson(req);
      const task = parseTask(body);
      if (run?.status === "running") throw new WorkbenchError(409, "已有任务正在运行，请等待完成或先取消。");
      if (!options.run) throw new WorkbenchError(503, "当前工作台未提供生成操作。");
      controller = new AbortController();
      const signal = controller.signal;
      const current: WorkbenchRun = { id: randomBytes(12).toString("hex"), label: task.label, status: "running", output: "", exitCode: null, startedAt: new Date().toISOString(), finishedAt: null };
      run = current;
      const write = (chunk: string) => { current.output = (current.output + redact(String(chunk))).slice(-24000); };
      running = Promise.resolve().then(() => options.run!(task.args, { stdout: { write }, stderr: { write } }, signal)).then(code => {
        current.exitCode = code;
        current.status = signal.aborted ? "cancelled" : code === 0 ? "completed" : "failed";
      }).catch(error => {
        write(`\n${error instanceof Error ? error.message : "任务执行失败"}\n`);
        current.exitCode = 1;
        current.status = signal.aborted ? "cancelled" : "failed";
      }).finally(() => { current.finishedAt = new Date().toISOString(); });
      return json(res, 202, { run: current });
    }
    if (method === "POST" && route === "api/runs/cancel") {
      const body = await readJson(req);
      if (!run || run.id !== body.id || run.status !== "running") throw new WorkbenchError(409, "该任务已结束或已改变，请刷新状态。");
      controller?.abort();
      return json(res, 202, { run });
    }
    if (method === "GET") {
      const asset = Object.hasOwn(options.assets || {}, route || "index.html") ? options.assets?.[route || "index.html"] : undefined;
      if (asset) {
        res.writeHead(200, { "Content-Type": asset.type });
        return res.end(asset.encoding === "base64" ? Buffer.from(asset.body, "base64") : asset.body);
      }
    }
    throw new WorkbenchError(404, "找不到页面或操作。");
  }
  const server = createServer((req, res) => { void handle(req, res).catch(error => {
    if (res.headersSent) { res.destroy(); return; }
    json(res, error instanceof WorkbenchError ? error.status : 500, { error: error instanceof WorkbenchError ? error.message : "读取失败，请检查文件权限或稍后重试。" });
  }); });
  server.requestTimeout = 30000;
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(options.port ?? 0, "127.0.0.1", () => { server.off("error", reject); resolve(); });
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Missing address");
  origin = `http://127.0.0.1:${address.port}`;
  return {
    url: `${origin}${prefix}`,
    close: async () => {
      controller?.abort();
      await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
      await running;
    },
  };
}

async function readJson(req: IncomingMessage): Promise<Record<string, unknown>> {
  if (req.headers["content-type"]?.split(";")[0]?.trim() !== "application/json") throw new WorkbenchError(415, "请发送 JSON 请求。");
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 8192) throw new WorkbenchError(413, "请求过大。");
    chunks.push(Buffer.from(chunk));
  }
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Expected object");
    return value as Record<string, unknown>;
  } catch { throw new WorkbenchError(400, "请求内容无效。"); }
}
function parseTask(body: Record<string, unknown>): { args: string[]; label: string } {
  if (body.kind === "daily") {
    if (body.date === undefined) return { args: ["run", "--today"], label: "生成今日日报" };
    if (typeof body.date === "string" && /^\d{4}-\d{2}-\d{2}$/.test(body.date)) {
      const parsed = new Date(body.date);
      if (Number.isFinite(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === body.date) return { args: ["run", "--date", body.date], label: `${body.date} 日报` };
    }
  }
  if (body.kind === "paper" && typeof body.id === "string") {
    const resources = modernArxivResources(body.id);
    if (resources) return { args: ["run", "--id", resources.id], label: `${resources.id} 详细总结` };
  }
  throw new WorkbenchError(400, "请选择有效日期或输入 arXiv ID。");
}
