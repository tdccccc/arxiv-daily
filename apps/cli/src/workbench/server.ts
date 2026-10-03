import { isWeekendReportDate, WEEKEND_REPORT_MESSAGE } from "./announcement-calendar";
import { randomBytes } from "node:crypto";
import { createReadStream } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { pipeline } from "node:stream/promises";
import { formatDate, modernArxivResources, redactText, todayInTz } from "@arxiv-daily/core";
import { loadCliConfig, type CliRuntimeConfig } from "../config";
import { inspectWorkbenchLibrary, performSettingsAction } from "./settings-actions";
import type { CliIo } from "../main-types";
import { inspectProduct } from "../inspect-cmd";
import { WorkbenchDocuments, WorkbenchError } from "./documents";
import { inspectCalendar } from "./calendar";
import { validateFrameOrigin } from "./embedding";
import { WorkbenchPapers } from "./papers";
import { readWorkbenchSettings, saveWorkbenchSettings } from "./settings";
import { WorkbenchSchedule } from "./schedule";
import { readPreferences, savePreferences } from "./preferences";

export interface WorkbenchAsset { type: string; body: string; encoding?: "base64" }
export interface WorkbenchOptions {
  config?: CliRuntimeConfig;
  configPath?: string;
  onConfigSaved?: (config: CliRuntimeConfig) => void;
  port?: number;
  assets?: Record<string, WorkbenchAsset>;
  run?: (args: string[], io: CliIo, signal: AbortSignal) => Promise<number>;
  now?: () => Date;
  beforeWrite?: () => Promise<void>;
  frameOrigin?: string;
}

export interface WorkbenchRun {
  id: string;
  label: string;
  /** Configured-timezone date for daily jobs; paper-note jobs have no calendar date. */
  date: string | null;
  status: "running" | "completed" | "failed" | "cancelled" | "skipped";
  output: string;
  exitCode: number | null;
  startedAt: string;
  finishedAt: string | null;
}

/** Ephemeral reading server. Durable state stays owned by the existing product. */
export async function startWorkbench(options: WorkbenchOptions) {
  let config = options.config;
  const configPath = options.configPath ?? config?.configPath;
  if (!configPath) throw new Error("Missing configuration path");
  const frameAncestor = options.frameOrigin === undefined ? "'none'" : validateFrameOrigin(options.frameOrigin);
  const now = options.now ?? (() => new Date());
  if (!Number.isInteger(options.port ?? 0) || (options.port ?? 0) < 0 || (options.port ?? 0) > 65535) throw new Error("Port must be 0..65535");
  let documents = config ? new WorkbenchDocuments(config) : undefined;
  let papers = config && documents ? new WorkbenchPapers(config, documents) : undefined;
  const prefix = `/${randomBytes(24).toString("hex")}/`;
  const secrets: string[] = [];
  const rememberSecrets = (value: CliRuntimeConfig) => { secrets.push(...[value.settings.llm.apiKey, value.settings.embedding.apiKey, value.settings.email.apiKey, value.settings.email.hostedToken].filter((key): key is string => Boolean(key))); };
  if (config) rememberSecrets(config);
  let saving = false;
  const redact = (text: string) => redactText(text, { secrets });
  let origin = "";
  let run: WorkbenchRun | null = null;
  let controller: AbortController | null = null;
  let running: Promise<void> | null = null;
  const json = (res: ServerResponse, code: number, value: unknown) => {
    res.writeHead(code, { "Content-Type": "application/json; charset=utf-8" });
    res.end(redact(JSON.stringify(value)));
  };

  const autoSchedule = new WorkbenchSchedule(() => saving || run?.status === "running" || !options.run, () => {
    beginRun("Automatic daily report check", null, (io, signal) => options.run!(["run", "--scheduled"], io, signal));
  });
  function activate(next: CliRuntimeConfig) {
    config = next;
    documents = new WorkbenchDocuments(next);
    papers = new WorkbenchPapers(next, documents);
    rememberSecrets(next);
    options.onConfigSaved?.(next);
    autoSchedule.update(next.workbenchSchedule);
  }
  function beginRun(label: string, date: string | null, execute: (io: CliIo, signal: AbortSignal) => Promise<number>) {
    if (saving || run?.status === "running") throw new WorkbenchError(409, "已有操作正在运行，请等待完成。");
    controller = new AbortController();
    const signal = controller.signal;
    const current: WorkbenchRun = { id: randomBytes(12).toString("hex"), label: label, date: date, status: "running", output: "", exitCode: null, startedAt: now().toISOString(), finishedAt: null };
    run = current;
    const write = (chunk: string) => { current.output = (current.output + redact(String(chunk))).slice(-24000); };
    running = Promise.resolve().then(() => execute({ stdout: { write }, stderr: { write } }, signal)).then(code => {
      current.exitCode = code;
      current.status = signal.aborted ? "cancelled" : code === 0 ? "completed" : "failed";
    }).catch(error => {
      write(`\n${error instanceof Error ? error.message : "任务执行失败"}\n`);
      current.exitCode = 1;
      current.status = signal.aborted ? "cancelled" : "failed";
    }).finally(() => { current.finishedAt = now().toISOString(); });
    return current;
  }

  async function handle(req: IncomingMessage, res: ServerResponse) {
    res.setHeader("Cache-Control", "no-store");
    res.setHeader("Referrer-Policy", "no-referrer");
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("Content-Security-Policy", `default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' https: http: data:; font-src 'self'; connect-src 'self'; frame-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors ${frameAncestor}`);
    if (req.headers.host !== new URL(origin).host || (req.headers.origin && req.headers.origin !== origin)) throw new WorkbenchError(403, "此工作台只接受本机页面的请求。");
    const url = new URL(req.url || "/", origin);
    if (!url.pathname.startsWith(prefix)) throw new WorkbenchError(404, "请使用启动时显示的工作台链接。");
    const route = url.pathname.slice(prefix.length);
    const method = req.method || "GET";
    const landing = method === "GET" && (route === "" || route === "index.html");
    if (req.headers["sec-fetch-site"] === "cross-site" && !landing) throw new WorkbenchError(403, "此工作台只接受本机页面的请求。");
    // Static assets and harmless preferences remain available before first-run setup.
    if (method === "GET") {
      const asset = Object.hasOwn(options.assets || {}, route || "index.html") ? options.assets?.[route || "index.html"] : undefined;
      if (asset) {
        res.writeHead(200, { "Content-Type": asset.type });
        return res.end(asset.encoding === "base64" ? Buffer.from(asset.body, "base64") : asset.body);
      }
    }
    if (method === "GET" && route === "api/settings") return json(res, 200, await readWorkbenchSettings(configPath!));
    if (method === "POST" && route === "api/settings") {
      const body = await readJson(req);
      if (saving || run?.status === "running") throw new WorkbenchError(409, "请等待当前操作完成后再保存设置。");
      saving = true;
      try {
        const next = await saveWorkbenchSettings(configPath!, body);
        activate(next);
        return json(res, 200, await readWorkbenchSettings(configPath!));
      } finally { saving = false; }
    }
    if (method === "GET" && route === "api/preferences") return json(res, 200, await readPreferences(configPath!));
    if (method === "POST" && route === "api/preferences") return json(res, 200, await savePreferences(configPath!, await readJson(req)));
    if (method === "GET" && route === "api/runs/current") return json(res, 200, { run });
    if (!config || !documents || !papers) {
      if (method === "GET" && route === "api/status") return json(res, 200, { setupRequired: true });
      return json(res, 409, { setupRequired: true, error: "请先完成设置，再使用文献工作台。" });
    }
    if (saving && method === "POST") throw new WorkbenchError(409, "正在保存设置，请稍后重试。");
    if (method === "GET" && route === "api/settings/library") return json(res, 200, { ...inspectWorkbenchLibrary(config), ...(run?.label === "Build index" ? { run } : {}) });
    if (method === "POST" && route === "api/settings/secret") {
      const body = await readJson(req);
      if (typeof body.field !== "string" || Object.keys(body).some(key => !["revision", "field"].includes(key)) || !["apiKey", "embedding.apiKey", "email.apiKey", "email.hostedToken"].includes(String(body.field))) throw new WorkbenchError(400, "请选择有效的密钥字段。");
      const current = await loadCliConfig({ configPath: configPath! });
      if (body.revision !== current.configRevision) throw new WorkbenchError(409, "配置已改变，请重新打开设置后查看。");
      const values: Record<string, string | undefined> = { apiKey: current.settings.llm.apiKey, "embedding.apiKey": current.settings.embedding.apiKey, "email.apiKey": current.settings.email.apiKey, "email.hostedToken": current.settings.email.hostedToken };
      // Deliberate, user-initiated reveal only. Normal JSON projections stay redacted.
      res.writeHead(200, { "Content-Type": "application/json; charset=utf-8" });
      return res.end(JSON.stringify({ value: values[String(body.field)] ?? "" }));
    }
    if (method === "POST" && route === "api/settings/action") {
      const { revision, ...body } = await readJson(req);
      if (saving || run?.status === "running") throw new WorkbenchError(409, "请等待当前操作完成。");
      if (!["models", "library-connect", "library-revoke", "library-build", "email-test", "email-verify"].includes(String(body.action))) throw new WorkbenchError(400, "未知设置操作。");
      saving = true;
      let current: CliRuntimeConfig;
      try {
        current = await loadCliConfig({ configPath: configPath! });
        if (revision !== current.configRevision || config.configRevision !== current.configRevision) throw new WorkbenchError(409, "配置已改变，请重新打开设置并保存。");
        if (body.action === "library-build") {
          const library = inspectWorkbenchLibrary(current);
          if (library.status.kind !== "authorized" && (!library.disclosure || body.fingerprint !== library.disclosure.authorizationFingerprint)) throw new WorkbenchError(409, "请先审核并确认文献库的当前处理授权。");
        }
        if (["library-build", "email-test", "email-verify"].includes(String(body.action))) {
          saving = false;
          const labels: Record<string, string> = { "library-build": "Build index", "email-test": "Send test", "email-verify": "Send verification" };
          const started = beginRun(labels[String(body.action)]!, null, async (io, signal) => {
            let completed = false;
            try {
              const result = await performSettingsAction(current, body, io, signal);
              if (result.message) io.stdout.write(result.message + "\n");
              completed = true;
              return 0;
            } finally {
              // Authorization may have committed even when processing fails.
              try { activate(await loadCliConfig({ configPath: configPath! })); }
              catch {
                io.stderr.write("配置重新读取失败，请重新打开工作台。\n");
                if (completed) throw new WorkbenchError(409, "配置重新读取失败，请重新打开工作台。");
              }
            }
          });
          return json(res, 202, { run: started });
        }
        controller = new AbortController();
        const result = await performSettingsAction(current, body, { stdout: { write: () => {} }, stderr: { write: () => {} } }, controller.signal);
        if (result.config) activate(result.config);
        return json(res, 200, { ...(result.models ? { models: result.models } : {}), ...(result.library ? { library: result.library } : {}), settings: await readWorkbenchSettings(configPath!), ...(result.message ? { message: result.message } : {}) });
      } catch (error) {
        if (error instanceof WorkbenchError) throw error;
        throw new WorkbenchError(400, redact(error instanceof Error ? error.message : "设置操作失败。"));
      } finally { saving = false; }
    }
    if (method === "GET" && route === "api/status") return json(res, 200, await inspectProduct(config));
    if (method === "GET" && route === "api/calendar") return json(res, 200, await inspectCalendar(config, documents, url.searchParams.get("month"), now(), run));
    if (method === "GET" && route === "api/papers") return json(res, 200, await papers.list(url.searchParams, now(), run));
    if (method === "GET" && route === "api/paper") return json(res, 200, { paper: await papers.paper(url.searchParams.get("key") || "") });
    if (method === "POST" && route === "api/paper/mark") return json(res, 200, { paper: await papers.mark(await readJson(req), options.beforeWrite) });
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
      const task = parseTask(body, formatDate(todayInTz(now(), config.settings.arxiv.timezone)));
      if (run?.status === "running") throw new WorkbenchError(409, "已有任务正在运行，请等待完成或先取消。");
      if (saving) throw new WorkbenchError(409, "正在保存设置，请稍后重试。");
      if (task.date && isWeekendReportDate(task.date)) {
        run = { id: randomBytes(12).toString("hex"), label: task.label, date: task.date, status: "skipped", output: WEEKEND_REPORT_MESSAGE, exitCode: 0, startedAt: now().toISOString(), finishedAt: now().toISOString() };
        return json(res, 202, { run });
      }
      if (!options.run) throw new WorkbenchError(503, "当前工作台未提供生成操作。");
      const current = beginRun(task.label, task.date, (io, signal) => options.run!(task.args, io, signal));
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
  autoSchedule.update(config?.workbenchSchedule);
  return {
    url: `${origin}${prefix}`,
    close: async () => {
      autoSchedule.close();
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
    if (size > 65536) throw new WorkbenchError(413, "请求过大。");
    chunks.push(Buffer.from(chunk));
  }
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Expected object");
    return value as Record<string, unknown>;
  } catch { throw new WorkbenchError(400, "请求内容无效。"); }
}
function parseTask(body: Record<string, unknown>, today: string): { args: string[]; label: string; date: string | null } {
  if (body.kind === "daily") {
    if (body.date === undefined) return { args: ["run", "--today"], label: "生成今日日报", date: today };
    if (typeof body.date === "string" && /^\d{4}-\d{2}-\d{2}$/.test(body.date)) {
      const parsed = new Date(body.date);
      if (Number.isFinite(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === body.date) return { args: ["run", "--date", body.date], label: `${body.date} 日报`, date: body.date };
    }
  }
  if (body.kind === "paper" && typeof body.id === "string") {
    const resources = modernArxivResources(body.id);
    if (resources) return { args: ["run", "--id", resources.id], label: `${resources.id} 详细总结`, date: null };
  }
  throw new WorkbenchError(400, "请选择有效日期或输入 arXiv ID。");
}
