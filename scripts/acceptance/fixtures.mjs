import http from "node:http";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

export const ACCEPTANCE_DATE = "2026-10-01";
export const PAPERS = Object.freeze([
  { id: "2610.10001", title: "Reliable measurement with controlled experiments", abstract: "Controlled experiments compare repeatable research measurements." },
  { id: "2610.10002", title: "Reproducible evidence for research workflows", abstract: "A bounded reproducible experiment measures research reliability." },
]);

const DATES = ["2026-10-09", "2026-10-08", "2026-10-07", "2026-10-06", "2026-10-05", "2026-10-02", ACCEPTANCE_DATE];
const xml = value => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");

function recentHtml(empty) {
  return `<html><body><dl id="articles">${DATES.map(date => {
    const label = new Intl.DateTimeFormat("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }).format(new Date(`${date}T12:00:00Z`));
    return `<h3>${label} (showing ${empty ? 0 : PAPERS.length} of ${empty ? 0 : PAPERS.length} entries)</h3>${empty ? "" : PAPERS.map(p => `<dt><a title="Abstract" href="/abs/${p.id}">arXiv:${p.id}</a></dt><dd><div class="list-title">Title: ${xml(p.title)}</div><div class="list-authors"><a>A. Researcher</a></div></dd>`).join("")}`;
  }).join("")}</dl></body></html>`;
}

function atom(ids) {
  return `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">${ids.map(id => {
    const p = PAPERS.find(paper => paper.id === id.replace(/v\d+$/, "")) ?? { id, title: `Research paper ${id}`, abstract: "A controlled research measurement and its reproducible evidence." };
    return `<entry><id>https://arxiv.org/abs/${p.id}v1</id><title>${xml(p.title)}</title><author><name>A. Researcher</name></author><summary>${xml(p.abstract)}</summary><published>${ACCEPTANCE_DATE}T00:00:00Z</published><updated>${ACCEPTANCE_DATE}T00:00:00Z</updated><arxiv:primary_category term="cs.AI"/><category term="cs.AI"/></entry>`;
  }).join("")}</feed>`;
}

function paperHtml(id) {
  const p = PAPERS.find(paper => paper.id === id.replace(/v\d+$/, "")) ?? PAPERS[0];
  return `<html><body><h1>${xml(p.title)}</h1><div class="ltx_abstract">${xml(p.abstract)}</div><h2>Introduction</h2><p>Reliable research needs controlled and repeatable measurements. We study a fixed collection of experimental tasks.</p><h2>Methods</h2><p>The procedure compares a controlled intervention with a repeated baseline over one hundred observations.</p><h2>Results</h2><p>We measure a ten percent improvement on the fixed sample. Table 1 reports the controlled comparison.</p><h2>Conclusions</h2><p>The method improves repeatability within this experiment. Generalization beyond the sample remains untested.</p></body></html>`;
}

function paperIds(text) {
  return [...new Set([...text.matchAll(/(?:ID:|arXiv(?: ID)?[:：]?|\/abs\/|"id"\s*:\s*")\s*(\d{4}\.\d{4,5})/g)].map(match => match[1]))];
}

function completion(payload) {
  const messages = Array.isArray(payload.messages) ? payload.messages : [];
  const system = typeof payload.system === "string" ? payload.system : messages.filter(m => m.role === "system").map(m => m.content).join("\n");
  const user = messages.filter(m => m.role === "user").map(m => typeof m.content === "string" ? m.content : (m.content ?? []).map(block => block.text ?? "").join("\n")).join("\n");
  const ids = paperIds(user);
  const evidence = { model: payload.model, prompt: system, paperIds: ids };
  if (system.includes("为所有命中论文评分") || system.includes("relevanceScore")) {
    const match = /^\s*-\s+([^\s:]+#\d+):/m.exec(system);
    if (!match) throw new Error("Filter fixture requires an actual direction reference");
    const ref = match[1], tag = ref.slice(0, ref.lastIndexOf("#"));
    return { ...evidence, kind: "filter", content: JSON.stringify({ papers: ids.map((id, i) => ({ id, category: tag, directions: [ref], relevanceScore: 95 - i * 10 })) }) };
  }
  if (system.includes("separate deep-dive note")) {
    return { ...evidence, kind: "detail-selection", content: JSON.stringify({ papers: ids.map(id => ({ id, score: 30, reason: "Controlled fixture does not automatically request a separate paper note." })) }) };
  }
  if (system.includes("严格 JSON 对象") || system.includes("strict JSON object")) {
    if (ids.length !== 1) throw new Error("Summary fixture requires exactly one paper ID");
    return { ...evidence, kind: "summary", content: JSON.stringify({ id: ids[0], coreProblem: "Reliable and repeatable research measurement", keyMethod: "A controlled intervention compared with a repeated baseline", mainResult: "A ten percent improvement on the fixed sample", whyRelevant: "Evidence for the configured research direction", limitations: "Only the fixed experimental sample was evaluated" }) };
  }
  if (system.includes("You organize a personal literature library")) {
    const raw = /<paper_data>\s*([\s\S]*?)\s*<\/paper_data>/.exec(user)?.[1];
    const data = JSON.parse(raw ?? "{}");
    if (!Array.isArray(data.groups)) throw new Error("Proposal fixture requires supplied evidence groups");
    return { ...evidence, kind: "proposal", content: JSON.stringify({ topics: data.groups.map((group, i) => ({ suggestedName: `Research evidence ${i + 1}`, directions: [{ text: "Reliable research methods", discoveryCues: ["controlled experiments", "reproducibility"], groupIds: [group.id], representativePaperKeys: group.papers.slice(0, 3).map(p => p.paperKey) }] })) }) };
  }
  if (system.includes("详细的中文论文总结") || system.includes("detailed") && system.includes("paper")) {
    const id = ids[0] ?? PAPERS[0].id, p = PAPERS.find(paper => paper.id === id) ?? PAPERS[0];
    const sections = ["研究问题", "方法设计", "核心结果", "主要结论", "关键图表", "适用边界"];
    return { ...evidence, kind: "detail", content: `# ${p.title}\n\n- **arXiv**: https://arxiv.org/abs/${id}\n\n${sections.map(section => `## ${section}\n\nThis controlled fixture describes the same research measurement, intervention, comparison and bounded result as the supplied paper. The sample contains one hundred observations and supports only the reported experiment. Further evidence is required to establish broader applicability.\n`).join("\n")}` };
  }
  if (user.trim() === "Hello") return { ...evidence, kind: "connection-test", content: "Hello. The local acceptance model is ready." };
  throw new Error("Unexpected model task in acceptance fixture");
}

async function readBody(request) {
  const chunks = []; let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > 2 * 1024 * 1024) throw new Error("Fixture request body exceeds 2 MiB");
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

/** Owned loopback service. There is deliberately no upstream forwarding path. */
export async function startFixtureServer(options = {}) {
  const requests = [], held = new Set();
  let origin, closed;
  const mode = { llm: "normal", arxiv: "normal", ...options.mode };
  const setMode = update => {
    if (update.llm !== undefined && !["normal", "unauthorized", "hold"].includes(update.llm)) throw new TypeError("Unknown LLM fixture mode");
    if (update.arxiv !== undefined && !["normal", "empty", "unavailable"].includes(update.arxiv)) throw new TypeError("Unknown arXiv fixture mode");
    Object.assign(mode, update);
  };
  const releaseHeld = () => { mode.llm = "normal"; for (const release of held) release(); held.clear(); };
  const server = http.createServer(async (request, response) => {
    let record;
    const send = (status, body, type = "application/json") => {
      if (record) record.status = status;
      if (!response.destroyed && !response.writableEnded) { response.writeHead(status, { "content-type": type, "cache-control": "no-store" }); response.end(typeof body === "string" || Buffer.isBuffer(body) ? body : JSON.stringify(body)); }
    };
    try {
      const incoming = new URL(request.url, origin);
      const target = new URL(incoming.pathname === "/proxy" ? incoming.searchParams.get("url") : incoming.href);
      record = { method: request.method, url: target.href, kind: "unexpected", timestamp: new Date().toISOString() };
      requests.push(record);
      const body = await readBody(request);
      const isArxiv = ["arxiv.org", "export.arxiv.org"].includes(target.hostname);
      const isModel = ["fixture-model.test", "model.example", "api.example.com", "model.test"].includes(target.hostname) || target.origin === origin;
      if (isArxiv && mode.arxiv === "unavailable") { record.kind = "arxiv-unavailable"; return send(503, { error: "Controlled arXiv outage" }); }
      if (isArxiv && /^\/list\/[^/]+\/recent$/.test(target.pathname)) { record.kind = "recent"; return send(200, recentHtml(mode.arxiv === "empty"), "text/html"); }
      if (isArxiv && target.pathname === "/api/query") {
        record.kind = "metadata";
        return send(200, atom(mode.arxiv === "empty" ? [] : (target.searchParams.get("id_list")?.split(",") ?? PAPERS.map(p => p.id))), "application/atom+xml");
      }
      if (isArxiv && /^\/html\/\d{4}\.\d{4,5}(?:v\d+)?$/.test(target.pathname)) { record.kind = "content"; return send(200, paperHtml(target.pathname.slice(6)), "text/html"); }
      if (isArxiv && /^\/abs\/\d{4}\.\d{4,5}(?:v\d+)?$/.test(target.pathname)) { record.kind = "abstract"; return send(200, `<blockquote class="abstract">Abstract: ${PAPERS[0].abstract}</blockquote>`, "text/html"); }
      if (isModel && /\/models$/.test(target.pathname) && request.method === "GET") { record.kind = "models"; return send(200, { object: "list", data: ["fixture-model", "fixture-model-2"].map(id => ({ id, object: "model" })) }); }
      if (isModel && /\/(?:chat\/completions|messages)$/.test(target.pathname) && request.method === "POST") {
        const payload = JSON.parse(body);
        record.kind = "model"; record.model = payload.model;
        if (mode.llm === "unauthorized") return send(401, { error: { message: "Controlled acceptance authentication failure", type: "authentication_error" } });
        if (mode.llm === "hold") {
          record.held = true;
          await new Promise(resolve => {
            const release = () => { held.delete(release); response.off("close", cancelled); resolve(); };
            const cancelled = () => { record.cancelled = true; release(); };
            held.add(release); response.once("close", cancelled);
          });
          if (response.destroyed) return;
        }
        const answer = completion(payload); Object.assign(record, answer); delete record.content;
        if (target.pathname.endsWith("/messages")) {
          return send(200, { id: "fixture-message", type: "message", role: "assistant", content: [{ type: "text", text: answer.content }], stop_reason: "end_turn", usage: { input_tokens: 100, output_tokens: 80 } });
        }
        if (payload.stream === false) return send(200, { id: "fixture-completion", choices: [{ index: 0, message: { role: "assistant", content: answer.content }, finish_reason: "stop" }], usage: { prompt_tokens: 100, completion_tokens: 80 } });
        return send(200, `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: answer.content }, finish_reason: "stop" }], usage: { prompt_tokens: 100, completion_tokens: 80 } })}\n\ndata: [DONE]\n\n`, "text/event-stream");
      }
      if (isModel && /\/embeddings$/.test(target.pathname) && request.method === "POST") {
        record.kind = "embeddings"; const payload = JSON.parse(body); record.model = payload.model;
        const input = Array.isArray(payload.input) ? payload.input : [payload.input];
        return send(200, { data: input.map((value, index) => ({ object: "embedding", index, embedding: String(value).includes("measurement") ? [1, 0, 0] : [0.8, 0.6, 0] })), model: payload.model, usage: { prompt_tokens: input.length * 10, total_tokens: input.length * 10 } });
      }
      if (target.hostname === "api.resend.com" && target.pathname === "/emails" && request.method === "POST") { record.kind = "email"; return send(200, { id: "acceptance-message-never-sent" }); }
      send(502, { error: `Unexpected fixture request: ${target.origin}${target.pathname}` });
    } catch (error) {
      if (record) record.kind = "fixture-error";
      send(400, { error: error.message });
    }
  });
  await new Promise((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
  origin = `http://127.0.0.1:${server.address().port}`;
  return { url: origin, requests, setMode, releaseHeld, get mode() { return { ...mode }; }, close() {
    if (!closed) { releaseHeld(); closed = new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }); }
    return closed;
  } };
}

/** Small valid PDF owned by the test; useful for native PDF navigation and scanning. */
function fixturePdf(paper) {
  const objects = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Count 4 /Kids [4 0 R 6 0 R 8 0 R 10 0 R] >>", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"];
  const literal = value => value.replaceAll("\\", "\\\\").replaceAll("(", "\\(").replaceAll(")", "\\)");
  for (let page = 0; page < 4; page++) {
    const contentId = 5 + page * 2;
    objects.push(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents ${contentId} 0 R >>`);
    const lines = [paper.title, "A. Researcher", `arXiv:${paper.id}`, "Abstract", paper.abstract, `Page ${page + 1}: controlled methods, evidence and limitations.`];
    const stream = `BT /F1 11 Tf 40 730 Td ${lines.map((line, i) => `${i ? "0 -24 Td " : ""}(${literal(line)}) Tj`).join("\n")} ET\n`;
    objects.push(`<< /Length ${Buffer.byteLength(stream)} >>\nstream\n${stream}endstream`);
  }
  let result = "%PDF-1.4\n", offsets = [0];
  objects.forEach((object, i) => { offsets.push(Buffer.byteLength(result)); result += `${i + 1} 0 obj\n${object}\nendobj\n`; });
  const start = Buffer.byteLength(result);
  result += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n${offsets.slice(1).map(offset => `${String(offset).padStart(10, "0")} 00000 n \n`).join("")}trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${start}\n%%EOF\n`;
  return result;
}

export function fixtureConfig({ vaultRoot, cacheDir }) {
  return `schema_version = 1\nvault_root = ${JSON.stringify(vaultRoot)}\ncache_dir = ${JSON.stringify(cacheDir)}\n\n[llm]\nprovider = "openai"\nbase_url = "https://fixture-model.test/v1"\napi_key = "fixture-key"\nmodel = "fixture-model"\nthinking_mode = false\n\n[arxiv]\ncategories = ["cs.AI"]\ntimezone = "UTC"\n\n[[arxiv.topics]]\nid = "acceptance-topic"\nname = "Research"\ntag = "research"\ndetail = false\ndirections = [{ id = "acceptance-direction", text = "Reliable research methods", origin = "manual" }]\n\n[output]\ndaily_dir = "arxiv-daily/daily"\npapers_dir = "arxiv-daily/papers"\nsummary_language = "zh"\nlink_style = "relative"\nmax_daily_papers = 2\n\n[email]\nenabled = false\n\n[advanced]\nrequest_delay_ms = 3000\n\n[workbench_schedule]\nenabled = false\n`;
}

export async function createFixtureEnvironment({ configured = true, mode } = {}) {
  const root = await mkdtemp(join(tmpdir(), "arxiv-acceptance-"));
  let server;
  try {
    const vaultRoot = join(root, "vault"), libraryRoot = join(vaultRoot, "library"), configHome = join(root, "config"), cacheDir = join(root, "cache");
    const configPath = join(configHome, "arxiv-daily", "config.toml");
    await Promise.all([mkdir(libraryRoot, { recursive: true }), mkdir(join(vaultRoot, ".obsidian"), { recursive: true }), mkdir(join(configHome, "arxiv-daily"), { recursive: true }), mkdir(cacheDir, { recursive: true })]);
    await writeFile(join(vaultRoot, ".obsidian", "community-plugins.json"), '["arxiv-daily"]\n');
    await Promise.all(PAPERS.map(p => writeFile(join(libraryRoot, `${p.id}.pdf`), fixturePdf(p))));
    if (configured) await writeFile(configPath, fixtureConfig({ vaultRoot, cacheDir }), { mode: 0o600 });
    server = await startFixtureServer({ mode });
    const preload = fileURLToPath(new URL("./fixture-preload.cjs", import.meta.url));
    const env = { ...process.env, XDG_CONFIG_HOME: configHome, XDG_CACHE_HOME: cacheDir, ARXIV_ACCEPTANCE_FIXTURE_URL: server.url, NODE_OPTIONS: `--require=${JSON.stringify(preload)}` };
    let disposed;
    return { root, vaultRoot, libraryRoot, configHome, configPath, cacheDir, env, server, dispose() {
      disposed ??= (async () => { await server.close(); await rm(root, { recursive: true, force: true }); })();
      return disposed;
    } };
  } catch (error) { await server?.close(); await rm(root, { recursive: true, force: true }); throw error; }
}
