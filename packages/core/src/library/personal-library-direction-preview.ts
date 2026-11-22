import type { ChatMessage, CallOptions } from "../llm/client";
import { buildPaperFilterRequest, decodePaperFilterRecords } from "../pipeline/paper-filter-contract";
import { unwrapSingleCodeFence } from "../pipeline/paper-filter";
import { normalizeTopic } from "../settings/topics";
import { throwIfCancelled } from "../services/cancellation";

export interface LibraryPreviewPaper {
  paperKey: string;
  title: string;
  abstract: string;
  categories: string[];
}

export interface LibraryDirectionPreview {
  directionText: string;
  categories: string[];
  missingCategories: string[];
  papers: Array<LibraryPreviewPaper & {
    matched: boolean;
    directionText: string;
    categoryCoverage: "inside" | "outside" | "unknown";
  }>;
}

export async function previewPersonalLibraryDirection(input: {
  text: string;
  papers: readonly LibraryPreviewPaper[];
  categories: readonly string[];
  llm: { call(messages: ChatMessage[], options?: CallOptions): Promise<string> };
  signal?: AbortSignal;
}): Promise<LibraryDirectionPreview> {
  throwIfCancelled(input.signal);
  const text = input.text.trim();
  if (!text || text.length > 1000 || /[\r\n]/u.test(text)
    || input.papers.length === 0 || input.papers.length > 20) throw new Error("Invalid direction preview input");
  const papers = structuredClone([...input.papers]);
  const categories = [...input.categories];
  const topic = normalizeTopic({ id: "preview", name: "Preview", tag: "preview", detail: false,
    directions: [{ id: "preview-direction", text, origin: "manual" }],
  });
  const request = buildPaperFilterRequest(papers.map((paper, index) => ({
    id: `sample-${index + 1}`, title: paper.title, abstract: paper.abstract, authors: "",
  })), { topics: [topic], categories, category: categories[0] ?? "", timezone: "UTC" });
  // The classification contract is shared; identify the input honestly as a
  // library sample rather than today's complete arXiv stream.
  request.messages[1]!.content = request.messages[1]!.content.replace(/^[\s\S]*?<paper_data>/u,
    "Library sample for a research-direction preview. Classify using the supplied direction.\n<paper_data>");
  if (request.messages.reduce((total, message) => total + message.content.length, 0) > 60_000) {
    throw new Error("Direction preview evidence is too large");
  }
  const options = { ...request.options, signal: input.signal, maxCompletionTokens: 4096, maxOutputCodeUnits: 64_000 };
  const knownIds = new Set(request.identity.knownIds);
  const validTags = new Set(request.identity.validTags);
  let messages = request.messages;
  let records: ReturnType<typeof decodePaperFilterRecords> | undefined;
  let failure = "Invalid direction preview response";
  // One format-correction retry, mirroring the daily filter: the model sees
  // only which rule it broke, never its own rejected answer.
  for (let attempt = 1; attempt <= 2; attempt += 1) {
    const raw = await input.llm.call(messages, options);
    throwIfCancelled(input.signal);
    let value: unknown;
    try {
      value = JSON.parse(unwrapSingleCodeFence(raw));
    } catch {
      failure = "Invalid direction preview response";
      messages = withCorrection(request.messages, "回复不是严格 JSON");
      continue;
    }
    const decoded = decodePaperFilterRecords(value, knownIds, validTags, request.identity.directions);
    if (decoded.ok) {
      records = decoded;
      break;
    }
    failure = `Invalid direction preview response: ${decoded.reason}`;
    messages = withCorrection(request.messages, `回复违反了输出格式（${decoded.reason}）`);
  }
  if (!records?.ok) throw new Error(failure);
  const byId = new Map(records.value.map((record) => [record.id, record]));
  const missing = new Set<string>();
  const results = papers.map((paper, index) => {
    const matched = byId.get(`sample-${index + 1}`)?.category === "preview";
    const inside = paper.categories.some((category) => categories.some((selected) =>
      category === selected || category.startsWith(`${selected}.`)));
    if (matched && !inside) for (const category of paper.categories) missing.add(category);
    return { ...paper, matched, directionText: matched ? text : "",
      categoryCoverage: paper.categories.length === 0 ? "unknown" as const : inside ? "inside" as const : "outside" as const,
    };
  });
  return { directionText: text, categories, missingCategories: [...missing].sort(), papers: results };
}

/**
 * The same request with the broken rule appended to its user turn. Only this
 * module's words go in; the model's rejected answer is never echoed back.
 */
function withCorrection(messages: readonly ChatMessage[], reason: string): ChatMessage[] {
  const guidance = [
    `上一次回复未通过校验：${reason}。`,
    "请重新输出，并且只输出一个 JSON 对象：不要用 Markdown 代码块包裹，不要附加任何其他文本；",
    "根对象只有 papers 数组；每条记录恰好包含 id、category、directions、relevanceScore；",
    "id 必须来自输入样本且不重复。",
  ].join("\n");
  const last = messages[messages.length - 1];
  if (!last || last.role !== "user") return [...messages, { role: "user", content: guidance }];
  return [...messages.slice(0, -1), { role: "user", content: `${last.content}\n\n${guidance}` }];
}
