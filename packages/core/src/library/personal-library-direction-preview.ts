import type { ChatMessage, CallOptions } from "../llm/client";
import { buildPaperFilterRequest, decodePaperFilterRecords } from "../pipeline/paper-filter-contract";
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
  const raw = await input.llm.call(request.messages, { ...request.options, signal: input.signal, maxCompletionTokens: 4096, maxOutputCodeUnits: 64_000 });
  throwIfCancelled(input.signal);
  let value: unknown;
  try { value = JSON.parse(raw); } catch { throw new Error("Invalid direction preview response"); }
  const decoded = decodePaperFilterRecords(value, new Set(request.identity.knownIds), new Set(request.identity.validTags), request.identity.directions);
  if (!decoded.ok) throw new Error(`Invalid direction preview response: ${decoded.reason}`);
  const byId = new Map(decoded.value.map((record) => [record.id, record]));
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
