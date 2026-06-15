import { describe, expect, it, vi } from "vitest";
import { summarizeDaily } from "../src/pipeline/summarizer";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { Logger } from "../src/services/logger";

describe("summarizeDaily link style", () => {
  it("passes relative detail links through the prompt", async () => {
    const calls: any[] = [];
    const llm = {
      call: vi.fn(async (messages: any[]) => {
        calls.push(messages);
        return [
          "# arXiv astro-ph 每日追踪 2026-06-13",
          "## Topic",
          "### Relative Link Paper → [2606.12345](../papers/2606.12345.md)",
          "- **arXiv**: [2606.12345](https://arxiv.org/abs/2606.12345)",
        ].join("\n");
      }),
    };

    await summarizeDaily(
      [
        {
          id: "2606.12345",
          title: "Relative Link Paper",
          authors: "A. Author",
          abstract: "abstract",
          category: "topic",
          isDetail: true,
          abstractConclusion: "## Abstract\nabstract",
          fullSections: null,
          detailLink: "[2606.12345](../papers/2606.12345.md)",
        },
      ],
      "2026-06-13",
      {
        llm: llm as any,
        logger: new Logger("error"),
        arxivSettings: {
          ...DEFAULT_SETTINGS.arxiv,
          topics: [
            {
              id: "topic",
              name: "Topic",
              tag: "topic",
              description: "topic",
              detail: true,
            },
          ],
        },
        advanced: DEFAULT_SETTINGS.advanced,
        llmTemperature: DEFAULT_SETTINGS.llm.temperature,
        linkStyle: "relative",
      },
    );

    const systemPrompt = calls[0][0].content;
    const userPrompt = calls[0][1].content;
    expect(systemPrompt).toContain(
      "### <实际论文标题> → [YYMM.NNNNN](../papers/YYMM.NNNNN.md)",
    );
    expect(systemPrompt).not.toContain("[[YYMM.NNNNN]]");
    expect(userPrompt).toContain(
      "=== Paper: 2606.12345 [category: topic] → [2606.12345](../papers/2606.12345.md) ===",
    );
  });

  it("does not inject legacy daily selection controls", async () => {
    const dailyMarkdown = [
      "# arXiv astro-ph 每日追踪 2026-06-13",
      "## Topic",
      "### Example Paper",
      "- **arXiv**: [2606.12345](https://arxiv.org/abs/2606.12345)",
      "- **核心问题**: 原文未说明",
    ].join("\n");
    const llm = {
      call: vi.fn(async () => dailyMarkdown),
    };

    const out = await summarizeDaily(
      [
        {
          id: "2606.12345",
          title: "Example Paper",
          authors: "A. Author",
          abstract: "abstract",
          category: "topic",
          isDetail: false,
          abstractConclusion: "## Abstract\nabstract",
          fullSections: null,
          inboxStatus: "to_read",
        },
      ],
      "2026-06-13",
      {
        llm: llm as any,
        logger: new Logger("error"),
        arxivSettings: {
          ...DEFAULT_SETTINGS.arxiv,
          topics: [
            {
              id: "topic",
              name: "Topic",
              tag: "topic",
              description: "topic",
              detail: false,
            },
          ],
        },
        advanced: DEFAULT_SETTINGS.advanced,
        llmTemperature: DEFAULT_SETTINGS.llm.temperature,
      },
    );

    expect(out).toBe(dailyMarkdown);
    expect(out).not.toContain("arxiv-daily:2606.12345:watch");
    expect(out).not.toContain("关注");
    expect(out).not.toContain("重点");
  });
});
