import type { GenerationMetrics } from "@arxiv-daily/core";
import { t } from "./i18n";
import { h } from "./dom";

const count = (value: number | undefined): string => value == null ? t("未记录") : String(value);
const duration = (value: number | undefined): string => value == null ? t("未记录") : value < 1000 ? `${value} ms` : `${(value / 1000).toFixed(1)} s`;

/** Persisted measurements only; neither file mtime nor estimated token counts. */
export function generationFooter(metrics: GenerationMetrics | null | undefined, scope?: "daily" | "paper"): HTMLElement {
  const timestamp = metrics?.generatedAt && Number.isFinite(Date.parse(metrics.generatedAt)) ? new Date(metrics.generatedAt).toISOString() : null;
  const fields: Array<[string, string]> = [
    [t("输入 Token"), count(metrics?.inputTokens)], [t("输出 Token"), count(metrics?.outputTokens)],
    [t("总 Token"), count(metrics?.totalTokens)], [t("生成耗时"), duration(metrics?.pipelineElapsedMs)],
    [t("LLM 累计耗时"), duration(metrics?.elapsedMs)],
  ];
  return h("footer", { class: "generation-footer" },
    h("h2", null, t("生成信息")),
    scope ? h("p", { class: "generation-scope" }, t(scope === "daily" ? "统计范围：整份来源日报" : "统计范围：本篇详细总结")) : null,
    h("dl", null,
      ...fields.map(([label, value]) => h("div", null, h("dt", null, label), h("dd", null, value))),
      h("div", { class: "generation-timestamp" },
        h("dt", null, t("生成时间")),
        h("dd", null, timestamp ? h("time", { datetime: timestamp }, timestamp.replace("T", " ").replace("Z", " UTC")) : t("未记录")),
      ),
    ),
    metrics && !metrics.usageComplete ? h("p", { class: "generation-incomplete" }, t("Token 用量记录不完整")) : null,
  );
}
