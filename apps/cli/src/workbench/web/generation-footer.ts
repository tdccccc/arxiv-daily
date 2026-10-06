import type { GenerationMetrics } from "@arxiv-daily/core";
import { t } from "./i18n";

const escape = (value: string): string => value.replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]!);
const count = (value: number | undefined): string => value == null ? t("未记录") : String(value);
const duration = (value: number | undefined): string => value == null ? t("未记录") : value < 1000 ? `${value} ms` : `${(value / 1000).toFixed(1)} s`;

/** Persisted measurements only; neither file mtime nor estimated token counts. */
export function generationFooter(metrics: GenerationMetrics | null | undefined, scope?: "daily" | "paper"): string {
  const timestamp = metrics?.generatedAt && Number.isFinite(Date.parse(metrics.generatedAt)) ? new Date(metrics.generatedAt).toISOString() : null;
  const fields = [
    ["输入 Token", count(metrics?.inputTokens)], ["输出 Token", count(metrics?.outputTokens)],
    ["总 Token", count(metrics?.totalTokens)], ["生成耗时", duration(metrics?.pipelineElapsedMs)],
    ["LLM 累计耗时", duration(metrics?.elapsedMs)],
  ];
  return `<footer class="generation-footer"><h2>${escape(t("生成信息"))}</h2>${scope ? `<p class="generation-scope">${escape(t(scope === "daily" ? "统计范围：整份来源日报" : "统计范围：本篇详细总结"))}</p>` : ""}<dl>${fields.map(([label, value]) => `<div><dt>${escape(t(label!))}</dt><dd>${escape(value!)}</dd></div>`).join("")}<div class="generation-timestamp"><dt>${escape(t("生成时间"))}</dt><dd>${timestamp ? `<time datetime="${escape(timestamp)}">${escape(timestamp.replace("T", " ").replace("Z", " UTC"))}</time>` : escape(t("未记录"))}</dd></div></dl>${metrics && !metrics.usageComplete ? `<p class="generation-incomplete">${escape(t("Token 用量记录不完整"))}</p>` : ""}</footer>`;
}
