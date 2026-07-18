import type { PluginSettings, RunState } from "@arxiv-daily/core";
import { arxivCategories } from "@arxiv-daily/core";
import { validateFilterConfig, validateSchedulerConfig } from "@arxiv-daily/core";
import type { Logger } from "@arxiv-daily/core";

export interface SetupStatus {
  llmReady: boolean;
  categoriesReady: boolean;
  topicsReady: boolean;
  readyToRun: boolean;
  firstReportComplete: boolean;
  latestCompletedReportDate?: string;
  reasons: string[];
  schedulerReasons: string[];
}

export function shouldRenderSetupGuide(
  status: Pick<SetupStatus, "readyToRun" | "firstReportComplete">,
): boolean {
  return !status.readyToRun || !status.firstReportComplete;
}

export function getSetupStatus(
  settings: PluginSettings,
  runState: RunState = {},
): SetupStatus {
  const llmReady = Boolean(
    settings.llm.apiKey.trim() &&
      settings.llm.baseUrl.trim() &&
      settings.llm.model.trim(),
  );
  const categoriesReady = arxivCategories(settings.arxiv).length > 0;
  const topicsReady =
    settings.arxiv.topics.length > 0 &&
    settings.arxiv.topics.every(
      (topic) =>
        topic.name.trim() &&
        topic.tag.trim() &&
        topic.description.trim(),
    );
  const validation = validateFilterConfig(settings);
  const schedulerValidation = validateSchedulerConfig(settings);
  const latestCompletedReportDate = Object.entries(runState)
    .filter(([, entry]) => entry?.status === "completed")
    .map(([date]) => date)
    .sort()
    .at(-1);

  return {
    llmReady,
    categoriesReady,
    topicsReady,
    readyToRun: validation.ok,
    firstReportComplete: latestCompletedReportDate !== undefined,
    latestCompletedReportDate,
    reasons: validation.reasons,
    schedulerReasons: schedulerValidation.reasons,
  };
}

export function logSetupStatus(
  logger: Logger,
  context: string,
  status: SetupStatus,
): void {
  logger.info(
    `onboarding: ${context}: ready=${status.readyToRun}, llm=${status.llmReady}, categories=${status.categoriesReady}, topics=${status.topicsReady}, reasons=${status.reasons.join("; ") || "none"}`,
  );
}
