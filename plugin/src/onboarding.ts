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
  scheduleEnabled: boolean;
  latestCompletedReportDate?: string;
  reasons: string[];
  schedulerReasons: string[];
}

/** True once every guide milestone is complete at the same time. */
export function isSetupComplete(
  status: Pick<SetupStatus, "readyToRun" | "firstReportComplete" | "scheduleEnabled">,
): boolean {
  return status.readyToRun && status.firstReportComplete && status.scheduleEnabled;
}

/**
 * The guide is for first-time setup only: it shows until every milestone is
 * complete at once, then stays hidden for good (tracked by the persisted
 * `settings.onboarding.guideCompleted` marker), even if the user later turns
 * the schedule off or their configuration becomes invalid again.
 */
export function shouldRenderSetupGuide(
  status: Pick<SetupStatus, "readyToRun" | "firstReportComplete" | "scheduleEnabled">,
  guideCompleted: boolean,
): boolean {
  if (guideCompleted) return false;
  return !isSetupComplete(status);
}

/**
 * Persist the "guide completed" marker the first time all milestones are
 * true at once. Idempotent; returns whether it just flipped from false to
 * true so a caller can show a one-time completion confirmation.
 */
export function markSetupGuideCompleteIfDone(
  settings: PluginSettings,
  status: Pick<SetupStatus, "readyToRun" | "firstReportComplete" | "scheduleEnabled">,
): boolean {
  if (settings.onboarding.guideCompleted || !isSetupComplete(status)) return false;
  settings.onboarding.guideCompleted = true;
  return true;
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
  const topicTags = settings.arxiv.topics.map((topic) => topic.tag.trim());
  const topicsReady =
    settings.arxiv.topics.length > 0 &&
    settings.arxiv.topics.every(
      (topic) =>
        topic.name.trim() &&
        topic.tag.trim() &&
        topic.directions.some((direction) => direction.text.trim()),
    ) &&
    new Set(topicTags).size === topicTags.length;
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
    scheduleEnabled: settings.schedule.enabled,
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
