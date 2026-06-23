import type { PluginSettings } from "./settings/types";
import { arxivCategories } from "./settings/categories";
import { validateFilterConfig } from "./settings/validation";

export interface SetupStatus {
  llmReady: boolean;
  categoriesReady: boolean;
  topicsReady: boolean;
  readyToRun: boolean;
  reasons: string[];
}

export function shouldRenderSetupGuide(status: Pick<SetupStatus, "readyToRun">): boolean {
  return !status.readyToRun;
}

export function getSetupStatus(settings: PluginSettings): SetupStatus {
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

  return {
    llmReady,
    categoriesReady,
    topicsReady,
    readyToRun: validation.ok,
    reasons: validation.reasons,
  };
}
