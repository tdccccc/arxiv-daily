import { t } from "./i18n";
import { DEFAULT_SETTINGS, validateFilterConfig } from '@arxiv-daily/core';
import type { WorkbenchSettings } from '../settings';
import { h } from "./dom";
/** Obsidian's five setup steps, projected from the same persisted settings. */
export function settingsSetupGuide(snapshot: WorkbenchSettings, firstReportComplete = false): HTMLElement | null {
  const v = snapshot.values;
  const llmReady = Boolean(v.apiKeyConfigured && v.baseUrl && v.model);
  const categoriesReady = v.categories.length > 0;
  const tags = v.topics.map(t => t.tag.trim());
  const topicsReady = v.topics.length > 0 && v.topics.every(t => t.name.trim() && t.tag.trim() && t.description.trim()) && new Set(tags).size === tags.length;
  const configured = { ...DEFAULT_SETTINGS, llm: { ...DEFAULT_SETTINGS.llm, apiKey: v.apiKeyConfigured ? 'configured' : '', baseUrl: v.baseUrl, model: v.model }, arxiv: { ...DEFAULT_SETTINGS.arxiv, categories: v.categories, topics: v.topics, timezone: v.timezone }, output: { ...DEFAULT_SETTINGS.output, dailyDir: v.dailyDir, papersDir: v.papersDir } };
  const validation = validateFilterConfig(configured);
  const enabled = v.schedule?.enabled ?? false;
  // Scheduling is optional: a successful manual report also completes setup.
  if (validation.ok && firstReportComplete) return null;
  const button = (action: string, text: string) => h("button", { type: "button", "data-setup-action": action }, t(text));
  const steps = [
    { done: llmReady, name: 'Connect AI', text: 'Add an API key, API base URL, and model under LLM.', action: button('llm', 'Connect AI') },
    { done: categoriesReady, name: 'Choose paper sources', text: 'Select at least one arXiv category under arXiv categories.', action: button('arxiv', 'Choose sources') },
    { done: topicsReady, name: 'Describe your research interests', text: 'Add at least one complete research topic under Research topics.', action: button('topics', 'Describe interests') },
    { done: firstReportComplete, name: 'Generate your first report', text: validation.ok ? 'Your configuration is ready. Generate a report to finish setup.' : 'Complete the earlier configuration steps before generating a report.', action: validation.ok ? button('generate', 'Generate first report') : null },
    { done: enabled, name: 'Turn on daily reports', text: validation.ok ? 'Reports then run by themselves on weekdays, inside the run window below.' : 'Available once the configuration above is complete.', action: validation.ok ? button('enable', 'Turn on daily reports') : null },
  ];
  const count = steps.filter(s => s.done).length;
  return h("div", { class: "settings-setup" },
    h("h2", null, t('Getting started')),
    h("p", null, t('{0} of 5 complete', count)),
    h("progress", { max: 5, value: count, "aria-label": t('Setup progress') }),
    h("ol", null, ...steps.map(s => h("li", { "data-complete": String(s.done) }, h("strong", null, t(s.name)), h("p", null, t(s.text)), s.action))),
    validation.reasons.length ? h("details", null, h("summary", null, t('Configuration details')), h("p", null, validation.reasons.join('; '))) : null,
    button('dashboard', 'Open dashboard'),
  );
}
