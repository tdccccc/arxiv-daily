// @vitest-environment happy-dom
import { expect, it } from "vitest";
import { readWorkbenchSettings } from "../src/workbench/settings";
import { settingsSetupGuide } from "../src/workbench/web/settings-setup";
it("matches the five Obsidian setup steps, gates generation and hides only after completion", async () => {
  const snapshot = await readWorkbenchSettings('/tmp/arxiv-missing-setup-fixture/config.toml');
  const root = document.createElement('div'); root.innerHTML = settingsSetupGuide(snapshot);
  expect(Array.from(root.querySelectorAll('li strong')).map(e=>e.textContent)).toEqual(['Connect AI','Choose paper sources','Describe your research interests','Generate your first report','Turn on daily reports']);
  expect(root.querySelector('[data-setup-action="generate"]')).toBeNull();
  snapshot.values.apiKeyConfigured=true; snapshot.values.baseUrl='https://example.test/v1'; snapshot.values.model='model';
  snapshot.values.topics=[{id:'focus',name:'Test',tag:'test',description:'Research',detail:true}];
  root.innerHTML=settingsSetupGuide(snapshot);
  expect(root.querySelector('[data-setup-action="generate"]')).toBeTruthy();
  snapshot.values.schedule.enabled=true;
  expect(settingsSetupGuide(snapshot,true)).toBe('');
});
