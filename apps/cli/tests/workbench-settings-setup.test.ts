import { setUiLanguage } from "../src/workbench/web/i18n";
// @vitest-environment happy-dom
import { expect, it, vi } from "vitest";
import { readWorkbenchSettings } from "../src/workbench/settings";
import { settingsSetupGuide } from "../src/workbench/web/settings-setup";
import { bindSettings, settingsForm } from "../src/workbench/web/settings";
it("matches the five Obsidian setup steps, gates generation and hides only after completion", async () => {
  setUiLanguage("en");
  const snapshot = await readWorkbenchSettings('/tmp/arxiv-missing-setup-fixture/config.toml');
  const root = document.createElement('div'); root.replaceChildren(settingsSetupGuide(snapshot)!);
  expect(Array.from(root.querySelectorAll('li strong')).map(e=>e.textContent)).toEqual(['Connect AI','Choose paper sources','Describe your research interests','Generate your first report','Turn on daily reports']);
  expect(root.querySelector('[data-setup-action="generate"]')).toBeNull();
  snapshot.values.apiKeyConfigured=true; snapshot.values.baseUrl='https://example.test/v1'; snapshot.values.model='model';
  snapshot.values.topics=[{id:'focus',name:'Test',tag:'test',description:'Research',directions:[{id:'d1',text:'Research',origin:'manual'}],detail:true}];
  root.replaceChildren(settingsSetupGuide(snapshot)!);
  expect(root.querySelector('[data-setup-action="generate"]')).toBeTruthy();
  snapshot.values.schedule.enabled=true;
  expect(settingsSetupGuide(snapshot,true)).toBeNull();
});

it("renders the setup guide in Chinese without changing model content", async()=>{
 setUiLanguage('zh');const snapshot=await readWorkbenchSettings('/tmp/arxiv-missing-setup-fixture/config.toml');
 const root=document.createElement('div');root.replaceChildren(settingsSetupGuide(snapshot)!);
 expect(root.querySelector('h2')?.textContent).toBe('开始使用');expect(root.querySelector('li strong')?.textContent).toBe('连接 AI');
 setUiLanguage('en');
});

it("does not reopen Enable onboarding for a configured manual-report user", async () => {
 setUiLanguage('en');
 const snapshot=await readWorkbenchSettings('/tmp/arxiv-missing-setup-fixture/config.toml');
 snapshot.values.apiKeyConfigured=true;snapshot.values.baseUrl='https://example.test/v1';snapshot.values.model='model';
 snapshot.values.topics=[{id:'focus',name:'Test',tag:'test',description:'Research',directions:[{id:'d1',text:'Research',origin:'manual'}],detail:true}];
 snapshot.values.schedule.enabled=false;
 expect(settingsSetupGuide(snapshot,true)).toBeNull();
 expect(snapshot.values.schedule.enabled).toBe(false);
 expect(settingsSetupGuide(snapshot,false)?.textContent).toContain('Getting started');
});

// The settings dialog is modal and covers the run tray, so the step itself has to say what its button did.
const running = { id: "first", label: "2026-10-09 日报", date: "2026-10-09", status: "running", output: "", exitCode: null, startedAt: "2026-10-21T00:00:00Z", finishedAt: null } as const;

async function guide(request: (url: string, body?: unknown) => Promise<unknown>) {
  setUiLanguage("en");
  const snapshot = await readWorkbenchSettings('/tmp/arxiv-missing-setup-fixture/config.toml');
  snapshot.values.apiKeyConfigured = true; snapshot.values.baseUrl = 'https://example.test/v1'; snapshot.values.model = 'model';
  snapshot.values.topics = [{ id: 'focus', name: 'Test', tag: 'test', description: 'Research', directions: [{ id: 'd1', text: 'Research', origin: 'manual' }], detail: true }];
  const form = settingsForm(snapshot);
  document.body.replaceChildren(form);
  const onRun = vi.fn();
  const binding = bindSettings(form, snapshot, request as never, async () => {}, onRun);
  const generate = () => form.querySelector<HTMLButtonElement>('[data-setup-action="generate"]')!;
  const status = () => form.querySelector('.settings-setup-status')?.textContent ?? '';
  const finish = (run: object) => form.dispatchEvent(new CustomEvent('workbench-run', { detail: run }));
  return { form, binding, onRun, generate, status, finish };
}

it("shows the first report's progress in its setup step and cannot start it twice", async () => {
  let accept!: (value: unknown) => void;
  const request = vi.fn(async (url: string) => { if (url === 'api/runs') return new Promise(resolve => { accept = resolve; }); throw new Error(`Unexpected request ${url}`); });
  const { binding, onRun, generate, status, finish } = await guide(request);
  // The step sits low in the scrolling dialog, so its status is brought into view.
  const reveal = vi.fn();
  Object.assign(Element.prototype, { scrollIntoView: reveal });
  try {
    generate().click();
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('api/runs', { kind: 'daily', latest: true }));
    expect(generate().disabled).toBe(true);
    expect(generate().textContent).toBe('Generating…');
    generate().click();
    accept({ run: running });
    await vi.waitFor(() => expect(onRun).toHaveBeenCalledWith(running));
    expect(reveal).toHaveBeenCalledOnce();
    expect(reveal.mock.contexts[0]).toBe(document.querySelector('.settings-setup-status'));
    expect(status()).toContain('Generating the report for 2026-10-09…');
    expect(generate().disabled).toBe(true);
    expect(generate().textContent).toBe('Generating…');
    generate().click();
    expect(request.mock.calls.filter(([url]) => url === 'api/runs')).toHaveLength(1);

    finish({ ...running, status: 'completed', outcome: 'papers_written', exitCode: 0 });
    expect(status()).toContain('2026-10-09 · Daily report saved');
    expect(generate().disabled).toBe(false);
    expect(generate().textContent).toBe('Generate first report');
  } finally { binding.dispose(); document.body.replaceChildren(); setUiLanguage('en'); Reflect.deleteProperty(Element.prototype, 'scrollIntoView'); }
});

it("says what to do next when the first report ends before its day is announced", async () => {
  const request = vi.fn(async () => ({ run: running }));
  const { binding, generate, status, finish } = await guide(request);
  try {
    generate().click();
    await vi.waitFor(() => expect(status()).toContain('Generating'));
    finish({ ...running, status: 'pending', outcome: 'awaiting_announcement', exitCode: 0 });
    expect(status()).toContain('2026-10-09 · Awaiting announcement');
    expect(status()).toContain('Check again later, or pick an announced day in the calendar.');
    expect(generate().disabled).toBe(false);
  } finally { binding.dispose(); document.body.replaceChildren(); }
});

it("explains why the first report is for an earlier day, and why it failed", async () => {
  const request = vi.fn(async () => ({ run: running, fellBackFrom: '2026-10-21' }));
  const { binding, generate, status, finish } = await guide(request);
  try {
    generate().click();
    await vi.waitFor(() => expect(status()).toContain('Generating'));
    expect(status()).toContain("Today's papers are not announced yet, so the first report uses the latest announced day, 2026-10-09.");
    finish({ ...running, status: 'failed', exitCode: 1, output: 'run 2026-10-09: failed_transient (network error)\n\n' });
    expect(status()).toContain('2026-10-09 · Generation failed');
    expect(status()).toContain('run 2026-10-09: failed_transient (network error)');
    expect(generate().disabled).toBe(false);
  } finally { binding.dispose(); document.body.replaceChildren(); }
});

it("keeps the step usable when the run cannot be started", async () => {
  const request = vi.fn(async () => { throw new Error('已有任务正在运行，请等待完成或先取消。'); });
  const { form, binding, onRun, generate } = await guide(request);
  try {
    generate().click();
    await vi.waitFor(() => expect(form.querySelector<HTMLElement>('[role="alert"]')?.hidden).toBe(false));
    expect(form.querySelector('[role="alert"]')?.textContent).toContain('已有任务正在运行');
    expect(generate().disabled).toBe(false);
    expect(generate().textContent).toBe('Generate first report');
    expect(onRun).not.toHaveBeenCalled();
  } finally { binding.dispose(); document.body.replaceChildren(); }
});
