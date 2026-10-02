const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { spawn } = require('node:child_process');

test('copied CLI serves its complete reading UI and dispatches original product generation', { timeout: 60000 }, async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'arxiv-workbench-package-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const binary = path.join(root, 'standalone.cjs');
  await fs.copyFile(path.resolve(__dirname, '../dist/arxiv-daily-cli.cjs'), binary);
  const configHome = path.join(root, 'config');
  const configDir = path.join(configHome, 'arxiv-daily');
  const vault = path.join(root, 'vault with spaces');
  await fs.mkdir(configDir, { recursive: true });
  await fs.mkdir(path.join(vault, 'arxiv-daily/papers'), { recursive: true });
  const source = '# Reading fixture\n\n## Method\n\n$x^2$\n\n| Method | Score |\n| --- | --- |\n| Ours | 42 |\n';
  await fs.writeFile(path.join(vault, 'arxiv-daily/papers/existing.md'), source);
  const configPath = path.join(configDir, 'config.toml');
  await fs.writeFile(configPath, `vault_root = ${JSON.stringify(vault)}\ncache_dir = ${JSON.stringify(path.join(root, 'cache'))}\n[llm]\nprovider = "openai"\napi_key = "fixture-secret-never-log"\nbase_url = "https://fixture.invalid/v1"\nmodel = "fixture-model"\nthinking_mode = false\n[arxiv]\ncategories = ["astro-ph"]\ntimezone = "UTC"\n[[arxiv.topics]]\nname = "Photometric redshifts"\ntag = "photo-z"\ndescription = "Photometric redshift estimation and calibration"\ndetail = true\n[output]\nsummary_language = "en"\nlink_style = "relative"\n[advanced]\nlog_level = "error"\n`);
  const log = path.join(root, 'requests.jsonl');
  await fs.writeFile(log, '');
  const env = { ...process.env, XDG_CONFIG_HOME: configHome, APPDATA: configHome,
    NODE_OPTIONS: `--require=${JSON.stringify(path.join(__dirname, 'fixtures/core-http.cjs'))}`,
    ARXIV_CORE_FIXTURE_SCENARIO: 'selected', ARXIV_CORE_FIXTURE_LOG: log };
  const child = spawn(process.execPath, [binary, 'ui', '--no-open'], { cwd: root, env, stdio: ['ignore', 'pipe', 'pipe'] });
  const exited = new Promise(resolve => child.once('exit', (code, signal) => resolve({ code, signal })));
  t.after(async () => { if (child.exitCode === null && child.signalCode === null) { child.kill('SIGTERM'); await exited; } });
  let output = '';
  child.stderr.on('data', chunk => { output += chunk; });
  const url = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`No workbench URL: ${output}`)), 10000);
    child.stdout.on('data', chunk => {
      output += chunk;
      const found = /Workbench: (http:\/\/127\.0\.0\.1:\d+\/[a-f0-9]+\/)/.exec(output);
      if (found) { clearTimeout(timer); resolve(found[1]); }
    });
    child.once('error', error => { clearTimeout(timer); reject(error); });
    child.once('exit', code => { clearTimeout(timer); reject(new Error(`Workbench exited ${code}: ${output}`)); });
  });
  const get = async endpoint => { const response = await fetch(new URL(endpoint, url)); assert.equal(response.status, 200, await response.clone().text()); return response; };
  assert.match(await (await get('')).text(), /arXiv Daily/);
  for (const asset of ['app.js', 'style.css', 'katex.css']) assert.ok((await (await get(asset)).text()).length > 100);
  const css = await (await get('katex.css')).text();
  const font = /url\(["']?(fonts\/[^)'" ]+\.woff2)/.exec(css)?.[1];
  assert.ok(font, 'math CSS references a bundled font');
  assert.ok((await (await get(font)).arrayBuffer()).byteLength > 100);
  const document = await (await get('api/document?path=arxiv-daily/papers/existing.md')).json();
  assert.match(document.html, /katex/);
  assert.match(document.html, /<table>/);
  const calendarBefore = await (await get('api/calendar?month=2026-05')).json();
  assert.equal(calendarBefore.timezone, 'UTC');
  assert.equal(calendarBefore.cells.filter(Boolean).length, 31);
  assert.equal(calendarBefore.cells.find(day => day?.date === '2026-05-11').state, 'not-generated');
  assert.equal((await fetch(new URL('api/calendar?month=2026-13', url))).status, 400);
  assert.equal(await fs.readFile(log, 'utf8'), '', 'reading must not trigger any HTTP/model request');

  const launch = await fetch(new URL('api/runs', url), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ kind: 'daily', date: '2026-05-11' }) });
  assert.equal(launch.status, 202, await launch.clone().text());
  let run;
  for (let i = 0; i < 400; i++) {
    run = (await (await get('api/runs/current')).json()).run;
    if (run.status !== 'running') break;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  assert.equal(run.status, 'completed', run.output);
  assert.doesNotMatch(run.output, /fixture-secret-never-log/);
  const daily = await fs.readFile(path.join(vault, 'arxiv-daily/daily/2026-05-11.md'), 'utf8');
  assert.match(daily, /Fixture calibrated redshifts/);
  assert.match(await fs.readFile(path.join(vault, 'arxiv-daily/papers/2605.08080.md'), 'utf8'), /Method Design/);
  assert.equal(await fs.readFile(path.join(vault, 'arxiv-daily/papers/existing.md'), 'utf8'), source);
  const listed = await (await get('api/documents?kind=daily')).json();
  assert.equal(listed.total, 1);
  const calendarAfter = await (await get('api/calendar?month=2026-05')).json();
  const completedDay = calendarAfter.cells.find(day => day?.date === '2026-05-11');
  assert.equal(completedDay.state, 'has-report');
  assert.equal(completedDay.reportPath, 'arxiv-daily/daily/2026-05-11.md');
  assert.equal(completedDay.papers, 2);
  assert.equal(completedDay.canGenerate, false);

  const papers = await (await get('api/papers?date=2026-05-11')).json();
  assert.equal(papers.total, 2);
  const indexOnly = papers.papers.find(paper => paper.arxivId === '2605.08068');
  assert.equal(indexOnly.detailPath, null, 'discoveries without a note remain visible');
  assert.ok(indexOnly.summary.whyRelevant);
  const post = (endpoint, body) => fetch(new URL(endpoint, url), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  assert.equal((await post('api/paper/mark', { key: indexOnly.key, action: 'status', value: 'to_read', expected: 'inbox' })).status, 200);
  assert.equal((await post('api/paper/mark', { key: indexOnly.key, action: 'star', value: true, expected: 'normal' })).status, 200);
  assert.equal((await post('api/preferences', { sidebarWidth: 570, sidebarCollapsed: true })).status, 200);
  assert.match(await (await get('style.css')).text(), /sidebar-resize/);
  assert.match(await (await get('app.js')).text(), /paper-overview/);

  // A fresh copied-CLI process observes durable state, independent of browser ports.
  const fresh = spawn(process.execPath, [binary, 'ui', '--no-open'], { cwd: root, env, stdio: ['ignore', 'pipe', 'pipe'] });
  const freshExit = new Promise(resolve => fresh.once('exit', resolve));
  t.after(async () => { if (fresh.exitCode === null && fresh.signalCode === null) { fresh.kill('SIGTERM'); await freshExit; } });
  const freshUrl = await new Promise((resolve, reject) => {
    let text = '';
    const timer = setTimeout(() => reject(new Error('Fresh workbench failed to start')), 10000);
    fresh.stdout.on('data', chunk => { text += chunk; const found = /Workbench: (http:\/\/127\.0\.0\.1:\d+\/[a-f0-9]+\/)/.exec(text); if (found) { clearTimeout(timer); resolve(found[1]); } });
    fresh.once('error', error => { clearTimeout(timer); reject(error); });
  });
  const reread = await (await fetch(new URL(`api/paper?key=${encodeURIComponent(indexOnly.key)}`, freshUrl))).json();
  assert.equal(reread.paper.status, 'to_read'); assert.equal(reread.paper.starred, true);
  assert.deepEqual(await (await fetch(new URL('api/preferences', freshUrl))).json(), { sidebarWidth: 570, sidebarCollapsed: true });
  fresh.kill('SIGTERM'); await freshExit;

  await fs.appendFile(configPath, '\n# changed configuration\n');
  assert.equal((await post('api/paper/mark', { key: indexOnly.key, action: 'status', value: 'read', expected: 'to_read' })).status, 409);
  await fetch(new URL('api/runs', url), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ kind: 'paper', id: '2605.09999' }) });
  for (let i = 0; i < 50; i++) {
    run = (await (await get('api/runs/current')).json()).run;
    if (run.status !== 'running') break;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  assert.equal(run.status, 'failed');
  assert.match(run.output, /配置.*改变|configuration.*changed/i);
  child.kill('SIGTERM');
  const exit = await exited;
  assert.equal(exit.code, 0, output);
});
