import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { installedDsh } from './dsh-environment.mjs';
const execute = promisify(execFile);
const dsh = installedDsh();
const project = resolve(import.meta.dirname, '../../..');
for (const firstRun of [false, true]) test(`${firstRun ? 'first-run' : 'configured'}: packed plugin installs into actual DSH and serves authenticated research workflows without agent turns`, { timeout: 60000, skip: !dsh && 'Install DSH or set DSH_CLI for integration acceptance' }, async t => {
  const root = await mkdtemp(join(tmpdir(), 'arxiv-dsh-integration-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const manifest = JSON.parse(await readFile(resolve(project, 'extensions/dsh-arxiv-daily/dist/package/package.json'), 'utf8'));
  assert.ok(manifest.os?.includes(process.platform), 'local artifact declares its native storage platform');
  assert.ok(manifest.cpu?.includes(process.arch));
  const configHome = join(root, 'config'), vault = join(root, 'vault');
  await mkdir(join(configHome, 'arxiv-daily'), { recursive: true });
  await mkdir(join(vault, 'arxiv-daily/papers'), { recursive: true });
  await writeFile(join(vault, 'arxiv-daily/papers/manual.md'), '# Standalone Markdown\n\n$x^2$\n');
  if (!firstRun) await writeFile(join(configHome, 'arxiv-daily/config.toml'), `vault_root = ${JSON.stringify(vault)}\ncache_dir = ${JSON.stringify(join(root, 'cache'))}\n[llm]\nprovider = "openai"\napi_key = "fixture-secret-never-log"\nbase_url = "https://fixture.invalid/v1"\nmodel = "fixture-model"\nthinking_mode = true\nreasoning_effort = "vendor-effort"\n[arxiv]\ncategories = ["astro-ph"]\ntimezone = "UTC"\n[[arxiv.topics]]\nname = "Photometric redshifts"\ntag = "photo-z"\ndescription = "Photometric redshift estimation and calibration"\ndetail = true\n[output]\nsummary_language = "en"\nlink_style = "relative"\n[advanced]\nlog_level = "error"\n`);
  const env = { ...process.env, DSH_HOME: join(root, 'dsh'), XDG_CONFIG_HOME: configHome, APPDATA: configHome };
  const packed = JSON.parse((await execute('npm', ['pack', './dist/package', '--pack-destination', root, '--json'], { cwd: resolve(project, 'extensions/dsh-arxiv-daily'), env })).stdout)[0];
  assert.equal(packed.files.some(file => /(^|\/)src\//.test(file.path)), false);
  assert.equal(packed.files.some(file => file.path === 'lib/arxiv-daily-cli.cjs'), true);
  const installed = await execute(process.execPath, [dsh.cli, 'plugin', '--profile', 'web', 'add', join(root, packed.filename)], { env, cwd: root, timeout: 20000 });
  assert.match(installed.stdout, /dsh-arxiv-daily/);
  const profile = JSON.parse(await readFile(join(env.DSH_HOME, 'profiles/web/package.json'), 'utf8'));
  assert.ok(profile.dsh.profile.bundles.includes('dsh-arxiv-daily'));
  const log = join(root, 'requests.jsonl'); await writeFile(log, '');
  const child = spawn(process.execPath, [dsh.cli, 'web', '--no-open', '--port', '0'], { cwd: root, env: { ...env, NODE_OPTIONS: `--require=${JSON.stringify(resolve(project, 'extensions/claude-code-arxiv-daily/tests/fixtures/core-http.cjs'))}`, ARXIV_CORE_FIXTURE_SCENARIO: 'selected', ARXIV_CORE_FIXTURE_LOG: log }, stdio: ['ignore', 'pipe', 'pipe'] });
  const exited = new Promise(resolve => child.once('close', resolve));
  t.after(async () => { if (child.exitCode === null && child.signalCode === null) { child.kill('SIGTERM'); const timer = setTimeout(() => child.kill('SIGKILL'), 5000); await exited; clearTimeout(timer); } });
  let output = ''; child.stderr.on('data', chunk => { output = (output + chunk).slice(-10000); });
  const login = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`DSH did not start: ${output}`)), 15000);
    child.stdout.on('data', chunk => { output += chunk; const match = /dsh web: (http:\/\/127\.0\.0\.1:\d+\/\?token=[\w-]+)/.exec(output); if (match) { clearTimeout(timer); resolve(match[1]); } });
    child.once('error', error => { clearTimeout(timer); reject(error); });
    child.once('close', () => { clearTimeout(timer); reject(new Error('DSH exited before readiness')); });
  });
  const origin = new URL(login).origin;
  const exchange = await fetch(login, { redirect: 'manual' }); assert.equal(exchange.status, 303);
  const cookie = exchange.headers.getSetCookie().map(value => value.split(';')[0]).join('; ');
  const rootPage = await fetch(origin, { headers: { Cookie: cookie } });
  assert.match(await rootPage.text(), /dsh-arxiv-daily/);
  const gatewayResponse = await fetch(origin + '/api/pluginManager/listPlugins', { method: 'POST', headers: { 'Content-Type': 'application/json', Cookie: cookie, Origin: origin }, body: JSON.stringify({ type: 'client-request', rpcId: 'gateway-check', method: 'pluginManager/listPlugins', payload: { args: {} } }) });
  assert.equal(gatewayResponse.status, 200);
  const gateway = await gatewayResponse.json(); assert.equal(gateway.result.ok, true);
  for (const moduleName of ['dsh-arxiv-daily', '@deepseek-ai/dsh-api-gateway']) assert.equal(gateway.result.value.find(plugin => plugin.moduleName === moduleName)?.fiberPhase, 'active', moduleName + ': ' + output);
  const request = { method: 'POST', headers: { 'Content-Type': 'application/json', Cookie: cookie, Origin: origin }, body: JSON.stringify({ type: 'client-request', rpcId: 'arxiv-test', method: 'arxiv-daily/open', payload: { frameOrigin: origin } }) };
  assert.equal((await fetch(origin + '/api/arxiv-daily/open', { ...request, headers: { 'Content-Type': 'application/json' } })).status, 401);
  assert.equal((await fetch(origin + '/api/arxiv-daily/open', { ...request, headers: { ...request.headers, Origin: 'https://example.com' } })).status, 403);
  const [first, second] = await Promise.all([fetch(origin + '/api/arxiv-daily/open', request).then(r => r.json()), fetch(origin + '/api/arxiv-daily/open', request).then(r => r.json())]);
  assert.equal(first.result.ok, true, JSON.stringify(first)); assert.equal(first.result.value.url, second.result.value.url);
  const url = first.result.value.url;
  const get = async route => { const response = await fetch(new URL(route, url)); assert.equal(response.status, 200); return response; };
  assert.ok((await get('')).headers.get('content-security-policy').includes(`frame-ancestors ${origin}`));
  if (firstRun) {
    assert.equal((await (await get('api/status')).json()).setupRequired, true);
    const initial = await (await get('api/settings')).json();
    const response = await fetch(new URL('api/settings', url), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ revision: null, values: { ...initial.values, vaultRoot: vault, provider: 'openai', apiKey: 'fixture-secret-never-log', baseUrl: 'https://fixture.invalid/v1', model: 'fixture-model', categories: ['astro-ph'], timezone: 'UTC', summaryLanguage: 'en', topics: [{ id: 'photo-z', name: 'Photometric redshifts', tag: 'photo-z', description: 'Photometric redshift estimation and calibration', detail: true }] } }) });
    assert.equal(response.status, 200, await response.text());
    assert.equal((await (await get('api/settings')).json()).setupRequired, false);
  }
  // Exercise the packed host's real persistence boundary in both configured and first-run cases.
  const beforeSettings = await (await get('api/settings')).json();
  if (!firstRun) assert.equal(beforeSettings.values.reasoningEffort, 'vendor-effort', 'legacy custom reasoning survives the shared projection');
  const extendedValues = {
    ...beforeSettings.values,
    apiKey: 'fixture-settings-secret-never-log',
    reasoningEffort: 'high', detailProfile: 'conservative', linkStyle: 'relative', summaryLanguage: 'en',
    schedule: { enabled: false, tickIntervalMin: 7, runAtLocal: '10:15', runUntilLocal: '17:45' },
    embedding: { mode: 'local', baseUrl: '', model: '', dimension: 384, apiKey: 'fixture-embedding-secret-never-log' },
    pdfParserSidecar: { enabled: false, capabilitiesUrl: 'http://127.0.0.1:5001/v1/capabilities', parseUrl: 'http://127.0.0.1:5001/v1/parse' },
    email: { enabled: false, mode: 'self', to: 'reader@example.test', fromEmail: 'papers@example.test', fromName: 'Research reports', apiKey: 'fixture-email-secret-never-log', hostedToken: 'fixture-hosted-secret-never-log' },
    logLevel: 'warn',
  };
  const updateSettings = await fetch(new URL('api/settings', url), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ revision: beforeSettings.revision, values: extendedValues }) });
  const updatedText = await updateSettings.text();
  assert.equal(updateSettings.status, 200, updatedText);
  const afterSettingsResponse = await get('api/settings');
  const afterSettingsText = await afterSettingsResponse.text();
  const afterSettings = JSON.parse(afterSettingsText);
  for (const key of ['reasoningEffort', 'detailProfile', 'linkStyle', 'summaryLanguage', 'schedule', 'pdfParserSidecar', 'logLevel']) {
    assert.deepEqual(afterSettings.values[key], extendedValues[key], `${key} must round-trip through the installed DSH host`);
  }
  assert.deepEqual(afterSettings.values.embedding, { mode: 'local', baseUrl: '', model: '', dimension: 384, apiKeyConfigured: true });
  assert.deepEqual(afterSettings.values.email, { enabled: false, mode: 'self', to: 'reader@example.test', fromEmail: 'papers@example.test', fromName: 'Research reports', apiKeyConfigured: true, hostedTokenConfigured: true });
  assert.equal(afterSettings.values.apiKeyConfigured, true);
  assert.notEqual(afterSettings.revision, beforeSettings.revision);
  assert.doesNotMatch(updatedText + afterSettingsText, /fixture-(?:settings|embedding|email|hosted)-secret-never-log/);
  const revealResponse = await fetch(new URL('api/settings/secret',url), {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision:afterSettings.revision,field:'apiKey'})});
  assert.equal(revealResponse.status,200);
  assert.deepEqual(await revealResponse.json(),{value:'fixture-settings-secret-never-log'});
  assert.equal(revealResponse.headers.get('cache-control'),'no-store');

  const librarySettings = await (await get('api/settings/library')).json();
  assert.equal(librarySettings.status.kind, 'disconnected');
  assert.match((await (await get('api/document?path=arxiv-daily/papers/manual.md')).json()).html, /katex/);
  assert.equal(await readFile(log, 'utf8'), '', 'opening and reading makes no provider requests');
  const post = (route, body) => fetch(new URL(route, url), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  assert.equal((await post('api/runs', { kind: 'daily', date: '2026-05-11' })).status, 202);
  let run;
  for (let i = 0; i < 250; i++) { run = (await (await get('api/runs/current')).json()).run; if (run.status !== 'running') break; await new Promise(resolve => setTimeout(resolve, 50)); }
  assert.equal(run.status, 'completed', run.output);
  const list = await (await get('api/papers?date=2026-05-11')).json(); assert.equal(list.total, 2);
  const paper = list.papers.find(p => !p.detailPath); assert.ok(paper, 'index-only discoveries remain available');
  assert.equal((await post('api/paper/mark', { key: paper.key, action: 'status', value: 'to_read', expected: 'inbox' })).status, 200);
  assert.equal((await post('api/paper/mark', { key: paper.key, action: 'star', value: true, expected: 'normal' })).status, 200);
  const actual = (await (await get(`api/paper?key=${encodeURIComponent(paper.key)}`)).json()).paper;
  assert.equal(actual.status, 'to_read'); assert.equal(actual.starred, true);
  assert.equal((await post('api/preferences', { appearance: { theme: 'dark', language: 'en' } })).status, 200);
  assert.equal((await post('api/preferences', { sidebarWidth: 530, sidebarCollapsed: false })).status, 200);
  assert.deepEqual((await (await get('api/preferences')).json()).appearance, { theme: 'dark', language: 'en' });
  assert.equal((await (await get('api/preferences')).json()).sidebarWidth, 530);
  const gatewayCall = async (method, args) => {
    const response = await fetch(origin + '/api/' + method, { ...request, body: JSON.stringify({ type: 'client-request', rpcId: 'host-action', method, payload: { args } }) });
    assert.equal(response.status, 200); const envelope = await response.json(); assert.equal(envelope.result.ok, true, JSON.stringify(envelope)); return envelope.result.value;
  };
  const id = gateway.result.value.find(plugin => plugin.moduleName === 'dsh-arxiv-daily').entryId;
  await gatewayCall('pluginManager/setPluginEnabled', { id, enabled: false });
  await assert.rejects(fetch(url), 'disabling the plugin stops its workbench');
  assert.equal((await gatewayCall('pluginManager/listPlugins', {})).find(plugin => plugin.moduleName === '@deepseek-ai/dsh-api-gateway').fiberPhase, 'active');
  await gatewayCall('pluginManager/setPluginEnabled', { id, enabled: true });
  const reopened = await (await fetch(origin + '/api/arxiv-daily/open', request)).json();
  assert.equal(reopened.result.ok, true); assert.notEqual(reopened.result.value.url, url);
  const saved = await (await fetch(new URL(`api/paper?key=${encodeURIComponent(paper.key)}`, reopened.result.value.url))).json();
  assert.equal(saved.paper.status, 'to_read'); assert.equal(saved.paper.starred, true);
  const restoredPreferences = await (await fetch(new URL('api/preferences', reopened.result.value.url))).json();
  assert.deepEqual(restoredPreferences.appearance, { theme: 'dark', language: 'en' });
  assert.doesNotMatch(output, /fixture(?:-(?:settings|embedding|email|hosted))?-secret-never-log|did not activate|already has an interceptor/);
  child.kill('SIGTERM'); await exited;
  await assert.rejects(fetch(reopened.result.value.url), 'DSH shutdown must close its workbench listener');
});
