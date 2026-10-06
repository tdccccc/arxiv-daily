const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { spawnSync } = require('node:child_process');

const binary = path.resolve(__dirname, '../dist/arxiv-daily-cli.cjs');
const preload = path.join(__dirname, 'fixtures/core-http.cjs');
const date = '2026-05-11';

async function fixture(t, scenario = 'selected') {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'arxiv-core-workflow-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const configRoot = path.join(root, 'configuration with spaces');
  const vault = path.join(root, 'research with spaces');
  const configDir = path.join(configRoot, 'arxiv-daily');
  const requestLog = path.join(root, 'requests.jsonl');
  await fs.mkdir(configDir, { recursive: true });
  await fs.writeFile(requestLog, '');
  // This deliberately has no personal-library path or confirmed profile.
  await fs.writeFile(path.join(configDir, 'config.toml'), `
schema_version = 1
vault_root = ${JSON.stringify(vault)}
cache_dir = ${JSON.stringify(path.join(root, 'cache'))}
[llm]
provider = "openai"
api_key = "fixture-secret-never-log"
base_url = "https://fixture.invalid/v1"
model = "fixture-model"
thinking_mode = false
[arxiv]
categories = ["astro-ph"]
timezone = "UTC"
[[arxiv.topics]]
name = "Photometric redshifts"
tag = "photo-z"
description = "Photometric redshift estimation and calibration"
detail = true
[output]
daily_dir = "arxiv-daily/daily"
papers_dir = "arxiv-daily/papers"
summary_language = "en"
link_style = "relative"
[advanced]
log_level = "error"
`);
  const env = {
    ...process.env,
    XDG_CONFIG_HOME: configRoot,
    APPDATA: configRoot,
    ARXIV_CORE_FIXTURE_SCENARIO: scenario,
    ARXIV_CORE_FIXTURE_LOG: requestLog,
  };
  delete env.NODE_OPTIONS;
  function run(argv, overrides = {}) {
    const result = spawnSync(process.execPath, ['--require', preload, binary, ...argv], {
      cwd: root, env: { ...env, ...overrides }, encoding: 'utf8', timeout: 45_000,
    });
    assert.ifError(result.error);
    assert.equal(result.signal, null, result.stderr);
    assert.doesNotMatch(result.stdout + result.stderr, /fixture-secret-never-log/);
    return result;
  }
  const read = relative => fs.readFile(path.join(vault, relative), 'utf8');
  const json = async relative => JSON.parse(await read(relative));
  const requests = async () => (await fs.readFile(requestLog, 'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
  return { root, vault, run, read, json, requests };
}

function succeeded(result) {
  assert.equal(result.status, 0, result.stderr || result.stdout);
}

test('standalone product filters, writes daily and automatic/manual paper notes, and resumes across processes', async t => {
  const f = await fixture(t);
  const daily = f.run(['run', '--date', date]);
  succeeded(daily);
  assert.match(daily.stdout, /completed \(2 papers written\)/);
  const report = await f.read(`arxiv-daily/daily/${date}.md`);
  assert.match(report, /Fixture calibrated redshifts/);
  assert.match(report, /Fixture uncertainty estimates/);
  assert.doesNotMatch(report, /Fixture unrelated quantum paper/);
  assert.match(report, /Structured problem for 2605\.08080/);
  assert.match(report, /Structured result for 2605\.08068/);
  assert.match(report, /\]\(\.\.\/papers\/2605\.08080\.md\)/);
  assert.doesNotMatch(report, /\]\(\.\.\/papers\/2605\.08068\.md\)/);
  const automaticNote = await f.read('arxiv-daily/papers/2605.08080.md');
  assert.match(automaticNote, /## Method Design/);
  assert.match(automaticNote, /Fixture full-text assessment/);
  await assert.rejects(f.read('arxiv-daily/papers/2605.08068.md'), { code: 'ENOENT' });
  let index = await f.json('arxiv-daily/.index/papers.json');
  assert.equal(index.papers['arxiv:2605.08080'].paperPath, 'arxiv-daily/papers/2605.08080.md');
  assert.equal(index.papers['arxiv:2605.08080'].detail, true);
  assert.equal(index.papers['arxiv:2605.08068'].detail, false);
  assert.equal(index.papers['arxiv:2605.08080'].summary.coreProblem, 'Structured problem for 2605.08080');
  assert.ok(index.papers['arxiv:2605.08080'].seenDates.includes(date));
  assert.ok(index.papers['arxiv:2605.08080'].dailyReports.includes(`arxiv-daily/daily/${date}.md`));
  const state = await f.json('arxiv-daily/.index/run-state.json');
  assert.equal(state.runState[date].status, 'completed');
  assert.equal(state.runState[date].papersWritten, 2);
  assert.match(await f.read('arxiv-daily/.index/run-history.jsonl'), /completed/);
  const firstRequests = await f.requests();
  assert.equal(firstRequests.filter(r => r.kind === 'filter').length, 1);
  assert.equal(firstRequests.filter(r => r.kind === 'selector').length, 1);
  assert.equal(firstRequests.filter(r => r.kind === 'daily-summary').length, 2);
  assert.equal(firstRequests.filter(r => r.kind === 'paper-note').length, 1);

  const manual = f.run(['run', '--id', '2605.09999v2', '--date', date]);
  succeeded(manual);
  assert.match(manual.stdout, /wrote arxiv-daily\/papers\/2605\.09999\.md/);
  const manualNote = await f.read('arxiv-daily/papers/2605.09999.md');
  assert.match(manualNote, /Fixture manual follow-up/);
  assert.match(manualNote, /Fixture full-text assessment/);
  index = await f.json('arxiv-daily/.index/papers.json');
  assert.equal(index.papers['arxiv:2605.09999'].paperPath, 'arxiv-daily/papers/2605.09999.md');
  assert.equal(index.papers['arxiv:2605.09999'].detail, true);
  const requestCount = (await f.requests()).length;
  const offline = { ARXIV_CORE_FIXTURE_SCENARIO: 'offline' };
  const repeatDaily = f.run(['run', '--date', date], offline);
  succeeded(repeatDaily);
  assert.match(repeatDaily.stdout, /skipped \(already done\)/);
  const repeatManual = f.run(['run', '--id', '2605.09999'], offline);
  succeeded(repeatManual);
  assert.match(repeatManual.stdout, /already exists/);
  assert.equal((await f.requests()).length, requestCount, 'completed runs must avoid network and model work');
  assert.equal(await f.read(`arxiv-daily/daily/${date}.md`), report);
  assert.equal(await f.read('arxiv-daily/papers/2605.09999.md'), manualNote);
  await assert.rejects(fs.access(path.join(f.vault, 'arxiv-daily-agent')), { code: 'ENOENT' });
});

test('zero matches complete durably without writing an empty report or repeating paid filtering', async t => {
  const f = await fixture(t, 'zero');
  const result = f.run(['run', '--date', date]);
  succeeded(result);
  assert.match(result.stdout, /completed \(0 papers written\)/);
  await assert.rejects(f.read(`arxiv-daily/daily/${date}.md`), { code: 'ENOENT' });
  const state = await f.json('arxiv-daily/.index/run-state.json');
  assert.equal(state.runState[date].status, 'completed');
  assert.equal(state.runState[date].papersWritten, 0);
  const requests = await f.requests();
  assert.equal(requests.filter(r => r.kind === 'filter').length, 1);
  assert.equal(requests.filter(r => ['selector', 'daily-summary', 'paper-note'].includes(r.kind)).length, 0);
  succeeded(f.run(['run', '--date', date], { ARXIV_CORE_FIXTURE_SCENARIO: 'offline' }));
  assert.equal((await f.requests()).length, requests.length);
});

test('an unpublished announce date remains retryable without an empty report or LLM work', async t => {
  const f = await fixture(t);
  const unpublishedDate = '2026-05-12';
  const result = f.run(['run', '--date', unpublishedDate]);
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.match(result.stdout, /awaiting_announcement/);
  assert.match(result.stdout, /newer than newest/);
  await assert.rejects(f.read(`arxiv-daily/daily/${unpublishedDate}.md`), { code: 'ENOENT' });
  const state = await f.json('arxiv-daily/.index/run-state.json');
  assert.equal(state.runState[unpublishedDate].status, 'pending');
  assert.equal(state.runState[unpublishedDate].outcome, 'awaiting_announcement');
  assert.equal(state.runState[unpublishedDate].failureAttempts, 0);
  assert.deepEqual((await f.requests()).map(r => r.kind), ['recent']);
});

test('manual detail protects an existing user note before network or model work', async t => {
  const f = await fixture(t, 'offline');
  const relative = 'arxiv-daily/papers/2605.09999.md';
  const note = '---\narxiv_id: "2605.09999"\n---\n# My reading notes\nKeep this user-authored judgment unchanged.\n';
  await fs.mkdir(path.dirname(path.join(f.vault, relative)), { recursive: true });
  await fs.writeFile(path.join(f.vault, relative), note);
  const result = f.run(['run', '--id', '2605.09999', '--date', date]);
  assert.equal(result.status, 1, result.stdout + result.stderr);
  assert.match(result.stderr, /protected|user-authored|unverified/);
  assert.equal(await f.read(relative), note);
  assert.deepEqual(await f.requests(), []);
  await assert.rejects(f.read('arxiv-daily/.index/papers.json'), { code: 'ENOENT' });
});
