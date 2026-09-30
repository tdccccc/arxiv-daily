const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { spawnSync } = require('node:child_process');

const plugin = path.resolve(__dirname, '..');
const binary = path.join(plugin, 'dist/arxiv-agent.cjs');

test('Claude plugin has a discoverable research skill and a bundled runnable command', async () => {
  const manifest = JSON.parse(await fs.readFile(path.join(plugin, '.claude-plugin/plugin.json'), 'utf8'));
  assert.equal(manifest.name, 'arxiv-daily');
  const skill = await fs.readFile(path.join(plugin, 'skills/research/SKILL.md'), 'utf8');
  assert.match(skill, /name: research/);
  assert.match(skill, /CLAUDE_PLUGIN_ROOT/);
  const help = spawnSync(process.execPath, [binary, 'help'], { encoding: 'utf8' });
  assert.equal(help.status, 0, help.stderr);
  assert.equal(JSON.parse(help.stdout).ok, true);
});

test('CLI saves and restores a research record across processes, with JSON-only output', async t => {
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'arxiv-agent-cli-'));
  t.after(() => fs.rm(temp, { recursive: true, force: true }));
  const library = path.join(temp, 'library with spaces');
  const workspace = path.join(temp, 'research with spaces');
  await fs.mkdir(library);
  await fs.writeFile(path.join(library, 'one.pdf'), '%PDF-1.4');
  function call(command, input = {}) {
    const result = spawnSync(process.execPath, [binary, command, '--workspace', workspace], { input: JSON.stringify(input), encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const parsed = JSON.parse(result.stdout);
    assert.equal(parsed.ok, true);
    return parsed.data;
  }
  call('init', { library });
  assert.equal(call('library').total, 1);
  const record = call('save', { kind: 'reading', slug: 'one', title: 'Read later', body: '# Reading decision\nRead the methods next.', sources: ['one.pdf'] });
  assert.match(record.path, /reading/);
  assert.equal(call('status').records.reading.length, 1);
  assert.match(call('read', { kind: 'reading', slug: 'one' }).markdown, /Read the methods next/);
  const malformed = spawnSync(process.execPath, [binary, 'save', '--workspace', workspace], { input: 'not json', encoding: 'utf8' });
  assert.notEqual(malformed.status, 0);
  assert.equal(JSON.parse(malformed.stdout).ok, false);
});

test('packaged plugin is relocatable outside the repository with no node_modules', async t => {
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'arxiv-plugin-package-'));
  t.after(() => fs.rm(temp, { recursive: true, force: true }));
  await fs.cp(path.join(plugin, 'dist/plugin'), temp, { recursive: true });
  const manifest = JSON.parse(await fs.readFile(path.join(temp, '.claude-plugin/plugin.json'), 'utf8'));
  assert.equal(manifest.name, 'arxiv-daily');
  const result = spawnSync(process.execPath, [path.join(temp, 'dist/arxiv-agent.cjs'), 'help'], { cwd: temp, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(JSON.parse(result.stdout).ok, true);
});
