const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { spawnSync } = require('node:child_process');

const plugin = path.resolve(__dirname, '..');
const binary = path.join(plugin, 'dist/arxiv-daily-cli.cjs');

test('plugin carries the exact product CLI build, including daily/detail commands', async () => {
  assert.deepEqual(await fs.readFile(binary), await fs.readFile(path.resolve(plugin, '../../apps/cli/dist/arxiv-daily-cli.cjs')));
  const result = spawnSync(process.execPath, [binary, 'help'], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /run --today/);
  assert.match(result.stdout, /run --id ARXIV_ID/);
  assert.match(result.stdout, /arxiv-daily status/);
  const manifest = JSON.parse(await fs.readFile(path.join(plugin, '.claude-plugin/plugin.json'), 'utf8'));
  assert.equal(manifest.name, 'arxiv-daily');
  const skill = await fs.readFile(path.join(plugin, 'skills/research/SKILL.md'), 'utf8');
  assert.ok(skill.includes('${CLAUDE_PLUGIN_ROOT}/dist/arxiv-daily-cli.cjs'));
  await assert.rejects(fs.access(path.join(plugin, 'dist/arxiv-agent.cjs')), { code: 'ENOENT' });
});

test('packaged CLI runs outside the repository and resolves product config without a library', async t => {
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'arxiv-product-package-'));
  t.after(() => fs.rm(temp, { recursive: true, force: true }));
  await fs.cp(path.join(plugin, 'dist/plugin'), path.join(temp, 'plugin'), { recursive: true });
  const target = path.join(temp, 'plugin/dist/arxiv-daily-cli.cjs');
  const env = { ...process.env, XDG_CONFIG_HOME: path.join(temp, 'config'), APPDATA: path.join(temp, 'config') };
  delete env.NODE_OPTIONS;
  const help = spawnSync(process.execPath, [target, 'help'], { cwd: temp, env, encoding: 'utf8' });
  assert.equal(help.status, 0, help.stderr);
  const status = spawnSync(process.execPath, [target, 'status'], { cwd: temp, env, encoding: 'utf8' });
  assert.equal(status.status, 2);
  assert.match(status.stderr, /CLI config not found/);
  assert.match(status.stderr, /arxiv-daily init/);
  assert.doesNotMatch(status.stderr, /library directory|Workspace is not connected/);
  const old = spawnSync(process.execPath, [target, 'save'], { cwd: temp, env, encoding: 'utf8' });
  assert.equal(old.status, 2);
  assert.match(old.stderr, /Unknown command/);
  await assert.rejects(fs.access(path.join(temp, 'arxiv-daily-agent')), { code: 'ENOENT' });
});
