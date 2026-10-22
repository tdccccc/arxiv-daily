import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { existsSync, readFileSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { WorkbenchProcess } from '../src/workbench-process.mjs';
const cliPath = fileURLToPath(new URL('./fixtures/child.cjs', import.meta.url));
function fixture(t, mode = 'ready') {
  const calls = [];
  const manager = new WorkbenchProcess({ cliPath, startupTimeoutMs: 1000, stopTimeoutMs: 150, env: { ...process.env, DSH_TEST_CHILD: mode }, spawn: (file, args, options) => {
    const child = spawn(file, args, options); calls.push({ file, args, options, child }); return child;
  } });
  t.after(() => manager.dispose()); return { manager, calls };
}
test('starts lazily once across concurrent opens and stops its owned child on dispose', async t => {
  const { manager, calls } = fixture(t); assert.equal(calls.length, 0);
  const [a, b] = await Promise.all([manager.open('http://127.0.0.1:3080'), manager.open('http://127.0.0.1:3080')]);
  assert.equal(a, b); assert.match(a, /^http:\/\/127\.0\.0\.1:8123\/[a-f0-9]{48}\/$/); assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].args, [cliPath, 'ui', '--no-open', '--exit-on-stdin-close', '--frame-origin', 'http://127.0.0.1:3080']);
  assert.equal(calls[0].options.shell, false);
  // The workbench stops itself when this pipe closes, which is what a killed host does to it.
  assert.equal(calls[0].options.stdio[0], 'pipe'); assert.equal(calls[0].child.stdin.writable, true);
  await assert.rejects(manager.open('http://localhost:3080'), /origin|地址/);
  await manager.dispose(); assert.ok(calls[0].child.exitCode !== null || calls[0].child.signalCode !== null);
  await assert.rejects(manager.open(), /stopped|卸载/);
});
test('reports missing setup without forwarding raw stderr or secrets', async t => {
  const { manager } = fixture(t, 'missing');
  await assert.rejects(manager.open(), error => error.code === 'setup-required' && !error.message.includes('fixture-secret'));
});
test('bounds startup time and can be disposed during startup', async t => {
  const { manager, calls } = fixture(t, 'silent');
  await assert.rejects(manager.open(), error => error.code === 'start-timeout');
  await manager.dispose(); assert.ok(calls[0].child.exitCode !== null || calls[0].child.signalCode !== null);
  const next = fixture(t, 'silent'); const pending = next.manager.open();
  const rejected = assert.rejects(pending, /stopped|卸载/); await next.manager.dispose(); await rejected;
});
test('restarts after an unexpected exit rather than returning an obsolete capability', async t => {
  const { manager, calls } = fixture(t); await manager.open();
  calls[0].child.kill('SIGTERM'); await once(calls[0].child, 'close');
  await manager.open(); assert.equal(calls.length, 2);
});

function running(pid) {
  try { process.kill(pid, 0); } catch { return false; }
  // A killed host's child is reaped by init; where nothing reaps it a finished child still answers signal 0.
  if (process.platform === 'linux') { try { return !/^\d+ \(.*\) Z /.test(readFileSync(`/proc/${pid}/stat`, 'utf8')); } catch { return false; } }
  return true;
}
async function reachable(url) { try { await fetch(url, { signal: AbortSignal.timeout(1000) }); return true; } catch { return false; } }
async function until(condition, ms = 15000) {
  for (const deadline = Date.now() + ms; Date.now() < deadline; await sleep(100)) if (!await condition()) continue; else return true;
  return false;
}
// The packed CLI is what DSH runs; the plugin build (node extensions/dsh-arxiv-daily/build.mjs) places it here.
const packedCli = fileURLToPath(new URL('../dist/package/lib/arxiv-daily-cli.cjs', import.meta.url));
test('the real workbench stops by itself when its DSH host is killed instead of disposed', { skip: !existsSync(packedCli) && 'build the plugin first: node extensions/dsh-arxiv-daily/build.mjs' }, async t => {
  const config = await mkdtemp(join(tmpdir(), 'dsh-killed-host-'));
  const host = spawn(process.execPath, [fileURLToPath(new URL('./fixtures/host.mjs', import.meta.url)), packedCli], { env: { ...process.env, XDG_CONFIG_HOME: config, APPDATA: config, HOME: config }, stdio: ['ignore', 'pipe', 'inherit'] });
  let child;
  t.after(async () => {
    host.kill('SIGKILL');
    // A failing test must not leave a workbench running behind it.
    if (child && running(child.pid)) { try { process.kill(process.platform === 'win32' ? child.pid : -child.pid, 'SIGKILL'); } catch { /* already gone */ } }
    await rm(config, { recursive: true, force: true });
  });
  let report = '';
  host.stdout.setEncoding('utf8');
  const reported = new Promise((resolve, reject) => {
    host.stdout.on('data', chunk => { report += chunk; if (report.includes('\n')) resolve(); });
    host.once('close', () => reject(new Error('the host exited before it reported its workbench')));
  });
  await Promise.race([reported, sleep(20000).then(() => { throw new Error('the host did not report its workbench'); })]);
  child = JSON.parse(report.split('\n')[0]);
  assert.equal(running(child.pid), true); assert.equal(await reachable(child.url), true);
  await sleep(500);
  assert.equal(running(child.pid), true, 'the workbench stays up while its host lives');

  host.kill('SIGKILL'); await once(host, 'close');
  assert.equal(await until(() => !running(child.pid)), true, 'the workbench process exits after its host is killed');
  assert.equal(await reachable(child.url), false);
});
