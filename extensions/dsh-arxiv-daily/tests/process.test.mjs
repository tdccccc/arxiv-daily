import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
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
  assert.deepEqual(calls[0].args, [cliPath, 'ui', '--no-open', '--frame-origin', 'http://127.0.0.1:3080']);
  assert.equal(calls[0].options.shell, false);
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
