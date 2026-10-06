import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { installedDsh } from './dsh-environment.mjs';
import { installClient } from '../src/client.mjs';
const dsh = installedDsh();
test('entry registrations are accepted by the installed DSH SlotCore and cleanly removed', { skip: !dsh && 'DSH required' }, async () => {
  const require = createRequire(dsh.cli);
  const { SlotCore } = await import(require.resolve('@deepseek-ai/dsh-client-ui-slots'));
  const core = new SlotCore(), cleanup = [];
  const specs = { 'sidebar.footer.action': { kind: 'list', scope: 'root' }, 'shell.overlay': { kind: 'list', scope: 'root' }, 'sidebar.right.pane.tab': { kind: 'keyed', scope: 'session' }, 'sidebar.right.pane.tab.title': { kind: 'keyed', scope: 'session' } };
  const removeRoot = core.register({ name: 'root', children: specs }, () => null);
  const ctx = { effect(fn) { cleanup.push(fn()); }, locale: { register() { return () => {}; }, bind() { return key => key; } }, sidebarRightTabs: { register() { return () => {}; } }, slots: { inject(name, fn) { assert.ok(core.spec(name)); return fn(); }, register: (options, component) => core.register(options, component) } };
  installClient(ctx, { createElement() {} }, {});
  for (const name of Object.keys(specs)) assert.equal(core.entries(name).length, 1, name);
  cleanup.reverse().forEach(fn => fn());
  for (const name of Object.keys(specs)) assert.equal(core.entries(name).length, 0, name);
  removeRoot();
});
