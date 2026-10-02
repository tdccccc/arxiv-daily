import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installHost } from '../src/host.mjs';
import { createOpener, installClient } from '../src/client.mjs';
const URL = 'http://127.0.0.1:8123/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/';
function host() {
  const calls = [], cleanup = []; let route;
  const manager = { async open(origin) { calls.push(origin); return URL; }, async dispose() { calls.push('disposed'); } };
  const ctx = { webServer: { port: 3080 }, effect(fn) { cleanup.push(fn()); }, connection: { fetch: { register(value) { route = value; return () => calls.push('unregistered'); } } } };
  installHost(ctx, manager);
  const handler = async (method, payload, signal) => {
    const response = await route.fetch(new Request('http://127.0.0.1:3080' + route.path, { method: 'POST', signal, body: JSON.stringify({ type: 'client-request', rpcId: 'test', method, payload }) }));
    const result = await response.json(); return response.ok ? result.result : { ok: false };
  };
  return { calls, manager, cleanup, handler, route };
}
test('registers only its authenticated endpoint, validates frame origin and disposes its process', async () => {
  const h = host(); assert.equal(h.route.path, '/api/arxiv-daily/open'); assert.deepEqual(h.route.methods, ['POST']);
  assert.equal((await h.handler('session/list', {}, new AbortController().signal)).ok, false);
  assert.deepEqual(await h.handler('arxiv-daily/open', { frameOrigin: 'http://127.0.0.1:3080' }, new AbortController().signal), { ok: true, value: { url: URL } });
  for (const body of [{ frameOrigin: 'https://example.com' }, { frameOrigin: 'http://127.0.0.1:9999' }, { executable: 'bad' }, null]) {
    assert.equal((await h.handler('arxiv-daily/open', body, new AbortController().signal)).ok, false);
  }
  assert.equal(h.calls.length, 1); await h.cleanup[0](); assert.deepEqual(h.calls.slice(-2), ['unregistered', 'disposed']);
});
test('does not leak unexpected process errors through the RPC', async () => {
  const h = host(); h.manager.open = async () => { throw new Error('private-secret'); };
  const result = await h.handler?.('arxiv-daily/open', {}, new AbortController().signal);
  assert.equal(result?.ok, false); assert.doesNotMatch(JSON.stringify(result), /private-secret/);
});
function client(result = { ok: true, value: { url: URL } }, location = { protocol: 'http:', origin: 'http://127.0.0.1:3080' }) {
  const calls = [];
  const ctx = { connection: { isLoopback: true, rpc: { async call(...args) { calls.push(['rpc', ...args]); return typeof result === 'function' ? result() : result; } } }, sidebarRight: { openTab(...args) { calls.push(['tab', ...args]); } } };
  return { calls, ctx, opener: createOpener(ctx, location) };
}
test('resolves a workbench address without a conversation or model request', async () => {
  const { opener, calls } = client(); assert.equal(await opener.open(), URL);
  assert.deepEqual(calls[0].slice(0, 4), ['rpc', '/api', 'arxiv-daily/open', { frameOrigin: 'http://127.0.0.1:3080' }]);
  assert.equal(calls.length, 1, 'loading an address does not need a mounted conversation');
  const desktop = client(undefined, { protocol: 'dsh-app:', origin: 'null' }); await desktop.opener.open(); assert.deepEqual(desktop.calls[0][3], { frameOrigin: 'dsh-app://app' });
});
test('rejects unsafe destinations, remote clients and late navigation after disposal', async () => {
  const bad = client({ ok: true, value: { url: 'https://example.com' } }); await assert.rejects(bad.opener.open()); assert.equal(bad.calls.length, 1);
  const remote = client(); remote.ctx.connection.isLoopback = false; await assert.rejects(remote.opener.open()); assert.equal(remote.calls.length, 0);
  let resolve; const slow = client(() => new Promise(r => { resolve = r; }));
  const pending = slow.opener.open(); slow.opener.dispose(); resolve({ ok: true, value: { url: URL } }); await pending;
  assert.equal(slow.calls.filter(call => call[0] === 'tab').length, 0);
});
test('registers a global footer/overlay and right Sidebar guide without a composer entry', () => {
  const slots = [], definitions = [], cleanup = [], removed = [];
  const ctx = { effect(fn) { cleanup.push(fn()); }, locale: { register() { return () => removed.push('locale'); }, bind() { return key => key; } },
    sidebarRightTabs: { register(definition) { definitions.push(definition); return () => removed.push('tab'); } },
    slots: { inject(_name, fn) { return fn(); }, register(meta, component) { slots.push({ meta, component }); return () => removed.push(meta.name); } } };
  installClient(ctx, { createElement() {} }, {});
  assert.deepEqual(slots.map(slot => slot.meta.name).sort(), ['shell.overlay', 'sidebar.footer.action', 'sidebar.right.pane.tab', 'sidebar.right.pane.tab.title'].sort());
  assert.equal(definitions[0].kind, 'dsh-arxiv-daily'); assert.equal(definitions[0].guide[0].id, 'workbench');
  assert.equal(slots.find(slot => slot.meta.name === 'sidebar.right.pane.tab').meta.key, definitions[0].id);
  cleanup.reverse().forEach(fn => fn()); assert.equal(removed.length, 6);
});

test('capability validation handles malformed ports without throwing', async () => {
  const { isWorkbenchUrl } = await import('../src/protocol.mjs');
  assert.equal(isWorkbenchUrl('http://127.0.0.1:99999/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/'), false);
});


test('the exact route rejects invalid RPC envelopes before starting a process', async () => {
  const h = host();
  for (const body of ['not json', '{}', JSON.stringify({ type: 'client-request', rpcId: '', method: 'arxiv-daily/open', payload: {} }), JSON.stringify({ type: 'client-request', rpcId: 'test', method: 'session/list', payload: {} })]) {
    const response = await h.route.fetch(new Request('http://127.0.0.1:3080' + h.route.path, { method: 'POST', body }));
    assert.equal(response.status, 400);
  }
  assert.equal(h.calls.length, 0);
});

test('permits the fixed Desktop frame origin without permitting arbitrary schemes', async () => {
  const h = host();
  assert.equal((await h.handler('arxiv-daily/open', { frameOrigin: 'dsh-app://app' }, new AbortController().signal)).ok, true);
  assert.equal((await h.handler('arxiv-daily/open', { frameOrigin: 'dsh-app://evil' }, new AbortController().signal)).ok, false);
  assert.deepEqual(h.calls, ['dsh-app://app']);
});
