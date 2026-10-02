import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installHost } from '../src/host.mjs';
import { createOpener, installClient } from '../src/client.mjs';
const URL = 'http://127.0.0.1:8123/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/';
function host() {
  const calls = [], cleanup = []; let handler, matches;
  const manager = { async open(origin) { calls.push(origin); return URL; }, async dispose() { calls.push('disposed'); } };
  const ctx = { webServer: { port: 3080 }, effect(fn) { cleanup.push(fn()); }, connection: { rpc: { intercept(channel, match, handle) { assert.equal(channel, '/api'); matches = match; handler = handle; return () => calls.push('unregistered'); } } } };
  installHost(ctx, manager); return { calls, manager, cleanup, get handler() { return handler; }, get matches() { return matches; } };
}
test('registers only its authenticated endpoint, validates frame origin and disposes its process', async () => {
  const h = host(); assert.equal(h.matches?.('arxiv-daily/open'), true); assert.equal(h.matches('session/list'), false);
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
test('a click invokes host code and opens the exact browser tab without a model request', async () => {
  const { opener, calls } = client(); await opener.open();
  assert.deepEqual(calls[0].slice(0, 4), ['rpc', '/api', 'arxiv-daily/open', { frameOrigin: 'http://127.0.0.1:3080' }]);
  assert.deepEqual(calls[1], ['tab', 'browser', { params: { url: URL } }]);
  const desktop = client(undefined, { protocol: 'dsh-app:', origin: 'null' }); await desktop.opener.open(); assert.deepEqual(desktop.calls[0][3], {});
});
test('rejects unsafe destinations, remote clients and late navigation after disposal', async () => {
  const bad = client({ ok: true, value: { url: 'https://example.com' } }); await assert.rejects(bad.opener.open()); assert.equal(bad.calls.length, 1);
  const remote = client(); remote.ctx.connection.isLoopback = false; await assert.rejects(remote.opener.open()); assert.equal(remote.calls.length, 0);
  let resolve; const slow = client(() => new Promise(r => { resolve = r; }));
  const pending = slow.opener.open(); slow.opener.dispose(); resolve({ ok: true, value: { url: URL } }); await pending;
  assert.equal(slow.calls.filter(call => call[0] === 'tab').length, 0);
});
test('registers a native composer entry and releases its slots and dictionaries', () => {
  const slots = [], cleanup = [], removed = [];
  const ctx = { effect(fn) { cleanup.push(fn()); }, locale: { register() { return () => removed.push('locale'); }, bind() { return key => key; } }, slots: { inject(name, fn) { assert.equal(name, 'conversation.composer.dock'); return fn(); }, register(meta, component) { slots.push({ meta, component }); return () => removed.push('slot'); } } };
  installClient(ctx, { createElement() {} }, {});
  assert.equal(slots.length, 1); assert.equal(slots[0].meta.id, 'dsh-arxiv-daily'); assert.equal(typeof slots[0].component, 'function');
  cleanup.reverse().forEach(fn => fn()); assert.deepEqual(removed.sort(), ['locale', 'slot']);
});


test('capability validation handles malformed ports without throwing', async () => {
  const { isWorkbenchUrl } = await import('../src/protocol.mjs');
  assert.equal(isWorkbenchUrl('http://127.0.0.1:99999/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/'), false);
});
