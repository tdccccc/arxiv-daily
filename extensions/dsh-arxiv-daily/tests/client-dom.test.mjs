import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { runInNewContext } from 'node:vm';
import { Window } from 'happy-dom';
import { installClient } from '../src/client.mjs';
import React from 'react';
import { createRoot } from 'react-dom/client';
const workbenchUrl = 'http://127.0.0.1:8123/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/';
function fixture(t) {
  const window = new Window({ url: 'http://127.0.0.1:3080', settings: { disableIframePageLoading: true } });
  const previous = ['window', 'document', 'navigator', 'IS_REACT_ACT_ENVIRONMENT'].map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]);
  for (const [key, value] of Object.entries({ window, document: window.document, navigator: window.navigator, IS_REACT_ACT_ENVIRONMENT: true })) Object.defineProperty(globalThis, key, { value, configurable: true });
  const slots = new Map(), definitions = [], cleanup = [], calls = []; let finish;
  const ctx = { effect(fn) { cleanup.push(fn()); }, locale: { register(_ns, dictionaries) { this.dict = dictionaries.zh; return () => {}; }, bind() { return key => this.dict[key]; } },
    slots: { inject(_name, fn) { return fn(); }, register(meta, body) { slots.set(meta.name, body); return () => {}; } },
    sidebarRightTabs: { register(value) { definitions.push(value); return () => {}; } },
    connection: { isLoopback: true, rpc: { call(...args) { calls.push(args); return new Promise(resolve => { finish = resolve; }); } } } };
  installClient(ctx, React, window.location);
  const element = window.document.createElement('div'); window.document.body.append(element); const root = createRoot(element);
  t.after(async () => { try { await React.act(async () => root.unmount()); cleanup.reverse().forEach(fn => fn()); } finally { window.happyDOM.abort(); for (const [name, descriptor] of previous) { if (descriptor) Object.defineProperty(globalThis, name, descriptor); else delete globalThis[name]; } } });
  return { root, element, window, slots, definitions, calls, finish: value => finish(value) };
}
test('footer opens the workbench before any conversation, supports retry and closes the overlay', async t => {
  const f = fixture(t); assert.ok(f.slots.has('sidebar.footer.action')); assert.ok(f.slots.has('shell.overlay'));
  await React.act(async () => f.root.render(React.createElement(React.StrictMode, null, React.createElement(f.slots.get('sidebar.footer.action'), { wide: true }), React.createElement(f.slots.get('shell.overlay')))));
  assert.match(f.element.textContent, /arxiv-daily/); assert.equal(f.calls.length, 0);
  await React.act(async () => f.element.querySelector('button').click());
  assert.ok(f.element.querySelector('[role=dialog]')); assert.match(f.element.textContent, /正在打开/);
  assert.equal(f.calls.length, 2, 'StrictMode retries mount safely without a conversation');
  await React.act(async () => f.finish({ ok: false, error: { message: '请先完成设置' } }));
  assert.match(f.element.querySelector('[role=alert]').textContent, /设置/);
  await React.act(async () => [...f.element.querySelectorAll('button')].find(button => button.textContent === '重试').click());
  await React.act(async () => f.finish({ ok: true, value: { url: workbenchUrl } }));
  assert.equal(f.element.querySelector('iframe').src, workbenchUrl);
  assert.equal(f.calls.at(-1)[2].frameOrigin, 'http://127.0.0.1:3080');
  await React.act(async () => f.element.querySelector('[aria-label="关闭 arxiv-daily"]').click());
  assert.equal(f.element.querySelector('[role=dialog]'), null);
});
test('right Sidebar guide names the shared reader tab and collapsed footer keeps an accessible icon', async t => {
  const f = fixture(t); assert.equal(f.definitions.length, 1); assert.equal(f.definitions[0].guide[0].title(), 'arxiv-daily');
  await React.act(async () => f.root.render(React.createElement(f.slots.get('sidebar.footer.action'), { wide: false })));
  assert.equal(f.element.querySelector('button').getAttribute('aria-label'), 'arxiv-daily');
  assert.ok(f.element.querySelector('svg')); assert.equal(f.element.textContent, '');
  await React.act(async () => f.root.render(React.createElement(f.slots.get('sidebar.right.pane.tab'))));
  await React.act(async () => f.finish({ ok: true, value: { url: workbenchUrl } }));
  assert.equal(f.element.querySelector('iframe').src, workbenchUrl);
});
test('distributed client declares the registry needed for fixed entries', async () => {
  let registered;
  runInNewContext(await readFile(new URL('../dist/package/lib/client.js', import.meta.url), 'utf8'), { window: { __ModuleLoader__: { load(value) { registered = value; } } }, URL });
  assert.equal(registered.id, 'dsh-arxiv-daily');
  const module = registered.factory(name => { assert.equal(name, 'react'); return {}; });
  assert.deepEqual(Array.from(module.inject), ['slots', 'locale', 'connection', 'sidebarRightTabs']);
});
