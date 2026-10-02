import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { runInNewContext } from 'node:vm';
import { Window } from 'happy-dom';
import { installClient } from '../src/client.mjs';
import React from 'react';
import { createRoot } from 'react-dom/client';
test('native button renders busy/error/retry and opens the workbench through React DOM', async t => {
  const window = new Window({ url: 'http://127.0.0.1:3080' });
  const globals = ['window', 'document', 'navigator', 'IS_REACT_ACT_ENVIRONMENT'];
  const previous = globals.map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]);
  for (const [key, value] of Object.entries({ window, document: window.document, navigator: window.navigator, IS_REACT_ACT_ENVIRONMENT: true })) Object.defineProperty(globalThis, key, { value, configurable: true });
  const restoreGlobals = () => { window.happyDOM.abort(); for (const [name, descriptor] of previous) { if (descriptor) Object.defineProperty(globalThis, name, descriptor); else delete globalThis[name]; } };
  let component, finish; const cleanup = [], opened = [];
  const ctx = { effect(fn) { cleanup.push(fn()); }, locale: { register(_ns, dictionaries) { this.dict = dictionaries.zh; return () => {}; }, bind() { return key => this.dict[key]; } },
    slots: { inject(_name, fn) { return fn(); }, register(_meta, body) { component = body; return () => {}; } },
    connection: { isLoopback: true, rpc: { call() { return new Promise(resolve => { finish = resolve; }); } } }, sidebarRight: { openTab(...args) { opened.push(args); } } };
  installClient(ctx, React, window.location);
  const element = window.document.createElement('div'); window.document.body.append(element); const root = createRoot(element);
  t.after(async () => { try { await React.act(async () => root.unmount()); cleanup.reverse().forEach(fn => fn()); } finally { restoreGlobals(); } });
  await React.act(async () => root.render(React.createElement(React.StrictMode, null, React.createElement(component))));
  assert.match(element.textContent, /文献/);
  await React.act(async () => element.querySelector('button').click()); assert.equal(element.querySelector('button').disabled, true);
  await React.act(async () => finish({ ok: false, error: { message: '请先完成设置' } }));
  assert.match(element.querySelector('[role=alert]').textContent, /设置/); assert.equal(element.querySelector('button').disabled, false);
  await React.act(async () => element.querySelector('button').click());
  const url = 'http://127.0.0.1:8123/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/';
  await React.act(async () => finish({ ok: true, value: { url } })); assert.deepEqual(opened, [['browser', { params: { url } }]]);
});
test('the distributed client uses the DSH module loader and declares the services it reads', async () => {
  let registered;
  runInNewContext(await readFile(new URL('../dist/package/lib/client.js', import.meta.url), 'utf8'), { window: { __ModuleLoader__: { load(value) { registered = value; } } }, URL });
  assert.equal(registered.id, 'dsh-arxiv-daily');
  const module = registered.factory(name => { assert.equal(name, 'react'); return {}; });
  assert.deepEqual(Array.from(module.inject), ['slots', 'locale', 'connection', 'sidebarRight']);
});
