const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { executeAgentCommand: run } = require('../dist/agent.cjs');

async function fixture(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'arxiv-agent-test-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const library = path.join(root, 'library');
  const workspace = path.join(root, 'research');
  await fs.mkdir(library);
  await fs.writeFile(path.join(library, '2606.12345.pdf'), '%PDF-1.4 fixture');
  await fs.writeFile(path.join(library, 'unidentified.pdf'), '%PDF-1.4 other');
  await fs.writeFile(path.join(library, 'private-notes.md'), 'not opted in');
  return { root, library, workspace };
}

test('connect a read-only library, paginate it, and recover setup in a later call', async t => {
  const { library, workspace } = await fixture(t);
  const before = await fs.readdir(library);
  const connected = await run('init', workspace, { library });
  assert.equal(connected.library, await fs.realpath(library));
  const page = await run('library', workspace, { limit: 1 });
  assert.equal(page.total, 2);
  assert.equal(page.items.length, 1);
  assert.equal(page.nextOffset, 1);
  assert.equal(page.items[0].paperKey, 'arxiv:2606.12345');
  const tail = await run('library', workspace, { offset: 1, limit: 1 });
  assert.equal(tail.nextOffset, null);
  assert.equal((await run('status', workspace)).library, connected.library);
  assert.deepEqual(await fs.readdir(library), before);
});

test('do not switch a connected library silently or include symbolic links', async t => {
  const { library, workspace, root } = await fixture(t);
  const other = path.join(root, 'other');
  await fs.mkdir(other);
  await fs.symlink(path.join(library, '2606.12345.pdf'), path.join(library, 'linked.pdf'));
  await run('init', workspace, { library });
  await assert.rejects(run('init', workspace, { library: other }), /already connected/i);
  const page = await run('library', workspace);
  assert.equal(page.total, 2);
  assert.ok(page.ignored >= 1);
});

test('persist direction drafts, require a matching revision for confirmation and edits', async t => {
  const { library, workspace } = await fixture(t);
  await run('init', workspace, { library });
  const draft = await run('save', workspace, {
    kind: 'direction', slug: 'efficient-inference', title: 'Efficient inference',
    body: 'Representative: arxiv:2606.12345. Evidence: title only.',
    sources: ['arxiv:2606.12345'],
  });
  assert.equal(draft.status, 'draft');
  assert.equal((await run('status', workspace)).confirmedDirections.length, 0);
  await assert.rejects(run('confirm-direction', workspace, { slug: 'efficient-inference', expectedSha256: 'old' }), /changed|conflict/i);
  const confirmed = await run('confirm-direction', workspace, { slug: 'efficient-inference', expectedSha256: draft.sha256 });
  assert.equal(confirmed.status, 'confirmed');
  assert.equal((await run('status', workspace)).confirmedDirections.length, 1);
  const read = await run('read', workspace, { kind: 'direction', slug: 'efficient-inference' });
  assert.match(read.markdown, /Efficient inference/);
  assert.match(read.markdown, /2606.12345/);
  await assert.rejects(run('save', workspace, { kind: 'direction', slug: 'efficient-inference', title: 'New', body: 'changed', sources: [] }), /changed|conflict/i);
  const edited = await run('save', workspace, { kind: 'direction', slug: 'efficient-inference', title: 'New', body: 'changed', sources: [], expectedSha256: confirmed.sha256 });
  assert.equal(edited.status, 'draft', 'changed direction needs review again');
});

test('persist paper notes and reading judgments, reject traversal and stale updates', async t => {
  const { library, workspace } = await fixture(t);
  await run('init', workspace, { library });
  const input = { kind: 'paper', slug: '2606.12345', title: 'A paper', body: '## Evidence\nBased on the abstract only.', sources: ['arxiv:2606.12345'] };
  const saved = await run('save', workspace, input);
  const same = await run('save', workspace, input);
  assert.equal(same.sha256, saved.sha256);
  await run('save', workspace, { kind: 'reading', slug: '2606.12345', title: 'Read closely', body: 'User judgment: useful method; inspect the experiments.', sources: ['arxiv:2606.12345'] });
  const status = await run('status', workspace);
  assert.equal(status.records.paper.length, 1);
  assert.equal(status.records.reading.length, 1);
  await assert.rejects(run('save', workspace, { ...input, slug: '../escape' }), /slug|path/i);
  await assert.rejects(run('read', workspace, { kind: 'paper', slug: '../escape' }), /slug|path/i);
  const results = await Promise.allSettled([
    run('save', workspace, { ...input, body: 'edit A', expectedSha256: saved.sha256 }),
    run('save', workspace, { ...input, body: 'edit B', expectedSha256: saved.sha256 }),
  ]);
  assert.equal(results.filter(r => r.status === 'fulfilled').length, 1);
  assert.equal(results.filter(r => r.status === 'rejected').length, 1);
});

test('reject output under the source library and unsafe stored config', async t => {
  const { library, workspace } = await fixture(t);
  await assert.rejects(run('init', library, { library }), /overlap|inside|source/i);
  await run('init', workspace, { library });
  await fs.rename(library, `${library}-old`);
  await fs.symlink(`${library}-old`, library);
  await assert.rejects(run('library', workspace), /symbolic|root|changed/i);
});
