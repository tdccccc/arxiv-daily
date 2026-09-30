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

const recentHtml = `<html><dl id="articles"><h3>Wed, 30 Sep 2026</h3>
<dt><a title="Abstract">arXiv:2609.12345</a></dt><dd><div class="list-title">Title: Efficient inference</div><div class="list-authors"><a>A. Author</a></div></dd>
<dt><a title="Abstract">arXiv:2609.12346</a></dt><dd><div class="list-title">Title: Other work</div><div class="list-authors"><a>B. Author</a></div></dd>
</dl></html>`;

test('recent uses core listing parsing, exposes available dates, and labels missing abstracts', async t => {
  const { library, workspace } = await fixture(t);
  await run('init', workspace, { library });
  const requests = [];
  const http = { async request(req) { requests.push(req.url); return { status: 200, headers: {}, bodyText: recentHtml }; } };
  const page = await run('recent', workspace, { category: 'cs.AI', date: '2026-09-30', limit: 1 }, { http });
  assert.equal(requests[0], 'https://arxiv.org/list/cs.AI/recent?skip=0&show=2000');
  assert.equal(page.total, 2);
  assert.equal(page.items.length, 1);
  assert.equal(page.items[0].paperKey, 'arxiv:2609.12345');
  assert.equal(page.items[0].evidenceDepth, 'listing-metadata');
  assert.equal(page.nextOffset, 1);
  const unavailable = await run('recent', workspace, { category: 'cs.AI', date: '2026-10-01' }, { http });
  assert.equal(unavailable.state, 'date-unavailable');
  assert.deepEqual(unavailable.availableDates, ['2026-09-30']);
  assert.equal(unavailable.total, 0);
  await assert.rejects(run('recent', workspace, { category: '../../etc' }, { http }), /category/i);
  await assert.rejects(run('recent', workspace, { category: 'cs.AI', date: '2026-02-30' }, { http }), /date/i);
});

test('paper canonicalizes identifiers and returns actual metadata from the core Atom parser', async t => {
  const { library, workspace } = await fixture(t);
  await run('init', workspace, { library });
  const atom = `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/2606.12345v2</id><title>Efficient inference</title><summary>An abstract about inference.</summary><published>2026-06-13T00:00:00Z</published><updated>2026-06-14T00:00:00Z</updated><author><name>A. Author</name></author><category term="cs.AI"/></entry></feed>`;
  const urls = [];
  const http = { async request(req) { urls.push(req.url); return { status: 200, headers: {}, bodyText: atom }; } };
  const paper = await run('paper', workspace, { id: 'https://arxiv.org/abs/2606.12345v2' }, { http });
  assert.match(urls[0], /id_list=2606.12345/);
  assert.equal(paper.paperKey, 'arxiv:2606.12345');
  assert.equal(paper.metadata.title, 'Efficient inference');
  assert.equal(paper.metadata.abstract, 'An abstract about inference.');
  assert.equal(paper.evidenceDepth, 'metadata-and-abstract');
  assert.equal(paper.fullText, null);
  await assert.rejects(run('paper', workspace, { id: 'https://evil.example/2606.12345' }, { http }), /arxiv/i);
});

test('full-text retrieval uses bounded shared extraction and reports sections rather than the whole PDF', async t => {
  const { library, workspace } = await fixture(t);
  await run('init', workspace, { library });
  const atom = `<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/2606.12347</id><title>Test paper</title><summary>A test abstract.</summary><published>2026-06-13T00:00:00Z</published><updated>2026-06-13T00:00:00Z</updated><author><name>A. Author</name></author><category term="cs.AI"/></entry></feed>`;
  const html = `<html><body><div class="ltx_abstract">A test abstract.</div><h2>Methods</h2><p>${'method '.repeat(4000)}</p><h2>Conclusion</h2><p>A finding.</p><h2>References</h2><p>Not evidence text.</p></body></html>`;
  const http = { async request(req) { return { status: 200, headers: {}, bodyText: req.url.includes('/api/query') ? atom : html }; } };
  const paper = await run('paper', workspace, { id: '2606.12347', fullText: true }, { http });
  assert.equal(paper.evidenceDepth, 'extracted-sections');
  assert.equal(paper.fullTextSource, 'arxiv-html');
  assert.match(paper.fullText, /Methods/);
  assert.ok(paper.fullText.length < 60000);
  assert.doesNotMatch(paper.fullText, /Not evidence text/);
  assert.equal(paper.extractionLimits.sectionCharacters, 12000);
});
