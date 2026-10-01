// Preloaded only by the integration tests: intercept the HTTP boundary while
// leaving the actual CLI, scheduler, pipeline, parsers, writer and stores intact.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const scenario = process.env.ARXIV_CORE_FIXTURE_SCENARIO;
const log = process.env.ARXIV_CORE_FIXTURE_LOG;
assert.ok(log, 'fixture request log is required');

const papers = {
  '2605.08080': { title: 'Fixture calibrated redshifts', abstract: 'Photometric redshift calibration improves accuracy by twelve percent.' },
  '2605.08068': { title: 'Fixture uncertainty estimates', abstract: 'Photometric redshift uncertainty estimates are evaluated on a validation sample.' },
  '2605.08001': { title: 'Fixture unrelated quantum paper', abstract: 'An unrelated quantum circuit construction.' },
  '2605.09999': { title: 'Fixture manual follow-up', abstract: 'Independent manual paper on photometric redshift inference.' },
};

function record(kind, url, extra = {}) {
  fs.appendFileSync(log, `${JSON.stringify({ kind, url, ...extra })}\n`);
}

function paper(id) {
  assert.ok(papers[id], `unexpected fixture paper: ${id}`);
  return papers[id];
}

function atom(ids) {
  return `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">${ids.map(id => {
    const p = paper(id);
    return `<entry><id>http://arxiv.org/abs/${id}v1</id><title>${p.title}</title><author><name>Fixture Author</name></author><summary>${p.abstract}</summary><published>2026-05-08T00:00:00Z</published><updated>2026-05-09T00:00:00Z</updated><arxiv:primary_category term="astro-ph.CO"/><category term="astro-ph.CO"/></entry>`;
  }).join('')}</feed>`;
}

function paperHtml(id) {
  const p = paper(id);
  return `<html><body><div class="ltx_abstract">${p.abstract}</div>
  <h2>Introduction</h2><p>We investigate photometric redshift calibration on a validation dataset of one thousand galaxies.</p>
  <h2>Methods</h2><p>We fit a calibrated regression model and evaluate predictions on held-out galaxies against the baseline.</p>
  <h2>Results</h2><p>The measured error falls by twelve percent on the validation sample. The comparison uses identical training data.</p>
  <h2>Conclusions</h2><p>The calibration helps this dataset. Broader survey generalization remains untested.</p></body></html>`;
}

function note(id) {
  const p = paper(id);
  return `# ${p.title}\n\n- **arXiv**: https://arxiv.org/abs/${id}\n\n` + [
    ['Research Problem', 'The paper studies photometric redshift calibration and the errors that affect inference on the provided validation sample. The motivation is reducing estimation error under the observed data conditions.'],
    ['Method Design', 'Fixture full-text assessment: the authors fit a calibrated regression model and compare held-out predictions against a baseline with identical training data. This design isolates the measured calibration effect.'],
    ['Core Results', 'The reported prediction error falls by twelve percent on a validation dataset of one thousand galaxies. The described experiment supports a comparison on this sample; no broader survey measurement is supplied.'],
    ['Main Conclusions', 'The calibration improves prediction error within this dataset. The paper does not establish generalization to other surveys, so the result should be interpreted within the reported evaluation conditions.'],
    ['Key Figures and Tables', 'No figure or table information is included in the input.'],
    ['Contributions and Novelty', 'Not specified in the source text.'],
    ['Scope and Limits', 'Broader survey generalization remains untested. The provided sections do not describe additional datasets or uncertainty intervals, so no claim about those evaluations is justified.'],
    ['Academic Value Assessment', 'The supplied baseline comparison supports a bounded calibration improvement on the reported sample. This paper can serve as a method reference, while evidence for transfer to other datasets remains unavailable.'],
  ].map(([heading, body]) => `## ${heading}\n${body}\n`).join('\n');
}

function llmResponse(body, url) {
  const system = body.messages?.find(m => m.role === 'system')?.content ?? '';
  const user = body.messages?.filter(m => m.role === 'user').map(m => m.content).join('\n') ?? '';
  let result;
  if (system.includes('选择最匹配的主题')) {
    record('filter', url);
    assert.match(user, /2605\.08080/);
    assert.match(user, /2605\.08068/);
    assert.match(user, /2605\.08001/);
    result = JSON.stringify({ papers: scenario === 'zero' ? [] : [
      { id: '2605.08080', category: 'photo-z' },
      { id: '2605.08068', category: 'photo-z' },
      { id: '2605.08001', category: 'skip' },
    ] });
  } else if (system.includes('strict research-paper evaluator')) {
    record('selector', url);
    result = JSON.stringify({ papers: [
      { id: '2605.08080', score: 85, reason: 'A supported calibration improvement.' },
      { id: '2605.08068', score: 40, reason: 'Insufficient evidence for automatic detail.' },
    ] });
  } else if (system.includes('strict JSON object') || system.includes('严格 JSON 对象')) {
    const id = /ID: (\d{4}\.\d{4,5})/.exec(user)?.[1];
    paper(id);
    record('daily-summary', url, { id });
    result = JSON.stringify({ id, coreProblem: `Structured problem for ${id}`, keyMethod: `Structured method for ${id}`, mainResult: `Structured result for ${id}`, whyRelevant: 'Directly relates to photometric redshift calibration.', limitations: 'Broader survey generalization is untested.' });
  } else if (system.includes('generate a detailed English paper summary')) {
    const id = /https:\/\/arxiv\.org\/abs\/(\d{4}\.\d{4,5})/.exec(user)?.[1];
    paper(id);
    record('paper-note', url, { id });
    result = note(id);
  } else {
    record('unexpected-prompt', url);
    throw new Error(`Unrecognized fixture LLM prompt: ${system.slice(0, 160)}`);
  }
  return new Response(`data: ${JSON.stringify({ choices: [{ delta: { content: result } }] })}\n\ndata: [DONE]\n\n`, {
    status: 200, headers: { 'content-type': 'text/event-stream' },
  });
}

globalThis.fetch = async function fixtureFetch(input, init = {}) {
  const url = new URL(String(input));
  if (scenario === 'offline') {
    record('unexpected-offline-request', url.href);
    throw new Error(`Network forbidden during fixture replay: ${url.href}`);
  }
  if (url.origin === 'https://arxiv.org' && url.pathname === '/list/astro-ph/recent') {
    record('recent', url.href);
    return new Response(`<dl id="articles"><h3>Mon, 11 May 2026 (showing 3 of 3 entries)</h3>${['2605.08080', '2605.08068', '2605.08001'].map(id => `<dt><a title="Abstract" href="/abs/${id}">arXiv:${id}</a></dt><dd><div class="list-title">Title: ${paper(id).title}</div><div class="list-authors"><a>Fixture Author</a></div></dd>`).join('')}</dl>`, { status: 200 });
  }
  if (url.origin === 'https://export.arxiv.org' && url.pathname === '/api/query' && url.searchParams.has('id_list')) {
    const ids = url.searchParams.get('id_list').split(',').map(id => id.replace(/v\d+$/, ''));
    record('metadata', url.href, { ids });
    return new Response(atom(ids), { status: 200 });
  }
  if (url.origin === 'https://arxiv.org' && /^\/html\/\d{4}\.\d{4,5}(?:v\d+)?$/.test(url.pathname)) {
    const id = url.pathname.split('/').pop().replace(/v\d+$/, '');
    record('paper-html', url.href, { id });
    return new Response(paperHtml(id), { status: 200 });
  }
  if (url.href === 'https://fixture.invalid/v1/chat/completions' && init.method === 'POST') {
    return llmResponse(JSON.parse(init.body), url.href);
  }
  record('unexpected-url', url.href);
  throw new Error(`No fixture HTTP response for ${url.href}; public network is disabled`);
};
