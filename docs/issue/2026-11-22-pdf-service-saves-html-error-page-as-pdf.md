# PdfService 下载论文 PDF 时未校验 %PDF- 文件头，会把 arXiv 返回的 HTTP 200 HTML 错误/校验页保存为 .pdf

状态：未修复

> 来源：2026-11-22 main `2557b90a` 代码审查（核心 PDF 下载服务）
> 严重程度：中 · 置信度：已复现（离线实测：HTML 正文被以 `kind: "done"` 写入 `.pdf`）

## 问题与依据

`packages/core/src/services/pdf.ts:56-84`（`PdfService.downloadById`）调用 `this.deps.fetcher.fetchPdf(arxivId, signal)` 拿到 `ArrayBuffer` 后，未做任何内容格式校验就直接落盘并更新索引：

```ts
pdf = await this.deps.fetcher.fetchPdf(arxivId, signal);
...
await this.deps.storage.writeBinary(path, pdf);
if (this.deps.paperIndex) {
  const updated = await this.deps.paperIndex.setPdfPath(arxivId, path);
  entryUpdated = Boolean(updated);
}
```

而上游 `ArxivFetcher.fetchPdf`（`packages/core/src/pipeline/arxiv-fetcher.ts`）同样只检查 HTTP 状态码 `200` 和响应体非空，不校验 `Content-Type` 或 PDF 魔数。

arXiv 在 PDF 尚未生成、触发限流/人机验证（reCAPTCHA）、论文仅提供 PostScript/withdrawn 等情况下，常会返回 HTTP 200 的 HTML 页面（如 `<!DOCTYPE html>...`）。当前实现会：
1. 把这份 HTML 源码原封不动写成 `<root>/pdfs/<id>.pdf`；
2. 将该路径写入 `papers.json` 的 `pdfPath`；
3. 向用户汇报下载成功，Dashboard 显示 “PDF saved”，但用户点击打开时外部 PDF 阅读器或 Obsidian PDF 视图立即报文件损坏。

## 原因

`fetchPdf` 与 `PdfService.downloadById` 仅把 HTTP 200 + 非空字节当成有效 PDF，缺少对 PDF 文件魔数（`%PDF-`）和空/HTML 响应的检查。

## 建议和验收

- 在 `ArxivFetcher.fetchPdf` 或 `PdfService.downloadById`（推荐两处同时把关：fetcher 拒绝非 PDF 响应，service 落盘前做最终守门）检查字节内容：
  - 按 PDF 规范，合法 PDF 头部（通常在前 1024 字节内）必须包含 ASCII 签名 `%PDF-`（`0x25 0x50 0x44 0x46 0x2D`）；若缺少 `%PDF-`（或响应头 `content-type` 明确为 `text/html`），拒绝落盘并返回 `{ kind: "fetch_error", reason: "..." }`，不写入 `.pdf` 也不更新 `papers.json` 的 `pdfPath`。
- 验收：当 HTTP 响应返回状态 200 且正文为 `<!DOCTYPE html><html>...` 时，`PdfService.downloadById()` 返回 `fetch_error`，不生成 `.pdf` 文件，且不修改 `papers.json`。

## 修复后应补的测试

- `packages/core/tests/pdf.test.ts`：
  - 模拟 `fetchPdf` 返回 HTML 字节（HTTP 200），断言返回 `{ kind: "fetch_error" }`、未写入 `.pdf` 文件且 `paperIndex.setPdfPath` 未被调用；
  - 模拟返回以 `%PDF-1.7` 开头的合法字节，断言正常写入并更新 `pdfPath`。
