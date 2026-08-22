# P6 — 可选本地 sidecar 高质量 PDF 解析

goal_ref: ../goal.md
updated: 2026-08-22

## Outcome

用户可显式启用一个仅限 loopback 的高质量 PDF parser sidecar；宿主先探测能力，再逐篇传递已获准的 PDF bytes 并把受限结构化响应接入既有 `DocumentParser` 契约。sidecar 不接收 library root 或文件路径，任何不可用、超限、取消或 schema 失败都可靠回退 PDF.js。

## Assumptions

- Core 的 `DocumentParser`、`ParsedDocument`、parser provenance 与结构化 chunker 已足以消费 sidecar 结果，P6 不改变它们的 host-neutral 边界或 retrieval 排序。
- 当前机器有 Python 3.12、Docker 与 Podman 命令，但没有 Docling、GROBID 或可用镜像；安装依赖、拉取镜像和模型下载需要单独授权。
- 首个 sidecar 只能绑定 loopback HTTP endpoint，并逐篇接收 bounded PDF bytes；不得收到目录路径、glob、library inventory、Vault storage access 或用户未显式启用的远程 URL。
- Docling-only 是默认候选；只有真实复杂论文评测显示其缺少需要且 GROBID enrichment 能在同一 privacy/operation 边界内补足时，才引入第二服务。

## Approach

先定义 versioned capability/parse JSON contract、endpoint allowlist、byte/request/response/block/locator 上限和 fail-closed decoder，并用 fake loopback server 取得 Red/Green。sidecar client 仅在用户配置且显式启用时选择，向其发送单份已读取 PDF bytes，不传路径；production parser 把成功结果标记为独立 provenance，任何 probe/transport/validation/cancel 失败退回现有 PDF.js。最后以 Docling-only 对复杂论文 corpus 的布局、标题、heading、table/caption 以及页码定位作对照，按预先定义的选择门决定是否需要 GROBID enrichment。

## Test strategy

- change kind: optional host integration plus parser behavior change
- strategy: strict Red–Green–Refactor for protocol, capability and fallback chunks；当前 PDF.js parser 是兼容 Green baseline
- Red / baseline signal: contract/client tests先因 sidecar parser/capability endpoint不存在失败；production orchestration tests先显示未启用时不触网、失败时未回退或路径泄漏
- Green / regression checks: focused Core/plugin protocol and indexing tests，随后 complete Core/Plugin/Node/CLI、typecheck、boundaries、production build和`git diff --check`
- exception: Docling/GROBID quality comparison needs downloaded local dependencies and real corpus；在获得安装授权前，只接受 fake-sidecar contract evidence，不接受 provider-choice 或 P6 completion

## Tasks

- [ ] 定义并验收 loopback-only sidecar capability 与 parse transport contract：无 path/root 字段、严格 byte/response 上限、version/capability/provenance 验证和 fail-closed decode。
- [ ] 实现并验收可选 client/parser adapter：未启用时不调用 endpoint，probe/transport/schema/cancel failure 全部回退 PDF.js，sidecar 不影响 Core 或 default local path。
- [ ] 将用户显式 enable、endpoint loopback 校验、能力诊断与 parser selection 接入 host，并验证 consent/配置改变会停用或重建派生索引而不扫描任意目录。
- [ ] 在获授权的本地 Docling-only environment 对真实复杂论文 corpus 运行预先定义的结构/定位评测，记录选择证据；仅在缺口可复现时评估 GROBID enrichment。
- [ ] 如评测选择保留 Docling-only，完成 production hardening；如选择 enrichment，先 L2 reshape P6 protocol/operation plan，再实现第二 provider。
- [ ] 完成 P6 全量回归与实际 host validation。

## Verification

- Capability baseline (2026-08-22): Python 3.12.3、Docker/Podman 命令可用；`docling`、`grobid` 与现成 sidecar 均不可用。尚未取得 provider-choice 或 real-corpus evidence。

## Abort / reshape triggers

- 如果任何方案需要把 selected library root、logical path 或目录扫描能力交给 sidecar，停止并 reshape transport。
- 如果 loopback/explicit enable 无法由当前 host 强制，或 sidecar 可以被配置为任意 remote endpoint，停止接线并先收紧 capability boundary。
- 如果 sidecar failure 让已接受 PDF.js parser 失效、改变既有 parser 输出却不触发 provenance reindex，停止并修复 fallback/derivation 选择。
- 如果 Docling-only 的真实评测不优于 PDF.js，或 GROBID 需要独立未界定的文件/网络权限，停止 provider integration 并进行 L2 reshape。
