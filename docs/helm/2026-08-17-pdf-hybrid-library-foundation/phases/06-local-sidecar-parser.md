# P6 — 可选本地 sidecar 高质量 PDF 解析

goal_ref: ../goal.md
updated: 2026-08-22

## Outcome

用户可显式启用一个仅限 loopback 的高质量 PDF parser sidecar；宿主先探测能力，再逐篇传递已获准的 PDF bytes 并把受限结构化响应接入既有 `DocumentParser` 契约。sidecar 不接收 library root 或文件路径，任何不可用、超限、取消或 schema 失败都可靠回退 PDF.js。

## Assumptions

- `ParsedDocument`、parser provenance 与结构化 chunker 可消费 sidecar 结果，但现有 `DocumentParser` 的静态 capabilities/provenance 无法表达单篇 sidecar failure 后的 PDF.js 回退；P6 需要最小的 host-neutral per-parse selector identity，且不改变旧 parser 调用方或 retrieval 排序。
- 当前机器有 Python 3.12、Docker 与 Podman 命令，但没有 Docling、GROBID 或可用镜像；安装依赖、拉取镜像和模型下载需要单独授权。
- 首个 sidecar 只能绑定 loopback HTTP endpoint，并逐篇接收 bounded PDF bytes；不得收到目录路径、glob、library inventory、Vault storage access 或用户未显式启用的远程 URL。
- Docling-only 是默认候选；只有真实复杂论文评测显示其缺少需要且 GROBID enrichment 能在同一 privacy/operation 边界内补足时，才引入第二服务。

## Approach

先定义 versioned capability/parse JSON contract、endpoint allowlist、byte/request/response/block/locator 上限和 fail-closed decoder，并用 fake loopback server 取得 Red/Green。随后增加 host-neutral parser selector：每篇返回实际选中的 document、capabilities 与 provenance；旧 `DocumentParser` 保持兼容。sidecar client 仅在用户配置且显式启用时选择，向其发送单份已读取 PDF bytes，不传路径；sidecar 成功以其 identity 入库，probe/transport/validation failure 回退 PDF.js identity，取消仍向上传播。最后以 Docling-only 对复杂论文 corpus 的布局、标题、heading、table/caption 以及页码定位作对照，按预先定义的选择门决定是否需要 GROBID enrichment。

## Test strategy

- change kind: optional host integration plus parser behavior change
- strategy: strict Red–Green–Refactor for protocol, capability and fallback chunks；当前 PDF.js parser 是兼容 Green baseline
- Red / baseline signal: contract/client tests先因 sidecar parser/capability endpoint不存在失败；production orchestration tests先显示未启用时不触网、失败时未回退或路径泄漏
- Green / regression checks: focused Core/plugin protocol and indexing tests，随后 complete Core/Plugin/Node/CLI、typecheck、boundaries、production build和`git diff --check`
- exception: Docling/GROBID quality comparison needs downloaded local dependencies and real corpus；在获得安装授权前，只接受 fake-sidecar contract evidence，不接受 provider-choice 或 P6 completion

## Tasks

- [x] 定义并验收 loopback-only sidecar capability 与 parse transport contract：无 path/root 字段、严格 byte/response 上限、version/capability/provenance 验证和 fail-closed decode。
- [x] 实现并验收 host-neutral per-parse selector 与可选 client/parser adapter：每次 parse 持久化实际获胜的 capabilities/provenance，per-parse sidecar failure 以 PDF.js identity 回退，cancel 不降级，旧 `DocumentParser` 路径保持兼容。
- [x] 将用户显式 enable、endpoint loopback 校验、能力诊断与 parser selection 接入 host：未启用时绝不 probe/request，probe/transport/schema failure 以 PDF.js identity 继续；sidecar 是本地 byte-only 处理，不扩大 remote consent，配置改变会取消活跃 index，后续运行按实际 parser derivation 重建受影响投影且不扫描任意目录。
- [x] 在获授权的本地 Docling-only environment 对真实复杂论文 corpus 运行预先定义的结构/定位评测，记录选择证据；仅在缺口可复现时评估 GROBID enrichment。
- [x] 如评测选择保留 Docling-only，完成 production hardening；如选择 enrichment，先 L2 reshape P6 protocol/operation plan，再实现第二 provider。
- [x] 完成 P6 自动化全量回归与实际 loopback host validation；桌面 Obsidian 交互验证留给 P7 跨平台验收。

## Verification

- Capability baseline (2026-08-22): Python 3.12.3、Docker/Podman 命令可用；`docling`、`grobid` 与现成 sidecar 均不可用。尚未取得 provider-choice 或 real-corpus evidence。
- P6.1 observed Red：Core 没有 sidecar contract/client，无法证明 capability probe 不带 PDF/path，或 parse request 不会传递 library root/logical path；未经 schema、response-size、provenance 与 declared capability 验证的 JSON 不能进入 `ParsedDocument`。
- P6.1 Green：Core 定义 protocol v1 capability/parse decoder 和同 origin、IP-literal loopback HTTP client。capability probe 为无 body GET；parse 为含 `application/pdf` body 的单一 `ArrayBuffer` POST。未知字段（含 path）、非 loopback/cross-origin、越界 request/response、HTTP/JSON/schema/provenance/capability mismatch 均 typed fail-closed。定向 Core 2 files / 8 tests、Core typecheck、`check:boundaries` 与 `git diff --check` 通过。实现提交为 `7ec3e2e` 与 `3a5426a`。
- P6.2 L2 evidence：现有 `DocumentParser` 将 capabilities/provenance 固定在实例上，`index-orchestration` 也从该静态字段派生 document identity。简单 try-sidecar/catch-PDF.js wrapper 会把 fallback document 错标为 sidecar 或用 union capability 错误驱动 chunker，破坏 P2 的 derivation/reindex contract。保留 P6.1 协议/client；重划为每次 parse 返回实际 parser identity 的 selector，旧 parser API 不变。
- Provider evaluation (2026-08-22): 在临时、CPU-only 的 Docling 2.121.0 / torch 2.7.1+cpu / torchvision 0.22.1+cpu environment 中，以 byte stream 解析 `1706.03762`（15 页、21.562 s、511 text / 4 table / 6 picture / 28 heading / 9 caption）、`1810.04805`（16 页、18.973 s、622 / 8 / 5 / 33 / 13）和 `2206.01062`（9 页、17.984 s、543 / 5 / 6 / 18 / 10）。与 `pdftotext` 页级基线对照，标题、heading、table/figure caption 与页码均一致，没有 locator 超过文档页数；记录位于临时评测输出，未写入用户文献库。
- Provider decision (2026-08-22): Docling-only 已满足结构、表格、caption、heading 和页定位要求，未观察到可复现的质量缺口；不引入会扩大操作和隐私边界的 GROBID。
- Production hardening (2026-08-22): 实际 service 在 `127.0.0.1:5001` 启动、probe 后解析 Attention PDF，返回 53,656 bytes、156 blocks、27 headings、4 tables、9 captions，block id 连续且页 locator 有效。服务只接受 `DocumentStream` 内存 PDF bytes，拒绝 query/path-like route、非 PDF、transfer encoding、空/缺失/超限 body，request/response 上限为 25 MiB / 16 MiB；`python3 -m py_compile tools/docling-sidecar/server.py` 和 `python3 -m unittest tools/docling-sidecar/tests/test_server.py`（5 tests）通过。
- P6 final regression (2026-08-22): Core 113 files / 2,015 tests、Plugin 41 files / 622 tests、Node runtime 3 files / 45 tests、CLI 7 files / 71 tests 通过；workspace typecheck、`check:boundaries`、`check:product-units`、production build 和 `git diff --check` 通过。lint 无新 error，仍为仓库既有 65 warnings / 60-cap baseline；桌面 Obsidian 的实际 viewer 与 setting interaction 不可在本容器运行，明确移交 P7。

## Abort / reshape triggers

- 如果任何方案需要把 selected library root、logical path 或目录扫描能力交给 sidecar，停止并 reshape transport。
- 如果 loopback/explicit enable 无法由当前 host 强制，或 sidecar 可以被配置为任意 remote endpoint，停止接线并先收紧 capability boundary。
- 如果 sidecar failure 让已接受 PDF.js parser 失效、改变既有 parser 输出却不触发 provenance reindex，停止并修复 fallback/derivation 选择。
- 如果 Docling-only 的真实评测不优于 PDF.js，或 GROBID 需要独立未界定的文件/网络权限，停止 provider integration 并进行 L2 reshape。
