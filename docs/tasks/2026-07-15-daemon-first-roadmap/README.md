# Daemon-First 架构路线图

> 日期: 2026-07-15 | 状态: Draft

---

## 背景

项目当前有三个前端（Obsidian 插件、Rust CLI、egui 桌面版）共享一个 Rust daemon。经过 Round 1 + Round 2 UI/UX 优化实践，确认 egui 即时模式框架存在难以逾越的美观度天花板，投入产出比低。决定放弃 egui 桌面版，聚焦 **daemon + Obsidian 插件 + CLI** 三端协同架构。

## 架构总览

```
┌─────────────────────────────────────────┐
│           Rust Daemon (核心)              │
│  ┌─────────────────────────────────────┐ │
│  │ protocol crate (API 类型，单一来源)    │ │
│  ├─────────────────────────────────────┤ │
│  │ 调度 / arXiv 抓取 / LLM 总结         │ │
│  │ 论文索引 / 推荐引擎 / 全文检索        │ │
│  │ HTTP REST API + SSE 事件流           │ │
│  └─────────────────────────────────────┘ │
└────────────┬────────────┬────────────────┘
             │            │
     HTTP + SSE       HTTP + SSE
             │            │
    ┌────────┴───┐  ┌────┴──────────┐
    │ Obsidian   │  │ Rust CLI      │
    │ 插件       │  │ arxiv-dailyctl│
    │ (知识工作)  │  │ (脚本/cron)   │
    └────────────┘  └───────────────┘
```

**前端和 CLI 只是 API 消费者。** 所有业务逻辑、数据一致性、调度策略全在 daemon 里。改一次 daemon，三个端同时受益。

---

## Phase 0: 核心功能对齐 (当前)

**目标**：确保 Rust daemon 核心功能与 Obsidian 插件版行为完全一致。

### 0.1 创建新 worktree

```bash
git worktree add .worktree/core-stabilize refactor/rust-standalone
```

基于 rust 分支，标记 egui 为 deprecated。后续如删除桌面版，另开 PR。

### 0.2 核心功能审计清单

逐项对比 Rust daemon 与 Obsidian 插件实现。

| 模块 | 检查项 | Obsidian 参考 | daemon 参考 |
|------|--------|-------------|------------|
| **arXiv 抓取** | Feed 解析、去重、日期过滤 | `pipeline/fetch.ts` | `crates/pipeline/` |
| **LLM 总结** | Prompt 一致性、thinking mode、重试 | `llm/client.ts` | `crates/pipeline/` |
| **Topic 匹配** | 分类规则、detail 标记、primary topic | `dashboard/model.ts` | `crates/search/` |
| **调度系统** | 时区、run window、tick interval | `services/scheduling/` | `crates/scheduler/` |
| **输出格式** | Daily report Markdown、link style、语言 | `pipeline/` | daemon executor |
| **论文状态** | Star/priority 映射、status 状态机 | `dashboard/model.ts` | `crates/domain/` |
| **日历/历史** | Run state、retry 逻辑 | `services/run-history.ts` | `crates/storage/` |
| **日志系统** | 级别、格式、持久化 | `services/logger.ts` | daemon |

### 0.3 CLI 命令补齐

需新增的命令：

| 命令 | 优先级 | 说明 |
|------|--------|------|
| `config show` | 高 | 展示完整配置 |
| `config set KEY VALUE` | 高 | 修改单项配置 |
| `papers list [--search] [--topic]` | 高 | 搜索/列出论文 |
| `papers star ARXIV_ID` | 中 | 星标论文 |
| `logs [--level] [--follow]` | 中 | 查看日志 |
| `diagnostics` | 中 | 健康检查 |
| `rec` | 低 | 推荐论文 (Phase 2 后) |
| `search QUERY` | 低 | 全文搜索 (Phase 2 后) |
| `export [--format]` | 低 | 导出数据 |
| `backup` | 低 | 备份数据目录 |

### 0.4 egui 桌面版处理

- 不在新 worktree 中继续投入
- `refactor/rust-standalone` 分支保留历史存档
- Phase 0 完成后从 workspace 中移除 `apps/desktop/`

---

## Phase 1: API 稳定

**目标**：protocol crate 定版，确保 Obsidian 插件和 CLI 无痛升级。

### 1.1 protocol 定版 (0.1.0 → 0.2.0)

- 一次性处理所有 breaking changes
- Review 所有 endpoint 的 request/response 类型完整性
- 确保 SSE 事件类型完整且有文档
- 错误响应格式统一

### 1.2 Obsidian 插件同步

- 升级插件侧 API 调用适配 0.2.0
- 确保 HTTP client 和 SSE client 兼容

### 1.3 版本策略

```
protocol 版本      daemon 兼容范围
─────────────────────────────────
0.2.x              Obsidian 插件 >= 0.2.0, CLI >= 0.1.0
```

---

## Phase 2: daemon 功能深化

**目标**：纯 daemon 侧新功能，所有前端自动受益。

### 2.1 论文推荐引擎

- 基于用户 star 历史的协同/内容推荐
- 推荐结果发布为 SSE 事件 (`paper.recommended`)
- 提供 `GET /v1/recommendations` endpoint

### 2.2 跨库去重

- arXiv ↔ Semantic Scholar ↔ 其他源
- 基于 DOI / title 相似度
- `POST /v1/dedup-check` endpoint

### 2.3 PDF 全文检索

- 集成 Tantivy / LanceDB 做索引
- `GET /v1/search?q=...` endpoint
- CLI: `search "transformer attention"`

### 2.4 性能优化

- 增量索引（新论文只处理增量）
- 查询缓存
- 大论文库下的 Dashboard 响应时间优化

---

## Phase 3: Obsidian 插件做精

**目标**：展示层升级，利用 daemon 新能力。

### 3.1 推荐卡片

- Dashboard 顶部展示 daemon 传来的推荐论文
- 一键 star / dismiss

### 3.2 搜索结果高亮

- 用 daemon 返回的全文搜索结果
- 关键词高亮、摘要片段

### 3.3 交互优化

- 批量操作 undo 支持
- 拖拽排序 (Phase 2 去重功能)
- 通知面板 (SSE 事件驱动的原生通知 + UI 消息中心)

### 3.4 性能

- 虚拟滚动大论文列表
- 增量渲染
- Tab 切换即时响应

---

## Phase 4: CLI 做全

**目标**：远程/脚本/自动化场景全覆盖。

### 4.1 核心命令

```
arxiv-dailyctl run today          # 运行今日
arxiv-dailyctl run date 2026-07   # 运行指定日期
arxiv-dailyctl config show        # 查看配置
arxiv-dailyctl config set llm.model gpt-4  # 修改配置
arxiv-dailyctl papers list --starred        # 论文列表
arxiv-dailyctl papers star 2607.01234       # 星标
arxiv-dailyctl logs --follow               # 实时日志
arxiv-dailyctl diagnostics                 # 健康检查
```

### 4.2 自动化场景

- `cron`: 定时运行 + 结果推送 (webhook/email)
- `systemd timer`: 替代内置 scheduler，更灵活
- CI/CD: `papers export` → 数据库/报告

### 4.3 输出格式

- `--json`: 可管道给 `jq`
- `--table`: 人类可读
- `--quiet`: 仅返回退出码

---

## 不做的

| 项目 | 原因 |
|------|------|
| egui 桌面版 | 即时模式框架美观度天花板低，投入产出比差 |
| Electron/Tauri 独立应用 | 插件用户体验已足够好，新增前端 ROI 低 |
| Web 版 | 非当前用户需求 |

---

## 实施节奏

```
现在 → Phase 0: 核心对齐 + CLI 补齐 (2-3 周)
   ↓
Phase 1: API 定版 + 插件同步 (1 周)
   ↓
Phase 2: daemon 功能深化 (持续)
   ↓
Phase 3 + Phase 4: 并行推进，按需切换 (持续)
```
