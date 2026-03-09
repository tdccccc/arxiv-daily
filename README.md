# arxiv-daily

arXiv astro-ph 每日论文自动追踪。用 LLM 从当日全部新论文中语义筛选相关论文，生成中文日报和详细论文报告。

## 工作流程

1. 北京时间 9:30 起轮询 arXiv astro-ph/new，等待当日更新
2. 解析所有新论文的标题 + 摘要
3. LLM 一次性筛选相关论文，标记分类 (category) 和是否详细收录 (detail)
4. 对筛选出的论文抓取 HTML 全文内容（带本地缓存，避免重复请求）
5. 生成日报 → `daily/YYYY-MM-DD.md`（所有相关论文的简要总结，按分类分组）
6. 对详细收录论文生成单独报告 → `papers/YYMM.NNNNN.md`

## 快速开始

```bash
# 安装依赖
pip install requests beautifulsoup4 pytz openai python-dotenv

# 配置
cp .env.example .env
# 编辑 .env，填入 API Key，修改研究兴趣等

# 运行
python arxiv_daily.py
```

## 配置说明

所有配置通过 `.env` 文件管理：

| 变量 | 说明 | 默认值 |
|---|---|---|
| `LLM_API_KEY` | LLM API Key | **必填** |
| `LLM_BASE_URL` | API 端点 | `https://api.openai.com/v1` |
| `LLM_MODEL` | 模型名称 | `gpt-4o` |
| `LLM_TEMPERATURE` | 生成温度 | `0.3` |
| `LLM_TIMEOUT` | LLM 请求超时（秒），推理模型建议调大 | `300` |
| `WORK_DIR` | 输出目录（日报和论文报告） | `./output` |
| `RESEARCH_INTERESTS` | 研究兴趣描述 | 有默认值 |
| `DETAIL_CRITERIA` | 详细收录标准 | 有默认值 |
| `CATEGORY_TAG_MAP` | 分类→标签映射 (JSON) | 有默认值 |
| `CATEGORY_DISPLAY_MAP` | 分类→日报显示名称 (JSON) | 有默认值 |
| `REQUEST_DELAY` | arXiv 请求间隔（秒） | `3` |
| `POLL_INTERVAL` | 轮询间隔（秒） | `1800` |
| `MAX_RETRIES` | 最大轮询次数 | `16` |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `LOG_FILE` | 日志文件路径 | 脚本目录下 `arxiv_daily.log` |
| `CACHE_DIR` | 缓存目录 | 脚本目录下 `.cache/` |
| `CACHE_EXPIRY_DAYS` | 缓存过期天数 | `7` |

多行文本在 `.env` 中用双引号包裹即可直接换行，参见 `.env.example`。

## 健壮性

- **结构化日志**：所有输出通过 `logging` 模块，支持文件 + 控制台双输出
- **HTTP 重试**：所有网络请求带指数退避重试，404 等不可恢复状态码直接跳过
- **LLM 重试**：LLM 调用失败自动重试（5s/10s/20s），使用流式请求避免服务端空闲断连
- **HTML 缓存**：论文 HTML 和 Abstract 页面缓存到本地，避免重复请求，自动过期清理
- **容错**：单篇论文抓取或总结失败不影响其余论文处理

## 自定义研究方向

修改 `.env` 中的以下变量：

- `RESEARCH_INTERESTS`：用自然语言描述你的研究兴趣，LLM 据此判断论文相关性
- `DETAIL_CRITERIA`：哪些类型的论文需要生成详细报告
- `CATEGORY_DISPLAY_MAP`：自定义日报中各分类的显示名称
