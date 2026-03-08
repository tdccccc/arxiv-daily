# arxiv-daily

arXiv astro-ph 每日论文自动追踪。用 LLM 从当日全部新论文中语义筛选相关论文，生成中文日报和详细论文报告。

## 工作流程

1. 北京时间 9:30 起轮询 arXiv astro-ph/new，等待当日更新
2. 解析所有新论文的标题 + 摘要
3. LLM 一次性筛选相关论文，标记分类 (category) 和是否详细收录 (detail)
4. 对筛选出的论文抓取 HTML 全文内容
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

| 变量 | 说明 | 必填 |
|---|---|---|
| `LLM_API_KEY` | LLM API Key | 是 |
| `LLM_BASE_URL` | API 端点 | 否，默认 OpenAI |
| `LLM_MODEL` | 模型名称 | 否，默认 gpt-4o |
| `WORK_DIR` | 输出目录 | 否，默认 ./output |
| `RESEARCH_INTERESTS` | 研究兴趣描述 | 否，有默认值 |
| `DETAIL_CRITERIA` | 详细收录标准 | 否，有默认值 |
| `CATEGORY_TAG_MAP` | 分类→标签映射 (JSON) | 否，有默认值 |
| `REQUEST_DELAY` | arXiv 请求间隔（秒） | 否，默认 3 |
| `POLL_INTERVAL` | 轮询间隔（秒） | 否，默认 1800 |
| `MAX_RETRIES` | 最大重试次数 | 否，默认 16 |

多行文本在 `.env` 中用双引号包裹即可直接换行，参见 `.env.example`。

## 自定义研究方向

修改 `.env` 中的 `RESEARCH_INTERESTS` 和 `DETAIL_CRITERIA`，用自然语言描述你的研究兴趣和详细收录标准即可。LLM 会据此判断哪些论文相关。
