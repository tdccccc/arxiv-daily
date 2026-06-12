#!/usr/bin/env python3
"""
arXiv astro-ph 每日论文追踪脚本

功能：
1. 北京时间 9:30 起，每 30 分钟轮询 arXiv 是否已更新为当日内容
2. 确认更新后，解析所有论文元数据，用 LLM 全量筛选相关论文
3. 日常追踪层：所有相关论文尽量用 Abstract + Conclusion + 可用正文摘录快速总结 → daily/
4. 详细阅读层：特别相关论文提取全文有用章节做详细总结 → papers/
"""

import os
import sys
import time
import datetime
import re
import json
import shutil
import logging

import pytz
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ================= 配置（全部从 .env 读取） =================
# LLM
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))
THINKING_MODE = os.getenv("LLM_THINKING_MODE", "true").lower() == "true"
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "high")

# 输出路径
WORK_DIR = os.getenv("WORK_DIR", os.path.join(os.path.dirname(__file__), "output"))
DAILY_DIR = os.path.join(WORK_DIR, "daily")
PAPERS_DIR = os.path.join(WORK_DIR, "papers")

# arXiv
ARXIV_CATEGORY = os.getenv("ARXIV_CATEGORY", "astro-ph")
TIMEZONE = os.getenv("TIMEZONE", "Asia/Shanghai")

# 筛选
RESEARCH_INTERESTS = os.getenv("RESEARCH_INTERESTS", """\
Describe your research interests here — what kinds of papers should the LLM look for?""")

DETAIL_CRITERIA = os.getenv("DETAIL_CRITERIA", """\
Papers proposing or comparing new methods / catalogs / benchmarks""")

CATEGORY_TAG_MAP = json.loads(os.getenv("CATEGORY_TAG_MAP",
    '{"example":"example","other":"other"}'))

CATEGORY_DISPLAY_MAP = json.loads(os.getenv("CATEGORY_DISPLAY_MAP",
    '{"example":"Example Section","other":"Other"}'))

# 允许标记 detail 的 category 列表（不在此列表中的 category 即使 LLM 标了 detail 也会降为 false）
DETAIL_CATEGORIES = json.loads(os.getenv("DETAIL_CATEGORIES",
    '["example"]'))

# 内容提取
SECTION_CHAR_LIMIT = int(os.getenv("SECTION_CHAR_LIMIT", "8000"))
PAPER_CHAR_LIMIT = int(os.getenv("PAPER_CHAR_LIMIT", "50000"))
DAILY_CHAR_LIMIT = int(os.getenv("DAILY_CHAR_LIMIT", "400000"))
SKIP_SECTIONS = json.loads(os.getenv("SKIP_SECTIONS",
    '["reference","bibliography","appendix","acknowledgement","acknowledgment",'
    '"author contribution","data availability","conflict of interest","orcid"]'))
PRIORITY_SECTIONS = json.loads(os.getenv("PRIORITY_SECTIONS",
    '["abstract","conclusion","summary"]'))

# 网络 & 轮询
REQUEST_DELAY = int(os.getenv("REQUEST_DELAY", "3"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "1800"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "16"))

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 日志
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "") or os.path.join(_SCRIPT_DIR, "arxiv_daily.log")

# 缓存
CACHE_DIR = os.getenv("CACHE_DIR", "") or os.path.join(_SCRIPT_DIR, ".cache")
CACHE_EXPIRY_DAYS = int(os.getenv("CACHE_EXPIRY_DAYS", "7"))
# =============================================================

BEIJING_TZ = pytz.timezone(TIMEZONE)
MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
}
WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


# ================= Logging =================
def _setup_logging():
    handlers = [logging.StreamHandler()]
    if LOG_FILE:
        handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger("arxiv_daily")


logger = _setup_logging()


# ================= 共享 Session & LLM 客户端 =================
_session = requests.Session()
_session.headers.update(HEADERS)

_llm_client = None


def _get_llm_client():
    """Lazy-create the LLM client so pure imports do not require an API key."""
    global _llm_client
    if _llm_client is None:
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("LLM_API_KEY is required for LLM calls")
        _llm_client = OpenAI(api_key=api_key, base_url=BASE_URL,
                             timeout=LLM_TIMEOUT, max_retries=0)
    return _llm_client


def _retry_request(url, *, timeout=15, max_retries=3, backoff=2, no_retry_statuses=()):
    """带指数退避的 HTTP GET 重试"""
    for attempt in range(max_retries):
        try:
            res = _session.get(url, timeout=timeout)
            if res.status_code in no_retry_statuses:
                res.raise_for_status()
            res.raise_for_status()
            return res
        except requests.RequestException as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code in no_retry_statuses:
                raise
            if attempt == max_retries - 1:
                raise
            wait = backoff ** attempt
            logger.warning(f"请求失败 ({url}), {wait}s 后重试: {e}")
            time.sleep(wait)


def _call_llm(*, messages, temperature, max_retries=3, backoff=5):
    """带指数退避的 LLM 调用重试，使用流式请求避免服务端空闲断连"""
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
            )
            if THINKING_MODE:
                kwargs["reasoning_effort"] = REASONING_EFFORT
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            else:
                kwargs["temperature"] = temperature

            stream = _get_llm_client().chat.completions.create(**kwargs)
            chunks = []
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)
            return "".join(chunks)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = backoff * (2 ** attempt)  # 5s, 10s, 20s
            logger.warning(f"LLM 调用失败, {wait}s 后重试: {e}")
            time.sleep(wait)


# ================= 缓存 =================
def _cache_path(arxiv_id, kind="html"):
    """返回缓存文件路径"""
    return os.path.join(CACHE_DIR, kind, f"{arxiv_id}.html")


def _read_cache(arxiv_id, kind="html"):
    """读取缓存，过期则删除并返回 None"""
    path = _cache_path(arxiv_id, kind)
    if not os.path.exists(path):
        return None
    age_days = (time.time() - os.path.getmtime(path)) / 86400
    if age_days > CACHE_EXPIRY_DAYS:
        os.remove(path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_cache(arxiv_id, content, kind="html"):
    """写入缓存"""
    path = _cache_path(arxiv_id, kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ================= 核心函数 =================
def get_beijing_now():
    return datetime.datetime.now(BEIJING_TZ)


def weekday_name(date_str):
    return WEEKDAY_NAMES[datetime.date.fromisoformat(date_str).weekday()]


def check_arxiv_update():
    """
    检查 arXiv astro-ph/new 页面日期是否与北京时间今天一致。
    返回 (是否更新, soup 对象, 信息字符串)。
    """
    url = f"https://arxiv.org/list/{ARXIV_CATEGORY}/new"
    response = _retry_request(url, timeout=30)
    soup = BeautifulSoup(response.text, "html.parser")

    today = get_beijing_now().date()
    arxiv_date = None

    # 寻找包含日期的 h3 标签
    h3 = soup.find("h3")
    if h3:
        date_text = h3.text.strip()
        match = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", date_text)
        if match:
            day, month_str, year = match.groups()
            month = MONTH_MAP.get(month_str.lower())
            if month is not None:
                arxiv_date = datetime.date(int(year), month, int(day))
    else:
        date_text = "未找到日期头 (h3)"

    # 备选：从论文 ID 推断年月
    if arxiv_date is None:
        first_link = soup.select_one("dl dt a[title=Abstract]")
        if first_link:
            id_match = re.match(r"(\d{2})(\d{2})\.", first_link.text.strip().replace("arXiv:", ""))
            if id_match:
                yy, mm = int(id_match.group(1)), int(id_match.group(2))
                if 2000 + yy == today.year and mm == today.month:
                    arxiv_date = today
                    date_text = f"从论文ID推断: {today}"

    if arxiv_date is None:
        return False, None, f"日期解析失败: {date_text}"

    if arxiv_date == today:
        return True, soup, date_text
    else:
        return False, None, f"页面日期({arxiv_date}) != 今天({today})"


def manage_existing_file(file_path):
    """如果目标文件已存在，备份为 .bak.md"""
    if os.path.exists(file_path):
        backup_path = file_path.replace(".md", ".bak.md")
        shutil.move(file_path, backup_path)
        logger.info(f"  已备份旧文件: {backup_path}")


SKIP_SECTION_KINDS = {"reference", "appendix", "acknowledgement"}


def _normalize_section_text(text):
    text = text.lower()
    text = re.sub(r"\\[a-z]+", " ", text)
    text = re.sub(r"^\s*(\d+(\.\d+)*|[ivxlcdm]+|[a-z])\s*[\).:-]?\s+", "", text, flags=re.I)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def _score_patterns(scores, kind, text, amount, patterns):
    for pattern in patterns:
        if re.search(pattern, text):
            scores[kind] = scores.get(kind, 0) + amount


def _classify_section(title, body_preview=""):
    title_text = _normalize_section_text(title)
    body_text = _normalize_section_text((body_preview or "")[:1200])
    scores = {}

    _score_patterns(scores, "reference", title_text, 6, [r"\b(references?|bibliography)\b"])
    _score_patterns(scores, "appendix", title_text, 6, [r"\bappendix\b"])
    _score_patterns(scores, "acknowledgement", title_text, 6, [
        r"\b(acknowledgements?|acknowledgments?|author contributions?|data availability|conflict of interest|orcid)\b",
    ])
    _score_patterns(scores, "conclusion", title_text, 5, [
        r"\b(conclusions?|summary|concluding remarks?|final remarks?)\b",
    ])
    _score_patterns(scores, "abstract", title_text, 5, [r"\babstract\b"])
    _score_patterns(scores, "related", title_text, 4, [
        r"\b(related work|previous work|literature review)\b",
    ])
    _score_patterns(scores, "introduction", title_text, 4, [
        r"\b(introduction|background|motivation|overview)\b",
    ])
    _score_patterns(scores, "data", title_text, 4, [
        r"\b(data|datasets?|samples?|observations?|survey|surveys|catalogs?|catalogues?|spectra|spectroscopic|photometry|photometric|imaging|images?|light curves?|data release)\b",
        r"\b(the\s+)?\w+\s+(catalog|catalogue|sample|survey)\b",
    ])
    _score_patterns(scores, "method", title_text, 4, [
        r"\b(methods?|methodology|approach|models?|modelling|modeling|algorithm|framework|pipeline|inference|calibration|estimator|likelihood|selection function|selection effects|forward model|training|architecture|network|reconstruction|synthesis|fitting)\b",
    ])
    _score_patterns(scores, "experiment", title_text, 4, [
        r"\b(experiments?|evaluation|benchmark|validation|tests?|setup|baselines?|ablation|comparison)\b",
    ])
    _score_patterns(scores, "result", title_text, 4, [
        r"\b(results?|findings?|measurements?|constraints?|performance|detections?|properties|estimates?)\b",
    ])
    _score_patterns(scores, "discussion", title_text, 4, [
        r"\b(discussion|implications?|interpretation|analysis)\b",
    ])
    _score_patterns(scores, "limitation", title_text, 4, [
        r"\b(limitations?|caveats?|uncertaint(y|ies)|systematics?|future work|robustness|biases?)\b",
    ])

    _score_patterns(scores, "data", body_text, 1, [
        r"\b(we use|we used|our sample|observed with|observations were|data release|catalogue|catalog|survey|spectra|photometry)\b",
    ])
    _score_patterns(scores, "method", body_text, 1, [
        r"\b(we model|we estimate|we infer|we train|we calibrate|algorithm|likelihood|pipeline|selection function|forward model)\b",
    ])
    _score_patterns(scores, "experiment", body_text, 1, [
        r"\b(we evaluate|we validate|benchmark|baseline|test set|simulation setup|experimental setup)\b",
    ])
    _score_patterns(scores, "result", body_text, 1, [
        r"\b(we find|we found|we measure|we measured|we show|our results|we obtain|we detect|we constrain|we report|improves?|outperforms?)\b",
    ])
    _score_patterns(scores, "discussion", body_text, 1, [
        r"\b(we discuss|this suggests|this implies|interpretation|implication)\b",
    ])
    _score_patterns(scores, "limitation", body_text, 1, [
        r"\b(uncertainty|uncertainties|limitation|limitations|caveat|caveats|systematic|systematics|bias|future work)\b",
    ])

    kinds = [kind for kind, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
    return kinds or ["other"]


def _section_rank(kinds, configured_priority=False):
    if configured_priority or any(k in ("abstract", "conclusion") for k in kinds):
        return 0
    if any(k in ("result", "experiment", "method", "data", "limitation", "discussion") for k in kinds):
        return 1
    if "other" in kinds:
        return 2
    if any(k in ("introduction", "related") for k in kinds):
        return 3
    return 2


def _compact_node_text(node, max_len=1600):
    return node.get_text(separator=" ", strip=True)[:max_len]


def _preserve_figure_table_text(soup):
    for tag in list(soup.select("figure, .ltx_figure, table, .ltx_table")):
        if not tag.parent:
            continue
        is_table = tag.name == "table" or "ltx_table" in tag.get("class", [])
        caption = tag.select_one("figcaption, caption, .ltx_caption")
        text = _compact_node_text(caption) if caption else _compact_node_text(tag)
        if not text:
            tag.decompose()
            continue
        replacement = soup.new_tag("p")
        replacement.string = f"{'Table text' if is_table else 'Figure caption'}: {text}"
        tag.replace_with(replacement)


def _strip_noise(soup):
    _preserve_figure_table_text(soup)
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()


def _extract_sections(soup):
    """
    从 arXiv HTML 论文中按章节提取内容。
    策略：按标题与正文预览自动分类，优先保留结果/方法/数据/实验/讨论/局限等高密度章节。
    如果总长度超限，仍保留一部分 other，Introduction/Related Work 优先级最低。
    返回 "## Section Title\ncontent\n\n..." 格式的文本。
    """
    _strip_noise(soup)

    # 查找所有章节标题 (h2, h3, h4)
    headers = soup.find_all(re.compile(r"^h[2-4]$"))
    if not headers:
        return None

    # 第一遍：收集所有有效章节
    all_sections = []
    for index, hdr in enumerate(headers):
        title = hdr.get_text(strip=True)
        title_lower = title.lower()

        if any(s in title_lower for s in SKIP_SECTIONS):
            continue

        content_parts = []
        for sibling in hdr.find_next_siblings():
            if sibling.name and re.match(r"^h[2-4]$", sibling.name):
                break
            text = sibling.get_text(separator=" ", strip=True)
            if text:
                content_parts.append(text)

        section_text = "\n".join(content_parts)[:SECTION_CHAR_LIMIT]
        if not section_text.strip():
            continue

        kinds = _classify_section(title, section_text)
        if any(k in SKIP_SECTION_KINDS for k in kinds):
            continue

        is_priority = (
            any(k in title_lower for k in PRIORITY_SECTIONS)
            or any(k in ("abstract", "conclusion") for k in kinds)
        )
        rank = _section_rank(kinds, is_priority)
        all_sections.append({
            "title": title,
            "text": section_text,
            "priority": is_priority,
            "rank": rank,
            "index": index,
        })

    if not all_sections:
        return None

    # 第二遍：先放入高优先级章节，再按原始顺序填充其余章节
    priority = [s for s in all_sections if s["priority"]]
    normal = sorted(
        [s for s in all_sections if not s["priority"]],
        key=lambda s: (s["rank"], s["index"]),
    )

    reserved = sum(len(s["text"]) for s in priority)
    budget = PAPER_CHAR_LIMIT - reserved

    selected = []
    total = 0
    for section in normal:
        text = section["text"]
        if total + len(text) > budget:
            # 预算不够，截断当前章节放入剩余空间
            remaining = budget - total
            if remaining > 500:
                selected.append({**section, "text": text[:remaining]})
                total += remaining
            continue
        selected.append(section)
        total += len(text)

    # 按原始顺序合并输出
    merged = selected + priority
    merged.sort(key=lambda s: s["index"])

    return "\n\n".join(f"## {s['title']}\n{s['text']}" for s in merged) if merged else None


def _extract_abstract_conclusion(soup):
    """
    从 arXiv HTML 论文中仅提取 Abstract + Conclusion/Summary 章节。
    返回 "## Abstract\n...\n\n## Conclusions\n..." 格式文本，或 None。
    """
    _strip_noise(soup)

    sections = []

    # Abstract 通常在 <div class="ltx_abstract"> 中，无 h2-h4 标题
    abstract_div = soup.find("div", class_="ltx_abstract")
    if abstract_div:
        text = abstract_div.get_text(separator=" ", strip=True)[:SECTION_CHAR_LIMIT]
        if text:
            sections.append(f"## Abstract\n{text}")

    # Conclusion/Summary 用 h2-h4 标题查找
    headers = soup.find_all(re.compile(r"^h[2-4]$"))
    conclusion_keywords = ["conclusion", "summary"]
    for hdr in headers:
        title = hdr.get_text(strip=True)
        title_lower = title.lower()
        if not any(k in title_lower for k in conclusion_keywords):
            continue

        content_parts = []
        for sibling in hdr.find_next_siblings():
            if sibling.name and re.match(r"^h[2-4]$", sibling.name):
                break
            text = sibling.get_text(separator=" ", strip=True)
            if text:
                content_parts.append(text)

        section_text = "\n".join(content_parts)[:SECTION_CHAR_LIMIT]
        if section_text.strip():
            sections.append(f"## {title}\n{section_text}")

    return "\n\n".join(sections) if sections else None


def fetch_paper_content(arxiv_id, is_detail=False):
    """
    抓取单篇论文内容。
    返回 (abstract_conclusion, full_sections)。
    - abstract_conclusion: 所有论文都有（Abstract + Conclusion 文本）
    - full_sections: is_detail=True 时提取所有有用章节，否则为 None
    调用方可为日报传入 is_detail=True，以便在不生成单篇详情时也利用正文摘录。
    """
    html_url = f"https://arxiv.org/html/{arxiv_id}"
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"

    # 尝试 HTML 全文（带缓存）
    try:
        cached_html = _read_cache(arxiv_id, "html")
        if cached_html:
            logger.info(f"    使用缓存: html/{arxiv_id}")
            html_text = cached_html
        else:
            res = _retry_request(html_url, no_retry_statuses=(404,))
            html_text = res.text
            _write_cache(arxiv_id, html_text, "html")

        # 提取 Abstract + Conclusion（用独立 soup，因为 decompose 会修改树）
        abstract_conclusion = _extract_abstract_conclusion(
            BeautifulSoup(html_text, "html.parser")
        )

        # 如果是 detail 论文，同时提取全文章节
        full_sections = None
        if is_detail:
            full_sections = _extract_sections(
                BeautifulSoup(html_text, "html.parser")
            )

        if abstract_conclusion:
            return abstract_conclusion, full_sections

        # abstract_conclusion 解析失败，回退到纯文本截取
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        plain = soup.get_text(separator="\n", strip=True)[:PAPER_CHAR_LIMIT]
        return plain, full_sections
    except Exception as e:
        logger.warning(f"    HTML 获取失败 ({arxiv_id}): {e}")

    # 回退到 Abstract 页面（带缓存）
    try:
        cached_abs = _read_cache(arxiv_id, "abs")
        if cached_abs:
            logger.info(f"    使用缓存: abs/{arxiv_id}")
            abs_text = cached_abs
        else:
            res = _retry_request(abs_url)
            abs_text = res.text
            _write_cache(arxiv_id, abs_text, "abs")

        soup = BeautifulSoup(abs_text, "html.parser")
        abstract_tag = soup.find("blockquote", class_="abstract")
        abstract = abstract_tag.text.replace("Abstract:", "").strip() if abstract_tag else "N/A"
        return f"## Abstract\n{abstract}", None
    except Exception as e:
        logger.error(f"    Abstract 获取也失败 ({arxiv_id}): {e}")
        return f"[获取失败] arXiv ID: {arxiv_id}", None


def parse_papers(soup):
    """
    从 arXiv 页面解析所有论文元数据（不做筛选、不抓取内容）。
    返回 list[dict]，每篇包含：id, title, authors, abstract
    """
    dl = soup.find("dl")
    if not dl:
        logger.warning("未找到论文列表 (dl 标签)")
        return []

    dt_list = dl.find_all("dt")
    dd_list = dl.find_all("dd")
    logger.info(f"发现 {len(dt_list)} 篇新论文")

    papers = []
    for dt, dd in zip(dt_list, dd_list):
        link = dt.find("a", title="Abstract")
        if not link:
            continue
        arxiv_id = link.text.replace("arXiv:", "").strip()

        # 提取标题
        title_div = dd.find("div", class_="list-title")
        title = title_div.text.replace("Title:", "").strip() if title_div else ""

        # 提取作者（第一作者 + et al.）
        authors_div = dd.find("div", class_="list-authors")
        if authors_div:
            author_links = authors_div.find_all("a")
            if author_links:
                first_author = author_links[0].text.strip()
                authors = f"{first_author} et al." if len(author_links) > 1 else first_author
            else:
                authors = authors_div.text.replace("Authors:", "").strip()
        else:
            authors = "Unknown"

        # 提取摘要（列表页的 mathjax 段落）
        abstract_p = dd.find("p", class_="mathjax")
        abstract = abstract_p.text.strip() if abstract_p else ""

        papers.append({
            "id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
        })

    return papers


def llm_filter_papers(papers):
    """
    用 LLM 一次性筛选所有论文，返回相关论文列表。
    每篇论文附带 is_detail 和 category 字段。
    """
    valid_categories = _configured_categories()
    category_options = "|".join(valid_categories)
    category_list = "\n".join(
        f"- {cat} → {CATEGORY_DISPLAY_MAP.get(cat, cat)}"
        for cat in valid_categories
    )
    category_values = ", ".join(valid_categories)

    # 构建论文列表文本
    papers_text = ""
    for p in papers:
        papers_text += (
            f"---\nID: {p['id']}\n"
            f"Title: {p['title']}\n"
            f"Abstract: {p['abstract']}\n"
        )

    system_prompt = f"""\
你是一位天体物理学研究者的助手。请根据研究兴趣，从下方论文列表中筛选出相关论文。

## 研究兴趣
{RESEARCH_INTERESTS}

## 详细收录标准
以下类型的论文应标记 detail: true（会生成详细报告）：
{DETAIL_CRITERIA}

## 可选 category
{category_list}

## 输出格式
请只输出一个 JSON 对象，不要输出任何其他内容：
{{"papers": [
  {{"id": "YYMM.NNNNN", "category": "{category_options}", "detail": true/false}},
  ...
]}}

规则：
- 只收录与研究兴趣相关的论文，不相关的直接忽略
- category 必须从 {category_values} 中选择最匹配的一个
- detail 判定要从严：只有论文的核心主题直接匹配详细收录标准时才设为 true。仅在摘要中提及相关概念但主题不同的论文，应设为 false
- 宁可漏选 detail 也不要错选——不确定时设为 false，日报已包含所有相关论文的总结
- 如果没有任何相关论文，返回 {{"papers": []}}"""

    user_content = f"以下是今日 arXiv astro-ph 的所有新论文：\n\n{papers_text}"

    # 构建 id → paper 的查找表
    paper_map = {p["id"]: p for p in papers}

    try:
        raw = _call_llm(messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ], temperature=0)
    except Exception as e:
        logger.error(f"  LLM 筛选失败: {e}，返回空列表")
        return []

    # 尝试提取 JSON（兜底：用正则匹配最外层大括号）
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError as e:
                logger.error(f"  JSON 解析失败: {e}")
                return []
        else:
            logger.error("  无法从 LLM 响应中提取 JSON")
            return []

    # 组装筛选结果
    filtered = []
    for item in result.get("papers", []):
        pid = item.get("id", "")
        if pid not in paper_map:
            logger.warning(f"  警告: LLM 返回的 ID {pid} 不在论文列表中，跳过")
            continue
        paper = dict(paper_map[pid])  # 复制一份
        paper["is_detail"] = bool(item.get("detail", False))
        category = str(item.get("category", "")).strip()
        if not category:
            category = "other" if "other" in valid_categories else valid_categories[0]
        if category not in valid_categories:
            logger.warning(f"  警告: LLM 返回未知 category {category}，跳过 {pid}")
            continue
        paper["category"] = category
        # 硬约束：只有指定 category 才允许 detail
        if paper["is_detail"] and paper["category"] not in DETAIL_CATEGORIES:
            paper["is_detail"] = False
            logger.info(f"  detail 降级 ({paper['category']} 不在 DETAIL_CATEGORIES): {pid}")
        label = "详细" if paper["is_detail"] else "日常"
        logger.info(f"  LLM 筛选[{label}|{paper['category']}]: {pid} | {paper['title'][:60]}")
        filtered.append(paper)

    logger.info(f"LLM 筛选完成: {len(filtered)}/{len(papers)} 篇相关")
    return filtered


def _configured_categories():
    """Return configured category keys in stable prompt/report order."""
    categories = []
    for source in (CATEGORY_DISPLAY_MAP, CATEGORY_TAG_MAP):
        for category in source.keys():
            if category and category not in categories:
                categories.append(category)
    for category in DETAIL_CATEGORIES:
        if category and category not in categories:
            categories.append(category)
    return categories or ["other"]


def _extract_section_titles(markdown_text):
    """Extract markdown section titles from paper excerpts."""
    if not markdown_text:
        return []
    titles = []
    for m in re.finditer(r"^##\s+(.+)$", markdown_text, re.M):
        title = m.group(1).strip()
        if title and title not in titles:
            titles.append(title)
    return titles


def _paper_source_sections(p):
    titles = []
    for title in _extract_section_titles(p.get("abstract_conclusion", "")):
        if title not in titles:
            titles.append(title)
    for title in _extract_section_titles(p.get("full_sections", "")):
        if title not in titles:
            titles.append(title)
    if titles:
        return ", ".join(titles)
    if p.get("abstract_conclusion", "").startswith("[获取失败]"):
        return "获取失败"
    return "正文摘录"


def _build_daily_content(p):
    parts = []
    abstract_conclusion = (p.get("abstract_conclusion") or "").strip()
    if abstract_conclusion:
        parts.append(abstract_conclusion)
    full_sections = (p.get("full_sections") or "").strip()
    if full_sections:
        parts.append(f"Full-text excerpts:\n{full_sections}")
    return "\n\n".join(parts)


def _build_paper_block(p):
    """构建单篇论文的信息文本块"""
    detail_mark = f" → [[{p['id']}]]" if p["is_detail"] else ""
    return (
        f"=== Paper: {p['id']} [category: {p.get('category', 'other')}]{detail_mark} ===\n"
        f"Title: {p['title']}\n"
        f"Authors: {p['authors']}\n"
        f"Source sections: {_paper_source_sections(p)}\n"
        f"{_build_daily_content(p)}\n\n"
    )


def _split_paper_batches(papers):
    """按 DAILY_CHAR_LIMIT 将论文分批，返回 list[list[dict]]"""
    batches = []
    current_batch = []
    current_size = 0
    for p in papers:
        block_size = len(_build_paper_block(p))
        if current_batch and current_size + block_size > DAILY_CHAR_LIMIT:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append(p)
        current_size += block_size
    if current_batch:
        batches.append(current_batch)
    return batches


def _call_daily_llm(papers, date_str, n_total, n_detail, is_partial=False):
    """单次日报 LLM 调用"""
    # 构建所有 category 的显示名称列表
    all_categories = _configured_categories()
    category_list = "\n".join(
        f"- {cat} → {CATEGORY_DISPLAY_MAP.get(cat, cat)}"
        for cat in all_categories
    )

    papers_info = "".join(_build_paper_block(p) for p in papers)

    partial_note = ""
    if is_partial:
        partial_note = (
            f"\n注意：这是分批处理的一部分（本批 {len(papers)} 篇），"
            f"请只为本批论文生成总结，不要输出标题头和统计行。\n"
        )

    header_fmt = "" if is_partial else (
        f"# arXiv astro-ph 每日追踪 {date_str}\n"
        f"共 {n_total} 篇相关论文，其中 {n_detail} 篇详细收录。\n\n"
    )

    system_prompt = f"""\
你是一个专业的天体物理学家助手。请根据提供的论文摘要、结论与可用正文摘录，生成 arXiv 每日论文追踪日报。

你的任务不是复述摘要，而是帮助研究者快速判断这篇论文的核心价值：它解决了什么具体问题、用了什么关键方法、得到什么证据、结论边界在哪里。

## Category 与显示名称对应关系
{category_list}
{partial_note}
请严格按照以下 Markdown 格式输出（不要输出 Markdown 代码块标记，直接输出内容）：

{header_fmt}## [显示名称]
### <实际论文标题> → [[YYMM.NNNNN]]
> 信息来源：<按输入的 Source sections 填写，例如 Abstract, Conclusion；不要编造>
- **作者**: First Author et al.
- **arXiv**: [ID](https://arxiv.org/abs/ID)
- **核心问题**: 论文试图解决的具体问题是什么，为什么值得研究（1句）
- **关键方法**: 作者用了什么方法、数据、模型、观测、模拟或理论工具（1-2句）
- **主要结果**: 优先写数值、误差、显著性、提升幅度、样本规模、参数范围、与前人/基线的对比；没有数值则写清作者声称的定性结果（1-2句）
- **为什么值得看**: 这篇论文具体改变了什么判断、解决了什么问题、约束了什么范围，或适用于什么场景（1句）
- **局限或边界**: 适用条件、不确定性、未覆盖的问题；若原文未说明，写"原文未说明"（1句）

注意：
- 所有论文（无论是否详细收录）都必须按上述完整格式输出，包含信息来源与五个核心字段，不得省略或只列标题
- 使用中文撰写，保留关键英文术语（如专有名词、物理量、巡天名称）
- 数学公式、物理量和符号必须使用 LaTeX 格式：行内用 $...$，独立公式用 $$...$$（如 $\\sigma_z = 0.03$、$M_{{500c}}$、$\\Omega_m$）
- 必须输出所有 category 的二级标题（使用上面的显示名称），如果某个 category 今日无论文，在标题下写"今日无相关论文更新。"
- 标题后带 → [[YYMM.NNNNN]] 的论文为详细收录论文（已在输入中标记），请保留此标记
- 未标记的论文不要加 [[]] 链接
- 先在内部判断论文属于方法、观测、理论、模拟、数据发布、综述等哪类，但不要输出类型；根据论文类型提取最核心的信息
- 只基于输入内容回答，不要引入外部知识，不要补全输入中没有说明的数据、实验、指标或结论
- 如果输入没有说明某项信息，请写"原文未说明"，不要猜测
- 如果输入只有摘要或摘要+结论，请按摘要级信息生成快速筛选摘要；如果输入包含正文结果、实验、方法或讨论章节，请优先使用这些高密度证据
- 区分作者已经用数据/实验/理论推导支持的结果和仅由作者声称的结果；证据细节不足时写"作者声称"
- 不要写"具有重要意义""提高了理解"这类空泛句子；每个价值判断必须说明具体改变了什么判断、约束了什么问题、或适用于什么场景"""

    return _call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"以下是今日筛选出的论文：\n\n{papers_info}"},
        ],
        temperature=LLM_TEMPERATURE,
    )


def summarize_daily(papers, date_str):
    """
    调用 LLM 生成日报总结。
    若总字符数超过 DAILY_CHAR_LIMIT，分批调用后拼接。
    """
    n_total = len(papers)
    n_detail = sum(1 for p in papers if p["is_detail"])

    total_chars = sum(len(_build_paper_block(p)) for p in papers)
    logger.info(f"  日报输入总长度: {total_chars} 字符 (限制: {DAILY_CHAR_LIMIT})")

    if total_chars <= DAILY_CHAR_LIMIT:
        return _call_daily_llm(papers, date_str, n_total, n_detail)

    # 分批处理
    batches = _split_paper_batches(papers)
    logger.info(f"  超过限制，分 {len(batches)} 批处理 ({[len(b) for b in batches]})")

    header = (
        f"# arXiv astro-ph 每日追踪 {date_str}\n"
        f"共 {n_total} 篇相关论文，其中 {n_detail} 篇详细收录。\n"
    )
    parts = [header]
    for i, batch in enumerate(batches):
        logger.info(f"  正在处理第 {i+1}/{len(batches)} 批 ({len(batch)} 篇)...")
        part = _call_daily_llm(batch, date_str, n_total, n_detail, is_partial=True)
        parts.append(part)

    return "\n\n".join(parts)


def summarize_paper_detail(paper, date_str):
    """调用 LLM 为单篇 detail 论文生成详细报告"""
    system_prompt = f"""\
你是一个专业的天体物理学家助手。请根据提供的论文各章节内容，生成一篇详细的中文论文总结。

你的任务不是复述摘要，而是还原论文的贡献链条：研究问题 -> 方法设计 -> 关键证据 -> 主要结论 -> 适用边界。

请严格按照以下 Markdown 格式输出（不要输出 Markdown 代码块标记，不要输出 YAML frontmatter，直接从 # 标题开始）：

# {paper['title']}

- **arXiv**: [{paper['id']}](https://arxiv.org/abs/{paper['id']})

## 研究问题
论文要解决的具体问题是什么？为什么这个问题值得研究？

## 方法设计
作者采用了什么核心方法、模型、数据、实验、观测、模拟或理论框架？

## 关键证据
作者用什么证据支持结论？优先保留数值、样本规模、误差、显著性、参数范围、基线对比或实验设置。

## 主要结论
论文最核心的发现或贡献是什么？区分作者已经证明的结果和作者提出的解释。

## 适用边界
结论在哪些条件下成立？有哪些限制、不确定性或未覆盖的问题？

## 一句话价值判断
用一句话说明这篇论文最值得关注的点，避免空泛评价。

注意：
- 使用中文撰写
- 保留关键英文术语（如专有名词、物理量）
- 数学公式、物理量和符号必须使用 LaTeX 格式：行内用 $...$，独立公式用 $$...$$（如 $\\sigma_z = 0.03$、$M_{{500c}}$、$\\Omega_m$）
- 只基于输入内容回答，不要引入外部知识，不要补全输入中没有说明的数据、实验、指标或结论
- 如果某项信息在输入中没有说明，请写"原文未说明"
- 先在内部判断论文属于方法、观测、理论、模拟、数据发布、综述等哪类，但不要输出类型；根据论文类型组织重点
- 优先提取数值、误差、显著性、提升幅度、样本规模、参数范围、与前人/基线的对比
- 区分作者已经用数据/实验/理论推导支持的结果和作者提出的解释；证据细节不足时写"作者声称"
- 不要写"具有重要意义""提高了理解"这类空泛句子；每个价值判断必须说明具体改变了什么判断、约束了什么问题、或适用于什么场景"""

    user_content = (
        f"论文 ID: {paper['id']}\n"
        f"标题: {paper['title']}\n"
        f"作者: {paper['authors']}\n\n"
        f"以下是论文各章节内容：\n\n{paper['full_sections']}"
    )

    return _call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=LLM_TEMPERATURE,
    )


def _generate_paper_tags(paper):
    """根据 LLM 返回的 category 生成 tags"""
    tags = ["arxiv", "paper"]
    category = paper.get("category", "")
    tag = CATEGORY_TAG_MAP.get(category)
    if tag:
        tags.append(tag)
    return tags


def wait_until_930():
    """如果当前北京时间早于 9:30，则等待到 9:30"""
    now = get_beijing_now()
    cutoff = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now < cutoff:
        wait_seconds = (cutoff - now).total_seconds()
        logger.info(f"当前北京时间 {now.strftime('%H:%M:%S')}，等待到 9:30 再开始检查...")
        time.sleep(wait_seconds)


def poll_arxiv_update():
    """
    从 9:30 开始，每 30 分钟检查一次 arXiv 是否更新为当日内容。
    更新后返回 (soup, msg)，超过最大重试次数则返回 None。
    """
    for attempt in range(1, MAX_RETRIES + 1):
        now = get_beijing_now()
        logger.info(f"[第 {attempt}/{MAX_RETRIES} 次检查] {now.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            is_updated, soup, msg = check_arxiv_update()
        except Exception as e:
            logger.warning(f"  检查出错: {e}")
            if attempt < MAX_RETRIES:
                logger.info(f"  {POLL_INTERVAL // 60} 分钟后重试...")
                time.sleep(POLL_INTERVAL)
            continue

        if is_updated:
            logger.info(f"  arXiv 已更新: {msg}")
            return soup, msg

        logger.info(f"  未更新: {msg}")
        if attempt < MAX_RETRIES:
            logger.info(f"  {POLL_INTERVAL // 60} 分钟后重试...")
            time.sleep(POLL_INTERVAL)

    logger.warning("已达最大重试次数，今日放弃。")
    return None


def main():
    now = get_beijing_now()
    logger.info(f"脚本启动: {now.strftime('%Y-%m-%d %H:%M:%S')} (北京时间)")

    # 1. 等待到 9:30
    wait_until_930()

    # 2. 轮询 arXiv 更新
    result = poll_arxiv_update()
    if result is None:
        sys.exit(0)

    soup, msg = result

    today_str = get_beijing_now().date().strftime("%Y-%m-%d")
    daily_file = os.path.join(DAILY_DIR, f"{today_str}.md")

    # 3. 确保目录存在 & 备份旧文件
    os.makedirs(DAILY_DIR, exist_ok=True)
    os.makedirs(PAPERS_DIR, exist_ok=True)
    manage_existing_file(daily_file)

    # 4. 解析所有论文元数据（不过滤）
    all_papers = parse_papers(soup)

    if not all_papers:
        logger.info("今日未发现论文。")
        header = f"---\ndate: {today_str}\nweekday: {weekday_name(today_str)}\ntags: [arxiv, daily]\n---\n\n"
        body = f"# arXiv astro-ph 每日追踪 {today_str}\n\n今日未发现论文。\n"
        with open(daily_file, "w", encoding="utf-8") as f:
            f.write(header + body)
        logger.info(f"空日报已保存: {daily_file}")
        return

    # 5. LLM 筛选相关论文
    logger.info(f"\n共 {len(all_papers)} 篇论文，开始 LLM 筛选...")
    filtered_papers = llm_filter_papers(all_papers)

    if not filtered_papers:
        logger.info("LLM 筛选后无相关论文，跳过日报生成。")
        return

    detail_papers = [p for p in filtered_papers if p["is_detail"]]
    logger.info(f"筛选结果: {len(filtered_papers)} 篇相关，其中 {len(detail_papers)} 篇详细收录。")

    # 6. 对筛选出的论文抓取内容
    logger.info("正在抓取论文内容...")
    for p in filtered_papers:
        time.sleep(REQUEST_DELAY)
        try:
            # 日报也尽量利用可获取的正文摘录；is_detail 仍只控制是否生成单篇详情。
            ac, fs = fetch_paper_content(p["id"], True)
            p["abstract_conclusion"] = ac
            p["full_sections"] = fs
        except Exception as e:
            logger.error(f"  抓取 {p['id']} 失败: {e}")
            p["abstract_conclusion"] = f"[获取失败] arXiv ID: {p['id']}"
            p["full_sections"] = None

    # 7. 日报总结
    logger.info("正在生成日报总结...")
    daily_summary = summarize_daily(filtered_papers, today_str)

    header = f"---\ndate: {today_str}\nweekday: {weekday_name(today_str)}\ntags: [arxiv, daily]\n---\n\n"
    with open(daily_file, "w", encoding="utf-8") as f:
        f.write(header + daily_summary)
    logger.info(f"日报已保存: {daily_file}")

    # 8. 详细论文报告（逐篇）
    for p in detail_papers:
        if not p["full_sections"]:
            logger.warning(f"  跳过 {p['id']}（无法提取全文章节）")
            continue

        logger.info(f"正在生成详细报告: {p['id']}...")
        paper_file = os.path.join(PAPERS_DIR, f"{p['id']}.md")
        manage_existing_file(paper_file)

        try:
            detail_summary = summarize_paper_detail(p, today_str)
        except Exception as e:
            logger.error(f"  详细报告生成失败 ({p['id']}): {e}")
            continue

        tags = _generate_paper_tags(p)
        frontmatter = (
            f"---\n"
            f"title: \"{p['title']}\"\n"
            f"authors: \"{p['authors']}\"\n"
            f"arxiv: \"{p['id']}\"\n"
            f"date: {today_str}\n"
            f"weekday: {weekday_name(today_str)}\n"
            f"tags: [{', '.join(tags)}]\n"
            f"---\n\n"
        )
        with open(paper_file, "w", encoding="utf-8") as f:
            f.write(frontmatter + detail_summary)
        logger.info(f"  详细报告已保存: {paper_file}")

    # 9. 统计
    logger.info(f"\n完成！共 {len(filtered_papers)} 篇相关，其中 {len(detail_papers)} 篇详细收录。")


if __name__ == "__main__":
    main()
