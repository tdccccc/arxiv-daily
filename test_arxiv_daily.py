#!/usr/bin/env python3
"""测试 arxiv_daily.py 的核心流程（跳过时间等待和日期检查）"""

from bs4 import BeautifulSoup

# 导入主脚本的所有函数和配置
from arxiv_daily import (
    parse_papers, fetch_paper_content,
    REQUEST_DELAY, logger, _retry_request, _call_llm,
)
import time


def test():
    # 1. 抓取 arXiv 页面（不检查日期）
    logger.info("=" * 60)
    logger.info("Step 1: 抓取 arXiv astro-ph/new 页面")
    logger.info("=" * 60)
    url = "https://arxiv.org/list/astro-ph/new"
    response = _retry_request(url, timeout=30)
    soup = BeautifulSoup(response.text, "html.parser")

    h3 = soup.find("h3")
    logger.info(f"页面日期: {h3.text.strip() if h3 else '未找到'}")

    # 2. 测试 parse_papers（应返回所有论文，无过滤）
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2: parse_papers — 解析所有论文元数据")
    logger.info("=" * 60)
    all_papers = parse_papers(soup)
    logger.info(f"解析到 {len(all_papers)} 篇论文")

    if all_papers:
        logger.info("\n前 3 篇示例:")
        for p in all_papers[:3]:
            logger.info(f"  ID: {p['id']}")
            logger.info(f"  Title: {p['title'][:80]}")
            logger.info(f"  Authors: {p['authors']}")
            logger.info(f"  Abstract: {p['abstract'][:100]}...")
            logger.info("")

    # 3. 测试 LLM 连通性（轻量调用，不消耗大量 token）
    logger.info("=" * 60)
    logger.info("Step 3: LLM 连通性测试（询问模型名称）")
    logger.info("=" * 60)
    try:
        reply = _call_llm(
            messages=[{"role": "user", "content": "你是什么模型？请用一句话回答。"}],
            temperature=0,
        )
        logger.info(f"  LLM 回复: {reply}")
    except Exception as e:
        logger.error(f"  LLM 调用失败: {e}")

    # 4. 测试内容抓取 + LLM 总结：取第 1 篇论文
    if all_papers:
        p = all_papers[0]
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Step 4: fetch + LLM 总结 — {p['id']}")
        logger.info("=" * 60)

        time.sleep(REQUEST_DELAY)
        ac, _ = fetch_paper_content(p["id"], is_detail=False)
        logger.info(f"  abstract_conclusion: {len(ac or '')} chars")

        try:
            summary = _call_llm(
                messages=[
                    {"role": "system", "content": "请用中文 3-5 句话总结这篇论文的核心内容。"},
                    {"role": "user", "content": f"标题: {p['title']}\n\n{ac}"},
                ],
                temperature=0.3,
            )
            logger.info(f"  LLM 总结:\n{summary}")
        except Exception as e:
            logger.error(f"  LLM 总结失败: {e}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("测试完成！")


if __name__ == "__main__":
    test()
