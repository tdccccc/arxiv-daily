#!/usr/bin/env python3
"""测试 arxiv_daily.py 的核心流程（跳过时间等待和日期检查）"""

import requests
from bs4 import BeautifulSoup

# 导入主脚本的所有函数和配置
from arxiv_daily import (
    HEADERS, parse_papers, llm_filter_papers, fetch_paper_content,
    REQUEST_DELAY,
)
import time


def test():
    # 1. 抓取 arXiv 页面（不检查日期）
    print("=" * 60)
    print("Step 1: 抓取 arXiv astro-ph/new 页面")
    print("=" * 60)
    url = "https://arxiv.org/list/astro-ph/new"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    h3 = soup.find("h3")
    print(f"页面日期: {h3.text.strip() if h3 else '未找到'}")

    # 2. 测试 parse_papers（应返回所有论文，无过滤）
    print("\n" + "=" * 60)
    print("Step 2: parse_papers — 解析所有论文元数据")
    print("=" * 60)
    all_papers = parse_papers(soup)
    print(f"解析到 {len(all_papers)} 篇论文")

    if all_papers:
        print(f"\n前 3 篇示例:")
        for p in all_papers[:3]:
            print(f"  ID: {p['id']}")
            print(f"  Title: {p['title'][:80]}")
            print(f"  Authors: {p['authors']}")
            print(f"  Abstract: {p['abstract'][:100]}...")
            print()

    # 3. 测试 llm_filter_papers
    print("=" * 60)
    print("Step 3: llm_filter_papers — LLM 筛选")
    print("=" * 60)
    filtered = llm_filter_papers(all_papers)
    print(f"\n筛选结果: {len(filtered)} 篇相关")

    if filtered:
        detail_count = sum(1 for p in filtered if p["is_detail"])
        print(f"  其中 {detail_count} 篇详细收录")
        for p in filtered:
            label = "详细" if p["is_detail"] else "日常"
            print(f"  [{label}|{p['category']}] {p['id']} | {p['title'][:60]}")

    # 4. 测试内容抓取：优先测详细论文 + 1 篇日常论文
    if filtered:
        detail = [p for p in filtered if p["is_detail"]]
        normal = [p for p in filtered if not p["is_detail"]]
        test_papers = detail[:1] + normal[:1]  # 1 篇详细 + 1 篇日常

        print(f"\n{'=' * 60}")
        print(f"Step 4: fetch_paper_content — 测试抓取 {len(test_papers)} 篇（详细+日常各1）")
        print("=" * 60)
        for p in test_papers:
            time.sleep(REQUEST_DELAY)
            ac, fs = fetch_paper_content(p["id"], p["is_detail"])
            p["abstract_conclusion"] = ac
            p["full_sections"] = fs
            print(f"\n  {p['id']} (detail={p['is_detail']}):")
            print(f"    abstract_conclusion: {len(ac or '')} chars")
            if fs:
                print(f"    full_sections: {len(fs)} chars")
                print(f"    full_sections 前 200 字: {fs[:200]}...")
            else:
                print(f"    full_sections: None")

    print(f"\n{'=' * 60}")
    print("测试完成！")


if __name__ == "__main__":
    test()
