# P17 — Scientific Markdown reading

goal_ref: ../goal.md
created: 2026-10-05T11:07:30+08:00
updated: 2026-10-05T11:17:00+08:00
revision: 2

## Outcome

DSH工作台论文概览、摘要、推荐理由及Markdown正文一致显示Markdown/LaTeX；复用既有安全渲染器与离线KaTeX资源。

## Assumptions

- 当前概览/列表只escape文本，完整Markdown已有KaTeX；用户具体样例待补充，不据此假定唯一故障。
- 不改研究原文，不使用Computer Use，不调用真实模型/邮件。

## Chunks

1. [x] Strict Red→Green：DOM复现概览/列表公式原样显示，复用纯Markdown渲染模块；文字属性/用户输入继续转义，限制列表标题中的交互标签。
2. [x] Strict Red→Green：测试常见公式分隔符、多行display、代码/货币/错误公式回退；仅修已复现解析缺口。
3. [x] 构建资源/HTTP合同验证本地KaTeX CSS和字体；相关CLI回归、typecheck与DSH隔离Host通过后打包。

## Verification

DOM与字符串渲染回归，恶意HTML/链接合同，复制安装包实际Host的CSS/font HTTP检查。无Electron视觉验证时明确说明。

## Abort / reshape triggers

若用户样例显示其他入口或语法，保留已验证修复并继续定位；不把所有反斜杠文本当作公式，不把整个页面作为HTML信任。

## Accepted result

- User confirmed raw $/LaTeX text was displayed. Reproduced plain-text rendering in list/overview and missing cross-blank-line display math; heading metadata also lost math boundaries. All now reuse the existing pure Markdown/KaTeX reader (shared browser/server module), without rewriting stored source or invoking a model.
- Added safe inline projection for titles/snippets, Markdown blocks for overview summaries/abstracts/novelty, preserved heading math for titles/TOC, and prevented duplicate mathematical first headings through parser metadata rather than KaTeX textContent.
- Standalone $$ and backslash-bracket blocks support blank lines, aligned/matrix, lists and quotes. Inline dollar/backslash-parentheses, currency, escaped syntax, fences/inline code and malformed readable fallback retain regression coverage. Arbitrary source HTML and trusted TeX commands remain disabled.
- Observed Red→Green: overview/list lacked KaTeX DOM; display blocks/inline helper missing; heading math was lost/unrendered; mathematical heading duplicated. One early UI helper expected the old fixture title, corrected before accepting behavioral Red.
- Green:63 tests in7 Markdown/UI/HTTP suites; final title correction38 tests in3 suites; CLI typecheck, boundary check and build pass. DSH20 tests passed with zero skipped; both actual packed Host scenarios rerun on final artifact2/2. Host verifies local katex.css, self font CSP and every referenced WOFF2 font signature. No Electron visual walkthrough, Computer Use, real model/email or user-vault writes.
- DSH0.1.15 archive: extensions/dsh-arxiv-daily/dist/dsh-arxiv-daily-0.1.15.tgz (linux/x64), sha1 e64b34963ed3897a7476be4864b5880853ec24bf. User installation was not changed; install/restart to use it. Existing reports need no regeneration.

Implementation commits17ca820 and0f79e90. P2/P3/P4 remain pending outside this display fix.
