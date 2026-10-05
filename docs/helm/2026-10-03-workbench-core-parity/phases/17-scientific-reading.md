# P17 — Scientific Markdown reading

goal_ref: ../goal.md
created: 2026-10-05T11:07:30+08:00
updated: 2026-10-05T11:07:30+08:00
revision: 1

## Outcome

DSH工作台论文概览、摘要、推荐理由及Markdown正文一致显示Markdown/LaTeX；复用既有安全渲染器与离线KaTeX资源。

## Assumptions

- 当前概览/列表只escape文本，完整Markdown已有KaTeX；用户具体样例待补充，不据此假定唯一故障。
- 不改研究原文，不使用Computer Use，不调用真实模型/邮件。

## Chunks

1. Strict Red→Green：DOM复现概览/列表公式原样显示，复用纯Markdown渲染模块；文字属性/用户输入继续转义，限制列表标题中的交互标签。
2. Strict Red→Green：测试常见公式分隔符、多行display、代码/货币/错误公式回退；仅修已复现解析缺口。
3. 构建资源/HTTP合同验证本地KaTeX CSS和字体；相关CLI回归、typecheck与DSH隔离Host通过后打包。

## Verification

DOM与字符串渲染回归，恶意HTML/链接合同，复制安装包实际Host的CSS/font HTTP检查。无Electron视觉验证时明确说明。

## Abort / reshape triggers

若用户样例显示其他入口或语法，保留已验证修复并继续定位；不把所有反斜杠文本当作公式，不把整个页面作为HTML信任。
