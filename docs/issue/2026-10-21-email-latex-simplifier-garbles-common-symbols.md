# 邮件摘要里常见的 LaTeX 记号被拼成乱码单词：`M_\odot` 变成 `M_odot`，`\mathcal{O}(n)` 变成 `mathcalO(n)`，`\ge` 变成 `ge`

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 MAIL-07
> 范围：日报邮件正文（HTML 与纯文本）中标题、作者、五个摘要字段的公式简化 `emailProse` · 严重程度：低 · 状态：未修复 · 置信度：已复现

## 问题

邮件里没有 MathJax，`emailProse` 把 `$…$` 里的 LaTeX 简化成纯文本。符号表只覆盖一部分命令；没有覆盖的命令，最后一步会**去掉反斜杠并保留命令名**，于是命令名和后面的参数粘在一起，读者看到的是一个不存在的单词。已复现（一次性测试，已删除；左边是邮件里实际得到的结果）：

| 输入 | 邮件里的结果 |
| --- | --- |
| `$M_\star > 10^{10}\,M_\odot$` | `M_star > 10^10 M_odot` |
| `$50^\circ$`、`$T_\oplus$` | `50^circ`、`T_oplus` |
| `$F \ge 3\sigma$`、`$F \le 1$`、`$\Omega_m \ne 0.3$` | `F ge 3σ`、`F le 1`、`Ω_m ne 0.3` |
| `$\dot{M} \ll 1$`、`$x \gg 1$`、`$\forall i$` | `dotM ll 1`、`gg 1`、`forall i` |
| `$\mathcal{O}(n \log n)$`、`$\mathbb{R}^d$` | `mathcalO(n log n)`、`mathbbR^d` |
| `$\sqrt{d}$`、`$\hat{\theta}$`、`$\bar{x}$`、`$\vec{v}$`、`$\tilde{y}$` | `sqrtd`、`hatθ`、`barx`、`vecv`、`tildey` |
| `$\left( \frac{a}{b} \right)^2$` | `left((a)/(b) right)^2` |
| `$\langle x \rangle$` | `langle x rangle` |
| `\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}` | `beginpmatrix 1 & 2 3 & 4 endpmatrix` |

`\le`/`\ge`/`\ne` 这几个最常用的写法没有映射（只映射了 `\leq`、`\geq`、`\neq`）；`\odot`、`\star`、`\circ`、`\oplus`、`\ll`、`\gg` 在天文类论文里几乎每篇都有。项目自带的测试邮件样例就是 astro-ph 的公式探测，目标读者显然包括这类用户。

影响：邮件是日报的“手机版投影”，用户在邮件里读摘要时会看到 `M_odot`、`ll 1`、`mathcalO(n)` 这类词，不均衡的符号可能让不等式方向和含义变得难以判断（`F le 1`、`ge 3σ`）。完整日报（vault 里的 Markdown）不受影响。

## 原因

- `packages/core/src/delivery/email-render.ts:268-339`：符号表缺少 `\le \ge \ne \ll \gg \odot \star \circ \oplus \forall \exists \langle \rangle` 等常用命令。
- `email-render.ts:362-363` `GROUP_CMD_OPEN`：只展开 `text/mathrm/mathbf/mathit/textrm/textit/textbf/operatorname/mathsf/mathtt`，没有 `mathcal/mathbb/mathscr/mathfrak/boldsymbol/bm`。
- `email-render.ts:349-353`：对所有其余命令执行 `replace(/\\([a-zA-Z]+)/g, "$1")` 去掉反斜杠，再删掉花括号，命令名就与参数连在一起（`\sqrt{d}` → `sqrtd`）。`\left`、`\right`、`\hat`、`\bar`、`\vec`、`\tilde`、`\dot`、`\begin`/`\end` 都走这条路径。

## 可能的解决方案

推荐：

1. 补充符号映射：`\le ≤`、`\ge ≥`、`\ne ≠`、`\ll ≪`、`\gg ≫`、`\odot ⊙`、`\star ⋆`、`\oplus ⊕`、`\circ °`、`\forall ∀`、`\exists ∃`、`\langle ⟨`、`\rangle ⟩`、`\mathbb{R} ℝ`（同理 `ℕ ℤ ℚ ℂ`）。
2. 对纯排版命令直接丢弃命令名、保留参数：`\left`、`\right`、`\big*`、`\displaystyle`、`\mathcal`、`\mathbb`、`\mathscr`、`\mathfrak`、`\boldsymbol`、`\bm`、`\hat`、`\bar`、`\vec`、`\tilde`、`\widetilde`、`\dot`、`\ddot`、`\overline`、`\underline`；`\sqrt{x}` 输出 `√(x)`。
3. 对仍然未知的命令，不要再把命令名拼到参数上：保留反斜杠原样输出（`\foo{x}`），读者至少知道这是 TeX，而不是一个英文单词；`\begin{…}…\end{…}` 环境整体按原样保留。

备选：只做第 3 点，作为最低限度的“不制造假单词”保险。

## 修复后应补的测试

- 上表每一行作为表驱动测试，断言新的预期结果。
- 不变量测试：对随机的 `$…$` 片段，输出中不应出现“已知 TeX 命令名紧跟字母或数字”的粘连（例如 `/(mathcal|mathbb|sqrt|hat|bar|vec|tilde|left|right)[A-Za-z0-9(]/`）。
- 现有“emailProse simplifies common LaTeX”“display/block math delimiters”“completes quickly on repeated unterminated delimiters”测试保持通过。
