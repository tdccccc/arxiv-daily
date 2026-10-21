# 工作台设置自动保存、`library connect`、接受方向都会重写整个 `config.toml`，`init` 生成的说明和用户自己的注释全部丢失

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 CLI-04
> 范围：CLI 配置文件写入（工作台设置、`library connect/authorize/revoke`、库方向接受） · 严重程度：低 · 状态：未修复 · 置信度：已复现

## 问题

ADR 0003 把 `config.toml` 定位为"用户可以直接编辑的唯一配置文件"：`init` 生成的文件带有完整的英文注释（编辑提示、"May contain secrets — do not commit or share this file publicly"、`[schedule]` 与 `schedule install` 的关系、`schema_version` 的说明），README 和 `init` 结尾也都提示用户手动改 `[[arxiv.topics]]`、`[schedule]`。

但有三条写入路径用 `smol-toml` 的 `parse` → `stringify` 整份重写文件，TOML 注释在解析时就被丢弃，写回后不再存在：

- 工作台设置里改任何一项（自动保存）；
- `arxiv-daily library connect|authorize|revoke`（工作台"连接文献库"同路径）；
- 接受文献库提出的方向（`library confirm` / `library review`，工作台"方向审核"）。

实测（用真实构建产物 + 真实界面）：

1. 配置里有头部说明、`# my note: switch to gpt-x on Fridays`、`[schedule]` 下的注释。打开工作台设置，把"每日论文上限"改成 25，自动保存完成后，`config.toml` 里一行注释都不剩；
2. 同样的文件，只执行 `arxiv-daily library connect <目录>`，结果相同：所有注释消失，只多了 `[library]` 表。

注释丢失之后，用户留给自己的备注（例如备用模型、临时关掉的选项的原因）没了；"含密钥、勿提交"的提醒也一并消失；`[schedule]`（外部 cron 用）与 `[workbench_schedule]`（工作台进程内定时器）的区别只存在于模板注释里，丢失后文件本身不再说明。

次要：工作台保存总是把 `vault_root` 重写成展开后的绝对路径（`settings.ts` 里无条件 `vault_root: vaultRoot`），手写成 `~/my-vault` 的值第一次保存后变成 `/home/<用户>/my-vault`。`init` 自己写的就是绝对路径，所以只影响手写 `~` 的人；如果这份配置通过 dotfiles 同步到另一个用户名不同的机器，保存过一次之后就指向了错误的目录。

## 原因

- `apps/cli/src/workbench/settings.ts:59-63`：`const document = parse(raw)` → `const updated = { ...document, vault_root: vaultRoot }` → `stringify(updated)`；整个文档对象重新序列化。
- `apps/cli/src/library-connection-cmd.ts:106`：`stringify({ ...parse(raw), library: next })`。
- `apps/cli/src/library-topic-settings.ts:43-46`：`stringify({ ...document, arxiv: {…, topics: next.topics}, library_proposal_acceptances: … })`。
- 三处都已经持有 `cli-config:<文件名>` 锁并校验过 revision，写入本身是安全的，缺的只是"保留未改动内容"的编辑方式。

## 可能的解决方案

推荐：抽出一个共享的"补丁式写配置"函数，用保留注释和格式的 TOML 编辑库（例如 `@decimalturn/toml-patch`，MIT；或类似的 CST 编辑库）对原文做最小修改，只改变确实变化的键，上面三处统一调用。`vault_root` 只在用户确实改了该字段时才写回。

备选：保持现有序列化，但在保存前把原文件复制为 `config.toml.bak`（0600，覆盖上一份）。这能找回注释，但不能避免每次保存都"重排"文件，也让密钥多一份副本。

## 修复后应补的测试

- 带注释的配置：工作台保存一个无关字段后，注释、未改动键的顺序和格式不变，只有被改的值变化。
- `library connect` / 接受方向之后，原有注释保留，新增的 `[library]`、`library_proposal_acceptances` 追加在文件末尾。
- 手写 `vault_root = "~/x"` 保存无关字段后保持 `~/x`。
- 并发写入（锁和 revision 检查）行为不变。
