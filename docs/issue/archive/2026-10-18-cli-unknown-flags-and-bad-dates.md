# CLI 忽略未知参数、不校验 `--date` / `--id`：拼错参数会触发真实运行，非法日期污染运行状态

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 CLI-6
> 范围：CLI 参数解析（run / email / data / schedule） · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 995d0ad） · 置信度：已复现

## 问题

- `run --today --dry-run`（或任何拼错的参数）会静默执行一次真实运行，产生 LLM 费用和网络请求。
- `data import a.zip --yses` 静默丢弃拼错的参数；第二个位置参数被忽略；`-y` 被当作 zip 路径。
- `run --date garbage`、`run --date 2026-02-30` 被接受。调度器随后以非法日期为键写入运行状态/历史（`garbage` 记为 `failed_transient`，`2026-02-30` 在一次网络抓取后记为 `failed_permanent`），这些记录出现在 `status` 的最近运行和工作台日历/历史里。工作台路径（`server.ts parseTask`）会校验日期，CLI 路径不会。

## 原因

- `apps/cli/src/main.ts:389-446`：`parseCli` 对 run / email / data / schedule 用 `includes()` / `optionValue` 查找，而不是严格遍历参数。
- 日期格式与日历有效性只在 `MarkdownWriter.dailyPath` 中较晚才检查（仅正则），此时调度器已记录了这次尝试。

## 可能的解决方案

推荐：每个子命令拒绝未知参数和多余位置参数（以用法错误退出，退出码 2）；在 `parseCli` 中把 `--date` 校验为真实日历日期（复用 `parseTask` 的检查），`--id` 用现有 arXiv 编号校验。

代价：依赖多余参数的脚本会失败，可以接受，退出码 2 本就是文档中的用法错误码。

## 修复后应补的测试

- `run --today --bogus` → 退出码 2，不运行。
- `run --date 2026-02-30`、`--date garbage` → 退出码 2。
- `data import a.zip --yses` → 退出码 2。
- 现有合法参数组合全部仍能解析。

## 修复情况

新增 `cli-args.ts`，按每个子命令的参数规格逐个检查：未知、重复或多余参数以退出码 2 结束并打印用法；`--date` 必须是真实日历日期，`--id` 必须是合法 arXiv 编号；`data import` 支持 `-y`。工作台复用同一日期校验。
