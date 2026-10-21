# 扫描文献库时识别 PDF 身份会先把压缩流完整解压进内存再检查大小，一个很小的异常 PDF 就能耗尽内存

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 LIB-05
> 范围：个人文献库扫描（对没有 arXiv 编号的 PDF 做内容识别，插件、CLI 共用） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch1`，提交 8abeba9） · 置信度：已复现

## 问题

对文件名没有 arXiv 编号的 PDF，扫描会读取文件头 4 MiB 和尾 1 MiB，把其中所有 FlateDecode 流解压后找 arXiv 编号和标题。代码想用 `MAX_INFLATED_STREAM_BYTES`（8 MiB）和 `MAX_TOTAL_INFLATED_BYTES`（32 MiB）限制解压量，但这两个上限都是在整个流已经解压完成之后才检查：压缩前的长度只限制到 8 MiB，真正的解压输出没有上限。

一个压缩比很高的流就能绕过这些限制。实测：一个 261,006 字节的 PDF（里面一个流解压后是 256 MiB 的零）让识别函数同步运行 0.6 秒，进程峰值内存增长约 517 MiB；识别窗口是 4 MiB，deflate 最高约 1000:1，理论上一个窗口内的流可解压到数 GiB。在 Obsidian 里这发生在渲染进程，同步解压会冻结界面，内存耗尽会让整个 Obsidian 崩溃；CLI 则是进程被系统杀掉。扫描会逐个处理库里的所有 PDF，所以一个来路不明的 PDF（下载的、别人发来的）或者一个含超大空白图像的正常 PDF 就能让每次扫描都崩溃。

注意这一步发生在扫描阶段，不需要任何模型授权，而且文件不变时识别结果会被缓存，崩溃发生时还没有保存目录，下次扫描会再次读到同一个文件。

## 原因

- `packages/core/src/library/pdf-identification-evidence.ts:79`：只检查 `end - start`（压缩后长度）是否超过 8 MiB。
- `packages/core/src/library/pdf-identification-evidence.ts:82-85`：`inflate(bytes.subarray(start, end))` 一次性返回完整输出，之后才比较 `inflated.byteLength > MAX_INFLATED_STREAM_BYTES` 并累加 `total`；超限时内存已经分配。
- 识别只需要每个流开头 512 个字符的文字（`STREAM_TEXT_PREFIX_CHARS`）以及 XMP 元数据，不需要整个流的内容，却解压了整个流。
- `packages/core/src/library/pdf-library-file-identifier.ts:22-32`：读取窗口为 4 MiB + 1 MiB，没有其它保护。

## 可能的解决方案

推荐：改用 pako 的流式 `Inflate`（`onData` 分块），累计输出超过上限（建议单流 1–2 MiB 即够识别用，XMP 一般在前几十 KiB）就立即停止并丢弃该流；总量上限同样在累加过程中生效，而不是事后比较。

备选：在解压前读取 zlib 头部之外没有大小信息，无法预判；因此不建议只靠压缩长度或"压缩比"启发式判断。

## 修复后应补的测试

- 构造一个 256 MiB 全零内容、压缩后约 250 KB 的 PDF：识别函数在固定内存预算内返回（可断言峰值输出块数或使用带计数的 inflate 桩），耗时有界，结果为"无编号、无标题"而不是崩溃。
- 正常 PDF（XMP 中有 `dc:identifier`、Info 字典有 `/Title`）识别结果不变。
- 有多个正常流和一个炸弹流时，炸弹流被丢弃，其余流的证据仍被使用。

## 修复情况

识别没有编号的 PDF 时，压缩流改用 pako 的流式解压：输出一旦超过单流上限就立即放弃这个流，不再先完整解压再检查大小。单流上限从 8 MiB 降到 2 MiB（识别只用内容流开头的文字和 XMP 开头的元素）；32 MiB 的总量上限现在把被放弃的流也算进去，所以大量小的"解压炸弹"也无法让一次扫描一直忙。一个 64 KiB、解压后 64 MiB 的零填充流现在只会解压约 2 MiB；正常 PDF 的识别结果不变（5 个系统 PDF 前后结果一致）。

测试：`pdf-identification-evidence.test.ts` 新增"inflation budget"一组：用对 `Inflate.push` 的计数包装断言实际解压字节数有界，覆盖单个炸弹流、夹在正常流之间的炸弹、24 个炸弹、40 个普通流四种情形，并确认炸弹之外的证据（编号、XMP、Info 标题）仍被使用，正常 PDF 的结果不变。
