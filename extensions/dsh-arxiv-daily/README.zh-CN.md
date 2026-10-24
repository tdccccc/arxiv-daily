[English](README.md) · 简体中文

# arXiv Daily for DeepSeek Harness

在 DeepSeek Harness（DSH）里提供 [arXiv Daily](../../README.md) 的阅读工作台——左侧栏“设置”上方有固定的 **arxiv-daily** 入口，右侧栏同样有对应标签。两处打开的是同一个完整工作台：日历、论文列表、日报、详细总结，以及待读/收藏标记。打开、搜索和标记都不会触发 Agent 对话，也不会调用模型。

工作台使用自己保存的模型设置，与 DSH 对话模型分开；并复用你已有的 CLI 配置——如果你已经在用 [CLI](../../apps/cli/README.md)，不需要再单独配置一遍。[Obsidian 插件](../../plugin/README.md)有自己的设置和 API 密钥，不会共享，但可以使用同一个输出目录。

## 环境要求

- DSH，Node.js **22.19.0** 及以上
- `PATH` 中可用的 [pnpm](https://pnpm.io)（DSH 的插件安装流程依赖它）

## 安装

**DSH Web**：

```sh
dsh plugin --profile web add dsh-arxiv-daily
```

然后重启 DSH，使 Host 和客户端都加载新插件。

**DSH Desktop**：打开 **插件 / Plugins → 添加插件 / Add plugin**，在包名输入框中填写 `dsh-arxiv-daily`，点击 **Install**，再点击 **Enable now**。DSH Desktop 的“添加插件”对话框接受纯 npm 包名（即 Web profile 下 `dsh plugin add` 后面的那部分），不需要本地文件或路径。

插件会自行注册 **arxiv-daily** 侧栏标签，不需要额外配置才能看到入口。

## 升级

**DSH Web**：

```sh
dsh plugin --profile web add dsh-arxiv-daily@latest
```

然后重启 DSH。

**DSH Desktop**：已安装插件不会自动更新。请先在 **Plugins** 里卸载该插件，再按全新安装的方式重新添加（Desktop 没有版本选择器，也不支持原地升级）。

## 卸载

卸载插件包本身（Desktop：**Plugins → arxiv-daily → Uninstall**；Web：`dsh plugin --profile web remove dsh-arxiv-daily`）只会移除插件。arXiv Daily 的配置、Markdown 日报/总结和论文索引不受影响——它们保存在插件之外，位于你的 CLI 配置和输出目录里（见 [CLI 配置](../../apps/cli/README.zh-CN.md#配置)）。

## 首次使用

点击左侧栏“设置”上方的 **arxiv-daily**（或右侧栏对应标签），无需先发消息。如果你已经有 CLI 配置，工作台会直接复用；否则会自动打开设置页，流程与[独立工作台](../../docs/workbench.zh-CN.md#首次使用)的首次使用一致。

## 使用

- DSH 内的工作台就是 [docs/workbench.zh-CN.md](../../docs/workbench.zh-CN.md) 中描述的那一个——日历/筛选侧栏、论文列表、带公式的 Markdown 阅读器、设置、个人文献库和方向审核在这里的用法完全相同。
- 可以用 DSH 自带的侧栏放大/浮动功能调整面板大小。较窄时通过专门的切换按钮打开日历/筛选。
- 模型请求使用 arXiv Daily 自己保存的 API 设置，与 DSH 对话模型分开。

完整功能说明见 [docs/workbench.zh-CN.md](../../docs/workbench.zh-CN.md)。

## 限制

- **仅支持本机 DSH** —— 工作台连接的是本机运行的 `arxiv-daily` 进程，不支持远程 DSH 反向访问你电脑上的文献库。
- **更新后需要重启** —— 只替换磁盘上的插件包不会替换已经在运行的工作台进程；需要退出并重启 DSH，再刷新浏览器标签。工作台进程会随 DSH 一起停止，DSH 被强制结束或崩溃时也一样，所以退出 DSH 就够了。
- **一次工作台生命周期只绑定一个来源** —— 一次工作台会话只绑定一个页面来源（origin）。如果在 DSH Web 的不同会话间混用 `localhost` 和 `127.0.0.1`，请回到最初使用的地址，或在会话结束后重新启用插件。

## 从源码构建

如果想试用尚未发布的改动，或使用本地构建：

```sh
npm ci
npm ci --prefix extensions/dsh-arxiv-daily --ignore-scripts
node extensions/dsh-arxiv-daily/build.mjs
npm pack ./extensions/dsh-arxiv-daily/dist/package --pack-destination ./extensions/dsh-arxiv-daily/dist
```

构建原生存储模块需要 CMake、C++ 编译器和 Node-API 头文件；产物 `.tgz` 只包含当前操作系统和架构的构建，不能跨平台分发。

安装本地产物：

```sh
dsh plugin --profile web add /absolute/path/dsh-arxiv-daily-<version>.tgz
```

**DSH Desktop** 同样可以在“添加插件”对话框中填写绝对路径或 `.tgz` 文件。

## 许可证

MIT —— 见 [LICENSE](../../LICENSE)。
