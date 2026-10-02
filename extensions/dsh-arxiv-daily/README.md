# arXiv Daily for DeepSeek Harness

在 DSH 输入区点击“文献”，直接在右侧内置浏览器打开文献工作台。日历、论文列表、日报、详细总结和待读/收藏使用现有 arXiv Daily 核心及本地 Markdown。打开、搜索和标记不需要 Agent 对话，也不调用模型。

## 构建与安装

从 arxiv-daily 仓库运行：

```sh
node extensions/dsh-arxiv-daily/build.mjs
```

本地构建包含当前操作系统和架构的原生存储模块，产物会标注平台限制。其他平台应在对应系统构建；完整跨平台分发需要先准备原项目的原生模块发布矩阵。

构建输出 `extensions/dsh-arxiv-daily/dist/package`。将该目录打包：

```sh
npm pack ./extensions/dsh-arxiv-daily/dist/package
```

**DSH Desktop：** 打开“插件 / Plugins → 添加插件 / Add plugin”，输入生成的 `dsh-arxiv-daily-0.1.0.tgz` 的绝对路径，安装后选择“立即启用 / Enable now”。桌面 profile 由 Electron 管理，不要使用 `dsh plugin --profile desktop`。

**DSH Web：** 使用下面的命令安装到 Web profile：

```sh
dsh plugin --profile web add /absolute/path/dsh-arxiv-daily-0.1.0.tgz
```

重新启动 DSH，使新的插件包和客户端模块一起加载。包尚未发布到 npm；目前请安装本地产物。插件会启用 DSH 自带的侧栏 Browser 组件。

## 使用

1. 沿用已有 arXiv Daily CLI 配置。首次使用时，在终端运行随包提供的 CLI：`node /absolute/plugin/path/lib/arxiv-daily-cli.cjs init`；从仓库试用也可运行 `node apps/cli/dist/arxiv-daily-cli.cjs init`。配置默认位于系统用户配置目录中的 `arxiv-daily/config.toml`。
2. 在本机 DSH 打开一个会话，点击输入区下方的“文献”。插件自动启动工作台，首次启动失败会显示提示，可完成设置后重试。
3. 在右侧直接选择日期、搜索论文、阅读日报/总结、保存待读或收藏。“生成”按钮调用既有 CLI 流程，使用 arXiv Daily 配置的模型与邮件设置。
4. 可使用 DSH 自带的侧栏放大或浮动功能。较窄时工作台通过“日历与筛选”切换导航。

工作台进程在同一插件实例的会话间共享。关闭阅读标签不会停止生成；禁用插件或退出 DSH 会停止它。插件重载后旧工作台链接会失效，再次点击“文献”获取新链接。

已在 DSH `0.1.7-alpha.1` 的真实 Host 上完成安装和 RPC 验证；未自动操作 Electron 桌面界面，其他版本仍需实测。

## 实现边界

- `src/workbench-process.mjs` 只管理现有 CLI 的启动、就绪和停止，未来 Claude Mod 可复用这个边界；DSH 的 RPC/按钮代码留在 `host.mjs` 和 `client.mjs`。
- Desktop 使用 DSH 的隔离 webview；Web 使用 iframe，并仅向当前本机 DSH 的精确来源开放嵌入。工作台仍验证 capability、Host 和 API 请求 Origin。
- 目前面向本机 DSH；不支持远程 DSH 网页访问用户电脑的文献服务。
- 一次工作台生命周期只绑定一个网页来源；混用 localhost 和 127.0.0.1 时请回到原地址，或在任务结束后重新启用插件。
- 保留独立 CLI/浏览器入口，不复制论文库，不把研究数据存到会话历史中。
- 首版保留 CLI 初始化；未提供新的 GUI 设置向导、自动切换 DSH 模型或 Claude Mod 视图。

## 验证

```sh
npm ci --prefix extensions/dsh-arxiv-daily --ignore-scripts
node extensions/dsh-arxiv-daily/build.mjs
node --test extensions/dsh-arxiv-daily/tests/*.test.mjs
```

具体 DSH 版本、真实 Host 验收结果与桌面视觉验证范围记录在本仓库 `docs/helm/2026-10-02-dsh-research-plugin`。
