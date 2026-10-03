# arXiv Daily for DeepSeek Harness

左侧栏“设置”上方提供固定的“arxiv-daily”入口，空白新建页面也能打开。右侧栏入口页同时提供“arxiv-daily”标签，两处显示同一工作台。日历、论文列表、日报、详细总结和待读/收藏使用现有 arXiv Daily 核心及本地 Markdown。打开、搜索和标记不需要 Agent 对话，也不调用模型。

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

**DSH Desktop：** 打开“插件 / Plugins → 添加插件 / Add plugin”，输入生成的 `dsh-arxiv-daily-0.1.6.tgz` 的绝对路径，安装后选择“立即启用 / Enable now”。桌面 profile 由 Electron 管理，不要使用 `dsh plugin --profile desktop`。

**DSH Web：** 使用下面的命令安装到 Web profile：

```sh
dsh plugin --profile web add /absolute/path/dsh-arxiv-daily-0.1.6.tgz
```

重新启动 DSH，使新的插件包和客户端模块一起加载。包尚未发布到 npm；目前请安装本地产物。插件会注册自己的“arxiv-daily”侧栏标签。

## 从旧版本升级

0.1.6 增加设置左侧跳转导航、当前区域高亮和清晰分区，窄屏导航位于顶部；原设置条目不变。0.1.5 按 Obsidian 1.13+ 设置原条目重做，保留原分组、顺序、下拉选项、开关和条件显示，并接通文献库、模型列表、邮件按钮及分钟级自动检查。0.1.4 增加图形首次使用与可保存设置页，直接在 DSH 内完成配置。0.1.3 将两侧入口和标签的显示名称统一为 `arxiv-daily`。0.1.2 已增加固定的左侧入口和右侧标签，移除会话输入区的旧按钮；同时保留 0.1.1 的 API Gateway 共存修复。Web profile 可用新包替换旧依赖：

```sh
dsh plugin --profile web add /absolute/path/dsh-arxiv-daily-0.1.6.tgz
```

然后退出并重新启动 DSH，使 Host 和客户端都加载新包。Desktop 使用插件管理器更新；若该版本的管理器不支持替换本地包，卸载旧插件后添加新包。插件卸载不会删除 arXiv Daily 的配置、Markdown 或论文索引。

## 使用

1. 点击左侧栏“设置”上方的“arxiv-daily”，无需先发消息；也可在右侧栏入口页选择“arxiv-daily”。
2. 首次使用会自动打开设置页。填写研究记录保存目录、模型 API 密钥与关注主题，点击“保存并开始使用”。已有 CLI 配置会直接沿用，不必再运行 `init`。
3. 工作台右上角“设置”遵循 Obsidian 1.13+ 主设置路径：LLM、arXiv categories、Research topics、Automatic detail notes、Timezone、Output & schedule、Personal library、Email delivery、Advanced、Help & feedback。分类、推理强度、时区、时间窗口等保留原选择控件；模型保留自由输入、候选列表及“Get models”。点击“保存设置”后生效，密钥留空保留且旧密钥不回显。独立运行必需的保存根目录放在原设置之前。

   Personal library 的 Choose folder / Build index / Revoke 复用共享文献库流程；Web 宿主通过路径框指定目录，首次处理会显示授权范围。Email delivery 的 Send verification / Send test 只有点击后才发送。Enable 开启后，仅在工作台进程运行时，按 Check every (minutes) 检查工作日报；原 CLI 外部 cron 配置独立保留，不会自动安装或修改系统任务。
4. 可使用 DSH 自带的侧栏放大或浮动功能。较窄时工作台通过“日历与筛选”切换导航。

工作台进程在同一插件实例的会话间共享。关闭阅读标签不会停止生成；禁用插件或退出 DSH 会停止它。插件重载后旧工作台链接会失效，再次点击“arxiv-daily”获取新链接。

旧版已在 DSH `0.1.7-alpha.1` 验证 Host 接口；本轮本机 DSH 已更新至 `0.2.0-rc.2`，在该版本验证固定入口的 SlotCore 注册和 Host 接口；未自动操作 Electron 桌面界面，其他版本仍需实测。

## 实现边界

- `src/workbench-process.mjs` 只管理现有 CLI 的启动、就绪和停止，未来 Claude Mod 可复用这个边界；DSH 的 RPC/按钮代码留在 `host.mjs` 和 `client.mjs`。
- 两种入口都使用隔离 iframe，Web 仅向当前本机 DSH 的精确来源开放嵌入，Desktop 仅允许固定的 `dsh-app://app` 来源。工作台仍验证 capability、Host 和 API 请求 Origin。
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

更新后必须退出并重新启动 `dsh web`，再刷新浏览器；仅替换磁盘上的插件包不会替换已经运行的工作台进程。
