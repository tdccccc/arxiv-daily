# 2026-10-01 修复验证与手测部署记录

## 已交付行为与提交

- `9e4bcb2 fix(library): scan before the first full-text index build`：只在未扫描、catalog缺失或与所选根目录不匹配时先扫描。已扫描资料库继续复用catalog；选目录本身不扫描。扫描与索引共用可取消操作，整体扫描失败或取消不继续索引。arXiv元数据查询失败仍允许可读取PDF参与全文索引；研究方向仍只用成功识别的论文。设置与命令共享完成提示，空库、全部失败、取消不误报成功。
- `0df3f95 fix(library): enlarge the research direction review dialog`：扩大审核外壳，限制宽高不超过视口，内容允许滚动、长按钮换行、小窗口单列表单。业务逻辑未改变；真实Obsidian布局尚待手测。
- 修改范围独立；CLI源码、版本文件、发行说明草稿均未纳入这两个提交。

## 实际验证

| 检查 | 观察结果 |
|---|---|
| 接手时6个plugin针对文件 | 105通过；不是历史Red/Green声明 |
| 修复后同6个plugin文件 | 110通过 |
| core fulltext-index-orchestration | 34通过 |
| plugin settings-tab + settings-declarative-tab | 182通过 |
| plugin library-connection-lifecycle | 15通过 |
| personal-library-interest-profile-modal | 修改前后均24通过 |
| NODE_OPTIONS=--max-old-space-size=8192 npm test | 沙箱外授权重跑exit 0：core 2073通过/2跳过；node-runtime 66；CLI 110；plugin 815 |
| npm run lint | exit 0，0 errors / 20 warnings，与历史基线一致 |
| npm run typecheck | exit 0，所有工作区通过 |
| npm run build | exit 0，CLI与插件产物生成 |
| npm run check:boundaries | Workspace boundaries OK |
| npm run check:obsidian-submission | PASS |
| git diff --check / staged范围检查 | 通过，无无关文件混入 |

第一次全量测试未通过：只读沙箱阻止既有测试在`~/.arxiv-daily/host-locks`创建锁，node-runtime 4、CLI 3、plugin 2项失败。按环境规则请求授权重跑，随后全部通过；未修改测试以绕过限制。

首次子任务误用根目录无配置Vitest，扫描到其他worktree测试并产生加载错误；已停止并改用指定plugin工作区配置。这不是行为Red，没有修改其他worktree。新增回归中的fixture错误也没有算作Red。

原始日志：`/tmp/arxiv-050-tests.log`（沙箱失败）、`/tmp/arxiv-050-tests-unrestricted.log`（通过）、`/tmp/arxiv-050-lint.log`、`/tmp/arxiv-050-typecheck.log`、`/tmp/arxiv-050-build.log`。子任务细节：`/tmp/arxiv-scan-fix-report.md`、`/tmp/arxiv-modal-fix-report.md`、`/tmp/arxiv-cli-settings-report.md`。

## 部署

目标：`/home/tiandc/Desktop/plugin_test/.obsidian/plugins/arxiv-daily/`。

先查看目标文件；以独占创建模式备份，不覆盖已有备份。实际创建：

- `main.js.bak-20261001-pre-build-scan`
- `styles.css.bak-20261001-pre-build-scan`

复制plugin/main.js、styles.css、manifest.json后逐件校验SHA-256相同：

| 文件 | 部署后SHA-256 |
|---|---|
| main.js | 9f9f720f6a852a8d6d180d46a658a19eb573ba7ef998eca3421a002e2a28fe49 |
| styles.css | e397f76082fc4ac786c258f9dde7dc500cd4411088ca20a697f8a1cb6a8958d3 |
| manifest.json | b54f7e409c97338a4b2b2457195f426b16d908a181457de2ffa410e8f9cdae91 |

完整记录：`/tmp/arxiv-050-deploy.json`。未打开或修改data.json，未启动Obsidian。当前manifest仍是0.4.6，这是手测通过前尚未进行0.5.0版本同步的预期状态。

## 用户手测清单

1. 重启Obsidian；选一个从未扫描的小PDF资料库，确认选择目录不会开始扫描，点击Build index才先扫描再索引。命令面板Index personal library full text应表现一致。
2. 对已扫描资料库再次Build index，确认不会自动重扫。添加新PDF后需要先Scan library再Build index。
3. 扫描中取消，确认不进入索引，提示为取消。空目录建索引应解释没有PDF；不可读取PDF不应提示已可搜索。
4. 在Review directions点Scan library，反馈成功识别、未识别和失败数量。网络查询失败不能直接解释为论文不受支持；可读取PDF的全文检索与方向识别数量分别判断。
5. 大窗口确认审核对话框更宽；缩小/缩短窗口，滚动到最后一条，确认按钮可见可点，切换Proposed/Confirmed和编辑保存正常。

本轮没有真实arXiv/模型下载/邮件调用验证，没有macOS/Windows桌面验证，也没有完成浏览器几何模拟。没有push、PR更新、合并、tag、发布、远程分支或stash清理；未修改受保护的其他工作树。

## 手测后才继续的发布准备

- 按用户决定在当前分支通过既有PR #51发布0.5.0；不另建发布分支。
- 运行sync:release-version与check:release-version，核对并展示发行说明，再执行docs/release.md要求的本地发布检查和独立提交。
- 草稿当前仍未跟踪、未修改、未作为实现证据：须补上较大审核窗口与首次本地模型约130 MB下载；核对无镜像设置、扫描发送ID/标题到arXiv且无单独同意弹窗、可读PDF/识别论文覆盖差异。
- 草稿“Everything here has landed on main”不符合当前仍在修复分支的状态，最终表述须核对；“每次文件变化全部重新嵌入”也须对照实际复用行为再确认，不能仅沿用旧报告。
- 保留Linux真实桌面与其他平台CI覆盖的边界；本轮不把旧平台记录当作新运行结果。
- 本轮未运行sync/check-release-version、test:release-tools、smoke:build、smoke:install、重新npm ci及远程CI验证；它们仍属于手测后的发布准备阶段。

## CLI 后续版本决定

用户已确认首批：主题查看/添加/修改/启停/删除，以及分类、时区、LLM、输出、邮件配置、计划时间；交互编辑＋非敏感单项命令，密钥隐藏输入。发邮件与安装定时任务继续使用已有专用入口。放后续版本，先收尾并手测0.5.0，授权届时逐项本地提交。

调查确认目前没有配置/主题修改入口；init默认覆盖配置，Keep existing不是编辑；配置为XDG/APPDATA规则下的TOML，--config已移除。实现时保留未管理字段、采用稳定主题ID、新增真正的启停语义、保存前严格校验并沿用原子私密保存；注释保留取舍需明确。候选命名config edit/set/show与topics管理尚未实现，未修改用户真实配置。
