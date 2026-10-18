# 发布工作流不校验标签所指提交的来源分支

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 CLI-7
> 范围：`.github/workflows/release.yml`、`publish-cli.yml`、`publish-dsh-plugin.yml` · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 1978632） · 置信度：代码推断

## 问题

任何推送的 `*.*.*` 标签（`dsh-v*` 除外）都会创建带签名证明的公开 GitHub Release，随后 `publish-cli` 通过可信发布把该提交发布到 npm。"Verify stable tag"步骤只检查标签解析到 `github.sha`（对任何标签都成立）、`docs/releases/<tag>.md` 存在、版本文件一致。于是一个打在未审查分支提交上（或早于某个安全修复的旧提交上）的标签，只要自身文件自洽，就会被发布，main 的"只能走 PR + 必需检查"保护被绕过。

`publish-dsh-plugin` 有同样的缺口，而且跳过了 `publish-cli` 会跑的 npm audit、lint 和 core/工作区测试。

注意：按 `docs/release.md`，正式发布流程是在 `release/<version>` 分支上打标签，发布成功后才把该分支合入 main，所以不能简单要求"标签提交必须在 main 上"。

## 原因

工作流中没有任何来源分支检查（`fetch-depth: 0` 已设置，但默认只取到标签引用），仓库里也没有记录标签保护规则。

## 可能的解决方案

推荐：在三个工作流里加入检查——拉取 `origin/main` 与 `origin/release/<version>`（DSH 对应其发布分支约定），要求标签提交可从其中之一到达（`git merge-base --is-ancestor`），并且 `origin/main` 当前提交是标签提交的祖先（发布分支包含 main 上的全部修复，防止从旧提交发布）。`publish-dsh-plugin` 补上与 `publish-cli` 一致的 audit / lint / 测试门禁。同步更新 `docs/release.md` 说明这些检查。

备选：只做"可从 main 或 release/* 到达"一项检查，较宽松。另外可在 GitHub 设置标签保护规则（仓库外配置，需要用户操作）。

## 修复后应补的测试

工作流难以单测；在 `scripts/tests/` 现有的工作流静态检查（若有）中断言三个工作流都包含来源检查步骤，或提取为脚本 `scripts/check-release-source.mjs` 并为其写 `node --test` 测试。

## 修复情况

新增 `scripts/check-release-source.mjs`，由 `release.yml`、`publish-cli.yml`、`publish-dsh-plugin.yml` 调用。标签推送：提交必须可从 `origin/main` 或 `origin/release/<version>` 到达，且包含当前 main。手动重新发布：只要求可到达。`publish-cli` 使用 `origin/main` 上的检查脚本副本，防止标签提交自带削弱版。DSH 发布接受 main 或 `release/dsh-<version>`，并补上与 `publish-cli` 相同的 audit / lint / 测试 / 构建门禁。`docs/release.md` 已更新。

局限：能同时推标签又能改工作流的人仍可绕过；文档建议在 GitHub 上设置标签保护规则，这需要仓库所有者在设置里操作。
