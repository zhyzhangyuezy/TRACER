# GitHub 上传说明（TRACER 公开代码与数据仓库）

这份说明对应当前已经发布的 TRACER 公开仓库与 Release，主要用于留档和后续更新。

## 当前公开地址

- 仓库主页：`https://github.com/zhyzhangyuezy/TRACER`
- Release 页面：`https://github.com/zhyzhangyuezy/TRACER/releases/tag/v1.0.0`

## 当前发布包位置

- 干净发布目录：`dist/gh_repo/TRACER/`
- 仓库压缩包：`dist/TRACER_github_repo_2026-04-20.zip`
- 大文件 Release 附件：`dist/gh_assets/ait_ads_canonical_events.zip`
- 发布清单：`dist/TRACER_github_release_manifest.json`
- 校验和：`dist/SHA256SUMS.txt`

## 当前发布策略

- GitHub 仓库只包含模型代码、实验脚本、处理后数据、结果摘要和文档。
- 论文源码与 PDF 不进入公开仓库。
- `data/ait_ads_public/ait_ads_canonical_events.csv` 作为单独的 Release 附件发布。
- 小型参考标签表直接包含在仓库的 `data/reference_labels/` 中。
- 其余未直接纳入 git 的公共上游输入，可通过
  `python scripts/fetch_external_data.py --targets all`
  自动恢复到期望路径。

## 如果以后需要重新发布

1. 在工作区根目录运行：

```powershell
python .\scripts\build_github_release.py
```

2. 检查生成结果：

- `dist/gh_repo/TRACER/`
- `dist/TRACER_github_repo_*.zip`
- `dist/gh_assets/ait_ads_canonical_events.zip`
- `dist/TRACER_github_release_manifest.json`

3. 如果要更新线上仓库内容，把 `dist/gh_repo/TRACER/` 中的内容推送到
   GitHub 仓库根目录。

4. 如果大文件附件有变化，在 GitHub Release 页面同步替换附件。

## 论文同步点

当前论文已经写入真实链接，重点文件是：

- `paper/sections/7_data_availability.tex`
- `paper/KBS_submission_statements_template.md`

如果未来仓库地址或 Release tag 变化，需要同步更新这两处并重新编译论文。

## 当前发布包的关键信息

- 主仓库压缩包约 `48.89 MB`
- staged repo 中超过 `100 MB` 的文件数：`0`
- 当前最大的 git 跟踪文件是
  `data/splunk_attack_data_public_probe/canonical_events.csv`
  ，约 `81.28 MB`

## 建议保留的证明材料

- `dist/TRACER_github_release_manifest.json`
- `dist/SHA256SUMS.txt`
- GitHub 仓库链接
- GitHub Release 链接
