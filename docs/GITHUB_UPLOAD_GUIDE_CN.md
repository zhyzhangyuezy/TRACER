# GitHub 上传流程（TRACER 代码与数据发布）

这份说明对应当前已经生成好的发布产物，目标是把项目的模型代码、实验脚本、处理后数据和结果摘要按标准形式发布到 GitHub，并把超大文件作为 GitHub Release 附件上传。

这版公共仓库**不包含论文源码与 PDF**。论文文件只保留在本地工作目录中，用于你上传完成后回填正式仓库链接并重新编译投稿版本。

## 当前已经准备好的文件

- GitHub 仓库目录版：`dist/gh_repo/TRACER/`
- GitHub 仓库压缩包：`dist/TRACER_github_repo_2026-04-20.zip`
- 大文件 Release 附件：`dist/gh_assets/ait_ads_canonical_events.zip`
- 发布清单：`dist/TRACER_github_release_manifest.json`
- 校验和：`dist/SHA256SUMS.txt`

## 推荐的仓库命名

建议仓库名：

- `TRACER-alert-escalation-warning`
- 或 `TRACER-code-data-release`

建议仓库描述：

`Archival code, processed data bundles, stored results, and experiment artifacts for TRACER: Knowledge-Guided Case-Based Warning for Alert Escalation Forecasting.`

建议 topics：

- `cybersecurity`
- `time-series`
- `retrieval-augmented`
- `case-based-reasoning`
- `alert-escalation`
- `knowledge-based-systems`

## 方式一：GitHub 网页直接上传

这是最省事的一种。

1. 登录 GitHub。
2. 点击右上角 `New repository`。
3. 仓库名填上面建议名之一。
4. 选择 `Public`。
5. 不要勾选自动生成 `README`、`.gitignore`、`LICENSE`。
6. 创建仓库。
7. 打开新建好的空仓库页面，选择 `uploading an existing file`。
8. 把 `dist/gh_repo/TRACER/` 目录里的**所有内容**拖进去。  
   注意：上传的是 `TRACER` 目录里的内容，不是把 `TRACER` 这个文件夹本身再包一层。
9. 提交说明可写：
   `Initial public code-and-data release for the TRACER paper`
10. 点击提交。

## 方式二：本地 git 上传

如果你的机器已经安装了 git，也可以用命令行。

先进入干净发布目录：

```powershell
cd "C:\Users\Administrator\OneDrive\跑程序\auto科研\Auto-claude-code-research-in-sleep-main\dist\gh_repo\TRACER"
```

然后执行：

```powershell
git init
git add .
git commit -m "Initial public code-and-data release for the TRACER paper"
git branch -M main
git remote add origin https://github.com/<your-account>/<your-repo>.git
git push -u origin main
```

## 创建 GitHub Release

主仓库上传完成后，再做 Release。

1. 进入仓库页面。
2. 点击右侧或顶部的 `Releases`。
3. 点击 `Draft a new release`。
4. Tag 建议填：`v1.0.0`
5. Release title 建议填：
   `TRACER code-and-data release v1.0.0`
6. Release notes 可以直接使用 `docs/GITHUB_RELEASE_NOTES_TEMPLATE.md` 里的内容。
7. 把 `dist/gh_assets/ait_ads_canonical_events.zip` 上传为附件。
8. 发布 release。

## 上传完成后要拿到的两个链接

上传结束后，请记录：

1. GitHub 仓库主页链接  
   例如：`https://github.com/<your-account>/<your-repo>`

2. GitHub Release 页面或大文件附件链接  
   例如：`https://github.com/<your-account>/<your-repo>/releases/tag/v1.0.0`

## 上传完成后要做的论文同步

打开：

- `paper/sections/7_data_availability.tex`

把其中这段占位符：

- `[repository URL to be inserted before final submission]`

替换成真实 GitHub 仓库链接。

如果你希望 data availability statement 同时明确指向大文件，也可以把 release 页面链接补进 `paper/KBS_submission_statements_template.md` 里的 `[release asset URL]` 占位符，并同步到最终投稿版本。

然后重新编译：

```powershell
cd "C:\Users\Administrator\OneDrive\跑程序\auto科研\Auto-claude-code-research-in-sleep-main\paper"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main_kbs.tex
```

## 上传前最后核对

建议核对以下几点：

- 仓库主页能看到 `README.md`
- 仓库里有 `LICENSE`、`CITATION.cff`
- `data/README.md` 和 `outputs/README.md` 可读
- 仓库没有超过 GitHub 限制的大文件
- Release 页面已经挂上 `ait_ads_canonical_events.zip`
- GitHub 仓库中不包含 `paper/` 目录
- 论文里的 data availability statement 已替换成真实链接

## 当前这版发布包的关键事实

- 主仓库压缩包：约 `68.78 MB`
- staged repo 中单文件超过 `100 MB` 的数量：`0`
- 单独 Release 附件：`ait_ads_canonical_events.zip`
- 对应原始大文件：`data/ait_ads_public/ait_ads_canonical_events.csv`

## 建议保留的证据文件

如果编辑或审稿人之后追问可用性证明，可以保留这些文件：

- `dist/TRACER_github_release_manifest.json`
- `dist/SHA256SUMS.txt`
- GitHub 仓库链接
- GitHub Release 链接
