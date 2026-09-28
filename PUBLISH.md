# PUBLISH.md —— 把 `repro/` 发布成公开 artifact 时，发什么、不发什么

这份说明与稿件里的 **Data Availability / Code Availability** 两段是同一条口径。投稿前请读一遍。

## 1. 结论：发这一份（**18,311 个文件 / 实测 ≈28,854 KB ≈28.2 MB 内容**，占盘约 37 MB）

| 目录 / 文件 | 文件数 | 体积 | 发？ | 说明 |
|---|---|---|---|---|
| `src/` | 11 | 153 KB | ✅ | 抓取、建图、门槛、四个 cell、训练、汇总、审计（10 个 `.py` + `fetch.log`） |
| `tools/` | 26 | 158 KB | ✅ | 诊断与产物脚本（含改稿包用的表/图生成器、机制图重绘与遮挡审计） |
| `tools/ext/` | 14 | 148 KB | ✅ | **v18 新增**：扩展实验 E1–E8 的脚本 + 九张扩展表的生成器 `make_extension_tables.py` |
| `results/` | 256 | 1.1 MB | ✅ | 每条 run 的 json（主结果、门槛、配对对比的来源） |
| `results_ckpt_last/` | 31 | 71 KB | ✅ | `CHECKPOINT_RULE=last` 的原始 runs（`tab:checkpoint` 的来源） |
| `results_alt_fixedwindow/` | 242 | 639 KB | ✅ | 固定窗口切分的稳健性 runs |
| `logs/` | 47 | 427 KB | ✅ | 建图日志、诊断、`floor_pool_readings*.json`、`checkpoint_rules.json` |
| `results_ext/` | 525 | 6.0 MB | ✅ | **v18 新增**：E1–E8 的原始产物（每个 (数据集, 模型, 种子, 条件) 一个 JSON/NPZ，含 per-source 数组与门控插桩量） |
| `data/graphs_ext/` | 4 | 2.3 MB | ✅ | **v18 新增**：从一手数据重建的引用网络（HepTh/HepPh 图 + 统计），E7 的输入 |
| `data/raw_citation/` | 15,093 | 14.6 MB | ⚠️ | **v18 新增**：arXiv Atom API 抓来的标题/摘要/发布日期缓存（8,852 篇），见 §2.5 的口径提醒 |
| `data/graphs/` | 5 | 1.8 MB | ✅ | **冻结图快照**（npm/maven 的 `*_graph.json.gz` + `*_stats.json`） |
| `data/seeds/` | 1 | 41 KB | ✅ | Maven 的额外种子记录 |
| `data/raw_maven/` | 2,022 | 1.05 MB | ✅ | Maven Central 原始元数据快照（见 §3 的口径说明；小文件多，占盘约 8 MB） |
| `assets/` | 1 | 81 KB | ✅ | 机制图**原图**（作者 762 pt 版，供对照；新图由 `tools/make_fig_mechanism.py` 生成） |
| `.gitignore`、`.gitattributes`、`LICENSE`、`CITATION.cff`、`MANIFEST.json`、`requirements.txt`、`run_all.sh`、`README.md`、`PUBLISH.md`、**`EXT-EXPERIMENTS.md`** | 10 | 92 KB | ✅ | 配置、依赖（**v18 起含 torch 与 PyG**）、授权、引用信息、一键复现、主文档、本文档，以及 E1–E8 的实验说明书 |
| **`data/raw_npm/`** | 4,959 | **2.7 GB** | ❌ | 见 §2 |
| **`data/bundles/`** | 22 | 98 MB | ❌ | 见 §2 |

合计 **18,311 个文件 / 实测 ≈28,854 KB ≈ 28.2 MB 内容**（`du` 口径约 37 MB，因为 `raw_maven` 有 2,022 个、
`raw_citation` 有 15,093 个小文件；
KB 数按各文件字节数求和 ÷1024 —— 含本文件自身，所以再改动本文档会让末位有几 KB 的自指浮动）；
`raw_npm` 一个目录就占全仓体积的 96%（2.7 GB / 2.8 GB）。

> `CITATION.cff` 与 `.gitattributes` 是 2026-09-27 补的：前者给 GitHub/Zenodo 提供正确的引用信息，
> 后者让冻结产物保持字节稳定（原样存储全部文件，只把 `summary.csv` 规范成 LF —— `csv.writer`
> 在各平台都写 CRLF，不统一的话在 Linux 上重跑 `summarize.py` 会无意义地显示该文件被改动）。
>
> `results/revision_figures/fig_mechanism.pdf` 2026-09-27 被**重绘版**取代：先前是在作者原图上
> 改字（`fix_fig_mechanism_labels.py`）再修箭头（`fix_fig_mechanism_arrows.py`），只治了标；真正的
> 病根是**尺度**——原图 762.12 pt 宽，`\includegraphics[width=\textwidth]` 缩到 **0.4865** 倍，
> 9.5 pt 的字号实际只印出 **≈4.6 pt**。新图 `tools/make_fig_mechanism.py` 把画布缩到 415 × 232 pt，
> 实测在本模板下按 **0.893 倍**印出（单栏 `\textwidth` = 370.7 pt），图内正文 **5.0–6.6 pt**、
> 上下标 3.7–4.1 pt——是改善，但不是「字号即印刷字号」。`tools/check_fig_occlusion.py` 做几何审计
> （文字墨迹框两两求交 + 线段穿字 + 越框），实测 0 重叠 / 0 穿字 / 0 越框、最小净距 1.05 pt；
> `FIG_FONT_SCALE=1.06` 起（`out-neighbours` ↔ `out` 重叠）即 FAIL，所以 1.0 是本版式能过审计的上限。
> 文件数因此从 2,647 变成 2,651（+2 工具、+SVG、+PNG），`assets/` 与 `original/` 里的作者原图未动。
> 2026-09-27 晚又补了印刷尺寸的**实测更正**（`FIG_FONT_SCALE` 旋钮 + 文档订正，见 `repro/README.md` §8 与改稿包
> `CHANGELOG-numbers.md` §17.8），内容体积随之从约 5,542 KB 变成约 **5,548 KB**（文件数不变）。
>
> 2026-09-27 第九次重打（改稿包表格重排，见 `CHANGELOG-numbers.md` §18 / §19）之前又改了这里的四个文件：
> `results/revision_tables/*.tex` 四张生成表与生成器 `tools/make_revision_tables.py` 同步到新的列结构，
> 内容体积 5,548 → **5,551 KB**（+2,546 B；文件数仍 2,651，`tools/` 因此 156 → 158 KB）。
> 数字、标签与结论一个都没变，改的只是列宽、表头词与分组标签行。

### 口径提醒：`data/raw_citation/`（v18 新增）

它是 E7 引用网络对照的**可复现锚**：8,852 篇 arXiv 论文的标题、摘要与提交日期，由
`tools/ext/e7_citation_control.py --fetch` 从 arXiv 官方 Atom API 按 ≤150 篇/批、每批间隔 3 s 抓取，
落盘缓存；`--build` 与整条链在缓存命中时**不会重新请求** arXiv。

- **为什么留在包里**：不留的话，E7 就得重抓（限速下约 10–30 分钟，且依赖 arXiv 可达），
  `data/graphs_ext/*.json.gz` 又**内含**这些标题与摘要文本，删缓存并不能让包"不含"这些内容。
- **权利归属**：与 `raw_maven` 同类，属**上游公开元数据**，MIT 不覆盖（见 §6）。
  发布到公开仓库/Zenodo 之前，请按 arXiv 当时的 API/OAI 使用条款确认批量元数据的再分发口径；
  若你选择不发布这一目录，就把它加进 `.gitignore`，并让 E7 的重跑走 `--fetch` 先抓一遍。
- **可核对性**：即使不发布缓存，`data/graphs_ext/*_stats.json` + `results_ext/e7_*` 也在包内，
  E7 的每一个数字仍能对着论文复算。

## 2. 为什么排除这两个

### `data/raw_npm/`（4,959 个 JSON，2.7 GB）

这是 npm registry 的**原始包元数据**，每个包一个文件。不发的理由：

1. **可完全重抓**：`src/fetch_npm.py` 默认走华为云镜像，约 0.36 s/包 × 4,959 ≈ **30 分钟**。
   发出的代码里已含该脚本与快照时点记录，任何审稿人都能自行重建。
2. **体量与收益不成比例**：为验证一篇论文下 2.7 GB raw dump 不现实；
   Zenodo 虽允许 50 GB，但下载体验会吓退读者。
3. **与声明自洽**：Data Availability 写的是"图、划分与结果随包归档；原始元数据来自公开
   registry，**重抓脚本与快照时点随包发布**"——即声明的是**脚本可重建**，不是"原始 dump 随包"。

### `data/bundles/`（22 个 `.pt`，98 MB）

TF-IDF + truncated-SVD 的**特征缓存**，由 `src/content_floor.py` 的 `build_bundle()` 生成，
`src/audit.py` 也是**重新构建**它再复算门槛（check D3），**不是读缓存**。所以缺了会按需重建。

唯一的代价：为了让 SVD **bitwise 稳定**，重跑时要单线程 BLAS —— `run_all.sh` 已经强制
`OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1`，所以按 `run_all.sh` 跑就不会有偏差。
（若你想让审稿人不依赖 BLAS 设置也能逐位复现 floor，可以额外把 `data/bundles/` 单独发一个
98 MB 的附件——**可选**，不是必需。）

## 3. 关于 `raw_maven` 的不对称（评审可能会问，先备好答案）

`raw_maven/` 只有 **8 MB / 2,022 个文件**，而 `raw_npm/` 是 2.7 GB，所以两者处理不同：

- 全文（含 `table_dataset_overview` 的"上一轮快照，未重跑"一块与局限节）已经披露
  **Maven 快照与节点日期口径不可比**这一事实；
- 发出 Maven 原始快照，是为了让这个披露**可被逐一核对**（快照时点、工件数、窗口下界
  `MAVEN_MIN_DATE=2021-04-19`），成本只有 8 MB；
- npm 侧同样可核对，方式不同：**跑 `fetch_npm.py` 重抓**（§2 已说明）。

## 4. 发布后怎么用

```bash
# 全量重跑（会自动抓 npm + Maven，然后建图 / 门槛 / 训练 / 汇总 / 审计）
bash run_all.sh

# 已有 data/raw_* 与 data/graphs/ 时，跳过抓取
SKIP_FETCH=1 bash run_all.sh

# 只补抓 npm（约 30 分钟）
python3 -u src/fetch_npm.py
```

```bash
# 只核对审计，不重跑训练
python3 src/audit.py            # 期望 414/414 PASS

# 只重生成论文里的表与图
python3 tools/make_revision_tables.py --out tables
python3 tools/make_revision_figures.py
python3 tools/make_revision_figure_datasets.py

# 机制图：重绘（PDF + 可编辑 SVG + 预览 PNG），再跑一次遮挡审计
python3 tools/make_fig_mechanism.py --out results/revision_figures --stem fig_mechanism
python3 tools/check_fig_occlusion.py results/revision_figures/fig_mechanism.pdf   # 期望 RESULT: PASS

# 历史路径（改字版 + 修箭头版，已被重绘取代，仅用于复现当时产物）：
#   python3 tools/fix_fig_mechanism_labels.py --src assets/fig_mechanism.original.pdf --out <新图>
#   python3 tools/fix_fig_mechanism_arrows.py --src <改字版> --out <新图>
```

## 5. 发布载体建议

- **Zenodo（推荐）**：给一个 DOI，可写进 Data/Code Availability，长期有效、可引用；
  接收后可把 DOI 写死进最终稿。
- **GitHub**：审稿阶段可用匿名仓库；注意 `data/raw_maven/` 的 2,022 个小文件会让仓库变大，
  如果你更希望仓库干净，就把 `data/raw_maven/` 与 `data/graphs/` 一起放进 Zenodo 附件，
  GitHub 仓库只放代码 + `MANIFEST.json` + 复现说明。
- 两处都要写清 **snapshot 时点**与 `MAVEN_MIN_DATE`，因为论文的 Maven 披露依赖它。

## 6. 授权与第三方数据来源

- **代码（本仓库全部 .py/.sh/.json）**：MIT，见 `LICENSE`（Copyright (c) 2026 Runxiao Jiang）。
- **派生产物**（`data/graphs/`、`data/seeds/`、`results*/`、`logs/`）：随上述 MIT 一并提供，
  可自由复用；引用时请指本论文。
- **第三方原始数据**：`data/raw_maven/` 与（若你另行发布）`data/raw_npm/` 是 Maven Central 与
  npm registry 的**公开元数据**，其权利归各自上游；MIT **不覆盖**这部分。
  仓库里应保留 **快照时点**与下界（`MAVEN_MIN_DATE=2021-04-19`），并在 README 说明它们的来源，
  而不是对上游数据再声明一次授权。
- Zenodo 上传时，在"License"字段填 **MIT**，"Related identifiers"填本论文 DOI（若有），
  并在描述里写明数据来源与快照时点。

---

## 7. 发布前的实跑验证（2026-09-26，全部由复现包自己产出）

这份包不是"静态检查过"而已，下面这些结果是**实际跑出来的**（环境：`torch 2.9.1+cpu`、
`torch_geometric 2.8.0.post1`、numpy 2.4.6、scipy 1.17.1、sklearn 1.8.0）：

```bash
python3 src/audit.py --datasets npm maven                      # 414 PASS / 0 FAIL
BUNDLE_DIR=<空目录> python3 src/audit.py --datasets npm maven   # 真重建 20 个 bundle，仍 414 / 0
python3 tools/make_revision_tables.py --out <tmp>              # 四张表 sha256 与发布副本相同
python3 tools/make_revision_figures.py --out <tmp>             # 三张图归一化时间戳后字节相同
python3 tools/make_revision_figure_datasets.py                 # 同上
python3 tools/compare_vs_paper.py                              # 逐数字对照表
python3 tools/compare_checkpoint_rules.py --dataset npm        # best_val vs last 两套策略
python3 tools/gate_within_config.py                            # 同配置门控复核
python3 tools/check_fig_occlusion.py results/revision_figures/fig_mechanism.pdf  # RESULT: PASS
FIG_FONT_SCALE=1.06 python3 tools/make_fig_mechanism.py --out <tmp> --stem f_106   # 反例：审计 FAIL
FIG_FONT_SCALE=1.12 python3 tools/make_fig_mechanism.py --out <tmp> --stem f_112   # 反例：审计 FAIL
```

**机制图重绘也过了实跑验证**（2026-09-27）：在包内重跑
`py tools/make_fig_mechanism.py --out results/revision_figures --stem fig_mechanism`，产出的
`fig_mechanism.pdf` 与随包发布的那一份 **sha256 逐位相同**（`27edebd7…d6d7c6`，51.4 KB）；
`check_fig_occlusion.py` 报 `text_text_overlaps 0 / line_ink_hits 0 / words_crossing_a_border 0`、
最小净距 1.05 pt。生成器用 `freeze_pdf_id()` 冻结了 PyMuPDF 的随机 trailer `/ID`，所以是字节级可复现，
不是"看起来一样"。

**印刷尺寸是量出来的，不是算出来的**（2026-09-27 晚补）：把改稿包真装进稿件、用用户态 TeX Live 2026
编译出 `main.pdf`（37 页、0 error、0 undefined ref）后，第 8 页机制图的文字 span 实测为
**5.00–6.61 pt**（上下标 3.69–4.13 pt），除以「画布 415 ÷ 实测 `\textwidth` 370.7」即得设计字号 5.6–7.4 pt，
**与脚本里的数字对得上**——这条反过来说明缩放系数是 **0.893 倍**而非 1.0。同一测量给出作者原图的实际印刷字号：
762.12 pt 画布 → 0.4865 倍 → 统一的 9.5 pt 印成 **≈4.6 pt**。想把这 0.893 买回来只有两条路：
在 370.7 pt 画布上重排，或整体调大字号；后者已用 `FIG_FONT_SCALE` 做反例验证——
`1.06` 让 `out-neighbours` 与 `out` 重叠、`1.12` 让 `out-view aggregation` 越框（两次审计都 FAIL），
所以 **1.0 就是本版式能过审计的上限**。

**最重要的一条**：把 `BUNDLE_DIR` 指向空目录后强制从 `data/raw_npm/` 与 `data/raw_maven/`
重建 20 个 bundle，审计**仍然 414/414**，而且 414 行输出与"读 98 MB 缓存"那一次
**逐字相同**。所以 §2 里"`data/bundles/` 只是可重建的缓存"是有证据的结论，不是推测 ——
不发布它不影响任何数字。

另外，把 `original/` 的原稿搭成本地嵌套树后跑改稿包安装脚本，零 WARNING、零 MISSING，
旧数字残留从 109 处归零。

**边界（不要过度声称）**：审计验证的是冻结产物之间的一致性，外加一次未训练模型的
前向复算；它**不重新训练**那 10 个种子。所以"发布包自洽"可以声称，"从零训练可逐位复现"
不可以。`run_all.sh` 的完整端到端（含约 30 分钟抓取与数小时训练）未在此次验证中运行。

## 8. `CITATION.cff` 必须通过 schema 校验（2026-09-27 复核，实测）

CFF 1.2.0 的根对象是 **`additionalProperties: false`**，顶层*只*允许这 21 个键：

```
abstract  authors  cff-version  commit  contact  date-released  doi  identifiers
keywords  license  license-url  message  preferred-citation  references
repository  repository-artifact  repository-code  title  type  url  version
```

由此有三条容易踩的坑（第一版仓库里正好踩中第一条，文件实测为**非法**）：

1. **`journal` / `year` 不能放顶层** —— 它们是论文（reference）的字段，必须放进
   `preferred-citation` 里。放顶层会让整个文件非法，GitHub 的「Cite this repository」
   面板会**直接不显示**，Zenodo 解析也会失败。
2. **顶层 `doi` 是"本软件这个版本"的 DOI**（Zenodo 铸出来的 `10.5281/zenodo.*`），
   **不是**论文 DOI；论文 DOI 写 `preferred-citation.doi`，接受后再填。
3. **`authors[].orcid` 必须是完整 URL**：`https://orcid.org/0000-0000-0000-0000`。
   裸数字串不符合 schema（该字段带 `format: uri` 与 pattern），占位时保持注释掉即可。

改动后一条命令自查（本仓库当前文件实测输出 `Citation metadata are valid according to
schema version 1.2.0.`，退出码 0）：

```bash
pip install cffconvert
cffconvert --validate -i CITATION.cff
```

> 作者的 **ORCID 不要臆造**：没注册就保持注释掉；注册是免费的，投稿/归档前花两分钟建一个即可。
> 作者顺序与通讯作者以稿件 `main.tex` 的 `\author*` 与 Author Contributions 为准。
> **2026-09-27（第十次重打）起稿件是两位作者**：`Runxiao Jiang` 为第一作者，`Jialiang Xie`
> （谢加良老师）为**通讯作者**；`CITATION.cff` 的 `authors` 与 `preferred-citation.authors`
> 已同步为两人（`authors[].corresponding: true` 挂在 Xie 上，CFF 1.2.0 的 person 字段支持它）。
