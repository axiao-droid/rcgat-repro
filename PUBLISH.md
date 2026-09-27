# PUBLISH.md —— 把 `repro/` 发布成公开 artifact 时，发什么、不发什么

这份说明与稿件里的 **Data Availability / Code Availability** 两段是同一条口径。投稿前请读一遍。

## 1. 结论：发这一份（约 13 MB / 2,650 个文件）

| 目录 / 文件 | 文件数 | 体积 | 发？ | 说明 |
|---|---|---|---|---|
| `src/` | 18 | 329 KB | ✅ | 抓取、建图、门槛、四个 cell、训练、汇总、审计 |
| `tools/` | 29 | 233 KB | ✅ | 诊断与产物脚本（含改稿包用的表/图生成器） |
| `results/` | 254 | 1.4 MB | ✅ | 每条 run 的 json（主结果、门槛、配对对比的来源） |
| `results_ckpt_last/` | 31 | 125 KB | ✅ | `CHECKPOINT_RULE=last` 的原始 runs（`tab:checkpoint` 的来源） |
| `results_alt_fixedwindow/` | 242 | 1.1 MB | ✅ | 固定窗口切分的稳健性 runs |
| `logs/` | 47 | 541 KB | ✅ | 建图日志、诊断、`floor_pool_readings*.json`、`checkpoint_rules.json` |
| `data/graphs/` | 5 | 1.8 MB | ✅ | **冻结图快照**（npm/maven 的 `*_graph.json.gz` + `*_stats.json`） |
| `data/seeds/` | 1 | 45 KB | ✅ | Maven 的额外种子记录 |
| `data/raw_maven/` | 2,022 | 8.0 MB | ✅ | Maven Central 原始元数据快照（见 §3 的口径说明） |
| `assets/` | 1 | 85 KB | ✅ | 机制图原图（改标签前的版本，供对照） |
| `MANIFEST.json`、`requirements.txt`、`run_all.sh`、`README.md`、`PUBLISH.md` | 5 | — | ✅ | 配置、依赖、一键复现、主文档、本文档 |
| **`data/raw_npm/`** | 4,959 | **2.7 GB** | ❌ | 见 §2 |
| **`data/bundles/`** | 22 | 98 MB | ❌ | 见 §2 |

总计约 **13 MB**，`raw_npm` 一个目录就占全仓体积的 96%（2.7 GB / 2.8 GB）。

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
```

**最重要的一条**：把 `BUNDLE_DIR` 指向空目录后强制从 `data/raw_npm/` 与 `data/raw_maven/`
重建 20 个 bundle，审计**仍然 414/414**，而且 414 行输出与"读 98 MB 缓存"那一次
**逐字相同**。所以 §2 里"`data/bundles/` 只是可重建的缓存"是有证据的结论，不是推测 ——
不发布它不影响任何数字。

另外，把 `original/` 的原稿搭成本地嵌套树后跑改稿包安装脚本，零 WARNING、零 MISSING，
旧数字残留从 109 处归零。

**边界（不要过度声称）**：审计验证的是冻结产物之间的一致性，外加一次未训练模型的
前向复算；它**不重新训练**那 10 个种子。所以"发布包自洽"可以声称，"从零训练可逐位复现"
不可以。`run_all.sh` 的完整端到端（含约 30 分钟抓取与数小时训练）未在此次验证中运行。
