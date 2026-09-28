# RC-GAT 复现包（repro）

本目录是论文（RC-GAT，npm + Maven 依赖网络，EMSE 投稿版）实验部分的**可运行复现代码**。
项目里回收到的 `rc_*` 代码（`ranking.py` / `content_floor.py` / `run_phase4..6.py` / `gated.py` /
`train3.py` / `summarize6.py` / `audit6.py`）只有模型和评测片段，**没有数据层**：从注册表原始
JSON 到"节点/边/文本"的适配器、抓取脚本的种子表、以及节点日期约定全部缺失，而且已有的部分
和论文正文存在实质冲突。本包把整条链路补齐、逐项对账，并把每一处判断的依据写进代码注释和
本文档。

```
run_all.sh                     → 一键跑完整条链路（等价于下面各步按顺序执行）
fetch_npm.py / fetch_maven.py   → data/raw_*/                原始注册表快照（可续抓）
build_dataset.py               → data/graphs/*_graph.json.gz 冻结的有向图快照（含 sha256）
content_floor.py               → data/bundles/*.pt            每个 (数据集, 种子) 的切分 + 内容地板 + 描述子 + 边时间
run_experiments.py             → results/{floor,diag,tuning,runs,selections}  四个阶段 + 最终 10 种子
summarize.py                   → results/summary/            配对对比、置信区间、符号计数
audit.py                       → results/audit.json           独立复核（切分/泄漏/地板/账目/对比）
tools/                         → 标定与诊断脚本（见下）
```

## 1. 快速开始

```bash
bash run_all.sh                          # 一键跑完：fetch → 图快照 → 地板/切分 → 调参 → 10 种子 → 汇总/审计/表格

# 或者分步执行：
pip install -r requirements.txt          # 见文件内注释：CPU 版 torch 即可
export OMP_NUM_THREADS=1                 # 让 bundle 可复现（见 §5.4）

python src/fetch_npm.py                  # 抓 npm（华为云镜像，约 0.36 s/包）
python src/fetch_maven.py                # 抓 Maven Central（约 1–2 s/工件）

MAVEN_MIN_DATE=2021-04-19 python src/build_dataset.py --datasets npm maven
python src/run_experiments.py --phase floor --datasets npm maven
python src/run_experiments.py --phase diag  --datasets npm maven
python src/run_experiments.py --phase tune   --datasets npm maven --workers 4
python src/run_experiments.py --phase final  --datasets npm maven --workers 4
python src/summarize.py --datasets npm maven
python src/audit.py --datasets npm maven
python tools/make_tables.py --datasets npm maven   # 生成论文格式的 LaTeX 表
python tools/compare_vs_paper.py                   # 本包结果 vs 论文 Table 1/2 的对照表
```

所有阶段都是**可续跑**的：每个 (数据集, 模型, 种子) 单独写一个 JSON，已存在就跳过；被中断后
重跑不会重复训练，也不会丢已完成的配置。

资源参考（4 vCPU / 8 GB）：单个训练任务单线程约 1.8 s/epoch，4 路并行时约 3 s/epoch（CPU 争用，
所以仍以 `--workers 4` 单线程并行最划算；`ranking.evaluate_ranking` 向量化前后的对比见 §4），
一次 tuning 任务 1.5–4.5 分钟。完整账见 §6。

## 2. 数据：从零抓取（npm + Maven）

论文的数据集太大、无法随包上传，因此两个图都在本机从公开注册表重建。

### 2.1 npm

* 来源：华为云 npm 镜像 `https://repo.huaweicloud.com/repository/npm/`（比官方 registry 快）。
* 节点日期 = `time.created`（首次发布）；边 = `versions[dist-tags.latest].dependencies`
  （去掉 `node:` 内置模块）；文本 = `name + description + keywords[:400] + readme[:1500]`，
  **只折叠空白、不剥 HTML 标签、先截断后拼接**（这一条是标定出来的，见 §3.3）。
* 做法：从约 250 个热门包出发做 BFS（依赖即扩展边），队列耗尽为止。
  最终 `fetched=4959`、`failed=9`（`@babel/helper-*`、`uWebSockets.js` 等在镜像上确实不存在的种子拼写）。
* 冻结快照：**4 959 节点 / 13 602 边**，节点日期 2010-12-19 .. 2026-09-17。

### 2.2 Maven

* 来源：Maven Central `repo1.maven.org` 的目录列表 + `maven-metadata.xml` + POM。
  （项目里回收到的 `fetch_maven.py` 其实是无关程序的 `.pyc`，完全不能用，本包的抓取脚本是重写的。）
* 边 = 最新版本的 compile-scope 依赖（跳过 test/provided/system/import 与 optional）。
* 节点日期 = **最新非预发布版本的发布日期**，并只保留日期 ≥ `MAVEN_MIN_DATE` 的工件。
  这一步不是随意设定：见 §3.2 的对照实验，它是唯一同时命中论文 Maven 表格里
  "date span 起点 2021-04-19" 与 "max out-degree 87" 的读法。
* 种子 = 手写热门工件表（Spring/JUnit/Jackson/Netty/…）+ 由
  `tools/maven_extra_seeds.py` 从已抓到的 group 目录里扩展出的同族工件（每个 group 最多 20 个）。
  论文没有给出种子表，这一步是**重建**，不是复现，详见 §5.1。

## 3. 与论文表格的对账（reproduction check）

### 3.1 图规模：几乎逐项命中

| 统计量 | 论文 npm | 本包 npm | 论文 Maven | 本包 Maven |
|---|---|---|---|---|
| 节点 | 4 962 | 4 959 | 1 484 | 1 664 |
| 边 | 13 596 | 13 602 | 5 482 | 5 651 |
| 平均度 | 2.740024183796856 | 2.742892 | 3.6940700808625335 | 3.396034 |
| 最大出/入度 | 167 / 253 | **167** / 254 | 87 / 216 | 122 / **214** |
| 节点日期跨度 | 2010-12-19 .. 2026-09-17 | **完全一致** | 2021-04-19 .. 2026-09-21 | 2021-04-24 .. 2026-09-25 |
| 互惠率 | 0.001 | 0.000882 | 0.004 | 0.000 |

npm 侧的结论是明确的：**节点日期 = `time.created`、边 = 最新版本 dependencies** 这一读法同时命中了
日期跨度两端、最大出度（精确 167）和边数（差 0.04%），因此数据层的口径可以认为已经对上。

Maven 侧的读法用 `tools/diag_date_conventions.py` 逐一排除后确定：只有"最新发布日 + 2021-04-19 之后"
这一读法同时得到"跨度起点 2021-04-2x"和"最大出度 87"（论文 87，精确一致）；换成首次发布日则跨度
回到 2005 年、最大出度 90。本包的 Maven 节点数偏多 12%（因为我们额外用 group 目录做了种子扩展），
最大出度偏大（122），这两点属于**重建种子表的差异**，见 §5.1。

### 3.2 评测规模：候选池算术精确命中，来源数/正例数对不上

论文里"每个来源的平均候选数"是唯一能反推切分规则的数字：

```
npm  : 4 962 − 347 − 1 = 4 614    （本包 4 611 = 4 959 − 347 − 1）
Maven: 1 484 − 104 − 1 = 1 379    （论文写 1 376；本包 1 079，因节点集不同）
```

也就是说：**测试窗口 = 最新的 round(0.07·n) 个节点，候选池 = 测试窗口之外的全部节点减去来源自身**。
但这样一来"测试窗口内的目标节点"根本不在候选池里，无法参与排序，而论文 §5 又把真值定义为
Γ_out(u) = N_out(u) ∩ V_test —— 两者互斥，只能取一个（详见 §5.2）。

本包按"目标必须落在候选池内"实现（即预测新节点会依赖哪些**既有**节点），这也是唯一能让 MRR ≠ 0
的读法。实测规模：

| 量 | 论文 npm | 本包 npm | 论文 Maven | 本包 Maven |
|---|---|---|---|---|
| 每种子测试节点 | 347（7%） | 347 | ~104 | 116 |
| 测试来源数 | 234 | 181 | 94 | 29 |
| 正例数 | 686 | 494 | 352 | 63 |
| 平均候选数 | 4 614 | 4 611 | 1 376 | 1 547 |
| 测试边（含不可排序） | — | 693 | — | 327 |

论文的 (234 来源, 686 正例) 更接近"测试窗口内部边的数量和其来源数"（本包：205 个有出边的测试节点、
689 条窗口内边），而不是任何可排序的组合，这一条建议作者自己复核表格出处（见 §5.2）。

### 3.3 内容地板：recipe 必须标定，否则差 0.03

论文只写了"TF-IDF 只用训练文本拟合、截断 SVD 到 300 维、L2 归一化、权重 w = f·log(N/(1+df))"。
这句话不足以复现一个数：`tools/floor_sweep.py` 在 npm 固定切分上跑了 12 种自然读法，地板从 0.087 到 0.180。

| recipe（readme 长度 / 分词 / 其他） | npm 地板（10 种子） |
|---|---|
| `[a-zA-Z][a-zA-Z0-9_+#.-]*`（保留连字符、scope、点号）+ sublinear，readme 1500 | **0.1573** |
| sklearn 默认 `\b\w\w+\b` + sublinear，readme 1500 | 0.1633 |
| `\b\w+\b` + sublinear，readme 1500（最初实现） | 0.1669 |
| readme 500 / 无 readme / 只要 description | 0.1754 / 0.1384 / 0.0870 |
| 非 sublinear（即论文公式 w = f·log(N/(1+df))） | 0.1796 |
| max_features 5000 / min_df 2 / SVD-100 | 0.1645 / 0.1609 / 0.1394 |

论文的 npm 地板 0.1565 只有第一行命中（0.1573，差 0.5%）；注意到**论文给出的权重公式对应的是
"非 sublinear"**，而那个 recipe 给出 0.1796，与论文表格相差 15%。因此论文至少有一处描述与实际
实现不符：要么公式写错，要么表格里的地板来自另一套 recipe。

**第二处、也是更关键的一处**是文本怎么拼出来 —— 论文完全没写。同一套 TF-IDF 超参下，只改
"README 要不要先剥 HTML 标签 / 先截断还是后截断"，npm 地板就从 0.1559 变成 0.1624：

| 文本拼法（同一 recipe：code-aware 分词 + 英文停用词 + min_df 1 + sublinear + SVD300） | npm 地板（10 种子） |
|---|---|
| `name + desc + kw[:400] + readme[:1500]`，只折叠空白、不剥 HTML（**本包默认**） | **0.1559 ± 0.0018** |
| 同上，但先剥 HTML 标签、最后再截断到 1500（最初实现） | 0.1624 |

所以本包的最终口径是"论文表格能对上的那一套"，与论文值 0.1565 差 **0.0001**（十种子 0.1566 ± 0.0022）；`build_dataset._npm_record`
的注释里写明了这一点，`tools/floor_sweep.py` 可以重跑这张表。

### 3.4 四个阶段的口径

| 论文 | 本包实现 | 位置 |
|---|---|---|
| 每种子独立切分、十个不同的 train–test 划分 | 测试窗口固定为最新 7%（候选池算术要求：4,962−347−1 = 4,614）；种子间差异落在 fit/val 边界上，并**保证十个种子的验证窗两两不同**（取模单射构造，见 §5.9），十个训练边集互不相同 | `content_floor._split_indices` |
| 训练边 = fit 窗口的出边（边日期 = 源节点日期） | 同 | `content_floor.build_bundle` |
| 候选池排除来源自身与泄漏节点 | 排除测试窗口 + 来源自身；再排除来源在 fit 图中的出邻居 | `ranking.evaluate_ranking` |
| 每来源 MRR / hits@10 / hits@100 | 同，且只统计"至少有一个可排序正例"的来源 | `ranking.py` |
| 描述子 13 维、只用 fit 窗口统计量标准化 | 同（`gated.compute_descriptors`） | `gated.py` |
| 零初始化 U 与门控 → 未训练模型 = 内容地板 | 同，并在每次运行里记录 `init_val_mrr` 供审计核对 | `train3.py` |
| 调参种子 11、网格 hidden{64,128}×dropout{0.2,0.5}、lr 1e-3、wd 1e-5、heads 4 | 同（4 个网格点、lr 固定） | `MANIFEST.json` |
| 调参 ≤80 epoch patience 15；最终 ≤100 epoch patience 20；eval_every 2；10 负例 | 同 | `MANIFEST.json` |
| 逐 cell 选择超参（论文 Table 1 的 Config 列 c1/c2/c3/c4 各不相同） | 默认**逐 cell** 选择（每个 cell 取验证 MRR 最优的网格点），baseline 同样逐 cell；`--shared` 可切换成"每数据集一套超参"的稳健性对照 | `run_experiments.select_configs` |
| 节 5 的"learning-rate and dropout choices are identical across cells" | **与它自己的 Table 1 冲突**：Config 列逐 cell 不同，意味着 dropout 并不相同（论文只对 lr 固定这一点成立） | `README §5.8` |
| 检查点 = 验证 MRR 最优 | **只在训练过的 epoch 中选**（`best_epoch ≥ 1`），epoch 0 的地板分数单独记录 | `train3.py` |

### 5.9 十个种子的切分必须真正互不相同（RNG 抖动会撞车）

论文说十个种子各自独立切分。先前实现用 `rng.random()` 抽验证窗大小：npm（n=4 959、带宽 ±20）十次抽样撞车一次（9/10 不同），Maven（n=1 664、带宽 ±7）撞车**四次**（6/10 不同）——四个种子的切分逐位相同，那几对“种子间差异”里就没有数据划分差异，配对区间偏窄。现在改成“必有放回”的确定性构造：

```
半带宽 V = max(round(0.08·n·0.05), 12)          # 至少 12 个节点，保证带宽够大
偏移   = (5·(seed mod (2V+1))) mod (2V+1) − V    # 5 与 2V+1 互素 ⇒ 该映射是单射
```

只要十个种子的 `seed mod (2V+1)` 互不相同（本包必然如此：种子跨度 10 < 2V+1），就得到十个互不相同的 fit 窗口。审计 **B7** 查的就是这一条——上面那四次撞车正是它抓出来的。切分规则改动连同 `SPLIT_RULE_VERSION` 一起进 bundle 缓存键（D-4），旧 bundle 不会静默复用。

另一种读法（“十个独立划分”指**测试窗**也不同）由 `SPLIT_JITTER_TEST=0.05` 打开，此时候选池只在对均值上等于 4,614；审计 **B8** 会说明当前生效的是哪一种。改动前那一版（固定测试窗 + RNG 抖动）的完整结果保留在 `results_alt_fixedwindow/`，可与 `results/` 对照。

## 4. 工具与诊断脚本（`tools/`）

每一步判断都留了可重跑的脚本，而不是只写结论。

| 脚本 | 作用 | 关键输出 |
|---|---|---|
| `tools/floor_sweep.py` | 12 种文本 recipe × N 个种子，报地板 / 来源数 / 正例数 | §3.3 的表；用于标定默认 recipe |
| `tools/diag_date_conventions.py` | 用 created / modified / latest 三种节点日期约定重构图，报节点数、边数、度极值、日期跨度、测试规模；Maven 侧再叠加"首次发布日 ± 2021-04-19 过滤" | §3.1 的读法排除实验 |
| `tools/diag_splits.py` | 切分诊断：测试窗口大小、验证集占比是否落在 ±5% 带内、十个划分是否互不相同 | 与论文候选池算术对账 |
| `tools/maven_extra_seeds.py` | 从已抓到的 group 目录里扩展同族工件（每 group ≤20 个，总 ≤1,200）生成 `data/seeds/maven_extra.txt` | 解决 Maven BFS 队列耗尽（1,155 节点）的问题 |
| `tools/profile_train.py` | `cProfile` 跑若干 epoch，定位训练循环热点 | 发现 75% 的时间耗在逐元素评测里（见下） |
| `tools/test_ranking_equiv.py` + `tools/ranking_loop_reference.py` | 用随机投影打分（大量并列值）逐个来源比对"向量化评测"与"原逐元素循环" | 输出 `EQUIVALENT`，保证加速没有改变任何数字 |
| `tools/make_tables.py` | 从 `results/` 直接生成论文格式的三张 LaTeX 表（主结果 / 配对对比 / 数据集统计） | `results/tables/*.tex`，可直接替换投稿包里的同名文件 |
| `tools/compare_vs_paper.py` | 把本包的 `summary.json` 与论文 Table 1/2 并排放成 Markdown（主结果 / 配对对比 / 选中配置） | 结论对照：哪些数字复现、哪些不复现 |
| `tools/diag_epoch_curve.py` | **诊断，不是结果**：逐 epoch 记录 val/test MRR、温度、损失，回答"训练为什么可能低于地板"（它用测试集选点，属于泄漏，仅用于定位问题） | `logs/diag_epoch.json` + 每模型一行汇总 |
| `tools/compare_checkpoint_rules.py` | 并列比较不同检查点策略（`best_val` / `last` …）下的十条种子结果：地板、地板差、低于地板的种子数、配对对比 | `logs/checkpoint_rules.json` |
| `tools/gate_within_config.py` | 同一网格点内比较门控（`ragat_sym`−`gat_dir`、`ragat_time`−`gat_time`），排除"逐 cell 选超参"带来的混淆 | §5.10 的表 |
| `tools/diag_check_trajectory.py` | 用生产训练器重跑同一个 (数据, 种子, 模型, 配置)，和 `diag_epoch_curve.py` 的验证曲线逐位比对，证明诊断没有测另一只模型 | 输出 `IDENTICAL` / `DIFFERENT`（§5.11 第 3 条） |
| `tools/diag_alt_candidates.py` | **诊断**：同一次训练在两种候选池读法下打分（A：窗外全部节点；B：除来源外全部节点），看哪种读法能产生"模型低于地板" | `logs/diag_alt_candidates.json` |
| `tools/diag_floor_pool_readings.py` | **诊断**：十种子下两种读法的内容地板，再与已落盘的模型结果做"同读法 / 错配读法"交叉表 | `logs/floor_pool_readings{,_maven}.json`；§5.12 的表 |
| `tools/make_revision_tables.py` | 生成改稿包用的四张表（主表 / 配对 / 协议敏感性 / 检查点策略），表格结构与稿件一致，可直接替换 | `results/revision_tables/*.tex`；§8 |
| `tools/make_revision_figures.py` | 用重建结果重画三张结果图（主结果、配对对比、逐种子门控差），旧图与旧数字绑定、必须一起换 | `results/revision_figures/*.pdf`；§8 |
| `tools/make_revision_figure_datasets.py` | 重画数据集图 `fig_datasets.pdf`（改稿后只画 npm/Maven：出/入度 CCDF + 节点到达曲线，含两图的节点日期口径差异） | `results/revision_figures/fig_datasets.pdf`；§8 |
| `tools/fix_fig_mechanism_labels.py` | **（历史，已被下面的重绘取代；保留用于复现当时的产物）** 在**作者原图**（`assets/fig_mechanism.original.pdf`，即投稿树里的那一版）上局部改字：把门控框里的 `g_d = 1 + tanh(MLP(ρ(u)))` 与 `MLP zero-init` 改成单层线性写法（`w^T ρ(u) + b`，14 参数/gate），架构示意不动；以 Nimbus Roman 子集重排，字体全部内嵌 | `results/revision_figures/fig_mechanism.pdf`；§8 |
| `tools/fix_fig_mechanism_arrows.py` | **（历史，已被下面的重绘取代）** 在**改字版机制图**上修两处绘制缺陷：(a) 把 `×g_out` 标签整体平移 (+1.10, −25.66) pt 到 out-view 节点正上方——原位置正好被门控箭头的箭头尖压住下标 `out`，平移规则与 `×g_in` 完全一致（左缘 = 节点 x0 − 0.63，底 = 节点顶 − 3.44）；(b) 三处连接箭头原来只剩一个箭头尖（间隙 5.56 / 6.90 / 3.93 pt 装不下 4.4 pt 长的头加杆），重新按「底边贴源框、尖点内切进目标框描边」摆放，并给 embedding→cosine 补 1.8 pt 可见杆。只用 4 个小矩形做 redaction（`REMOVE_IF_COVERED`，不碰相邻的 ⊗ 与框），输出整数坐标加固定 trailer `/ID`，两次运行字节一致 | `results/revision_figures/fig_mechanism.pdf`；§8 |
| `tools/make_fig_mechanism.py` | 机制图**重绘**（取代上面改字 + 修箭头两步）：PDF 页幅 415 × 232 pt **设计画布**（不是印刷宽度）。作者原图 762.12 × 380.30 pt 在 `width=\textwidth` 下被缩到 0.4865 倍，9.5 pt 实际只印出 **≈4.6 pt**，这才是“挤、糊、压”的根因；新图实测按 **0.893 倍**印出（sn-jnl 单栏 `\textwidth` = 370.7 pt），图内正文 **5.0–6.6 pt**、上下标 3.7–4.1 pt。脚本内置 `FIG_FONT_SCALE`（默认 1.0；1.06 起审计即 FAIL，故 1.0 是本版式上限）。Nimbus Sans（Helvetica 同族）三个子集全部内嵌；`freeze_pdf_id()` 冻结 PyMuPDF 的随机 trailer `/ID`，连续三次构建字节一致 | `results/revision_figures/fig_mechanism.pdf` + `.svg` + `.png`；§8 |
| `tools/check_fig_occlusion.py` | 机制图**遮挡审计**（可验证，不靠肉眼）：按基线取全部文字墨迹框与全部线段两两求交，报文字互压 / 线段穿字 / 文字越框三类问题与最小净距。实测 111 词 / 208 线段 / 14 容器：`text_text_overlaps 0`、`line_ink_hits 0`、`words_crossing_a_border 0`、最小净距 **1.05 pt**，`RESULT: PASS` | 终端报告；§8 |
| `tools/scan_stale_numbers.py` | 扫描稿件树里**已被重跑退休**的数字与论断（旧门槛 0.1565/0.3377、旧门控增益 0.0297/0.0311、旧模型规模 4,962/1,484、旧测试规模 234/686、旧"引用网络作控制"表述…），逐条给出"原值 → 新值"和命中行；退出码非 0 便于当发布门 | 改稿前原稿命中 109 处 / 19 个文件；装完改稿包后 **0 处**；§8 |
| `tools/check_revision_package.py` | 自检改稿包：把包 `apply.sh` 装进一份临时稿件副本，再检查 `\input` 是否解析、`\ref` 是否有 `\label`、`\label` 是否重复、`\begin/\end`、括号与 `$` 是否配平；v18 起还要求 `tables/table_citation_control.tex` **存在且被 `\input`**（v17 时规则相反——那时这张表被删了，所以要求该 key 不存在） | 本次实测 28 个 .tex / 73 个 label / 128 个引用，`all structural checks passed`、**0 warning**；带 `--expect build/manuscript` 时再逐字节比对，**35 个安装文件全部一致**（`.revised` 头部注释块除外）；§8 |

`ranking.evaluate_ranking` 原来是逐节点 Python 循环（每个来源对 4 600 个候选做一次列表推导 + 字典
建排名），npm 上一次评测 5.5 s，占单个训练任务 wall time 的约 75%（7.6 s/epoch，一次 tuning 配置
530–607 s）。改成 numpy 掩码 + `argsort` 后：**1.8 s/epoch，一个 tuning 配置 78–261 s**，全套
（56 个 tuning + 140 个 final 任务）从约 9 小时降到约 2 小时。等价性由上述脚本逐来源核对。

### 4.1 扩展实验 `tools/ext/` 与 `EXT-EXPERIMENTS.md`（v18 新增）

v17 那批"有界陈述 + 未来工作"在 v18 里全部跑成了真结果，脚本与产物如下（**说明书就是
`EXT-EXPERIMENTS.md`**：每条命令、输出路径、头条数字、正文落点、以及"不能声称什么"）。

| 脚本 | 作用 | 关键输出 |
|---|---|---|
| `tools/ext/e1_variance_decomposition.py` | 两因素方差分解（图 × 单元，切分嵌套在图内）：报平方和份额、每种子里的单元排序与 Spearman ρ | `results_ext/e1_variance/e1_variance.json`；`table_variance.tex` |
| `tools/ext/e2_structural_baselines.py` + `e2_report.py` | 十种无训练结构启发式，**在两张图上各报一次**（归纳图 vs 含测试边的 `mp_edges_test` 图） | `results_ext/e2_structural*`；`table_structural_baselines.tex` |
| `tools/ext/e3_text_floor.py` | 四种文本配方（word / char 3–5 gram / union / 大词表）只换特征重算地板，`baseline` 行须逐种子复现冻结地板（对照锚） | `results_ext/e3_text_floor/`；`table_text_floor.tex` 上半 |
| `tools/ext/e3c_stratified.py` | 按描述长度四分位与命名空间分层报地板与增量（读 E3a/E4 的 per-source 数组） | `results_ext/e3c_stratified/`；`table_stratified.tex`（两张浮动表） |
| `tools/ext/e4_instrument.py` + `e4_report.py` | 门控插桩：取值范围、移动比例、门控/编码器梯度范数、描述子相关、零 vs 非零初始化、**epoch-0 锚**、`--graph-free` 无图消融；同时落盘 `ps_*.npz` per-source 数组 | `results_ext/e4_instrument/`、`e4_instrument_report.{json,md}`；`table_gate_diagnostics.tex` |
| `tools/ext/e5_per_source_inference.py` | 把复现单位从种子换成**来源**：bootstrap / 配对 t / 符号检验 / Wilcoxon / 1%·5% 截尾均值 / 效应集中度 | `results_ext/e5_persource/`；`table_persource.tex` |
| `tools/ext/e6_maven_firstpublish.py` + `e6_report.py` | Maven 按"首次发布日期"重构图并重跑整条链（含去窗口过滤的变体），与冻结口径并排 | `results_ext/e6_maven_firstpub_report.{json,md}`；`table_maven_dates.tex` |
| `tools/ext/e7_citation_control.py` + `e7_report.py` | 引用网络对照**从一手数据重建**：SNAP 边表 + arXiv Atom API（`--fetch` 抓元数据、`--build` 建图、`--sources` 跑链） | `data/graphs_ext/`、`results_ext/e7_*`；`table_citation_control.tex` |
| `tools/ext/e8_pairwise_ncn.py` | NCN/BUDDY 式**学习式成对结构排序器**（3 个一阶特征 / 8 个全特征两档） | `results_ext/e8_pairwise/`；`table_pairwise.tex` |
| `tools/ext/make_extension_tables.py` | 把上面九张表从 JSON 生成出来（`_stack()` 做"估计值/区间"上下排；模型名转义下划线） | `artifacts/rcgat-revision-package/tables/*.tex` |

**共同前提**：`pip install torch torch-geometric`（CPU 版即可），并导出
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`
——不钉住 BLAS 线程，同一份切分会给出不同的 SVD 结果，`baseline` 配方就不再逐种子复现地板。
所有脚本**可续跑**：已存在的 JSON/NPZ 会被跳过；E7 的抓取有落盘缓存，不重复请求 arXiv。

## 5. 已知偏差与论文内部不一致（请在投稿前逐条处理）

### 5.1 Maven 的数据层是**重建**，不是复现

npm 侧可以对账到"几乎逐项命中"（§3.1），Maven 侧不行，原因是数据本身无法复原：

1. **种子表缺失**。论文没有给出 Maven 的起始工件集，本包用"手写热门工件 + group 目录扩展"重建。
   用 group 目录扩展的副作用是：会把**当天刚发布**的工件大量拉进来（本包 1 664 个节点里 379 个
   的发布日在抓取前 7 天内，占 23%）。
2. **抓取时点差 4 天**。论文跨度止于 2026-09-21，本包抓到 2026-09-25。测试窗口取"最新 7% 节点"
   时，本包落进了 09-24 .. 09-25 这两天的发布train里。
3. 后果是这个测试切片**明显退化**：Maven 每个种子 116 个测试节点、327 条测试边，其中
   **264 条（81%）指向测试窗口内部**（同族模块同批发布），这些目标不在候选池里、无法参与排序；
   真正可排序的只有 29 个来源 / 63 个正例。论文报告的是 94 个来源 / 352 个正例，说明论文的测试
   切片里绝大多数新节点的依赖指向**窗口外的老节点**——这只有在节点日期分布不偏向抓取当天时才成立。
4. 因此 Maven 的**绝对数值不要与论文表格逐格对照**（地板 0.096 vs 论文 0.3377 主要就是第 3 点造成的，
   本包的地板是可排序目标很少的糟糕切分下的地板）。Maven 侧应作为"同一协议、不同数据快照"的
   独立复现看，重点看**方法之间的配对差**（gate / time / 组合），这部分对切分退化不敏感。
5. 本包另跑了一个"跨度对齐"变体（`MAVEN_MAX_DATE=2026-09-21`，节点 1 285 / 边 3 655），
   测试窗口退化程度类似（90 个测试节点、30 个来源、82 个正例），所以没有采用它——见
   `tools/diag_date_conventions.py` 的输出。

### 5.2 论文 §5 的 Γ_out(u) = N_out(u) ∩ V_test 与它自己的候选池定义互斥

候选池算术（§3.2）要求测试窗口内的节点**全部**排除在候选池之外，此时 Γ_out 里的节点一个都不在
候选池里，MRR 恒为 0。本包按"目标必须落在候选池内"实现：真值 = 测试窗口来源的、指向**窗口外**
节点的出边。论文需要明确改成后者，否则 MRR 无法计算。

### 5.3 论文的 TF-IDF 权重公式与它报告的内容地板不一致

论文写的是 w = f·log(N/(1+df))（即不做 sublinear-TF），但按该读法实现，npm 地板是 0.1796，
比表格里的 0.1565 高 15%；只有 sublinear-TF + 保留连字符/点号/scope 的分词（`(?u)[a-zA-Z][a-zA-Z0-9_+#.-]*`）
才落在 0.1573（差 0.5%）。另外本包实测：文本拼法（README 是否剥 HTML）比分词器影响还大（0.1559 vs 0.1624），
论文对此完全没有描述。**公式或表格至少有一处需要改**，本包默认用后者，并在
`feature_metadata` 里记录所有文本相关超参，便于作者统一口径。

### 5.4 随机 SVD 的数值噪声会把地板推动 ±0.006

`TruncatedSVD` 的随机算法对输入**行顺序**敏感：同一 recipe、同一数据，仅把 fit 文本的排列顺序换成
按日期排序，npm 地板就从 0.1573 变成 0.1655。修法是固定 fit 文本为上升索引序 + `n_iter=10` +
`OMP_NUM_THREADS=1`（`content_floor.build_bundle` 已固化）。`audit.py` 的 D3 检查会用同一 bundle
重算地板并要求逐位一致，所以任何人复现时都会得到同一个数。

### 5.5 检查点选择必须排除 epoch 0

门控 MLP 与残差投影 U 都是零初始化，所以**未训练模型就等于内容地板**。若让 epoch 0 参与
"验证 MRR 最优"的检查点选择，模型会直接选初始权重（本包第一次 npm 调参就卡在 `best_epoch=-1`，即整条曲线都没能超过地板）。
`train3.py` 现在只在训练过的 epoch 里选检查点，并把 `init_val_mrr` 单独记录下来供审计核对
（`SET SELECT_FROM_INIT=1` 可以恢复旧行为做对照）。

### 5.6 来源数 / 正例数对不上（234 / 686）

实测可排序规模是 181 来源 / 494 正例；论文的 234 / 686 更接近"测试窗口**内部**边的来源数与边数"
（本包：205 个有出边的测试节点、689 条窗口内边），但那些边按论文自己的候选池定义不可排序。
这一条与 §5.1 的第 3 点是同一个成因（测试窗口内的同批发布），建议作者核对该表格的原始口径。

### 5.8 "dropout 在四个 cell 间一致"与 Table 1 自相矛盾

论文实验设置里写"learning-rate and dropout choices are identical across cells"，但 Table 1 的 Config
列逐 cell 不同（npm：gat_dir=c3、rcgat_sym=c2、gat_time=c1、rcgat_time=c4；Maven 同样四处不同），
而 c1..c4 恰恰就是 hidden × dropout 的四个组合。所以**只有学习率是跨 cell 固定的**。本包默认按
Table 1 的实际做法逐 cell 选（`--shared` 保留另一读法做稳健性对照），并建议作者改掉这句话。

### 5.10 检查点策略是论文没写的一个自由度（`CHECKPOINT_RULE`）

论文只说"检查点 = 验证 MRR 最优"，但**没有说最优是在哪个范围内取**。这一点不是细节：门控 MLP 与残差投影都是零初始化，未训练模型就等于内容地板，因此

* 若允许 epoch 0 参选，模型会退化成地板，而论文里"显著低于地板"的结论在数学上就不可能成立（D-6）；
* 若取"最后一个 epoch"，训练可以被允许跑过最优点，**这才是唯一能让模型低于地板的自然读法**；
* 若取"验证最优的训练 epoch"（本包默认），检查点至少在验证集上是最优的，低于地板只能来自验证/测试的错位。

`train3.py` 现在把这三种读法都做成了开关，并写进每个 run JSON（`checkpoint_rule` / `selected_epoch` / `best_epoch`），所以跑出来的是哪一种在产物里可查，不用猜：

```bash
CHECKPOINT_RULE=best_val            python src/run_experiments.py --phase final ...   # 默认
CHECKPOINT_RULE=last                python src/run_experiments.py --phase final ...   # 最后 epoch
CHECKPOINT_RULE=first               python src/run_experiments.py --phase final ...   # 第 1 个 epoch
CHECKPOINT_RULE=best_val_with_init  python src/run_experiments.py --phase final ...   # 允许 epoch 0（旧行为）
```

复现一套与主结果**同一套超参**的对照不需要重跑调参：把 `results/selections.json` 拷到另一个结果根目录即可，`select_configs` 会在缺少 tuning 结果时沿用已冻结的逐 cell 配置。

```bash
mkdir -p results_ckpt_last && cp results/selections.json results_ckpt_last/
RESULTS_ROOT=$PWD/results_ckpt_last CHECKPOINT_RULE=last \
  OMP_NUM_THREADS=1 python src/run_experiments.py --phase final --datasets npm \
    --models gat_dir rcgat_sym gat_time --workers 4
python tools/compare_checkpoint_rules.py --dataset npm best_val=results last=results_ckpt_last \
  --out logs/checkpoint_rules.json
```

`RESULTS_ROOT` 是相对回收代码新增的开关（回收的 `run_phase4/5/6.py` 把输出路径写死），保证对照实验永远不会覆盖主结果。结论见表 §5.11。

### 5.11 诊断：论文"模型低于地板"从哪来（检查点与温度都排除）

论文 npm 上除 time cell 外**所有模型都低于内容地板**（gat_dir 0.1214 vs 地板 0.1565，即训练让 MRR 掉了 0.035），而本包同一模型、同一损失、同一预算比地板高 0.008。相差 0.05 的符号翻转只有三个可能来源：检查点策略、评分器温度、数据/切分。`tools/diag_epoch_curve.py`（**诊断脚本，不是结果**：它逐 epoch 看测试集，等于用测试集选点，属于泄漏，只用于定位问题）在 npm、种子 101、测试地板 0.1559 上给出：

| 模型 | 预算 | 验证最优 epoch | 该点测试 MRR | 最后 epoch 测试 | 逐 epoch 最低 | 低于地板的 epoch 数 |
|---|---|---|---|---|---|---|
| gat_dir | 100 | 100 | 0.1599 | 0.1599 | 0.1545（ep6） | 1/50 |
| gat_dir | 60 | 56 | 0.1679 | 0.1641 | 0.1545（ep6） | 1/30 |
| ragat_sym | 60 | 58 | 0.1571 | 0.1562 | 0.1560（ep56） | 0/30 |
| gat_time | 60 | 58 | 0.1796 | 0.1847 | 0.1577（ep6） | 0/30 |

1. **检查点策略排除**。100 epoch 预算下验证 MRR 到 ep100 仍在上升（0.1695），所以 `best_val` 与 `last` 取到同一个模型、同一个测试值；五十次评测里只有 ep6 一次低于地板，且只低 0.0014（噪声量级）。十条种子的 `last` 规则另跑了一遍（`results_ckpt_last/`，`tools/compare_checkpoint_rules.py`）：

| cell | best_val 测试 MRR | Δ地板 | 低于地板种子 | last 测试 MRR | Δ地板 | 低于地板种子 |
|---|---|---|---|---|---|---|
| gat_dir | 0.1644 | +0.0078 | 0/10 | 0.1642 | +0.0076 | 0/10 |
| ragat_sym | 0.1623 | +0.0057 | 0/10 | 0.1615 | +0.0049 | 1/10 |
| gat_time | 0.1811 | +0.0245 | 1/10 | 0.1775 | +0.0209 | 1/10 |
| 配对 gate | −0.0021 [−0.0055,+0.0014] 5/10 | | | −0.0027 [−0.0068,+0.0014] 5/10 | | |
| 配对 time | +0.0167 [+0.0057,+0.0277] 8/10 | | | +0.0133 [+0.0030,+0.0236] 8/10 | | |

   两种规则给出同样的定性结论：**所有 cell 都在地板上方**，−0.035 不是任何检查点能取到的值。

2. **温度排除**。可学习温度在 100 epoch 里只从 9.98 漂到 9.29，"温度发散把排序压平"不成立。

3. **诊断本身先被验过**。`tools/diag_check_trajectory.py` 用生产训练器（`train3.train_model`）在完全相同的 (数据, 种子, 模型, 配置) 上重跑 100 epoch，验证曲线与诊断脚本**逐位一致**（51 个评测点 max|Δval| = 0，ep100 双方都是 0.169476）。所以第 1、2 条排除不是脚本误差造成的。

4. 剩下只能是数据/切分 —— 见下一节。

### 5.12 最可能的出处：论文的地板与模型可能不是同一个候选池读法

论文 §5 的真值定义 $\Gamma_{out}(u) = N_{out}(u) \cap V_{test}$ 与它自己的候选池算术（4,962 − 347 − 1 = 4,614，即整个测试窗都在池外）互斥（§5.2）：前者要求窗内节点可排序，后者要求它们不可排序。回收到的 `content_floor.py` 里写的却是 `core_mask = np.ones(nm, dtype=bool)  # all present nodes are core`，即**候选池 = 除来源外全部节点**。该文件自称 "Reconstructed from the protocol spec (not verbatim)"，所以这条只是"另一种读法确实会被写出来"的证据，不是原始实现的证据——但它说明这个自由度是真实存在的。

两种读法的内容地板差很多（正例集也不同）：

| 数据集 | 读法 A：池 = 窗外全部节点（本包默认，候选 4,611 / 1,547） | 读法 B：池 = 除来源外全部节点（候选 4,958 / 1,663） |
|---|---|---|
| npm | 0.1566 ± 0.0022（181 来源 / 494 正例） | 0.1731 ± 0.0033（205 / 693） |
| Maven | 0.0922 ± 0.0042（29 / 63） | 0.1433 ± 0.0031（106 / 327） |

**地板与模型用同一读法时，所有 cell 都在地板上方**；把两者**错配**（模型按 A 打分，地板按 B 记）则精确复现论文的符号与量级（`tools/diag_floor_pool_readings.py`，十种子，括号内为低于地板的种子数）：

| 数据集 | 组合 | gat_dir | ragat_sym | gat_time | ragat_time |
|---|---|---|---|---|---|
| npm | 同读法 A/A | +0.0078（0/10） | +0.0057（0/10） | +0.0245（1/10） | +0.0087（3/10） |
| npm | A 模型 / B 地板 | −0.0088（**9/10**） | −0.0108（**10/10**） | +0.0079（4/10） | −0.0079（7/10） |
| Maven | 同读法 A/A | +0.0116（1/10） | +0.0102（2/10） | +0.0115（1/10） | +0.0108（1/10） |
| Maven | A 模型 / B 地板 | −0.0394（**10/10**） | −0.0409（**10/10**） | −0.0395（**10/10**） | −0.0403（**10/10**） |

论文 npm 报的是 `gat_dir − floor = −0.0351`：**符号、正例关系、量级都与"两个池子错配"这一行一致**（Maven 那一行的 −0.040 与论文 npm 的 −0.035 同量级），而且这个机制顺手解释了为什么只有 time cell 还留在地板之上。

结论：本包在任何**自洽**读法（A/A 或 B/B）、任何检查点规则下都复现不出"模型低于地板"，而池子错配可以精确复制这个现象的符号与量级。所以论文至少有一处是**地板与模型用不同口径算的**，需要作者核对（`tools/diag_floor_pool_readings.py` 两分钟可复算）。这直接决定"训练反而有害"这一叙述能不能保留。

### 5.7 未复现的项

* 论文里的**引用量对照实验**（citation control）没有对应代码，也没有被抓取的数据；本包未实现。
* 论文未给出各表的 per-seed 明细，本包只能给出自跑结果的均值/标准差与配对区间。
* 论文没有公开种子表、原始快照与超参搜索日志，所以 §5.1/§5.6 的偏差无法进一步收敛。

## 6. 运行时间与资源（4 vCPU / 8 GB，CPU-only）

| 阶段 | 任务数 | 单任务 | 4 workers 总时长 |
|---|---|---|---|
| `floor` + `diag` | 22 | 8–20 s | < 5 min |
| `tune`（两数据集） | 56 | 78–261 s | 约 45 min |
| `final`（两数据集，7 模型 × 10 种子） | 140 | 100–330 s | 约 2 h |
| `final`（切分规则 v2，两数据集） | 140 | 35–410 s | 约 50 min（含 Maven 的快模型） |
| `summarize` + `audit` + 制表 | — | 秒级 | < 1 min（审计 414 项） |
| 诊断（§5.11/§5.12，非结果） | 6 次训练 + 地板复算 | 1–5 min | `diag_epoch` ≈ 12 min、`diag_alt_candidates` ≈ 18 min、两个地板对照 ≈ 3 min、`last` 十种子 ≈ 85 min |

`data/bundles/*.pt` 每个约 2–7 MB，22 个 bundle 约 60 MB；原始快照 `data/raw_npm`、
`data/raw_maven` 各约数十 MB，均已压缩存储、不入库。

## 7. 修复清单（相对回收到的 `rc_*` 代码）

| # | 回收代码的症状 | 影响 | 本包的修法与证据 |
|---|---|---|---|
| D-1 | 整条数据层不存在：没有"注册表 JSON → 节点/边/文本"的适配器，没有种子表，没有节点日期约定 | 任何脚本都跑不起来 | 新增 `fetch_npm.py` / `fetch_maven.py` / `build_dataset.py`；npm 口径用 `time.created` + 最新版 dependencies，逐项命中论文统计（§3.1） |
| D-2 | `content_floor.py` 按**边**做 70/10/20 时间切分，还用 `y*365+m*30+d` 的假日索引 | 与论文的"按节点日期切分"不是同一任务 | 改为按节点日期切分；测试窗口固定为最新 7%（候选池算术要求），每种子抖动只作用于 fit/val 边界；原先的实现会让测试集变成 491 个节点而不是 347 |
| D-3 | TF-IDF/SVD 在**全体**文本上拟合（论文与自己的 docstring 都说是 train-only） | 文本泄漏 | 只在 fit 窗口文本上拟合；并固定 fit 文本为升序索引序 + `n_iter=10`，消除随机 SVD 的行序噪声（±0.006 MRR） |
| D-4 | bundle 缓存键不含文本 recipe | 改了 recipe 仍会静默复用旧 bundle（本次标定中真实踩到） | 缓存键加入 recipe 指纹（token pattern / stopwords / max_features / min_df / sublinear / 维度） |
| D-5 | `ranking.evaluate_ranking`：正例全在池外时 `min()` 抛 `ValueError`；候选池不排除来源自身；逐元素 Python 循环 | 崩溃 + 口径错误 + 占 75% 训练时间 | 逐项修正；向量化后每次评测 5.5 s → 0.4 s，单个训练任务 7.6 s/epoch → 1.8 s/epoch；等价性由 `tools/test_ranking_equiv.py`（含大量并列值的鲁棒打分）逐来源比对，输出 `EQUIVALENT` |
| D-6 | `train3.py` 允许 epoch 0（零初始化 ⇒ 等于内容地板）参与"验证最优"检查点选择 | 模型退化成地板，且"显著低于地板"的结论不可能成立 | 只在训练过的 epoch 里选检查点，epoch 0 的分数单独记为 `init_val_mrr`；审计 H 项要求它逐位等于地板 |
| D-7 | `summarize6.py` 把地板**硬编码**为 `{'npm':0.1565,'maven':0.3377}`，t 临界值用手打表（df 只到 20） | Δ floor 一列不可核验，数据一变就悄悄错 | 地板从 `results/floor/` 读取（与模型同一划分），区间用 `scipy.stats.t` |
| D-8 | `run_phase4/5/6.py` 三个近似重复的 runner，导入路径不一致，注册表与论文矛盾 | 无法一次跑完整条流水线 | 合并为 `run_experiments.py`：`floor / diag / tune / select / final`，可续跑、`--workers` 并行 |
| D-9 | 论文声称"审计逐项通过"，但回收到的 `audit6.py` 不能失败（只打印，不比对） | 审计等于没有 | `audit.py` 实现 A–H 共 **414 项**可失败的检查（本轮 414 通过 / 0 失败）（图快照/切分/泄漏/地板/账目/对比/选择复算/初始化锚点）；本次修复中正是它先暴露了评测口径写错（4 项 FAIL） |
| D-10 | 论文正文说 dropout 跨 cell 一致，但它的 Table 1 逐 cell 配置不同 | 超参口径自相矛盾 | 默认**逐 cell** 选择（与 Table 1 一致），`--shared` 保留"每数据集一套超参"的对照读法（§5.8） |
| D-11 | 切分抖动用 `rng.random()` 抽验证窗大小：Maven 上十个种子只有 6 个不同的划分（四个逐位相同） | “十个独立切分”名不副实，配对区间偏窄 | 改为取模单射构造 + 半带宽下限 12 节点，审计 B7 由 FAIL 转 PASS（§5.9） |
| D-12 | `run_experiments.py` 把命令行里的**论文别名**原样写进 run JSON 的 `registry_model`（`rcgat_sym`），而所有下游工具按注册表名（`ragat_sym`）索引 | 用论文名跑出来的结果会从汇总/对照表里**静默消失**（本次 10 个 run 真的消失过，§5.11 的 `last` 批次才发现） | 写盘前统一 `gated.canonical`；`tools/compare_checkpoint_rules.py` 对旧产物也做同样映射 |
| D-13 | 回收的 runner 把输出路径写死，无法做"同一套超参、换一个检查点策略"的对照 | 任何对照实验都会覆盖主结果 | 新增 `RESULTS_ROOT`（独立结果根）与 `CHECKPOINT_RULE`（§5.10）；配置延续靠拷贝 `selections.json`，`select_configs` 在缺 tuning 结果时沿用已冻结的逐 cell 配置 |

## 8. 改稿包（路线 A）

结果节与全部结果表/图已按本包的重建结果重写，交付物在 `artifacts/rcgat-revision-package/`：

```
07_results.revised.tex                  # 结果节（sec:results/gate/time/protocol/citation）
sections/{01_abstract,03_introduction,05b_datasets,05_methodology,
          06_experimental_setup,08_discussion,09_conclusion}.revised.tex   # 其余受影响的章节
                                         # 05_methodology：Γ_out 目标集、指标分母、候选池指针、
                                         # 门控写成单层线性（56 参数）、训练段、图注
sections/highlights.revised.txt         # 亮点
tables/table_main_results.tex           # tab:main   （同一 \label）
tables/table_paired.tex                 # tab:paired （同一 \label）
tables/table_protocol_sensitivity.tex   # 新增 tab:protocol
tables/table_checkpoint_rule.tex        # 新增 tab:checkpoint
tables/table_dataset_stats.tex          # tab:datasets（本次快照重算）
tables/table_dataset_overview.tex       # tab:data_overview（上块重算，下块标注"未重跑"）
tables/table_configs.tex                # 网格表（只改 caption：调参种子 11、按图与 cell 选择）
figures/fig_{main_results,gate_contrast,gate_perseed,datasets}.pdf
figures/fig_mechanism.pdf               # 机制图：重绘版（make_fig_mechanism.py 生成，真矢量 + 真文字，零遮挡）
figures/fig_mechanism.svg               # 同一份坐标导出的可编辑 SVG（114 个真 <text>）
original/                               # 改稿前的原始文件 + 未改字的 fig_mechanism.original.pdf
CHANGELOG-numbers.md                    # 入口文档：逐句变更 + 需作者确认的 S-28、S-32..S-34（§25 是 v18 补记）
tables/table_{variance,structural_baselines,text_floor,stratified,gate_diagnostics,\
        persource,maven_dates,citation_control,pairwise}.tex   # v18 九张扩展表（8 → 17 张）
apply.sh                                # 自动判定布局、自动 .bak、支持 --dry-run
```

生成、复算与自检（本仓库内，无需 GPU）：

```bash
python tools/make_revision_tables.py --out results/revision_tables   # 四张结果表
python tools/make_revision_figures.py --out results                  # 三张结果图
python tools/make_revision_figure_datasets.py                        # 数据集图
python tools/make_fig_mechanism.py --out results/revision_figures --stem fig_mechanism
                                                                     # 机制图重绘（PDF + 可编辑 SVG + 预览 PNG）
#   FIG_FONT_SCALE 旋钮（默认 1.0）用来复跑字号上限的反例：=1.06 时 out-neighbours 与 out 文字重叠、
#   =1.12 时 out-view aggregation 压到卡片边框（两次 check_fig_occlusion.py 都 FAIL），故 1.0 是本版式上限。
python tools/check_fig_occlusion.py results/revision_figures/fig_mechanism.pdf
                                                                     # 遮挡审计，期望 RESULT: PASS
# 历史工具（改字版 + 修箭头版，已被重绘取代，仅用于复现当时产物）：
#   python tools/fix_fig_mechanism_labels.py --src assets/fig_mechanism.original.pdf --out <新图>
#   python tools/fix_fig_mechanism_arrows.py --src <改字版> --out <新图>
python tools/gate_within_config.py                                   # 同配置门控复核（逐 cell 调参的混淆）
python tools/check_revision_package.py                               # 改稿包结构自检（实测 28 文件 / 73 label / 128 ref，0 warning）
#   v18 另可用 --expect <构建树> 逐字节比对（剥掉 .revised 头部注释块）：35 个安装文件全部一致
python tools/ext/make_extension_tables.py --out ../artifacts/rcgat-revision-package/tables
                                                                     # 九张扩展表重生成（E1–E8）
python tools/scan_stale_numbers.py /path/to/manuscript               # 旧数字残留扫描（装包后应为 0）
python src/audit.py                                                  # 414/414 PASS
```

要点：重写后的结果节中没有任何"低于内容门槛"的 cell（与 D-6/D-7 及 `content_floor` 的候选池口径直接相关，见 §5.12 的 `tab:protocol`），预注册的门控对比在两图都不通过判定规则，两图最强的模型都是外部 GCN 基线。这些结论不依赖检查点策略（§5.10 的 `tab:checkpoint`：`best_val` 与 `last` 判定一致），也不依赖逐 cell 的超参选择（`tools/gate_within_config.py` 同配置复核：npm −0.0042、Maven −0.0109，都是负的）。

---

## 发布这份仓库时（PUBLISH.md）

本仓库包含一个 2.7 GB 的 `data/raw_npm/`（4,959 个原始包元数据 JSON）与一个 98 MB 的
`data/bundles/`（TF-IDF+SVD 特征缓存）。**发布公开 artifact 时这两个目录都不发**，理由与替代方案见
**`PUBLISH.md`**：

- `data/raw_npm/` 可由 `src/fetch_npm.py` 重抓（约 30 分钟，华为云镜像），
  与稿件 Data Availability 里"原始元数据来自公开 registry、重抓脚本与快照时点随包发布"的口径一致；
- `data/bundles/` 是缓存，`src/content_floor.py::build_bundle()` 会按需重建，
  `src/audit.py` 的 check D3 本来就是**重建后再复算**门槛（不读缓存），
  所以缺失不影响任何审计；`run_all.sh` 已强制单线程 BLAS 保证 SVD bitwise 稳定。

排除这两个目录后，可发布子集体积约 **37 MB / 18,311 个文件**（内容 ≈28,854 KB；v18 新增的
`results_ext/` 6.0 MB、`data/raw_citation/` 14.6 MB 与 `data/graphs_ext/` 2.3 MB 是主要增量，
`data/raw_citation/` 的再分发口径见 `PUBLISH.md` §2.5）（`data/raw_maven/` 8 MB 建议保留：
论文对 Maven 快照有披露，发出快照才能让该披露被逐一核对）。

一键打包（在本仓库的上一级目录执行）：

```bash
tar --sort=name --mtime='2026-09-27 00:00:00Z' --owner=0 --group=0 --numeric-owner \
  --exclude='repro/data/raw_npm' --exclude='repro/data/bundles' \
  --exclude='__pycache__' --exclude='*.pyc' \
  -czf rcgat-repro-publishable.tar.gz repro
```

打包结果是**逐字节可复现的**：四个 tar 参数把条目顺序（`--sort=name`）、mtime（`--mtime`）
与 uid/gid（`--owner=0 --group=0 --numeric-owner`）全部钉死，而 `tar -z` 是管道模式，gzip 不写时间戳。
同一命令连跑两次 sha256 相同（实测）；**去掉这四个参数，内容一样但字节不一样**，别指望两者哈希相等
（换 tar/gzip 版本也不保证一致）。发布的那一份文件名带日期（`rcgat-repro-publishable-2026-09-27.tar.gz`），
文件名、字节数与 sha256 记在 `AUTHOR-ACTIONS.md` §15 与 `CHANGELOG-numbers.md` §16，
不与包内容互相引用，避免自指。
