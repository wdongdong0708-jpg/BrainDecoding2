# BrainDecoding

非侵入式脑电/脑磁（EEG/MEG）连续语义解码与延迟承诺（Delayed Commitment）研究代码库。

---

## 1. 仓库顶层架构规范（严格限定 6 个顶层目录）

为保持仓库长期整洁、职责单一，全仓库顶层目录严格固定为以下 **6 个目录**（由 `tests/core/test_repository_conventions.py` 自动守卫）：

```text
BrainDecoding/
├── braindecoding/              # 唯一核心 Python 源码包
│   ├── data/                   # 数据集加载、事件对齐、通道坐标与派生缓存校验
│   ├── training/               # 神经骨干网络、对比损失、滚动编码器与分布式训练运行时
│   ├── decoding/               # 词级/句级解码器、延迟承诺搜索、修正增益与语言模型重打分
│   ├── evaluation/             # 检索评测、候选召回、配对显著性检验與跨数据集留出集评估
│   └── tasks/                  # 任务级训练入口与配置适配
├── configs/                    # 实验超参数与协议配置（仅限 .yaml，零缩写见名知意）
│   ├── training/               # 训练工作流配置（如 pallier2025_word_encoder.yaml）与训练协议
│   ├── decoding/               # 在线滚动句级解码与修正增益预览配置
│   ├── evaluation/             # 留出测试集主评测与各项对照消融配置
│   └── word_decoding/          # 数据集底层契约 (base.yaml) 与词级阶段配置
├── data/                       # 唯一允许存放 JSON 清单、实验结果与派生缓存的数据中心
│   ├── manifests/              # 实验协议注册表 (protocol_index.yaml)、数据切分与产物哈希清单
│   ├── results/                # 正式实验结果（studies/<study>/results.json 与 tables/exports/*.csv）
│   └── cache/                  # 本地派生数据清单 (derived/) 与对齐产物缓存 (artifacts/)
├── paper/                      # 论文 LaTeX 源码 (manuscript/) 与最终定稿图片 (figures/)
├── scripts/                    # 命令行工具与论文出图脚本 (scripts/paper/)
└── tests/                      # 镜像源码结构的自动化单元测试与仓库规范守卫
    ├── core/                   # CLI、配置、协议清单与仓库规范守卫测试
    ├── data/                   # 数据层单元测试
    ├── training/               # 训练层单元测试
    ├── decoding/               # 解码层单元测试
    └── evaluation/             # 评测与统计检验单元测试
```

---

## 2. JSON 与 Markdown 文件治理准则

针对科研代码库常见的“JSON 和 Markdown 随处生成导致目录失控”问题，本仓库执行以下硬性限制：

1. **Markdown 零散落原则**：
   - 全仓库仅保留 **2 个 `.md` 文件**：根目录 `README.md` 与主实验协议契约 `data/manifests/rolling_sentence/unified_protocol_v2.md`。
   - 严禁在子目录中创建临时说明文档、审计笔记或实验过程 `.md` 报告；所有模块说明统一写入 Python docstring，使用规范统一写入本 `README.md`。
2. **JSON 严格归口 `data/` 原则**：
   - `braindecoding/`、`configs/`、`scripts/`、`paper/`、`tests/` 及仓库根目录下**严禁存放任何 `.json` 文件**。
   - 配置文件统一使用 `.yaml` 并放入 `configs/`。
   - 每个实验在 `data/results/studies/<study_name>/` 下**仅允许保留唯一的 `results.json`**，严禁混放图片、日志、进程号（`.pid`）或冗余溯源快照。

---

## 3. 新增代码与文件命名规范（Self-Explanatory Naming Standard）

新增或重构任何 Python 模块、脚本、测试文件与 `.yaml` 配置文件时，必须遵守以下命名规范（由自动化测试实时拦截违规命名）：

1. **见名知意，禁用内部缩写**：
   - 文件名必须采用完整的英文单词（小写 `snake_case`），让人仅凭文件名即可明确模块职责。
   - **严禁使用内部代号或缩写词**：如 `c1`（应写为 `single_step_revision`）、`f_local`（应写为 `neural_only_local`）、`k0`（应写为 `candidate_pool_search`）、`k2_d2`（应写为 `local_window_revision`）、`bge`（应写为 `multilingual_embedding`）、`eval`（应写为 `evaluation`）。
2. **严禁版本号与临时补丁后缀**：
   - **禁止**在文件名末尾追加 `_v1`、`_v2`、`_test_v2`、`_stable`、`_patch`、`_new`、`_old`、`_tmp`、`_retry1` 等后缀。
   - 演进算法应直接演进对应模块（版本历史由 Git 追踪），补充函数应合并入所属职责模块，不得为单个辅助函数新建碎片文件。

### 核心模块速查表

| 子包 | 模块文件名 | 核心职责说明 |
| :--- | :--- | :--- |
| `braindecoding.data` | `dataset_builder.py` | 派生数据集构建与完整性校验入口 |
| `braindecoding.data` | `derived_cache.py` | 派生数据缓存路径与清单解析 |
| `braindecoding.data` | `event_table.py` | 刺激事件时间表读取与对齐验证 |
| `braindecoding.data` | `sensor_coordinates.py` | EEG/MEG 传感器空间坐标归一化与通道掩码 |
| `braindecoding.data` | `text_embeddings.py` | 文本特征与词向量提取 |
| `braindecoding.data` | `pallier2025_dataset.py` | Pallier 2025 法语 MEG 数据集加载器 |
| `braindecoding.data` | `chineseeeg2_dataset.py` | ChineseEEG2 中文脑电数据集加载器 |
| `braindecoding.data` | `synthetic_timing_dataset.py` | 合成时间偏移诊断基准数据集 |
| `braindecoding.data` | `long_window_dataset.py` | 单字长窗口边界效应诊断数据集 |
| `braindecoding.training` | `neural_backbone.py` | BrainModule 多被试空间注意力与卷积骨干网络 |
| `braindecoding.training` | `contrastive_losses.py` | CLIP / SigLIP / DINO 对比学习损失函数 |
| `braindecoding.training` | `rolling_encoder_training.py` | 统一滚动窗口神经编码器训练与缓存导出 |
| `braindecoding.training` | `gwilliams_forgetting_training.py` | Gwilliams 2023 遗忘门控训练流程 |
| `braindecoding.training` | `distributed_runtime.py` | 训练运行时环境、随机种子与断点恢复 |
| `braindecoding.training` | `environment_preflight.py` | 训练前数据、模型与硬件环境预检 |
| `braindecoding.decoding` | `delayed_commitment_decoder.py` | 延迟承诺（Delayed Commitment）句级波束解码器 |
| `braindecoding.decoding` | `revision_gain_sentence_decoder.py` | 修正增益（Revision Gain）句级解码与评测生成 |
| `braindecoding.decoding` | `unified_sentence_decoder.py` | 统一句级批处理解码核心实现 |
| `braindecoding.decoding` | `fixed_grid_sequence_decoder.py` | 固定网格 CTC 序列解码与快速推理 |
| `braindecoding.decoding` | `windowed_sequence_baseline.py` | 滑动窗口序列解码基线（Jayalath 复现） |
| `braindecoding.decoding` | `word_language_model_rescorer.py` | 自然词级语言模型与提示词条件重打分 |
| `braindecoding.evaluation` | `retrieval_metrics.py` | Top-K 检索准确率与被试内置换检验指标 |
| `braindecoding.evaluation` | `statistical_significance.py` | 配对分块置换检验与 Holm 多重比较校正 |
| `braindecoding.evaluation` | `fixed_horizon_rolling_search.py` | 固定视野滚动窗口超参数搜索与评测 |
| `braindecoding.evaluation` | `chinese_candidate_heldout_evaluation.py` | ChineseEEG2 候选集留出评测 |
| `braindecoding.evaluation` | `chinese_forgetting_heldout_evaluation.py` | ChineseEEG2 遗忘门控留出评测 |
| `braindecoding.evaluation` | `gwilliams_candidate_evaluation.py` | Gwilliams 2023 候选集留出评测 |
| `braindecoding.evaluation` | `gwilliams_forgetting_evaluation.py` | Gwilliams 2023 遗忘门控留出评测 |
| `braindecoding.evaluation` | `bilingual_embedding_reranker.py` | 双语文本嵌入重排序器评测 |
| `braindecoding.evaluation` | `heldout_selection_evaluation.py` | 留出集超参数选择一致性评测 |
| `braindecoding.training` | `workflow_runner.py` | 本地小样本极速联调与云端 CPU 预热 / GPU 训练统一工作流执行器 |
| `scripts` | `cache_archive.py` | 预处理特征缓存单文件 `.tar` 打包 (`pack`) 与秒级解包 (`unpack`) 工具 |
| `scripts` | `colab_setup.py` | 防崩版 Google Colab 虚拟环境初始化、PyTorch 约束锁定与本地软链接桥工具 |

---

## 4. 本地小样本联调 + Google Colab 云端双模工作流

### 4.1 环境变量路径解耦

所有路径统一由环境变量驱动（未设置时自动回退到本地合理默认值）：

| 环境变量 | 说明 | 本地默认值 | Colab 推荐值 |
| :--- | :--- | :--- | :--- |
| `BRAINDECODING_DATA_DIR`（兼容 `BRAINDATA_ROOT`） | 原始数据目录 | `D:/dataset` | `/content/studies_links`（指向云盘 `data/` 的本地软链接桥） |
| `BRAINDECODING_CACHE_DIR` | 预处理特征缓存目录 | `data/cache/derived` | `/content/local_cache`（本地高速盘） |
| `BRAINDECODING_RESULTS_DIR` | 训练 Checkpoint 与日志目录 | `data/cache/runs` | `/content/drive/MyDrive/BrainDecoding/runs` |
| `BRAINDECODING_MODEL_DIR`（兼容 `BRAINDECODING_MODEL_ROOT`） | 预训练文本/语言模型目录 | `D:/code/dascoli-word-decoding/models` | `/content/drive/MyDrive/BrainDecoding/models` |

### 4.2 统一 CLI 入口（`smoke` / `cache` / `train` / `eval-run`）与三阶段训练 YAML 配置

项目训练划分为三个递进阶段，均支持通过 `--config` / `-c` 直接指定 `configs/training/*.yaml` 配置文件（或通过 `--dataset` + `--stage` 指定）：
- **阶段 1 (`word` / `1`)**：词级脑编码器（纯 CNN，`use_transformer: false`），从头训练。
- **阶段 2.1 (`context` / `2.1`)**：词级脑编码器 + 16 层全局上下文 Transformer（`use_transformer: true`，使用完整窗口，不启用局部窗口滚动修订）；加载阶段 1 最佳检查点初始化脑编码器，第 1 轮冻结 CNN，随后联合训练。
- **阶段 2.2 (`revision` / `2.2`)**：词级脑编码器 + 16 层局部窗口滚动修订（`use_transformer: true`，`K0/K1/K2` 局部窗口滚动回修）；加载阶段 1 最佳检查点初始化脑编码器，第 1 轮冻结 CNN，随后联合训练。

```bash
# 1. 本地单受试者 2 Epoch 极速冒烟测试（验证前反向传播、bf16-mixed、指标计算与 Checkpoint 保存）
python -m braindecoding smoke --config configs/training/chineseeeg2_word_encoder.yaml --num-subjects 1 --epochs 2 --require-cache

# 2. 仅预热特征缓存（支持全量或指定受试者子集，--include-test 同时预热测试集）
python -m braindecoding cache --dataset chineseeeg2_littleprince --subject all --include-test
python -m braindecoding cache --dataset pallier2025 --subject all --include-test

# 3. 正式三阶段训练（直接传入 configs/training/*.yaml，内置 L4 GPU 调优超参，支持 --resume auto 自动从 last.ckpt 断点续训）
# Pallier2025 法语 MEG（阶段 1 -> 阶段 2.1 / 阶段 2.2）
python -m braindecoding train --config configs/training/pallier2025_word_encoder.yaml --seed 0 --subject all --require-cache --resume auto
python -m braindecoding train --config configs/training/pallier2025_context_transformer.yaml --seed 0 --subject all --require-cache --resume auto
python -m braindecoding train --config configs/training/pallier2025_rolling_revision.yaml --seed 0 --subject all --require-cache --resume auto

# ChineseEEG-2 中文《小王子》EEG（阶段 1 -> 阶段 2.1 / 阶段 2.2）
python -m braindecoding train --config configs/training/chineseeeg2_word_encoder.yaml --seed 0 --subject all --require-cache --resume auto
python -m braindecoding train --config configs/training/chineseeeg2_context_transformer.yaml --seed 0 --subject all --require-cache --resume auto
python -m braindecoding train --config configs/training/chineseeeg2_rolling_revision.yaml --seed 0 --subject all --require-cache --resume auto

# 4. 验证集 (val) 与测试集 (test) 评估及可视化绘图（生成 2x2 综合看板图与跨阶段对比图）
python -m braindecoding eval-run --config configs/training/pallier2025_context_transformer.yaml --seed 0 --split val --require-cache
python -m braindecoding eval-run --dataset pallier2025 --seed 0 --split test --all-stages --require-cache

# 5. 缓存单文件打包与解包（Colab 本地高速盘 <-> Google Drive）
python scripts/cache_archive.py pack --cache-dir /content/local_cache --archive-dir /content/drive/MyDrive/BrainDecoding/archives --modality meg
python scripts/cache_archive.py unpack --archive-dir /content/drive/MyDrive/BrainDecoding/archives --cache-dir /content/local_cache --modality meg --skip-existing

# 6. Colab 云端 Notebook
# - scripts/colab/colab_drive_init.ipynb     : 一键创建 Google Drive 标准目录并核验上传的 .tar 缓存包
# - scripts/colab/colab_cpu_cache.ipynb      : CPU 实例预热特征缓存（含 train/val/test）并打包为单文件 .tar
# - scripts/colab/colab_gpu_train.ipynb      : GPU 实例秒级解包缓存、按 --config 启动三阶段训练并可视化验证集评估结果
# - scripts/colab/colab_gpu_evaluation.ipynb : GPU 实例专门用于评估留出测试集（支持单阶段或三阶段横向对比及可视化图表）

# 7. 查看当前注册的标准实验协议与运行单元测试
python -m braindecoding list
pytest tests -q
```


