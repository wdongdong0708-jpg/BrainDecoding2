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
├── configs/                    # 实验超参数配置（仅限 .yaml）
│   ├── datasets/               # 数据集与词级别基线实验配置
│   └── unified_rolling/        # 统一滚动句级解码实验配置
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

新增或重构任何 Python 模块、脚本与测试文件时，必须遵守以下命名规范（由自动化测试实时拦截违规命名）：

1. **见名知意，禁用内部缩写**：
   - 文件名必须采用完整的英文单词（小写 `snake_case`），让人仅凭文件名即可明确模块职责。
   - **严禁使用内部代号或缩写词**：如 `c1`（应写为 `candidate`）、`f`（应写为 `forgetting`）、`k0`（应写为 `fixed_horizon`）、`lm`（应写为 `language_model`）、`bge`（应写为 `bilingual_embedding`）、`ctc`（应写为 `connectionist_temporal_classification` 或 `sequence_decoder`）、`eval`（应写为 `evaluation`）。
2. **严禁版本号与临时补丁后缀**：
   - **禁止**在文件名末尾追加 `_v1`、`_v2`、`_test_v2`、`_stable`、`_patch`、`_new`、`_old`、`_tmp`、`_io_guard` 等后缀。
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

---

## 4. 常用命令

```bash
# 查看当前注册的标准实验协议
python -m braindecoding list

# 运行全量自动化单元测试与仓库规范守卫
pytest tests -q
```
