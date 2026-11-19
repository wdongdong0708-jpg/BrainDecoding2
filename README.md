# BrainDecoding

自然连续语流脑磁（MEG）词对齐神经解码与局部跨词上下文编码研究代码库。

核心科学主线：
- **固定词窗 1.0 s 词对齐解码**：第一阶段独立训练并全程冻结单窗卷积基座（`Word-Only`），第二阶段引入同词间间隔错位脑信号窗口抑制相邻重叠窗口的隐式时长捷径，测试与推理统一采用无错位支路单次前向计算（`η_eval = 0.0`）。
- **局部跨词联合编码（`Rolling-K2`）**：仅利用已到达前序词与最多 `r = 2` 个后续词窗直接定稿每个词的最终神经表征，并接入冻结 `Qwen3-8B-Base` 从左到右单遍束搜索解码句子。
- **统一 MEG 数据底座（PNPL + 本地 BIDS）**：全仓统一复用唯一词窗数据集 `braindecoding/data/word_window_dataset.py`（`WordWindowDataset`），原生支持 `pnpl` 驱动的 `libribrain100`、`pallier2025_full`、`armeni2022` 以及兼容既有 `.pt` 权重的 10 被试 `pallier2025`。

---

## 1. 仓库顶层架构（5 个核心业务顶层目录）

```text
BrainDecoding/
├── braindecoding/              # 唯一核心 Python 源码包
│   ├── data/                   # 统一 WordWindowDataset、PNPL 适配器、Pallier2025 BIDS 物化、事件表与词向量缓存
│   ├── models/                 # 传感器空间投影、膨胀卷积脑编码器、分组上下文 Transformer 与主词嵌入模型
│   ├── training/               # 通用 MEG 词任务、孪生去偏训练（损失/预提取/定长前向/评估解耦）与工作流编排
│   ├── decoding/               # 神经词表得分导出、增量 KV-Cache 语言模型打分、单遍静态束搜索、多尺度扫描与 S3 流水线
│   └── evaluation/             # 检索指标、统计显著性（scipy.stats）、词级检索评测与序列解码多维评测
├── configs/                    # 实验超参数与协议配置（.yaml）
│   ├── controls/               # 负对照配置（temporal_shift / zero_signal）
│   ├── training/               # Stage 1 / 2.1 / 2.2 / 2.3 与孪生去偏训练配置
│   ├── decoding/               # S3 序列解码协议配置（sequence_decoding_protocol.yaml）
│   ├── evaluation/             # 留出集评测与长窗对照配置
│   └── word_decoding/          # 数据集底层契约（base.yaml）与词级阶段配置
├── data/                       # 冻结协议清单（manifests/）与本地缓存目录（cache/、results/ 均由 .gitignore 忽略）
├── scripts/                    # 薄层 CLI 入口（build_derived_data.py、decode_sentences.py）与云端缓存/环境工具
└── tests/                      # 自动化单元测试（core / data / training / decoding / evaluation）
```

---

## 2. 支持的 MEG 数据集与核心模块速查

| 数据集标识 | 来源与说明 | 默认预处理契约 |
| :--- | :--- | :--- |
| `pallier2025` | Pallier 2025 法语《小王子》10 被试本地 BIDS 派生缓存（严格保持既有 `.pt` 权重零重训加载） | 50 Hz + 单记录 `RobustScaler` + 0.5 s 基线扣除 + `[-5, 5]` 截断 |
| `pallier2025_full` | 经 `pnpl` 拉取的完整 Pallier 2025 法语 MEG 数据集 | 强制 `standardize=False`，单记录 `RobustScaler`，运行时切窗 |
| `libribrain100` | 经 `pnpl` 拉取的 LibriBrain 英语 MEG 数据集（单/多被试动态提取） | 同上，按会话/章节隔离划分单元防泄漏 |
| `armeni2022` | 经 `pnpl` 拉取的 Armeni 2022 英语 MEG 数据集（3 被试动态提取） | 同上，支持 `sensor_xyz` 坐标投影 |

| 子包 | 代表模块 | 核心职责 |
| :--- | :--- | :--- |
| `braindecoding.data` | `word_window_dataset.py` | 全仓唯一 MEG 词窗 Dataset（`WordWindowDataset` / `DerivedMegWordDataset`），支持 `baseline_seconds` 与严格过去 `preroll` |
| `braindecoding.data` | `pnpl_dataset_adapter.py` | 调用 `pnpl` 物化 `libribrain100` / `pallier2025_full` / `armeni2022` 派生缓存与规范中文事件表 |
| `braindecoding.data` | `pallier2025_dataset.py` | 本地 BIDS 原始文件解析与事件表/信号物化，Dataset 轻量继承 `WordWindowDataset` |
| `braindecoding.models` | `embedding.py`, `brain_encoder.py`, `spatial.py`, `transformer.py` | 冻结权重契约的核心神经架构（由 `tests/core/test_checkpoint_state_dict_compat.py` 锁定参数键哈希） |
| `braindecoding.training` | `meg_word_task.py`, `workflow_runner.py` | 通用 MEG 词任务构建、事件表动态受试者提取与四阶段工作流执行器 |
| `braindecoding.training` | `siamese_debiased_training.py`, `debiased_{losses,preextraction,padded_forward,preextracted_forward,preextracted_eval}.py` | 冻结 CNN 特征预提取与三档去偏训练（`null_space` / `amortized_timing` / `legacy_siamese`） |
| `braindecoding.decoding` | `neural_score_exporter.py`, `language_model_scoring.py`, `rolling_beam_search.py`, `beam_search_sweeps.py`, `sentence_decoding_pipeline.py` | 单遍左到右束搜索（`Brain-Only` / `LM-Only` / `Brain + LM`）、验证集校准、跨被试 $k$ 聚合与多尺度词表扫描 |
| `braindecoding.evaluation` | `word_retrieval_{evaluation,statistics,metrics,controls,common_horizon,single_stage,control_selection,multiseed_controls,multiseed,cross_model,runner}.py`, `sequence_decoding_evaluation.py` | 词级检索评估（基于 `scipy.stats`）、负对照与跨模型配对检验、序列解码 6 维指标与材料块置换检验 |

---

## 3. 环境变量与常用命令

### 环境变量路径解耦

| 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `BRAINDECODING_DATA_DIR`（兼容 `BRAINDATA_ROOT`） | 原始数据目录 | `D:/dataset` |
| `BRAINDECODING_CACHE_DIR` | 派生特征缓存目录 | `data/cache/derived` |
| `BRAINDECODING_RESULTS_DIR` | 训练检查点与评估输出目录 | `data/cache/runs` |
| `BRAINDECODING_MODEL_DIR`（兼容 `BRAINDECODING_MODEL_ROOT`） | 预训练文本/语言模型目录 | `D:/code/dascoli-word-decoding/models` |

### 常用 CLI 命令

```bash
# 1. 本地冒烟测试（支持 pallier2025 / pallier2025_full / libribrain100 / armeni2022）
python -m braindecoding smoke --dataset pallier2025 --stage 1 --num-subjects 1 --epochs 2 --require-cache

# 2. 预热派生缓存
python -m braindecoding cache --dataset pallier2025 --subject all --include-test

# 3. 正式阶段训练（Stage 1 Word-Only -> Stage 2.1 Full-Context / Stage 2.2 Rolling-K2 / Stage 2.3 Prefix-Revision）
python -m braindecoding train --config configs/training/pallier2025_word_encoder.yaml --seed 0 --subject all --require-cache --resume auto
python -m braindecoding train --config configs/training/pallier2025_rolling_revision.yaml --seed 0 --subject all --require-cache --resume auto

# 4. 留出集词级评估与时间错位负对照
python -m braindecoding eval-run --dataset pallier2025 --seed 0 --split test --all-stages --control-config configs/controls/temporal_shift/temporal_shift_control.yaml --require-cache

# 5. Stage 3 序列解码流水线（export -> calibrate -> decode -> evaluate）
python -m braindecoding decode-sentences --dataset pallier2025 --split test --seed 0,1,2

# 6. 缓存单文件打包与解包（Colab / 云盘迁移）
python scripts/cache_archive.py pack --cache-dir /content/local_cache --archive-dir /content/drive/MyDrive/BrainDecoding/archives --modality meg
python scripts/cache_archive.py unpack --archive-dir /content/drive/MyDrive/BrainDecoding/archives --cache-dir /content/local_cache --modality meg --skip-existing

# 7. 运行测试与权重键名兼容门禁
pytest tests -q
pytest tests/core/test_checkpoint_state_dict_compat.py -q
```
