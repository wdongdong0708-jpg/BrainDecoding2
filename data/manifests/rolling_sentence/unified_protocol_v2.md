# 自然连续语流局部跨词去偏编码与句子解码：当前生效实验协议快照

> **维护原则（State Snapshot）**：本文件是 `BrainDecoding` 代码仓唯一生效的实验协议与基准结果状态快照，随主线演进**原地覆写更新**，不追加历史流水账（历史版本留存 `git log`）。此前 `2026-09-27` 的 `BGE-M3 + Qwen2.5-1.5B + A/B/C` 在线句子改写草案及 `C-Gated` 屏幕闪烁门控支线已彻底废弃。

- **当前生效主数据集**：`Pallier2025`（法语连续语音听觉 MEG《小王子》，10 名受试者 `sub-01..10`，随机种子 `Seeds [0, 1, 2]`）
- **科学主线一句话**：第一阶段独立训练并冻结单词卷积基座（`Word-Only`）；第二阶段在跨词 Transformer 训练期引入同词间间隔错位脑窗遏制隐式词长捷径，测试期统一采用无错位支路单次前向（$\eta_{\text{eval}} = 0.0$）；将跨词视野收敛为已到达前序词与最多 $r = 2$ 个后续词窗的 `Local (r = 2)` 直接定稿输出，在将平均输入等待时间缩短一半以上的同时，获得与整句联合编码相当的词表征质量与更优的句子解码性能。

---

## 1. `[ACTIVE]` vs `[DEPRECATED]` 组件与符号映射表

| 代码工程阶段 / 符号 | 协议状态 | 论文标准术语 | 对应配置文件 / 说明 |
| :--- | :--- | :--- | :--- |
| `stage: word` (`S0`) | `[ACTIVE]` | **单词独立解码（`Word-Only`）** | `configs/training/pallier2025_word_encoder.yaml`；第二阶段全程冻结作底 |
| `stage: context`（端到端微调） | `[ACTIVE Baseline]` | **`Full-Context (Joint)`** | 未冻结 CNN 且未去偏的整句双向基线（暴露隐式时长泄露） |
| `stage: context`（冻结 CNN，无去偏） | `[ACTIVE Baseline]` | **`+ Freeze CNN`** | 仅冻结 Stage 1 CNN、未施加错位约束的整句基线 |
| `stage: context`（孪生错位去偏） | `[ACTIVE Reference]` | **`+ Debias, Full-Context`** | `configs/training/siamese_debiased/pallier2025_siamese_debiased_context.yaml` |
| `stage: prefix_revision`（$r \in \{0, 1, 2\}$） | **`[ACTIVE Primary]`** | **局部跨词联合编码（`Local (r = 0/1/2)` / `+ Debias, Local (r = 2)`）** | `configs/training/siamese_debiased/pallier2025_siamese_debiased_prefix_revision.yaml`；可见范围 $z_{1:\min(n, i+r)}$ |
| `stage: revision`（$R = 2$） | `[ACTIVE Appendix A]` | **对称短窗 `Local-Window (R = 2)`** | 局部窗 $[\max(1, i-2), \min(n, i+r)]$；附录对比消融 |
| `raw_simple` / `raw_full` 序列解码 | **`[ACTIVE Primary]`** | **第三阶段束搜索句子解码（$\eta_{\text{eval}} = 0.0$）** | `configs/decoding/raw_simple_sequence_decoding_protocol.yaml` 与 `raw_full_sequence_decoding_protocol.yaml` |
| `*_brain_only`（$\lambda = 0$ 神经贪心解码） | **`[ACTIVE Diagnostic]`** | **`Brain-Only`** | 不依赖 `Qwen3-8B-Base`（允许 `scorer=None`），直接逐词取神经得分矩阵 $\arg\max$ |
| `shifted`（Pre-Roll $-75\text{ s}$ 错位序列） | **`[ACTIVE Baseline]`** | **`LM-Only`** | 同配置与语言模型束搜索下的无真实脑信号对齐负对照基线 |
| `C-Gated` / `tau_gate` / `sweep_confidence_gate_pareto` | **`[DEPRECATED]`** | 严禁出现在论文中 | 已砍除“0 秒先出词 + 屏幕改写 + 防闪烁门控”支线，代码函数仅作兼容保留 |
| `F_local` / 修订率 / 有害覆写 | **`[DEPRECATED]`** | 严禁出现在论文中 | 同上；导出脚本自动过滤（`decoded_bundles.pop("F_local", None)`） |
| `BGE-M3` + `Qwen2.5-1.5B` + `A/B/C` 协议 | **`[DEPRECATED]`** | 已由下节 `T5-Large` + `Qwen3-8B-Base` + 固定 $r=2$ 定稿取代 | 旧 `2026-09-27` 草案仅留存于 Git 历史 |

---

## 2. 当前生效三阶段数据、模型与训练协议

### 2.1 数据预处理与静态文本目标
- **脑磁信号预处理**：原始 $1{,}000\text{ Hz}$ 滤波（$0.1\text{--}40\text{ Hz}$）后降采样至 $50\text{ Hz}$；按采集段以 `RobustScaler` 标准化；从词起始时刻 $\tau_i$ 截取固定时长 $T_{\text{win}} = 1.0\text{ s}$（$S = 50$ 采样点），扣除窗口前 $0.5\text{ s}$ 均值完成基线校正并截断至 $[-5, 5]$。测试划分共 $1{,}940$ 个连续片段（每受试者 $194$ 个）、$18{,}450$ 个测试词位；完整训练词表 $\lvert V\rvert = 2{,}020$（覆盖测试集 $82.49\%$ 词位）。
- **目标词向量 $e_w$**：冻结 `t5-large` 第 12 层（`layer_fraction: 0.5`）子词均值池化并 $\ell_2$ 归一化（$d = 1{,}024$，`data/cache/derived/pallier2025/text/t5_large_layer_0_5/embeddings.npz`）。
- **事件时间戳与四分类词性元数据**：通过 `neural_score_exporter.py`（`build_segment_inventory` / `enrich_segments_from_event_table`）为每个测试片段持久化词级 `onset_seconds`、`offset_seconds`、`pos_tags` 与 `is_content_word_mask`；`SyntheticTimingDataset` 支持可选 `include_onset_seconds=True` 输出浮点型词起始时间戳以计算 $\Delta_{\text{JS}}$ 与物理等待时长。

### 2.2 Stage 1 与 Stage 2 训练超参
- **Stage 1（`Word-Only`）**：`configs/training/pallier2025_word_encoder.yaml`。$306 \to 270$ 空间注意力 + 5 个残差膨胀卷积块（隐藏维度 $160$，核大小 $3$）+ 时间注意力池化；`D-SigLIP` 去重对比损失（初始温度 $10.0$、偏置 $-10.0$、重复屏蔽阈值 $0.999$）；`AdamW`（`lr = 4e-4`, `weight_decay = 0.0`, `batch_size = 256`, `bf16-mixed`）。
- **Stage 2（去偏跨词联合编码）**：加载同种子 Stage 1 `best.pt` 并全程冻结 CNN（`freeze_brain_encoder_epochs = 30`）；上层为 16 层 16 头带 RoPE 的非预训练 Transformer（$d = 1{,}024$，注意力丢弃率 $0.1$）；`AdamW`（`lr = 1e-4`, `weight_decay = 0.01`, `batch_size = 128`）。
  - 同段错位负锚点：$M = 6$ 组循环平移 $\delta_m \in \{-15, -30, -45, -60, -90, -120\}\text{ s}$，训练期扣除系数 $\eta = 0.50$，零化惩罚权重 $\lambda_{\text{null}} = 10.0$；
  - 局部前序前缀加后续词模型 `Local (r = 0/1/2)`（`configs/training/siamese_debiased/pallier2025_siamese_debiased_prefix_revision.yaml`）额外施加相对 $r = 0$ 的跨步一致性惩罚：$\lambda_\Delta = 8.0$、$\lambda_{\text{dir}} = 0.50$（配置中对应字段 `lambda_delta_null: 8.0`、`beta_delta_cm: 0.50`），等权优化 $r \in \{0, 1, 2\}$ 三步均值损失。

### 2.3 Stage 3 序列解码与后处理评估协议
- **语言模型**：冻结 `Qwen/Qwen3-8B-Base`。
- **束搜索与 `Brain-Only` 对照**：每步保留脑信号余弦 Top $K_{\text{cand}} = 20$ 候选，局部 Softmax 温度 $T_r \in \{0.06, 0.07\}$（验证集网格选择），束宽 $B = 5$；语言权重满足 $\lambda_{K_0} \ge \lambda_{K_1} \ge \lambda_{K_2}$ 且测试推理固定 $\eta_{\text{eval}} = 0.0$。同时开启 `include_brain_only: true`（`S0_brain_only`, `K0_brain_only`, `K1_brain_only`, `K2_brain_only`, `F_brain_only`），在 $\lambda = 0$、`scorer=None` 下执行纯神经 $\arg\max$ 解码；`shifted`（Pre-Roll $-75\text{ s}$）对应 `LM-Only` 基线。
- **跨被试多观测扩展（$k \in \{1, 2, 5, 10\}$）**：在 `rolling_beam_search.py` 与 `sequence_decoding_protocol.yaml` 中配置默认 `multi_subject_aggregation_mode: agreement_weighted`（`multi_subject_agreement_alpha: 2.0`，对标 SimpleB2T Eq. 7–8：按各受试者句段/词位平均余弦相似度与群体均值矢量的一致性加权聚合），同时兼容保留 `rms_preserved` 与 `F` 状态回填（`K2` / `F` 齐头并进）。
- **两套正式评估协议与细粒度指标**：
  1. **主基准（无 OOV 长句闭集 `L >= 5, 0% OOV`）**：`configs/decoding/raw_simple_sequence_decoding_protocol.yaml`，覆盖基准全集 $\lvert V\rvert = 131\ (n = 300,\ 2{,}300\text{ 词位})$ 及贪心覆盖子集 $\lvert V\rvert = 50\ (n = 100)$、$\lvert V\rvert = 100\ (n = 220)$。
  2. **完整测试集协议**：`configs/decoding/raw_full_sequence_decoding_protocol.yaml`，覆盖未筛选全部 $n = 1{,}940$ 个测试片段、完整训练词表 $\lvert V\rvert = 2{,}020$。
  3. **真实物理等待与四分类词性指标**：基于事件时间戳由 `compute_mean_physical_wait_seconds` 精确计算每个词从起始时刻到定稿窗口结束的物理等待时长
     $$\text{Wait}_i(r) = \tau_{\min(n, i+r)} - \tau_i + T_{\text{win}}$$
     并由 `classify_part_of_speech_four_way` 统计 **名词 (`Noun`)、动词 (`Verb`)、形容词/副词 (`Adj/Adv`) 与功能词 (`Function`)** 四大词类词准确率（适配 French TreeBank 与 Universal Dependencies 标签）。

---

## 3. 锁定测试评估口径与权威锚点结果（Seeds `0, 1, 2`, 均值 ± 标准差）

### 3.1 词级检索主表（主文词级检索口径）
- **锁定口径**：共同测试词位 $N = 14{,}590$，50 词查询位 $N_{50} = 7{,}440$；错位对照采用严格不接触未来的前序 $45.0\text{ s}$ 缓冲（`shift_mode=preroll`, $\delta_{\text{eval}} \in \{-10, -20, -30, -40\}\text{ s}$），单次前向推理 $\eta_{\text{eval}} = 0.0$；词类等权平衡准确率（随机机会 Top-10 $= 20.00\%$，Top-1 $= 2.00\%$）。

| 模型设置 | Top-10 Clean (%) ↑ | Top-10 Shifted (%) ↓ | Top-10 净增益 (pp) ↑ | Top-1 Clean (%) ↑ | Top-1 Shifted (%) ↓ | Top-1 净增益 (pp) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Word-Only` | `40.64 ± 0.15` | `20.07 ± 0.16` | `+20.57 ± 0.04` | `6.64 ± 0.13` | `1.97 ± 0.11` | `+4.67 ± 0.23` |
| `Full-Context (Joint)` | `68.42 ± 0.42` | `61.14 ± 0.78` | `+7.28 ± 1.17` | `20.74 ± 0.42` | `14.88 ± 1.01` | `+5.86 ± 1.34` |
| `+ Freeze CNN` | `64.33 ± 0.88` | `50.49 ± 3.45` | `+13.84 ± 2.61` | `17.24 ± 0.77` | `9.60 ± 1.29` | `+7.64 ± 0.54` |
| `+ Debias, Full-Context` | `57.95 ± 2.51` | `22.61 ± 0.63` | `+35.33 ± 3.14` | `13.58 ± 1.57` | `2.62 ± 0.13` | `+10.96 ± 1.69` |
| **`+ Debias, Local (r = 2)`** | **`58.70 ± 1.02`** | `23.65 ± 0.17` | `+35.05 ± 1.17` | **`14.42 ± 0.35`** | `3.11 ± 0.11` | **`+11.31 ± 0.25`** |

- **核心消融锚点（同 $N_{50} = 7{,}440$ 口径）**：
  - `Local-Window (R = 2)` 对称短窗：Top-10 Clean `61.46 ± 0.88%` / Shifted `29.96 ± 1.03%` / Net `+31.50 ± 0.23 pp`；
  - `Local (r = 2) w/o L_delta-null`：Top-10 Clean `57.89 ± 1.15%` / Shifted `24.62 ± 0.44%` / Net `+33.27 ± 1.44 pp`；
  - `Local (r = 2) + 测试在线均值扣除 (ema_decay=0.99, eta_eval=0.5)`：Top-10 Clean `58.42 ± 0.97%` / Shifted `23.74 ± 0.15%` / Net `+34.69 ± 1.10 pp`。

### 3.2 句子解码主表与全测试集锚点（$k = 1, \eta_{\text{eval}} = 0.0$）

| 解码策略 | 验证集 $\lambda_r$ | $\lvert V\rvert=100$ BLEU-1 (%) | $\lvert V\rvert=131$ BLEU-1 (%) | $\lvert V\rvert=131$ Top-1 (%) | $\lvert V\rvert=131$ 平均等待 (s) | 全集 $\lvert V\rvert=2{,}020$ BLEU-1 (%) | 全集平均等待 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Word-Only` | `0.13` | `15.90 ± 0.83` | `13.26 ± 0.47` | `6.43 ± 0.16` | `1.00` | `9.15 ± 0.56` | `1.00` |
| `Local (r = 0)` | `0.12` | `17.09 ± 0.83` | `14.78 ± 0.77` | `7.49 ± 0.58` | `1.00` | `12.20 ± 0.64` | `1.00` |
| `Local (r = 1)` | `0.08` | `20.15 ± 1.23` | `17.10 ± 0.59` | `10.38 ± 0.32` | `1.87` | `13.63 ± 0.32` | `1.89` |
| **`Local (r = 2)`** | `0.08` | **`21.74 ± 0.99`** | **`18.70 ± 0.94`** | **`12.10 ± 0.29`** | `2.61` | **`14.09 ± 0.42`** | `2.69` |
| `Full-Context` | `0.37` | `20.90 ± 2.01` | `17.55 ± 1.75` | `10.51 ± 1.94` | `5.33` | `12.38 ± 0.97` | `8.31` |

*(注：本文全部核心与对比实验统一基于 `T5-Large` 文本表征与统一测试口径执行，严禁混入已废弃的历史 `BAAI/bge-m3` 存档。5 模型合成脉冲与先验对齐诊断由当前训练好的 `T5-Large` 检查点直接通过 `SyntheticTimingDataset` 纯前向评测生成；孤立单字拉宽窗对照基于同一 `T5-Large` Stage-1 `Word-Only` 架构分别训练 `1.00 s` / `1.28 s` / `1.50 s` / `2.50 s` 并于 `common_horizon_delay = 2` 口径评测；完整论文数据导出由 `scripts/export_manuscript_studies.py` 输出至 Google Drive 与 `data/results/exports/manuscript_studies/`。)*

---

## 4. 已落地的论文重构诊断与定性导出管线（`scripts/export_manuscript_studies.py`）

对应代码实现模块为 [`braindecoding/evaluation/manuscript_study_exporter.py`](../../../braindecoding/evaluation/manuscript_study_exporter.py)，一键命令行入口为 [`scripts/export_manuscript_studies.py`](../../../scripts/export_manuscript_studies.py)，并在 Colab 流水线单元格 `任务 9`–`任务 11` 自动执行：

1. **5 模型 `T5-Large` 前置去偏与词长先验对齐诊断（图 2 + 表 S1a，零主模型重训）**：
   - 调用 `evaluate_five_model_t5_diagnostics` 基于当前 `T5-Large` 检查点（`Word-Only`、`Full-Context (Joint)`、`+ Freeze CNN`、`+ Debias, Full-Context`、`+ Debias, Local (r = 2)`）在同一 `common_horizon_delay = 2` 测试子集上执行纯前向 `SyntheticTimingDataset`（`shared` / `independent`，`evaluation_seed = 20261009`）脉冲响应评测，并结合 `causal_fast_audit` 的 Clean / Shifted 指标导出 `five_model_debiasing_ledger.csv`、`synthetic_timing_summary.csv` 与 `five_model_t5_diagnostics_test.json`；
   - 跨全部 5 组模型计算预测词长高斯核密度与真实物理词长先验分布之间的 Jensen-Shannon 散度及降低量 $\Delta_{\text{JS}}$（单位：`bits`，含多随机种子均值±标准差与 Holm 校正置换检验 $p_{\text{Holm}}$）及 Top-10 词长误差下降（ms），导出 `duration_prior_alignment.csv`；
   - 基于 5 模型 `.pt` 词表得分抽取典型内容词（含不同词长）在未去偏模型与 `Local (r = 2)` 下的 Top-10 近邻词、目标词排名与真实物理词长，导出 `word_top10_neighborhoods.json` 与 `.csv`。
2. **词级视野对比、孤立单窗拉宽与词频四分位分层（图 3）**：
   - 调用 `collect_t5_long_window_comparison_rows` 汇总同一 `T5-Large` 体系下的 Stage-1 孤立拉宽词窗对照（`1.00 s` / `1.28 s` / `1.50 s` / `2.50 s`）与 Stage-2 去偏跨词家族（`Local (r = 0/1/2)`、`Full-Context`），导出 `long_window_comparison.csv`；
   - 基于事件时间戳导出 `physical_wait_distribution.csv`（`Word-Only` 与 `Local (r = 0..4)`、`Full-Context` 真实等待时长）；
   - 按训练集词频等分为 `Q1`–`Q4` 四分位统计各状态 Clean / Shifted / Net Top-10 与 Top-1，导出 `word_frequency_quartiles.csv`。
3. **句子解码定性例句、逐词候选矩阵与跨受试者耦合（图 4 + 表 1–2）**：
   - 导出单句代表性逐词 Top-5 神经概率矩阵（`sentence_neural_top5_matrix.json`）；
   - 按句子 BLEU-1 分位数抽取 `P90` / `P50` / `P10` 真实法语测试例句对比 `Word-Only`、`Local (r = 0/2)` 与 `Full-Context` 解码输出（`decoded_sentence_quantile_examples.json`）；
   - 计算受试者维度“词级 Top-10 (%) vs 句级 BLEU-1 (%) / Word Acc (%)”Pearson $r$、$p$ 值与 OLS 斜率，导出 `subject_retrieval_sentence_coupling.csv` 与 `.json`。
