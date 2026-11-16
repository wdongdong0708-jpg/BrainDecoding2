# 自然连续语流局部跨词去偏编码与句子解码：当前生效实验协议快照

> **维护原则（State Snapshot）**：本文件是 `BrainDecoding` 代码仓唯一生效的实验协议与基准结果状态快照，随主线演进**原地覆写更新**，不追加历史流水账（历史版本留存 `git log`）。此前 `2026-09-27` 的 `BGE-M3 + Qwen2.5-1.5B + A/B/C` 在线句子改写草案及 `C-Gated` 屏幕闪烁门控支线已彻底废弃。

- **当前生效主数据集**：`Pallier2025`（法语连续语音听觉 MEG《小王子》，10 名受试者 `sub-01..10`，随机种子 `Seeds [0, 1, 2]`）
- **科学主线一句话**：第一阶段独立训练并冻结单词卷积基座（`Word-Only`）；第二阶段在跨词 Transformer 训练期引入同词间间隔错位脑窗遏制隐式词长捷径，测试期统一采用无错位支路单次前向（$\eta_{\text{eval}} = 0.0$）；将跨词视野收敛为已到达前序词与最多 $r = 2$ 个后续词窗的 `Rolling-K2` 直接定稿输出，在将平均输入等待时间缩短一半以上的同时，获得与整句联合编码相当的词表征质量与更优的句子解码性能。

---

## 1. `[ACTIVE]` vs `[DEPRECATED]` 组件与符号映射表

| 代码工程阶段 / 符号 | 协议状态 | 论文标准术语 | 对应配置文件 / 说明 |
| :--- | :--- | :--- | :--- |
| `stage: word` (`S0`) | `[ACTIVE]` | **单词独立解码（`Word-Only`）** | `configs/training/pallier2025_word_encoder.yaml`；第二阶段全程冻结作底 |
| `stage: context`（端到端微调） | `[ACTIVE Baseline]` | **`Full-Context (Joint)`** | 未冻结 CNN 且未去偏的整句双向基线（暴露隐式时长泄露） |
| `stage: context`（冻结 CNN，无去偏） | `[ACTIVE Baseline]` | **`+ Freeze CNN`** | 仅冻结 Stage 1 CNN、未施加错位约束的整句基线 |
| `stage: context`（孪生错位去偏） | `[ACTIVE Reference]` | **`+ Debias (Full-Context)`** | `configs/training/siamese_debiased/pallier2025_siamese_debiased_context.yaml` |
| `stage: prefix_revision`（$r \in \{0, 1, 2\}$） | **`[ACTIVE Primary]`** | **局部跨词联合编码（`Rolling-K0/K1/K2` / `+ Debias (Rolling-K2)`）** | `configs/training/siamese_debiased/pallier2025_siamese_debiased_prefix_revision.yaml`；可见范围 $z_{1:\min(n, i+r)}$ |
| `stage: revision`（$R = 2$） | `[ACTIVE Appendix A]` | **对称短窗 `Local-Window (R = 2)`** | 局部窗 $[\max(1, i-2), \min(n, i+r)]$；附录 A 对比消融 |
| `raw_simple` / `raw_full` 序列解码 | **`[ACTIVE Primary]`** | **第三阶段束搜索句子解码（$\eta_{\text{eval}} = 0.0$）** | `configs/decoding/raw_simple_sequence_decoding_protocol.yaml` 与 `raw_full_sequence_decoding_protocol.yaml` |
| `C-Gated` / `tau_gate` / `sweep_confidence_gate_pareto` | **`[DEPRECATED]`** | 严禁出现在论文中 | 已砍除“0 秒先出词 + 屏幕改写 + 防闪烁门控”支线，代码函数仅作兼容保留 |
| `F_local` / 修订率 / 有害覆写 | **`[DEPRECATED]`** | 严禁出现在论文中 | 同上；导出脚本自动过滤或人工制表时剔除 |
| `BGE-M3` + `Qwen2.5-1.5B` + `A/B/C` 协议 | **`[DEPRECATED]`** | 已由下节 `T5-Large` + `Qwen3-8B-Base` + 固定 $r=2$ 定稿取代 | 旧 `2026-09-27` 草案仅留存于 Git 历史 |

---

## 2. 当前生效三阶段数据、模型与训练协议

### 2.1 数据预处理与静态文本目标
- **脑磁信号预处理**：原始 $1{,}000\text{ Hz}$ 滤波（$0.1\text{--}40\text{ Hz}$）后降采样至 $50\text{ Hz}$；按采集段以 `RobustScaler` 标准化；从词起始时刻 $\tau_i$ 截取固定时长 $T_{\text{win}} = 1.0\text{ s}$（$S = 50$ 采样点），扣除窗口前 $0.5\text{ s}$ 均值完成基线校正并截断至 $[-5, 5]$。测试划分共 $1{,}940$ 个连续片段（每受试者 $194$ 个）、$18{,}450$ 个测试词位；完整训练词表 $\lvert V\rvert = 2{,}020$（覆盖测试集 $82.49\%$ 词位）。
- **目标词向量 $e_w$**：冻结 `t5-large` 第 12 层（`layer_fraction: 0.5`）子词均值池化并 $\ell_2$ 归一化（$d = 1{,}024$，`data/cache/derived/pallier2025/text/t5_large_layer_0_5/embeddings.npz`）。

### 2.2 Stage 1 与 Stage 2 训练超参
- **Stage 1（`Word-Only`）**：`configs/training/pallier2025_word_encoder.yaml`。$306 \to 270$ 空间注意力 + 5 个残差膨胀卷积块（隐藏维度 $160$，核大小 $3$）+ 时间注意力池化；`D-SigLIP` 去重对比损失（初始温度 $10.0$、偏置 $-10.0$、重复屏蔽阈值 $0.999$）；`AdamW`（`lr = 4e-4`, `weight_decay = 0.0`, `batch_size = 256`, `bf16-mixed`）。
- **Stage 2（去偏跨词联合编码）**：加载同种子 Stage 1 `best.pt` 并全程冻结 CNN（`freeze_brain_encoder_epochs = 30`）；上层为 16 层 16 头带 RoPE 的非预训练 Transformer（$d = 1{,}024$，注意力丢弃率 $0.1$）；`AdamW`（`lr = 1e-4`, `weight_decay = 0.01`, `batch_size = 128`）。
  - 同段错位负锚点：$M = 6$ 组循环平移 $\delta_m \in \{-15, -30, -45, -60, -90, -120\}\text{ s}$，训练期扣除系数 $\eta = 0.50$，零化惩罚权重 $\lambda_{\text{null}} = 10.0$；
  - 局部前序前缀加后续词模型 `Rolling-K0/K1/K2`（`configs/training/siamese_debiased/pallier2025_siamese_debiased_prefix_revision.yaml`）额外施加相对 $r = 0$ 的跨步一致性惩罚：$\lambda_\Delta = 8.0$、$\lambda_{\text{dir}} = 0.50$（配置中对应字段 `lambda_delta_null: 8.0`、`beta_delta_cm: 0.50`），等权优化 $r \in \{0, 1, 2\}$ 三步均值损失。

### 2.3 Stage 3 序列解码协议
- **语言模型**：冻结 `Qwen/Qwen3-8B-Base`。
- **束搜索参数**：每步仅保留脑信号余弦 Top $K_{\text{cand}} = 20$ 候选，局部 Softmax 温度 $T_r \in \{0.06, 0.07\}$（验证集网格选择），束宽 $B = 5$；语言权重满足 $\lambda_{K_0} \ge \lambda_{K_1} \ge \lambda_{K_2}$ 且测试推理固定 $\eta_{\text{eval}} = 0.0$。
- **两套正式评估协议**：
  1. **主基准（无 OOV 长句闭集 `L >= 5, 0% OOV`）**：`configs/decoding/raw_simple_sequence_decoding_protocol.yaml`，覆盖基准全集 $\lvert V\rvert = 131\ (n = 300,\ 2{,}300\text{ 词位})$ 及贪心覆盖子集 $\lvert V\rvert = 50\ (n = 100)$、$\lvert V\rvert = 100\ (n = 220)$；支持跨受试者多观测平均 $k \in \{1, 2, 5, 10\}$。
  2. **附录 B 全测试集协议**：`configs/decoding/raw_full_sequence_decoding_protocol.yaml`，覆盖未筛选全部 $n = 1{,}940$ 个测试片段、完整训练词表 $\lvert V\rvert = 2{,}020$。

---

## 3. 锁定测试评估口径与权威锚点结果（Seeds `0, 1, 2`, 均值 ± 标准差）

### 3.1 词级检索主表（表 1 口径）
- **锁定口径**：共同测试词位 $N = 14{,}590$，50 词查询位 $N_{50} = 7{,}440$；错位对照采用严格不接触未来的前序 $45.0\text{ s}$ 缓冲（`shift_mode=preroll`, $\delta_{\text{eval}} \in \{-10, -20, -30, -40\}\text{ s}$），单次前向推理 $\eta_{\text{eval}} = 0.0$；词类等权平衡准确率（随机机会 Top-10 $= 20.00\%$，Top-1 $= 2.00\%$）。

| 模型设置 | Top-10 Clean (%) ↑ | Top-10 Shifted (%) ↓ | Top-10 净增益 (pp) ↑ | Top-1 Clean (%) ↑ | Top-1 Shifted (%) ↓ | Top-1 净增益 (pp) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Word-Only` | `40.64 ± 0.15` | `20.07 ± 0.16` | `+20.57 ± 0.04` | `6.64 ± 0.13` | `1.97 ± 0.11` | `+4.67 ± 0.23` |
| `Full-Context (Joint)` | `68.42 ± 0.42` | `61.14 ± 0.78` | `+7.28 ± 1.17` | `20.74 ± 0.42` | `14.88 ± 1.01` | `+5.86 ± 1.34` |
| `+ Freeze CNN` | `64.33 ± 0.88` | `50.49 ± 3.45` | `+13.84 ± 2.61` | `17.24 ± 0.77` | `9.60 ± 1.29` | `+7.64 ± 0.54` |
| `+ Debias (Full-Context)` | `57.95 ± 2.51` | `22.61 ± 0.63` | `+35.33 ± 3.14` | `13.58 ± 1.57` | `2.62 ± 0.13` | `+10.96 ± 1.69` |
| **`+ Debias (Rolling-K2)`** | **`58.70 ± 1.02`** | `23.65 ± 0.17` | `+35.05 ± 1.17` | **`14.42 ± 0.35`** | `3.11 ± 0.11` | **`+11.31 ± 0.25`** |

- **附录 A.1 核心消融锚点（同 $N_{50} = 7{,}440$ 口径）**：
  - `Local-Window (R = 2)` 对称短窗：Top-10 Clean `61.46 ± 0.88%` / Shifted `29.96 ± 1.03%` / Net `+31.50 ± 0.23 pp`；
  - `Rolling-K2 (w/o L_delta-null)`：Top-10 Clean `57.89 ± 1.15%` / Shifted `24.62 ± 0.44%` / Net `+33.27 ± 1.44 pp`；
  - `Rolling-K2 + 测试在线均值扣除 (ema_decay=0.99, eta_eval=0.5)`：Top-10 Clean `58.42 ± 0.97%` / Shifted `23.74 ± 0.15%` / Net `+34.69 ± 1.10 pp`。

### 3.2 句子解码主表与全测试集锚点（表 2 与表 B.1 口径，$k = 1, \eta_{\text{eval}} = 0.0$）

| 解码策略 | 验证集 $\lambda_r$ | $\lvert V\rvert=100$ BLEU-1 (%) | $\lvert V\rvert=131$ BLEU-1 (%) | $\lvert V\rvert=131$ Top-1 (%) | $\lvert V\rvert=131$ 平均等待 (s) | 全集 $\lvert V\rvert=2{,}020$ BLEU-1 (%) | 全集平均等待 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Word-Only` | `0.13` | `15.90 ± 0.83` | `13.26 ± 0.47` | `6.43 ± 0.16` | `1.00` | `9.15 ± 0.56` | `1.00` |
| `Rolling-K0` | `0.12` | `17.09 ± 0.83` | `14.78 ± 0.77` | `7.49 ± 0.58` | `1.00` | `12.20 ± 0.64` | `1.00` |
| `Rolling-K1` | `0.08` | `20.15 ± 1.23` | `17.10 ± 0.59` | `10.38 ± 0.32` | `1.87` | `13.63 ± 0.32` | `1.89` |
| **`Rolling-K2`** | `0.08` | **`21.74 ± 0.99`** | **`18.70 ± 0.94`** | **`12.10 ± 0.29`** | `2.61` | **`14.09 ± 0.42`** | `2.69` |
| `Full-Context` | `0.37` | `20.90 ± 2.01` | `17.55 ± 1.75` | `10.51 ± 1.94` | `5.33` | `12.38 ± 0.97` | `8.31` |

*(完整汇总导出报告源位于独立论文仓 `C:\Users\Administrator\Desktop\论文\笔记本\实验数据\`，代码仓轻量化结构表已归档于 `data/results/studies/pallier2025/`，含 7 组核心实验 `tables/*.csv`、`causal_fast_audit_summary/`、`raw_*_sequence_decoding/tables/*.csv` 与 `word_and_synthetic_diagnostics/`。)*

---

## 4. 待办增量评估与消融路线图（不改变主线前提下的补强项）

1. **P0 — 定性词检索与句子解码样本导出**（纯读已保存测试预测，无需重训）：抽取典型法语测试词的 Top-10 语义/近音邻域预测与代表性句子对比样本（`Word-Only`、`Rolling-K0/K2`、`Full-Context`，附中英文释义）。
2. **P1 — `Joint Synthetic` 纯脉冲负对照与词长分布一致性诊断 ($\Delta_{\text{JS}}$)**：基于现有 `braindecoding/data/synthetic_timing_dataset.py`（$60/120/240\text{ ms}$ 平滑泊松脉冲）评估已训练模型并计算与高斯物理词长先验分布的 Jensen-Shannon 散度对齐度（对标 SimpleB2T Sec. 2.3 & App. A.2/A.5）。
3. **P1/P2 — 跨受试者一致性范数加权与额外对照整理**：审查 `braindecoding/decoding/rolling_beam_search.py` 的 `_average_subject_score_rows_rms_preserved`（拟从逐词位逆 RMS 缩放改为统一标量加权平均以贴合 SimpleB2T Eq. 7–8 物理依据）；补充 `pallier2025_causal_null_space_prefix_revision.yaml`（纯前序 $\eta=0$ 简化训练）与长窗对比诊断 `word_and_synthetic_diagnostics/long_window_comparison.csv`（$1.0\text{ s}/1.27\text{ s}/1.5\text{ s}$）纳入附录核对。
