"""BrainDecoding 公共代码包（MEG 连续语音神经词级与局部前瞻序列解码）。

分层子包结构：
- braindecoding.data: 统一 50 Hz 词窗数据集、PNPL 适配器、事件表契约与文本向量缓存
- braindecoding.models: 空间投影、扩张卷积脑编码器与局部前瞻 Transformer
- braindecoding.training: Stage 1 冻结基座训练、Stage 2 错位去偏与工作流编排
- braindecoding.decoding: 神经词表得分导出、KV-Cache 语言模型打分与单遍静态束搜索
- braindecoding.evaluation: 检索指标、词级/长窗/序列解码评估与配对统计显著性检验
"""

from __future__ import annotations
