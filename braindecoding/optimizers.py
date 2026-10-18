"""跨任务共享的优化器与学习率调度。"""

import math

import torch


def build_optimizer(model, config):
    """构造 AdamW 优化器。"""
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )


def build_scheduler(optimizer, total_steps, warmup_ratio=0.05):
    """使用短预热后的余弦衰减。"""
    warmup_steps = int(total_steps * float(warmup_ratio))

    def learning_rate_scale(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = max(total_steps - warmup_steps, 1)
        progress = min(max(step - warmup_steps, 0) / remaining, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)


def trainable_parameters(*modules):
    """从一个或多个模块中逐个返回不重复的可训练参数。"""
    seen = set()
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            identifier = id(parameter)
            if parameter.requires_grad and identifier not in seen:
                seen.add(identifier)
                yield parameter


def build_adamw_for_modules(modules, config):
    """为模型和损失中的可学习参数构建 AdamW。"""
    parameters = list(trainable_parameters(*modules))
    if not parameters:
        raise ValueError("没有可训练参数。")
    return torch.optim.AdamW(
        parameters,
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def build_cosine_annealing_scheduler(optimizer, epochs, minimum_learning_rate=0.0):
    """构建源 LibriBrain 基线使用的轮次级余弦调度器。"""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(epochs), 1),
        eta_min=float(minimum_learning_rate),
    )
