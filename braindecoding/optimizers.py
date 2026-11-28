"""跨任务共享的优化器、学习率调度器与模型参数指数滑动平均（EMA）。"""

from contextlib import contextmanager
import math

import torch


_NO_DECAY_NAME_TOKENS = (
    "bias",
    "rel_pos_logits",
    "pos_logits",
    "neg_logits",
    "sink_logits",
    "log_temperature",
    "past_bias",
    "future_bias",
    "correction",
)


def build_optimizer(model, config):
    """构造 AdamW 优化器。"""
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )


def build_scheduler(
    optimizer,
    total_steps,
    warmup_ratio=0.05,
    minimum_learning_rate=0.0,
):
    """使用短预热后的步级余弦衰减（支持可选最小学习率下限）。"""
    total_steps = max(int(total_steps), 1)
    warmup_steps = int(total_steps * float(warmup_ratio))
    min_lr = max(0.0, float(minimum_learning_rate))

    def _make_group_lambda(base_lr: float):
        min_scale = (
            min(1.0, min_lr / float(base_lr))
            if (min_lr > 0.0 and float(base_lr) > 0.0)
            else 0.0
        )

        def learning_rate_scale(step):
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            remaining = max(total_steps - warmup_steps, 1)
            progress = min(max(step - warmup_steps, 0) / remaining, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_scale + (1.0 - min_scale) * cosine

        return learning_rate_scale

    lambdas = [
        _make_group_lambda(float(group.get("lr", 0.0)))
        for group in optimizer.param_groups
    ]
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambdas if len(lambdas) > 1 else lambdas[0]
    )


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


def _is_no_decay_parameter(name: str, parameter: torch.Tensor) -> bool:
    if parameter.ndim < 2:
        return True
    lowered = str(name).lower()
    return any(tok in lowered for tok in _NO_DECAY_NAME_TOKENS)


def build_adamw_for_modules(modules, config):
    """为模型和损失中的可学习参数构建 AdamW（可选解耦权重衰减分组）。"""
    lr = float(config.get("learning_rate", 1e-4))
    wd = float(config.get("weight_decay", 0.0))
    raw_betas = config.get("betas")
    adamw_kwargs = {}
    if raw_betas is not None and len(raw_betas) == 2:
        adamw_kwargs["betas"] = (float(raw_betas[0]), float(raw_betas[1]))

    if not bool(config.get("decoupled_weight_decay", False)):
        parameters = list(trainable_parameters(*modules))
        if not parameters:
            raise ValueError("没有可训练参数。")
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=wd,
            **adamw_kwargs,
        )

    raw_swiglu_wd = config.get("neighbor_swiglu_weight_decay")
    swiglu_wd = None if raw_swiglu_wd is None else float(raw_swiglu_wd)
    raw_decollapse_wd = config.get("self_decollapse_weight_decay")
    decollapse_wd = (
        None if raw_decollapse_wd is None else float(raw_decollapse_wd)
    )

    decay_params = []
    swiglu_decay_params = []
    decollapse_decay_params = []
    no_decay_params = []
    seen = set()
    for module in modules:
        if module is None:
            continue
        for name, parameter in module.named_parameters():
            identifier = id(parameter)
            if not parameter.requires_grad or identifier in seen:
                continue
            seen.add(identifier)
            lowered_name = str(name).lower()
            if _is_no_decay_parameter(name, parameter):
                no_decay_params.append(parameter)
            elif swiglu_wd is not None and "neighbor_swiglu" in lowered_name:
                swiglu_decay_params.append(parameter)
            elif (
                decollapse_wd is not None
                and "self_decollapse" in lowered_name
            ):
                decollapse_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

    if (
        not decay_params
        and not swiglu_decay_params
        and not decollapse_decay_params
        and not no_decay_params
    ):
        raise ValueError("没有可训练参数。")
    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": wd})
    if swiglu_decay_params:
        param_groups.append(
            {"params": swiglu_decay_params, "weight_decay": swiglu_wd}
        )
    if decollapse_decay_params:
        param_groups.append(
            {"params": decollapse_decay_params, "weight_decay": decollapse_wd}
        )
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return torch.optim.AdamW(param_groups, lr=lr, **adamw_kwargs)


def build_cosine_annealing_scheduler(optimizer, epochs, minimum_learning_rate=0.0):
    """构建源 LibriBrain 基线使用的轮次级余弦调度器。"""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(epochs), 1),
        eta_min=float(minimum_learning_rate),
    )


class ModelEMA:
    """仅跟踪可训练浮点参数的指数滑动平均（零 state_dict 键名侵入，bf16 下维持 fp32 影子精度）。"""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        decay: float = 0.999,
        trainable_only: bool = True,
    ):
        decay_f = float(decay)
        if not (0.0 < decay_f < 1.0):
            raise ValueError(f"ModelEMA decay 必须位于 (0, 1) 之间，收到：{decay_f}")
        self.decay = decay_f
        self.trainable_only = bool(trainable_only)
        self.num_updates = 0
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if not param.is_floating_point():
                continue
            if self.trainable_only and not param.requires_grad:
                continue
            self.shadow[name] = param.detach().float().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        eff_decay = min(
            self.decay,
            (1.0 + float(self.num_updates)) / (10.0 + float(self.num_updates)),
        )
        alpha = 1.0 - eff_decay
        param_map = dict(model.named_parameters())
        for name, shadow_tensor in self.shadow.items():
            param = param_map.get(name)
            if param is None:
                continue
            src = param.detach()
            if shadow_tensor.device != src.device:
                shadow_tensor = shadow_tensor.to(device=src.device)
                self.shadow[name] = shadow_tensor
            shadow_tensor.lerp_(src.float(), alpha)

    @contextmanager
    def apply_to(self, model: torch.nn.Module):
        """临时将 EMA 影子权重切入模型（用于验证与保存最优检查点），退出时无损恢复原始训练权重。"""
        backup: dict[str, torch.Tensor] = {}
        param_map = dict(model.named_parameters())
        with torch.no_grad():
            for name, shadow_tensor in self.shadow.items():
                param = param_map.get(name)
                if param is None:
                    continue
                backup[name] = param.detach().clone()
                param.copy_(shadow_tensor.to(device=param.device, dtype=param.dtype))
        try:
            yield model
        finally:
            with torch.no_grad():
                for name, raw_tensor in backup.items():
                    param_map[name].copy_(raw_tensor)

    def state_dict(self) -> dict:
        return {
            "decay": float(self.decay),
            "trainable_only": bool(self.trainable_only),
            "num_updates": int(self.num_updates),
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.trainable_only = bool(state.get("trainable_only", self.trainable_only))
        self.num_updates = int(state.get("num_updates", 0))
        loaded_shadow = state.get("shadow") or {}
        for name, tensor in loaded_shadow.items():
            if name in self.shadow:
                self.shadow[name] = tensor.to(
                    device=self.shadow[name].device, dtype=torch.float32
                ).clone()
            else:
                self.shadow[name] = tensor.float().clone()
