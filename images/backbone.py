"""
cnn_utils.py — утилиты для использования/дообучения CNN-бэкбонов

Идеи:
- создать backbone (через timm или torchvision)
- заморозить/разморозить части
- прикрутить простую голову (linear probe / MLP)
- обучить в несколько эпох (train/eval)
- аккуратно отделить LR для backbone и head

Точки, которые почти точно придётся править под задачу, помечены в комментариях.
"""

import math
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# ==========
# Общие утилиты
# ==========

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ==========
# Создание backbone'а
# ==========

def create_vision_backbone(
    model_name: str = "resnet18",
    pretrained: bool = True,
    in_chans: int = 3,
    use_timm_first: bool = True,
):
    """
    Создаёт CNN-backbone и возвращает (backbone, feature_dim).

    Вариант 1: timm (если установлен).
    Вариант 2: torchvision (если timm нет или не хочется).

    feature_dim — размер эмбеддинга после global pooling.

    Примеры модельных имён для timm:
    - "resnet18", "resnet50"
    - "tf_efficientnet_b0_ns", "tf_efficientnet_b3_ns"
    - "convnext_tiny", "convnext_base"
    """

    backbone = None
    feature_dim = None

    # ---- timm-вариант ----
    if use_timm_first:
        try:
            import timm
            # num_classes=0 -> возвращает эмбеддинги, а не logits
            backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=0,
                global_pool="avg",
            )
            feature_dim = backbone.num_features
            print(f"[create_vision_backbone] Using timm model={model_name}, feat_dim={feature_dim}")
            return backbone, feature_dim
        except Exception as e:
            print(f"[create_vision_backbone] timm path failed: {e}. Falling back to torchvision...")

    # ---- torchvision-вариант (пример для resnet18/50) ----
    import torchvision.models as tvm
    from torchvision.models import ResNet18_Weights, ResNet50_Weights

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = tvm.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = tvm.resnet50(weights=weights)
    else:
        raise ValueError(f"Unknown model_name for torchvision path: {model_name}")

    # если у тебя инпут 1-канальный (например, спектрограммы), нужно заменить conv1
    if in_chans != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_chans,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        print(f"[create_vision_backbone] Replaced conv1 for in_chans={in_chans}")

    feature_dim = model.fc.in_features
    model.fc = nn.Identity()  # выкидываем классификатор -> backbone возвращает эмбеддинги

    backbone = model
    print(f"[create_vision_backbone] Using torchvision {model_name}, feat_dim={feature_dim}")
    return backbone, feature_dim


# ==========
# Фриз/разморозка backbone'а
# ==========

def freeze_backbone(model: nn.Module, freeze: bool = True):
    """
    Ожидается, что у модели есть атрибут .backbone.
    """
    for p in model.backbone.parameters():
        p.requires_grad = not freeze
    print(f"[freeze_backbone] freeze={freeze}")


def unfreeze_last_n_layers_backbone(model: nn.Module, n: int = 1):
    """
    Очень грубая эвристика: размораживаем последние n "слоёв" backbone.
    Для ResNet / ConvNeXt это может означать последние блоки.
    Не гарантируется, что будет идеально для любой архитектуры — подгоняй под задачу.
    """
    layers = list(model.backbone.children())
    print(f"[unfreeze_last_n_layers_backbone] total layers={len(layers)}, unfreeze last={n}")
    # сначала фризим всё
    for p in model.backbone.parameters():
        p.requires_grad = False
    # потом размораживаем последние n
    for layer in layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True


# ==========
# Классификационная голова и общий CNN-класс
# ==========

class LinearHead(nn.Module):
    """
    Простая голова: Linear -> (опц.) Dropout -> (опц.) BatchNorm -> (опц.) активация.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        layers = []
        d_in = in_dim

        if hidden_dim is not None:
            layers.append(nn.Linear(d_in, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "gelu":
                layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))
        else:
            layers.append(nn.Linear(d_in, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNNClassifier(nn.Module):
    """
    Обёртка над backbone + классификационная голова.

    Ожидается:
    - backbone возвращает эмбеддинг размера feature_dim
    - head: LinearHead или любой другой nn.Module
    """

    def __init__(self, backbone: nn.Module, feature_dim: int, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = LinearHead(feature_dim, n_classes)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        feats = self.backbone(x)  # (B, feature_dim)
        logits = self.head(feats) # (B, n_classes)
        return logits


# ==========
# Optimizer / scheduler
# ==========

def create_optimizer(
    model: CNNClassifier,
    lr_backbone: float = 1e-4,
    lr_head: float = 1e-3,
    weight_decay: float = 1e-4,
):

    """
    Два набора параметров:
    - backbone (обычно меньший lr)
    - head (обычно lr побольше)
    """

    params = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": lr_backbone},
        {"params": [p for p in model.head.parameters() if p.requires_grad], "lr": lr_head},
    ]
    optimizer = optim.AdamW(params, weight_decay=weight_decay)
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    num_epochs: int,
    warmup_epochs: int = 0,
):
    """
    Простейший cosine scheduler.
    На туре можно вообще обойтись без scheduler, но пусть будет заготовка.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        t = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


# ==========
# Train / Eval лупы
# ==========

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    log_interval: int = 50,
    use_amp: bool = True,
):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    running_loss = 0.0
    n_samples = 0

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        if (step + 1) % log_interval == 0:
            print(f"[train] step {step+1}/{len(loader)}, loss={loss.item():.4f}")

    return running_loss / max(1, n_samples)


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    total_loss = 0.0
    n_samples = 0

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        if criterion is not None:
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = float((all_preds == all_labels).mean())
    metrics = {"accuracy": acc}
    if criterion is not None:
        metrics["loss"] = total_loss / max(1, n_samples)
    return metrics


# ==========
# Высокоуровневая функция обучения
# ==========

def train_cnn_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    n_classes: int,
    in_chans: int = 3,
    num_epochs: int = 10,
    device_str: str = "cuda",
    lr_backbone: float = 1e-4,
    lr_head: float = 1e-3,
    freeze_epochs: int = 0,
    use_timm_first: bool = True,
):
    """
    Высокоуровневый сценарий:

    1) создать backbone
    2) собрать CNNClassifier
    3) (опц.) заморозить backbone на первые freeze_epochs
    4) обучать, логировать accuracy
    """

    set_seed(42)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    backbone, feat_dim = create_vision_backbone(
        model_name=model_name,
        pretrained=True,
        in_chans=in_chans,
        use_timm_first=use_timm_first,
    )
    model = CNNClassifier(backbone, feat_dim, n_classes).to(device)
    print("[train_cnn_classifier] total params:", count_params(model, trainable_only=False))
    print("[train_cnn_classifier] trainable params:", count_params(model, trainable_only=True))

    optimizer = create_optimizer(
        model,
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        weight_decay=1e-4,
    )
    scheduler = create_scheduler(
        optimizer,
        num_epochs=num_epochs,
        warmup_epochs=0,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")

        # простая логика: первые freeze_epochs — backbone фрозен
        if freeze_epochs > 0 and epoch <= freeze_epochs:
            freeze_backbone(model, freeze=True)
        else:
            freeze_backbone(model, freeze=False)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            log_interval=50,
            use_amp=True,
        )
        val_metrics = eval_one_epoch(
            model,
            val_loader,
            device,
            criterion=criterion,
        )

        scheduler.step()

        val_acc = val_metrics["accuracy"]
        val_loss = val_metrics["loss"]
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "model_name": model_name,
                "feature_dim": feat_dim,
                "n_classes": n_classes,
                "in_chans": in_chans,
            }
            torch.save(best_state, f"cnn_classifier_best_{model_name}.pth")
            print(f"  >>> New best val_acc={val_acc:.4f}, checkpoint saved.")

    print(f"\n[train_cnn_classifier] best_val_acc={best_val_acc:.4f}")
    return best_state
