"""
audio_cnn_train.py — шаблон обучения CNN для аудио (KWS/классификация клипов)

Ожидания:
- есть audio_toolbox.py с:
  - KWSDataset
  - set_seed
  - kws_metrics (accuracy + confusion matrix, опц.)

Тебе останется:
- собрать список items = [(path, label_int), ...]
- задать num_classes, labels2id/id2labels
- запустить train_cnn(...)
"""

import os
import math
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

from audio_toolbox import (
    set_seed,
    KWSDataset,
    kws_metrics,  # можно не использовать, если не нужно
)


# =====================
# Модель CNN
# =====================

class AudioCNN(nn.Module):
    """
    Простой CNN по лог-мел спектрам.
    Вход: (B, 1, n_mels, T)
    Выход: logits (B, n_classes)
    """

    def __init__(
        self,
        n_mels: int,
        n_classes: int,
        base_channels: int = 32,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (n_mels/2, T/2)

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (n_mels/4, T/4)

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            # усредняем по freq/time
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(base_channels * 4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, n_mels, T)
        """
        x = self.features(x)          # (B, C, 1, 1)
        x = x.view(x.size(0), -1)     # (B, C)
        x = self.classifier(x)        # (B, n_classes)
        return x


# =====================
# Train / Eval лупы
# =====================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int = 50,
) -> float:
    """
    Обучение за одну эпоху.
    Возвращает средний loss.
    """
    model.train()
    running_loss = 0.0
    n_samples = 0

    for batch_idx, (spec, labels) in enumerate(loader):
        spec = spec.to(device)        # (B, 1, F, T)
        labels = labels.to(device)    # (B,)

        optimizer.zero_grad()
        logits = model(spec)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [train] batch {batch_idx+1}/{len(loader)} loss={loss.item():.4f}")

    return running_loss / max(1, n_samples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Оценка модели: loss + accuracy.
    Для подробных метрик можно использовать kws_metrics.
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0

    all_preds = []
    all_labels = []

    for spec, labels in loader:
        spec = spec.to(device)
        labels = labels.to(device)

        logits = model(spec)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = kws_metrics(all_labels, all_preds)  # accuracy + per-class
    avg_loss = total_loss / max(1, n_samples)

    out = {
        "val_loss": float(avg_loss),
        "val_accuracy": float(metrics["accuracy"]),
    }
    return out


# =====================
# Основная функция обучения
# =====================

def train_cnn(
    items: List[Tuple[str, int]],
    num_classes: int,
    labels: Optional[List[str]] = None,
    sr: int = 16000,
    clip_duration: float = 1.0,
    n_mels: int = 64,
    feature_type: str = "logmel",   # или "mfcc"
    batch_size: int = 64,
    num_epochs: int = 20,
    lr: float = 1e-3,
    val_fraction: float = 0.15,
    num_workers: int = 2,
    device_str: str = "cuda",
    random_state: int = 42,
    model_out_path: str = "audio_cnn_best.pth",
):
    """
    items: список (path, label_int)
    num_classes: число классов
    labels: список имён классов (опционально, для логов)
    """

    set_seed(random_state)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- Train/val split ---
    n_total = len(items)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    print(f"Total items={n_total}, train={n_train}, val={n_val}")

    # Чтобы не возиться с индексами — random_split по Dataset позже:
    # здесь просто перемешаем список
    random.shuffle(items)
    train_items = items[:n_train]
    val_items = items[n_train:]

    # --- Dataset / DataLoader ---
    train_dataset = KWSDataset(
        train_items,
        sr=sr,
        clip_duration=clip_duration,
        feature_type=feature_type,
        waveform_transform=None,   # сюда можно повесить add_white_noise/random_gain/time_shift
        spec_transform=None,       # сюда — random_time_mask/random_freq_mask
        n_mels=n_mels,
    )
    val_dataset = KWSDataset(
        val_items,
        sr=sr,
        clip_duration=clip_duration,
        feature_type=feature_type,
        waveform_transform=None,
        spec_transform=None,
        n_mels=n_mels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # --- Модель / оптимизатор / лосс ---
    model = AudioCNN(
        n_mels=n_mels,
        n_classes=num_classes,
        base_channels=32,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",     # по accuracy
        factor=0.5,
        patience=3,
        verbose=True,
    )

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            log_interval=50,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        val_loss = val_metrics["val_loss"]
        val_acc = val_metrics["val_accuracy"]

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # шаг lr scheduler
        scheduler.step(val_acc)

        # сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }
            torch.save(best_state, model_out_path)
            print(f"  >>> New best val_acc={val_acc:.4f}, saved to {model_out_path}")

    print("\nTraining finished.")
    print(f"Best val_acc={best_val_acc:.4f}")
    if best_state is None:
        print("Warning: best_state is None (что-то пошло не так с обучением)")

    return best_state


# =====================
# Пример вызова
# =====================

if __name__ == "__main__":
    """
    Здесь псевдокод: ты сам подставишь путь к датасету.
    Предположим, у тебя есть:
    - root_dir с папками по классам или csv с (path, label)
    """

    # Пример: если есть словарь label_name -> id
    label2id = {
        "yes": 0,
        "no": 1,
        "up": 2,
        "down": 3,
        # ...
    }
    id2label = {v: k for k, v in label2id.items()}

    # TODO: собрать items = [(wav_path, label_id), ...]
    items = []  # заполняешь сам под задачу

    if len(items) == 0:
        print("Заполни items перед запуском обучения!")
    else:
        best = train_cnn(
            items=items,
            num_classes=len(label2id),
            labels=list(label2id.keys()),
            sr=16000,
            clip_duration=1.0,
            n_mels=64,
            feature_type="logmel",
            batch_size=64,
            num_epochs=20,
            lr=1e-3,
            val_fraction=0.15,
            num_workers=2,
            device_str="cuda",
            random_state=42,
            model_out_path="audio_cnn_best.pth",
        )
