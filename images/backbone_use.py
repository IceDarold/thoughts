# 1) Собираешь DataLoader'ы под картинки/спектры
train_loader = ...
val_loader = ...

# 2) Запускаешь обучение
best = train_cnn_classifier(
    train_loader=train_loader,
    val_loader=val_loader,
    model_name="resnet18",   # или "resnet50", "tf_efficientnet_b0_ns", "convnext_tiny"
    n_classes=num_classes,
    in_chans=1,              # если спектрограммы (1 канал), 3 — если RGB
    num_epochs=10,
    freeze_epochs=3,         # сначала linear probe, потом чуть fine-tune backbone
)
