 """
autogluon_template.py — минимальный шаблон под табличку (Kaggle-стайл)

Что делает:
- грузит train/test
- задаёт label и id
- тренирует AutoGluon TabularPredictor
- показывает leaderboard
- делает сабмит

Дальше сам подставляешь пути, названия колонок и метрику.
"""

import os
import pandas as pd

from autogluon.tabular import TabularPredictor


# =========================
# CONFIG
# =========================

TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"

ID_COL = "id"           # колонка с идентификатором объекта
TARGET_COL = "target"   # таргет

# "regression" / "binary" / "multiclass"
PROBLEM_TYPE = "binary"

# подставь свою метрику:
# regression: "rmse", "mae"
# binary: "roc_auc", "log_loss", "accuracy"
# multiclass: "log_loss", "accuracy", и т.д.
EVAL_METRIC = "roc_auc"

# "medium_quality", "best_quality", "high_quality_fast_inference", ...
PRESETS = "medium_quality"

TIME_LIMIT = 60 * 60   # сек, тут 1 час на fit


# =========================
# MAIN
# =========================

def main():
    # --- загрузка данных ---
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    print("train shape:", train.shape)
    print("test  shape:", test.shape)
    print("train columns:", train.columns.tolist())

    # sanity-check
    assert TARGET_COL in train.columns, f"{TARGET_COL} не найден в train"
    assert ID_COL in train.columns and ID_COL in test.columns, "ID_COL не найден в train/test"

    # --- настройка и обучение ---
    predictor = TabularPredictor(
        label=TARGET_COL,
        problem_type=PROBLEM_TYPE,
        eval_metric=EVAL_METRIC,
    )

    predictor = predictor.fit(
        train_data=train,
        presets=PRESETS,
        time_limit=TIME_LIMIT,
        # при наличии GPU можно добавить:
        # ag_args_fit={"num_gpus": 1},
    )

    # --- leaderboard на train (OOF-оценка) ---
    lb = predictor.leaderboard(train, silent=False)
    print(lb)

    # --- предсказания на test ---
    if PROBLEM_TYPE == "binary":
        # для бинарки часто нужны вероятности положительного класса
        proba = predictor.predict_proba(test)[1]
        preds_for_submit = proba
    else:
        # для регрессии / multiclass чаще всего достаточно predict
        preds_for_submit = predictor.predict(test)

    # --- сабмит ---
    submission = pd.DataFrame({
        ID_COL: test[ID_COL],
        TARGET_COL: preds_for_submit,
    })

    out_path = "submission_autogluon.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    main()
