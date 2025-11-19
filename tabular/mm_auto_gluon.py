from autogluon.multimodal import MultiModalPredictor
import pandas as pd

TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
TARGET_COL = "target"

def main():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # Например, если есть колонка image_path с путями к картинкам
    # и текстовая колонка text
    # AutoGluon сам разрулит типы фичей

    predictor = MultiModalPredictor(
        label=TARGET_COL,
        problem_type="regression",  # или "binary"/"multiclass"
        eval_metric="rmse",         # поменяй под задачу
    )

    predictor.fit(
        train_data=train,
        time_limit=60 * 60,
        presets="medium_quality",
    )

    preds = predictor.predict(test)
    preds.to_csv("submission_mm.csv", index=False)

if __name__ == "__main__":
    main()
