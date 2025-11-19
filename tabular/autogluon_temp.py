"""
autogluon_geo_image_template.py — шаблон под табличку + geo + image-эмбеддинги

Что делает:
- грузит train/test
- при необходимости — мёрджит отдельные эмбеддинги по id
- строит простые geo-фичи из lat/lon
- оставляет эмбеддинги как числовые фичи (AutoGluon сам их съест)
- тренирует TabularPredictor
- делает сабмит

Подправь CONFIG-блок под свою задачу (пути, имена столбцов, метрика и т.п.).
"""

import os
import math
import pandas as pd

from autogluon.tabular import TabularPredictor


# =========================
# CONFIG
# =========================

# Базовые файлы
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"

ID_COL = "id"           # идентификатор объекта
TARGET_COL = "target"   # таргет

# Если geo есть — выставь имена колонок; если нет — поставь None
LAT_COL = "lat"         # или None, если нет
LON_COL = "lon"         # или None, если нет

# Если эмбеддинги в отдельных файлах — укажи пути
# Если эмбеды уже внутри train/test (столбцы вида emb_0, emb_1, ...),
# оставь тут None, а EMB_PREFIX будет искать их по префиксу.
TRAIN_EMB_CSV = None          # например: "train_image_emb.csv"
TEST_EMB_CSV = None           # например: "test_image_emb.csv"
EMB_ID_COL = "id"             # id-колонка в файлах эмбеддингов

EMB_PREFIX = "emb_"           # префикс колонок эмбеддингов, если они уже в train/test

# Тип задачи и метрика
PROBLEM_TYPE = "binary"       # "regression" / "binary" / "multiclass"
EVAL_METRIC = "roc_auc"       # подставь свою

PRESETS = "medium_quality"    # "medium_quality", "best_quality", "high_quality_fast_inference", ...
TIME_LIMIT = 60 * 60          # секунд, здесь 1 час

# Опционально: дополнительные колонки, которые надо игнорировать
EXTRA_IGNORE_COLS = []        # например ["some_leaky_col"]


# =========================
# Фичи
# =========================

def haversine_distance(lat1, lon1, lat2, lon2, radius=6371.0):
    """
    Классическое расстояние по сфере (км).
    Можно использовать как geo-фичу до "центра".
    """
    import numpy as np

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    return radius * c


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Простые geo-фичи из lat/lon, если они заданы.
    """
    import numpy as np

    df = df.copy()

    if LAT_COL is None or LON_COL is None:
        return df
    if LAT_COL not in df.columns or LON_COL not in df.columns:
        return df

    lat = df[LAT_COL].astype("float32")
    lon = df[LON_COL].astype("float32")

    # Нормализация
    df["geo_lat_norm"] = (lat - lat.mean()) / (lat.std() + 1e-6)
    df["geo_lon_norm"] = (lon - lon.mean()) / (lon.std() + 1e-6)

    # Полушария
    df["geo_hemisphere_ns"] = (lat > 0).astype("int8")
    df["geo_hemisphere_ew"] = (lon > 0).astype("int8")

    # Расстояние до условного "центра" (средней точки)
    center_lat = lat.mean()
    center_lon = lon.mean()
    df["geo_center_dist_km"] = haversine_distance(
        lat.values,
        lon.values,
        center_lat,
        center_lon,
    ).astype("float32")

    return df


def merge_external_embeddings(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_emb_path: str,
    test_emb_path: str,
    id_col: str = ID_COL,
    emb_id_col: str = EMB_ID_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Если эмбеддинги лежат в отдельных csv — пример джойна по id.
    Ожидает, что emb_id_col уникален в каждом emb-файле.
    """
    emb_tr = pd.read_csv(train_emb_path)
    emb_te = pd.read_csv(test_emb_path)

    assert emb_tr[emb_id_col].is_unique, "train эмбеддинги: id не уникален"
    assert emb_te[emb_id_col].is_unique, "test эмбеддинги: id не уникален"

    print("train_emb shape:", emb_tr.shape)
    print("test_emb  shape:", emb_te.shape)

    train = train.merge(
        emb_tr,
        left_on=id_col,
        right_on=emb_id_col,
        how="left",
        validate="1:1",
    )
    test = test.merge(
        emb_te,
        left_on=id_col,
        right_on=emb_id_col,
        how="left",
        validate="1:1",
    )

    if emb_id_col != id_col:
        train = train.drop(columns=[emb_id_col])
        test = test.drop(columns=[emb_id_col])

    return train, test


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Общий билдер фич:
    - geo (если есть LAT_COL/LON_COL)
    - эмбеддинги уже считаем готовыми числовыми фичами (AutoGluon сам разберётся)
    """
    df = df.copy()

    # GEO
    df = add_geo_features(df)

    # Можно воткнуть сюда любые доп. фичи под задачу

    return df


# =========================
# MAIN
# =========================

def main():
    # --- загрузка данных ---
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    print("train shape:", train.shape)
    print("test  shape:", test.shape)
    print("train columns:", train.columns.tolist()[:20], "...")
    print("test  columns:", test.columns.tolist()[:20], "...")

    assert TARGET_COL in train.columns, f"{TARGET_COL} не найден в train"
    assert ID_COL in train.columns and ID_COL in test.columns, "ID_COL не найден в train/test"

    # --- при необходимости: мёрджим отдельные эмбеддинги ---
    if TRAIN_EMB_CSV is not None and TEST_EMB_CSV is not None:
        train, test = merge_external_embeddings(
            train,
            test,
            train_emb_path=TRAIN_EMB_CSV,
            test_emb_path=TEST_EMB_CSV,
            id_col=ID_COL,
            emb_id_col=EMB_ID_COL,
        )

    # --- добавляем geo-фичи и прочие ---
    train_fe = build_features(train)
    test_fe = build_features(test)

    # --- список колонок, которые надо игнорировать при обучении ---
    ignore_cols = [ID_COL] + EXTRA_IGNORE_COLS
    ignore_cols = [c for c in ignore_cols if c in train_fe.columns]

    print("Ignored columns:", ignore_cols)

    # --- TabularPredictor ---
    predictor = TabularPredictor(
        label=TARGET_COL,
        problem_type=PROBLEM_TYPE,
        eval_metric=EVAL_METRIC,
    )

    predictor = predictor.fit(
        train_data=train_fe,
        presets=PRESETS,
        time_limit=TIME_LIMIT,
        ignored_columns=ignore_cols,
        # если есть GPU и оно нужно:
        # ag_args_fit={"num_gpus": 1},
    )

    # --- Leaderboard на train (OOF) ---
    print("==== Leaderboard (train) ====")
    lb = predictor.leaderboard(train_fe, silent=False)
    print(lb)

    # --- Предсказания на test ---
    if PROBLEM_TYPE == "binary":
        # для бинарки по умолчанию ожидаем, что таргет {0,1},
        # и берём вероятность класса "1"
        proba = predictor.predict_proba(test_fe)[1]
        preds_for_submit = proba
    else:
        preds_for_submit = predictor.predict(test_fe)

    # --- Сабмит ---
    submission = pd.DataFrame({
        ID_COL: test_fe[ID_COL],
        TARGET_COL: preds_for_submit,
    })

    out_path = "submission_autogluon_geo_image.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    main()
