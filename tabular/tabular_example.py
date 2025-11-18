"""
main_tabular_geo_image.py — пример полного пайплайна:
- загрузка train/test
- geo-фичи
- PCA по эмбеддингам
- LightGBM с CV
- сабмит + диагностика по подгруппам

Предполагается, что функции из toolbox.py либо:
- лежат в модуле `toolbox`, либо
- скопированы выше по ноутбуку.
"""

import numpy as np
import pandas as pd

from toolbox import (
    set_seed,
    load_train_test,
    quick_target_info,
    compare_train_test,

    add_geo_basic_features,
    add_geo_center_distance,
    add_geo_clusters,

    get_embedding_cols,
    add_embedding_norms,
    pca_embeddings,

    get_num_cat_cols,
    lgbm_cv_train,

    make_submission,
    save_submission,
    subgroup_performance,
    # reduce_memory_usage,  # можно подключить по желанию
)


# =========================
# CONFIG
# =========================

# Имена колонок и файлов — ПЛЕЙСХОЛДЕРЫ, под задачу заменишь руками
ID_COL = "id"
TARGET_COL = "target"

LAT_COL = "lat"
LON_COL = "lon"

# если эмбеддинги уже в train/test и колоноки начинаются с "emb_"
EMB_PREFIX = "emb_"

TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"

# если эмбеддинги лежат отдельно — заготовка (ниже есть merge-функция)
TRAIN_EMB_CSV = None  # например "train_image_emb.csv"
TEST_EMB_CSV = None   # например "test_image_emb.csv"
EMB_ID_COL = "id"

TASK_TYPE = "reg"      # "reg" / "bin" / "multiclass"
METRIC_NAME = "rmse"   # "rmse", "auc", "f1", "logloss", "mae"
N_SPLITS = 5
SEED = 42


# =========================
# HELPER: merge эмбеддингов
# =========================

def merge_external_embeddings(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_emb_path: str,
    test_emb_path: str,
    id_col: str = ID_COL,
    emb_id_col: str = EMB_ID_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Если эмбеддинги лежат в отдельных csv — пример того, как их джоинить.
    Ожидает, что в emb-таблице есть уникальный emb_id_col.
    """
    emb_tr = pd.read_csv(train_emb_path)
    emb_te = pd.read_csv(test_emb_path)

    # sanity-чек
    assert emb_tr[emb_id_col].is_unique, "train эмбеддинги: id не уникален"
    assert emb_te[emb_id_col].is_unique, "test эмбеддинги: id не уникален"

    print("train emb shape:", emb_tr.shape)
    print("test  emb shape:", emb_te.shape)

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

    # можно удалить дублирующий id из эмбеддингов
    if emb_id_col != id_col:
        train = train.drop(columns=[emb_id_col])
        test = test.drop(columns=[emb_id_col])

    return train, test


# =========================
# FEATURE BUILDER
# =========================

def build_features(
    df: pd.DataFrame,
    is_train: bool = True,
) -> pd.DataFrame:
    """
    Единая функция построения фичей:
    - базовые geo-фичи
    - расстояние до центра
    - geo-кластеры
    - агрегаты по эмбеддингам
    - PCA по эмбеддингам

    Тут шаблонно всё включено; под конкретную задачу можно части отключать.
    """
    df = df.copy()

    # --- GEO ---
    if LAT_COL in df.columns and LON_COL in df.columns:
        df = add_geo_basic_features(df, LAT_COL, LON_COL)
        df = add_geo_center_distance(df, LAT_COL, LON_COL)
        # KMeans по координатам (можно уменьшить/увеличить число кластеров)
        df = add_geo_clusters(df, LAT_COL, LON_COL, n_clusters=20)

    # --- EMBEDDINGS ---
    emb_cols = get_embedding_cols(df, prefix=EMB_PREFIX)
    if emb_cols:
        # агрегаты: норма, среднее, максимум
        df = add_embedding_norms(df, emb_cols, prefix=EMB_PREFIX)

        # PCA-сжатие (например, до 32 измерений)
        df = pca_embeddings(
            df,
            emb_cols,
            n_components=32,
            prefix=f"{EMB_PREFIX}pca_",
        )

        # Если память критична — можно удалить сырые эмбеддинги:
        # df = df.drop(columns=emb_cols)

    # --- можно воткнуть ещё что-то кастомное ---
    # df = reduce_memory_usage(df)  # если нужно

    return df


# =========================
# MAIN PIPELINE
# =========================

def main():
    # 1. Сид
    set_seed(SEED)

    # 2. Загрузка базовых train/test
    train, test = load_train_test(
        TRAIN_CSV,
        TEST_CSV,
        id_col=ID_COL,
        target_col=TARGET_COL,
    )

    # Если эмбеддинги лежат отдельно — смёржить тут
    if TRAIN_EMB_CSV is not None and TEST_EMB_CSV is not None:
        train, test = merge_external_embeddings(
            train,
            test,
            train_emb_path=TRAIN_EMB_CSV,
            test_emb_path=TEST_EMB_CSV,
            id_col=ID_COL,
            emb_id_col=EMB_ID_COL,
        )

    # 3. Быстрый взгляд на таргет и train/test shift
    quick_target_info(train, TARGET_COL)
    compare_train_test(
        train,
        test,
        exclude_cols=[ID_COL, TARGET_COL, LAT_COL, LON_COL],
        max_cols=50,
    )

    # 4. Построение фичей
    train_fe = build_features(train, is_train=True)
    test_fe = build_features(test, is_train=False)

    # 5. Разделение на num/cat
    num_cols, cat_cols = get_num_cat_cols(
        train_fe,
        target_col=TARGET_COL,
        id_cols=[ID_COL],
    )
    feature_cols = num_cols + cat_cols

    print("num_cols:", len(num_cols))
    print("cat_cols:", len(cat_cols))
    print("total feature_cols:", len(feature_cols))

    # 6. Матрицы X, y, X_test
    X = train_fe[feature_cols]
    y = train_fe[TARGET_COL].values
    X_test = test_fe[feature_cols]

    # 7. CV + LightGBM (шаблон)
    oof, test_pred, models, oof_metric = lgbm_cv_train(
        X=X,
        y=y,
        X_test=X_test,
        params=None,          # можно оставить None → возьмутся дефолты из функции
        n_splits=N_SPLITS,
        task_type=TASK_TYPE,
        metric_name=METRIC_NAME,
        groups=None,          # при необходимости подставить group-col
        verbose=200,
    )

    print(f"Final OOF {METRIC_NAME}: {oof_metric:.6f}")

    # 8. Сабмит
    sub = make_submission(
        test_ids=test[ID_COL],
        preds=test_pred,
        id_col=ID_COL,
        target_col=TARGET_COL,
    )
    save_submission(sub, path="submission_lgbm_geo_emb.csv")

    # 9. Робастность по подгруппам (пример — гео-кластеры)
    cluster_col = "geo_cluster_20"
    if cluster_col in train_fe.columns:
        print("Subgroup performance by geo cluster:")
        subgroup_performance(
            y_true=y,
            y_pred=oof,
            group=train_fe[cluster_col],
            metric_name=METRIC_NAME,
            top_n=20,
        )


if __name__ == "__main__":
    main()
