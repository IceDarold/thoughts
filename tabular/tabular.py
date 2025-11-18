"""
toolbox.py — набор утилит для табличных задач с geo и эмбеддингами.

Ничего из этого не обязанo работать «из коробки» — это шаблоны,
которые ты можешь копировать, править и подгонять под конкретный тур.
"""

import os
import gc
import time
import math
import random
from contextlib import contextmanager
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    f1_score,
    log_loss,
)
from sklearn.decomposition import PCA

# lightgbm / xgboost / catboost могут быть не установлены,
# так что оставляю импорт внутри функций как пример.


# ==========
# Общие утилиты
# ==========

def set_seed(seed: int = 42):
    """Фиксируем все сиды, чтобы хоть как-то контролировать рандом."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


@contextmanager
def timer(name: str):
    """Простой таймер-контекст: with timer("fit lgbm"): ..."""
    start = time.time()
    print(f"[{name}] start")
    yield
    end = time.time()
    print(f"[{name}] done in {end - start:.2f} s")


def memory_usage(df: pd.DataFrame, name: str = "df"):
    """Грубая оценка памяти датафрейма."""
    mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"{name} memory usage: {mb:.2f} MB")
    return mb


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Агрессивный даункаст числовых колонок.
    Иногда даёт артефакты — использовать аккуратно.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type).startswith("int"):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= 0:
                if c_max < 255:
                    df[col] = df[col].astype("uint8")
                elif c_max < 65535:
                    df[col] = df[col].astype("uint16")
                elif c_max < 4294967295:
                    df[col] = df[col].astype("uint32")
                else:
                    df[col] = df[col].astype("uint64")
            else:
                if np.iinfo("int8").min < c_min < np.iinfo("int8").max:
                    df[col] = df[col].astype("int8")
                elif np.iinfo("int16").min < c_min < np.iinfo("int16").max:
                    df[col] = df[col].astype("int16")
                elif np.iinfo("int32").min < c_min < np.iinfo("int32").max:
                    df[col] = df[col].astype("int32")
                else:
                    df[col] = df[col].astype("int64")
        elif str(col_type).startswith("float"):
            df[col] = df[col].astype("float32")
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        print(f"Mem usage {start_mem:.2f} -> {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df


# ==========
# Загрузка и базовый анализ данных
# ==========

def load_train_test(
    train_path: str,
    test_path: str,
    id_col: str,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Быстрая загрузка train/test с базовыми проверками."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    assert id_col in train.columns, f"{id_col} не найден в train"
    assert id_col in test.columns, f"{id_col} не найден в test"

    if target_col is not None:
        assert target_col in train.columns, f"{target_col} не найден в train"

    print("train shape:", train.shape)
    print("test  shape:", test.shape)
    memory_usage(train, "train")
    memory_usage(test, "test")
    return train, test


def quick_target_info(train: pd.DataFrame, target_col: str):
    """Быстрая сводка по таргету."""
    y = train[target_col]
    print("Target dtype:", y.dtype)
    print(y.describe())

    if y.nunique() < 20:
        print("Value counts:")
        print(y.value_counts(normalize=True))


def compare_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    max_cols: int = 50,
):
    """
    Сравнение распределений train / test по числовым фичам,
    чтобы поймать сильный сдвиг.
    """
    if exclude_cols is None:
        exclude_cols = []

    num_cols = [
        c for c in train.columns
        if train[c].dtype != "O" and c not in exclude_cols
    ]
    num_cols = num_cols[:max_cols]

    rows = []
    for col in num_cols:
        tr = train[col].dropna()
        te = test[col].dropna()
        rows.append(
            {
                "feature": col,
                "train_mean": tr.mean(),
                "test_mean": te.mean(),
                "train_std": tr.std(),
                "test_std": te.std(),
                "train_min": tr.min(),
                "test_min": te.min(),
                "train_max": tr.max(),
                "test_max": te.max(),
                "train_n": tr.shape[0],
                "test_n": te.shape[0],
            }
        )
    diff_df = pd.DataFrame(rows)
    diff_df["mean_ratio"] = diff_df["test_mean"] / (diff_df["train_mean"] + 1e-6)
    diff_df = diff_df.sort_values("mean_ratio")
    print(diff_df.head(20))
    print(diff_df.tail(20))
    return diff_df


# ==========
# Работа с geo-фичами
# ==========

def haversine_distance(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
    radius: float = 6371.0,
) -> Union[float, np.ndarray]:
    """
    Расстояние по сфере между двумя точками (в километрах по умолчанию).
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return radius * c


def add_geo_basic_features(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    prefix: str = "geo_",
) -> pd.DataFrame:
    """
    Простейшие geo-фичи: нормализованные координаты + "полушария".
    """
    df = df.copy()
    df[f"{prefix}lat_norm"] = (df[lat_col] - df[lat_col].mean()) / (df[lat_col].std() + 1e-6)
    df[f"{prefix}lon_norm"] = (df[lon_col] - df[lon_col].mean()) / (df[lon_col].std() + 1e-6)
    df[f"{prefix}hemisphere_ns"] = (df[lat_col] > 0).astype("int8")
    df[f"{prefix}hemisphere_ew"] = (df[lon_col] > 0).astype("int8")
    return df


def add_geo_center_distance(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    col_name: str = "geo_center_dist_km",
) -> pd.DataFrame:
    """
    Фича: расстояние до некого "центра" (средняя точка, столица и т.п.).
    """
    df = df.copy()
    if center_lat is None:
        center_lat = df[lat_col].mean()
    if center_lon is None:
        center_lon = df[lon_col].mean()

    df[col_name] = haversine_distance(
        df[lat_col].values,
        df[lon_col].values,
        center_lat,
        center_lon,
    )
    return df


def add_geo_clusters(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    n_clusters: int = 20,
    prefix: str = "geo_cluster_",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Кластеризация по координатам (KMeans).
    Требует sklearn, но можно взять за шаблон.
    """
    from sklearn.cluster import KMeans

    df = df.copy()
    coords = df[[lat_col, lon_col]].values
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    df[f"{prefix}{n_clusters}"] = km.fit_predict(coords).astype("int32")
    return df


# ==========
# Работа с эмбеддингами
# ==========

def get_embedding_cols(df: pd.DataFrame, prefix: str = "emb_") -> List[str]:
    """Находит столбцы эмбеддингов по префиксу."""
    return [c for c in df.columns if c.startswith(prefix)]


def add_embedding_norms(
    df: pd.DataFrame,
    emb_cols: List[str],
    prefix: str = "emb_",
) -> pd.DataFrame:
    """
    Добавляет агрегаты по эмбеддингам: L2-норму, среднее, максимум.
    """
    df = df.copy()
    emb = df[emb_cols].values.astype("float32")
    norms = np.linalg.norm(emb, axis=1)
    df[f"{prefix}l2"] = norms
    df[f"{prefix}mean"] = emb.mean(axis=1)
    df[f"{prefix}max"] = emb.max(axis=1)
    return df


def pca_embeddings(
    df: pd.DataFrame,
    emb_cols: List[str],
    n_components: int = 32,
    prefix: str = "emb_pca_",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    PCA-сжатие эмбеддингов: превращаем, например, 512-мерный в 32-мерный.
    """
    df = df.copy()
    emb = df[emb_cols].values.astype("float32")
    pca = PCA(n_components=n_components, random_state=random_state)
    emb_pca = pca.fit_transform(emb)
    for i in range(n_components):
        df[f"{prefix}{i}"] = emb_pca[:, i]
    return df


# ==========
# Фичи: табличные, категории, target encoding
# ==========

def get_num_cat_cols(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    id_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Грубое разбиение фич: числовые / категориальные."""
    if id_cols is None:
        id_cols = []
    exclude = set(id_cols)
    if target_col is not None:
        exclude.add(target_col)

    num_cols, cat_cols = [], []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


def freq_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    min_count: int = 1,
    suffix: str = "_freq",
):
    """
    Частотное кодирование категориального признака.
    """
    vc = train[col].value_counts()
    vc = vc[vc >= min_count]
    train[col + suffix] = train[col].map(vc).fillna(0).astype("float32")
    test[col + suffix] = test[col].map(vc).fillna(0).astype("float32")
    return train, test


def target_encoding_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    target_col: str,
    cv,
    prior_weight: float = 100.0,
    suffix: str = "_te",
):
    """
    Пример target encoding с CV, без утечки.
    prior_weight задаёт силу глобального среднего.
    """
    train = train.copy()
    test = test.copy()

    global_mean = train[target_col].mean()
    train[f"{col}{suffix}"] = 0.0
    test_te_values = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(train, train[target_col])):
        tr = train.iloc[tr_idx]
        val = train.iloc[val_idx]
        stats = tr.groupby(col)[target_col].agg(["mean", "count"])
        # сглаживание к глобальному среднему
        stats["te"] = (stats["mean"] * stats["count"] + global_mean * prior_weight) / (
            stats["count"] + prior_weight
        )
        train.loc[val_idx, f"{col}{suffix}"] = val[col].map(stats["te"]).fillna(global_mean)

        # для test считаем по full-data stats
        if fold == 0:
            full_stats = train.groupby(col)[target_col].agg(["mean", "count"])
            full_stats["te"] = (full_stats["mean"] * full_stats["count"] + global_mean * prior_weight) / (
                full_stats["count"] + prior_weight
            )
            test_te = test[col].map(full_stats["te"]).fillna(global_mean)
            test_te_values = test_te.values

    test[f"{col}{suffix}"] = test_te_values
    return train, test


# ==========
# CV-сплиттеры и метрики
# ==========

def get_cv(
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    task_type: str = "reg",
    y: Optional[pd.Series] = None,
    groups: Optional[pd.Series] = None,
):
    """
    Возвращает подходящий сплиттер:
    - 'reg'   -> KFold
    - 'bin'   -> StratifiedKFold
    - 'multiclass' -> StratifiedKFold
    - 'group' -> GroupKFold
    """
    if groups is not None:
        print("Using GroupKFold")
        return GroupKFold(n_splits=n_splits)

    if task_type in ["bin", "multiclass"] and y is not None:
        print("Using StratifiedKFold")
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    print("Using plain KFold")
    return KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )


def get_metric_fn(metric_name: str):
    """
    Возвращает функцию метрики по имени.
    Это шаблон: при желании можно расширять.
    """
    metric_name = metric_name.lower()

    def rmse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

    if metric_name == "rmse":
        return rmse
    if metric_name == "mae":
        return mean_absolute_error
    if metric_name == "auc":
        return roc_auc_score
    if metric_name == "f1":
        return lambda yt, yp: f1_score(yt, (yp > 0.5).astype(int))
    if metric_name == "logloss":
        return log_loss

    # fallback
    return rmse


# ==========
# CV-обучение моделей (LightGBM / CatBoost как шаблон)
# ==========

def lgbm_cv_train(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    X_test: Optional[pd.DataFrame] = None,
    params: Optional[Dict] = None,
    n_splits: int = 5,
    task_type: str = "reg",
    metric_name: str = "rmse",
    groups: Optional[pd.Series] = None,
    verbose: int = 100,
):
    """
    Шаблон CV-обучения LightGBM.
    Не обязательно будет работать «как есть» — это заготовка.
    """
    import lightgbm as lgb

    if params is None:
        if task_type == "reg":
            params = dict(
                objective="regression",
                metric="rmse",
                learning_rate=0.05,
                num_leaves=64,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=1,
                seed=42,
                n_jobs=-1,
            )
        elif task_type == "bin":
            params = dict(
                objective="binary",
                metric="auc",
                learning_rate=0.05,
                num_leaves=64,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=1,
                seed=42,
                n_jobs=-1,
            )
        else:
            params = dict(
                objective="multiclass",
                metric="multi_logloss",
                learning_rate=0.05,
                num_leaves=64,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=1,
                seed=42,
                n_jobs=-1,
            )

    cv = get_cv(
        n_splits=n_splits,
        task_type=task_type,
        y=y,
        groups=groups,
    )
    metric_fn = get_metric_fn(metric_name)

    oof = np.zeros(len(X), dtype="float32")
    test_pred = np.zeros(len(X_test), dtype="float32") if X_test is not None else None

    models = []
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        print(f"Fold {fold+1}/{n_splits}")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        train_set = lgb.Dataset(X_tr, label=y_tr)
        valid_set = lgb.Dataset(X_val, label=y_val)

        with timer(f"lgbm fold {fold+1}"):
            model = lgb.train(
                params,
                train_set,
                num_boost_round=10000,
                valid_sets=[train_set, valid_set],
                valid_names=["train", "valid"],
                early_stopping_rounds=200,
                verbose_eval=verbose,
            )

        y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
        oof[val_idx] = y_pred_val

        if X_test is not None:
            test_pred += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

        fold_metric = metric_fn(y_val, y_pred_val)
        print(f"Fold {fold+1} {metric_name}: {fold_metric:.5f}")
        models.append(model)
        gc.collect()

    oof_metric = metric_fn(y, oof)
    print(f"OOF {metric_name}: {oof_metric:.5f}")
    return oof, test_pred, models, oof_metric


def catboost_cv_train(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    X_test: Optional[pd.DataFrame] = None,
    cat_features: Optional[List[str]] = None,
    params: Optional[Dict] = None,
    n_splits: int = 5,
    task_type: str = "reg",
    metric_name: str = "rmse",
    groups: Optional[pd.Series] = None,
    verbose: int = 200,
):
    """
    Шаблон CV-обучения CatBoost.
    """
    from catboost import CatBoostRegressor, CatBoostClassifier, Pool

    if params is None:
        if task_type == "reg":
            params = dict(
                loss_function="RMSE",
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=42,
                verbose=False,
            )
            ModelCls = CatBoostRegressor
        else:
            params = dict(
                loss_function="Logloss",
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=42,
                verbose=False,
            )
            ModelCls = CatBoostClassifier

    cv = get_cv(
        n_splits=n_splits,
        task_type=task_type,
        y=y,
        groups=groups,
    )
    metric_fn = get_metric_fn(metric_name)

    oof = np.zeros(len(X), dtype="float32")
    test_pred = np.zeros(len(X_test), dtype="float32") if X_test is not None else None

    if cat_features is None:
        cat_features = [
            i for i, c in enumerate(X.columns)
            if str(X[c].dtype) == "object" or "category" in str(X[c].dtype)
        ]

    models = []
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        print(f"Fold {fold+1}/{n_splits}")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        valid_pool = Pool(X_val, y_val, cat_features=cat_features)
        model = ModelCls(**params)

        with timer(f"catboost fold {fold+1}"):
            model.fit(
                train_pool,
                eval_set=valid_pool,
                verbose=verbose,
            )

        y_val_pred = model.predict(valid_pool).reshape(-1)
        oof[val_idx] = y_val_pred

        if X_test is not None:
            test_pool = Pool(X_test, cat_features=cat_features)
            test_pred += model.predict(test_pool).reshape(-1) / n_splits

        fold_metric = metric_fn(y_val, y_val_pred)
        print(f"Fold {fold+1} {metric_name}: {fold_metric:.5f}")
        models.append(model)
        gc.collect()

    oof_metric = metric_fn(y, oof)
    print(f"OOF {metric_name}: {oof_metric:.5f}")
    return oof, test_pred, models, oof_metric


# ==========
# Блендинг и стэкинг
# ==========

def blend_equal_weight(preds: List[np.ndarray]) -> np.ndarray:
    """
    Простое усреднение нескольких предсказаний (одинаковые веса).
    """
    stacked = np.vstack(preds)
    return stacked.mean(axis=0)


def blend_search_weight(
    y_true: np.ndarray,
    preds: List[np.ndarray],
    metric_name: str = "rmse",
    grid: Optional[List[float]] = None,
):
    """
    Тупой перебор веса двух моделей w и (1-w).
    Можно расширить до более сложного поиска.
    """
    metric_fn = get_metric_fn(metric_name)
    if grid is None:
        grid = [i / 10 for i in range(0, 11)]

    best_score = None
    best_w = None
    p1, p2 = preds
    for w in grid:
        blend = w * p1 + (1 - w) * p2
        score = metric_fn(y_true, blend)
        print(f"w={w:.2f}: score={score:.5f}")
        if best_score is None or score < best_score:
            best_score = score
            best_w = w
    print(f"Best w={best_w:.2f} score={best_score:.5f}")
    return best_w, best_score


# ==========
# Сабмиты и служебки
# ==========

def make_submission(
    test_ids: Union[pd.Series, np.ndarray],
    preds: Union[pd.Series, np.ndarray],
    id_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Собираем датафрейм сабмита."""
    sub = pd.DataFrame({id_col: test_ids, target_col: preds})
    return sub


def save_submission(
    df_sub: pd.DataFrame,
    path: str = "submission.csv",
):
    """Сохраняем сабмит и выводим небольшую сводку."""
    df_sub.to_csv(path, index=False)
    print(f"Saved submission to {path}")
    print(df_sub.head())


# ==========
# Диагностика по подгруппам (например, по гео)
# ==========

def subgroup_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: Union[pd.Series, np.ndarray],
    metric_name: str = "rmse",
    top_n: int = 20,
):
    """
    Робастность по подгруппам: меряем метрику внутри каждой группы.
    """
    metric_fn = get_metric_fn(metric_name)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": group})
    rows = []
    for g, part in df.groupby("group"):
        if len(part) < 10:
            continue
        score = metric_fn(part["y_true"], part["y_pred"])
        rows.append({"group": g, "n": len(part), "score": score})
    res = pd.DataFrame(rows).sort_values("score")
    print(res.head(top_n))
    print(res.tail(top_n))
    return res
