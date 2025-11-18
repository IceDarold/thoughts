"""
boosting_utils.py

Набор утилит для:
- CV-обучения бустингов (LightGBM / XGBoost / CatBoost)
- простого блендинга предсказаний (average / rank / logit / stacking)
- хранения пресетов параметров для разных моделей под ансамбли
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Literal, Tuple, Union, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import Ridge, LogisticRegression

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier, Pool


TaskType = Literal["regression", "binary", "multiclass"]
MetricFn = Callable[[np.ndarray, np.ndarray], float]


# ===============================
#  Общие структуры и утилиты
# ===============================

@dataclass
class CVResult:
    models: List[Any]
    oof: np.ndarray
    scores: List[float]
    score_mean: float


def make_folds(
    y: Union[pd.Series, np.ndarray],
    n_splits: int = 5,
    task_type: TaskType = "regression",
    random_state: int = 42,
    shuffle: bool = True,
):
    """
    Возвращает список (train_idx, valid_idx) для CV.
    """
    y_arr = np.asarray(y)
    if task_type == "regression":
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        splits = list(kf.split(y_arr))
    else:
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
        splits = list(skf.split(np.zeros_like(y_arr), y_arr))
    return splits


# ===============================
#  LightGBM
# ===============================

def train_lgbm_cv(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    task_type: TaskType = "regression",
    params: Optional[Dict] = None,
    metric_fn: Optional[MetricFn] = None,
    num_boost_round: int = 10_000,
    early_stopping_rounds: int = 200,
    verbose_eval: int = 200,
    categorical_features: Optional[List[str]] = None,
    predict_proba_for_classification: bool = True,
) -> CVResult:
    """
    CV-обучение LightGBM.

    X, y       — полный train
    folds      — список (train_idx, valid_idx) из make_folds(...)
    metric_fn  — функция метрики: metric_fn(y_true, y_pred) -> float
                 (для классификации сюда можно передавать вероятности)

    Для классификации по умолчанию возвращает probabilities.
    """
    X_np = X
    y_np = np.asarray(y)

    if params is None:
        params = {}

    models = []
    oof = np.zeros((len(X),) if task_type != "multiclass" else (len(X), len(np.unique(y_np))))
    scores = []

    cat_idx = None
    if categorical_features is not None:
        cat_idx = [X.columns.get_loc(col) for col in categorical_features]

    for fold_i, (trn_idx, val_idx) in enumerate(folds):
        print(f"LightGBM | Fold {fold_i + 1}/{len(folds)}")

        X_trn, X_val = X_np.iloc[trn_idx], X_np.iloc[val_idx]
        y_trn, y_val = y_np[trn_idx], y_np[val_idx]

        if task_type == "regression":
            model = lgb.LGBMRegressor(**params)
        elif task_type == "binary":
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMClassifier(**params)

        fit_params = dict(
            X=X_trn,
            y=y_trn,
            eval_set=[(X_val, y_val)],
        )
        if categorical_features is not None:
            fit_params["categorical_feature"] = cat_idx

        model.fit(
            **fit_params,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=bool(verbose_eval)),
                lgb.log_evaluation(verbose_eval),
            ],
        )

        if task_type == "regression":
            val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
        else:
            if predict_proba_for_classification and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_val, num_iteration=model.best_iteration_)
                if task_type == "binary":
                    val_pred = proba[:, 1]
                else:
                    val_pred = proba
            else:
                val_pred = model.predict(X_val, num_iteration=model.best_iteration_)

        oof[val_idx] = val_pred

        if metric_fn is not None:
            score = metric_fn(y_val, val_pred)
            scores.append(score)
            print(f"Fold {fold_i + 1} score: {score:.6f}")

        models.append(model)

    score_mean = float(np.mean(scores)) if scores else float("nan")
    print(f"LightGBM CV mean score: {score_mean:.6f}")
    return CVResult(models=models, oof=oof, scores=scores, score_mean=score_mean)


# ===============================
#  XGBoost
# ===============================

def train_xgb_cv(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    task_type: TaskType = "regression",
    params: Optional[Dict] = None,
    metric_fn: Optional[MetricFn] = None,
    num_boost_round: int = 10_000,
    early_stopping_rounds: int = 200,
    verbose_eval: int = 200,
    predict_proba_for_classification: bool = True,
) -> CVResult:
    """
    CV-обучение XGBoost (sklearn API).
    """
    X_np = X
    y_np = np.asarray(y)

    if params is None:
        params = {}

    models = []
    oof = np.zeros((len(X),) if task_type != "multiclass" else (len(X), len(np.unique(y_np))))
    scores = []

    for fold_i, (trn_idx, val_idx) in enumerate(folds):
        print(f"XGBoost | Fold {fold_i + 1}/{len(folds)}")

        X_trn, X_val = X_np.iloc[trn_idx], X_np.iloc[val_idx]
        y_trn, y_val = y_np[trn_idx], y_np[val_idx]

        if task_type == "regression":
            model = xgb.XGBRegressor(
                **params,
                n_estimators=num_boost_round,
            )
        elif task_type == "binary":
            model = xgb.XGBClassifier(
                **params,
                n_estimators=num_boost_round,
            )
        else:
            model = xgb.XGBClassifier(
                **params,
                n_estimators=num_boost_round,
            )

        model.fit(
            X_trn,
            y_trn,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=bool(verbose_eval),
        )

        best_ntree = model.best_ntree_limit if hasattr(model, "best_ntree_limit") else None

        if task_type == "regression":
            val_pred = model.predict(X_val, ntree_limit=best_ntree)
        else:
            if predict_proba_for_classification and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_val, ntree_limit=best_ntree)
                if task_type == "binary":
                    val_pred = proba[:, 1]
                else:
                    val_pred = proba
            else:
                val_pred = model.predict(X_val, ntree_limit=best_ntree)

        oof[val_idx] = val_pred

        if metric_fn is not None:
            score = metric_fn(y_val, val_pred)
            scores.append(score)
            print(f"Fold {fold_i + 1} score: {score:.6f}")

        models.append(model)

    score_mean = float(np.mean(scores)) if scores else float("nan")
    print(f"XGBoost CV mean score: {score_mean:.6f}")
    return CVResult(models=models, oof=oof, scores=scores, score_mean=score_mean)


# ===============================
#  CatBoost
# ===============================

def train_cat_cv(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    task_type: TaskType = "regression",
    params: Optional[Dict] = None,
    metric_fn: Optional[MetricFn] = None,
    num_boost_round: int = 10_000,
    early_stopping_rounds: int = 200,
    verbose_eval: int = 200,
    cat_features: Optional[List[str]] = None,
    predict_proba_for_classification: bool = True,
) -> CVResult:
    """
    CV-обучение CatBoost.

    Важно: CatBoost любит работать с Pool, сюда можно передавать cat_features.
    """
    X_np = X
    y_np = np.asarray(y)

    if params is None:
        params = {}

    models = []
    oof = np.zeros((len(X),) if task_type != "multiclass" else (len(X), len(np.unique(y_np))))
    scores = []

    cat_idx = None
    if cat_features is not None:
        cat_idx = [X.columns.get_loc(col) for col in cat_features]

    for fold_i, (trn_idx, val_idx) in enumerate(folds):
        print(f"CatBoost | Fold {fold_i + 1}/{len(folds)}")

        X_trn, X_val = X_np.iloc[trn_idx], X_np.iloc[val_idx]
        y_trn, y_val = y_np[trn_idx], y_np[val_idx]

        train_pool = Pool(X_trn, y_trn, cat_features=cat_idx)
        val_pool = Pool(X_val, y_val, cat_features=cat_idx)

        if task_type == "regression":
            model = CatBoostRegressor(
                **params,
                iterations=num_boost_round,
                od_type="Iter",
                od_wait=early_stopping_rounds,
                verbose=verbose_eval,
            )
        else:
            model = CatBoostClassifier(
                **params,
                iterations=num_boost_round,
                od_type="Iter",
                od_wait=early_stopping_rounds,
                verbose=verbose_eval,
            )

        model.fit(train_pool, eval_set=val_pool)

        if task_type == "regression":
            val_pred = model.predict(val_pool)
        else:
            if predict_proba_for_classification:
                proba = model.predict_proba(val_pool)
                if task_type == "binary":
                    val_pred = proba[:, 1]
                else:
                    val_pred = proba
            else:
                val_pred = model.predict(val_pool)

        oof[val_idx] = val_pred

        if metric_fn is not None:
            score = metric_fn(y_val, val_pred)
            scores.append(score)
            print(f"Fold {fold_i + 1} score: {score:.6f}")

        models.append(model)

    score_mean = float(np.mean(scores)) if scores else float("nan")
    print(f"CatBoost CV mean score: {score_mean:.6f}")
    return CVResult(models=models, oof=oof, scores=scores, score_mean=score_mean)


# ===============================
#  Предсказание на тесте
# ===============================

def predict_mean(models: List[Any], X_test: pd.DataFrame, task_type: TaskType = "regression",
                 predict_proba_for_classification: bool = True) -> np.ndarray:
    """
    Прогоняет список моделей по тесту и усредняет предсказания.
    """
    preds = []
    for model in models:
        if task_type == "regression":
            p = model.predict(X_test)
        else:
            if predict_proba_for_classification and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if task_type == "binary":
                    p = proba[:, 1]
                else:
                    p = proba
            else:
                p = model.predict(X_test)
        preds.append(np.asarray(p))

    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0)


# ===============================
#  Блендинг предсказаний
# ===============================

def average_blend(
    preds_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Простейший бленд: взвешенное среднее нескольких предсказаний.
    Подходит и для (N,) и для (N, C).
    """
    arrs = [np.asarray(p) for p in preds_list]
    stacked = np.stack(arrs, axis=0)  # (M, N) или (M, N, C)

    if weights is None:
        weights = np.ones(len(arrs), dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    # tensordot удобен и для 2D/3D
    blended = np.tensordot(weights, stacked, axes=(0, 0))
    return blended


def rank_average_blend(
    preds_list: List[np.ndarray],
) -> np.ndarray:
    """
    Rank-averaging бленд (для регрессии / бинарки):
    - каждое предсказание переводим в ранги
    - ранги усредняем
    - можно потом нормировать обратно, если нужно
    """
    arrs = [np.asarray(p).ravel() for p in preds_list]
    n = arrs[0].shape[0]
    ranks = []
    for p in arrs:
        order = p.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(n, dtype=float)
        ranks.append(r)

    ranks = np.stack(ranks, axis=0)
    mean_ranks = ranks.mean(axis=0)
    return mean_ranks  # при необходимости можно ещё нормировать в [0,1]


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logit_blend_binary(
    preds_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Блендинг для бинарной классификации:
    - переводим вероятности через logit
    - делаем взвешенное среднее в logit-пространстве
    - возвращаем сигмоиду
    """
    arrs = [np.asarray(p).ravel() for p in preds_list]

    if weights is None:
        weights = np.ones(len(arrs), dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    logits = np.stack([_logit(p) for p in arrs], axis=0)
    z = np.tensordot(weights, logits, axes=(0, 0))
    return _sigmoid(z)


def stacking_blend_regression(
    oof_preds: np.ndarray,
    y: Union[pd.Series, np.ndarray],
    test_preds: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, Ridge]:
    """
    Stacking для регрессии:
    - oof_preds: (N, M) — OOF предсказания M моделей
    - y: (N,)
    - test_preds: (T, M) — предсказания M моделей на тесте
    """
    oof_preds = np.asarray(oof_preds)
    test_preds = np.asarray(test_preds)
    y_arr = np.asarray(y)

    meta = Ridge(alpha=alpha)
    meta.fit(oof_preds, y_arr)
    blended_test = meta.predict(test_preds)
    return blended_test, meta


def stacking_blend_binary(
    oof_preds: np.ndarray,
    y: Union[pd.Series, np.ndarray],
    test_preds: np.ndarray,
    C: float = 1.0,
) -> Tuple[np.ndarray, LogisticRegression]:
    """
    Stacking для бинарной классификации:
    - oof_preds / test_preds — вероятности (N, M) / (T, M).
    """
    oof_preds = np.asarray(oof_preds)
    test_preds = np.asarray(test_preds)
    y_arr = np.asarray(y)

    meta = LogisticRegression(C=C, max_iter=10_000)
    meta.fit(oof_preds, y_arr)
    blended_test = meta.predict_proba(test_preds)[:, 1]
    return blended_test, meta


# ===============================
#  Пресеты параметров моделей
# ===============================

# LightGBM пресеты
LGBM_PARAM_PRESETS: Dict[str, Dict] = {
    # Быстрый регрессор, часто как baseline
    "lgbm_reg_fast": {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "num_leaves": 64,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    },
    # Более тяжёлый регрессор
    "lgbm_reg_strong": {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "n_estimators": 6000,
        "num_leaves": 128,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 2.0,
        "random_state": 42,
        "n_jobs": -1,
    },
    # Бинарная классификация (fast)
    "lgbm_bin_fast": {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "num_leaves": 64,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    },
    # Бинарная классификация (поусилее)
    "lgbm_bin_strong": {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "n_estimators": 6000,
        "num_leaves": 128,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 2.0,
        "random_state": 42,
        "n_jobs": -1,
    },
}

# XGBoost пресеты
XGB_PARAM_PRESETS: Dict[str, Dict] = {
    "xgb_reg_fast": {
        "booster": "gbtree",
        "tree_method": "hist",  # если можно, поменять на "gpu_hist"
        "learning_rate": 0.05,
        "max_depth": 7,
        "min_child_weight": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "random_state": 42,
        "n_jobs": -1,
    },
    "xgb_reg_strong": {
        "booster": "gbtree",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 9,
        "min_child_weight": 3.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "random_state": 42,
        "n_jobs": -1,
    },
    "xgb_bin_fast": {
        "booster": "gbtree",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "random_state": 42,
        "n_jobs": -1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    },
    "xgb_bin_strong": {
        "booster": "gbtree",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 3.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "random_state": 42,
        "n_jobs": -1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    },
}

# CatBoost пресеты
CAT_PARAM_PRESETS: Dict[str, Dict] = {
    "cat_reg_fast": {
        "loss_function": "RMSE",
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 1.0,
        "random_state": 42,
        "task_type": "CPU",  # при необходимости CPU/GPU
        "thread_count": -1,
    },
    "cat_reg_strong": {
        "loss_function": "RMSE",
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 1.0,
        "random_state": 42,
        "task_type": "CPU",
        "thread_count": -1,
    },
    "cat_bin_fast": {
        "loss_function": "Logloss",
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 1.0,
        "random_state": 42,
        "task_type": "CPU",
        "thread_count": -1,
    },
    "cat_bin_strong": {
        "loss_function": "Logloss",
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 1.0,
        "random_state": 42,
        "task_type": "CPU",
        "thread_count": -1,
    },
}


# Для удобства — всё в одном месте
BOOSTING_PARAM_PRESETS = {
    **{f"lgbm::{k}": v for k, v in LGBM_PARAM_PRESETS.items()},
    **{f"xgb::{k}": v for k, v in XGB_PARAM_PRESETS.items()},
    **{f"cat::{k}": v for k, v in CAT_PARAM_PRESETS.items()},
}
