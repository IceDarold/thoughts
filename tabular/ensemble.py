from typing import List, Dict, Optional, Tuple, Any, Union, Literal, Callable
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit,
)
from sklearn.linear_model import Ridge, LogisticRegression

TaskType = Literal["regression", "binary", "multiclass"]
MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _build_folds(
    train_df: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    task_type: TaskType,
    val_type: str = "skf",
    n_splits: int = 5,
    group_col: Optional[str] = None,
    time_col: Optional[str] = None,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Строим фолды под разные типы валидации:
      - 'skf': StratifiedKFold (для классификации) или KFold (для регрессии)
      - 'kfold': всегда KFold
      - 'group': GroupKFold по group_col
      - 'time': TimeSeriesSplit по порядку строк (или по time_col, если задан)
    """
    y_arr = np.asarray(y)
    val_type = val_type.lower()

    if val_type in ("skf", "stratified"):
        if task_type == "regression":
            kf = KFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            folds = list(kf.split(train_df))
        else:
            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            folds = list(skf.split(train_df, y_arr))
        return folds

    if val_type in ("kfold", "kf"):
        kf = KFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        return list(kf.split(train_df))

    if val_type == "group":
        if group_col is None:
            raise ValueError("group_col must be provided for group validation")
        groups = train_df[group_col].values
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(train_df, y_arr, groups=groups))

    if val_type == "time":
        # считаем, что строки уже отсортированы по времени
        # либо явно сортируем по time_col, если он есть
        if time_col is not None:
            order = np.argsort(train_df[time_col].values)
        else:
            order = np.arange(len(train_df))

        tscv = TimeSeriesSplit(n_splits=n_splits)
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        for tr_rel, val_rel in tscv.split(order):
            tr_idx = order[tr_rel]
            val_idx = order[val_rel]
            folds.append((tr_idx, val_idx))
        return folds

    raise ValueError(f"Unknown val_type: {val_type}")


def run_boosting_ensemble(
    train_df: pd.DataFrame,
    target_col: str,
    task_type: TaskType,
    val_type: str = "skf",
    n_splits: int = 5,
    group_col: Optional[str] = None,
    time_col: Optional[str] = None,
    cat_features: Optional[List[str]] = None,
    metric_fn: Optional[MetricFn] = None,
    test_df: Optional[pd.DataFrame] = None,
    model_keys: Optional[List[str]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Универсальный раннер:
      - строит фолды
      - обучает кучу бустингов из BOOSTING_PARAM_PRESETS
      - делает стеккинг (Ridge/LogReg) поверх OOF
      - возвращает OOF/TEST бленды и детали.

    Аргументы:
      train_df: DataFrame с train
      target_col: название таргета в train_df
      task_type: 'regression' / 'binary' / 'multiclass'
      val_type: 'skf' / 'kfold' / 'group' / 'time'
      group_col: колонка для group validation
      time_col: колонка для time validation (если нужно)
      cat_features: список категориальных фич (для LGBM/CatBoost)
      metric_fn: функция метрики metric_fn(y_true, y_pred)
      test_df: DataFrame test (если есть, тогда считаем предсказания и на test)
      model_keys: список ключей из BOOSTING_PARAM_PRESETS, если хочешь руками
      random_state: сид для фолдов

    Возвращает dict:
      {
        "cv_results": {model_key: CVResult, ...},
        "folds": список фолдов (train_idx, val_idx),
        "oof_blend": np.ndarray или None,
        "test_blend": np.ndarray или None,
        "used_models_for_blend": список model_keys,
        "blend_score": float или None,
        "per_model_scores": {model_key: score_mean}
      }
    """
    # --- разделяем X/y ---
    y = np.asarray(train_df[target_col].values)
    X = train_df.drop(columns=[target_col]).copy()

    if test_df is not None:
        # Обеспечиваем одинаковый набор и порядок колонок
        X_test = test_df[X.columns].copy()
    else:
        X_test = None

    # --- строим фолды ---
    folds = _build_folds(
        train_df=train_df,
        y=y,
        task_type=task_type,
        val_type=val_type,
        n_splits=n_splits,
        group_col=group_col,
        time_col=time_col,
        random_state=random_state,
    )

    # --- подбираем набор моделей ---
    global BOOSTING_PARAM_PRESETS  # предполагаем, что уже объявлен
    all_keys = list(BOOSTING_PARAM_PRESETS.keys())

    if model_keys is None:
        # Фильтруем по типу задачи
        if task_type == "regression":
            candidate_keys = [
                k for k in all_keys
                if "_reg_" in k  # примитивный фильтр по имени
            ]
        elif task_type == "binary":
            candidate_keys = [
                k for k in all_keys
                if "_bin_" in k
            ]
        else:
            # если когда-то заведёшь отдельные _mc_ пресеты – сюда
            candidate_keys = [
                k for k in all_keys
                if "multiclass" in k or "_mc_" in k
            ]

        # чуть ограничим количество моделей по умолчанию,
        # чтобы не запускать всё подряд
        # (при желании можешь руками передать model_keys)
        # возьмём до 8 штук
        candidate_keys = sorted(candidate_keys)
        model_keys = candidate_keys[:8]

    print("Will train models:")
    for k in model_keys:
        print("  ", k)

    cv_results: Dict[str, Any] = {}
    test_preds: Dict[str, np.ndarray] = {}
    per_model_scores: Dict[str, float] = {}

    # Для стеккинга
    oof_for_stacking: List[np.ndarray] = []
    test_for_stacking: List[np.ndarray] = []
    used_for_stacking: List[str] = []

    # --- обучение всех моделей ---
    for key in model_keys:
        model_type, preset_name = key.split("::", 1)
        params = BOOSTING_PARAM_PRESETS[key].copy()

        print(f"\n=== Training {key} ===")

        if model_type == "lgbm":
            res = train_lgbm_cv(
                X=X,
                y=y,
                folds=folds,
                task_type=task_type,
                params=params,
                metric_fn=metric_fn,
                categorical_features=cat_features,
            )
        elif model_type == "xgb":
            res = train_xgb_cv(
                X=X,
                y=y,
                folds=folds,
                task_type=task_type,
                params=params,
                metric_fn=metric_fn,
            )
        elif model_type == "cat":
            res = train_cat_cv(
                X=X,
                y=y,
                folds=folds,
                task_type=task_type,
                params=params,
                metric_fn=metric_fn,
                cat_features=cat_features,
            )
        else:
            print(f"Unknown model_type: {model_type}, skipping")
            continue

        cv_results[key] = res
        per_model_scores[key] = res.score_mean

        # собираем предсказания на тест (если есть)
        if X_test is not None:
            test_pred = predict_mean(
                models=res.models,
                X_test=X_test,
                task_type=task_type,
            )
            test_preds[key] = np.asarray(test_pred)

        # подготавливаем фичи для стеккинга (только regression/binary)
        if task_type in ("regression", "binary"):
            oof_pred = np.asarray(res.oof)
            if oof_pred.ndim == 1:
                oof_col = oof_pred.reshape(-1, 1)
            else:
                # На всякий случай: если вдруг многомерный output,
                # для бинарки берём последний столбец
                if task_type == "binary" and oof_pred.shape[1] >= 2:
                    oof_col = oof_pred[:, [-1]]
                else:
                    # для регрессии многомерный output не ждём -> скипаем из стеккинга
                    print(f"{key}: oof has unexpected shape {oof_pred.shape}, skip in stacking")
                    continue

            oof_for_stacking.append(oof_col)
            used_for_stacking.append(key)

            if X_test is not None and key in test_preds:
                test_pred = test_preds[key]
                test_pred = np.asarray(test_pred)
                if test_pred.ndim == 1:
                    test_col = test_pred.reshape(-1, 1)
                else:
                    if task_type == "binary" and test_pred.shape[1] >= 2:
                        test_col = test_pred[:, [-1]]
                    else:
                        print(f"{key}: test_pred has unexpected shape {test_pred.shape}, skip in stacking")
                        continue
                test_for_stacking.append(test_col)

    # --- блендинг: сначала пытаемся стеккинг, иначе простое среднее ---
    oof_blend: Optional[np.ndarray] = None
    test_blend: Optional[np.ndarray] = None
    blend_score: Optional[float] = None

    if task_type in ("regression", "binary") and len(oof_for_stacking) >= 2:
        print("\n=== Stacking blend ===")
        oof_stack = np.concatenate(oof_for_stacking, axis=1)  # (N, M)
        if X_test is not None and len(test_for_stacking) == len(oof_for_stacking):
            test_stack = np.concatenate(test_for_stacking, axis=1)  # (T, M)
        else:
            test_stack = None

        if task_type == "regression":
            meta = Ridge(alpha=1.0)
            meta.fit(oof_stack, y)
            oof_blend = meta.predict(oof_stack)
            if test_stack is not None:
                test_blend = meta.predict(test_stack)
        else:
            meta = LogisticRegression(C=1.0, max_iter=10_000)
            meta.fit(oof_stack, y)
            oof_proba = meta.predict_proba(oof_stack)[:, 1]
            oof_blend = oof_proba
            if test_stack is not None:
                test_proba = meta.predict_proba(test_stack)[:, 1]
                test_blend = test_proba

        if metric_fn is not None:
            blend_score = float(metric_fn(y, oof_blend))
            print(f"Blend score (stacking): {blend_score:.6f}")

    else:
        print("\n=== Simple average blend (no stacking) ===")
        # fallback: среднее всех доступных моделей
        # (подходит и для multiclass, и для reg/binary)
        if len(cv_results) > 0:
            # OOF
            oof_arrs = []
            for k, res in cv_results.items():
                arr = np.asarray(res.oof)
                oof_arrs.append(arr)
            oof_stack = np.stack(oof_arrs, axis=0)
            oof_blend = oof_stack.mean(axis=0)

            # TEST
            if X_test is not None and len(test_preds) > 0:
                test_arrs = []
                for k, p in test_preds.items():
                    test_arrs.append(np.asarray(p))
                test_stack = np.stack(test_arrs, axis=0)
                test_blend = test_stack.mean(axis=0)

            if metric_fn is not None and task_type in ("regression", "binary"):
                blend_score = float(metric_fn(y, oof_blend))
                print(f"Blend score (avg): {blend_score:.6f}")

        used_for_stacking = list(cv_results.keys())

    return {
        "cv_results": cv_results,
        "folds": folds,
        "oof_blend": oof_blend,
        "test_blend": test_blend,
        "used_models_for_blend": used_for_stacking,
        "blend_score": blend_score,
        "per_model_scores": per_model_scores,
    }
