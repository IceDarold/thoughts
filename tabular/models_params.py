# ====================================================
# ДОПОЛНИТЕЛЬНЫЕ ПРЕСЕТЫ ДЛЯ БЛЕНДИНГА
# ====================================================

EXTRA_LGBM_PARAM_PRESETS = {
    # --- REGRESSION ---

    # Очень быстрый, низко-Variance, высоко-Bias
    "lgbm_reg_tiny":
    {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "n_estimators": 800,
        "num_leaves": 31,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 2.0,
        "random_state": 41,
        "n_jobs": -1,
    },

    # Более "robust": маленькие листья, сильная регуляризация
    "lgbm_reg_robust":
    {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.04,
        "n_estimators": 5000,
        "num_leaves": 48,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 5.0,
        "random_state": 43,
        "n_jobs": -1,
    },

    # GOSS — другой тип бустинга, хорошо для блендинга
    "lgbm_reg_goss":
    {
        "boosting_type": "goss",
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "n_estimators": 4000,
        "num_leaves": 96,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,  # в goss bagging_fraction не используется
        "lambda_l1": 0.0,
        "lambda_l2": 2.0,
        "top_rate": 0.2,
        "other_rate": 0.1,
        "random_state": 44,
        "n_jobs": -1,
    },

    # DART — стохастический бустинг с дропаутом деревьев
    "lgbm_reg_dart":
    {
        "boosting_type": "dart",
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "n_estimators": 3000,
        "num_leaves": 64,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "drop_rate": 0.1,
        "skip_drop": 0.5,
        "random_state": 45,
        "n_jobs": -1,
    },

    # --- BINARY ---

    "lgbm_bin_tiny":
    {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.1,
        "n_estimators": 800,
        "num_leaves": 31,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 2.0,
        "random_state": 51,
        "n_jobs": -1,
    },

    "lgbm_bin_robust":
    {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.04,
        "n_estimators": 5000,
        "num_leaves": 48,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 5.0,
        "random_state": 53,
        "n_jobs": -1,
    },

    # Для несбалансированных классов – пример, scale_pos_weight потом можно подправить
    "lgbm_bin_imbalanced":
    {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.04,
        "n_estimators": 4000,
        "num_leaves": 64,
        "min_data_in_leaf": 60,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 2.0,
        "is_unbalance": True,
        "scale_pos_weight": 3.0,
        "random_state": 54,
        "n_jobs": -1,
    },
}

LGBM_PARAM_PRESETS.update(EXTRA_LGBM_PARAM_PRESETS)


EXTRA_XGB_PARAM_PRESETS = {
    # --- REGRESSION ---

    # Очень лёгкая модель для стэкинга (мало деревьев, высокий lr)
    "xgb_reg_tiny":
    {
        "booster": "gbtree",
        "tree_method": "hist",
        "learning_rate": 0.12,
        "max_depth": 5,
        "min_child_weight": 6.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "random_state": 61,
        "n_jobs": -1,
    },

    # Более широкий, чуть более «жёсткий» регрессор
    "xgb_reg_wide":
    {
        "booster": "gbtree",
        "tree_method": "hist",
        "learning_rate": 0.035,
        "max_depth": 10,
        "min_child_weight": 3.0,
        "subsample": 0.85,
        "colsample_bytree": 0.7,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "random_state": 62,
        "n_jobs": -1,
    },

    # Сильная L1 для sparse/шумных фич
    "xgb_reg_l1heavy":
    {
        "booster": "gbtree",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 7,
        "min_child_weight": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.5,
        "reg_alpha": 5.0,
        "random_state": 63,
        "n_jobs": -1,
    },

    # --- BINARY ---

    "xgb_bin_tiny":
    {
        "booster": "gbtree",
        "tree_method": "hist",
        "learning_rate": 0.12,
        "max_depth": 5,
        "min_child_weight": 6.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "random_state": 71,
        "n_jobs": -1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    },

    "xgb_bin_robust":
    {
        "booster": "gbtree",
        "tree_method": "hist",
        "learning_rate": 0.04,
        "max_depth": 8,
        "min_child_weight": 4.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "reg_alpha": 0.0,
        "random_state": 72,
        "n_jobs": -1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    },

    # Несбалансированный класс (пример – scale_pos_weight подстраиваешь под задачу)
    "xgb_bin_imbalanced":
    {
        "booster": "gbtree",
        "tree_method": "hist",
        "learning_rate": 0.04,
        "max_depth": 7,
        "min_child_weight": 4.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "random_state": 73,
        "n_jobs": -1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": 4.0,
    },
}

XGB_PARAM_PRESETS.update(EXTRA_XGB_PARAM_PRESETS)


EXTRA_CAT_PARAM_PRESETS = {
    # --- REGRESSION ---

    # Совсем лёгкий CatBoost (быстрый OOF, хороший как слабый лёрнер)
    "cat_reg_tiny":
    {
        "loss_function": "RMSE",
        "learning_rate": 0.15,
        "depth": 5,
        "l2_leaf_reg": 4.0,
        "bagging_temperature": 1.0,
        "random_state": 81,
        "task_type": "CPU",
        "thread_count": -1,
    },

    # Более глубокий, сильный регрессор
    "cat_reg_deep":
    {
        "loss_function": "RMSE",
        "learning_rate": 0.04,
        "depth": 10,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 0.5,
        "random_state": 82,
        "task_type": "CPU",
        "thread_count": -1,
        "grow_policy": "Depthwise",
    },

    # --- BINARY ---

    "cat_bin_tiny":
    {
        "loss_function": "Logloss",
        "learning_rate": 0.15,
        "depth": 5,
        "l2_leaf_reg": 4.0,
        "bagging_temperature": 1.0,
        "random_state": 91,
        "task_type": "CPU",
        "thread_count": -1,
    },

    "cat_bin_robust":
    {
        "loss_function": "Logloss",
        "learning_rate": 0.04,
        "depth": 8,
        "l2_leaf_reg": 5.0,
        "bagging_temperature": 0.5,
        "random_state": 92,
        "task_type": "CPU",
        "thread_count": -1,
    },

    # Для несбалансированных задач
    "cat_bin_imbalanced":
    {
        "loss_function": "Logloss",
        "learning_rate": 0.05,
        "depth": 7,
        "l2_leaf_reg": 4.0,
        "bagging_temperature": 0.5,
        "random_state": 93,
        "task_type": "CPU",
        "thread_count": -1,
        "auto_class_weights": "Balanced",
    },
}

CAT_PARAM_PRESETS.update(EXTRA_CAT_PARAM_PRESETS)


# Обновляем общий словарь
BOOSTING_PARAM_PRESETS.update({
    **{f"lgbm::{k}": v for k, v in EXTRA_LGBM_PARAM_PRESETS.items()},
    **{f"xgb::{k}": v for k, v in EXTRA_XGB_PARAM_PRESETS.items()},
    **{f"cat::{k}": v for k, v in EXTRA_CAT_PARAM_PRESETS.items()},
})
