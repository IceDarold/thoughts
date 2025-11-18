def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

res = run_boosting_ensemble(
    train_df=train,
    target_col="target",
    task_type="regression",
    val_type="kfold",
    n_splits=5,
    group_col=None,
    time_col=None,
    cat_features=["cat_col1", "cat_col2"],  # или None
    metric_fn=rmse,
    test_df=test,
)

submission = pd.DataFrame({
    "id": test["id"],
    "target": res["test_blend"],
})
