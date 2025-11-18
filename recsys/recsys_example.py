import recsys_utils as ru

ru.set_seed(42)

# чтение
interactions = ru.read_csv_sorted("interactions.csv", time_col="ts")
items = ru.read_csv_sorted("items.csv")

# сплит
train, val = ru.train_val_last_k_per_user(interactions, "user_id", "ts", k=1)

# истории и популярность
user_hist = ru.build_user_histories(train, "user_id", "item_id", time_col="ts")
pop = ru.compute_item_popularity(train, "item_id")

# эмбеддинги
emb_index = ru.ItemEmbeddingIndex(items, item_id_col="item_id", emb_cols_prefix="emb_")

# кандидаты
with ru.Timer("candidates"):
    candidates = ru.generate_candidates_mixed(
        user_histories=user_hist,
        popularity=pop,
        emb_index=emb_index,
        max_candidates_per_user=300
    )

# правда и метрики
truth = ru.build_truth_from_df(val, "user_id", "item_id")
# допустим, пока ранкера нет — просто используем кандидатов как ранжирование:
metrics = ru.evaluate_all(truth, candidates, ks=(5, 10, 20))
print(metrics)
