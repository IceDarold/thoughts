# recsys_utils.py
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Iterable, Tuple, Optional
import random
import time
import math


# ======================
# Общие утилиты
# ======================

def set_seed(seed: int = 42) -> None:
    """Фиксируем сиды для повторяемости."""
    random.seed(seed)
    np.random.seed(seed)


class Timer:
    """Контекстный менеджер для замера времени.

    Пример:
    with Timer("build_candidates"):
        candidates = ...
    """
    def __init__(self, name: str = "") -> None:
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.start is None:
            return
        elapsed = time.time() - self.start
        print(f"[TIMER] {self.name}: {elapsed:.2f}s")


# ======================
# Загрузка и препроцесс
# ======================

def read_csv_sorted(path: str,
                    time_col: Optional[str] = None,
                    **read_csv_kwargs) -> pd.DataFrame:
    """Простое чтение CSV + сортировка по времени (если колонка есть)."""
    df = pd.read_csv(path, **read_csv_kwargs)
    if time_col is not None and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
    return df


def add_user_time_rank(df: pd.DataFrame,
                       user_col: str = "user_id",
                       time_col: str = "ts",
                       rank_col: str = "rank_ts") -> pd.DataFrame:
    """Добавляет порядковый номер взаимодействия пользователя по времени."""
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not in df")
    df = df.sort_values([user_col, time_col])
    df[rank_col] = df.groupby(user_col)[time_col].rank(method="first").astype(int)
    return df


# ======================
# Сплиты train/val
# ======================

def train_val_last_k_per_user(df: pd.DataFrame,
                              user_col: str = "user_id",
                              time_col: str = "ts",
                              k: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Сплит: для каждого пользователя последние k событий -> val, остальное -> train."""
    df = df.sort_values([user_col, time_col])
    grp = df.groupby(user_col, sort=False)
    val_idx = grp.tail(k).index
    val_df = df.loc[val_idx]
    train_df = df.drop(val_idx)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def train_val_time_cutoff(df: pd.DataFrame,
                          time_col: str = "ts",
                          cutoff_ts: float = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Сплит по времени: все до cutoff -> train, после/равно -> val."""
    if cutoff_ts is None:
        raise ValueError("cutoff_ts must be provided")
    train_df = df[df[time_col] < cutoff_ts].copy()
    val_df = df[df[time_col] >= cutoff_ts].copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ======================
# Истории пользователей
# ======================

def build_user_histories(df: pd.DataFrame,
                         user_col: str = "user_id",
                         item_col: str = "item_id",
                         time_col: Optional[str] = None,
                         min_len: int = 1) -> Dict:
    """Строит словарь user -> [item1, item2, ...] в хронологическом порядке."""
    if time_col is not None and time_col in df.columns:
        df = df.sort_values([user_col, time_col])
    hist: Dict = defaultdict(list)
    for u, it in zip(df[user_col].values, df[item_col].values):
        hist[u].append(it)
    if min_len <= 1:
        return dict(hist)
    return {u: items for u, items in hist.items() if len(items) >= min_len}


# ======================
# Популярность
# ======================

def compute_item_popularity(df: pd.DataFrame,
                            item_col: str = "item_id",
                            weight_col: Optional[str] = None,
                            min_interactions: int = 1,
                            normalize: bool = True) -> pd.Series:
    """Считает популярность айтемов (по count или по сумме веса)."""
    if weight_col is None:
        pop = df.groupby(item_col)[item_col].count()
    else:
        pop = df.groupby(item_col)[weight_col].sum()

    pop = pop[pop >= min_interactions].astype(float).sort_values(ascending=False)

    if normalize and len(pop) > 0:
        pop = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)

    return pop


def get_top_popular_items(popularity: pd.Series,
                          k: int = 50) -> List:
    """Возвращает список top-k популярных айтемов."""
    return popularity.sort_values(ascending=False).index.to_list()[:k]


# ======================
# Эмбеддинги айтемов
# ======================

class ItemEmbeddingIndex:
    """Простой индекс эмбеддингов айтемов с косинусной похожестью."""
    def __init__(self,
                 items_df: pd.DataFrame,
                 item_id_col: str = "item_id",
                 emb_cols_prefix: str = "emb_"):
        self.item_id_col = item_id_col
        emb_cols = [c for c in items_df.columns if c.startswith(emb_cols_prefix)]
        if not emb_cols:
            raise ValueError(f"No embedding columns starting with '{emb_cols_prefix}'")

        self.item_ids = items_df[item_id_col].values
        emb = items_df[emb_cols].to_numpy(dtype=np.float32)

        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        self.emb = emb / norms
        self.id2idx = {i: idx for idx, i in enumerate(self.item_ids)}

    def get_vector(self, item_id) -> Optional[np.ndarray]:
        idx = self.id2idx.get(item_id)
        if idx is None:
            return None
        return self.emb[idx]

    def most_similar(self,
                     item_id,
                     top_n: int = 50,
                     exclude_self: bool = True) -> List[Tuple]:
        """Находит top_n наиболее похожих айтемов по косинусной метрике."""
        idx = self.id2idx.get(item_id)
        if idx is None:
            return []

        vec = self.emb[idx]  # уже нормирован
        sims = self.emb @ vec  # [n_items]

        if exclude_self:
            sims[idx] = -1.0

        top_n = min(top_n, len(self.item_ids))
        # быстрый top-k + сортировка
        idxs = np.argpartition(-sims, top_n - 1)[:top_n]
        idxs = idxs[np.argsort(-sims[idxs])]

        return [(self.item_ids[i], float(sims[i])) for i in idxs]

    def user_profile(self,
                     item_ids: Iterable,
                     weights: Optional[Iterable] = None) -> Optional[np.ndarray]:
        """Строит эмбеддинг пользователя как взвешенное среднее эмбеддингов его айтемов."""
        vecs = []
        wts = []
        item_ids = list(item_ids)
        if not item_ids:
            return None

        for k, it in enumerate(item_ids):
            v = self.get_vector(it)
            if v is None:
                continue
            vecs.append(v)
            if weights is None:
                wts.append(1.0)
            else:
                if isinstance(weights, (list, tuple, np.ndarray)):
                    if k < len(weights):
                        wts.append(float(weights[k]))
                    else:
                        wts.append(1.0)
                else:
                    wts.append(float(weights))

        if not vecs:
            return None

        vecs = np.stack(vecs, axis=0)
        wts = np.array(wts, dtype=np.float32).reshape(-1, 1)
        prof = (vecs * wts).sum(axis=0) / (wts.sum() + 1e-9)
        prof = prof / (np.linalg.norm(prof) + 1e-9)
        return prof

    def most_similar_to_vector(self,
                               vec: np.ndarray,
                               top_n: int = 50) -> List[Tuple]:
        """Находит айтемы, похожие на произвольный вектор (например, профиль пользователя)."""
        if vec is None:
            return []
        vec = vec.astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        sims = self.emb @ vec
        top_n = min(top_n, len(self.item_ids))
        idxs = np.argpartition(-sims, top_n - 1)[:top_n]
        idxs = idxs[np.argsort(-sims[idxs])]
        return [(self.item_ids[i], float(sims[i])) for i in idxs]


# ======================
# Генерация кандидатов
# ======================

def generate_candidates_mixed(
    user_histories: Dict,
    popularity: pd.Series,
    emb_index: Optional[ItemEmbeddingIndex] = None,
    max_hist_items: int = 20,
    n_pop: int = 100,
    n_sim_per_hist: int = 30,
    max_candidates_per_user: int = 300,
    exclude_seen: bool = True,
    alpha_sim: float = 1.0,
    beta_pop: float = 0.3,
) -> Dict:
    """Генерирует кандидатов: popular + похожие по эмбеддингу на историю пользователя.

    Возвращает: user_id -> [item1, item2, ...] (отсортировано по score).
    """
    if not isinstance(popularity, pd.Series):
        popularity = pd.Series(popularity)

    pop = popularity.astype(float).sort_values(ascending=False)

    if len(pop) > 0:
        pop_norm = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)
    else:
        pop_norm = pop

    top_pop_items = pop_norm.index.to_list()

    result: Dict = {}

    for u, hist in user_histories.items():
        hist = list(hist) if hist is not None else []
        if exclude_seen:
            seen = set(hist)
        else:
            seen = set()

        cand_scores: Dict = {}

        # 1) популярность
        for it in top_pop_items[:n_pop]:
            if it in seen:
                continue
            cand_scores[it] = cand_scores.get(it, 0.0) + beta_pop * pop_norm[it]

        # 2) похожие айтемы на последние просмотренные
        if emb_index is not None and len(hist) > 0:
            recent_hist = hist[-max_hist_items:]
            for it_hist in recent_hist:
                if it_hist not in emb_index.id2idx:
                    continue
                sim_items = emb_index.most_similar(it_hist, top_n=n_sim_per_hist)
                for it_cand, sim in sim_items:
                    if it_cand in seen:
                        continue
                    base = cand_scores.get(it_cand, 0.0)
                    cand_scores[it_cand] = base + alpha_sim * sim

        if not cand_scores:
            # супер cold-start: просто top popular
            cands_sorted = top_pop_items[:max_candidates_per_user]
        else:
            cands_sorted = sorted(cand_scores.items(), key=lambda x: -x[1])
            cands_sorted = [it for it, _ in cands_sorted[:max_candidates_per_user]]

        result[u] = cands_sorted

    return result


# ======================
# Метрики: Recall / MAP / NDCG
# ======================

def build_truth_from_df(df: pd.DataFrame,
                        user_col: str = "user_id",
                        item_col: str = "item_id") -> Dict:
    """user -> список истинных айтемов (может быть несколько)."""
    return df.groupby(user_col)[item_col].apply(list).to_dict()


def recall_at_k(truth: Dict,
                preds: Dict,
                k: int = 20) -> float:
    """Recall@K по user-ам.

    Всего = количество всех истинных пар (user, item),
    зачёт = сколько из них попало в top-K.
    """
    num = 0.0
    den = 0.0
    for u, true_items in truth.items():
        if not true_items:
            continue
        true_set = set(true_items)
        den += len(true_set)
        pred = preds.get(u, [])
        if not pred:
            continue
        hit = len(set(pred[:k]) & true_set)
        num += hit
    return float(num / den) if den > 0 else 0.0


def map_at_k(truth: Dict,
             preds: Dict,
             k: int = 20) -> float:
    """MAP@K по user-ам."""
    ap_sum = 0.0
    n_users = 0

    for u, true_items in truth.items():
        if not true_items:
            continue
        true_set = set(true_items)
        pred = preds.get(u, [])
        if not pred:
            continue

        hits = 0.0
        ap = 0.0
        for idx, it in enumerate(pred[:k], start=1):
            if it in true_set:
                hits += 1.0
                ap += hits / idx

        ap /= min(len(true_set), k)
        ap_sum += ap
        n_users += 1

    return float(ap_sum / n_users) if n_users > 0 else 0.0


def ndcg_at_k(truth: Dict,
              preds: Dict,
              k: int = 20) -> float:
    """NDCG@K по user-ам (бинарные релевантности)."""
    ndcg_sum = 0.0
    n_users = 0

    for u, true_items in truth.items():
        if not true_items:
            continue
        true_set = set(true_items)
        pred = preds.get(u, [])
        if not pred:
            continue

        dcg = 0.0
        for idx, it in enumerate(pred[:k], start=1):
            if it in true_set:
                dcg += 1.0 / math.log2(idx + 1)

        ideal_hits = min(len(true_set), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
        if idcg == 0.0:
            continue

        ndcg_sum += dcg / idcg
        n_users += 1

    return float(ndcg_sum / n_users) if n_users > 0 else 0.0


def evaluate_all(truth: Dict,
                 preds: Dict,
                 ks: Iterable[int] = (5, 10, 20)) -> Dict[str, float]:
    """Считает сразу Recall/MAP/NDCG для нескольких K."""
    res = {}
    for k in ks:
        res[f"recall@{k}"] = recall_at_k(truth, preds, k)
        res[f"map@{k}"] = map_at_k(truth, preds, k)
        res[f"ndcg@{k}"] = ndcg_at_k(truth, preds, k)
    return res


# ======================
# Из score-таблицы в предсказания
# ======================

def scores_df_to_preds(df_scores: pd.DataFrame,
                       user_col: str = "user_id",
                       item_col: str = "item_id",
                       score_col: str = "score",
                       k: int = 20) -> Dict:
    """Преобразует таблицу (user, item, score) в словарь user -> список item по убыванию score."""
    df_sorted = df_scores.sort_values([user_col, score_col],
                                      ascending=[True, False])
    topk = df_sorted.groupby(user_col).head(k)
    preds = topk.groupby(user_col)[item_col].apply(list).to_dict()
    return preds


# ======================
# Негативный сэмплинг и датасет для ранкера
# ======================

def sample_negative_items_for_user(user_items_set: Iterable,
                                   all_items_array: Iterable,
                                   num_negatives: int = 4) -> List:
    """Сэмплирование отрицательных айтемов (не из истории юзера)."""
    user_items_set = set(user_items_set)
    all_items = np.array(list(all_items_array))
    if len(all_items) == 0:
        return []

    mask = ~np.isin(all_items, list(user_items_set))
    candidates = all_items[mask]
    if len(candidates) == 0:
        return []

    if num_negatives >= len(candidates):
        return candidates.tolist()
    idxs = np.random.choice(len(candidates), size=num_negatives, replace=False)
    return candidates[idxs].tolist()


def make_pointwise_training_samples(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    all_items: Iterable = None,
    num_negatives: int = 4,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Генерирует pointwise выборку (user, item, label) с negative sampling.

    all_items: полный список item_id (например, из items_df).
    """
    if all_items is None:
        all_items = interactions[item_col].unique()
    all_items = np.array(list(all_items))

    if drop_duplicates:
        interactions = interactions.drop_duplicates([user_col, item_col])

    user2items = interactions.groupby(user_col)[item_col].apply(set).to_dict()

    rows = []
    for u, pos_items_set in user2items.items():
        pos_items = list(pos_items_set)
        for pos in pos_items:
            rows.append((u, pos, 1))
            neg_items = sample_negative_items_for_user(
                pos_items_set, all_items, num_negatives
            )
            for neg in neg_items:
                rows.append((u, neg, 0))

    train_df = pd.DataFrame(rows, columns=[user_col, item_col, "label"])
    return train_df
