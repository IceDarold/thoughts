# image_utils.py
import os
import math
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterable

import numpy as np

try:
    from PIL import Image
except ImportError as e:
    raise ImportError(
        "Pillow (PIL) не установлен. Установи его командой `pip install pillow`."
    ) from e


# ======================
# Общие утилиты
# ======================

def set_seed(seed: int = 42) -> None:
    """Фиксим сиды для повторяемости."""
    random.seed(seed)
    np.random.seed(seed)


class Timer:
    """Контекстный менеджер для замера времени.

    Пример:
    with Timer("load_images"):
        imgs = load_image_batch(...)
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
# Работа с путями и списком картинок
# ======================

def list_images(root: str,
                exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
                recursive: bool = True) -> List[Path]:
    """Возвращает список путей до картинок в директории."""
    root_path = Path(root)
    exts_lower = tuple(e.lower() for e in exts)

    if recursive:
        paths = [
            p for p in root_path.rglob("*")
            if p.suffix.lower() in exts_lower and p.is_file()
        ]
    else:
        paths = [
            p for p in root_path.iterdir()
            if p.suffix.lower() in exts_lower and p.is_file()
        ]
    return sorted(paths)


def build_image_table(
    root: str,
    infer_label_from_parent: bool = False,
    exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
):
    """Строит простую табличку по картинкам: id, path, (опционально label).

    Возвращает pandas.DataFrame, но импорт делает локально,
    чтобы не тянуть pandas без надобности.
    """
    import pandas as pd  # локальный импорт, чтобы модуль не падал без pandas

    paths = list_images(root, exts=exts, recursive=True)
    records = []
    for p in paths:
        img_id = p.stem
        if infer_label_from_parent:
            label = p.parent.name
        else:
            label = None
        records.append({"image_id": img_id, "path": str(p), "label": label})

    df = pd.DataFrame(records)
    return df


# ======================
# Загрузка и сохранение картинок
# ======================

def load_image(path: str,
               mode: str = "RGB",
               max_side: Optional[int] = None) -> Image.Image:
    """Читает картинку через PIL, опционально сжимает так, чтобы max(H, W) <= max_side."""
    img = Image.open(path)
    if mode is not None:
        img = img.convert(mode)

    if max_side is not None:
        w, h = img.size
        scale = max(w, h) / float(max_side)
        if scale > 1.0:
            new_w = int(round(w / scale))
            new_h = int(round(h / scale))
            img = img.resize((new_w, new_h), Image.BILINEAR)
    return img


def save_image(img: Image.Image,
               path: str,
               quality: int = 95) -> None:
    """Сохраняет картинку на диск, создавая директорию при необходимости."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), quality=quality)


# ======================
# Конвертации PIL <-> numpy
# ======================

def pil_to_numpy(img: Image.Image,
                 normalize: bool = False,
                 chw: bool = False,
                 dtype=np.float32) -> np.ndarray:
    """Перевод PIL.Image в numpy array.

    normalize: делить на 255
    chw: возвращать (C, H, W), иначе (H, W, C)
    """
    arr = np.array(img)
    if normalize:
        arr = arr.astype(dtype) / 255.0
    else:
        arr = arr.astype(dtype)

    if chw:
        if arr.ndim == 2:
            arr = arr[None, ...]  # (1, H, W)
        else:
            arr = np.transpose(arr, (2, 0, 1))
    return arr


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Перевод numpy (H, W, C) или (C, H, W) в PIL.Image."""
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[1]:
        # (C, H, W) -> (H, W, C)
        arr = np.transpose(arr, (1, 2, 0))

    arr = np.asarray(arr)
    if arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr)


# ======================
# Ресайзы и кропы
# ======================

def resize_to_max_side(img: Image.Image,
                       max_side: int) -> Image.Image:
    """Масштабирует так, чтобы max(H, W) == max_side (или меньше, если картинка уже маленькая)."""
    w, h = img.size
    scale = max(w, h) / float(max_side)
    if scale <= 1.0:
        return img
    new_w = int(round(w / scale))
    new_h = int(round(h / scale))
    return img.resize((new_w, new_h), Image.BILINEAR)


def resize_shorter_side(img: Image.Image,
                        short_side: int) -> Image.Image:
    """Масштабирует так, чтобы мин(H, W) == short_side."""
    w, h = img.size
    s = min(w, h)
    if s == 0:
        return img
    scale = short_side / float(s)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), Image.BILINEAR)


def center_crop(img: Image.Image,
                size: Tuple[int, int]) -> Image.Image:
    """Центральный кроп до size=(W, H)."""
    target_w, target_h = size
    w, h = img.size
    if w < target_w or h < target_h:
        # сначала масштабируем до min >= size
        scale = max(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        w, h = img.size

    left = (w - target_w) // 2
    top = (h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    return img.crop((left, top, right, bottom))


def pad_to_square(img: Image.Image,
                  pad_color=(0, 0, 0)) -> Image.Image:
    """Добавляет паддинги, чтобы картинка стала квадратной (без ресайза)."""
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    new_img = Image.new(img.mode, (size, size), pad_color)
    offset = ((size - w) // 2, (size - h) // 2)
    new_img.paste(img, offset)
    return new_img


def resize_and_pad(img: Image.Image,
                   size: Tuple[int, int] = (224, 224),
                   pad_color=(0, 0, 0)) -> Image.Image:
    """Сначала масштабирует по большей стороне, потом паддит до size."""
    target_w, target_h = size
    # масштабируем так, чтобы max(H, W) == max(target_w, target_h)
    max_side = max(target_w, target_h)
    img = resize_to_max_side(img, max_side=max_side)
    img = pad_to_square(img, pad_color=pad_color)
    img = img.resize((target_w, target_h), Image.BILINEAR)
    return img


# ======================
# Простые аугментации
# ======================

def random_horizontal_flip(img: Image.Image,
                           p: float = 0.5) -> Image.Image:
    """Случайный горизонтальный флип."""
    if random.random() < p:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def random_rotate(img: Image.Image,
                  max_deg: float = 10.0,
                  p: float = 0.5) -> Image.Image:
    """Случайный поворот в диапазоне [-max_deg, max_deg]."""
    if random.random() < p and max_deg > 0:
        angle = random.uniform(-max_deg, max_deg)
        return img.rotate(angle, resample=Image.BILINEAR)
    return img


def random_color_jitter(img: Image.Image,
                        brightness: float = 0.1,
                        contrast: float = 0.1,
                        saturation: float = 0.1) -> Image.Image:
    """Очень простая реализация ColorJitter без зависимостей от torchvision."""
    arr = np.asarray(img).astype(np.float32) / 255.0

    # яркость
    if brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        arr = arr * factor

    # контраст
    if contrast > 0:
        mean = arr.mean(axis=(0, 1), keepdims=True)
        factor = 1.0 + random.uniform(-contrast, contrast)
        arr = (arr - mean) * factor + mean

    # насыщенность (через перевод в "оттенок серого" и смешивание)
    if saturation > 0 and arr.shape[-1] == 3:
        gray = arr.mean(axis=2, keepdims=True)
        factor = 1.0 + random.uniform(-saturation, saturation)
        arr = (arr - gray) * factor + gray

    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def random_augment(
    img: Image.Image,
    flip_p: float = 0.5,
    max_rotate_deg: float = 10.0,
    color_jitter_strength: float = 0.1,
) -> Image.Image:
    """Комбо-ауга: флип, поворот, немного цвета."""
    img = random_horizontal_flip(img, p=flip_p)
    img = random_rotate(img, max_deg=max_rotate_deg, p=0.7 if max_rotate_deg > 0 else 0.0)
    if color_jitter_strength > 0:
        img = random_color_jitter(
            img,
            brightness=color_jitter_strength,
            contrast=color_jitter_strength,
            saturation=color_jitter_strength,
        )
    return img


# ======================
# Батч-препроцессинг
# ======================

def preprocess_image(
    path: str,
    size: Tuple[int, int] = (224, 224),
    augment: bool = False,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Полный пайплайн для одной картинки: load -> resize+pad -> (aug) -> to numpy CHW.

    Возвращает массив shape (3, H, W) либо (H, W, 3) если надо, но
    здесь делаем именно (C, H, W) под типичный CV-пайплайн.
    """
    img = load_image(path, mode="RGB")
    img = resize_and_pad(img, size=size)

    if augment:
        img = random_augment(img)

    arr = pil_to_numpy(img, normalize=True, chw=True)  # (C, H, W), [0, 1]

    if normalize:
        # ImageNet-статы по умолчанию
        mean_arr = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        std_arr = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
        arr = (arr - mean_arr) / (std_arr + 1e-9)

    return arr.astype(np.float32)


def load_image_batch(
    paths: List[str],
    size: Tuple[int, int] = (224, 224),
    augment: bool = False,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Готовит батч картинок: (N, C, H, W)."""
    batch = []
    for p in paths:
        arr = preprocess_image(
            p, size=size, augment=augment, normalize=normalize,
            mean=mean, std=std,
        )
        batch.append(arr)
    if not batch:
        return np.zeros((0, 3, size[1], size[0]), dtype=np.float32)
    return np.stack(batch, axis=0)


# ======================
# Статистики датасета
# ======================

def compute_dataset_channel_stats(
    paths: List[str],
    size: Tuple[int, int] = (224, 224),
    sample_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Оценивает mean/std по каналам на подвыборке картинок.

    Возвращает:
        mean: shape (3,)
        std: shape (3,)
    """
    if len(paths) == 0:
        raise ValueError("paths is empty")

    if sample_size > len(paths):
        sample_paths = list(paths)
    else:
        sample_paths = random.sample(paths, sample_size)

    sums = np.zeros(3, dtype=np.float64)
    sq_sums = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for p in sample_paths:
        img = load_image(p, mode="RGB")
        img = resize_and_pad(img, size=size)
        arr = pil_to_numpy(img, normalize=True, chw=True)  # (3, H, W), [0, 1]
        c, h, w = arr.shape
        arr = arr.reshape(c, -1)
        sums += arr.sum(axis=1)
        sq_sums += (arr ** 2).sum(axis=1)
        n_pixels += arr.shape[1]

    mean = sums / n_pixels
    var = (sq_sums / n_pixels) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


# ======================
# Работа с эмбеддингами картинок
# ======================

def normalize_embeddings(emb: np.ndarray,
                         axis: int = 1,
                         eps: float = 1e-9) -> np.ndarray:
    """L2-нормализация эмбеддингов."""
    norm = np.linalg.norm(emb, axis=axis, keepdims=True) + eps
    return emb / norm


def cosine_similarity_matrix(a: np.ndarray,
                             b: Optional[np.ndarray] = None) -> np.ndarray:
    """Косинусная похожесть между всеми парами строк a и b.

    a: (N, D)
    b: (M, D) или None (тогда b = a)
    """
    if b is None:
        b = a
    a_n = normalize_embeddings(a, axis=1)
    b_n = normalize_embeddings(b, axis=1)
    return a_n @ b_n.T


def topk_similar(
    emb: np.ndarray,
    query_idx: int,
    k: int = 10,
    exclude_self: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Находит top-k ближайших к emb[query_idx] по косинусу.

    Возвращает:
        indices: (k,)
        scores: (k,)
    """
    sims = cosine_similarity_matrix(emb[query_idx:query_idx + 1], emb)[0]  # (N,)
    if exclude_self:
        sims[query_idx] = -1.0

    k = min(k, len(sims))
    idxs = np.argpartition(-sims, k - 1)[:k]
    idxs = idxs[np.argsort(-sims[idxs])]
    return idxs, sims[idxs]


def knn_graph_from_embeddings(
    emb: np.ndarray,
    k: int = 20,
    exclude_self: bool = True,
) -> Dict[int, List[Tuple[int, float]]]:
    """Строит простой kNN-граф по эмбеддингам: i -> [(j, sim_ij), ...]."""
    n = emb.shape[0]
    graph: Dict[int, List[Tuple[int, float]]] = {}
    for i in range(n):
        idxs, sims = topk_similar(emb, i, k=k, exclude_self=exclude_self)
        graph[i] = list(zip(idxs.tolist(), sims.tolist()))
    return graph
