# image_augs.py
"""
Набор аугментаций картинок для олимпиад / kaggle:
- геометрические (flip, rotate, crop, resize+pad)
- цветовые (яркость, контраст, насыщенность, hue, grayscale и т.п.)
- blur / noise / cutout
- mixup / cutmix
- простые классы Compose, RandomApply, OneOf
"""

import random
import math
from typing import Tuple, List, Sequence, Callable, Optional
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter


# =====================================
# Общие утилиты
# =====================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


class Compose:
    """Последовательно применяет список трансформов к картинке."""
    def __init__(self, transforms: Sequence[Callable[[Image.Image], Image.Image]]):
        self.transforms = list(transforms)

    def __call__(self, img: Image.Image) -> Image.Image:
        for t in self.transforms:
            img = t(img)
        return img


class RandomApply:
    """Применяет трансформ с вероятностью p."""
    def __init__(self, transform: Callable[[Image.Image], Image.Image], p: float = 0.5):
        self.transform = transform
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return self.transform(img)
        return img


class OneOf:
    """Случайно выбирает один трансформ из списка и применяет его с вероятностью p."""
    def __init__(self, transforms: Sequence[Callable[[Image.Image], Image.Image]], p: float = 0.8):
        self.transforms = list(transforms)
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.transforms or random.random() >= self.p:
            return img
        t = random.choice(self.transforms)
        return t(img)


# =====================================
# БАЗОВЫЕ ГЕОМЕТРИЧЕСКИЕ АУГИ
# =====================================

def hflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def vflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def random_horizontal_flip(p: float = 0.5) -> Callable[[Image.Image], Image.Image]:
    def _t(img: Image.Image) -> Image.Image:
        if random.random() < p:
            return hflip(img)
        return img
    return _t


def random_vertical_flip(p: float = 0.5) -> Callable[[Image.Image], Image.Image]:
    def _t(img: Image.Image) -> Image.Image:
        if random.random() < p:
            return vflip(img)
        return img
    return _t


def random_rotate(max_degrees: float = 15.0, p: float = 0.5) -> Callable[[Image.Image], Image.Image]:
    """Случайный поворот в диапазоне [-max_deg, max_deg]."""
    def _t(img: Image.Image) -> Image.Image:
        if random.random() >= p or max_degrees <= 0:
            return img
        angle = random.uniform(-max_degrees, max_degrees)
        return img.rotate(angle, resample=Image.BILINEAR)
    return _t


def resize(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Ресайз до (W, H)."""
    w, h = size
    return img.resize((w, h), Image.BILINEAR)


def pad_to_square(img: Image.Image, pad_color=(0, 0, 0)) -> Image.Image:
    """Паддинг до квадрата без ресайза."""
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
    """Сначала масштабируем по большей стороне, потом паддим до квадрата и ресайзим до size."""
    target_w, target_h = size
    max_side = max(target_w, target_h)
    w, h = img.size
    scale = max(w, h) / float(max_side)
    if scale > 1.0:
        new_w = int(round(w / scale))
        new_h = int(round(h / scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
    img = pad_to_square(img, pad_color=pad_color)
    img = img.resize((target_w, target_h), Image.BILINEAR)
    return img


def center_crop(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
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


def random_crop(size: Tuple[int, int]) -> Callable[[Image.Image], Image.Image]:
    """Случайный кроп размера size=(W,H)."""
    target_w, target_h = size

    def _t(img: Image.Image) -> Image.Image:
        w, h = img.size
        if w < target_w or h < target_h:
            # масштабируем до min >= size
            scale = max(target_w / w, target_h / h)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img2 = img.resize((new_w, new_h), Image.BILINEAR)
        else:
            img2 = img

        w2, h2 = img2.size
        if w2 == target_w and h2 == target_h:
            return img2

        left = random.randint(0, w2 - target_w)
        top = random.randint(0, h2 - target_h)
        right = left + target_w
        bottom = top + target_h
        return img2.crop((left, top, right, bottom))

    return _t


def random_resized_crop(size: Tuple[int, int],
                        scale: Tuple[float, float] = (0.6, 1.0),
                        ratio: Tuple[float, float] = (3. / 4., 4. / 3.),
                        trials: int = 10) -> Callable[[Image.Image], Image.Image]:
    """Аналог torchvision RandomResizedCrop (упрощённый)."""
    target_w, target_h = size

    def _t(img: Image.Image) -> Image.Image:
        w, h = img.size
        area = w * h
        for _ in range(trials):
            target_area = random.uniform(scale[0], scale[1]) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect = math.exp(random.uniform(*log_ratio))

            cw = int(round(math.sqrt(target_area * aspect)))
            ch = int(round(math.sqrt(target_area / aspect)))

            if cw <= w and ch <= h:
                left = random.randint(0, w - cw)
                top = random.randint(0, h - ch)
                img_c = img.crop((left, top, left + cw, top + ch))
                return img_c.resize((target_w, target_h), Image.BILINEAR)

        # fallback — центр-кроп + ресайз
        return center_crop(img, (target_w, target_h)).resize((target_w, target_h), Image.BILINEAR)

    return _t


# =====================================
# ЦВЕТОВЫЕ АУГИ
# =====================================

def _to_float_np(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def _from_float_np(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0) * 255.0
    arr = arr.astype(np.uint8)
    return Image.fromarray(arr)


def random_brightness(max_delta: float = 0.2, p: float = 0.8) -> Callable[[Image.Image], Image.Image]:
    """Умножаем картинку на [1-max_delta, 1+max_delta]."""
    def _t(img: Image.Image) -> Image.Image:
        if random.random() >= p or max_delta <= 0:
            return img
        factor = 1.0 + random.uniform(-max_delta, max_delta)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    return _t


def random_contrast(max_delta: float = 0.2, p: float = 0.8) -> Callable[[Image.Image], Image.Image]:
    def _t(img: Image.Image) -> Image.Image:
        if random.random() >= p or max_delta <= 0:
            return img
        factor = 1.0 + random.uniform(-max_delta, max_delta)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    return _t


def random_saturation(max_delta: float = 0.2, p: float = 0.8) -> Callable[[Image.Image], Image.Image]:
    def _t(img: Image.Image) -> Image.Image:
        if random.random() >= p or max_delta <= 0:
            return img
        factor = 1.0 + random.uniform(-max_delta, max_delta)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)
    return _t


def random_hue(max_delta: float = 0.05, p: float = 0.8) -> Callable[[Image.Image], Image.Image]:
    """Сдвиг hue в пространстве HSV. max_delta ~ [0, 0.5]."""
    def _t(img: Image.Image) -> Image.Image:
        if random.random() >= p or max_delta <= 0:
            return img
        # переходим в HSV
        hsv = img.convert("HSV")
        arr = np.array(hsv).astype(np.int16)
        h = arr[..., 0]
        delta = int(random.uniform(-max_delta, max_delta) * 255)
        h = (h + delta) % 256
        arr[..., 0] = h
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        hsv = Image.fromarray(arr, mode="HSV")
        return hsv.convert("RGB")
    return _t


def random_color_jitter(brightness: float = 0.2,
                        contrast: float = 0.2,
                        saturation: float = 0.2,
                        hue: float = 0.05,
                        p: float = 0.8) -> Callable[[Image.Image], Image.Image]:
    """Комбинированный jitter как в torchvision."""
    transforms = []
    if brightness > 0:
        transforms.append(random_brightness(brightness, p=1.0))
    if contrast > 0:
        transforms.append(random_contrast(contrast, p=1.0))
    if saturation > 0:
        transforms.append(random_saturation(saturation, p=1.0))
    if hue > 0:
        transforms.append(random_hue(hue, p=1.0))

    def _t(img: Image.Image) -> Image.Image:
        if not transforms or random.random() >= p:
            return img
        order = list(range(len(transforms)))
        random.shuffle(order)
        out = img
        for idx in order:
            out = transforms[idx](out)
        return out

    return _t


def random_grayscale(p: float = 0.2) -> Callable[[Image.Image], Image.Image]:
    def _t(img: Image.Image) -> Image.Image:
        if random.random() >= p:
            return img
        return ImageOps.grayscale(img).convert("RGB")
    return _t


def random_solarize(threshold: int = 128, p: float = 0.2) -> Callable[[Image.Image], Image.Image]:
    def _t(img: Image.Image) -> Image.Image:
        if random.random() >= p:
            return img
        return ImageOps.solarize(img, threshold=threshold)
    return _t


def random_posterize(bits: int = 4, p: float = 0.2) -> Callable[[Image.Image], Image.Image]:
    def _t(img: Image.Image) -> Image.Image:
        if random.random() >= p:
            return img
        return ImageOps.posterize(img, bits=bits)
    return _t


def random_equalize(p: float = 0.2) -> Callable[[Image.Image], Image.Image]:
    def _t(img: Image.Image) -> Image.Image:
        if random.random() >= p:
            return img
        return ImageOps.equalize(img)
    return _t


def random_autocontrast(p: float = 0.2) -> Callable[[Image.Image], Image.Image]:
    def _t(img: Image.Image) -> Image.Image:
        if random.random() >= p:
            return img
        return ImageOps.autocontrast(img)
    return _t


# =====================================
# BLUR / SHARPNESS / NOISE
# =====================================

def random_gaussian_blur(max_radius: float = 1.5, p: float = 0.5) -> Callable[[Image.Image], Image.Image]:
    def _t(img: Image.Image) -> Image.Image:
        if random.random() >= p or max_radius <= 0:
            return img
        radius = random.uniform(0.1, max_radius)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    return _t


def random_sharpness(max_factor: float = 0.5, p: float = 0.5) -> Callable[[Image.Image], Image.Image]:
    """factor=1 — оригинал; <1 размытие, >1 усиление резкости."""
    def _t(img: Image.Image) -> Image.Image:
        if random.random() >= p or max_factor <= 0:
            return img
        factor = 1.0 + random.uniform(-max_factor, max_factor)
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)
    return _t


def random_gaussian_noise(std: float = 0.05, p: float = 0.5) -> Callable[[Image.Image], Image.Image]:
    """Добавляет гауссов шум (std относительный, ~0.05 ок)."""
    def _t(img: Image.Image) -> Image.Image:
        if random.random() >= p or std <= 0:
            return img
        arr = _to_float_np(img)
        noise = np.random.normal(0.0, std, size=arr.shape).astype(np.float32)
        arr = arr + noise
        return _from_float_np(arr)
    return _t


def random_salt_and_pepper(amount: float = 0.01, p: float = 0.5) -> Callable[[Image.Image], Image.Image]:
    """Salt & pepper шум: небольшая доля пикселей -> 0 или 1."""
    def _t(img: Image.Image) -> Image.Image:
        if random.random() >= p or amount <= 0:
            return img
        arr = _to_float_np(img)
        h, w, c = arr.shape
        num = int(amount * h * w)
        if num <= 0:
            return img
        ys = np.random.randint(0, h, size=num)
        xs = np.random.randint(0, w, size=num)
        mask = np.random.rand(num) < 0.5
        arr[ys[mask], xs[mask], :] = 0.0
        arr[ys[~mask], xs[~mask], :] = 1.0
        return _from_float_np(arr)
    return _t


# =====================================
# CUTOUT / RANDOM ERASING
# =====================================

def random_cutout(num_holes: int = 1,
                  max_hole_frac: float = 0.5,
                  fill_value: Tuple[int, int, int] = (0, 0, 0),
                  p: float = 0.5) -> Callable[[Image.Image], Image.Image]:
    """Cutout: несколько прямоугольных дыр заполняем цветом."""
    def _t(img: Image.Image) -> Image.Image:
        if random.random() >= p or num_holes <= 0 or max_hole_frac <= 0:
            return img
        w, h = img.size
        arr = np.array(img).copy()

        for _ in range(num_holes):
            hole_w = int(random.uniform(0.1, max_hole_frac) * w)
            hole_h = int(random.uniform(0.1, max_hole_frac) * h)
            if hole_w <= 0 or hole_h <= 0:
                continue
            x0 = random.randint(0, max(0, w - hole_w))
            y0 = random.randint(0, max(0, h - hole_h))
            arr[y0:y0 + hole_h, x0:x0 + hole_w, :] = np.array(fill_value, dtype=arr.dtype)[None, None, :]

        return Image.fromarray(arr)

    return _t


def random_erasing(p: float = 0.5,
                   scale: Tuple[float, float] = (0.02, 0.33),
                   ratio: Tuple[float, float] = (0.3, 3.3),
                   fill_value: Optional[Tuple[int, int, int]] = None,
                   trials: int = 10) -> Callable[[Image.Image], Image.Image]:
    """Random Erasing (как в статье) — одна область, случайный размер/аспект."""
    def _t(img: Image.Image) -> Image.Image:
        if random.random() >= p:
            return img

        w, h = img.size
        area = w * h
        arr = np.array(img).copy()

        for _ in range(trials):
            target_area = random.uniform(scale[0], scale[1]) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect = math.exp(random.uniform(*log_ratio))

            ew = int(round(math.sqrt(target_area * aspect)))
            eh = int(round(math.sqrt(target_area / aspect)))

            if ew <= 0 or eh <= 0 or ew > w or eh > h:
                continue

            x0 = random.randint(0, w - ew)
            y0 = random.randint(0, h - eh)

            if fill_value is None:
                # шум
                patch = np.random.randint(0, 256, size=(eh, ew, arr.shape[2]), dtype=arr.dtype)
            else:
                patch = np.array(fill_value, dtype=arr.dtype)[None, None, :]

            arr[y0:y0 + eh, x0:x0 + ew, :] = patch
            return Image.fromarray(arr)

        return img

    return _t


# =====================================
# MIXUP / CUTMIX (для пар картинок)
# =====================================

def mixup_images(img1: Image.Image,
                 img2: Image.Image,
                 alpha: float = 0.4) -> Tuple[Image.Image, float]:
    """Mixup двух картинок (одинакового размера).

    Возвращает (смешанная_картинка, lambda), где lambda — вес первой.
    """
    if alpha <= 0:
        lam = 1.0
    else:
        lam = np.random.beta(alpha, alpha)

    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.BILINEAR)

    arr1 = _to_float_np(img1)
    arr2 = _to_float_np(img2)
    mixed = lam * arr1 + (1.0 - lam) * arr2
    mixed = np.clip(mixed, 0.0, 1.0)
    return _from_float_np(mixed), float(lam)


def cutmix_images(img1: Image.Image,
                  img2: Image.Image,
                  alpha: float = 1.0) -> Tuple[Image.Image, float]:
    """CutMix двух картинок (одинакового размера).

    Возвращает (картинка, lambda), где lambda — доля площади от первой картинки.
    """
    if alpha <= 0:
        lam = 1.0
    else:
        lam = np.random.beta(alpha, alpha)

    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.BILINEAR)

    w, h = img1.size
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    cut_w = int(w * math.sqrt(1 - lam))
    cut_h = int(h * math.sqrt(1 - lam))

    x0 = np.clip(cx - cut_w // 2, 0, w)
    y0 = np.clip(cy - cut_h // 2, 0, h)
    x1 = np.clip(x0 + cut_w, 0, w)
    y1 = np.clip(y0 + cut_h, 0, h)

    arr1 = np.array(img1)
    arr2 = np.array(img2)

    arr1[y0:y1, x0:x1, :] = arr2[y0:y1, x0:x1, :]

    # Фактическая доля площади от первой (может чуть отличаться из-за границ)
    new_lam = 1.0 - (float((x1 - x0) * (y1 - y0)) / float(w * h))
    return Image.fromarray(arr1), float(new_lam)


# =====================================
# ГОТОВАЯ "СИЛЬНАЯ" АУГА ДЛЯ КЛАССИФИКАЦИИ
# =====================================

def build_default_strong_aug(
    size: Tuple[int, int] = (224, 224)
) -> Callable[[Image.Image], Image.Image]:
    """Типичный сильный pipeline для классификации: RRCrop + color jitter + blur + cutout."""
    return Compose([
        random_resized_crop(size=size, scale=(0.6, 1.0)),
        random_horizontal_flip(p=0.5),
        OneOf([
            random_color_jitter(0.3, 0.3, 0.3, 0.05, p=1.0),
            random_grayscale(p=1.0),
        ], p=0.7),
        OneOf([
            random_gaussian_blur(max_radius=1.5, p=1.0),
            random_sharpness(max_factor=0.7, p=1.0),
        ], p=0.5),
        OneOf([
            random_cutout(num_holes=1, max_hole_frac=0.4, fill_value=(0, 0, 0), p=1.0),
            random_erasing(p=1.0, scale=(0.02, 0.2)),
        ], p=0.5),
    ])
