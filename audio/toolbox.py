"""
audio_toolbox.py — набор утилит для работы со звуком на олимпиадах

Идея:
- загрузка и нормализация аудио
- нарезка/паддинг до нужной длины
- вычисление спектрограмм / лог-мел / MFCC
- простые аугментации (шум, сдвиг, SpecAugment)
- простейший VAD по энергии
- шаблон под KWS-датасет и метрики

Это не обязанo работать «из коробки» — это шаблоны.
Точки, где надо править под задачу, помечены в комментариях.
"""

import os
import math
import random
from typing import List, Tuple, Optional, Callable, Dict, Union

import numpy as np

# Попытка импортировать типичные аудио-библиотеки.
# На олимпиаде можешь либо использовать их, либо переписать на torchaudio.
try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None

# torch — опционально, для Dataset/тензоров
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None


# ==========
# Общие утилиты
# ==========

def set_seed(seed: int = 42):
    """Фиксируем сиды для numpy/random/torch (если есть)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        _torch.cuda.manual_seed_all(seed)
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ==========
# Загрузка и базовая подготовка аудио
# ==========

def load_audio(
    path: str,
    sr: Optional[int] = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Загружаем аудио как numpy-массив float32, возвращаем (waveform, sr).

    sr=None -> оставить родной sample rate.
    mono=True -> сводим к одному каналу.
    """

    if librosa is not None:
        # librosa.load сразу даёт float32 и умеет ресемплить
        wav, orig_sr = librosa.load(path, sr=sr, mono=mono)
        return wav.astype("float32"), orig_sr
    elif sf is not None:
        wav, orig_sr = sf.read(path)  # (samples, channels?) или (samples,) если mono
        wav = wav.astype("float32")
        if mono and wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr is not None and sr != orig_sr and librosa is not None:
            wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
            return wav.astype("float32"), sr
        return wav, orig_sr
    else:
        raise ImportError("Нет ни librosa, ни soundfile — подставь свой loader.")


def normalize_peak(wav: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """
    Нормализация по пику: приводим максимум |x| к заданному значению.
    """
    wav = wav.astype("float32")
    mx = np.max(np.abs(wav)) + 1e-9
    return wav * (peak / mx)


def fix_length(
    wav: np.ndarray,
    target_len: int,
    mode: str = "pad_trunc",
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Приводим сигнал к фиксированной длине в сэмплах.
    mode="pad_trunc": сначала режем, потом паддим, если нужно.
    """
    x = wav
    if x.shape[0] > target_len:
        x = x[:target_len]
    elif x.shape[0] < target_len:
        pad = np.full(target_len - x.shape[0], pad_value, dtype=x.dtype)
        x = np.concatenate([x, pad], axis=0)
    return x


def random_crop(
    wav: np.ndarray,
    crop_len: int,
) -> np.ndarray:
    """
    Случайный кроп фиксированной длины.
    Если сигнал короче — допадиваем справа.
    """
    if wav.shape[0] <= crop_len:
        return fix_length(wav, crop_len)
    max_start = wav.shape[0] - crop_len
    start = random.randint(0, max_start)
    return wav[start:start + crop_len]


# ==========
# STFT, мел-спектрограммы, MFCC
# ==========

def compute_stft(
    wav: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
) -> np.ndarray:
    """
    Возвращает комплексный спектр (частота x время).
    Для лог-мела ниже можно использовать его или сразу librosa.feature.melspectrogram.
    """
    if librosa is None:
        raise ImportError("Нужна librosa для STFT, либо реализуй сам через scipy/torch.")
    if win_length is None:
        win_length = n_fft
    stft = librosa.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
    )
    return stft


def amplitude_to_db(S: np.ndarray, ref: float = 1.0, amin: float = 1e-10) -> np.ndarray:
    """
    Перевод амплитуды/мощности в dB.
    Если есть librosa, лучше использовать librosa.amplitude_to_db / power_to_db.
    """
    if librosa is not None:
        # Пытаемся угадать, мощность это или амплитуда
        return librosa.power_to_db(S, ref=ref)
    S = np.maximum(S, amin)
    return 20.0 * np.log10(S / ref)


def logmel_spectrogram(
    wav: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 64,
    fmin: float = 50.0,
    fmax: Optional[float] = None,
    power: float = 2.0,
) -> np.ndarray:
    """
    Лог-мел спектрограмма (n_mels x time).
    Суперклассика для KWS/ASR.
    """
    if librosa is None:
        raise ImportError("Для log-mel нужен librosa. Либо перенеси это на torchaudio.")
    if fmax is None:
        fmax = sr / 2
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel.astype("float32")


def mfcc_from_waveform(
    wav: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 40,
    fmin: float = 50.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """
    MFCC по сигналу (n_mfcc x time).
    Иногда MFCC вполне хватает для простых KWS задач.
    """
    if librosa is None:
        raise ImportError("Для MFCC нужен librosa.")
    if fmax is None:
        fmax = sr / 2
    mfcc = librosa.feature.mfcc(
        y=wav,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    return mfcc.astype("float32")


# ==========
# Простые аудио-аугментации
# ==========

def add_white_noise(
    wav: np.ndarray,
    snr_db: float = 20.0,
) -> np.ndarray:
    """
    Добавляем белый шум с заданным соотношением сигнал/шум в dB.
    """
    sig_power = np.mean(wav ** 2) + 1e-9
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=wav.shape).astype("float32")
    return wav + noise


def random_gain(
    wav: np.ndarray,
    min_gain_db: float = -6.0,
    max_gain_db: float = 6.0,
) -> np.ndarray:
    """
    Случайное усиление/ослабление (в dB).
    """
    gain_db = random.uniform(min_gain_db, max_gain_db)
    gain = 10 ** (gain_db / 20.0)
    return (wav * gain).astype("float32")


def time_shift(
    wav: np.ndarray,
    max_shift_frac: float = 0.2,
) -> np.ndarray:
    """
    Случайный циклический сдвиг сигнала во времени.
    max_shift_frac=0.2 -> сдвиг до 20% длины.
    """
    n = wav.shape[0]
    max_shift = int(n * max_shift_frac)
    if max_shift <= 0:
        return wav
    shift = random.randint(-max_shift, max_shift)
    return np.roll(wav, shift)


# ==========
# SpecAugment-подобные аугментации для спектрограмм
# ==========

def random_time_mask(
    spec: np.ndarray,
    max_width: int = 20,
    p: float = 0.5,
) -> np.ndarray:
    """
    Маскирование по времени: spec (freq x time).
    """
    if random.random() > p:
        return spec
    spec = spec.copy()
    n_frames = spec.shape[1]
    if n_frames <= 1:
        return spec
    width = random.randint(1, min(max_width, n_frames))
    start = random.randint(0, n_frames - width)
    spec[:, start:start + width] = 0.0
    return spec


def random_freq_mask(
    spec: np.ndarray,
    max_width: int = 8,
    p: float = 0.5,
) -> np.ndarray:
    """
    Маскирование по частоте: spec (freq x time).
    """
    if random.random() > p:
        return spec
    spec = spec.copy()
    n_freq = spec.shape[0]
    if n_freq <= 1:
        return spec
    width = random.randint(1, min(max_width, n_freq))
    start = random.randint(0, n_freq - width)
    spec[start:start + width, :] = 0.0
    return spec


# ==========
# Простейший VAD (voice activity detection) по энергии
# ==========

def frame_energy(
    wav: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    """
    Энергия по фреймам (mean(x^2)).
    """
    n = wav.shape[0]
    energies = []
    for start in range(0, n - frame_length + 1, hop_length):
        frame = wav[start:start + frame_length]
        energies.append(np.mean(frame ** 2))
    return np.array(energies, dtype="float32")


def simple_energy_vad(
    wav: np.ndarray,
    frame_length: int,
    hop_length: int,
    energy_thresh: float,
) -> np.ndarray:
    """
    Возвращает вектор маски фреймов (1 = речь, 0 = тишина).
    Ожидается, что energy_thresh выберешь по валидации.
    """
    eng = frame_energy(wav, frame_length, hop_length)
    mask = (eng > energy_thresh).astype("int8")
    return mask


# ==========
# KWS / аудио-датасет (шаблон)
# ==========

class KWSDataset(Dataset):
    """
    Шаблон датасета для задачи keyword spotting / классификации коротких аудио-клипов.

    Ожидает:
    - items: список (path, label)
    - sr: sample rate, до которого приводим
    - clip_duration: длина клипа в секундах (обрежем/допадим)
    - waveform_transform: функции типа augment (работают по wav)
    - spec_transform: функции типа random_time_mask/freq_mask (по спектру)
    - feature_type: "logmel" / "mfcc"
    """

    def __init__(
        self,
        items: List[Tuple[str, int]],
        sr: int = 16000,
        clip_duration: float = 1.0,
        feature_type: str = "logmel",
        waveform_transform: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        spec_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        n_mels: int = 64,
    ):
        if torch is None:
            raise ImportError("Для этого класса нужен torch.")

        self.items = items
        self.sr = sr
        self.clip_len = int(clip_duration * sr)
        self.feature_type = feature_type
        self.waveform_transform = waveform_transform
        self.spec_transform = spec_transform
        self.n_mels = n_mels

    def __len__(self):
        return len(self.items)

    def _compute_features(self, wav: np.ndarray) -> np.ndarray:
        if self.feature_type == "logmel":
            spec = logmel_spectrogram(
                wav,
                sr=self.sr,
                n_fft=1024,
                hop_length=256,
                n_mels=self.n_mels,
                fmin=50.0,
                fmax=None,
            )
        elif self.feature_type == "mfcc":
            spec = mfcc_from_waveform(
                wav,
                sr=self.sr,
                n_mfcc=self.n_mels,
                n_fft=1024,
                hop_length=256,
                n_mels=self.n_mels,
                fmin=50.0,
                fmax=None,
            )
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

        if self.spec_transform is not None:
            spec = self.spec_transform(spec)
        return spec

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        wav, orig_sr = load_audio(path, sr=self.sr, mono=True)
        wav = normalize_peak(wav, peak=0.99)
        wav = fix_length(wav, self.clip_len)

        if self.waveform_transform is not None:
            wav = self.waveform_transform(wav, self.sr) if callable(
                getattr(self.waveform_transform, "__call__", None)
            ) else self.waveform_transform(wav)

        spec = self._compute_features(wav)   # (freq x time)
        # torch.Tensor [1, F, T] — удобно под CNN
        spec_tensor = torch.from_numpy(spec).unsqueeze(0)  # (1, F, T)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return spec_tensor, label_tensor


# ==========
# Метрики для KWS/классификации
# ==========

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Обычная accuracy. y_pred — предсказанный класс.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape
    return float((y_true == y_pred).mean())


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Конфматрица [n_classes x n_classes], без sklearn.
    """
    y_true = np.asarray(y_true, dtype="int64")
    y_pred = np.asarray(y_pred, dtype="int64")
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max()) + 1)
    cm = np.zeros((n_classes, n_classes), dtype="int32")
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    """
    Recall по строкам конфматрицы.
    """
    tp = np.diag(cm)
    denom = cm.sum(axis=1) + 1e-9
    return tp / denom


def kws_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Сводка метрик для KWS:
    - accuracy
    - per_class_recall
    """
    cm = confusion_matrix(y_true, y_pred)
    rec = per_class_recall(cm)
    acc = accuracy(y_true, y_pred)
    metrics = {
        "accuracy": acc,
        "recall_per_class": rec,
        "confusion_matrix": cm,
    }
    if labels is not None and len(labels) == cm.shape[0]:
        # Можно распечатать красиво при желании
        pass
    return metrics
