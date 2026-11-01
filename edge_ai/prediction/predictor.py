from __future__ import annotations

from typing import Protocol

import numpy as np


class TimeSeriesPredictor(Protocol):
    def predict(self, values: list[float], horizon: int) -> list[float]: ...


class FFTPredictor:
    def __init__(self, top_k: int = 8):
        self._top_k = top_k

    def predict(self, values: list[float], horizon: int) -> list[float]:
        n = len(values)
        if n == 0:
            return [0.0] * horizon
        if n < 4:
            mean = sum(values) / max(len(values), 1)
            return [mean] * horizon

        arr = np.array(values, dtype=np.float64)
        freqs = np.fft.rfft(arr)

        magnitudes = np.abs(freqs)
        magnitudes[0] = 0
        top_indices = np.argsort(magnitudes)[-self._top_k :]
        top_indices = np.append([0], top_indices)

        filtered = np.zeros_like(freqs)
        for idx in top_indices:
            if idx < len(freqs):
                filtered[idx] = freqs[idx]

        extended_len = n + horizon
        padded = np.zeros(extended_len // 2 + 1, dtype=np.complex128)
        copy_len = min(len(filtered), len(padded))
        padded[:copy_len] = filtered[:copy_len]

        scale = extended_len / n
        reconstructed = np.fft.irfft(padded, n=extended_len) * scale
        return reconstructed[n:].tolist()
