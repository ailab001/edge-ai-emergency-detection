from __future__ import annotations

from collections import deque
from typing import Protocol

import numpy as np
from sklearn.ensemble import IsolationForest

from shared.schemas import ProcessedChannel


class AnomalyDetector(Protocol):
    def fit_predict(
        self, channels: list[ProcessedChannel]
    ) -> list[ProcessedChannel]: ...


class IsolationForestDetector:
    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        window_size: int = 300,
        min_samples: int = 30,
    ):
        self._n_estimators = n_estimators
        self._contamination = contamination
        self._window_size = window_size
        self._min_samples = min_samples
        self._buffer: deque[list[float]] = deque(maxlen=window_size)
        self._model: IsolationForest | None = None

    def fit_predict(
        self, channels: list[ProcessedChannel]
    ) -> list[ProcessedChannel]:
        if not channels:
            return channels

        feature_vector = [ch.ewma_value for ch in channels]
        self._buffer.append(feature_vector)

        if len(self._buffer) < self._min_samples:
            return channels

        dim = len(feature_vector)
        consistent = [v for v in self._buffer if len(v) == dim]
        if len(consistent) < self._min_samples:
            return channels

        X = np.array(consistent)
        self._model = IsolationForest(
            n_estimators=self._n_estimators,
            contamination=self._contamination,
            random_state=42,
        )
        self._model.fit(X)

        current = np.array(feature_vector).reshape(1, -1)
        raw_score = self._model.decision_function(current)[0]
        label = self._model.predict(current)[0]
        composite_score = float(np.clip(-raw_score, -1.0, 1.0))
        is_anomaly = label == -1

        contributions = self._channel_contributions(X, feature_vector)
        for i, ch in enumerate(channels):
            ch.anomaly_score = float(np.clip(
                composite_score * contributions[i], -1.0, 1.0,
            ))
            ch.is_anomaly = is_anomaly and contributions[i] > 1.0 / len(channels)

        return channels

    @staticmethod
    def _channel_contributions(
        X: np.ndarray, current: list[float]
    ) -> list[float]:
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds = np.where(stds < 1e-9, 1.0, stds)

        deviations = np.abs((np.array(current) - means) / stds)
        total = deviations.sum()
        if total < 1e-9:
            n = len(current)
            return [1.0 / n] * n
        return (deviations / total).tolist()
