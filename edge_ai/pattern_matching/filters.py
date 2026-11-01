from __future__ import annotations

from typing import Protocol


class SignalFilter(Protocol):
    def update(self, value: float, ts: float) -> float: ...
    def reset(self) -> None: ...


class KalmanFilter:
    def __init__(
        self,
        process_noise: float = 1e-3,
        measurement_noise: float = 0.1,
    ):
        self._q = process_noise
        self._r = measurement_noise
        self._x: float | None = None
        self._p: float = 1.0

    def update(self, value: float, ts: float) -> float:
        if self._x is None:
            self._x = value
            self._p = 1.0
            return value

        self._p += self._q
        k = self._p / (self._p + self._r)
        self._x += k * (value - self._x)
        self._p *= 1.0 - k
        return self._x

    def reset(self) -> None:
        self._x = None
        self._p = 1.0


class EWMAFilter:
    def __init__(self, span: int = 10):
        self._alpha = 2.0 / (span + 1)
        self._value: float | None = None

    def update(self, value: float, ts: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value = self._alpha * value + (1.0 - self._alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._value = None
