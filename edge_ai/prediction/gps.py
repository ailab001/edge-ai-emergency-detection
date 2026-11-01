from __future__ import annotations

import math

from shared.schemas import GPSSummary, HomeLocation, MobileGPS

EARTH_RADIUS_M = 6_371_000

DEFAULT_DISTANCE_THRESHOLD_M = 500.0
DEFAULT_DURATION_THRESHOLD_SEC = 7200.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


class GPSAnalyzer:
    def __init__(
        self,
        home: HomeLocation,
        distance_threshold_m: float = DEFAULT_DISTANCE_THRESHOLD_M,
        duration_threshold_sec: float = DEFAULT_DURATION_THRESHOLD_SEC,
    ):
        self._home = home
        self._dist_threshold = distance_threshold_m
        self._dur_threshold = duration_threshold_sec
        self._last_home_ts: float | None = None
        self._first_gps_ts: float | None = None
        self._latest: MobileGPS | None = None

    def update(self, gps: MobileGPS) -> None:
        self._latest = gps
        if self._first_gps_ts is None:
            self._first_gps_ts = gps.ts
        dist = haversine(self._home.lat, self._home.lon, gps.lat, gps.lon)
        if dist <= self._dist_threshold:
            self._last_home_ts = gps.ts

    def evaluate(self, now: float) -> GPSSummary | None:
        if self._latest is None:
            return None

        dist = haversine(
            self._home.lat, self._home.lon,
            self._latest.lat, self._latest.lon,
        )

        if self._last_home_ts is not None:
            away_duration = now - self._last_home_ts
        elif self._first_gps_ts is not None:
            away_duration = now - self._first_gps_ts
        else:
            away_duration = 0.0

        is_risk = dist > self._dist_threshold and away_duration > self._dur_threshold

        return GPSSummary(
            last_lat=self._latest.lat,
            last_lon=self._latest.lon,
            home_lat=self._home.lat,
            home_lon=self._home.lon,
            distance_from_home_m=round(dist, 1),
            is_no_return_risk=is_risk,
        )
