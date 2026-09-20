"""Conventional non-LLM behavioral baseline: Multinomial Logit (MNL) station choice.

Utility: U_i = -beta_time * time_i - beta_price * price_i - beta_distance * distance_i
  time_i = queue wait (min) + travel time (distance / avg_speed * 60)
Choice probability: P_i = softmax(U_i / temperature), sampled by probability.
"""

import math
import random


class MNLStationSelector:
    """MNL station selector: sample a station by softmax(utility) over candidates."""

    def __init__(self, seed=None):
        # Each agent gets its own Random instance, deterministically seeded for reproducibility.
        self.rng = random.Random(seed)

    def _utility(self, station, beta_time, beta_price, beta_distance, avg_speed_kmh):
        """Linear utility of a single station (higher is better); station is ChargingStationInfo."""
        travel_min = station.distance_km / avg_speed_kmh * 60.0
        time_cost = travel_min + station.estimated_wait_time_min
        price = station.price_per_kwh
        dist = station.distance_km
        return -beta_time * time_cost - beta_price * price - beta_distance * dist

    def select(self, stations, beta_time, beta_price, beta_distance,
               avg_speed_kmh=30.0, temperature=1.0):
        """Sample a station id by MNL probability; returns None for an empty list."""
        if not stations:
            return None

        utils = [
            self._utility(s, beta_time, beta_price, beta_distance, avg_speed_kmh)
            for s in stations
        ]

        # Softmax (subtract max to avoid overflow); temperature controls randomness.
        max_u = max(utils)
        scale = 1.0 / max(temperature, 1e-6)
        exps = [math.exp((u - max_u) * scale) for u in utils]
        total = sum(exps)

        r = self.rng.random()
        cum = 0.0
        for station, e in zip(stations, exps):
            cum += e / total
            if r <= cum:
                return station.station_id
        return stations[-1].station_id
