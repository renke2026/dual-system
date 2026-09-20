"""The dispatch center: wires together topology, solver, and charging stations."""

import math
from typing import Dict, Any, List
from src.env.grid import GridTopology
from src.env.solver import PowerFlowSolver
from src.env.stations import CSManager
from src.common import NewsMessage
import datetime
import collections
import numpy as np
from config import CONFIG


class GridManager:
    def __init__(self, grid_file: str, cs_manager: CSManager):
        """Dispatch center: computes voltages, sets prices, and issues power limits."""

        self.topology = GridTopology(grid_file)
        self.solver = PowerFlowSolver(self.topology, base_mva=self.topology.sb_mva)
        self.cs_manager = cs_manager

        self.last_voltages: Dict[str, float] = {}  # BusID -> Voltage (p.u.)
        self.min_voltage = 1.0

        self.V_SAFE = CONFIG.grid.v_safe          # below this, prices start rising
        self.V_CRITICAL = CONFIG.grid.v_critical  # below this, load limiting kicks in

        # Price history per bus for moving-average filtering.
        self.price_history = collections.defaultdict(
            lambda: collections.deque(maxlen=CONFIG.grid.price_history_len)
        )
        self.last_broadcast_time = None

    def update_power_flow(self, current_time):
        """Run one power-flow calculation (dynamic load indexed by time of day)."""
        minute_idx = current_time.hour * 60 + current_time.minute

        load_p_mw = {}
        load_q_mvar = {}

        # A. Base load, looked up directly from the precomputed arrays.
        for bid, bus in self.topology.buses.items():
            bus.pd_base_mw = bus.pd_array[minute_idx]
            bus.qd_base_mvar = bus.qd_array[minute_idx]

            load_p_mw[bid] = bus.pd_base_mw
            load_q_mvar[bid] = bus.qd_base_mvar

        # B. Add charging-station load (already includes the prior grid limit).
        for station in self.cs_manager.stations.values():
            bus_id = station.bus_id
            if bus_id and bus_id in self.topology.buses:
                station_load_mw = station.get_current_load_kw() / 1000.0
                load_p_mw[bus_id] += station_load_mw

        # Only P is passed; the solver fills in Q assuming PF=0.95.
        self.last_voltages = self.solver.solve(load_p_mw)

        if self.last_voltages:
            self.min_voltage = min(self.last_voltages.values())
        else:
            self.min_voltage = 0.0

    def calculate_price_multiplier(self) -> Dict[str, float]:
        """Compute per-bus price multipliers (LMP signal).

        Physical layer: piecewise function of voltage.
          - V >= 0.95: 1.0x (normal)
          - 0.90 <= V < 0.95: linear growth to 3.0x (mild warning)
          - V < 0.90: exponential growth to 10.0x (strong deterrent)
        Signal layer: moving average over the last few minutes to smooth spikes.
        """
        final_multipliers = {}
        if not self.last_voltages:
            return {}

        for bus_id, v in self.last_voltages.items():
            raw_mult = 1.0

            if v >= self.V_SAFE:
                # Zone 1: safe.
                raw_mult = 1.0

            elif v >= self.V_CRITICAL:
                # Zone 2: linear price growth (v=0.95 -> 1.0, v=0.90 -> 3.0).
                slope = (CONFIG.grid.price_linear_high - 1.0) / (self.V_SAFE - self.V_CRITICAL)
                deviation = self.V_SAFE - v
                raw_mult = 1.0 + (deviation * slope)

            else:
                # Zone 3: exponential surge on top of the linear base.
                deviation = self.V_CRITICAL - v
                raw_mult = CONFIG.grid.price_linear_high * math.exp(CONFIG.grid.price_surge_exp * deviation)

            # Cap to prevent numerical blow-up.
            raw_mult = min(CONFIG.grid.price_max, raw_mult)

            self.price_history[bus_id].append(raw_mult)

            history_queue = self.price_history[bus_id]
            avg_mult = sum(history_queue) / len(history_queue)

            final_multipliers[bus_id] = round(avg_mult, 2)

        return final_multipliers

    def apply_smart_charging(self):
        """Closed-loop control: dispatch price signals and physical load limits."""
        if not self.last_voltages:
            return

        price_map = self.calculate_price_multiplier()

        limit_triggered_count = 0

        for station in self.cs_manager.stations.values():
            bus_id = station.bus_id
            if not bus_id or bus_id not in self.last_voltages:
                continue

            v = self.last_voltages[bus_id]

            # A. Dynamic price (1.0 if no multiplier).
            mult = price_map.get(bus_id, 1.0)
            station.set_price_multiplier(mult)

            # B. Physical load limit: P-controller below V_CRITICAL.
            if v < self.V_CRITICAL:
                factor = 1.0 - (self.V_CRITICAL - v) * CONFIG.grid.limit_slope
                factor = max(CONFIG.grid.limit_floor, min(1.0, factor))

                station.set_grid_limit(factor)
                limit_triggered_count += 1
            else:
                station.set_grid_limit(1.0)

    def get_current_multiplier(self) -> float:
        """Return the highest price multiplier across the network (for observations)."""
        multipliers = self.calculate_price_multiplier()
        if not multipliers:
            return 1.0
        return max(multipliers.values())

    def detect_and_generate_events(self, current_time: datetime.datetime) -> List[NewsMessage]:
        """Detect grid anomalies and generate broadcast news (with cooldown to avoid spam)."""
        if self.last_broadcast_time:
            delta_seconds = (current_time - self.last_broadcast_time).total_seconds()
            if delta_seconds < CONFIG.grid.broadcast_cooldown_s:
                return []

        events = []

        if not self.last_voltages:
            return events

        # A. Price surge.
        price_map = self.calculate_price_multiplier()
        if price_map:
            max_multiplier = max(price_map.values())
            if max_multiplier > CONFIG.grid.price_surge_threshold:
                print(f"⚡ [Grid Detection] Price-surge trigger: network-wide max multiplier {max_multiplier:.2f}x (threshold: {CONFIG.grid.price_surge_threshold})")
                msg = NewsMessage(
                    timestamp=current_time,
                    source="Grid Operator",
                    category="PRICE_SURGE",
                    content=f"URGENT: Grid load is extremely high! Electricity prices have surged to {max_multiplier:.1f}x base rate. Charging is NOT recommended.",
                    priority="CRITICAL"
                )
                events.append(msg)

        # B. Voltage-collapse risk (grid alert).
        if self.min_voltage < CONFIG.grid.grid_alert_voltage:
            print(f"⚡ [Grid Detection] Voltage-alert trigger: local minimum voltage {self.min_voltage:.3f} p.u. (critical line: {CONFIG.grid.grid_alert_voltage})")
            msg = NewsMessage(
                timestamp=current_time,
                source="Grid Operator",
                category="GRID_ALERT",
                content=f"CRITICAL WARNING: Brownout imminent. Voltage dropped to {self.min_voltage:.3f} p.u. Please stop charging immediately.",
                priority="CRITICAL"
            )
            events.append(msg)

        # Only update the cooldown timestamp when an event was actually sent.
        if events:
            self.last_broadcast_time = current_time

        return events

    def get_grid_intuition(self) -> str:
        """Map physical data to a qualitative macro-environment perception."""
        if not self.last_voltages:
            return "STABLE (Off-peak)"

        if self.min_voltage < CONFIG.grid.intuition_critical:
            return "CRITICAL OVERLOAD (Extreme prices, Risk of outage)"
        elif self.min_voltage < CONFIG.grid.intuition_heavy:
            return "HEAVY LOAD (Peak Hours, Prices surging)"
        else:
            return "STABLE (Normal Operation)"

    def get_total_system_load(self) -> float:
        """Total active load (MW) = base load + charging load."""
        total_mw = 0.0

        for bus in self.topology.buses.values():
            total_mw += bus.pd_base_mw

        for station in self.cs_manager.stations.values():
            total_mw += station.get_current_load_kw() / 1000.0

        return total_mw

    def reset(self):
        """Clear price history and cached state for a fresh day."""
        self.price_history.clear()
        self.last_voltages = {}
        self.min_voltage = 1.0
        self.last_broadcast_time = None
        print("⚡ [Grid] Grid state cache cleared.")
