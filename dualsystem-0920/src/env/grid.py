import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from config import CONFIG


@dataclass
class Bus:
    """A grid node (bus)."""
    id: str
    base_kv: float
    v_pu: float = 1.0
    v_min: float = 0.9
    v_max: float = 1.1

    # Precomputed 1440-minute load profiles (per minute).
    pd_array: np.ndarray = None  # active power
    qd_array: np.ndarray = None  # reactive power

    # Base load (excluding charging stations).
    pd_base_mw: float = 0.0
    qd_base_mvar: float = 0.0


@dataclass
class Line:
    """A transmission/distribution line."""
    id: str
    from_bus: str
    to_bus: str
    r_ohm: float
    x_ohm: float
    max_i_ka: float = float('inf')


@dataclass
class Gen:
    """A generator."""
    id: str
    bus_id: str
    p_max_mw: float
    p_min_mw: float


class GridTopology:
    def __init__(self, grid_file: str):
        self.grid_file = grid_file

        # Global base values (defaults from config; XML may override sb_mva).
        self.sb_mva: float = CONFIG.grid.sb_mva
        self.ub_kv: float = CONFIG.grid.ub_kv

        self.buses: Dict[str, Bus] = {}
        self.lines: List[Line] = []
        self.gens: List[Gen] = []

        # Bus ID -> list of station IDs connected to it (filled by CSManager).
        self.bus_station_map: Dict[str, List[str]] = {}

        print(f"⚡ [Grid] Loading grid topology: {grid_file}")
        self._load_xml()
        self._precompute_profiles()

    def _parse_value(self, val_str: str) -> float:
        """Strip a unit suffix and convert to float (e.g. '10.2MW' -> 10.2)."""
        if val_str == "inf":
            return float("inf")

        if val_str is None:
            return 0.0

        units = ["MW", "Mvar", "MVA", "kV", "ohm", "$/MWh2", "$/MWh", "$"]
        clean_str = val_str
        for u in units:
            clean_str = clean_str.replace(u, "")

        try:
            return float(clean_str)
        except ValueError:
            return 0.0

    def _load_xml(self):
        tree = ET.parse(self.grid_file)
        root = tree.getroot()

        sb_str = root.get("Sb", "1.0MVA")
        self.sb_mva = float(sb_str.replace("MVA", ""))

        for elem in root.findall("bus"):
            bid = elem.get("ID")
            v_pu = self._parse_value(elem.get("V", "1.0"))
            v_min = self._parse_value(elem.get("MinV", "0.9"))
            v_max = self._parse_value(elem.get("MaxV", "1.1"))

            # Load profiles support both a constant value and time-series items.
            pd_profile = {}
            qd_profile = {}

            pd_elem = elem.find("Pd")
            if pd_elem is not None:
                if "const" in pd_elem.attrib:
                    val = self._parse_value(pd_elem.get("const"))
                    for h in range(25):
                        pd_profile[h * 3600] = val
                else:
                    for item in pd_elem.findall("item"):
                        pd_profile[int(item.get("time"))] = self._parse_value(item.get("value"))

            qd_elem = elem.find("Qd")
            if qd_elem is not None:
                if "const" in qd_elem.attrib:
                    val = self._parse_value(qd_elem.get("const"))
                    for h in range(25):
                        qd_profile[h * 3600] = val
                else:
                    for item in qd_elem.findall("item"):
                        qd_profile[int(item.get("time"))] = self._parse_value(item.get("value"))

            bus = Bus(id=bid, base_kv=self.ub_kv, v_pu=v_pu, v_min=v_min, v_max=v_max)
            bus._temp_pd = pd_profile
            bus._temp_qd = qd_profile
            self.buses[bid] = bus

        for elem in root.findall("line"):
            line = Line(
                id=elem.get("ID"),
                from_bus=elem.get("From"),
                to_bus=elem.get("To"),
                r_ohm=self._parse_value(elem.get("R")),
                x_ohm=self._parse_value(elem.get("X")),
                max_i_ka=self._parse_value(elem.get("MaxIkA", "inf"))
            )
            self.lines.append(line)

        for elem in root.findall("gen"):
            pmin = elem.find("Pmin")
            pmax = elem.find("Pmax")

            gen = Gen(
                id=elem.get("ID"),
                bus_id=elem.get("Bus"),
                p_min_mw=self._parse_value(pmin.get("const")) if pmin is not None else 0.0,
                p_max_mw=self._parse_value(pmax.get("const")) if pmax is not None else 0.0
            )
            self.gens.append(gen)

        print(f"✅ Grid loaded: {len(self.buses)} Nodes, {len(self.lines)} Lines, {self.sb_mva} MVA Base.")

    def get_bus(self, bus_id: str) -> Optional[Bus]:
        return self.buses.get(bus_id)

    def _precompute_profiles(self):
        """Interpolate the full-day load onto a per-minute array for fast lookup."""
        minutes_in_day = 1440
        x_minutes = np.arange(minutes_in_day)

        for bus in self.buses.values():
            times = sorted(bus._temp_pd.keys())
            vals = [bus._temp_pd[t] for t in times]
            time_mins = [t // 60 for t in times]
            bus.pd_array = np.interp(x_minutes, time_mins, vals)

            q_times = sorted(bus._temp_qd.keys())
            q_vals = [bus._temp_qd[t] for t in q_times]
            q_time_mins = [t // 60 for t in q_times]
            bus.qd_array = np.interp(x_minutes, q_time_mins, q_vals)

            del bus._temp_pd
            del bus._temp_qd
