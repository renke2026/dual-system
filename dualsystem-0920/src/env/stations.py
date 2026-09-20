import bisect
import collections
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional, Deque
from abc import ABC, abstractmethod
import datetime
from src.env.vehicle import ChargeableVehicle
from config import CONFIG


class BaseStation(ABC):
    """Charging-station base class: shared queue management and pricing, no energy dynamics."""

    def __init__(self,
                 station_id: str,
                 edge_id: str,
                 bus_id: str,
                 capacity: int,
                 max_power_kw: float,
                 station_type: str,
                 price_table: List[Tuple[int, float]] = None):

        self.id = station_id
        self.edge_id = edge_id        # corresponding SUMO edge id
        self.bus_id = bus_id          # corresponding PDN bus id
        self.capacity = capacity      # number of slots
        self.max_power_kw = max_power_kw
        self.station_type = station_type  # "FCS" or "SCS"

        # Price table: [(start second, price), ...], e.g. [(3600, 1.2), ...].
        self.price_table = price_table if price_table else [(0, 1.0)]
        self.price_times = [p[0] for p in self.price_table]

        # Dynamic control (smart charging & dynamic pricing).
        self._grid_limit_factor: float = 1.0
        self._price_multiplier: float = 1.0

        self.queue: Deque[ChargeableVehicle] = collections.deque()
        self.charging_vehicles: List[ChargeableVehicle] = []

    def get_wait_count(self) -> int:
        return len(self.queue)

    def set_grid_limit(self, factor: float):
        """Apply a grid load-limit factor."""
        self._grid_limit_factor = max(0.0, min(1.0, factor))

    def set_price_multiplier(self, mult: float):
        """Apply a dynamic price multiplier."""
        self._price_multiplier = mult

    def get_price(self, current_time_sec: int) -> float:
        """Query the current price; the time is wrapped modulo 86400 for multi-day runs."""
        time_in_day = current_time_sec % 86400

        idx = bisect.bisect_right(self.price_times, time_in_day)

        if idx == 0:
            base_price = self.price_table[0][1]
        else:
            base_price = self.price_table[idx - 1][1]

        return round(base_price * self._price_multiplier, 2)

    def has_vehicle(self, veh_id: str) -> bool:
        all_vehs = list(self.queue) + self.charging_vehicles
        return any(v.id == veh_id for v in all_vehs)

    def add_vehicle(self, vehicle: ChargeableVehicle) -> bool:
        """Vehicle enters: to a charging slot if free, otherwise to the queue."""
        if self.has_vehicle(vehicle.id):
            return False

        if len(self.charging_vehicles) < self.capacity:
            vehicle.status = "Charging"
            self.charging_vehicles.append(vehicle)
        else:
            vehicle.status = "Waiting"
            self.queue.append(vehicle)
        return True

    def remove_vehicle(self, veh_id: str) -> Optional[ChargeableVehicle]:
        """Vehicle leaves: remove and backfill the vacancy (FIFO)."""
        for i, veh in enumerate(self.charging_vehicles):
            if veh.id == veh_id:
                removed_veh = self.charging_vehicles.pop(i)
                removed_veh.last_charging_power_kw = 0.0
                self._fill_vacancy()
                return removed_veh

        try:
            target = next(v for v in self.queue if v.id == veh_id)
            self.queue.remove(target)
            target.last_charging_power_kw = 0.0
            return target
        except StopIteration:
            return None

    def _fill_vacancy(self):
        """Backfill an empty slot from the queue (FIFO)."""
        if self.queue and len(self.charging_vehicles) < self.capacity:
            next_veh = self.queue.popleft()
            next_veh.status = "Charging"
            self.charging_vehicles.append(next_veh)

    def get_current_load_kw(self) -> float:
        """Current total active power, for the grid calculation."""
        total_load = 0.0
        for veh in self.charging_vehicles:
            if veh.status == "Charging":
                limit = min(self.max_power_kw, veh.max_charge_power_kw)
                actual = limit * self._grid_limit_factor
                total_load += actual
            elif veh.status == "Discharging":
                # Discharge counts as negative load (V2G).
                limit = min(self.max_power_kw, veh.max_charge_power_kw)
                total_load -= limit

        return total_load

    def get_role_counts(self):
        """Count commuter/gig workers in queue and slots; returns (c_q, c_occ, g_q, g_occ)."""
        c_q = 0
        g_q = 0
        c_occ = 0
        g_occ = 0

        for v in self.queue:
            if v.role == "COMMUTER":
                c_q += 1
            elif v.role == "GIG_WORKER":
                g_q += 1

        for v in self.charging_vehicles:
            if v.role == "COMMUTER":
                c_occ += 1
            elif v.role == "GIG_WORKER":
                g_occ += 1

        return c_q, c_occ, g_q, g_occ

    def reset(self):
        """Daily reset: clear all vehicles (and parking for SCS)."""
        self.queue.clear()
        self.charging_vehicles.clear()
        if hasattr(self, 'parking_list'):
            self.parking_list.clear()

    @abstractmethod
    def update(self, dt_minutes: int, current_time_sec: int):
        """Physical energy update."""
        pass


class FCS(BaseStation):
    """Fast charging station: like a gas station, no V2G, full battery means done."""

    def __init__(self, station_id, edge_id, bus_id, capacity, price_table):
        super().__init__(station_id, edge_id, bus_id, capacity, CONFIG.station.fcs_max_power_kw, "FCS", price_table)

    def update(self, dt_minutes: int, current_time_sec: int):
        dt_hours = dt_minutes / 60.0
        current_price = self.get_price(current_time_sec)

        for veh in self.charging_vehicles:
            charging_efficiency = 1.0
            if veh.soc > 0.8:
                # Constant-voltage stage: power tapers off (80%->1.0, 90%->0.6, 95%->0.4).
                charging_efficiency = 1.0 - (veh.soc - 0.8) * 4

            if veh.status == "Charging":
                if veh.soc < 1.0:
                    hardware_limit = min(self.max_power_kw, veh.max_charge_power_kw)
                    actual_power = hardware_limit * self._grid_limit_factor * charging_efficiency
                    veh.last_charging_power_kw = actual_power

                    energy_added = actual_power * dt_hours
                    step_cost = energy_added * current_price

                    veh.soc += energy_added / veh.capacity
                    veh.soc = min(1.0, veh.soc)

                    veh.added_energy_kwh += energy_added
                    veh.incurred_cost += step_cost
                else:
                    # Full: wait for the agent to notice and leave.
                    pass


class SCS(BaseStation):
    """Slow charging station: like a parking lot; supports V2G, full means Done (still occupies a slot)."""

    def __init__(self, station_id, edge_id, bus_id, capacity, price_table):
        super().__init__(station_id, edge_id, bus_id, capacity, CONFIG.station.scs_max_power_kw, "SCS", price_table)
        self.parking_list: List[ChargeableVehicle] = []

    def update(self, dt_minutes: int, current_time_sec: int):
        dt_hours = dt_minutes / 60.0
        current_price = self.get_price(current_time_sec)

        for veh in self.charging_vehicles:

            if veh.status == "Charging":
                if veh.soc < 1.0:
                    hardware_limit = min(self.max_power_kw, veh.max_charge_power_kw)
                    actual_power = hardware_limit * self._grid_limit_factor
                    veh.last_charging_power_kw = actual_power

                    energy_added = actual_power * dt_hours
                    step_cost = energy_added * current_price

                    veh.soc += energy_added / veh.capacity
                    veh.soc = min(1.0, veh.soc)

                    veh.added_energy_kwh += energy_added
                    veh.incurred_cost += step_cost
                else:
                    # Full: keep occupying the slot but stop drawing power.
                    veh.status = "Done"

            elif veh.status == "Discharging":
                # Only discharge above 20% SoC (battery protection).
                if veh.soc > 0.2:
                    discharge_power = min(self.max_power_kw, veh.max_charge_power_kw)

                    energy_removed = discharge_power * dt_hours
                    step_earn = energy_removed * current_price

                    veh.soc -= energy_removed / veh.capacity
                    veh.incurred_cost -= step_earn
                    veh.added_energy_kwh -= energy_removed
                else:
                    veh.status = "Done"

            elif veh.status == "Done":
                # Idle: occupies a slot, no power, no cost.
                pass

    def has_vehicle(self, veh_id: str) -> bool:
        if super().has_vehicle(veh_id):
            return True
        return any(v.id == veh_id for v in self.parking_list)

    def remove_vehicle(self, veh_id: str) -> Optional[ChargeableVehicle]:
        """Scan and clean all containers: charging slots, queue, and parking list."""
        found_veh = None

        for i, veh in enumerate(self.charging_vehicles):
            if veh.id == veh_id:
                found_veh = self.charging_vehicles.pop(i)
                self._fill_vacancy()
                break

        try:
            target = next(v for v in self.queue if v.id == veh_id)
            self.queue.remove(target)
            if not found_veh:
                found_veh = target
        except StopIteration:
            pass

        for i, veh in enumerate(self.parking_list):
            if veh.id == veh_id:
                self.parking_list.pop(i)
                if not found_veh:
                    found_veh = veh
                break

        if found_veh:
            found_veh.last_charging_power_kw = 0.0
        return found_veh

    def add_vehicle(self, vehicle: ChargeableVehicle) -> bool:
        """SCS entry: if full, the vehicle parks without charging instead of waiting."""
        if self.has_vehicle(vehicle.id):
            return False

        if len(self.charging_vehicles) < self.capacity:
            vehicle.status = "Charging"
            self.charging_vehicles.append(vehicle)
            return True
        else:
            vehicle.status = "ParkingWithoutCharging"
            self.parking_list.append(vehicle)
            self.queue.append(vehicle)
            return True


class CSManager:
    def __init__(self, station_file: str = "case/12nodes.cs.price.xml"):
        """Charging-station manager; loads a single XML with all stations and prices."""
        self.stations: Dict[str, BaseStation] = {}
        print(f"🔌 [Env] Loading charging station data: {station_file} ...")
        self._load_stations(station_file)
        print(f"✅ Charging stations loaded, {len(self.stations)} stations total.")

    def _load_stations(self, filename: str):
        try:
            tree = ET.parse(filename)
            root = tree.getroot()

            for elem in root:
                tag = elem.tag.lower()

                if tag not in ['fcs', 'scs']:
                    continue

                s_id = elem.get('name')
                edge_id = elem.get('edge')
                bus_id = elem.get('bus', None)

                slots = int(elem.get('slots', CONFIG.station.default_slots))

                price_table = self._parse_price_table(elem.find('pbuy'))

                if tag == 'fcs':
                    station = FCS(s_id, edge_id, bus_id, slots, price_table)
                elif tag == 'scs':
                    station = SCS(s_id, edge_id, bus_id, slots, price_table)
                else:
                    continue

                self.stations[s_id] = station

        except FileNotFoundError:
            print(f"❌ Error: file not found {filename}. Run 'python scripts/overwrite_prices.py' first to generate it.")
        except Exception as e:
            print(f"❌ Parse error {filename}: {e}")

    def _parse_price_table(self, pbuy_elem) -> List[Tuple[int, float]]:
        """Parse <pbuy><item btime="0" price="1.0" />...</pbuy>."""
        table = []
        if pbuy_elem is None:
            return [(0, 1.0)]

        for item in pbuy_elem.findall('item'):
            time = int(item.get('btime'))
            price = float(item.get('price'))
            table.append((time, price))

        table.sort(key=lambda x: x[0])
        return table

    def get_station(self, station_id: str) -> Optional[BaseStation]:
        return self.stations.get(station_id)

    def find_station_on_edge(self, edge_id: str) -> Optional[BaseStation]:
        for s in self.stations.values():
            if s.edge_id == edge_id:
                return s
        return None

    def update_all(self, dt_minutes: int, current_time: datetime.datetime):
        day_seconds = current_time.hour * 3600 + current_time.minute * 60 + current_time.second

        for station in self.stations.values():
            station.update(dt_minutes, day_seconds)

    def get_observation_data(self, current_time: datetime.datetime) -> List[dict]:
        infos = []
        day_seconds = current_time.hour * 3600 + current_time.minute * 60

        for s in self.stations.values():
            current_price = s.get_price(day_seconds)
            wait_min = (s.get_wait_count() / s.capacity) * CONFIG.station.wait_time_factor
            infos.append({
                "station_id": s.id,
                "station_type": s.station_type,
                "distance_km": 0.0,  # filled in by the map
                "price_per_kwh": current_price,
                "queue_length": s.get_wait_count(),
                "estimated_wait_time_min": int(wait_min)
            })
        return infos

    def reset_all(self):
        print("⚡ [System] Performing network-wide hard reset of charging stations...")
        for station in self.stations.values():
            station.reset()
        print("✅ All stations cleared.")
