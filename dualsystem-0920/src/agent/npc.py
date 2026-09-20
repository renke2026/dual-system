import random
import datetime
import traci
from typing import Literal, Optional

from src.env.stations import CSManager
from src.env.map import CityMap
from src.env.vehicle import ChargeableVehicle


class NPCAgent:
    """Rule-based NPC: COMMUTER does a pendulum commute (Home <-> Work); GIG_WORKER wanders between random jobs."""

    def __init__(self, npc_id: str, role_type: str, strategy: str, initial_soc: float,
                 initial_edge_id: str, work_edge_id: str = None,
                 max_daily_trips: int = 2, consumption_rate: float = 0.01, seed: int = 42, capacity: float = 60.0,
                 first_trip_time: str = "08:00",
                 return_trip_time: Optional[str] = None):

        self.id = npc_id
        self.role_type = role_type
        self.avatar = ChargeableVehicle(npc_id, initial_soc, capacity, role=role_type)
        self.strategy = strategy
        self.seed_value = seed
        self.rng = random.Random(seed)
        self.initial_soc_backup = initial_soc

        self.home_edge = initial_edge_id
        self.work_edge = work_edge_id
        self.current_edge_id = initial_edge_id

        self.max_daily_trips = max_daily_trips
        self.completed_trips = 0

        # Parse "HH:MM" into an (hour, minute) tuple for comparisons.
        try:
            h, m = map(int, first_trip_time.split(':'))
            self.start_time_tuple = (h, m)
        except:
            self.start_time_tuple = (8, 0)

        self.return_time_tuple = None
        if return_trip_time:
            try:
                h, m = map(int, return_trip_time.split(':'))
                self.return_time_tuple = (h, m)
            except:
                pass

        self.state = "IDLE"
        self.target_station_id = None
        self.is_in_sumo = False

        self.parked_station_id = None
        self.current_trip_destination = None
        self.stuck_counter = 0

    def reset_daily_state(self, sumo_manager, cs_manager):
        """Daily reset: restore the initial state (ghost-occupancy cleanup first)."""

        # Clear any station occupancy (queue / charging / SCS parking).
        if self.target_station_id:
            station = cs_manager.get_station(self.target_station_id)
            if station:
                station.remove_vehicle(self.id)
        if self.parked_station_id:
            station = cs_manager.get_station(self.parked_station_id)
            if station:
                station.remove_vehicle(self.id)

        # Remove from SUMO if still on the road.
        if self.is_in_sumo:
            try:
                sumo_manager.remove_vehicle_from_sumo(self.id)
            except:
                pass
            self.is_in_sumo = False

        self.state = "IDLE"
        self.target_station_id = None
        self.current_edge_id = self.home_edge
        self.avatar.soc = self.initial_soc_backup
        self.avatar.status = "Waiting"
        self._has_initialized_parking = False
        self.completed_trips = 0

        # Re-seed the RNG so each day repeats the same sequence.
        if hasattr(self, 'seed_value'):
            self.rng = random.Random(self.seed_value)
        else:
            self.rng = random.Random(hash(self.id))

    def _update_physics_consumption(self, dt_minutes: int):
        """Physical energy consumption."""
        base_consumption = 0.0001 * dt_minutes
        drive_consumption = 0.0

        if self.state in ["DRIVING", "TRANSIT_TO_CHARGE"] and self.is_in_sumo:
            try:
                if self.id in traci.vehicle.getIDList():
                    speed_mps = traci.vehicle.getSpeed(self.id)
                    if speed_mps > 0:
                        dist_km = (speed_mps * (dt_minutes * 60)) / 1000.0
                        speed_kmh = speed_mps * 3.6
                        rate_kwh_100km = 15.0
                        if speed_kmh < 10:
                            rate_kwh_100km = 20.0
                        elif speed_kmh > 100:
                            rate_kwh_100km = 18.0 + (speed_kmh - 100) * 0.2
                        energy_kwh = dist_km * (rate_kwh_100km / 100.0)
                        drive_consumption = energy_kwh / self.avatar.capacity
                        drive_consumption = drive_consumption * 1.3
            except Exception:
                pass

        total_drop = base_consumption + drive_consumption
        self.avatar.soc -= total_drop
        self.avatar.soc = max(0.0, min(1.0, self.avatar.soc))

    def _is_time_to_act(self, current_time: datetime.datetime, target_tuple):
        if not target_tuple:
            return False
        curr_tuple = (current_time.hour, current_time.minute)
        return curr_tuple >= target_tuple

    def update(self, cs_manager, city_map, sumo_manager,
               dt_minutes, current_time, logger=None):

        # Initial parking: force overnight SCS parking on the first idle step.
        if self.state == "IDLE" and self.completed_trips == 0 and not self.is_in_sumo:
            if not getattr(self, '_has_initialized_parking', False):
                self._has_initialized_parking = True

                station = cs_manager.find_station_on_edge(self.current_edge_id)
                if station and station.station_type == "SCS":
                    if station.add_vehicle(self.avatar):
                        self.state = "IDLE"
                        self.parked_station_id = station.id
                        if self.avatar.soc < 0.95:
                            self.arrival_time = current_time
                            self.avatar.reset_charging_session()

        hour = current_time.hour
        self._update_physics_consumption(dt_minutes)

        # Schedule override: allowed only when stationary and interruptible.
        is_interruptible = False
        if self.state == "IDLE":
            is_interruptible = True
        elif self.state == "CHARGING":
            station = cs_manager.get_station(self.target_station_id)
            if station and station.station_type == "SCS":
                is_interruptible = True  # slow charging can be interrupted anytime

        if self.state == "WAITING_INIT":
            is_interruptible = True

        if is_interruptible and not self.is_in_sumo:
            target_dest = None

            if self.role_type == "COMMUTER":
                if self.completed_trips == 0 and hour < 16:
                    if self._is_time_to_act(current_time, self.start_time_tuple):
                        target_dest = self.work_edge
                elif self.completed_trips >= 1:
                    if self._is_time_to_act(current_time, self.return_time_tuple):
                        target_dest = self.home_edge
            else:
                if self.completed_trips == 0 and hour < 16:
                    if self._is_time_to_act(current_time, self.start_time_tuple):
                        candidate_edges = [loc.edge_id for loc in city_map.locations]
                        if candidate_edges:
                            dest = random.choice(candidate_edges)
                            if dest != self.current_edge_id:
                                target_dest = dest
                elif 6 <= hour <= 21 and self.completed_trips < self.max_daily_trips:
                    if self.rng.random() < 0.1:
                        candidate_edges = [loc.edge_id for loc in city_map.locations]
                        if candidate_edges:
                            dest = self.current_edge_id
                            for _ in range(20):
                                possible_dest = self.rng.choice(candidate_edges)
                                if possible_dest == self.current_edge_id:
                                    continue
                                if "rev" in possible_dest:
                                    continue
                                if possible_dest.startswith(":"):
                                    continue
                                dest = possible_dest
                                break
                            if dest != self.current_edge_id:
                                target_dest = dest
                elif hour >= 22 and self.current_edge_id != self.home_edge:
                    target_dest = self.home_edge

            if target_dest:
                is_started = self._start_trip(sumo_manager, target_dest)

                if not is_started:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = 0

                # Anti-deadlock: force teleport after ~30 minutes stuck.
                force_leave = False
                if self.stuck_counter > 30:
                    force_leave = True
                    self.state = "DRIVING"
                    self.is_in_sumo = False
                    self.current_trip_destination = target_dest
                    self.stuck_counter = 0

                # Only physically detach when the trip actually started (or teleported).
                if is_started or force_leave:
                    if self.parked_station_id:
                        station = cs_manager.get_station(self.parked_station_id)
                        if station:
                            station.remove_vehicle(self.id)
                        self.parked_station_id = None

                    if self.target_station_id:
                        station = cs_manager.get_station(self.target_station_id)
                        if station:
                            station.remove_vehicle(self.id)
                        self.target_station_id = None

        # Driving / charging state machine.
        if self.state in ["DRIVING", "TRANSIT_TO_CHARGE"]:
            self._check_physical_arrival(sumo_manager, cs_manager, city_map, current_time)

            if self.state == "DRIVING" and self.avatar.soc < 0.15:
                self._make_charging_decision(cs_manager, city_map, current_time, sumo_manager)

        elif self.state == "CHARGING":
            # Only FCS enters CHARGING; SCS parking keeps IDLE (see _enter_charging_station).
            station = cs_manager.get_station(self.target_station_id)
            if station:
                if station.station_type == "FCS":
                    if self.avatar.soc >= 0.95:
                        station.remove_vehicle(self.id)
                        self.state = "IDLE"
                        self.target_station_id = None
            else:
                self.state = "IDLE"

        self.avatar.soc = max(0.0, min(1.0, self.avatar.soc))

    def _start_trip(self, sumo_manager, dest_edge) -> bool:
        if dest_edge == self.current_edge_id:
            return False

        if sumo_manager.add_vehicle_to_sumo(self.id, self.current_edge_id, dest_edge):
            self.state = "DRIVING"
            self.is_in_sumo = True
            self.current_trip_destination = dest_edge
            return True
        else:
            return False

    def _check_physical_arrival(self, sumo_manager, cs_manager, city_map, current_time):
        active_ids = traci.vehicle.getIDList()
        pending_ids = traci.simulation.getPendingVehicles()

        if self.id not in active_ids and self.id not in pending_ids:
            if self.is_in_sumo or self.current_trip_destination:
                self.is_in_sumo = False

                if self.state == "TRANSIT_TO_CHARGE":
                    station = cs_manager.get_station(self.target_station_id)
                    if station:
                        self._enter_charging_station(station, current_time, sumo_manager)
                else:
                    self.state = "IDLE"
                    self.completed_trips += 1

                    # Vehicle gone from SUMO, so update the edge manually (teleport).
                    if self.current_trip_destination:
                        self.current_edge_id = self.current_trip_destination
                        self.current_trip_destination = None

                    # Auto-park in SCS on arrival (commuters only).
                    station = cs_manager.find_station_on_edge(self.current_edge_id)
                    if station and station.station_type == "SCS" and self.role_type == "COMMUTER":
                        if station.add_vehicle(self.avatar):
                            self.parked_station_id = station.id

                            if self.avatar.soc >= 0.95:
                                self.state = "IDLE"
                            else:
                                self.target_station_id = station.id
                                self.state = "CHARGING"
                                self.arrival_time = current_time
                                self.avatar.reset_charging_session()
            return

        if self.id in active_ids:
            try:
                curr_edge = traci.vehicle.getRoadID(self.id)
                if curr_edge and not curr_edge.startswith(":"):
                    self.current_edge_id = curr_edge

                    if self.state == "TRANSIT_TO_CHARGE":
                        station = cs_manager.get_station(self.target_station_id)
                        if station and curr_edge == station.edge_id:
                            self._enter_charging_station(station, current_time, sumo_manager)
            except traci.exceptions.TraCIException:
                pass

    def _make_charging_decision(self, cs_manager, city_map, current_time, sumo_manager):
        all_stations = list(cs_manager.stations.values())
        best_station = None
        min_score = float('inf')
        seconds = current_time.hour * 3600 + current_time.minute * 60

        for station in all_stations:
            if self.strategy == "NEAREST":
                dist = city_map.get_route_distance(self.current_edge_id, station.edge_id)
                if dist < min_score:
                    min_score, best_station = dist, station
            elif self.strategy == "CHEAPEST":
                price = station.get_price(seconds)
                if price < min_score:
                    min_score, best_station = price, station

        if best_station:
            if sumo_manager.change_target(self.id, best_station.edge_id):
                self.state = "TRANSIT_TO_CHARGE"
                self.target_station_id = best_station.id

    def _enter_charging_station(self, station, current_time, sumo_manager):
        if station.add_vehicle(self.avatar):
            self.arrival_time = current_time
            self.avatar.reset_charging_session()
            sumo_manager.remove_vehicle_from_sumo(self.id)
            self.is_in_sumo = False

            # Any trip ending at a station counts as a completed trip.
            if self.state in ["DRIVING", "TRANSIT_TO_CHARGE"]:
                if self.role_type == "COMMUTER":
                    if station.edge_id == self.work_edge or station.edge_id == self.home_edge:
                        self.completed_trips += 1
                else:
                    self.completed_trips += 1

            if station.station_type == "FCS":
                self.state = "CHARGING"  # FCS is a task: wait until full
                self.target_station_id = station.id
            else:
                self.state = "IDLE"  # SCS is parking: leave anytime
                self.parked_station_id = station.id
        else:
            self.state = "IDLE"

    def _leave_charging_station(self, cs_manager, current_time, sumo_manager, logger):
        station = cs_manager.get_station(self.target_station_id)
        if station:
            if logger:
                duration = 0
                if hasattr(self, 'arrival_time'):
                    duration = (current_time - self.arrival_time).total_seconds() / 60

                time_str = current_time.strftime("%H:%M")
                logger.log_charging_session(
                    time_str, self.id, self.role_type,
                    station,
                    self.avatar.added_energy_kwh,
                    duration
                )

            station.remove_vehicle(self.id)

        self.state = "IDLE"
