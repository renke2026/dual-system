from typing import Dict, Any, Optional, List, Tuple
import datetime
import traci
import asyncio
from src.common import EnvironmentObservation, ChargingStationInfo, AgentProfile, NewsMessage
from src.agent.memory import MemoryStream
from src.agent.planner import Planner
from src.agent.baselines import MNLStationSelector
from src.env.vehicle import ChargeableVehicle
from src.env.map import DailySchedule, CityMap
from src.env.stations import CSManager
from src.env.sumo_interface import SumoManager
from datetime import timedelta
from src.common import PerceptionContext
from config import CONFIG

class SimulationAgent:
    def __init__(self, profile: AgentProfile, schedule: DailySchedule, current_time: datetime.datetime):
        self.id = profile.agent_id
        self.profile = profile
        self.schedule = schedule

        self.news_inbox: List[NewsMessage] = []

        self.memory = MemoryStream(self.id, profile)
        trip_strs = [f"{t.depart_time.strftime('%H:%M')} go to {t.dest_loc.id}" for t in schedule.trips]
        self.memory.add_snapshot(
            description=f"Initial Plan: {'; '.join(trip_strs)}",
            importance=5,
            tags={"type": "daily_plan"}
        )
        self.planner = Planner(profile, self.memory)

        self._mnl_selector = MNLStationSelector(
            seed=(CONFIG.generation.agents_seed + sum(ord(c) for c in self.id)) % (2 ** 31)
        )

        self.avatar = ChargeableVehicle(
            vehicle_id=profile.agent_id,
            soc=schedule.initial_soc,
            capacity_kwh=profile.vehicle.capacity_kwh,
            role=profile.role_type,
            max_charge_power_kw=profile.vehicle.max_charge_power_kw,
        )

        self.state = "IDLE"
        self.current_trip_idx = 0
        self.current_location = schedule.trips[0].origin_loc.edge_id if schedule.trips else "Home"
        self.target_station_id = None
        self.last_decision = None
        self.is_thinking = False
        self.last_thought_time = None

        self.current_mental_state = "ALERT"
        self.next_wake_up_time = None
        self.news_inbox: List[NewsMessage] = []
        self.daily_news_archive: List[NewsMessage] = []
        self.latest_urgency = "NONE"

        self.daily_trip_logs = []
        self.daily_energy_strategy = "STANDARD"
        self.daily_expense = 0.0

        self.feedback_buffer = None

        self.expected_price = None
        self.expected_queue = None

        self.target_soc_trigger = None

        self.daily_expense = 0.0
        self.daily_energy = 0.0
        self.history_avg_price = CONFIG.agent.initial_history_avg_price

        self.pending_decision_context: Optional[Dict] = None
        self.has_thought_this_step = False
        self.current_position_xy = (0.0, 0.0)

    def _eff_anxiety(self) -> float:
        """Return a fixed anxiety value when uniform_numeric_params=True, else the agent's real anxiety level."""
        if CONFIG.ablation.uniform_numeric_params:
            return CONFIG.ablation.uniform_anxiety_level
        return self.profile.range_anxiety_level

    def _eff_price_sensitivity(self) -> float:
        """Return a fixed price sensitivity when uniform_numeric_params=True, else the agent's real sensitivity."""
        if CONFIG.ablation.uniform_numeric_params:
            return CONFIG.ablation.uniform_price_sensitivity
        return self.profile.price_sensitivity

    def _mnl_select_station(self, obs: EnvironmentObservation) -> Optional[str]:
        """Select a station from nearby_stations via MNL (personalized weights: anxiety scales time, price sensitivity scales cost)."""
        stations = obs.nearby_stations if obs is not None else []
        if not stations:
            return None
        beta_time = CONFIG.mnl.beta_time * (1.0 + self._eff_anxiety() * CONFIG.mnl.anxiety_time_scale)
        beta_price = CONFIG.mnl.beta_price * (1.0 + self._eff_price_sensitivity() * CONFIG.mnl.price_sensitivity_scale)
        beta_distance = CONFIG.mnl.beta_distance
        return self._mnl_selector.select(
            stations, beta_time, beta_price, beta_distance,
            CONFIG.mnl.avg_speed_kmh, CONFIG.mnl.temperature,
        )

    def _mnl_conventional_decision(self, state: str, triggers: Dict[str, Any], obs: EnvironmentObservation) -> Dict[str, Any]:
        """[Baseline B] Fully non-LLM decision: System-1 rules pick the action, MNL picks the station.

        Returns a dict matching ActionOutput.model_dump() for reuse by handlers.
        """
        urgency = triggers.get("urgency", "NONE")
        decision = None
        target_id = None
        recheck = 15
        target_soc = None

        if state == "IDLE":
            if urgency != "NONE":
                decision = "FIND_CHARGER"
                target_id = self._mnl_select_station(obs)
            else:
                decision = "STAY_AND_WAIT"
        elif state == "DRIVING":
            if urgency != "NONE":
                decision = "REROUTE_TO_CHARGER"
                target_id = self._mnl_select_station(obs)
            else:
                decision = "KEEP_ROUTE"
        elif state == "DRIVING_TO_CHARGE":
            if self.feedback_buffer:
                decision = "CHANGE_STATION"
                target_id = self._mnl_select_station(obs)
            else:
                decision = "CONTINUE_TO_STATION"
        elif state == "CHARGING":
            if triggers.get("is_late"):
                decision = "UNPLUG_AND_LEAVE"
            else:
                decision = "CONTINUE_CHARGING"

        return {
            "thought_process": "[Baseline B] Conventional MNL decision (non-LLM)",
            "decision": decision,
            "target_station_id": target_id,
            "mental_state": "ALERT",
            "recheck_delay_min": recheck,
            "target_soc": target_soc,
            "reasoning_summary": "MNL discrete-choice station selection",
        }

    def _interpret_sensory_input(self, obs: EnvironmentObservation) -> PerceptionContext:
        """[Step 5 Refined v2] Fuzzy sensory interpreter.

        Fixes a missing price field and refines the three battery-perception tiers.
        """
        surge = obs.grid_surge_multiplier

        tolerance = CONFIG.agent.price_tolerance_base - (self._eff_price_sensitivity() * CONFIG.agent.price_tolerance_scale)

        if surge <= 1.0:
            price_tag = "FAIR (Base Price)"
            if surge < 0.9: price_tag = "BARGAIN (Discounted)"
        elif surge < tolerance:
            price_tag = "ELEVATED (Surge)"
        else:
            price_tag = "EXTREMELY OVERPRICED (High Surge)"

        dist_to_go = CONFIG.agent.dist_to_go_default_km

        thresholds = self._calculate_dynamic_thresholds(dist_to_go)
        survival_soc = thresholds["survival_soc"]
        panic_soc = thresholds["panic_soc"]
        comfort_soc = thresholds["comfort_soc"]

        if obs.soc < survival_soc:
            batt_tag = "DEADLY (Physical Risk)"
            margin = obs.soc - survival_soc

        elif obs.soc < panic_soc:
            batt_tag = "PANIC (High Anxiety)"
            margin = obs.soc - panic_soc

        elif obs.soc < comfort_soc:
            batt_tag = "CONCERNED (Below Comfort)"
            margin = obs.soc - comfort_soc

        else:
            batt_tag = "COMFORTABLE"
            margin = obs.soc - comfort_soc

        lat = obs.lateness_min
        if lat <= 0: time_tag = "ON TIME"
        elif lat < CONFIG.agent.slight_late_min: time_tag = "SLIGHTLY LATE"
        else: time_tag = "SEVERELY LATE"

        desc = (
            f"Sensory Check: "
            f"Battery is {batt_tag} (Margin: {margin*100:+.1f}%). "
            f"Grid Price is {price_tag} ({surge:.1f}x Surge). "
            f"Schedule is {time_tag}."
        )

        return PerceptionContext(
            price_status=price_tag,
            price_ratio=surge,
            battery_status=batt_tag,
            safety_margin=margin,
            schedule_status=time_tag,
            description_text=desc
        )

    def _calculate_dynamic_thresholds(self, distance_km: float) -> dict:
        """Compute all physical and psychological thresholds for the current trip.

        Keeps _determine_charging_need (entry) and is_ready_to_depart (exit) on one standard to avoid deadlock.
        """
        consumption_per_km = CONFIG.agent.consumption_per_km
        trip_consumption = distance_km * consumption_per_km

        survival_soc = trip_consumption + CONFIG.agent.survival_buffer

        comfort_buffer = CONFIG.agent.comfort_base + (self._eff_anxiety() * CONFIG.agent.comfort_anxiety_scale)
        comfort_soc = trip_consumption + comfort_buffer

        base_panic_level = CONFIG.agent.panic_base + (self._eff_anxiety() * CONFIG.agent.panic_anxiety_scale)

        if self.profile.role_type == "GIG_WORKER":
            base_panic_level += CONFIG.agent.panic_gig_offset
        if self.profile.price_trait == "SENSITIVE":
            base_panic_level += CONFIG.agent.panic_sensitive_offset

        return {
            "trip_consumption": trip_consumption,
            "survival_soc": survival_soc,
            "comfort_soc": comfort_soc,
            "panic_soc": base_panic_level
        }

    def receive_news(self, news: NewsMessage, current_time: datetime.datetime):
        """Receive a broadcast message."""
        self.news_inbox.append(news)
        self.daily_news_archive.append(news)

        if news.priority in ["CRITICAL", "HIGH"]:
            imp = 9 if news.priority == "CRITICAL" else 5
            self.memory.add_snapshot(
                description=f"Received Broadcast [{news.priority}]: {news.content}",
                importance=imp,
                tags={"type": "news", "category": news.category}
            )

        if news.priority == "CRITICAL":
            if self.current_mental_state != "ALERT":
                print(f"❗ {self.id} was startled by news! State changed from {self.current_mental_state} -> ALERT")
                self.current_mental_state = "ALERT"
                self.next_wake_up_time = None

    def get_unread_high_priority_news(self) -> List[NewsMessage]:
        """Check for high-priority news to break silence.

        Only reads, does not consume; actual consumption happens in think_async.
        """
        urgent_news = [
            msg for msg in self.news_inbox
            if msg.priority in ["CRITICAL", "HIGH"]
        ]
        return urgent_news

    def clear_inbox(self):
        """Clear the inbox (called after thinking completes)."""
        self.news_inbox = []


    def get_observation(self, current_time: datetime.datetime, cs_manager: CSManager, grid_manager: Any, city_map: CityMap,
                        focus_type: str = "ALL",processing_level: str = "FULL") -> EnvironmentObservation:
        """[Research-grade observation builder]

        processing_level:
          - "FULL": compute traffic, distance, Top-K (for decisions) - expensive
          - "SIMPLE": only base state and market averages (for reactions) - fast
        """
        current_edge_id = self.current_location
        if self.state in ["DRIVING", "DRIVING_TO_CHARGE"]:
            try:
                active_vehs = traci.vehicle.getIDList()
                if self.id in active_vehs:
                    road_id = traci.vehicle.getRoadID(self.id)
                    if road_id and not road_id.startswith(":"):
                        current_edge_id = road_id
            except Exception: pass

        traffic_status = "Unknown"
        current_speed_kmh = 0.0

        if processing_level == "FULL" and current_edge_id:
            try:
                mean_speed = traci.edge.getLastStepMeanSpeed(current_edge_id)
                current_speed_kmh = mean_speed * 3.6
                max_speed = traci.edge.getMaxSpeed(current_edge_id)
                if max_speed > 0:
                    ratio = mean_speed / max_speed
                    if ratio < 0.4: traffic_status = "Congested"
                    elif ratio < 0.8: traffic_status = "Moderate"
                    else: traffic_status = "Fluency"
            except: pass

        raw_infos = cs_manager.get_observation_data(current_time)

        fcs_prices = [s['price_per_kwh'] for s in raw_infos if s.get('station_type') == 'FCS']
        scs_prices = [s['price_per_kwh'] for s in raw_infos if s.get('station_type') == 'SCS']
        avg_fcs = sum(fcs_prices) / len(fcs_prices) if fcs_prices else 0.0
        avg_scs = sum(scs_prices) / len(scs_prices) if scs_prices else 0.0

        final_stations = []
        if processing_level == "FULL":
            fcs_candidates = []
            scs_candidates = []

            for info in raw_infos:
                station_id = info['station_id']
                s_type = info.get('station_type', 'FCS')
                s_obj = cs_manager.get_station(station_id)

                real_dist = 999.0
                if current_edge_id:
                    real_dist = city_map.get_route_distance(current_edge_id, s_obj.edge_id)

                if real_dist == float('inf'): continue

                s_info = ChargingStationInfo(
                    station_id=station_id,
                    station_type=s_type,
                    distance_km=round(real_dist, 2),
                    price_per_kwh=info['price_per_kwh'],
                    queue_length=info['queue_length'],
                    estimated_wait_time_min=info['estimated_wait_time_min']
                )

                if s_type == "FCS":
                    fcs_candidates.append(s_info)
                else:
                    scs_candidates.append(s_info)

            def get_top_k(candidates, sort_key, k):
                return sorted(candidates, key=sort_key)[:k]

            top_fcs_dist = get_top_k(fcs_candidates, lambda x: x.distance_km, 2)
            top_fcs_price = get_top_k(fcs_candidates, lambda x: x.price_per_kwh, 1)

            top_scs_dist = get_top_k(scs_candidates, lambda x: x.distance_km, 2)
            top_scs_price = get_top_k(scs_candidates, lambda x: x.price_per_kwh, 1)

            selection_map = {}
            for s in top_fcs_dist + top_fcs_price + top_scs_dist + top_scs_price:
                selection_map[s.station_id] = s

            final_stations = list(selection_map.values())

            final_stations.sort(key=lambda x: x.distance_km)

        dest_charger_str = "None"

        if processing_level == "FULL" and self.current_trip_idx < len(self.schedule.trips):
            trip = self.schedule.trips[self.current_trip_idx]
            d_edge = trip.dest_loc.edge_id
            d_type = trip.dest_loc.type

            d_station = cs_manager.find_station_on_edge(d_edge)

            is_valid_dest = False
            if d_station and d_station.station_type == "SCS":
                if self.profile.role_type == "COMMUTER" and d_type in ["Work", "Home"]:
                    is_valid_dest = True
                elif d_type == "Home":
                    is_valid_dest = True

            if is_valid_dest:
                dist_to_dest = trip.distance_km

                if self.state == "DRIVING":
                     dist_to_dest = city_map.get_route_distance(self.current_location, d_edge)

                if dist_to_dest == float('inf'): dist_to_dest = 999.0

                current_price = d_station.get_price(current_time.hour * 3600 + current_time.minute * 60)

                virtual_id = f"DEST_{d_type}"

                dest_info_obj = ChargingStationInfo(
                    station_id=virtual_id,
                    station_type="SCS (Dest)",
                    distance_km=round(dist_to_dest, 2),
                    price_per_kwh=current_price,
                    queue_length=d_station.get_wait_count(),
                    estimated_wait_time_min=0
                )

                final_stations.insert(0, dest_info_obj)

                dest_charger_str = f"Available at {d_type} (${current_price:.2f}/kWh)"
            else:
                dest_charger_str = "Not available at destination"

        dest_info_str = "Unknown"
        current_lateness = 0
        if processing_level == "FULL" and self.current_trip_idx < len(self.schedule.trips):
            trip = self.schedule.trips[self.current_trip_idx]
            if current_time > trip.depart_time:
                diff = current_time - trip.depart_time
                current_lateness = int(diff.total_seconds() / 60)

            d_edge = trip.dest_loc.edge_id
            d_type = trip.dest_loc.type
            d_station = cs_manager.find_station_on_edge(d_edge)

            if d_station and d_station.station_type == "SCS":
                is_valid = False
                if self.profile.role_type == "COMMUTER" and d_type in ["Work", "Home"]:
                    is_valid = True
                elif d_type == "Home":
                    is_valid = True

                if is_valid:
                    price = d_station.get_price(current_time.hour * 3600)
                    dest_info_str = f"AVAILABLE (Slow Charger at {d_type}, Price: {price:.2f} $/kWh)"
                else:
                    dest_info_str = f"NOT SUITABLE (Charger at {d_type} exists but staying time is short)"
            else:
                dest_info_str = "NOT AVAILABLE"


        grid_status = "Unknown"
        if grid_manager is not None:
            try: grid_status = grid_manager.get_grid_intuition()
            except: pass

        grid_mult = grid_manager.get_current_multiplier()

        schedule_text = "No upcoming trips."
        if self.current_trip_idx < len(self.schedule.trips):
            trip = self.schedule.trips[self.current_trip_idx]
            target_time_str = trip.depart_time.strftime('%H:%M')
            now_time_str = current_time.strftime('%H:%M')

            diff_min = int((trip.depart_time - current_time).total_seconds() / 60)

            if diff_min > 0:
                schedule_text = (f"NEXT_TRIP: To {trip.dest_loc.type}. "
                                 f"Scheduled at {target_time_str} (Now: {now_time_str}, Buffer: {diff_min} mins). "
                                 f"STATUS: Early - You can wait or start early if needed.")
            elif diff_min == 0:
                schedule_text = (f"NEXT_TRIP: To {trip.dest_loc.type}. "
                                 f"Scheduled at {target_time_str} (Now: {now_time_str}). "
                                 f"STATUS: On Time - Depart immediately recommended.")
            elif -15 <= diff_min <= 0:
                 schedule_text = (f"NEXT_TRIP: To {trip.dest_loc.type} at {target_time_str} (in {abs(diff_min)}m). "
                                 f"STATUS: READY - Preparing to depart.")
            else:
                schedule_text = (f"NEXT_TRIP: To {trip.dest_loc.type} at {target_time_str} (in {abs(diff_min)}m). "
                                 f"STATUS: TOO EARLY. Converting time to energy/rest is recommended.")

        obs = EnvironmentObservation(
            current_time=current_time,
            soc=round(self.avatar.soc, 2),
            location=current_edge_id if current_edge_id else "Unknown",
            charging_power_kw=round(self.avatar.last_charging_power_kw, 1),
            nearby_stations=final_stations,
            traffic_status=traffic_status,
            current_speed=round(current_speed_kmh, 1),
            last_action_feedback=self.feedback_buffer,
            dest_charger_info=dest_charger_str,
            lateness_min=current_lateness,
            market_avg_fcs_price=round(avg_fcs, 2),
            market_avg_scs_price=round(avg_scs, 2),
            grid_status_intuition=grid_status,
            grid_surge_multiplier=grid_mult,
            upcoming_schedule_text=schedule_text
        )

        self.feedback_buffer = None

        return obs

    def _determine_charging_need(self, next_trip_dist_km: float, dest_has_scs: bool) -> str:
        """[Three-tier funnel] Mixed filter over physical constraints, profession and personality.

        Assesses the "discomfort" of the current state instead of a simple threshold cut.
        """
        soc = self.avatar.soc

        thresholds = self._calculate_dynamic_thresholds(next_trip_dist_km)

        if soc < thresholds["survival_soc"]:
            return "CRITICAL"

        if self.state == "DRIVING" and self.current_mental_state == "COMMITTED":
            return "NONE"

        is_dest_charger_valid = False

        if dest_has_scs:
            if self.profile.role_type == "COMMUTER":
                is_dest_charger_valid = True

            elif self.profile.role_type == "GIG_WORKER":
                current_trip = self.schedule.trips[self.current_trip_idx] if self.current_trip_idx < len(self.schedule.trips) else None
                if current_trip and current_trip.dest_loc.type == "Home":
                    is_dest_charger_valid = True
                else:
                    is_dest_charger_valid = False

        if is_dest_charger_valid:
            comfort_buffer = CONFIG.agent.comfort_base + (self._eff_anxiety() * CONFIG.agent.comfort_anxiety_scale_alt)

            if soc >= thresholds["comfort_soc"]:
                if self.profile.price_trait == "INSENSITIVE" and soc < CONFIG.agent.insensitive_emerge_soc:
                    return "DECISION_NEEDED"
                else:
                    return "NONE"
            else:
                return "DECISION_NEEDED"

        else:
            limit = thresholds["panic_soc"]

            if self.state in ["DRIVING", "DRIVING_TO_CHARGE"]:
                limit *= CONFIG.agent.driving_discount

            if soc < limit:
                return "DECISION_NEEDED"

        return "NONE"

    def is_ready_to_depart(self, trip) -> Tuple[bool, str]:
        """[Standardized] Whether the agent is ready to leave, using the same numeric standard as _determine_charging_need."""
        current_soc = self.avatar.soc

        thresholds = self._calculate_dynamic_thresholds(trip.distance_km)

        needed = thresholds["comfort_soc"]

        if current_soc < needed:
            return False, f"Not Confident: Has {current_soc*100:.1f}%, Needs {needed*100:.1f}% (Comfort Level)"

        return True, "Ready to depart"

    def _sense_triggers(self, current_time: datetime.datetime, cs_manager: CSManager) -> Dict[str, Any]:
        """[Perception] Scan System-1 signals and build the decision context."""

        triggers = {
            "current_time": current_time,
            "is_late": False,
            "lateness_min": 0,
            "urgency": "NONE",
            "has_news": False,
            "trip": None,
            "dest_has_scs": False,
            "remaining_dist": 0.0
        }

        if self.current_trip_idx < len(self.schedule.trips):
            trip = self.schedule.trips[self.current_trip_idx]
            triggers["trip"] = trip

            time_diff_min = int((current_time - trip.depart_time).total_seconds() / 60)
            triggers["lateness_min"] = time_diff_min

            if time_diff_min > 0:
                triggers["is_late"] = True
            else:
                triggers["is_late"] = False

        else:
            home_edge = self.schedule.trips[0].origin_loc.edge_id if self.schedule.trips else None

            if home_edge and self.current_location != home_edge and self.state == "IDLE":
                from src.env.map import Trip, Location

                dummy_trip = Trip(
                    trip_id=f"{self.id}_return_home",
                    origin_loc=Location(id="Current", type="Other", x=0, y=0, edge_id=self.current_location, taz_id=""),
                    dest_loc=Location(id="Home", type="Home", x=0, y=0, edge_id=home_edge, taz_id=""),
                    depart_time=current_time,
                    latest_arrival_time = current_time + datetime.timedelta(minutes=30),
                    distance_km=5.0
                )

                triggers["trip"] = dummy_trip
                triggers["is_late"] = True
                triggers["is_end_of_day_return"] = True
                print(f"🏠 {self.id} detected wandering away from home, auto-generating a trip home: {self.current_location}-> {home_edge}")

        if self.state == "DRIVING":
            if triggers["trip"]:
                triggers["remaining_dist"] = triggers["trip"].distance_km
                dest_station = cs_manager.find_station_on_edge(triggers["trip"].dest_loc.edge_id)
                if dest_station and dest_station.station_type == "SCS":
                    triggers["dest_has_scs"] = True

        elif self.state in ["IDLE", "CHARGING"]:
            if triggers["trip"]:
                triggers["remaining_dist"] = triggers["trip"].distance_km

            if triggers["trip"]:
                dest_station = cs_manager.find_station_on_edge(triggers["trip"].dest_loc.edge_id)
                if dest_station and dest_station.station_type == "SCS":
                    triggers["dest_has_scs"] = True

        triggers["urgency"] = self._determine_charging_need(
            triggers["remaining_dist"],
            triggers["dest_has_scs"]
        )
        self.latest_urgency = triggers["urgency"]

        triggers["has_news"] = len(self.get_unread_high_priority_news()) > 0

        return triggers

    def _check_filter_logic(self, triggers: Dict[str, Any]) -> bool:
        """[Filter core] Decide whether to wake System 2 (LLM).

        Strict order: survival > sleep/focus > lateness > ordinary need.
        """
        if self.is_thinking: return False

        # [Ablation A] Event-driven baseline: strip the cognitive-state mechanism.
        # Keep only "event trigger + LLM decision"; drop COMMITTED/LIGHT_SLEEP/DEEP_FOCUS shields,
        # snooze alarms, the priority matrix and the curfew.
        if CONFIG.ablation.baseline_no_cognitive_state:
            return (triggers["urgency"] != "NONE" or triggers["has_news"] or triggers["is_late"])

        urgency = triggers["urgency"]
        has_news = triggers["has_news"]
        is_late = triggers["is_late"]
        current_time = triggers["current_time"]

        # Hard curfew: between curfew_start and curfew_end, only survival/breaking news wakes the agent.
        hour = current_time.hour
        if CONFIG.agent.curfew_start_hour <= hour < CONFIG.agent.curfew_end_hour:
            if triggers["urgency"] == "CRITICAL" or triggers["has_news"]:
                return True

            return False

        is_charging_late = (self.state == "CHARGING" and is_late)

        if has_news:
            return True

        if urgency == "CRITICAL":
            if self.state == "CHARGING" and self.current_mental_state == "LIGHT_SLEEP":
                if self.next_wake_up_time and current_time < self.next_wake_up_time:
                    return False

            return True

        if self.current_mental_state == "COMMITTED":
            return False

        if self.current_mental_state == "LIGHT_SLEEP":
            if self.next_wake_up_time:
                if current_time >= self.next_wake_up_time:
                    self.current_mental_state = "ALERT"
                    self.next_wake_up_time = None
                    return True
                else:
                    return False
            else:
                return True

        if self.current_mental_state == "DEEP_FOCUS":
            return False

        if is_charging_late:
            return True

        if self.state == "DRIVING_TO_CHARGE":
            return False

        if urgency != "NONE":
            return True

        return False


    async def update_state_machine(self, current_time: datetime.datetime,
                                   dt_minutes: int,
                                   cs_manager: CSManager,
                                   grid_manager: Any,
                                   sumo_manager: SumoManager,
                                   city_map: CityMap,
                                   semaphore: asyncio.Semaphore,
                                   logger=None):
        """[Main entry] The single state-update entry point."""
        self._current_sim_time = current_time
        self._update_physics_consumption(dt_minutes, sumo_manager)

        ctx = {
            "time": current_time,
            "dt": dt_minutes,
            "cs": cs_manager,
            "grid": grid_manager,
            "sumo": sumo_manager,
            "map": city_map,
            "sem": semaphore,
            "log": logger
        }

        transition = None

        if self.state == "IDLE":
            transition = await self._handle_state_idle(ctx)

        elif self.state == "DRIVING":
            transition = await self._handle_state_driving(ctx)

        elif self.state == "DRIVING_TO_CHARGE":
            transition = await self._handle_state_driving_to_charge(ctx)

        elif self.state == "CHARGING":
            transition = await self._handle_state_charging(ctx)

        else:
            print(f"⚠️ Unknown state: {self.state}, resetting to IDLE")
            transition = {"action": "RESET", "next_state": "IDLE", "reason": "Unknown State Error"}

        await self._execute_transition(transition, ctx)

    def _update_physics_consumption(self, dt_minutes: int, sumo_manager: SumoManager):
        """[Research-grade physical metabolism] Energy consumption based on measured SUMO data."""
        base_consumption = CONFIG.agent.base_consumption_per_min * dt_minutes

        drive_consumption = 0.0

        if self.state in ["DRIVING", "DRIVING_TO_CHARGE"]:
            try:
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
                    drive_consumption = drive_consumption * CONFIG.agent.drive_consumption_scale
            except Exception:
                pass

        total_drop = base_consumption + drive_consumption
        self.avatar.soc -= total_drop
        self.avatar.soc = max(0.0, min(1.0, self.avatar.soc))

        if self.state in ["DRIVING", "DRIVING_TO_CHARGE"]:
            try:
                pos = traci.vehicle.getPosition(self.id)
                self.current_position_xy = pos
            except:
                pass
        elif self.state in ["IDLE", "CHARGING"]:
            pass

    async def _handle_state_idle(self, ctx) -> Dict[str, Any]:
        """[IDLE] Decision space: START_TRIP, FIND_CHARGER, STAY_AND_WAIT."""
        triggers = self._sense_triggers(ctx['time'], ctx['cs'])

        if triggers.get('is_end_of_day_return'):
            return {
                    "source": "SYS1_REFLEX",
                    "action": "AUTO_DEPART",
                    "next_state": "DRIVING",
                    "new_mental_state": "COMMITTED" if triggers.get('is_end_of_day_return') else "ALERT",
                    "reason": "End of Day Return" if triggers.get('is_end_of_day_return') else "Schedule Trigger",
                    "trip": triggers['trip'],
                    "target_id": None,
                    "wait_duration": None,
                    "target_soc": None,
                    "station": None,
                    "llm_data": None
                }

        if triggers['is_late']:
            is_ready, _ = self.is_ready_to_depart(triggers['trip'])
            if is_ready:
                return {
                    "source": "SYS1_REFLEX",
                    "action": "AUTO_DEPART",
                    "next_state": "DRIVING",
                    "new_mental_state": "ALERT",
                    "reason": "Schedule Trigger (Ready)",
                    "trip": triggers['trip'],
                    "target_id": None,
                    "wait_duration": None,
                    "target_soc": None,
                    "station": None,
                    "llm_data": None
                }

        if self._check_filter_logic(triggers):
            self.is_thinking = True
            self.has_thought_this_step = True
            try:
                current_news = list(self.news_inbox)
                self.clear_inbox()

                focus = "ALL"
                obs = self.get_observation(ctx['time'], ctx['cs'], ctx['grid'], ctx['map'], focus_type=focus, processing_level="FULL")
                perception = self._interpret_sensory_input(obs)
                print(f"🕒 {ctx['time'].strftime('%H:%M')} |🧠 {self.id} (🔋{self.avatar.soc})(IDLE) thinking: depart or find a charger?")

                if CONFIG.ablation.baseline_mode == "mnl":
                    llm_result = self._mnl_conventional_decision("IDLE", triggers, obs)
                else:
                    async with ctx['sem']:
                        llm_result = await self.planner.plan_idle_action(obs, current_news, perception)

                decision = llm_result.get("decision")
                mental_state = llm_result.get("mental_state", "ALERT")
                print(f"🕒 {ctx['time'].strftime('%H:%M')} |🧠 {self.id} (🔋{self.avatar.soc})(IDLE) decision: {decision}|mental:{mental_state}")

                if decision == "START_TRIP":
                    return {
                        "source": "SYS2_COGNITION",
                        "action": "FORCE_DEPART",
                        "next_state": "DRIVING",
                        "new_mental_state": mental_state,
                        "reason": "Cognitive Decision: Force Start",
                        "trip": triggers['trip'],
                        "llm_data": llm_result,
                        "target_id": None, "wait_duration": None, "target_soc": None, "station": None
                    }

                elif decision == "FIND_CHARGER":
                    return {
                        "source": "SYS2_COGNITION",
                        "action": "FIND_CHARGER",
                        "next_state": "DRIVING_TO_CHARGE",
                        "new_mental_state": mental_state,
                        "reason": "Cognitive Decision: Find Charger",
                        "target_id": llm_result.get("target_station_id"),
                        "llm_data": llm_result,
                        "trip": None, "wait_duration": None, "target_soc": None, "station": None
                    }
                elif decision == "STAY_AND_WAIT":
                    return {
                        "source": "SYS2_COGNITION",
                        "action": "STAY_AND_WAIT",
                        "next_state": "IDLE",
                        "new_mental_state": mental_state,
                        "reason": "Cognitive Decision: Wait",
                        "wait_duration": llm_result.get("recheck_delay_min", 15),
                        "llm_data": llm_result,
                        "trip": None, "target_id": None, "target_soc": None, "station": None
                    }

            finally:
                self.is_thinking = False

        return {
            "source": "SYS1_REFLEX",
            "action": "STAY",
            "next_state": "IDLE",
            "new_mental_state": self.current_mental_state,
            "reason": "Idle Inertia",
            "target_id": None, "wait_duration": None, "target_soc": None, "trip": None, "station": None, "llm_data": None
        }

    async def _handle_state_driving(self, ctx) -> Dict[str, Any]:
        """[DRIVING] Decision space: KEEP_ROUTE, REROUTE_TO_CHARGER."""
        triggers = self._sense_triggers(ctx['time'], ctx['cs'])

        if self._check_filter_logic(triggers):
            self.is_thinking = True
            self.has_thought_this_step = True
            try:
                current_news = list(self.news_inbox)
                self.clear_inbox()

                focus = "ALL"
                obs = self.get_observation(ctx['time'], ctx['cs'], ctx['grid'], ctx['map'], focus_type=focus, processing_level="FULL")
                perception = self._interpret_sensory_input(obs)

                print(f"🕒 {ctx['time'].strftime('%H:%M')} |🧠 {self.id}(🔋{self.avatar.soc}) (DRIVING) thinking: keep going or reroute?")

                if CONFIG.ablation.baseline_mode == "mnl":
                    llm_result = self._mnl_conventional_decision("DRIVING", triggers, obs)
                else:
                    async with ctx['sem']:
                        llm_result = await self.planner.plan_driving_action(obs, current_news, perception)

                decision = llm_result.get("decision")
                mental_state = llm_result.get("mental_state", "ALERT")

                print(f"🕒 {ctx['time'].strftime('%H:%M')} |🧠 {self.id}(🔋{self.avatar.soc}) (DRIVING) thinking: keep going or reroute?(decision:{decision}|mental:{mental_state})")

                if decision == "REROUTE_TO_CHARGER":
                     return {
                        "source": "SYS2_COGNITION",
                        "action": "CHANGE_STATION",
                        "next_state": "DRIVING_TO_CHARGE",
                        "new_mental_state": mental_state,
                        "reason": "Cognitive Decision: Reroute",
                        "target_id": llm_result.get("target_station_id"),
                        "llm_data": llm_result,
                        "trip": None, "wait_duration": None, "target_soc": None, "station": None
                    }
                elif decision == "KEEP_ROUTE":
                    return {
                        "source": "SYS2_COGNITION",
                        "action": "KEEP_DRIVING",
                        "next_state": "DRIVING",
                        "new_mental_state": mental_state,
                        "reason": "Cognitive Decision: Keep Route",
                        "llm_data": llm_result,
                        "trip": None, "target_id": None, "wait_duration": None, "target_soc": None, "station": None
                    }

            finally:
                self.is_thinking = False

        return {
            "source": "SYS1_REFLEX",
            "action": "KEEP_DRIVING",
            "next_state": "DRIVING",
            "new_mental_state": self.current_mental_state,
            "reason": "Driving Inertia",
            "trip": None, "target_id": None, "wait_duration": None, "target_soc": None, "station": None, "llm_data": None
        }

    async def _handle_state_driving_to_charge(self, ctx) -> Dict[str, Any]:
        """[DRIVING_TO_CHARGE] Decision space: CONTINUE_TO_STATION, CHANGE_STATION, ABORT_AND_RETURN."""
        triggers = self._sense_triggers(ctx['time'], ctx['cs'])
        should_think = bool(self.feedback_buffer) or triggers['urgency'] == "CRITICAL"

        if should_think:
            self.is_thinking = True
            self.has_thought_this_step = True
            try:
                current_news = list(self.news_inbox)
                self.clear_inbox()
                focus = "ALL"
                obs = self.get_observation(ctx['time'], ctx['cs'], ctx['grid'], ctx['map'], focus_type=focus, processing_level="FULL")
                perception = self._interpret_sensory_input(obs)

                print(f"🕒 {ctx['time'].strftime('%H:%M')} |🧠 {self.id}(🔋{self.avatar.soc}) (HUNTING) thinking: navigation blocked / regret?")

                if CONFIG.ablation.baseline_mode == "mnl":
                    llm_result = self._mnl_conventional_decision("DRIVING_TO_CHARGE", triggers, obs)
                else:
                    async with ctx['sem']:
                        llm_result = await self.planner.plan_hunting_action(obs, current_news, perception)

                decision = llm_result.get("decision")
                mental_state = llm_result.get("mental_state", "ALERT")

                if decision == "CHANGE_STATION":
                    return {
                        "source": "SYS2_COGNITION",
                        "action": "CHANGE_STATION",
                        "next_state": "DRIVING_TO_CHARGE",
                        "new_mental_state": mental_state,
                        "reason": "Cognitive Decision: Retry Reroute",
                        "target_id": llm_result.get("target_station_id"),
                        "llm_data": llm_result,
                        "trip": None, "wait_duration": None, "target_soc": None, "station": None
                    }
                elif decision == "ABORT_AND_RETURN":
                    return {
                        "source": "SYS2_COGNITION",
                        "action": "ABORT_CHARGE",
                        "next_state": "DRIVING",
                        "new_mental_state": mental_state,
                        "reason": "Cognitive Decision: Abort Charging",
                        "trip": triggers['trip'],
                        "llm_data": llm_result,
                        "target_id": None, "wait_duration": None, "target_soc": None, "station": None
                    }
                elif decision == "CONTINUE_TO_STATION":
                     return {
                        "source": "SYS2_COGNITION",
                        "action": "KEEP_DRIVING",
                        "next_state": "DRIVING_TO_CHARGE",
                        "new_mental_state": mental_state,
                        "reason": "Cognitive Decision: Persist",
                        "llm_data": llm_result,
                        "trip": None, "target_id": None, "wait_duration": None, "target_soc": None, "station": None
                    }
            finally:
                self.is_thinking = False

        return {
            "source": "SYS1_REFLEX",
            "action": "KEEP_DRIVING",
            "next_state": "DRIVING_TO_CHARGE",
            "new_mental_state": self.current_mental_state,
            "reason": "Hunting Inertia",
            "trip": None, "target_id": None, "wait_duration": None, "target_soc": None, "station": None, "llm_data": None
        }

    async def _handle_state_charging(self, ctx) -> Dict[str, Any]:
        """[CHARGING] Decision space: UNPLUG_AND_LEAVE, CONTINUE_CHARGING."""
        triggers = self._sense_triggers(ctx['time'], ctx['cs'])
        station = ctx['cs'].find_station_on_edge(self.current_location)

        if not station and self.target_station_id:
            station = ctx['cs'].get_station(self.target_station_id)

        if not station:
            print(f"❌ {self.id} is in CHARGING state but cannot find the station handle! Forcing reset to IDLE.")
            return {
                "source": "SYSTEM_ERR",
                "action": "RESET",
                "next_state": "IDLE",
                "new_mental_state": "ALERT",
                "reason": "Lost Station Handle",
                "trip": None, "target_id": None, "wait_duration": None, "target_soc": None, "station": None, "llm_data": None
            }

        if station:
            should_auto_leave = False
            if self.target_soc_trigger and self.avatar.soc >= self.target_soc_trigger:
                should_auto_leave = True
            elif station.station_type == "FCS" and self.avatar.soc >= CONFIG.station.fcs_full_soc:
                should_auto_leave = True

            if should_auto_leave:
                return {
                    "source": "SYS1_REFLEX",
                    "action": "AUTO_UNPLUG",
                    "next_state": "IDLE",
                    "new_mental_state": "ALERT",
                    "reason": "Reflex: Battery Full/Target Reached",
                    "station": station,
                    "trip": None, "target_id": None, "wait_duration": None, "target_soc": None, "llm_data": None
                }

        if self._check_filter_logic(triggers):
            self.is_thinking = True
            self.has_thought_this_step = True
            try:
                obs = self.get_observation(ctx['time'], ctx['cs'], ctx['grid'], ctx['map'], processing_level="SIMPLE")
                perception = self._interpret_sensory_input(obs)

                duration = 0
                if hasattr(self, 'arrival_time') and self.arrival_time:
                    duration = int((ctx['time'] - self.arrival_time).total_seconds() / 60)

                session_info = {
                    "Duration": f"{duration} mins",
                    "Energy Added": f"{self.avatar.added_energy_kwh:.2f} kWh",
                    "Cost Incurred": f"${self.avatar.incurred_cost:.2f}"
                }

                print(f"🕒 {ctx['time'].strftime('%H:%M')} |🧠 {self.id}(🔋{self.avatar.soc}) (CHARGING) thinking: unplug or keep charging?")

                if CONFIG.ablation.baseline_mode == "mnl":
                    llm_result = self._mnl_conventional_decision("CHARGING", triggers, obs)
                else:
                    async with ctx['sem']:
                        llm_result = await self.planner.plan_charging_action(obs, session_info, perception)

                decision = llm_result.get("decision")
                mental_state = llm_result.get("mental_state", "ALERT")
                print(f"🕒 {ctx['time'].strftime('%H:%M')} |🧠 {self.id}(🔋{self.avatar.soc}) (CHARGING) decision: {decision}|mental:{mental_state}")
                recheck_delay_min_s = llm_result.get("recheck_delay_min")
                target_soc_s = llm_result.get("target_soc")
                print(f"recheck_delay_min{recheck_delay_min_s}|target_soc:{target_soc_s}")

                if decision == "UNPLUG_AND_LEAVE":
                    return {
                        "source": "SYS2_COGNITION",
                        "action": "COGNITIVE_DEPART",
                        "next_state": "DRIVING",
                        "new_mental_state": "COMMITTED",
                        "reason": "Cognitive Decision: Depart",
                        "station": station,
                        "llm_data": llm_result,
                        "trip": None, "target_id": None, "wait_duration": None, "target_soc": None
                    }
                elif decision == "CONTINUE_CHARGING":
                    return {
                        "source": "SYS2_COGNITION",
                        "action": "KEEP_CHARGING",
                        "next_state": "CHARGING",
                        "new_mental_state": mental_state,
                        "reason": "Cognitive Decision: Keep Charging",
                        "wait_duration": llm_result.get("recheck_delay_min"),
                        "target_soc": llm_result.get("target_soc"),
                        "llm_data": llm_result,
                        "trip": None, "target_id": None, "station": None
                    }
            finally:
                self.is_thinking = False

        return {
            "source": "SYS1_REFLEX",
            "action": "KEEP_CHARGING",
            "next_state": "CHARGING",
            "new_mental_state": self.current_mental_state,
            "reason": "Charging Inertia",
            "trip": None, "target_id": None, "wait_duration": None, "target_soc": None, "station": None, "llm_data": None
        }


    async def _execute_transition(self, trans: Dict[str, Any], ctx: Dict[str, Any]):
        """[Atomic executor] Turn a state-transition packet into physical actions.

        Supported actions: AUTO_DEPART, FORCE_DEPART, AUTO_UNPLUG, REROUTE, ABORT_CHARGE,
        WAIT, KEEP_CHARGING, COGNITIVE_DEPART, STAY, KEEP_DRIVING.
        """
        source = trans["source"]
        action = trans["action"]
        reason: str = trans["reason"]
        llm_data = trans.get("llm_data")

        prev_state = self.state
        time_str = ctx['time'].strftime("%H:%M")

        target_state = trans.get("next_state")
        target_mental_state = trans["new_mental_state"]

        # [Ablation A] Strip cognitive state: force ALERT so the LLM's COMMITTED/LIGHT_SLEEP/DEEP_FOCUS have no effect.
        if CONFIG.ablation.baseline_no_cognitive_state:
            target_mental_state = "ALERT"

        # Resolve virtual "DEST_*" IDs back to the real physical station ID.
        raw_target_id = trans.get("target_id")
        resolved_target_id = raw_target_id

        if raw_target_id and raw_target_id.startswith("DEST_"):
            if self.current_trip_idx < len(self.schedule.trips):
                trip = self.schedule.trips[self.current_trip_idx]
                dest_edge = trip.dest_loc.edge_id
                real_station = ctx['cs'].find_station_on_edge(dest_edge)
                if real_station:
                    resolved_target_id = real_station.id

        if source == "SYS2_COGNITION":
            trigger_desc = trans["reason"]
            if trans.get("llm_data"):
                pass

            self.pending_decision_context = {
                "trigger": trigger_desc,
                "mental_state": trans["new_mental_state"],
                "decision": trans["action"],
                "start_time": ctx['time']
            }

        else:
            if action not in ["STAY", "KEEP_DRIVING", "KEEP_CHARGING"]:
                 print(f"🤖 {time_str} | {self.id}(🔋{self.avatar.soc}) [Reflex] {action} | Reason: {reason}")

        if action in ["AUTO_DEPART", "FORCE_DEPART"]:
            trip = trans["trip"]
            success = False

            if not trip:
                print(f"⚠️ {time_str} | {self.id} Action '{action}' ignored: No trip data available in context.")
                self.state = "IDLE"
                self.current_mental_state = "ALERT"
                return

            if trip.origin_loc.edge_id == trip.dest_loc.edge_id:
                self.on_arrive_at_destination(ctx['time'])
                success = True
            else:
                if ctx['sumo'].add_vehicle_to_sumo(self.id, trip.origin_loc.edge_id, trip.dest_loc.edge_id):
                    success = True
                else:
                    self.feedback_buffer = f"{action} Failed: SUMO gridlock."
                    print(f"❌ {time_str} | {self.id} {action} Failed!")

            if success:
                if resolved_target_id:
                    self.target_station_id = resolved_target_id
                if target_state: self.state = target_state
                self.current_mental_state = target_mental_state
            else:
                self.current_mental_state = "ALERT"

        elif action in ["AUTO_UNPLUG", "COGNITIVE_DEPART"]:
            station = trans.get("station")
            if station:
                success = await self._execute_departure_async(station, ctx['sumo'], ctx['cs'], ctx['time'], ctx['map'], logger=ctx.get('log'), source=source)

                if success:
                    if target_state: self.state = target_state
                    self.current_mental_state = target_mental_state
                else:
                    self.feedback_buffer = f"{action} executed, but re-entry to road failed."
                    self.current_mental_state = "ALERT"

        elif action in ["FIND_CHARGER", "CHANGE_STATION"]:
            target_id = trans.get("target_id")
            station = ctx['cs'].get_station(target_id)
            success = False

            if station:
                seconds = ctx['time'].hour * 3600 + ctx['time'].minute * 60
                self.expected_price = station.get_price(seconds)
                self.expected_queue = station.get_wait_count()

                if self.state in ["DRIVING", "DRIVING_TO_CHARGE"]:
                    success = ctx['sumo'].change_target(self.id, station.edge_id)
                elif self.state == "IDLE":
                    success = ctx['sumo'].add_vehicle_to_sumo(self.id, self.current_location, station.edge_id)

            if success:
                self.target_station_id = target_id
                if target_state: self.state = target_state
                self.current_mental_state = target_mental_state
            else:
                self.state = prev_state
                self.feedback_buffer = f"Navigation to {target_id} failed."
                self.current_mental_state = "ALERT"
                print(f"❌ {self.id} {action} physically failed! Feedback: {self.feedback_buffer}")

        elif action == "ABORT_CHARGE":
            trip = trans["trip"]
            success = False
            if trip and ctx['sumo'].change_target(self.id, trip.dest_loc.edge_id):
                self.target_station_id = None
                success = True

            if success:
                if target_state: self.state = target_state
                self.current_mental_state = target_mental_state
            else:
                self.feedback_buffer = "Abort Charging failed (SUMO Error)."
                self.current_mental_state = "ALERT"

        elif action in ["STAY", "STAY_AND_WAIT", "KEEP_CHARGING"]:
            delay = trans.get("wait_duration")
            if delay:
                self.next_wake_up_time = ctx['time'] + datetime.timedelta(minutes=delay)

            target_soc = trans.get("target_soc")
            if target_soc:
                self.target_soc_trigger = target_soc

            if target_state: self.state = target_state
            self.current_mental_state = target_mental_state

        elif action == "KEEP_DRIVING":
            if resolved_target_id:
                self.target_station_id = resolved_target_id

            if target_state: self.state = target_state
            self.current_mental_state = target_mental_state

        elif action == "RESET":
            self.state = "IDLE"
            self.target_station_id = None
            self.current_mental_state = "ALERT"
            print(f"⚠️ {time_str} | {self.id} System Reset performed.")


    async def _execute_departure_async(self, station, sumo_manager, cs_manager, current_time, city_map, logger=None, source="UNKNOWN"):
        """[Spinal layer - unified departure executor]

        Handles unplug, billing, memory generation, and SUMO injection.
        """
        time_str = current_time.strftime("%H:%M")
        station_id = station.id

        await self.react_to_departure_async(station_id, cs_manager, city_map, current_time)

        station.remove_vehicle(self.id)

        if logger is not None:
            seconds = current_time.hour * 3600 + current_time.minute * 60
            try:
                price = station.get_price(seconds)
            except Exception:
                price = None
            duration_min = 0
            if self.avatar.charge_start_time is not None:
                duration_min = int((current_time - self.avatar.charge_start_time).total_seconds() / 60)
            logger.log_charging_event(
                day=logger.current_day,
                time_str=time_str,
                agent_id=self.id,
                role=self.profile.role_type,
                station_id=station_id,
                station_type=getattr(station, 'station_type', 'Unknown'),
                entry_soc=self.avatar.entry_soc,
                exit_soc=self.avatar.soc,
                charge_start_time=self.avatar.charge_start_time,
                charge_end_time=current_time,
                duration_min=duration_min,
                energy_kwh=self.avatar.added_energy_kwh,
                cost=self.avatar.incurred_cost,
                price_per_kwh=price,
                decision_source=source,
            )

        success = False
        if self.current_trip_idx < len(self.schedule.trips):
            curr_trip = self.schedule.trips[self.current_trip_idx]
            start_edge = station.edge_id
            dest_edge = curr_trip.dest_loc.edge_id

            if start_edge == dest_edge:
                self.state = "IDLE"
                self.current_location = start_edge
                success = True
                print(f"🅿️ {self.id} (🔋{self.avatar.soc}) is already at destination ({station_id}), ending charge and entering standby.")
                self.on_arrive_at_destination(current_time)
            else:
                if sumo_manager.add_vehicle_to_sumo(self.id, start_edge, dest_edge):
                    self.state = "DRIVING"
                    success = True
                    print(f"🕒 {time_str} |🚗 {self.id} leaving {station_id}, heading to destination!")
                else:
                    print(f"❌ {self.id}(🔋{self.avatar.soc}) failed to inject into SUMO when leaving the station! Forced into roadside IDLE state.")
                    self.state = "IDLE"
                    self.current_location = start_edge
        else:
            self.state = "IDLE"
            self.current_location = station.edge_id
            success = True

        self.target_station_id = None
        self.last_decision = None

        return success


    async def react_to_arrival_async(self, station_id: str, cs_manager: CSManager, city_map: CityMap, current_time: datetime.datetime):
        """[Async] On-arrival psychology with expectation-violation detection.

        Triggers LLM thinking only when reality deviates sharply from expectation.
        """
        station = cs_manager.get_station(station_id)
        if not station: return

        seconds = current_time.hour * 3600 + current_time.minute * 60
        actual_price = station.get_price(seconds)
        actual_queue = station.get_wait_count()

        is_surprised = False
        surprise_reason = ""

        if self.expected_price is not None:
            if actual_price > self.expected_price + CONFIG.agent.price_surprise_tolerance:
                is_surprised = True
                surprise_reason += f"Price surged! (Exp: {self.expected_price}, Act: {actual_price}). "

            if self.expected_queue == 0 and actual_queue > 0:
                is_surprised = True
                surprise_reason += f"Unexpected Queue! (Exp: 0, Act: {actual_queue}). "

            if actual_queue > self.expected_queue + CONFIG.agent.queue_surprise_extra:
                is_surprised = True
                surprise_reason += f"Queue got much worse! (Exp: {self.expected_queue}, Act: {actual_queue}). "

        if not is_surprised:
            self.memory.add_snapshot(
                description=f"Arrived at {station.id}. Conditions normal.",
                importance=1,
                tags={"type": "routine_arrival"}
            )
        else:
            self.memory.add_episode(
                trigger="Arrival Check",
                mental_state=self.current_mental_state,
                decision="Enter Station",
                outcome=f"Surprise: {surprise_reason}",
                evaluation="REGRET"
            )
            print(f"😲 {self.id} arrival surprise: {surprise_reason}")

        self.expected_price = None
        self.expected_queue = None


    async def react_to_departure_async(self, station_id: str, cs_manager: CSManager, city_map: CityMap, current_time: datetime.datetime):
        """[Departure settlement] Close the loop: combine the prior decision intent with the actual outcome into an Episode."""
        cost = self.avatar.incurred_cost
        energy = self.avatar.added_energy_kwh
        duration_min = 0
        if hasattr(self, 'arrival_time') and self.arrival_time:
            duration_min = int((current_time - self.arrival_time).total_seconds() / 60)

        lateness_min = 0
        if self.current_trip_idx < len(self.schedule.trips):
            trip = self.schedule.trips[self.current_trip_idx]
            if current_time > trip.depart_time:
                lateness_min = int((current_time - trip.depart_time).total_seconds() / 60)

        outcome_str = f"Cost ${cost:.2f}, Charged {energy:.1f}kWh, Late {lateness_min}m"

        eval_result = self._calculate_evaluation(
            lateness_min=lateness_min,
            cost=cost,
            final_soc=self.avatar.soc,
            energy_gained=energy
        )

        if self.pending_decision_context:
            ctx = self.pending_decision_context

            self.memory.add_episode(
                trigger=ctx["trigger"],
                mental_state=ctx["mental_state"],
                decision=ctx["decision"],
                outcome=outcome_str,
                evaluation=eval_result
            )

            print(f"🧠 [Loop Closed] {self.id}: {ctx['decision']} -> {eval_result} ({outcome_str})")

            self.pending_decision_context = None
        else:
            self.memory.add_snapshot(
                description=f"Finished charging at {station_id}. {outcome_str}",
                importance=3,
                tags={"type": "auto_charge_report"}
            )

        self.daily_expense += cost
        self.daily_energy += energy
        self.arrival_time = None

    def on_arrive_at_charging_station(self, station_id: str, cs_manager: CSManager, city_map: CityMap, current_time: datetime.datetime):
        """Main-thread callback: physical station entry + intent verification and memory generation."""
        station = cs_manager.get_station(station_id)
        if not station: return

        if station.has_vehicle(self.id):
            print(f"⚠️ {self.id} is already in the station ({station_id}), correcting state.")
            if self.avatar.status == "Charging":
                self.state = "CHARGING"
            else:
                self.state = "IDLE"
            self.current_location = station.edge_id
            self.target_station_id = None
            return

        if station.add_vehicle(self.avatar):
            if self.avatar.status == "Charging":
                self.state = "CHARGING"
                self.current_location = station.edge_id
                self.arrival_time = current_time
                self.avatar.reset_charging_session()

                self.avatar.entry_soc = self.avatar.soc
                self.avatar.charge_start_time = current_time
                self.avatar.entry_station_id = station_id

                asyncio.create_task(self.react_to_arrival_async(station_id, cs_manager, city_map, current_time))

                print(f"🔌 {self.id} (🔋{self.avatar.soc:.2f}) successfully plugged into {station_id}")

                self.target_station_id = None

            elif self.avatar.status == "ParkingWithoutCharging":
                self.state = "IDLE"
                self.current_location = station.edge_id
                self.current_mental_state = "ALERT"

                is_targeted_failure = (self.target_station_id == station_id)

                if is_targeted_failure:
                    print(f"📉 {self.id} decision collapsed! Target station {station_id} is full. Writing FAILURE memory.")
                    self.memory.add_episode(
                        trigger="Arrival at Target Station",
                        mental_state="COMMITTED",
                        decision="Charge Here",
                        outcome="Station Full (Forced Parking)",
                        evaluation="FAILURE"
                    )
                else:
                    pass

                self.target_station_id = None

        else:
            print(f"❌ Entry to station {station_id} rejected. Forcing to IDLE.")
            self.state = "IDLE"
            self.current_mental_state = "ALERT"

            if self.target_station_id == station_id:
                self.memory.add_episode(
                    trigger="Charging Attempt",
                    mental_state="COMMITTED",
                    decision="Enter Station",
                    outcome="Access Denied",
                    evaluation="FAILURE"
                )

            self.target_station_id = None

    def on_arrive_at_destination(self, current_time: datetime.datetime):
        time_str = current_time.strftime("%H:%M")

        if self.current_trip_idx < len(self.schedule.trips):
            trip = self.schedule.trips[self.current_trip_idx]
            delay_min = (current_time - trip.latest_arrival_time).total_seconds() / 60
            self.current_location = trip.dest_loc.edge_id

            log_entry = {
                "trip_id": trip.trip_id,
                "trip_idx": self.current_trip_idx,
                "origin": trip.origin_loc.type,
                "dest": trip.dest_loc.type,
                "distance_km": trip.distance_km,
                "latest_arrival_time": trip.latest_arrival_time.strftime("%H:%M"),
                "actual_arrival": current_time.strftime("%H:%M"),
                "delay_min": int(delay_min)
            }
            self.daily_trip_logs.append(log_entry)
            print(f"🕒 {time_str} |🏁 {self.id} (🔋{self.avatar.soc:.2f}) arrived at destination ({trip.dest_loc.type}) (Delay: {int(delay_min)}m)")

        self.current_trip_idx += 1

        if self.current_trip_idx < len(self.schedule.trips):
            self.state = "IDLE"
            self.current_mental_state = "ALERT"
        else:
            self.state = "IDLE"
            self.current_mental_state = "DEEP_FOCUS"

            print(f"🛌 {self.id} is back home (schedule complete), entering deep sleep.")

        if self.pending_decision_context:
            ctx = self.pending_decision_context
            if ctx["decision"] in ["KEEP_DRIVING", "STAY", "STAY_AND_WAIT", "START_TRIP", "FORCE_DEPART"]:
                final_soc = self.avatar.soc
                lateness_min = 0
                if self.daily_trip_logs:
                    lateness_min = self.daily_trip_logs[-1].get('delay_min', 0)

                eval_result = self._calculate_evaluation(
                    lateness_min=max(0, lateness_min),
                    cost=0.0,
                    final_soc=final_soc,
                    energy_gained=0.0
                )
                self.memory.add_episode(
                    trigger=ctx["trigger"],
                    mental_state=ctx["mental_state"],
                    decision=ctx["decision"],
                    outcome=f"Arrived safely. Final SoC: {final_soc*100:.1f}%",
                    evaluation=eval_result
                )
                print(f"🧠 [Loop Closed] {self.id}: {ctx['decision']} -> {eval_result}")
            self.pending_decision_context = None

    async def perform_daily_reflection_async(self,
                                             current_time: datetime.datetime,
                                             cs_manager: CSManager,
                                             city_map: CityMap,
                                             day_index: int,
                                             logger=None,
                                             avg_price: float = 1.0):
        """[Step 4: Orchestrator] Three-stage reflection pipeline (refactored).

        Flow: Accountant -> Data Collector -> Psychologist (Intent) -> Scheduler (Calc).
        """
        print(f"🌙 {self.id} running three-level deep reflection (Day {day_index})...")

        stats = self.planner._preprocess_daily_stats(
            self.daily_trip_logs,
            self.daily_expense,
            self.daily_energy,
            avg_price
        )

        today_avg = stats.get('avg_cost_per_kwh', 0.0)
        if today_avg > 0.01:
            old_anchor = self.history_avg_price
            self.history_avg_price = CONFIG.agent.price_anchor_new * today_avg + CONFIG.agent.price_anchor_old * old_anchor
            print(f"🧠 {self.id} price anchor reshaped: {old_anchor:.2f} -> {self.history_avg_price:.2f}")

        negative_episodes = []
        if hasattr(self.memory, 'recent_memories_buffer'):
            for mem in self.memory.recent_memories_buffer:
                if mem.type == "EPISODE" and mem.created_at.date() == current_time.date():
                    if hasattr(mem, 'evaluation') and mem.evaluation in ["FAILURE", "REGRET"]:
                        negative_episodes.append(mem.get_semantic_content())

        if negative_episodes:
            print(f"📉 [Reflection] Collected {len(negative_episodes)} negative episodes.")

        if CONFIG.ablation.baseline_mode == "mnl":
            evolution = {
                "delta_anxiety": 0.0,
                "delta_price_sensitivity": 0.0,
                "new_energy_strategy": self.profile.energy_strategy,
                "strategic_intent": "CONVENTIONAL_BASELINE",
                "learned_rules": [],
            }
        else:
            evolution = await self.planner._analyze_evolution_async(
                stats,
                self.profile,
                self.daily_news_archive,
                negative_episodes,
                current_time=current_time
            )

        old_a = self.profile.range_anxiety_level
        old_p = self.profile.price_sensitivity

        new_a = max(0.0, min(1.0, old_a + evolution['delta_anxiety']))
        new_p = max(0.0, min(1.0, old_p + evolution['delta_price_sensitivity']))

        self.profile.range_anxiety_level = new_a
        self.profile.price_sensitivity = new_p
        self.profile.energy_strategy = evolution['new_energy_strategy']

        if logger:
            logger.log_evolution(
                day=day_index,
                agent_id=self.id,
                role=self.profile.role_type,
                old_a=old_a, delta_a=evolution['delta_anxiety'], new_a=new_a,
                old_p=old_p, delta_p=evolution['delta_price_sensitivity'], new_p=new_p,
                insight=evolution['strategic_intent']
            )

        if CONFIG.ablation.baseline_mode == "mnl":
            plan = {"trip_adjustments": []}
        else:
            plan = await self.planner._plan_schedule_async(
                self.daily_trip_logs,
                evolution['strategic_intent'],
                current_time=current_time
            )

        for trip in self.schedule.trips:
            trip.depart_time += datetime.timedelta(days=1)
            if hasattr(trip, 'latest_arrival_time'):
                trip.latest_arrival_time += datetime.timedelta(days=1)

        shift_log = []
        for adj in plan['trip_adjustments']:
            idx = adj['trip_index']
            units = adj['shift_units']

            offset_min = units * CONFIG.agent.schedule_shift_unit_min

            if 0 <= idx < len(self.schedule.trips) and offset_min != 0:
                target_trip = self.schedule.trips[idx]
                target_trip.depart_time += datetime.timedelta(minutes=offset_min)


                sign = "+" if offset_min > 0 else ""
                shift_log.append(f"T{idx}({sign}{offset_min}m)")

        if shift_log:
            print(f"📈 {self.id} itinerary optimized: {evolution['strategic_intent']} -> [{', '.join(shift_log)}]")
        else:
            print(f"📈 {self.id} itinerary kept: {evolution['strategic_intent']}")

        if logger:
            total_dist_km = sum([log.get('distance_km', 0.0) for log in self.daily_trip_logs])

            avg_cost_per_km = 0.0
            if total_dist_km > 0.1:
                avg_cost_per_km = self.daily_expense / total_dist_km

            role = getattr(self.profile, 'role_type', 'Unknown')
            for trip_log in self.daily_trip_logs:
                logger.log_trip(
                    day=day_index,
                    agent_id=self.id,
                    role=role,
                    trip_data=trip_log,
                    daily_cost_per_km=avg_cost_per_km
                )

        for rule in evolution['learned_rules']:
            self.memory.add_heuristic(rule_text=rule, day_index=day_index)

        summary_text = (
            f"Day {day_index} Summary: Cost ${self.daily_expense:.2f}, "
            f"Delay {stats['total_delay_min']}m. "
            f"Intent: {evolution['strategic_intent']}. "
            f"Shifts: {', '.join(shift_log) if shift_log else 'None'}."
        )
        self.memory.add_snapshot(
            description=summary_text,
            importance=8,
            tags={"type": "daily_summary", "day": day_index}
        )

        self.daily_expense = 0.0
        self.daily_energy = 0.0
        self.daily_trip_logs = []
        self.daily_news_archive = []

    def _calculate_evaluation(self, lateness_min: int, cost: float, final_soc: float, energy_gained: float = 0) -> str:
        """[Research-grade utility evaluation] Compute subjective utility from profile traits and map it to a label.

        Utility = w_time*(-Delay) + w_money*(-Cost) + w_anxiety*(SoC_Comfort) + w_gain*Energy
        """
        w_time = CONFIG.agent.w_time_base
        w_money = CONFIG.agent.w_money_base
        w_anxiety = CONFIG.agent.w_anxiety_base

        eff_role = CONFIG.ablation.uniform_param_role if CONFIG.ablation.uniform_numeric_params else self.profile.role_type
        eff_price_trait = CONFIG.ablation.uniform_param_price_trait if CONFIG.ablation.uniform_numeric_params else self.profile.price_trait
        eff_anxiety_trait = CONFIG.ablation.uniform_param_anxiety_trait if CONFIG.ablation.uniform_numeric_params else self.profile.anxiety_trait

        if eff_role == "COMMUTER":
            w_time = CONFIG.agent.w_time_commuter
        elif eff_role == "GIG_WORKER":
            w_time = CONFIG.agent.w_time_gig

        if eff_price_trait == "SENSITIVE":
            w_money = CONFIG.agent.w_money_sensitive
        else:
            w_money = CONFIG.agent.w_money_insensitive

        if eff_anxiety_trait == "ANXIOUS":
            w_anxiety = CONFIG.agent.w_anxiety_anxious
        else:
            w_anxiety = CONFIG.agent.w_anxiety_calm

        u_time = CONFIG.agent.u_time_per_min * max(0, lateness_min) * w_time

        u_cost = CONFIG.agent.u_cost_per_dollar * cost * w_money

        if final_soc < 0.2:
            u_soc = CONFIG.agent.u_soc_critical_scale * (0.2 - final_soc) * 100 * w_anxiety
        elif final_soc < 0.5:
            u_soc = CONFIG.agent.u_soc_climb * (final_soc - 0.2) * 100 * w_anxiety
        else:
            u_soc = CONFIG.agent.u_soc_saturated * w_anxiety

        u_gain = 0
        if energy_gained > 0:
            u_gain = CONFIG.agent.u_gain_per_kwh * energy_gained

        total_utility = u_time + u_cost + u_soc + u_gain

        if total_utility < CONFIG.agent.eval_failure:
            return "FAILURE"
        elif total_utility < CONFIG.agent.eval_regret:
            return "REGRET"
        elif total_utility < CONFIG.agent.eval_acceptable:
            return "ACCEPTABLE"
        else:
            return "SUCCESS"
