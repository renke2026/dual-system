import sys
import os
import time
import datetime
import asyncio
from pathlib import Path
from typing import List

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))
from config import CONFIG

from src.env.stations import CSManager
from src.env.map import CityMap
from src.env.sumo_interface import SumoManager
from src.agent.factory import AgentFactory
from src.agent.core import SimulationAgent
from src.env.grid_manager import GridManager
from src.agent.npc import NPCAgent
from src.agent.memory import MemoryStream
from src.agent.planner import Planner
import json
from src.utils.logger import SimulationLogger
import sys
import traci

# Max API concurrency (Gemini free tier: 2-5 recommended).
MAX_API_CONCURRENCY = CONFIG.simulation.max_api_concurrency

class LoggerWriter:
    def __init__(self, file_path, stream=None, mode="w", shared_log=None):
        self.terminal = stream if stream is not None else sys.stdout
        self.log = shared_log if shared_log is not None else open(file_path, mode, encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

class MultiAgentSimulation:
    def __init__(self, agents_file: str = CONFIG.paths.agents_file, npcs_file_rel: str = CONFIG.paths.npcs_file):

        self.logger = SimulationLogger()
        self.semaphore = asyncio.Semaphore(MAX_API_CONCURRENCY)

        print(f"📊 Data logged to: {self.logger.base_dir}")
        print(f"🚀 Async simulation engine ready, max concurrency: {MAX_API_CONCURRENCY}")


        console_log_path = self.logger.base_dir / "console_output.log"

        _console = LoggerWriter(console_log_path, stream=sys.stdout, mode="w")
        sys.stdout = _console
        sys.stderr = LoggerWriter(console_log_path, stream=sys.stderr, shared_log=_console.log)


        MemoryStream.set_log_path(self.logger.memory_csv_path)
        MemoryStream.set_db_path(self.logger.base_dir / "chroma_db")

        Planner.set_logger(self.logger)

        print(f"📊 Data logged to: {self.logger.base_dir}")

        self.sumo_net_file = CONFIG.paths.net_file
        self.city_map = CityMap(
            net_file=self.sumo_net_file,
            taz_file=CONFIG.paths.taz_file,
            type_file=CONFIG.paths.type_file
        )
        self.cs_manager = CSManager(
            station_file=CONFIG.paths.station_file
        )

        self.sumo = SumoManager(net_file=self.sumo_net_file, gui=CONFIG.simulation.sumo_gui)
        self.sumo.start()

        print("⚡ Initializing smart grid system...")
        self.grid_manager = GridManager(
            grid_file=CONFIG.paths.grid_file,
            cs_manager=self.cs_manager
        )

        self.factory = AgentFactory(self.city_map)
        self.current_time = datetime.datetime.combine(CONFIG.simulation.base_date, datetime.time(0, 0, 0))

        self.agents: List[SimulationAgent] = []
        loaded_data = self.factory.load_agents_from_file(agents_file)

        for profile, schedule in loaded_data:
            agent = SimulationAgent(profile, schedule, self.current_time)
            self.agents.append(agent)

        print(f"✅ World created, {len(self.agents)} agents ready.")

        self.npcs: List[NPCAgent] = []
        npcs_path = project_root / npcs_file_rel
        if npcs_path.exists():
            print(f"🤖 Loading NPCs...")
            with open(npcs_path, 'r', encoding='utf-8') as f:
                npcs_data_list = json.load(f)
            for item in npcs_data_list:
                npc = NPCAgent(
                    npc_id=item['id'],
                    role_type=item.get('role_type', 'COMMUTER'),
                    strategy=item['strategy'],
                    initial_soc=item['initial_soc'],
                    initial_edge_id=item['initial_edge_id'],
                    work_edge_id=item.get('work_edge_id'),
                    max_daily_trips=item.get('max_daily_trips', CONFIG.npc.max_daily_trips_default),
                    consumption_rate=item.get('consumption_rate', CONFIG.npc.consumption_rate_default),
                    seed=item.get('seed', CONFIG.generation.npcs_seed),
                    capacity=item.get('capacity', CONFIG.npc.capacity_kwh),
                    first_trip_time=item.get('first_trip_time', "08:00"),
                    return_trip_time=item.get('return_trip_time', None)
                )
                self.npcs.append(npc)

        print(f"🕵️ [DEBUG] Loaded {len(self.npcs)} NPC objects in memory.")
        if len(self.npcs) > 0:
            print(f"   - First NPC ID: {self.npcs[0].id}")
            print(f"   - Its departure logic: Role={self.npcs[0].role_type}, IDLE={self.npcs[0].state=='IDLE'}")


        self.daily_prices = []

    async def run(self, days: int = CONFIG.simulation.days):
        """Async main run loop."""
        step_minutes = CONFIG.simulation.step_minutes
        daily_steps = (24 * 60) // step_minutes

        START_HOUR = CONFIG.simulation.start_hour
        END_HOUR = CONFIG.simulation.end_hour
        daily_steps = (END_HOUR - START_HOUR) * 60

        base_date = datetime.datetime.combine(CONFIG.simulation.base_date, datetime.time(0, 0, 0))

        print(f"🚀 Experiment config: daily run {START_HOUR}:00 - {END_HOUR}:00 ({daily_steps} minutes total)")
        print(f"🔄 Experiment mechanism: force SoC reset at 06:00 daily (Repeated Games Mode)\n")

        abl = CONFIG.ablation
        print("🧪 ═══ Decision engine & ablation switches ═══")
        print(f"   baseline_mode (decision engine)  = {abl.baseline_mode}   (full / event_llm / mnl)")
        print(f"   no_cognitive_state (strip cognitive state) = {abl.baseline_no_cognitive_state}")
        print(f"   disable_memory (disable memory)   = {abl.disable_memory}")
        print(f"   disable_pydantic (disable Pydantic validation) = {abl.disable_pydantic_validation}")
        print(f"   disable_fsm (disable FSM intercept) = {abl.disable_fsm_intercept}")
        print(f"   uniform_persona (unify personas)  = {abl.uniform_persona}")
        print(f"   uniform_numeric_params (unify params) = {abl.uniform_numeric_params}")
        print("🎲 ═══ Random seed ═══")
        print(f"   agents_seed = {CONFIG.generation.agents_seed}   npcs_seed = {CONFIG.generation.npcs_seed}")
        print("🗺️  ═══ Run scale ═══")
        print(f"   Case = {CONFIG.paths.case_name}   Days = {CONFIG.simulation.days}   LLM concurrency = {MAX_API_CONCURRENCY}")
        print("=" * 70 + "\n")

        try:
            run_start = time.time()
            for day in range(days):
                day_start = time.time()

                current_date = base_date + datetime.timedelta(days=day)
                self.current_time = datetime.datetime.combine(current_date.date(), datetime.time(START_HOUR, 0))
                current_day_idx = day + 1
                print(f"\n🌞🌞🌞 === Day {day + 1}/{days} begins ({self.current_time.date()}) === 🌞🌞🌞")
                if day > 0:
                    self.sumo.reload()

                self.logger.set_day(current_day_idx)

                MemoryStream.set_current_day(current_day_idx)
                Planner.set_current_day(current_day_idx)

                print("🔋 [System] Executing daily SoC reset & cognitive cache cleanup...")
                self.grid_manager.reset()
                for agent in self.agents:
                    original_soc = agent.schedule.initial_soc
                    agent.avatar.soc = original_soc

                    agent.current_trip_idx = 0

                    if agent.schedule.trips:
                        start_edge = agent.schedule.trips[0].origin_loc.edge_id
                        agent.current_location = start_edge

                    agent.state = "IDLE"
                    agent.target_station_id = None
                    agent.last_decision = None
                    agent.is_thinking = False

                    agent.current_mental_state = "ALERT"
                    agent.feedback_buffer = None

                    agent.news_inbox = []
                    agent.daily_news_archive = []

                    agent.expected_price = None
                    agent.expected_queue = None
                    agent.target_soc_trigger = None
                    agent.pending_decision_context = None

                    agent.daily_trip_logs = []

                print(f"✅ All agents reset.\n")

                self.daily_prices = []

                for step in range(daily_steps):
                    self.current_time += datetime.timedelta(minutes=step_minutes)

                    for _ in range(step_minutes * 60):
                        self.sumo.step()
                        self._check_sumo_events(self.current_time)

                    active_sumo_ids = set(traci.vehicle.getIDList())
                    pending_sumo_ids = set(traci.simulation.getPendingVehicles())

                    for agent in self.agents:
                        if agent.state in ["DRIVING", "DRIVING_TO_CHARGE"]:
                            if agent.id not in active_sumo_ids and agent.id not in pending_sumo_ids:
                                print(f"👻 Fixing ghost state: {agent.id} is no longer on the network, forcing arrival settlement.")
                                if agent.state == "DRIVING_TO_CHARGE":
                                    station = self.cs_manager.find_station_on_edge(agent.current_location)
                                    if station:
                                        agent.on_arrive_at_charging_station(station.id, self.cs_manager, self.city_map, self.current_time)
                                    else:
                                        agent.state = "IDLE"
                                        agent.current_mental_state = "ALERT"
                                else:
                                    agent.on_arrive_at_destination(self.current_time)

                    self.cs_manager.update_all(step_minutes, self.current_time)
                    self.grid_manager.update_power_flow(self.current_time)
                    self.grid_manager.apply_smart_charging()

                    grid_news = self.grid_manager.detect_and_generate_events(self.current_time)

                    if grid_news:
                        print(f"📢 [Broadcast] Broadcasting {len(grid_news)} urgent notices to all agents.")
                        for agent in self.agents:
                            for msg in grid_news:
                                agent.receive_news(msg, self.current_time)

                    self.npcs.sort(key=lambda x: x.id)
                    for npc in self.npcs:
                        npc.update(self.cs_manager, self.city_map, self.sumo, step_minutes, self.current_time, self.logger)

                    active_veh_ids = traci.vehicle.getIDList()
                    total_vehs = len(active_veh_ids)

                    current_speeds_mps = []
                    halting_count = 0

                    for veh_id in active_veh_ids:
                        try:
                            speed = traci.vehicle.getSpeed(veh_id)
                            if speed >= 0:
                                current_speeds_mps.append(speed)
                                if speed < 0.1:
                                    halting_count += 1
                        except traci.exceptions.TraCIException:
                            continue

                    net_mean_speed_kmh = 0.0
                    if current_speeds_mps:
                        avg_mps = sum(current_speeds_mps) / len(current_speeds_mps)
                        net_mean_speed_kmh = avg_mps * 3.6

                    time_str = self.current_time.strftime("%H:%M")
                    self.logger.log_traffic_stats(
                        time_str=time_str,
                        active_count=total_vehs,
                        halting_count=halting_count,
                        mean_speed_kmh=net_mean_speed_kmh
                    )

                    current_loads = []

                    for agent in self.agents:
                        tag = f"{agent.profile.role_type}_{agent.profile.price_trait}_{agent.profile.anxiety_trait}"

                        power = agent.avatar.last_charging_power_kw

                        current_loads.append((agent.id, tag, power))

                    time_str = self.current_time.strftime("%H:%M")
                    self.logger.log_individual_loads(time_str, current_loads)

                    tasks = []
                    for agent in self.agents:
                        task = asyncio.create_task(
                            agent.update_state_machine(
                                current_time=self.current_time,
                                dt_minutes=step_minutes,
                                cs_manager=self.cs_manager,
                                grid_manager=self.grid_manager,
                                sumo_manager=self.sumo,
                                city_map=self.city_map,
                                semaphore=self.semaphore,
                                logger=self.logger
                            )
                        )
                        tasks.append(task)

                    if tasks:
                        await asyncio.gather(*tasks)
                    time_str = self.current_time.strftime("%H:%M")

                    if hasattr(self.logger, 'log_trajectory'):
                        for agent in self.agents:
                            self.logger.log_trajectory(time_str, agent)
                            agent.has_thought_this_step = False

                    real_total_load = self.grid_manager.get_total_system_load()

                    current_mult = self.grid_manager.get_current_multiplier()
                    self.logger.log_grid(
                        time_str,
                        self.grid_manager.min_voltage,
                        real_total_load,
                        current_mult
                    )

                    for s in self.cs_manager.stations.values():
                        self.logger.log_station(time_str, s)

                    if (step + 1) % 30 == 0:
                        print()
                        self._print_stats()

                    current_mult = list(self.grid_manager.calculate_price_multiplier().values())[0] if self.grid_manager.last_voltages else 1.0
                    self.daily_prices.append(current_mult)

                print(f"zzz... End-of-day {day + 1} review ...")
                for npc in self.npcs:
                    npc.reset_daily_state(self.sumo, self.cs_manager)

                avg_price_today = sum(self.daily_prices) / len(self.daily_prices) if self.daily_prices else 1.0
                print(f"📊 Average price multiplier today: {avg_price_today:.2f}x")

                night_tasks = []
                for agent in self.agents:
                    night_tasks.append(agent.perform_daily_reflection_async(
                        self.current_time,
                        self.cs_manager,
                        self.city_map,
                        day_index=day+1,
                        logger=self.logger,
                        avg_price=avg_price_today
                    ))

                if night_tasks:
                    await asyncio.gather(*night_tasks)

                self.cs_manager.reset_all()
                day_elapsed = time.time() - day_start
                total_elapsed = time.time() - run_start
                remaining = (total_elapsed / (day + 1)) * (days - day - 1)
                print(f"✅ Day {day + 1}/{days} review complete (today {day_elapsed:.0f}s | total {total_elapsed:.0f}s | remaining ~{remaining:.0f}s).")

        except KeyboardInterrupt:
            print("\n🛑 Simulation interrupted by user.")
        except Exception as e:
            import traceback
            print("\n❌ Simulation hit a critical error (CRITICAL ERROR)!")
            print("="*40)
            traceback.print_exc()
            print("="*40)

        finally:
            self.close()


    def _check_sumo_events(self, current_time):
        """Handle SUMO arrival events."""
        arrived_ids = self.sumo.get_arrived_vehicles()
        for veh_id in arrived_ids:
            agent = next((a for a in self.agents if a.id == veh_id), None)
            if not agent: continue

            if agent.state == "DRIVING_TO_CHARGE":
                agent.on_arrive_at_charging_station(
                    agent.target_station_id,
                    self.cs_manager,
                    self.city_map,
                    self.current_time
                )
            elif agent.state == "DRIVING":
                agent.on_arrive_at_destination(self.current_time)

    def _print_stats(self):
        """[Panoramic dashboard] Monitor agent physical, cognitive and crisis states in real time."""
        loc_stats = {"home": 0, "work": 0, "other": 0, "road": 0, "charge": 0}

        mental_stats = {"FOCUS": 0, "LIGHT": 0, "ALERT": 0, "COMMIT": 0}

        risk_stats = {"safe": 0, "low": 0, "critical": 0}

        charge_type = {"FCS": 0, "SCS": 0}

        for agent in self.agents:
            if agent.state == "CHARGING":
                loc_stats["charge"] += 1
                if agent.target_station_id:
                    charge_type["SCS"] += 1
            elif agent.state in ["DRIVING", "DRIVING_TO_CHARGE"]:
                loc_stats["road"] += 1
            else:
                curr = agent.current_location
                loc_stats["other"] += 1

            ms = agent.current_mental_state
            if "FOCUS" in ms: mental_stats["FOCUS"] += 1
            elif "LIGHT" in ms: mental_stats["LIGHT"] += 1
            elif "ALERT" in ms: mental_stats["ALERT"] += 1
            elif "COMMIT" in ms: mental_stats["COMMIT"] += 1

            soc = agent.avatar.soc
            if soc < 0.2: risk_stats["critical"] += 1
            elif soc < 0.4: risk_stats["low"] += 1
            else: risk_stats["safe"] += 1

        time_str = self.current_time.strftime('%H:%M')

        print(f"🕒 {time_str} | [Phys] 🅿️ :{loc_stats['other']} 🚗:{loc_stats['road']} 🔌:{loc_stats['charge']}")

        print(f"            | [Mind] 💤:{mental_stats['FOCUS']+mental_stats['LIGHT']} 👀:{mental_stats['ALERT']} 🔥:{mental_stats['COMMIT']}")

        risk_str = ""
        if risk_stats['critical'] > 0:
            risk_str = f" 💀CRITICAL:{risk_stats['critical']}"
        if risk_stats['low'] > 0:
            risk_str += f" ⚠️LOW:{risk_stats['low']}"

        if risk_str:
            print(f"            | [Risk]{risk_str}")

    def close(self):
        self.sumo.close()
        self.logger.save_agent_memories(self.agents)
        print("Simulation finished.")


if __name__ == "__main__":
    sim = MultiAgentSimulation()
    asyncio.run(sim.run())
