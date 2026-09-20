"""CSV logging for all simulation outputs."""

import csv
import os
import json
import datetime
from pathlib import Path
from config import CONFIG


class SimulationLogger:
    def __init__(self, log_dir=CONFIG.paths.results_dir):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(log_dir) / timestamp
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_day = 1

        # [Env] Grid stats.
        self.grid_log_path = self.base_dir / "grid_stats.csv"
        self._init_csv(self.grid_log_path, ["Day", "Time", "Min_Voltage", "Total_Load_MW", "Price_Multiplier"])

        # [Env] Station stats.
        self.station_log_path = self.base_dir / "station_stats.csv"
        self._init_csv(self.station_log_path, ["Day", "Time", "Station_ID", "Type",
                                               "Queue_Len", "Occupied_Slots", "Limit_Factor",
                                               "Commuter_Q", "Commuter_Occ", "Gig_Q", "Gig_Occ"])

        # [Agent] Behavior-loop episodes.
        self.episode_log_path = self.base_dir / "episode_logs.csv"
        self._init_csv(self.episode_log_path, [
            "Day", "Time", "Agent_ID", "Role",
            "Trigger", "Mental_State", "Decision",
            "Outcome_Text", "Evaluation",
            "Importance"
        ])

        # [Agent] Trajectories (with archetype traits).
        self.traj_log_path = self.base_dir / "agent_trajectories.csv"
        self._init_csv(self.traj_log_path, [
            "Day",
            "Time",
            "Agent_ID",
            "Role",
            "Price_Trait",
            "Anxiety_Trait",
            "SoC",
            "Current_Edge",
            "Status",
            "Is_Thinking",
            "X", "Y"
        ])

        # [Agent] Cognitive evolution (daily reflection).
        self.evolution_log_path = self.base_dir / "agent_evolution.csv"
        self._init_csv(self.evolution_log_path, [
            "Day", "Agent_ID", "Role",
            "Old_Anxiety", "Delta_Anxiety", "New_Anxiety",
            "Old_PriceSens", "Delta_PriceSens", "New_PriceSens",
            "Strategic_Intent"
        ])

        # [System] Raw memory stream.
        self.memory_csv_path = self.base_dir / "raw_memories.csv"
        self._init_csv(self.memory_csv_path, [
            "Day", "Time", "Agent_ID", "Type", "Source", "Importance", "Content_Str"
        ])

        # [Global] Charging sessions (agents + NPCs).
        self.session_log_path = self.base_dir / "charging_sessions.csv"
        self._init_csv(self.session_log_path, [
            "Time", "ID", "Role", "Station_ID", "Station_Type", "Energy_kWh", "Duration_Min"
        ])

        # [Network] Macro traffic indicators (for the fundamental diagram).
        self.traffic_log_path = self.base_dir / "network_traffic.csv"
        self._init_csv(self.traffic_log_path, [
            "Day",
            "Time",
            "Active_Vehicles_Count",
            "Halting_Vehicles_Count",
            "Network_Mean_Speed_kmh",
            "Network_Flow_Proxy",
            "Congestion_Index"
        ])

        # [Trip] Trip-level OD statistics (lateness CDF and per-km cost).
        self.trips_log_path = self.base_dir / "trips.csv"
        self._init_csv(self.trips_log_path, [
            "Day",
            "Agent_ID",
            "Role",
            "Trip_ID",
            "Origin_Type",
            "Dest_Type",
            "Distance_km",
            "Planned_Arr_Time",
            "Actual_Arr_Time",
            "Delay_Min",
            "Lateness_Status",
            "Daily_Avg_Cost_Per_Km"
        ])

        # [Agent] Per-agent load breakdown (atomic level).
        self.agent_load_log_path = self.base_dir / "agent_loads.csv"
        self._init_csv(self.agent_load_log_path, [
            "Day", "Time", "Agent_ID", "Archetype", "Load_kW"
        ])

        # [Micro] Per-event charging logs (one row per charging session).
        self.charging_event_log_path = self.base_dir / "charging_events.csv"
        self._init_csv(self.charging_event_log_path, [
            "Day", "Time", "ID", "Role", "Station_ID", "Station_Type",
            "Entry_SoC", "Exit_SoC", "Charge_Start_Time", "Charge_End_Time",
            "Duration_Min", "Energy_kWh", "Cost", "Price_per_kwh", "Decision_Source"
        ])

        # [Micro] LLM usage (tokens / latency / raw-output validity).
        self.llm_usage_log_path = self.base_dir / "llm_usage.csv"
        self._init_csv(self.llm_usage_log_path, [
            "Day", "Time", "Agent_ID", "Call_Type", "Model",
            "Prompt_Tokens", "Completion_Tokens", "Total_Tokens", "API_Latency_ms",
            "Prompt_Len", "Raw_Decision", "Validated_Decision", "Is_Illegal",
            "Pydantic_Changed", "FSM_Intercepted", "Retrieved_Memories", "Raw_JSON"
        ])

    def _init_csv(self, path, headers):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def set_day(self, day: int):
        self.current_day = day

    # --- Env logs ---
    def log_grid(self, time_str, min_v, load, price_mult):
        with open(self.grid_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([self.current_day, time_str, f"{min_v:.3f}", f"{load:.3f}", f"{price_mult:.2f}"])

    def log_station(self, time_str, station):
        c_q, c_occ, g_q, g_occ = station.get_role_counts()
        with open(self.station_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_day, time_str, station.id, station.station_type,
                len(station.queue), len(station.charging_vehicles),
                f"{getattr(station, '_grid_limit_factor', 1.0):.2f}",
                c_q, c_occ, g_q, g_occ
            ])

    # --- Agent logs ---
    def log_trajectory(self, time_str, agent):
        with open(self.traj_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            role = getattr(agent.profile, 'role_type', 'Unknown')
            did_think = 1 if agent.has_thought_this_step else 0

            x, y = agent.current_position_xy if hasattr(agent, 'current_position_xy') else (0.0, 0.0)

            writer.writerow([
                self.current_day,
                time_str,
                agent.id,
                role,
                getattr(agent.profile, 'price_trait', 'Unknown'),
                getattr(agent.profile, 'anxiety_trait', 'Unknown'),
                f"{agent.avatar.soc:.4f}",
                agent.current_location,
                agent.state,
                did_think,
                f"{x:.2f}", f"{y:.2f}"
            ])

    def log_evolution(self, day, agent_id, role, old_a, delta_a, new_a, old_p, delta_p, new_p, insight):
        """Daily reflection result."""
        with open(self.evolution_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                day, agent_id, role,
                f"{old_a:.3f}", f"{delta_a:.3f}", f"{new_a:.3f}",
                f"{old_p:.3f}", f"{delta_p:.3f}", f"{new_p:.3f}",
                insight
            ])

    def save_agent_memories(self, agents):
        """Dump agent memories to a human-readable text file at the end of the run."""
        memory_file = self.base_dir / "final_memory_dump.txt"
        with open(memory_file, 'w', encoding='utf-8') as f:
            for agent in agents:
                role = getattr(agent.profile, 'role_type', 'Unknown')
                f.write(f"=== Agent: {agent.id} ({role}) ===\n")
                if hasattr(agent.memory, 'recent_memories_buffer'):
                    for mem in agent.memory.recent_memories_buffer:
                        desc = getattr(mem, 'description', 'No Desc')
                        if mem.type == 'EPISODE':
                            desc = mem.get_semantic_content()
                        elif mem.type == 'HEURISTIC':
                            desc = f"[RULE] {mem.rule_text}"
                        f.write(f"[{mem.created_at.strftime('%H:%M')}] {desc} (Imp: {mem.importance})\n")
                f.write("\n")

    def log_charging_session(self, time_str, agent_id, role, station, energy, duration):
        """Record one complete charging session (agent or NPC)."""
        with open(self.session_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                time_str, agent_id, role,
                station.id, station.station_type,
                f"{energy:.2f}", int(duration)
            ])

    def log_traffic_stats(self, time_str, active_count, halting_count, mean_speed_kmh):
        """
        Record macro traffic indicators.
        :param active_count: total active vehicles on the network (including queued)
        :param halting_count: stopped/congested vehicles (speed < 0.1 m/s)
        :param mean_speed_kmh: network-wide mean speed
        """
        # Flow proxy = density * speed (same trend as veh/h, usable for the fundamental diagram).
        flow_proxy = active_count * mean_speed_kmh

        # Congestion index = stopped vehicles / total vehicles.
        congestion_index = 0.0
        if active_count > 0:
            congestion_index = halting_count / active_count

        with open(self.traffic_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_day,
                time_str,
                active_count,
                halting_count,
                f"{mean_speed_kmh:.2f}",
                f"{flow_proxy:.2f}",
                f"{congestion_index:.4f}"
            ])

    def log_trip(self, day, agent_id, role, trip_data, daily_cost_per_km):
        """
        Record one trip's detailed data.
        :param trip_data: dict with per-trip metadata (from Agent.daily_trip_logs)
        :param daily_cost_per_km: the agent's mean per-km cost that day
        """
        delay = trip_data.get('delay_min', 0)
        status = "ON_TIME"
        if delay > 30:
            status = "SEVERE_LATE"
        elif delay > 5:
            status = "LATE"

        with open(self.trips_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                day,
                agent_id,
                role,
                trip_data.get('trip_id', 'Unknown'),
                trip_data.get('origin', 'Unknown'),
                trip_data.get('dest', 'Unknown'),
                f"{trip_data.get('distance_km', 0.0):.2f}",
                trip_data.get('latest_arrival_time', 'N/A'),
                trip_data.get('actual_arrival', 'N/A'),
                int(delay),
                status,
                f"{daily_cost_per_km:.4f}"
            ])

    def log_individual_loads(self, time_str, agent_load_list):
        """
        :param agent_load_list: List of tuples [(agent_id, archetype_tag, load_kw), ...]
        """
        with open(self.agent_load_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for (agent_id, archetype, load) in agent_load_list:
                writer.writerow([
                    self.current_day,
                    time_str,
                    agent_id,
                    archetype,
                    f"{load:.2f}"
                ])

    def log_charging_event(self, day, time_str, agent_id, role, station_id, station_type,
                           entry_soc, exit_soc, charge_start_time, charge_end_time, duration_min,
                           energy_kwh, cost, price_per_kwh, decision_source):
        """Record one LLM-agent charging session (entry/exit SoC, times, station, cost)."""

        def _fmt_soc(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else ""

        def _fmt_time(v):
            return v.strftime("%H:%M") if v is not None else ""

        with open(self.charging_event_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                day, time_str, agent_id, role, station_id, station_type,
                _fmt_soc(entry_soc),
                _fmt_soc(exit_soc),
                _fmt_time(charge_start_time),
                _fmt_time(charge_end_time),
                int(duration_min) if duration_min is not None else "",
                f"{energy_kwh:.2f}" if energy_kwh is not None else "",
                f"{cost:.2f}" if cost is not None else "",
                f"{price_per_kwh:.2f}" if price_per_kwh is not None else "",
                decision_source
            ])

    def log_llm_usage(self, day, time_str, agent_id, call_type, model,
                      prompt_tokens, completion_tokens, total_tokens, latency_ms,
                      prompt_len, raw_decision, validated_decision, is_illegal,
                      pydantic_changed, fsm_intercepted, retrieved_memories="", raw_json=""):
        """Record one LLM call: tokens, latency, prompt length, raw/validated decision, FSM correction."""
        with open(self.llm_usage_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                day, time_str, agent_id, call_type, model,
                prompt_tokens if prompt_tokens is not None else "",
                completion_tokens if completion_tokens is not None else "",
                total_tokens if total_tokens is not None else "",
                f"{latency_ms:.1f}" if latency_ms is not None else "",
                prompt_len if prompt_len is not None else "",
                raw_decision if raw_decision is not None else "",
                validated_decision if validated_decision is not None else "",
                1 if is_illegal else 0,
                1 if pydantic_changed else 0,
                1 if fsm_intercepted else 0,
                (retrieved_memories or "").replace("\n", " ").replace("\r", " ")[:300],
                (raw_json or "").replace("\n", " ").replace("\r", " ")[:500]
            ])
