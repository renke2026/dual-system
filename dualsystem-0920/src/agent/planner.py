import json
import logging
import asyncio
import os
import random
import time
from openai import AsyncOpenAI
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.common import AgentProfile, EnvironmentObservation, ChargingStationInfo, NewsMessage, EnergyStrategyType
from src.agent.memory import MemoryStream
from src.env.map import DailySchedule
from src.env.map import DailySchedule, CityMap
from src.env.stations import CSManager
import datetime
from src.common import PerceptionContext
from config import CONFIG

DEEPSEEK_API_KEY = CONFIG.secrets.deepseek_api_key
DEEPSEEK_BASE_URL = CONFIG.secrets.deepseek_base_url


class PsychologicalEvolution(BaseModel):
    delta_anxiety: float = Field(..., description="Change in anxiety (-0.1 to 0.1).")
    delta_price_sensitivity: float = Field(..., description="Change in price sensitivity (-0.1 to 0.1).")
    new_energy_strategy: EnergyStrategyType = Field(..., description="Updated charging strategy for tomorrow.")
    learned_rules: List[str] = Field(default_factory=list, description="New rules derived from today's experience.")
    strategic_intent: str = Field(..., description="High-level instruction for the Scheduler. E.g., 'PRIORITIZE_TIME: Aggressively shift trips to avoid lateness', 'MAINTAIN: Current schedule is fine', 'COST_SAVE: Shift trips to off-peak if possible'.")


class TripAdjustment(BaseModel):
    trip_index: int = Field(..., description="Index of the trip in the schedule list.")
    shift_units: int = Field(..., description="Number of 5-minute units to shift. -4 means 20 mins earlier.")


class SchedulePlan(BaseModel):
    trip_adjustments: List[TripAdjustment] = Field(default_factory=list)


class ReflectionOutput(BaseModel):
    trip_adjustments: List[TripAdjustment] = Field(..., description="Adjustments for tomorrow's trips.")

    delta_anxiety: float = Field(..., description="Change in Range Anxiety.")
    delta_price_sensitivity: float = Field(..., description="Change in Price Sensitivity.")

    daily_summary: str = Field(..., description="A short summary of today for logging.")

    learned_rules: List[str] = Field(..., description="List of high-level rules learned today. E.g., 'Avoid Station X at 8am', 'Battery drains faster in traffic'.")


def _normalize_soc(value):
    """Normalize target_soc for either convention: 0-1 fractions stay as-is; 0-100 percentages are divided by 100.

    The result is clamped to [0.0, 1.0]; unparseable input returns None.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v > 1.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))


class ActionOutput(BaseModel):
    thought_process: Optional[str] = Field(..., description="Internal monologue analyzing Trade-offs (Time vs Cost vs Anxiety).")
    decision: str = Field(..., description="Action: CHARGE | DRIVE | WAIT")
    target_station_id: Optional[str] = Field(None, description="Station ID if CHARGE is chosen, else null")
    mental_state: Literal["DEEP_FOCUS", "LIGHT_SLEEP", "ALERT", "COMMITTED"] = Field(..., description="Attention level")
    recheck_delay_min: Optional[int] = Field(default=15, description="Wake up delay")
    target_soc: Optional[float] = Field(default=None, description="Target SoC as a fraction in [0,1]. E.g. 0.8 = 80%.")
    reasoning_summary: str = Field(..., description="Short explanation.")

    @field_validator("target_soc", mode="before")
    @classmethod
    def _normalize_target_soc(cls, v):
        return _normalize_soc(v)


class Planner:
    # Class-level log bridge: scripts/main.py injects a SimulationLogger and the current day,
    # shared by every agent's Planner instance.
    _logger = None
    _current_day = 1

    @classmethod
    def set_logger(cls, logger):
        cls._logger = logger

    @classmethod
    def set_current_day(cls, day):
        cls._current_day = day

    # Decisions that must carry a valid target station (used for legality checks).
    NEED_TARGET_DECISIONS = ("FIND_CHARGER", "REROUTE_TO_CHARGER", "CHANGE_STATION")

    def __init__(self, profile: AgentProfile, memory_stream: MemoryStream):
        self.profile = profile
        self.memory_stream = memory_stream

        self.model_name = CONFIG.llm.deepseek_model
        self.USE_MOCK_LLM = CONFIG.llm.use_mock_llm

        self.client = None
        if not self.USE_MOCK_LLM:
            if not DEEPSEEK_API_KEY:
                raise ValueError(
                    "DEEPSEEK_API_KEY is not set. Provide your DeepSeek API key via the "
                    "DEEPSEEK_API_KEY environment variable before running."
                )
            self.client = AsyncOpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL
            )

    def _retrieve_relevant_memories(self, state_query: str, current_time: datetime.datetime) -> List[str]:
        """Dual-channel retrieval: similar past episodes plus a forced trauma (failure/regret) channel."""
        # Channel A: similar episodes (top-k).
        similar_mems = self.memory_stream.retrieve(
            query=state_query,
            current_time=current_time,
            top_k=CONFIG.llm.retrieve_similar_top_k,
            metadata_filter={"type": "EPISODE"}
        )

        # Channel B: forced trauma recall of past FAILURE/REGRET episodes, even at lower similarity.
        trauma_mems = self.memory_stream.retrieve(
            query=state_query + " failure regret mistake late expensive",
            current_time=current_time,
            top_k=CONFIG.llm.retrieve_trauma_top_k,
            metadata_filter={"evaluation": {"$in": ["FAILURE", "REGRET"]}}
        )

        # Channel C: high-importance environment snapshots (e.g. CRITICAL news).
        critical_snapshots = self.memory_stream.retrieve(
            query=state_query,
            current_time=current_time,
            top_k=CONFIG.llm.retrieve_critical_top_k,
            metadata_filter={
                "$and": [
                    {"type": "SNAPSHOT"},
                    {"importance": {"$gte": CONFIG.llm.critical_snapshot_importance}}
                ]
            }
        )

        all_mems = list(set(similar_mems + trauma_mems + critical_snapshots))
        return all_mems

    def _get_rule_query_keywords(self, state: str) -> str:
        """Map a physical state to intent keywords for rule retrieval (state -> rules, not question -> rules)."""
        if state == "IDLE":
            return "Planning, Schedule, Departure Time, Morning Traffic, Commute Strategy"
        elif state == "DRIVING":
            return "Range Anxiety, Traffic Congestion, Route Selection, Speed vs Energy"
        elif state == "DRIVING_TO_CHARGE":
            return "Charging Station Availability, Navigation Errors, Queueing, Station Reliability"
        elif state == "CHARGING":
            return "Charging Price, Waiting Time, Target Battery Level, When to leave"
        else:
            return "General Strategy, Cost Saving, Time Management"

    def _get_basic_env_str(self, obs: Optional[EnvironmentObservation]) -> str:
        """Block 1: basic physical environment in a compact CSV/KV style to save tokens."""
        if not obs:
            return ""

        extras = []
        if obs.charging_power_kw > 0:
            extras.append(f"ChgSpeed={obs.charging_power_kw:.1f}kW")
        if obs.dest_charger_info and obs.dest_charger_info != "Unknown":
            short_dest = obs.dest_charger_info.replace("AVAILABLE", "Avail").replace("NOT SUITABLE", "Unsuitable").replace("NOT AVAILABLE", "None")
            extras.append(f"DestChgr={short_dest}")

        extra_str = ", ".join(extras)
        if extra_str:
            extra_str = ", " + extra_str

        return f"""
        [Self State]
        Time={obs.current_time.strftime('%H:%M')}, Loc={obs.location}, SoC={obs.soc*100:.1f}%, Traffic={obs.traffic_status}{extra_str}
        """

    def _get_market_str(self, obs: Optional[EnvironmentObservation]) -> str:
        """Block 2: market intuition as a one-line summary."""
        if not obs:
            return ""
        grid_status = obs.grid_status_intuition.split('(')[0].strip()

        return f"[Market] Grid={grid_status}, AvgPrice: FCS=${obs.market_avg_fcs_price:.2f}, SCS=${obs.market_avg_scs_price:.2f}"

    def _get_station_str(self, obs: Optional[EnvironmentObservation], focus_type: str) -> str:
        """Block 3: nearby stations as a CSV table, saving ~40% tokens vs prose."""
        if not obs or not obs.nearby_stations:
            return "\n[Nearby Stations]\nNone."

        title = "[Nearby Stations (Smart Mix: Nearest & Cheapest)]"

        lines = [title, "ID, Type, Dist(km), Price($), Queue, Wait(m)"]

        for s in obs.nearby_stations:
            s_type = "FCS" if s.station_type == "FCS" else "SCS"

            line = f"{s.station_id}, {s_type}, {s.distance_km:.1f}, {s.price_per_kwh:.2f}, {s.queue_length}, {s.estimated_wait_time_min}"
            lines.append(line)

        return "\n" + "\n".join(lines) + "\n"

    def _get_crisis_str(self, obs: Optional[EnvironmentObservation]) -> str:
        """Block 4: crisis and feedback; keep only the highlighted states."""
        if not obs:
            return ""

        alerts = []

        if obs.lateness_min > 0:
            if obs.lateness_min < 15:
                tag = f"LATE ({obs.lateness_min}m)"
            elif obs.lateness_min < 30:
                tag = f"SEVERELY LATE ({obs.lateness_min}m)"
            else:
                tag = f"CRITICAL LATE ({obs.lateness_min}m)"
            alerts.append(f"SCHEDULE_STATUS: {tag}")

        if obs.last_action_feedback:
            alerts.append(f"PREV_ACTION_FAILED: {obs.last_action_feedback}")

        if not alerts:
            return ""

        return "\n[ALERTS]\n" + "\n".join(alerts) + "\n"

    def _get_schedule_str(self, obs: EnvironmentObservation) -> str:
        """Block 5: upcoming schedule preview."""
        if not obs or not obs.upcoming_schedule_text:
            return ""
        return f"""
        [Upcoming Schedule]
        {obs.upcoming_schedule_text}
        """

    def _build_context(self,
                       obs: Optional[EnvironmentObservation],
                       news: List[NewsMessage],
                       memories: List[str],
                       focus_type: str = "ALL",
                       perception: Optional[PerceptionContext] = None,
                       rules: List[str] = None,
                       include_schedule: bool = True,
                       include_stations: bool = True,
                       include_market: bool = True,
                       include_crisis: bool = True) -> str:
        """Assemble perception, memory, rules, and news into the LLM's decision context."""
        sections = []

        if perception and perception.description_text:
            alert_section = f"""
        [⚠️ INSTINCTIVE PERCEPTION]
        {perception.description_text}
        (This is your immediate situational awareness. Trust your gut.)
            """
            sections.append(alert_section)

        if rules:
            rules_text = "\n".join([f"- {r}" for r in rules])
            sections.append(f"""
        [🧠 STRATEGIC RULES (Heuristics)]
        (Lessons learned from previous days. FOLLOW THESE RULES to avoid repeating mistakes.)
        {rules_text}
            """)

        if memories:
            mem_text = "\n".join([f"- {m}" for m in memories])
            sections.append(f"""
        [📜 RELEVANT EXPERIENCES (Episodes)]
        (Similar situations from your memory. Pay attention to 'REGRET' or 'FAILURE' outcomes.)
        {mem_text}
            """)
        else:
            sections.append("\n[Relevant Experiences]\nNone recalled.")

        if obs:
            sections.append(self._get_basic_env_str(obs))

            if include_crisis:
                sections.append(self._get_crisis_str(obs))

            if include_schedule:
                sections.append(self._get_schedule_str(obs))

            if include_market:
                sections.append(self._get_market_str(obs))

            if include_stations:
                sections.append(self._get_station_str(obs, focus_type))

        if news:
            news_lines = [f"!!! {n.source}: [{n.category}] {n.content}" for n in news]
            sections.append(f"\n[BREAKING NEWS - READ CAREFULLY]\n" + "\n".join(news_lines))

        reflection_prompt = """
        [Self-Correction Protocol]
        - Check your [Relevant Experiences]: Did you fail recently in a similar situation?
        - Check [Strategic Rules]: Are you violating any learned rule?
        - If you are LATE, stop optimizing for Price. Speed is priority.
        - If you are POOR, stop optimizing for Speed. Price is priority.
        """
        sections.append(reflection_prompt)

        return "\n".join(filter(None, sections))

    def _validate_logic(self, action: ActionOutput, obs: EnvironmentObservation) -> ActionOutput:
        valid_station_ids = {s.station_id for s in obs.nearby_stations}
        if action.decision == "CHARGE":
            if not action.target_station_id or action.target_station_id not in valid_station_ids:
                action.decision = "DRIVE"
                action.target_station_id = None
                action.reasoning_summary += " (System corrected: Invalid Target)"

        if action.decision == "CHARGE" and obs.soc > CONFIG.station.fcs_full_soc:
             action.decision = "DRIVE"
             action.target_station_id = None

        if action.mental_state == "ALERT":
            action.recheck_delay_min = 1

        return action

    async def _safe_api_call(self, system_prompt: str, user_prompt: str, retries=CONFIG.llm.deepseek_retries, json_mode=True):
        """Call the LLM; returns (content, usage_dict, latency_ms).

        usage_dict holds prompt/completion/total tokens; None on mock/failure, latency 0.0.
        """
        if self.USE_MOCK_LLM:
            await asyncio.sleep(0.1)
            if "thought_process" in user_prompt:
                content = json.dumps({
                    "thought_process": "Mock thinking...",
                    "decision": "DRIVE",
                    "target_station_id": None,
                    "mental_state": "LIGHT_SLEEP",
                    "recheck_delay_min": 15,
                    "reasoning_summary": "Mock mode."
                })
            else:
                content = json.dumps({"thought": "Mock thought."})
            return content, None, 0.0

        for i in range(retries):
            try:
                t0 = time.perf_counter()
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"} if json_mode else None,
                    temperature=CONFIG.llm.deepseek_temperature
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0

                content = response.choices[0].message.content

                usage = None
                if getattr(response, "usage", None) is not None:
                    usage = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(response.usage, "completion_tokens", None),
                        "total_tokens": getattr(response.usage, "total_tokens", None),
                    }

                await asyncio.sleep(CONFIG.llm.deepseek_request_interval_s)
                return content, usage, latency_ms

            except Exception as e:
                print(f"⚠️ [DeepSeek] Error: {e}. Retrying ({i+1}/{retries})...")
                await asyncio.sleep(CONFIG.llm.deepseek_backoff_base_s * (i + 1))

        raise Exception("DeepSeek API Failed")

    async def plan_idle_action(self, obs: EnvironmentObservation, news: List[NewsMessage], perception: PerceptionContext) -> Dict[str, Any]:
        """[IDLE] Decide the next action while parked: trip trigger, low battery, or news."""
        query_context = f"I am IDLE. {perception.description_text}"
        memories = self._retrieve_relevant_memories(query_context, obs.current_time)

        state_keywords = self._get_rule_query_keywords("IDLE")
        rules = self.memory_stream.retrieve_relevant_rules(state_keywords, obs.current_time)

        context = self._build_context(
            obs, news, memories,
            perception=perception,
            rules=rules,
            include_schedule=True,
            include_stations=True,
            include_market=True,
            include_crisis=True
        )
        task_instruction = """
        [Task] Decision in IDLE state (Parked).
        [Objective] Balance Schedule (Lateness) vs Battery (Range) vs Cost.

         [Input Data Reference]
        Look at `[Upcoming Schedule]` above. Find the text: "(in X mins)".
        - **X** is your time buffer.

        [State Space]
        - DEEP_FOCUS: Long stay (Work/Sleep). Ignore minor price changes.
        - LIGHT_SLEEP: Short wait. Set `recheck_delay_min`.
        - COMMITTED: Force departure. Suppress anxiety.
        - ALERT: Active scanning.

        [Action Space]
        1. START_TRIP:
           - Description: Depart for the next destination immediately.
           - Early Departure: If current time is before schedule, you choose to leave early.
           - Requirement: Set mental_state to "COMMITTED" to lock in this trip.
           - **Warning**: Do NOT start if "TOO EARLY" unless you have a specific strategic reason (e.g. dodging predicted traffic)
        2. FIND_CHARGER:
           - Description: Reroute to a charging station instead of your destination.
           - Logic: Use this if SoC is too low for the next trip or prices are currently acceptable.
           - Requirement: Pick a `target_station_id`.
        3. STAY_AND_WAIT:
           - Description: Remain parked.
            - **CRITICAL STRATEGY for Long Waits**:
             - If the next trip is far away (e.g., > 30 mins) and you decide **NOT to charge**:
             - You MUST set `mental_state` to **"DEEP_FOCUS"**.
             - **Reason**: Do NOT set a short delay (15m). Trust the System to auto-wake you when it's time to depart.
             - Only use "LIGHT_SLEEP" with `recheck_delay_min` if you are actively waiting for a price drop or a specific station slot.


        [Logic Heuristics]
        - IF LATE -> START_TRIP (Risk) or FIND_CHARGER (Safe).
        - IF LOW BAT -> FIND_CHARGER.
        - IF HIGH PRICE -> STAY_AND_WAIT.

        Output JSON:
        {
            "thought_process": "Brief trade-off analysis...",
            "decision": "START_TRIP" | "FIND_CHARGER" | "STAY_AND_WAIT",
            "target_station_id": "ID" | null,
            "mental_state": "COMMITTED" | "ALERT" | "LIGHT_SLEEP" | "DEEP_FOCUS",
            "recheck_delay_min": int,
            "reasoning_summary": "string"
        }
        """
        return await self._execute_llm_call(
            context, task_instruction,
            allowed_decisions=["START_TRIP", "FIND_CHARGER", "STAY_AND_WAIT"],
            obs=obs,
            retrieved_memories=(memories + rules)
        )

    async def plan_driving_action(self, obs: EnvironmentObservation, news: List[NewsMessage], perception: PerceptionContext) -> Dict[str, Any]:
        """[DRIVING] Decide whether to reroute, focusing on range anxiety and remaining range."""
        query_context = f"I am DRIVING. {perception.description_text}"
        memories = self._retrieve_relevant_memories(query_context, obs.current_time)

        state_keywords = self._get_rule_query_keywords("DRIVING")
        rules = self.memory_stream.retrieve_relevant_rules(state_keywords, obs.current_time)

        context = self._build_context(
            obs, news, memories,
            rules=rules,
            perception=perception,
            include_schedule=False,
            include_stations=True,
            include_market=True,
            include_crisis=False
        )

        task_instruction = """
        [Task] Decision in DRIVING state.
        [Objective] Assess Range Anxiety vs Schedule.

        [State Space]
        - COMMITTED: Determined to reach dest. BLOCK anxiety warnings.
        - ALERT: Worried about range. Check options.

        [Action Space]
        1. KEEP_ROUTE: Continue to destination. (Req: COMMITTED/ALERT).
        2. REROUTE_TO_CHARGER: Divert immediately. (Req: ALERT). Pick `target_station_id`.

        [Logic Heuristics]
        - IF Bat > Comfort -> KEEP_ROUTE (Set COMMITTED).
        - IF Bat Critical -> REROUTE_TO_CHARGER.

        Output JSON:
        {
            "thought_process": "Brief analysis...",
            "decision": "KEEP_ROUTE" | "REROUTE_TO_CHARGER",
            "target_station_id": "ID" | null,
            "mental_state": "ALERT" | "COMMITTED",
            "reasoning_summary": "string"
        }
        """

        return await self._execute_llm_call(
            context, task_instruction,
            allowed_decisions=["KEEP_ROUTE", "REROUTE_TO_CHARGER"],
            obs=obs,
            retrieved_memories=(memories + rules)
        )

    async def plan_hunting_action(self, obs: EnvironmentObservation, news: List[NewsMessage], perception: PerceptionContext) -> Dict[str, Any]:
        """[HUNTING] Handle navigation failure or a change of mind en route to a charger."""
        query_context = f"I am DRIVING_TO_CHARGE. {perception.description_text}"
        memories = self._retrieve_relevant_memories(query_context, obs.current_time)

        state_keywords = self._get_rule_query_keywords("DRIVING_TO_CHARGE")
        rules = self.memory_stream.retrieve_relevant_rules(state_keywords, obs.current_time)

        context = self._build_context(
            obs, news, [],
            rules=rules,
            perception=perception,
            include_schedule=False,
            include_stations=True,
            include_market=True,
            include_crisis=True
        )

        task_instruction = """
        [Task] Decision in HUNTING state (En route to charger).
        [Context] Navigation failed or reconsidering plan.

        [State Space]
        - ALERT: Active hunting. Handle errors immediately.
        - COMMITTED: Abort charging, return to trip.

        [Action Space]
        1. CONTINUE_TO_STATION: Retry original target. (Transient error).
        2. CHANGE_STATION: Select NEW target. (Target unreachable/full).
        3. ABORT_AND_RETURN: Cancel charge. Go to destination. (Req: COMMITTED).

        Output JSON:
        {
            "thought_process": "Brief analysis...",
            "decision": "CONTINUE_TO_STATION" | "CHANGE_STATION" | "ABORT_AND_RETURN",
            "target_station_id": "ID" | null,
            "mental_state": "ALERT" | "COMMITTED",
            "reasoning_summary": "string"
        }
        """

        return await self._execute_llm_call(
            context, task_instruction,
            allowed_decisions=["CONTINUE_TO_STATION", "CHANGE_STATION", "ABORT_AND_RETURN"],
            obs=obs,
            retrieved_memories=(memories + rules)
        )

    async def plan_charging_action(self, obs: EnvironmentObservation, session_info: Dict[str, str], perception: PerceptionContext) -> Dict[str, Any]:
        """[CHARGING] Decide whether to stay or leave the charger."""
        query_context = f"I am CHARGING. {perception.description_text}"
        memories = self._retrieve_relevant_memories(query_context, obs.current_time)

        state_keywords = self._get_rule_query_keywords("CHARGING")
        rules = self.memory_stream.retrieve_relevant_rules(state_keywords, obs.current_time)

        rules_str = ""
        if rules:
            r_text = "\n".join([f"- {r}" for r in rules])
            rules_str = f"[🧠 STRATEGY RULES]\n{r_text}\n"

        session_str = "\n".join([f"- {k}: {v}" for k, v in session_info.items()])
        dest_str = obs.dest_charger_info if obs.dest_charger_info else "Unknown"

        schedule_status = "On Time"
        if obs.lateness_min > 0:
            schedule_status = f"⚠️ LATE by {obs.lateness_min} mins"
            if obs.lateness_min > 30: schedule_status = f"💀 CRITICAL LATE ({obs.lateness_min}m)"

        context = f"""
        {memories}
        {rules_str}
        [Current Status]
        - Time: {obs.current_time.strftime('%H:%M')}
        - Battery: {obs.soc*100:.1f}%
        - Schedule: {schedule_status}
        - Session: {session_str}
        - Dest Charger: {dest_str}
        """

        task_instruction = """
        [Task] Decision in CHARGING state.
        [Objective] Trade-off: Time (Schedule) vs Energy (SoC).

        [State Space]
        - LIGHT_SLEEP: Wait for battery. (Set `target_soc` OR `recheck_delay_min`).
        - COMMITTED: Unplug and leave NOW.

        [Action Space]
        1. UNPLUG_AND_LEAVE: Depart. (Late / Expensive / Enough Charge).
        2. CONTINUE_CHARGING: Stay. (Critical Bat / Cheap / Early).

        [Logic Heuristics]
        - LATE? -> UNPLUG_AND_LEAVE.
        - SoC < Target? -> CONTINUE_CHARGING.

        Output JSON:
        {
            "thought_process": "Brief analysis...",
            "decision": "UNPLUG_AND_LEAVE" | "CONTINUE_CHARGING",
            "mental_state": "COMMITTED" | "LIGHT_SLEEP",
            "recheck_delay_min": int,
            "target_soc": float in [0,1] (fraction, e.g. 0.8 = 80%; DO NOT use 0-100),
            "reasoning_summary": "string"
        }
        """

        return await self._execute_llm_call(
            context, task_instruction,
            allowed_decisions=["UNPLUG_AND_LEAVE", "CONTINUE_CHARGING"],
            obs=obs,
            retrieved_memories=(memories + rules)
        )

    async def _execute_llm_call(self, context, instruction, allowed_decisions=None, obs=None, retrieved_memories=None):
        """Unified LLM call: parse the raw JSON, record illegality, then apply pydantic and FSM corrections."""
        system_p = self.profile.get_system_prompt()
        full_prompt = f"{context}\n\n{instruction}"
        current_time = obs.current_time if obs is not None else None
        valid_station_ids = {s.station_id for s in obs.nearby_stations} if obs is not None else set()

        raw_decision = None
        is_illegal = False
        raw_json = ""

        try:
            json_str, usage, latency_ms = await self._safe_api_call(system_p, full_prompt, json_mode=True)
            raw_json = json_str or ""

            try:
                raw = json.loads(json_str) if isinstance(json_str, str) else {}
                if not isinstance(raw, dict):
                    raw = {}
            except Exception:
                raw = {}
            raw_decision = raw.get("decision")

            allowed = allowed_decisions or []
            if raw_decision not in allowed:
                is_illegal = True
            if raw_decision in self.NEED_TARGET_DECISIONS:
                tid = raw.get("target_station_id")
                if not tid or (valid_station_ids and tid not in valid_station_ids):
                    is_illegal = True
            ts = raw.get("target_soc")
            if ts is not None:
                try:
                    if not (0.0 <= float(ts) <= 1.0):
                        is_illegal = True
                except (TypeError, ValueError):
                    is_illegal = True

            if CONFIG.ablation.disable_pydantic_validation:
                validated = {
                    "thought_process": raw.get("thought_process"),
                    "decision": raw_decision if raw_decision else "ERROR",
                    "target_station_id": raw.get("target_station_id"),
                    "mental_state": raw.get("mental_state", "ALERT"),
                    "recheck_delay_min": raw.get("recheck_delay_min", 15),
                    "target_soc": _normalize_soc(raw.get("target_soc")),
                    "reasoning_summary": raw.get("reasoning_summary", ""),
                }
            else:
                validated = ActionOutput.model_validate_json(json_str).model_dump()

            pydantic_decision = validated.get("decision")
            pydantic_changed = (pydantic_decision != raw_decision)

            if not CONFIG.ablation.disable_fsm_intercept:
                validated = self._correct_decision(validated, valid_station_ids, obs)

            final_decision = validated.get("decision")
            fsm_intercepted = (final_decision != pydantic_decision)

            prompt_len = len(system_p) + len(full_prompt)
            self._log_llm_usage(
                current_time, "decision", usage, latency_ms,
                prompt_len, raw_decision, final_decision, is_illegal,
                pydantic_changed, fsm_intercepted, retrieved_memories, raw_json
            )

            return validated
        except Exception as e:
            print(f"❌ [Planner] LLM Call Failed: {e}")
            return {"decision": "ERROR", "reasoning_summary": str(e)}

    def _correct_decision(self, validated: Dict[str, Any], valid_station_ids, obs) -> Dict[str, Any]:
        """[FSM intercept] Correct decisions with an invalid target to a safe action, avoiding a physical crash.

        Skipped when disable_fsm_intercept=True (invalid target passes through -> navigation failure).
        """
        d = validated.get("decision")
        if d in self.NEED_TARGET_DECISIONS:
            tid = validated.get("target_station_id")
            if not tid or (valid_station_ids and tid not in valid_station_ids):
                validated["decision"] = "ERROR"
                validated["target_station_id"] = None
                validated["reasoning_summary"] = (
                    validated.get("reasoning_summary", "") + " (FSM corrected: Invalid Target)"
                )
        return validated

    def _log_llm_usage(self, current_time, call_type, usage, latency_ms,
                       prompt_len, raw_decision, validated_decision, is_illegal,
                       pydantic_changed, fsm_intercepted, retrieved_memories=None, raw_json=""):
        """Write one LLM usage record (tokens / latency / prompt length / raw-output legality / FSM intercept / memory hits)."""
        if Planner._logger is None:
            return
        if usage is None:
            usage = {}
        time_str = current_time.strftime("%H:%M") if current_time else ""
        mem_str = " || ".join(str(m) for m in retrieved_memories) if retrieved_memories else ""
        Planner._logger.log_llm_usage(
            day=Planner._current_day,
            time_str=time_str,
            agent_id=self.profile.agent_id,
            call_type=call_type,
            model=self.model_name,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            latency_ms=latency_ms,
            prompt_len=prompt_len,
            raw_decision=raw_decision,
            validated_decision=validated_decision,
            is_illegal=is_illegal,
            pydantic_changed=pydantic_changed,
            fsm_intercepted=fsm_intercepted,
            retrieved_memories=mem_str,
            raw_json=raw_json,
        )

    async def generate_reaction_async(self, obs: EnvironmentObservation, news: List[NewsMessage], event_type: str, station_name: str, context_details: Dict[str, str]):
        """[Async] Generate a reaction monologue (reuses the context blocks)."""
        system_p = self.profile.get_system_prompt()

        base_ctx = self._build_context(
            obs, news, [],
            include_stations=False,
            include_crisis=False,
            include_market=True
        )

        details_text = "\n".join([f"- {k}: {v}" for k, v in context_details.items()])

        task_instruction = f"""
        [Reaction Task]
        Event: **{event_type}** at **{station_name}**.

        [Event Details]
        {details_text}

        [Instruction]
        Write a short (1-2 sentences) internal monologue.
        **COMPARE** the event details (e.g., Price) with the `[Macro Market Intuition]` above.
        - Is it cheaper or more expensive than average?
        - Are you happy or annoyed?

        Output strictly in JSON: {{'thought': '...'}}
        """

        full_prompt = f"{base_ctx}\n\n{task_instruction}"

        try:
            res, usage, latency_ms = await self._safe_api_call(system_p, full_prompt, json_mode=True)
            self._log_llm_usage(obs.current_time, "reaction", usage, latency_ms,
                                prompt_len=None, raw_decision=None, validated_decision=None,
                                is_illegal=False, pydantic_changed=False, fsm_intercepted=False,
                                retrieved_memories=None, raw_json=res or "")
            data = json.loads(res)
            return data.get("thought", f"I am at {station_name}.")
        except:
            return f"I am at {station_name}."

    def _preprocess_daily_stats(self,
                                trip_logs: List[Dict],
                                daily_expense: float,
                                total_energy_kwh: float,
                                market_avg_price: float) -> Dict[str, Any]:
        """[Step 1: The Accountant] Objective data preprocessing, no value judgment."""
        stats = {
            "total_delay_min": 0,
            "max_single_delay_min": 0,
            "on_time_rate": 1.0,
            "avg_cost_per_kwh": 0.0,
            "price_competitiveness": 1.0
        }

        if trip_logs:
            delays = [log.get('delay_min', 0) for log in trip_logs]

            stats["total_delay_min"] = sum(delays)

            stats["max_single_delay_min"] = max(delays)

            on_time_count = sum(1 for d in delays if d <= 0)
            stats["on_time_rate"] = round(on_time_count / len(delays), 2)

        if total_energy_kwh > 0.01:
            my_avg_price = daily_expense / total_energy_kwh
            stats["avg_cost_per_kwh"] = round(my_avg_price, 2)

            if market_avg_price > 0.01:
                stats["price_competitiveness"] = round(my_avg_price / market_avg_price, 2)
            else:
                stats["price_competitiveness"] = 1.0
        else:
            stats["avg_cost_per_kwh"] = 0.0
            stats["price_competitiveness"] = 1.0

        return stats

    async def _analyze_evolution_async(self,
                                       stats: Dict[str, Any],
                                       profile: AgentProfile,
                                       news: List[NewsMessage],
                                       negative_episodes: List[str] = [],
                                       current_time: datetime.datetime = None) -> Dict[str, Any]:
        """[Step 2: The Psychologist] Decide personality evolution and strategy from the stats."""
        system_p = profile.get_system_prompt()

        stats_str = json.dumps(stats, indent=2)
        news_str = "\n".join([str(n) for n in news]) if news else "None"

        mistakes_str = "None. (Good job!)"
        if negative_episodes:
            mistakes_str = "\n".join([f"- {ep}" for ep in negative_episodes])

        task_instruction = f"""
        [Reflection Step 1: Psychological & Strategic Analysis]
        You are the agent's "Psychologist". Your job is to analyze performance and set the STRATEGY for tomorrow.
        DO NOT calculate specific schedule times. Leave that to the Scheduler.

        [Data Inputs]
        - Daily Stats: {stats_str}
        - Major Events: {news_str}
        - Current Strategy: {profile.energy_strategy}

        [Specific Mistakes Today (Negative Episodes)]
        {mistakes_str}

        [Analysis Tasks]
        1. **Personality Evolution**:
           - If `total_delay_min` > 20: FAILED schedule -> Increase Anxiety.
           - If `price_competitiveness` > 1.2: OVERSPENT -> Increase Price Sensitivity.

        2. **Root Cause Analysis & Rules**:
           - Analyze the `Negative Episodes`. Why did they happen?
           - Generate `learned_rules` to prevent recurrence. (e.g., "If battery < 20%, ignore price.")

        3. **Strategic Intent (The Mandate)**:
           - Give a high-level command to the Scheduler based on today's pain points.
           - If LATE: Intent = "PRIORITIZE_TIME: Aggressively shift departure times earlier."
           - If EXPENSIVE: Intent = "COST_OPTIMIZE: Shift trips to cheaper windows if buffer allows."
           - If OK: Intent = "MAINTAIN: Minor tweaks only."

        Output JSON:
        {{
            "delta_anxiety": 0.05,  // Float between -0.1 and 0.1
            "delta_price_sensitivity": -0.02, // Float between -0.1 and 0.1
            "new_energy_strategy": "COST_MINIMIZER", // Must be one of: CONVENIENCE_FIRST, COST_MINIMIZER, RANGE_PROTECTOR, GRID_TRADER
            "learned_rules": ["Avoid charging at Station A during morning rush hour."],
            "strategic_intent": "MAINTAIN: System error fallback."
        }}
        """

        try:
            json_str, usage, latency_ms = await self._safe_api_call(system_p, task_instruction, json_mode=True)
            self._log_llm_usage(current_time, "reflection_evolution", usage, latency_ms,
                                prompt_len=None, raw_decision=None, validated_decision=None,
                                is_illegal=False, pydantic_changed=False, fsm_intercepted=False,
                                retrieved_memories=None, raw_json=json_str or "")
            return PsychologicalEvolution.model_validate_json(json_str).model_dump()
        except Exception as e:
            print(f"⚠️ [Planner] Step 2 Evolution Failed: {e}")
            return {
                "delta_anxiety": 0.0,
                "delta_price_sensitivity": 0.0,
                "new_energy_strategy": profile.energy_strategy,
                "learned_rules": [],
                "strategic_intent": "MAINTAIN: System error fallback."
            }

    async def _plan_schedule_async(self,
                                   trip_logs: List[Dict[str, Any]],
                                   strategic_intent: str,
                                   current_time: datetime.datetime = None) -> Dict[str, Any]:
        """[Step 3: The Scheduler] Execute the strategic intent as precise 5-min-unit adjustments."""
        if "MAINTAIN" in strategic_intent and "Minor tweaks" in strategic_intent:
            return {"trip_adjustments": []}

        log_lines = []
        for log in trip_logs:
            idx = log.get('trip_idx', '?')
            org = log.get('origin', 'Unknown')
            dst = log.get('dest', 'Unknown')
            planned = log.get('latest_arrival_time', 'N/A')
            actual = log.get('actual_arrival', 'N/A')
            delay = log.get('delay_min', 0)

            status = "ON TIME"
            if delay > 5: status = f"LATE (+{delay}m)"
            elif delay < -5: status = f"EARLY ({delay}m)"

            line = f"- Trip {idx} ({org}->{dst}): Planned Arr {planned}, Actual Arr {actual} -> {status}"
            log_lines.append(line)

        report_str = "\n".join(log_lines) if log_lines else "No trip data available."

        system_p = "You are an expert Transport Scheduler. You optimize schedules based on performance data. Output JSON only."

        task_instruction = f"""
        [Reflection Step 2: Precision Schedule Adjustment]

        [Strategic Mandate (From Psychologist)]
        "{strategic_intent}"

        [Yesterday's Performance Report]
        {report_str}

        [Calculation Rules]
        1. **Unit System**: All adjustments are in **5-minute units**.
           - `shift_units = -1` => 5 mins EARLIER.
           - `shift_units = +2` => 10 mins LATER.
        2. **Adjustment Logic**:
           - If a trip was LATE by X mins: You usually need to shift DEPARTURE earlier by roughly X mins (or slightly more for buffer).
           - Example: Late by 15m -> Shift approx -3 or -4 units (-15 to -20m).
           - If strategic intent is "PRIORITIZE_TIME", add extra buffer units.
        3. **Chain Reaction**:
           - If Trip 0 is shifted earlier, subsequent trips might need adjustment too, OR they might be fine. Use your judgment.

        [Task]
        Output a list of `trip_adjustments` to fix the delays and align with the strategic mandate.

        Output JSON:
        {{
            "trip_adjustments": [
                {{ "trip_index": 0, "shift_units": -4 }}, // Shift Trip 0 earlier by 20 mins
                {{ "trip_index": 1, "shift_units": 0 }}   // No change
            ]
        }}
        """

        try:
            json_str, usage, latency_ms = await self._safe_api_call(system_p, task_instruction, json_mode=True)
            self._log_llm_usage(current_time, "reflection_schedule", usage, latency_ms,
                                prompt_len=None, raw_decision=None, validated_decision=None,
                                is_illegal=False, pydantic_changed=False, fsm_intercepted=False,
                                retrieved_memories=None, raw_json=json_str or "")
            return SchedulePlan.model_validate_json(json_str).model_dump()
        except Exception as e:
            print(f"⚠️ [Planner] Step 3 Schedule Failed: {e}")
            return {"trip_adjustments": []}
