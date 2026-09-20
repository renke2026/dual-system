"""Shared data structures used across the simulation."""

import datetime
from typing import List, Optional, Literal, TypedDict, Any, Dict
from uuid import uuid4, UUID
from pydantic import BaseModel, Field
from config import CONFIG


# Memory type enum
MemoryType = Literal["SNAPSHOT", "EPISODE", "HEURISTIC"]


class MemoryBase(BaseModel):
    """Base class for all memories, holding common metadata."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    type: MemoryType
    importance: int = Field(default=1, ge=0, le=10)
    source: str = Field(default="System", description="Source: System, Cognition, Reflection")

    def get_semantic_content(self) -> str:
        """Return the text used for embedding; must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement get_semantic_content")


class MemorySnapshot(MemoryBase):
    """[Short-term] Records an environment observation or transient state.

    Short-lived; most snapshots are filtered out and never enter long-term storage.
    """
    type: Literal["SNAPSHOT"] = "SNAPSHOT"
    description: str = Field(..., description="Natural-language description, e.g. 'Arrived at Station A'.")
    context_tags: Dict[str, Any] = Field(default_factory=dict, description="Auxiliary tags")

    def get_semantic_content(self) -> str:
        return f"Snap: {self.description}"


class MemoryEpisode(MemoryBase):
    """[Core memory] CASE model (Context-Action-State-Effect).

    The most research-valuable data, used for System-2 retrieval of similar dilemmas.
    """
    type: Literal["EPISODE"] = "EPISODE"

    trigger_event: str = Field(..., description="Event that triggered the reasoning, e.g. 'Low Battery (15%)' or 'Price Surge'")
    mental_state: str = Field(..., description="Mental state at decision time, e.g. 'PANIC', 'COMMITTED', 'LIGHT_SLEEP'")
    decision: str = Field(..., description="The decision made, e.g. 'Reroute to FCS'")
    outcome: str = Field(..., description="Objective outcome, e.g. 'Late by 10min', 'Cost $15'")
    evaluation: Literal["SUCCESS", "ACCEPTABLE", "FAILURE", "REGRET"] = Field(..., description="Self-evaluation of this decision")

    def get_semantic_content(self) -> str:
        """Compressed causal text: trigger (mental) -> decision -> outcome [eval]."""
        return (
            f"Epi: {self.trigger_event} ({self.mental_state}) -> "
            f"{self.decision} -> "
            f"{self.outcome} [{self.evaluation}]"
        )


class MemoryHeuristic(MemoryBase):
    """[Long-term] A rule distilled through reflection."""
    type: Literal["HEURISTIC"] = "HEURISTIC"
    rule_text: str = Field(..., description="Abstract rule, e.g. 'Avoid FCS during morning rush'.")
    derived_from_day: int = Field(default=0, description="Which day the rule was summarized from")

    def get_semantic_content(self) -> str:
        return f"[RULE] {self.rule_text}"


# Charging strategy enum (global, shared by planner and other modules)
EnergyStrategyType = Literal[
    "CONVENIENCE_FIRST",    # go to the nearest, ignore price
    "COST_MINIMIZER",       # compare Home/Work/Public prices, pick the lowest
    "RANGE_PROTECTOR",      # anxiety-first: look for a charger at low SoC regardless of price
    "GRID_TRADER"           # V2G arbitrage: seek discharging opportunities (advanced)
]


class NewsMessage(BaseModel):
    """A broadcast message from the environment."""
    timestamp: datetime.datetime = Field(..., description="News publication time")
    source: str = Field(..., description="Source: Grid, Traffic, Weather")
    category: Literal["PRICE_SURGE", "GRID_ALERT", "TRAFFIC_JAM", "WEATHER"]
    content: str = Field(..., description="Natural-language description, readable by the LLM")
    priority: Literal["CRITICAL", "HIGH", "NORMAL"] = Field(default="NORMAL", description="Priority")

    def __str__(self):
        return f"[{self.category}] {self.content}"


class ChargingStationInfo(BaseModel):
    """A snapshot of a single charging station."""
    station_id: str
    station_type: str = "FCS"
    distance_km: float
    price_per_kwh: float
    queue_length: int = 0
    estimated_wait_time_min: int = 0


class EnvironmentObservation(BaseModel):
    """The world as seen by an agent at a given time step.

    The numeric environment is converted into this object, then into a prompt.
    """
    current_time: datetime.datetime

    soc: float = Field(..., ge=0.0, le=1.0, description="Current battery (State of Charge), 0.0-1.0")
    location: str = Field(..., description="Current location coordinates or description")
    current_speed: float = Field(default=0.0, description="Current speed km/h")
    charging_power_kw: float = Field(default=0.0, description="Current charging speed if plugged in.")
    traffic_status: Literal["Fluency", "Moderate", "Congested", "Unknown"] = "Unknown"
    nearby_stations: List[ChargingStationInfo] = Field(default_factory=list, description="List of charging stations within sight")
    last_action_feedback: Optional[str] = Field(default=None, description="Result of the previous action (e.g., 'Navigation Failed').")
    dest_charger_info: Optional[str] = Field(default=None, description="Info about charger at destination, e.g., 'SCS Available at Work'")
    market_avg_fcs_price: float = Field(default=0.0, description="Market average FCS (fast-charging) price")
    market_avg_scs_price: float = Field(default=0.0, description="Market average SCS (slow-charging) price")
    grid_status_intuition: str = Field(default="Unknown", description="Macro-level grid status description")
    grid_surge_multiplier: float = Field(default=1.0, description="Current grid surge multiplier (The Surge Factor)")
    lateness_min: int = Field(default=0, description="Minutes the current trip is delayed (Current Time - Scheduled Departure)")
    upcoming_schedule_text: Optional[str] = Field(default=None, description="Description of the next trip")


class VehicleSpec(BaseModel):
    """Physical vehicle parameters (defaults come from config)."""
    capacity_kwh: float = Field(default=CONFIG.vehicle.capacity_kwh, description="Total battery capacity (Tesla Model Y Long Range)")
    max_charge_power_kw: float = Field(default=CONFIG.vehicle.max_charge_power_kw, description="Maximum fast-charging power")
    max_slow_power_kw: float = Field(default=CONFIG.vehicle.max_slow_power_kw, description="Maximum slow-charging power")
    car_model: str = Field(default=CONFIG.vehicle.car_model, description="Vehicle model description")


class AgentProfile(BaseModel):
    """An agent's persona: qualitative traits (for the LLM) plus quantitative values (for the physics engine)."""
    agent_id: str
    name: str

    role_type: Literal["COMMUTER", "GIG_WORKER"] = Field(..., description="Occupational role that determines schedule constraints")
    price_trait: Literal["SENSITIVE", "INSENSITIVE"] = Field(..., description="Price preference label")
    anxiety_trait: Literal["ANXIOUS", "CALM"] = Field(..., description="Anxiety preference label")
    personality_traits: str = Field(..., description="Detailed personality description text")

    range_anxiety_level: float = Field(..., ge=0.0, le=1.0, description="Anxiety threshold (0-1)")
    price_sensitivity: float = Field(..., ge=0.0, le=1.0, description="Price sensitivity (0-1)")

    vehicle: VehicleSpec = Field(default_factory=VehicleSpec)
    energy_strategy: EnergyStrategyType = Field(default="CONVENIENCE_FIRST", description="Current charging strategy mode")

    def get_system_prompt(self) -> str:
        """Build the system prompt from the qualitative trait labels."""

        # Ablation: a uniform persona overrides every agent's own traits.
        if CONFIG.ablation.uniform_persona:
            role = CONFIG.ablation.uniform_persona_role
            price = CONFIG.ablation.uniform_persona_price_trait
            anxiety = CONFIG.ablation.uniform_persona_anxiety_trait
        else:
            role = self.role_type
            price = self.price_trait
            anxiety = self.anxiety_trait

        role_desc = ""
        if role == "COMMUTER":
            role_desc = "You have a strict 9-to-5 schedule. Being late for work is NOT acceptable."
        else:
            role_desc = "You drive for a living. Time is money, but charging costs eat into your profits."

        price_desc = ""
        if price == "SENSITIVE":
            price_desc = (
                "You are HIGHLY SENSITIVE to charging prices. "
                "You prefer waiting for off-peak hours or finding cheaper slow chargers. "
                "You hate paying surge prices."
            )
        else:
            price_desc = (
                "You are NOT sensitive to price. "
                "You prioritize convenience and speed over cost. "
                "You are willing to pay extra for fast charging to save time."
            )

        anxiety_desc = ""
        if anxiety == "ANXIOUS":
            anxiety_desc = (
                "You have HIGH range anxiety. "
                "You get nervous when battery drops below 50%. "
                "You prefer to keep your battery high just in case."
            )
        else:
            anxiety_desc = (
                "You are CALM about your range. "
                "You are comfortable driving until the battery is very low (e.g., 10-20%). "
                "You trust you can find a charger when really needed."
            )

        return f"""
        Role: You are {self.name}, a {role}.

        [Core Personality]
        {self.personality_traits}

        [Behavioral Guidelines]
        1. **Schedule**: {role_desc}
        2. **Spending Habit**: {price_desc}
        3. **Risk Tolerance**: {anxiety_desc}

        Your decisions MUST reflect these traits. Do not break character.
        """


# Which system made the decision.
TransitionSource = Literal[
    "SYS1_REFLEX",      # rules/reflexes/physiological needs (no LLM)
    "SYS2_COGNITION"    # deliberate LLM reasoning (spends tokens)
]

# Cognitive intent: System-2 configuring how System-1 should behave.
MentalState = Literal[
    "ALERT",        # check needs every frame, ready to trigger thinking
    "COMMITTED",    # block ordinary distractions, execute the current plan
    "LIGHT_SLEEP",  # set an alarm and wait, ignore small fluctuations
    "DEEP_FOCUS"    # asleep at home or long-haul driving, ignore all non-fatal signals
]

# Physical action instruction (attribution prefix removed).
PhysicsAction = Literal[
    "AUTO_DEPART",       # System 1: time/rule triggered departure
    "FORCE_DEPART",      # System 2: force departure, ignoring anxiety

    "AUTO_UNPLUG",       # System 1: full battery auto unplug
    "COGNITIVE_DEPART",  # System 2: not full, but decide to unplug and leave

    "FIND_CHARGER",      # from a standstill, decide to find a charger
    "CHANGE_STATION",    # switch stations mid-route
    "ABORT_CHARGE",      # give up charging, return to original destination
    "REROUTE",           # generic reroute (reserved)

    "STAY",              # inert stationary (no explicit purpose)
    "STAY_AND_WAIT",     # purposeful waiting (e.g., for a price drop)
    "KEEP_DRIVING",      # keep driving
    "KEEP_CHARGING",     # keep charging

    "RESET"
]


class TransitionPacket(TypedDict):
    """The only legal return structure of state-machine handlers (_handle_state_xxx)."""

    source: TransitionSource        # who decided (for attribution analysis)
    action: PhysicsAction           # what physical action to execute in SUMO
    new_mental_state: MentalState   # next-frame intent (configures System 1)
    reason: str                     # reason description (for debug)

    next_state: Optional[str]       # next physical state (IDLE, DRIVING, CHARGING...)
                                    # None usually means the executor decides dynamically

    target_id: Optional[str]        # target station or destination id (DEPART/REROUTE)
    wait_duration: Optional[int]    # wait/alarm duration (WAIT/LIGHT_SLEEP)
    target_soc: Optional[float]     # target SoC (KEEP_CHARGING)

    trip: Optional[Any]             # related Trip object (DEPART)
    station: Optional[Any]          # related Station object (DEPART/UNPLUG)

    llm_data: Optional[Dict[str, Any]]  # raw LLM output when SYS2 (for detailed logging)


class PerceptionContext(BaseModel):
    """Fuzzy perception context using linguistic variables instead of boolean alarms."""
    price_status: str = Field(..., description="e.g., 'Cheap', 'Fair', 'Expensive', 'Extremely Overpriced'")
    price_ratio: float = Field(..., description="Current/Average ratio")

    battery_status: str = Field(..., description="e.g., 'Comfortable', 'Anxious', 'Critical'")
    safety_margin: float = Field(..., description="SoC margin above panic threshold")

    schedule_status: str = Field(..., description="e.g., 'Early', 'On Time', 'Late', 'Severely Late'")

    description_text: str = Field(..., description="Natural language summary of the perception.")


if __name__ == "__main__":
    print("=== Testing Data Structures ===")

    try:
        driver = AgentProfile(
            agent_id="agent_001",
            name="Zhang Wei",
            role_type="GIG_WORKER",
            price_trait="SENSITIVE",
            anxiety_trait="ANXIOUS",
            personality_traits="Frugal and meticulous; willing to drive extra to save a few cents.",
            range_anxiety_level=0.7,
            price_sensitivity=0.9
        )
        print(f"✅ Agent created successfully: {driver.name} ({driver.role_type})")
        print(f"System Prompt preview:\n{driver.get_system_prompt()[:100]}...\n")
    except Exception as e:
        print(f"❌ AgentProfile test failed: {e}")

    try:
        snap = MemorySnapshot(description="Test Snapshot", importance=1)
        print(f"✅ Snapshot Semantic: {snap.get_semantic_content()}")
    except Exception as e:
        print(f"❌ MemorySnapshot test failed: {e}")

    try:
        epi = MemoryEpisode(
            trigger_event="Low Battery",
            mental_state="PANIC",
            decision="WAIT",
            outcome="Late",
            evaluation="FAILURE",
            importance=9
        )
        print(f"✅ Episode Semantic: {epi.get_semantic_content()}")
    except Exception as e:
        print(f"❌ MemoryEpisode test failed: {e}")

    print("=== Test Complete ===")
