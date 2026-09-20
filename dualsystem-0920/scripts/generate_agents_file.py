import sys
import os
import json
import random
import datetime
import argparse
from pathlib import Path
from typing import List, Dict, Any
from copy import deepcopy

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from config import CONFIG

from src.env.map import CityMap, Trip, Location, DailySchedule
from src.common import AgentProfile
from src.utils.llm_client import LLMFactory, AgentArchetype, RawActivity
import math


def generate_numerical_traits(price_trait: str, anxiety_trait: str):
    """Map qualitative LLM labels to concrete physical parameters (0.0 - 1.0)."""
    if price_trait == "SENSITIVE":
        p_val = random.uniform(*CONFIG.generation.sensitive_price_range)
    else:
        p_val = random.uniform(*CONFIG.generation.insensitive_price_range)

    if anxiety_trait == "ANXIOUS":
        a_val = random.uniform(*CONFIG.generation.anxious_range)
    else:
        a_val = random.uniform(*CONFIG.generation.calm_range)

    return round(p_val, 2), round(a_val, 2)


def estimate_physics_duration(dist_km: float) -> int:
    """Estimate trip duration with a nonlinear distance-dependent speed model plus Gaussian noise."""
    if dist_km <= 0.1:
        return 1

    # Piecewise nonlinear base speed by distance band.
    if dist_km < 3.0:
        base_speed = 45.0 + (dist_km / 3.0) * 5.0
    elif dist_km < 15.0:
        base_speed = 50.0 + ((dist_km - 3.0) / 12.0) * 15.0
    else:
        base_speed = 50.0 + (1 - math.exp(-0.05 * (dist_km - 15.0))) * 20.0

    speed_noise = random.gauss(0, CONFIG.generation.speed_noise_sigma)
    final_speed = max(CONFIG.generation.speed_floor, base_speed + speed_noise)

    duration_min = int((dist_km / final_speed) * 60)

    overhead = random.randint(*CONFIG.generation.overhead_min)

    return duration_min + overhead


def ground_schedule(agent_id: str, archetype: AgentArchetype, city_map: CityMap, base_date: datetime.date) -> DailySchedule:
    """Convert an abstract routine into physical trips with rigorous latest-arrival times."""
    trips: List[Trip] = []
    fixed_locations: Dict[str, Location] = {}

    def get_grounded_loc(loc_type: str) -> Location:
        if archetype.role_type == "GIG_WORKER" and loc_type == "Work":
            return city_map.get_random_location(None)
        if loc_type in ["Home", "Work"]:
            if loc_type not in fixed_locations:
                fixed_locations[loc_type] = city_map.get_random_location(loc_type)
            return fixed_locations[loc_type]
        return city_map.get_random_location(loc_type)

    routine = archetype.daily_routine

    # Force a closed loop: the last trip must end at Home.
    if routine and routine[-1].dest_type != "Home":
        last_trip = routine[-1]
        home_trip = deepcopy(last_trip)
        home_trip.time = "22:00"
        home_trip.origin_type = last_trip.dest_type
        home_trip.dest_type = "Home"
        home_trip.description = "Return Home (Auto-added)"
        routine.append(home_trip)

    current_loc = get_grounded_loc(routine[0].origin_type)

    for i, act in enumerate(routine):
        origin = current_loc
        dest = get_grounded_loc(act.dest_type)

        if origin.edge_id == dest.edge_id:
            continue

        dist_km = city_map.calculate_distance(origin, dest)
        if dist_km == float('inf'):
            dist_km = 10.0

        try:
            target_h, target_m = map(int, act.time.split(':'))
            depart_dt = datetime.datetime.combine(base_date, datetime.time(target_h, target_m))
        except:
            depart_dt = datetime.datetime.combine(base_date, datetime.time(8, 0))

        physics_duration_min = estimate_physics_duration(dist_km)

        # Anxious agents reserve a larger safety buffer, so their deadline is looser.
        if archetype.anxiety_trait == "ANXIOUS":
            safety_buffer = random.randint(*CONFIG.generation.anxious_safety_buffer)
        else:
            safety_buffer = random.randint(*CONFIG.generation.calm_safety_buffer)

        arrival_dt = depart_dt + datetime.timedelta(minutes=physics_duration_min + safety_buffer)

        trip = Trip(
            trip_id=f"{agent_id}_t_{i}",
            origin_loc=origin,
            dest_loc=dest,
            depart_time=depart_dt,
            latest_arrival_time=arrival_dt,
            distance_km=round(dist_km, 2)
        )
        trips.append(trip)
        current_loc = dest

    # Dummy fallback trip.
    if not trips:
        home = city_map.get_random_location("Home")
        dummy_time = datetime.datetime.combine(base_date, datetime.time(8, 0))
        trips.append(Trip(
            trip_id=f"{agent_id}_dummy",
            origin_loc=home,
            dest_loc=home,
            depart_time=dummy_time,
            latest_arrival_time=dummy_time + datetime.timedelta(minutes=10),
            distance_km=0
        ))

    return DailySchedule(
        agent_id=agent_id,
        trips=trips,
        initial_soc=round(random.uniform(CONFIG.generation.initial_soc_min, CONFIG.generation.initial_soc_max), 2)
    )


def load_or_generate_archetypes(force_new: bool) -> List[AgentArchetype]:
    file_path = Path(CONFIG.paths.archetypes_file)

    if file_path.exists() and not force_new:
        print(f"📂 Loading local archetype library: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if len(data) == 8:
                print("✅ Successfully loaded 8 experiment archetypes.")
                return [AgentArchetype(**item) for item in data]
            else:
                print(f"⚠️ Local file contains {len(data)} archetypes, which does not match the 8-class experiment requirement; regenerating.")
        except Exception as e:
            print(f"⚠️ Load failed ({e}), preparing to regenerate...")

    print("🧠 Calling the LLM to generate new experiment archetypes...")
    llm = LLMFactory()
    archetypes = llm.generate_archetypes()

    if archetypes:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding='utf-8') as f:
            data = [a.model_dump(mode='json') for a in archetypes]
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 New archetypes saved to {file_path}")

    return archetypes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, help="Total number of agents to generate")
    parser.add_argument("--seed", type=int, default=CONFIG.generation.agents_seed)
    parser.add_argument("--force_new", action="store_true", help="Force regenerate archetypes from LLM")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"🌍 Loading map data...")
    city_map = CityMap(CONFIG.paths.net_file, CONFIG.paths.taz_file, CONFIG.paths.type_file)

    archetypes = load_or_generate_archetypes(args.force_new)
    if not archetypes or len(archetypes) != 8:
        print("❌ Error: failed to obtain 8 valid archetypes.")
        return

    pool_commuter = [a for a in archetypes if a.role_type == "COMMUTER"]
    pool_gig = [a for a in archetypes if a.role_type == "GIG_WORKER"]

    print(f"📊 Archetype pool ready: Commuter({len(pool_commuter)}), Gig Worker({len(pool_gig)})")

    agents_data = []
    base_date = CONFIG.simulation.base_date

    print(f"\n🚀 Generating {args.count} agents (target ratio: 70% Commuter / 30% Gig)...")

    agents_data = []
    base_date = CONFIG.simulation.base_date

    INSTANCES_PER_ARCHETYPE = CONFIG.generation.instances_per_archetype
    total_agents = len(archetypes) * INSTANCES_PER_ARCHETYPE

    print(f"\n🚀 Generating {total_agents} agents (strict balanced mode: 8 archetypes x {INSTANCES_PER_ARCHETYPE} instances)...")
    print(f"ℹ️  Ignoring CLI argument --count, forcing the 8x2 generation strategy.")

    global_id_counter = 1

    for arch_idx, archetype in enumerate(archetypes):
        print(f"   - Processing archetype [{arch_idx+1}/8]: {archetype.name_tag} ({archetype.role_type}, {archetype.price_trait}, {archetype.anxiety_trait})")

        for instance_idx in range(INSTANCES_PER_ARCHETYPE):

            # Unique id embedding the trait combination.
            agent_id = f"agent_{global_id_counter}_{archetype.role_type}_{archetype.price_trait}_{archetype.anxiety_trait}"

            p_val, a_val = generate_numerical_traits(archetype.price_trait, archetype.anxiety_trait)

            profile = AgentProfile(
                agent_id=agent_id,
                name=f"{archetype.name_tag}_{instance_idx}",
                role_type=archetype.role_type,
                price_trait=archetype.price_trait,
                anxiety_trait=archetype.anxiety_trait,
                personality_traits=archetype.description,
                range_anxiety_level=a_val,
                price_sensitivity=p_val,
            )

            schedule = ground_schedule(agent_id, archetype, city_map, base_date)

            agent_json = {
                "id": agent_id,
                "type": "llm_generated",
                "profile": profile.model_dump(mode='json'),
                "schedule": schedule.model_dump(mode='json')
            }
            agents_data.append(agent_json)
            global_id_counter += 1

    output_path = Path(CONFIG.paths.agents_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(agents_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Generation complete!")
    print(f"   - Total: {len(agents_data)}")
    print(f"   - Output: {output_path}")


if __name__ == "__main__":
    main()
