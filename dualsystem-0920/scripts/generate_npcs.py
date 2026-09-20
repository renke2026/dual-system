import sys
import os
import json
import random
import argparse
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))
from config import CONFIG

from src.env.map import CityMap


def get_random_time_str(start_hour, end_hour, start_min=0, end_min=59):
    """Return a random HH:MM time string within [start, end]."""
    start_total = start_hour * 60 + start_min
    end_total = end_hour * 60 + end_min

    random_total = random.randint(start_total, end_total)

    h = random_total // 60
    m = random_total % 60
    return f"{h:02d}:{m:02d}"


def main():
    parser = argparse.ArgumentParser(description="Generate experiment control NPCs")
    parser.add_argument("--count", type=int, default=CONFIG.generation.npcs_count)
    parser.add_argument("--seed", type=int, default=CONFIG.generation.npcs_seed)
    parser.add_argument("--output", type=str, default=CONFIG.paths.npcs_file)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"🌍 Loading Map via CityMap...")
    city_map = CityMap(CONFIG.paths.net_file, CONFIG.paths.taz_file, CONFIG.paths.type_file)
    net = city_map.net

    print(f"🤖 Generating {args.count} NPCs (nuclear-grade path validation enabled)...")
    npcs_data = []
    failed_count = 0

    for i in range(args.count):
        role = "COMMUTER" if random.random() < CONFIG.generation.npc_commuter_ratio else "GIG_WORKER"
        strategy = "CHEAPEST" if random.random() < CONFIG.generation.npc_cheapest_ratio else "NEAREST"
        init_soc = round(random.uniform(CONFIG.generation.npc_soc_min, CONFIG.generation.npc_soc_max), 2)

        valid_pair_found = False
        home_loc = None
        work_loc = None

        for _ in range(CONFIG.generation.npc_path_check_attempts):
            home_loc = city_map.get_random_location("Home")

            if role == "COMMUTER":
                work_loc = city_map.get_random_location("Work")
                if work_loc.edge_id == home_loc.edge_id:
                    continue
            else:
                work_loc = city_map.get_random_location("Work")

            # Strict connectivity check against the underlying net (no straight-line fallback).
            try:
                edge_from = net.getEdge(home_loc.edge_id)
                edge_to = net.getEdge(work_loc.edge_id)

                path, cost = net.getShortestPath(edge_from, edge_to, vClass="passenger")

                if path is None:
                    continue

                path_back, cost_back = net.getShortestPath(edge_to, edge_from, vClass="passenger")
                if path_back is None:
                    continue

                valid_pair_found = True
                break

            except Exception:
                continue

        if not valid_pair_found:
            failed_count += 1
            continue

        first_trip_time = "08:00"
        return_trip_time = None

        if role == "COMMUTER":
            first_trip_time = get_random_time_str(*CONFIG.generation.npc_commuter_first_depart, 0, 59)
            return_trip_time = get_random_time_str(*CONFIG.generation.npc_commuter_return)
        else:
            first_trip_time = get_random_time_str(*CONFIG.generation.npc_gig_first_depart, 0, 59)
            return_trip_time = None

        max_daily_trips = CONFIG.npc.max_daily_trips_default
        work_edge_id = None

        if role == "COMMUTER":
            work_edge_id = work_loc.edge_id
        else:
            max_daily_trips = random.randint(*CONFIG.generation.npc_gig_max_trips)
            work_edge_id = work_loc.edge_id

        npc_json = {
            "id": f"NPC_{i}_{role}",
            "role_type": role,
            "strategy": strategy,
            "initial_soc": init_soc,
            "initial_edge_id": home_loc.edge_id,
            "work_edge_id": work_edge_id,
            "max_daily_trips": max_daily_trips,
            "first_trip_time": first_trip_time,
            "return_trip_time": return_trip_time,
            "consumption_rate": round(random.uniform(*CONFIG.generation.npc_consumption_rate), 4),
            "seed": random.randint(0, 1000000),
            "capacity": CONFIG.npc.capacity_kwh
        }
        npcs_data.append(npc_json)

    print(f"\n📊 Final report: generated {len(npcs_data)} / abandoned {failed_count}")

    output_path = Path(project_root) / args.output
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(npcs_data, f, indent=2, ensure_ascii=False)
    print(f"✅ NPC config file saved: {output_path}")


if __name__ == "__main__":
    main()
