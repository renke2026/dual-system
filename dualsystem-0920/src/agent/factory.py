import json
import random
import datetime
from uuid import uuid4
from pathlib import Path
from typing import List, Tuple
from pydantic import ValidationError

from src.common import AgentProfile
from src.env.map import CityMap, DailySchedule


class AgentFactory:
    def __init__(self, city_map: CityMap):
        """The factory depends on an externally provided CityMap so all agents share one map."""
        self.city_map = city_map

    def load_agents_from_file(self, filename: str) -> List[Tuple[AgentProfile, DailySchedule]]:
        """Load agent data from JSON and reconstruct objects (with qualitative trait labels)."""
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"❌ Agent config file not found: {path}")

        print(f"📂 Loading agents from {path}...")

        with open(path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        loaded_agents = []
        seen_ids = set()

        for item in data_list:
            agent_id = item.get('id', 'Unknown')

            try:
                if agent_id in seen_ids:
                    print(f"⚠️ Skipping duplicate agent ID: {agent_id}")
                    continue

                # Pydantic validates the new fields (role_type, price_trait, etc.).
                profile = AgentProfile(**item['profile'])

                schedule = DailySchedule(**item['schedule'])

                self._validate_schedule_edges(schedule)

                loaded_agents.append((profile, schedule))
                seen_ids.add(agent_id)

            except ValidationError as e:
                print(f"❌ [Data validation failed] Agent {agent_id} format error: {e}")
                print("💡 Hint: rerun generate_agents_file.py to generate data matching the new schema.")
            except Exception as e:
                print(f"⚠️ Skipping corrupted data item ({agent_id}): {e}")

        print(f"✅ Successfully loaded {len(loaded_agents)} agents.")
        return loaded_agents

    def _validate_schedule_edges(self, schedule: DailySchedule):
        """Check that trip edge ids exist in the loaded network."""
        if hasattr(self.city_map, 'net'):
            for trip in schedule.trips:
                if not self.city_map.net.hasEdge(trip.origin_loc.edge_id):
                    print(f"⚠️ Warning: Agent {schedule.agent_id} origin edge {trip.origin_loc.edge_id} does not exist in the map!")
                if not self.city_map.net.hasEdge(trip.dest_loc.edge_id):
                    print(f"⚠️ Warning: Agent {schedule.agent_id} destination edge {trip.dest_loc.edge_id} does not exist in the map!")
