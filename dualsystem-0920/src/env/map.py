"""Road-network map utilities built on sumolib."""

import random
import datetime
import math
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Literal
from pathlib import Path
from pydantic import BaseModel
import sumolib

ZoneType = Literal["Home", "Work", "Commercial", "Relax", "Other"]


class Location(BaseModel):
    id: str             # logical id, e.g. "Home_TAZ1_0"
    type: ZoneType      # functional type
    x: float            # coordinate X
    y: float            # coordinate Y
    edge_id: str        # corresponding SUMO edge id (e.g. "gneE1")
    taz_id: str         # traffic analysis zone id (e.g. "TAZ1")


class Trip(BaseModel):
    """A single trip."""
    trip_id: str
    origin_loc: Location
    dest_loc: Location
    depart_time: datetime.datetime
    latest_arrival_time: datetime.datetime
    distance_km: float


class DailySchedule(BaseModel):
    """A full day of trips for one agent."""
    agent_id: str
    trips: List[Trip]
    initial_soc: float = 0.8


class CityMap:
    def __init__(self,
                 net_file: str = "case/nanjing_nodes/37nodes.net.xml",
                 taz_file: str = "case/nanjing_nodes/37nodes.taz.xml",
                 type_file: str = "case/nanjing_nodes/37nodes_taz_type.txt"):

        self.locations: List[Location] = []
        self.taz_type_map: Dict[str, str] = {}

        print(f"🌍 [Sumolib] Loading road network: {net_file} ...")

        self.net = sumolib.net.readNet(net_file)

        self._load_taz_types(type_file)
        self._load_taz_xml(taz_file)

        print(f"✅ Map loaded, generated {len(self.locations)} valid locations.")

    def _load_taz_types(self, filename):
        """Parse taz_type.txt (custom format, not a SUMO standard)."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or ':' not in line:
                        continue
                    type_name, taz_ids = line.split(':')
                    for taz_id in taz_ids.split(','):
                        self.taz_type_map[taz_id.strip()] = type_name.strip()
        except FileNotFoundError:
            print(f"⚠️ Warning: cannot find {filename}, all TAZ will default to 'Other'")

    def _load_taz_xml(self, filename):
        """Parse .taz.xml and obtain coordinates via sumolib."""
        tree = ET.parse(filename)
        root = tree.getroot()

        for taz in root.findall('taz'):
            taz_id = taz.get('id')
            zone_type = self.taz_type_map.get(taz_id, "Other")

            edge_ids = set()
            for src in taz.findall('tazSource'):
                edge_ids.add(src.get('id'))
            for sink in taz.findall('tazSink'):
                edge_ids.add(sink.get('id'))

            for i, edge_id in enumerate(edge_ids):
                if self.net.hasEdge(edge_id):
                    # Filter out internal edges and station reverse connectors.
                    if edge_id.startswith(":"):
                        continue

                    if "rev" in edge_id or "CS" in edge_id:
                        if edge_id.endswith("rev"):
                            continue

                    edge_obj = self.net.getEdge(edge_id)
                    if edge_obj.getLength() < 5.0:
                        continue

                    # Use the first point of the edge shape as the location.
                    shape = edge_obj.getShape()
                    start_x, start_y = shape[0]

                    loc = Location(
                        id=f"{zone_type}_{taz_id}_{i}",
                        type=zone_type,
                        x=start_x,
                        y=start_y,
                        edge_id=edge_id,
                        taz_id=taz_id
                    )
                    self.locations.append(loc)

    def get_random_location(self, zone_type: ZoneType = None) -> Location:
        """Return a random location, optionally restricted to a zone type."""
        if zone_type:
            candidates = [l for l in self.locations if l.type == zone_type]
            if not candidates:
                fallback = random.choice(self.locations).model_copy()
                fallback.type = zone_type
                return fallback
            return random.choice(candidates)
        else:
            return random.choice(self.locations)

    def calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """Road-network distance (km) via sumolib's Dijkstra shortest path."""
        try:
            edge_from = self.net.getEdge(loc1.edge_id)
            edge_to = self.net.getEdge(loc2.edge_id)

            path, distance_m = self.net.getShortestPath(edge_from, edge_to)

            return distance_m / 1000.0
        except Exception:
            # Fall back to straight-line distance if the graph is disconnected.
            dist_m = math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)
            return (dist_m * 1.4) / 1000.0

    def get_route_distance(self, edge_id1: str, edge_id2: str) -> float:
        """Road-network distance (km) between two edge ids, for agent/NPC use."""
        if edge_id1 == edge_id2:
            return 0.0

        try:
            if not self.net.hasEdge(edge_id1) or not self.net.hasEdge(edge_id2):
                return float('inf')

            edge_from = self.net.getEdge(edge_id1)
            edge_to = self.net.getEdge(edge_id2)

            path, distance_m = self.net.getShortestPath(edge_from, edge_to)

            return distance_m / 1000.0
        except Exception as e:
            return float('inf')


if __name__ == "__main__":
    city = CityMap()

    home = city.get_random_location("Home")
    print(f"🏠 Home: {home.id} @ {home.edge_id} ({home.x:.1f}, {home.y:.1f})")

    work = city.get_random_location("Work")
    print(f"🏢 Work: {work.id} @ {work.edge_id}")

    dist = city.calculate_distance(home, work)
    print(f"🚗 Real road-network distance: {dist:.2f} km")
