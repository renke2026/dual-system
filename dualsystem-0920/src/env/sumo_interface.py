import traci
import os
import sys
from config import CONFIG

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


class SumoManager:
    def __init__(self, net_file: str, gui: bool = None):
        if gui is None:
            gui = CONFIG.simulation.sumo_gui
        abs_net_file = os.path.abspath(net_file)
        max_end_time = CONFIG.simulation.sumo_max_end_time
        log_file = os.path.abspath(CONFIG.paths.sumo_log)
        start_time_seconds = 0
        self.sumo_cmd = [
            "sumo-gui" if gui else "sumo",
            "-n", abs_net_file,
            "--step-length", str(CONFIG.simulation.sumo_step_length),
            "--begin", str(start_time_seconds),
            "--no-step-log", "true",
            "--waiting-time-memory", str(CONFIG.simulation.sumo_waiting_time_memory),
            "--start", "true",
            "--quit-on-end", "false",
            "--log", log_file,
            "--message-log", log_file,
            "--error-log", log_file,
            "--end", str(max_end_time)
        ]
        self.started = False

    def start(self):
        if not self.started:
            traci.start(self.sumo_cmd)
            self.started = True
            print(f"🚀 SUMO started, loading network: {self.sumo_cmd[2]}")

            try:
                # Copy the default vehicle type, then set physical/visual attributes.
                traci.vehicletype.copy("DEFAULT_VEHTYPE", "type_EV")

                traci.vehicletype.setLength("type_EV", CONFIG.vehicle.length_m)
                traci.vehicletype.setWidth("type_EV", CONFIG.vehicle.width_m)
                traci.vehicletype.setMinGap("type_EV", CONFIG.vehicle.min_gap_m)

                traci.vehicletype.setShapeClass("type_EV", "passenger")
                traci.vehicletype.setColor("type_EV", (0, 255, 0))

                print("✅ [SUMO] Successfully registered vehicle type: type_EV")

            except traci.exceptions.TraCIException as e:
                print(f"⚠️ [SUMO] Vehicle type definition warning (may already exist): {e}")
            except Exception as e:
                print(f"❌ [SUMO] Vehicle type definition failed: {e}")

    def close(self):
        traci.close()

    def step(self):
        traci.simulationStep()

    def add_vehicle_to_sumo(self, veh_id: str, start_edge: str, dest_edge: str):
        """Spawn (or respawn) a vehicle in SUMO, with fault tolerance."""
        if not start_edge or not dest_edge:
            return False
        if start_edge == dest_edge:
            return True

        import random
        route_id = f"route_{veh_id}_{int(traci.simulation.getTime())}_{random.randint(1000, 9999)}"

        try:
            stage = traci.simulation.findRoute(start_edge, dest_edge)
            if not stage.edges:
                print(f"❌ SUMO routing failed: {start_edge} -> {dest_edge} (network may be disconnected)")
                return False

            # Full-state de-duplication: running, pending, and loaded lists.
            running_list = traci.vehicle.getIDList()
            pending_list = traci.simulation.getPendingVehicles()
            loaded_list = traci.simulation.getLoadedIDList()

            vehicle_exists = (veh_id in running_list) or (veh_id in pending_list) or (veh_id in loaded_list)

            if vehicle_exists:
                # Branch A: vehicle already exists (update or respawn).
                try:
                    curr_edge = traci.vehicle.getRoadID(veh_id)

                    # If the vehicle is still pending, getRoadID returns empty; force a respawn.
                    if not curr_edge or curr_edge.startswith(":"):
                        raise traci.exceptions.TraCIException("Vehicle is Pending or in Junction")

                    if curr_edge == dest_edge:
                        return True

                    stage = traci.simulation.findRoute(curr_edge, dest_edge)
                    if stage.edges:
                        traci.vehicle.setRoute(veh_id, stage.edges)
                        return True
                    else:
                        raise traci.exceptions.TraCIException("Route not found")

                except traci.exceptions.TraCIException:
                    try:
                        traci.vehicle.remove(veh_id)
                        traci.route.add(route_id, stage.edges)
                        traci.vehicle.add(
                            veh_id,
                            route_id,
                            typeID="type_EV",
                            departPos="free",
                            departSpeed="0"
                        )
                        if "NPC" in veh_id:
                            traci.vehicle.setColor(veh_id, (255, 215, 0))
                        else:
                            traci.vehicle.setColor(veh_id, (0, 255, 127))
                        return True
                    except Exception as e:
                        print(f"❌ [SUMO] Forced respawn failed: {e}")
                        return False

            else:
                # Branch B: vehicle is not on the network.
                try:
                    stage = traci.simulation.findRoute(start_edge, dest_edge)
                    if not stage.edges:
                        return False

                    traci.route.add(route_id, stage.edges)

                    traci.vehicle.add(
                        veh_id,
                        route_id,
                        typeID="type_EV",
                        departPos="free",
                        departSpeed=0
                    )

                    if "NPC" in veh_id:
                        traci.vehicle.setColor(veh_id, (255, 215, 0))  # gold for NPCs
                    else:
                        traci.vehicle.setColor(veh_id, (0, 255, 127))  # spring green for agents
                        print(f"🚗 [SUMO] Vehicle {veh_id} on road: {start_edge} -> {dest_edge}")

                    return True
                except traci.exceptions.TraCIException as e:
                    print(f"⚠️ [SUMO] Add-vehicle exception: {e}")
                    return False

        except Exception as e:
            print(f"❌ SUMO Add Error: {e}")
            return False

    def remove_vehicle_from_sumo(self, veh_id: str):
        """Remove a vehicle from SUMO (entering a charging station)."""
        try:
            # REMOVE_PARKING marks the vehicle as parking, not finished.
            traci.vehicle.remove(veh_id, reason=traci.constants.REMOVE_PARKING)
        except Exception as e:
            print(f"❌ SUMO Remove Error: {e}")

    def change_target(self, veh_id: str, new_dest_edge: str):
        """Change a vehicle's destination mid-route."""
        try:
            if veh_id not in traci.vehicle.getIDList():
                return False

            curr_edge = traci.vehicle.getRoadID(veh_id)

            # Empty (pending) or internal (junction) edges cannot be rerouted.
            if not curr_edge or curr_edge.startswith(":"):
                return False

            if curr_edge == new_dest_edge:
                return True

            stage = traci.simulation.findRoute(curr_edge, new_dest_edge)

            if stage.edges:
                try:
                    traci.vehicle.setRoute(veh_id, stage.edges)
                    return True
                except traci.exceptions.TraCIException as e:
                    print(f"❌ SUMO setRoute physical rejection: {e}")
                    return False
            else:
                return False

        except Exception as e:
            print(f"❌ change_target exception: {e}")
            return False

    def get_arrived_vehicles(self):
        """Return the IDs of vehicles that arrived this step."""
        return traci.simulation.getArrivedIDList()

    def get_active_vehicles(self):
        return traci.vehicle.getIDList()

    def reload(self):
        """Reload the network, resetting time to 0 and the internal RNG."""
        try:
            traci.load(self.sumo_cmd[1:])
            print("✅ SUMO simulation environment reset to T=0")

            # The vehicle type is lost on reload, so re-register it.
            self._define_vehicle_type()
        except Exception as e:
            print(f"❌ SUMO reload failed: {e}")

    def _define_vehicle_type(self):
        try:
            traci.vehicletype.copy("DEFAULT_VEHTYPE", "type_EV")
            traci.vehicletype.setLength("type_EV", CONFIG.vehicle.length_m)
            traci.vehicletype.setWidth("type_EV", CONFIG.vehicle.width_m)
            traci.vehicletype.setMinGap("type_EV", CONFIG.vehicle.min_gap_m)
            traci.vehicletype.setShapeClass("type_EV", "passenger")
            traci.vehicletype.setColor("type_EV", (0, 255, 0))
        except:
            pass

    def start(self):
        if not self.started:
            traci.start(self.sumo_cmd)
            self.started = True
            self._define_vehicle_type()
