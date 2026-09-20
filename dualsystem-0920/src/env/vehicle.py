import random


class ChargeableVehicle:
    """The vehicle's physical avatar, storing current state (SoC, capacity, power limits)."""

    def __init__(self,
                 vehicle_id: str,
                 soc: float,
                 capacity_kwh: float,
                 role: str = "Unknown",
                 max_charge_power_kw: float = 120.0):
        """
        :param vehicle_id: Agent ID
        :param soc: current state of charge (0.0 - 1.0)
        :param capacity_kwh: total battery capacity
        :param max_charge_power_kw: max charging power the vehicle supports
        """
        self.id = vehicle_id
        self.soc = soc
        self.capacity = capacity_kwh
        self.max_charge_power_kw = max_charge_power_kw
        self.last_charging_power_kw = 0.0  # instantaneous power of the last step
        self.role = role
        # "Waiting", "Charging", "Done", or "Discharging" (V2G, reserved).
        self.status = "Waiting"
        # Cumulative cost of the current charging session.
        self.incurred_cost = 0.0
        # Cumulative energy charged this session (for average-power calculation).
        self.added_energy_kwh = 0.0
        # Micro charging-event fields: entry SoC / charge start time / station id.
        self.entry_soc = None
        self.charge_start_time = None
        self.entry_station_id = None

    def reset_charging_session(self):
        """Reset counters before starting a new charging session."""
        self.incurred_cost = 0.0
        self.added_energy_kwh = 0.0
        self.entry_soc = None
        self.charge_start_time = None
        self.entry_station_id = None

    def __repr__(self):
        return f"<Veh {self.id} SoC:{self.soc:.2f} Status:{self.status}>"
