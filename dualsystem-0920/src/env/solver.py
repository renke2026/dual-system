import cvxpy as cp
import numpy as np
import math
from typing import Dict, Optional
from src.env.grid import GridTopology
from config import CONFIG


class PowerFlowSolver:
    def __init__(self, grid: GridTopology, base_mva: float):
        """
        LinDistFlow solver (optimized).
        :param grid: grid topology
        :param base_mva: system base power (MVA)
        """
        self.grid = grid
        self.base_mva = base_mva

        self.bus_ids = list(grid.buses.keys())
        self.bus_map = {bid: i for i, bid in enumerate(self.bus_ids)}
        self.num_buses = len(self.bus_ids)
        self.num_lines = len(grid.lines)

        # Root (slack) bus; default to the first bus.
        self.root_idx = 0

        # Base impedance Z_base = (U_base_kV)^2 / S_base_MVA, from the root bus voltage.
        root_bus = grid.buses[self.bus_ids[self.root_idx]]
        self.base_kv = root_bus.base_kv
        self.z_base = (self.base_kv ** 2) / self.base_mva

        print(f"🧮 [Solver] Init: Base={self.base_mva}MVA, {self.base_kv}kV -> Z_base={self.z_base:.4f}Ω")

        self.line_from = []
        self.line_to = []
        self.line_r_pu = []
        self.line_x_pu = []

        for line in grid.lines:
            if line.from_bus in self.bus_map and line.to_bus in self.bus_map:
                f_idx = self.bus_map[line.from_bus]
                t_idx = self.bus_map[line.to_bus]

                self.line_from.append(f_idx)
                self.line_to.append(t_idx)

                # Physical value -> per-unit.
                self.line_r_pu.append(line.r_ohm / self.z_base)
                self.line_x_pu.append(line.x_ohm / self.z_base)
            else:
                print(f"⚠️ [Solver] Ignoring invalid line: {line.id}")

        self._build_model()

    def _build_model(self):
        """Build the LinDistFlow convex optimization model (parameterized to avoid recompilation)."""
        # U = V^2.
        self.U = cp.Variable(self.num_buses, name="U")
        # P_ij, Q_ij (line power flows).
        self.P_lines = cp.Variable(self.num_lines, name="P_lines")
        self.Q_lines = cp.Variable(self.num_lines, name="Q_lines")

        self.P_load = cp.Parameter(self.num_buses, name="P_load")
        self.Q_load = cp.Parameter(self.num_buses, name="Q_load")

        constraints = []

        # Slack bus voltage fixed at U_root = 1.0.
        constraints.append(self.U[self.root_idx] == 1.0)

        # Line voltage drop: U_j = U_i - 2(RP + XQ).
        for k in range(self.num_lines):
            i = self.line_from[k]
            j = self.line_to[k]
            r = self.line_r_pu[k]
            x = self.line_x_pu[k]

            constraints.append(
                self.U[j] == self.U[i] - 2 * (r * self.P_lines[k] + x * self.Q_lines[k])
            )

        # Nodal power balance (KCL): inflow - outflow = load, for all non-root buses.
        node_in_lines = [[] for _ in range(self.num_buses)]
        node_out_lines = [[] for _ in range(self.num_buses)]

        for k in range(self.num_lines):
            f_idx = self.line_from[k]
            t_idx = self.line_to[k]
            node_out_lines[f_idx].append(k)
            node_in_lines[t_idx].append(k)

        for j in range(self.num_buses):
            if j == self.root_idx:
                continue  # slack bus: no power-balance constraint

            p_bal = 0
            q_bal = 0

            for k in node_in_lines[j]:
                p_bal += self.P_lines[k]
                q_bal += self.Q_lines[k]

            for k in node_out_lines[j]:
                p_bal -= self.P_lines[k]
                q_bal -= self.Q_lines[k]

            constraints.append(p_bal == self.P_load[j])
            constraints.append(q_bal == self.Q_load[j])

        # Keep voltage non-negative (V >= 0.5 p.u.).
        constraints.append(self.U >= CONFIG.solver.u_min)

        # Objective: constant 0 (feasibility check) for numerical stability.
        prob = cp.Problem(cp.Minimize(0), constraints)

        self.prob = prob

    def solve(self, load_p_mw: Dict[str, float], load_q_mvar: Dict[str, float] = None) -> Dict[str, float]:
        """
        Run the power flow.
        :param load_p_mw: active load {bus_id: MW}
        :param load_q_mvar: reactive load {bus_id: MVar} (optional)
        :return: per-unit voltages {bus_id: v_pu}
        """
        p_vec = np.zeros(self.num_buses)
        q_vec = np.zeros(self.num_buses)

        tan_phi = CONFIG.solver.tan_phi  # tan(arccos(0.95)) ~= 0.328

        for bus_id, p_mw in load_p_mw.items():
            if bus_id in self.bus_map:
                idx = self.bus_map[bus_id]
                p_pu = p_mw / self.base_mva
                p_vec[idx] = p_pu

                if load_q_mvar and bus_id in load_q_mvar:
                    q_vec[idx] = load_q_mvar[bus_id] / self.base_mva
                else:
                    # No Q given: assume PF=0.95 (inductive load).
                    q_vec[idx] = p_pu * tan_phi

        self.P_load.value = p_vec
        self.Q_load.value = q_vec

        try:
            self.prob.solve(solver=getattr(cp, CONFIG.solver.solver_name), verbose=False)
        except Exception as e:
            print(f"❌ [Solver] Math Error: {e}")
            return self._fallback_result()

        return self._check_convergence()

    def _check_convergence(self) -> Dict[str, float]:
        """Validate solver status and physical plausibility."""
        if self.prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return self._fallback_result()

        u_val = self.U.value
        if u_val is None:
            return self._fallback_result()

        v_pu = np.sqrt(np.maximum(u_val, 0))  # guard against tiny negatives

        min_v = np.min(v_pu)
        max_v = np.max(v_pu)

        # Voltage outside the collapse band means the grid has collapsed.
        if min_v < CONFIG.solver.collapse_v_min or max_v > CONFIG.solver.collapse_v_max:
            return self._fallback_result()

        result = {}
        for i, idx in enumerate(self.bus_ids):
            result[idx] = float(v_pu[i])

        return result

    def _fallback_result(self) -> Dict[str, float]:
        """Return the fallback voltage for every bus."""
        return {bid: CONFIG.solver.fallback_voltage for bid in self.bus_ids}
