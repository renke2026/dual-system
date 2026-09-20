# Dual-System EV Charging Simulation

A multi-agent simulation of electric-vehicle charging behavior that couples a **System-1 rule engine** (fast, deterministic reflexes) with a **System-2 LLM planner** (deliberate reasoning). Agents drive on a SUMO road network, charge at stations attached to a distribution grid, and evolve their personality and schedule through daily reflection. The framework is built for behavioral experiments (e.g. congestion, price-surge, and memory effects) via configurable baselines and ablations.

## Architecture

Each agent is a finite-state machine (`IDLE → DRIVING → DRIVING_TO_CHARGE → CHARGING`) with two decision layers:

- **System-1** — threshold-based reflexes, curfew, snooze alarms, and a validity intercept. It decides *whether* the agent should think and enforces *safe* actions without any LLM call.
- **System-2** — an LLM planner that reasons about trade-offs (time vs. cost vs. range anxiety) and returns a structured decision.

System-2 also *configures* System-1 through a cognitive **mental state** (`ALERT` / `COMMITTED` / `LIGHT_SLEEP` / `DEEP_FOCUS`), which modulates how often the agent re-evaluates. Memories are stored in a vector database and retrieved to inform future decisions.

## 1. Prompt Templates

All prompts are in `src/agent/planner.py` (decisions, reflection), `src/utils/llm_client.py` (archetype generation), and `src/common.py` (persona).

- **Persona (system prompt).** Built from a 2×2×2 trait design — role (`COMMUTER`/`GIG_WORKER`), price sensitivity (`SENSITIVE`/`INSENSITIVE`), and range anxiety (`ANXIOUS`/`CALM`). The traits become behavioral guidelines the agent must follow "in character".
- **Decision context.** Each decision prompt is assembled from modular blocks: instinctive perception, learned strategic rules, relevant past episodes, self state (time/SoC/traffic), schedule preview, market intuition (grid status, average prices), a nearby-station table, breaking news, and a self-correction protocol.
- **State-specific task instructions.** Each FSM state defines an action space and a JSON output contract (e.g. IDLE: `START_TRIP` / `FIND_CHARGER` / `STAY_AND_WAIT`; DRIVING: `KEEP_ROUTE` / `REROUTE_TO_CHARGER`), validated by a Pydantic schema.
- **Daily reflection (three steps).** An objective *Accountant* computes daily statistics (no LLM); a *Psychologist* decides personality evolution and a strategic intent; a *Scheduler* converts that intent into concrete schedule adjustments.

## 2. System-1 Rule Set

The deterministic layer in `src/agent/core.py` implements the "fast" system:

- **Dynamic thresholds.** Three SoC thresholds (survival / comfort / panic) are recomputed per trip from the remaining distance, personality (anxiety, price sensitivity), and profession (gig workers carry a higher panic threshold).
- **Sensory interpretation.** Raw observations are mapped to qualitative tags — battery (`DEADLY`/`PANIC`/`CONCERNED`/`COMFORTABLE`), price (`BARGAIN`/`FAIR`/`ELEVATED`/`OVERPRICED`), schedule (`ON TIME`/`SLIGHTLY LATE`/`SEVERELY LATE`) — that feed both systems.
- **Charging-need funnel.** A three-tier filter (`NONE` / `DECISION_NEEDED` / `CRITICAL`) decides whether charging is worth considering, accounting for destination chargers and profession.
- **System-2 gate.** A strict priority order decides whether to wake the LLM: survival-critical needs and breaking news always wake the agent; committed plans, deep focus, and a nighttime curfew suppress thinking; snooze alarms schedule the next wake-up.
- **News reflex.** Critical broadcasts force the agent into an alert state regardless of its current mental state.
- **Safety intercept.** Invalid or unsafe LLM outputs (e.g. charging at a nonexistent station, or above a full battery) are corrected to safe actions.
- **MNL baseline.** A fully non-LLM fallback where the rules choose the action and a multinomial logit model chooses the station.

## 3. Network Topology

Two road-network cases are registered in `config.py` and selected with `--case` (default `12nodes`).

| | `12nodes` | `nanjing` (37 nodes) |
|---|---|---|
| SUMO junctions | 115 | 227 |
| Traffic zones (TAZ) | 12 | 37 |
| Grid buses | 3 | 33 |
| Fast chargers (FCS) | 8 | 10 |
| Slow chargers (SCS) | 40 | 130 |

Each case provides a SUMO road network, traffic-analysis-zone definitions with a `Home`/`Work`/`Relax`/`Other` classification, a radial distribution grid (solved with a LinDistFlow power-flow model), and fast/slow charging stations attached to grid buses. Grid voltage drives dynamic electricity prices: below safe voltage thresholds prices rise and eventually surge, with high-priority news broadcast to agents.

## 4. Post-processing Scripts

`generate.py` prepares the data (pricing schedules, agent profiles, NPCs) by invoking three scripts under `scripts/`: `overwrite_prices.py`, `generate_agents_file.py`, and `generate_npcs.py`. `run.py` then runs the simulation (`scripts/main.py`).

Every run writes a timestamped directory under `results/` with structured CSV logs covering grid and station statistics, agent trajectories and episode-level decisions, daily personality evolution, charging sessions and events, network traffic, and LLM usage (tokens, latency, raw-vs-validated decisions).

## Quick Start

```powershell
python generate.py                        # (optional) prepare data
python run.py --days 1 --baseline mnl     # smoke test, no LLM
python run.py --days 1                    # full dual-system run
```

Requirements: SUMO (with `SUMO_HOME` set) and the Python dependencies imported in `config.py` (traci/sumolib, cvxpy, chromadb, sentence-transformers, google-generativeai, openai, pydantic).

## API Keys

No API keys are stored in the code. Provide your own keys at run time via environment variables:

- `DEEPSEEK_API_KEY` — required for the System-2 planner (`run.py`, full dual-system run).
- `GOOGLE_API_KEY` — required only for archetype generation (`generate.py`).

```powershell
# PowerShell
$env:DEEPSEEK_API_KEY = "your-key"
$env:GOOGLE_API_KEY   = "your-key"
```

If a key is missing, the program stops with a clear message naming the variable to set.
