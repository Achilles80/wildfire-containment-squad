# Wildfire Containment Squad — Implementation Plan

> **For Claude Code:** This file is the full specification for a university course project
> (Foundations of AI, multi-agent case study). Read it fully before writing code.
> Build it **phase by phase** (Section 10). After each phase, run the tests for that phase and
> stop to report what was built before starting the next phase.
> When a value is specified here (grid size, probabilities, radii, weights), use it exactly and
> read it from `config.yaml`, never hard-code it in agent logic.

---

## 0. Course brief and full rubric (verbatim from the course)

**Course:** Foundations of Artificial Intelligence — Case Study

**Brief:** A multi agent implementation with demo and proper testing.

| Review | Criterion | Marks |
|---|---|---|
| Review 1 (design, already done) | PEAS Formulation | 3 |
| Review 1 | Environment & Agent Analysis | 3 |
| Review 1 | Algorithmic Modeling & Search Strategy | 3 |
| Review 1 | Q&A & Presentation Mechanics | 1 |
| **Review 2 (this implementation)** | **Tool/Package Selection & Setup** | **3** |
| **Review 2** | **Multi-Agent Execution & Interaction** | **3** |
| **Review 2** | **Demo Quality & Testing Scenarios** | **3** |
| **Review 2** | **Code Structure & Scalability** | **1** |

The project is software-only (Python simulation). Appendix A holds the Review 1 design that
this implementation must match.

---

## 1. Project context

A multi-agent simulation in which **Scout Drones** and **Firefighters**, coordinated by a
**Coordinator**, contain a wildfire that spreads stochastically across a 2D forest grid.

The project is graded on:

- Tool/package selection and setup
- Multi-agent execution and interaction (agents must genuinely depend on each other)
- Demo quality and testing scenarios (live visual demo + repeatable experiments with metrics)
- Code structure and scalability (modular, config-driven, works with 3 agents or 30)

The design was already presented in Review 1. **The implementation must match the design below**
(same algorithms, same parameters, same agent roles), because evaluators will compare them.

### Grading rubric (Review 2 — this implementation, 10 marks)

Every phase in Section 10 exists to earn one of these. When making trade-offs, protect the
high-mark items first.

| Criterion | Marks | What evaluators look for | Where this plan covers it | Evidence to produce |
|---|---|---|---|---|
| Tool/Package Selection & Setup | 3 | Justified choice of tools; clean, reproducible setup | Section 2 (stack + why), `requirements.txt`, `config.yaml`, README setup steps | README with one-command install and run; pinned requirements; tools justified in README |
| Multi-Agent Execution & Interaction | 3 | Agents genuinely interact: communication, negotiation, cooperation, conflict resolution — not agents running side by side | Sections 7 and 8: shared belief map, auction (ANNOUNCE/BID/AWARD/DONE), reservations, target exclusion, re-auctioning | Visible auction log and zone assignments in the demo panel; logged messages; `test_auction.py`, `test_conflicts.py` |
| Demo Quality & Testing Scenarios | 3 | Clear live demo; systematic tests with edge cases and metrics | Section 9: live app, 6 scenarios × 3 strategies × 30 seeds, charts, unit tests | Smooth `app.py` demo with controls; `results/summary.csv` + charts showing the full system beats both baselines; all pytest tests passing |
| Code Structure & Scalability | 1 | Modular code; works with few or many agents | Sections 3 and 11: package layout, pure algorithm functions, config-driven | Runs unchanged with 0 scouts or 30 firefighters (Scenario 5 proves it); type hints and docstrings |

For reference, Review 1 (design, already presented) was graded on PEAS formulation (3),
environment and agent analysis (3), algorithmic modeling and search strategy (3), and
presentation/Q&A (1). The implementation must stay consistent with that design.

### Core design summary

| Component | What it does | Algorithm |
|---|---|---|
| Scout Drone | Explores, senses fire, writes to the shared belief map | Frontier-based exploration |
| Firefighter | Bids on fire zones, travels, extinguishes cells, cuts firebreaks | Risk-aware A* + utility-based bidding |
| Coordinator | Clusters fire into zones, runs auctions, assigns firefighters | Connected components + sequential single-item auction |
| Shared Belief Map | Team's knowledge of every cell + when it was last seen | Newest timestamp wins |
| Environment | 50×50 grid, stochastic wind-driven fire spread | Probabilistic spread model |

---

## 2. Tech stack

- **Python 3.10+**
- **Mesa >= 3.0** — agent-based modelling framework + browser visualisation (SolaraViz).
  **Important:** Mesa 3 changed its API significantly from Mesa 2 (schedulers such as
  `RandomActivation` are deprecated/removed, agents are created with `Agent(model)` without a
  `unique_id` argument, visualisation uses `SolaraViz`). Check the installed version's API and
  docs before writing Mesa code, and do not use Mesa 2 patterns.
- **NumPy** — grid state arrays
- **heapq** (stdlib) — A* priority queue
- **PyYAML** — config loading
- **Matplotlib** — experiment charts
- **pandas** — experiment results tables
- **pytest** — unit tests

`requirements.txt`:

```
mesa>=3.0
solara
numpy
pyyaml
matplotlib
pandas
pytest
```

If SolaraViz causes persistent problems, fall back to a Matplotlib `FuncAnimation` or Pygame
viewer in `app.py`, keeping the model code unchanged. Ask before switching.

---

## 3. Project structure

```
wildfire_squad/
├── config.yaml                # ALL parameters (Section 4)
├── requirements.txt
├── README.md                  # how to install, run demo, run experiments, run tests
├── model.py                   # WildfireModel: grid, agents, step loop, data collection
├── belief_map.py              # BeliefMap class
├── environment/
│   ├── __init__.py
│   ├── cells.py               # CellState enum, constants
│   ├── forest.py              # forest/map generation incl. preset scenario maps
│   └── fire.py                # fire spread model
├── agents/
│   ├── __init__.py
│   ├── scout.py
│   ├── firefighter.py
│   └── coordinator.py
├── algorithms/
│   ├── __init__.py
│   ├── astar.py               # risk-aware A* + UCS (for comparison), node counters
│   ├── auction.py             # utility function + sequential single-item auction
│   ├── frontier.py            # frontier detection + scout target selection
│   └── zones.py               # connected-component clustering + threat score
├── strategies.py              # "independent", "greedy", "auction" strategy switches
├── scenarios.py               # the 6 test scenarios as config overrides
├── run_experiments.py         # batch runs, CSV output, charts
├── app.py                     # live visual demo
├── results/                   # generated CSVs and PNG charts (gitignored except examples)
└── tests/
    ├── test_astar.py
    ├── test_fire.py
    ├── test_belief_map.py
    ├── test_auction.py
    ├── test_zones.py
    ├── test_conflicts.py
    └── test_model_smoke.py
```

---

## 4. `config.yaml` (default values — use exactly these)

```yaml
seed: 42

grid:
  width: 50
  height: 50
  tree_density: 0.85          # fraction of land cells that are Tree
  sparse_region_fraction: 0.3 # fraction of trees in sparse regions (fuel 0.5)
  lakes: 2                    # random water bodies (ignored if a preset map is used)
  base_station: [2, 2]        # all agents start here

fire:
  ignitions: 1                # number of initial burning cells
  p_base: 0.3
  fuel_dense: 1.0
  fuel_sparse: 0.5
  wind_k: 0.8                 # wind strength factor k
  wind_direction_deg: 0       # 0 = wind blowing toward +x (east); 90 = toward +y
  burn_duration: 4            # steps a cell burns before becoming Burnt
  spread_neighbourhood: 8     # fire spreads to 8 neighbours

agents:
  n_scouts: 3
  n_firefighters: 5
  scout_sense_radius: 3
  firefighter_sense_radius: 1
  water_max: 10
  extinguish_water_cost: 1

scout:
  stale_after: 15             # cells older than this become frontier again
  target_exclusion_radius: 5  # ignore targets within 5 cells of another scout's target

firefighter:
  risk_lambda: 5.0
  replan_every: 10            # also replan immediately if next cell is burning

coordinator:
  cluster_every: 5            # re-cluster zones + run auctions every 5 steps
  large_zone_cells: 20        # zones larger than this are auctioned twice (2 firefighters)
  reauction_after_absent: 15  # re-auction a zone if its firefighter is away refilling > 15 steps

simulation:
  t_max: 300
  strategy: auction           # independent | greedy | auction

score:
  w1: 0.7
  w2: 0.2
  w3: 0.1

experiments:
  runs_per_scenario: 30
```

Load config into a dict/dataclass once; pass it to the model. Scenarios and experiments
override values by deep-merging a dict onto the defaults.

---

## 5. Environment specification

### 5.1 Cell states (`environment/cells.py`)

```python
class CellState(IntEnum):
    EMPTY = 0
    TREE = 1
    BURNING = 2
    BURNT = 3
    FIREBREAK = 4
    WATER = 5
UNKNOWN = -1   # used only in the belief map
```

Ground truth lives in NumPy arrays on the model:

- `state[x, y]` — CellState
- `fuel[x, y]` — 1.0 (dense), 0.5 (sparse), 0 for non-tree
- `burn_timer[x, y]` — steps remaining while BURNING

Use `(x, y)` indexing consistently everywhere, matching Mesa's grid coordinates.

### 5.2 Forest generation (`environment/forest.py`)

- Default: random map from seed — lakes as rough blobs of WATER, remaining land cells TREE with
  probability `tree_density` else EMPTY; mark `sparse_region_fraction` of trees as sparse using a few
  rectangular or blob regions (not per-cell noise, so sparse areas are visible in the demo).
- Keep the base station and a 3-cell radius around it free of fire (ignitions never placed there).
- Ignitions: random TREE cells at least 15 cells from the base station.
- Preset map for Scenario 6 (**river**): a vertical WATER column at `x = 25`, with two
  3-cell-wide crossings at `y = 10..12` and `y = 38..40` (EMPTY cells). Ignition on the far side
  from the base station.

### 5.3 Fire spread (`environment/fire.py`)

Each step, for every TREE cell `n` with at least one BURNING neighbour `b` (8-neighbourhood),
it ignites with probability:

```
P(ignite from b) = p_base × fuel[n] × (1 + k × cos θ)
```

- θ = angle between the wind vector and the direction vector `(n − b)` normalised.
- `cos θ = dot(unit(n − b), unit(wind))`.
- Combine multiple burning neighbours as independent chances:
  `P(ignite) = 1 − Π_b (1 − P(ignite from b))`.
- Clip each probability to [0, 1].
- BURNING cells decrement `burn_timer`; at 0 they become BURNT.
- FIREBREAK, WATER, EMPTY, BURNT never ignite.
- **Use the model's seeded RNG** so runs are reproducible.
- Expose a pure function `ignition_probability(b, n, fuel, wind, cfg)` — reused by the
  firefighter risk cost and the Coordinator's threat score.

Checks: with `k = 0.8`, downwind factor is 1.8× base and upwind is 0.2× base.

---

## 6. Shared belief map (`belief_map.py`)

```python
class BeliefMap:
    state: np.ndarray       # CellState or UNKNOWN (-1), initialised to UNKNOWN
    seen_step: np.ndarray   # step last observed, initialised to -1

    def observe(self, cells: dict[(x, y), CellState], step: int) -> None
        # write only if step >= seen_step for that cell (newest timestamp wins)
    def age(self, step) -> np.ndarray         # step - seen_step (UNKNOWN = very large)
    def stale_mask(self, step, stale_after) -> np.ndarray
    def burning_cells(self) -> list[(x, y)]
    def is_passable_for_firefighter(self, x, y) -> bool
        # not BURNING, not WATER; UNKNOWN treated as passable
```

- Scouts write every step (radius 3). Firefighters also write what they see (radius 1).
- In the **independent** strategy, each firefighter has its own private `BeliefMap` instead of the
  shared one, and scouts' observations are not available to firefighters.

---

## 7. Agents

Each step executes in this **fixed order** (implement in `WildfireModel.step()`):

1. Scouts: sense → write belief map → choose target → move.
2. Coordinator: every `cluster_every` steps, cluster zones and run auctions.
3. Firefighters: act in a fixed order (sorted by id), using cell reservations (Section 8.2).
4. Fire spread + burn timers update.
5. Firefighter safety check: any firefighter standing on a cell that is now BURNING is
   **caught** — remove it from the grid and count it in `A_lost`.
6. Data collection; check termination (no BURNING cells in ground truth, or `t == t_max`).

### 7.1 Scout Drone (`agents/scout.py`) — goal-based, model-based

- Moves one cell in 8 directions per step (or hovers). Flies over any cell; cannot be harmed.
- Sense: all ground-truth cells within Chebyshev radius `scout_sense_radius` → `belief.observe(...)`.
- Target selection (`algorithms/frontier.py`):
  - Frontier cells = UNKNOWN or stale cells (age > `stale_after`) that are adjacent to a
    known, non-stale cell. If no frontier exists, treat all UNKNOWN/stale cells as candidates.
  - Exclude cells within `target_exclusion_radius` of another scout's currently published target.
  - Choose the cell maximising `age / (1 + chebyshev_distance)` (UNKNOWN age = `t_max + 1`).
  - Publish the target (store on the model, e.g. `model.scout_targets[scout_id]`).
  - Re-select when the target is reached or has been observed since selection.
- Move one step straight toward the target (Chebyshev step; no pathfinding needed since scouts fly).

### 7.2 Firefighter (`agents/firefighter.py`) — utility-based, model-based

State: `pos`, `water` (starts at `water_max`), `assigned_zone`, `target_cell`, `path`,
`steps_since_plan`, `refilling_since`.

Moves one cell in 4 directions per step. Cannot enter BURNING or WATER cells (per its belief map
plus a ground-truth check at move time; if the next cell is actually burning, replan instead).

Senses radius 1 → writes to belief map.

**Action priority each step** (exactly one action per step):

1. If `water == 0`: go toward nearest WATER-adjacent passable cell; when adjacent to WATER,
   refill to `water_max` (refilling uses the step).
2. Else if any adjacent (4-neighbour) cell is BURNING: extinguish one of them
   (set to BURNT, costs `extinguish_water_cost`). Prefer the one with the highest downwind
   spread probability.
3. Else if at `target_cell` and it is TREE: cut a firebreak (TREE → FIREBREAK, no water cost),
   then pick the next firebreak target along the zone's downwind edge.
4. Else: take the next step on the A* path to `target_cell`.

**Replanning:** rerun A* when the next path cell is BURNING/blocked, every `replan_every` steps,
or when `target_cell` changes. If A* finds no path: send DONE(abandoned) and become free.

**Firebreak targets:** for the assigned zone, candidate cells = TREE cells adjacent to the zone's
burning cells on the downwind side (projection of `(cell − zone_centroid)` onto the wind vector > 0).
Target = the candidate with the largest downwind projection that is reachable; ties → nearest.
If no candidates, target = nearest cell adjacent to a burning cell in the zone.

**Bidding:** when free and an ANNOUNCE arrives, compute `U(f, z)` (Section 8.1) and send a BID.

**Strategies:**

- `auction` (full system): behaviour above.
- `greedy`: no bidding; Coordinator assigns each zone to the nearest free firefighter.
- `independent`: no Coordinator; firefighter uses its private belief map, targets the nearest
  known BURNING cell, same A* and action priority.

### 7.3 Coordinator (`agents/coordinator.py`) — utility-based

Has no position on the grid (not placed on the grid).

Every `cluster_every` steps:

1. `zones = cluster_zones(belief_map)` (`algorithms/zones.py`): connected components of BURNING
   cells using 8-connectivity. Give zones stable ids across steps by matching new zones to old
   ones by maximum cell overlap.
2. `threat(z) = len(z.cells) × mean downwind ignition probability of the zone's neighbouring
   TREE cells` (use `ignition_probability`).
3. For zones that are new, changed (merged/split), or whose firefighter is gone/absent
   (> `reauction_after_absent` steps refilling), send ANNOUNCE and run the auction (Section 8.1).
4. On DONE(contained): release the firefighter back to the free pool.
   On DONE(abandoned): release it and re-announce the zone next round.
5. Zones with no burning cells are marked contained and their firefighters released.

---

## 8. Algorithms

### 8.1 Auction (`algorithms/auction.py`)

Utility of firefighter `f` for zone `z`:

```
U(f, z) = threat(z) / (1 + pathcost(f, z)) × (water_f / water_max)
```

`pathcost(f, z)` = A* path cost from `f.pos` to `z`'s target cell (inf if unreachable → no bid).

Sequential single-item auction:

```python
for zone in sorted(zones, key=threat, reverse=True):
    slots = 2 if len(zone.cells) > large_zone_cells else 1
    for _ in range(slots):
        bids = {f: U(f, zone) for f in free_firefighters if U(f, zone) > 0}
        if not bids:
            break
        winner = max(bids, key=bids.get)   # tie-break: lower agent id
        assign(winner, zone)
        free_firefighters.remove(winner)
```

Log every auction round (zone id, bids, winner) to a list on the model for the demo side panel
and for debugging.

### 8.2 Messages and conflicts

Implement messages as simple dataclasses (`Observe`, `Target`, `Announce`, `Bid`, `Award`,
`Done`) passed through a message queue on the model (instant delivery, processed in the
step order above). This keeps interaction explicit and loggable.

Conflict rules:

| Conflict | Rule |
|---|---|
| Two firefighters want the same zone | Auction; higher utility wins, loser bids on next zone |
| Two agents want the same cell in one step | Firefighters move in id order; `model.reserved` set per step; a reserved cell counts as blocked for later agents, who wait (or replan if blocked 2 steps in a row) |
| Two scouts target the same area | Target exclusion radius (Section 7.1) |
| Conflicting observations of one cell | Newest timestamp wins (BeliefMap) |
| Path blocked by new fire | Replan; if no path → DONE(abandoned) |
| Out of water mid-task | Refill, keep zone; Coordinator re-auctions if absent > 15 steps |

### 8.3 Risk-aware A* (`algorithms/astar.py`)

- Graph: 4-connected grid; passable = `belief.is_passable_for_firefighter` and not reserved
  (reservation applies only to the immediate next step, not the full path).
- Step cost entering cell `n`: `c(n) = 1 + λ × risk(n)`,
  `risk(n) = max over BURNING neighbours b of ignition_probability(b, n, ...)` (0 if none;
  use the belief map's burning cells).
- Heuristic: Manhattan distance to goal (admissible and consistent since every step costs ≥ 1).
- Goal: reach `target_cell` (or a cell adjacent to it if the target itself is impassable).
- Use `heapq` with a tie-breaking counter.
- Return `(path, cost, nodes_expanded)`.
- Also implement `ucs(...)` = same function with `h = 0`, used only for the A* vs UCS comparison
  metric. Record `nodes_expanded` for both in experiments (run UCS on the same queries in a
  measurement-only mode so it does not affect behaviour).

---

## 9. Metrics, visualisation, experiments

### 9.1 Metrics (Mesa DataCollector, per step + final)

- `trees_initial`, `trees_saved` (TREE cells at the end; FIREBREAK cells do not count as saved)
- `pct_forest_saved = trees_saved / trees_initial × 100` (primary metric)
- `t_contain` (step when no BURNING cells remain; `t_max` if never)
- `agents_lost`
- `mean_belief_staleness` (mean age of known cells)
- `astar_nodes_expanded`, `ucs_nodes_expanded` (totals)
- `score = w1 × trees_saved/trees_initial − w2 × t_contain/t_max − w3 × agents_lost/n_firefighters`

### 9.2 Live demo (`app.py`)

- Grid view colours: Tree dense = dark green, Tree sparse = light green, Burning = red/orange,
  Burnt = dark grey, Firebreak = brown, Water = blue, Empty = beige.
- Agents: scouts = small white/cyan circles, firefighters = yellow squares with id labels.
- Toggle to show the belief map instead of ground truth (unknown = black, stale = faded).
- Side panel: live `% forest saved`, step count, current zone assignments, last auction log.
- Controls (sliders/dropdowns): strategy, number of scouts/firefighters, wind direction,
  wind strength k, ignitions, scenario preset, seed.
- Wind direction arrow drawn on the grid.

### 9.3 Scenarios (`scenarios.py`)

| # | Name | Overrides |
|---|---|---|
| 1 | calm_single | `wind_k: 0.0`, `ignitions: 1` |
| 2 | windy_single | `wind_k: 0.8`, `ignitions: 1` |
| 3 | multi_ignition | `wind_k: 0.4`, `ignitions: 3`, ignitions ≥ 15 cells apart |
| 4 | no_scouts | `n_scouts: 0` |
| 5 | scaling | `n_firefighters` ∈ {2, 5, 10} |
| 6 | river | preset river map |

### 9.4 Experiments (`run_experiments.py`)

- For each scenario × strategy (`independent`, `greedy`, `auction`) × `runs_per_scenario` seeds
  (seeds 0..29), run headless (no visualisation) and record final metrics.
- Same seed across strategies so they face the identical fire (paired comparison).
- Output: `results/results.csv` (one row per run) and `results/summary.csv`
  (mean ± std per scenario × strategy).
- Charts (`results/*.png`): grouped bar chart of mean `% forest saved` per scenario by strategy
  with std error bars; scaling line chart for Scenario 5; A* vs UCS nodes-expanded bar chart.
- CLI: `python run_experiments.py --scenarios all --runs 30`, plus `--quick` (3 runs) for testing.
- Should finish in a few minutes; if too slow, profile and optimise fire spread with NumPy
  vectorisation before reducing runs.

---

## 10. Build phases (do these in order)

Each phase ends with passing tests and a short report. Do not start the next phase until the
current one works.

**Phase 1 — Environment.**
`config.yaml`, config loader, `cells.py`, `forest.py`, `fire.py`, a bare `WildfireModel` with fire
spread only (no agents).
Tests: `test_fire.py` — downwind/upwind factors (1.8× / 0.2× with k=0.8), non-tree cells never
ignite, burning becomes burnt after 4 steps, same seed ⇒ identical fire.

**Phase 2 — Belief map + A*.**
`belief_map.py`, `astar.py` (A* + UCS).
Tests: `test_belief_map.py` (newest wins, older observation ignored, staleness);
`test_astar.py` (A* cost equals UCS cost on random grids, A* expands ≤ UCS nodes, avoids burning
cells, risk cost makes path detour around fire when λ is high, returns no path when blocked).

**Phase 3 — Firefighters (independent strategy).**
Firefighter agent with private belief map, action priority, water/refill, extinguish, caught-in-fire
check, reservations.
Tests: `test_conflicts.py` (two firefighters never occupy the same cell; reserved cell handling);
`test_model_smoke.py` (independent strategy runs 300 steps without error).

**Phase 4 — Scouts + shared belief map.**
Scout agent, frontier selection, target exclusion; switch firefighters to shared map in
`greedy`/`auction` modes.
Tests: scouts explore > 80% of the grid within `t_max` on an empty-fire map; two scouts never
hold targets within 5 cells of each other.

**Phase 5 — Coordinator + zones + auction.**
`zones.py`, `auction.py`, Coordinator, messages, greedy strategy, full auction strategy,
firebreak targeting, DONE handling, re-auction on absence.
Tests: `test_zones.py` (component clustering, stable ids after growth);
`test_auction.py` (highest utility wins, tie-break by id, large zone gets 2 firefighters,
unreachable firefighter doesn't bid).

**Phase 6 — Metrics + experiments.**
DataCollector, score, `scenarios.py`, `run_experiments.py`, charts.
Check: `--quick` run completes; full run produces CSVs and charts.

**Phase 7 — Visual demo.**
`app.py` with all controls and panels from Section 9.2.

**Phase 8 — Polish.**
README (setup, demo, experiments, tests, project structure, design summary), docstrings on all
public functions, type hints, remove dead code, `ruff`/`black` formatting if available.

---

## 11. Coding standards

- Type hints and docstrings on all public classes and functions.
- No magic numbers in logic — everything from config.
- All randomness through the model's seeded RNG (`model.random` / a `numpy.random.Generator`
  created from the seed). Never use the global `random` or `np.random` functions directly.
- Keep algorithms (`algorithms/`) as pure functions independent of Mesa where possible, so they are
  easy to unit test.
- Agents must work for any count (0 scouts, 1–30 firefighters) without code changes.
- Log key events (auctions, awards, abandons, agents lost) to a list on the model, not print().

## 12. Out of scope

3D terrain, real GIS data, continuous space, communication delays or failures, reinforcement
learning. Optional extension only if everything else is done: firefighters fall back to greedy
nearest-zone behaviour if no AWARD arrives within 10 steps (Coordinator failure robustness).

---

## Appendix A — Review 1 design reference (must stay consistent)

This is the design presented in Review 1. The code, demo labels and README should use the same
names, roles, formulas and parameters.

### A.1 Problem statement

A wildfire spreads faster than any single responder can act, visibility is limited, and wind
changes the spread direction. Containment needs three things at once: finding the fire, deciding
who fights which part, and moving responders there efficiently. Objectives:

1. Model the forest as a dynamic, partially observable grid with probabilistic, wind- and
   fuel-driven fire spread.
2. Two cooperating agent types with distinct roles (Scouts = perception, Firefighters = action),
   plus a lightweight Coordinator for task allocation.
3. Informed search (A*) for navigation and a market-based auction for task allocation.
4. Show through controlled scenarios that coordinated agents save measurably more forest than
   independent agents.

### A.2 PEAS

**System performance measure (shared by all agents):**
`Score = w1 × T_saved/T_initial − w2 × t_contain/t_max − w3 × A_lost/A_total`
with w1 = 0.7, w2 = 0.2, w3 = 0.1. Contained = no Burning cell remains.

| Agent | Performance | Environment | Actuators | Sensors |
|---|---|---|---|---|
| Scout Drone | Burning cells detected within 5 steps of ignition; grid explored; low belief-map staleness | Forest grid from above; fire, smoke, wind; other drones; immune to fire | Fly 1 cell in 8 directions or hover; broadcast observations | Camera radius 3 (cell states); wind sensor; own position |
| Firefighter | Cells extinguished; firebreaks cut ahead of the fire; zones contained; not caught in fire | Forest grid on the ground; tree/burning/burnt/water/firebreak cells; other firefighters; limited water | Move 1 cell in 4 directions; extinguish adjacent burning cell; cut firebreak on adjacent tree; refill at water; send bids | Local view radius 1; shared belief map; own position and water; task awards |
| Coordinator | Low total travel cost of assignments; zones covered; few idle firefighters | Shared belief map and the firefighter team; no body on the grid | Announce zones; award zones to winning bidders; re-open auctions when zones change | Belief map; bids; task-completion messages |

### A.3 Environment properties

| Property | Classification | Consequence in code |
|---|---|---|
| Observability | Partially observable | Shared belief map with timestamps |
| Determinism | Stochastic | Plans recomputed as fire moves |
| Episodic/sequential | Sequential | Firebreaks placed ahead of future spread |
| Static/dynamic | Dynamic | Fast A*, periodic and event-driven replanning |
| Discrete/continuous | Discrete | Grid search is well defined |
| Agents | Multi-agent, cooperative | Shared map + auctions + conflict rules |
| Known/unknown | Known rules, unknown state | Scouts explore; spread model used for risk and threat |

### A.4 Agent architectures

| Agent | Architecture | Internal state |
|---|---|---|
| Scout Drone | Model-based, goal-based | Belief map with last-seen time per cell |
| Firefighter | Model-based, utility-based | Position, water, assigned zone, current A* path |
| Coordinator | Utility-based | Zones, open auctions, assignments |

### A.5 Algorithms and justification

- **Navigation:** risk-aware A*, step cost `1 + λ × risk(n)`, λ = 5, Manhattan heuristic
  (admissible and consistent). Chosen over BFS (ignores costs), DFS (not optimal), greedy
  best-first (not optimal, walks into fire); UCS kept as the comparison baseline.
- **Task allocation:** sequential single-item auction with
  `U(f, z) = threat(z) / (1 + pathcost(f, z)) × water_f / water_max`. Chosen over independent
  agents and greedy nearest (both kept as baselines) and the Hungarian algorithm (O(n³) central
  recomputation every time zones change).
- **Exploration:** frontier-based, value = age / (1 + distance), 5-cell target exclusion.
- **Belief fusion:** newest timestamp wins; cells older than 15 steps become stale.

### A.6 Viva questions the implementation should make easy to demonstrate

- Why multi-agent? → demo toggle: `independent` vs `auction` strategy on the same seed.
- Why A* over BFS/UCS? → nodes-expanded chart (A* vs UCS).
- Is the heuristic admissible? → `test_astar.py` shows A* cost equals UCS cost.
- What if fire blocks the path? → replanning visible in the demo; abandon + re-auction logged.
- Partial observability? → belief-map view toggle in the demo.
- Does coordination help? → `results/summary.csv` and the grouped bar chart.
- Limitations: 2D grid, simplified spread, instant communication, no terrain slope.
