# Wildfire Containment Squad

A multi-agent simulation for the Foundations of AI case study. **Scout Drones** explore a
partially observable forest and share what they see. A **Coordinator** clusters the known fire
into zones and auctions them. **Firefighters** bid, travel with risk-aware A\*, put out burning
cells and cut firebreaks while the wildfire spreads stochastically with the wind.

The implementation follows the Review 1 design: the same agents, algorithms, formulas and
parameters. Where the implementation had to decide something Review 1 left open, the decision
is listed in [Implementation decisions](#implementation-decisions).

---

## 1. Setup (one command)

Requires Python 3.10+ (tested on 3.13, Windows 11).

```bash
cd wildfire_squad
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

All versions in `requirements.txt` are pinned to what the project was tested with.

| Tool | Used for | Why this one |
|---|---|---|
| **Mesa 3.5** | `Model`, `Agent`, `MultiGrid`, `DataCollector`, SolaraViz run controls | Purpose-built agent-based modelling framework; agent registry, seeded RNG and data collection come for free |
| **Solara** | Browser UI for the live demo | Mesa's own visualisation stack; reactive widgets for play/step/reset and parameters |
| **NumPy** | Grid state, vectorised fire spread and risk maps | Whole-grid spread computed with 8 array shifts instead of Python loops; a full 300-step run takes ~2.5 s |
| **heapq** (stdlib) | A\* / UCS priority queue | No extra dependency, O(log n) push/pop |
| **PyYAML** | `config.yaml` | Every parameter lives in one human-readable file |
| **Matplotlib / pandas** | Demo rendering, experiment charts and tables | Standard, reproducible PNG + CSV output |
| **pytest** | 94 unit, integration and regression tests | Standard; parametrised tests over seeds and strategies |

## 2. Run the live demo

```bash
solara run app.py          # opens http://localhost:8765
```

* **Left:** Reset / Play / Step, speed, and every parameter: strategy, scenario preset, number
  of scouts and firefighters, wind direction and strength, ignitions, fire time scale, seed.
  Press **Reset** to apply parameter changes.
* **Centre:** the forest. Dark green = dense, light green = sparse, orange-red = burning,
  grey = burnt, brown = firebreak, blue = water, beige = empty. Cyan circles are scouts,
  yellow squares are firefighters (with id), white × marks zone targets, and the arrow shows
  the wind. Tick **Show belief map** to see what the team knows: black = unknown,
  faded = stale (not seen for more than 15 steps).
* **Right:** live % forest saved, zones with their threat and assigned firefighters, each
  firefighter's position/water/activity, the **last auction rounds with every bid and the
  winner**, message counts (OBSERVE, TARGET, ANNOUNCE, BID, AWARD, DONE, REVOKE) and recent
  events (awards, abandons, agents lost, yields).
* **Below the map:** % forest saved and burning cells over time.

Suggested viva demo: seed 0, preset `windy_single`. Run `independent` and then `auction` on the
same seed. The auction log fills once scouts find the fire, and the firefighters spread over
the zones instead of wandering until they stumble on it.

## 3. Run the experiments

```bash
python run_experiments.py --quick                       # 3 seeds per cell, ~30 s
python run_experiments.py --scenarios all --runs 30     # full study: 720 runs, ~7 min on 4 workers
python run_experiments.py --replot                      # rebuild tables/charts from results.csv
```

The runner writes to `results/`:

| File | Content |
|---|---|
| `results.csv` | One row per run (scenario, strategy, seed, all metrics) |
| `summary.csv` | Mean and std of every metric per scenario × strategy |
| `paired_gains.csv` | Auction minus each baseline on the **same seed**, with ≈95% CI and win counts |
| `forest_saved_by_scenario.png` | Grouped bars: mean % forest saved per scenario by strategy |
| `paired_gains.png` | Paired gain of the auction over each baseline, per scenario |
| `scaling.png` | Scenario 5: % forest saved vs 2 / 5 / 10 firefighters |
| `astar_vs_ucs.png` | Nodes expanded per search, A\* vs UCS on identical queries |

Every strategy sees the same map and the same fire random stream for a given seed (separate
RNG streams for map, fire and agents), so the comparisons are **paired**.

## 4. Run the tests

```bash
python -m pytest            # 94 tests, ~45 s
```

| File | What it proves |
|---|---|
| `test_fire.py` | Downwind 1.8× / upwind 0.2× factors (k = 0.8), fuel scaling, clipping, independent-neighbour combination, non-Tree cells never ignite, Burning → Burnt after 4 updates, same seed ⇒ identical fire, ignition placement, river layout |
| `test_belief_map.py` | Newest observation wins, older observation ignored, age and staleness, passability |
| `test_astar.py` | A\* cost equals UCS cost on 25 random weighted grids (admissible heuristic), A\* never expands more nodes, avoids burning cells, detours around fire when λ is high, no path when blocked, reservations apply to the first move only |
| `test_zones.py` | 8-connected clustering, stable zone ids as zones grow, merge/split/removal handling, threat grows with size, downwind firebreak targets |
| `test_auction.py` | Utility formula, highest utility wins, tie-break by lower id, zones auctioned by threat and the loser bids on the next zone, large zone gets 2 firefighters, unreachable firefighter does not bid, greedy baseline ignores threat, end-to-end ANNOUNCE→BID→AWARD over the message bus |
| `test_conflicts.py` | Two firefighters never share a cell (all strategies, 10 firefighters), reserved-cell waiting in a corridor, scout targets always > 5 cells apart, replanning around new fire, no deadlock at the base (regression) |
| `test_scouts.py` | Frontier definition, stale cells become frontier, target exclusion, scouts explore > 80% of the grid, shared vs private maps, exclusion fallback (regression) |
| `test_model_smoke.py` | Every strategy runs 300 steps, every scenario runs, 0–30 agents without code changes, score formula matches Review 1, deterministic per seed, caught firefighters are removed |

## 5. Results (30 seeds per cell, 720 runs)

Mean % forest saved (± std across seeds):

| Scenario | Independent | Greedy nearest | **Auction (full system)** |
|---|---|---|---|
| S1 calm_single | 15.3 ± 14.6 | 19.1 ± 20.9 | **19.1 ± 22.0** |
| S2 windy_single | 43.3 ± 24.4 | 46.7 ± 28.2 | **48.1 ± 28.0** |
| S3 multi_ignition | **5.1 ± 4.6** | 2.8 ± 3.3 | 2.6 ± 3.0 |
| S4 no_scouts | 43.3 ± 24.4 | 46.9 ± 25.4 | **48.4 ± 25.3** |
| S5 scaling_2 | 41.1 ± 25.7 | **43.1 ± 26.6** | 42.7 ± 26.3 |
| S5 scaling_5 | 43.3 ± 24.4 | 46.7 ± 28.2 | **48.1 ± 28.0** |
| S5 scaling_10 | 44.1 ± 23.7 | 55.0 ± 25.5 | **55.7 ± 27.1** |
| S6 river | 65.1 ± 13.0 | **72.3 ± 14.4** | 71.4 ± 15.0 |

The large standard deviations are mostly the fire itself: some seeds' fires burn out early,
others run across the whole map. The paired comparison removes that variance.

Paired gain of the auction (percentage points, same seeds, ≈95% CI):

| Scenario | vs independent | better on | vs greedy |
|---|---|---|---|
| S1 calm_single | +3.8 ± 5.7 | 15/30 | +0.0 ± 1.6 |
| S2 windy_single | **+4.8 ± 4.1** | 20/30 | +1.5 ± 1.7 |
| S3 multi_ignition | **−2.6 ± 1.4** | 6/30 | −0.3 ± 0.8 |
| S4 no_scouts | **+5.0 ± 2.6** | 22/30 | +1.5 ± 1.6 |
| S5 scaling_2 | +1.6 ± 2.1 | 16/30 | −0.4 ± 2.0 |
| S5 scaling_5 | **+4.8 ± 4.1** | 20/30 | +1.5 ± 1.7 |
| S5 scaling_10 | **+11.6 ± 5.8** | 21/30 | +0.8 ± 2.7 |
| S6 river | **+6.3 ± 2.2** | 28/30 | −0.9 ± 1.9 |

**What the numbers say (for the report and viva):**

* **Coordination beats independent agents** wherever the CI excludes zero: strong wind, no
  scouts, 5 and 10 firefighters, and the river. The gain **grows with team size** (+1.6 → +4.8
  → +11.6 points for 2 → 5 → 10 firefighters). Independent firefighters crowd the nearest fire
  they happen to find, so extra firefighters add almost nothing (41 → 43 → 44%). Coordinated
  ones spread over zones (43 → 48 → 56%).
* **The river (S6) shows the value of shared perception plus A\*:** independent firefighters
  never reach the fire across the river (0 cells extinguished); coordinated ones path through
  the crossings. The auction wins on 28 of 30 seeds.
* **Without scouts (S4)** the coordinated team still wins, because firefighters that share one
  map and publish exploration targets search the forest far faster than five private searches.
* **Auction vs greedy is a statistical tie.** With one fire at a time there is rarely a real
  choice between zones, and the support rounds (see below) put every firefighter to work under
  both methods. The auction's advantage in theory (threat ordering, water, true path cost)
  needs several competing zones and spare capacity, which this map size rarely provides.
* **Three simultaneous ignitions (S3) overwhelm five firefighters under every strategy**
  (≤ 5% saved). The independent baseline is slightly better there: coordinated firefighters
  spend time travelling between zones that cannot be held anyway.
* **Firefighter safety:** coordinated firefighters engage the fire far more (≈60 cells
  extinguished vs ≈20) and are caught more often (≈0.9 vs 0.2 per run). That is the price of
  engaging, and the score's w3 term accounts for it.
* **A\* vs UCS:** on identical queries A\* expands **72% fewer nodes** overall (67–84% per
  scenario) and always returns the same optimal cost (`test_astar.py`).

## 6. Project structure

```
wildfire_squad/
├── config.yaml             # every parameter (Review 1 values + marked implementation details)
├── settings.py             # load config.yaml, deep-merge scenario/experiment overrides
├── model.py                # WildfireModel: grid, fixed step order, planning cache, metrics
├── belief_map.py           # BeliefMap: state + last-seen step per cell, newest-wins fusion
├── messages.py             # OBSERVE/TARGET/ANNOUNCE/BID/AWARD/DONE/REVOKE + message bus
├── strategies.py           # independent | greedy | auction switches
├── scenarios.py            # the six Review 1 scenarios as config overrides
├── environment/
│   ├── cells.py            # CellState enum, neighbourhoods, distances
│   ├── forest.py           # random forest (lakes, sparse patches), river preset, ignitions
│   └── fire.py             # ignition_probability, vectorised spread, risk map
├── agents/
│   ├── scout.py            # Scout Drone (goal-based)
│   ├── firefighter.py      # Firefighter (utility-based)
│   └── coordinator.py      # Coordinator (utility-based)
├── algorithms/             # pure functions, no Mesa dependency, unit-tested directly
│   ├── astar.py            # risk-aware A* + UCS, node counters
│   ├── auction.py          # utility, sequential single-item auction, greedy baseline
│   ├── frontier.py         # frontier detection + exploration target selection
│   └── zones.py            # connected components, stable ids, threat, firebreak targets
├── run_experiments.py      # parallel batch runs → CSVs + charts
├── app.py                  # live demo
├── results/                # generated CSVs and charts
└── tests/                  # 94 pytest tests
```

**Scalability.** No count is hard-coded. The same code runs 0 scouts, 1–30 firefighters and
any number of ignitions (`test_scales_to_any_agent_count`). The algorithms live in pure
functions that take arrays and callbacks, so they are tested without Mesa. Every parameter is
read from `config.yaml` and every random draw goes through the model's seeded generators.

## 7. How the design maps to code

| Review 1 element | Where |
|---|---|
| Score = w1·T_saved/T_initial − w2·t_contain/t_max − w3·A_lost/A_total (0.7, 0.2, 0.1) | `WildfireModel.score` |
| P(ignite) = p_base × fuel × (1 + k cos θ), p_base 0.3, k 0.8, fuel 1.0 / 0.5, Burnt after 4 | `environment/fire.py` |
| Partially observable: scouts radius 3, firefighters radius 1 | `agents/scout.py::sense_window` |
| Shared belief map, newest timestamp wins, stale after 15 steps | `belief_map.py` |
| Scout: frontier value = age / (1 + distance), 5-cell target exclusion | `algorithms/frontier.py` |
| Firefighter action priority: refill → extinguish → firebreak → A\* step | `Firefighter.step` |
| Risk-aware A\*: c(n) = 1 + λ·risk(n), λ = 5, Manhattan heuristic; re-plan when blocked or every 10 steps | `algorithms/astar.py`, `Firefighter._go_to` |
| Coordinator: connected components every 5 steps, threat = size × mean downwind ignition probability | `algorithms/zones.py`, `Coordinator.recluster` |
| Sequential single-item auction, U(f,z) = threat/(1+pathcost) × water/water_max, 2 slots for zones > 20 cells | `algorithms/auction.py`, `Coordinator.allocate` |
| Messages OBSERVE, TARGET, ANNOUNCE, BID, AWARD, DONE | `messages.py` (all traffic goes through `MessageBus`) |
| Conflicts: auction for zones, cell reservations in id order, scout target exclusion, newest-wins, replan/abandon, re-auction when absent > 15 steps | `Coordinator`, `WildfireModel.step`, `Firefighter._go_to`, `Coordinator.release_absent` |
| Baselines: independent (private maps, no Coordinator) and greedy nearest | `strategies.py`, `greedy_assignment` |
| Six scenarios, 30 seeds each, 3 strategies | `scenarios.py`, `run_experiments.py` |

## 8. Implementation decisions

These fill in details Review 1 did not fix. Every one is a switch or a value in
`config.yaml` marked `impl`.

1. **Fire time scale, `fire.spread_every: 5`.** This is the most important one to know for
   the viva. With the Review 1 probabilities (p_base 0.3, 85% tree density, 4-step burn), a
   fire updated every agent step burns ~90% of the 50 × 50 forest in about 100 steps. We measured that even **firefighters given
   perfect knowledge of the fire** save only ~10% under that timing. No strategy can make a
   difference, so the experiments could not test the Review 1 hypothesis. We keep every Review
   1 probability and the 4-update burn time unchanged, and let agents act 5 times per fire
   update (read it as: the fire front moves more slowly than a ground crew or a drone). Set
   `spread_every: 1` for the literal timing.
2. **Terrain is known a priori.** Agents know where lakes and rivers are (a forest map), but
   not the fire state. Every cell's fire state starts `UNKNOWN`. Without this, independent
   firefighters with a radius-1 view could never find water to refill.
3. **Safe footing** (`firefighter.secure_footing`). A firefighter next to fire on a Tree cell
   first cuts its own cell into a firebreak (an existing actuator), and does not step onto
   fuel beside the fire on a fire-update step. Without this, up to ~3 of 5 firefighters were
   caught per run, even with perfect knowledge of the fire.
4. **Support rounds** (`coordinator.assign_idle`). After the design slots (1 per zone, 2 for
   zones > 20 cells) are filled, leftover free firefighters are auctioned too, one extra slot
   per zone per round. The Coordinator's own PEAS measure is "few idle firefighters". The
   greedy baseline gets the same rounds, so the comparison stays fair.
5. **Idle firefighters explore** (`firefighter.explore_when_idle`) with the same frontier
   rule as scouts when the team knows of no fire. This is what lets the system work with
   0 scouts (Scenario 4).
6. **Zone changes.** Merged/split zones keep their firefighters and advertise any newly
   opened slots. Firefighters of a zone that disappears are released with a `REVOKE` message
   (the one message type added to the Review 1 protocol, also used for "absent > 15 steps").
   The zone table is published by the Coordinator on the shared blackboard next to the belief
   map.
7. **Start positions.** Firefighters start on distinct cells of the base-station area (the
   base and its nearest neighbours), so the "never share a cell" rule holds from step 0.
   Scouts all start on the base cell (they fly).
8. **Yielding.** An idle firefighter that blocks a teammate's only route steps aside
   (logged as a `yield` event). This fixed a deadlock where the firefighter starting on the
   base cell was boxed in by four idle teammates.
9. **Greedy baseline** uses straight-line (Manhattan) distance and ignores water and threat,
   exactly "Coordinator sends each zone its closest firefighter".
10. **Demo layout.** The sidebar reuses Mesa's SolaraViz controllers (`ModelController`,
    `ModelCreator`). The page layout is our own because Mesa 3.5's draggable grid does not
    render with the Vue 3 front end of current Solara.

## 9. Viva quick answers

| Question | Show |
|---|---|
| Why multi-agent? | Demo: `independent` vs `auction` on the same seed; `paired_gains.png` |
| Why A\* over BFS/UCS? | `astar_vs_ucs.png`: 72% fewer nodes; BFS ignores the risk costs |
| Is the heuristic admissible? | `test_astar.py`: A\* cost = UCS cost on 125 random queries |
| What if fire blocks the path? | Replanning in `Firefighter._go_to`; `done_abandoned` events in the demo |
| Partial observability? | Demo belief-map toggle (black = unknown, faded = stale) |
| Does coordination help? | Yes, by +4.8 to +11.6 points where significant, growing with team size; not against 3 simultaneous fires |
| Auction vs greedy? | Statistically tied here (§5). Say so honestly and explain why |
| Is the Coordinator a single point of failure? | Yes. The planned fallback: greedy nearest-zone if no AWARD within 10 steps |
| Limitations | 2D grid, simplified spread, instant communication, no slope, agents act 5× per fire update |
