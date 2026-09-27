# Wildfire Containment Squad

A multi-agent simulation for the Foundations of AI case study. **Scout Drones** explore a
partially observable forest and share what they see. A **Coordinator** clusters the known fire
into zones and auctions them. **Firefighters** bid, travel with risk-aware A\*, put out burning
cells and cut firebreaks while the wildfire spreads stochastically with the wind.

The implementation follows the Review 1 design: the same agents, algorithms, formulas and
parameters. Where building it forced a decision that Review 1 did not fix, or where testing
showed that a detail of the plan hurt performance, the change is listed with its evidence in
[§9 Implementation decisions](#9-implementation-decisions), and every difference from the wording
of the Review 1 report is listed in [§9.1](#91-differences-from-the-review-1-report).

**Team:** Adithya Ajay (CB.SC.U4CSE23102), Amrith B (CB.SC.U4CSE23105),
Gowreesh B (CB.SC.U4CSE23119), Vinaayak Kanagaraj (CB.SC.U4CSE23152).

**Presentation material** (in [`../presentation/`](../presentation/)): the Review 2 slides
(`Review 2 slides.pdf`) and the study guide (`Study guide.pdf`, `Study guide.md`).

### Where each Review 2 criterion is evidenced

| Criterion | Evidence |
|---|---|
| **Tool/Package Selection & Setup (3)** | §1: tools with reasons, pinned `requirements.txt`, one-click `setup.bat` / `setup.sh`, verified by a clean install on Python 3.12 and 3.13 |
| **Multi-Agent Execution & Interaction (3)** | Agents depend on each other through the shared belief map and an ANNOUNCE → BID → AWARD → DONE / REVOKE message protocol (`messages.py`), with conflict rules for zones, cells, scout targets and observations. It is all visible live in the demo (auction log, message counts, events) and tested in `test_auction.py`, `test_conflicts.py` and `test_coordination.py` |
| **Demo Quality & Testing Scenarios (3)** | Live demo with belief-map view, one-click strategy comparison and Coordinator-failure switch (§2). 114 tests, 89% coverage (§4). Six Review 1 scenarios × 3 strategies × 30 seeds, paired statistics, plus four supporting studies (§5–6) |
| **Code Structure & Scalability (1)** | Package layout with Mesa-free algorithm modules, everything in `config.yaml`, runs from 0 to 30 agents without code changes (§7) |

---

## 1. Setup

Requires **Python 3.12 or newer** (Mesa 3.5 needs it). Tested on Python 3.12.14 and 3.13.2,
Windows 11, from a clean virtual environment.

**Windows (one click):** double-click `setup.bat`, then `run_demo.bat`, `run_tests.bat` or
`run_experiments.bat`.

**macOS / Linux:**

```bash
./setup.sh                  # creates .venv and installs requirements
./run.sh demo               # or: ./run.sh tests | ./run.sh experiments [--quick]
```

**Manual:**

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

| Tool | Used for | Why this one |
|---|---|---|
| **Mesa 3.5** (`mesa[viz]`) | `Model`, `Agent`, `MultiGrid`, `DataCollector`, run controls | Purpose-built for agent-based models: agent registry, seeded RNG and data collection built in; Mesa 3 API throughout (no deprecated schedulers) |
| **Solara** | Browser UI for the live demo | Mesa's own visualisation stack; reactive widgets for play/step/reset and parameters |
| **NumPy** | Grid state, vectorised fire spread and risk maps | The whole-grid spread is 8 array shifts, not Python loops; a 300-step run takes ~1.4 s |
| **heapq** (stdlib) | A\* / UCS priority queue | O(log n) push/pop, no extra dependency |
| **PyYAML** | `config.yaml` | Every parameter in one readable file |
| **Matplotlib / pandas** | Demo rendering, charts, result tables | Standard, reproducible PNG + CSV output |
| **pytest** | 114 tests | Parametrised over seeds, strategies and scenarios |

`requirements-dev.txt` adds `pytest-cov` and `ruff` for coverage and linting.

## 2. The live demo

```bash
solara run app.py          # http://localhost:8765   (or run_demo.bat / ./run.sh demo)
```

* **Left:** Reset / Play / Step, speed, and every parameter: strategy, scenario preset, number
  of scouts and firefighters, wind direction and strength, ignitions, fire time scale,
  **Coordinator fails at step**, seed. Press **Reset** to apply.
* **Centre:** the forest. Dark green = dense, light green = sparse, orange-red = burning,
  grey = burnt, brown = firebreak, blue = water, beige = empty. Cyan circles are scouts,
  yellow squares are firefighters (with id), white × marks zone reference cells, and the arrow
  shows the wind. **Show belief map** switches to what the team knows: black = unknown,
  faded = stale (not seen for more than 15 steps).
* **Right:** the current settings in one line, % forest saved, zones with their threat and
  crews, each firefighter's water and activity, the **last auction rounds with every bid and
  the winner**, message counts, and events (awards, revokes, fallbacks, agents lost).
* **Below:** % forest saved and burning cells over time, and **"Compare all 3 strategies on
  this exact setup"**. This runs independent, greedy and auction on the same seed in about
  one second and shows their results and final maps side by side.

**Five-minute demo script:**

1. Preset `windy_single`, seed 0, strategy `auction` → Reset → Play. Scouts sweep, the fire is
   found, ANNOUNCE/BID/AWARD rounds appear on the right, crews converge; the fire is
   contained at step 195 with 84.7% saved.
2. Tick **Show belief map**: black unknown areas, faded stale areas (partial observability).
3. Press **Compare**: independent 49.5%, greedy 59.5%, auction 84.7% on this exact fire.
4. Preset `river`: independent firefighters never reach the far bank. Coordinated ones use A\*
   through the two crossings.
5. Set **Coordinator fails at step** = 60 → Reset → Play: `fallback` events appear and the
   crews keep fighting on their own (Review 1's "single point of failure" answer).

## 3. Experiments and studies

```bash
python run_experiments.py --quick                      # 3 seeds per cell, ~30 s
python run_experiments.py --scenarios all --runs 30    # 720 runs, ~5 min on 4 workers
python run_studies.py                                  # 4 supporting studies, 1090 runs, ~5 min
python run_experiments.py --replot                     # rebuild tables/charts from CSVs
python run_studies.py --replot
```

Outputs in `results/` (main study) and `results/studies/`:

| File | Content |
|---|---|
| `results.csv`, `summary.csv` | One row per run; mean and std per scenario × strategy |
| `paired_gains.csv`, `paired_gains.png` | Auction minus each baseline on the **same seed**, ≈95% CI, win counts |
| `forest_saved_by_scenario.png`, `scaling.png` | Grouped bars per scenario; Scenario 5 team-size curve |
| `astar_vs_ucs.png` | Nodes expanded per search, A\* vs UCS on identical queries |
| `studies/timing.png`, `tactics.png`, `allocation.png`, `robustness.png` | The four supporting studies (§6), each with `_runs.csv`, `_summary.csv`, `_paired.csv` |

Map, fire and agent decisions use separate seeded random streams. For a given seed every
strategy faces **the identical map and the identical fire random numbers**, so the comparisons
are paired.

## 4. Tests

```bash
python -m pytest                                  # 114 tests, ~40 s
python -m pytest --cov=. --cov-config=.coveragerc # 89% overall; simulation modules 93–100%
```

| File | What it proves |
|---|---|
| `test_fire.py` | Downwind 1.8× / upwind 0.2× (k = 0.8), fuel scaling, clipping, independent-neighbour combination, non-Tree cells never ignite, Burning → Burnt after 4 updates, same seed ⇒ same fire, ignition placement, river layout |
| `test_belief_map.py` | Newest observation wins, older observation ignored, age and staleness, passability |
| `test_astar.py` | A\* cost = UCS cost on 125 random weighted queries (admissible heuristic), A\* never expands more nodes, avoids fire, detours when λ is high, no path when blocked, reservations only affect the first move |
| `test_zones.py` | 8-connected clustering, stable ids as zones grow, merge / split / removal, threat, downwind firebreak targets |
| `test_auction.py` | Utility formula, highest bid wins, tie → lower id, zones by threat and the loser bids on the next zone, large zone → 2 firefighters, unreachable → no bid, greedy ignores threat, ANNOUNCE → BID → AWARD end to end |
| `test_conflicts.py` | Never two firefighters on one cell (all strategies, 10 firefighters), corridor reservations, scout targets > 5 cells apart, replanning around new fire, no deadlock at the base |
| `test_scouts.py` | Frontier definition, stale cells return to the frontier, target exclusion and fallback, scouts explore > 80% of the grid, shared vs private maps |
| `test_coordination.py` | Re-auction after > 15 steps absent, abandoned zone re-offered, event-driven re-allocation, greedy sends AWARD without BID, Coordinator failure → fallback, no fallback when disabled, safety-first footing, live zone tracking, zone linking, fire time scale, targeted preset, perfect-knowledge mode |
| `test_model_smoke.py` | Every strategy and scenario runs, 0–30 agents without code changes, score = Review 1 formula, deterministic per seed, caught firefighters removed |
| `test_tools.py` | Demo model/drawing/comparison, experiment and study runners end to end (CSV + charts + replot), scenario expansion |

## 5. Main results (6 scenarios × 3 strategies × 30 seeds = 720 runs)

Mean % forest saved (± std across seeds):

| Scenario | Independent | Greedy nearest | **Auction (full system)** |
|---|---|---|---|
| S1 calm_single | 15.2 ± 13.9 | 20.6 ± 17.3 | **24.5 ± 21.5** |
| S2 windy_single | 42.6 ± 24.7 | 54.3 ± 24.0 | **56.1 ± 26.4** |
| S3 multi_ignition | 5.0 ± 4.6 | 4.6 ± 4.8 | **5.8 ± 6.6** |
| S4 no_scouts | 42.6 ± 24.7 | 49.4 ± 25.6 | **49.5 ± 25.4** |
| S5 scaling_2 | 41.3 ± 25.6 | **46.6 ± 27.3** | 45.4 ± 28.1 |
| S5 scaling_5 | 42.6 ± 24.7 | 54.3 ± 24.0 | **56.1 ± 26.4** |
| S5 scaling_10 | 44.0 ± 23.7 | 66.7 ± 25.7 | **69.3 ± 26.0** |
| S6 river | 65.1 ± 13.0 | **77.2 ± 13.0** | 76.2 ± 13.0 |

The large standard deviations are the fire itself: some ignitions burn out early, others run
across the map. The paired comparison on the same seeds removes that:

| Scenario | Auction − independent | wins | Auction − greedy | wins |
|---|---|---|---|---|
| S1 calm_single | **+9.3 ± 6.1** | 23/30 | +3.9 ± 5.6 | 21/30 |
| S2 windy_single | **+13.5 ± 4.7** | 24/30 | +1.8 ± 3.7 | 16/30 |
| S3 multi_ignition | +0.8 ± 1.7 | 14/30 | **+1.2 ± 0.9** | 18/30 |
| S4 no_scouts | **+6.9 ± 3.2** | 22/30 | +0.1 ± 1.9 | 13/30 |
| S5 scaling_2 | **+4.1 ± 2.9** | 19/30 | −1.2 ± 2.7 | 15/30 |
| S5 scaling_5 | **+13.5 ± 4.7** | 24/30 | +1.8 ± 3.7 | 16/30 |
| S5 scaling_10 | **+25.3 ± 7.4** | 26/30 | +2.6 ± 3.9 | 15/30 |
| S6 river | **+11.1 ± 2.4** | 29/30 | −1.0 ± 2.2 | 10/30 |

Bold = the ≈95% confidence interval excludes zero. (S2 and S5-scaling_5 are the same default
configuration, so their numbers match. That is a check that runs are deterministic.)

Other measured facts:

* **A\* vs UCS:** on identical queries, with the same optimal cost, A\* expands **63% fewer
  nodes** overall (49–87% per scenario).
* **Safety:** 0.10 firefighters caught per run with the auction (0.16 greedy, 0.13 independent).
* **Work done:** coordinated crews extinguish ~75 cells per run vs ~29 for independent ones.
* **Runtime:** ~1.4 s per 300-step run including the UCS measurement.

### Findings vs the Review 1 hypothesis

Review 1 predicted: *"The full system should save the most forest, with the largest margin in
Scenarios 2 and 3."*

* **Confirmed: the full system saves the most.** It has the highest mean in 6 of 8
  configurations and beats independent agents significantly in 7 of 8, by up to +25 points.
* **Confirmed: the margin grows with team size.** +4.1 → +13.5 → +25.3 points for 2 → 5 → 10
  firefighters. Independent firefighters barely improve with numbers (41 → 43 → 44%) because
  they only find fire by stumbling on it and then crowd it.
* **Confirmed for Scenario 2:** one of the largest margins over independent agents (+13.5).
* **Partly confirmed for Scenario 3:** it is the only scenario where the **auction beats greedy
  significantly** (+1.2 ± 0.9), exactly the threat-aware allocation benefit Review 1 argued.
  But three simultaneous fires overwhelm five firefighters under every strategy (≤ 6% saved),
  so the margin over independent agents is small.
* **Not confirmed: auction vs greedy in general.** Elsewhere the difference is within noise.
  With one or two fire zones there is rarely a real choice to make. The targeted allocation
  study (§6.3) shows the same. **Most of the gain comes from the shared belief map plus
  coordinated targeting; the choice of allocation rule matters only when several zones compete.**

## 6. Supporting studies (`run_studies.py`)

### 6.1 Fire time scale — why agents act 5 times per fire update

With the literal Review 1 timing (fire updates every agent step), **every strategy saves the
same ~11.6%** (auction − independent = +0.0 ± 0.7). **Even firefighters given perfect
knowledge of the fire save only 15%.** The fire crosses the 50 × 50 forest in about 100
steps, faster than any team can act, so the literal timing cannot test the hypothesis at all.
As agents get more actions per fire update, coordination starts to matter: at 5 steps per
update the auction gains **+14.1 ± 5.7** over independent agents (`studies/timing.png`).

| Agent steps per fire update | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|
| Independent | 11.6 | 15.7 | 25.3 | 45.9 | 66.1 |
| Greedy | 11.6 | 17.7 | 28.9 | 56.6 | 88.8 |
| Auction | 11.6 | 18.7 | 30.5 | 60.0 | 90.3 |
| Auction + perfect knowledge | 15.0 | 23.0 | 44.2 | 79.9 | 96.8 |

### 6.2 What a firefighter targets inside its zone

The implementation plan sent each firefighter to the zone's downwind edge (the head of the
fire). Review 1's agent analysis describes the firefighter as utility-based, able to "prefer
cells that stop the most future spread". Implementing that as *spread threatened by a burning
cell ÷ (1 + distance)* saves **+8.1 ± 4.2** points more than the downwind-edge rule (windy,
auction; +7.1 ± 4.0 for greedy). It is now the default (`firefighter.zone_tactic: utility`).

### 6.3 Targeted allocation test

A low-threat fire in sparse forest near the base plus a high-threat fire in dense forest farther
away, with only 3 firefighters. Auction and greedy tie (−0.8 ± 2.0), and both beat
independent agents (+6.9 ± 3.4). Linking zone fragments within 3 cells helps both (+3
points). Threat-aware allocation did not outperform nearest-first here, most likely because
the near fire is found first and both methods commit crews to it before the far fire is known.

### 6.4 Coordinator failure (the single point of failure from Review 1's Q&A)

The Coordinator stops at step 60. Without a fallback the team collapses to independent level
(43.9%). With the implemented fallback (a firefighter that knows about fire but receives no
AWARD for 10 steps attacks the nearest known fire), the team keeps working: **+16.8 ± 4.9**
points over no fallback. It even edges out the working Coordinator here (+4.6 ± 4.1): with a
single fire, once every crew is engaged, "everyone to the nearest fire on the shared map" is
as good as zone allocation. That is consistent with §5: the shared map is the main source of
the gain.

## 7. Project structure and scalability

```
wildfire_squad/
├── config.yaml              # every parameter (Review 1 values + marked implementation details)
├── settings.py              # load config.yaml, deep-merge scenario/experiment overrides
├── model.py                 # WildfireModel: fixed step order, planning cache, metrics, score
├── belief_map.py            # BeliefMap: state + last-seen step per cell, newest-wins fusion
├── messages.py              # OBSERVE/TARGET/ANNOUNCE/BID/AWARD/DONE/REVOKE + message bus
├── strategies.py            # independent | greedy | auction switches
├── scenarios.py             # the six Review 1 scenarios as config overrides
├── environment/
│   ├── cells.py             # CellState enum, neighbourhoods, distances
│   ├── forest.py            # random forest, river preset, targeted "competing" preset, ignitions
│   └── fire.py              # ignition_probability, vectorised spread, risk map
├── agents/
│   ├── scout.py             # Scout Drone (model-based, goal-based)
│   ├── firefighter.py       # Firefighter (model-based, utility-based)
│   └── coordinator.py       # Coordinator (utility-based)
├── algorithms/              # pure functions, no Mesa dependency, unit-tested directly
│   ├── astar.py             # risk-aware A* + UCS, node counters
│   ├── auction.py           # utility, sequential single-item auction, greedy baseline
│   ├── frontier.py          # frontier detection + exploration target selection
│   └── zones.py             # connected components, stable ids, live zones, threat, firebreak targets
├── run_experiments.py       # main study → results/
├── run_studies.py           # supporting studies → results/studies/
├── app.py                   # live demo
├── setup.bat / setup.sh, run_*.bat / run.sh
├── results/                 # CSVs and charts
└── tests/                   # 114 pytest tests
```

**Scalability.** No agent count is hard-coded: the same code runs 0 scouts, 1–30
firefighters and any number of ignitions (`test_scales_to_any_agent_count`). The algorithms are
pure functions on arrays and callbacks. Every parameter is read from `config.yaml`, and every
random draw goes through the model's seeded generators.

## 8. How the design maps to code

| Review 1 element | Where |
|---|---|
| Score = w1·T_saved/T_initial − w2·t_contain/t_max − w3·A_lost/A_total (0.7, 0.2, 0.1) | `WildfireModel.score` |
| P(ignite) = p_base × fuel × (1 + k cos θ), p_base 0.3, k 0.8, fuel 1.0 / 0.5, Burnt after 4 | `environment/fire.py` |
| Partially observable: scouts radius 3, firefighters radius 1 | `agents/scout.py::sense_window` |
| Shared belief map, newest timestamp wins, stale after 15 steps | `belief_map.py` |
| Scout: frontier value = age / (1 + distance), 5-cell target exclusion | `algorithms/frontier.py` |
| Firefighter priority: (safety) → refill → extinguish → firebreak → A\* step | `Firefighter.step` |
| Risk-aware A\*: c(n) = 1 + λ·risk(n), λ = 5, Manhattan heuristic; replan when blocked or every 10 steps | `algorithms/astar.py`, `Firefighter._go_to` |
| Coordinator: connected components every 5 steps, threat = size × mean downwind ignition probability | `algorithms/zones.py`, `Coordinator.recluster` |
| Sequential single-item auction, U(f,z) = threat/(1+pathcost) × water/water_max, 2 slots above 20 cells | `algorithms/auction.py`, `Coordinator.allocate` |
| Messages OBSERVE, TARGET, ANNOUNCE, BID, AWARD, DONE | `messages.py` (all interaction goes through `MessageBus`) |
| Conflicts: auction for zones, reservations in id order, scout target exclusion, newest-wins, replan/abandon, re-auction after 15 steps absent | `Coordinator`, `WildfireModel.step`, `Firefighter._go_to`, `Coordinator.release_absent` |
| Coordinator single point of failure → greedy fallback after 10 steps without AWARD | `Firefighter._await_award_or_fallback`, `coordinator.fail_at_step` |
| Baselines: independent (private maps, no Coordinator) and greedy nearest | `strategies.py`, `greedy_assignment` |

## 9. Implementation decisions

Each one is a switch or a value in `config.yaml`, and each is backed by a study or a test.

1. **Fire time scale** (`fire.spread_every: 5`). All Review 1 probabilities are unchanged; agents
   get 5 actions per fire update. At 1 (literal timing) no strategy, not even perfect
   knowledge, can make a difference (§6.1).
2. **Utility-based targeting inside a zone** (`firefighter.zone_tactic: utility`), from Review 1's
   agent analysis. +8 points over the plan's downwind-edge rule (§6.2), which remains
   selectable as `firebreak`. The AWARD's target cell is used to price bids; the winner then
   picks its own cell inside the zone.
3. **Live zones and event-driven allocation.** A firefighter tracks its zone on the shared map
   between clustering rounds, and the Coordinator re-allocates a firefighter as soon as it is
   free ("periodic and event-driven replanning", Review 1 §4). Without this, crews chased
   burnt-out cells and waited idle for up to 5 steps.
4. **Safety first** (`firefighter.secure_footing`). A firefighter standing on fuel next to fire
   first cuts its own cell into a firebreak (an existing actuator), before anything else. On a
   fire-update step it does not step onto fuel within 2 cells of known fire. This cut losses
   from ~0.9 to ~0.1 per run.
5. **Coordinator-failure fallback** (`firefighter.award_timeout: 10`), the robustness extension
   Review 1 proposed (§6.4).
6. **Support rounds** (`coordinator.assign_idle`). After the design slots are filled, leftover
   free firefighters are auctioned one extra slot per zone ("few idle firefighters" is the
   Coordinator's PEAS measure). The greedy baseline gets the same rounds.
7. **Terrain known a priori.** Agents know where water is (a forest map) but not the fire;
   every cell's fire state starts unknown.
8. **Idle exploration** (`firefighter.explore_when_idle`). With no known fire, free firefighters
   explore the frontier. This is what makes Scenario 4 (no scouts) work.
9. **REVOKE message** (one addition to the protocol) for "absent > 15 steps" and "zone gone".
10. **Start cells and yielding.** Firefighters start on distinct base cells. An idle firefighter
    blocking a teammate's only route steps aside.
11. **Demo layout.** The run controls are Mesa's (`ModelController`, `ModelCreator`); the page
    layout is custom because Mesa 3.5's draggable grid does not render with current Solara.

### 9.1 Differences from the Review 1 report

Where the code does not do literally what the Review 1 report says, this is why:

| Review 1 report says | The code does | Why |
|---|---|---|
| "A burning cell becomes Burnt after 4 steps" (§4) | Burnt after 4 **fire updates**; the fire updates every 5 agent steps, so 20 agent steps | Decision 1: every probability is unchanged, only the time scale (§6.1) |
| Actuator: "cut a firebreak on an **adjacent** tree cell" (§3.2) | The firefighter walks onto the tree and converts the cell it is **standing on** (`Firefighter.cut_firebreak`) | Same actuator, one cell per action, no water. Standing on it makes that cell safe the moment it is cut, which is also the safety rule (decision 4) |
| Goal test: "position is **adjacent** to the assigned zone's target cell" (§6.1) | The goal is the target cell itself when it can be entered (a tree to cut), and its free 4-neighbours when it cannot (a burning cell to extinguish) (`algorithms/astar.py::goal_cells`) | A burning cell must be reached from beside it, as the report says; a tree must be stood on to be cut |
| Mesa provides "grid, **scheduler**" (§9.1) | Mesa 3 removed schedulers; `WildfireModel.step()` calls scouts, Coordinator and firefighters in a fixed order itself | The plan required the Mesa 3 API; a fixed order also makes the conflict rules deterministic |
| Firefighter: "if at the zone's downwind edge, cut a firebreak" (§5.2) | Targets the burning cell of its zone that stops the most spread per step of travel; the downwind edge is the fallback, and remains selectable (`zone_tactic: firebreak`) | Decision 2: Review 1 §5 calls the firefighter utility-based; +8.1 points (§6.2) |
| Six message types (§8.1) | Seven: REVOKE added | Decision 9 |

## 10. Viva quick answers

| Question | Answer / show |
|---|---|
| Why multi-agent? | Demo **Compare** button; `paired_gains.png`: +4 to +25 points over independent agents |
| Why A\* over BFS/UCS? | Costs differ (fire risk), so BFS is not optimal; `astar_vs_ucs.png`: 63% fewer nodes than UCS at the same cost |
| Is the heuristic admissible? | Every step costs ≥ 1 and moves one cell, so Manhattan never overestimates; `test_astar.py` checks A\* cost = UCS cost on 125 queries |
| What if fire blocks the path? | Replan; if no path, DONE(abandoned) and the zone is re-auctioned (`test_coordination.py`) |
| Partial observability? | Belief-map toggle in the demo; scouts vs no scouts (+6.9 with shared map even without scouts) |
| Does coordination help? | Yes: significant in 7 of 8 configurations, growing with team size |
| Auction vs greedy? | Significant only with multiple fires (S3); otherwise tied, because one or two zones leave no real choice (§5 findings) |
| Why not the literal fire timing? | §6.1: at literal timing even perfect knowledge saves only 15%; nothing can be compared |
| Single point of failure? | Switch "Coordinator fails at step" in the demo: the fallback keeps the team working (+16.8 vs no fallback) |
| Why does the code differ from the report here and there? | §9.1 lists every difference and its reason |
| Limitations | 2D grid, simplified spread, instant communication, no slope; agents act 5× per fire update; auction ≈ greedy with few zones |
