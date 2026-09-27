# Wildfire Containment Squad — Review 2 — speaker notes

## 1. Wildfire Containment Squad

Speaker: Adithya. Open with the one-line pitch: scouts find the fire, a coordinator auctions fire zones, firefighters bid and fight with risk-aware A*. On the same fires, the coordinated team saves up to 25 percentage points more forest than independent agents. Then hand over to the recap.

## 2. No single agent can see or reach the whole fire

Speaker: Adithya. The design is unchanged from Review 1: perception is split from action. Read the diagram from the bottom left: the environment only shows each agent what is near it, scouts write it into the shared belief map, the Coordinator turns the map into zones and auctions them, firefighters bid, travel and act. All three agent types share one score: 0.7 x forest saved, minus 0.2 x time to contain, minus 0.1 x firefighters lost.

## 3. Each tool was chosen for one job

Speaker: Adithya. Rubric: tool and package selection and setup. Stress that the setup was verified from scratch: a fresh Python 3.12 environment, install from requirements.txt, all 114 tests pass. That test caught a real problem (Mesa's visualisation needs its viz extra), which is now fixed. Mesa 3 API throughout, no deprecated schedulers.

## 4. Six fixed stages per step, seven messages

Speaker: Amrith. Rubric: multi-agent execution and interaction. The fixed order makes every run reproducible and gives every conflict a winner. Conflict rules to mention: the auction resolves two firefighters wanting one zone; reservations in id order resolve two wanting one cell; scouts exclude cells within 5 of each other's targets; the newest observation wins; a blocked path means replan or DONE(abandoned) and re-auction; a firefighter away refilling for more than 15 steps loses its zone. REVOKE is the one message added to Review 1's protocol.

## 5. Risk-aware A* and a threat-first auction

Speaker: Amrith. Admissibility in one breath: every move costs at least 1 and changes the Manhattan distance by at most 1, so the heuristic never overestimates. test_astar.py checks that A* and UCS return the same cost on 125 random weighted queries, and the chart shows A* expanding 63 percent fewer nodes overall. The auction: the Coordinator announces each zone, every free firefighter bids with its real A* path cost and its water, the highest bid wins, ties go to the lower id. Why not Hungarian: zones change every few steps; the auction is cheap to rerun and decentralised.

## 6. Same fire, three strategies

Speaker: Gowreesh. Switch to the browser (run_demo.bat). Script: 1) preset windy_single, seed 0, auction, Reset, Play. Point at scouts, then at the auction rounds with every bid on the right. 2) Pause, tick Show belief map: black unknown, faded stale. 3) Let it finish: contained at step 195 with 84.7 percent saved. 4) Press Compare: independent 49.5, greedy 59.5, auction 84.7 on this exact fire. 5) Optional: preset river, independent firefighters never reach the far bank. 6) Optional: Coordinator fails at step 60, fallback events appear and crews keep fighting. If the demo fails, stay on this slide: it is the backup.

## 7. Three layers of testing on identical fires

Speaker: Vinaayak. Rubric: demo quality and testing scenarios. Examples of tests worth naming: the fire formula gives exactly 1.8 times downwind and 0.2 times upwind; A* cost equals UCS cost on 125 random queries; two firefighters never share a cell even with 10 of them; the Coordinator failing triggers the fallback. Separate random streams for the map, the fire and the agents are what make the comparison paired.

## 8. Coordination wins in 7 of 8 configurations

Speaker: Vinaayak. How to read the chart: each dot is the average gain of the auction over a baseline on the same 30 fires; the whisker is the 95 percent confidence interval; a whisker entirely above zero means significantly better. Against independent agents (orange dots): significant in 7 of 8, up to +25.3 points. Against greedy (green dots): tied except with three fires. The gain grows with team size: +4.1, +13.5 and +25.3 points for 2, 5 and 10 firefighters, because extra firefighters only help if someone tells them where to go. Also: A* 63 percent fewer nodes than UCS; about 0.1 firefighters caught per run.

## 9. What we predicted, and what we found

Speaker: Vinaayak. Say this before anyone asks. Owning the finding reads as rigour. We tested the auction's claimed advantage directly with a targeted map, a small fire near the base and a dangerous one far away, and it still tied with greedy, most likely because the near fire is found first and both methods commit crews to it before the far one is known.

## 10. Every change from the plan has a study behind it

Speaker: Gowreesh. Each is a switch in config.yaml, so an evaluator can turn it off and see the difference. Time scale: every Review 1 probability is unchanged; only the unit of time changes. Utility targeting is what Review 1's own agent analysis describes: prefer cells that stop the most spread. The fallback answers Review 1's single-point-of-failure question; demo it with the Coordinator-fails slider. Also built in: live zones and event-driven allocation, so crews never chase burnt-out cells or wait idle.

## 11. 0 to 30 agents with no code change

Speaker: Amrith. Rubric: code structure and scalability. About 3,300 lines of code plus 900 lines of tests, type hints and docstrings throughout, formatted and linted with ruff. Patterns to name: a blackboard (the shared map), contract net (announce, bid, award), and a strategy switch (one code path, three strategies).

## 12. What this model leaves out

Speaker: Adithya. Close on the honest summary: coordination helps significantly and more with larger teams; the shared map carries most of the gain; the auction matters when fires compete. Then open for questions: whoever owns the slide in question answers first.
