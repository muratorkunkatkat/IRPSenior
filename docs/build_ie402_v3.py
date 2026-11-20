"""Builds docs/IE 402 v3.docx: the IE 401 report with the changes of the full project marked.

Green: added. Red inside { }: taken out (kept in the file so the change can be applied by
hand in Google Docs). Pictures get a green or red band. Text of IE 401 that is still true
is left as it is; only the parts that changed are marked.

    python docs/build_ie402_v3.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import docx
from docx.oxml.ns import qn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ie402 import Builder, caption, fmt, heading, label_line, para  # noqa: E402
from build_ie402_v2 import (MERGE, GREEN, added, add_after, append, block_until, element_text,  # noqa: E402
                            removed, strike, swap)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PLOTS = ROOT / "plots"
SRC = DOCS / "IE 401.docx"
OUT = DOCS / "IE 402 v3.docx"
REPORT = PLOTS / "report"
CFG = PLOTS / "analysis"
EXT = CFG / "extended"


def between(p, start_text, end_text):
    """Exact text of a paragraph from start_text up to and including end_text."""
    full = element_text(p)
    a = full.find(start_text)
    b = full.find(end_text, a)
    if a < 0 or b < 0:
        raise SystemExit(f"span not found: {start_text!r} .. {end_text!r}")
    return full[a:b + len(end_text)]


def nth(p, context, text):
    """Occurrence index of `text` at the place where `context` starts."""
    full = element_text(p)
    pos = full.find(context)
    if pos < 0:
        raise SystemExit(f"context not found: {context!r}")
    pos += context.find(text)
    count, start = 0, full.find(text)
    while start != -1 and start < pos:
        count += 1
        start = full.find(text, start + 1)
    return count


def pct(old, new):
    return 100.0 * (old - new) / old


def extended_rows():
    jobs = (
        ("greedy", "00_original", "sa_T75", "Original, greedy start", "SA T75"),
        ("greedy", "00_original", "sa_T25", "Original, greedy start", "SA T25"),
        ("greedy", "00_original", "sa_500", "Original, greedy start", "SA 500"),
        ("random", "00_original", "sa_T75", "Original, random start", "SA T75"),
        ("greedy", "02_growth_no_fleet", "sa_T75", "Growth, greedy start", "SA T75"),
        ("greedy", "02_growth_no_fleet", "sa_T25", "Growth, greedy start", "SA T25"),
        ("greedy", "02_growth_no_fleet", "sa_500", "Growth, greedy start", "SA 500"),
    )
    rows = []
    for start, config, key, title, label in jobs:
        stem = f"{start}__{config}__{key}.json"
        base = json.loads((CFG / "_cache" / stem).read_text())
        ext = json.loads((EXT / "_cache" / stem).read_text())
        if "cool_seconds" not in ext:
            raise SystemExit(f"1000 minute run missing: {stem}")
        greedy = base["greedy_total"]
        rows.append([title, label, fmt(base["best"]), fmt(ext["best"]),
                     f"{pct(greedy, base['best']):.1f}%", f"{pct(greedy, ext['best']):.1f}%"])
    return rows


def lint(document):
    """Report AI-looking marks in the added (green) text."""
    words = []
    for run in document.element.body.iter(qn("w:r")):
        mark = run.find(f"{qn('w:rPr')}/{qn('w:highlight')}")
        if mark is not None and mark.get(qn("w:val")) == GREEN:
            words.append("".join(t.text or "" for t in run.iter(qn("w:t"))))
    text = "\n".join(words)
    checks = {
        "semicolon": r";",
        "dash": r"[—–]",
        "hyphenated word": r"[A-Za-z]-[A-Za-z]",
        "british spelling": r"neighbour|behaviour|colour|organis",
        "dynamic": r"[Dd]ynamic",
        "tenure": r"[Tt]enure",
        "IE 40x": r"IE ?40[12]",
        "company scale": r"[Cc]ompany.scale",
    }
    found = False
    for name, pattern in checks.items():
        for m in re.finditer(pattern, text):
            found = True
            print(f"lint {name}: ...{text[max(0, m.start() - 40):m.end() + 40]!r}")
    if not found:
        print("lint: clean")


def main():
    work = DOCS / "_ie401_merged_v3.docx"
    done = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(MERGE), str(SRC), "-o", str(work)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"merge_runs failed: {done.stderr or done.stdout}")
    d = docx.Document(str(work))
    b = Builder(d)
    A = Builder.after

    # ------------------------------------------------------------ cover, abstract
    swap(b.find("SPRING 2026"), "SPRING 2026", "FALL 2026")
    add_after(b.find("This project focuses on optimizing"), "given operating constraints.",
              " For large problems, a greedy heuristic and two metaheuristics, simulated annealing and tabu "
              "search, were developed to find good solutions in reasonable time.")

    # ------------------------------------------------------------ chapter 1
    append(b.find("To achieve these objectives"),
           ". A greedy heuristic builds a first distribution plan, and two metaheuristics, simulated annealing "
           "and tabu search, improve this plan.")
    swap(b.find("The mathematical model serves as a first step"),
         "future studies with more advanced optimization techniques",
         "the heuristic methods used for the real scale problem")
    p = b.find("This report is divided into seven chapters")
    swap(p, "seven", "eight")
    swap(p, "Chapter 7 concludes the study.",
         "Chapter 7 explains the implementation, and Chapter 8 concludes the study.")

    # ------------------------------------------------------------ chapter 4
    strike(b.find("The aim of the model is to minimize"),
           " Additionally, a small penalty is applied for tour time to encourage more efficient route planning "
           "decisions.")
    for t in d.tables:
        for row in t.rows:
            if row.cells[0].text.strip().startswith("IE 366"):
                append(row.cells[1].paragraphs[-1]._p,
                       ", designing the greedy heuristic, simulated annealing and tabu search")

    # ------------------------------------------------------------ 5.1 iv) literature
    A(b.find("Sustainability has become a key concern"), *added(
        heading("iv) Metaheuristics and Their Parameters", 4),
        para("Because exact methods can only solve small IRP instances, metaheuristics that improve a feasible "
             "solution through small changes are widely used for large instances [2]. Simulated annealing (SA) "
             "accepts a worse neighboring solution with a probability that decreases as a temperature parameter is "
             "lowered, which lets the search leave local optima. Its performance depends strongly on the initial and "
             "final temperatures. Duman and Duman [13] set both through acceptance probabilities instead of fixed "
             "values: the initial temperature is chosen so that a solution 10% worse than a reference solution is "
             "accepted with a probability of 0.75, 0.50 or 0.25, and the final temperature so that a solution 1% "
             "worse is accepted with a probability of 0.0001, and the temperature is reduced geometrically between "
             "the two. This makes the temperatures independent of the scale of the objective function."),
        para("Tabu search (TS), introduced by Glover [14], moves in every iteration to the best solution in a "
             "neighborhood of the current solution, even if it is worse, and prevents the search from cycling by "
             "keeping recently performed moves in a tabu list. The tabu size determines how long a move stays in this "
             "list, and a tabu move can still be accepted if it leads to a solution better than the best one found "
             "so far (aspiration criterion). When the neighborhood is very large, only a sample of neighbors is "
             "evaluated in each iteration, so the sample size and the tabu size are the two main parameters of the "
             "method."),
        para("When a local search uses several types of moves, the choice of move affects its efficiency. Ropke and "
             "Pisinger [15] reward each operator according to the solutions it produces: 33 points for a new best "
             "solution, 9 for an improvement of the current solution, and 13 for a worse solution that is accepted, "
             "so that operators that lead the search out of local optima are also rewarded. Operators are then "
             "selected with probabilities based on their average scores. The SA temperatures, the structure of the "
             "TS and the move weights of this project are based on these three works (Section 5.3)."),
    ))

    # ------------------------------------------------------------ 5.2 modeling approach
    A(b.find("In this research, a MILP formulation"), *added(
        para("For this reason, the MILP is solved exactly only on small instances. For the real scale problem "
             "(71 dealers, 80 trucks, 5 car types, 30 days), the routing variables X alone number more than "
             "12 million (80 × 72 × 72 × 30), so a two stage heuristic approach is adopted. In the first stage, a "
             "constructive heuristic (greedy) produces a feasible plan in a fraction of a second. In the second "
             "stage, a local search metaheuristic repeatedly modifies this plan and keeps the changes that lower the "
             "total cost. Simulated annealing and tabu search were chosen because both are well established for "
             "routing problems, both can work on the same solution representation and neighborhood, and they "
             "escape local optima in two different ways: SA accepts worse plans with a probability that decreases "
             "over time, while TS always moves to the best sampled neighbor and forbids recently made moves. Using "
             "one neighborhood and one objective for both methods makes any difference in the results a difference "
             "in the search strategy rather than in the cost accounting."),
        para("The main advantages of this approach are scalability and flexibility: the objective can contain "
             "every cost term of the mathematical model, and a plan of any size can be improved within a chosen "
             "time limit. Its disadvantage is that optimality cannot be proven, and the quality of the result "
             "depends on the parameters and on the run time. For this reason, a substantial part of Chapter 6 is "
             "devoted to parameter selection and run length."),
    ))

    # ------------------------------------------------------------ 5.3 i) objective without the tour time penalty
    closing = removed(b.by_image("image23.png"))
    A(closing, *added(b.picture(DOCS / "assets" / "objective_402.png", 6.5)))

    # ------------------------------------------------------------ 5.3 ii) greedy: only the two changed rules
    p = b.find("The algorithm takes the transportation network parameters")
    swap(p, "it iterates through each car type sequentially",
         "each available truck is assigned to the car type with the largest total unmet demand")
    p = b.find("If available trucks and critical dealers exist")
    swap(p, between(p, "If no critical dealers remain but", "to the factory."),
         "If no critical dealers remain, the truck returns to the factory, so it only carries the cars it "
         "delivers.")
    greedy_tail = b.find("As output, it gives daily satisfied demand")
    swap(greedy_tail, "Figure 1.1", "Figure A.1")

    # ------------------------------------------------------------ 5.3 iii) to vii) metaheuristics
    framework = b.figure(REPORT / "framework.png", "Figure 5.1", "Solution framework of the metaheuristics", 6.0)
    weights = b.figure(REPORT / "weights.png", "Figure 5.2", "Move weights obtained from the pilot run", 5.6)
    moves = [
        ["Move", "Description"],
        ["rebuild", "Selects a random tour with at least two dealers and puts its dealers in the order with the "
                    "lowest driving cost. The order is found exactly for tours with up to 11 dealers."],
        ["hand over", "Selects a random dealer of a random tour and moves it, with its quantity, to another random "
                      "tour of the same car type that has enough free capacity. The dealer is inserted at the "
                      "position that adds the least driving cost."],
        ["exchange", "Selects one random dealer from each of two random tours of the same car type and swaps them "
                     "with their quantities. Each dealer is inserted into the other tour at the position that adds "
                     "the least driving cost."],
        ["add car", "Increases the number of cars dropped at a random dealer of a random tour by one, if the truck "
                    "has free capacity."],
        ["remove car", "Decreases the number of cars dropped at a random dealer of a random tour by one. The dealer "
                       "stays on the tour even if its quantity becomes zero."],
        ["add dealer", "Adds a random dealer with zero cars to a random tour, at the position that adds the least "
                       "driving cost, or opens a new tour with this dealer on a truck that is free on a random day."],
        ["remove dealer", "Removes a random dealer, with its quantity, from a random tour. A tour with no dealer "
                          "left is deleted."],
    ]
    temps = [
        ["Setting", "Initial temperature", "Acceptance of an average worsening move (Δ = 6,685) at the start", "Final temperature"],
        ["SA T75", "23,237", "75%", "72.58"],
        ["SA T50", "9,644", "50%", "72.58"],
        ["SA T25", "4,822", "25%", "72.58"],
        ["SA 500", "500", "< 0.001%", "72.58"],
        ["SA 50", "50", "≈ 0%", "0.75"],
    ]
    sa_code = [
        "Input: starting plan s0, initial temperature T0,",
        "       final temperature Tf, time limit τmax",
        "s ← s0",
        "best ← s0",
        "while elapsed time τ < τmax:",
        "    T ← T0 · (Tf / T0)^(τ / τmax)",
        "    select a move m with probability proportional to its weight",
        "    s' ← apply m to s           (select again if m cannot be applied)",
        "    Δ ← f(s') − f(s)",
        "    if Δ ≤ 0 or random(0,1) < exp(−Δ / T):  s ← s'",
        "    if f(s) < f(best):  best ← s",
        "Output: best",
    ]
    ts_code = [
        "Input: starting plan s0, sample size n, tabu size θ,",
        "       time limit τmax",
        "s ← s0",
        "best ← s0",
        "tabu list ← empty",
        "while elapsed time τ < τmax:",
        "    N ← n neighbors of s, each made by a move selected with the move weights",
        "    A ← { s' in N : move(s') is not tabu  or  f(s') < f(best) }",
        "    if A is empty:  A ← N",
        "    s ← the cheapest plan in A          (even if it is worse than s)",
        "    add move(s) to the tabu list for the next θ iterations",
        "    if f(s) < f(best):  best ← s",
        "Output: best",
    ]
    A(greedy_tail, *added(
        heading("iii) Solution Representation and Evaluation", 4),
        para("The metaheuristics start from the plan of the greedy algorithm and work on its list of tours. Each tour "
             "is described by its day, truck, car type, the dealers it visits and the number of cars dropped at each "
             "dealer. Every new plan is evaluated with the objective function of the mathematical model, and the "
             "dealer storage capacity is handled with a penalty of 15,000 per extra car per day. The overall "
             "solution framework is shown in Figure 5.1."),
        *framework,
        heading("iv) Neighborhood Structure", 4),
        para("A neighbor is obtained from the current plan by one of the seven moves in Table 5.1. If the selected "
             "move cannot be applied, for example because the truck capacity would be exceeded or two tours of the "
             "same truck would overlap in time, another move is selected."),
        caption("Table 5.1", "Moves of the neighborhood", before_table=True),
        b.table(moves, [1700, 7300], align="ll"),
        para(""),
        heading("v) Move Selection Weights", 4),
        para("The seven moves are not equally useful, so instead of being selected with equal probability, each move "
             "is selected with a fixed weight. The weights are derived from the adaptive scoring scheme of Ropke and "
             "Pisinger [15], which rewards an operator according to the outcome of the solutions it produces. Every "
             "neighbor produced by a move receives a score of 33 if it is a new best plan, 9 if it improves the "
             "current plan, 13 if it is worse but accepted by the search, and 0 if it is rejected. Accepted worse "
             "solutions score higher than ordinary improvements because they are the steps that let the search leave "
             "a local optimum, while ordinary improvements are frequent."),
        para("In the original method, the weights are updated continuously during the search. In this project, the "
             "scores are collected in a single pilot run and the weights are then fixed, so that every run in "
             "Chapter 6 uses exactly the same neighborhood and the differences between runs come only from the "
             "algorithm and the starting plan. The pilot was run on the original instance from a random plan with "
             "equal weights, using SA with T25 for 400 minutes. A random start was chosen so that every move had many "
             "chances to improve the plan, and the pilot reduced the cost of the random plan from 36.6 million to "
             "3.02 million. For each move, the total score is divided by the number of neighbors it produced, so "
             "that a move is not rewarded only for being selected often. These average scores are scaled so that "
             "the smallest becomes 1, and are capped at 5. The lower bound keeps every move in the search, because "
             "some moves are rarely useful on their own but enable others, and the upper bound prevents a single "
             "move from dominating the selection. The resulting weights are shown in Figure 5.2: rebuild is selected "
             "most often (24% of the selections), exchange, hand over and add dealer follow with 17 to 18% each, and "
             "remove car stays at the lower bound (6%)."),
        *weights,
        heading("vi) Simulated Annealing", 4),
        para("Simulated annealing selects one neighbor per iteration. A neighbor that lowers the total cost is always "
             "accepted, and a neighbor that increases it by Δ is accepted with probability exp(−Δ/T), where T is the "
             "current temperature (Metropolis criterion). High temperatures let the search move freely across the "
             "solution space, and low temperatures make it behave like a descent method."),
        para("The initial and final temperatures follow the rule proposed by Duman and Duman [13]: the initial "
             "temperature is chosen so that a solution worse than a reference solution by a given deviation is "
             "accepted with probability 0.75, 0.50 or 0.25, and the final temperature so that a solution worse by "
             "one tenth of that deviation is accepted with probability 0.0001. In the paper, the deviation is a "
             "percentage of the objective value of a good reference solution. In our problem, a single move changes "
             "the monthly cost by only a small fraction of the total, so the deviation was measured directly on the "
             "move scale: on the greedy plan of the original instance, the average cost increase of a worsening "
             "neighbor is Δ = 6,685. With T = −Δ / ln p, this gives the temperatures in Table 5.2."),
        caption("Table 5.2", "Temperature settings of simulated annealing", before_table=True),
        b.table(temps, [1500, 1900, 3800, 1800]),
        para(""),
        para("In addition to the three settings of the paper, two colder starting temperatures, 500 and 50, were "
             "included as reference settings. At these temperatures an average worsening move is almost never "
             "accepted, but small worsenings still are, so they test whether the hot phase of the paper's settings is "
             "needed at all. SA 500 cools to the same final temperature as the other settings. Since 50 is already "
             "below that value, SA 50 is cooled by the same ratio as T25, from 50 to 0.75."),
        para("The paper decreases the temperature geometrically, multiplying it by a factor α after every L "
             "neighbors, with α chosen so that the final temperature is reached when the iteration budget is used "
             "up. Because our runs are limited by computation time rather than by an iteration count, the same "
             "geometric schedule is applied to the elapsed time: T(τ) = T0 · (Tf / T0)^(τ/τmax), where τ is the "
             "elapsed time and τmax the time limit. Every run therefore reaches its final temperature exactly at the "
             "end of its time limit, independently of how fast the moves are. The procedure is summarized in "
             "Figure 5.3."),
        b.algorithm(sa_code),
        caption("Figure 5.3", "Pseudocode of simulated annealing"),
        heading("vii) Tabu Search", 4),
        para("Tabu search samples a fixed number of neighbors of the current plan in every iteration, evaluates all "
             "of them, and moves to the cheapest allowed one, even if it is worse than the current plan. A neighbor "
             "is allowed if the move that produced it is not tabu. The tabu attribute is the exact move, which means "
             "its type and the tours, dealers and positions it changes. Once a move is performed, it is placed in the "
             "tabu list and cannot be repeated for the next θ iterations, where θ is the tabu size. Following Glover "
             "[14], a tabu move is still allowed if it produces a plan better than the best plan found so far "
             "(aspiration criterion), and if all sampled neighbors are tabu, the cheapest of them is taken so that "
             "the search never stops (Figure 5.4)."),
        para("The two parameters were set by our own choice and then compared in a screening experiment (Section "
             "6.1). The number of sampled neighbors was set relative to the number of dealers: 70 is roughly one "
             "neighbor per dealer of the original instance, and 35 roughly one per two dealers. For the tabu size, "
             "the values 25 and 625 were chosen from the geometric series 5, 25, 125, 625, so that a small and a very "
             "large tabu list, differing by a factor of 25, are compared."),
        b.algorithm(ts_code),
        caption("Figure 5.4", "Pseudocode of tabu search"),
    ))

    # ------------------------------------------------------------ 5.4 and 5.5 caption numbers
    swap(b.find("Figure 2.1: Console Outputs"), "Figure 2.1", "Figure 5.5")
    swap(b.find("By visualizing the daily dispatch schedules"), "Figures 2.1 to 2.4", "Figures 5.6 to 5.9")
    p = b.find("Figure 3.1")
    swap(p, "Figure 3.1", "Figure 5.6")
    swap(p, "Figure 3.2", "Figure 5.7")
    p = b.find("Figure 3.3")
    swap(p, "Figure 3.3", "Figure 5.8")
    swap(p, "Figure 3.4", "Figure 5.9")
    heur_ver = b.find("For the verification and the validation of the heuristic model")
    old = "Figure 4.1 to\n4.4" if "Figure 4.1 to\n4.4" in element_text(heur_ver) else "Figure 4.1 to 4.4"
    swap(heur_ver, old, "Figures A.2 to A.5")

    # ------------------------------------------------------------ 5.6 metaheuristics
    small = [
        ["Trucks", "Exact model (Gurobi, 600 s)", "Greedy heuristic", "Best metaheuristic (120 s)", "Gap to exact"],
        ["4", "29,742", "78,073", "51,058 (SA 500)", "71.7%"],
        ["6", "19,133", "57,340", "26,975 (SA 500)", "41.0%"],
        ["8", "18,157", "47,474", "19,437 (SA 500)", "7.0%"],
    ]
    A(heur_ver, *added(
        heading("5.6) Verification and validation of the Metaheuristics", 3),
        para("For the verification of the metaheuristics, every plan they return is evaluated again with the same "
             "objective function. In all runs of Chapter 6, no truck carries more than its capacity, no truck has "
             "overlapping tours and no dealer exceeds its storage capacity. For the validation, simulated annealing "
             "and tabu search were run for 120 seconds on the small instances of Section 6.1 (10 dealers, 7 days, 2 "
             "car types), starting from the greedy plan, and compared with the exact solutions (Table 5.3)."),
        caption("Table 5.3", "Exact model, greedy heuristic and metaheuristics on the small instances", before_table=True),
        b.table(small, [1100, 2200, 1700, 2500, 1500]),
        para(""),
        para("The metaheuristics close a large part of the gap between the greedy plan and the exact solution, from "
             "56% of it with 4 trucks to 96% with 8 trucks. With 4 trucks the remaining gap is the largest, because "
             "the fleet is too small to serve the demand on time. Note that the exact values are the best solutions "
             "found within the time limit of 600 seconds."),
    ))

    # ------------------------------------------------------------ chapter 6 intro, 6.1 i to iii
    p = b.find("In this chapter, the computational experiments")
    swap(p, "two", "three", nth(p, "two main phases", "two"))
    strike(p, "and ", nth(p, "and a sensitivity analysis", "and "))
    add_after(p, "under varying parameters",
              ", and a computational study of the metaheuristics on the real scale problem and its variants")
    add_after(p, "time limit of 600 seconds.",
              " The metaheuristics are also written in Python, and each metaheuristic run uses one processor core, "
              "so that runs executed at the same time do not share computing power.")
    swap(b.find("First, we analyzed the exact model"), "5 dealers, 10 trucks", "10 dealers, 6 trucks")
    swap(b.find("The results show a clear trade off"), "Table 1.1", "Table A.1")
    swap(b.find("In the results, for the truck numbers"), "Table 2.1", "Table A.2")
    p = b.find("To compare with the exact model, the heuristic algorithm")
    swap(p, between(p, "Similar to the exact model, shifting", "figure 1.1)."),
         "Similar to the exact model, more trucks lead to fewer backorders and a lower objective value, and no "
         "cars remain unsold. However, the objective values are still 2.6 to 3 times higher than the exact "
         "solutions, mainly because of backorders and longer routes.")
    swap(p, "Table 2.2", "Table A.3")
    heur_sens = p

    # ------------------------------------------------------------ 6.1 iv to ix
    instances = [
        ["Instance", "Dealers", "Trucks", "Demand scale", "Monthly demand (cars)", "Greedy plan cost"],
        ["Original", "71", "80", "1.0×", "4,351", "2,076,165"],
        ["Growth, same fleet", "110", "80", "2.0×", "13,482", "29,242,970"],
        ["Full map, slack", "71", "140", "0.5×", "2,170", "1,321,492"],
        ["Far, tight", "24", "30", "1.0×", "1,475", "1,609,930"],
    ]
    greedy_rows = [
        ["Setting", "Original", "Growth, same fleet", "Full map, slack", "Far, tight"],
        ["Greedy plan", "2,076,165", "29,242,970", "1,321,492", "1,609,930"],
        ["SA T75", "2,076,165 (0.0%)", "24,531,505 (16.1%)", "832,875 (37.0%)", "612,738 (61.9%)"],
        ["SA T25", "1,759,381 (15.3%)", "**20,819,112 (28.8%)**", "836,980 (36.7%)", "610,408 (62.1%)"],
        ["SA 500", "1,569,090 (24.4%)", "22,814,942 (22.0%)", "806,229 (39.0%)", "**595,680 (63.0%)**"],
        ["SA 50", "1,517,212 (26.9%)", "27,824,335 (4.9%)", "816,774 (38.2%)", "873,009 (45.8%)"],
        ["TS 70/25", "1,518,535 (26.9%)", "28,835,232 (1.4%)", "820,085 (37.9%)", "876,403 (45.6%)"],
        ["TS 70/625", "**1,502,348 (27.6%)**", "28,907,726 (1.1%)", "**802,706 (39.3%)**", "868,934 (46.0%)"],
    ]
    random_rows = [
        ["Setting", "Original", "Growth, same fleet", "Full map, slack", "Far, tight"],
        ["Random plan", "36,603,643", "96,373,536", "43,038,795", "13,882,243"],
        ["Greedy plan (reference)", "2,076,165", "29,242,970", "1,321,492", "1,609,930"],
        ["SA T75", "4,716,117 (−127.2%)", "29,614,504 (−1.3%)", "852,098 (35.5%)", "638,738 (60.3%)"],
        ["SA T25", "**3,538,143 (−70.4%)**", "**27,251,977 (6.8%)**", "**847,058 (35.9%)**", "**575,622 (64.2%)**"],
        ["SA 500", "3,950,099 (−90.3%)", "33,196,988 (−13.5%)", "856,052 (35.2%)", "591,582 (63.3%)"],
        ["SA 50", "9,164,981 (−341.4%)", "54,237,757 (−85.5%)", "2,154,854 (−63.1%)", "2,842,792 (−76.6%)"],
        ["TS 70/25", "9,922,460 (−377.9%)", "67,393,932 (−130.5%)", "1,988,372 (−50.5%)", "2,931,797 (−82.1%)"],
        ["TS 70/625", "9,613,576 (−363.0%)", "65,283,198 (−123.2%)", "1,948,865 (−47.5%)", "2,974,009 (−84.7%)"],
    ]
    ext_rows = [["Instance and start", "Setting", "Best in 400 min", "Best in 1000 min",
                 "Improvement in 400 min", "Improvement in 1000 min"]] + extended_rows()
    names = (("00_original", "Original"), ("02_growth_no_fleet", "Growth, same fleet"),
             ("07_full_slack_thin", "Full map, slack"), ("09_far_tight", "Far, tight"))
    gfig, rfig = {}, {}
    for k, (folder, title) in enumerate(names):
        gfig[k] = b.figure(CFG / "greedy" / folder / "cost_vs_time.png", f"Figure 6.{k + 1}",
                           f"Best objective over time, {title} instance, greedy start", 5.4)
        rfig[k] = b.figure(CFG / "random" / folder / "cost_vs_time.png", f"Figure 6.{k + 5}",
                           f"Best objective over time, {title} instance, random start", 5.4)

    A(heur_sens, *added(
        heading("iv) Real Scale Test Instances", 4),
        para("The real scale instance was generated from the operating averages provided by Tofaş. The Bursa plant "
             "and 71 dealer sites are placed in 45 cities across Türkiye in proportion to the actual dealer network. "
             "Road distances are approximated by great circle distances multiplied by 1.35, and transportation costs "
             "1.5 per km. A leg of up to 280 km is completed within the same day, and longer legs take one day per "
             "500 km. The daily demand of each dealer for the five car types (Egea Cross, Egea Sedan, Scudo, Scudo "
             "Van and Ulysse) is drawn from {0, 1, 2} with the company's average daily demand per dealer (0.636, "
             "0.997, 0.296, 0.061 and 0.052 respectively). A truck carries 8 Egea cars or 6 of the larger types, "
             "dealer storage capacities range from 36 to 80 cars, the fleet consists of 80 trucks and the horizon "
             "is 30 days. The cost parameters are H = 20 per car per day, B = 500 per car per day and G = 15,000 "
             "per unsold car."),
        para("To test the algorithms on more than a single instance, ten variants of this instance were designed by "
             "changing three characteristics: the size and spread of the dealer network (24 cities across the "
             "country, the 23 dealers reachable from the plant within a day, the 71 original dealers, or 110 dealers "
             "after adding 39 sites in existing cities), the fleet size (30, 80, 120 or 140 trucks) and the demand "
             "intensity (0.5, 1 or 2 times the original). The full list is given in Table A.4 in the “Appendix”. "
             "Because each metaheuristic run takes several hours, four instances were selected to keep the study "
             "within the available computation time while preserving this diversity (Table 6.1): the original "
             "instance as the baseline, a growth scenario in which the network and the demand grow but the fleet "
             "stays the same, a slack scenario with the original map, more trucks and lower demand, and a sparse "
             "national network with a tight fleet. In the preliminary runs, these instances also led to different "
             "rankings of the algorithms."),
        caption("Table 6.1", "Instances of the main study", before_table=True),
        b.table(instances, [2100, 900, 900, 1300, 1800, 1700], align="lcccrr"),
        para(""),
        heading("v) Algorithm Screening", 4),
        para("Before the main experiment, nine settings were compared on the original instance with a time limit of "
             "100 minutes, starting from the greedy plan: SA with the five temperatures of Table 5.2, and TS with 35 "
             "or 70 sampled neighbors and a tabu size of 25 or 625 (Table A.5 and Figures A.6 to A.8 in the "
             "“Appendix”). The tabu settings gave the largest improvements, followed by SA 50 and SA 500, while SA "
             "T75, T50 and T25 did not improve the greedy plan within 100 minutes."),
        para("Based on these results, six settings were kept for the main experiment. TS with 35 neighbors was "
             "removed because 70 neighbors gave better results with both tabu sizes, and SA T50 was removed because "
             "it lies between T75 and T25 and behaved the same as them in the screening."),
        heading("vi) Design of the Main Experiment", 4),
        para("The main experiment runs the six selected settings (SA T75, SA T25, SA 500, SA 50, TS 70/25 and TS "
             "70/625, where the TS label gives the sample size and the tabu size) on the four instances of Table 6.1, "
             "from two different starting plans, with a time limit of 400 minutes per run. In total, 48 runs were "
             "executed in parallel on separate cores. The first starting plan is the greedy plan, which shows how "
             "much the metaheuristics add to the constructive heuristic. The second is a random plan, built by "
             "drawing tours with random days, trucks, car types, dealers and quantities while respecting truck "
             "capacities and truck availability. It shows whether an algorithm can find good plans without a good "
             "starting point. The random plans are very expensive (between 13.9 and 96.4 million) because they "
             "ignore demand. The move weights of Figure 5.2 and the temperatures of Table 5.2 are the same in every "
             "run."),
        heading("vii) Results from the Greedy Start", 4),
        para("Table 6.2 summarizes the best plan found by each setting from the greedy start. Figures 6.1 to 6.4 show "
             "how the best objective value changes over time, and the current objective plots are given in Figures "
             "A.9 to A.16 in the “Appendix”."),
        caption("Table 6.2", "Best objective in 400 minutes from the greedy start (improvement over the greedy plan)",
                before_table=True),
        b.table(greedy_rows, [1500, 1900, 2100, 1800, 1800], align="lrrrr", size=19),
        para(""),
        para("Figure 6.1 shows the results on the **original instance**, where the greedy plan costs 2,076,165.", keep_next=True),
        *gfig[0],
        para("TS 70/625 found the best plan (1,502,348, 27.6% below the greedy plan), followed closely by SA 50 and "
             "TS 70/25 (26.9% each). The tabu settings and SA 50 improve the plan quickly in the first minutes and "
             "then continue slowly, and SA 500 improves gradually and reaches 24.4%. The hot settings behave "
             "differently. SA T25 only starts to improve after 338 minutes, once its temperature has decreased "
             "enough, and is still improving quickly at the end of the run (15.3%), while SA T75 does not improve "
             "the greedy plan within 400 minutes."),
        para("Figure 6.2 shows the **growth scenario**. Here the fleet is far too small for twice the demand on a "
             "larger network, and backorders make up 27.5 million of the 29.2 million cost of the greedy plan.", keep_next=True),
        *gfig[1],
        para("In this case the ranking is reversed. SA T25 is the best setting (20,819,112, 28.8%), followed by SA "
             "500 (22.0%) and SA T75 (16.1%), while SA 50 (4.9%) and both tabu settings (1.1 to 1.4%) barely improve "
             "the plan. The hot SA settings accept many worse plans and in this way rebuild the tours so that more "
             "cars are delivered (11,514 instead of 9,842 for SA T25), with a higher transportation cost. All three "
             "are still decreasing steeply at 400 minutes."),
        para("Figure 6.3 shows the **full map, slack** instance. The fleet is large and the demand is low, so the "
             "greedy plan has no backorders and its cost is only transportation.", keep_next=True),
        *gfig[2],
        para("All six settings reduce the cost by 37 to 39%, and TS 70/625 (802,706) and SA 500 (806,229) are almost "
             "equal. The SA plans drive much less but keep more cars in stock at the dealers, while the tabu and SA "
             "50 plans drive more and hold less. Different balances between transportation and holding lead to "
             "similar total costs."),
        para("Figure 6.4 shows the **far, tight** instance, a sparse national network with a small fleet, where the "
             "greedy plan has both long routes and backorders.", keep_next=True),
        *gfig[3],
        para("The three hotter SA settings reduce the cost by about 62 to 63%, and SA 500 is the best (595,680, "
             "63.0%). TS and SA 50 stop at about 46%, with about five times as many backorders as the SA plans. "
             "Here, accepting worse plans is needed to rearrange the tours of the few available trucks."),
        heading("viii) Results from the Random Start", 4),
        para("From a random plan, every setting reduces the cost by a large amount (Table 6.3), but the important "
             "comparison is with the greedy plan. Figures 6.5 to 6.8 show the best objective over time, and the "
             "current objective plots are given in Figures A.17 to A.24 in the “Appendix”."),
        caption("Table 6.3", "Best objective in 400 minutes from the random start (improvement over the greedy plan, "
                             "negative values are above it)", before_table=True),
        b.table(random_rows, [1500, 2000, 2000, 1800, 1800], align="lrrrr", size=19),
        para(""),
        para("Figure 6.5 shows the **original instance** from the random start.", keep_next=True),
        *rfig[0],
        para("No setting reaches the greedy plan within 400 minutes. SA T25 ends at 3.54 million, 70% above the "
             "greedy plan, and SA T75 and SA T25 are still decreasing at the end of the run. TS and SA 50 stop much "
             "earlier, at more than four times the cost of the greedy plan."),
        para("Figure 6.6 shows the **growth instance** from the random start.", keep_next=True),
        *rfig[1],
        para("SA T25 ends 6.8% below the greedy plan, even though it started from a plan more than three times as "
             "expensive. SA T75 ends just above the greedy plan."),
        para("Figure 6.7 shows the **full map, slack** instance from the random start.", keep_next=True),
        *rfig[2],
        para("The three hotter SA settings end about 35 to 36% below the greedy plan, close to their results from "
             "the greedy start."),
        para("Figure 6.8 shows the **far, tight** instance from the random start.", keep_next=True),
        *rfig[3],
        para("SA T25 ends 64.2% below the greedy plan and finds the best plan for this instance (575,622), better "
             "than any result from the greedy start. Overall, SA T25 is the best setting from the random start on "
             "all four instances, while TS and SA 50, which perform well from the greedy plan, stay far above the "
             "greedy plan on every instance."),
        heading("ix) Extended 1000 Minute Runs", 4),
        para("Some settings were still improving at the end of the 400 minutes, especially the hot SA settings on "
             "the original and growth instances. Therefore, seven SA runs were extended to 1000 minutes. In its first "
             "400 minutes, an extended run follows the same cooling schedule as the 400 minute run, and after that "
             "the temperature keeps decreasing at the same geometric rate. Table 6.4 compares the results, and the "
             "plots are given in Figures A.25 to A.34 in the “Appendix”."),
        caption("Table 6.4", "Results of the 1000 minute runs (improvement over the greedy plan)", before_table=True),
        b.table(ext_rows, [1900, 900, 1550, 1550, 1500, 1500], align="llrrrr", size=19),
        para(""),
        para("With the additional time, SA 500 finds the best plan for the original instance (1,437,905, 30.7% below "
             "the greedy plan), and SA T25 reaches 28.7%. Both are better than the best 400 minute result of TS "
             "70/625 (27.6%). On the growth instance, SA T25 reduces the cost of the greedy plan by 41.3% instead "
             "of 28.8%. From the random start, SA T75 improves by a further 17% after minute 400 but still ends 93% "
             "above the greedy plan. Most of the extra improvement is reached by minute 800, and the runs change "
             "little in the last 200 minutes."),
    ))

    # ------------------------------------------------------------ 6.2 discussion
    p = b.find("As seen in the heuristic solution")
    strike(p, between(p, " The main reason behind this", "pilot run results)."))
    old_block = block_until(b.find("Another problem is that this algorithm"),
                            lambda e: "6.3)" in element_text(e))
    closing = removed(old_block[0], old_block[-1])
    A(closing, *added(
        para("To improve these solutions, the greedy plan is used as the starting point of simulated annealing and "
             "tabu search. Within 400 minutes, the metaheuristics reduce the cost of the greedy plan by 27.6% to "
             "63.0%, depending on the instance."),
        para("No single setting is the best on every instance, and the results depend on how tight the fleet is "
             "compared to the demand. When the fleet is sufficient, as in the original and full slack instances, the "
             "greedy plan already has a good structure, and the methods that accept few or no worse plans, TS and "
             "the cold SA settings, improve it quickly. When the fleet is tight, as in the growth and far tight "
             "instances, backorders dominate the cost and many tours have to be rearranged. Only the SA settings "
             "that accept worse plans can do this, and their advantage grows with time."),
        para("The hot SA settings are limited mainly by time. Their temperature has to fall before they start "
             "improving, which happened late or not at all within 400 minutes on the original instance. In the "
             "extended runs, SA 500 gave the best plan for the original instance and SA T25 reduced the cost of the "
             "growth scenario by 41.3%."),
        para("The random start experiments show how much the methods depend on the starting plan. SA T25 is the most "
             "robust setting, and it ends below the greedy plan on three of the four instances. TS and SA 50 depend "
             "strongly on a good starting plan, so the greedy plan is the recommended starting point."),
        para("For TS, the tabu size has little effect. TS 70/25 and TS 70/625 differ by at most 2.2% on every "
             "instance, because the tabu attribute is the exact move and the same move is rarely sampled again in "
             "such a large neighborhood. At most 2% of the sampled neighbors were tabu in the main study."),
        para("Based on these results, we recommend building the greedy plan first and then running TS with 70 "
             "neighbors when the computation time is short or the fleet is sufficient, and SA starting from T25 or "
             "500 when a longer run is possible, especially when the fleet is tight. Since every setting runs on a "
             "single core, several settings can be run at the same time and the best plan can be kept."),
    ))

    # ------------------------------------------------------------ 6.3 pilot implementation
    p = b.find("For pilot implementation, we run the heuristic")
    swap(p, "72 dealers", "71 dealers")
    swap(p, "1000 trucks", "80 trucks")
    console_old = b.by_image("image8.png")
    closing = removed(console_old, b.find("Figure 5.1: Console Outputs of the Heuristic Pilot Run"))
    A(closing, *added(*b.figure(REPORT / "pilot_end_of_month.png", "Figure 6.9",
                                "Console Outputs of the Heuristic Pilot Run", 4.3)))
    days_old = b.by_image("image7.png")
    closing = removed(days_old)
    A(closing, *added(
        *b.figure(REPORT / "pilot_day_04.png", "Figure 6.10", "Day 4 Outputs of Pilot Implementation", 4.6),
        *b.figure(REPORT / "pilot_day_29.png", "Figure 6.11", "Day 29 Outputs of Pilot Implementation", 4.6),
    ))
    p = b.find("In this pilot implementation, problems stated")
    swap(p, between(p, "problems stated beforeward", "car type deliveries."),
         "the outputs above show that the heuristic gives a plan for the real scale problem in less than a second. "
         "In Figure 6.9, no cars remain unsold at the end of the month, and only 45 of the 4,351 demanded cars are "
         "not delivered. Figure 6.10 shows a day on which the demand of every car type is fully met. On some days, "
         "such as day 29 in Figure 6.11, fewer cars are delivered because many trucks are still on their tours, and "
         "these cars become backorders.")
    pilot_rows = [
        ["Cost component", "Greedy plan", "TS 70/625 (400 min)", "SA 500 (1000 min)"],
        ["Transportation", "1,902,665", "1,418,288", "1,169,445"],
        ["Holding", "0", "61,560", "241,460"],
        ["Backorder", "173,500", "22,500", "27,000"],
        ["Unsold cars", "0", "0", "0"],
        ["**Total**", "**2,076,165**", "**1,502,348**", "**1,437,905**"],
        ["Cars delivered (of 4,351)", "4,306", "4,341", "4,341"],
        ["Backordered car days", "347", "45", "54"],
    ]
    A(p, *added(
        para("The best plans of the metaheuristics on the same instance are compared with the greedy plan in "
             "Table 6.5."),
        caption("Table 6.5", "Cost components of the greedy plan and the best plans on the original instance",
                before_table=True),
        b.table(pilot_rows, [3000, 1900, 2100, 2000], align="lrrr"),
        para(""),
        para("Both plans cut the backorder cost by more than 80% and drive much less. The transportation cost, which "
             "depends on the distance driven, falls by 25.5% with TS and by 38.5% with SA. The SA plan delivers cars "
             "earlier and keeps more stock at the dealers, but it needs much less driving. In both plans, no car "
             "remains unsold and no dealer exceeds its storage capacity."),
    ))

    # ------------------------------------------------------------ chapter 7
    swap(b.find("The implementation also requires appropriate software"), " but this model lacks accuracy",
         ", and the metaheuristics improve these solutions considerably when more computation time is given")
    closing = removed(b.find("Despite this study including a full scale implementation"))
    A(closing, *added(para(
        "Before the plans are used in practice, they should be tested with the actual data of Tofaş, since the real "
        "scale problem in this study was generated from company averages.")))

    # ------------------------------------------------------------ chapter 8
    append(b.find("In this study, we examined the inventory distribution system"),
           " To improve the heuristic solutions, simulated annealing and tabu search were developed, with parameters "
           "based on the literature [13], [14], [15]. The results show that the metaheuristics reduce the cost of the "
           "heuristic solutions considerably, by 30.7% on the real scale problem. Simulated annealing works better "
           "when the fleet is tight, while tabu search gives good results quickly when the fleet is sufficient.")
    p = b.find("The current work is limited to model development")
    swap(p, "exact & heuristic solutions", "exact, heuristic & metaheuristic solutions")
    swap(p, between(p, "In future work, the current heuristic", "(like genetic algorithm)."),
         "In future work, the plans can be tested with the actual data of Tofaş, the runs can be repeated with "
         "different random seeds, and the move weights can be updated during the search instead of being fixed "
         "after a pilot run. Longer runs and combinations of the two metaheuristics can also improve the results "
         "further.")

    # ------------------------------------------------------------ references
    removed(b.find("In this part, there should be a list of references"))
    A(b.find("[12]"), *added(
        para("[13] T. Duman and E. Duman, “Solving a new application of asymmetric TSP by modified migrating birds "
             "optimization algorithm,” *Evolutionary Intelligence*, vol. 17, no. 3, 2024.", after=240, before=240),
        para("[14] F. Glover, “Tabu search, Part I,” *ORSA Journal on Computing*, vol. 1, no. 3, pp. 190 to 206, "
             "1989.", after=240, before=240),
        para("[15] S. Ropke and D. Pisinger, “An adaptive large neighborhood search heuristic for the pickup and "
             "delivery problem with time windows,” *Transportation Science*, vol. 40, no. 4, pp. 455 to 472, 2006.",
             after=240, before=240),
    ))

    # ------------------------------------------------------------ appendix
    greedy_code = [
        "for period t (Days) in T:",
        "    for truck k in K:",
        "        if truck k is locked on day t: continue",
        "        for car type c in C:",
        "            unmet demand of c = sum over dealers i of",
        "                max(0, demand of dealer i at day t - inventory of dealer i)",
        "        c* = car type with the largest unmet demand",
        "        if unmet demand of c* = 0: continue",
        "        current = factory (N = 0)",
        "        available_capacity of truck k = capacity (given that car type is c*)",
        "        route = [0]",
        "        route_time = 0",
        "        while available_capacity of truck k > 0:",
        "            critical dealers = dealers not on route whose inventory of c*",
        "                               is below their demand at day t",
        "            if there is no critical dealer: break",
        "            j = critical dealer closest to current",
        "            route_time += time_matrix[current][j]",
        "            cars dropped off at node j = min(available_capacity of truck k,",
        "                                             cars needed for dealer j)",
        "            available_capacity of truck k -= cars dropped off at node j",
        "            current = j",
        "            route.append(j)",
        "        if route has a dealer:",
        "            route.append(0) & route_time += time_matrix[current][0]",
        "            lock truck k for route_time days",
        "    inventory of every dealer -= demand of the day",
    ]
    closing = removed(b.by_image("image12.jpg"), b.find("Figure 1.1: Console Outputs"))
    A(closing, *added(b.algorithm(greedy_code), caption("Figure A.1", "Pseudocode of the Greedy Heuristic")))

    old_plots = block_until(b.by_image("image25.png"), lambda e: element_text(e).startswith("Table 1.1"))
    closing = removed(old_plots[0], old_plots[-1])
    days = b.grid([
        (PLOTS / "heuristic_dynamic" / "day_00.png", "Figure A.2", "Routing Schedule on Day 0"),
        (PLOTS / "heuristic_dynamic" / "day_01.png", "Figure A.3", "Routing Schedule on Day 1"),
        (PLOTS / "heuristic_dynamic" / "day_02.png", "Figure A.4", "Routing Schedule on Day 2"),
        (PLOTS / "heuristic_dynamic" / "day_03.png", "Figure A.5", "Routing Schedule on Day 3"),
    ])
    A(closing, *added(days, para("")))
    swap(b.find("Table 1.1:"), "Table 1.1", "Table A.1")
    swap(b.find("Table 2.1:"), "Table 2.1", "Table A.2")

    heur_caption = b.find("Table: Truck Numbers and Objective Value, Heuristic")
    swap(heur_caption, "Table:", "Table A.3:")
    heur_table = None
    for t in d.tables:
        if t.cell(0, 0).text.strip().startswith("# of") and "Time" in t.cell(0, t._column_count - 1).text:
            heur_table = t
    new_values = [("78073.19", "16073.19", "62000.00", "0", "0.0001"),
                  ("57339.80", "28339.80", "29000.00", "0", "0.0001"),
                  ("47473.94", "33973.94", "13500.00", "0", "0.0001")]
    for r, row in enumerate(new_values, start=1):
        for c, value in enumerate(row, start=1):
            cell_p = heur_table.cell(r, c).paragraphs[0]._p
            old_value = element_text(cell_p).strip()
            if old_value != value:
                swap(cell_p, old_value, value)

    pool = [
        ["#", "Instance", "Dealer network", "Trucks", "Demand scale"],
        ["00", "Original", "71 dealers (original)", "80", "1.0×"],
        ["01", "Scaled firm", "110 dealers", "120", "1.0×"],
        ["02", "Growth, same fleet", "110 dealers", "80", "2.0×"],
        ["03", "Large map, tight fleet", "110 dealers", "30", "1.0×"],
        ["04", "Near region, congested", "23 dealers reachable in a day", "30", "2.0×"],
        ["05", "Sparse country, slack fleet", "24 cities across the country", "140", "0.5×"],
        ["06", "Same map, loaded", "71 dealers (original)", "30", "2.0×"],
        ["07", "Full map, slack", "71 dealers (original)", "140", "0.5×"],
        ["08", "Near region, thin demand", "23 dealers reachable in a day", "80", "0.5×"],
        ["09", "Far, tight", "24 cities across the country", "30", "1.0×"],
    ]
    screen = [
        ["Setting", "Best objective", "Improvement over the greedy plan"],
        ["TS 70/25", "1,576,074", "24.09%"],
        ["TS 70/625", "1,577,180", "24.03%"],
        ["TS 35/625", "1,589,912", "23.42%"],
        ["TS 35/25", "1,592,910", "23.28%"],
        ["SA 50", "1,622,358", "21.86%"],
        ["SA 500", "1,776,258", "14.45%"],
        ["SA T75", "2,076,165", "0.00%"],
        ["SA T50", "2,076,165", "0.00%"],
        ["SA T25", "2,076,165", "0.00%"],
    ]
    screen_dir = CFG / "greedy" / "00_original_100mins"
    appendix = [
        para(""),
        label_line("Test Instances"),
        caption("Table A.4", "The ten instance variants designed for the study (the instances of the main study are "
                             "00, 02, 07 and 09)", before_table=True),
        b.table(pool, [600, 3000, 3100, 1000, 1300], align="clccc"),
        para(""),
        label_line("Algorithm Screening (100 minutes, original instance, greedy start)"),
        caption("Table A.5", "Screening results of the nine settings (TS: sample size / tabu size)", before_table=True),
        b.table(screen, [2600, 2600, 3400], align="lrr"),
        para(""),
        *b.figure(screen_dir / "cost_vs_time.png", "Figure A.6", "Screening, best objective over time"),
        *b.figure(screen_dir / "curr_cost_vs_time.png", "Figure A.7", "Screening, current objective over time"),
        *b.figure(screen_dir / "curr_cost_vs_time_rescaled.png", "Figure A.8",
                  "Screening, current objective over time (rescaled)"),
        label_line("Current Objective Plots of the 400 Minute Study"),
    ]
    n = 9
    for start, word in (("greedy", "greedy start"), ("random", "random start")):
        for folder, title in names:
            appendix += b.figure(CFG / start / folder / "curr_cost_vs_time.png", f"Figure A.{n}",
                                 f"Current objective over time, {title} instance, {word}")
            appendix += b.figure(CFG / start / folder / "curr_cost_vs_time_rescaled.png", f"Figure A.{n + 1}",
                                 f"Current objective over time (rescaled), {title} instance, {word}")
            n += 2
    appendix.append(label_line("Extended 1000 Minute Runs"))
    groups = ((EXT / "greedy" / "00_original", "Original instance, greedy start", ("sa_T75", "sa_T25", "sa_500")),
              (EXT / "random" / "00_original", "Original instance, random start", ("sa_T75",)),
              (EXT / "greedy" / "02_growth_no_fleet", "Growth, same fleet instance, greedy start",
               ("sa_T75", "sa_T25", "sa_500")))
    labels = {"sa_T75": "SA T75", "sa_T25": "SA T25", "sa_500": "SA 500"}
    for folder, title, _keys in groups:
        appendix += b.figure(folder / "cost_vs_time.png", f"Figure A.{n}",
                             f"Best objective over time, {title} (extended settings 1000 minutes, others 400 minutes)")
        n += 1
    for folder, title, keys in groups:
        for key in keys:
            appendix += b.figure(folder / "convergence" / f"{key}.png", f"Figure A.{n}",
                                 f"Current and best objective of the 1000 minute {labels[key]} run, {title}")
            n += 1
    A(heur_table._tbl, *added(*appendix))

    d.save(str(OUT))
    work.unlink()
    lint(docx.Document(str(OUT)))
    print("saved", OUT)


if __name__ == "__main__":
    main()
