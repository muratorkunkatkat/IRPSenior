"""Builds docs/IE 402 v2.docx as a marked-up copy of the submitted IE 401 report.

Everything added for IE 402 is highlighted green. Everything taken out of IE 401
stays in the document, wrapped in {curly braces} and highlighted red. Pictures
are marked with a green or red band (paragraph shading) because Word does not
highlight images. Yellow marks open points (pending results, decisions).

    python docs/build_ie402_v2.py
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import docx
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ie402 import Builder, caption, fmt, heading, label_line, para  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PLOTS = ROOT / "plots"
SRC = DOCS / "IE 401.docx"
OUT = DOCS / "IE 402 v2.docx"
MERGE = Path(
    "/home/orkun/.config/Claude/local-agent-mode-sessions/skills-plugin/b6982539-c4c4-45f9-9af5-b6a8be32382d/"
    "8742d78a-80bd-4a7a-a2f7-b823a7bbe0bb/skills/docx/scripts/merge_runs.py"
)

GREEN, RED = "green", "red"
FINAL = False  # True: build the clean document (removals deleted, additions not marked)
FILL = {GREEN: "00FF00", RED: "FF0000"}

RPR_ORDER = ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike", "outline",
             "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish", "webHidden", "color", "spacing",
             "w", "kern", "position", "sz", "szCs", "highlight", "u", "effect", "bdr", "shd", "fitText",
             "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath"]
PPR_ORDER = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
             "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
             "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
             "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc",
             "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr",
             "sectPr", "pPrChange"]


# ---------------------------------------------------------------- low-level marking

def _local(element):
    return element.tag.split("}")[-1]


def _ordered_insert(parent, child, order):
    rank = order.index(_local(child))
    for existing in parent:
        name = _local(existing)
        if name in order and order.index(name) > rank:
            existing.addprevious(child)
            return
    parent.append(child)


def _rpr(run):
    rpr = run.find(qn("w:rPr"))
    if rpr is None:
        rpr = parse_xml(f'<w:rPr {nsdecls("w")}/>')
        run.insert(0, rpr)
    return rpr


def set_highlight(run, color, keep_yellow=False):
    rpr = _rpr(run)
    old = rpr.find(qn("w:highlight"))
    if old is not None:
        if keep_yellow and old.get(qn("w:val")) == "yellow":
            return
        rpr.remove(old)
    _ordered_insert(rpr, parse_xml(f'<w:highlight {nsdecls("w")} w:val="{color}"/>'), RPR_ORDER)


def shade_paragraph(p, color):
    ppr = p.find(qn("w:pPr"))
    if ppr is None:
        ppr = parse_xml(f'<w:pPr {nsdecls("w")}/>')
        p.insert(0, ppr)
    old = ppr.find(qn("w:shd"))
    if old is not None:
        ppr.remove(old)
    _ordered_insert(ppr, parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{FILL[color]}"/>'),
                    PPR_ORDER)


def paint(element, color, keep_yellow=False):
    """Highlight every run of an element; give paragraphs with pictures a coloured band."""
    for run in element.iter(qn("w:r")):
        if run.find(qn("w:t")) is not None:
            set_highlight(run, color, keep_yellow)
    paragraphs = [element] if element.tag == qn("w:p") else []
    paragraphs += list(element.iter(qn("w:p")))
    for p in paragraphs:
        if p.find(".//" + qn("w:drawing")) is not None:
            shade_paragraph(p, color)


def text_of(run):
    return "".join(t.text or "" for t in run.iter(qn("w:t")))


def make_run(text, like, color):
    run = copy.deepcopy(like)
    for child in list(run):
        if child.tag != qn("w:rPr"):
            run.remove(child)
    t = parse_xml(f'<w:t {nsdecls("w")} xml:space="preserve"/>')
    t.text = text
    run.append(t)
    if FINAL:
        old = run.find(f"{qn('w:rPr')}/{qn('w:highlight')}")
        if old is not None:
            old.getparent().remove(old)
    else:
        set_highlight(run, color)
    return run


def first_text_run(element):
    for run in element.iter(qn("w:r")):
        if run.find(qn("w:t")) is not None:
            return run
    return parse_xml(f'<w:r {nsdecls("w")}><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
                     f'w:cs="Times New Roman" w:eastAsia="Times New Roman"/><w:sz w:val="24"/><w:szCs w:val="24"/>'
                     f'</w:rPr></w:r>')


def last_text_run(element):
    runs = [run for run in element.iter(qn("w:r")) if run.find(qn("w:t")) is not None]
    return runs[-1] if runs else first_text_run(element)


def brace_paragraph(char):
    return parse_xml(
        f'<w:p {nsdecls("w")}><w:pPr><w:spacing w:after="0" w:before="0"/></w:pPr><w:r><w:rPr>'
        f'<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman" '
        f'w:eastAsia="Times New Roman"/><w:b w:val="1"/><w:sz w:val="24"/><w:szCs w:val="24"/>'
        f'<w:highlight w:val="red"/></w:rPr><w:t>{char}</w:t></w:r></w:p>'
    )


# ---------------------------------------------------------------- marking operations

def added(*elements):
    """Mark freshly built elements (paragraphs, tables, figures) as IE 402 additions."""
    if FINAL:
        flat = []
        for element in elements:
            flat += list(element) if isinstance(element, (list, tuple)) else [element]
        return flat
    out = []
    for element in elements:
        if isinstance(element, (list, tuple)):
            out += added(*element)
            continue
        paint(element, GREEN, keep_yellow=True)
        out.append(element)
    return out


def removed(first, last=None):
    """Mark a run of sibling elements of IE 401 as removed: red, inside { }."""
    block = [first]
    while last is not None and block[-1] is not last:
        nxt = block[-1].getnext()
        if nxt is None:
            raise SystemExit("removed(): last element not reached")
        block.append(nxt)
    if FINAL:
        anchor = block[0].getprevious()
        for element in block:
            element.getparent().remove(element)
        return anchor
    for element in block:
        paint(element, RED)
    head, tail = block[0], block[-1]
    if head.tag == qn("w:p"):
        like = first_text_run(head)
        ppr = head.find(qn("w:pPr"))
        brace = make_run("{", like, RED)
        if ppr is not None:
            ppr.addnext(brace)
        else:
            head.insert(0, brace)
    else:
        head.addprevious(brace_paragraph("{"))
    if tail.tag == qn("w:p"):
        tail.append(make_run("}", last_text_run(tail), RED))
        return tail
    closing = brace_paragraph("}")
    tail.addnext(closing)
    return closing


def _split(run, offset):
    t = run.find(qn("w:t"))
    text = t.text or ""
    twin = copy.deepcopy(run)
    t.text = text[:offset]
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    tt = twin.find(qn("w:t"))
    tt.text = text[offset:]
    tt.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    run.addnext(twin)
    return twin


def isolate(p, text, occurrence=0):
    """Split runs so that `text` is covered by whole runs; return those runs."""
    def spans():
        out, pos = [], 0
        for run in p.iter(qn("w:r")):
            if run.find(qn("w:t")) is None:
                continue
            if len(run.findall(qn("w:t"))) != 1:
                raise SystemExit(f"run with several text nodes near {text!r}")
            value = text_of(run)
            out.append((run, pos, pos + len(value)))
            pos += len(value)
        return out

    full = "".join(text_of(run) for run, _a, _b in spans())
    start = -1
    for _ in range(occurrence + 1):
        start = full.find(text, start + 1)
        if start < 0:
            raise SystemExit(f"text not found: {text!r}")
    end = start + len(text)
    for boundary in (end, start):
        for run, a, b in spans():
            if a < boundary < b:
                _split(run, boundary - a)
                break
    return [run for run, a, b in spans() if a >= start and b <= end and b > a]


def strike(p, text, occurrence=0):
    runs = isolate(p, text, occurrence)
    if FINAL:
        for run in runs:
            run.getparent().remove(run)
        return None
    for run in runs:
        set_highlight(run, RED)
    runs[0].addprevious(make_run("{", runs[0], RED))
    closing = make_run("}", runs[-1], RED)
    runs[-1].addnext(closing)
    return closing


def swap(p, old, new, occurrence=0):
    if FINAL:
        runs = isolate(p, old, occurrence)
        runs[0].addprevious(make_run(new, runs[0], GREEN))
        for run in runs:
            run.getparent().remove(run)
        return
    closing = strike(p, old, occurrence)
    closing.addnext(make_run(new, closing, GREEN))


def add_before(p, anchor_text, new):
    runs = isolate(p, anchor_text)
    runs[0].addprevious(make_run(new, runs[0], GREEN))


def add_after(p, anchor_text, new):
    runs = isolate(p, anchor_text)
    runs[-1].addnext(make_run(new, runs[-1], GREEN))


def append(p, new):
    p.append(make_run(new, last_text_run(p), GREEN))


def block_until(start, stop_test):
    """start and the following siblings up to (not including) the first one for which stop_test holds."""
    block = [start]
    nxt = start.getnext()
    while nxt is not None and not stop_test(nxt):
        block.append(nxt)
        nxt = nxt.getnext()
    while len(block) > 1 and block[-1].tag == qn("w:p") and not "".join(
            t.text or "" for t in block[-1].iter(qn("w:t"))).strip() and block[-1].find(".//" + qn("w:drawing")) is None:
        block.pop()
    return block


def element_text(element):
    return "".join(t.text or "" for t in element.iter(qn("w:t")))


# ---------------------------------------------------------------- results of the 1000-minute runs

EXT_JOBS = (
    ("greedy", "00_original", "sa_T75", "Original, greedy", "SA T75"),
    ("greedy", "00_original", "sa_T25", "Original, greedy", "SA T25"),
    ("greedy", "00_original", "sa_500", "Original, greedy", "SA 500"),
    ("random", "00_original", "sa_T75", "Original, random", "SA T75"),
    ("greedy", "02_growth_no_fleet", "sa_T75", "Growth, greedy", "SA T75"),
    ("greedy", "02_growth_no_fleet", "sa_T25", "Growth, greedy", "SA T25"),
    ("greedy", "02_growth_no_fleet", "sa_500", "Growth, greedy", "SA 500"),
)
PENDING = "[[pending]]"


def load_extended():
    rows = []
    for start, config, key, title, label in EXT_JOBS:
        stem = f"{start}__{config}__{key}.json"
        base = json.loads((PLOTS / "analysis" / "_cache" / stem).read_text())
        path = PLOTS / "analysis" / "extended" / "_cache" / stem
        ext = json.loads(path.read_text()) if path.exists() else {}
        if "cool_seconds" not in ext:
            ext = None
        greedy = base["greedy_total"]
        imp400 = 100 * (greedy - base["best"]) / greedy
        if ext is None:
            rows.append([title, label, fmt(base["best"]), PENDING, f"{imp400:.1f}% → {PENDING}"])
        else:
            imp1000 = 100 * (greedy - ext["best"]) / greedy
            rows.append([title, label, fmt(base["best"]), fmt(ext["best"]), f"{imp400:.1f}% → {imp1000:.1f}%"])
    ready = all(PENDING not in row[3] for row in rows)
    return rows, ready


# ---------------------------------------------------------------- document

def main():
    work = DOCS / "_ie401_merged_v2.docx"
    done = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(MERGE), str(SRC), "-o", str(work)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"merge_runs failed: {done.stderr or done.stdout}")
    d = docx.Document(str(work))
    b = Builder(d)
    A = Builder.after

    ext_rows, ext_ready = load_extended()

    # ------------------------------------------------------------ cover, abstract
    swap(b.find("SPRING 2026"), "SPRING 2026", "FALL 2026")
    A(b.find("This project focuses on optimizing"), *added(para(
        "In the second part of the project (IE 402), a dynamic greedy heuristic builds a feasible plan for the "
        "company-scale instance (71 dealers, 80 trucks, 5 vehicle types, 30 days) in less than a second, and two "
        "metaheuristics, simulated annealing and tabu search, improve this plan through a shared neighbourhood of "
        "seven moves. In a 400-minute computational study on four instances, the metaheuristics reduce the total cost "
        "of the greedy plan by 27.6% to 63.0%, and extended 1000-minute runs [[result pending]]. The study provides a "
        "systematic and data driven tool for distribution planning that lowers both cost and driving distance.")))

    # ------------------------------------------------------------ chapter 1
    p = b.find("To achieve these objectives")
    append(p, ". Because the exact model cannot be solved at the company scale, a dynamic greedy heuristic constructs "
              "a feasible 30-day distribution plan within seconds, and two metaheuristics, simulated annealing (SA) "
              "and tabu search (TS), improve that plan by searching a common neighbourhood of seven moves.")
    A(b.find("The mathematical model serves as a first step"), *added(para(
        "The first part of the project (IE 401) established the mathematical model and the constructive heuristic. "
        "This report extends that work with the metaheuristic layer and a computational study on the company-scale "
        "instance and three variants of it. Starting from the greedy plan, the best metaheuristic lowers the total "
        "monthly cost of the original 71-dealer instance by 27.6% within 400 minutes [[and by … within 1000 "
        "minutes, pending]]. Across the four instances, the improvement over the greedy plan ranges from 27.6% to "
        "63.0% in 400 minutes, and the best algorithm depends on how tight the fleet is relative to demand.")))
    p = b.find("This report is divided into seven chapters")
    swap(p, "seven", "eight")
    add_after(p, "Chapter 5 explains the solution approach", ": the literature review, the mathematical model, the "
                                                            "dynamic greedy heuristic and the two metaheuristics")
    add_after(p, "Chapter 6 presents the results.", " Chapter 7 discusses the implementation, and Chapter 8 "
                                                    "concludes the study.")
    strike(p, " Chapter 7 concludes the study.")

    # ------------------------------------------------------------ chapter 4
    strike(b.find("The aim of the model is to minimize"),
           " Additionally, a small penalty is applied for tour time to encourage more efficient route planning "
           "decisions.")
    append(b.find("The performance of the system is evaluated"),
           " In this project, a dynamic greedy heuristic and two metaheuristics, simulated annealing and tabu "
           "search, are developed for this purpose.")
    for t in d.tables:
        for row in t.rows:
            if row.cells[0].text.strip().startswith("IE 366"):
                append(row.cells[1].paragraphs[-1]._p, "; greedy construction, simulated annealing and tabu search")

    # ------------------------------------------------------------ 5.1 iv) literature
    A(b.find("Sustainability has become a key concern"), *added(
        heading("iv) Metaheuristics and Their Parameters", 4),
        para("Because exact methods can only solve small IRP instances, metaheuristics that improve a feasible "
             "solution through small changes are widely used for large instances [2]. Simulated annealing (SA) "
             "accepts a worse neighbouring solution with a probability that decreases as a temperature parameter is "
             "lowered, which lets the search leave local optima. Its performance depends strongly on the initial and "
             "final temperatures. Duman and Duman [13] set both through acceptance probabilities instead of fixed "
             "values: the initial temperature is chosen so that a solution 10% worse than a reference solution is "
             "accepted with a probability of 0.75, 0.50 or 0.25, and the final temperature so that a solution 1% "
             "worse is accepted with a probability of 0.0001, and the temperature is reduced geometrically between "
             "the two. This makes the temperatures independent of the scale of the objective function."),
        para("Tabu search (TS), introduced by Glover [14], moves in every iteration to the best solution in a "
             "neighbourhood of the current solution, even if it is worse, and prevents the search from cycling by "
             "keeping recently performed moves in a tabu list. The tabu size determines how long a move stays in this "
             "list, and a tabu move can still be accepted if it leads to a solution better than the best one found "
             "so far (aspiration criterion). When the neighbourhood is very large, only a sample of neighbours is "
             "evaluated in each iteration, so the sample size and the tabu size are the two main parameters of the "
             "method."),
        para("When a local search uses several types of moves, the choice of move affects its efficiency. Ropke and "
             "Pisinger [15] reward each operator according to the solutions it produces: 33 points for a new best "
             "solution, 9 for an improvement of the current solution, and 13 for a worse solution that is accepted, "
             "so that operators that lead the search out of local optima are also rewarded. Operators are then "
             "selected with probabilities based on their average scores. The SA temperatures, the TS parameters and "
             "the move weights of this project are based on these three works (Section 5.3)."),
    ))

    # ------------------------------------------------------------ 5.2 modeling approach
    A(b.find("In this research, a MILP formulation"), *added(
        para("For this reason, the MILP is solved exactly only on small instances. For the company-scale instance "
             "(71 dealers, 80 trucks, 5 vehicle types, 30 days), the routing variables X alone number more than "
             "12 million (80 × 72 × 72 × 30), so a two-stage heuristic approach is adopted. In the first stage, a "
             "constructive heuristic (dynamic greedy) produces a feasible plan in a fraction of a second. In the "
             "second stage, a local-search metaheuristic repeatedly modifies this plan and keeps the changes that "
             "lower the total cost. Simulated annealing and tabu search were chosen because both are "
             "well-established for routing problems, both can work on the same solution representation and "
             "neighbourhood, and they escape local optima in two different ways: SA accepts worse plans with a "
             "probability that decreases over time, while TS always moves to the best sampled neighbour and "
             "forbids recently made moves. Using one neighbourhood and one objective for both methods makes any "
             "difference in the results a difference in the search strategy rather than in the cost accounting."),
        para("The main advantages of this approach are scalability and flexibility: the objective can contain "
             "every cost term of the mathematical model, and a plan of any size can be improved within a chosen "
             "time budget. Its disadvantage is that optimality cannot be proven, and the quality of the result "
             "depends on the parameters and on the time budget. For this reason, a substantial part of Chapter 6 "
             "is devoted to parameter selection and run length."),
    ))

    # ------------------------------------------------------------ 5.3 i) objective without the tour-time penalty
    old_objective = b.by_image("image23.png")
    closing = removed(old_objective)
    A(closing, *added(b.picture(Path(__file__).with_name("assets") / "objective_402.png", 6.5)))

    # ------------------------------------------------------------ 5.3 ii) dynamic greedy
    add_before(b.find("ii) Heuristic Solution"), "Greedy", "Dynamic ")
    append(b.find("The heuristic algorithm below is designed"),
           " It is the dynamic version of the greedy algorithm: vehicle types compete for trucks according to "
           "their unmet demand, and no surplus vehicles are left at dealers.")
    p = b.find("The algorithm takes the transportation network parameters")
    swap(p, "For each day, it iterates through each car type sequentially. It identifies \"critical dealers\" whose "
            "current inventory is insufficient to meet the day's forecasted demand.",
         "On each day, every truck that is not locked by an earlier tour is considered in turn. The truck is "
         "assigned to the vehicle type with the largest total unmet demand over all dealers on that day, so trucks "
         "are shared among vehicle types according to their shortages instead of serving the types one after "
         "another.")
    p = b.find("If available trucks and critical dealers exist")
    swap(p, "It delivers the required quantity to satisfy the demand of that dealer. If a truck satisfies a dealer's "
            "demand but still retains inventory, it continues to the next nearest critical dealer. If no critical "
            "dealers remain but the truck still holds inventory, the truck delivers the remaining vehicles to the "
            "final dealer on the route to avoid returning inventory to the factory.",
         "It unloads the smaller of its remaining load and the dealer's shortage, and continues to the next closest "
         "short dealer until it is empty or no short dealer remains. Since a truck never unloads more than a dealer "
         "needs, no surplus vehicles accumulate at the last dealer of a route. When the tour is complete, the truck "
         "returns to the factory and is locked for the duration of the tour, and dealer inventories are reduced by "
         "the day's demand.")
    greedy_tail = b.find("As output, it gives daily satisfied demand")
    closing = removed(greedy_tail)
    greedy_new = para(
        "As output, the algorithm gives the list of tours (day, truck, vehicle type, visited dealers and the quantity "
        "dropped at each), which is evaluated with the objective function of the mathematical model. On the original "
        "instance, the dynamic greedy plan delivers 4,306 of the 4,351 vehicles demanded in the month with 608 tours, "
        "leaves no vehicle unsold, and costs 2,076,165. For the entire pseudocode, please check the “Appendix” "
        "(Figure A.1).")

    framework = b.figure(PLOTS / "report" / "framework.png", "Figure 5.1",
                         "Solution framework of the heuristic approach", 6.0)
    weights = b.figure(PLOTS / "initial" / "weights.png", "Figure 5.2",
                       "Move selection weights obtained from the 400-minute pilot run", 5.6)
    moves = [
        ["Move", "Description"],
        ["rebuild", "Re-orders the dealers of one tour into the cheapest driving order (exact for tours of up to 11 dealers)."],
        ["hand-over", "Moves one dealer, with its quantity, from its tour to another tour of the same vehicle type, at the cheapest position."],
        ["exchange", "Swaps two dealers between two tours of the same vehicle type, each at its cheapest position."],
        ["add-car", "Drops one more vehicle at a dealer of a tour, if the truck has free space."],
        ["remove-car", "Drops one vehicle less at a dealer of a tour; the dealer stays on the tour even with zero quantity."],
        ["add-dealer", "Inserts a dealer with zero quantity at the cheapest position of an existing tour, or opens a new tour on a truck that is free on that day."],
        ["remove-dealer", "Removes one dealer from a tour; the tour is deleted when no dealer is left on it."],
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
        "       final temperature Tf, time budget τmax",
        "s ← s0;  best ← s0",
        "while elapsed time τ < τmax:",
        "    T ← T0 · (Tf / T0)^(τ / τmax)",
        "    draw a move m with probability proportional to its weight",
        "    s' ← apply m to s           (draw again if m cannot be built)",
        "    Δ ← f(s') − f(s)",
        "    if Δ ≤ 0 or random(0,1) < exp(−Δ / T):  s ← s'",
        "    if f(s) < f(best):  best ← s",
        "Output: best",
    ]
    ts_code = [
        "Input: starting plan s0, sample size n, tabu size θ,",
        "       time budget τmax",
        "s ← s0;  best ← s0;  tabu list ← empty",
        "while elapsed time τ < τmax:",
        "    N ← n neighbours of s, each made by a move drawn with the move weights",
        "    A ← { s' in N : move(s') is not tabu  or  f(s') < f(best) }",
        "    if A is empty:  A ← N",
        "    s ← the cheapest plan in A          (even if it is worse than s)",
        "    add move(s) to the tabu list for the next θ iterations",
        "    if f(s) < f(best):  best ← s",
        "Output: best",
    ]
    small = [
        ["Trucks", "Exact model (Gurobi, 600 s)", "Dynamic greedy", "Best metaheuristic (120 s)", "Gap to exact"],
        ["4", "29,742", "78,073", "52,176 (SA 500)", "75.4%"],
        ["6", "19,133", "57,340", "23,442 (SA 500)", "22.5%"],
        ["8", "18,157", "47,474", "20,107 (SA 500)", "10.7%"],
    ]

    A(closing, *added(
        greedy_new,
        heading("iii) Solution Representation and Objective Evaluation", 4),
        para("The metaheuristics work on the list of tours produced by the greedy algorithm. A solution is a set of "
             "tours, and each tour is defined by its dispatch day, truck, vehicle type, ordered list of dealers and "
             "the quantity dropped at each dealer. From this list, the daily inventory of every dealer and vehicle "
             "type is reconstructed and the plan is scored with the objective of the mathematical model: "
             "transportation cost (R for every arc travelled), holding cost (H for every vehicle in stock at the end "
             "of a day), backorder cost (B for every vehicle short on a day) and unsold-vehicle cost (G for every "
             "vehicle left at the end of the horizon). Dealer storage capacity is handled as a penalty of 15,000 per "
             "vehicle above capacity per day, the same order as the unsold-vehicle cost, so that a plan that violates "
             "it is never preferred. Truck capacity and the single-type rule hold by construction, and a change that "
             "makes two tours of the same truck overlap in time is rejected. Dealer-to-dealer transfers, which the "
             "mathematical model allows, are not generated by the heuristic methods: every vehicle is delivered from "
             "the factory."),
        para("Every candidate plan is evaluated with the full objective, so a change that shortens a route but "
             "increases backorders is judged by its total effect. In all final plans reported in Chapter 6, the "
             "storage penalty and the unsold-vehicle cost are zero. The overall solution framework is shown in "
             "Figure 5.1."),
        *framework,
        heading("iv) Neighbourhood Structure", 4),
        para("A neighbour is obtained from the current plan by one of the seven moves in Table 5.1. Moves that change "
             "a route choose the cheapest insertion position in terms of driving cost, and the full cost of the new "
             "plan is then evaluated. If the move drawn cannot be built (for example, there is no second tour of the "
             "same vehicle type to exchange a dealer with), another move is drawn. Some moves work together: "
             "add-dealer creates a stop with zero quantity, which add-car can later fill, and remove-car can empty a "
             "stop that remove-dealer later deletes."),
        caption("Table 5.1", "Moves of the neighbourhood", before_table=True),
        b.table(moves, [1700, 7300], align="ll"),
        para(""),
        heading("v) Move Selection Weights", 4),
        para("The seven moves are not equally useful, so instead of being drawn with equal probability, each move is "
             "drawn with a fixed weight. The weights are derived from the adaptive scoring scheme of Ropke and "
             "Pisinger [15], which rewards an operator according to the outcome of the solutions it produces. Every "
             "neighbour produced by a move receives a score of 33 if it is a new best plan, 9 if it improves the "
             "current plan, 13 if it is worse but accepted by the search, and 0 if it is rejected. Accepted worse "
             "solutions score higher than ordinary improvements because they are the steps that let the search leave "
             "a local optimum, while ordinary improvements are frequent."),
        para("In the original method, the weights are updated continuously during the search. In this project, the "
             "scores are collected in a single pilot run and the weights are then fixed, so that every run in "
             "Chapter 6 uses exactly the same neighbourhood and the differences between runs come only from the "
             "algorithm and the starting plan. The pilot was run on the original instance from a random plan with "
             "equal weights, using SA with T25 for 400 minutes; a random start was chosen so that every move had many "
             "chances to improve the plan, and the pilot reduced the cost of the random plan from 36.6 million to "
             "3.02 million. For each move, the total score is divided by the number of neighbours it produced, so "
             "that a move is not rewarded only for being drawn often. These average scores are scaled so that the "
             "smallest becomes 1, and are capped at 5. The lower bound keeps every move in the search, because some "
             "moves are rarely useful on their own but enable others; the upper bound prevents a single move from "
             "dominating the draw. The resulting weights are shown in Figure 5.2: rebuild is drawn most often (24% of "
             "the draws), exchange, hand-over and add-dealer follow with 17–18% each, and remove-car stays at the "
             "lower bound (6%)."),
        *weights,
        heading("vi) Simulated Annealing", 4),
        para("Simulated annealing draws one neighbour per iteration. A neighbour that lowers the total cost is always "
             "accepted, and a neighbour that increases it by Δ is accepted with probability exp(−Δ/T), where T is the "
             "current temperature (Metropolis criterion). High temperatures let the search move freely across the "
             "solution space, and low temperatures make it behave like a descent method."),
        para("The initial and final temperatures follow the rule proposed by Duman and Duman [13]: the initial "
             "temperature is chosen so that a solution worse than a reference solution by a given deviation is "
             "accepted with probability 0.75, 0.50 or 0.25, and the final temperature so that a solution worse by "
             "one tenth of that deviation is accepted with probability 0.0001. In the paper, the deviation is a "
             "percentage of the objective value of a good reference solution. In our problem, a single move changes "
             "the monthly cost by only a small fraction of the total, so the deviation was measured directly on the "
             "move scale: on the greedy plan of the original instance, the average cost increase of a worsening "
             "neighbour is Δ = 6,685. With T = −Δ / ln p, this gives the temperatures in Table 5.2."),
        caption("Table 5.2", "Temperature settings of simulated annealing", before_table=True),
        b.table(temps, [1500, 1900, 3800, 1800]),
        para(""),
        para("In addition to the three settings of the paper, two colder starting temperatures, 500 and 50, were "
             "included as reference settings. At these temperatures an average worsening move is almost never "
             "accepted, but small worsenings still are, so they test whether the hot phase of the paper's settings is "
             "needed at all. SA 500 cools to the same final temperature as the other settings. Since 50 is already "
             "below that value, SA 50 is cooled by the same ratio as T25, from 50 to 0.75."),
        para("The paper decreases the temperature geometrically, multiplying it by a factor α after every L "
             "neighbours, with α chosen so that the final temperature is reached when the iteration budget is used "
             "up. Because our runs are limited by computation time rather than by an iteration count, the same "
             "geometric schedule is applied to the elapsed time: T(τ) = T0 · (Tf / T0)^(τ/τmax), where τ is the "
             "elapsed time and τmax the time budget. Every run therefore reaches its final temperature exactly at the "
             "end of its time budget, independently of how fast the moves are. The procedure is summarized in "
             "Figure 5.3."),
        b.algorithm(sa_code),
        caption("Figure 5.3", "Pseudocode of simulated annealing"),
        heading("vii) Tabu Search", 4),
        para("Tabu search samples a fixed number of neighbours of the current plan in every iteration, scores all of "
             "them, and moves to the cheapest admissible one, even if it is worse than the current plan. A neighbour "
             "is admissible if the move that produced it is not tabu. The tabu attribute is the exact move, that is, "
             "its type and the tours, dealers and positions it changes; once performed, the move is placed in the "
             "tabu list and cannot be repeated for the next θ iterations, where θ is the tabu size. Following Glover "
             "[14], a tabu move is still allowed if it produces a plan better than the best plan found so far "
             "(aspiration criterion), and if all sampled neighbours are tabu, the cheapest of them is taken so that "
             "the search never stalls (Figure 5.4)."),
        para("The two parameters were set by our own choice and then compared in a screening experiment (Section "
             "6.1). The number of sampled neighbours was set relative to the number of dealers: 70 is roughly one "
             "neighbour per dealer of the original instance, and 35 roughly one per two dealers. For the tabu size, "
             "the values 25 and 625 were chosen from the geometric series 5, 25, 125, 625, so that a small and a very "
             "large tabu list, differing by a factor of 25, are compared."),
        b.algorithm(ts_code),
        caption("Figure 5.4", "Pseudocode of tabu search"),
    ))

    # ------------------------------------------------------------ 5.4 caption numbers
    swap(b.find("Figure 2.1: Console Outputs"), "Figure 2.1", "Figure 5.5")
    swap(b.find("By visualizing the daily dispatch schedules"), "Figures 2.1 to 2.4", "Figures 5.6 to 5.9")
    p = b.find("Figure 3.1")
    swap(p, "Figure 3.1", "Figure 5.6")
    swap(p, "Figure 3.2", "Figure 5.7")
    p = b.find("Figure 3.3")
    swap(p, "Figure 3.3", "Figure 5.8")
    swap(p, "Figure 3.4", "Figure 5.9")

    # ------------------------------------------------------------ 5.5 and 5.6
    heur_ver = b.find("For the verification and the validation of the heuristic model")
    swap(heur_ver, "Figure 4.1 to\n4.4", "Figures A.2 to A.5") if "Figure 4.1 to\n4.4" in element_text(heur_ver) \
        else swap(heur_ver, "Figure 4.1 to 4.4", "Figures A.2 to A.5")
    add_after(heur_ver, "builds up its road according to that logic.",
              " In the dynamic version, each truck takes the vehicle type with the largest shortage on its dispatch "
              "day, visits the nearest short dealers in sequence, returns to the factory and stays locked until its "
              "tour is complete.")
    A(heur_ver, *added(
        heading("5.6) Verification and validation of the Metaheuristics", 3),
        para("The metaheuristics were verified in two ways. First, every plan they return is evaluated from scratch "
             "with the same objective function as the starting plan, and its cost components are checked: in all "
             "runs reported in Chapter 6, no truck has overlapping tours or carries more than its capacity, no dealer "
             "exceeds its storage capacity and no vehicle is left unsold. Second, the metaheuristics were run on the "
             "small instances of Section 6.1 (10 dealers, 7 days, 2 vehicle types, 4 to 8 trucks), for which "
             "time-limited exact solutions are available. Each setting was given 120 seconds, starting from the "
             "dynamic greedy plan; the best results are shown in Table 5.3."),
        caption("Table 5.3", "Exact model, dynamic greedy and metaheuristics on the small instances", before_table=True),
        b.table(small, [1100, 2200, 1700, 2500, 1500]),
        para(""),
        para("The metaheuristics remove most of the gap between the greedy plan and the exact solution when the fleet "
             "is sufficient (6 and 8 trucks). With 4 trucks the remaining gap is larger: the fleet is too small to "
             "serve the demand on time, and the exact model can also use dealer-to-dealer transfers, which the "
             "neighbourhood does not generate. Note that the exact values are the best solutions found within the "
             "600-second limit and are not necessarily optimal. These results confirm that the search improves the "
             "plan in the expected direction and stays consistent with the exact model, and they already indicate "
             "that instances with a tight fleet are the hardest for local search, which is confirmed on the "
             "company-scale instances in Chapter 6."),
    ))

    # ------------------------------------------------------------ chapter 6 intro and 6.1 i-iii
    A(b.find("In this chapter, the computational experiments"), *added(para(
        "In IE 402, two further parts were added: a computational study of the metaheuristics on the company-scale "
        "instance and three variants of it, and extended runs for the settings that were still improving at the end "
        "of that study. The heuristic and the metaheuristics are implemented in Python; every metaheuristic run uses "
        "a single processor core of a 16-core workstation, so that runs executed in parallel do not share computing "
        "power, and all runs use the same random seed (44).")))
    p = b.find("First, we analyzed the exact model")
    swap(p, "5 dealers, 10 trucks", "10 dealers, 6 trucks")
    swap(b.find("The results show a clear trade off"), "Table 1.1", "Table A.1")
    swap(b.find("In the results, for the truck numbers"), "Table 2.1", "Table A.2")
    heur_sens = b.find("To compare with the exact model, the heuristic algorithm")
    closing = removed(heur_sens)

    instances = [
        ["Instance", "Dealers", "Trucks", "Demand scale", "Monthly demand (vehicles)", "Greedy plan cost"],
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
    extended_rows = [["Instance, start", "Setting", "Best in the 400-minute run", "Best in the 1000-minute run",
                      "Improvement over the greedy plan (400 → 1000 min)"]] + ext_rows
    pilot_rows = [
        ["Cost component", "Dynamic greedy", "TS 70/625 (400 min)", "[[best 1000-minute plan, pending]]"],
        ["Transportation", "1,902,665", "1,418,288", PENDING],
        ["Holding", "0", "61,560", PENDING],
        ["Backorder", "173,500", "22,500", PENDING],
        ["Unsold vehicles", "0", "0", PENDING],
        ["**Total**", "**2,076,165**", "**1,502,348**", PENDING],
        ["Vehicles delivered (of 4,351)", "4,306", "4,341", PENDING],
        ["Backorders (vehicle-days)", "347", "45", PENDING],
    ]
    cfg = PLOTS / "analysis"
    greedy_figs, random_figs = {}, {}
    names = (("00_original", "Original"), ("02_growth_no_fleet", "Growth, same fleet"),
             ("07_full_slack_thin", "Full map, slack"), ("09_far_tight", "Far, tight"))
    for k, (folder, title) in enumerate(names):
        greedy_figs[k] = b.figure(cfg / "greedy" / folder / "cost_vs_time.png", f"Figure 6.{k + 1}",
                                  f"Best objective over time, {title} instance, greedy start", 5.4)
        random_figs[k] = b.figure(cfg / "random" / folder / "cost_vs_time.png", f"Figure 6.{k + 5}",
                                  f"Best objective over time, {title} instance, random start", 5.4)

    A(closing, *added(
        para("To compare with the exact model, the dynamic greedy heuristic was tested across the same truck numbers, "
             "and its plans were evaluated with the same objective function. The heuristic solves every scenario in "
             "less than 0.001 seconds, and its objective values are 78,073, 57,340 and 47,474 for 4, 6 and 8 trucks. "
             "As in the exact model, more trucks lead to fewer backorders and a lower total cost, and no vehicle is "
             "left unsold in any scenario. However, the greedy plans remain 2.6 to 3.0 times as expensive as the "
             "exact solutions, because of both backorders and longer routes: each truck only serves dealers that are "
             "short on its dispatch day, in nearest-neighbour order (the full table is available as Table A.3 in the "
             "“Appendix”). This gap is what the metaheuristics address; their results on the same instances are "
             "given in Section 5.6."),
        heading("iv) Company-Scale Test Instances", 4),
        para("The company-scale instance was generated from the operating averages provided by Tofaş. The Bursa plant "
             "and 71 dealer sites are placed in 45 cities across Türkiye in proportion to the actual dealer network. "
             "Road distances are approximated by great-circle distances multiplied by 1.35, and transportation costs "
             "1.5 per km. A leg of up to 280 km is completed within the same day, and longer legs take one day per "
             "500 km. The daily demand of each dealer for the five vehicle types (Egea Cross, Egea Sedan, Scudo, "
             "Scudo Van and Ulysse) is drawn from {0, 1, 2} with the company's average daily demand per dealer "
             "(0.636, 0.997, 0.296, 0.061 and 0.052 respectively). A truck carries 8 Egea vehicles or 6 of the larger "
             "types, dealer storage capacities range from 36 to 80 vehicles, the fleet consists of 80 trucks and the "
             "horizon is 30 days. The cost parameters are H = 20 per vehicle per day, B = 500 per vehicle per day and "
             "G = 15,000 per unsold vehicle."),
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
        para("Nine algorithm settings were first compared on the original instance, starting from the greedy plan, "
             "with a time limit of 100 minutes each: SA with the five starting temperatures of Table 5.2, and TS with "
             "35 or 70 sampled neighbours and a tabu size of 25 or 625. All nine runs were executed in parallel. "
             "Within 100 minutes, the four tabu settings achieved the largest improvements (23.3% to 24.1%), followed "
             "by SA 50 (21.9%) and SA 500 (14.4%), while SA T75, T50 and T25 did not improve the greedy plan at all: "
             "at these temperatures, the current plan was still moving above the cost of the greedy plan when the "
             "time limit was reached (full results in Table A.5 and Figures A.6 to A.8 in the “Appendix”)."),
        para("Based on this screening, the number of settings was reduced from nine to six. Among the tabu settings, "
             "70 sampled neighbours performed better than 35 for both tabu sizes, so the two settings with 70 "
             "neighbours were kept; since they share the same sample size, their comparison isolates the effect of "
             "the tabu size. Among the SA settings, T50 was removed: it lies between T75 and T25, which already "
             "represent the two ends of the acceptance range of the paper, and the three behaved identically in the "
             "screening. The cold settings SA 500 and SA 50 were kept because they were the only SA settings that "
             "improved the plan within the screening time."),
        heading("vi) Design of the Main Experiment", 4),
        para("The main experiment runs the six selected settings (SA T75, SA T25, SA 500, SA 50, TS 70/25 and TS "
             "70/625, where the TS label gives the sample size and the tabu size) on the four instances of Table 6.1, "
             "from two different starting plans, with a time limit of 400 minutes per run: 48 runs in total, "
             "executed in parallel on separate cores. The first starting plan is the dynamic greedy plan, which shows "
             "how much the metaheuristics add to the constructive heuristic. The second is a random plan, built by "
             "drawing tours with random days, trucks, vehicle types, dealers and quantities while respecting truck "
             "capacities and truck availability; it shows whether an algorithm can find good plans without a good "
             "starting point. The random plans are very expensive (between 13.9 and 96.4 million) because they "
             "ignore demand. The move weights of Figure 5.2 and the temperatures of Table 5.2 are the same in every "
             "run."),
        heading("vii) Results from the Greedy Start", 4),
        para("Table 6.2 summarizes the best plan found by each setting from the greedy start, and Figures 6.1 to 6.4 "
             "show how the best objective value develops over time. The corresponding current-objective plots are "
             "given in Figures A.9 to A.16 in the “Appendix”."),
        caption("Table 6.2", "Best objective in 400 minutes from the greedy start (improvement over the greedy plan)", before_table=True),
        b.table(greedy_rows, [1500, 1900, 2100, 1800, 1800], align="lrrrr", size=19),
        para(""),
        para("On the **original instance**, TS 70/625 found the best plan (1,502,348, 27.6% below the greedy plan), "
             "followed closely by SA 50 and TS 70/25 (26.9% each). The tabu settings and SA 50 improve the plan "
             "quickly in the first minutes and then continue slowly, and SA 500 improves gradually and reaches 24.4%. "
             "The hot settings behave differently: SA T25 only starts to improve after 338 minutes, once its "
             "temperature has decreased sufficiently, and is still improving quickly at the end of the run (15.3%), "
             "while SA T75 does not improve the greedy plan within 400 minutes (Figure 6.1)."),
        *greedy_figs[0],
        para("In the **growth scenario**, the fleet is far too small for twice the demand on a larger network, and "
             "backorders dominate the cost (27.5 million of the 29.2 million of the greedy plan). Here the ranking "
             "is reversed: SA T25 is the best setting (20,819,112, 28.8%), followed by SA 500 (22.0%) and SA T75 "
             "(16.1%), while SA 50 (4.9%) and both tabu settings (1.1–1.4%) barely improve the plan. The hot SA "
             "settings accept long sequences of worse plans and thereby restructure the tours so that more vehicles "
             "are delivered (11,514 instead of 9,842 for SA T25), at the price of a higher transportation cost. All "
             "three are still decreasing steeply at 400 minutes (Figure 6.2)."),
        *greedy_figs[1],
        para("When the fleet is ample and demand is low (**full map, slack**), the greedy plan has no backorders and "
             "its cost consists only of transportation. All six settings reduce it by 37% to 39%, and TS 70/625 "
             "(802,706) and SA 500 (806,229) are practically tied. The SA plans drive much less but keep more vehicles "
             "in stock at the dealers, while the tabu and SA 50 plans drive more and hold less; different trade-offs "
             "between transportation and holding lead to similar total costs (Figure 6.3)."),
        *greedy_figs[2],
        para("On the sparse national network with a tight fleet (**far, tight**), the greedy plan suffers from both "
             "long routes and backorders. The three SA settings with higher temperatures reduce the cost by about "
             "62–63%, SA 500 being the best (595,680, 63.0%), while TS and SA 50 stop at about 46%, with about five "
             "times as many backorders as the SA plans (around 650 vehicle-days against 109–142). Here, accepting "
             "worse plans is necessary to rearrange the tours of the few available trucks (Figure 6.4)."),
        *greedy_figs[3],
        heading("viii) Results from the Random Start", 4),
        para("From a random plan, every setting reduces the cost by a large factor (Table 6.3), but the relevant "
             "comparison is with the greedy plan. SA T25 is the best setting on all four instances, while TS and "
             "SA 50, which perform well from the greedy plan, remain far above the greedy plan on every instance: "
             "without a high acceptance probability, they settle in the first deep local optimum they reach. The "
             "current-objective plots are given in Figures A.17 to A.24 in the “Appendix”."),
        caption("Table 6.3", "Best objective in 400 minutes from the random start (improvement over the greedy plan; negative values are above it)", before_table=True),
        b.table(random_rows, [1500, 2000, 2000, 1800, 1800], align="lrrrr", size=19),
        para(""),
        para("On the **original instance**, no setting reaches the greedy plan within 400 minutes. SA T25 ends at "
             "3.54 million, 70% above the greedy plan, and SA T75 and SA T25 are still decreasing at the end of the "
             "run (Figure 6.5)."),
        *random_figs[0],
        para("On the **growth instance**, SA T25 ends 6.8% below the greedy plan, even though it started from a plan "
             "more than three times as expensive; SA T75 ends just above it (Figure 6.6)."),
        *random_figs[1],
        para("On the **full-slack instance**, the three hotter SA settings end about 35–36% below the greedy plan and "
             "close to the results obtained from the greedy start (Figure 6.7)."),
        *random_figs[2],
        para("On the **far-tight instance**, SA T25 ends 64.2% below the greedy plan and finds the best plan of the "
             "entire study (575,622), better than any result obtained from the greedy start (Figure 6.8)."),
        *random_figs[3],
        heading("ix) Extended 1000-Minute Runs", 4),
        para("Figures 6.1 to 6.8 show that some settings were still improving steeply when the 400-minute limit was "
             "reached. In the last hour of the run, SA T25 on the original instance improved its best value by "
             "14.6%, and the three hot SA settings on the growth instance by 1.9% to 5.9%, whereas the tabu settings "
             "and SA 50 improved by less than 1% in their last hour from the greedy start; SA T75 on the original "
             "instance had not improved at all. To measure what these settings gain from more time, seven runs were "
             "extended to 1000 minutes: SA T75, SA T25 and SA 500 on the original instance, SA T75 from the random "
             "start on the original instance, and SA T75, SA T25 and SA 500 on the growth instance."),
        para("An extended run continues the 400-minute run. For its first 400 minutes, it follows the same cooling "
             "schedule and reaches the final temperature of Table 5.2 at minute 400; after that, the temperature "
             "keeps decreasing at the same geometric rate, T(τ) = T0 · (Tf / T0)^(τ/400) for τ up to 1000 minutes, so "
             "that the additional 600 minutes are spent at ever lower temperatures, close to a pure descent. Table "
             "6.4 compares the results; the time plots, together with the 400-minute curves of the other settings, "
             "are given in Figures A.25 to A.34 in the “Appendix”."),
        caption("Table 6.4", "Results of the 1000-minute runs", before_table=True),
        b.table(extended_rows, [1650, 1000, 1800, 1800, 2750], align="llrrc", size=19),
        para(""),
        para("[[Discussion of the 1000-minute results, to be written when the runs finish (20 November 2026, about "
             "15:15).]]"),
    ))

    # ------------------------------------------------------------ 6.2 discussion
    p62 = b.find("The exact algorithm is not applicable to the original problem")
    strike(p62, " The current heuristic algorithm is designed to be simple and easily applicable. As expected, "
                "simplicity comes with a cost.")
    old_block = block_until(p62.getnext(), lambda e: "6.3)" in element_text(e))
    closing = removed(old_block[0], old_block[-1])
    discussion = [
        "The proposed solution is therefore the combination of the dynamic greedy heuristic and a metaheuristic. The "
        "greedy plan is available in less than a second and is already feasible, with no unsold vehicles; the "
        "metaheuristics then reduce its cost by 27.6% to 63.0% within 400 minutes, depending on the instance.",
        "No single setting is best on every instance, and the results are explained by how tight the fleet is "
        "relative to demand. When the fleet is sufficient (original and full-slack instances), the greedy plan "
        "already has a good structure, and methods that accept few or no worse plans (TS and the cold SA settings) "
        "improve it quickly; TS 70/625 gives the best 400-minute result on both. When the fleet is tight (growth and "
        "far-tight instances), backorders dominate the cost and improving the plan requires rearranging many tours; "
        "only the SA settings that accept worse plans achieve this, and their advantage grows with time.",
        "The hot SA settings are limited mainly by time. Their temperature has to fall before they start improving, "
        "which in 400 minutes happened late or not at all on the original instance. [[What the 1000-minute runs "
        "show, pending.]]",
        "The random-start experiments show how robust the methods are. SA T25 is the most robust setting: from a plan "
        "that ignores demand, it ends below the greedy plan on three of the four instances. TS and SA 50 depend "
        "strongly on a good starting plan. For practical use, the greedy plan is therefore the recommended starting "
        "point.",
        "For TS, the tabu size has little influence: TS 70/25 and TS 70/625 differ by at most 2.2% on every instance. "
        "Because the tabu attribute is the exact move and the neighbourhood is very large, the same move is rarely "
        "sampled again; at most 2% of the sampled neighbours were tabu in the main study. The number of sampled "
        "neighbours (70 against 35) had a clearer effect in the screening.",
        "Based on these results, the recommended procedure is to build the dynamic greedy plan and then to run TS "
        "with 70 neighbours when only a short computation time is available or the fleet is sufficient, and SA "
        "starting from T25 or 500 for an overnight or longer run, especially when the fleet is tight relative to "
        "demand. Since every setting runs on a single core, several settings can be run in parallel and the best "
        "plan kept. [[To be revised with the 1000-minute results.]]",
        "All results are based on a single random seed and on one weight vector obtained from a single pilot run. "
        "Repeated runs with different seeds would show how much of the difference between close settings, such as "
        "the near-ties on the full-slack instance, is due to chance.",
    ]
    A(closing, *added(*[para(text) for text in discussion]))

    # ------------------------------------------------------------ 6.3 pilot
    pilot = b.find("For pilot implementation, we run the heuristic")
    old_block = block_until(pilot, lambda e: "Chapter 7" in element_text(e))
    closing = removed(old_block[0], old_block[-1])
    A(closing, *added(
        para("The original instance has the size of the real Fiat network (71 dealers, 80 trucks, 5 vehicle types, "
             "30 days) and serves as the pilot implementation of the approach. Table 6.5 compares the greedy plan "
             "with the best plans found in 400 and in 1000 minutes."),
        caption("Table 6.5", "Cost components of the greedy plan and the best plans on the original instance", before_table=True),
        b.table(pilot_rows, [3000, 1900, 2100, 2000], align="lrrr"),
        para(""),
        para("The best 400-minute plan cuts the backorder cost by 87% and the transportation cost, which is "
             "proportional to the distance driven, by 25.5%. It keeps inventories low and delivers 4,341 of the "
             "4,351 vehicles demanded; no vehicle remains unsold and no dealer exceeds its storage capacity. Since "
             "every kilometre saved also reduces fuel consumption and emissions, the shorter routes support the "
             "environmental goal stated in Section 4.5. [[Comparison with the best 1000-minute plan, pending.]]"),
    ))

    # ------------------------------------------------------------ chapter 7
    A(b.find("Model deployment involves the regular collection"), *added(para(
        "In practice, we suggest a monthly planning cycle. At the beginning of each month, the demand forecasts, "
        "dealer inventories and truck availability are collected; the dynamic greedy heuristic produces a first plan; "
        "a metaheuristic improves this plan overnight; and planners review the resulting tours before dispatching. "
        "When forecasts or inventories change significantly during the month, the remaining days can be re-planned "
        "in the same way, since the metaheuristics can start from any feasible plan.")))
    p = b.find("The implementation also requires appropriate software")
    closing = removed(p)
    A(closing, *added(para(
        "The implementation requires a standard computer with Python. The dynamic greedy heuristic and the "
        "metaheuristics do not need a commercial solver; a Gurobi license is only needed if the exact model is used "
        "for small what-if analyses. The greedy plan is produced in less than a second, and a metaheuristic run of 400 "
        "minutes fits into an overnight batch. Since each run uses a single processor core, several settings can be "
        "run in parallel on a multi-core workstation and the best plan selected, so the approach is technically and "
        "financially feasible with a low investment. Basic training is needed for planners to prepare the input data "
        "and to interpret the resulting tours.")))
    p = b.find("Despite this study including a full scale implementation")
    closing = removed(p)
    A(closing, *added(para(
        "The main difficulty of implementation is data. The instance used in this study was generated from company "
        "averages (daily demand per vehicle type, truck capacities and fleet size) and approximate road distances. "
        "Before deployment, the model must be fed with actual dealer locations, road distances, storage capacities "
        "and forecasts, and its plans must be compared with the routes currently used by Tofaş. A second difficulty "
        "is computation time: good plans for large or tight instances require several hours, so the planning cycle "
        "must leave time for an overnight run. Finally, demand forecasts change during the month, which requires "
        "re-planning on a rolling basis.")))

    # ------------------------------------------------------------ chapter 8
    A(b.find("In this study, we examined the inventory distribution system"), *added(
        para("In IE 402, the greedy heuristic was revised into a dynamic version that assigns trucks according to "
             "unmet demand and leaves no unsold vehicles, and two metaheuristics, simulated annealing and tabu search, "
             "were developed on a shared neighbourhood of seven moves. Their parameters were set with methods from "
             "the literature: the initial and final temperatures of SA follow the acceptance-probability rule of "
             "Duman and Duman [13], tabu search follows Glover [14], and the move weights were derived from the "
             "scoring scheme of Ropke and Pisinger [15] in a single pilot run."),
        para("On four company-scale instances, the metaheuristics reduced the cost of the greedy plan by 27.6% to "
             "63.0% in 400 minutes [[and … in 1000 minutes, pending]]. On the original instance, the best 400-minute "
             "plan cuts the transportation cost by 25.5% and the backorder cost by 87% compared with the greedy plan. "
             "Tabu search was the best choice for short run times and sufficient fleets, while simulated annealing "
             "with higher starting temperatures was the only method able to improve plans for tight fleets and from "
             "random starting plans, provided that it was given enough time."),
    ))
    append(b.find("While the primary objective of this study is to minimize costs"),
           " Fewer backorders also mean that customers receive their vehicles with shorter delays. The approach does "
           "not raise legal issues, since it only uses operational data of the company, and it does not affect the "
           "safety of drivers as long as the travel times used in the model respect the legal driving limits.")
    p = b.find("The current work is limited to model development")
    closing = removed(p)
    A(closing, *added(para(
        "The current work has some limitations that point to future work. The results are based on a single random "
        "seed and on an instance generated from company averages; repeating the runs with several seeds and "
        "validating the plans with real dealer and route data would strengthen the conclusions. Several settings were "
        "still improving at the end of their runs, so longer runs, parallel runs and hybrid strategies (for example, "
        "a fast tabu search phase followed by simulated annealing) could improve the results further. The move "
        "weights could also be updated during the search instead of being fixed after a pilot run. Finally, the "
        "approach could be extended with dealer-to-dealer transfers in the neighbourhood, stochastic demand with "
        "re-planning during the month, and an explicit CO₂ emission term, bringing the analysis closer to the real "
        "operating conditions of Tofaş.")))

    # ------------------------------------------------------------ references
    removed(b.find("In this part, there should be a list of references"))
    A(b.find("[12]"), *added(
        para("[13] T. Duman and E. Duman, “Solving a new application of asymmetric TSP by modified migrating birds "
             "optimization algorithm,” *Evolutionary Intelligence*, vol. 17, no. 3, 2024.", after=240, before=240),
        para("[14] F. Glover, “Tabu search—Part I,” *ORSA Journal on Computing*, vol. 1, no. 3, pp. 190–206, 1989.",
             after=240, before=240),
        para("[15] S. Ropke and D. Pisinger, “An adaptive large neighborhood search heuristic for the pickup and "
             "delivery problem with time windows,” *Transportation Science*, vol. 40, no. 4, pp. 455–472, 2006.",
             after=240, before=240),
    ))

    # ------------------------------------------------------------ appendix
    greedy_code = [
        "Input: dealers N, types C, trucks K, days T, travel times A,",
        "       capacities Q, daily demand F",
        "I[i][c] ← 0 for every dealer i and type c",
        "locked_until[k] ← 0 for every truck k",
        "for each day t in T:",
        "    for each truck k in K:",
        "        if locked_until[k] > t: continue",
        "        c* ← type with the largest total shortage",
        "              sum over i of max(0, F[i][c][t] - I[i][c])",
        "        if no type is short: continue",
        "        current ← factory; load ← Q[c*]; route ← []; time ← 0",
        "        while load > 0:",
        "            j ← nearest dealer to current, not on route,",
        "                 with I[j][c*] < F[j][c*][t]",
        "            if there is no such dealer: break",
        "            drop ← min(load, F[j][c*][t] - I[j][c*])",
        "            I[j][c*] ← I[j][c*] + drop; load ← load - drop",
        "            time ← time + A[current][j]; current ← j",
        "            append (j, drop) to route",
        "        if route is not empty:",
        "            time ← time + A[current][factory]",
        "            locked_until[k] ← t + time",
        "            record the tour (t, k, c*, route)",
        "    for every dealer i and type c: I[i][c] ← I[i][c] - F[i][c][t]",
        "Output: the list of tours",
    ]
    pseudo_old = b.by_image("image12.jpg")
    closing = removed(pseudo_old, b.find("Figure 1.1: Console Outputs"))
    A(closing, *added(b.algorithm(greedy_code), caption("Figure A.1", "Pseudocode of the Dynamic Greedy Heuristic")))

    add_before(b.find("Heuristic Solution Plots"), "Heuristic", "Dynamic Greedy ")
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
    heur_table = None
    for t in d.tables:
        if t.cell(0, 0).text.strip().startswith("# of") and "Time" in t.cell(0, t._column_count - 1).text:
            heur_table = t._tbl
    closing = removed(heur_caption, heur_table)
    heur_rows = [
        ["# of Trucks", "Obj. Value", "Trans. Cost", "Backorder Cost", "Unsold Cost", "Time"],
        ["4", "78073.19", "16073.19", "62000.00", "0", "0.0001"],
        ["6", "57339.80", "28339.80", "29000.00", "0", "0.0001"],
        ["8", "47473.94", "33973.94", "13500.00", "0", "0.0001"],
    ]
    pool = [
        ["#", "Instance", "Dealer network", "Trucks", "Demand scale"],
        ["00", "Original", "71 dealers (original)", "80", "1.0×"],
        ["01", "Scaled firm", "110 dealers", "120", "1.0×"],
        ["02", "Growth, same fleet", "110 dealers", "80", "2.0×"],
        ["03", "Large map, tight fleet", "110 dealers", "30", "1.0×"],
        ["04", "Near region, congested", "23 same-day dealers", "30", "2.0×"],
        ["05", "Sparse country, slack fleet", "24 cities across the country", "140", "0.5×"],
        ["06", "Same map, loaded", "71 dealers (original)", "30", "2.0×"],
        ["07", "Full map, slack", "71 dealers (original)", "140", "0.5×"],
        ["08", "Near region, thin demand", "23 same-day dealers", "80", "0.5×"],
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
    screen_dir = cfg / "greedy" / "00_original_100mins"
    appendix = [
        para(""),
        caption("Table A.3", "Truck Numbers and Objective Value, Dynamic Greedy Heuristic", before_table=True),
        b.table(heur_rows, [1300, 1500, 1500, 1700, 1400, 1100]),
        para(""),
        label_line("Test Instances"),
        caption("Table A.4", "The ten instance variants designed for the study (the instances of the main study are 00, 02, 07 and 09)", before_table=True),
        b.table(pool, [600, 3000, 3100, 1000, 1300], align="clccc"),
        para(""),
        label_line("Algorithm Screening (100 minutes, original instance, greedy start)"),
        caption("Table A.5", "Screening results of the nine settings (TS: sample size / tabu size)", before_table=True),
        b.table(screen, [2600, 2600, 3400], align="lrr"),
        para(""),
        *b.figure(screen_dir / "cost_vs_time.png", "Figure A.6", "Screening, best objective over time"),
        *b.figure(screen_dir / "curr_cost_vs_time.png", "Figure A.7", "Screening, current objective over time"),
        *b.figure(screen_dir / "curr_cost_vs_time_rescaled.png", "Figure A.8", "Screening, current objective on the best-cost axis"),
        label_line("Current-Objective Plots of the 400-Minute Study"),
    ]
    n = 9
    for start, word in (("greedy", "greedy start"), ("random", "random start")):
        for folder, title in names:
            appendix += b.figure(cfg / start / folder / "curr_cost_vs_time.png", f"Figure A.{n}",
                                 f"Current objective over time, {title} instance, {word}")
            appendix += b.figure(cfg / start / folder / "curr_cost_vs_time_rescaled.png", f"Figure A.{n + 1}",
                                 f"Current objective on the best-cost axis, {title} instance, {word}")
            n += 2
    ext = cfg / "extended"
    appendix.append(label_line("Extended 1000-Minute Runs"))
    groups = ((ext / "greedy" / "00_original", "Original instance, greedy start", ("sa_T75", "sa_T25", "sa_500")),
              (ext / "random" / "00_original", "Original instance, random start", ("sa_T75",)),
              (ext / "greedy" / "02_growth_no_fleet", "Growth, same fleet instance, greedy start",
               ("sa_T75", "sa_T25", "sa_500")))
    labels = {"sa_T75": "SA T75", "sa_T25": "SA T25", "sa_500": "SA 500"}
    if ext_ready:
        for folder, title, _keys in groups:
            appendix += b.figure(folder / "cost_vs_time.png", f"Figure A.{n}",
                                 f"Best objective over time, {title}; extended settings 1000 minutes, others 400 minutes")
            n += 1
        for folder, title, keys in groups:
            for key in keys:
                appendix += b.figure(folder / "convergence" / f"{key}.png", f"Figure A.{n}",
                                     f"Current and best objective of the 1000-minute {labels[key]} run, {title}")
                n += 1
    else:
        appendix.append(para("[[Figures A.25 to A.34 (best objective over time for the three extended cases, and "
                             "current and best objective of the seven 1000-minute runs) will be added when the runs "
                             "finish.]]"))
    A(closing, *added(*appendix))

    # ------------------------------------------------------------ save
    d.save(str(OUT))
    work.unlink()
    print("saved", OUT, "| extended results", "in" if ext_ready else "pending")


if __name__ == "__main__":
    main()
