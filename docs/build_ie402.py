"""Builds docs/IE 402 v1.docx on top of the submitted IE 401 report.

The IE 401 document is copied, its runs are merged, and the IE 402 changes are
applied in place so that fonts, spacing and heading styles stay identical.

    python docs/build_ie402.py
"""

import copy
import re
import shutil
import subprocess
import sys
from pathlib import Path

import docx
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PLOTS = ROOT / "plots"
SRC = DOCS / "IE 401.docx"
OUT = DOCS / "IE 402 v1.docx"
MERGE = Path(sys.argv[1]) if len(sys.argv) > 1 else None

FONT = '<w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>'
MONO = '<w:rFonts w:ascii="Courier New" w:cs="Courier New" w:eastAsia="Courier New" w:hAnsi="Courier New"/>'


# ---------------------------------------------------------------- xml helpers

def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def runs_xml(markup, size=24, font=FONT, base_bold=False, base_italic=False, raw=False):
    """**bold**, *italic*, [[highlighted placeholder]]; raw=True keeps the text as it is."""
    out = []
    tokens = [markup] if raw else re.split(r"(\*\*.+?\*\*|\[\[.+?\]\]|\*.+?\*)", markup)
    for token in tokens:
        if not token:
            continue
        bold, italic, mark = base_bold, base_italic, False
        if raw:
            pass
        elif token.startswith("**"):
            token, bold = token[2:-2], True
        elif token.startswith("[["):
            token, mark = token[2:-2], True
        elif token.startswith("*"):
            token, italic = token[1:-1], True
        props = font
        if bold:
            props += '<w:b w:val="1"/><w:bCs w:val="1"/>'
        if italic:
            props += '<w:i w:val="1"/><w:iCs w:val="1"/>'
        props += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        if mark:
            props += '<w:highlight w:val="yellow"/>'
        out.append(f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{esc(token)}</w:t></w:r>')
    return "".join(out)


def para(markup, jc="both", after=200, before=0, line=360, size=24, keep_next=False):
    keep = '<w:keepNext w:val="1"/>' if keep_next else ""
    return parse_xml(
        f'<w:p {nsdecls("w")}><w:pPr>{keep}<w:spacing w:after="{after}" w:before="{before}" '
        f'w:line="{line}" w:lineRule="auto"/><w:jc w:val="{jc}"/></w:pPr>{runs_xml(markup, size)}</w:p>'
    )


def heading(text, level):
    if level == 3:
        ppr = (f'<w:pStyle w:val="Heading3"/><w:spacing w:line="360" w:lineRule="auto"/><w:jc w:val="both"/>'
               f'<w:rPr>{FONT}<w:b w:val="1"/><w:bCs w:val="1"/><w:color w:val="000000"/>'
               f'<w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr>')
        rpr = f'{FONT}<w:b w:val="1"/><w:bCs w:val="1"/><w:color w:val="000000"/><w:sz w:val="26"/><w:szCs w:val="26"/>'
    else:
        ppr = (f'<w:pStyle w:val="Heading4"/><w:spacing w:after="200" w:line="360" w:lineRule="auto"/>'
               f'<w:jc w:val="both"/><w:rPr>{FONT}<w:b w:val="1"/><w:bCs w:val="1"/><w:color w:val="000000"/></w:rPr>')
        rpr = f'{FONT}<w:b w:val="1"/><w:bCs w:val="1"/><w:color w:val="000000"/>'
    return parse_xml(
        f'<w:p {nsdecls("w")}><w:pPr>{ppr}</w:pPr><w:r><w:rPr>{rpr}</w:rPr>'
        f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>'
    )


def caption(label, text, before_table=False):
    after = 120 if before_table else 240
    return parse_xml(
        f'<w:p {nsdecls("w")}><w:pPr><w:keepNext w:val="{1 if before_table else 0}"/>'
        f'<w:spacing w:after="{after}" w:before="{60}" w:line="276" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>'
        f'{runs_xml(label + ": ", 22)}{runs_xml(text, 22, base_italic=True)}</w:p>'
    )


def label_line(text):
    return para(f"**{text}**", jc="center", after=120, before=240, keep_next=True)


class Builder:
    def __init__(self, document):
        self.doc = document
        self.body = document.element.body

    # ------------------------------------------------------------ lookups
    def find(self, start, occurrence=0):
        hits = [p for p in self.doc.paragraphs if p.text.strip().startswith(start)]
        if len(hits) <= occurrence:
            raise SystemExit(f"anchor not found: {start!r}")
        return hits[occurrence]._p

    def by_image(self, name):
        for p in self.doc.paragraphs:
            for blip in p._p.iter(qn("a:blip")):
                if self.doc.part.rels[blip.get(qn("r:embed"))].target_ref.endswith(name):
                    return p._p
        raise SystemExit(f"image not found: {name}")

    # ------------------------------------------------------------ editing
    @staticmethod
    def after(anchor, *elements):
        cursor = anchor
        for element in elements:
            cursor.addnext(element)
            cursor = element
        return cursor

    @staticmethod
    def remove(*elements):
        for element in elements:
            element.getparent().remove(element)

    @staticmethod
    def replace_text(p, markup, size=24):
        for r in list(p):
            if r.tag in (qn("w:r"), qn("w:hyperlink")):
                p.remove(r)
        for r in parse_xml(f'<w:p {nsdecls("w")}>{runs_xml(markup, size)}</w:p>'):
            p.append(r)

    def swap_heading(self, start, text, level):
        old = self.find(start)
        old.addprevious(heading(text, level))
        self.remove(old)

    @staticmethod
    def substitute(p, old, new):
        found = False
        for t in p.iter(qn("w:t")):
            if t.text and old in t.text:
                t.text = t.text.replace(old, new)
                found = True
        if not found:
            raise SystemExit(f"text not found: {old!r}")

    # ------------------------------------------------------------ figures and tables
    def picture(self, path, width):
        p = self.doc.add_paragraph()
        p.alignment = 1
        p.paragraph_format.space_after = 0
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Inches(width))
        element = p._p
        element.getparent().remove(element)
        return element

    def figure(self, path, label, text, width=6.0):
        return [self.picture(path, width), caption(label, text)]

    def media_blob(self, name):
        for rel in self.doc.part.rels.values():
            if "image" in rel.reltype and rel.target_ref.endswith(name):
                return rel.target_part.blob
        raise SystemExit(f"media not found: {name}")

    def grid(self, items, width=3.0):
        """items: (image path or bytes, label, text); laid out two per row in a borderless table."""
        import io

        rows = (len(items) + 1) // 2
        t = self.doc.add_table(rows=rows, cols=2)
        for k, (image, label, text) in enumerate(items):
            cell = t.cell(k // 2, k % 2)
            p = cell.paragraphs[0]
            p.alignment = 1
            source = io.BytesIO(image) if isinstance(image, bytes) else str(image)
            p.add_run().add_picture(source, width=Inches(width))
            cell._tc.append(caption(label, text))
        element = t._tbl
        element.getparent().remove(element)
        return element

    def table(self, rows, widths, header=True, size=20, align=None):
        total = sum(widths)
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        border = '<w:{0} w:color="000000" w:space="0" w:sz="8" w:val="single"/>'
        borders = "".join(border.format(x) for x in ("top", "left", "bottom", "right", "insideH", "insideV"))
        xml = [f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/><w:jc w:val="center"/>'
               f'<w:tblBorders>{borders}</w:tblBorders><w:tblLayout w:type="fixed"/>'
               f'<w:tblCellMar><w:top w:w="60" w:type="dxa"/><w:left w:w="90" w:type="dxa"/>'
               f'<w:bottom w:w="60" w:type="dxa"/><w:right w:w="90" w:type="dxa"/></w:tblCellMar>'
               f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>']
        for r, row in enumerate(rows):
            head = header and r == 0
            keep = '<w:keepNext w:val="1"/>' if r < len(rows) - 1 else ""
            xml.append('<w:tr><w:trPr><w:cantSplit w:val="1"/>' + ('<w:tblHeader w:val="1"/>' if head else '') + '</w:trPr>')
            for c, cell in enumerate(row):
                shade = '<w:shd w:fill="d9d9d9" w:val="clear"/>' if head else ""
                jc = "center" if (head or align is None or align[c] == "c") else ("left" if align[c] == "l" else "right")
                text = f"**{cell}**" if head and cell else str(cell)
                xml.append(
                    f'<w:tc><w:tcPr><w:tcW w:w="{widths[c]}" w:type="dxa"/>{shade}<w:vAlign w:val="center"/></w:tcPr>'
                    f'<w:p><w:pPr>{keep}<w:spacing w:after="0" w:before="0" w:line="240" w:lineRule="auto"/>'
                    f'<w:jc w:val="{jc}"/></w:pPr>{runs_xml(text, size)}</w:p></w:tc>'
                )
            xml.append("</w:tr>")
        xml.append("</w:tbl>")
        return parse_xml("".join(xml))

    def algorithm(self, lines, width=9000):
        paras = []
        for line in lines:
            indent = len(line) - len(line.lstrip(" "))
            text = line.strip()
            bold = text.startswith("Input") or text.startswith("Output")
            paras.append(
                f'<w:p><w:pPr><w:spacing w:after="0" w:before="0" w:line="240" w:lineRule="auto"/>'
                f'<w:ind w:left="{indent * 100}"/></w:pPr>{runs_xml(text, 17, MONO, base_bold=bold, raw=True)}</w:p>'
            )
        border = '<w:{0} w:color="000000" w:space="0" w:sz="8" w:val="single"/>'
        borders = "".join(border.format(x) for x in ("top", "left", "bottom", "right"))
        return parse_xml(
            f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:w="{width}" w:type="dxa"/><w:jc w:val="center"/>'
            f'<w:tblBorders>{borders}</w:tblBorders><w:tblLayout w:type="fixed"/>'
            f'<w:tblCellMar><w:top w:w="100" w:type="dxa"/><w:left w:w="140" w:type="dxa"/>'
            f'<w:bottom w:w="100" w:type="dxa"/><w:right w:w="140" w:type="dxa"/></w:tblCellMar></w:tblPr>'
            f'<w:tblGrid><w:gridCol w:w="{width}"/></w:tblGrid><w:tr><w:trPr><w:cantSplit w:val="0"/></w:trPr><w:tc>'
            f'<w:tcPr><w:tcW w:w="{width}" w:type="dxa"/><w:shd w:fill="f7f7f7" w:val="clear"/></w:tcPr>'
            + "".join(paras) + "</w:tc></w:tr></w:tbl>"
        )

    def swap_image(self, name, path, keep_width=True):
        """Replace the bytes of media/<name> and fix the drawing height to the new aspect ratio."""
        from PIL import Image

        target = None
        for rel in self.doc.part.rels.values():
            if "image" in rel.reltype and rel.target_ref.endswith(name):
                target = rel
        if target is None:
            raise SystemExit(f"media not found: {name}")
        target.target_part._blob = Path(path).read_bytes()
        width_px, height_px = Image.open(path).size
        for blip in self.body.iter(qn("a:blip")):
            if blip.get(qn("r:embed")) != target.rId:
                continue
            drawing = blip
            while drawing.tag not in (qn("wp:inline"), qn("wp:anchor")):
                drawing = drawing.getparent()
            extent = drawing.find(qn("wp:extent"))
            cx = int(extent.get("cx"))
            cy = int(cx * height_px / width_px)
            extent.set("cy", str(cy))
            for ext in drawing.iter(qn("a:ext")):
                if ext.get("cx") == str(cx):
                    ext.set("cy", str(cy))


def fmt(value):
    return f"{value:,.0f}"


def main():
    work = DOCS / "_ie401_merged.docx"
    merge_script = MERGE or Path(
        "/home/orkun/.config/Claude/local-agent-mode-sessions/skills-plugin/b6982539-c4c4-45f9-9af5-b6a8be32382d/"
        "8742d78a-80bd-4a7a-a2f7-b823a7bbe0bb/skills/docx/scripts/merge_runs.py"
    )
    merged = merge_script.exists() and subprocess.run(
        [sys.executable, str(merge_script), str(SRC), "-o", str(work)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not merged:
        shutil.copy(SRC, work)
    d = docx.Document(str(work))
    b = Builder(d)

    # ------------------------------------------------------------ cover, abstract
    b.substitute(b.find("SPRING 2026"), "SPRING 2026", "FALL 2026")
    b.replace_text(b.find("This project focuses on optimizing"), (
        "This project optimizes the vehicle distribution network of Tofaş for the 71 Fiat dealerships in Türkiye. "
        "Vehicles of five types are shipped from the Bursa plant on car carriers that carry a single vehicle type per "
        "tour, and a plan has to balance transportation cost, dealer inventory, backorders and unsold vehicles over a "
        "30-day horizon. The problem is formulated as a multi-period, multi-product inventory routing problem and "
        "solved exactly with Gurobi on small instances. For the company-scale instance, a dynamic greedy heuristic "
        "builds a feasible plan in less than a second, and simulated annealing and tabu search improve this plan "
        "through a shared neighbourhood of seven moves. In a 400-minute computational study on four instances, the "
        "metaheuristics reduce the total cost of the greedy plan by 27.6% to 63.0%, and extended 1000-minute runs show "
        "that further improvement is possible when more time is available. The study provides a systematic and data "
        "driven tool for distribution planning that lowers both cost and driving distance."
    ))

    # ------------------------------------------------------------ chapter 1
    b.replace_text(b.find("To achieve these objectives"), (
        "To achieve these objectives, we analyzed company data, examined relevant literature, and developed solution "
        "approaches at three levels. A mixed-integer linear programming (MILP) model formulates the routing, transfer "
        "and inventory decisions exactly and is solved with Gurobi Optimizer on small instances. Because the exact "
        "model cannot be solved at the company scale, a dynamic greedy heuristic constructs a feasible 30-day "
        "distribution plan within seconds, and two metaheuristics, simulated annealing (SA) and tabu search (TS), "
        "improve that plan by searching a common neighbourhood of seven moves."
    ))
    b.replace_text(b.find("The mathematical model serves as a first step"), (
        "The first part of the project (IE 401) established the mathematical model and the constructive heuristic. "
        "This report extends that work with the metaheuristic layer and a computational study on the company-scale "
        "instance and three variants of it. Starting from the greedy plan, the best metaheuristic lowers the total "
        "monthly cost of the original 71-dealer instance by 27.6% within 400 minutes and by 29.1% within 1000 "
        "minutes. Across the four instances, the improvement over the greedy plan ranges from 27.6% to 63.0% in 400 "
        "minutes, and the best algorithm depends on how tight the fleet is relative to demand."
    ))
    b.replace_text(b.find("This report is divided into seven chapters"), (
        "This report is divided into eight chapters. Chapter 1 introduces the project. Chapter 2 presents general "
        "information about Tofaş, and Chapter 3 examines the current system and its main issues. Chapter 4 defines "
        "the problem. Chapter 5 reviews the literature and explains the solution approach: the mathematical model, "
        "the dynamic greedy heuristic and the two metaheuristics. Chapter 6 presents the computational results, "
        "Chapter 7 discusses the implementation, and Chapter 8 concludes the study."
    ))

    # ------------------------------------------------------------ chapter 4
    b.substitute(b.find("The aim of the model is to minimize"),
                 " Additionally, a small penalty is applied for tour time to encourage more efficient route planning decisions.",
                 "")
    p = b.find("The performance of the system is evaluated")
    b.substitute(p, "Therefore, heuristic approaches based on demands and routing are also considered to obtain "
                    "high-quality solutions within reasonable computation times.",
                 "Therefore, a constructive heuristic and two metaheuristics, simulated annealing and tabu search, are "
                 "developed to obtain high-quality solutions for the company-scale instance within reasonable "
                 "computation times.")
    for t in d.tables:
        for row in t.rows:
            if row.cells[0].text.strip().startswith("IE 366"):
                b.replace_text(row.cells[1].paragraphs[0]._p,
                               "Identifying the VRP model; greedy construction, simulated annealing and tabu search")

    # ------------------------------------------------------------ 5.1 literature placeholder
    b.after(b.find("Sustainability has become a key concern"),
            heading("iv) Metaheuristics and Their Parameter Settings", 4),
            para("[[Placeholder, to be written after the team's literature consultation. Planned content: the "
                 "simulated annealing temperature rule of Duman and Duman [13]; the source used for the tabu search "
                 "neighbourhood size and tenure [14]; the adaptive operator scoring of Ropke and Pisinger [15].]]"))

    # ------------------------------------------------------------ 5.2 modeling approach
    b.after(b.find("In this research, a MILP formulation"),
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
                 "is devoted to parameter selection and run length."))

    # ------------------------------------------------------------ 5.3 i) objective without the tour-time penalty
    b.swap_image("image23.png", Path(__file__).with_name("assets") / "objective_402.png")

    # ------------------------------------------------------------ 5.3 ii) dynamic greedy
    b.swap_heading("ii) Heuristic Solution- Greedy Algorithm", "ii) Heuristic Solution - Dynamic Greedy Algorithm", 4)
    b.replace_text(b.find("The heuristic algorithm below is designed"), (
        "The heuristic algorithm below is designed to quickly generate feasible vehicle distribution plans for large "
        "instances (like our original problem) where the exact solution is impossible to compute. It is the dynamic "
        "version of the greedy algorithm developed in IE 401: vehicle types compete for trucks according to their "
        "unmet demand, and no surplus vehicles are left at dealers."
    ))
    b.replace_text(b.find("The algorithm takes the transportation network parameters"), (
        "The algorithm takes the transportation network parameters (travel times, costs), the truck capacity for each "
        "vehicle type, and the daily forecasted dealer demands. It operates day by day. On each day, every truck that "
        "is not locked by an earlier tour is considered in turn. The truck is assigned to the vehicle type with the "
        "largest total unmet demand over all dealers on that day, so trucks are shared among vehicle types according "
        "to their shortages instead of serving the types one after another."
    ))
    b.replace_text(b.find("If available trucks and critical dealers exist"), (
        "The route is constructed using a nearest-neighbour rule. Starting from the factory, the truck moves to the "
        "closest dealer that is short of the assigned type and unloads the smaller of its remaining load and the "
        "dealer's shortage. It then continues to the next closest short dealer until it is empty or no short dealer "
        "remains. Since a truck never unloads more than a dealer needs, no surplus vehicles accumulate at the last "
        "dealer of a route. When the tour is complete, the truck returns to the factory and is locked for the "
        "duration of the tour, and dealer inventories are reduced by the day's demand."
    ))
    greedy_tail = b.find("As output, it gives daily satisfied demand")
    b.replace_text(greedy_tail, (
        "As output, the algorithm gives the list of tours (day, truck, vehicle type, visited dealers and the quantity "
        "dropped at each), which is evaluated with the objective function of the mathematical model. On the original "
        "instance, the dynamic greedy plan delivers 4,306 of the 4,351 vehicles demanded in the month with 608 tours, "
        "leaves no vehicle unsold, and costs 2,076,165. For the entire pseudocode, please check the “Appendix” "
        "(Figure A.1)."
    ))

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
        "Input: starting plan s0, sample size n, tenure θ,",
        "       time budget τmax",
        "s ← s0;  best ← s0;  tabu list ← empty",
        "while elapsed time τ < τmax:",
        "    N ← n neighbours of s, each made by a move drawn with the move weights",
        "    A ← { s' in N : move(s') is not tabu  or  f(s') < f(best) }",
        "    if A is empty:  A ← N",
        "    s ← the cheapest plan in A          (even if it is worse than s)",
        "    make move(s) tabu for the next θ iterations",
        "    if f(s) < f(best):  best ← s",
        "Output: best",
    ]
    small = [
        ["Trucks", "Exact model (Gurobi, 600 s)", "Dynamic greedy", "Best metaheuristic (120 s)", "Gap to exact"],
        ["4", "29,742", "78,073", "52,176 (SA 500)", "75.4%"],
        ["6", "19,133", "57,340", "23,442 (SA 500)", "22.5%"],
        ["8", "18,157", "47,474", "20,107 (SA 500)", "10.7%"],
    ]

    b.after(
        greedy_tail,
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
             "its type and the tours, dealers and positions it changes; once performed, the move cannot be repeated "
             "for a number of iterations called the tenure. A tabu move is still allowed if it produces a plan better "
             "than the best plan found so far (aspiration criterion), and if all sampled neighbours are tabu, the "
             "cheapest of them is taken so that the search never stalls (Figure 5.4)."),
        para("The number of sampled neighbours was set relative to the number of dealers: 70 is roughly one "
             "neighbour per dealer of the original instance, and 35 roughly one per two dealers. The tenures 25 and "
             "625 are two members of a geometric series that differ by a factor of 25 [[[14], source to be "
             "confirmed]]. Section 6.1 explains which of these settings were used in the main study."),
        b.algorithm(ts_code),
        caption("Figure 5.4", "Pseudocode of tabu search"),
    )

    # ------------------------------------------------------------ 5.4 captions
    b.substitute(b.find("Figure 2.1: Console Outputs"), "Figure 2.1", "Figure 5.5")
    p = b.find("By visualizing the daily dispatch schedules")
    b.substitute(p, "Figures 2.1 to 2.4", "Figures 5.6 to 5.9")
    days = b.grid([
        (b.media_blob("image28.png"), "Figure 5.6", "Routing Schedule on Day 0"),
        (b.media_blob("image27.png"), "Figure 5.7", "Routing Schedule on Day 1"),
        (b.media_blob("image22.png"), "Figure 5.8", "Routing Schedule on Day 2"),
        (b.media_blob("image30.png"), "Figure 5.9", "Routing Schedule on Day 3"),
    ])
    for run in list(p.iter(qn("w:r"))):
        if run.find(qn("w:drawing")) is not None:
            run.getparent().remove(run)
    b.remove(b.find("Figure 3.1"), b.find("Figure 3.3"))
    b.after(p, days)

    # ------------------------------------------------------------ 5.5 and 5.6
    heur_ver = b.find("For the verification and the validation of the heuristic model")
    b.replace_text(heur_ver, (
        "For the verification and the validation of the heuristic model, the logic is implemented in a Python script, "
        "and the algorithm was applied to exactly the same problem (10 nodes, 5 trucks, 30 days instance). As seen in "
        "the plots given in the “Appendix” section (Figures A.2 to A.5), each truck takes the vehicle type with the "
        "largest shortage on its dispatch day, visits the nearest short dealers in sequence, returns to the factory "
        "and stays locked until its tour is complete. By using the GitHub link in the “Appendix”, all 30 days' plots "
        "can be checked for both the exact and the heuristic algorithms."
    ))
    b.after(
        heur_ver,
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
    )

    # ------------------------------------------------------------ chapter 6 intro and 6.1 i-iii
    b.replace_text(b.find("In this chapter, the computational experiments"), (
        "In this chapter, the computational experiments conducted to evaluate the performance, scalability, and "
        "parameters of the models are provided. The experiments are divided into three parts: a sensitivity analysis "
        "of the exact model and the greedy heuristic on small instances, a computational study of the metaheuristics "
        "on the company-scale instance and three variants of it, and extended runs for the settings that were still "
        "improving at the end of that study. All exact models were solved using Gurobi Optimizer in Python with a "
        "time limit of 600 seconds. The heuristic and the metaheuristics are implemented in Python; every "
        "metaheuristic run uses a single processor core of a 16-core workstation, so that runs executed in parallel "
        "do not share computing power, and all runs use the same random seed (44). The whole script is available on "
        "GitHub, and is mentioned in the “Appendix” section."
    ))
    b.substitute(b.find("The results show a clear trade off"), "Table 1.1", "Table A.1")
    b.substitute(b.find("In the results, for the truck numbers"), "Table 2.1", "Table A.2")
    heur_sens = b.find("To compare with the exact model, the heuristic algorithm")
    b.replace_text(heur_sens, (
        "To compare with the exact model, the dynamic greedy heuristic was tested across the same truck numbers, and "
        "its plans were evaluated with the same objective function. The heuristic solves every scenario in less than "
        "0.001 seconds, and its objective values are 78,073, 57,340 and 47,474 for 4, 6 and 8 trucks. As in the exact "
        "model, more trucks lead to fewer backorders and a lower total cost, and no vehicle is left unsold in any "
        "scenario. However, the greedy plans remain 2.6 to 3.0 times as expensive as the exact solutions, because of "
        "both backorders and longer routes: each truck only serves dealers that are short on its dispatch day, in "
        "nearest-neighbour order (the full table is available as Table A.3 in the “Appendix”). This gap is what the "
        "metaheuristics address; their results on the same instances are given in Section 5.6."
    ))

    # ------------------------------------------------------------ 6.1 iv-ix (company-scale study)
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
    extended_rows = [
        ["Instance, start", "Setting", "Best in 400 min", "Best in 1000 min", "Improvement over greedy (400 → 1000 min)", "First improvement in the 1000-min run"],
        ["Original, greedy", "SA 500", "1,569,090", "1,471,750", "24.4% → 29.1%", "immediately"],
        ["Original, greedy", "SA T75", "2,076,165", "1,702,077", "0.0% → 18.0%", "minute 871"],
        ["Original, random", "SA T75", "4,716,117", "2,573,460", "−127.2% → −24.0%", "immediately"],
        ["Growth, greedy", "SA T25", "20,819,112", "13,471,818", "28.8% → 53.9%", "minute 121"],
        ["Growth, greedy", "SA T75", "24,531,505", "15,152,473", "16.1% → 48.2%", "minute 393"],
        ["Growth, greedy", "SA 500", "22,814,942", "20,105,295", "22.0% → 31.3%", "immediately"],
    ]
    pilot_rows = [
        ["Cost component", "Dynamic greedy", "TS 70/625 (400 min)", "SA 500 (1000 min)"],
        ["Transportation", "1,902,665", "1,418,288", "1,174,130"],
        ["Holding", "0", "61,560", "267,620"],
        ["Backorder", "173,500", "22,500", "30,000"],
        ["Unsold vehicles", "0", "0", "0"],
        ["**Total**", "**2,076,165**", "**1,502,348**", "**1,471,750**"],
        ["Vehicles delivered (of 4,351)", "4,306", "4,341", "4,327"],
        ["Backorders (vehicle-days)", "347", "45", "60"],
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

    b.after(
        heur_sens,
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
             "35 or 70 sampled neighbours and a tenure of 25 or 625. All nine runs were executed in parallel. Within "
             "100 minutes, the four tabu settings achieved the largest improvements (23.3% to 24.1%), followed by "
             "SA 50 (21.9%) and SA 500 (14.4%), while SA T75, T50 and T25 did not improve the greedy plan at all: at "
             "these temperatures, the current plan was still moving above the cost of the greedy plan when the time "
             "limit was reached (full results in Table A.5 and Figures A.6 to A.8 in the “Appendix”)."),
        para("Based on this screening, the number of settings was reduced from nine to six. Among the tabu settings, "
             "70 sampled neighbours performed better than 35 for both tenures, so the two settings with 70 "
             "neighbours were kept; since they share the same sample size, their comparison isolates the effect of "
             "the tenure. Among the SA settings, T50 was removed: it lies between T75 and T25, which already "
             "represent the two ends of the acceptance range of the paper, and the three behaved identically in the "
             "screening. The cold settings SA 500 and SA 50 were kept because they were the only SA settings that "
             "improved the plan within the screening time."),
        heading("vi) Design of the Main Experiment", 4),
        para("The main experiment runs the six selected settings (SA T75, SA T25, SA 500, SA 50, TS 70/25 and TS "
             "70/625) on the four instances of Table 6.1, from two different starting plans, with a time limit of "
             "400 minutes per run: 48 runs in total, executed in parallel on separate cores. The first starting plan "
             "is the dynamic greedy plan, which shows how much the metaheuristics add to the constructive heuristic. "
             "The second is a random plan, built by drawing tours with random days, trucks, vehicle types, dealers "
             "and quantities while respecting truck capacities and truck availability; it shows whether an algorithm "
             "can find good plans without a good starting point. The random plans are very expensive (between 13.9 "
             "and 96.4 million) because they ignore demand. The move weights of Figure 5.2 and the temperatures of "
             "Table 5.2 are the same in every run."),
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
             "instance had not improved at all. To measure how much these settings lose to the time limit, six of "
             "them were run again with a time limit of 1000 minutes: SA 500 and SA T75 on the original instance, "
             "SA T75 from the random start on the original instance, and SA T25, SA T75 and SA 500 on the growth "
             "instance. Each extended run is a new run whose cooling schedule is spread over 1000 minutes, so it is "
             "not a continuation of the 400-minute run. Table 6.4 compares the results; the time plots, together with "
             "the 400-minute curves of the other settings, are given in Figures A.25 to A.30 in the “Appendix”."),
        caption("Table 6.4", "Results of the 1000-minute runs", before_table=True),
        b.table(extended_rows, [1550, 950, 1500, 1500, 1900, 1700], align="llrrcc", size=19),
        para(""),
        para("With 1000 minutes, SA 500 finds the best plan for the original instance (1,471,750, 29.1% below the "
             "greedy plan), better than the best 400-minute result of TS 70/625 (1,502,348). On the growth instance "
             "the gain is much larger: SA T25 reaches 13.47 million, 53.9% below the greedy plan and 35% below its "
             "own 400-minute result, mainly by reducing backorders from 16.0 to 6.8 million (13,625 vehicle-days "
             "instead of 32,063). SA T75 follows the same pattern and reaches 15.15 million. Because the cooling is "
             "spread over a longer time, the hot settings start to improve later in the extended runs (SA T75 on "
             "the original instance only after 871 minutes), but they then improve faster and further. From the "
             "random start, SA T75 reduces its distance to the greedy plan from 127% to 24%, but does not reach it. "
             "These results show that the hot SA settings are limited by the time budget rather than by their "
             "design, and that their position in Tables 6.2 and 6.3 improves with longer runs."),
    )

    # ------------------------------------------------------------ 6.2 discussion
    old = b.find("The exact algorithm is not applicable to the original problem")
    first = old
    discussion = [
        "The exact model cannot be applied to the company-scale instance, so the proposed solution is the "
        "combination of the dynamic greedy heuristic and a metaheuristic. The greedy plan is available in less than "
        "a second and is already feasible, with no unsold vehicles; the metaheuristics then reduce its cost by 27.6% "
        "to 63.0% within 400 minutes, depending on the instance.",
        "No single setting is best on every instance, and the results are explained by how tight the fleet is "
        "relative to demand. When the fleet is sufficient (original and full-slack instances), the greedy plan "
        "already has a good structure, and methods that accept few or no worse plans (TS and the cold SA settings) "
        "improve it quickly; TS 70/625 gives the best 400-minute result on both. When the fleet is tight (growth and "
        "far-tight instances), backorders dominate the cost and improving the plan requires rearranging many tours; "
        "only the SA settings that accept worse plans achieve this, and their advantage grows with time.",
        "The hot SA settings are limited mainly by time. Their temperature has to fall before they start improving, "
        "which in 400 minutes happened late or not at all on the original instance. The extended runs show that "
        "with 1000 minutes, SA 500 gives the best plan for the original instance and SA T25 more than halves the "
        "cost of the growth scenario.",
        "The random-start experiments show how robust the methods are. SA T25 is the most robust setting: from a plan "
        "that ignores demand, it ends below the greedy plan on three of the four instances. TS and SA 50 depend "
        "strongly on a good starting plan. For practical use, the greedy plan is therefore the recommended starting "
        "point.",
        "For TS, the tenure has little influence: TS 70/25 and TS 70/625 differ by at most 2.2% on every instance. "
        "Because the tabu attribute is the exact move and the neighbourhood is very large, the same move is rarely "
        "sampled again; at most 2% of the sampled neighbours were tabu in the main study. The number of sampled "
        "neighbours (70 against 35) had a clearer effect in the screening.",
        "Based on these results, the recommended procedure is to build the dynamic greedy plan and then to run TS "
        "with 70 neighbours when only a short computation time is available or the fleet is sufficient, and SA "
        "starting from T25 or 500 for an overnight or longer run, especially when the fleet is tight relative to "
        "demand. Since every setting runs on a single core, several settings can be run in parallel and the best "
        "plan kept.",
        "All results are based on a single random seed and on one weight vector obtained from a single pilot run. "
        "Repeated runs with different seeds would show how much of the difference between close settings, such as "
        "the near-ties on the full-slack instance, is due to chance.",
    ]
    stale = [old]
    nxt = old.getnext()
    while nxt is not None and not (nxt.tag == qn("w:p") and "6.3)" in "".join(t.text or "" for t in nxt.iter(qn("w:t")))):
        stale.append(nxt)
        nxt = nxt.getnext()
    anchor = first.getprevious()
    b.remove(*stale)
    b.after(anchor, *[para(text) for text in discussion])

    # ------------------------------------------------------------ 6.3 pilot
    pilot_start = b.find("For pilot implementation, we run the heuristic")
    stale = [pilot_start]
    nxt = pilot_start.getnext()
    while nxt is not None and "Chapter 7" not in "".join(t.text or "" for t in nxt.iter(qn("w:t"))):
        stale.append(nxt)
        nxt = nxt.getnext()
    anchor = pilot_start.getprevious()
    b.remove(*stale)
    b.after(
        anchor,
        para("The original instance has the size of the real Fiat network (71 dealers, 80 trucks, 5 vehicle types, "
             "30 days) and serves as the pilot implementation of the approach. Table 6.5 compares the greedy plan "
             "with the best plans found in 400 and in 1000 minutes."),
        caption("Table 6.5", "Cost components of the greedy plan and the best plans on the original instance", before_table=True),
        b.table(pilot_rows, [3000, 1900, 2100, 2000], align="lrrr"),
        para(""),
        para("Both improved plans cut the backorder cost by more than 80% and drive considerably less: the "
             "transportation cost, which is proportional to the distance driven, falls by 25.5% with TS and by 38.3% "
             "with SA. The two plans reach this in different ways. The TS plan keeps inventories low and delivers "
             "4,341 of the 4,351 vehicles demanded. The SA plan delivers vehicles earlier, so it holds more stock at "
             "the dealers, but it needs much less driving. In both plans no vehicle remains unsold and no dealer "
             "exceeds its storage capacity. Since every kilometre saved also reduces fuel consumption and emissions, "
             "the shorter routes support the environmental goal stated in Section 4.5."),
    )

    # ------------------------------------------------------------ chapter 7
    b.after(b.find("Model deployment involves the regular collection"),
            para("In practice, we suggest a monthly planning cycle. At the beginning of each month, the demand "
                 "forecasts, dealer inventories and truck availability are collected; the dynamic greedy heuristic "
                 "produces a first plan; a metaheuristic improves this plan overnight; and planners review the "
                 "resulting tours before dispatching. When forecasts or inventories change significantly during the "
                 "month, the remaining days can be re-planned in the same way, since the metaheuristics can start "
                 "from any feasible plan."))
    b.replace_text(b.find("The implementation also requires appropriate software"), (
        "The implementation requires a standard computer with Python. The dynamic greedy heuristic and the "
        "metaheuristics do not need a commercial solver; a Gurobi license is only needed if the exact model is used "
        "for small what-if analyses. The greedy plan is produced in less than a second, and a metaheuristic run of 400 "
        "minutes fits into an overnight batch. Since each run uses a single processor core, several settings can be "
        "run in parallel on a multi-core workstation and the best plan selected, so the approach is technically and "
        "financially feasible with a low investment. Basic training is needed for planners to prepare the input data "
        "and to interpret the resulting tours."
    ))
    b.replace_text(b.find("Despite this study including a full scale implementation"), (
        "The main difficulty of implementation is data. The instance used in this study was generated from company "
        "averages (daily demand per vehicle type, truck capacities and fleet size) and approximate road distances. "
        "Before deployment, the model must be fed with actual dealer locations, road distances, storage capacities "
        "and forecasts, and its plans must be compared with the routes currently used by Tofaş. A second difficulty "
        "is computation time: good plans for large or tight instances require several hours, so the planning cycle "
        "must leave time for an overnight run. Finally, demand forecasts change during the month, which requires "
        "re-planning on a rolling basis."
    ))

    # ------------------------------------------------------------ chapter 8
    b.replace_text(b.find("In this study, we examined the inventory distribution system"), (
        "In this study, we examined the vehicle distribution system of Tofaş for the Fiat dealerships in Türkiye. In "
        "IE 401, the problem was formulated as a multi-period, multi-product inventory routing problem and solved with "
        "Gurobi on small instances, and a greedy heuristic was developed for large instances. In IE 402, the greedy "
        "heuristic was revised into a dynamic version that assigns trucks according to unmet demand and leaves no "
        "unsold vehicles, and two metaheuristics, simulated annealing and tabu search, were developed on a shared "
        "neighbourhood of seven moves. Their parameters were set with methods from the literature: the initial and "
        "final temperatures of SA follow the acceptance-probability rule of Duman and Duman [13], and the move "
        "weights were derived from the scoring scheme of Ropke and Pisinger [15] in a single pilot run."
    ))
    impact = b.find("While the primary objective of this study is to minimize costs")
    b.replace_text(impact, (
        "While the primary objective of this study is to minimize costs, its environmental and social impacts are "
        "also important. Shorter routes reduce fuel consumption and carbon emissions, and fewer backorders mean that "
        "customers receive their vehicles with shorter delays. The approach does not raise legal issues, since it only "
        "uses operational data of the company, and it does not affect the safety of drivers as long as the travel "
        "times used in the model respect the legal driving limits."
    ))
    impact.addprevious(para(
        "On four company-scale instances, the metaheuristics reduced the cost of the greedy plan by 27.6% to 63.0% in "
        "400 minutes, and by up to 53.9% on the growth scenario in 1000 minutes. On the original instance, the best "
        "plan cuts the transportation cost by 38.3% and the backorder cost by 83% compared with the greedy plan. Tabu "
        "search was the best choice for short run times and sufficient fleets, while simulated annealing with higher "
        "starting temperatures was the only method able to improve plans for tight fleets and from random starting "
        "plans, provided that it was given enough time."
    ))
    b.replace_text(b.find("The current work is limited to model development"), (
        "The current work has some limitations that point to future work. The results are based on a single random "
        "seed and on an instance generated from company averages; repeating the runs with several seeds and "
        "validating the plans with real dealer and route data would strengthen the conclusions. Several settings were "
        "still improving at the end of their runs, so longer runs, parallel runs and hybrid strategies (for example, "
        "a fast tabu search phase followed by simulated annealing) could improve the results further. The move "
        "weights could also be updated during the search instead of being fixed after a pilot run. Finally, the "
        "approach could be extended with dealer-to-dealer transfers in the neighbourhood, stochastic demand with "
        "re-planning during the month, and an explicit CO₂ emission term, bringing the analysis closer to the real "
        "operating conditions of Tofaş."
    ))

    # ------------------------------------------------------------ references
    b.remove(b.find("In this part, there should be a list of references"))
    b.after(b.find("[12]"),
            para("[13] T. Duman and E. Duman, “Solving a new application of asymmetric TSP by modified migrating birds "
                 "optimization algorithm,” *Evolutionary Intelligence*, vol. 17, no. 3, 2024.", after=240, before=240),
            para("[14] [[Source of the tabu search neighbourhood size and tenure, to be confirmed.]]", after=240, before=240),
            para("[15] S. Ropke and D. Pisinger, “An adaptive large neighborhood search heuristic for the pickup and "
                 "delivery problem with time windows,” *Transportation Science*, vol. 40, no. 4, pp. 455–472, 2006.",
                 after=240, before=240))

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
    b.after(pseudo_old, b.algorithm(greedy_code))
    b.remove(pseudo_old)
    b.replace_text(b.find("Figure 1.1: Console Outputs"), "Figure A.1: *Pseudocode of the Dynamic Greedy Heuristic*", 22)
    b.replace_text(b.find("Heuristic Solution Plots"), "**Dynamic Greedy Heuristic Solution Plots**")
    for name, day in (("image25.png", 0), ("image29.png", 1), ("image26.png", 2), ("image24.png", 3)):
        b.swap_image(name, PLOTS / "heuristic_dynamic" / f"day_{day:02d}.png")
    p = b.find("Figure 4.1:")
    b.substitute(p, "Figure 4.1", "Figure A.2")
    b.substitute(p, "Figure 4.2", "Figure A.3")
    p = b.find("Figure 4.3:")
    b.substitute(p, "Figure 4.3", "Figure A.4")
    b.substitute(p, "Figure 4.4", "Figure A.5")
    b.substitute(b.find("Table 1.1:"), "Table 1.1", "Table A.1")
    b.substitute(b.find("Table 2.1:"), "Table 2.1", "Table A.2")
    heur_caption = b.find("Table: Truck Numbers and Objective Value, Heuristic")
    b.replace_text(heur_caption, "Table A.3: *Truck Numbers and Objective Value, Dynamic Greedy Heuristic*")
    heur_table = None
    for t in d.tables:
        if t.cell(0, 0).text.strip().startswith("# of") and "Time" in t.cell(0, t._column_count - 1).text:
            heur_table = t
    values = [("78073.19", "16073.19", "62000.00", "0", "0.0001"),
              ("57339.80", "28339.80", "29000.00", "0", "0.0001"),
              ("47473.94", "33973.94", "13500.00", "0", "0.0001")]
    for r, row in enumerate(values, start=1):
        for c, value in enumerate(row, start=1):
            b.replace_text(heur_table.cell(r, c).paragraphs[0]._p, value, 22)
    heur_table_end = heur_table._tbl
    for shd in heur_table_end.iter(qn("w:shd")):
        if shd.get(qn("w:fill")) in ("ffff00", "00ff00"):
            shd.set(qn("w:fill"), "auto")
    for start in ("Table A.1:", "Table A.2:", "Table A.3:"):
        ppr = b.find(start).find(qn("w:pPr"))
        ppr.insert(0, parse_xml(f'<w:keepNext {nsdecls("w")} w:val="1"/>'))

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
        label_line("Test Instances"),
        caption("Table A.4", "The ten instance variants designed for the study (the instances of the main study are 00, 02, 07 and 09)", before_table=True),
        b.table(pool, [600, 3000, 3100, 1000, 1300], align="clccc"),
        para(""),
        label_line("Algorithm Screening (100 minutes, original instance, greedy start)"),
        caption("Table A.5", "Screening results of the nine settings", before_table=True),
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
    for folder, title in ((ext / "greedy" / "00_original", "Original instance, greedy start"),
                          (ext / "greedy" / "02_growth_no_fleet", "Growth, same fleet instance, greedy start"),
                          (ext / "random" / "00_original", "Original instance, random start")):
        appendix += b.figure(folder / "cost_vs_time_combined.png", f"Figure A.{n}",
                             f"Best objective over time, {title}; extended settings 1000 minutes, others 400 minutes")
        n += 1
    for folder, title in ((ext / "greedy" / "00_original", "Original instance, greedy start"),
                          (ext / "greedy" / "02_growth_no_fleet", "Growth, same fleet instance, greedy start"),
                          (ext / "random" / "00_original", "Original instance, random start")):
        appendix += b.figure(folder / "curr_cost_vs_time_rescaled.png", f"Figure A.{n}",
                             f"Current objective of the 1000-minute runs on the best-cost axis, {title}")
        n += 1
    b.after(heur_table_end, *appendix)

    # ------------------------------------------------------------ fields and save
    settings = d.settings.element
    if settings.find(qn("w:updateFields")) is None:
        settings.append(parse_xml(f'<w:updateFields {nsdecls("w")} w:val="true"/>'))
    d.save(str(OUT))
    work.unlink()
    print("saved", OUT)


if __name__ == "__main__":
    main()
