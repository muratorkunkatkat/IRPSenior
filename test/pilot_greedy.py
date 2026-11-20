"""Pilot run of the greedy heuristic on the real scale instance (71 dealers, 80 trucks, 5 car types, 30 days).

Prints the same report as test1_heuristic.py, using the per-dealer quantities of the
greedy plan, and saves the end-of-month report and the day 4 / day 29 sections as
console-style images for the report.

    python test/pilot_greedy.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "test"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from PIL import Image, ImageDraw, ImageFont

import analysis_long as study
from analysis_grid import build_instance
from calibrate_metaheuristics import load_instance
from heuristic.search_problem import Solution, build_initial_trips, clone_trips, evaluate

OUT = ROOT / "plots" / "report"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def report(data, trips):
    types = data["C"]
    days = data["T"]
    dealers = data["nodes_no_factory"]
    F = data["F"]
    lines = {"daily": {}, "month": []}
    for t in days:
        rows = [f"Day {t}"]
        for c in types:
            demanded = sum(F[j][c][t] for j in dealers)
            delivered = sum(sum(trip.qty) for trip in trips if trip.day == t and trip.car_type == c)
            rate = 100.0 * delivered / demanded if demanded else 0.0
            rows.append(f"Car Type {c}: Demanded = {demanded:>4} | Delivered = {delivered:>4} | "
                        f"Satisfaction = {rate:>5.1f}%")
        lines["daily"][t] = rows

    month = ["END OF MONTH INVENTORY REPORT", "=" * 60]
    total_unmet = total_unsold = 0
    for c in types:
        unmet = unsold = 0
        for i in dealers:
            demanded = sum(F[i][c][t] for t in days)
            delivered = sum(q for trip in trips if trip.car_type == c
                            for stop, q in zip(trip.stops, trip.qty) if stop == i)
            unsold += max(0, delivered - demanded)
            unmet += max(0, demanded - delivered)
        month.append(f"Car Type {c}: Unmet Demand = {unmet:>5} | Unsold = {unsold:>5}")
        total_unmet += unmet
        total_unsold += unsold
    month += ["-" * 60, f"Unmet Demand = {total_unmet}, Unsold Cars = {total_unsold}", "=" * 60]
    lines["month"] = month
    return lines


def render(rows, path, size=15):
    font = ImageFont.truetype(FONT, size)
    pad, gap = 10, 6
    width = max(int(font.getlength(row)) for row in rows) + 2 * pad
    height = len(rows) * (size + gap) + 2 * pad - gap + 4
    image = Image.new("RGB", (width, height), (25, 26, 27))
    draw = ImageDraw.Draw(image)
    for k, row in enumerate(rows):
        draw.text((pad, pad + k * (size + gap)), row, font=font, fill=(204, 204, 204))
    image.save(path)


def main():
    base, _path = load_instance()
    data, _meta = build_instance(base, study.SPEC_BY_NAME["00_original"])
    started = time.perf_counter()
    trips = build_initial_trips(data)
    seconds = time.perf_counter() - started
    cost, _aux = evaluate(Solution(data, clone_trips(trips)))
    lines = report(data, trips)

    print(f"Greedy plan in {seconds:.3f} s")
    print(f"OBJ. VALUE (total cost): {cost.total:,.2f}  transport {cost.transport:,.2f}  "
          f"holding {cost.holding:,.2f}  backorder {cost.backorder:,.2f}  unsold {cost.unsold:,.2f}")
    print(f"TOTAL DISPATCHES: {sum(1 for trip in trips if trip.stops)}  delivered {cost.delivered}")
    for t in data["T"]:
        print("\n".join(lines["daily"][t]))
    print("\n".join(lines["month"]))

    OUT.mkdir(parents=True, exist_ok=True)
    render(lines["month"], OUT / "pilot_end_of_month.png")
    render(lines["daily"][4], OUT / "pilot_day_04.png")
    render(lines["daily"][29], OUT / "pilot_day_29.png")


if __name__ == "__main__":
    main()
