"""Results document and the updated speech, written after the 100-minute grid."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "test") not in sys.path:
    sys.path.insert(0, str(ROOT / "test"))

import analysis_long as grid

OUT = grid.OUT


def _title(spec):
    return grid.ENGLISH_TITLES.get(spec["name"], spec["name"])
DOC_PATH = OUT / "results.docx"
SPEECH_PATH = ROOT / "docs" / "speech2.docx"

# Previous 30-minute study, build-rate weights, greedy start. Context only.
PREV = {
    "00_original": {
        "sa_T75": 2076165, "sa_T25": 2076165, "sa_500": 1887095, "sa_50": 1642568,
        "tabu_n70_t25": 1632472, "tabu_n70_t625": 1626820,
        "why": (
            "Orijinal problem taban. Otuz dakikada formül sıcaklıkları greedy plandan hiç ayrılmadı. "
            "İnişi tabu 70 ve SA 50 yaptı."
        ),
    },
    "02_growth_no_fleet": {
        "sa_T75": 29242970, "sa_T25": 27684017, "sa_500": 27205891, "sa_50": 28779523,
        "tabu_n70_t25": 29122725, "tabu_n70_t625": 29120576,
        "why": (
            "Otuz dakikalık çalışmada T25 ve T50 inişe geçmiş, SA 500'ün önüne geçememişti. "
            "T25'in en iyi planı son adımdaydı. T50'nin bütün inişi son on dakikadaydı. "
            "T50 bu turda yok. Soru, T25'e yetmiş dakika daha verilince o geçişin olup olmadığıdır."
        ),
    },
    "07_full_slack_thin": {
        "sa_T75": 1188111, "sa_T25": 967367, "sa_500": 969935, "sa_50": 947451,
        "tabu_n70_t25": 951763, "tabu_n70_t625": 946510,
        "why": (
            "Aynı 71 bayi, filo bol, talep seyrek. Otuz dakikada tabu 70/625 ile SA 50 öndeydi. "
            "T25 son on dakikada hızlanıp liderin 21 bin lira yakınına geldi. T75 ve T50 da inişe yeni başlamıştı."
        ),
    },
    "09_far_tight": {
        "sa_T75": 1314028, "sa_T25": 774679, "sa_500": 794831, "sa_50": 1038299,
        "tabu_n70_t25": 1073677, "tabu_n70_t625": 1055967,
        "why": (
            "Uzak iskelet ve dar filo. Otuz dakikada T25 zaten öndeydi, SA 500 yirminci dakikanın lideriydi. "
            "Bu problem, sıralamanın süre uzayınca durup durmadığına bakar."
        ),
    },
}

SPEAKERS = ["Ece", "Zeynep", "Bahar", "Orkun", "Onur"]


def _en(value, digits=0):
    return f"{value:,.{digits}f}"


def _tr(value, digits=0):
    return f"{value:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(value, digits=1):
    return _tr(value, digits) + "%"


def _load_weights():
    if not grid.WEIGHTS_PATH.exists():
        return None
    return json.loads(grid.WEIGHTS_PATH.read_text())


def _rows(start, spec):
    found = []
    for algo in grid.ALGOS:
        summary = grid._load_summary(start, spec["name"], algo["key"])
        if summary is None:
            return None
        found.append(summary)
    return found


def _complete():
    return all(_rows(start, spec) is not None for start in ("greedy", "random") for spec in grid.SPECS)


def _done_count():
    total = 0
    for start in ("greedy", "random"):
        for spec in grid.SPECS:
            for algo in grid.ALGOS:
                if grid._summary_path(start, spec["name"], algo["key"]).exists():
                    total += 1
    return total


def _best(rows):
    return min(rows, key=lambda row: row["best"])


def _ordered(rows):
    return sorted(rows, key=lambda row: row["best"])


def _tie(a, b):
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= 0.005 * scale


def _bill(cost):
    total = cost["total"] or 1.0
    back = 100.0 * cost["backorder"] / total
    transport = 100.0 * cost["transport"] / total
    if back >= 70:
        return f"faturanın {_pct(back)} kadarı karşılayamama"
    if transport >= 70:
        return f"faturanın {_pct(transport)} kadarı nakliye"
    return f"fatura karışık: nakliye {_pct(transport)}, karşılayamama {_pct(back)}"


def _piece_sentence(start_cost, best):
    pieces = (
        ("nakliye", start_cost["transport"] - best["transport"]),
        ("elde tutma", start_cost["holding"] - best["holding"]),
        ("karşılayamama", start_cost["backorder"] - best["backorder"]),
        ("ay sonu satılamayan", start_cost["unsold"] - best["unsold"]),
        ("kapasite cezası", start_cost["capacity"] - best["capacity"]),
    )
    dropped = sorted(((name, delta) for name, delta in pieces if delta > 1), key=lambda item: -item[1])
    raised = sorted(((name, -delta) for name, delta in pieces if delta < -1), key=lambda item: -item[1])
    text = ""
    if dropped:
        text += "İnen kalemler: " + ", ".join(f"{name} {_tr(delta)}" for name, delta in dropped[:3]) + ". "
    if raised:
        text += "Yükselen kalemler: " + ", ".join(f"{name} {_tr(delta)}" for name, delta in raised[:3]) + ". "
    gap = best["delivered"] - start_cost["delivered"]
    if gap > 0:
        text += f"Teslim {int(gap)} arttı. "
    elif gap < 0:
        text += f"Teslim {int(abs(gap))} azaldı. "
    else:
        text += "Teslim aynı kaldı. "
    return text


def _clock_sentence(rows):
    if any("best_at_6000" not in row or "best_at_18000" not in row for row in rows):
        return "100. ve 300. dakika işaretleri bu koşuda yok."
    at_100m = min(rows, key=lambda row: row["best_at_6000"])
    at_300 = min(rows, key=lambda row: row["best_at_18000"])
    at_end = _best(rows)
    text = (
        f"Bu koşunun 100. dakikasındaki lider {at_100m['label']} ({_tr(at_100m['best_at_6000'])}). "
        f"300. dakikada {at_300['label']} ({_tr(at_300['best_at_18000'])}). "
        f"400. dakikada {at_end['label']} ({_tr(at_end['best'])}). "
    )
    text += "Lider değişmedi. " if at_100m["algo_key"] == at_end["algo_key"] else "Lider süre uzayınca değişti. "
    moving = []
    for row in rows:
        gained = row["initial"] - row["best"]
        if gained <= 1:
            continue
        late = row["best_at_18000"] - row["best"]
        if late / gained >= 0.15:
            moving.append(row["label"])
    text += (
        "Son 100 dakikada inişi sürenler: " + ", ".join(moving) + ". "
        if moving else
        "Son 100 dakikada sıralamayı değiştirecek bir iniş kalmadı. "
    )
    flat = [row["label"] for row in rows if row["improvement"] < 0.5]
    if flat:
        text += "Başlangıca yapışanlar: " + ", ".join(flat) + ". "
    return text


def _tenure_sentence(rows):
    by_key = {row["algo_key"]: row for row in rows}
    short = by_key.get("tabu_n70_t25")
    long = by_key.get("tabu_n70_t625")
    if short is None or long is None:
        return ""
    gap = abs(short["best"] - long["best"])
    better = short if short["best"] <= long["best"] else long
    if _tie(short["best"], long["best"]):
        return (
            f"Tabu 70/25 ile 70/625 arasındaki fark {_tr(gap)}. "
            "Komşu sayısı ikisinde de 70. Tenure bu problemde ayrışmıyor. "
        )
    return (
        f"İki tabu da 70 komşu çekiyor. Daha ucuz olan {better['label']}. Fark {_tr(gap)}. "
        f"Yasaklı aday payı 25 sürede {_tr(short.get('forbidden_share') or 0, 2)}%, "
        f"625 sürede {_tr(long.get('forbidden_share') or 0, 2)}%. "
    )


def _prev_sentence(spec, rows):
    prev = PREV.get(spec["name"])
    if not prev:
        return ""
    winner = _best(rows)
    old = prev.get(winner["algo_key"])
    text = prev["why"] + " "
    if old is not None:
        text += (
            f"Eski çalışmada bu algoritmanın 30. dakika sonu {_tr(old)} idi. "
            f"Bu çalışmanın 400. dakika sonu {_tr(winner['best'])}. "
        )
    return text


def _rank_line(rows):
    parts = [f"{row['label']} {_tr(row['best'])} ({_pct(row['improvement'])})" for row in _ordered(rows)]
    return "Sıra: " + "; ".join(parts) + "."


def _interesting(row):
    gained = row["initial"] - row["best"]
    late = 0.0
    if gained > 1 and "best_at_18000" in row:
        late = (row["best_at_18000"] - row["best"]) / gained
    return row["improvement"] < 0.5 or late >= 0.15


def write_results():
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    section = doc.sections[0]
    for margin in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, margin, Cm(1.6))
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    for style_name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 12)):
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    def para(text, bold=False, size=11, center=False, space_after=8):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(space_after)
        paragraph.paragraph_format.space_before = Pt(0)
        if center:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(size)
        run.bold = bold
        return paragraph

    def heading(text, level):
        doc.add_heading(text, level=level)

    def shade(cell, color):
        props = cell._tc.get_or_add_tcPr()
        fill = OxmlElement("w:shd")
        fill.set(qn("w:fill"), color)
        fill.set(qn("w:val"), "clear")
        props.append(fill)

    def write_cell(cell, text, bold=False, size=8, white=False):
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.space_before = Pt(0)
        run = paragraph.add_run("" if text is None else str(text))
        run.font.name = "Times New Roman"
        run.font.size = Pt(size)
        run.bold = bold
        if white:
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    def table(headers, data, font=8):
        grid_table = doc.add_table(rows=1 + len(data), cols=len(headers))
        grid_table.style = "Table Grid"
        for index, header in enumerate(headers):
            cell = grid_table.rows[0].cells[index]
            write_cell(cell, header, bold=True, size=font, white=True)
            shade(cell, "1F4E79")
        for r, row in enumerate(data):
            for c, value in enumerate(row):
                write_cell(grid_table.rows[r + 1].cells[c], value, size=font)
        doc.add_paragraph().paragraph_format.space_after = Pt(6)

    def image(path, caption):
        if path is None or not Path(path).exists():
            return
        doc.add_picture(str(path), width=Cm(16.2))
        para(caption, size=9, center=True, space_after=10)

    done = _done_count()
    total_jobs = 2 * len(grid.SPECS) * len(grid.ALGOS)
    weights = _load_weights()

    para("Four problems, six algorithms, 400 minutes, two starts", bold=True, size=18, space_after=4)
    para(
        "Ağırlıklar, orijinal problemin rastgele başlangıcında 400 dakikalık T25 koşusundan geliyor. "
        "Aramalar bu vektör kilitlendikten sonra yapıldı. Greedy başlangıç ve rastgele başlangıç ayrıdır."
    )
    para(f"Tamamlanan koşu {done}/{total_jobs}.", bold=True)

    heading("1. Ağırlıklar", 1)
    para(
        "Önceki kural, hareketin kurulma oranının tersiydi. Kurulması zor olan daha sık çekiliyordu. "
        "Ağırlık, hareket maliyeti indirdiği için değil, teklif geometrisi tuttuğu için büyüyordu. "
        "Ropke ve Pisinger (Transportation Science, 2006) operatör ağırlığını aramanın sonucundan kurar. "
        "Bir operatör yeni en iyi planı bulursa 33, mevcut plandan iyi bir plan bulursa 9, "
        "daha kötü ama kabul edilen bir plan bulursa 13 puan alır. Reddedilen komşu puan almaz. "
        "Makalede arama 100 iterasyonluk dilimlere bölünür. Dilim sonu ağırlığı "
        "(1 − r) çarpı eski ağırlık artı r çarpı (puan / kullanım) olur. Kullandıkları r, 0,1'dir. "
        "Puan, daha önce kabul edilmemiş plana yazılır. Operatörleri sil-ve-ekle çiftidir."
    )
    para(
        "Bizim hareketler rebuild, hand-over, exchange, add-car, remove-car, add-dealer ve remove-dealer'dır. "
        "Ağırlık bir kez kurulup dondurulur. 400 dakikalık eşit ağırlıklı T25 tek dilimdir ve r, 1 alınır. "
        "r'yi 0,1 bırakıp sonra 1 ile 5 arasına germek, tepki katsayısını siler. "
        "Ziyaret edilmiş planların hash tablosu tutulmadı. Her kabul edilen planı 400 dakika saklamak ayrı bir deneydir. "
        "Kabul kararı Metropolis'te kalır. Hareket mevcut planı yine iyileştirirse puan yazılır."
    )
    para(
        "Alt ve üst sınır duruyor. En küçük pozitif ortalama puan 1 olacak şekilde bölünür, 5'te kesilir. "
        "Puanı sıfır olan hareket 1'de kalır. Kurulma oranı ağırlığa girmez. "
        "Sıcaklık bu pilottan kurulmaz. T75, T25, 500 ve 50, ölçülmüş 6.685 liralık kötüleşme ölçeğindedir. "
        f"T75 = {_tr(grid.T75, 2)}, T25 = {_tr(grid.T25, 2)}, ortak taban {_tr(grid.T_FLOOR, 2)}. "
        f"SA 50 kendi oranıyla {_tr(50.0 * grid.RATIO, 2)} değerine iner. "
        "T50 bu turda yok. SA 50 duruyor."
    )
    if weights is None:
        para("Ağırlık dosyası henüz yok.")
    else:
        para(
            f"Pilotun rastgele başlangıcı {_tr(weights['initial_total'], 2)}, "
            f"400 dakika sonu {_tr(weights['best_total'], 2)}, iniş {_pct(weights['improvement'])}. "
            f"Adım {weights.get('iterations', '—')}, en iyi adım {weights.get('best_iteration', '—')}."
        )
        names = weights["names"]
        vector = weights["weights"]
        total_w = sum(vector) or 1.0
        table_rows = []
        for index, name in enumerate(names):
            outcome = weights["outcomes"][name]
            uses = sum(outcome.values())
            score = 33 * outcome["new_best"] + 9 * outcome["improved"] + 13 * outcome["accepted_worse"]
            raw = score / uses if uses else 0.0
            stat = weights.get("stats", {}).get(name, {})
            attempts = stat.get("attempts", 0)
            built = stat.get("built", 0)
            rate = 100.0 * built / attempts if attempts else 0.0
            table_rows.append([
                name, uses, outcome["new_best"], outcome["improved"], outcome["accepted_worse"],
                outcome["rejected"], _tr(raw, 3), _tr(vector[index], 3),
                _pct(100.0 * vector[index] / total_w), _pct(rate) if attempts else "—",
            ])
        table(
            ["Hareket", "Puanlanan", "Yeni en iyi", "İyileştiren", "Kabul, kötü", "Red", "Ham", "Ağırlık", "Çekiliş", "Kurulum"],
            table_rows, font=7,
        )
        para(
            "Ham sütun puan bölü puanlanan komşudur. Ağırlık, ham değerin en küçük pozitife bölünüp 5 ile kesilmiş halidir. "
            "Kurulum sütunu eski kuralın oranıdır. Tabloda durur, ağırlığa yazılmaz."
        )

    heading("2. Dört problem ve altı algoritma", 1)
    para(
        "Önceki tur on problem ve dokuz algoritmaydı. Otuz dakika, inişi süren sıcaklıkları kesiyordu. "
        "Bu turda dört problem var. Orijinal duruyor. "
        "Büyüme, filo aynı: T25 ve T50 inişe geçmiş, SA 500'ü geçemeden süre bitmişti. "
        "Aynı harita, boşalmış: T25 son on dakikada soğuk bandın yanına gelmişti. "
        "Uzak iskelet, dar filo: T25 otuz dakikada zaten öndeydi. "
        "Biri 110 bayi ve ceza faturası, biri aynı 71 bayide bol filo, biri 24 bayi ve uzun yol."
    )
    para(
        "Algoritmalar SA T75, SA T25, SA 500, SA 50, Tabu 70/25 ve Tabu 70/625'tir. "
        "SA T50 çıkarıldı. Otuz dakikalık on problemde 70 komşuluk, 35 komşuluktan iyiydi. "
        "Kalan iki tabu bu yüzden 70 komşudur. Tenure 25 ve 625'tir. "
        "Her koşu 400 dakikadır, tohum 44'tür. Sekiz arama aynı anda yürür. Her arama kendi çekirdeğinde 400 dakika alır."
    )
    para(
        "Greedy başlangıç, o verinin dinamik açgözlü planıdır. "
        "Rastgele başlangıç günü, tırı, modeli ve bırakılan adedi çeker. Çakışan tur tutulmaz. "
        "İki başlangıç kilitlenmiş aynı ağırlığı kullanır."
    )

    heading("3. Greedy planlar", 1)
    greedy_rows = []
    for spec in grid.SPECS:
        sample = None
        for start in ("greedy", "random"):
            found = _rows(start, spec)
            if found:
                sample = found[0]
                break
        if sample is None:
            greedy_rows.append([_title(spec), "—", "—", "—", "—", "—", "—"])
            continue
        meta = sample["meta"]
        greedy = sample["greedy"]
        greedy_rows.append([
            _title(spec), meta["dealers"], meta["trucks"], _tr(meta["demand_mode"], 1),
            meta["forecast"], _tr(greedy["total"]), _bill(greedy),
        ])
    table(["Problem", "Bayi", "Tır", "Talep", "Toplam talep", "Greedy", "Fatura"], greedy_rows, font=8)
    para("Yüzde, o problemin kendi greedy planına göredir.")

    def write_start(start, number):
        title = "Greedy başlangıç" if start == "greedy" else "Rastgele başlangıç"
        heading(f"{number}. {title}", 1)
        if start == "random":
            para(
                "İyileşme sütunu koşunun kendi rastgele planına göredir. "
                "Greedy'ye göre sütunu, aramanın firmanın planını geçip geçmediğini söyler."
            )
        else:
            para("Başlangıç, o problemin dinamik greedy planıdır.")
        image(
            OUT / start / "improvement_by_config.png",
            f"Şekil. {title}, greedy plana göre iyileşme. {start}/improvement_by_config.png",
        )
        for spec in grid.SPECS:
            rows = _rows(start, spec)
            heading(_title(spec), 2)
            if rows is None:
                para("Bu problemin altı koşusu henüz kapanmadı.")
                continue
            meta = rows[0]["meta"]
            para(
                f"{spec['note']} Bayi {meta['dealers']}, tır {meta['trucks']}, "
                f"talep {_tr(meta['demand_mode'], 1)}, toplam talep {meta['forecast']}. "
                f"Greedy {_tr(rows[0]['greedy']['total'])}, {_bill(rows[0]['greedy'])}."
            )
            para(_prev_sentence(spec, rows))
            para(_rank_line(rows))
            para(_clock_sentence(rows))
            para(_tenure_sentence(rows))
            para(_piece_sentence(rows[0]["start_cost"], _best(rows)))
            data = []
            for row in rows:
                data.append([
                    row["label"], _tr(row["best"]), _pct(row["improvement"]),
                    _pct(row["improvement_vs_greedy"]),
                    _tr(row.get("best_at_6000", row["best"])),
                    _tr(row.get("best_at_18000", row["best"])),
                    _tr(row["best"]), row["best_iteration"], row["iterations"],
                ])
            table(
                ["Algoritma", "En iyi", "Başlangıca göre", "Greedy'ye göre", "100. dk", "300. dk", "400. dk", "En iyi adım", "Adım"],
                data, font=7,
            )
            data2 = [[
                "Başlangıç",
                _tr(rows[0]["start_cost"]["transport"]), _tr(rows[0]["start_cost"]["holding"]),
                _tr(rows[0]["start_cost"]["backorder"]), _tr(rows[0]["start_cost"]["unsold"]),
                _tr(rows[0]["start_cost"]["capacity"]), rows[0]["start_cost"]["delivered"], "—",
            ]]
            for row in rows:
                data2.append([
                    row["label"], _tr(row["transport"]), _tr(row["holding"]), _tr(row["backorder"]),
                    _tr(row["unsold"]), _tr(row["capacity"]), row["delivered"],
                    _tr(row["end_current"] - row["best"]),
                ])
            table(
                ["Algoritma", "Nakliye", "Elde tutma", "Karşılayamama", "Satılamayan", "Kapasite", "Teslim", "Güncel − en iyi"],
                data2, font=7,
            )
            folder = OUT / start / spec["name"]
            image(folder / "cost_vs_time.png", f"Şekil. {_title(spec)}, {title}, en iyi maliyet.")
            image(folder / "improvement_vs_time.png", f"Şekil. {_title(spec)}, {title}, başlangıca göre iyileşme.")
            image(folder / "total_cost.png", f"Şekil. {_title(spec)}, {title}, bitiş maliyeti.")
            for row in rows:
                if not _interesting(row):
                    continue
                note = "Çizgi başlangıçta kalmıştır." if row["improvement"] < 0.5 else "Son 100 dakikada iniş sürmüştür."
                image(folder / "convergence" / f"{row['algo_key']}.png", f"Şekil. {row['label']} yakınsama. {note}")
            para(f"Altı ayrı yakınsama grafiği {start}/{spec['name']}/convergence altındadır.")

    write_start("greedy", 4)
    write_start("random", 5)

    heading("6. İki başlangıç yan yana", 1)
    side = []
    notes = []
    for spec in grid.SPECS:
        grow = _rows("greedy", spec)
        rand = _rows("random", spec)
        if grow is None or rand is None:
            side.append([_title(spec), "—", "—", "—", "—"])
            continue
        gwin = _best(grow)
        rwin = _best(rand)
        side.append([
            _title(spec),
            f"{gwin['label']} {_tr(gwin['best'])}",
            _pct(gwin["improvement"]),
            f"{rwin['label']} {_tr(rwin['best'])}",
            _pct(rwin["improvement_vs_greedy"]),
        ])
        same = gwin["algo_key"] == rwin["algo_key"]
        beat = rwin["best"] + 0.5 < grow[0]["greedy"]["total"]
        notes.append(
            f"{_title(spec)}: greedy başlangıcın birincisi {gwin['label']}, "
            f"rastgele başlangıcın birincisi {rwin['label']}. "
            + ("İsim aynı. " if same else "İsim değişti. ")
            + ("Rastgele arama greedy planın altına inmiştir." if beat else "Rastgele arama greedy planın altında bitmemiştir.")
        )
    table(
        ["Problem", "Greedy başlangıç birincisi", "İyileşme", "Rastgele başlangıç birincisi", "Greedy plana göre"],
        side, font=8,
    )
    for note in notes:
        para(note)
    para(
        "Rastgele başlangıcın kendi planına göre yüzdesi büyüktür, çünkü plan kötüdür. "
        "Okunacak sütun, greedy plana göre olandır. "
        "İki maliyet iyinin binde beşinin içindeyse bu bir ayrışma sayılmaz. Tohum tektir, 44."
    )

    heading("7. Dosyalar", 1)
    para("plots/analysis/results.docx bu dosyadır. Greedy ve random klasörlerinde ayrı bir sonuç belgesi yoktur.")
    para("plots/analysis/greedy ve plots/analysis/random altında dört problem klasörü vardır. Her birinde cost_vs_time, improvement_vs_time, total_cost ve convergence klasörü durur.")
    para("Bütün koşu günlükleri plots/analysis/logs içindedir. Ağırlık listesi plots/analysis/weights.json dosyasındadır.")
    para("Eski otuz dakikalık konuşma docs/speech2_30min.docx dosyasına alındı. Güncel konuşma docs/speech2.docx dosyasındadır.")

    OUT.mkdir(parents=True, exist_ok=True)
    doc.save(DOC_PATH)
    print(f"Wrote {DOC_PATH}  runs={done}/{total_jobs}", flush=True)


def write_speech():
    """docs/speech2.docx is the curated talk. Do not replace it with a row dump."""
    if SPEECH_PATH.exists():
        print(f"Speech kept at {SPEECH_PATH}", flush=True)
        return
    print("docs/speech2.docx is missing.", flush=True)


if __name__ == "__main__":
    write_results()
    write_speech()
