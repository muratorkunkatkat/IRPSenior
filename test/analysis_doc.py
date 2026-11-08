"""Lecturer-facing writeup of the 10-config reweighted study.

    /home/orkun/anaconda3/envs/ml/bin/python test/analysis_doc.py
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "test") not in sys.path:
    sys.path.insert(0, str(ROOT / "test"))

from data.generate_json_data import CITIES
import analysis_grid as ag

ML_PYTHON = Path("/home/orkun/anaconda3/envs/ml/bin/python")
OUT = ag.OUT
DOC_PATH = OUT / "results.docx"


def write_doc():
    try:
        from docx import Document
    except ImportError:
        if ML_PYTHON.exists() and Path(sys.executable).resolve() != ML_PYTHON.resolve():
            subprocess.check_call([str(ML_PYTHON), "-u", str(Path(__file__).resolve())], cwd=str(ROOT))
            return
        raise
    _build(Document)


def _tr(value, digits=0):
    text = f"{value:,.{digits}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(value):
    return _tr(value, 1) + "%"


def _load_weight_payload():
    path = OUT / "weights.json"
    if path.exists():
        return json.loads(path.read_text())
    src = ag.INITIAL / "weights.json"
    if src.exists():
        return json.loads(src.read_text())
    return None


def _city_of():
    mapping = {}
    cursor = 1
    for name, _lat, _lon, count, _yard in CITIES:
        for _ in range(count):
            mapping[cursor] = name
            cursor += 1
    return mapping


def _place_list(ids):
    cities = _city_of()
    counts = {}
    order = []
    for dealer in ids:
        name = cities[dealer]
        if name not in counts:
            order.append(name)
            counts[name] = 0
        counts[name] += 1
    return ", ".join(f"{name} ({counts[name]})" for name in order)


def _summaries():
    found = []
    for spec in ag.CONFIGS:
        row = []
        missing = False
        for algo in ag.ALGOS:
            summary = ag._load_summary(spec["name"], algo["key"])
            if summary is None:
                missing = True
                break
            row.append(summary)
        if not missing:
            found.append((spec, row))
    return found


def _ranks(rows):
    ordered = sorted(rows, key=lambda item: item["best"])
    rank = {}
    place = 1
    for index, row in enumerate(ordered):
        if index and row["best"] > ordered[index - 1]["best"] + 0.5:
            place = index + 1
        rank[row["algo_key"]] = place
    return rank


def _share(part, total):
    if not total:
        return 0.0
    return 100.0 * part / total


def _late_fraction(row):
    gained = row["initial"] - row["best"]
    late = row.get("best_at_1200", row["best"]) - row["best"]
    if gained <= 1:
        return 0.0
    return max(0.0, late / gained)


def _leader(rows, key):
    return min(rows, key=lambda item: item[key])


def _piece_sentence(greedy, best):
    pieces = (
        ("nakliye", greedy["transport"] - best["transport"]),
        ("elde tutma", greedy["holding"] - best["holding"]),
        ("karşılayamama", greedy["backorder"] - best["backorder"]),
        ("ay sonu satılamayan", greedy["unsold"] - best["unsold"]),
        ("kapasite cezası", greedy["capacity"] - best["capacity"]),
    )
    dropped = [(name, delta) for name, delta in pieces if delta > 1]
    raised = [(name, -delta) for name, delta in pieces if delta < -1]
    dropped.sort(key=lambda item: -item[1])
    raised.sort(key=lambda item: -item[1])
    text = ""
    if dropped:
        text += "İniş şu kalemlerden geldi: " + ", ".join(
            f"{name} {_tr(delta)} lira" for name, delta in dropped
        ) + ". "
    if raised:
        text += "Şu kalemler yükseldi: " + ", ".join(
            f"{name} {_tr(delta)} lira" for name, delta in raised
        ) + ". "
    delivered_gap = best["delivered"] - greedy["delivered"]
    if delivered_gap > 0:
        text += f"Teslim edilen araç {delivered_gap} adet arttı. "
    elif delivered_gap < 0:
        text += f"Teslim edilen araç {abs(delivered_gap)} adet azaldı. "
    else:
        text += "Teslim edilen araç sayısı değişmedi. "
    return text


def _sa_sentence(rows):
    sa = [row for row in rows if row["family"] == "sa"]
    hot_to_cold = sorted(sa, key=lambda row: -row["t0"])
    names = " → ".join(row["label"] for row in hot_to_cold)
    costs = " → ".join(_tr(row["best"]) for row in hot_to_cold)
    best = min(sa, key=lambda row: row["best"])
    worst = max(sa, key=lambda row: row["best"])
    monotone = all(
        hot_to_cold[i]["best"] + 0.5 >= hot_to_cold[i + 1]["best"]
        for i in range(len(hot_to_cold) - 1)
    )
    if monotone:
        order = (
            f"Beş SA ayarı sıcaklıktan soğuğa {names} sırasında dizildiğinde "
            f"en iyi maliyet de aynı yönde iniyor ({costs}). "
            "Bu configde daha sıcak başlamak, yarım saatin sonunda daha iyi bir plan bırakmıyor."
        )
    else:
        order = (
            f"SA sıralaması sıcaklık sırası değil. En iyi SA {best['label']} "
            f"({_tr(best['best'])}), en zayıf SA {worst['label']} ({_tr(worst['best'])}). "
            f"Sıcaktan soğuğa maliyetler: {costs}."
        )
    glued = []
    for row in sa:
        gap = row["end_current"] - row["best"]
        if gap <= max(50.0, 0.002 * row["best"]):
            glued.append(row["label"])
    if glued:
        order += " Şu ayarlarda süre bittiğinde güncel plan en iyi plana yapışmış durumda: " + ", ".join(glued) + "."
    wandering = [row["label"] for row in sa if row["end_current"] > row["best"] * 1.05]
    if wandering:
        order += (
            " Şu ayarlarda arama bitişte hâlâ en iyinin belirgin biçimde üstünde dolaşıyor: "
            + ", ".join(wandering) + "."
        )
    return order


def _tabu_sentence(rows):
    tabu = [row for row in rows if row["family"] == "tabu"]
    by_key = {row["algo_key"]: row for row in tabu}
    lines = []
    for neighbors in (35, 70):
        short = by_key[f"tabu_n{neighbors}_t25"]
        long = by_key[f"tabu_n{neighbors}_t625"]
        gap = short["best"] - long["best"]
        base = min(short["best"], long["best"])
        rel = 100.0 * gap / base if base else 0.0
        if abs(rel) < 0.5:
            lines.append(
                f"Komşu örneklemi {neighbors} iken tenure 25 ile 625 birbirine {_tr(abs(gap))} lira mesafede "
                f"({_pct(abs(rel))}). Bu örneklemde yasak süresi sonucu taşımıyor."
            )
        elif gap > 0:
            lines.append(
                f"Komşu örneklemi {neighbors} iken uzun tenure ({_tr(long['best'])}) "
                f"kısa tenure'dan {_tr(gap)} lira iyi ({_pct(rel)})."
            )
        else:
            lines.append(
                f"Komşu örneklemi {neighbors} iken kısa tenure ({_tr(short['best'])}) "
                f"uzun tenure'dan {_tr(-gap)} lira iyi ({_pct(-rel)})."
            )
    n35 = min(by_key["tabu_n35_t25"]["best"], by_key["tabu_n35_t625"]["best"])
    n70 = min(by_key["tabu_n70_t25"]["best"], by_key["tabu_n70_t625"]["best"])
    if n70 < n35 * 0.995:
        lines.append(f"Daha geniş örneklem (70) dar örneklemden (35) iyi: {_tr(n70)} karşılık {_tr(n35)}.")
    elif n35 < n70 * 0.995:
        lines.append(f"Dar örneklem (35) geniş örneklemden (70) iyi: {_tr(n35)} karşılık {_tr(n70)}.")
    else:
        lines.append("35 ve 70 komşu örnekleminin en iyi tenure'ları birbirine çok yakın.")
    forbidden = max(row.get("forbidden_share") or 0.0 for row in tabu)
    all_tabu = sum(row.get("all_tabu_iterations") or 0 for row in tabu)
    lines.append(
        f"Dört tabu koşusunun en yüksek yasaklı aday payı {_pct(forbidden)}. "
        f"Tüm adayları yasaklı olup da en ucuza zorlanan adım sayısı toplam {all_tabu}."
    )
    return " ".join(lines)


def _clock_sentence(rows):
    marks = (("10. dakika", "best_at_600"), ("20. dakika", "best_at_1200"), ("30. dakika", "best"))
    parts = []
    leaders = []
    for label, key in marks:
        leader = _leader(rows, key)
        leaders.append(leader["label"])
        parts.append(f"{label} lideri {leader['label']} ({_tr(leader[key])})")
    text = ". ".join(parts) + ". "
    if len(set(leaders)) == 1:
        text += "Lider yarım saat boyunca değişmedi."
    else:
        text += "Lider değişti: " + " → ".join(leaders) + "."
    return text


def _still_sentence(rows):
    moving = []
    flat = []
    for row in rows:
        fraction = _late_fraction(row)
        if fraction >= 0.15 and row["initial"] - row["best"] > 1:
            moving.append(f"{row['label']} (iyileşmenin {_pct(100 * fraction)} kadarı son 10 dakikada)")
        else:
            flat.append(row["label"])
    if not moving:
        return "Dokuz aramanın da son 10 dakikası, o ana kadar aldıkları iyileşmenin küçük bir parçası. Sıra, süre uzasa da bu yarım saatlik resme göre oturmuş görünüyor."
    return (
        "Son 10 dakikada iyileşmenin kayda değer kısmını hâlâ alan ayarlar: "
        + "; ".join(moving)
        + ". Bunlarda süreyi uzatmak sırayı değiştirebilir. Diğerleri düzleşmiş durumda."
    )


def _greedy_kind(greedy, forecast):
    total = greedy["total"] or 1.0
    back = _share(greedy["backorder"], total)
    hold = _share(greedy["holding"], total)
    ship = _share(greedy["transport"], total)
    unsold = _share(greedy["unsold"], total)
    service = 100.0 * greedy["delivered"] / forecast if forecast else 0.0
    if back >= 50:
        kind = "Greedy plan talebi taşımıyor. Maliyetin büyük parçası karşılayamama cezası. Arama burada önce hizmeti onarmak zorunda."
    elif ship >= 70 and back < 10:
        kind = "Greedy planda ceza küçük. Kalan iş, nakliye maliyetini indirmek."
    else:
        kind = "Greedy plan hem nakliye hem ceza taşıyor. İyi bir arama birini indirirken diğerini şişirmemeli."
    return (
        f"Greedy {_tr(greedy['total'], 2)} lira. Bunun {_pct(ship)} kadarı nakliye, "
        f"{_pct(hold)} kadarı elde tutma, {_pct(back)} kadarı karşılayamama, "
        f"{_pct(unsold)} kadarı satılamayan. Teslim {greedy['delivered']} / talep {forecast} "
        f"({_pct(service)}). " + kind
    )


def _move_sentence(row):
    counts = row.get("move_counts") or {}
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda item: -item[1])
    shown = ", ".join(f"{name} {count}" for name, count in ordered if count)
    if not shown:
        return ""
    return f"Kabul edilen hareketler: {shown}."


def _config_reading(spec, rows):
    greedy = rows[0]["greedy"]
    meta = rows[0]["meta"]
    winner = min(rows, key=lambda row: row["best"])
    ordered = sorted(rows, key=lambda row: row["best"])
    second = ordered[1]
    gap = second["best"] - winner["best"]
    rel = 100.0 * gap / winner["best"] if winner["best"] else 0.0
    sa_best = min((row for row in rows if row["family"] == "sa"), key=lambda row: row["best"])
    tabu_best = min((row for row in rows if row["family"] == "tabu"), key=lambda row: row["best"])
    family_gap = tabu_best["best"] - sa_best["best"]
    if abs(family_gap) <= 0.5:
        family = "En iyi SA ile en iyi tabu aynı maliyette."
    elif family_gap > 0:
        family = (
            f"En iyi SA ({sa_best['label']}, {_tr(sa_best['best'])}) en iyi tabudan "
            f"({tabu_best['label']}, {_tr(tabu_best['best'])}) {_tr(family_gap)} lira iyi."
        )
    else:
        family = (
            f"En iyi tabu ({tabu_best['label']}, {_tr(tabu_best['best'])}) en iyi SA'dan "
            f"({sa_best['label']}, {_tr(sa_best['best'])}) {_tr(-family_gap)} lira iyi."
        )
    if gap <= 1:
        lead = f"Birincilik berabere: {winner['label']} ve {second['label']} aynı maliyette."
    elif rel < 0.5:
        lead = (
            f"Birinci {winner['label']} ({_tr(winner['best'], 2)}), ikinci {second['label']} "
            f"({_tr(second['best'], 2)}). Fark {_tr(gap)} lira, yani {_pct(rel)}. "
            "Bu fark, birinciliği ilan edecek kadar büyük değil."
        )
    else:
        lead = (
            f"Birinci {winner['label']} ({_tr(winner['best'], 2)}), ikinci {second['label']} "
            f"({_tr(second['best'], 2)}). Fark {_tr(gap)} lira ({_pct(rel)})."
        )
    worse = [row for row in rows if row["best"] > greedy["total"] + 1]
    if worse:
        worse_text = " Greedy'nin üstünde biten ayarlar: " + ", ".join(
            f"{row['label']} ({_tr(row['best'])})" for row in worse
        ) + "."
    else:
        worse_text = " Dokuz ayarın hepsi greedy maliyetinin altında bitti."
    if gap <= 1:
        verdict = (
            f"{spec['title']} için yarım saatlik hüküm bir birincilik ilan etmez. "
            f"{winner['label']} ve {second['label']} aynı maliyettedir. " + family
        )
    else:
        verdict = (
            f"{spec['title']} için yarım saatlik hüküm: {winner['label']}, "
            f"greedy'ye göre {_pct(winner['improvement'])} iniş. " + family
        )
    return " ".join([
        _greedy_kind(greedy, meta["forecast"]),
        lead + worse_text,
        _piece_sentence(greedy, winner),
        _clock_sentence(rows),
        _still_sentence(rows),
        _sa_sentence(rows),
        _tabu_sentence(rows),
        _move_sentence(winner),
        verdict,
    ])


def _build(document_cls):
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    doc = document_cls()
    section = doc.sections[0]
    section.top_margin = Cm(1.6)
    section.bottom_margin = Cm(1.6)
    section.left_margin = Cm(1.6)
    section.right_margin = Cm(1.6)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.space_before = Pt(0)
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
        grid = doc.add_table(rows=1 + len(data), cols=len(headers))
        grid.style = "Table Grid"
        for index, header in enumerate(headers):
            cell = grid.rows[0].cells[index]
            write_cell(cell, header, bold=True, size=font, white=True)
            shade(cell, "1F4E79")
        for r, row in enumerate(data):
            for c, value in enumerate(row):
                write_cell(grid.rows[r + 1].cells[c], value, size=font)
        doc.add_paragraph().paragraph_format.space_after = Pt(6)

    def image(path, caption):
        if not path.exists():
            para(f"Grafik henüz yok: {path.name}", size=10)
            return
        doc.add_picture(str(path), width=Cm(16.2))
        para(caption, size=9, center=True, space_after=10)

    finished = _summaries()
    done = sum(1 for spec in ag.CONFIGS for algo in ag.ALGOS if ag._summary_path(spec["name"], algo["key"]).exists())
    weights = _load_weight_payload()

    para("IRP deneyi: on problem, dokuz algoritma, otuz dakika", bold=True, size=18, space_after=4)
    para("Ağırlıklar, orijinal problemin rastgele başlangıcında T25 ile kalibre edildi. Aramalar greedy başlangıçtadır.", size=11)
    if done < 90:
        para(
            f"Bu dosya ara rapordur. Tamamlanan koşu {done}/90. "
            "Bitmemiş configlerin bölümü, o configin dokuz koşusu da kapanınca dolar. Dosya her config bitiminde yenilenir.",
            bold=True,
        )
    else:
        para("Doksan koşunun hepsi kapandı. Aşağıdaki sayılar bu dosyayla birlikte üretilen grafiklerden okunmuştur.", bold=True)

    heading("1. Bu deney neyi ölçüyor", 1)
    para(
        "Soru şudur. Hareket ağırlıkları tek bir kalibrasyondan geliyor ve sabitleniyor. "
        "Bundan sonra sıcaklığın, tabu örnekleminin ve tabu süresinin sıralaması, problem değişince duruyor mu? "
        "Problem üç yerden değişiyor: bayi sayısı, tır sayısı, talep yoğunluğu. "
        "Mesafe matrisi yeniden üretilmiyor. Eski bayi çiftlerinin süre ve maliyetleri veri dosyasındaki değerin aynısıdır."
    )
    para(
        "Dokuz algoritma var. Beşi benzetilmiş tavlama: ölçülmüş kötüleşme ortalaması 6.685 lira üzerinden "
        "kabul olasılığı 0,75, 0,50 ve 0,25 veren üç sıcaklık, artı soğuk referans olarak 500 ve 50. "
        "Dördü tabu: komşu örneklemi 35 ve 70, tabu süresi 25 ve 625. "
        "Önceki 3×3 ızgarada 150 komşu ve 125 süre, maliyeti taşımıyordu. O iki düzey bu deneyde yok."
    )
    para(
        "Her koşu 30 dakikadır. Tohum 44'tür. Başlangıç, o configin greedy planıdır. "
        "Rastgele başlangıç bu turda yok. On config ve dokuz algoritma zaten 45 saat işlemci ister. "
        "İkinci bir başlangıç bunu ikiye katlardı. "
        "Dört koşu aynı anda yürür. Her koşunun saati yine 30 dakikadır. Dört çekirdek, duvar saatini kısaltır, her aramaya verilen süreyi kısaltmaz."
    )
    para(
        "Sıcaklıklar, maliyetin yüzde 10'undan kurulmadı. Yüzde 10, iyi bir turun içine tek bir eklemenin payı olarak okunur. "
        "Bizim kötüleşme ölçümümüz turun yüzde 8–15'i değil, greedy planın binde 3'ü düzeyindeydi. "
        "Formül aynı, girdi değişti. T(p) = −6.685 / ln(p). "
        f"T75 = {_tr(ag.T75, 2)}, T50 = {_tr(ag.T50, 2)}, T25 = {_tr(ag.T25, 2)}. "
        f"İlk dört ayar ortak tabana soğur: {_tr(ag.T_FLOOR, 2)}. "
        f"T0 = 50 zaten bu tabanın altında olduğu için kendi oranıyla soğur, tabanı {_tr(50.0 * ag.RATIO, 2)}. "
        "Saat soğutması 30 dakikanın sonunda tabana varır. T0 = 50, kâğıttaki sıcaklık değildir. "
        "Kâğıttaki üç hedef 0,75, 0,50 ve 0,25'tir. 500 ve 50, formülün soğuk ucunda ne kaldığını görmek için durur."
    )

    heading("2. On config nasıl kuruldu", 1)
    para(
        "Orijinal dosya 30 gün, 71 bayi, 80 tır, beş modeldir. Modeller Egea Cross, Egea Sedan, Scudo, Scudo kamyonet ve Ulysse. "
        "Kapasiteler 8, 8, 6, 6, 6. Fabrika Bursa'dır. Talep her bayi, her model, her gün için 0, 1 veya 2'dir. "
        "Orijinal talebin toplamı 4.351'dir."
    )
    para(
        "110 bayilik configler, bu 71 bayinin üstüne 39 yeni bayi ekler. Yeni bayiler mevcut şehirlerin içine konur: "
        "İstanbul +12, Ankara +6, İzmir +4, Bursa +3, Antalya +3, Adana +2, Konya +2, Gaziantep +2, Kocaeli +2, Mersin +2, Kayseri +1. "
        "Konum, şehrin koordinatı etrafında sabit bir çember üzerindedir. Tohum 44 + 110. "
        "Yeni bayinin depo kapasitesi, o şehrin mevcut depo kapasitesidir. "
        "Yeni talebin günlük ortalaması, orijinal dosyayı üreten model ortalamalarıdır. "
        "Eski çiftlerin A ve R değerleri kopyalanır. Yalnızca yeni bir uç içeren çiftler hesaplanır: "
        "kuş uçuşu × 1,35, 280 km altı aynı gün, sonrası 500 km/gün, maliyet km × 1,5 ve iki ondalık."
    )
    base, _path = ag.load_instance()
    near_ids = ag.near_dealer_ids(base)
    national_ids = ag.national_dealer_ids()
    para(
        "23 bayilik config, fabrikaya gidiş süresi 0 olan bayilerdir. Aynı gün varılır. Şehir dağılımı: "
        + _place_list(near_ids) + "."
    )
    para(
        "24 bayilik config, her çok şubeli şehirden bir bayi ve boylam boyunca seçilmiş 14 tek şehirlik bayidir. "
        "Boylamı tek başına dilimlemek İstanbul ve Ankara'yı düşürürdü. Onlar listeye şehir olarak girdikleri için düşmez. Şehir dağılımı: "
        + _place_list(national_ids) + "."
    )
    para(
        "Talep çarpanı bayi başına yoğunluktur, ülke toplamını sabitlemez. 2× her hücreyi ikiye katlar. "
        "0,5× hücredeki 2'yi 1 yapar. Hücre 1 ise ve bayi numarası + model + gün çiftse o hücre silinir. "
        "Gün sayısı, model kapasitesi, ceza katsayıları ve tutulan bayilerin depo kapasitesi değişmez."
    )
    para(
        "Tır sayısı, eldeki filo uzunluğudur. 30, 80, 120 ve 140 aynı tır tanımının adedidir. "
        "Tırın o turda taşıdığı modele göre kapasite 8 veya 6'dır."
    )

    heading("3. Ağırlıklar", 1)
    para(
        "Kalibrasyon, büyük ızgaradan önce ve ondan ayrıdır. Orijinal problem, rastgele başlangıç, tohum 44, "
        "eşit ağırlık, T0 = T25, taban 72,58, 30 dakika. "
        "Ağırlık, bir hareketin işe yarayıp yaramasından değil, teklifin kurulabilme oranından gelir. "
        "Ham ağırlık = 1 / (kurulan / denenen). En küçük ham ağırlık 1 olacak şekilde bölünür. Tavan 5'tir. "
        "Kurulma oranı sıfıra yakınsa ham değer, bölmeden önce 5 alınır. "
        "Kabul oranı bu hesaba girmez. Aynı vektör on configin hem tavlamasında hem tabusunda kullanılır. "
        "Config başına yeniden kalibrasyon yoktur. Tabu da komşusunu bu ağırlıkla çeker."
    )
    if weights is None:
        para("Ağırlık dosyası henüz yazılmadı. Kalibrasyon kapanınca bu bölüm dolar.")
    else:
        names = weights.get("names") or weights.get("moves") or list(ag.MOVE_NAMES)
        vector = weights["weights"]
        stats = weights.get("stats") or {}
        proposed = weights.get("proposed_counts") or {}
        accepted = weights.get("move_counts") or {}
        initial = weights.get("initial_total", weights.get("initial"))
        best = weights.get("best_total", weights.get("best"))
        if initial is not None and best is not None:
            para(
                f"Kalibrasyon koşusunun rastgele başlangıcı {_tr(initial, 2)} lira, "
                f"30. dakikadaki en iyi plan {_tr(best, 2)} liradır. "
                f"İniş {_pct(ag.improvement(initial, best))}. "
                "Aynı tohumla, sekiz koşu aynı anda yürürken alınan eşit ağırlıklı rastgele T25, 6.883.988 lirada kalmıştı. "
                "Bu kalibrasyon tek başına yürüdü. Saat soğutması duvar saatine bağlı olduğu için 30 dakikada daha çok adım attı ve daha aşağı indi. "
                "İki yörünge aynı değildir. Ağırlıklar bu tek koşunun kurulma oranlarından gelir. "
                "Kalibrasyonun kendi maliyeti, aşağıdaki on configin skoru değildir."
            )
        raw = []
        for name in names:
            stat = stats.get(name, {})
            attempts = stat.get("attempts", 0)
            built = stat.get("built", 0)
            rate = built / attempts if attempts else 0.0
            raw.append(5.0 if rate <= 1e-12 else 1.0 / rate)
        lowest = min(raw) if raw else 1.0
        rows = []
        total_w = sum(vector) or 1.0
        for index, name in enumerate(names):
            stat = stats.get(name, {})
            attempts = stat.get("attempts", 0)
            built = stat.get("built", 0)
            rate = 100.0 * built / attempts if attempts else 0.0
            rows.append([
                name,
                attempts,
                built,
                _pct(rate) if attempts else "—",
                _tr(raw[index], 3),
                _tr(vector[index], 3),
                _pct(100.0 * vector[index] / total_w),
                proposed.get(name, "—"),
                accepted.get(name, "—"),
            ])
        table(
            ["Hareket", "Deneme", "Kurulan", "Kurulum", "Ham", "Ağırlık", "Çekiliş", "Teklif", "Kabul"],
            rows,
            font=8,
        )
        para(
            "Ham sütunu 1 / kurulma oranıdır. Ağırlık, ham değerin en küçüğe bölünüp 5 ile kesilmiş halidir. "
            "Çekiliş, bir sonraki hareketin o türden gelme olasılığıdır. "
            "Teklif ve kabul, kalibrasyon koşusunda Metropolis'e ulaşan ve kabul edilen adetlerdir. "
            "Kabul yüksek diye ağırlık büyütülmedi."
        )
        para(
            "Grafikler plots/initial içindeki eski dosyaların yerine geçti. "
            "sa_convergence güncel maliyet, en iyi maliyet ve sıcaklığı birlikte gösterir. "
            "Ağırlık, kabul eğrisinden değil kurulma çubuğundan okunur."
        )
        image(ag.INITIAL / "sa_convergence.png", "Şekil. Kalibrasyon. Rastgele başlangıç, eşit ağırlık, T25, 30 dakika. plots/initial/sa_convergence.png")
        image(ag.INITIAL / "sa_components_vs_time.png", "Şekil. Kalibrasyonda en iyi planın maliyet kalemleri. plots/initial/sa_components_vs_time.png")
        image(ag.INITIAL / "move_acceptance.png", "Şekil. Kurulma oranı ve kabul oranı. Ağırlık soldaki çubuktan gelir. plots/initial/move_acceptance.png")
        image(ag.INITIAL / "accepted_mix.png", "Şekil. Kabul edilen hareketlerin payı. Bu pay ağırlığa yazılmaz. plots/initial/accepted_mix.png")

    heading("4. Greedy planlar", 1)
    para(
        "Her configin başlangıcı, o verinin dinamik greedy planıdır. Boş tır, kalan talebi en büyük model, "
        "o modelde hâlâ açık olan en yakın bayi. Aynı gün varılan bayiler süre olarak berabere kalır. "
        "Bırakılan miktar, ihtiyacın ve tırda kalan yerin küçük olanıdır. Teslim, tırın çıktığı güne yazılır. "
        "Greedy deterministiktir. Tohum onu değiştirmez."
    )
    greedy_rows = []
    for spec in ag.CONFIGS:
        sample = None
        for algo in ag.ALGOS:
            sample = ag._load_summary(spec["name"], algo["key"])
            if sample:
                break
        if sample is None:
            data, meta = ag.build_instance(base, spec)
            from heuristic.search_problem import Solution, build_initial_trips, clone_trips, evaluate
            trips = build_initial_trips(data)
            cost, _aux = evaluate(Solution(data, clone_trips(trips)))
            greedy_rows.append([
                spec["title"], spec["note"], meta["dealers"], meta["trucks"],
                _tr(meta["demand_mode"], 1), meta["forecast"], len(trips), _tr(cost.total),
            ])
        else:
            greedy = sample["greedy"]
            meta = sample["meta"]
            greedy_rows.append([
                spec["title"], spec["note"], meta["dealers"], meta["trucks"],
                _tr(meta["demand_mode"], 1), meta["forecast"], sample.get("initial_tours", "—"),
                _tr(greedy["total"]),
            ])
    table(
        ["Config", "Ne değişti", "Bayi", "Tır", "Talep", "Toplam talep", "Tur", "Greedy"],
        greedy_rows,
        font=7,
    )
    para(
        "Greedy maliyeti configler arasında 100 bin lira ile 30 milyon lira arasında dağılır. "
        "Yüzde iyileşmeyi configler arasında tek bir ortalamayla kıyaslamak, kolay problemle tıkanmış problemi birbirine katar. "
        "Asıl kıyas, configin içinde dokuz algoritmanın sırasıdır. Yüzde, o configin kendi greedy'sine göredir."
    )

    heading("5. Config config sonuçlar", 1)
    if not finished:
        para("Henüz dokuz koşusu birden kapanmış bir config yok. İlk config kapanınca bu bölüm ve grafikler dolar.")
    for spec, rows in finished:
        meta = rows[0]["meta"]
        heading(f"{spec['title']}", 2)
        para(spec["note"] + f" Bayi {meta['dealers']}, tır {meta['trucks']}, talep çarpanı {meta['demand_mode']}, toplam talep {meta['forecast']}.")
        para(_config_reading(spec, rows))
        header = ["Algoritma", "En iyi", "İyileşme", "10 dk", "20 dk", "30 dk", "En iyi adım", "Adım", "Durdu"]
        data = []
        for row in rows:
            data.append([
                row["label"],
                _tr(row["best"], 0),
                _pct(row["improvement"]),
                _tr(row["best_at_600"], 0),
                _tr(row["best_at_1200"], 0),
                _tr(row["best"], 0),
                row["best_iteration"],
                row["iterations"],
                row["stopped"],
            ])
        table(header, data, font=7)
        header2 = ["Algoritma", "Nakliye", "Elde tutma", "Karşılayamama", "Satılamayan", "Kapasite", "Teslim", "Güncel − en iyi"]
        data2 = []
        greedy = rows[0]["greedy"]
        data2.append([
            "Greedy", _tr(greedy["transport"], 0), _tr(greedy["holding"], 0),
            _tr(greedy["backorder"], 0), _tr(greedy["unsold"], 0), _tr(greedy["capacity"], 0),
            greedy["delivered"], "—",
        ])
        for row in rows:
            data2.append([
                row["label"],
                _tr(row["transport"], 0),
                _tr(row["holding"], 0),
                _tr(row["backorder"], 0),
                _tr(row["unsold"], 0),
                _tr(row["capacity"], 0),
                row["delivered"],
                _tr(row["end_current"] - row["best"], 0),
            ])
        table(header2, data2, font=7)
        para(
            "Güncel − en iyi sütunu, süre bittiğinde aramanın en iyi plandan ne kadar uzakta olduğunu gösterir. "
            "Sıfır, aramanın tabana yapıştığı anlamına gelir. Büyük bir pozitif sayı, tavlamanın hâlâ kötü planları kabul ettiğini gösterir. "
            "En iyi adım, toplam adıma yakınsa iyileşme sona doğru gelmiştir."
        )
        folder = OUT / spec["name"]
        image(folder / "cost_vs_time.png", f"Şekil. {spec['title']}, dokuz algoritmanın en iyi maliyeti. {spec['name']}/cost_vs_time.png")
        image(folder / "improvement_vs_time.png", f"Şekil. {spec['title']}, greedy'ye göre iyileşme. {spec['name']}/improvement_vs_time.png")
        image(folder / "total_cost.png", f"Şekil. {spec['title']}, bitiş maliyeti. {spec['name']}/total_cost.png")

    heading("6. Sıra nerede değişiyor", 1)
    if len(finished) < len(ag.CONFIGS):
        para(f"Bu bölüm {len(finished)}/{len(ag.CONFIGS)} config ile yazıldı. Eksikler kapanınca yenilenecek.")
    if not finished:
        para("Karşılaştırılacak kapanmış config yok.")
    else:
        image(OUT / "overview" / "improvement_by_config.png", "Şekil. İyileşme yüzdesi, config ekseninde. Yüzde, o configin kendi greedy'sine göredir. overview/improvement_by_config.png")
        overview = []
        wins = {algo["key"]: [] for algo in ag.ALGOS}
        rank_sum = {algo["key"]: 0 for algo in ag.ALGOS}
        rank_n = 0
        family_flip = []
        clock_flip = []
        late_flags = []
        for spec, rows in finished:
            rank = _ranks(rows)
            winner = min(rows, key=lambda row: row["best"])
            wins[winner["algo_key"]].append(spec["title"])
            rank_n += 1
            for row in rows:
                rank_sum[row["algo_key"]] += rank[row["algo_key"]]
            sa_best = min((row for row in rows if row["family"] == "sa"), key=lambda row: row["best"])
            tabu_best = min((row for row in rows if row["family"] == "tabu"), key=lambda row: row["best"])
            if abs(tabu_best["best"] - sa_best["best"]) <= 0.5:
                family_flip.append(f"{spec['title']} SA ve tabu berabere")
            elif tabu_best["best"] < sa_best["best"]:
                family_flip.append(f"{spec['title']} tabu ({tabu_best['label']})")
            else:
                family_flip.append(f"{spec['title']} SA ({sa_best['label']})")
            early = _leader(rows, "best_at_1200")["label"]
            if early != winner["label"]:
                clock_flip.append(f"{spec['title']}: 20. dakikada {early}, 30. dakikada {winner['label']}")
            for row in rows:
                if _late_fraction(row) >= 0.15 and row["initial"] - row["best"] > 1:
                    late_flags.append(f"{spec['title']} / {row['label']}")
            overview.append([
                spec["title"],
                _tr(rows[0]["greedy"]["total"], 0),
                winner["label"],
                _tr(winner["best"], 0),
                _pct(winner["improvement"]),
                sa_best["label"],
                tabu_best["label"],
            ])
        table(
            ["Config", "Greedy", "Birinci", "En iyi", "İyileşme", "En iyi SA", "En iyi tabu"],
            overview,
            font=7,
        )
        rank_rows = []
        for algo in ag.ALGOS:
            mean_rank = rank_sum[algo["key"]] / rank_n
            titles = ", ".join(wins[algo["key"]]) if wins[algo["key"]] else "—"
            rank_rows.append([algo["label"], _tr(mean_rank, 2), len(wins[algo["key"]]), titles])
        table(["Algoritma", "Ortalama sıra", "Birincilik", "Birinci olduğu configler"], rank_rows, font=8)
        para(
            "Ortalama sıra 1'e yakınsa algoritma configlerin çoğunda üsttedir. "
            "Birincilik sayısı, berabere kalınan durumlarda yalnız en küçük maliyeti sayar. "
            "İki maliyet 0,5 liranın içindeyse ikisi de aynı sırayı paylaşır."
        )
        para("Aile karşılaştırması, her configde en iyi SA ile en iyi tabudan ucuz olanı söyler. " + " ".join(family_flip) + ".")
        if clock_flip:
            para("20. dakikadan 30. dakikaya lider değişen configler: " + "; ".join(clock_flip) + ".")
        else:
            para("Kapanan configlerin hiçbirinde 20. dakikanın lideri, 30. dakikada yerini bırakmadı.")
        if late_flags:
            para(
                "İyileşmenin en az %15'i son 10 dakikada gelmiş koşular: "
                + "; ".join(late_flags)
                + ". Bunların sırası, süre uzarsa değişebilir. Diğer koşular bu yarım saatte oturmuş sayılır."
            )
        else:
            para("Kapanan koşuların hiçbirinde iyileşmenin büyük parçası son 10 dakikaya yığılmadı.")

        hot_wins = [title for title in wins["sa_T75"] + wins["sa_T50"] + wins["sa_T25"]]
        cold_wins = wins["sa_500"] + wins["sa_50"]
        para(
            "Formül sıcaklıkları (T75, T50, T25) ile soğuk referanslar (500 ve 50) ayrı okunmalı. "
            + (
                "Formül sıcaklıklarının birinci olduğu yerler: " + ", ".join(hot_wins) + ". "
                if hot_wins else "Formül sıcaklıkları hiçbir kapanmış configde birinci olmadı. "
            )
            + (
                "Soğuk referansların birinci olduğu yerler: " + ", ".join(cold_wins) + ". "
                if cold_wins else "Soğuk referanslar hiçbir kapanmış configde birinci olmadı. "
            )
            + "Greedy iyi bir plan bıraktıysa soğuk inişin öne geçmesi beklenir. "
            "Greedy cezaya sıkıştıysa sıcak aramanın, cezayı kırması beklenir. "
            "Yukarıdaki liste bu beklentiyi config config doğrular ya da bozar. Beklentiye uymayan satır, o configin bölümünde kalem kalem yazılıdır."
        )

    heading("7. Sorulacak yerler", 1)
    qa = [
        (
            "Neden on config, neden tek tek değiştirmek değil?",
            "Tek düğme, firmanın aynı anda büyümesini göstermez. "
            "Listede filonun da büyüdüğü bir firma, bayi ve talep büyüyüp filonun kaldığı bir firma, "
            "harita büyüyüp filonun daraldığı bir firma, aynı haritada yüklenen ve boşalan iki uç, "
            "yakın bölgede sıkışık ve seyrek iki uç, uzak iskelette rahat ve dar iki uç var. "
            "Mesafe matrisi bu değişimlerin dışında tutuldu.",
        ),
        (
            "110 bayi verisi uydurma mı?",
            "71 bayinin koordinatı ve eski çiftlerin mesafesi duruyor. "
            "39 bayi, zaten çok şubeli olan şehirlere, o şehrin depo kapasitesiyle eklendi. "
            "Talep, orijinal dosyanın model ortalamalarından, tohum 44 + 110 ile üretildi. "
            "Yeni çiftlerin yol kuralı, eski dosyayı üreten kuralın aynısıdır.",
        ),
        (
            "Neden ağırlık bir kez, rastgele başlangıçtan?",
            "Ağırlık, hareketin o andaki işe yararlığı değil, kurulabilme zorluğudur. "
            "Zor kurulan hareket eşit ağırlıkta az teklif edilir. Kalibrasyon bunu tersine çevirir, tavan 5'tir. "
            "Rastgele plan, kurulma oranını greedy'nin düzgün turlarına hapsetmesin diye seçildi. "
            "Sıcaklık T25'tir çünkü ölçülmüş ortalama kötüleşmeyi dört seferden birinde kabul eden formül sıcaklığı budur. "
            "Aynı vektörü her configde kullanmak, algoritma farkını problem farkından ayırır. "
            "Her config kendi ağırlığını kursaydı, sıra değişince bunun sıcaklıktan mı ağırlıktan mı geldiği karışırdı.",
        ),
        (
            "Kabul oranı neden ağırlıkta yok?",
            "Kabul oranı sıcaklığa ve o anki plana bağlıdır. Onu ağırlığa yazmak, soğuk bir koşunun reddettiği hareketi bir sonraki koşuda da seyrek teklif eder. "
            "Kurulma oranı ise hareketin geometrisidir. Teklif edilebilen bir hareketin kabulü, tavlamanın ve tabunun kendi kuralına bırakıldı.",
        ),
        (
            "Neden bu beş sıcaklık?",
            "Üçü, ölçülmüş ortalama yokuşu 0,75, 0,50 ve 0,25 olasılıkla kabul eder. "
            "500 ve 50, formülün altında kalan iniş bölgesidir. "
            "T0 = 50, kâğıdın önerdiği sıcaklık değildir. Kâğıt, problemi değişince sıcaklığın yeniden kurulmasını söyler. "
            "Burada yeniden kurulan girdi, turun yüzde 10'u değil, yedi hareketten ölçülen 6.685 liradır.",
        ),
        (
            "Neden tabu 2×2?",
            "Önceki 30 dakikalık 3×3 ızgarada 150 komşu, 70 komşudan iyi değildi. Tenure 125, 25 ile 625'in arasında bir şey söylemedi. "
            "İşlemci sınırı 48 saatti. Beş tavlama dururken tabuya dört hücre kaldı. Kalan hücreler, örneklemin ve sürenin iki ucudur.",
        ),
        (
            "Neden yalnız greedy başlangıç?",
            "Rastgele başlangıç her configi bir kez daha 30 dakika koştururdu. Bütçe 45,5 saattir: yarım saat kalibrasyon, 90 koşu. "
            "Greedy, firmanın elindeki planın karşılığıdır. Rastgele planın aynı ağırlıkla ne yaptığı, kalibrasyon koşusunda ve daha önceki eşit ağırlık ızgarasında duruyor. Bu dosya onu yeniden koşturmaz.",
        ),
        (
            "30 dakika yetti mi?",
            "Her config bölümünde 10, 20 ve 30. dakika lideri ve son 10 dakikanın iyileşme payı yazılıdır. "
            "Lider değişmediyse ve son 10 dakika düzse, bu problem için yarım saat sıralamayı oturtmuş demektir. "
            "Son 10 dakikada iniş sürüyorsa, o satırın birinciliği süreye bağlıdır. Bunu gizlemeden yazdım.",
        ),
        (
            "İyileşme yüzdeyse hizmet bozulmuş olabilir mi?",
            "Evet, olabilir. Onun için her configde nakliye, elde tutma, karşılayamama, satılamayan ve teslim adedi ayrı duruyor. "
            "Nakliye inip karşılayamama çıktıysa, metin bunu iniş diye satmaz. Teslim adedi de ayrıca yazılıdır.",
        ),
        (
            "Tek tohum yeter mi?",
            "Hayır, bir güven aralığı değildir. Tohum 44, önceki koşularla aynı olsun diye sabitlendi. "
            "Sıra iki algoritma arasında birkaç bin liraysa, bu dosya onu birincilik diye büyütmez. "
            "Fark yüzde 0,5'in altındaysa metin bunu söyler.",
        ),
        (
            "Tabu süresi işe yarıyor mu?",
            "Her configin tabu paragrafı, 25 ile 625'i aynı örneklemde lira lira karşılaştırır. "
            "Yasaklı aday payı ve tüm adayların yasak olduğu adım sayısı da oradadır. "
            "Pay küçükse süre pratikte bağlanmıyor demektir. Büyükse ve maliyet de değişiyorsa, süre kararı taşımış demektir.",
        ),
        (
            "Ağırlıklar eşit koşudan iyi mi?",
            "Bu deney, eşit ağırlığın aynısını her configde bir kez daha koşturmaz. "
            "Orijinal problemin eşit ağırlık sonuçları önceki ızgaradadır. "
            "Bu dosya, sabitlenmiş ağırlıkla sıcaklığın ve tabunun problem değişince nasıl sıraya girdiğini okur. "
            "Orijinal configin satırı, önceki eşit ağırlık sayısının yanına konabilir. Ağırlığın tek başına ne kazandırdığı, yalnız o satırdan ve yalnız aynı başlangıç için okunur.",
        ),
    ]
    for question, answer in qa:
        para(question, bold=True, space_after=2)
        para(answer)

    heading("8. Dosyalar", 1)
    para("Kalibrasyon grafikleri plots/initial içindedir. Ağırlık listesi plots/initial/weights.json ve plots/analysis/weights.json dosyalarındadır.")
    para("Her configin üç grafiği plots/analysis/<config>/ altındadır: cost_vs_time.png, improvement_vs_time.png, total_cost.png.")
    para("Configler arası iyileşme grafiği plots/analysis/overview/improvement_by_config.png dosyasındadır.")
    para("Her koşunun ayrı günlüğü plots/analysis/<config>__<algoritma>.log dosyasındadır. Ortak günlük log.txt, o anki durum progress.txt içindedir.")
    para(
        "Koşu yarıda kesilirse aynı komut kaldığı yerden sürer. Özeti yazılmış koşu yeniden başlamaz. "
        "Özeti yazılmamış koşu 30 dakikasına baştan başlar."
    )

    OUT.mkdir(parents=True, exist_ok=True)
    doc.save(DOC_PATH)
    print(f"Wrote {DOC_PATH}  finished_configs={len(finished)}  runs={done}", flush=True)


if __name__ == "__main__":
    write_doc()
