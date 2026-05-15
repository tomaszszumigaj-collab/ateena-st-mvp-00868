from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from io import StringIO
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

GENERIC_DRESS_CHART = {
    "XS": {"bust": [80, 84], "waist": [62, 66], "hips": [86, 90]},
    "S": {"bust": [84, 88], "waist": [66, 70], "hips": [90, 94]},
    "M": {"bust": [88, 92], "waist": [70, 74], "hips": [94, 98]},
    "L": {"bust": [92, 97], "waist": [74, 79], "hips": [98, 103]},
    "XL": {"bust": [97, 103], "waist": [79, 85], "hips": [103, 109]},
}

TRACKING_PARAMS = {
    "gclid", "fbclid", "gbraid", "wbraid", "mc_cid", "mc_eid", "srsltid", "_gl"
}
HYBRID_SIZES = {"XS/S", "S/M", "M/L", "L/XL", "XL/XXL"}


@dataclass
class ReviewSummary:
    review_count: int
    runs_small: float
    runs_large: float
    true_to_size: float
    tight_areas: List[str]
    short_length: bool
    summary_lines: List[str]
    raw_reviews: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProductProfile:
    brand: str
    name: str
    source_url: str
    image_url: Optional[str]
    dress_type: str
    fit_type: str
    stretch_level: str
    length_type: str
    style_effect: str
    tight_areas: List[str]
    runs_small: float
    runs_large: float
    true_to_size: float
    review_count: int
    review_lines: List[str]
    size_chart: Dict[str, Dict[str, List[float]]]
    used_fallback_chart: bool
    parsing_notes: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_first_url(url_or_text: str) -> str:
    text = (url_or_text or "").strip()
    match = re.search(r"https?://\S+", text)
    if not match:
        return text
    url = match.group(0)
    return url.rstrip(').,;]\"\'')


def sanitize_product_url(url_or_text: str) -> tuple[str, List[str]]:
    notes: List[str] = []
    raw = extract_first_url(url_or_text)
    if raw != (url_or_text or "").strip():
        notes.append("W polu wykryto więcej niż jeden adres lub dodatkowy tekst — użyto pierwszego poprawnego linku.")

    parsed = urlparse(raw)
    if not parsed.scheme:
        raw = "https://" + raw.lstrip("/")
        parsed = urlparse(raw)

    kept_params = []
    removed = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.startswith("utm_") or key in TRACKING_PARAMS:
            removed.append(key)
        else:
            kept_params.append((key, value))
    clean_query = urlencode(kept_params, doseq=True)
    sanitized = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, clean_query, ""))
    if removed:
        notes.append("Usunięto parametry trackingowe z linku, żeby poprawić stabilność pobierania strony.")
    return sanitized, notes


def fetch_page(url: str, timeout: int = 18) -> tuple[str, str]:
    session = requests.Session()
    retries = Retry(
        total=2,
        read=2,
        connect=2,
        backoff_factor=0.6,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    resp = session.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text, resp.url


def _walk_jsonld(data):
    if isinstance(data, list):
        for item in data:
            yield from _walk_jsonld(item)
    elif isinstance(data, dict):
        yield data
        for value in data.values():
            yield from _walk_jsonld(value)


def _extract_title_brand(soup: BeautifulSoup, url: str) -> tuple[str, str]:
    title = ""
    brand = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = _clean(og_title["content"])
    if not title and soup.title and soup.title.string:
        title = _clean(soup.title.string)
    if not title:
        h1 = soup.find("h1")
        title = _clean(h1.get_text(" ")) if h1 else "Sukienka"

    jsonld_brand = None
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except Exception:
            continue
        for obj in _walk_jsonld(data):
            if isinstance(obj, dict) and str(obj.get("@type", "")).lower() == "product":
                if isinstance(obj.get("brand"), dict):
                    jsonld_brand = obj.get("brand", {}).get("name")
                elif isinstance(obj.get("brand"), str):
                    jsonld_brand = obj.get("brand")
                if obj.get("name"):
                    title = _clean(str(obj["name"]))
        if jsonld_brand:
            break

    domain_brand = re.sub(r"^www\.", "", re.sub(r"https?://", "", url).split("/")[0]).split(".")[0].title()
    brand = _clean(jsonld_brand or domain_brand)
    return brand, title


def _extract_image_url(soup: BeautifulSoup, final_url: str) -> Optional[str]:
    candidates = []
    meta = soup.find("meta", property="og:image")
    if meta and meta.get("content"):
        candidates.append((10, meta["content"]))
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("srcset", "").split(" ")[0]
        if not src:
            continue
        alt = (img.get("alt") or "").lower()
        cls = " ".join(img.get("class", []))
        score = 0
        if any(k in alt for k in ["dress", "sukien", "kombinezon"]):
            score += 4
        if any(token in cls.lower() for token in ["product", "gallery", "hero", "main", "image"]):
            score += 2
        w = img.get("width")
        h = img.get("height")
        try:
            if int(w or 0) >= 300 or int(h or 0) >= 300:
                score += 2
        except Exception:
            pass
        candidates.append((score, src))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return urljoin(final_url, candidates[0][1])
    return None


def _infer_dress_profile(text_blob: str) -> tuple[str, str, str, str, str, List[str]]:
    text = text_blob.lower()
    dress_type = "a_line"
    if any(k in text for k in ["bodycon", "figure-hugging", "dopasowana", "obcisła"]):
        dress_type = "bodycon"
    elif any(k in text for k in ["wrap", "kopertow", "surplice", "przekładany dekolt"]):
        dress_type = "wrap"
    elif any(k in text for k in ["shift", "prosta sukienka", "boxy"]):
        dress_type = "shift"
    elif any(k in text for k in ["fit and flare", "rozklosz", "fit-and-flare"]):
        dress_type = "fit_and_flare"
    elif any(k in text for k in ["slip", "satyn", "bias cut"]):
        dress_type = "slip"

    fit_type = "regular"
    if any(k in text for k in ["slim", "dopas", "bodycon", "tailored"]):
        fit_type = "slim"
    elif any(k in text for k in ["oversize", "loose", "swobod", "relaxed", "zwiewna"]):
        fit_type = "loose"

    stretch_level = "medium"
    if any(k in text for k in ["no stretch", "non-stretch", "sztywn", "rigid", "woven"]):
        stretch_level = "low"
    elif any(k in text for k in ["high stretch", "very stretchy", "super stretch", "elastane 8", "elastane 10", "elastan 8", "elastan 10", "5% elastan", "elastyczna talia"]):
        stretch_level = "high"

    length_type = "midi"
    if any(k in text for k in ["mini", "above knee", "krótka", "długość mini"]):
        length_type = "mini"
    elif any(k in text for k in ["maxi", "floor length", "long dress", "długa"]):
        length_type = "maxi"

    style_effect = "neutralny_fason"
    if dress_type in {"wrap", "fit_and_flare"}:
        style_effect = "podkresla_talie"
    elif dress_type == "a_line":
        style_effect = "maskuje_biodra"
    elif dress_type == "bodycon":
        style_effect = "podkresla_talie"
    elif dress_type == "slip":
        style_effect = "wydluza_sylwetke"

    tight_areas: List[str] = []
    if fit_type == "slim":
        tight_areas.extend(["bust", "waist", "hips"])
    elif dress_type == "shift":
        tight_areas.extend(["shoulders", "length"])
    elif dress_type == "wrap":
        tight_areas.extend(["waist"])
    elif dress_type == "a_line":
        tight_areas.extend(["bust", "waist"])
    elif dress_type == "fit_and_flare":
        tight_areas.extend(["bust", "waist"])
    elif dress_type == "slip":
        tight_areas.extend(["bust", "hips", "length"])
    return dress_type, fit_type, stretch_level, length_type, style_effect, sorted(set(tight_areas))


def _normalize_size_label(value: str) -> Optional[str]:
    txt = _clean(str(value)).upper().replace("SIZE", "").replace("ROZMIAR", "").strip()
    txt = txt.replace(" ", "")
    mapping = {
        "EXTRASMALL": "XS", "X-SMALL": "XS", "XSMALL": "XS",
        "SMALL": "S", "MEDIUM": "M", "LARGE": "L", "EXTRALARGE": "XL",
        "XLARGE": "XL", "X-LARGE": "XL", "XXLARGE": "XXL"
    }
    txt = mapping.get(txt, txt)
    if txt in {"XS", "S", "M", "L", "XL", "XXL"} or txt in HYBRID_SIZES:
        return txt
    return None


def _normalize_measure_name(name: str) -> Optional[str]:
    n = _clean(name).lower()
    if any(k in n for k in ["bust", "chest", "biust", "klat", "pod pachami"]):
        return "bust"
    if any(k in n for k in ["waist", "talia"]):
        return "waist"
    if any(k in n for k in ["hip", "biodr"]):
        return "hips"
    if any(k in n for k in ["length", "długość całkowita", "długość"]):
        return "length"
    if any(k in n for k in ["size", "rozmiar"]):
        return "size"
    return None


def _numbers_from_cell(value) -> List[float]:
    if value is None:
        return []
    text = _clean(str(value)).replace(',', '.')
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    return [float(n) for n in nums]


def _row_value_to_range(zone: str, raw_text: str) -> Optional[List[float]]:
    text = _clean(raw_text).replace(',', '.')
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return None
    is_width_x2 = 'x 2' in text.lower() or 'x2' in text.lower()
    if zone in {'bust', 'waist', 'hips'} and is_width_x2:
        nums = [n * 2 for n in nums]
    if len(nums) >= 2:
        return [min(nums[0], nums[1]), max(nums[0], nums[1])]
    if len(nums) == 1:
        return [nums[0] - 2, nums[0] + 2]
    return None


def _chart_from_df(df: pd.DataFrame) -> Dict[str, Dict[str, List[float]]]:
    cols = [_normalize_measure_name(c) for c in df.columns]
    if 'size' not in cols:
        return {}
    chart: Dict[str, Dict[str, List[float]]] = {}
    size_col_idx = cols.index('size')
    for _, row in df.iterrows():
        size = _normalize_size_label(row.iloc[size_col_idx])
        if not size:
            continue
        row_chart: Dict[str, List[float]] = {}
        for idx, col in enumerate(cols):
            if col in {'bust', 'waist', 'hips'}:
                rng = _row_value_to_range(col, row.iloc[idx])
                if rng:
                    row_chart[col] = rng
        if row_chart:
            chart[size] = row_chart
    return chart


def _textual_size_chart(html_text: str) -> Dict[str, Dict[str, List[float]]]:
    cleaned = _clean(html_text).replace(',', '.')
    pattern = re.compile(
        r"\b(XS/S|S/M|M/L|L/XL|XL/XXL|XS|S|M|L|XL|XXL)\b[^\d]{0,35}(\d{2,3})(?:\s*[-–/]\s*(\d{2,3}))?[^\d]{0,30}(\d{2,3})(?:\s*[-–/]\s*(\d{2,3}))?[^\d]{0,30}(\d{2,3})(?:\s*[-–/]\s*(\d{2,3}))?",
        re.I,
    )
    chart: Dict[str, Dict[str, List[float]]] = {}
    for match in pattern.finditer(cleaned):
        size = _normalize_size_label(match.group(1))
        nums = [match.group(i) for i in range(2, 8)]
        floats = [float(n) if n else None for n in nums]
        if not size:
            continue
        bust_vals = [floats[0], floats[1] or (floats[0] + 4 if floats[0] else None)]
        waist_vals = [floats[2], floats[3] or (floats[2] + 4 if floats[2] else None)]
        hips_vals = [floats[4], floats[5] or (floats[4] + 4 if floats[4] else None)]
        row: Dict[str, List[float]] = {}
        if all(v is not None for v in bust_vals):
            row['bust'] = [min(bust_vals), max(bust_vals)]
        if all(v is not None for v in waist_vals):
            row['waist'] = [min(waist_vals), max(waist_vals)]
        if all(v is not None for v in hips_vals):
            row['hips'] = [min(hips_vals), max(hips_vals)]
        if row:
            chart[size] = row
    return chart


def _extract_text_between(text: str, start_patterns: List[str], end_patterns: List[str]) -> str:
    lower = text.lower()
    start_pos = -1
    for pat in start_patterns:
        pos = lower.find(pat.lower())
        if pos != -1 and (start_pos == -1 or pos < start_pos):
            start_pos = pos
    if start_pos == -1:
        return ''
    end_pos = len(text)
    for pat in end_patterns:
        pos = lower.find(pat.lower(), start_pos + 1)
        if pos != -1:
            end_pos = min(end_pos, pos)
    return text[start_pos:end_pos]


def _extract_pakuten_chart(text: str) -> tuple[Dict[str, Dict[str, List[float]]], List[str]]:
    notes: List[str] = []
    block = _extract_text_between(
        text,
        start_patterns=['WYMIARY:', 'Wymiary:'],
        end_patterns=['SKŁAD MATERIAŁU', 'Skład materiału', 'Nasza modelka', 'Wiemy, że cenisz']
    )
    if not block:
        return {}, notes

    size_block = _extract_text_between(block, ['Rozmiar:'], ['Długość całkowita:', 'Szerokość pod pachami:', 'Szerokość w talii:'])
    size_labels = []
    for s in re.findall(r"\b(?:XS|S|M|L|XL|XXL)(?:\s*/\s*(?:XS|S|M|L|XL|XXL))?\b", size_block, flags=re.I):
        clean = _normalize_size_label(s)
        if clean and clean not in size_labels:
            size_labels.append(clean)
    if not size_labels:
        return {}, notes

    def parse_width_series(label: str) -> List[List[float]]:
        section = _extract_text_between(block, [label], ['Długość całkowita:', 'Szerokość pod pachami:', 'Szerokość w talii:', 'Długość rękawa od szyi:'])
        if not section:
            return []
        values: List[List[float]] = []
        for raw in re.findall(r"(\d+(?:[\.,]\d+)?)\s*cm\s*x\s*2(?:\s*\(\s*max\.?\s*(\d+(?:[\.,]\d+)?)\s*cm\s*x\s*2\s*\))?", section, flags=re.I):
            first = float(raw[0].replace(',', '.')) * 2
            second = float(raw[1].replace(',', '.')) * 2 if raw[1] else None
            if second is not None:
                values.append([min(first, second), max(first, second)])
            else:
                values.append([first - 2, first + 2])
        return values

    bust_values = parse_width_series('Szerokość pod pachami:')
    waist_values = parse_width_series('Szerokość w talii:')

    chart: Dict[str, Dict[str, List[float]]] = {}
    for idx, size in enumerate(size_labels):
        row: Dict[str, List[float]] = {}
        if idx < len(bust_values):
            row['bust'] = [round(bust_values[idx][0], 1), round(bust_values[idx][1], 1)]
        if idx < len(waist_values):
            row['waist'] = [round(waist_values[idx][0], 1), round(waist_values[idx][1], 1)]
        if row:
            chart[size] = row

    if chart:
        notes.append('Odczytano sekcję Wymiary i skład ze strony Pakuten.')
        if any('hips' not in row for row in chart.values()):
            notes.append('Producent nie podał osobnego obwodu bioder — rekomendacja opiera się mocniej na fasonie, talii i biuście.')
    return chart, notes


def extract_size_chart(html: str, url: str) -> tuple[Dict[str, Dict[str, List[float]]], bool, List[str]]:
    notes: List[str] = []
    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        tables = []

    best_chart = {}
    for df in tables:
        chart = _chart_from_df(df)
        if len(chart) > len(best_chart):
            best_chart = chart

    if best_chart:
        notes.append('Odczytano tabelę rozmiarów z tabeli HTML na stronie produktu.')
        return best_chart, False, notes

    text_only = BeautifulSoup(html, 'html.parser').get_text(' ', strip=True)

    if 'pakuten.pl' in urlparse(url).netloc.lower():
        pak_chart, pak_notes = _extract_pakuten_chart(text_only)
        if pak_chart:
            notes.extend(pak_notes)
            return pak_chart, False, notes

    text_chart = _textual_size_chart(text_only)
    if len(text_chart) >= 1:
        notes.append('Nie znaleziono klasycznej tabeli HTML — odczytano rozmiary z tekstu strony.')
        return text_chart, False, notes

    domain = urlparse(url).netloc.lower()
    if 'zalando.' in domain:
        notes.append('Zalando pokazuje tabelę rozmiarów w warstwie dynamicznej; w statycznym HTML nie było liczbowej tabeli do zczytania.')
    notes.append('Nie znaleziono wiarygodnej tabeli rozmiarów — użyto referencyjnej tabeli sukienek.')
    return GENERIC_DRESS_CHART, True, notes


def extract_reviews(soup: BeautifulSoup) -> List[str]:
    reviews: List[str] = []

    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script.string or '')
        except Exception:
            continue
        for obj in _walk_jsonld(data):
            obj_type = str(obj.get('@type', '')).lower() if isinstance(obj, dict) else ''
            if obj_type == 'review':
                body = obj.get('reviewBody') or obj.get('description')
                if body:
                    reviews.append(_clean(str(body)))
            elif obj_type == 'product':
                aggregate = obj.get('review')
                if isinstance(aggregate, list):
                    for item in aggregate:
                        if isinstance(item, dict):
                            body = item.get('reviewBody') or item.get('description')
                            if body:
                                reviews.append(_clean(str(body)))

    if reviews:
        return list(dict.fromkeys(reviews))[:30]

    selectors = [re.compile('review', re.I), re.compile('opini', re.I), re.compile('testimonial', re.I), re.compile('comment', re.I)]
    for tag in soup.find_all(True):
        attrs = ' '.join([' '.join(tag.get('class', [])), tag.get('id', '')])
        if not attrs:
            continue
        if any(p.search(attrs) for p in selectors):
            text = _clean(tag.get_text(' '))
            if 30 <= len(text) <= 450 and text not in reviews:
                reviews.append(text)
    return reviews[:30]


def analyze_reviews(review_texts: List[str], fallback_text: str = '', source_url: str = '') -> ReviewSummary:
    texts = [t.lower() for t in review_texts if t]
    fallback_lower = fallback_text.lower() if fallback_text else ''

    small_patterns = [r'runs small', r'size up', r'zaniż', r'mała rozmiar', r'ciasn']
    large_patterns = [r'runs large', r'size down', r'zawyż', r'duża rozmiar', r'za luźn']
    true_patterns = [r'true to size', r'zgodn.*rozmiar', r'normalna rozmiar', r'rozmiar ok']
    hips_patterns = [r'hip', r'biodr']
    waist_patterns = [r'waist', r'tali']
    bust_patterns = [r'bust', r'chest', r'biust']
    length_patterns = [r'short', r'too long', r'length', r'dług', r'krótk']

    def count(patterns: List[str]) -> int:
        total = 0
        for txt in texts:
            if any(re.search(p, txt) for p in patterns):
                total += 1
        return total

    total_reviews = max(len(texts), 1)
    small = count(small_patterns)
    large = count(large_patterns)
    true = count(true_patterns)
    hips = count(hips_patterns)
    waist = count(waist_patterns)
    bust = count(bust_patterns)
    length = count(length_patterns)

    summary_lines: List[str] = []
    if not texts and 'zalando.' in urlparse(source_url).netloc.lower() and 'najrzadziej zwracanych' in fallback_lower:
        true = 1
        summary_lines.append('Strona sygnalizuje, że to artykuł rzadko zwracany, co zwykle działa na korzyść stabilności rozmiaru.')
    elif not texts:
        summary_lines.append('Na stronie nie znaleziono czytelnych, tekstowych opinii klientów.')

    tight_areas: List[str] = []
    if hips:
        tight_areas.append('hips')
    if waist:
        tight_areas.append('waist')
    if bust:
        tight_areas.append('bust')

    if small > large and small >= 1:
        summary_lines.append('Opinie sugerują lekko zaniżoną rozmiarówkę.')
    elif large > small and large >= 1:
        summary_lines.append('Opinie sugerują lekko zawyżoną rozmiarówkę.')
    elif texts:
        summary_lines.append('Opinie nie pokazują mocnego odchylenia rozmiarówki.')
    if true >= max(small, large) and true > 0:
        summary_lines.append('Część sygnałów wskazuje, że model jest dość zgodny z tabelą producenta.')
    if tight_areas:
        pretty = {'hips': 'biodra', 'waist': 'talia', 'bust': 'biust'}
        summary_lines.append('Najczęściej wracające uwagi dotyczą: ' + ', '.join(pretty[x] for x in tight_areas) + '.')
    if length:
        summary_lines.append('W recenzjach pojawiają się też komentarze o długości sukienki.')

    runs_small = round(min(1.0, small / total_reviews), 2)
    runs_large = round(min(1.0, large / total_reviews), 2)
    true_to_size = round(min(1.0, max(true / total_reviews, 1.0 - max(runs_small, runs_large) * 0.75)), 2)

    return ReviewSummary(
        review_count=len(review_texts),
        runs_small=runs_small,
        runs_large=runs_large,
        true_to_size=true_to_size,
        tight_areas=tight_areas,
        short_length=length > 0,
        summary_lines=summary_lines,
        raw_reviews=review_texts[:10],
    )


def _fallback_product_profile(clean_url: str, notes: List[str], error_text: str) -> ProductProfile:
    parsed = urlparse(clean_url)
    brand = parsed.netloc.replace('www.', '').split('.')[0].title() or 'Sklep online'
    slug = parsed.path.strip('/').split('/')[-1] or 'produkt'
    slug = re.sub(r'[-_]+', ' ', slug)
    name = _clean(re.sub(r'\.html?$', '', slug)) or 'Sukienka'
    if 'sukien' not in name.lower() and 'kombinezon' not in name.lower():
        name = f'Sukienka / produkt z linku — {name}'
    text_blob = f'{brand} {name}'
    dress_type, fit_type, stretch_level, length_type, style_effect, inferred_tight_areas = _infer_dress_profile(text_blob)
    notes = list(notes) + [f'Nie udało się pobrać pełnej treści strony: {error_text}', 'Użyto bezpiecznego profilu awaryjnego produktu i referencyjnej tabeli rozmiarów.']
    return ProductProfile(
        brand=brand,
        name=name,
        source_url=clean_url,
        image_url=None,
        dress_type=dress_type,
        fit_type=fit_type,
        stretch_level=stretch_level,
        length_type=length_type,
        style_effect=style_effect,
        tight_areas=inferred_tight_areas,
        runs_small=0.0,
        runs_large=0.0,
        true_to_size=0.55,
        review_count=0,
        review_lines=['Nie udało się pobrać opinii z tej strony — rekomendacja opiera się głównie na fasonie i bezpiecznej tabeli referencyjnej.'],
        size_chart=GENERIC_DRESS_CHART,
        used_fallback_chart=True,
        parsing_notes=notes,
    )


def ingest_product_from_url(url_input: str) -> ProductProfile:
    clean_url, notes = sanitize_product_url(url_input)
    try:
        html, final_url = fetch_page(clean_url)
    except Exception as exc:
        return _fallback_product_profile(clean_url, notes, str(exc))

    soup = BeautifulSoup(html, 'html.parser')
    brand, name = _extract_title_brand(soup, final_url)
    image_url = _extract_image_url(soup, final_url)
    meta_text = ' '.join(meta.get('content', '') for meta in soup.find_all('meta') if meta.get('name') in {'description', 'keywords'})
    body_text = soup.get_text(' ', strip=True)[:12000]
    text_blob = _clean(' '.join([name, meta_text, body_text]))
    dress_type, fit_type, stretch_level, length_type, style_effect, inferred_tight_areas = _infer_dress_profile(text_blob)
    size_chart, used_fallback_chart, size_notes = extract_size_chart(html, final_url)
    notes.extend(size_notes)
    review_texts = extract_reviews(soup)
    review_summary = analyze_reviews(review_texts, fallback_text=text_blob, source_url=final_url)
    tight_areas = sorted(set(inferred_tight_areas + review_summary.tight_areas + (['length'] if review_summary.short_length else [])))
    notes.extend(review_summary.summary_lines)
    if review_texts:
        notes.append(f'Odczytano {len(review_texts)} przykładowych opinii klientów.')

    return ProductProfile(
        brand=brand,
        name=name,
        source_url=final_url,
        image_url=image_url,
        dress_type=dress_type,
        fit_type=fit_type,
        stretch_level=stretch_level,
        length_type=length_type,
        style_effect=style_effect,
        tight_areas=tight_areas,
        runs_small=review_summary.runs_small,
        runs_large=review_summary.runs_large,
        true_to_size=review_summary.true_to_size,
        review_count=review_summary.review_count,
        review_lines=review_summary.summary_lines + (["Przykładowe opinie: " + " | ".join(review_summary.raw_reviews[:2])] if review_summary.raw_reviews else []),
        size_chart=size_chart,
        used_fallback_chart=used_fallback_chart,
        parsing_notes=notes,
    )
