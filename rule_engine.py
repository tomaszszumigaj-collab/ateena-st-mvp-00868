from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

BASE_SIZE_RANK = {
    "XXS": 0,
    "XS": 1,
    "XS/S": 1.5,
    "S": 2,
    "S/M": 2.5,
    "M": 3,
    "M/L": 3.5,
    "L": 4,
    "L/XL": 4.5,
    "XL": 5,
    "XL/XXL": 5.5,
    "XXL": 6,
}

STRETCH_CM = {"low": 0, "medium": 2, "high": 4}

CATEGORY_PROFILES: Dict[str, Dict] = {
    "sukienka casualowa": {
        "family": "dress",
        "zones": ["bust", "waist", "hips"],
        "zone_weights": {"bust": 1.0, "waist": 1.0, "hips": 1.0},
        "style_scores": {"gruszka": 90, "jabłko": 81, "klepsydra": 92, "prostokąt": 79, "odwrócony trójkąt": 84},
        "leniency": 0.0,
    },
    "sukienka elegancka": {
        "family": "dress",
        "zones": ["bust", "waist", "hips"],
        "zone_weights": {"bust": 1.1, "waist": 1.15, "hips": 1.05},
        "style_scores": {"gruszka": 88, "jabłko": 76, "klepsydra": 94, "prostokąt": 77, "odwrócony trójkąt": 83},
        "leniency": -0.5,
    },
    "spódniczka": {
        "family": "bottom",
        "zones": ["waist", "hips"],
        "zone_weights": {"waist": 1.2, "hips": 1.1},
        "style_scores": {"gruszka": 89, "jabłko": 76, "klepsydra": 90, "prostokąt": 78, "odwrócony trójkąt": 84},
        "leniency": 0.5,
    },
    "legginsy": {
        "family": "bottom",
        "zones": ["waist", "hips", "thigh"],
        "zone_weights": {"waist": 1.0, "hips": 1.0, "thigh": 1.15},
        "style_scores": {"gruszka": 74, "jabłko": 68, "klepsydra": 84, "prostokąt": 76, "odwrócony trójkąt": 80},
        "leniency": 1.5,
    },
    "żakiet": {
        "family": "outerwear_top",
        "zones": ["chest", "waist", "arm_biceps"],
        "zone_weights": {"chest": 1.2, "waist": 0.8, "arm_biceps": 1.0},
        "style_scores": {"gruszka": 93, "jabłko": 84, "klepsydra": 90, "prostokąt": 87, "odwrócony trójkąt": 80},
        "leniency": 0.0,
    },
    "marynarka": {
        "family": "outerwear_top",
        "zones": ["chest", "waist", "arm_biceps"],
        "zone_weights": {"chest": 1.25, "waist": 0.85, "arm_biceps": 1.0},
        "style_scores": {"gruszka": 92, "jabłko": 85, "klepsydra": 91, "prostokąt": 88, "odwrócony trójkąt": 79},
        "leniency": -0.2,
    },
    "spodnie eleganckie": {
        "family": "bottom",
        "zones": ["waist", "hips", "thigh"],
        "zone_weights": {"waist": 1.2, "hips": 1.05, "thigh": 0.95},
        "style_scores": {"gruszka": 80, "jabłko": 75, "klepsydra": 86, "prostokąt": 79, "odwrócony trójkąt": 88},
        "leniency": -0.2,
    },
    "spodnie jeansowe": {
        "family": "bottom",
        "zones": ["waist", "hips", "thigh"],
        "zone_weights": {"waist": 1.15, "hips": 1.0, "thigh": 1.1},
        "style_scores": {"gruszka": 83, "jabłko": 72, "klepsydra": 88, "prostokąt": 80, "odwrócony trójkąt": 84},
        "leniency": 0.5,
    },
    "spodnie casualowe": {
        "family": "bottom",
        "zones": ["waist", "hips", "thigh"],
        "zone_weights": {"waist": 1.1, "hips": 1.0, "thigh": 1.0},
        "style_scores": {"gruszka": 82, "jabłko": 76, "klepsydra": 86, "prostokąt": 80, "odwrócony trójkąt": 86},
        "leniency": 1.0,
    },
    "gorset": {
        "family": "structured_top",
        "zones": ["bust", "waist"],
        "zone_weights": {"bust": 1.2, "waist": 1.4},
        "style_scores": {"gruszka": 84, "jabłko": 60, "klepsydra": 95, "prostokąt": 82, "odwrócony trójkąt": 76},
        "leniency": -1.0,
    },
    "T-shirt": {
        "family": "top",
        "zones": ["chest", "waist"],
        "zone_weights": {"chest": 1.05, "waist": 0.7},
        "style_scores": {"gruszka": 82, "jabłko": 78, "klepsydra": 80, "prostokąt": 84, "odwrócony trójkąt": 78},
        "leniency": 1.5,
    },
    "bluza": {
        "family": "top",
        "zones": ["chest", "waist", "arm_biceps"],
        "zone_weights": {"chest": 1.0, "waist": 0.5, "arm_biceps": 0.8},
        "style_scores": {"gruszka": 78, "jabłko": 82, "klepsydra": 74, "prostokąt": 86, "odwrócony trójkąt": 75},
        "leniency": 2.5,
    },
    "bluza z kapturem": {
        "family": "top",
        "zones": ["chest", "waist", "arm_biceps"],
        "zone_weights": {"chest": 1.0, "waist": 0.45, "arm_biceps": 0.8},
        "style_scores": {"gruszka": 76, "jabłko": 82, "klepsydra": 72, "prostokąt": 86, "odwrócony trójkąt": 74},
        "leniency": 3.0,
    },
    "garnitur": {
        "family": "suit",
        "zones": ["chest", "waist", "hips", "thigh", "arm_biceps"],
        "zone_weights": {"chest": 1.2, "waist": 1.0, "hips": 0.8, "thigh": 0.8, "arm_biceps": 0.9},
        "style_scores": {"gruszka": 78, "jabłko": 80, "klepsydra": 84, "prostokąt": 88, "odwrócony trójkąt": 86},
        "leniency": 0.0,
    },
}

# aliases from ingest / demo
DRESS_TYPE_STYLE_MAP = {
    "wrap": {"gruszka": 91, "jabłko": 84, "klepsydra": 96, "prostokąt": 85, "odwrócony trójkąt": 84},
    "a_line": {"gruszka": 93, "jabłko": 86, "klepsydra": 90, "prostokąt": 80, "odwrócony trójkąt": 92},
    "fit_and_flare": {"gruszka": 95, "jabłko": 82, "klepsydra": 92, "prostokąt": 90, "odwrócony trójkąt": 91},
    "bodycon": {"gruszka": 58, "jabłko": 52, "klepsydra": 89, "prostokąt": 68, "odwrócony trójkąt": 70},
    "shift": {"gruszka": 74, "jabłko": 91, "klepsydra": 74, "prostokąt": 82, "odwrócony trójkąt": 79},
    "slip": {"gruszka": 70, "jabłko": 66, "klepsydra": 78, "prostokąt": 72, "odwrócony trójkąt": 71},
}


@dataclass
class UserProfile:
    height_cm: float
    weight_kg: float
    age: Optional[int]
    body_type: str
    fit_preference: Optional[str]
    build_type: str
    bust_cm: float
    waist_cm: float
    hips_cm: float
    extra_measurements: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProductProfileLite:
    brand: str
    name: str
    dress_type: str
    fit_type: str
    stretch_level: str
    length_type: str
    style_effect: str
    runs_small: float
    runs_large: float
    true_to_size: float
    tight_areas: List[str]
    review_count: int
    review_lines: List[str]
    size_chart: Dict[str, Dict[str, List[float]]]
    product_kind: str = "sukienka casualowa"
    search_group: str = "damskie"
    style_branch: str = "casual"


def _normalize_fit_preference(value: Optional[str]) -> str:
    if value in {"standard", "dopasowane", "luźniejsze"}:
        return value
    return "standard"


def _size_rank(label: str) -> float:
    txt = (label or "").upper().strip()
    return BASE_SIZE_RANK.get(txt, 99.0)


def ordered_size_labels(labels: List[str]) -> List[str]:
    return sorted(labels, key=lambda x: (_size_rank(x), x))


def get_category_profile(product_kind: str) -> Dict:
    return CATEGORY_PROFILES.get(product_kind, CATEGORY_PROFILES["sukienka casualowa"])


def shape_match_score(body_type: str, product: ProductProfileLite) -> float:
    # if selected category is a dress and parser found a better sub-type, use style map for that cut
    if get_category_profile(product.product_kind).get("family") == "dress" and product.dress_type in DRESS_TYPE_STYLE_MAP:
        return float(DRESS_TYPE_STYLE_MAP[product.dress_type].get(body_type, 78))
    profile = get_category_profile(product.product_kind)
    return float(profile["style_scores"].get(body_type, 78))


def _review_modifier(product: ProductProfileLite) -> float:
    return round((product.runs_small - product.runs_large) * 1.8, 2)


def _preference_bonus(user: UserProfile, product: ProductProfileLite) -> float:
    pref = _normalize_fit_preference(user.fit_preference)
    bonus = 0.0
    if pref == "luźniejsze":
        bonus -= 4.0
    elif pref == "dopasowane":
        bonus += 2.0
    if product.fit_type == "loose" and pref == "luźniejsze":
        bonus += 1.5
    return bonus


def _chart_aliases(zone: str, product_kind: str) -> List[str]:
    if zone == "chest":
        return ["chest", "bust"]
    if zone == "bust":
        return ["bust", "chest"]
    if zone == "waist":
        return ["waist", "abdomen"]
    if zone == "hips":
        return ["hips"]
    if zone == "thigh":
        return ["thigh", "hips"]  # fallback via hips if chart lacks thigh
    if zone == "arm_biceps":
        return ["arm", "bust", "chest"]
    return [zone]


def _user_measure(zone: str, user: UserProfile, product_kind: str) -> float:
    extra = user.extra_measurements or {}
    if zone == "bust":
        return float(user.bust_cm)
    if zone == "chest":
        return float(extra.get("chest_cm") or user.bust_cm)
    if zone == "waist":
        return float(user.waist_cm)
    if zone == "hips":
        return float(user.hips_cm)
    if zone == "thigh":
        return float(extra.get("thigh_cm") or max(user.hips_cm * 0.62, 48.0))
    if zone == "arm_biceps":
        return float(extra.get("arm_biceps_cm") or max((extra.get("chest_cm") or user.bust_cm) * 0.34, 24.0))
    if zone == "abdomen":
        return float(extra.get("abdomen_cm") or max(user.waist_cm, user.waist_cm + 4))
    return 0.0


def _effective_chart_range(chart_row: Dict[str, List[float]], zone: str, product_kind: str) -> Optional[Tuple[float, float, str]]:
    for alias in _chart_aliases(zone, product_kind):
        if alias in chart_row:
            low, high = chart_row[alias]
            # heuristic conversions when chart misses category-specific measure
            if zone == "thigh" and alias == "hips":
                return low * 0.58, high * 0.62, alias
            if zone == "arm_biceps" and alias in {"bust", "chest"}:
                return low * 0.30, high * 0.36, alias
            if zone == "waist" and alias == "abdomen":
                return low * 0.93, high * 0.98, alias
            return float(low), float(high), alias
    return None


def _fit_chart(user: UserProfile, product: ProductProfileLite, chart_row: Dict[str, List[float]], stretch_bonus: float, review_modifier: float, user_bonus: float) -> tuple[bool, Dict[str, float], List[str], List[str], Dict[str, str]]:
    category = get_category_profile(product.product_kind)
    deltas: Dict[str, float] = {}
    issues: List[str] = []
    unavailable: List[str] = []
    sources: Dict[str, str] = {}
    for zone in category["zones"]:
        chart_range = _effective_chart_range(chart_row, zone, product.product_kind)
        measure = _user_measure(zone, user, product.product_kind)
        if not chart_range or measure <= 0:
            unavailable.append(zone)
            continue
        lower, upper, source_alias = chart_range
        zone_weight = category["zone_weights"].get(zone, 1.0)
        fit_type_penalty = 0.7 if product.fit_type == "slim" else 0.0
        effective_upper = upper + stretch_bonus + category.get("leniency", 0.0) - review_modifier - user_bonus - fit_type_penalty
        deltas[zone] = round(effective_upper - measure, 1)
        sources[zone] = source_alias
        if measure > effective_upper:
            issues.append(zone)
    fits = not issues and bool(deltas)
    return fits, deltas, issues, unavailable, sources


def _base_fit_score(deltas: Dict[str, float], issues: List[str], unavailable: List[str], product: ProductProfileLite) -> float:
    category = get_category_profile(product.product_kind)
    zone_weights = category["zone_weights"]
    if not deltas:
        return 56.0
    score = 82.0
    for zone, delta in deltas.items():
        weight = zone_weights.get(zone, 1.0)
        if delta >= 4:
            score += 4 * weight
        elif delta >= 2:
            score += 2 * weight
        elif delta >= 0:
            score += 0
        elif delta >= -2:
            score -= 10 * weight
        else:
            score -= 18 * weight
    score -= len(issues) * 5.5
    score -= len(unavailable) * 2.5
    return max(0.0, min(100.0, score))


def _translate_flags(issues: List[str], product: ProductProfileLite, unavailable: List[str], sources: Dict[str, str]) -> List[str]:
    pretty = {
        "bust": "biust",
        "chest": "klatka piersiowa",
        "waist": "talia",
        "hips": "biodra",
        "thigh": "udo",
        "arm_biceps": "ramię / biceps",
    }
    flags = [f"Ryzyko napięcia w strefie: {pretty.get(i, i)}." for i in issues]
    for zone in unavailable:
        flags.append(f"Producent nie podał czytelnego wymiaru dla strefy: {pretty.get(zone, zone)} — ta część oceny jest mniej pewna.")
    for zone, alias in sources.items():
        if zone != alias and zone in {"thigh", "arm_biceps"}:
            flags.append(f"Dla strefy {pretty.get(zone, zone)} użyto przybliżenia na podstawie pola tabeli: {alias}.")
    if product.product_kind in {"gorset", "żakiet", "marynarka", "garnitur"} and product.fit_type == "slim":
        flags.append("To fason konstrukcyjny o mniejszym marginesie błędu — rozmiar musi być dobrze trafiony.")
    if product.product_kind in {"legginsy", "spodnie jeansowe"} and "thigh" in issues:
        flags.append("W tej kategorii obwód uda i bioder bywa ważniejszy niż sama talia.")
    if product.runs_small >= 0.45:
        flags.append("Recenzje sugerują, że model bywa odczuwalnie mniejszy niż tabela producenta.")
    if product.fit_type == "slim" and product.stretch_level == "low":
        flags.append("To fason mało wybaczający — margines błędu rozmiaru jest niewielki.")
    if product.product_kind in {"bluza", "bluza z kapturem", "T-shirt"} and product.fit_type == "loose":
        flags.append("W tej kategorii większy luz bywa częścią zamierzonego efektu, więc niewielki zapas nie musi oznaczać złego fitu.")
    return flags


def decide_verdict(score: float) -> str:
    if score >= 81:
        return "bierz"
    if score >= 63:
        return "ostrożnie"
    return "raczej odpuść"


def _product_label(product: ProductProfileLite) -> str:
    return product.product_kind or product.name or "produkt"


def build_explanation(user: UserProfile, product: ProductProfileLite, verdict: str, chosen_size: str, alternate_size: str, fit_score: float, style_score: float, flags: List[str], measures: Dict[str, float], used_zones: List[str]) -> str:
    seed_src = f"{user.body_type}-{product.brand}-{product.name}-{product.product_kind}-{chosen_size}-{verdict}"
    seed = int(hashlib.md5(seed_src.encode('utf-8')).hexdigest()[:8], 16)
    rnd = random.Random(seed)
    product_label = _product_label(product)

    openers = {
        "bierz": [
            f"Ten produkt wypada całkiem mocno dla Twojej sylwetki i wymiarów.",
            f"Na papierze {product_label} wygląda jak sensowny wybór dla Ciebie.",
            f"Tu konstrukcja produktu i Twoje proporcje dość dobrze się zazębiają.",
        ],
        "ostrożnie": [
            f"{product_label.capitalize()} jest do obrony, ale wymaga uważnego wyboru rozmiaru.",
            f"To może zadziałać, choć nie jest to najłatwiejszy zakup w ciemno.",
            f"Ten fason nie jest zły, ale decyzja rozmiarowa ma tu spore znaczenie.",
        ],
        "raczej odpuść": [
            f"W obecnym układzie {product_label} wygląda ryzykownie dla Twoich wymiarów.",
            f"To raczej nie jest najbezpieczniejszy wybór bez przymiarki.",
            f"Ten krój i tabela rozmiarów nie układają się tu zbyt korzystnie.",
        ],
    }
    size_lines = [
        f"Najbardziej spójnie wypada u Ciebie rozmiar {chosen_size}.",
        f"Najmocniejsza rekomendacja to rozmiar {chosen_size}.",
        f"Dla Twoich wymiarów najlepiej wygląda rozmiar {chosen_size}.",
    ]
    alt_lines = [
        f"Rozmiar {alternate_size} warto trzymać jako plan B, jeśli lubisz więcej luzu.",
        f"Alternatywnie możesz rozważyć {alternate_size}, gdy cenisz bardziej swobodny komfort.",
        f"{alternate_size} ma sens jako zapasowy wariant przy bardziej luźnym efekcie.",
    ]
    style_good = [
        "Sam typ produktu dobrze współpracuje z Twoją sylwetką i nie walczy z proporcjami.",
        "Kategoria i fason są dość zgodne z Twoim typem sylwetki.",
        "To jest jedna z tych form, które zazwyczaj dobrze układają się przy takich proporcjach.",
    ]
    style_mid = [
        "Kategoria jest przyzwoita, ale nie daje maksymalnej przewagi stylistycznej.",
        "Ten typ produktu jest okej, choć nie jest najbardziej uprzywilejowany dla Twojej figury.",
        "Forma produktu jest poprawna, ale bez dużego bonusu estetycznego.",
    ]
    style_low = [
        "To nie jest najbardziej korzystny typ produktu dla Twojej sylwetki.",
        "Ta forma może podkreślić strefy, których nie chcesz najmocniej eksponować.",
        "Kategoria produktu daje tu słabszy efekt wizualny niż inne możliwe wybory.",
    ]

    parts = [rnd.choice(openers[verdict]), rnd.choice(size_lines)]
    if alternate_size != chosen_size:
        parts.append(rnd.choice(alt_lines))
    if style_score >= 88:
        parts.append(rnd.choice(style_good))
    elif style_score >= 75:
        parts.append(rnd.choice(style_mid))
    else:
        parts.append(rnd.choice(style_low))

    if product.runs_small >= 0.45:
        parts.append("Duży sygnał z opinii: model ma tendencję do zaniżonej rozmiarówki, więc tabela producenta może być zbyt optymistyczna.")
    elif product.runs_large >= 0.45:
        parts.append("Recenzje sugerują lekko zawyżoną rozmiarówkę, więc uważaj, żeby nie skończyć z nadmiarem luzu.")
    elif product.review_count > 0:
        parts.append("Sygnały z opinii są raczej spokojne i nie pokazują dużego konfliktu z tabelą rozmiarów.")
    else:
        parts.append("Tu opieramy się głównie na tabeli i konstrukcji produktu, bo sygnał z opinii jest słaby albo niewidoczny.")

    if flags:
        parts.append("Największe punkty uwagi: " + " ".join(flags))
    else:
        parts.append("Nie widać czerwonej flagi, która sama w sobie dyskwalifikowałaby ten model.")

    readable_measures = []
    pretty = {"bust": "biust", "chest": "klatka", "waist": "talia", "hips": "biodra", "thigh": "udo", "arm_biceps": "ramię"}
    for zone in used_zones:
        if zone in measures:
            readable_measures.append(f"{pretty.get(zone, zone)} {measures[zone]:.1f} cm")
    if readable_measures:
        parts.append("Do decyzji przyjęto: " + ", ".join(readable_measures) + ".")
    parts.append(f"Łączny wynik dopasowania to {fit_score:.0f}/100, a zgodność kategorii z sylwetką {style_score:.0f}/100.")
    return " ".join(parts)


def recommend_size(user: UserProfile, product: ProductProfileLite) -> Dict:
    category = get_category_profile(product.product_kind)
    stretch_bonus = STRETCH_CM.get(product.stretch_level, 0)
    review_modifier = _review_modifier(product)
    preference_bonus = _preference_bonus(user, product)
    size_order = ordered_size_labels(list(product.size_chart.keys()))

    user_measures = {zone: _user_measure(zone, user, product.product_kind) for zone in category["zones"]}

    candidates = []
    for size in size_order:
        if size not in product.size_chart:
            continue
        fits, deltas, issues, unavailable, sources = _fit_chart(user, product, product.size_chart[size], stretch_bonus, review_modifier, preference_bonus)
        base_score = _base_fit_score(deltas, issues, unavailable, product)
        candidates.append({
            "size": size,
            "fits": fits,
            "deltas": deltas,
            "issues": issues,
            "unavailable": unavailable,
            "sources": sources,
            "base_score": round(base_score, 1),
        })

    if not candidates:
        return {
            "recommended_size": "brak danych",
            "alternate_size": "brak danych",
            "fit_score": 55.0,
            "dress_match_score": shape_match_score(user.body_type, product),
            "verdict": "ostrożnie",
            "risk_flags": ["Nie udało się odczytać żadnej użytecznej tabeli rozmiarów dla tego produktu."],
            "estimated_measurements": user_measures,
            "explanation": f"Aplikacja nie znalazła czytelnych danych rozmiarowych dla kategorii: {product.product_kind}. W tej sytuacji rekomendacja byłaby zbyt słaba, więc lepiej oprzeć decyzję na przymiarce albo zdjęciu tabeli rozmiarów ze strony.",
        }

    chosen = next((c for c in candidates if c["fits"]), None)
    if not chosen:
        chosen = max(candidates, key=lambda c: c["base_score"])

    pref = _normalize_fit_preference(user.fit_preference)
    idx = size_order.index(chosen["size"])
    if pref == "luźniejsze" and idx < len(size_order) - 1:
        chosen_size = size_order[idx + 1]
    elif pref == "dopasowane" and idx > 0 and chosen["base_score"] > 76:
        chosen_size = chosen["size"]
    else:
        chosen_size = chosen["size"]

    if pref == "dopasowane":
        alternate_size = chosen_size
    else:
        alt_idx = min(size_order.index(chosen_size) + 1, len(size_order) - 1)
        alternate_size = size_order[alt_idx]

    style_score = shape_match_score(user.body_type, product)
    issue_flags = _translate_flags(chosen["issues"], product, chosen["unavailable"], chosen["sources"])
    fit_score = round(chosen["base_score"] * 0.64 + style_score * 0.26 + (100 - min(len(issue_flags) * 7, 28)) * 0.10, 1)
    verdict = decide_verdict(fit_score)
    explanation = build_explanation(user, product, verdict, chosen_size, alternate_size, fit_score, style_score, issue_flags, user_measures, category["zones"])

    return {
        "recommended_size": chosen_size,
        "alternate_size": alternate_size,
        "fit_score": fit_score,
        "dress_match_score": round(style_score, 1),
        "verdict": verdict,
        "risk_flags": issue_flags,
        "estimated_measurements": user_measures,
        "explanation": explanation,
    }
