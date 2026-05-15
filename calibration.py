from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

from body_analysis import classify_body_type
from database import calibration_summary, get_calibration_profile, recent_calibration_examples


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def confidence_band(confidence: float) -> str:
    if confidence >= 0.82:
        return 'high'
    if confidence >= 0.62:
        return 'medium'
    return 'low'


def normalize_clothing_fit_bucket(clothing_fit: str) -> str:
    mapping = {
        'obcisłe': 'fitted',
        'raczej dopasowane': 'fitted',
        'średnie': 'regular',
        'nie wiem': 'regular',
        'luźne': 'loose',
        'bardzo luźne': 'loose',
    }
    return mapping.get(clothing_fit, 'regular')


def photo_quality_bucket(confidence: float, front_capture: Dict | None = None, profile_capture: Dict | None = None) -> str:
    front_capture = front_capture or {}
    profile_capture = profile_capture or {}
    min_score = min(float(front_capture.get('score', confidence or 0.0)), float(profile_capture.get('score', confidence or 0.0)))
    status_codes = {front_capture.get('status_code', ''), profile_capture.get('status_code', '')}
    if 'reject' in status_codes:
        return 'low_usable'
    if confidence >= 0.82 and min_score >= 0.84 and status_codes <= {'accept', ''}:
        return 'excellent'
    if confidence >= 0.62 and min_score >= 0.68:
        return 'good'
    return 'low_usable'


def sample_weight(confidence: float, clothing_fit: str = 'średnie', photo_quality: str = 'good') -> float:
    base = 0.35 + 0.9 * float(confidence)
    clothing_mult = {'fitted': 1.0, 'regular': 0.82, 'loose': 0.45}
    quality_mult = {'excellent': 1.0, 'good': 0.85, 'low_usable': 0.55}
    base *= clothing_mult.get(normalize_clothing_fit_bucket(clothing_fit), 0.82)
    base *= quality_mult.get(photo_quality, 0.85)
    return round(_clamp(base, 0.10, 1.35), 2)


@dataclass
class CalibrationCorrection:
    applied: bool
    raw: Dict[str, float]
    corrected: Dict[str, float]
    offsets_cm: Dict[str, float]
    used_scopes: List[str]
    sample_count: float
    confidence_band: str
    photo_quality_bucket: str
    product_kind: str
    clothing_fit_bucket: str
    strength: float
    note: str
    post_abs_error: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


def get_calibration_correction(
    *,
    raw_measures: Dict[str, float],
    gender: str,
    confidence: float,
    front_capture: Dict | None,
    profile_capture: Dict | None,
    product_kind: str,
    clothing_fit: str,
) -> CalibrationCorrection:
    raw = {
        'bust': round(float(raw_measures['bust']), 1),
        'waist': round(float(raw_measures['waist']), 1),
        'hips': round(float(raw_measures['hips']), 1),
    }
    band = confidence_band(confidence)
    quality = photo_quality_bucket(confidence, front_capture, profile_capture)
    fit_bucket = normalize_clothing_fit_bucket(clothing_fit)
    profile = get_calibration_profile(
        gender=gender,
        photo_quality_bucket=quality,
        product_kind=product_kind,
        clothing_fit_bucket=fit_bucket,
    )

    if not profile.get('found'):
        return CalibrationCorrection(
            applied=False,
            raw=raw,
            corrected=dict(raw),
            offsets_cm={'bust': 0.0, 'waist': 0.0, 'hips': 0.0},
            used_scopes=[],
            sample_count=0.0,
            confidence_band=band,
            photo_quality_bucket=quality,
            product_kind=product_kind,
            clothing_fit_bucket=fit_bucket,
            strength=0.0,
            note='Brak wystarczającej historii dla segmentu płeć × jakość zdjęcia × typ produktu × dopasowanie ubrania — na tym etapie nie zastosowano korekty kalibracyjnej.',
            post_abs_error={'bust': 0.0, 'waist': 0.0, 'hips': 0.0},
        )

    raw_offsets = profile['offsets']
    caps = {'bust': 12.0, 'waist': 10.0, 'hips': 12.0}
    offsets = {zone: round(_clamp(float(raw_offsets.get(zone, 0.0)), -caps[zone], caps[zone]), 2) for zone in caps}
    corrected = {
        'bust': round(_clamp(raw['bust'] + offsets['bust'], 60.0, 170.0), 1),
        'waist': round(_clamp(raw['waist'] + offsets['waist'], 48.0, 160.0), 1),
        'hips': round(_clamp(raw['hips'] + offsets['hips'], 70.0, 180.0), 1),
    }
    note = (
        f"Zastosowano segmentowaną korektę kalibracyjną na bazie {profile['sample_count']} podobnych przypadków "
        f"(scope: {profile['scope']}, jakość: {quality}, produkt: {product_kind}, strój: {fit_bucket}, strength: {profile['strength']:.2f})."
    )
    return CalibrationCorrection(
        applied=True,
        raw=raw,
        corrected=corrected,
        offsets_cm=offsets,
        used_scopes=list(profile.get('used_scopes', [])) or [profile.get('scope', 'unknown')],
        sample_count=float(profile['sample_count']),
        confidence_band=band,
        photo_quality_bucket=quality,
        product_kind=product_kind,
        clothing_fit_bucket=fit_bucket,
        strength=float(profile.get('strength', 0.0)),
        note=note,
        post_abs_error=profile.get('displayed_abs_error', {'bust': 0.0, 'waist': 0.0, 'hips': 0.0}),
    )


def compare_ai_vs_manual(*, raw_measures: Dict[str, float], displayed_measures: Dict[str, float], manual_measures: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    raw = {k: round(float(v), 1) for k, v in raw_measures.items()}
    displayed = {k: round(float(v), 1) for k, v in displayed_measures.items()}
    manual = {k: round(float(v), 1) for k, v in manual_measures.items()}
    return {
        'raw_ai': raw,
        'displayed_ai': displayed,
        'manual': manual,
        'raw_error': {k: round(manual[k] - raw[k], 1) for k in manual},
        'displayed_error': {k: round(manual[k] - displayed[k], 1) for k in manual},
        'manual_body_type': classify_body_type(manual['bust'], manual['waist'], manual['hips']),
    }


def calibration_overview(limit: int = 30):
    return recent_calibration_examples(limit)
