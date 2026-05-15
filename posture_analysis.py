from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

from body_analysis import HAS_MEDIAPIPE, _extract_silhouette, _mean_pts, _pt, _distance


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _score_from_value(value: float, low: float, high: float) -> int:
    if value <= low:
        return 0
    scale = (value - low) / max(high - low, 1e-6)
    return int(round(_clamp(1 + scale * 6, 1, 7)))


def _torso_len(landmarks: Dict[str, Tuple[float, float, float]]) -> float:
    sh = _mean_pts(landmarks, ["left_shoulder", "right_shoulder"])
    hp = _mean_pts(landmarks, ["left_hip", "right_hip"])
    if sh and hp:
        return max(1.0, _distance(sh, hp))
    return 1.0


def _mean_visibility(landmarks: Dict[str, Tuple[float, float, float]], names: List[str]) -> float:
    vals = [landmarks[n][2] for n in names if n in landmarks]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _make_finding(code: str, label: str, score: int, confidence: float, description: str, evidence: str) -> Dict:
    return {
        'code': code,
        'label': label,
        'score_1_10': int(score),
        'confidence': round(_clamp(confidence, 0.25, 0.95), 2),
        'description': description,
        'evidence': evidence,
    }


def _front_findings(landmarks: Dict[str, Tuple[float, float, float]]) -> List[Dict]:
    findings: List[Dict] = []
    torso = _torso_len(landmarks)

    lsh = _pt(landmarks, 'left_shoulder')
    rsh = _pt(landmarks, 'right_shoulder')
    lhp = _pt(landmarks, 'left_hip')
    rhp = _pt(landmarks, 'right_hip')
    lkn = _pt(landmarks, 'left_knee')
    rkn = _pt(landmarks, 'right_knee')
    lan = _pt(landmarks, 'left_ankle')
    ran = _pt(landmarks, 'right_ankle')
    shoulder_mid = _mean_pts(landmarks, ['left_shoulder', 'right_shoulder'])
    hip_mid = _mean_pts(landmarks, ['left_hip', 'right_hip'])
    ankle_mid = _mean_pts(landmarks, ['left_ankle', 'right_ankle', 'left_foot_index', 'right_foot_index'])

    if lsh and rsh:
        ratio = abs(lsh[1] - rsh[1]) / torso
        score = _score_from_value(ratio, 0.02, 0.08)
        if score:
            conf = _mean_visibility(landmarks, ['left_shoulder', 'right_shoulder', 'nose'])
            findings.append(_make_finding(
                'shoulder_asymmetry',
                'Możliwa asymetria barków',
                score,
                0.45 + conf * 0.4,
                'Linia barków wygląda na nierówną w ujęciu frontalnym.',
                f'Różnica wysokości barków ≈ {ratio*100:.1f}% długości tułowia.',
            ))

    if lhp and rhp:
        ratio = abs(lhp[1] - rhp[1]) / torso
        score = _score_from_value(ratio, 0.02, 0.08)
        if score:
            conf = _mean_visibility(landmarks, ['left_hip', 'right_hip'])
            findings.append(_make_finding(
                'hip_asymmetry',
                'Możliwa asymetria bioder / miednicy',
                score,
                0.42 + conf * 0.35,
                'Linia bioder wygląda na nierówną w ujęciu frontalnym.',
                f'Różnica wysokości bioder ≈ {ratio*100:.1f}% długości tułowia.',
            ))

    if shoulder_mid and ankle_mid:
        dx = shoulder_mid[0] - ankle_mid[0]
        dy = max(1.0, ankle_mid[1] - shoulder_mid[1])
        tilt_deg = abs(math.degrees(math.atan2(dx, dy)))
        score = _score_from_value(tilt_deg, 3.0, 12.0)
        if score:
            conf = _mean_visibility(landmarks, ['left_shoulder', 'right_shoulder', 'left_ankle', 'right_ankle'])
            findings.append(_make_finding(
                'axis_asymmetry',
                'Możliwa asymetria osi sylwetki',
                score,
                0.40 + conf * 0.35,
                'Oś barki–tułów–stopy nie wygląda na całkowicie pionową.',
                f'Odchylenie osi od pionu ≈ {tilt_deg:.1f}°.',
            ))

    if lan and ran:
        ratio = abs(lan[1] - ran[1]) / torso
        score = _score_from_value(ratio, 0.015, 0.05)
        if score:
            conf = _mean_visibility(landmarks, ['left_ankle', 'right_ankle'])
            findings.append(_make_finding(
                'leg_length_asymmetry',
                'Możliwa asymetria kończyn dolnych',
                score,
                0.34 + conf * 0.35,
                'Poziom kostek wygląda na nierówny.',
                f'Różnica wysokości kostek ≈ {ratio*100:.1f}% długości tułowia.',
            ))

    if lkn and rkn and lan and ran:
        knee_gap = abs(lkn[0] - rkn[0])
        ankle_gap = abs(lan[0] - ran[0])
        if ankle_gap > 1:
            ratio = knee_gap / ankle_gap
            conf = _mean_visibility(landmarks, ['left_knee', 'right_knee', 'left_ankle', 'right_ankle'])
            if ratio < 0.82:
                score = _score_from_value(0.82 - ratio, 0.02, 0.22)
                if score:
                    findings.append(_make_finding(
                        'knee_valgus_tendency',
                        'Możliwa tendencja do koślawości kolan',
                        score,
                        0.34 + conf * 0.35,
                        'Kolana wizualnie zbiegają się bardziej niż kostki.',
                        f'Stosunek rozstawu kolan do kostek ≈ {ratio:.2f}.',
                    ))
            elif ratio > 1.22:
                score = _score_from_value(ratio - 1.22, 0.02, 0.28)
                if score:
                    findings.append(_make_finding(
                        'knee_varus_tendency',
                        'Możliwa tendencja do szpotawości kolan',
                        score,
                        0.34 + conf * 0.35,
                        'Kolana wizualnie są szerzej niż kostki.',
                        f'Stosunek rozstawu kolan do kostek ≈ {ratio:.2f}.',
                    ))

    return findings


def _profile_direction(landmarks: Dict[str, Tuple[float, float, float]]) -> int:
    nose = _pt(landmarks, 'nose')
    shoulder = _mean_pts(landmarks, ['left_shoulder', 'right_shoulder'])
    if not (nose and shoulder):
        return 1
    return 1 if nose[0] >= shoulder[0] else -1


def _profile_findings(landmarks: Dict[str, Tuple[float, float, float]]) -> List[Dict]:
    findings: List[Dict] = []
    torso = _torso_len(landmarks)
    dir_sign = _profile_direction(landmarks)
    nose = _pt(landmarks, 'nose')
    ear = _mean_pts(landmarks, ['left_ear', 'right_ear'])
    shoulder = _mean_pts(landmarks, ['left_shoulder', 'right_shoulder'])
    hip = _mean_pts(landmarks, ['left_hip', 'right_hip'])
    ankle = _mean_pts(landmarks, ['left_ankle', 'right_ankle', 'left_foot_index', 'right_foot_index'])

    if nose and shoulder:
        forward = ((nose[0] - shoulder[0]) * dir_sign) / torso
        score = _score_from_value(forward, 0.06, 0.18)
        if score:
            conf = _mean_visibility(landmarks, ['nose', 'left_shoulder', 'right_shoulder'])
            findings.append(_make_finding(
                'forward_head_tendency',
                'Możliwa tendencja do wysunięcia głowy',
                score,
                0.42 + conf * 0.35,
                'Głowa wygląda na ustawioną bardziej do przodu względem barków.',
                f'Wysunięcie głowy ≈ {forward*100:.1f}% długości tułowia.',
            ))

    if shoulder and hip:
        shoulder_forward = ((shoulder[0] - hip[0]) * dir_sign) / torso
        score = _score_from_value(shoulder_forward, 0.05, 0.16)
        if score:
            conf = _mean_visibility(landmarks, ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'])
            findings.append(_make_finding(
                'rounded_shoulders_tendency',
                'Możliwe zaokrąglenie barków',
                score,
                0.40 + conf * 0.35,
                'Linia barków wygląda na ustawioną bardziej do przodu.',
                f'Przesunięcie barków względem bioder ≈ {shoulder_forward*100:.1f}% długości tułowia.',
            ))

    if hip and ankle:
        pelvis_shift = ((hip[0] - ankle[0]) * dir_sign) / torso
        score = _score_from_value(abs(pelvis_shift), 0.05, 0.16)
        if score:
            conf = _mean_visibility(landmarks, ['left_hip', 'right_hip', 'left_ankle', 'right_ankle'])
            findings.append(_make_finding(
                'pelvic_tilt_tendency',
                'Możliwa tendencja ustawienia miednicy poza osią',
                score,
                0.32 + conf * 0.30,
                'Miednica wygląda na przesuniętą względem linii bark–kostka.',
                f'Przesunięcie miednicy względem kostek ≈ {pelvis_shift*100:.1f}% długości tułowia.',
            ))

    return findings


def _dedupe_findings(findings: List[Dict]) -> List[Dict]:
    best: Dict[str, Dict] = {}
    for f in findings:
        current = best.get(f['code'])
        if current is None or (f['score_1_10'], f['confidence']) > (current['score_1_10'], current['confidence']):
            best[f['code']] = f
    return sorted(best.values(), key=lambda x: (-x['score_1_10'], -x['confidence'], x['label']))


def analyze_posture_from_images(
    front_image_bytes: bytes,
    profile_image_bytes: bytes,
    gender: str = 'kobieta',
    clothing_fit_answer: str = 'średnie',
    back_image_bytes: bytes | None = None,
) -> Dict:
    if not HAS_MEDIAPIPE:
        return {
            'available': False,
            'detected': False,
            'message': 'Analiza postawy wymaga MediaPipe. W tej konfiguracji działa tylko pomiar sylwetki bez posture screening.',
            'findings': [],
            'overall_score_1_10': 0,
            'scale_note': 'Skala 1–10, ale w wersji konsumenckiej aplikacja pokazuje maksymalnie 7/10.',
        }

    front = _extract_silhouette(front_image_bytes, 'front', clothing_fit_answer)
    profile = _extract_silhouette(profile_image_bytes, 'profile', clothing_fit_answer)
    back = _extract_silhouette(back_image_bytes, 'front', clothing_fit_answer) if back_image_bytes is not None else None
    if front.quality.status_code == 'reject' or profile.quality.status_code == 'reject':
        return {
            'available': False,
            'detected': False,
            'message': 'Analiza postawy nie została wykonana, bo co najmniej jedno zdjęcie nie przeszło quality gate.',
            'findings': [],
            'overall_score_1_10': 0,
            'scale_note': 'Skala 1–10, ale w wersji konsumenckiej aplikacja pokazuje maksymalnie 7/10.',
        }

    findings = _front_findings(front.landmarks) + _profile_findings(profile.landmarks)
    if back is not None and back.quality.status_code == 'accept':
        back_findings = _front_findings(back.landmarks)
        for f in back_findings:
            f['confidence'] = min(0.97, round(f['confidence'] + 0.08, 2))
            f['description'] = f['description'] + ' Dodatkowe potwierdzenie pochodzi z ujęcia TYŁ.'
        findings += back_findings
    findings = _dedupe_findings(findings)
    if not findings:
        return {
            'available': True,
            'detected': False,
            'message': 'Nie wykryto istotnej cechy postawy wpływającej wizualnie na dobór kreacji.',
            'findings': [],
            'overall_score_1_10': 0,
            'scale_note': 'Skala 1–10, ale w wersji konsumenckiej aplikacja pokazuje maksymalnie 7/10.',
            'quality_context': {
                'front_score': front.quality.score,
                'profile_score': profile.quality.score,
                'back_score': back.quality.score if back is not None else None,
            },
        }

    overall = max(f['score_1_10'] for f in findings)
    return {
        'available': True,
        'detected': True,
        'message': 'Wykryto wizualne cechy postawy, które mogą wpływać na to, jak układa się kreacja. To screening wizualny, nie diagnoza medyczna.',
        'findings': findings,
        'overall_score_1_10': int(overall),
        'scale_note': 'Skala 1–10, ale w wersji konsumenckiej aplikacja pokazuje maksymalnie 7/10.',
        'quality_context': {
            'front_score': front.quality.score,
            'profile_score': profile.quality.score,
            'back_score': back.quality.score if back is not None else None,
        },
    }


def evaluate_visual_compensation(posture_summary: Dict, product) -> Dict:
    if not posture_summary or not posture_summary.get('available'):
        return {
            'available': False,
            'status': 'unknown',
            'message': 'Brak analizy postawy — nie da się ocenić, czy fason pomoże wizualnie skompensować cechy postawy.',
            'helps': [],
            'risks': [],
            'neutral': [],
        }

    if not posture_summary.get('detected'):
        return {
            'available': True,
            'status': 'neutral',
            'message': 'Nie wykryto istotnej cechy postawy, która wymagałaby specjalnej kompensacji fasonem. Ocenę opieramy głównie na sylwetce i rozmiarze.',
            'helps': [],
            'risks': [],
            'neutral': ['Fason oceniamy głównie przez typ sylwetki i dopasowanie rozmiaru.'],
        }

    dress_type = getattr(product, 'dress_type', '') or product.get('dress_type', '')
    length_type = getattr(product, 'length_type', '') or product.get('length_type', '')
    fit_type = getattr(product, 'fit_type', '') or product.get('fit_type', '')
    product_kind = getattr(product, 'product_kind', '') or getattr(product, 'name', '') or 'produkt'
    kind = str(product_kind).lower()

    is_dress = 'sukien' in kind
    is_skirt = 'spódnic' in kind or 'spodnic' in kind
    is_leggings = 'leggins' in kind
    is_blazer = kind in {'marynarka', 'żakiet', 'zakiet'}
    is_suit = 'garnitur' in kind
    is_pants = 'spodnie' in kind or is_leggings
    is_top = kind in {'t-shirt', 'bluza', 'bluza z kapturem', 'gorset'} or is_blazer or is_suit

    helps: List[str] = []
    risks: List[str] = []
    neutral: List[str] = []

    for f in posture_summary.get('findings', []):
        code = f['code']
        label = f['label']
        if code in {'rounded_shoulders_tendency', 'forward_head_tendency', 'shoulder_asymmetry'}:
            if is_blazer or is_suit or (is_dress and dress_type in {'wrap', 'a_line', 'fit_and_flare'}):
                helps.append(f'{label}: ten krój ma większą szansę optycznie uporządkować górę sylwetki i linię barków.')
            elif kind in {'gorset'} or (is_dress and dress_type in {'bodycon', 'slip'}) or (is_top and fit_type == 'slim'):
                risks.append(f'{label}: bardziej dopasowana góra może mocniej pokazać linię barków i górnej części tułowia.')
            else:
                neutral.append(f'{label}: wpływ fasonu na górę sylwetki wygląda raczej neutralnie.')
        elif code in {'hip_asymmetry', 'axis_asymmetry', 'pelvic_tilt_tendency'}:
            if is_dress and dress_type in {'wrap', 'a_line', 'fit_and_flare'}:
                helps.append(f'{label}: ten fason ma większą szansę wygładzić optycznie okolice talii i bioder.')
            elif is_skirt and fit_type != 'slim':
                helps.append(f'{label}: mniej sztywny dół może wizualnie łagodniej pracować w okolicy bioder i talii.')
            elif is_pants and not is_leggings and fit_type != 'slim':
                neutral.append(f'{label}: wpływ kroju spodni na oś sylwetki będzie umiarkowany, ważne są też długość i linia nogawki.')
            elif is_leggings or (is_dress and dress_type in {'bodycon', 'slip'}) or fit_type == 'slim':
                risks.append(f'{label}: bardziej dopasowany fason może mocniej pokazać linię talii, bioder albo oś sylwetki.')
            else:
                neutral.append(f'{label}: wpływ fasonu na oś sylwetki jest umiarkowany.')
        elif code in {'knee_valgus_tendency', 'knee_varus_tendency', 'leg_length_asymmetry'}:
            if is_dress and length_type in {'midi', 'maxi'}:
                helps.append(f'{label}: długość {length_type} może lepiej zneutralizować optykę linii nóg.')
            elif is_pants and not is_leggings:
                helps.append(f'{label}: dobrze dobrana nogawka może lepiej uspokoić wizualnie linię nóg niż krótki dół.')
            elif is_skirt and length_type == 'mini':
                risks.append(f'{label}: krótka długość może mocniej wyeksponować linię nóg.')
            elif is_leggings:
                risks.append(f'{label}: legginsy często mocno pokazują linię nóg, więc mogą uwidocznić tę cechę.')
            else:
                neutral.append(f'{label}: wpływ długości fasonu na nogi wygląda raczej neutralnie.')
        else:
            neutral.append(f'{label}: brak silnej reguły wizualnej dla tej kategorii produktu.')

    score = len(helps) - len(risks)
    if score > 0:
        status = 'helps'
        message = 'Wybrany fason ma kilka cech, które mogą wizualnie pomóc przy wykrytych cechach postawy.'
    elif score < 0:
        status = 'risk'
        message = 'Wybrany fason może podkreślić część wykrytych cech postawy, więc warto podejść do zakupu ostrożnie.'
    else:
        status = 'neutral'
        message = 'Wpływ fasonu na wykryte cechy postawy wygląda raczej neutralnie.'

    return {
        'available': True,
        'status': status,
        'message': message,
        'helps': helps,
        'risks': risks,
        'neutral': neutral,
    }

