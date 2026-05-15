from __future__ import annotations

from typing import Dict, List


def _flag(code: str, label: str, message: str, level: str = 'warn', affected: List[str] | None = None) -> Dict:
    return {
        'code': code,
        'label': label,
        'message': message,
        'level': level,
        'affected': affected or [],
    }


def analyze_measurement_sanity(measures: Dict[str, float], gender: str, body_type: str, height_cm: float) -> Dict:
    bust = float(measures.get('bust', 0) or 0)
    waist = float(measures.get('waist', 0) or 0)
    hips = float(measures.get('hips', 0) or 0)
    abdomen = float(measures.get('abdomen_cm', 0) or 0)
    thigh = float(measures.get('thigh_cm', 0) or 0)
    arm = float(measures.get('arm_biceps_cm', 0) or 0)
    neck = float(measures.get('neck_cm', 0) or 0)
    wrist = float(measures.get('wrist_cm', 0) or 0)
    calf_max = float(measures.get('calf_max_cm', 0) or 0)
    calf_min = float(measures.get('calf_min_cm', 0) or 0)

    flags: List[Dict] = []
    weak_points: List[str] = []

    if bust > 0 and waist > 0 and waist > bust * 1.10:
        flags.append(_flag('waist_gt_bust', 'Talia nienaturalnie duża względem klatki/biustu', 'Talia wyszła zauważalnie większa od klatki / biustu. Warto potwierdzić talię ręcznie.', 'warn', ['waist']))
        weak_points.append('waist')
    if hips > 0 and waist > 0 and waist > hips * 1.08:
        flags.append(_flag('waist_gt_hips', 'Talia nienaturalnie duża względem bioder', 'Talia wyszła większa od bioder. To często oznacza błąd kadru lub zbyt luźne ubranie.', 'warn', ['waist', 'hips']))
        weak_points.extend(['waist', 'hips'])
    if hips > 0 and bust > 0 and (hips < 0.78 * bust or hips > 1.35 * bust):
        flags.append(_flag('hips_vs_bust', 'Relacja bioder do klatki/biustu wygląda nietypowo', 'Relacja bioder do klatki/biustu jest na granicy sensowności. Potwierdź te strefy ręcznie.', 'warn', ['bust', 'hips']))
        weak_points.extend(['bust', 'hips'])
    if abdomen > 0 and waist > 0 and abdomen < waist - 2:
        flags.append(_flag('abdomen_lt_waist', 'Brzuch mniejszy niż talia', 'Obwód brzucha wyszedł mniejszy niż obwód talii, co zwykle jest mało prawdopodobne. Potwierdź brzuch ręcznie.', 'warn', ['abdomen_cm']))
        weak_points.append('abdomen_cm')
    if calf_max > 0 and calf_min > 0 and calf_min >= calf_max:
        flags.append(_flag('calf_order', 'Łydka min jest większa lub równa łydce max', 'Najwęższe miejsce łydki nie może być większe od najszerszego. Sprawdź oba pola.', 'warn', ['calf_max_cm', 'calf_min_cm']))
        weak_points.extend(['calf_max_cm', 'calf_min_cm'])
    if arm > 0 and wrist > 0 and wrist > arm * 0.80:
        flags.append(_flag('wrist_vs_arm', 'Nadgarstek wygląda zbyt duży względem ramienia', 'Nadgarstek wyszedł bardzo zbliżony do obwodu ramienia. Potwierdź nadgarstek i ramię ręcznie.', 'warn', ['wrist_cm', 'arm_biceps_cm']))
        weak_points.extend(['wrist_cm', 'arm_biceps_cm'])
    if thigh > 0 and calf_max > 0 and calf_max > thigh:
        flags.append(_flag('calf_gt_thigh', 'Łydka większa od uda', 'Najszersza łydka wyszła większa od uda. To zwykle oznacza błąd pomiaru lub kadru.', 'warn', ['thigh_cm', 'calf_max_cm']))
        weak_points.extend(['thigh_cm', 'calf_max_cm'])
    if neck > 0 and bust > 0 and neck > bust * 0.52:
        flags.append(_flag('neck_vs_bust', 'Szyja wygląda zbyt duża względem klatki/biustu', 'Obwód szyi jest bardzo duży względem klatki/biustu. Potwierdź szyję ręcznie.', 'warn', ['neck_cm']))
        weak_points.append('neck_cm')
    if wrist > 0 and (wrist < 11 or wrist > 25):
        flags.append(_flag('wrist_range', 'Nadgarstek poza typowym zakresem', 'Obwód nadgarstka jest poza typowym zakresem. Warto potwierdzić go ręcznie.', 'warn', ['wrist_cm']))
        weak_points.append('wrist_cm')
    if gender == 'kobieta' and bust > 0 and bust < 70:
        flags.append(_flag('bust_low', 'Biust wygląda bardzo mały', 'Biust wyszedł bardzo niski. Sprawdź zdjęcia albo potwierdź ręcznie.', 'warn', ['bust']))
        weak_points.append('bust')
    if gender == 'mężczyzna' and bust > 0 and bust < 80:
        flags.append(_flag('chest_low', 'Klatka piersiowa wygląda bardzo mała', 'Klatka piersiowa wyszła bardzo niska. Potwierdź ją ręcznie.', 'warn', ['bust']))
        weak_points.append('bust')
    # height sanity
    wh_ratio = waist / max(height_cm, 1)
    if waist > 0 and (wh_ratio < 0.28 or wh_ratio > 0.75):
        flags.append(_flag('waist_height_ratio', 'Relacja talii do wzrostu wygląda nietypowo', 'Wynik talii jest nietypowy względem wzrostu. Warto potwierdzić ten wymiar ręcznie.', 'warn', ['waist']))
        weak_points.append('waist')

    # body-type consistency (soft)
    if body_type == 'gruszka' and bust > hips > 0:
        flags.append(_flag('body_type_conflict_pear', 'Wyniki są słabo zgodne z typem sylwetki „gruszka”', 'AI wykryło gruszkę, ale relacja klatki do bioder nie jest typowa. Potwierdź klatkę i biodra.', 'warn', ['bust', 'hips']))
        weak_points.extend(['bust', 'hips'])
    if body_type == 'odwrócony trójkąt' and hips > bust > 0:
        flags.append(_flag('body_type_conflict_triangle', 'Wyniki są słabo zgodne z typem „odwrócony trójkąt”', 'AI wykryło odwrócony trójkąt, ale biodra wyszły większe niż klatka. Potwierdź klatkę i biodra.', 'warn', ['bust', 'hips']))
        weak_points.extend(['bust', 'hips'])

    weak_points = sorted(set(weak_points), key=lambda x: ['bust','waist','hips','abdomen_cm','thigh_cm','arm_biceps_cm','neck_cm','wrist_cm','calf_max_cm','calf_min_cm'].index(x) if x in ['bust','waist','hips','abdomen_cm','thigh_cm','arm_biceps_cm','neck_cm','wrist_cm','calf_max_cm','calf_min_cm'] else 99)

    return {
        'ok': len(flags) == 0,
        'flags': flags,
        'weak_points': weak_points,
        'summary': 'Brak istotnych niespójności pomiarowych.' if not flags else 'Wykryto strefy, które warto potwierdzić ręcznie.',
    }
