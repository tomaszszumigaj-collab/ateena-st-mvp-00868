
from __future__ import annotations

from typing import Dict, Any
import pandas as pd

REQUIRED_COLUMNS = [
    'gender','product_kind','photo_quality_bucket','clothing_fit_bucket',
    'raw_bust_cm','raw_waist_cm','raw_hips_cm',
    'manual_bust_cm','manual_waist_cm','manual_hips_cm'
]

OPTIONAL_COLUMNS = [
    'raw_abdomen_cm','manual_abdomen_cm',
    'raw_thigh_cm','manual_thigh_cm',
    'raw_arm_biceps_cm','manual_arm_biceps_cm',
    'raw_calf_max_cm','manual_calf_max_cm',
    'predicted_accept','manual_should_accept'
]

def template_dataframe() -> pd.DataFrame:
    rows = [
        {'gender':'kobieta','product_kind':'sukienka casualowa','photo_quality_bucket':'good','clothing_fit_bucket':'fitted','raw_bust_cm':92.0,'raw_waist_cm':72.0,'raw_hips_cm':98.0,'manual_bust_cm':90.0,'manual_waist_cm':70.0,'manual_hips_cm':96.0,'raw_abdomen_cm':76.0,'manual_abdomen_cm':74.0,'raw_thigh_cm':57.0,'manual_thigh_cm':55.0,'raw_arm_biceps_cm':29.0,'manual_arm_biceps_cm':28.0,'raw_calf_max_cm':36.0,'manual_calf_max_cm':35.0,'predicted_accept':1,'manual_should_accept':1},
        {'gender':'kobieta','product_kind':'spódniczka','photo_quality_bucket':'excellent','clothing_fit_bucket':'regular','raw_bust_cm':88.0,'raw_waist_cm':69.0,'raw_hips_cm':97.0,'manual_bust_cm':87.0,'manual_waist_cm':68.0,'manual_hips_cm':95.0,'raw_abdomen_cm':73.0,'manual_abdomen_cm':72.0,'raw_thigh_cm':56.0,'manual_thigh_cm':54.0,'raw_arm_biceps_cm':27.0,'manual_arm_biceps_cm':27.0,'raw_calf_max_cm':35.0,'manual_calf_max_cm':34.0,'predicted_accept':1,'manual_should_accept':1},
        {'gender':'mężczyzna','product_kind':'garnitur','photo_quality_bucket':'good','clothing_fit_bucket':'regular','raw_bust_cm':104.0,'raw_waist_cm':88.0,'raw_hips_cm':100.0,'manual_bust_cm':102.0,'manual_waist_cm':86.0,'manual_hips_cm':99.0,'raw_abdomen_cm':91.0,'manual_abdomen_cm':89.0,'raw_thigh_cm':60.0,'manual_thigh_cm':58.0,'raw_arm_biceps_cm':33.0,'manual_arm_biceps_cm':32.0,'raw_calf_max_cm':38.0,'manual_calf_max_cm':37.0,'predicted_accept':1,'manual_should_accept':1},
        {'gender':'mężczyzna','product_kind':'spodnie jeansowe','photo_quality_bucket':'good','clothing_fit_bucket':'fitted','raw_bust_cm':101.0,'raw_waist_cm':84.0,'raw_hips_cm':101.0,'manual_bust_cm':100.0,'manual_waist_cm':82.0,'manual_hips_cm':99.0,'raw_abdomen_cm':87.0,'manual_abdomen_cm':85.0,'raw_thigh_cm':59.0,'manual_thigh_cm':57.0,'raw_arm_biceps_cm':32.0,'manual_arm_biceps_cm':31.0,'raw_calf_max_cm':39.0,'manual_calf_max_cm':38.0,'predicted_accept':1,'manual_should_accept':1},
        {'gender':'kobieta','product_kind':'legginsy','photo_quality_bucket':'usable','clothing_fit_bucket':'fitted','raw_bust_cm':90.0,'raw_waist_cm':71.0,'raw_hips_cm':96.0,'manual_bust_cm':88.0,'manual_waist_cm':69.0,'manual_hips_cm':94.0,'raw_abdomen_cm':75.0,'manual_abdomen_cm':73.0,'raw_thigh_cm':58.0,'manual_thigh_cm':56.0,'raw_arm_biceps_cm':28.0,'manual_arm_biceps_cm':27.0,'raw_calf_max_cm':37.0,'manual_calf_max_cm':35.0,'predicted_accept':0,'manual_should_accept':0},
        {'gender':'mężczyzna','product_kind':'marynarka','photo_quality_bucket':'excellent','clothing_fit_bucket':'regular','raw_bust_cm':107.0,'raw_waist_cm':90.0,'raw_hips_cm':101.0,'manual_bust_cm':105.0,'manual_waist_cm':88.0,'manual_hips_cm':100.0,'raw_abdomen_cm':93.0,'manual_abdomen_cm':90.0,'raw_thigh_cm':61.0,'manual_thigh_cm':59.0,'raw_arm_biceps_cm':34.0,'manual_arm_biceps_cm':33.0,'raw_calf_max_cm':39.0,'manual_calf_max_cm':38.0,'predicted_accept':1,'manual_should_accept':1},
        {'gender':'kobieta','product_kind':'sukienka elegancka','photo_quality_bucket':'usable','clothing_fit_bucket':'loose','raw_bust_cm':95.0,'raw_waist_cm':77.0,'raw_hips_cm':102.0,'manual_bust_cm':91.0,'manual_waist_cm':71.0,'manual_hips_cm':97.0,'raw_abdomen_cm':80.0,'manual_abdomen_cm':75.0,'raw_thigh_cm':60.0,'manual_thigh_cm':56.0,'raw_arm_biceps_cm':30.0,'manual_arm_biceps_cm':28.0,'raw_calf_max_cm':38.0,'manual_calf_max_cm':35.0,'predicted_accept':1,'manual_should_accept':0},
        {'gender':'mężczyzna','product_kind':'T-shirt','photo_quality_bucket':'excellent','clothing_fit_bucket':'fitted','raw_bust_cm':103.0,'raw_waist_cm':86.0,'raw_hips_cm':98.0,'manual_bust_cm':102.0,'manual_waist_cm':85.0,'manual_hips_cm':98.0,'raw_abdomen_cm':89.0,'manual_abdomen_cm':88.0,'raw_thigh_cm':58.0,'manual_thigh_cm':57.0,'raw_arm_biceps_cm':33.0,'manual_arm_biceps_cm':32.0,'raw_calf_max_cm':38.0,'manual_calf_max_cm':37.0,'predicted_accept':1,'manual_should_accept':1},
    ]
    return pd.DataFrame(rows)

def template_csv_bytes() -> bytes:
    return template_dataframe().to_csv(index=False).encode('utf-8')

def _pair_metrics(df: pd.DataFrame, label: str, raw_col: str, man_col: str) -> Dict[str, Any]:
    dd = df.dropna(subset=[raw_col, man_col]).copy()
    if dd.empty:
        return {}
    dd[f'err_{label}'] = (dd[raw_col] - dd[man_col]).abs()
    return {
        f'mae_{label}_cm': round(float(dd[f"err_{label}"].mean()), 2),
        f'median_{label}_cm': round(float(dd[f"err_{label}"].median()), 2),
        f'p90_{label}_cm': round(float(dd[f"err_{label}"].quantile(0.9)), 2),
        f'n_{label}': int(len(dd)),
    }

def evaluate_real_benchmark(df: pd.DataFrame) -> Dict[str, Any]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'Brak wymaganych kolumn benchmarku: {", ".join(missing)}')
    d = df.copy()
    numeric_cols = [c for c in d.columns if c.startswith('raw_') or c.startswith('manual_') or c in ('predicted_accept','manual_should_accept')]
    for c in numeric_cols:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=['raw_bust_cm','raw_waist_cm','raw_hips_cm','manual_bust_cm','manual_waist_cm','manual_hips_cm'])
    if d.empty:
        raise ValueError('Po walidacji nie zostały żadne kompletne rekordy benchmarku.')

    metrics: Dict[str, Any] = {'row_count': int(len(d))}
    pairs = [
        ('bust','raw_bust_cm','manual_bust_cm'),
        ('waist','raw_waist_cm','manual_waist_cm'),
        ('hips','raw_hips_cm','manual_hips_cm'),
        ('abdomen','raw_abdomen_cm','manual_abdomen_cm'),
        ('thigh','raw_thigh_cm','manual_thigh_cm'),
        ('arm_biceps','raw_arm_biceps_cm','manual_arm_biceps_cm'),
        ('calf_max','raw_calf_max_cm','manual_calf_max_cm'),
    ]
    part_summary = []
    for label, raw_col, man_col in pairs:
        if raw_col in d.columns and man_col in d.columns:
            pm = _pair_metrics(d, label, raw_col, man_col)
            metrics.update(pm)
            if pm:
                part_summary.append({
                    'part': label,
                    'mae_cm': pm.get(f'mae_{label}_cm'),
                    'median_cm': pm.get(f'median_{label}_cm'),
                    'p90_cm': pm.get(f'p90_{label}_cm'),
                    'n': pm.get(f'n_{label}')
                })
    metrics['per_part_summary'] = part_summary
    metrics['segments'] = d.groupby(['gender','product_kind']).size().reset_index(name='n').to_dict(orient='records')
    metrics['quality_breakdown'] = d.groupby(['photo_quality_bucket']).size().reset_index(name='n').to_dict(orient='records')
    metrics['clothing_breakdown'] = d.groupby(['clothing_fit_bucket']).size().reset_index(name='n').to_dict(orient='records')
    seg = d.groupby(['gender','product_kind']).apply(
        lambda g: pd.Series({
            'mae_bust_cm': round(float((g['raw_bust_cm']-g['manual_bust_cm']).abs().mean()),2),
            'mae_waist_cm': round(float((g['raw_waist_cm']-g['manual_waist_cm']).abs().mean()),2),
            'mae_hips_cm': round(float((g['raw_hips_cm']-g['manual_hips_cm']).abs().mean()),2),
            'n': int(len(g))
        })
    ).reset_index()
    metrics['segment_mae'] = seg.to_dict(orient='records')
    if 'predicted_accept' in d.columns and 'manual_should_accept' in d.columns:
        dd = d.dropna(subset=['predicted_accept','manual_should_accept']).copy()
        if not dd.empty:
            fp = int(((dd['predicted_accept'] == 1) & (dd['manual_should_accept'] == 0)).sum())
            fn = int(((dd['predicted_accept'] == 0) & (dd['manual_should_accept'] == 1)).sum())
            tp = int(((dd['predicted_accept'] == 1) & (dd['manual_should_accept'] == 1)).sum())
            tn = int(((dd['predicted_accept'] == 0) & (dd['manual_should_accept'] == 0)).sum())
            metrics['capture_truth_set'] = {
                'n': int(len(dd)),
                'false_accept_rate_pct': round(fp / max(len(dd),1) * 100, 1),
                'false_reject_rate_pct': round(fn / max(len(dd),1) * 100, 1),
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            }
    metrics['notes'] = 'Truth set 3.0: core + extended parts + segment MAE + optional false accept/reject.'
    return metrics
