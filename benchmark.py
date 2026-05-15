
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from rule_engine import ProductProfileLite, UserProfile, recommend_size
from posture_analysis import evaluate_visual_compensation
from product_ingest import ProductProfile


def load_sample_products(app_dir: Path) -> List[Dict[str, Any]]:
    return json.loads((app_dir / 'data' / 'sample_products.json').read_text(encoding='utf-8'))


def run_benchmark(app_dir: Path) -> Dict[str, Any]:
    samples = load_sample_products(app_dir)
    synthetic_users = [
        {'body_type':'gruszka','build_type':'średnia','bust':90,'waist':72,'hips':100,'height':168,'weight':62},
        {'body_type':'klepsydra','build_type':'średnia','bust':92,'waist':68,'hips':94,'height':170,'weight':60},
        {'body_type':'jabłko','build_type':'pełniejsza','bust':102,'waist':90,'hips':104,'height':167,'weight':82},
        {'body_type':'prostokąt','build_type':'szczupła','bust':84,'waist':70,'hips':88,'height':172,'weight':55},
    ]
    fit_scores = []
    visual_scores = []
    caution = 0
    buy = 0
    avoid = 0
    for s in samples[:8]:
        p = ProductProfileLite(
            brand=s['brand'], name=s['name'], dress_type=s['dress_type'], fit_type=s['fit_type'], stretch_level=s['stretch_level'],
            length_type=s['length_type'], style_effect=s['style_effect'], runs_small=float(s['runs_small']), runs_large=float(s['runs_large']),
            true_to_size=float(s['true_to_size']), tight_areas=s['tight_areas'], review_count=len(s.get('review_texts',[])), review_lines=s.get('review_texts',[]),
            size_chart=s['size_chart'], product_kind='sukienka casualowa', search_group='damskie', style_branch='casual'
        )
        for u in synthetic_users:
            rec = recommend_size(UserProfile(
                height_cm=u['height'], weight_kg=u['weight'], age=30, body_type=u['body_type'], fit_preference='standard', build_type=u['build_type'],
                bust_cm=u['bust'], waist_cm=u['waist'], hips_cm=u['hips'], extra_measurements={}
            ), p)
            fit_scores.append(rec['fit_score'])
            visual_scores.append(rec['dress_match_score'])
            if rec['verdict'] == 'bierz':
                buy += 1
            elif rec['verdict'] == 'ostrożnie':
                caution += 1
            else:
                avoid += 1
    n = max(len(fit_scores), 1)
    return {
        'scenario_count': n,
        'avg_technical_fit': round(sum(fit_scores)/n, 1),
        'avg_visual_fit': round(sum(visual_scores)/n, 1),
        'buy_share_pct': round(100*buy/n, 1),
        'caution_share_pct': round(100*caution/n, 1),
        'avoid_share_pct': round(100*avoid/n, 1),
        'notes': 'To syntetyczny benchmark heurystyczny na katalogu demo i profilach testowych; nie jest to walidacja produkcyjna na danych rzeczywistych.'
    }
