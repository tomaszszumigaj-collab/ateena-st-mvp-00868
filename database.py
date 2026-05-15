from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).parent / 'data' / 'ateena_mvp_v8_6_2.db'


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            gender TEXT,
            unit_system TEXT,
            capture_method TEXT,
            height_cm REAL,
            weight_kg REAL,
            age INTEGER,
            fit_preference TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS body_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            gender TEXT,
            body_type TEXT,
            build_type TEXT,
            confidence REAL,
            measurement_source TEXT,
            bust_cm REAL,
            waist_cm REAL,
            hips_cm REAL,
            raw_ai_bust_cm REAL,
            raw_ai_waist_cm REAL,
            raw_ai_hips_cm REAL,
            calibration_info_json TEXT,
            measurement_confidence_json TEXT,
            extra_measurements_json TEXT,
            posture_summary_json TEXT,
            front_capture_json TEXT,
            profile_capture_json TEXT,
            back_capture_json TEXT,
            notes_json TEXT,
            sanity_report_json TEXT,
            weak_points_json TEXT,
            front_image_path TEXT,
            profile_image_path TEXT,
            back_image_path TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_url TEXT,
            brand TEXT,
            name TEXT,
            image_url TEXT,
            dress_type TEXT,
            fit_type TEXT,
            length_type TEXT,
            stretch_level TEXT,
            style_effect TEXT,
            tight_areas_json TEXT,
            size_chart_json TEXT,
            runs_small REAL,
            runs_large REAL,
            true_to_size REAL,
            review_count INTEGER,
            review_lines_json TEXT,
            used_fallback_chart INTEGER,
            parsing_notes_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            body_analysis_id INTEGER,
            product_id INTEGER,
            recommended_size TEXT,
            alternate_size TEXT,
            fit_score REAL,
            dress_match_score REAL,
            verdict TEXT,
            risk_flags_json TEXT,
            explanation TEXT,
            estimated_measurements_json TEXT,
            visual_fit_json TEXT,
            model_version TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (body_analysis_id) REFERENCES body_analyses(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recommendation_id INTEGER,
            purchased INTEGER,
            chosen_size TEXT,
            overall_fit_label TEXT,
            problem_areas_json TEXT,
            returned INTEGER,
            comment TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (recommendation_id) REFERENCES recommendations(id)
        );

        CREATE TABLE IF NOT EXISTS calibration_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            body_analysis_id INTEGER,
            recommendation_id INTEGER,
            gender TEXT,
            body_type TEXT,
            build_type TEXT,
            confidence REAL,
            confidence_band TEXT,
            photo_quality_bucket TEXT,
            photo_quality_score REAL,
            product_kind TEXT,
            clothing_fit TEXT,
            clothing_fit_bucket TEXT,
            sample_weight REAL,
            ai_raw_bust_cm REAL,
            ai_raw_waist_cm REAL,
            ai_raw_hips_cm REAL,
            ai_displayed_bust_cm REAL,
            ai_displayed_waist_cm REAL,
            ai_displayed_hips_cm REAL,
            manual_bust_cm REAL,
            manual_waist_cm REAL,
            manual_hips_cm REAL,
            raw_error_bust_cm REAL,
            raw_error_waist_cm REAL,
            raw_error_hips_cm REAL,
            displayed_error_bust_cm REAL,
            displayed_error_waist_cm REAL,
            displayed_error_hips_cm REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (body_analysis_id) REFERENCES body_analyses(id),
            FOREIGN KEY (recommendation_id) REFERENCES recommendations(id)
        );

        CREATE INDEX IF NOT EXISTS idx_calib_segment ON calibration_examples(
            gender, photo_quality_bucket, product_kind, clothing_fit_bucket
        );
        CREATE INDEX IF NOT EXISTS idx_calib_gender ON calibration_examples(gender);
        CREATE INDEX IF NOT EXISTS idx_calib_product ON calibration_examples(product_kind);
        CREATE INDEX IF NOT EXISTS idx_calib_quality ON calibration_examples(photo_quality_bucket);
        

        
        CREATE TABLE IF NOT EXISTS privacy_consents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            consent_analysis INTEGER,
            consent_store_images INTEGER,
            consent_training INTEGER,
            locale_code TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

CREATE TABLE IF NOT EXISTS translation_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            locale_code TEXT,
            screen_name TEXT,
            source_text TEXT,
            comment TEXT,
            reporter_email TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS ocr_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_kind TEXT,
            product_image_path TEXT,
            size_chart_image_path TEXT,
            extracted_text_json TEXT,
            parsing_notes_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT,
            metrics_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS capture_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            gender TEXT,
            product_kind TEXT,
            photo_clothing_fit TEXT,
            capture_method TEXT,
            accepted INTEGER,
            measurement_ready INTEGER,
            posture_ready INTEGER,
            front_image_path TEXT,
            profile_image_path TEXT,
            back_image_path TEXT,
            front_capture_json TEXT,
            profile_capture_json TEXT,
            back_capture_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS calibration_part_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            body_analysis_id INTEGER,
            recommendation_id INTEGER,
            gender TEXT,
            photo_quality_bucket TEXT,
            product_kind TEXT,
            clothing_fit_bucket TEXT,
            measure_key TEXT,
            sample_weight REAL,
            ai_raw_value_cm REAL,
            ai_displayed_value_cm REAL,
            manual_value_cm REAL,
            raw_error_cm REAL,
            displayed_error_cm REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (body_analysis_id) REFERENCES body_analyses(id),
            FOREIGN KEY (recommendation_id) REFERENCES recommendations(id)
        );
        CREATE INDEX IF NOT EXISTS idx_calib_part_segment ON calibration_part_examples(
            gender, photo_quality_bucket, product_kind, clothing_fit_bucket, measure_key
        );
        """
    )
    conn.commit()
    conn.close()


def save_user(email: str, gender: str, unit_system: str, capture_method: str, height_cm: float, weight_kg: float, age: Optional[int], fit_preference: Optional[str]) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO users(email, gender, unit_system, capture_method, height_cm, weight_kg, age, fit_preference) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (email, gender, unit_system, capture_method, height_cm, weight_kg, age, fit_preference),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def save_body_analysis(
    user_id: int,
    gender: str,
    body: Dict[str, Any],
    front_path: Optional[str],
    profile_path: Optional[str],
    back_path: Optional[str] = None,
    extra_measurements: Optional[Dict[str, Any]] = None,
    posture_summary: Optional[Dict[str, Any]] = None,
) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO body_analyses(
            user_id, gender, body_type, build_type, confidence, measurement_source,
            bust_cm, waist_cm, hips_cm,
            raw_ai_bust_cm, raw_ai_waist_cm, raw_ai_hips_cm,
            calibration_info_json, measurement_confidence_json, extra_measurements_json, posture_summary_json,
            front_capture_json, profile_capture_json, back_capture_json, notes_json, sanity_report_json, weak_points_json, front_image_path, profile_image_path, back_image_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            gender,
            body['body_type'],
            body['build_type'],
            float(body['confidence']),
            body.get('measurement_source'),
            float(body['suggested_bust_cm']),
            float(body['suggested_waist_cm']),
            float(body['suggested_hips_cm']),
            float(body.get('raw_bust_cm', body['suggested_bust_cm'])),
            float(body.get('raw_waist_cm', body['suggested_waist_cm'])),
            float(body.get('raw_hips_cm', body['suggested_hips_cm'])),
            json.dumps(body.get('calibration_info', {}), ensure_ascii=False),
            json.dumps(body.get('measurement_confidence', {}), ensure_ascii=False),
            json.dumps(extra_measurements or {}, ensure_ascii=False),
            json.dumps(posture_summary or {}, ensure_ascii=False),
            json.dumps(body.get('front_capture', {}), ensure_ascii=False),
            json.dumps(body.get('profile_capture', {}), ensure_ascii=False),
            json.dumps(body.get('back_capture', {}), ensure_ascii=False),
            json.dumps(body.get('notes', []), ensure_ascii=False),
            json.dumps(body.get('sanity_report', {}), ensure_ascii=False),
            json.dumps(body.get('weak_points', []), ensure_ascii=False),
            front_path,
            profile_path,
            back_path,
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def save_product(product: Dict[str, Any]) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO products(source_url, brand, name, image_url, dress_type, fit_type, length_type, stretch_level, style_effect,
                             tight_areas_json, size_chart_json, runs_small, runs_large, true_to_size, review_count, review_lines_json,
                             used_fallback_chart, parsing_notes_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            product.get('source_url'),
            product.get('brand'),
            product.get('name'),
            product.get('image_url'),
            product.get('dress_type'),
            product.get('fit_type'),
            product.get('length_type'),
            product.get('stretch_level'),
            product.get('style_effect'),
            json.dumps(product.get('tight_areas', []), ensure_ascii=False),
            json.dumps(product.get('size_chart', {}), ensure_ascii=False),
            product.get('runs_small'),
            product.get('runs_large'),
            product.get('true_to_size'),
            product.get('review_count'),
            json.dumps(product.get('review_lines', []), ensure_ascii=False),
            int(bool(product.get('used_fallback_chart'))),
            json.dumps(product.get('parsing_notes', []), ensure_ascii=False),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def save_recommendation(user_id: int, body_analysis_id: int, product_id: int, rec: Dict[str, Any]) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO recommendations(user_id, body_analysis_id, product_id, recommended_size, alternate_size, fit_score,
                                    dress_match_score, verdict, risk_flags_json, explanation, estimated_measurements_json, visual_fit_json, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            body_analysis_id,
            product_id,
            rec.get('recommended_size'),
            rec.get('alternate_size'),
            rec.get('fit_score'),
            rec.get('dress_match_score'),
            rec.get('verdict'),
            json.dumps(rec.get('risk_flags', []), ensure_ascii=False),
            rec.get('explanation'),
            json.dumps(rec.get('estimated_measurements', {}), ensure_ascii=False),
            json.dumps(rec.get('visual_fit', {}), ensure_ascii=False),
            rec.get('model_version', 'mvp_v7_8'),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def save_feedback(recommendation_id: int, purchased: bool, chosen_size: Optional[str], overall_fit_label: str, problem_areas: List[str], returned: bool, comment: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO feedback(recommendation_id, purchased, chosen_size, overall_fit_label, problem_areas_json, returned, comment)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (recommendation_id, int(purchased), chosen_size, overall_fit_label, json.dumps(problem_areas, ensure_ascii=False), int(returned), comment),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def save_calibration_sample(
    *,
    user_id: int,
    body_analysis_id: int,
    recommendation_id: Optional[int],
    gender: str,
    body_type: str,
    build_type: str,
    confidence: float,
    confidence_band: str,
    photo_quality_bucket: str,
    photo_quality_score: float,
    product_kind: str,
    clothing_fit: str,
    clothing_fit_bucket: str,
    sample_weight: float,
    ai_raw_bust_cm: float,
    ai_raw_waist_cm: float,
    ai_raw_hips_cm: float,
    ai_displayed_bust_cm: float,
    ai_displayed_waist_cm: float,
    ai_displayed_hips_cm: float,
    manual_bust_cm: float,
    manual_waist_cm: float,
    manual_hips_cm: float,
) -> int:
    raw_error_bust = round(float(manual_bust_cm) - float(ai_raw_bust_cm), 2)
    raw_error_waist = round(float(manual_waist_cm) - float(ai_raw_waist_cm), 2)
    raw_error_hips = round(float(manual_hips_cm) - float(ai_raw_hips_cm), 2)
    displayed_error_bust = round(float(manual_bust_cm) - float(ai_displayed_bust_cm), 2)
    displayed_error_waist = round(float(manual_waist_cm) - float(ai_displayed_waist_cm), 2)
    displayed_error_hips = round(float(manual_hips_cm) - float(ai_displayed_hips_cm), 2)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO calibration_examples(
            user_id, body_analysis_id, recommendation_id, gender, body_type, build_type,
            confidence, confidence_band, photo_quality_bucket, photo_quality_score,
            product_kind, clothing_fit, clothing_fit_bucket, sample_weight,
            ai_raw_bust_cm, ai_raw_waist_cm, ai_raw_hips_cm,
            ai_displayed_bust_cm, ai_displayed_waist_cm, ai_displayed_hips_cm,
            manual_bust_cm, manual_waist_cm, manual_hips_cm,
            raw_error_bust_cm, raw_error_waist_cm, raw_error_hips_cm,
            displayed_error_bust_cm, displayed_error_waist_cm, displayed_error_hips_cm
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id, body_analysis_id, recommendation_id, gender, body_type, build_type,
            float(confidence), confidence_band, photo_quality_bucket, float(photo_quality_score),
            product_kind, clothing_fit, clothing_fit_bucket, float(sample_weight),
            float(ai_raw_bust_cm), float(ai_raw_waist_cm), float(ai_raw_hips_cm),
            float(ai_displayed_bust_cm), float(ai_displayed_waist_cm), float(ai_displayed_hips_cm),
            float(manual_bust_cm), float(manual_waist_cm), float(manual_hips_cm),
            raw_error_bust, raw_error_waist, raw_error_hips,
            displayed_error_bust, displayed_error_waist, displayed_error_hips,
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


save_calibration_example = save_calibration_sample


def _aggregate_offsets(where_clause: str = '', params: tuple = ()) -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        f"""
        SELECT COUNT(*) AS sample_count,
               COALESCE(SUM(sample_weight), 0) AS total_weight,
               CASE WHEN SUM(sample_weight) > 0 THEN SUM(raw_error_bust_cm * sample_weight) / SUM(sample_weight) ELSE 0 END AS bust_offset,
               CASE WHEN SUM(sample_weight) > 0 THEN SUM(raw_error_waist_cm * sample_weight) / SUM(sample_weight) ELSE 0 END AS waist_offset,
               CASE WHEN SUM(sample_weight) > 0 THEN SUM(raw_error_hips_cm * sample_weight) / SUM(sample_weight) ELSE 0 END AS hips_offset,
               AVG(ABS(displayed_error_bust_cm)) AS displayed_abs_bust,
               AVG(ABS(displayed_error_waist_cm)) AS displayed_abs_waist,
               AVG(ABS(displayed_error_hips_cm)) AS displayed_abs_hips
        FROM calibration_examples
        {where_clause}
        """,
        params,
    ).fetchone()
    conn.close()
    return {
        'sample_count': int(row['sample_count'] or 0),
        'total_weight': round(float(row['total_weight'] or 0.0), 2),
        'offsets': {
            'bust': round(float(row['bust_offset'] or 0.0), 2),
            'waist': round(float(row['waist_offset'] or 0.0), 2),
            'hips': round(float(row['hips_offset'] or 0.0), 2),
        },
        'displayed_abs_error': {
            'bust': round(float(row['displayed_abs_bust'] or 0.0), 2),
            'waist': round(float(row['displayed_abs_waist'] or 0.0), 2),
            'hips': round(float(row['displayed_abs_hips'] or 0.0), 2),
        },
    }


def get_calibration_profile(*, gender: str, photo_quality_bucket: str, product_kind: str, clothing_fit_bucket: str) -> Dict[str, Any]:
    candidates = [
        ('gender+quality+product+fit', 'WHERE gender = ? AND photo_quality_bucket = ? AND product_kind = ? AND clothing_fit_bucket = ?', (gender, photo_quality_bucket, product_kind, clothing_fit_bucket), 4, 2.2),
        ('gender+quality+product', 'WHERE gender = ? AND photo_quality_bucket = ? AND product_kind = ?', (gender, photo_quality_bucket, product_kind), 5, 2.8),
        ('gender+quality+fit', 'WHERE gender = ? AND photo_quality_bucket = ? AND clothing_fit_bucket = ?', (gender, photo_quality_bucket, clothing_fit_bucket), 5, 2.8),
        ('gender+product+fit', 'WHERE gender = ? AND product_kind = ? AND clothing_fit_bucket = ?', (gender, product_kind, clothing_fit_bucket), 5, 2.8),
        ('gender+quality', 'WHERE gender = ? AND photo_quality_bucket = ?', (gender, photo_quality_bucket), 6, 3.2),
        ('gender+product', 'WHERE gender = ? AND product_kind = ?', (gender, product_kind), 6, 3.2),
        ('gender+fit', 'WHERE gender = ? AND clothing_fit_bucket = ?', (gender, clothing_fit_bucket), 6, 3.2),
        ('gender_only', 'WHERE gender = ?', (gender,), 8, 4.5),
        ('global', '', tuple(), 12, 6.0),
    ]

    for scope, where_clause, params, min_count, min_weight in candidates:
        agg = _aggregate_offsets(where_clause, params)
        if agg['sample_count'] >= min_count and agg['total_weight'] >= min_weight:
            strength = min(1.0, agg['total_weight'] / max(min_weight * 1.8, 5.0))
            offsets = {k: round(v * strength, 2) for k, v in agg['offsets'].items()}
            return {
                'found': True,
                'scope': scope,
                'used_scopes': [scope],
                'sample_count': agg['sample_count'],
                'total_weight': agg['total_weight'],
                'strength': round(strength, 2),
                'offsets': offsets,
                'raw_average_offsets': agg['offsets'],
                'displayed_abs_error': agg['displayed_abs_error'],
                'photo_quality_bucket': photo_quality_bucket,
                'product_kind': product_kind,
                'clothing_fit_bucket': clothing_fit_bucket,
            }

    return {
        'found': False,
        'scope': 'none',
        'used_scopes': [],
        'sample_count': 0,
        'total_weight': 0.0,
        'strength': 0.0,
        'offsets': {'bust': 0.0, 'waist': 0.0, 'hips': 0.0},
        'raw_average_offsets': {'bust': 0.0, 'waist': 0.0, 'hips': 0.0},
        'displayed_abs_error': {'bust': 0.0, 'waist': 0.0, 'hips': 0.0},
        'photo_quality_bucket': photo_quality_bucket,
        'product_kind': product_kind,
        'clothing_fit_bucket': clothing_fit_bucket,
    }


def recent_recommendations(limit: int = 30) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT r.id, u.email, u.gender, p.brand, p.name, r.recommended_size, r.fit_score, r.verdict, r.model_version, r.created_at
        FROM recommendations r
        JOIN users u ON u.id = r.user_id
        JOIN products p ON p.id = r.product_id
        ORDER BY r.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def recommendations_by_email(email: str) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT r.id, p.brand, p.name, r.recommended_size, r.verdict, r.created_at
        FROM recommendations r
        JOIN users u ON u.id = r.user_id
        JOIN products p ON p.id = r.product_id
        WHERE lower(u.email) = lower(?)
        ORDER BY r.id DESC
        """,
        (email,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def recent_calibration_examples(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT id, gender, photo_quality_bucket, product_kind, clothing_fit_bucket, clothing_fit, sample_weight,
               ai_raw_bust_cm, ai_raw_waist_cm, ai_raw_hips_cm,
               ai_displayed_bust_cm, ai_displayed_waist_cm, ai_displayed_hips_cm,
               manual_bust_cm, manual_waist_cm, manual_hips_cm,
               raw_error_bust_cm, raw_error_waist_cm, raw_error_hips_cm,
               displayed_error_bust_cm, displayed_error_waist_cm, displayed_error_hips_cm,
               created_at
        FROM calibration_examples
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _grouped_summary(column: str) -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        f"""
        SELECT {column} AS bucket,
               COUNT(*) AS sample_count,
               COALESCE(SUM(sample_weight), 0) AS total_weight,
               CASE WHEN SUM(sample_weight) > 0 THEN SUM(raw_error_bust_cm * sample_weight) / SUM(sample_weight) ELSE 0 END AS bust_offset,
               CASE WHEN SUM(sample_weight) > 0 THEN SUM(raw_error_waist_cm * sample_weight) / SUM(sample_weight) ELSE 0 END AS waist_offset,
               CASE WHEN SUM(sample_weight) > 0 THEN SUM(raw_error_hips_cm * sample_weight) / SUM(sample_weight) ELSE 0 END AS hips_offset
        FROM calibration_examples
        GROUP BY {column}
        ORDER BY total_weight DESC, sample_count DESC
        """
    ).fetchall()
    conn.close()
    out = {}
    for row in rows:
        key = row['bucket'] if row['bucket'] is not None else 'unknown'
        out[key] = {
            'sample_count': int(row['sample_count'] or 0),
            'total_weight': round(float(row['total_weight'] or 0.0), 2),
            'offsets': {
                'bust': round(float(row['bust_offset'] or 0.0), 2),
                'waist': round(float(row['waist_offset'] or 0.0), 2),
                'hips': round(float(row['hips_offset'] or 0.0), 2),
            },
        }
    return out


def calibration_summary() -> Dict[str, Any]:
    overall = _aggregate_offsets()
    return {
        'overall': overall,
        'by_gender': _grouped_summary('gender'),
        'by_photo_quality': _grouped_summary('photo_quality_bucket'),
        'by_product_kind': _grouped_summary('product_kind'),
        'by_clothing_fit_bucket': _grouped_summary('clothing_fit_bucket'),
    }


def list_products_for_visual_search(limit: int = 300) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT id, source_url, brand, name, image_url, dress_type, fit_type, length_type, stretch_level, style_effect,
               tight_areas_json, size_chart_json, runs_small, runs_large, true_to_size, review_count, review_lines_json,
               used_fallback_chart, parsing_notes_json
        FROM products
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in ['tight_areas_json', 'size_chart_json', 'review_lines_json', 'parsing_notes_json']:
            if key in item and item[key]:
                try:
                    parsed = json.loads(item[key])
                except Exception:
                    parsed = [] if key.endswith('_json') else {}
                item[key.replace('_json','')] = parsed
            else:
                item[key.replace('_json','')] = [] if key in ['tight_areas_json','review_lines_json','parsing_notes_json'] else {}
        out.append(item)
    return out



def save_privacy_consent(
    user_id: int,
    consent_analysis: bool,
    consent_store_images: bool,
    consent_training: bool,
    locale_code: str | None = None,
) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO privacy_consents(user_id, consent_analysis, consent_store_images, consent_training, locale_code) VALUES (?, ?, ?, ?, ?)',
        (user_id, int(consent_analysis), int(consent_store_images), int(consent_training), locale_code),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def consent_overview() -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    try:
        total = cur.execute('SELECT COUNT(*) c FROM privacy_consents').fetchone()['c']
        analysis_yes = cur.execute('SELECT COUNT(*) c FROM privacy_consents WHERE consent_analysis = 1').fetchone()['c']
        images_yes = cur.execute('SELECT COUNT(*) c FROM privacy_consents WHERE consent_store_images = 1').fetchone()['c']
        training_yes = cur.execute('SELECT COUNT(*) c FROM privacy_consents WHERE consent_training = 1').fetchone()['c']
    except Exception:
        total = analysis_yes = images_yes = training_yes = 0
    conn.close()
    return {
        'total': int(total),
        'analysis_yes': int(analysis_yes),
        'images_yes': int(images_yes),
        'training_yes': int(training_yes),
    }


def save_translation_report(locale_code: str, screen_name: str, source_text: str, comment: str, reporter_email: str | None = None) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO translation_reports(locale_code, screen_name, source_text, comment, reporter_email) VALUES (?, ?, ?, ?, ?)',
        (locale_code, screen_name, source_text, comment, reporter_email),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def recent_translation_reports(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute('SELECT * FROM translation_reports ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_ocr_session(product_kind: str, product_image_path: str | None, size_chart_image_path: str | None, extracted_text: Dict[str, Any], parsing_notes: List[str]) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO ocr_sessions(product_kind, product_image_path, size_chart_image_path, extracted_text_json, parsing_notes_json) VALUES (?, ?, ?, ?, ?)',
        (product_kind, product_image_path, size_chart_image_path, json.dumps(extracted_text, ensure_ascii=False), json.dumps(parsing_notes, ensure_ascii=False)),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def recent_ocr_sessions(limit: int = 30) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute('SELECT * FROM ocr_sessions ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_benchmark_run(run_name: str, metrics: Dict[str, Any]) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('INSERT INTO benchmark_runs(run_name, metrics_json) VALUES (?, ?)', (run_name, json.dumps(metrics, ensure_ascii=False)))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def recent_benchmark_runs(limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute('SELECT * FROM benchmark_runs ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d['metrics_json'] = json.loads(d['metrics_json']) if d.get('metrics_json') else {}
        except Exception:
            pass
        out.append(d)
    return out


def save_capture_session(
    *,
    user_email: str,
    gender: str,
    product_kind: str,
    photo_clothing_fit: str,
    capture_method: str,
    accepted: bool,
    measurement_ready: bool,
    posture_ready: bool,
    front_image_path: str | None,
    profile_image_path: str | None,
    back_image_path: str | None,
    front_capture: Dict[str, Any],
    profile_capture: Dict[str, Any],
    back_capture: Dict[str, Any] | None = None,
) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO capture_sessions(
            user_email, gender, product_kind, photo_clothing_fit, capture_method,
            accepted, measurement_ready, posture_ready,
            front_image_path, profile_image_path, back_image_path,
            front_capture_json, profile_capture_json, back_capture_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_email, gender, product_kind, photo_clothing_fit, capture_method,
            int(bool(accepted)), int(bool(measurement_ready)), int(bool(posture_ready)),
            front_image_path, profile_image_path, back_image_path,
            json.dumps(front_capture or {}, ensure_ascii=False),
            json.dumps(profile_capture or {}, ensure_ascii=False),
            json.dumps(back_capture or {}, ensure_ascii=False),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def recent_capture_sessions(limit: int = 30) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT id, user_email, gender, product_kind, photo_clothing_fit, capture_method,
               accepted, measurement_ready, posture_ready,
               created_at
        FROM capture_sessions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def save_calibration_part_sample(
    *,
    user_id: int,
    body_analysis_id: int,
    recommendation_id: int | None,
    gender: str,
    photo_quality_bucket: str,
    product_kind: str,
    clothing_fit_bucket: str,
    measure_key: str,
    sample_weight: float,
    ai_raw_value_cm: float,
    ai_displayed_value_cm: float,
    manual_value_cm: float,
) -> int:
    raw_error = round(float(manual_value_cm) - float(ai_raw_value_cm), 2)
    displayed_error = round(float(manual_value_cm) - float(ai_displayed_value_cm), 2)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO calibration_part_examples(
            user_id, body_analysis_id, recommendation_id, gender,
            photo_quality_bucket, product_kind, clothing_fit_bucket, measure_key,
            sample_weight, ai_raw_value_cm, ai_displayed_value_cm, manual_value_cm,
            raw_error_cm, displayed_error_cm
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id, body_analysis_id, recommendation_id, gender,
            photo_quality_bucket, product_kind, clothing_fit_bucket, measure_key,
            float(sample_weight), float(ai_raw_value_cm), float(ai_displayed_value_cm), float(manual_value_cm),
            raw_error, displayed_error,
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def _aggregate_part_offsets(where_clause: str = '', params: tuple = ()) -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        f"""
        SELECT measure_key,
               COUNT(*) AS sample_count,
               COALESCE(SUM(sample_weight), 0) AS total_weight,
               CASE WHEN SUM(sample_weight) > 0 THEN SUM(raw_error_cm * sample_weight) / SUM(sample_weight) ELSE 0 END AS offset,
               AVG(ABS(displayed_error_cm)) AS displayed_abs_error
        FROM calibration_part_examples
        {where_clause}
        GROUP BY measure_key
        """,
        params,
    ).fetchall()
    conn.close()
    out: Dict[str, Any] = {}
    for row in rows:
        out[row['measure_key']] = {
            'sample_count': int(row['sample_count'] or 0),
            'total_weight': round(float(row['total_weight'] or 0.0), 2),
            'offset': round(float(row['offset'] or 0.0), 2),
            'displayed_abs_error': round(float(row['displayed_abs_error'] or 0.0), 2),
        }
    return out


def get_calibration_part_offsets(*, gender: str, photo_quality_bucket: str, product_kind: str, clothing_fit_bucket: str) -> Dict[str, Any]:
    candidates = [
        ('gender+quality+product+fit', 'WHERE gender = ? AND photo_quality_bucket = ? AND product_kind = ? AND clothing_fit_bucket = ?', (gender, photo_quality_bucket, product_kind, clothing_fit_bucket), 3, 1.4),
        ('gender+quality+product', 'WHERE gender = ? AND photo_quality_bucket = ? AND product_kind = ?', (gender, photo_quality_bucket, product_kind), 4, 1.8),
        ('gender+quality', 'WHERE gender = ? AND photo_quality_bucket = ?', (gender, photo_quality_bucket), 5, 2.2),
        ('gender+product', 'WHERE gender = ? AND product_kind = ?', (gender, product_kind), 5, 2.2),
        ('gender_only', 'WHERE gender = ?', (gender,), 6, 3.0),
        ('global', '', tuple(), 8, 4.0),
    ]
    for scope, where_clause, params, min_count, min_weight in candidates:
        agg = _aggregate_part_offsets(where_clause, params)
        if not agg:
            continue
        qualified = {k:v for k,v in agg.items() if v['sample_count'] >= min_count and v['total_weight'] >= min_weight}
        if qualified:
            offsets = {k: round(v['offset'] * min(1.0, v['total_weight']/max(min_weight*1.5, 3.0)), 2) for k,v in qualified.items()}
            return {
                'found': True,
                'scope': scope,
                'offsets': offsets,
                'stats': qualified,
            }
    return {'found': False, 'scope': 'none', 'offsets': {}, 'stats': {}}


def calibration_part_summary() -> Dict[str, Any]:
    return {
        'by_measure': _aggregate_part_offsets(),
    }


def _ensure_annotation_reviews_table() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS annotation_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reviewer_email TEXT,
            source_image_path TEXT,
            annotation_csv_path TEXT,
            summary_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )
    conn.commit()
    conn.close()

_ensure_annotation_reviews_table()

def save_annotation_review(reviewer_email: str | None, source_image_path: str | None, annotation_csv_path: str | None, summary: Dict[str, Any]) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO annotation_reviews(reviewer_email, source_image_path, annotation_csv_path, summary_json) VALUES (?, ?, ?, ?)',
        (reviewer_email, source_image_path, annotation_csv_path, json.dumps(summary, ensure_ascii=False)),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id

def recent_annotation_reviews(limit: int = 30) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        'SELECT id, reviewer_email, source_image_path, annotation_csv_path, summary_json, created_at FROM annotation_reviews ORDER BY id DESC LIMIT ?',
        (limit,),
    ).fetchall()
    conn.close()
    out = []
    for row in rows:
        d = dict(row)
        try:
            d['summary'] = json.loads(d.pop('summary_json') or '{}')
        except Exception:
            d['summary'] = {}
            d.pop('summary_json', None)
        out.append(d)
    return out
