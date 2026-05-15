
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List

from database import get_conn


def qa_overview() -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    out: Dict[str, Any] = {}
    out['users'] = cur.execute('SELECT COUNT(*) c FROM users').fetchone()['c']
    out['body_analyses'] = cur.execute('SELECT COUNT(*) c FROM body_analyses').fetchone()['c']
    out['products'] = cur.execute('SELECT COUNT(*) c FROM products').fetchone()['c']
    out['recommendations'] = cur.execute('SELECT COUNT(*) c FROM recommendations').fetchone()['c']
    out['feedback'] = cur.execute('SELECT COUNT(*) c FROM feedback').fetchone()['c']
    out['calibration_examples'] = cur.execute('SELECT COUNT(*) c FROM calibration_examples').fetchone()['c']
    try:
        out['translation_reports'] = cur.execute('SELECT COUNT(*) c FROM translation_reports').fetchone()['c']
    except Exception:
        out['translation_reports'] = 0
    try:
        out['ocr_sessions'] = cur.execute('SELECT COUNT(*) c FROM ocr_sessions').fetchone()['c']
    except Exception:
        out['ocr_sessions'] = 0
    try:
        out['benchmark_runs'] = cur.execute('SELECT COUNT(*) c FROM benchmark_runs').fetchone()['c']
    except Exception:
        out['benchmark_runs'] = 0
    try:
        out['capture_sessions'] = cur.execute('SELECT COUNT(*) c FROM capture_sessions').fetchone()['c']
    except Exception:
        out['capture_sessions'] = 0
    try:
        out['privacy_consents'] = cur.execute('SELECT COUNT(*) c FROM privacy_consents').fetchone()['c']
    except Exception:
        out['privacy_consents'] = 0
    try:
        out['calibration_part_examples'] = cur.execute('SELECT COUNT(*) c FROM calibration_part_examples').fetchone()['c']
    except Exception:
        out['calibration_part_examples'] = 0

    rows = cur.execute('SELECT front_capture_json, profile_capture_json FROM body_analyses ORDER BY id DESC LIMIT 200').fetchall()
    quality = Counter()
    common_msgs = Counter()
    for r in rows:
        for k in ['front_capture_json', 'profile_capture_json']:
            payload = r[k]
            if not payload:
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            quality[obj.get('status_code','unknown')] += 1
            for msg in obj.get('messages', [])[:2]:
                common_msgs[msg] += 1
    out['quality_distribution'] = dict(quality)
    out['top_capture_messages'] = common_msgs.most_common(10)
    conn.close()
    return out
