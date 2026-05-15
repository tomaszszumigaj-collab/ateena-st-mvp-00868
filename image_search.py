
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse

import cv2
import numpy as np
import requests


@dataclass
class VisualSearchResult:
    product_id: str
    brand: str
    name: str
    product_url: str
    image_path: str
    category: str
    score: float
    notes: str
    source_type: str = "demo"
    payload: Dict[str, Any] = field(default_factory=dict)


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Nie udało się odczytać obrazu.')
    return img


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f'Nie udało się odczytać obrazu katalogowego: {path}')
    return img


def _avg_hash(gray: np.ndarray, hash_size: int = 16) -> np.ndarray:
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    return (small > mean).astype(np.uint8).flatten()


def _color_hist(img: np.ndarray, bins: int = 8) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def _edge_profile(gray: np.ndarray) -> np.ndarray:
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(gray, 80, 160)
    grid = 8
    h, w = edges.shape
    out = []
    for r in range(grid):
        for c in range(grid):
            cell = edges[r*h//grid:(r+1)*h//grid, c*w//grid:(c+1)*w//grid]
            out.append(float(cell.mean()) / 255.0)
    return np.array(out, dtype=np.float32)


def _feature_vector(img: np.ndarray) -> Dict[str, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {'hash': _avg_hash(gray), 'hist': _color_hist(img), 'edge': _edge_profile(gray)}


def _similarity(q: Dict[str, np.ndarray], c: Dict[str, np.ndarray]) -> float:
    hash_dist = np.mean(q['hash'] != c['hash'])
    hist_sim = cv2.compareHist(q['hist'].astype(np.float32), c['hist'].astype(np.float32), cv2.HISTCMP_CORREL)
    hist_sim = max(-1.0, min(1.0, float(hist_sim)))
    edge_dist = float(np.mean(np.abs(q['edge'] - c['edge'])))
    score = 100.0 - (hash_dist * 45.0 + (1.0 - (hist_sim + 1.0)/2.0) * 35.0 + edge_dist * 40.0)
    return round(max(0.0, min(100.0, score)), 1)


def _kind_boost(preferred_kind: str | None, category: str, payload: dict) -> float:
    if not preferred_kind:
        return 0.0
    pk = str(preferred_kind).lower()
    hay = ' '.join([str(category or ''), str(payload.get('product_kind') or ''), str(payload.get('dress_type') or ''), str(payload.get('name') or '')]).lower()
    if pk in hay:
        return 8.0
    if 'sukienka' in pk and 'dress' in hay:
        return 5.0
    if 'spodnie' in pk and ('trousers' in hay or 'bottom' in hay or 'jeans' in hay):
        return 5.0
    if 'marynarka' in pk and ('blazer' in hay or 'structured_top' in hay):
        return 5.0
    return 0.0

def _infer_category(payload: dict) -> str:
    kind = payload.get('product_kind') or payload.get('dress_type') or payload.get('name') or 'odzież'
    mapping = {
        'wrap': 'sukienka / wrap', 'a_line': 'sukienka / a-line', 'bodycon': 'sukienka / bodycon',
        'shift': 'sukienka / shift', 'fit_and_flare': 'sukienka / fit-and-flare', 'slip': 'sukienka / slip',
    }
    return mapping.get(kind, str(kind))


def _cache_remote_image(url: str, cache_dir: Path) -> Optional[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(url).path).suffix or '.jpg'
    filename = hashlib.md5(url.encode('utf-8')).hexdigest() + suffix
    dest = cache_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest
    except Exception:
        return None


def _candidate_path(image_ref: str, app_dir: Path) -> Optional[Path]:
    if not image_ref:
        return None
    if image_ref.startswith('http://') or image_ref.startswith('https://'):
        return _cache_remote_image(image_ref, app_dir / 'data' / 'visual_search_cache')
    p = Path(image_ref)
    if p.exists():
        return p
    p2 = app_dir / image_ref
    if p2.exists():
        return p2
    return None


def search_index_by_photo(image_bytes: bytes, app_dir: Path, db_products: List[Dict[str, Any]] | None = None, top_k: int = 10, preferred_kind: str | None = None) -> List[VisualSearchResult]:
    sample_products = json.loads((app_dir / 'data' / 'sample_products.json').read_text(encoding='utf-8'))
    q_feat = _feature_vector(_decode_image_bytes(image_bytes))
    results: List[VisualSearchResult] = []
    seen = set()

    for row in sample_products:
        img_path = app_dir / row['image_path']
        try:
            c_feat = _feature_vector(_read_image(img_path))
            category = _infer_category(row)
            score = _similarity(q_feat, c_feat) + _kind_boost(preferred_kind, category, row)
            key = ('demo', row.get('product_url') or row.get('name'))
            if key in seen:
                continue
            seen.add(key)
            results.append(VisualSearchResult(
                product_id=str(row['product_id']),
                brand=row['brand'],
                name=row['name'],
                product_url=row['product_url'],
                image_path=str(img_path),
                category=category,
                score=score,
                notes='Wynik z katalogu demo.',
                source_type='demo',
                payload=row,
            ))
        except Exception:
            continue

    for row in db_products or []:
        try:
            img_ref = row.get('image_url') or ''
            img_path = _candidate_path(img_ref, app_dir)
            if not img_path or not img_path.exists():
                continue
            c_feat = _feature_vector(_read_image(img_path))
            category = _infer_category(row)
            score = _similarity(q_feat, c_feat) + _kind_boost(preferred_kind, category, row)
            key = ('db', row.get('source_url') or row.get('name') or str(row.get('id')))
            if key in seen:
                continue
            seen.add(key)
            results.append(VisualSearchResult(
                product_id=str(row.get('id')),
                brand=row.get('brand') or 'Unknown brand',
                name=row.get('name') or 'Unknown product',
                product_url=row.get('source_url') or '',
                image_path=str(img_path),
                category=category,
                score=score,
                notes='Wynik z lokalnego indeksu produktów zapisanych wcześniej z linków i analiz.',
                source_type='db',
                payload=row,
            ))
        except Exception:
            continue

    results.sort(key=lambda x: (-x.score, x.name))
    return results[:top_k]
