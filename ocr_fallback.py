
from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
try:
    import pytesseract
except Exception:
    pytesseract = None
from PIL import Image

from product_ingest import ProductProfile


def _configure_windows_tesseract() -> None:
    if pytesseract is None:
        return
    try:
        current = getattr(pytesseract.pytesseract, 'tesseract_cmd', '') or ''
        if current and Path(current).exists():
            return
    except Exception:
        pass
    candidates = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                pytesseract.pytesseract.tesseract_cmd = str(candidate)
                return
            except Exception:
                return


def ocr_engine_status() -> tuple[bool, str]:
    if pytesseract is None:
        return False, "Brakuje pakietu Python `pytesseract`."
    _configure_windows_tesseract()
    try:
        cmd = getattr(pytesseract.pytesseract, 'tesseract_cmd', '') or 'tesseract'
        if 'tesseract.exe' in cmd.lower() and Path(cmd).exists():
            return True, f"Tesseract skonfigurowany: {cmd}"
        return True, "Pakiet pytesseract jest dostępny, ale systemowy Tesseract może wymagać instalacji."
    except Exception:
        return True, "Pakiet pytesseract jest dostępny."


def _bytes_to_temp_image(image_bytes: bytes, suffix: str = '.png') -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with open(path, 'wb') as f:
        f.write(image_bytes)
    return path


def _ocr_text(image_bytes: Optional[bytes]) -> str:
    if not image_bytes:
        return ''
    if pytesseract is None:
        return ''
    _configure_windows_tesseract()
    path = _bytes_to_temp_image(image_bytes)
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img, lang='eng')
        return re.sub(r'\s+', ' ', text or '').strip()
    except Exception:
        return ''
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


def _infer_kind_profile(product_kind: str) -> tuple[str, str, str, str, str, List[str]]:
    kind = (product_kind or '').lower()
    if 'garnitur' in kind or 'marynarka' in kind or 'żakiet' in kind:
        return 'structured_top', 'regular', 'low', 'regular', 'structured_upper_body', ['shoulders', 'chest', 'arm']
    if 'spodnie' in kind or 'leggins' in kind:
        return 'bottom', 'regular', 'medium', 'regular', 'lower_body_focus', ['waist', 'hips', 'thigh']
    if 'gorset' in kind:
        return 'corset', 'slim', 'low', 'regular', 'waist_emphasis', ['bust', 'waist']
    if 'spódniczka' in kind:
        return 'skirt', 'regular', 'low', 'regular', 'hips_and_waist', ['waist', 'hips']
    if 'bluza' in kind or 'hoodie' in kind:
        return 'top_loose', 'loose', 'medium', 'regular', 'casual_upper_body', ['chest', 'arm']
    if 't-shirt' in kind:
        return 'top_basic', 'regular', 'medium', 'regular', 'casual_upper_body', ['chest']
    if 'sukienka' in kind:
        if 'elegancka' in kind:
            return 'dress_elegant', 'slim', 'low', 'midi', 'waist_emphasis', ['bust', 'waist', 'hips']
        return 'dress_casual', 'regular', 'medium', 'midi', 'balanced', ['bust', 'waist', 'hips']
    return 'generic_apparel', 'regular', 'medium', 'regular', 'balanced', ['waist', 'hips']


def _normalize_chart_from_ocr(ocr_text: str, fallback_chart: Dict[str, Dict]) -> tuple[Dict[str, Dict], bool, List[str]]:
    notes = []
    if not ocr_text:
        return fallback_chart, True, ['OCR nie odczytał czytelnego tekstu tabeli — użyto fallbacku kategorii.']
    sizes = ['XXS','XS','S','M','L','XL','XXL','XS/S','S/M','M/L','L/XL']
    found = {s: [] for s in sizes}
    for size in sizes:
        for match in re.finditer(rf'(?<![A-Z]){re.escape(size)}(?![A-Z])', ocr_text):
            snippet = ocr_text[match.end():match.end()+60]
            nums = re.findall(r'(\d{2,3})', snippet)
            if len(nums) >= 2:
                found[size].extend([int(nums[0]), int(nums[1])])
    chart = {}
    for size, nums in found.items():
        if len(nums) >= 2:
            lo, hi = min(nums[0], nums[1]), max(nums[0], nums[1])
            row = fallback_chart.get(size, {}) if size in fallback_chart else {}
            if 'bust' in row or 'chest' in row:
                if 'bust' in row:
                    chart[size] = {'bust': [lo, hi]}
                else:
                    chart[size] = {'chest': [lo, hi]}
            else:
                chart[size] = {'waist': [lo, hi]}
    if not chart:
        return fallback_chart, True, ['OCR odczytał tekst, ale nie udało się wiarygodnie sparsować tabeli — użyto fallbacku kategorii.']
    notes.append('Tabela rozmiarów została częściowo odczytana z OCR. Sprawdź wynik przed użyciem produkcyjnym.')
    # merge with fallback to fill gaps
    merged = {}
    for size, row in fallback_chart.items():
        merged[size] = dict(row)
        if size in chart:
            merged[size].update(chart[size])
    return merged, False, notes


def ingest_product_from_screenshots(product_image_bytes: Optional[bytes], size_chart_bytes: Optional[bytes], product_kind: str) -> ProductProfile:
    fallback = {}
    # app.py will adapt with category chart anyway, but we try to return something coherent
    kind_profile = _infer_kind_profile(product_kind)
    dress_type, fit_type, stretch_level, length_type, style_effect, tight_areas = kind_profile
    product_text = _ocr_text(product_image_bytes)
    chart_text = _ocr_text(size_chart_bytes)
    name = product_kind.title() if product_kind else 'Produkt ze zrzutu ekranu'
    brand = 'OCR fallback'
    notes = ['Źródło produktu: screenshot / OCR fallback.']
    if product_text:
        notes.append('OCR produktu odczytał część tekstu z obrazu.')
        m = re.search(r'([A-Za-z][A-Za-z0-9\- ]{3,40})', product_text)
        if m:
            name = m.group(1).strip()[:60]
    size_chart, used_fallback, more_notes = _normalize_chart_from_ocr(chart_text, fallback)
    notes.extend(more_notes)
    return ProductProfile(
        brand=brand,
        name=name,
        source_url='screenshot://local-upload',
        image_url=None,
        dress_type=dress_type,
        fit_type=fit_type,
        stretch_level=stretch_level,
        length_type=length_type,
        style_effect=style_effect,
        tight_areas=tight_areas,
        runs_small=0.20,
        runs_large=0.05,
        true_to_size=0.45,
        review_count=0,
        review_lines=['Brak opinii klientów — użyto fallbacku OCR / screenshot.'],
        size_chart=size_chart,
        used_fallback_chart=used_fallback,
        parsing_notes=notes,
    )



def ingest_product_from_texts(product_text: str | None, chart_text: str | None, product_kind: str) -> ProductProfile:
    fallback = {}
    dress_type, fit_type, stretch_level, length_type, style_effect, tight_areas = _infer_kind_profile(product_kind)
    notes = ['Źródło produktu: ręcznie wklejony tekst OCR / fallback.']
    product_text = re.sub(r'\s+', ' ', product_text or '').strip()
    chart_text = re.sub(r'\s+', ' ', chart_text or '').strip()
    name = product_kind.title() if product_kind else 'Produkt z OCR tekstowego'
    if product_text:
        m = re.search(r'([A-Za-zÀ-ž][A-Za-zÀ-ž0-9\- ]{3,60})', product_text)
        if m:
            name = m.group(1).strip()[:60]
        notes.append('Nazwa i część profilu produktu pochodzą z ręcznie wklejonego tekstu.')
    size_chart, used_fallback, more_notes = _normalize_chart_from_ocr(chart_text, fallback)
    notes.extend(more_notes)
    return ProductProfile(
        brand='OCR text fallback',
        name=name,
        source_url='ocrtext://manual-paste',
        image_url=None,
        dress_type=dress_type,
        fit_type=fit_type,
        stretch_level=stretch_level,
        length_type=length_type,
        style_effect=style_effect,
        tight_areas=tight_areas,
        runs_small=0.15,
        runs_large=0.05,
        true_to_size=0.50,
        review_count=0,
        review_lines=['Brak opinii klientów — użyto ręcznie wklejonego tekstu / fallbacku OCR.'],
        size_chart=size_chart,
        used_fallback_chart=used_fallback,
        parsing_notes=notes,
    )


def extract_review_lines_from_sources(review_image_bytes: Optional[bytes], review_text: str | None) -> List[str]:
    raw = ''
    if review_text and str(review_text).strip():
        raw += ' ' + str(review_text).strip()
    if review_image_bytes:
        raw += ' ' + _ocr_text(review_image_bytes)
    raw = re.sub(r'\s+', ' ', raw or '').strip()
    if not raw:
        return []
    chunks = re.split(r'(?:\s*[•\-–]\s*|(?<=[\.!?])\s+)', raw)
    out = []
    seen = set()
    for chunk in chunks:
        line = re.sub(r'\s+', ' ', chunk).strip(' -–•')
        if len(line) < 12:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line[:180])
    return out[:8]


def score_ocr_quality(product_text: str | None, chart_text: str | None, review_lines: List[str] | None = None, size_chart: Optional[Dict[str, Dict]] = None) -> Dict[str, object]:
    product_text = re.sub(r'\s+', ' ', product_text or '').strip()
    chart_text = re.sub(r'\s+', ' ', chart_text or '').strip()
    review_lines = review_lines or []
    size_chart = size_chart or {}
    flags: List[str] = []

    product_score = 15
    if len(product_text) >= 20:
        product_score = 45
    elif len(product_text) >= 8:
        product_score = 30
        flags.append('krótki tekst produktu')
    else:
        flags.append('brak pewnego tekstu produktu')

    chart_rows = sum(1 for _, row in size_chart.items() if row)
    chart_score = min(35, chart_rows * 8)
    if chart_rows == 0:
        flags.append('brak pewnej tabeli rozmiarów')
    elif chart_rows < 2:
        flags.append('tabela rozmiarów jest bardzo uboga')

    review_score = min(20, len(review_lines) * 4)
    if not review_lines:
        flags.append('brak opinii / review lines')
    elif len(review_lines) < 2:
        flags.append('słaby sygnał z opinii')

    total = max(0, min(100, product_score + chart_score + review_score))
    band = 'high' if total >= 75 else 'medium' if total >= 50 else 'low'
    return {
        'total_score': int(total),
        'band': band,
        'product_score': int(product_score),
        'chart_score': int(chart_score),
        'review_score': int(review_score),
        'flags': flags,
        'chart_rows': chart_rows,
        'review_lines': len(review_lines),
    }


def chart_to_rows(size_chart: Dict[str, Dict]) -> List[Dict[str, object]]:
    rows = []
    for size, vals in (size_chart or {}).items():
        rows.append({
            'size': size,
            'bust_min': (vals.get('bust') or [None, None])[0],
            'bust_max': (vals.get('bust') or [None, None])[1],
            'chest_min': (vals.get('chest') or [None, None])[0],
            'chest_max': (vals.get('chest') or [None, None])[1],
            'waist_min': (vals.get('waist') or [None, None])[0],
            'waist_max': (vals.get('waist') or [None, None])[1],
            'hips_min': (vals.get('hips') or [None, None])[0],
            'hips_max': (vals.get('hips') or [None, None])[1],
            'thigh_min': (vals.get('thigh') or [None, None])[0],
            'thigh_max': (vals.get('thigh') or [None, None])[1],
            'arm_min': (vals.get('arm') or [None, None])[0],
            'arm_max': (vals.get('arm') or [None, None])[1],
        })
    return rows


def rows_to_chart(rows: List[Dict[str, object]]) -> Dict[str, Dict]:
    chart: Dict[str, Dict] = {}
    for row in rows or []:
        size = str(row.get('size') or '').strip()
        if not size:
            continue
        out: Dict[str, List[int]] = {}
        for key in ['bust','chest','waist','hips','thigh','arm']:
            lo = row.get(f'{key}_min')
            hi = row.get(f'{key}_max')
            if lo in (None, '') or hi in (None, ''):
                continue
            try:
                lo_i, hi_i = int(float(lo)), int(float(hi))
            except Exception:
                continue
            if hi_i < lo_i:
                lo_i, hi_i = hi_i, lo_i
            out[key] = [lo_i, hi_i]
        if out:
            chart[size] = out
    return chart
