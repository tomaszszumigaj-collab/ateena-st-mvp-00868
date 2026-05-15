from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, Optional
import io
from datetime import datetime
import zipfile

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HAS_HEIF = True
except Exception:
    HAS_HEIF = False

from body_analysis import BodyAnalysisResult, HAS_MEDIAPIPE, analyze_body, anthropometric_prior, classify_body_type, derive_build_type
from calibration import (
    calibration_overview,
    compare_ai_vs_manual,
    confidence_band,
    get_calibration_correction,
    normalize_clothing_fit_bucket,
    photo_quality_bucket,
    sample_weight as calibration_sample_weight,
)
from database import (
    calibration_summary,
    calibration_part_summary,
    get_calibration_part_offsets,
    init_db,
    list_products_for_visual_search,
    recommendations_by_email,
    recent_recommendations,
    recent_capture_sessions,
    save_body_analysis,
    save_calibration_part_sample,
    save_calibration_sample,
    save_capture_session,
    save_feedback,
    save_product,
    save_recommendation,
    save_user,
    save_translation_report,
    recent_translation_reports,
    save_ocr_session,
    recent_ocr_sessions,
    save_benchmark_run,
    recent_benchmark_runs,
    save_privacy_consent,
    consent_overview,
    save_annotation_review,
    recent_annotation_reviews,
)
from product_ingest import ProductProfile, ingest_product_from_url
from ocr_fallback import ingest_product_from_screenshots, ingest_product_from_texts, extract_review_lines_from_sources, score_ocr_quality, chart_to_rows, rows_to_chart, ocr_engine_status
from qa_panel import qa_overview
from benchmark import run_benchmark
from benchmark_real import evaluate_real_benchmark, template_csv_bytes
from annotation_review import template_annotation_csv_bytes, parse_annotation_csv, compare_annotations
from capture_pro import render_capture_pro_component
from posture_analysis import analyze_posture_from_images, evaluate_visual_compensation
from rule_engine import ProductProfileLite, UserProfile, recommend_size, get_category_profile
from image_search import search_index_by_photo
from landmark_schema import schema_rows_for_ui
from landmark_schema import LANDMARK_SCHEMA_VERSION, schema_rows_for_ui

APP_DIR = Path(__file__).parent
SAMPLE_PRODUCTS = json.loads((APP_DIR / 'data' / 'sample_products.json').read_text(encoding='utf-8'))
UPLOADS_DIR = APP_DIR / 'uploads'
UPLOADS_DIR.mkdir(exist_ok=True)

GUIDE_ASSETS = {
    'kobieta': {
        'front': APP_DIR / 'assets' / 'female_front_guide.png',
        'profile': APP_DIR / 'assets' / 'female_profile_guide.png',
        'back': APP_DIR / 'assets' / 'female_back_guide.png',
        'manual': APP_DIR / 'assets' / 'female_manual_measurements.png',
    },
    'mężczyzna': {
        'front': APP_DIR / 'assets' / 'male_front_guide.png',
        'profile': APP_DIR / 'assets' / 'male_profile_guide.png',
        'back': APP_DIR / 'assets' / 'male_back_guide.png',
        'manual': APP_DIR / 'assets' / 'male_manual_measurements.png',
    },
}

EXTRA_MEASURE_LABELS = [
    ('arm_biceps_cm', 'Ramię / biceps (cm)'),
    ('neck_cm', 'Szyja (cm)'),
    ('chest_cm', 'Klatka piersiowa (cm)'),
    ('abdomen_cm', 'Brzuch (cm)'),
    ('thigh_cm', 'Udo (cm)'),
    ('wrist_cm', 'Nadgarstek (cm)'),
    ('calf_max_cm', 'Łydka — najszersze miejsce (cm)'),
    ('calf_min_cm', 'Łydka — najwęższe miejsce (cm)'),
]

CM_PER_IN = 2.54
KG_PER_LB = 0.45359237

SUPPORTED_IMAGE_UPLOAD_TYPES = ['jpg','jpeg','png','webp','heic','heif']

PRODUCT_TREE = {
    'damskie': {
        'eleganckie': ['sukienka elegancka', 'żakiet', 'marynarka', 'spodnie eleganckie', 'gorset'],
        'casual': ['sukienka casualowa', 'spódniczka', 'spodnie casualowe', 'spodnie jeansowe', 'T-shirt', 'bluza', 'bluza z kapturem', 'legginsy'],
    },
    'męskie': {
        'eleganckie': ['garnitur', 'marynarka', 'spodnie eleganckie'],
        'casual': ['spodnie casualowe', 'spodnie jeansowe', 'T-shirt', 'bluza', 'bluza z kapturem'],
    },
    'unisex': {
        'basic': ['T-shirt', 'bluza', 'bluza z kapturem', 'legginsy'],
    },
}


GENERIC_CATEGORY_CHARTS = {
    'spódniczka': {
        'XS': {'waist': [62, 66], 'hips': [86, 90]},
        'S': {'waist': [66, 70], 'hips': [90, 94]},
        'M': {'waist': [70, 74], 'hips': [94, 98]},
        'L': {'waist': [74, 79], 'hips': [98, 103]},
        'XL': {'waist': [79, 85], 'hips': [103, 109]},
    },
    'legginsy': {
        'XS': {'waist': [60, 65], 'hips': [84, 89], 'thigh': [49, 53]},
        'S': {'waist': [65, 70], 'hips': [89, 94], 'thigh': [53, 56]},
        'M': {'waist': [70, 75], 'hips': [94, 99], 'thigh': [56, 59]},
        'L': {'waist': [75, 81], 'hips': [99, 105], 'thigh': [59, 63]},
        'XL': {'waist': [81, 88], 'hips': [105, 112], 'thigh': [63, 67]},
    },
    'żakiet': {
        'XS': {'chest': [82, 86], 'waist': [64, 68], 'arm': [26, 29]},
        'S': {'chest': [86, 90], 'waist': [68, 72], 'arm': [28, 31]},
        'M': {'chest': [90, 95], 'waist': [72, 77], 'arm': [30, 33]},
        'L': {'chest': [95, 101], 'waist': [77, 83], 'arm': [32, 35]},
        'XL': {'chest': [101, 108], 'waist': [83, 90], 'arm': [34, 37]},
    },
    'marynarka': {
        'XS': {'chest': [82, 86], 'waist': [64, 68], 'arm': [26, 29]},
        'S': {'chest': [86, 91], 'waist': [68, 73], 'arm': [28, 31]},
        'M': {'chest': [91, 97], 'waist': [73, 79], 'arm': [30, 33]},
        'L': {'chest': [97, 104], 'waist': [79, 86], 'arm': [32, 35]},
        'XL': {'chest': [104, 112], 'waist': [86, 94], 'arm': [34, 37]},
    },
    'spodnie eleganckie': {
        'XS': {'waist': [62, 66], 'hips': [86, 90], 'thigh': [49, 53]},
        'S': {'waist': [66, 70], 'hips': [90, 94], 'thigh': [53, 56]},
        'M': {'waist': [70, 74], 'hips': [94, 98], 'thigh': [56, 59]},
        'L': {'waist': [74, 79], 'hips': [98, 103], 'thigh': [59, 63]},
        'XL': {'waist': [79, 85], 'hips': [103, 109], 'thigh': [63, 67]},
    },
    'spodnie jeansowe': {
        'XS': {'waist': [61, 65], 'hips': [85, 89], 'thigh': [50, 54]},
        'S': {'waist': [65, 69], 'hips': [89, 94], 'thigh': [54, 57]},
        'M': {'waist': [69, 74], 'hips': [94, 99], 'thigh': [57, 60]},
        'L': {'waist': [74, 80], 'hips': [99, 105], 'thigh': [60, 64]},
        'XL': {'waist': [80, 87], 'hips': [105, 112], 'thigh': [64, 68]},
    },
    'spodnie casualowe': {
        'XS': {'waist': [62, 67], 'hips': [86, 91], 'thigh': [50, 54]},
        'S': {'waist': [67, 72], 'hips': [91, 96], 'thigh': [54, 57]},
        'M': {'waist': [72, 77], 'hips': [96, 101], 'thigh': [57, 60]},
        'L': {'waist': [77, 83], 'hips': [101, 107], 'thigh': [60, 64]},
        'XL': {'waist': [83, 90], 'hips': [107, 114], 'thigh': [64, 68]},
    },
    'gorset': {
        'XS': {'bust': [78, 82], 'waist': [58, 62]},
        'S': {'bust': [82, 86], 'waist': [62, 66]},
        'M': {'bust': [86, 90], 'waist': [66, 70]},
        'L': {'bust': [90, 95], 'waist': [70, 75]},
        'XL': {'bust': [95, 101], 'waist': [75, 81]},
    },
    'T-shirt': {
        'XS': {'chest': [82, 87], 'waist': [66, 71], 'arm': [25, 28]},
        'S': {'chest': [87, 93], 'waist': [71, 77], 'arm': [27, 30]},
        'M': {'chest': [93, 100], 'waist': [77, 84], 'arm': [29, 32]},
        'L': {'chest': [100, 108], 'waist': [84, 92], 'arm': [31, 34]},
        'XL': {'chest': [108, 117], 'waist': [92, 101], 'arm': [33, 36]},
    },
    'bluza': {
        'XS': {'chest': [86, 92], 'waist': [72, 78], 'arm': [27, 30]},
        'S': {'chest': [92, 99], 'waist': [78, 85], 'arm': [29, 32]},
        'M': {'chest': [99, 107], 'waist': [85, 93], 'arm': [31, 34]},
        'L': {'chest': [107, 116], 'waist': [93, 102], 'arm': [33, 36]},
        'XL': {'chest': [116, 126], 'waist': [102, 112], 'arm': [35, 38]},
    },
    'bluza z kapturem': {
        'XS': {'chest': [88, 94], 'waist': [74, 80], 'arm': [27, 30]},
        'S': {'chest': [94, 101], 'waist': [80, 87], 'arm': [29, 32]},
        'M': {'chest': [101, 109], 'waist': [87, 95], 'arm': [31, 34]},
        'L': {'chest': [109, 118], 'waist': [95, 104], 'arm': [33, 36]},
        'XL': {'chest': [118, 128], 'waist': [104, 114], 'arm': [35, 38]},
    },
    'garnitur': {
        'XS': {'chest': [86, 90], 'waist': [70, 74], 'hips': [88, 92], 'thigh': [50, 54], 'arm': [28, 31]},
        'S': {'chest': [90, 95], 'waist': [74, 79], 'hips': [92, 97], 'thigh': [54, 57], 'arm': [30, 33]},
        'M': {'chest': [95, 101], 'waist': [79, 85], 'hips': [97, 103], 'thigh': [57, 60], 'arm': [32, 35]},
        'L': {'chest': [101, 108], 'waist': [85, 92], 'hips': [103, 110], 'thigh': [60, 64], 'arm': [34, 37]},
        'XL': {'chest': [108, 116], 'waist': [92, 100], 'hips': [110, 118], 'thigh': [64, 68], 'arm': [36, 39]},
    },
}


LANGUAGE_CHOICES = {
    'Auto / browser': 'auto',
    'Polski': 'pl',
    'English': 'en',
    'Español': 'es',
    'Italiano': 'it',
    'Français': 'fr',
    'Deutsch': 'de',
    'العربية': 'ar',
    '日本語': 'ja',
    '中文（普通话）': 'zh',
    'Nederlands': 'nl',
    'Svenska': 'sv',
    'Čeština': 'cs',
    'Português': 'pt',
    'Русский': 'ru',
    'Ελληνικά': 'el',
    'हिन्दी': 'hi',
    'বাংলা': 'bn',
    'اردو': 'ur',
    '한국어': 'ko',
    'Basa Jawa': 'jv',
    'Türkçe': 'tr',
    'Tiếng Việt': 'vi',
    'தமிழ்': 'ta',
    'فارسی': 'fa',
    'Bahasa Melayu': 'ms',
    'Українська': 'uk',
    'Dansk': 'da',
    'Norsk': 'no',
    'Română': 'ro',
    'Български': 'bg',
    'Српски': 'sr',
    'Hrvatski': 'hr',
    'Slovenčina': 'sk',
    'Slovenščina': 'sl',
}

BROWSER_LANG_MAP = {
    'pl': 'pl', 'en': 'en', 'es': 'es', 'it': 'it', 'fr': 'fr', 'de': 'de', 'ar': 'ar',
    'ja': 'ja', 'zh': 'zh', 'nl': 'nl', 'sv': 'sv', 'cs': 'cs', 'pt': 'pt', 'ru': 'ru', 'el': 'el',
    'hi': 'hi', 'bn': 'bn', 'ur': 'ur', 'ko': 'ko', 'jv': 'jv', 'tr': 'tr', 'vi': 'vi', 'ta': 'ta',
    'fa': 'fa', 'ms': 'ms', 'uk': 'uk', 'da': 'da', 'no': 'no', 'nb': 'no', 'nn': 'no', 'ro': 'ro',
    'bg': 'bg', 'sr': 'sr', 'hr': 'hr', 'sk': 'sk', 'sl': 'sl'
}

LOCALE_CHOICES = {
    'Auto / browser': 'auto',
    'Polski (Polska)': 'pl-PL',
    'English (United States)': 'en-US',
    'English (United Kingdom)': 'en-GB',
    'Español (España)': 'es-ES',
    'Español (Latinoamérica)': 'es-419',
    'Italiano (Italia)': 'it-IT',
    'Français (France)': 'fr-FR',
    'Français (Canada)': 'fr-CA',
    'Deutsch (Deutschland)': 'de-DE',
    'Deutsch (Österreich)': 'de-AT',
    'Deutsch (Schweiz)': 'de-CH',
    'العربية (السعودية)': 'ar-SA',
    '日本語 (日本)': 'ja-JP',
    '中文（简体，中国）': 'zh-Hans-CN',
    '中文（繁體，台灣）': 'zh-Hant-TW',
    'Nederlands (Nederland)': 'nl-NL',
    'Nederlands (België)': 'nl-BE',
    'Svenska (Sverige)': 'sv-SE',
    'Čeština (Česko)': 'cs-CZ',
    'Português (Brasil)': 'pt-BR',
    'Português (Portugal)': 'pt-PT',
    'Русский (Россия)': 'ru-RU',
    'Ελληνικά (Ελλάδα)': 'el-GR',
    'हिन्दी (भारत)': 'hi-IN',
    'বাংলা (বাংলাদেশ)': 'bn-BD',
    'বাংলা (ভারত)': 'bn-IN',
    'اردو (پاکستان)': 'ur-PK',
    'اردو (भारत)': 'ur-IN',
    '한국어 (대한민국)': 'ko-KR',
    'Basa Jawa (Indonesia)': 'jv-ID',
    'Türkçe (Türkiye)': 'tr-TR',
    'Tiếng Việt (Việt Nam)': 'vi-VN',
    'தமிழ் (இந்தியா)': 'ta-IN',
    'தமிழ் (ශ්‍රී ලංකා / இலங்கை)': 'ta-LK',
    'தமிழ் (Singapore)': 'ta-SG',
    'فارسی (ایران)': 'fa-IR',
    'Bahasa Melayu (Malaysia)': 'ms-MY',
    'Bahasa Melayu (Singapore)': 'ms-SG',
    'Bahasa Melayu (Brunei)': 'ms-BN',
    'Українська (Україна)': 'uk-UA',
    'Dansk (Danmark)': 'da-DK',
    'Norsk Bokmål (Norge)': 'nb-NO',
    'Norsk Nynorsk (Noreg)': 'nn-NO',
    'Română (România)': 'ro-RO',
    'Български (България)': 'bg-BG',
    'Српски (ћирилица, Србија)': 'sr-Cyrl-RS',
    'Srpski (latinica, Srbija)': 'sr-Latn-RS',
    'Hrvatski (Hrvatska)': 'hr-HR',
    'Slovenčina (Slovensko)': 'sk-SK',
    'Slovenščina (Slovenija)': 'sl-SI',
    'Bahasa Indonesia (Indonesia)': 'id-ID',
    'Filipino (Pilipinas)': 'fil-PH',
    'मराठी (भारत)': 'mr-IN',
    'తెలుగు (భారతదేశం)': 'te-IN',
    'ગુજરાતી (ભારત)': 'gu-IN',
    'ਪੰਜਾਬੀ (ਗੁਰਮੁਖੀ, ਭਾਰਤ)': 'pa-Guru-IN',
    'پنجابی (پاکستان)': 'pnb-Arab-PK',
    'پښتو (افغانستان)': 'ps-AF',
    'پښتو (پاکستان)': 'ps-PK',
    'Nigerian Pidgin (Nigeria)': 'pcm-NG',
    '粵語（繁體，香港）': 'yue-Hant-HK',
    '吴语（简体，中国）': 'wuu-Hans-CN',
}

LOCALE_TO_LANG = {
    'pl-PL': 'pl', 'en-US': 'en', 'en-GB': 'en', 'es-ES': 'es', 'es-419': 'es', 'it-IT': 'it',
    'fr-FR': 'fr', 'fr-CA': 'fr', 'de-DE': 'de', 'de-AT': 'de', 'de-CH': 'de', 'ar-SA': 'ar', 'ja-JP': 'ja',
    'zh-Hans-CN': 'zh', 'zh-Hant-TW': 'zh', 'nl-NL': 'nl', 'nl-BE': 'nl', 'sv-SE': 'sv', 'cs-CZ': 'cs',
    'pt-BR': 'pt', 'pt-PT': 'pt', 'ru-RU': 'ru', 'el-GR': 'el', 'hi-IN': 'hi', 'bn-BD': 'bn', 'bn-IN': 'bn',
    'ur-PK': 'ur', 'ur-IN': 'ur', 'ko-KR': 'ko', 'jv-ID': 'jv', 'tr-TR': 'tr', 'vi-VN': 'vi',
    'ta-IN': 'ta', 'ta-LK': 'ta', 'ta-SG': 'ta', 'fa-IR': 'fa', 'ms-MY': 'ms', 'ms-SG': 'ms', 'ms-BN': 'ms',
    'uk-UA': 'uk', 'da-DK': 'da', 'nb-NO': 'no', 'nn-NO': 'no', 'ro-RO': 'ro', 'bg-BG': 'bg',
    'sr-Cyrl-RS': 'sr', 'sr-Latn-RS': 'sr', 'hr-HR': 'hr', 'sk-SK': 'sk', 'sl-SI': 'sl',
    'id-ID': 'id', 'fil-PH': 'fil', 'mr-IN': 'mr', 'te-IN': 'te', 'gu-IN': 'gu',
    'pa-Guru-IN': 'pa', 'pnb-Arab-PK': 'pnb', 'ps-AF': 'ps', 'ps-PK': 'ps',
    'pcm-NG': 'pcm', 'yue-Hant-HK': 'yue', 'wuu-Hans-CN': 'wuu',
}

BROWSER_LOCALE_PREFIXES = {
    'pl-pl': 'pl-PL', 'pl': 'pl-PL', 'en-gb': 'en-GB', 'en-us': 'en-US', 'en': 'en-US',
    'es-es': 'es-ES', 'es-419': 'es-419', 'es-mx': 'es-419', 'es-ar': 'es-419', 'es-co': 'es-419', 'es-cl': 'es-419', 'es-pe': 'es-419', 'es': 'es-ES',
    'it-it': 'it-IT', 'it': 'it-IT', 'fr-fr': 'fr-FR', 'fr-ca': 'fr-CA', 'fr': 'fr-FR',
    'de-de': 'de-DE', 'de-at': 'de-AT', 'de-ch': 'de-CH', 'de': 'de-DE', 'ar-sa': 'ar-SA', 'ar': 'ar-SA',
    'ja-jp': 'ja-JP', 'ja': 'ja-JP', 'zh-hans-cn': 'zh-Hans-CN', 'zh-cn': 'zh-Hans-CN', 'zh-sg': 'zh-Hans-CN',
    'zh-hant-tw': 'zh-Hant-TW', 'zh-tw': 'zh-Hant-TW', 'zh-hk': 'zh-Hant-TW', 'zh': 'zh-Hans-CN',
    'nl-nl': 'nl-NL', 'nl-be': 'nl-BE', 'nl': 'nl-NL', 'sv-se': 'sv-SE', 'sv': 'sv-SE',
    'cs-cz': 'cs-CZ', 'cs': 'cs-CZ', 'pt-br': 'pt-BR', 'pt-pt': 'pt-PT', 'pt': 'pt-BR',
    'ru-ru': 'ru-RU', 'ru': 'ru-RU', 'el-gr': 'el-GR', 'el': 'el-GR', 'hi-in': 'hi-IN', 'hi': 'hi-IN',
    'bn-bd': 'bn-BD', 'bn-in': 'bn-IN', 'bn': 'bn-BD', 'ur-pk': 'ur-PK', 'ur-in': 'ur-IN', 'ur': 'ur-PK',
    'ko-kr': 'ko-KR', 'ko': 'ko-KR', 'jv-id': 'jv-ID', 'jv': 'jv-ID', 'tr-tr': 'tr-TR', 'tr': 'tr-TR',
    'vi-vn': 'vi-VN', 'vi': 'vi-VN', 'ta-in': 'ta-IN', 'ta-lk': 'ta-LK', 'ta-sg': 'ta-SG', 'ta': 'ta-IN',
    'fa-ir': 'fa-IR', 'fa': 'fa-IR', 'ms-my': 'ms-MY', 'ms-sg': 'ms-SG', 'ms-bn': 'ms-BN', 'ms': 'ms-MY',
    'uk-ua': 'uk-UA', 'uk': 'uk-UA', 'da-dk': 'da-DK', 'da': 'da-DK', 'nb-no': 'nb-NO', 'nn-no': 'nn-NO', 'no': 'nb-NO',
    'ro-ro': 'ro-RO', 'ro': 'ro-RO', 'bg-bg': 'bg-BG', 'bg': 'bg-BG', 'sr-cyrl-rs': 'sr-Cyrl-RS', 'sr-latn-rs': 'sr-Latn-RS', 'sr': 'sr-Cyrl-RS',
    'hr-hr': 'hr-HR', 'hr': 'hr-HR', 'sk-sk': 'sk-SK', 'sk': 'sk-SK', 'sl-si': 'sl-SI', 'sl': 'sl-SI',
    'id-id': 'id-ID', 'id': 'id-ID', 'fil-ph': 'fil-PH', 'fil': 'fil-PH', 'tl-ph': 'fil-PH', 'tl': 'fil-PH',
    'mr-in': 'mr-IN', 'mr': 'mr-IN', 'te-in': 'te-IN', 'te': 'te-IN', 'gu-in': 'gu-IN', 'gu': 'gu-IN',
    'pa-guru-in': 'pa-Guru-IN', 'pa-in': 'pa-Guru-IN', 'pa': 'pa-Guru-IN',
    'pnb-arab-pk': 'pnb-Arab-PK', 'pnb-pk': 'pnb-Arab-PK', 'pnb': 'pnb-Arab-PK',
    'ps-af': 'ps-AF', 'ps-pk': 'ps-PK', 'ps': 'ps-AF',
    'pcm-ng': 'pcm-NG', 'pcm': 'pcm-NG',
    'yue-hant-hk': 'yue-Hant-HK', 'yue-hk': 'yue-Hant-HK', 'yue': 'yue-Hant-HK',
    'wuu-hans-cn': 'wuu-Hans-CN', 'wuu-cn': 'wuu-Hans-CN', 'wuu': 'wuu-Hans-CN',
}

UI_I18N = {
    'en': {
        'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8',
        'Nowa analiza': 'New analysis', 'Feedback po zakupie': 'Post-purchase feedback', 'Historia testów': 'Test history', 'Wyszukiwanie po zdjęciu': 'Search by photo', 'Wgraj zdjęcie produktu': 'Upload product photo', 'Uruchom aparat produktu': 'Open product camera', 'Wyszukaj podobne produkty': 'Search similar products', 'Wyniki wyszukiwania wizualnego': 'Visual search results', 'Brak wyników wyszukiwania': 'No visual search results', 'To jest beta lokalnego wyszukiwania po zdjęciu — działa na indeksie katalogu demo, podobnie do mini Google Lens dla MVP.': 'This is a beta local photo search — it works on the demo catalog index, similar to a mini Google Lens for MVP.', 'Podpowiedź: najlepiej działa zdjęcie samego produktu lub produktu na jasnym tle.': 'Tip: it works best with a photo of the product alone or on a clean background.',
        '1. Dane użytkownika': '1. User data', '2. Dane produktu': '2. Product data',
        'Płeć': 'Gender', 'kobieta': 'female', 'mężczyzna': 'male',
        'System jednostek': 'Unit system', 'metryczny': 'metric', 'imperialny': 'imperial',
        'Wiek': 'Age', 'Wzrost (cm)': 'Height (cm)', 'Waga (kg)': 'Weight (kg)',
        'Jak chcesz określić wymiary?': 'How do you want to determine measurements?',
        'AI scan': 'AI scan', 'AI scan + ręczna korekta': 'AI scan + manual correction', 'Wpiszę ręcznie': 'I will enter them manually',
        'Preferencja dopasowania (opcjonalnie)': 'Fit preference (optional)',
        'automatycznie': 'automatic', 'standard': 'standard', 'dopasowane': 'fitted', 'luźniejsze': 'looser',
        'Instrukcja ustawienia do zdjęcia': 'Photo positioning guide',
        'Pokaż instrukcję ręcznych pomiarów': 'Show manual measurement guide',
        'Skąd bierzemy zdjęcia?': 'Where do we get the photos from?', 'Aparat (zalecane)': 'Camera (recommended)', 'Upload z galerii': 'Upload from gallery',
        'Jak powstanie zdjęcie?': 'How will the photo be taken?',
        'inna osoba trzyma telefon': 'another person holds the phone', 'telefon oparty stabilnie + timer': 'phone placed steadily + timer',
        'statyw': 'tripod', 'zdjęcie wykonane samemu': 'self-taken photo',
        'Czy ubranie, w którym jesteś, jest wystarczająco dopasowane, żeby pomiar był dokładny?': 'Is the outfit you are wearing fitted enough for an accurate measurement?',
        'obcisłe': 'tight / fitted', 'raczej dopasowane': 'rather fitted', 'średnie': 'medium', 'luźne': 'loose', 'bardzo luźne': 'very loose', 'nie wiem': 'I do not know',
        'Zrób zdjęcie FRONT': 'Take FRONT photo', 'Zrób zdjęcie PROFIL': 'Take PROFILE photo', 'Zrób zdjęcie TYŁ (opcjonalne, zalecane)': 'Take BACK photo (optional, recommended)',
        'Dodaj zdjęcie FRONT': 'Add FRONT photo', 'Dodaj zdjęcie PROFIL': 'Add PROFILE photo', 'Dodaj zdjęcie TYŁ (opcjonalne, zalecane)': 'Add BACK photo (optional, recommended)',
        'Analizuj sylwetkę': 'Analyze body', 'Wynik analizy sylwetki': 'Body analysis result', 'Typ sylwetki': 'Body type',
        'Talia': 'Waist', 'Biodra': 'Hips', 'Pełny profil wymiarów': 'Full measurement profile',
        'Dla kogo szukasz produktu?': 'Who is the product for?', 'Gałąź': 'Branch', 'Typ produktu': 'Product type',
        'Źródło produktu': 'Product source', 'Link do produktu': 'Product link', 'Katalog demo': 'Demo catalog',
        'Wklej link do produktu': 'Paste product link', 'Pobierz dane produktu z linku': 'Fetch product data from link',
        'Profil produktu': 'Product profile', 'Sygnały z opinii klientów': 'Customer review signals', 'Tabela rozmiarów': 'Size chart',
        '3. Generuj rekomendację i zapisz do bazy': '3. Generate recommendation and save to database',
        'Werdykt': 'Verdict', 'Komentarz AI': 'AI commentary', 'Punkty uwagi': 'Points to watch', 'Wymiary użyte do decyzji': 'Measurements used for the decision',
        'Feedback po zakupie': 'Post-purchase feedback', 'Historia testów': 'Test history', 'Wyszukiwanie po zdjęciu': 'Search by photo', 'Wgraj zdjęcie produktu': 'Upload product photo', 'Uruchom aparat produktu': 'Open product camera', 'Wyszukaj podobne produkty': 'Search similar products', 'Wyniki wyszukiwania wizualnego': 'Visual search results', 'Brak wyników wyszukiwania': 'No visual search results', 'To jest beta lokalnego wyszukiwania po zdjęciu — działa na indeksie katalogu demo, podobnie do mini Google Lens dla MVP.': 'This is a beta local photo search — it works on the demo catalog index, similar to a mini Google Lens for MVP.', 'Podpowiedź: najlepiej działa zdjęcie samego produktu lub produktu na jasnym tle.': 'Tip: it works best with a photo of the product alone or on a clean background.', 'Email': 'Email',
        'Dla kogo szukasz produktu?': 'Who is the product for?',
        'Co nowego w v8.6.8': 'What is new in v8.6.8',
        'Język': 'Language', 'Zgody i prywatność': 'Consent & privacy', 'Akceptuję analizę zdjęć ciała do działania aplikacji': 'I agree to body image analysis for app functionality', 'Zgadzam się na zapis zdjęć w systemie': 'I agree to storing photos in the system', 'Zgadzam się na wykorzystanie danych do poprawy modelu': 'I agree to using data to improve the model', 'Bez zgody na analizę aplikacja nie może przejść dalej.': 'Without consent for analysis the app cannot continue.', 'Jeśli nie wyrażasz zgody na zapis zdjęć, system zapisze tylko wynik analizy bez plików zdjęć.': 'If you do not consent to storing images, the system will only save the analysis result without image files.', 'Jeśli nie wyrażasz zgody na użycie danych do poprawy modelu, Twoje dane nie trafią do calibration loop.': 'If you do not consent to using data to improve the model, your data will not enter the calibration loop.', 'Prywatność i zgody': 'Privacy & consent',
    },
    'es': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8', 'Nowa analiza':'Nuevo análisis','Feedback po zakupie':'Feedback tras la compra','Historia testów':'Historial de pruebas','1. Dane użytkownika':'1. Datos del usuario','2. Dane produktu':'2. Datos del producto','Płeć':'Sexo','kobieta':'mujer','mężczyzna':'hombre','System jednostek':'Sistema de unidades','metryczny':'métrico','imperialny':'imperial','Wiek':'Edad','Email':'Correo electrónico','Język':'Idioma'},
    'it': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Nuova analisi','Feedback po zakupie':'Feedback post-acquisto','Historia testów':'Cronologia test','1. Dane użytkownika':'1. Dati utente','2. Dane produktu':'2. Dati prodotto','Płeć':'Genere','kobieta':'donna','mężczyzna':'uomo','System jednostek':'Sistema di unità','metryczny':'metrico','imperialny':'imperiale','Wiek':'Età','Email':'Email','Język':'Lingua'},
    'fr': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Nouvelle analyse','Feedback po zakupie':'Retour après achat','Historia testów':'Historique des tests','1. Dane użytkownika':'1. Données utilisateur','2. Dane produktu':'2. Données produit','Płeć':'Sexe','kobieta':'femme','mężczyzna':'homme','System jednostek':'Système d’unités','metryczny':'métrique','imperialny':'impérial','Wiek':'Âge','Email':'E-mail','Język':'Langue'},
    'de': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Neue Analyse','Feedback po zakupie':'Feedback nach dem Kauf','Historia testów':'Testverlauf','1. Dane użytkownika':'1. Benutzerdaten','2. Dane produktu':'2. Produktdaten','Płeć':'Geschlecht','kobieta':'Frau','mężczyzna':'Mann','System jednostek':'Einheitensystem','metryczny':'metrisch','imperialny':'imperial','Wiek':'Alter','Email':'E-Mail','Język':'Sprache'},
    'ar': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'تحليل جديد','Feedback po zakupie':'ملاحظات بعد الشراء','Historia testów':'سجل الاختبارات','1. Dane użytkownika':'1. بيانات المستخدم','2. Dane produktu':'2. بيانات المنتج','Płeć':'الجنس','kobieta':'أنثى','mężczyzna':'ذكر','System jednostek':'نظام الوحدات','metryczny':'متري','imperialny':'إمبراطوري','Wiek':'العمر','Email':'البريد الإلكتروني','Język':'اللغة'},
    'ja': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'新規分析','Feedback po zakupie':'購入後フィードバック','Historia testów':'テスト履歴','1. Dane użytkownika':'1. ユーザーデータ','2. Dane produktu':'2. 商品データ','Płeć':'性別','kobieta':'女性','mężczyzna':'男性','System jednostek':'単位系','metryczny':'メートル法','imperialny':'ヤード・ポンド法','Wiek':'年齢','Email':'メール','Język':'言語'},
    'zh': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'新分析','Feedback po zakupie':'购买后反馈','Historia testów':'测试历史','1. Dane użytkownika':'1. 用户数据','2. Dane produktu':'2. 产品数据','Płeć':'性别','kobieta':'女性','mężczyzna':'男性','System jednostek':'单位制','metryczny':'公制','imperialny':'英制','Wiek':'年龄','Email':'电子邮件','Język':'语言'},
    'nl': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Nieuwe analyse','Feedback po zakupie':'Feedback na aankoop','Historia testów':'Testgeschiedenis','1. Dane użytkownika':'1. Gebruikersgegevens','2. Dane produktu':'2. Productgegevens','Płeć':'Geslacht','kobieta':'vrouw','mężczyzna':'man','System jednostek':'Eenhedensysteem','metryczny':'metrisch','imperialny':'imperiaal','Wiek':'Leeftijd','Email':'E-mail','Język':'Taal'},
    'sv': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Ny analys','Feedback po zakupie':'Feedback efter köp','Historia testów':'Testhistorik','1. Dane użytkownika':'1. Användardata','2. Dane produktu':'2. Produktdata','Płeć':'Kön','kobieta':'kvinna','mężczyzna':'man','System jednostek':'Enhetssystem','metryczny':'metriskt','imperialny':'imperialt','Wiek':'Ålder','Email':'E-post','Język':'Språk'},
    'cs': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Nová analýza','Feedback po zakupie':'Zpětná vazba po nákupu','Historia testów':'Historie testů','1. Dane użytkownika':'1. Údaje uživatele','2. Dane produktu':'2. Údaje o produktu','Płeć':'Pohlaví','kobieta':'žena','mężczyzna':'muž','System jednostek':'Systém jednotek','metryczny':'metrický','imperialny':'imperiální','Wiek':'Věk','Email':'E-mail','Język':'Jazyk'},
    'pt': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Nova análise','Feedback po zakupie':'Feedback pós-compra','Historia testów':'Histórico de testes','1. Dane użytkownika':'1. Dados do usuário','2. Dane produktu':'2. Dados do produto','Płeć':'Género','kobieta':'mulher','mężczyzna':'homem','System jednostek':'Sistema de unidades','metryczny':'métrico','imperialny':'imperial','Wiek':'Idade','Email':'E-mail','Język':'Idioma'},
    'ru': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Новый анализ','Feedback po zakupie':'Отзыв после покупки','Historia testów':'История тестов','1. Dane użytkownika':'1. Данные пользователя','2. Dane produktu':'2. Данные товара','Płeć':'Пол','kobieta':'женщина','mężczyzna':'мужчина','System jednostek':'Система единиц','metryczny':'метрическая','imperialny':'имперская','Wiek':'Возраст','Email':'Email','Język':'Язык'},
    'el': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Νέα ανάλυση','Feedback po zakupie':'Ανατροφοδότηση μετά την αγορά','Historia testów':'Ιστορικό δοκιμών','1. Dane użytkownika':'1. Στοιχεία χρήστη','2. Dane produktu':'2. Στοιχεία προϊόντος','Płeć':'Φύλο','kobieta':'γυναίκα','mężczyzna':'άνδρας','System jednostek':'Σύστημα μονάδων','metryczny':'μετρικό','imperialny':'ιμπεριακό','Wiek':'Ηλικία','Email':'Email','Język':'Γλώσσα'},
    'hi': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'नया विश्लेषण','Feedback po zakupie':'खरीद के बाद प्रतिक्रिया','Historia testów':'टेस्ट इतिहास','1. Dane użytkownika':'1. उपयोगकर्ता डेटा','2. Dane produktu':'2. उत्पाद डेटा','Płeć':'लिंग','kobieta':'महिला','mężczyzna':'पुरुष','System jednostek':'मापन प्रणाली','metryczny':'मेट्रिक','imperialny':'इम्पीरियल','Wiek':'आयु','Email':'ईमेल','Język':'भाषा'},
    'bn': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'নতুন বিশ্লেষণ','Feedback po zakupie':'ক্রয়ের পর মতামত','Historia testów':'পরীক্ষার ইতিহাস','1. Dane użytkownika':'1. ব্যবহারকারীর তথ্য','2. Dane produktu':'2. পণ্যের তথ্য','Płeć':'লিঙ্গ','kobieta':'নারী','mężczyzna':'পুরুষ','System jednostek':'একক পদ্ধতি','metryczny':'মেট্রিক','imperialny':'ইম্পেরিয়াল','Wiek':'বয়স','Email':'ইমেল','Język':'ভাষা'},
    'ur': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'نیا تجزیہ','Feedback po zakupie':'خرید کے بعد فیڈبیک','Historia testów':'ٹیسٹ ہسٹری','1. Dane użytkownika':'1. صارف کا ڈیٹا','2. Dane produktu':'2. پروڈکٹ ڈیٹا','Płeć':'جنس','kobieta':'خاتون','mężczyzna':'مرد','System jednostek':'یونٹ سسٹم','metryczny':'میٹرک','imperialny':'امپیریل','Wiek':'عمر','Email':'ای میل','Język':'زبان'},
    'ko': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'새 분석','Feedback po zakupie':'구매 후 피드백','Historia testów':'테스트 기록','1. Dane użytkownika':'1. 사용자 데이터','2. Dane produktu':'2. 제품 데이터','Płeć':'성별','kobieta':'여성','mężczyzna':'남성','System jednostek':'단위 체계','metryczny':'미터법','imperialny':'야드파운드법','Wiek':'나이','Email':'이메일','Język':'언어'},
    'jv': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Analisis anyar','Feedback po zakupie':'Umpan balik sawise tuku','Historia testów':'Riwayat tes','1. Dane użytkownika':'1. Data pangguna','2. Dane produktu':'2. Data produk','Płeć':'Jinis kelamin','kobieta':'wadon','mężczyzna':'lanang','System jednostek':'Sistem ukuran','metryczny':'metrik','imperialny':'imperial','Wiek':'Umur','Email':'Email','Język':'Basa'},
    'tr': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Yeni analiz','Feedback po zakupie':'Satın alma sonrası geri bildirim','Historia testów':'Test geçmişi','1. Dane użytkownika':'1. Kullanıcı verileri','2. Dane produktu':'2. Ürün verileri','Płeć':'Cinsiyet','kobieta':'kadın','mężczyzna':'erkek','System jednostek':'Birim sistemi','metryczny':'metrik','imperialny':'emperyal','Wiek':'Yaş','Email':'E-posta','Język':'Dil'},
    'vi': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Phân tích mới','Feedback po zakupie':'Phản hồi sau mua','Historia testów':'Lịch sử kiểm tra','1. Dane użytkownika':'1. Dữ liệu người dùng','2. Dane produktu':'2. Dữ liệu sản phẩm','Płeć':'Giới tính','kobieta':'nữ','mężczyzna':'nam','System jednostek':'Hệ đơn vị','metryczny':'mét','imperialny':'anh mỹ','Wiek':'Tuổi','Email':'Email','Język':'Ngôn ngữ'},
    'ta': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'புதிய பகுப்பாய்வு','Feedback po zakupie':'வாங்கிய பிறகான கருத்து','Historia testów':'சோதனை வரலாறு','1. Dane użytkownika':'1. பயனர் தரவு','2. Dane produktu':'2. தயாரிப்பு தரவு','Płeć':'பாலினம்','kobieta':'பெண்','mężczyzna':'ஆண்','System jednostek':'அலகு அமைப்பு','metryczny':'மெட்ரிக்','imperialny':'இம்பீரியல்','Wiek':'வயது','Email':'மின்னஞ்சல்','Język':'மொழி'},
    'fa': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'تحلیل جدید','Feedback po zakupie':'بازخورد پس از خرید','Historia testów':'تاریخچه تست‌ها','1. Dane użytkownika':'1. داده‌های کاربر','2. Dane produktu':'2. داده‌های محصول','Płeć':'جنسیت','kobieta':'زن','mężczyzna':'مرد','System jednostek':'سیستم واحدها','metryczny':'متریک','imperialny':'امپریال','Wiek':'سن','Email':'ایمیل','Język':'زبان'},
    'ms': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Analisis baharu','Feedback po zakupie':'Maklum balas selepas pembelian','Historia testów':'Sejarah ujian','1. Dane użytkownika':'1. Data pengguna','2. Dane produktu':'2. Data produk','Płeć':'Jantina','kobieta':'wanita','mężczyzna':'lelaki','System jednostek':'Sistem unit','metryczny':'metrik','imperialny':'imperial','Wiek':'Umur','Email':'E-mel','Język':'Bahasa'},
    'uk': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Новий аналіз','Feedback po zakupie':'Відгук після покупки','Historia testów':'Історія тестів','1. Dane użytkownika':'1. Дані користувача','2. Dane produktu':'2. Дані продукту','Płeć':'Стать','kobieta':'жінка','mężczyzna':'чоловік','System jednostek':'Система одиниць','metryczny':'метрична','imperialny':'імперська','Wiek':'Вік','Email':'Email','Język':'Мова'},
    'da': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Ny analyse','Feedback po zakupie':'Feedback efter køb','Historia testów':'Testhistorik','1. Dane użytkownika':'1. Brugerdata','2. Dane produktu':'2. Produktdata','Płeć':'Køn','kobieta':'kvinde','mężczyzna':'mand','System jednostek':'Enhedssystem','metryczny':'metrisk','imperialny':'imperialt','Wiek':'Alder','Email':'E-mail','Język':'Sprog'},
    'no': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Ny analyse','Feedback po zakupie':'Tilbakemelding etter kjøp','Historia testów':'Testhistorikk','1. Dane użytkownika':'1. Brukerdata','2. Dane produktu':'2. Produktdata','Płeć':'Kjønn','kobieta':'kvinne','mężczyzna':'mann','System jednostek':'Enhetssystem','metryczny':'metrisk','imperialny':'imperial','Wiek':'Alder','Email':'E-post','Język':'Språk'},
    'ro': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Analiză nouă','Feedback po zakupie':'Feedback după cumpărare','Historia testów':'Istoric teste','1. Dane użytkownika':'1. Date utilizator','2. Dane produktu':'2. Date produs','Płeć':'Gen','kobieta':'femeie','mężczyzna':'bărbat','System jednostek':'Sistem de unități','metryczny':'metric','imperialny':'imperial','Wiek':'Vârstă','Email':'Email','Język':'Limbă'},
    'bg': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Нов анализ','Feedback po zakupie':'Обратна връзка след покупка','Historia testów':'История на тестовете','1. Dane użytkownika':'1. Данни за потребителя','2. Dane produktu':'2. Данни за продукта','Płeć':'Пол','kobieta':'жена','mężczyzna':'мъж','System jednostek':'Система единици','metryczny':'метрична','imperialny':'имперска','Wiek':'Възраст','Email':'Имейл','Język':'Език'},
    'sr': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Нова анализа','Feedback po zakupie':'Повратна информација након куповине','Historia testów':'Историја тестова','1. Dane użytkownika':'1. Подаци о кориснику','2. Dane produktu':'2. Подаци о производу','Płeć':'Пол','kobieta':'жена','mężczyzna':'мушкарац','System jednostek':'Систем јединица','metryczny':'метрички','imperialny':'империјални','Wiek':'Узраст','Email':'Имејл','Język':'Језик'},
    'hr': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Nova analiza','Feedback po zakupie':'Povratne informacije nakon kupnje','Historia testów':'Povijest testova','1. Dane użytkownika':'1. Podaci korisnika','2. Dane produktu':'2. Podaci o proizvodu','Płeć':'Spol','kobieta':'žena','mężczyzna':'muškarac','System jednostek':'Sustav jedinica','metryczny':'metrički','imperialny':'imperijalni','Wiek':'Dob','Email':'E-pošta','Język':'Jezik'},
    'sk': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Nová analýza','Feedback po zakupie':'Spätná väzba po nákupe','Historia testów':'História testov','1. Dane użytkownika':'1. Údaje používateľa','2. Dane produktu':'2. Údaje o produkte','Płeć':'Pohlavie','kobieta':'žena','mężczyzna':'muž','System jednostek':'Systém jednotiek','metryczny':'metrický','imperialny':'imperiálny','Wiek':'Vek','Email':'E-mail','Język':'Jazyk'},
    'sl': {'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8','Nowa analiza':'Nova analiza','Feedback po zakupie':'Povratne informacije po nakupu','Historia testów':'Zgodovina testov','1. Dane użytkownika':'1. Podatki uporabnika','2. Dane produktu':'2. Podatki o izdelku','Płeć':'Spol','kobieta':'ženska','mężczyzna':'moški','System jednostek':'Sistem enot','metryczny':'metrični','imperialny':'imperialni','Wiek':'Starost','Email':'E-pošta','Język':'Jezik'},

    'id': {
        'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8',
        'Nowa analiza': 'Analisis baru',
        'Feedback po zakupie': 'Umpan balik setelah pembelian',
        'Historia testów': 'Riwayat tes',
        '1. Dane użytkownika': '1. Data pengguna',
        '2. Dane produktu': '2. Data produk',
        'Płeć': 'Jenis kelamin',
        'kobieta': 'perempuan',
        'mężczyzna': 'laki-laki',
        'System jednostek': 'Sistem satuan',
        'metryczny': 'metrik',
        'imperialny': 'imperial'
    },
    'fil': {
        'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8',
        'Nowa analiza': 'Bagong pagsusuri',
        'Feedback po zakupie': 'Feedback pagkatapos ng pagbili',
        'Historia testów': 'Kasaysayan ng mga test',
        '1. Dane użytkownika': '1. Datos ng user',
        '2. Dane produktu': '2. Datos ng produkto',
        'Płeć': 'Kasarian',
        'kobieta': 'babae',
        'mężczyzna': 'lalaki',
        'System jednostek': 'Sistema ng sukat',
        'metryczny': 'metric',
        'imperialny': 'imperial'
    },
    'mr': {
        'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8',
        'Nowa analiza': 'नवीन विश्लेषण',
        'Feedback po zakupie': 'खरेदीनंतर अभिप्राय',
        'Historia testów': 'चाचणी इतिहास',
        '1. Dane użytkownika': '1. वापरकर्ता डेटा',
        '2. Dane produktu': '2. उत्पादन डेटा',
        'Płeć': 'लिंग',
        'kobieta': 'स्त्री',
        'mężczyzna': 'पुरुष',
        'System jednostek': 'मापन पद्धती',
        'metryczny': 'मेट्रिक',
        'imperialny': 'इम्पीरियल'
    },
    'te': {
        'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8',
        'Nowa analiza': 'కొత్త విశ్లేషణ',
        'Feedback po zakupie': 'కొనుగోలు తర్వాత అభిప్రాయం',
        'Historia testów': 'పరీక్షల చరిత్ర',
        '1. Dane użytkownika': '1. వినియోగదారు డేటా',
        '2. Dane produktu': '2. ఉత్పత్తి డేటా',
        'Płeć': 'లింగం',
        'kobieta': 'మహిళ',
        'mężczyzna': 'పురుషుడు',
        'System jednostek': 'కొలమాన వ్యవస్థ',
        'metryczny': 'మెట్రిక్',
        'imperialny': 'ఇంపీరియల్'
    },
    'gu': {
        'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8',
        'Nowa analiza': 'નવી વિશ્લેષણ',
        'Feedback po zakupie': 'ખરીદી પછી પ્રતિસાદ',
        'Historia testów': 'ટેસ્ટ ઇતિહાસ',
        '1. Dane użytkownika': '1. વપરાશકર્તા ડેટા',
        '2. Dane produktu': '2. ઉત્પાદન ડેટા',
        'Płeć': 'લિંગ',
        'kobieta': 'સ્ત્રી',
        'mężczyzna': 'પુરુષ',
        'System jednostek': 'માપ પ્રણાલી',
        'metryczny': 'મેટ્રિક',
        'imperialny': 'ઇમ્પિરિયલ'
    },
    'pa': {
        'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8',
        'Nowa analiza': 'ਨਵਾਂ ਵਿਸ਼ਲੇਸ਼ਣ',
        'Feedback po zakupie': 'ਖਰੀਦ ਤੋਂ ਬਾਅਦ ਫੀਡਬੈਕ',
        'Historia testów': 'ਟੈਸਟ ਇਤਿਹਾਸ',
        '1. Dane użytkownika': '1. ਵਰਤੋਂਕਾਰ ਡਾਟਾ',
        '2. Dane produktu': '2. ਉਤਪਾਦ ਡਾਟਾ',
        'Płeć': 'ਲਿੰਗ',
        'kobieta': 'ਮਹਿਲਾ',
        'mężczyzna': 'ਮਰਦ',
        'System jednostek': 'ਇਕਾਈ ਪ੍ਰਣਾਲੀ',
        'metryczny': 'ਮੈਟ੍ਰਿਕ',
        'imperialny': 'ਇੰਪੀਰੀਅਲ'
    },
    'pnb': {
        'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8',
        'Nowa analiza': 'نواں تجزیہ',
        'Feedback po zakupie': 'خرید کے بعد رائے',
        'Historia testów': 'ٹیسٹ ہسٹری',
        '1. Dane użytkownika': '1. صارف ڈیٹا',
        '2. Dane produktu': '2. پراڈکٹ ڈیٹا',
        'Płeć': 'جنس',
        'kobieta': 'عورت',
        'mężczyzna': 'مرد',
        'System jednostek': 'یونٹ سسٹم',
        'metryczny': 'میٹرک',
        'imperialny': 'امپیریل'
    },
    'ps': {
        'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8',
        'Nowa analiza': 'نوې شننه',
        'Feedback po zakupie': 'د پېرلو وروسته نظر',
        'Historia testów': 'د ازموینو تاریخ',
        '1. Dane użytkownika': '1. د کارونکي معلومات',
        '2. Dane produktu': '2. د محصول معلومات',
        'Płeć': 'جنس',
        'kobieta': 'ښځه',
        'mężczyzna': 'سړی',
        'System jednostek': 'د واحدونو سیستم',
        'metryczny': 'مېټريک',
        'imperialny': 'امپيريال'
    },
    'pcm': {
        'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8',
        'Nowa analiza': 'New analysis',
        'Feedback po zakupie': 'Feedback afta purchase',
        'Historia testów': 'Test history', 'Wyszukiwanie po zdjęciu': 'Search by photo', 'Wgraj zdjęcie produktu': 'Upload product photo', 'Uruchom aparat produktu': 'Open product camera', 'Wyszukaj podobne produkty': 'Search similar products', 'Wyniki wyszukiwania wizualnego': 'Visual search results', 'Brak wyników wyszukiwania': 'No visual search results', 'To jest beta lokalnego wyszukiwania po zdjęciu — działa na indeksie katalogu demo, podobnie do mini Google Lens dla MVP.': 'This is a beta local photo search — it works on the demo catalog index, similar to a mini Google Lens for MVP.', 'Podpowiedź: najlepiej działa zdjęcie samego produktu lub produktu na jasnym tle.': 'Tip: it works best with a photo of the product alone or on a clean background.',
        '1. Dane użytkownika': '1. User data',
        '2. Dane produktu': '2. Product data',
        'Płeć': 'Gender',
        'kobieta': 'woman',
        'mężczyzna': 'man',
        'System jednostek': 'Unit system',
        'metryczny': 'metric',
        'imperialny': 'imperial'
    },
    'yue': {
        'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8',
        'Nowa analiza': '新分析',
        'Feedback po zakupie': '購買後回饋',
        'Historia testów': '測試紀錄',
        '1. Dane użytkownika': '1. 用戶資料',
        '2. Dane produktu': '2. 產品資料',
        'Płeć': '性別',
        'kobieta': '女性',
        'mężczyzna': '男性',
        'System jednostek': '單位系統',
        'metryczny': '公制',
        'imperialny': '英制'
    },
    'wuu': {
        'ATEENA-ST MVP — Apparel v8.6.8': 'ATEENA-ST MVP — Apparel v8.6.8',
        'Nowa analiza': '新分析',
        'Feedback po zakupie': '购买后反馈',
        'Historia testów': '测试历史',
        '1. Dane użytkownika': '1. 用户数据',
        '2. Dane produktu': '2. 产品数据',
        'Płeć': '性别',
        'kobieta': '女性',
        'mężczyzna': '男性',
        'System jednostek': '单位系统',
        'metryczny': '公制',
        'imperialny': '英制'
    },
}

TEXT_REPLACEMENTS = {
    'en': [('Nie wykryto', 'No detected'), ('Wykryto', 'Detected'), ('Front', 'Front'), ('Profil', 'Profile'), ('Tył', 'Back'), ('Brak', 'No'), ('Tak', 'Yes')],
    'es': [('Nie wykryto', 'No se detectó'), ('Wykryto', 'Detectado'), ('Front', 'Frontal'), ('Profil', 'Perfil'), ('Tył', 'Espalda')],
    'it': [('Nie wykryto', 'Non rilevato'), ('Wykryto', 'Rilevato')],
    'fr': [('Nie wykryto', 'Non détecté'), ('Wykryto', 'Détecté')],
    'de': [('Nie wykryto', 'Nicht erkannt'), ('Wykryto', 'Erkannt')],
    'ar': [('Nie wykryto', 'لم يتم اكتشاف'), ('Wykryto', 'تم اكتشاف')],
    'ja': [('Nie wykryto', '検出されませんでした'), ('Wykryto', '検出あり')],
    'zh': [('Nie wykryto', '未检测到'), ('Wykryto', '已检测到')],
    'nl': [('Nie wykryto', 'Niet gedetecteerd'), ('Wykryto', 'Gedetecteerd')],
    'sv': [('Nie wykryto', 'Inte upptäckt'), ('Wykryto', 'Upptäckt')],
    'cs': [('Nie wykryto', 'Nebyla zjištěna'), ('Wykryto', 'Zjištěno')],
    'pt': [('Nie wykryto', 'Não detetado'), ('Wykryto', 'Detetado')],
    'ru': [('Nie wykryto', 'Не обнаружено'), ('Wykryto', 'Обнаружено')],
    'el': [('Nie wykryto', 'Δεν εντοπίστηκε'), ('Wykryto', 'Εντοπίστηκε')],
    'hi': [('Nie wykryto', 'पता नहीं चला'), ('Wykryto', 'पता चला')],
    'bn': [('Nie wykryto', 'সনাক্ত হয়নি'), ('Wykryto', 'সনাক্ত হয়েছে')],
    'ur': [('Nie wykryto', 'پتہ نہیں چلا'), ('Wykryto', 'پتہ چلا')],
    'ko': [('Nie wykryto', '감지되지 않음'), ('Wykryto', '감지됨')],
    'jv': [('Nie wykryto', 'Ora dideteksi'), ('Wykryto', 'Dideteksi')],
    'tr': [('Nie wykryto', 'Tespit edilmedi'), ('Wykryto', 'Tespit edildi')],
    'vi': [('Nie wykryto', 'Không phát hiện'), ('Wykryto', 'Đã phát hiện')],
    'ta': [('Nie wykryto', 'கண்டறியப்படவில்லை'), ('Wykryto', 'கண்டறியப்பட்டது')],
    'fa': [('Nie wykryto', 'تشخیص داده نشد'), ('Wykryto', 'تشخیص داده شد')],
    'ms': [('Nie wykryto', 'Tidak dikesan'), ('Wykryto', 'Dikesan')],
    'uk': [('Nie wykryto', 'Не виявлено'), ('Wykryto', 'Виявлено')],
    'da': [('Nie wykryto', 'Ikke registreret'), ('Wykryto', 'Registreret')],
    'no': [('Nie wykryto', 'Ikke oppdaget'), ('Wykryto', 'Oppdaget')],
    'ro': [('Nie wykryto', 'Nu a fost detectat'), ('Wykryto', 'Detectat')],
    'bg': [('Nie wykryto', 'Не е открито'), ('Wykryto', 'Открито')],
    'sr': [('Nie wykryto', 'Није откривено'), ('Wykryto', 'Откривено')],
    'hr': [('Nie wykryto', 'Nije otkriveno'), ('Wykryto', 'Otkriveno')],
    'sk': [('Nie wykryto', 'Nebolo zistené'), ('Wykryto', 'Zistené')],
    'sl': [('Nie wykryto', 'Ni zaznano'), ('Wykryto', 'Zaznano')],
}


def _browser_locale_code() -> str:
    header = ''
    try:
        headers = getattr(getattr(st, 'context', None), 'headers', None)
        if headers is not None:
            if hasattr(headers, 'get'):
                header = headers.get('accept-language', '') or headers.get('Accept-Language', '') or ''
            else:
                header = str(headers)
    except Exception:
        header = ''
    if header:
        raw = header.split(',')[0].strip().replace('_', '-').lower()
        for prefix, locale in sorted(BROWSER_LOCALE_PREFIXES.items(), key=lambda x: -len(x[0])):
            if raw.startswith(prefix):
                return locale
    return 'pl-PL'


def _selected_locale_code() -> str:
    choice = st.session_state.get('ui_language_choice', 'Auto / browser')
    code = LOCALE_CHOICES.get(choice, 'auto')
    if code == 'auto':
        return _browser_locale_code()
    return code


def _selected_lang_code() -> str:
    locale_code = _selected_locale_code()
    return LOCALE_TO_LANG.get(locale_code, locale_code.split('-')[0].lower())


def tr(text: str) -> str:
    lang = _selected_lang_code()
    if lang == 'pl':
        return text
    return UI_I18N.get(lang, {}).get(text, UI_I18N.get('en', {}).get(text, text))


def tr_text(text: str) -> str:
    lang = _selected_lang_code()
    if lang == 'pl' or not isinstance(text, str):
        return text
    exact = UI_I18N.get(lang, {}).get(text)
    if exact:
        return exact
    out = text
    for src, dst in TEXT_REPLACEMENTS.get(lang, []):
        out = out.replace(src, dst)
    return out

PRODUCT_OPTION_TRANSLATIONS = {
    'damskie':'women', 'męskie':'men', 'unisex':'unisex',
    'eleganckie':'elegant', 'casual':'casual', 'basic':'basic',
    'sukienka elegancka':'elegant dress', 'sukienka casualowa':'casual dress', 'spódniczka':'skirt', 'legginsy':'leggings',
    'żakiet':'jacket', 'marynarka':'blazer', 'spodnie eleganckie':'dress trousers', 'spodnie jeansowe':'jeans', 'spodnie casualowe':'casual trousers',
    'gorset':'corset', 'T-shirt':'T-shirt', 'bluza':'sweatshirt', 'bluza z kapturem':'hoodie', 'garnitur':'suit'
}

def tr_option(text: str) -> str:
    lang = _selected_lang_code()
    if lang == 'pl':
        return text
    base = PRODUCT_OPTION_TRANSLATIONS.get(text, text)
    return base


def adapt_product_to_selected_kind(product_result, product_kind: str):
    generic = GENERIC_CATEGORY_CHARTS.get(product_kind)
    if not generic:
        return product_result
    current = getattr(product_result, 'size_chart', {}) or {}
    if not current or getattr(product_result, 'used_fallback_chart', False):
        product_result.size_chart = generic
        return product_result
    merged = {}
    for size, row in current.items():
        base = dict(generic.get(size, {}))
        base.update(row)
        merged[size] = base
    # add missing sizes from generic only if chart is very sparse
    if len(merged) < 2:
        for size, row in generic.items():
            merged.setdefault(size, row)
    product_result.size_chart = merged
    return product_result


def cm_to_in(value_cm: float) -> float:
    return float(value_cm) / CM_PER_IN


def in_to_cm(value_in: float) -> float:
    return float(value_in) * CM_PER_IN


def kg_to_lb(value_kg: float) -> float:
    return float(value_kg) / KG_PER_LB


def lb_to_kg(value_lb: float) -> float:
    return float(value_lb) * KG_PER_LB


def format_measure(value_cm: float, unit_system: str) -> str:
    return f"{cm_to_in(value_cm):.1f} in" if unit_system == 'imperialny' else f"{value_cm:.1f} cm"


def metric_key_label(label_cm: str, unit_system: str) -> str:
    return label_cm.replace('(cm)', '(in)') if unit_system == 'imperialny' else label_cm


def input_measure(label_cm: str, base_value_cm: float, unit_system: str, disabled: bool, key: str) -> float:
    if unit_system == 'imperialny':
        value_in = st.number_input(metric_key_label(label_cm, unit_system), min_value=0.0, value=float(round(cm_to_in(base_value_cm), 1)), step=0.5, disabled=disabled, key=key)
        return round(in_to_cm(value_in), 1)
    return round(st.number_input(label_cm, min_value=0.0, value=float(round(base_value_cm, 1)), step=0.5, disabled=disabled, key=key), 1)

REQUIRED_PLACEHOLDER = '— wybierz —'
OPTIONAL_PLACEHOLDER = '— pozostaw puste —'

def req_label(label: str) -> str:
    return f"{label} *"

def render_missing_fields_panel(section_map: Dict[str, list]):
    normalized = {k: [x for x in (v or []) if x] for k, v in (section_map or {}).items()}
    total_missing = sum(len(v) for v in normalized.values())
    if total_missing == 0:
        st.success('Wszystkie wymagane pola są uzupełnione.')
        return
    st.markdown('### Brakuje Ci jeszcze tych pól')
    for section, items in normalized.items():
        if not items:
            continue
        st.markdown(f"**{section}**")
        for item in items:
            st.write(f"- {item}")
    next_missing = next((item for items in normalized.values() for item in items), None)
    if next_missing:
        st.info(f'Następne pole do uzupełnienia: {next_missing}')
        if st.button('Pokaż następne brakujące pole', key='next_missing_field_btn_v866', use_container_width=True):
            st.session_state['next_missing_field_hint_v866'] = next_missing
            st.warning(f'Uzupełnij teraz: {next_missing}')


def required_selectbox(label: str, options, key: str, format_func=None):
    values = [REQUIRED_PLACEHOLDER] + list(options)
    def _fmt(v):
        if v == REQUIRED_PLACEHOLDER:
            return REQUIRED_PLACEHOLDER
        return format_func(v) if format_func else v
    selected = st.selectbox(req_label(label), values, key=key, format_func=_fmt)
    return None if selected == REQUIRED_PLACEHOLDER else selected

def optional_selectbox(label: str, options, key: str, format_func=None):
    values = [OPTIONAL_PLACEHOLDER] + list(options)
    def _fmt(v):
        if v == OPTIONAL_PLACEHOLDER:
            return OPTIONAL_PLACEHOLDER
        return format_func(v) if format_func else v
    selected = st.selectbox(req_label(label), values, key=key, format_func=_fmt)
    return None if selected == OPTIONAL_PLACEHOLDER else selected

def parse_number_text(raw: str, integer: bool = False):
    if raw is None:
        return None
    s = str(raw).strip().replace(',', '.')
    if s == '':
        return None
    try:
        val = int(float(s)) if integer else float(s)
        return val
    except Exception:
        return None

def required_number_text_input(label: str, key: str, placeholder: str = ''):
    raw = st.text_input(req_label(label), value='', placeholder=placeholder, key=key)
    return parse_number_text(raw, integer=False)

def required_int_text_input(label: str, key: str, placeholder: str = ''):
    raw = st.text_input(req_label(label), value='', placeholder=placeholder, key=key)
    return parse_number_text(raw, integer=True)

def blank_measure_input(label_cm: str, unit_system: str, key: str, placeholder: str = '', help_text: str = '') -> Optional[float]:
    label = metric_key_label(label_cm, unit_system)
    raw = st.text_input(label, value='', placeholder=placeholder, key=key, help=help_text)
    val = parse_number_text(raw, integer=False)
    if val is None:
        return None
    if unit_system == 'imperialny':
        return round(in_to_cm(float(val)), 1)
    return round(float(val), 1)

ocr_ready, ocr_msg = ocr_engine_status()

st.set_page_config(page_title='ATEENA-ST MVP — Apparel v8.6.8', page_icon='👗', layout='wide')
init_db()

APP_VERSION = '8.6.8'
BUILD_ID = datetime.now().strftime('%Y%m%d-%H%M%S')
BUILD_LABEL = f'ATEENA {APP_VERSION} | build {BUILD_ID}'


def _normalize_uploaded_image(uploaded_file) -> tuple[bytes, str, Dict]:
    raw = uploaded_file.getvalue()
    name = getattr(uploaded_file, 'name', None) or 'upload.jpg'
    ext = Path(name).suffix.lower()
    meta: Dict = {
        'original_name': name,
        'original_ext': ext or '',
        'normalized_ext': ext or '.jpg',
        'original_size_bytes': len(raw),
        'normalized_size_bytes': len(raw),
        'width_px': None,
        'height_px': None,
        'exif_fixed': False,
        'converted': False,
        'payload_ok': False,
        'error': '',
    }
    if ext in {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif'}:
        try:
            img = Image.open(io.BytesIO(raw))
            w0, h0 = img.size
            img = ImageOps.exif_transpose(img)
            meta['exif_fixed'] = True
            if img.mode != 'RGB':
                img = img.convert('RGB')
            out = io.BytesIO()
            img.save(out, format='JPEG', quality=95)
            normalized = out.getvalue()
            meta['normalized_ext'] = '.jpg'
            meta['normalized_size_bytes'] = len(normalized)
            meta['width_px'], meta['height_px'] = img.size
            meta['converted'] = ext in {'.png', '.webp', '.heic', '.heif'} or ext == '.jpeg'
            meta['payload_ok'] = True
            return normalized, '.jpg', meta
        except Exception as exc:
            meta['error'] = str(exc)
            meta['payload_ok'] = len(raw) > 0
            return raw, ext or '.jpg', meta
    meta['payload_ok'] = len(raw) > 0
    return raw, ext or '.jpg', meta


def _image_meta_from_bytes(image_bytes: Optional[bytes], fallback_name: str = '') -> Dict:
    meta = {
        'original_name': fallback_name,
        'original_ext': Path(fallback_name).suffix.lower() if fallback_name else '',
        'normalized_ext': '.jpg',
        'original_size_bytes': len(image_bytes or b''),
        'normalized_size_bytes': len(image_bytes or b''),
        'width_px': None,
        'height_px': None,
        'exif_fixed': True,
        'converted': True,
        'payload_ok': bool(image_bytes),
        'error': '',
    }
    if image_bytes:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            meta['width_px'], meta['height_px'] = img.size
        except Exception as exc:
            meta['error'] = str(exc)
    return meta


def save_upload(uploaded_file, prefix: str) -> Optional[str]:
    if uploaded_file is None:
        return None
    normalized, ext, _meta = _normalize_uploaded_image(uploaded_file)
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}{ext or '.jpg'}"
    dest = UPLOADS_DIR / filename
    dest.write_bytes(normalized)
    return str(dest)


def image_to_bytes(uploaded_file) -> bytes:
    normalized, _ext, _meta = _normalize_uploaded_image(uploaded_file)
    return normalized


def persist_uploaded_slot(slot_name: str, uploaded_file) -> Optional[bytes]:
    payload_key = f"upload_slot_{slot_name}_bytes"
    name_key = f"upload_slot_{slot_name}_name"
    meta_key = f"upload_slot_{slot_name}_meta"
    if uploaded_file is not None:
        try:
            normalized, _ext, meta = _normalize_uploaded_image(uploaded_file)
            st.session_state[payload_key] = normalized
            st.session_state[name_key] = getattr(uploaded_file, 'name', slot_name)
            meta['payload_ok'] = bool(normalized)
            st.session_state[meta_key] = meta
        except Exception as exc:
            st.session_state[payload_key] = None
            st.session_state[name_key] = ''
            st.session_state[meta_key] = {'payload_ok': False, 'error': str(exc)}
    return st.session_state.get(payload_key)


def get_persisted_slot(slot_name: str) -> Optional[bytes]:
    return st.session_state.get(f"upload_slot_{slot_name}_bytes")


def get_persisted_slot_meta(slot_name: str) -> Dict:
    return st.session_state.get(f"upload_slot_{slot_name}_meta", {}) or {}


def clear_persisted_slot(slot_name: str):
    st.session_state.pop(f"upload_slot_{slot_name}_bytes", None)
    st.session_state.pop(f"upload_slot_{slot_name}_name", None)
    st.session_state.pop(f"upload_slot_{slot_name}_meta", None)


def clear_all_upload_slots():
    for slot_name in ['front','profile_left','profile_right','back']:
        clear_persisted_slot(slot_name)


def build_debug_bundle(slot_payloads: Dict[str, Optional[bytes]], slot_meta: Dict[str, Dict], body_result: Optional[BodyAnalysisResult]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('build_info.json', json.dumps({'app_version': APP_VERSION, 'build_id': BUILD_ID}, ensure_ascii=False, indent=2))
        zf.writestr('slot_meta.json', json.dumps(slot_meta, ensure_ascii=False, indent=2))
        if body_result is not None:
            zf.writestr('body_result.json', json.dumps(body_result.to_dict(), ensure_ascii=False, indent=2))
        for slot, payload in slot_payloads.items():
            if payload:
                zf.writestr(f'uploads/{slot}.jpg', payload)
    mem.seek(0)
    return mem.getvalue()


def save_upload_bytes(image_bytes: Optional[bytes], prefix: str, ext: str = '.jpg') -> Optional[str]:
    if not image_bytes:
        return None
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}{ext}"
    dest = UPLOADS_DIR / filename
    dest.write_bytes(image_bytes)
    return str(dest)


def load_demo_df() -> pd.DataFrame:
    return pd.DataFrame(SAMPLE_PRODUCTS)


def build_demo_product(row: dict) -> ProductProfile:
    return ProductProfile(
        brand=row['brand'],
        name=row['name'],
        source_url=row['product_url'],
        image_url=str(APP_DIR / row['image_path']),
        dress_type=row['dress_type'],
        fit_type=row['fit_type'],
        stretch_level=row['stretch_level'],
        length_type=row['length_type'],
        style_effect=row['style_effect'],
        tight_areas=row['tight_areas'],
        runs_small=float(row['runs_small']),
        runs_large=float(row['runs_large']),
        true_to_size=float(row['true_to_size']),
        review_count=len(row.get('review_texts', [])),
        review_lines=row.get('review_texts', []),
        size_chart=row['size_chart'],
        used_fallback_chart=False,
        parsing_notes=['To dane z katalogu demo.'],
    )



def build_db_product(row: dict) -> ProductProfile:
    return ProductProfile(
        brand=row.get('brand', 'Unknown brand'),
        name=row.get('name', 'Unknown product'),
        source_url=row.get('source_url', ''),
        image_url=row.get('image_url'),
        dress_type=row.get('dress_type', row.get('product_kind', 'odzież')),
        fit_type=row.get('fit_type', 'regular'),
        stretch_level=row.get('stretch_level', 'medium'),
        length_type=row.get('length_type', 'regular'),
        style_effect=row.get('style_effect', 'neutral'),
        tight_areas=row.get('tight_areas', []),
        runs_small=float(row.get('runs_small') or 0.0),
        runs_large=float(row.get('runs_large') or 0.0),
        true_to_size=float(row.get('true_to_size') or 0.5),
        review_count=int(row.get('review_count') or 0),
        review_lines=row.get('review_lines', []),
        size_chart=row.get('size_chart', {}),
        used_fallback_chart=bool(row.get('used_fallback_chart')),
        parsing_notes=row.get('parsing_notes', []),
    )


def import_capture_mobile_json(json_bytes: bytes) -> dict:
    data = json.loads(json_bytes.decode('utf-8'))
    settings = data.get('settings', {})
    captures = data.get('captures', {})
    front = captures.get('front') or {}
    profile = captures.get('profile') or {}
    back = captures.get('back') or {}

    def _extract_bytes(obj):
        image = obj.get('image')
        if not image or not isinstance(image, str) or ',' not in image:
            return None
        import base64
        try:
            return base64.b64decode(image.split(',', 1)[1])
        except Exception:
            return None

    front_bytes = _extract_bytes(front)
    profile_bytes = _extract_bytes(profile)
    back_bytes = _extract_bytes(back)
    front_eval = ((front.get('meta') or {}).get('evaluation') or {})
    profile_eval = ((profile.get('meta') or {}).get('evaluation') or {})
    back_eval = ((back.get('meta') or {}).get('evaluation') or {})
    accepted = bool(front_eval.get('status') == 'ACCEPT' and profile_eval.get('status') == 'ACCEPT' and not front_eval.get('hardReject') and not profile_eval.get('hardReject'))
    return {
        'source': 'capture_mobile_pilot_json',
        'accepted': accepted,
        'measurement_ready': accepted,
        'posture_ready': accepted,
        'capture_method': settings.get('captureRole', 'druga_osoba'),
        'photo_clothing_fit': settings.get('clothingFit', 'unknown'),
        'gender': settings.get('gender', 'kobieta'),
        'height_cm': settings.get('height'),
        'weight_kg': settings.get('weight'),
        'age': settings.get('age'),
        'front_bytes': front_bytes,
        'profile_bytes': profile_bytes,
        'back_bytes': back_bytes,
        'body': {
            'front_capture': {
                'status': front_eval.get('status', '—'),
                'status_code': str(front_eval.get('status', '')).lower() if front_eval.get('status') else 'unknown',
                'score': float(front_eval.get('score', 0)) / 100 if float(front_eval.get('score', 0)) > 1 else float(front_eval.get('score', 0)),
                'detected_orientation': front_eval.get('orientation'),
                'measurement_ready': accepted,
                'posture_ready': accepted,
                'accept_ready': accepted,
                'blockers': front_eval.get('messages', []),
            },
            'profile_capture': {
                'status': profile_eval.get('status', '—'),
                'status_code': str(profile_eval.get('status', '')).lower() if profile_eval.get('status') else 'unknown',
                'score': float(profile_eval.get('score', 0)) / 100 if float(profile_eval.get('score', 0)) > 1 else float(profile_eval.get('score', 0)),
                'detected_orientation': profile_eval.get('orientation'),
                'measurement_ready': accepted,
                'posture_ready': accepted,
                'accept_ready': accepted,
                'blockers': profile_eval.get('messages', []),
            },
            'back_capture': {
                'status': back_eval.get('status', '—') if back else '—',
                'status_code': str(back_eval.get('status', '')).lower() if back_eval.get('status') else 'unknown',
                'score': float(back_eval.get('score', 0)) / 100 if float(back_eval.get('score', 0)) > 1 else float(back_eval.get('score', 0)),
                'detected_orientation': back_eval.get('orientation'),
                'measurement_ready': bool(back and accepted),
                'posture_ready': bool(back and accepted),
                'accept_ready': bool(back and accepted),
                'blockers': back_eval.get('messages', []),
            },
            'notes': ['Sesja zaimportowana z Capture Mobile Studio JSON.'],
        },
    }




def mobile_session_quality_summary(imported: dict) -> dict:
    body = imported.get('body') or {}
    scores = []
    ready = {'front': False, 'profile': False, 'back': False}
    for key in ['front_capture', 'profile_capture', 'back_capture']:
        cap = body.get(key) or {}
        if cap:
            try:
                scores.append(float(cap.get('score', 0)))
            except Exception:
                pass
            if key == 'front_capture':
                ready['front'] = bool(cap.get('measurement_ready') or cap.get('accept_ready'))
            elif key == 'profile_capture':
                ready['profile'] = bool(cap.get('measurement_ready') or cap.get('accept_ready'))
            elif key == 'back_capture':
                ready['back'] = bool(cap.get('posture_ready') or cap.get('accept_ready'))
    score = int(round((sum(scores) / max(len(scores),1)) * 100)) if scores and max(scores) <= 1 else int(round(sum(scores) / max(len(scores),1))) if scores else 0
    return {'score': score, 'ready': ready}

def render_demo_catalog():
    demo_df = load_demo_df()
    selected = None
    st.markdown('### Katalog demo')
    rows = [demo_df.iloc[i:i + 2] for i in range(0, len(demo_df), 2)]
    for chunk in rows:
        cols = st.columns(len(chunk))
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                st.image(str(APP_DIR / row['image_path']), use_container_width=True)
                st.markdown(f"**{row['brand']} — {row['name']}**")
                st.caption(f"{row['dress_type']} • {row['fit_type']} • {row['stretch_level']} • {row['length_type']}")
                st.write(f"Link demo: {row['product_url']}")
                if st.button(f"Wybierz: {row['name']}", key=f"demo_{row['name']}", use_container_width=True):
                    selected = row.to_dict()
    return selected


def render_capture_status(title: str, capture: dict):
    if not capture:
        return
    status = capture.get('status', '')
    status_code = capture.get('status_code', '')
    score = int(float(capture.get('score', 0)) * 100)
    if status_code == 'accept':
        st.success(f"{title}: {status} ({score}%)")
    elif status_code == 'retry':
        st.warning(f"{title}: {status} ({score}%)")
    else:
        st.error(f"{title}: {status} ({score}%)")
    meta = []
    if capture.get('detected_orientation'):
        meta.append(f"orientacja: {capture.get('detected_orientation')}")
    if capture.get('camera_height_hint'):
        meta.append(f"kamera: {capture.get('camera_height_hint')}")
    if capture.get('camera_angle_hint'):
        meta.append(f"kąt: {capture.get('camera_angle_hint')}")
    if capture.get('roll_deg') is not None:
        meta.append(f"roll: {capture.get('roll_deg')}°")
    if capture.get('orientation_confidence') is not None:
        meta.append(f"pewność orientacji: {int(float(capture.get('orientation_confidence', 0))*100)}%")
    if capture.get('selfie_risk'):
        meta.append('ryzyko selfie-perspective')
    if capture.get('vision_clothing_fit'):
        meta.append(f"strój(wizja): {capture.get('vision_clothing_fit')}")
    if capture.get('distance_hint'):
        meta.append(f"dystans: {capture.get('distance_hint')}")
    if capture.get('camera_pitch_hint'):
        meta.append(f"perspektywa: {capture.get('camera_pitch_hint')}")
    if capture.get('background_cleanup_score') is not None:
        meta.append(f"cleanup tła: {int(float(capture.get('background_cleanup_score', 0))*100)}%")
    if meta:
        st.caption(' | '.join(meta))
    if capture.get('landmark_visibility_score') is not None:
        st.write(f"- Jakość landmarków: {int(float(capture.get('landmark_visibility_score', 0))*100)}%")
    if capture.get('hand_position_hint'):
        st.write(f"- Ułożenie rąk: {capture.get('hand_position_hint')}")
    missing = capture.get('missing_points') or []
    if missing:
        st.write('- Brak landmarków: ' + ', '.join(missing[:8]) + ('...' if len(missing) > 8 else ''))
    for msg in capture.get('messages', [])[:5]:
        st.write(f"- {msg}")



def render_capture_gate_summary(capture: dict, label: str = ''):
    if not capture:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric(f'{label} accept_ready', 'tak' if capture.get('accept_ready') else 'nie')
    c2.metric(f'{label} measurement_ready', 'tak' if capture.get('measurement_ready') else 'nie')
    c3.metric(f'{label} posture_ready', 'tak' if capture.get('posture_ready') else 'nie')
    blockers = capture.get('blockers') or []
    if capture.get('background_cleanup_applied'):
        st.caption('Oczyszczanie tła zostało zastosowane przed analizą tej klatki.')
    if blockers:
        st.caption('Blokery jakości: ' + '; '.join(blockers[:6]))




def render_landmark_schema_reference():
    st.subheader('ATEENA Body Landmark Schema v1')
    rows = schema_rows_for_ui()
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption('Schema v1 integruje centralne partie FRONT, kończyny L/R, profil głębokości i tył dla symetrii / postawy.')

def render_landmark_segment_report(body_result: BodyAnalysisResult):
    st.subheader('Status segmentów landmarków')
    report = body_result.landmark_segment_report or {}
    tabs = []
    labels = []
    for key, label in [('front','FRONT'), ('profile','PROFIL'), ('back','TYŁ')]:
        if report.get(key):
            labels.append(label)
            tabs.append(report.get(key))
    if not tabs:
        st.info('Brak raportu segmentów.')
        return
    st_tabs = st.tabs(labels)
    for tab, key, label in zip(st_tabs, [k for k in ['front','profile','back'] if report.get(k)], labels):
        with tab:
            df = pd.DataFrame(report[key])
            if not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)


def render_background_cleanup_preview(body_result: BodyAnalysisResult):
    if not body_result:
        return
    imgs = []
    caps = []
    if body_result.front_measure_overlay_image is not None:
        imgs.append(body_result.front_measure_overlay_image[:, :, ::-1])
        caps.append('Front — po oczyszczeniu tła + miejsca odczytu pomiarów')
    elif body_result.front_cleaned_image is not None:
        imgs.append(body_result.front_cleaned_image[:, :, ::-1])
        caps.append('Front — po oczyszczeniu tła')
    if body_result.profile_measure_overlay_image is not None:
        imgs.append(body_result.profile_measure_overlay_image[:, :, ::-1])
        caps.append('Profil lewy — po oczyszczeniu tła + miejsca odczytu pomiarów')
    elif body_result.profile_cleaned_image is not None:
        imgs.append(body_result.profile_cleaned_image[:, :, ::-1])
        caps.append('Profil lewy — po oczyszczeniu tła')
    if getattr(body_result, 'right_profile_measure_overlay_image', None) is not None:
        imgs.append(body_result.right_profile_measure_overlay_image[:, :, ::-1])
        caps.append('Profil prawy — po oczyszczeniu tła + miejsca odczytu pomiarów')
    elif getattr(body_result, 'right_profile_cleaned_image', None) is not None:
        imgs.append(body_result.right_profile_cleaned_image[:, :, ::-1])
        caps.append('Profil prawy — po oczyszczeniu tła')
    if body_result.back_measure_overlay_image is not None:
        imgs.append(body_result.back_measure_overlay_image[:, :, ::-1])
        caps.append('Tył — po oczyszczeniu tła + miejsca odczytu pomiarów')
    elif body_result.back_cleaned_image is not None:
        imgs.append(body_result.back_cleaned_image[:, :, ::-1])
        caps.append('Tył — po oczyszczeniu tła')
    if not imgs:
        return
    st.markdown('### Podgląd po oczyszczeniu tła')
    st.caption('Na obrazach zaznaczone są miejsca, z których aplikacja odczytała albo oszacowała pomiary. Wszystkie wartości zapisują się w cm. W 8.6.3 możesz też łatwiej debugować brakujące sloty i zdjęcia z iPhone (HEIC/HEIF).')
    cols = st.columns(len(imgs))
    for col, img, cap in zip(cols, imgs, caps):
        with col:
            st.image(img, caption=cap, use_container_width=True)


def render_upload_slot_status(label: str, uploaded_file, image_bytes: Optional[bytes] = None, slot_name: Optional[str] = None, diagnostic_mode: bool = False):
    persisted = get_persisted_slot(slot_name) if slot_name else None
    preview_bytes = image_bytes or persisted
    meta = get_persisted_slot_meta(slot_name) if slot_name else {}
    if not preview_bytes and uploaded_file is not None:
        try:
            normalized, _ext, uploaded_meta = _normalize_uploaded_image(uploaded_file)
            preview_bytes = normalized
            meta = uploaded_meta
        except Exception as exc:
            preview_bytes = None
            meta = {'payload_ok': False, 'error': str(exc)}
    has_file = bool(preview_bytes)
    if has_file:
        st.success(f'{label}: dodane')
        try:
            st.image(preview_bytes, caption=label, use_container_width=True)
        except Exception:
            st.info(f'{label}: plik dodany, ale podgląd nie jest dostępny.')
        rows = []
        rows.append({'pole': 'plik', 'wartość': meta.get('original_name') or (st.session_state.get(f"upload_slot_{slot_name}_name") if slot_name else '') or '—'})
        rows.append({'pole': 'format', 'wartość': f"{meta.get('original_ext', '—')} → {meta.get('normalized_ext', '—')}"})
        rows.append({'pole': 'rozmiar', 'wartość': f"{round(float(meta.get('normalized_size_bytes', len(preview_bytes or b'')))/1024,1)} KB"})
        dims = f"{meta.get('width_px') or '—'} × {meta.get('height_px') or '—'} px"
        rows.append({'pole': 'wymiary', 'wartość': dims})
        rows.append({'pole': 'EXIF naprawiony', 'wartość': 'tak' if meta.get('exif_fixed') else 'nie'})
        rows.append({'pole': 'payload OK', 'wartość': 'tak' if meta.get('payload_ok', bool(preview_bytes)) else 'nie'})
        if diagnostic_mode:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if meta.get('error'):
                st.caption('debug: ' + str(meta.get('error')))
        else:
            st.caption(' | '.join([f"{r['pole']}: {r['wartość']}" for r in rows]))
        if slot_name and st.button(f'Wyczyść {label}', key=f'clear_{slot_name}', use_container_width=True):
            clear_persisted_slot(slot_name)
            st.rerun()
    else:
        st.warning(f'{label}: brak')
        if diagnostic_mode and meta:
            st.json(meta)

WEAK_POINT_LABELS = {
    'bust': 'Biust / klatka piersiowa',
    'waist': 'Talia',
    'hips': 'Biodra',
    'abdomen_cm': 'Brzuch',
    'thigh_cm': 'Udo',
    'arm_biceps_cm': 'Ramię / biceps',
    'neck_cm': 'Szyja',
    'wrist_cm': 'Nadgarstek',
    'calf_max_cm': 'Łydka max',
    'calf_min_cm': 'Łydka min',
}


def weak_label(key: str) -> str:
    return WEAK_POINT_LABELS.get(key, key)


def render_live_capture_overlay_helper(gender: str):
    html = f"""
    <div style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif'>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>
        <strong>Live capture overlay — beta helper</strong>
        <span style='font-size:12px;background:#ccfbf1;color:#115e59;padding:4px 8px;border-radius:999px'>kamera live</span>
      </div>
      <div style='font-size:13px;color:#475569;margin-bottom:10px'>Ten helper nie zapisuje zdjęcia do aplikacji — pomaga ustawić sylwetkę przed użyciem pola Aparat. Ustaw całe ciało, stopy, ręce i wysokość kamery.</div>
      <video id='v63b_video' playsinline autoplay muted style='width:100%;max-height:420px;background:#0f172a;border-radius:16px;object-fit:cover'></video>
      <canvas id='v63b_canvas' style='position:relative;margin-top:-420px;width:100%;max-height:420px;border-radius:16px;pointer-events:none'></canvas>
      <div id='v63b_status' style='margin-top:12px;font-size:13px;padding:10px 12px;border-radius:12px;background:#f8fafc;border:1px solid #e2e8f0;color:#0f172a'>Kliknij poniżej i pozwól na dostęp do kamery.</div>
      <button id='v63b_start' style='margin-top:10px;width:100%;padding:12px 14px;border:none;border-radius:12px;background:#14b8a6;color:#06231f;font-weight:800'>Uruchom live overlay</button>
      <script type='module'>
        const gender = {json.dumps(gender)};
        const statusEl = document.getElementById('v63b_status');
        const video = document.getElementById('v63b_video');
        const canvas = document.getElementById('v63b_canvas');
        const ctx = canvas.getContext('2d');
        let poseLandmarker = null, stream = null;
        function setStatus(txt) {{ statusEl.textContent = txt; }}
        document.getElementById('v63b_start').onclick = async () => {{
          try {{
            const {{ PoseLandmarker, FilesetResolver }} = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14');
            const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
            poseLandmarker = await PoseLandmarker.createFromOptions(vision, {{
              baseOptions: {{ modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task' }},
              runningMode: 'VIDEO', numPoses: 1
            }});
            stream = await navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: 'environment' }}, audio:false }});
            video.srcObject = stream; await video.play();
            setStatus('Ustaw FRONT albo PROFIL: głowa i stopy w kadrze, ręce lekko odsunięte, telefon na wysokości bioder.');
            const tick = () => {{
              if (!poseLandmarker || video.readyState < 2) return requestAnimationFrame(tick);
              canvas.width = video.clientWidth; canvas.height = video.clientHeight;
              const res = poseLandmarker.detectForVideo(video, performance.now());
              ctx.clearRect(0,0,canvas.width,canvas.height);
              ctx.strokeStyle = 'rgba(20,184,166,.95)'; ctx.lineWidth = 3; ctx.setLineDash([8,6]);
              ctx.strokeRect(canvas.width*0.18, canvas.height*0.05, canvas.width*0.64, canvas.height*0.88);
              ctx.setLineDash([]);
              const lm = res.landmarks?.[0];
              if (lm) {{
                const draw = (a,b) => {{ ctx.beginPath(); ctx.moveTo(lm[a].x*canvas.width,lm[a].y*canvas.height); ctx.lineTo(lm[b].x*canvas.width,lm[b].y*canvas.height); ctx.stroke(); }};
                [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[27,31],[24,26],[26,28],[28,32]].forEach(p=>{{ if(lm[p[0]]&&lm[p[1]]) draw(p[0],p[1]); }});
                const feet = (lm[31]?.visibility||0)>0.4 && (lm[32]?.visibility||0)>0.4;
                const hands = (lm[15]?.visibility||0)>0.3 && (lm[16]?.visibility||0)>0.3;
                const head = (lm[0]?.visibility||0)>0.4;
                if (head && feet && hands) setStatus('Wygląda dobrze: sprawdź jeszcze, czy to pełny front albo pełny profil i zrób zdjęcie w polu Aparat poniżej.');
                else setStatus('Pokaż pełne ciało: głowę, obie dłonie i obie stopy.');
              }} else {{ setStatus('Nie wykryto sylwetki — stań na prostym tle i pokaż całe ciało.'); }}
              requestAnimationFrame(tick);
            }};
            requestAnimationFrame(tick);
          }} catch(e) {{ setStatus('Nie udało się uruchomić live overlay: ' + e.message); }}
        }};
      </script>
    </div>
    """
    components.html(html, height=620)



def confidence_band(score: float) -> str:
    if score >= 0.80:
        return 'wysoki'
    if score >= 0.62:
        return 'średni'
    return 'niski'

def confidence_action(score: float) -> str:
    if score >= 0.80:
        return 'OK'
    if score >= 0.62:
        return 'warto obserwować'
    return 'potwierdź ręcznie'

def render_capture_action_plan(capture: dict, label: str):
    if not capture:
        return
    blockers = capture.get('blockers') or []
    if not blockers:
        st.success(f'{label}: nie ma krytycznych blockerów quality gate.')
        return
    st.warning(f'{label}: co poprawić przed kolejnym ujęciem')
    for item in blockers[:5]:
        st.write(f'- {item}')


def measurement_confidence_reasons(body_result: BodyAnalysisResult) -> Dict[str, str]:
    reasons: Dict[str, str] = {}
    front = body_result.front_capture or {}
    profile = body_result.profile_capture or {}
    back = body_result.back_capture or {}
    for key, score in body_result.measurement_confidence.items():
        if score >= 0.80:
            reasons[key] = 'wysoki confidence: komplet kluczowych ujęć i dobra jakość landmarków'
        elif score >= 0.62:
            reasons[key] = 'średni confidence: wynik używalny, ale warto obserwować jakość ujęcia i stroju'
        else:
            bits = []
            if front.get('selfie_risk') or profile.get('selfie_risk'):
                bits.append('ryzyko selfie')
            if (front.get('vision_clothing_fit') == 'loose') or (profile.get('vision_clothing_fit') == 'loose'):
                bits.append('luźny strój')
            if not back or back.get('status_code') not in {'accept', 'ok'}:
                bits.append('brak lub słaby TYŁ')
            if front.get('status_code') != 'accept' or profile.get('status_code') != 'accept':
                bits.append('graniczny capture')
            if float(front.get('background_cleanup_score', 1)) < 0.46 or float(profile.get('background_cleanup_score', 1)) < 0.46:
                bits.append('tło nadal utrudnia odcięcie postaci')
            reasons[key] = 'niski confidence: ' + (', '.join(bits) if bits else 'wymaga ręcznego potwierdzenia')
    return reasons

def product_confidence_score(product_result) -> float:
    score = 55.0
    if product_result is None:
        return score
    if not getattr(product_result, 'used_fallback_chart', False):
        score += 15
    if getattr(product_result, 'review_count', 0) >= 3:
        score += 10
    if len(getattr(product_result, 'size_chart', {}) or {}) >= 3:
        score += 10
    if getattr(product_result, 'source_url', None):
        score += 5
    return max(0.0, min(100.0, score))

def postural_fit_score(posture_summary: Dict, visual_fit: Dict) -> float:
    if not posture_summary or not posture_summary.get('available'):
        return 55.0
    if not posture_summary.get('detected'):
        return 92.0
    status = (visual_fit or {}).get('status')
    if status == 'helps':
        return 86.0
    if status == 'risk':
        return 52.0
    return 68.0


def source_mode_label(body_result: BodyAnalysisResult) -> str:
    source = (body_result.measurement_source or '').lower()
    if source.startswith('pose_landmarks_plus_prior'):
        return 'photo_based_estimate'
    if source.startswith('anthropometric_prior'):
        return 'fallback_prior_only'
    return 'unknown'

def render_runtime_self_test():
    st.markdown('### Runtime self-test')
    if HAS_MEDIAPIPE:
        st.success('MediaPipe: OK — vision pipeline dostępny.')
    else:
        st.error('MediaPipe: MISSING — vision pipeline disabled, fallback prior only.')

def render_measurement_source_status(body_result: BodyAnalysisResult):
    mode = source_mode_label(body_result)
    if mode == 'photo_based_estimate':
        st.success('Źródło wyniku: photo-based estimate — wynik oparty na analizie zdjęć + priors + calibration.')
    elif mode == 'fallback_prior_only':
        st.error('Źródło wyniku: fallback prior only — wynik NIE pochodzi z pełnej analizy zdjęć i nie powinien być używany jako właściwa rekomendacja.')
    else:
        st.warning(f"Źródło wyniku: {body_result.measurement_source}")

def render_confidence_reasons(body_result: BodyAnalysisResult):
    reasons = measurement_confidence_reasons(body_result)
    if not reasons:
        return
    st.markdown('**Confidence reason per part**')
    rows = []
    label_map = {'bust':'Biust / klatka','waist':'Talia','hips':'Biodra'}
    for key, val in body_result.measurement_confidence.items():
        rows.append({
            'strefa': label_map.get(key, key.replace('_cm','').replace('_',' ')),
            'powód': reasons.get(key, '—'),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render_measurement_confidence_table(body_result: BodyAnalysisResult):
    rows = []
    rows.append({'strefa': 'Biust / klatka', 'wartość': body_result.suggested_bust_cm, 'confidence': int(body_result.measurement_confidence.get('bust', 0) * 100), 'pasmo': confidence_band(body_result.measurement_confidence.get('bust', 0)), 'zalecenie': confidence_action(body_result.measurement_confidence.get('bust', 0))})
    rows.append({'strefa': 'Talia', 'wartość': body_result.suggested_waist_cm, 'confidence': int(body_result.measurement_confidence.get('waist', 0) * 100), 'pasmo': confidence_band(body_result.measurement_confidence.get('waist', 0)), 'zalecenie': confidence_action(body_result.measurement_confidence.get('waist', 0))})
    rows.append({'strefa': 'Biodra', 'wartość': body_result.suggested_hips_cm, 'confidence': int(body_result.measurement_confidence.get('hips', 0) * 100), 'pasmo': confidence_band(body_result.measurement_confidence.get('hips', 0)), 'zalecenie': confidence_action(body_result.measurement_confidence.get('hips', 0))})
    for key, label in EXTRA_MEASURE_LABELS:
        if key in body_result.extra_estimates:
            score = body_result.measurement_confidence.get(key, 0)
            rows.append({'strefa': label.replace(' (cm)', ''), 'wartość': body_result.extra_estimates[key], 'confidence': int(score * 100), 'pasmo': confidence_band(score), 'zalecenie': confidence_action(score)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_sanity_report(body_result: BodyAnalysisResult):
    report = body_result.sanity_report or {}
    st.subheader(tr_text('Sanity engine'))
    if report.get('ok', False):
        st.success(tr_text(report.get('summary', 'Brak istotnych niespójności pomiarowych.')))
        return
    st.warning(tr_text(report.get('summary', 'Wykryto niespójności pomiarowe.')))
    for flag in report.get('flags', []):
        st.write(f"- {flag.get('label')}: {flag.get('message')}")


def manual_body_result(height_cm: float, weight_kg: float, age: int, gender: str, bust_cm: float, waist_cm: float, hips_cm: float) -> BodyAnalysisResult:
    body_type = classify_body_type(float(bust_cm), float(waist_cm), float(hips_cm))
    build_type = derive_build_type(float(height_cm), float(weight_kg))
    notes = [
        'Użytkownik wybrał tryb ręcznego podania wymiarów.',
        'Aplikacja nie estymowała wymiarów ze zdjęcia w tym przebiegu.',
        'Dla najwyższej jakości ręcznych pomiarów: miarka powinna przylegać do ciała, ale nie uciskać; mięśnie rozluźnione; oddech naturalny.',
    ]
    return BodyAnalysisResult(
        body_type=body_type,
        suggested_bust_cm=round(float(bust_cm), 1),
        suggested_waist_cm=round(float(waist_cm), 1),
        suggested_hips_cm=round(float(hips_cm), 1),
        raw_bust_cm=round(float(bust_cm), 1),
        raw_waist_cm=round(float(waist_cm), 1),
        raw_hips_cm=round(float(hips_cm), 1),
        build_type=build_type,
        confidence=0.96,
        notes=notes,
        measurement_source=f'manual_entry_{gender}',
        front_capture={'status': 'pominięto', 'status_code': 'accept', 'score': 1.0, 'messages': ['Tryb ręczny — bez zdjęcia FRONT.'], 'checks': {}},
        profile_capture={'status': 'pominięto', 'status_code': 'accept', 'score': 1.0, 'messages': ['Tryb ręczny — bez zdjęcia PROFIL.'], 'checks': {}},
        back_capture={'status': 'pominięto', 'status_code': 'accept', 'score': 1.0, 'messages': ['Tryb ręczny — bez zdjęcia TYŁ.'], 'checks': {}},
        calibration_info={'applied': False, 'used_scopes': [], 'offsets_cm': {'bust': 0.0, 'waist': 0.0, 'hips': 0.0}, 'sample_count': 0.0},
        measurement_confidence={'bust': 0.99, 'waist': 0.99, 'hips': 0.99, 'abdomen_cm': 0.99, 'thigh_cm': 0.99, 'arm_biceps_cm': 0.99, 'neck_cm': 0.99, 'wrist_cm': 0.99, 'calf_max_cm': 0.99, 'calf_min_cm': 0.99},
        extra_estimates={},
        weak_points=[],
        sanity_report={'ok': True, 'flags': [], 'weak_points': [], 'summary': 'Brak istotnych niespójności pomiarowych.'},
    )


def clean_extra_measurements(extra: Dict[str, float], gender: str, primary_bust_or_chest: float, primary_waist: float, primary_hips: float) -> Dict[str, float]:
    cleaned = {k: round(float(v), 1) for k, v in extra.items() if float(v) > 0}
    cleaned['gender'] = gender
    cleaned['bust_or_chest_primary_cm'] = round(float(primary_bust_or_chest), 1)
    cleaned['waist_primary_cm'] = round(float(primary_waist), 1)
    cleaned['hips_primary_cm'] = round(float(primary_hips), 1)
    return cleaned

def apply_part_calibration_to_body_result(body_result: BodyAnalysisResult, *, gender: str, product_kind: str, clothing_fit: str):
    photo_bucket = photo_quality_bucket(float(body_result.confidence), body_result.front_capture, body_result.profile_capture)
    fit_bucket = normalize_clothing_fit_bucket(clothing_fit)
    part_profile = get_calibration_part_offsets(
        gender=gender,
        photo_quality_bucket=photo_bucket,
        product_kind=product_kind,
        clothing_fit_bucket=fit_bucket,
    )
    if not part_profile.get('found'):
        return body_result

    ranges = {
        'abdomen_cm': (55.0, 180.0),
        'thigh_cm': (38.0, 100.0),
        'arm_biceps_cm': (18.0, 70.0),
        'neck_cm': (24.0, 65.0),
        'wrist_cm': (10.0, 28.0),
        'calf_max_cm': (22.0, 65.0),
        'calf_min_cm': (16.0, 50.0),
        'chest_cm': (60.0, 180.0),
    }
    part_info = {
        'scope': part_profile.get('scope'),
        'offsets': {},
        'stats': part_profile.get('stats', {}),
    }
    for key, offset in (part_profile.get('offsets') or {}).items():
        if key in body_result.extra_estimates:
            low, high = ranges.get(key, (0.0, 999.0))
            corrected = max(low, min(high, float(body_result.extra_estimates[key]) + float(offset)))
            body_result.extra_estimates[key] = round(corrected, 1)
            part_info['offsets'][key] = float(offset)
            if key in body_result.measurement_confidence:
                body_result.measurement_confidence[key] = round(min(0.96, float(body_result.measurement_confidence[key]) + 0.03), 2)
    existing = body_result.calibration_info or {}
    existing['part_calibration'] = part_info
    body_result.calibration_info = existing
    return body_result


def editable_chart_json(product_result) -> str:
    try:
        return json.dumps(getattr(product_result, 'size_chart', {}) or {}, ensure_ascii=False, indent=2)
    except Exception:
        return "{}"



def extra_measurements_summary(extra: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in extra.items() if isinstance(v, (int, float)) and v > 0}


def apply_calibration_to_result(body_result: BodyAnalysisResult, gender: str, clothing_fit: str, product_kind: str) -> BodyAnalysisResult:
    correction = get_calibration_correction(
        raw_measures={
            'bust': float(body_result.raw_bust_cm),
            'waist': float(body_result.raw_waist_cm),
            'hips': float(body_result.raw_hips_cm),
        },
        gender=gender,
        confidence=float(body_result.confidence),
        front_capture=body_result.front_capture,
        profile_capture=body_result.profile_capture,
        product_kind=product_kind,
        clothing_fit=clothing_fit,
    )
    body_result.calibration_info = correction.to_dict()
    if correction.applied:
        body_result.suggested_bust_cm = correction.corrected['bust']
        body_result.suggested_waist_cm = correction.corrected['waist']
        body_result.suggested_hips_cm = correction.corrected['hips']
        body_result.body_type = classify_body_type(body_result.suggested_bust_cm, body_result.suggested_waist_cm, body_result.suggested_hips_cm)
        body_result.measurement_source = f"{body_result.measurement_source}_calibrated"
        body_result.notes.append(correction.note)
        body_result.notes.append(
            f"Korekta AI: biust/klatka {correction.offsets_cm['bust']:+.1f} cm, talia {correction.offsets_cm['waist']:+.1f} cm, biodra {correction.offsets_cm['hips']:+.1f} cm."
        )
    else:
        body_result.notes.append(correction.note)
    return body_result




def render_posture_summary(posture_summary: Dict, body_result: Optional[BodyAnalysisResult] = None):
    st.subheader('Ocena postawy (screening wizualny)')
    if body_result is not None:
        if not body_result.front_capture.get('posture_ready') or not body_result.profile_capture.get('posture_ready'):
            st.info('Ocena postawy została ograniczona, bo front albo profil nie mają posture_ready. Popraw zdjęcia, jeśli chcesz dokładniejszy screening.')
            return
    if not posture_summary:
        st.info('Brak danych o postawie.')
        return
    if not posture_summary.get('available'):
        st.info(posture_summary.get('message', 'Analiza postawy nie była dostępna.'))
        return
    if not posture_summary.get('detected'):
        st.success(tr_text(posture_summary.get('message', 'Nie wykryto istotnej cechy postawy wpływającej wizualnie na dobór kreacji.')))
        if posture_summary.get('scale_note'):
            st.caption(posture_summary['scale_note'])
        return
    st.warning(tr_text(posture_summary.get('message', 'Wykryto wizualne cechy postawy.')))
    if posture_summary.get('scale_note'):
        st.caption(posture_summary['scale_note'])
    rows = []
    for f in posture_summary.get('findings', []):
        rows.append({
            'cecha': f.get('label'),
            'skala 1–10': f.get('score_1_10'),
            'confidence': f"{int(float(f.get('confidence', 0))*100)}%",
            'opis': f.get('description'),
            'evidence': f.get('evidence'),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_visual_fit(visual_fit: Dict):
    st.subheader(tr_text('Impact of the cut on appearance for detected posture traits'))
    if not visual_fit:
        st.info('Brak oceny wizualnej produktu.')
        return
    status = visual_fit.get('status')
    msg = visual_fit.get('message', '')
    if status == 'helps':
        st.success(msg)
    elif status == 'risk':
        st.warning(msg)
    else:
        st.info(msg)
    if visual_fit.get('helps'):
        st.markdown('**Co może pomagać**')
        for line in visual_fit['helps']:
            st.write(f'- {line}')
    if visual_fit.get('risks'):
        st.markdown('**Na co uważać**')
        for line in visual_fit['risks']:
            st.write(f'- {line}')
    if visual_fit.get('neutral'):
        st.markdown('**Neutralne obserwacje**')
        for line in visual_fit['neutral'][:5]:
            st.write(f'- {line}')


lang_left, lang_right = st.columns([1.2, 3])
with lang_left:
    _lang_choice = st.selectbox('Language / auto', list(LOCALE_CHOICES.keys()), key='ui_language_choice', index=0)
with lang_right:
    st.caption(f"Auto locale uses your browser settings when available. Current locale: {_selected_locale_code()} | base language: {_selected_lang_code().upper()}")
st.title(tr('ATEENA-ST MVP — Apparel v8.6.8'))
st.caption(tr_text('V8.6.8 adds to V7.4: integrated Capture Pro session save into main analysis, optional BACK capture in Capture Pro, stronger hand-off from mobile capture to product analysis, and stricter photo reuse flow.'))
st.caption(f'Aktualny build: {BUILD_LABEL}')
render_runtime_self_test()
if not HAS_MEDIAPIPE:
    st.error('Vision pipeline disabled — measurement fallback only. Dopóki MediaPipe nie działa, aplikacja nie powinna generować właściwych rekomendacji ze zdjęć.')

with st.expander(tr('Co nowego w v8.6.8'), expanded=False):
    st.markdown(
        """
        - **Hard quality gate**: zdjęcie może zostać zaakceptowane, odesłane do poprawy albo odrzucone.
        - **Pełna sylwetka**: aplikacja wymaga widocznej głowy, rąk, kolan, kostek i stóp.
        - **Front / profil / półprofil**: aplikacja rozróżnia orientację i nie powinna przepuszczać półprofilu jako prawdziwego profilu.
        - **Pozycja stojąca**: zdjęcia siedząc są odrzucane.
        - **Kąt kamery**: aplikacja ocenia przechylenie i czy telefon jest około wysokości bioder.
        - **Wpływ stroju do zdjęcia**: luźne ubranie obniża wiarygodność lub blokuje pomiar.
        - **Posture screening**: aplikacja opisuje brak wykrytej istotnej cechy postawy albo pokazuje wizualne cechy postawy w skali 1–10 (w UI maks. 7/10).
        - **Visual compensation**: po wyborze produktu aplikacja ocenia, czy fason pomaga, jest neutralny czy może podkreślać wykryte cechy postawy.
        - **Stabilizacja uploadu 4 zdjęć**: sloty zapisują się w session state i pokazują diagnostykę pliku.
        - **Tryb diagnostyczny**: możesz przełączyć dokładny widok statusu slotów i eksportować paczkę debug.
        - **Analiza poglądowa**: wynik może zostać policzony mimo warningów quality gate, jeśli chcesz debugować pipeline.
        - **Runtime self-test**: aplikacja pokazuje, czy MediaPipe / vision pipeline działa.
        - **Źródło wyniku**: rozróżnienie `photo-based estimate` vs `fallback prior only`.
        - **Blokada mylącego fallbacku**: jeśli wynik pochodzi tylko z fallback prior, rekomendacja jest zatrzymywana.
        """
    )

analyze_tab, capture_tab, mobile_tab, search_tab, annotation_tab, qa_tab, benchmark_tab, feedback_tab, history_tab = st.tabs([tr('Nowa analiza'), 'Capture Pro (beta)', tr('Capture Mobile Studio'), tr('Wyszukiwanie po zdjęciu'), 'Annotation Review', 'QA / Admin', 'Benchmark', tr('Feedback po zakupie'), tr('Historia testów')])

with analyze_tab:
    left, right = st.columns([1.08, 1])
    with left:
        st.header(tr('1. Dane użytkownika'))
        email = st.text_input(req_label(tr('Email')), placeholder='anna@example.com')

        st.subheader(tr_text('Prywatność i zgody'))
        consent_analysis = st.checkbox(tr_text('Akceptuję analizę zdjęć ciała do działania aplikacji'), value=True)
        consent_store_images = st.checkbox(tr_text('Zgadzam się na zapis zdjęć w systemie'), value=True)
        consent_training = st.checkbox(tr_text('Zgadzam się na wykorzystanie danych do poprawy modelu'), value=True)
        if not consent_analysis:
            st.error(tr_text('Bez zgody na analizę aplikacja nie może przejść dalej.'))
        if not consent_store_images:
            st.info(tr_text('Jeśli nie wyrażasz zgody na zapis zdjęć, system zapisze tylko wynik analizy bez plików zdjęć.'))
        if not consent_training:
            st.info(tr_text('Jeśli nie wyrażasz zgody na użycie danych do poprawy modelu, Twoje dane nie trafią do calibration loop.'))

        gender = required_selectbox(tr('Płeć'), ['kobieta', 'mężczyzna'], key='gender_required_v865', format_func=tr)
        unit_system = required_selectbox(tr('System jednostek'), ['metryczny', 'imperialny'], key='unit_system_required_v865', format_func=tr)
        c1, c2, c3 = st.columns(3)
        if unit_system == 'metryczny':
            with c1:
                height_cm = required_number_text_input(tr('Wzrost (cm)'), key='height_cm_required_v865', placeholder='np. 178')
            with c2:
                weight_kg = required_number_text_input(tr('Waga (kg)'), key='weight_kg_required_v865', placeholder='np. 82')
            with c3:
                age = required_int_text_input(tr('Wiek'), key='age_required_v865', placeholder='np. 34')
        elif unit_system == 'imperialny':
            with c1:
                height_ft = required_int_text_input(tr_text('Height — feet (ft)'), key='height_ft_required_v865', placeholder='e.g. 5')
                height_in = required_int_text_input(tr_text('Height — inches (in)'), key='height_in_required_v865', placeholder='e.g. 11')
            with c2:
                weight_lb = required_number_text_input(tr_text('Weight (lb)'), key='weight_lb_required_v865', placeholder='e.g. 180')
            with c3:
                age = required_int_text_input(tr('Wiek'), key='age_required_v865_imp', placeholder='np. 34')
            height_cm = round((float(height_ft or 0) * 12 + float(height_in or 0)) * CM_PER_IN, 1) if height_ft is not None and height_in is not None else None
            weight_kg = round(lb_to_kg(float(weight_lb)), 1) if weight_lb is not None else None
            if height_cm is not None and weight_kg is not None:
                st.caption(f'Przeliczenie wewnętrzne: {height_cm:.1f} cm / {weight_kg:.1f} kg')
        else:
            height_cm = None
            weight_kg = None
            age = None

        measurement_mode = required_selectbox(
            tr('Jak chcesz określić wymiary?'),
            ['AI scan', 'AI scan + ręczna korekta', 'Wpiszę ręcznie'],
            key='measurement_mode_required_v865',
            format_func=tr,
        )
        fit_preference = optional_selectbox(tr('Preferencja dopasowania (opcjonalnie)'), ['automatycznie', 'standard', 'dopasowane', 'luźniejsze'], key='fit_preference_optional_v865', format_func=tr)
        st.caption('Wersja 8.6.5 startuje z pustymi polami. Uzupełnij je świadomie — aplikacja nie zakłada domyślnych danych użytkownika.')

        required_user_fields = []
        if not gender:
            required_user_fields.append(tr('Płeć'))
        if not unit_system:
            required_user_fields.append(tr('System jednostek'))
        if height_cm is None:
            required_user_fields.append(tr('Wzrost (cm)') if unit_system != 'imperialny' else 'Height')
        if weight_kg is None:
            required_user_fields.append(tr('Waga (kg)') if unit_system != 'imperialny' else 'Weight')
        if age is None:
            required_user_fields.append(tr('Wiek'))
        if not measurement_mode:
            required_user_fields.append(tr('Jak chcesz określić wymiary?'))
        user_core_ready = len(required_user_fields) == 0
        if not user_core_ready:
            st.warning('Uzupełnij wymagane pola użytkownika: ' + ', '.join(required_user_fields))

        guide_pack = GUIDE_ASSETS[gender] if gender in GUIDE_ASSETS else GUIDE_ASSETS['kobieta']
        prior = anthropometric_prior(float(height_cm or 170), float(weight_kg or 62), int(age or 30), gender=gender or 'kobieta')
        primary_measure_label = 'Biust (cm)' if gender == 'kobieta' else 'Klatka piersiowa (cm)'
        primary_measure_label_display = metric_key_label(primary_measure_label, unit_system or 'metryczny')

        st.markdown('### Instrukcja ustawienia do zdjęcia')
        g1, g2, g3 = st.columns(3)
        with g1:
            st.image(str(guide_pack['front']), caption=f'Kadr FRONT — {gender}', use_container_width=True)
        with g2:
            st.image(str(guide_pack['profile']), caption=f'Kadr PROFIL — {gender}', use_container_width=True)
        with g3:
            st.image(str(guide_pack['back']), caption=f'Kadr TYŁ — {gender}', use_container_width=True)
        st.info('Najlepszy kompromis dla AI: ręce lekko odsunięte od ciała (ok. 10–15 cm), mięśnie rozluźnione, telefon w dłoni osoby robiącej zdjęcie albo na stabilnym podparciu, zawsze na wysokości bioder, proste tło, cała sylwetka w kadrze. W 8.6.1 prosimy o komplet 4 zdjęć: FRONT, PROFIL LEWY, PROFIL PRAWY i TYŁ. Dwa profile poprawiają stabilność odczytu głębokości, a TYŁ poprawia ocenę symetrii i części stref dolnych.')

        show_manual_guide = measurement_mode in ['AI scan + ręczna korekta', 'Wpiszę ręcznie']
        if show_manual_guide:
            with st.expander('Pokaż instrukcję ręcznych pomiarów', expanded=(measurement_mode == 'Wpiszę ręcznie')):
                st.image(str(guide_pack['manual']), caption=f'Instrukcja ręcznego mierzenia — {gender}', use_container_width=True)

        body_result = st.session_state.get('body_result')
        posture_summary = st.session_state.get('posture_summary')
        front_photo = None
        profile_photo = None
        right_profile_photo = None
        back_photo = None
        front_photo_bytes = None
        profile_photo_bytes = None
        right_profile_photo_bytes = None
        back_photo_bytes = None
        capture_method = 'inna osoba trzyma telefon'
        clothing_fit = 'średnie'
        diagnostic_mode = st.session_state.get('diagnostic_mode_v864', False)

        if measurement_mode != 'Wpiszę ręcznie':
            with st.expander('Live capture overlay (beta helper)', expanded=False):
                st.caption('To pomocnik live — nie zapisuje zdjęć do aplikacji, ale pomaga ustawić sylwetkę zanim użyjesz pola Aparat.')
                render_live_capture_overlay_helper(gender)

            stored_capture = st.session_state.get('capture_pro_saved')
            capture_options = ['Aparat (zalecane)', 'Upload z galerii']
            if stored_capture and stored_capture.get('front_bytes') and stored_capture.get('profile_bytes'):
                capture_options.append('Capture Pro session')
            capture_mode = required_selectbox(tr('Skąd bierzemy zdjęcia?'), capture_options, key='capture_mode_required_v865', format_func=tr)

            if capture_mode == 'Capture Pro session':
                if not stored_capture:
                    st.error('Brak zapisanej sesji Capture Pro.')
                else:
                    st.success('Używasz zapisanej sesji Capture Pro z measurement_ready.')
                    capture_method = stored_capture.get('capture_method', capture_method)
                    clothing_fit = stored_capture.get('photo_clothing_fit', clothing_fit)
                    front_photo_bytes = stored_capture.get('front_bytes')
                    profile_photo_bytes = stored_capture.get('profile_bytes')
                    right_profile_photo_bytes = stored_capture.get('right_profile_bytes')
                    back_photo_bytes = stored_capture.get('back_bytes')
                    st.caption(f"Metoda capture: {capture_method} | Strój podczas zdjęcia: {clothing_fit}")
                    cp1, cp2, cp3, cp4 = st.columns(4)
                    with cp1:
                        if front_photo_bytes:
                            st.image(front_photo_bytes, caption='FRONT z Capture Pro', use_container_width=True)
                    with cp2:
                        if profile_photo_bytes:
                            st.image(profile_photo_bytes, caption='PROFIL z Capture Pro', use_container_width=True)
                    with cp3:
                        if right_profile_photo_bytes:
                            st.image(right_profile_photo_bytes, caption='PROFIL PRAWY z Capture Pro', use_container_width=True)
                    with cp4:
                        if back_photo_bytes:
                            st.image(back_photo_bytes, caption='TYŁ z Capture Pro', use_container_width=True)
                    if not stored_capture.get('accepted'):
                        st.warning('Ta sesja Capture Pro nie ma pełnego measurement_ready dla FRONT, obu PROFILI i TYŁ. Możesz ją obejrzeć, ale nie powinna być użyta do właściwego pomiaru.')
            else:
                capture_method = required_selectbox(
                    tr('Jak powstanie zdjęcie?'),
                    ['inna osoba trzyma telefon', 'telefon oparty stabilnie + timer', 'statyw', 'zdjęcie wykonane samemu'],
                    key='capture_method_required_v865',
                    format_func=tr,
                )
                clothing_fit = required_selectbox(
                    tr('Czy ubranie, w którym jesteś, jest wystarczająco dopasowane, żeby pomiar był dokładny?'),
                    ['obcisłe', 'raczej dopasowane', 'średnie', 'luźne', 'bardzo luźne', 'nie wiem'],
                    key='clothing_fit_required_v865',
                    format_func=tr,
                )
                if clothing_fit in {'luźne', 'bardzo luźne'}:
                    st.warning('Luźne ubranie może uniemożliwić wiarygodny pomiar. Najlepszy efekt daje dopasowany strój sportowy albo bielizna.')
                elif clothing_fit == 'nie wiem':
                    st.info('Jeśli materiał odstaje od ciała, wynik będzie mniej pewny. Najlepszy efekt daje dopasowany strój.')
                elif not clothing_fit:
                    st.info('Wybierz dopasowanie stroju, zanim przejdziesz dalej.')
                st.caption('Statyw nie jest wymagany. Najlepiej, gdy druga osoba trzyma telefon lub gdy telefon jest stabilnie oparty i ustawiony na wysokości bioder.')
                if capture_method == 'zdjęcie wykonane samemu':
                    st.warning('Tryb „zdjęcie wykonane samemu” traktuj jako awaryjny — selfie pełnej sylwetki często zniekształca proporcje i obniża jakość pomiaru.')

                diagnostic_mode = st.checkbox('Pokaż dane diagnostyczne uploadu i analizy', value=False, key='diagnostic_mode_v864')

                if capture_mode == 'Aparat (zalecane)':
                    front_photo = st.camera_input(req_label(tr('Zrób zdjęcie FRONT')), key='front_camera_v863')
                    profile_photo = st.camera_input(req_label(tr('Zrób zdjęcie PROFIL LEWY')), key='profile_left_camera_v863')
                    right_profile_photo = st.camera_input(req_label(tr('Zrób zdjęcie PROFIL PRAWY')), key='profile_right_camera_v863')
                    back_photo = st.camera_input(req_label(tr('Zrób zdjęcie TYŁ')), key='back_camera_v863')
                else:
                    front_photo = st.file_uploader(req_label(tr('Dodaj zdjęcie FRONT')), type=SUPPORTED_IMAGE_UPLOAD_TYPES, key='front_upload_v863')
                    profile_photo = st.file_uploader(req_label(tr('Dodaj zdjęcie PROFIL LEWY')), type=SUPPORTED_IMAGE_UPLOAD_TYPES, key='profile_left_upload_v863')
                    right_profile_photo = st.file_uploader(req_label(tr('Dodaj zdjęcie PROFIL PRAWY')), type=SUPPORTED_IMAGE_UPLOAD_TYPES, key='profile_right_upload_v863')
                    back_photo = st.file_uploader(req_label(tr('Dodaj zdjęcie TYŁ')), type=SUPPORTED_IMAGE_UPLOAD_TYPES, key='back_upload_v863')

                # persist uploads across reruns, especially on mobile Safari/Chrome
                if front_photo is not None:
                    front_photo_bytes = persist_uploaded_slot('front', front_photo)
                else:
                    front_photo_bytes = front_photo_bytes or get_persisted_slot('front')
                if profile_photo is not None:
                    profile_photo_bytes = persist_uploaded_slot('profile_left', profile_photo)
                else:
                    profile_photo_bytes = profile_photo_bytes or get_persisted_slot('profile_left')
                if right_profile_photo is not None:
                    right_profile_photo_bytes = persist_uploaded_slot('profile_right', right_profile_photo)
                else:
                    right_profile_photo_bytes = right_profile_photo_bytes or get_persisted_slot('profile_right')
                if back_photo is not None:
                    back_photo_bytes = persist_uploaded_slot('back', back_photo)
                else:
                    back_photo_bytes = back_photo_bytes or get_persisted_slot('back')

                st.markdown('#### Status slotów zdjęć')
                if st.button('Wyczyść wszystkie sloty zdjęć', key='clear_all_slots_btn', use_container_width=True):
                    clear_all_upload_slots()
                    st.rerun()
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    render_upload_slot_status('FRONT', front_photo, front_photo_bytes, 'front', diagnostic_mode=diagnostic_mode)
                with s2:
                    render_upload_slot_status('PROFIL LEWY', profile_photo, profile_photo_bytes, 'profile_left', diagnostic_mode=diagnostic_mode)
                with s3:
                    render_upload_slot_status('PROFIL PRAWY', right_profile_photo, right_profile_photo_bytes, 'profile_right', diagnostic_mode=diagnostic_mode)
                with s4:
                    render_upload_slot_status('TYŁ', back_photo, back_photo_bytes, 'back', diagnostic_mode=diagnostic_mode)
                slot_payloads = {'front': front_photo_bytes, 'profile_left': profile_photo_bytes, 'profile_right': right_profile_photo_bytes, 'back': back_photo_bytes}
                slot_meta = {
                    'front': get_persisted_slot_meta('front'),
                    'profile_left': get_persisted_slot_meta('profile_left'),
                    'profile_right': get_persisted_slot_meta('profile_right'),
                    'back': get_persisted_slot_meta('back'),
                }
                if front_photo_bytes and profile_photo_bytes and right_profile_photo_bytes and back_photo_bytes:
                    st.success('Wszystkie 4 sloty zdjęć są wczytane i gotowe do analizy.')
                else:
                    loaded = [k for k,v in slot_payloads.items() if v]
                    st.info(f'Załadowane sloty: {len(loaded)}/4')
                required_capture_fields = []
                if not capture_mode:
                    required_capture_fields.append(tr('Skąd bierzemy zdjęcia?'))
                if measurement_mode != 'Wpiszę ręcznie' and capture_mode != 'Capture Pro session':
                    if not capture_method:
                        required_capture_fields.append(tr('Jak powstanie zdjęcie?'))
                    if not clothing_fit:
                        required_capture_fields.append(tr('Czy ubranie, w którym jesteś, jest wystarczająco dopasowane, żeby pomiar był dokładny?'))
                if measurement_mode != 'Wpiszę ręcznie':
                    if not front_photo_bytes:
                        required_capture_fields.append('FRONT')
                    if not profile_photo_bytes:
                        required_capture_fields.append('PROFIL LEWY')
                    if not right_profile_photo_bytes:
                        required_capture_fields.append('PROFIL PRAWY')
                    if not back_photo_bytes:
                        required_capture_fields.append('TYŁ')
                if diagnostic_mode:
                    debug_zip = build_debug_bundle(slot_payloads, slot_meta, st.session_state.get('body_result'))
                    st.download_button('Eksportuj pakiet debug sesji', data=debug_zip, file_name=f'ateena_debug_{BUILD_ID}.zip', mime='application/zip', use_container_width=True)

            if st.button(tr('Analizuj sylwetkę'), use_container_width=True, type='primary'):
                if not user_core_ready:
                    st.error('Uzupełnij wszystkie wymagane pola użytkownika oznaczone gwiazdką przed analizą.')
                    st.stop()
                if not capture_mode:
                    st.error('Wybierz wymagane źródło zdjęć przed analizą.')
                    st.stop()
                if capture_mode != 'Capture Pro session' and (not capture_method or not clothing_fit):
                    st.error('Uzupełnij wymagane pola zdjęć oznaczone gwiazdką przed analizą.')
                    st.stop()
                front_payload = front_photo_bytes or (image_to_bytes(front_photo) if front_photo else None)
                profile_payload = profile_photo_bytes or (image_to_bytes(profile_photo) if profile_photo else None)
                right_profile_payload = right_profile_photo_bytes or (image_to_bytes(right_profile_photo) if right_profile_photo else None)
                back_payload = back_photo_bytes or (image_to_bytes(back_photo) if back_photo else None)
                missing_slots = []
                if not front_payload:
                    missing_slots.append('FRONT')
                if not profile_payload:
                    missing_slots.append('PROFIL LEWY')
                if not right_profile_payload:
                    missing_slots.append('PROFIL PRAWY')
                if not back_payload:
                    missing_slots.append('TYŁ')
                st.session_state['last_upload_diagnostics'] = {
                    'missing_slots': missing_slots,
                    'slot_payload_present': {
                        'FRONT': bool(front_payload),
                        'PROFIL LEWY': bool(profile_payload),
                        'PROFIL PRAWY': bool(right_profile_payload),
                        'TYŁ': bool(back_payload),
                    },
                    'slot_meta': {
                        'front': get_persisted_slot_meta('front'),
                        'profile_left': get_persisted_slot_meta('profile_left'),
                        'profile_right': get_persisted_slot_meta('profile_right'),
                        'back': get_persisted_slot_meta('back'),
                    }
                }
                if missing_slots:
                    st.error('Dodaj komplet 4 zdjęć. Brakuje dokładnie: ' + ', '.join(missing_slots) + '. Jeśli używasz iPhone, zdjęcia HEIC/HEIF są konwertowane automatycznie po uploadzie. Jeśli slot pokazuje miniaturę, ale analiza nadal twierdzi że brakuje zdjęcia, kliknij „Wyczyść wszystkie sloty zdjęć” i dodaj je ponownie po kolei.')
                    if diagnostic_mode:
                        st.json(st.session_state['last_upload_diagnostics'])
                else:
                    with st.spinner('Analizuję sylwetkę, ustawienie ciała, jakość kadru i strefy słabe...'):
                        body_result = analyze_body(
                            front_payload,
                            profile_payload,
                            float(height_cm),
                            float(weight_kg),
                            int(age),
                            gender=gender,
                            clothing_fit_answer=clothing_fit,
                            capture_method=capture_method,
                            back_image_bytes=back_payload,
                            right_profile_image_bytes=right_profile_payload,
                        )
                        posture_summary = analyze_posture_from_images(
                            front_payload,
                            profile_payload,
                            gender=gender,
                            clothing_fit_answer=clothing_fit,
                            back_image_bytes=back_payload,
                        )
                        st.session_state['body_result'] = body_result
                        st.session_state['posture_summary'] = posture_summary
                        st.session_state['capture_method'] = capture_method
                        st.session_state['photo_clothing_fit'] = clothing_fit
                        accept_map = {
                            'FRONT': body_result.front_capture.get('status_code') == 'accept',
                            'PROFIL LEWY': body_result.left_profile_capture.get('status_code') == 'accept',
                            'PROFIL PRAWY': body_result.right_profile_capture.get('status_code') == 'accept',
                            'TYŁ': body_result.back_capture.get('status_code') == 'accept',
                        }
                        failed = [k for k, ok in accept_map.items() if not ok]
                        st.session_state['last_quality_gate_failed'] = failed
                        if failed:
                            st.warning('Analiza poglądowa została policzona, ale nie wszystkie 4 zdjęcia przeszły quality gate ACCEPT. Problem dotyczy: ' + ', '.join(failed) + '. Sprawdź sekcje statusów poniżej.')
                            if diagnostic_mode:
                                st.info('Tryb diagnostyczny: wynik pozostaje widoczny mimo warningów quality gate.')
                        else:
                            st.success('Sylwetka przeanalizowana. Wszystkie 4 ujęcia zostały przyjęte: FRONT, PROFIL LEWY, PROFIL PRAWY i TYŁ.')
        else:
            st.info('W trybie ręcznym zdjęcia są opcjonalne i nie są potrzebne do wygenerowania rekomendacji.')
            st.session_state['body_result'] = None
            st.session_state['posture_summary'] = {'available': False, 'detected': False, 'message': 'Tryb ręczny — analiza postawy wymaga zdjęć.'}
            body_result = None
            posture_summary = st.session_state.get('posture_summary')

        if body_result:
            st.subheader(tr('Wynik analizy sylwetki'))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(tr('Typ sylwetki'), tr_text(body_result.body_type))
            m2.metric(primary_measure_label_display.replace(' (cm)', ' (AI)').replace(' (in)', ' (AI)'), format_measure(body_result.suggested_bust_cm, unit_system))
            m3.metric('Talia (AI)', format_measure(body_result.suggested_waist_cm, unit_system))
            m4.metric('Biodra (AI)', format_measure(body_result.suggested_hips_cm, unit_system))
            st.caption(f"Pewność odczytu: {int(body_result.confidence * 100)}% | Budowa: {body_result.build_type} | Źródło: {body_result.measurement_source}")
            render_measurement_source_status(body_result)
            render_capture_gate_summary(body_result.front_capture, 'FRONT')
            render_capture_action_plan(body_result.front_capture, 'FRONT')
            render_capture_gate_summary(body_result.left_profile_capture, 'PROFIL LEWY')
            render_capture_action_plan(body_result.left_profile_capture, 'PROFIL LEWY')
            render_capture_gate_summary(body_result.right_profile_capture, 'PROFIL PRAWY')
            render_capture_action_plan(body_result.right_profile_capture, 'PROFIL PRAWY')
            if body_result.back_capture:
                render_capture_gate_summary(body_result.back_capture, 'TYŁ')
                render_capture_action_plan(body_result.back_capture, 'TYŁ')
            raw_cols = st.columns(3)
            raw_cols[0].metric('Surowy biust/klatka', format_measure(body_result.raw_bust_cm, unit_system))
            raw_cols[1].metric('Surowa talia', format_measure(body_result.raw_waist_cm, unit_system))
            raw_cols[2].metric('Surowe biodra', format_measure(body_result.raw_hips_cm, unit_system))
            st.markdown('**Confidence per part**')
            render_measurement_confidence_table(body_result)
            render_confidence_reasons(body_result)
            if body_result.weak_points:
                st.warning('AI ma słabszą pewność dla tych stref: ' + ', '.join(weak_label(k) for k in body_result.weak_points))
            if body_result.calibration_info.get('applied'):
                st.info(
                    f"Kalibracja aktywna: warstwy {', '.join(body_result.calibration_info.get('used_scopes', []))} | "
                    f"offsety: biust/klatka {body_result.calibration_info['offsets_cm']['bust']:+.1f}, "
                    f"talia {body_result.calibration_info['offsets_cm']['waist']:+.1f}, biodra {body_result.calibration_info['offsets_cm']['hips']:+.1f} cm | "
                    f"segment: {body_result.calibration_info.get('photo_quality_bucket', '—')} / {body_result.calibration_info.get('product_kind', '—')} / {body_result.calibration_info.get('clothing_fit_bucket', '—')}."
                )
            if measurement_mode != 'Wpiszę ręcznie':
                render_capture_status('Front', body_result.front_capture)
                render_capture_status('Profil lewy', body_result.left_profile_capture)
                render_capture_status('Profil prawy', body_result.right_profile_capture)
                render_capture_status('Tył', body_result.back_capture)
            for note in body_result.notes:
                st.write(f"- {tr_text(note)}")
            if diagnostic_mode:
                st.markdown('### Diagnostyka uploadu i analizy')
                if st.session_state.get('last_upload_diagnostics'):
                    st.json(st.session_state.get('last_upload_diagnostics'))
                st.write('Quality gate failed:', st.session_state.get('last_quality_gate_failed', []))
            d1, d2, d3, d4 = st.columns(4)
            if body_result.front_debug_image is not None:
                with d1:
                    st.image(body_result.front_debug_image[:, :, ::-1], caption='Front — jakość kadru i linie pomiarowe', use_container_width=True)
            if body_result.profile_debug_image is not None:
                with d2:
                    st.image(body_result.profile_debug_image[:, :, ::-1], caption='Profil lewy — jakość kadru i linie pomiarowe', use_container_width=True)
            if getattr(body_result, 'right_profile_debug_image', None) is not None:
                with d3:
                    st.image(body_result.right_profile_debug_image[:, :, ::-1], caption='Profil prawy — jakość kadru i linie pomiarowe', use_container_width=True)
            if body_result.back_debug_image is not None:
                with d4:
                    st.image(body_result.back_debug_image[:, :, ::-1], caption='Tył — jakość kadru i linie pomiarowe', use_container_width=True)
            render_background_cleanup_preview(body_result)
            render_landmark_segment_report(body_result)
            st.markdown('#### Wyniki zapisane w cm')
            values_cm = {
                'Biust / klatka': round(float(body_result.suggested_bust_cm), 1),
                'Talia': round(float(body_result.suggested_waist_cm), 1),
                'Biodra': round(float(body_result.suggested_hips_cm), 1),
            }
            for key, label in EXTRA_MEASURE_LABELS:
                if key in body_result.extra_estimates:
                    values_cm[label.replace(' (cm)', '')] = round(float(body_result.extra_estimates[key]), 1)
            st.json(values_cm)
            render_sanity_report(body_result)
            render_posture_summary(posture_summary, body_result)

        st.markdown('### Wymiary użyte do rekomendacji')
        extra_measurements_raw: Dict[str, float] = {k: 0.0 for k, _ in EXTRA_MEASURE_LABELS}
        weak_core_overrides: Dict[str, float] = {}
        weak_points = body_result.weak_points if body_result else []

        use_manual_core = measurement_mode == 'Wpiszę ręcznie'
        if measurement_mode == 'Wpiszę ręcznie':
            st.caption('Wpisz swoje rzeczywiste wymiary ręcznie. Wersja 8.6.5 nie podpowiada wartości domyślnych.')
        elif measurement_mode == 'AI scan + ręczna korekta':
            st.caption('Jeśli chcesz poprawić wynik AI, wpisz ręcznie tylko te wymiary, które znasz. Pola startują puste.')
            use_manual_core = st.checkbox('Chcę nadpisać cały rdzeń AI ręcznymi wymiarami', value=False)

        c1, c2, c3 = st.columns(3)
        if use_manual_core:
            with c1:
                bust_val = blank_measure_input(primary_measure_label, unit_system or 'metryczny', key='bust_manual_core_v865', placeholder='np. 102')
                bust_cm = float(bust_val or 0.0)
            with c2:
                waist_val = blank_measure_input('Talia (cm)', unit_system or 'metryczny', key='waist_manual_core_v865', placeholder='np. 86')
                waist_cm = float(waist_val or 0.0)
            with c3:
                hips_val = blank_measure_input('Biodra (cm)', unit_system or 'metryczny', key='hips_manual_core_v865', placeholder='np. 101')
                hips_cm = float(hips_val or 0.0)
        else:
            bust_cm = 0.0
            waist_cm = 0.0
            hips_cm = 0.0
            with c1:
                st.metric(primary_measure_label_display.replace(' (cm)', ' (AI)').replace(' (in)', ' (AI)'), format_measure(body_result.suggested_bust_cm, unit_system or 'metryczny') if body_result else '—')
            with c2:
                st.metric('Talia (AI)', format_measure(body_result.suggested_waist_cm, unit_system or 'metryczny') if body_result else '—')
            with c3:
                st.metric('Biodra (AI)', format_measure(body_result.suggested_hips_cm, unit_system or 'metryczny') if body_result else '—')

        if measurement_mode != 'Wpiszę ręcznie' and body_result:
            if weak_points:
                st.warning('Rekomendowane ręczne potwierdzenie tylko dla słabych stref: ' + ', '.join(weak_label(k) for k in weak_points))
                with st.expander('Confirm only weak points', expanded=True):
                    st.caption('Te pola startują puste. Uzupełnij tylko te, które znasz — brak wpisu nie nadpisze AI.')
                    weak_core_keys = [k for k in weak_points if k in {'bust', 'waist', 'hips'}]
                    if weak_core_keys:
                        wc1, wc2, wc3 = st.columns(3)
                        if 'bust' in weak_core_keys:
                            with wc1:
                                val = blank_measure_input(primary_measure_label, unit_system or 'metryczny', key='weak_bust_v865', placeholder='np. 102')
                                if val is not None:
                                    weak_core_overrides['bust'] = float(val)
                        if 'waist' in weak_core_keys:
                            with wc2:
                                val = blank_measure_input('Talia (cm)', unit_system or 'metryczny', key='weak_waist_v865', placeholder='np. 86')
                                if val is not None:
                                    weak_core_overrides['waist'] = float(val)
                        if 'hips' in weak_core_keys:
                            with wc3:
                                val = blank_measure_input('Biodra (cm)', unit_system or 'metryczny', key='weak_hips_v865', placeholder='np. 101')
                                if val is not None:
                                    weak_core_overrides['hips'] = float(val)
                    weak_extra = [k for k in weak_points if k not in {'bust', 'waist', 'hips'}]
                    if weak_extra:
                        ex_cols = st.columns(2)
                        for idx, (key, label) in enumerate(EXTRA_MEASURE_LABELS):
                            if key not in weak_extra:
                                continue
                            with ex_cols[idx % 2]:
                                val = blank_measure_input(label, unit_system or 'metryczny', key=f'weak_{key}_v865')
                                extra_measurements_raw[key] = float(val or 0.0)
            else:
                st.success('AI nie wskazało słabych punktów do ręcznego potwierdzenia.')

        full_measurements_on = st.checkbox('Pokaż pełny profil wymiarów', value=(measurement_mode == 'Wpiszę ręcznie'))
        if full_measurements_on:
            with st.expander('Pełny profil wymiarów', expanded=True):
                st.caption('Wszystkie pola startują puste. Uzupełnij tylko te wymiary, które rzeczywiście znasz.')
                ex_cols = st.columns(2)
                for idx, (key, label) in enumerate(EXTRA_MEASURE_LABELS):
                    with ex_cols[idx % 2]:
                        val = blank_measure_input(label, unit_system or 'metryczny', key=f'extra_{key}_v865')
                        extra_measurements_raw[key] = float(val or 0.0)
                st.caption('Szyja: mierz pod jabłkiem Adama. Brzuch: obwód na wysokości pępka. Łydka: podaj osobno najszersze i najwęższe miejsce.')
    with right:
        st.header(tr('2. Dane produktu'))
        search_group = required_selectbox(tr('Dla kogo szukasz produktu?'), list(PRODUCT_TREE.keys()), key='search_group_required_v865', format_func=tr_option)
        style_branch = required_selectbox(tr('Gałąź'), list(PRODUCT_TREE[search_group].keys()), key='style_branch_required_v865', format_func=tr_option) if search_group else None
        search_product_kind = required_selectbox(tr('Typ produktu'), PRODUCT_TREE[search_group][style_branch], key='product_kind_required_v865', format_func=tr_option) if search_group and style_branch else None
        if search_product_kind:
            st.session_state['last_search_product_kind'] = search_product_kind
            st.caption(f"{tr_text('Selected path')}: {tr_option(search_group)} → {tr_option(style_branch)} → {tr_option(search_product_kind)}")
            st.info('Logika fitu w tej wersji działa już dla wielu kategorii ubrań. Pamiętaj jednak, że dokładność nadal zależy od jakości tabeli rozmiarów i danych na stronie produktu.')
        else:
            st.info('Wybierz kolejno: dla kogo, gałąź i typ produktu.')
        mode = required_selectbox(tr('Źródło produktu'), ['Link do produktu', 'Zrzut ekranu produktu', 'Katalog demo'], key='product_source_required_v865', format_func=tr)
        product_result = st.session_state.get('product_result')
        required_product_fields = []
        if not search_group:
            required_product_fields.append(tr('Dla kogo szukasz produktu?'))
        if not style_branch:
            required_product_fields.append(tr('Gałąź'))
        if not search_product_kind:
            required_product_fields.append(tr('Typ produktu'))
        if not mode:
            required_product_fields.append(tr('Źródło produktu'))

        if mode == 'Link do produktu' and search_product_kind:
            product_url = st.text_area(tr('Wklej link do produktu'), placeholder='https://shop.com/product', height=110)
            if st.button(tr('Pobierz dane produktu z linku'), use_container_width=True):
                if not product_url:
                    st.error('Wklej link do produktu.')
                else:
                    with st.spinner('Czytam stronę, szukam zdjęcia, tabeli rozmiarów i opinii...'):
                        try:
                            product_result = adapt_product_to_selected_kind(ingest_product_from_url(product_url), search_product_kind)
                            st.session_state['product_result'] = product_result
                            st.success('Dane produktu pobrane.')
                        except Exception as exc:
                            st.error(f'Nie udało się pobrać danych produktu: {exc}')
            st.caption('W tej wersji MVP link jest głównym źródłem produktu. Jeśli wkleisz kilka adresów naraz, aplikacja użyje pierwszego poprawnego linku i usunie parametry trackingowe.')
        elif mode == 'Zrzut ekranu produktu' and search_product_kind:
            st.caption('Fallback OCR 4.0: wgraj screenshot produktu, opcjonalnie tabeli rozmiarów i opinii. Jeśli OCR nie odczyta tabeli, aplikacja użyje fallbacku kategorii, a jeśli odczyta opinie — dołoży je do sygnałów produktu.')
            ocr_product_photo = st.file_uploader('Wgraj screenshot produktu', type=['jpg','jpeg','png'], key='ocr_product_photo')
            ocr_chart_photo = st.file_uploader('Wgraj screenshot tabeli rozmiarów (opcjonalne)', type=['jpg','jpeg','png'], key='ocr_chart_photo')
            ocr_review_photo = st.file_uploader('Wgraj screenshot opinii / recenzji (opcjonalne)', type=['jpg','jpeg','png'], key='ocr_review_photo')
            with st.expander('Ręczne wklejenie tekstu OCR / fallback', expanded=False):
                ocr_product_text = st.text_area('Wklej tekst odczytany z produktu (opcjonalne)', key='ocr_product_text', height=100)
                ocr_chart_text = st.text_area('Wklej tekst odczytany z tabeli rozmiarów (opcjonalne)', key='ocr_chart_text', height=120)
                ocr_review_text = st.text_area('Wklej tekst odczytany z opinii / review (opcjonalne)', key='ocr_review_text', height=120)
            if st.button('Analizuj screenshoty / tekst produktu', use_container_width=True):
                if not ocr_product_photo and not ocr_chart_photo and not ocr_review_photo and not (ocr_product_text or '').strip() and not (ocr_chart_text or '').strip() and not (ocr_review_text or '').strip():
                    st.error('Dodaj screenshot produktu, screenshot tabeli, screenshot opinii albo wklej tekst OCR.')
                else:
                    with st.spinner('Uruchamiam OCR fallback dla produktu, tabeli i opinii...'):
                        try:
                            if (ocr_product_text or '').strip() or (ocr_chart_text or '').strip():
                                product_result = adapt_product_to_selected_kind(ingest_product_from_texts(
                                    (ocr_product_text or '').strip(),
                                    (ocr_chart_text or '').strip(),
                                    search_product_kind,
                                ), search_product_kind)
                            else:
                                product_result = adapt_product_to_selected_kind(ingest_product_from_screenshots(
                                    image_to_bytes(ocr_product_photo) if ocr_product_photo else None,
                                    image_to_bytes(ocr_chart_photo) if ocr_chart_photo else None,
                                    search_product_kind,
                                ), search_product_kind)
                            review_lines = extract_review_lines_from_sources(image_to_bytes(ocr_review_photo) if ocr_review_photo else None, (ocr_review_text or '').strip())
                            if review_lines:
                                existing = list(getattr(product_result, 'review_lines', []) or [])
                                product_result.review_lines = existing + review_lines
                                product_result.review_count = max(int(getattr(product_result, 'review_count', 0) or 0), len(product_result.review_lines))
                                product_result.parsing_notes = list(getattr(product_result, 'parsing_notes', []) or []) + ['Review screenshot / OCR applied in V8.6.8.']
                            st.session_state['product_result'] = product_result
                            product_img_path = save_upload(ocr_product_photo, 'ocr_product') if ocr_product_photo else None
                            chart_img_path = save_upload(ocr_chart_photo, 'ocr_chart') if ocr_chart_photo else None
                            save_ocr_session(search_product_kind, product_img_path, chart_img_path, {'product_present': bool(ocr_product_photo), 'chart_present': bool(ocr_chart_photo), 'review_present': bool(ocr_review_photo), 'manual_text_present': bool((ocr_product_text or '').strip() or (ocr_chart_text or '').strip() or (ocr_review_text or '').strip()), 'review_lines_added': len(review_lines)}, product_result.parsing_notes)
                            st.success('Produkt przygotowany z Product Rescue 4.0: link / screenshot / OCR / review OCR / quality score / ręczna korekta.')
                        except Exception as exc:
                            st.error(f'Nie udało się zinterpretować danych produktu: {exc}')
        else:
            selected_row = render_demo_catalog()
            if selected_row:
                product_result = adapt_product_to_selected_kind(build_demo_product(selected_row), search_product_kind)
                st.session_state['product_result'] = product_result
                st.success('Wybrano produkt z katalogu demo.')

        if product_result:
            st.subheader('Profil produktu')
            st.caption(f'Logika fitu będzie liczona dla kategorii: {search_product_kind}')
            if getattr(product_result, 'image_url', None):
                st.image(product_result.image_url, caption='Zdjęcie produktu', use_container_width=True)
            st.write(f"**{product_result.brand} — {product_result.name}**")
            st.write(f"Fason: **{product_result.dress_type}**, fit: **{product_result.fit_type}**, stretch: **{product_result.stretch_level}**, długość: **{product_result.length_type}**")
            st.write(f"Efekt fasonu: **{product_result.style_effect.replace('_', ' ')}**")
            st.write(f"Strefy ryzyka: {', '.join(product_result.tight_areas) if product_result.tight_areas else 'brak wyraźnych sygnałów'}")
            if product_result.source_url:
                st.write(f"Źródło: {product_result.source_url}")
            st.markdown('**Sygnały z opinii klientów**')
            for line in product_result.review_lines or ['Brak odczytanych opinii lub strona nie pokazała recenzji.']:
                st.write(f"- {tr_text(line)}")

            ocr_quality = score_ocr_quality(
                getattr(product_result, 'name', ''),
                '\n'.join(getattr(product_result, 'parsing_notes', []) or []),
                getattr(product_result, 'review_lines', []) or [],
                getattr(product_result, 'size_chart', {}) or {},
            )
            st.markdown('**OCR / rescue quality**')
            q1, q2, q3, q4 = st.columns(4)
            q1.metric('OCR total', f"{ocr_quality['total_score']}/100")
            q2.metric('Product', ocr_quality['product_score'])
            q3.metric('Chart', ocr_quality['chart_score'])
            q4.metric('Reviews', ocr_quality['review_score'])
            st.caption(f"Band: {ocr_quality['band']} | rows: {ocr_quality['chart_rows']} | review lines: {ocr_quality['review_lines']}")
            if ocr_quality['flags']:
                for flag in ocr_quality['flags']:
                    st.write(f"- {flag}")

            st.markdown('**Tabela rozmiarów**')
            chart_df = pd.DataFrame([
                {
                    'rozmiar': size,
                    'biust': f"{vals['bust'][0]}–{vals['bust'][1]}" if 'bust' in vals else '—',
                    'klatka': f"{vals['chest'][0]}–{vals['chest'][1]}" if 'chest' in vals else '—',
                    'talia': f"{vals['waist'][0]}–{vals['waist'][1]}" if 'waist' in vals else '—',
                    'biodra': f"{vals['hips'][0]}–{vals['hips'][1]}" if 'hips' in vals else '—',
                    'udo': f"{vals['thigh'][0]}–{vals['thigh'][1]}" if 'thigh' in vals else '—',
                    'ramię': f"{vals['arm'][0]}–{vals['arm'][1]}" if 'arm' in vals else '—',
                }
                for size, vals in product_result.size_chart.items()
            ])
            st.dataframe(chart_df, use_container_width=True, hide_index=True)
            if product_result.used_fallback_chart:
                st.warning('Ta tabela jest fallbackiem referencyjnym, bo strona nie dała się wiarygodnie odczytać.')
            for note in product_result.parsing_notes:
                st.caption(tr_text(note))

            with st.expander('OCR fallback 4.0 — popraw dane produktu ręcznie', expanded=False):
                corr_brand = st.text_input('Marka (korekta)', value=getattr(product_result, 'brand', ''), key='ocr_corr_brand')
                corr_name = st.text_input('Nazwa produktu (korekta)', value=getattr(product_result, 'name', ''), key='ocr_corr_name')
                corr_dress_type = st.text_input('Typ / fason (korekta)', value=getattr(product_result, 'dress_type', ''), key='ocr_corr_dress_type')
                corr_fit = st.text_input('Fit (korekta)', value=getattr(product_result, 'fit_type', ''), key='ocr_corr_fit')
                corr_stretch = st.text_input('Stretch (korekta)', value=getattr(product_result, 'stretch_level', ''), key='ocr_corr_stretch')
                corr_length = st.text_input('Długość (korekta)', value=getattr(product_result, 'length_type', ''), key='ocr_corr_length')
                corr_reviews = st.text_area('Opinie / review lines — po jednej linii', value='\n'.join(getattr(product_result, 'review_lines', []) or []), key='ocr_corr_reviews', height=100)
                st.caption('Preferowana korekta tabeli rozmiarów: edytuj wiersze zamiast ręcznie modyfikować JSON.')
                chart_rows = chart_to_rows(getattr(product_result, 'size_chart', {}) or {})
                edited_rows = st.data_editor(pd.DataFrame(chart_rows), num_rows='dynamic', use_container_width=True, key='ocr_chart_editor_v83')
                with st.expander('Zaawansowany fallback JSON', expanded=False):
                    corr_chart = st.text_area('Tabela rozmiarów jako JSON', value=editable_chart_json(product_result), key='ocr_corr_chart', height=180)
                if st.button('Zastosuj korekty OCR / screenshot', key='apply_ocr_corrections_btn', use_container_width=True):
                    try:
                        parsed_chart = rows_to_chart(edited_rows.to_dict(orient='records'))
                        if not parsed_chart and (corr_chart or '').strip():
                            parsed_chart = json.loads(corr_chart)
                        product_result.brand = corr_brand or product_result.brand
                        product_result.name = corr_name or product_result.name
                        product_result.dress_type = corr_dress_type or product_result.dress_type
                        product_result.fit_type = corr_fit or product_result.fit_type
                        product_result.stretch_level = corr_stretch or product_result.stretch_level
                        product_result.length_type = corr_length or product_result.length_type
                        product_result.review_lines = [line.strip() for line in (corr_reviews or '').splitlines() if line.strip()]
                        if isinstance(parsed_chart, dict) and parsed_chart:
                            product_result.size_chart = parsed_chart
                            product_result.used_fallback_chart = False
                        product_result.parsing_notes = list(getattr(product_result, 'parsing_notes', []) or []) + ['Manual OCR correction applied in V8.6.8.']
                        st.session_state['product_result'] = product_result
                        st.success('Zastosowano ręczne korekty OCR.')
                    except Exception as exc:
                        st.error(f'Nie udało się zastosować korekty OCR: {exc}')

    render_missing_fields_panel({
        'Dane użytkownika': required_user_fields,
        'Zdjęcia': required_capture_fields if measurement_mode != 'Wpiszę ręcznie' else [],
        'Produkt': required_product_fields,
    })

    st.divider()
    if st.button(tr('3. Generuj rekomendację i zapisz do bazy'), use_container_width=True):
        body_result = st.session_state.get('body_result')
        posture_summary = st.session_state.get('posture_summary')
        product_result = st.session_state.get('product_result')
        if not consent_analysis:
            st.error(tr_text('Bez zgody na analizę aplikacja nie może przejść dalej.'))
        elif not user_core_ready:
            st.error('Uzupełnij wszystkie wymagane pola użytkownika oznaczone gwiazdką przed wygenerowaniem rekomendacji.')
        elif not email:
            st.error('Podaj email, żeby zapisać analizę i późniejszy feedback.')
        elif not search_group or not style_branch or not search_product_kind or not mode:
            st.error('Uzupełnij wszystkie wymagane pola produktu oznaczone gwiazdką przed wygenerowaniem rekomendacji.')
        elif not product_result:
            st.error('Najpierw pobierz dane produktu lub wybierz produkt z katalogu demo.')
        elif measurement_mode != 'Wpiszę ręcznie' and body_result and source_mode_label(body_result) == 'fallback_prior_only':
            st.error('Ta sesja działa w trybie fallback prior only. Rekomendacja została zablokowana, bo vision pipeline nie policzył właściwej estymacji ze zdjęć.')
            st.stop()
        else:
            calibration_candidate = False
            if body_result and measurement_mode != 'Wpiszę ręcznie':
                if body_result.front_capture.get('status_code') != 'accept' or body_result.left_profile_capture.get('status_code') != 'accept' or body_result.right_profile_capture.get('status_code') != 'accept' or body_result.back_capture.get('status_code') != 'accept':
                    st.warning('Generujesz wynik mimo że nie wszystkie 4 ujęcia mają ACCEPT. Rekomendację traktuj jako poglądową i sprawdź statusy capture.')
            extra_clean_draft = clean_extra_measurements(extra_measurements_raw if full_measurements_on else {}, gender, float(bust_cm), float(waist_cm), float(hips_cm))
            if measurement_mode == 'Wpiszę ręcznie':
                required_zones = get_category_profile(search_product_kind)['zones']
                missing_required = []
                if 'bust' in required_zones and bust_cm <= 0:
                    missing_required.append(primary_measure_label)
                if 'chest' in required_zones and (extra_clean_draft.get('chest_cm', 0) <= 0 and bust_cm <= 0):
                    missing_required.append('Klatka piersiowa')
                if 'waist' in required_zones and waist_cm <= 0:
                    missing_required.append('Talia')
                if 'hips' in required_zones and hips_cm <= 0:
                    missing_required.append('Biodra')
                if 'thigh' in required_zones and extra_clean_draft.get('thigh_cm', 0) <= 0:
                    missing_required.append('Udo')
                if 'arm_biceps' in required_zones and extra_clean_draft.get('arm_biceps_cm', 0) <= 0:
                    missing_required.append('Ramię / biceps')
                if missing_required:
                    st.error('W trybie ręcznym podaj brakujące kluczowe wymiary dla tej kategorii: ' + ', '.join(missing_required) + '.')
                    st.stop()
                body_result = manual_body_result(float(height_cm), float(weight_kg), int(age), gender, float(bust_cm), float(waist_cm), float(hips_cm))
                posture_summary = {'available': False, 'detected': False, 'message': 'Tryb ręczny — analiza postawy wymaga zdjęć.'}
            elif not body_result:
                st.error('Najpierw przeanalizuj sylwetkę albo przejdź do trybu ręcznego.')
                st.stop()
            else:
                if use_manual_core and (bust_cm <= 0 or waist_cm <= 0 or hips_cm <= 0):
                    st.error('Jeśli wybierasz ręczne nadpisanie rdzenia AI, uzupełnij ręcznie: klatkę/biust, talię i biodra.')
                    st.stop()
                calibration_candidate = measurement_mode == 'AI scan + ręczna korekta' and (use_manual_core or bool(weak_core_overrides))
                body_result = apply_calibration_to_result(body_result, gender=gender, clothing_fit=clothing_fit, product_kind=search_product_kind)
                body_result = apply_part_calibration_to_body_result(body_result, gender=gender, clothing_fit=clothing_fit, product_kind=search_product_kind)
                st.session_state['body_result'] = body_result

            used_bust = float(bust_cm) if use_manual_core else float(weak_core_overrides.get('bust', body_result.suggested_bust_cm))
            used_waist = float(waist_cm) if use_manual_core else float(weak_core_overrides.get('waist', body_result.suggested_waist_cm))
            used_hips = float(hips_cm) if use_manual_core else float(weak_core_overrides.get('hips', body_result.suggested_hips_cm))
            merged_extra = {k: v for k, v in extra_measurements_raw.items() if float(v) > 0}
            if body_result and getattr(body_result, 'extra_estimates', None):
                for k, v in body_result.extra_estimates.items():
                    merged_extra.setdefault(k, v)
            extra_clean = clean_extra_measurements(merged_extra, gender, used_bust, used_waist, used_hips)

            user = UserProfile(
                height_cm=float(height_cm),
                weight_kg=float(weight_kg),
                age=int(age),
                body_type=body_result.body_type,
                fit_preference=None if fit_preference == 'automatycznie' else fit_preference,
                build_type=body_result.build_type,
                bust_cm=used_bust,
                waist_cm=used_waist,
                hips_cm=used_hips,
                extra_measurements=extra_clean,
            )
            if hasattr(product_result, 'parsing_notes'):
                product_result.parsing_notes = list(product_result.parsing_notes) + [f'Search intent: {search_group} → {style_branch} → {search_product_kind}']
            setattr(product_result, 'product_kind', search_product_kind)
            setattr(product_result, 'search_group', search_group)
            setattr(product_result, 'style_branch', style_branch)
            product_lite = ProductProfileLite(
                brand=product_result.brand,
                name=product_result.name,
                dress_type=product_result.dress_type,
                fit_type=product_result.fit_type,
                stretch_level=product_result.stretch_level,
                length_type=product_result.length_type,
                style_effect=product_result.style_effect,
                runs_small=product_result.runs_small,
                runs_large=product_result.runs_large,
                true_to_size=product_result.true_to_size,
                tight_areas=product_result.tight_areas,
                review_count=product_result.review_count,
                review_lines=product_result.review_lines,
                size_chart=product_result.size_chart,
                product_kind=search_product_kind,
                search_group=search_group,
                style_branch=style_branch,
            )
            rec = recommend_size(user, product_lite)
            visual_fit = evaluate_visual_compensation(posture_summary, product_result)
            rec['visual_fit'] = visual_fit
            rec['model_version'] = 'mvp_v6_3b3_locale_matrix'
            rec['estimated_measurements'] = {
                'gender': gender,
                'unit_system': unit_system,
                'search_group': search_group,
                'search_branch': style_branch,
                'search_product_kind': search_product_kind,
                'bust_or_chest_cm': used_bust,
                'waist_cm': used_waist,
                'hips_cm': used_hips,
                'ai_raw_core_cm': {
                    'bust_or_chest': round(float(body_result.raw_bust_cm), 1),
                    'waist': round(float(body_result.raw_waist_cm), 1),
                    'hips': round(float(body_result.raw_hips_cm), 1),
                },
                'ai_corrected_core_cm': {
                    'bust_or_chest': round(float(body_result.suggested_bust_cm), 1),
                    'waist': round(float(body_result.suggested_waist_cm), 1),
                    'hips': round(float(body_result.suggested_hips_cm), 1),
                },
                'calibration_info': body_result.calibration_info,
                'measurement_confidence': body_result.measurement_confidence,
                'weak_points': body_result.weak_points,
                'sanity_report': body_result.sanity_report,
                **body_result.extra_estimates,
                **extra_clean,
            }

            front_path = (save_upload(front_photo, 'front') if front_photo else save_upload_bytes(front_photo_bytes, 'front')) if consent_store_images else None
            profile_path = (save_upload(profile_photo, 'profile_left') if profile_photo else save_upload_bytes(profile_photo_bytes, 'profile_left')) if consent_store_images else None
            right_profile_path = (save_upload(right_profile_photo, 'profile_right') if right_profile_photo else save_upload_bytes(right_profile_photo_bytes, 'profile_right')) if consent_store_images else None
            back_path = (save_upload(back_photo, 'back') if back_photo else save_upload_bytes(back_photo_bytes, 'back')) if consent_store_images else None
            user_id = save_user(email, gender, unit_system, capture_method, float(height_cm), float(weight_kg), int(age), None if fit_preference == 'automatycznie' else fit_preference)
            save_privacy_consent(user_id, consent_analysis, consent_store_images, consent_training, _selected_locale_code())
            body_payload = body_result.to_dict()
            body_payload['posture_summary'] = posture_summary
            body_payload['capture_method'] = capture_method
            body_payload['unit_system'] = unit_system
            body_id = save_body_analysis(user_id, gender, body_payload, front_path, profile_path, back_path=back_path, extra_measurements=extra_clean, posture_summary=posture_summary)
            product_id = save_product(product_result.to_dict())
            rec_id = save_recommendation(user_id, body_id, product_id, rec)
            st.session_state['last_recommendation_id'] = rec_id

            calibration_saved = None
            part_calibration_saved = []
            if calibration_candidate and consent_training:
                photo_bucket = photo_quality_bucket(float(body_result.confidence), body_result.front_capture, body_result.profile_capture)
                clothing_fit_bucket = normalize_clothing_fit_bucket(clothing_fit)
                weight = calibration_sample_weight(float(body_result.confidence), clothing_fit=clothing_fit, photo_quality=photo_bucket)
                calibration_saved = save_calibration_sample(
                    user_id=user_id,
                    body_analysis_id=body_id,
                    recommendation_id=rec_id,
                    gender=gender,
                    body_type=body_result.body_type,
                    build_type=body_result.build_type,
                    confidence=float(body_result.confidence),
                    confidence_band=confidence_band(float(body_result.confidence)),
                    photo_quality_bucket=photo_bucket,
                    photo_quality_score=float(min(body_result.front_capture.get('score', body_result.confidence), body_result.profile_capture.get('score', body_result.confidence))),
                    product_kind=search_product_kind,
                    clothing_fit=clothing_fit,
                    clothing_fit_bucket=clothing_fit_bucket,
                    sample_weight=weight,
                    ai_raw_bust_cm=float(body_result.raw_bust_cm),
                    ai_raw_waist_cm=float(body_result.raw_waist_cm),
                    ai_raw_hips_cm=float(body_result.raw_hips_cm),
                    ai_displayed_bust_cm=float(body_result.suggested_bust_cm),
                    ai_displayed_waist_cm=float(body_result.suggested_waist_cm),
                    ai_displayed_hips_cm=float(body_result.suggested_hips_cm),
                    manual_bust_cm=float(used_bust),
                    manual_waist_cm=float(used_waist),
                    manual_hips_cm=float(used_hips),
                )
                for mkey, mval in extra_clean.items():
                    ai_raw_part = float((body_result.extra_estimates or {}).get(mkey, 0.0))
                    if ai_raw_part > 0 and float(mval) > 0:
                        pid = save_calibration_part_sample(
                            user_id=user_id,
                            body_analysis_id=body_id,
                            recommendation_id=rec_id,
                            gender=gender,
                            photo_quality_bucket=photo_bucket,
                            product_kind=search_product_kind,
                            clothing_fit_bucket=clothing_fit_bucket,
                            measure_key=mkey,
                            sample_weight=weight,
                            ai_raw_value_cm=ai_raw_part,
                            ai_displayed_value_cm=float((body_result.extra_estimates or {}).get(mkey, ai_raw_part)),
                            manual_value_cm=float(mval),
                        )
                        part_calibration_saved.append(pid)

            st.success(f'Analiza gotowa i zapisana. ID rekomendacji: {rec_id}')
            comparison_payload = None
            if calibration_saved is not None:
                st.info(f'Próbka kalibracyjna została zapisana. ID kalibracji: {calibration_saved}. Segment uczenia: {photo_bucket} / {search_product_kind} / {clothing_fit_bucket}. Ta korekta będzie uwzględniana przy kolejnych podobnych analizach.')
                if part_calibration_saved:
                    st.caption('Zapisano również kalibrację per part dla: ' + str(len(part_calibration_saved)) + ' dodatkowych wymiarów.')
                comparison_payload = compare_ai_vs_manual(
                    raw_measures={'bust': float(body_result.raw_bust_cm), 'waist': float(body_result.raw_waist_cm), 'hips': float(body_result.raw_hips_cm)},
                    displayed_measures={'bust': float(body_result.suggested_bust_cm), 'waist': float(body_result.suggested_waist_cm), 'hips': float(body_result.suggested_hips_cm)},
                    manual_measures={'bust': float(used_bust), 'waist': float(used_waist), 'hips': float(used_hips)},
                )
            visual_status = (visual_fit or {}).get('status', 'neutral')
            visual_bonus = {'helps': 8, 'risk': -10, 'neutral': 0, 'not_available': 0}.get(visual_status, 0)
            technical_fit_score = float(rec['fit_score'])
            visual_fit_score = max(0.0, min(100.0, float(rec['dress_match_score']) + visual_bonus))
            rec['technical_fit_score'] = technical_fit_score
            rec['visual_fit_score'] = visual_fit_score

            a, b, c, d = st.columns(4)
            st.caption(f'Kategoria fitu: {search_group} → {style_branch} → {search_product_kind}')
            a.metric('Rozmiar', rec['recommended_size'])
            b.metric('Alternatywa', rec['alternate_size'])
            c.metric('Fit techniczny', f"{technical_fit_score:.0f}/100")
            d.metric('Fit wizualny', f"{visual_fit_score:.0f}/100")

            st.markdown('**Rozdzielony wynik końcowy**')
            split1, split2 = st.columns(2)
            with split1:
                st.success('Fit techniczny')
                st.write('Ocena tego, czy produkt powinien pasować rozmiarowo i gdzie jest ryzyko ciasnoty / luzu.')
                st.write(f"Wynik: **{technical_fit_score:.0f}/100**")
            with split2:
                st.info('Fit wizualny')
                st.write('Ocena tego, czy fason pasuje do sylwetki i czy pomaga lub podkreśla wykryte cechy postawy.')
                st.write(f"Wynik: **{visual_fit_score:.0f}/100**")
            split3, split4 = st.columns(2)
            pfit = postural_fit_score(posture_summary or {}, visual_fit or {})
            mconf = float(body_result.confidence) * 100
            pconf = 0.0 if not posture_summary else (70.0 if posture_summary.get('detected') else 85.0) if posture_summary.get('available') else 35.0
            prconf = product_confidence_score(product_result)
            with split3:
                st.info('Fit posturalny')
                st.write('Ocena tego, czy fason neutralizuje, maskuje lub podkreśla wizualne cechy postawy.')
                st.write(f"Wynik: **{pfit:.0f}/100**")
            with split4:
                st.info('Confidence')
                st.write(f"Pomiar: **{mconf:.0f}/100**")
                st.write(f"Produkt: **{prconf:.0f}/100**")
                st.write(f"Postawa: **{pconf:.0f}/100**")

            verdict_icon = {'bierz': '✅', 'ostrożnie': '⚠️', 'raczej odpuść': '❌'}[rec['verdict']]
            st.subheader(f"Werdykt: {verdict_icon} {rec['verdict']}")
            left2, right2 = st.columns([1.2, 1])
            with left2:
                st.markdown('**' + tr('Komentarz AI') + '**')
                st.write(rec['explanation'])
            with right2:
                st.markdown('**' + tr('Punkty uwagi') + '**')
                if rec['risk_flags']:
                    for flag in rec['risk_flags']:
                        st.write(f"- {flag}")
                else:
                    st.write('Brak istotnych czerwonych flag.')
                st.markdown('**' + tr('Wymiary użyte do decyzji') + '**')
                st.json(rec['estimated_measurements'])
                extra_only = extra_measurements_summary(extra_clean)
                if extra_only:
                    st.markdown('**Dodatkowe wymiary zapisane**')
                    st.json(extra_only)
            render_visual_fit(visual_fit)
            if comparison_payload is not None:
                st.markdown('**Calibration loop — porównanie AI vs centymetr**')
                cmp_df = pd.DataFrame([
                    {
                        'strefa': 'biust/klatka',
                        'AI surowe': comparison_payload['raw_ai']['bust'],
                        'AI po korekcie': comparison_payload['displayed_ai']['bust'],
                        'centymetr': comparison_payload['manual']['bust'],
                        'błąd surowy': comparison_payload['raw_error']['bust'],
                        'błąd po korekcie': comparison_payload['displayed_error']['bust'],
                    },
                    {
                        'strefa': 'talia',
                        'AI surowe': comparison_payload['raw_ai']['waist'],
                        'AI po korekcie': comparison_payload['displayed_ai']['waist'],
                        'centymetr': comparison_payload['manual']['waist'],
                        'błąd surowy': comparison_payload['raw_error']['waist'],
                        'błąd po korekcie': comparison_payload['displayed_error']['waist'],
                    },
                    {
                        'strefa': 'biodra',
                        'AI surowe': comparison_payload['raw_ai']['hips'],
                        'AI po korekcie': comparison_payload['displayed_ai']['hips'],
                        'centymetr': comparison_payload['manual']['hips'],
                        'błąd surowy': comparison_payload['raw_error']['hips'],
                        'błąd po korekcie': comparison_payload['displayed_error']['hips'],
                    },
                ])
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)




with capture_tab:
    st.header('Capture Pro — integrated beta')
    st.caption('V8.6.8 łączy Capture Pro z główną analizą. Możesz zrobić FRONT / PROFIL / TYŁ, przejść quality gate i jednym kliknięciem zapisać zdjęcia do „Nowa analiza”.')
    c_a, c_b = st.columns([1.25, 1])
    with c_a:
        capture_gender = st.radio('Płeć w trybie capture', ['kobieta', 'mężczyzna'], horizontal=True, key='capture_pro_gender')
        capture_height = st.number_input('Wzrost do Capture Pro (cm)', min_value=145, max_value=210, value=int(st.session_state.get('capture_pro_height', 170)), key='capture_pro_height')
        capture_weight = st.number_input('Waga do Capture Pro (kg)', min_value=38, max_value=220, value=int(st.session_state.get('capture_pro_weight', 65)), key='capture_pro_weight')
        capture_age = st.number_input('Wiek do Capture Pro', min_value=16, max_value=85, value=int(st.session_state.get('capture_pro_age', 30)), key='capture_pro_age')
        capture_method_live = st.radio('Jak powstanie zdjęcie?', ['inna osoba trzyma telefon', 'telefon oparty stabilnie + timer', 'statyw', 'zdjęcie wykonane samemu'], horizontal=True, key='capture_pro_method')
        capture_fit = st.radio('Czy ubranie, w którym jesteś, jest wystarczająco dopasowane, żeby pomiar był dokładny?', ['obcisłe', 'raczej dopasowane', 'średnie', 'luźne', 'bardzo luźne', 'nie wiem'], horizontal=True, index=1, key='capture_pro_fit')
        st.caption('W V8.6.8 możesz zapisać sesję Capture Pro i wykorzystać ją bezpośrednio w głównej analizie.')
        cp_front = st.camera_input('Capture Pro — FRONT', key='capture_pro_front_capture')
        cp_profile = st.camera_input('Capture Pro — PROFIL', key='capture_pro_profile_capture')
        cp_back = st.camera_input('Capture Pro — TYŁ (opcjonalne)', key='capture_pro_back_capture')

        if st.button('Zweryfikuj i zapisz do głównej analizy', use_container_width=True, type='primary', key='capture_pro_save_btn'):
            if not cp_front or not cp_profile:
                st.error('Do zapisania sesji Capture Pro potrzebne są FRONT i PROFIL.')
            else:
                with st.spinner('Weryfikuję zdjęcia Capture Pro i przygotowuję sesję do głównej analizy...'):
                    front_bytes = cp_front.getvalue()
                    profile_bytes = cp_profile.getvalue()
                    back_bytes = cp_back.getvalue() if cp_back else None
                    cp_body = analyze_body(
                        front_bytes,
                        profile_bytes,
                        float(capture_height),
                        float(capture_weight),
                        int(capture_age),
                        gender=capture_gender,
                        clothing_fit_answer=capture_fit,
                        capture_method=capture_method_live,
                        back_image_bytes=back_bytes,
                    )
                    cp_posture = analyze_posture_from_images(
                        front_bytes,
                        profile_bytes,
                        gender=capture_gender,
                        clothing_fit_answer=capture_fit,
                        back_image_bytes=back_bytes,
                    )
                    measurement_ready = bool(cp_body.front_capture.get('measurement_ready')) and bool(cp_body.profile_capture.get('measurement_ready'))
                    posture_ready = measurement_ready and (bool(cp_body.back_capture.get('posture_ready')) if cp_body.back_capture else measurement_ready)
                    ok = measurement_ready
                    st.session_state['capture_pro_saved'] = {
                        'front_bytes': front_bytes,
                        'profile_bytes': profile_bytes,
                        'back_bytes': back_bytes,
                        'gender': capture_gender,
                        'height_cm': float(capture_height),
                        'weight_kg': float(capture_weight),
                        'age': int(capture_age),
                        'capture_method': capture_method_live,
                        'photo_clothing_fit': capture_fit,
                        'body_result': cp_body.to_dict(),
                        'posture_summary': cp_posture,
                        'accepted': bool(ok),
                        'measurement_ready': bool(measurement_ready),
                        'posture_ready': bool(posture_ready),
                    }
                    if ok:
                        if posture_ready:
                            st.success('Sesja Capture Pro zapisana. FRONT i PROFIL mają measurement_ready, a ocena postawy ma posture_ready. To najlepszy wariant do dalszej analizy.')
                        else:
                            st.success('Sesja Capture Pro zapisana. FRONT i PROFIL mają measurement_ready, ale ocena postawy nadal ma ograniczenia. Zadbaj szczególnie o lepszy TYŁ i bardziej neutralną perspektywę.')
                    else:
                        st.error('Sesja została zapisana, ale FRONT / PROFIL nie mają measurement_ready. Najpierw popraw zdjęcia albo użyj sesji tylko poglądowo.')
        if st.session_state.get('capture_pro_saved'):
            saved = st.session_state['capture_pro_saved']
            st.markdown('### Ostatnia zapisana sesja')
            st.write(f"- accepted: {'tak' if saved.get('accepted') else 'nie'}")
            st.write(f"- measurement_ready: {'tak' if saved.get('measurement_ready') else 'nie'}")
            st.write(f"- posture_ready: {'tak' if saved.get('posture_ready') else 'nie'}")
            st.write(f"- metoda: {saved.get('capture_method')}")
            st.write(f"- dopasowanie stroju: {saved.get('photo_clothing_fit')}")
            if st.button('Wyczyść zapisaną sesję Capture Pro', use_container_width=True, key='capture_pro_clear'):
                st.session_state.pop('capture_pro_saved', None)
                st.success('Usunięto zapisaną sesję Capture Pro.')

    with c_b:
        st.markdown('### Live helper')
        render_capture_pro_component(capture_gender, allow_selfie=(capture_method_live == 'zdjęcie wykonane samemu'), key_suffix='v75')
        saved = st.session_state.get('capture_pro_saved')
        if saved:
            st.markdown('### Podgląd zapisanej sesji')
            p1, p2, p3 = st.columns(3)
            with p1:
                if saved.get('front_bytes'):
                    st.image(saved['front_bytes'], caption='FRONT', use_container_width=True)
            with p2:
                if saved.get('profile_bytes'):
                    st.image(saved['profile_bytes'], caption='PROFIL', use_container_width=True)
            with p3:
                if saved.get('back_bytes'):
                    st.image(saved['back_bytes'], caption='TYŁ', use_container_width=True)
            body = saved.get('body_result', {})
            if body:
                st.caption(f"FRONT: {body.get('front_capture',{}).get('status','—')} | PROFIL: {body.get('profile_capture',{}).get('status','—')} | TYŁ: {body.get('back_capture',{}).get('status','—')}")
                render_capture_gate_summary(body.get('front_capture', {}), 'FRONT')
                render_capture_gate_summary(body.get('profile_capture', {}), 'PROFIL')
                if body.get('back_capture'):
                    render_capture_gate_summary(body.get('back_capture', {}), 'TYŁ')
                for note in body.get('notes', [])[:6]:
                    st.write(f"- {note}")


with mobile_tab:
    st.header(tr('Capture Mobile Studio'))
    st.caption('W 8.5 dołączony jest osobny, lekki pilot mobilny do testowania capture flow na telefonie. Możesz uruchomić go osobno, wyeksportować JSON sesji i zaimportować go tutaj do głównej analizy. Import pokazuje też skrócone podsumowanie jakości całej sesji.')
    st.info(tr('Pakiet mobile capture v3 jest dołączony w folderze capture_mobile_pilot w ZIP-ie 8.5.'))
    st.markdown('**Co znajdziesz w pakiecie mobile pilot**')
    st.write('- statyczną aplikację webową do testowania jakości zdjęć FRONT / PROFIL / TYŁ na telefonie')
    st.write('- live quality gate')
    st.write('- eksport JSON z sesji capture')
    st.write('- możliwość użycia zaimportowanej sesji w głównej analizie')
    mobile_json = st.file_uploader(tr('Wgraj plik JSON wyeksportowany z mobilnego pilota capture'), type=['json'], key='capture_mobile_json')
    if mobile_json is not None:
        try:
            imported = import_capture_mobile_json(mobile_json.getvalue())
            st.session_state['capture_mobile_import'] = imported
            st.success('Sesja mobile została odczytana.')
            c1, c2, c3 = st.columns(3)
            with c1:
                if imported.get('front_bytes'):
                    st.image(imported['front_bytes'], caption='FRONT z mobile pilot', use_container_width=True)
            with c2:
                if imported.get('profile_bytes'):
                    st.image(imported['profile_bytes'], caption='PROFIL z mobile pilot', use_container_width=True)
            with c3:
                if imported.get('back_bytes'):
                    st.image(imported['back_bytes'], caption='TYŁ z mobile pilot', use_container_width=True)
            summary = mobile_session_quality_summary(imported)
            st.write(f"- session_quality_score: {summary['score']}/100")
            st.write(f"- accepted: {'tak' if imported.get('accepted') else 'nie'}")
            st.write(f"- measurement_ready: {'tak' if imported.get('measurement_ready') else 'nie'}")
            st.write(f"- gender: {imported.get('gender')}")
            st.write(f"- capture_method: {imported.get('capture_method')}")
            st.write(f"- clothing_fit: {imported.get('photo_clothing_fit')}")
            st.write(f"- gotowe ujęcia: front={summary['ready']['front']}, profil={summary['ready']['profile']}, tył={summary['ready']['back']}")
            if st.button(tr('Użyj zaimportowanej sesji w głównej analizie'), use_container_width=True, key='use_mobile_import'):
                st.session_state['capture_pro_saved'] = imported
                st.success('Zaimportowana sesja została ustawiona jako źródło „Capture Pro session” w zakładce Nowa analiza.')
        except Exception as exc:
            st.error(f'Nie udało się odczytać sesji mobile: {exc}')


with search_tab:
    st.header(tr('Wyszukiwanie po zdjęciu'))
    st.caption('To jest rozszerzona wersja lokalnego wyszukiwania po zdjęciu — działa na katalogu demo oraz na lokalnym indeksie produktów zapisanych wcześniej z linków i analiz. To nadal nie jest pełny internetowy reverse image search klasy Google Lens, ale wynik jest szerszy niż w MVP 7.1.')
    st.info(tr('Podpowiedź: najlepiej działa zdjęcie samego produktu lub produktu na jasnym tle.'))
    indexed_products = list_products_for_visual_search(500)
    st.caption(f'Aktywny indeks lokalny: katalog demo + {len(indexed_products)} zapisanych produktów z wcześniejszych analiz. Ranking uwzględnia też wybraną kategorię produktu.')

    search_source = st.radio('Źródło zdjęcia produktu', [tr('Aparat (zalecane)'), tr('Upload z galerii')], horizontal=True, key='search_source')
    query_photo = None
    if search_source == tr('Aparat (zalecane)'):
        query_photo = st.camera_input(tr('Uruchom aparat produktu'), key='product_search_camera')
    else:
        query_photo = st.file_uploader(tr('Wgraj zdjęcie produktu'), type=['jpg','jpeg','png'], key='product_search_upload')

    if st.button(tr('Wyszukaj podobne produkty'), use_container_width=True, key='search_by_photo_btn'):
        if not query_photo:
            st.error(tr('Wgraj zdjęcie produktu'))
        else:
            with st.spinner(tr('Wyszukaj podobne produkty')):
                try:
                    results = search_index_by_photo(image_to_bytes(query_photo), APP_DIR, db_products=indexed_products, top_k=12, preferred_kind=st.session_state.get('last_search_product_kind'))
                    st.session_state['visual_search_results'] = results
                    st.session_state['visual_search_query_preview'] = query_photo.getvalue()
                except Exception as exc:
                    st.session_state['visual_search_results'] = []
                    st.error(f'Visual search error: {exc}')

    if st.session_state.get('visual_search_query_preview'):
        st.image(st.session_state['visual_search_query_preview'], caption='Zapytanie', width=220)

    results = st.session_state.get('visual_search_results', [])
    st.subheader(tr('Wyniki wyszukiwania wizualnego'))
    if not results:
        st.write(tr('Brak wyników wyszukiwania'))
    else:
        cols = st.columns(2)
        for i, row in enumerate(results):
            with cols[i % 2]:
                st.image(row.image_path, use_container_width=True)
                source_tag = 'katalog demo' if row.source_type == 'demo' else 'lokalny indeks'
                st.markdown(f"**{row.brand} — {row.name}**")
                st.caption(f"{row.category} • similarity {row.score}/100 • {source_tag}")
                if row.product_url:
                    st.write(row.product_url)
                st.caption(row.notes)
                if st.button(f"Użyj w analizie: {row.name}", key=f"use_search_{row.source_type}_{row.product_id}", use_container_width=True):
                    if row.source_type == 'demo':
                        st.session_state['product_result'] = build_demo_product(row.payload)
                    else:
                        st.session_state['product_result'] = build_db_product(row.payload)
                    st.success('Wybrano produkt z wyszukiwania wizualnego i ustawiono go w analizie.')




with annotation_tab:
    st.header('Annotation Review')
    st.caption('Import ręcznych adnotacji landmarków i porównanie overlayu aplikacji z poprawną anotacją. To jest pierwsza warstwa dataset review pod model pomiarowy.')
    st.download_button(
        'Pobierz szablon adnotacji CSV',
        data=template_annotation_csv_bytes(),
        file_name='ateena_annotation_template_v8_6.csv',
        mime='text/csv',
        use_container_width=True,
    )
    reviewer_email = st.text_input('Email reviewera / anotatora', key='annotation_reviewer_email')
    ann_image = st.file_uploader('Wgraj referencyjny obraz z anotacją (opcjonalne)', type=['jpg','jpeg','png'], key='annotation_image')
    ann_csv = st.file_uploader('Wgraj CSV z adnotacjami landmarków', type=['csv'], key='annotation_csv')
    if ann_image is not None:
        st.image(ann_image, caption='Obraz referencyjny z anotacją', use_container_width=True)

    current_body = st.session_state.get('body_result')
    if current_body is None:
        st.info('Najpierw zrób analizę w zakładce „Nowa analiza”, żeby porównać aktualny overlay aplikacji z ręczną anotacją.')
    else:
        st.write('Aktualny body_result jest gotowy do porównania.')
        if ann_csv is not None:
            try:
                ann_df = parse_annotation_csv(ann_csv.getvalue())
                comparison = compare_annotations(current_body, ann_df)
                summary = comparison['summary']
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric('Landmark rows', summary.get('landmark_total', 0))
                c2.metric('Found by app', summary.get('landmark_found', 0))
                c3.metric('Visibility acc', f"{summary.get('visibility_accuracy', 0)}%")
                c4.metric('Occlusion acc', f"{summary.get('occlusion_accuracy', 0)}%")
                c5.metric('Confidence acc', f"{summary.get('confidence_accuracy', 0)}%")
                if summary.get('mae_cm') is not None:
                    st.metric('MAE manual vs app (cm)', summary.get('mae_cm'))
                st.subheader('Porównanie landmarków')
                st.dataframe(comparison['landmark_comparison'], use_container_width=True, hide_index=True)
                st.subheader('Porównanie wymiarów')
                metric_df = comparison['metric_comparison']
                if not metric_df.empty:
                    st.dataframe(metric_df, use_container_width=True, hide_index=True)
                else:
                    st.info('W CSV nie podano żadnych manualnych wartości cm do porównania.')
                if st.button('Zapisz review adnotacji do bazy', use_container_width=True):
                    image_path = save_upload(ann_image, 'annotation_reference') if ann_image is not None else None
                    csv_path = save_upload(ann_csv, 'annotation_csv') if ann_csv is not None else None
                    review_id = save_annotation_review(reviewer_email or None, image_path, csv_path, summary)
                    st.success(f'Zapisano review adnotacji. ID: {review_id}')
            except Exception as exc:
                st.error(f'Nie udało się przetworzyć anotacji: {exc}')

    st.subheader('Ostatnie review adnotacji')
    reviews = recent_annotation_reviews(20)
    if reviews:
        rows = []
        for r in reviews:
            s = r.get('summary', {}) or {}
            rows.append({
                'id': r.get('id'),
                'reviewer_email': r.get('reviewer_email'),
                'landmark_total': s.get('landmark_total'),
                'found_by_app': s.get('landmark_found'),
                'visibility_acc_%': s.get('visibility_accuracy'),
                'occlusion_acc_%': s.get('occlusion_accuracy'),
                'confidence_acc_%': s.get('confidence_accuracy'),
                'mae_cm': s.get('mae_cm'),
                'created_at': r.get('created_at'),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info('Brak zapisanych review adnotacji.')

with qa_tab:
    st.header('QA / Admin')
    qa = qa_overview()
    q1, q2, q3, q4 = st.columns(4)
    q1.metric('Użytkownicy', qa.get('users', 0))
    q2.metric('Analizy sylwetki', qa.get('body_analyses', 0))
    q3.metric('Produkty', qa.get('products', 0))
    q4.metric('Rekomendacje', qa.get('recommendations', 0))
    q5, q6, q7, q8, q9 = st.columns(5)
    q5.metric('Feedback', qa.get('feedback', 0))
    q6.metric('Kalibracje', qa.get('calibration_examples', 0))
    q7.metric('OCR sesje', qa.get('ocr_sessions', 0))
    q8.metric('Zgłoszenia tłumaczeń', qa.get('translation_reports', 0))
    q9.metric('Zgody prywatności', qa.get('privacy_consents', 0))
    st.markdown('**Rozkład quality gate**')
    st.json(qa.get('quality_distribution', {}))
    st.markdown('**Najczęstsze komunikaty capture**')
    for msg, cnt in qa.get('top_capture_messages', []):
        st.write(f'- {msg} ({cnt})')

    st.markdown('**Prywatność i zgody**')
    cstats = consent_overview()
    st.json(cstats)

    st.markdown('**Calibration per part**')
    st.json(calibration_part_summary())

    st.markdown('### Zgłoś problem z tłumaczeniem')
    current_locale = _selected_locale_code()
    t_screen = st.selectbox('Ekran / sekcja', ['Nowa analiza', 'Wyszukiwanie po zdjęciu', 'Produkt', 'Wynik rekomendacji', 'Feedback', 'Inne'], key='tr_screen')
    t_source = st.text_input('Tekst źródłowy widoczny w aplikacji', key='tr_source')
    t_comment = st.text_area('Co brzmi źle / nienaturalnie?', key='tr_comment')
    t_email = st.text_input('Email zgłaszającego (opcjonalnie)', key='tr_email')
    if st.button('Zapisz zgłoszenie tłumaczenia', key='save_translation_report_btn'):
        if not t_source or not t_comment:
            st.error('Uzupełnij tekst źródłowy i komentarz.')
        else:
            rid = save_translation_report(current_locale, t_screen, t_source, t_comment, t_email or None)
            st.success(f'Zgłoszenie zapisane. ID: {rid}')
    tr_rows = recent_translation_reports(20)
    if tr_rows:
        st.markdown('**Ostatnie zgłoszenia tłumaczeń**')
        st.dataframe(pd.DataFrame(tr_rows), use_container_width=True, hide_index=True)

    ocr_rows = recent_ocr_sessions(20)
    if ocr_rows:
        st.markdown('**Ostatnie sesje OCR fallback**')
        st.dataframe(pd.DataFrame(ocr_rows), use_container_width=True, hide_index=True)

    cap_rows = recent_capture_sessions(20)
    if cap_rows:
        st.markdown('**Ostatnie sesje Capture Pro**')
        st.dataframe(pd.DataFrame(cap_rows), use_container_width=True, hide_index=True)


    latest_bench = st.session_state.get('latest_benchmark_metrics')
    if latest_bench:
        st.markdown('**Ostatni benchmark — skrót truth set**')
        if latest_bench.get('capture_truth_set'):
            st.json(latest_bench.get('capture_truth_set'))
        if latest_bench.get('per_part_summary'):
            st.dataframe(pd.DataFrame(latest_bench.get('per_part_summary', [])), use_container_width=True, hide_index=True)

    part_summary = calibration_part_summary().get('by_measure', {})
    if part_summary:
        st.markdown('**Calibration Loop 2.0 — per part**')
        rows = []
        for key, stats in part_summary.items():
            rows.append({
                'partia': key,
                'sample_count': stats.get('sample_count', 0),
                'total_weight': stats.get('total_weight', 0.0),
                'offset_cm': stats.get('offset', 0.0),
                'displayed_abs_error': stats.get('displayed_abs_error', 0.0),
            })
        st.dataframe(pd.DataFrame(rows).sort_values(['sample_count','total_weight'], ascending=False), use_container_width=True, hide_index=True)

with benchmark_tab:
    st.header('Benchmark')
    st.caption('Masz tu dwa tryby: benchmark syntetyczny demo oraz benchmark na własnym secie referencyjnym CSV.')
    col_syn, col_real = st.columns(2)
    with col_syn:
        st.markdown('### Benchmark demo')
        if st.button('Uruchom benchmark demo', key='run_benchmark_demo', use_container_width=True):
            metrics = run_benchmark(APP_DIR)
            run_id = save_benchmark_run('demo_synthetic_v7_4', metrics)
            st.success(f'Benchmark zapisany. ID: {run_id}')
            st.session_state['latest_benchmark_metrics'] = metrics
    with col_real:
        st.markdown('### Benchmark na własnym CSV')
        st.download_button('Pobierz szablon CSV', data=template_csv_bytes(), file_name='ateena_benchmark_template.csv', mime='text/csv', use_container_width=True)
        real_csv = st.file_uploader('Wgraj benchmark CSV', type=['csv'], key='real_benchmark_csv')
        if st.button('Uruchom benchmark na CSV', key='run_benchmark_real', use_container_width=True):
            if not real_csv:
                st.error('Najpierw wgraj plik CSV z benchmarkiem.')
            else:
                try:
                    import pandas as _pd
                    df = _pd.read_csv(real_csv)
                    metrics = evaluate_real_benchmark(df)
                    run_id = save_benchmark_run('real_reference_csv_v7_4', metrics)
                    st.success(f'Benchmark referencyjny zapisany. ID: {run_id}')
                    st.session_state['latest_benchmark_metrics'] = metrics
                except Exception as exc:
                    st.error(f'Nie udało się policzyć benchmarku referencyjnego: {exc}')
    latest_metrics = st.session_state.get('latest_benchmark_metrics')
    if latest_metrics:
        if 'scenario_count' in latest_metrics:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric('Scenariusze', latest_metrics.get('scenario_count', 0))
            b2.metric('Śr. fit techniczny', latest_metrics.get('avg_technical_fit', 0))
            b3.metric('Śr. fit wizualny', latest_metrics.get('avg_visual_fit', 0))
            b4.metric('Buy share', f"{latest_metrics.get('buy_share_pct', 0)}%")
        else:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric('Rekordy', latest_metrics.get('row_count', 0))
            b2.metric('MAE biust', latest_metrics.get('mae_bust_cm', 0))
            b3.metric('MAE talia', latest_metrics.get('mae_waist_cm', 0))
            b4.metric('MAE biodra', latest_metrics.get('mae_hips_cm', 0))
            extra_cols = st.columns(4)
            extra_cols[0].metric('MAE brzuch', latest_metrics.get('mae_abdomen_cm', 0))
            extra_cols[1].metric('MAE udo', latest_metrics.get('mae_thigh_cm', 0))
            extra_cols[2].metric('MAE biceps', latest_metrics.get('mae_arm_biceps_cm', 0))
            extra_cols[3].metric('MAE łydka', latest_metrics.get('mae_calf_max_cm', 0))
            if latest_metrics.get('capture_truth_set'):
                st.markdown('**Capture truth set**')
                cts = latest_metrics['capture_truth_set']
                c1, c2, c3 = st.columns(3)
                c1.metric('False accept', f"{cts.get('false_accept_rate_pct', 0)}%")
                c2.metric('False reject', f"{cts.get('false_reject_rate_pct', 0)}%")
                c3.metric('N truth set', cts.get('n', 0))
            if latest_metrics.get('per_part_summary'):
                st.markdown('**Per-part summary**')
                st.dataframe(pd.DataFrame(latest_metrics['per_part_summary']), use_container_width=True, hide_index=True)
            if latest_metrics.get('segment_mae'):
                st.markdown('**Segment MAE (gender × product kind)**')
                st.dataframe(pd.DataFrame(latest_metrics['segment_mae']), use_container_width=True, hide_index=True)
        st.json(latest_metrics)
    run_rows = recent_benchmark_runs(20)
    if run_rows:
        st.markdown('**Historia benchmarków**')
        flat = []
        for r in run_rows:
            row = {k:v for k,v in r.items() if k != 'metrics_json'}
            row.update(r.get('metrics_json', {}))
            flat.append(row)
        st.dataframe(pd.DataFrame(flat), use_container_width=True, hide_index=True)

with feedback_tab:
    st.header('Feedback po zakupie')
    st.caption('Ta sekcja buduje data loop — dzięki niej system uczy się, czy rekomendacje faktycznie trafiają.')
    feedback_email = st.text_input('Email użyty przy analizie', key='feedback_email')
    recs = recommendations_by_email(feedback_email) if feedback_email else []
    if feedback_email and not recs:
        st.info('Nie znaleziono zapisanych rekomendacji dla tego adresu email.')
    if recs:
        label_map = {f"#{r['id']} | {r['brand']} — {r['name']} | {r['recommended_size']} | {r['verdict']} | {r['created_at']}": r['id'] for r in recs}
        selected_label = st.selectbox('Wybierz analizę do oceny', list(label_map.keys()))
        purchased = st.radio('Czy produkt został kupiony?', ['tak', 'nie']) == 'tak'
        chosen_size = st.text_input('Jaki rozmiar został wybrany?', placeholder='np. M') if purchased else ''
        overall_fit_label = st.selectbox('Jak finalnie leżał produkt?', ['idealnie', 'trochę za ciasny', 'dużo za ciasny', 'trochę za luźny', 'dużo za luźny', 'nie kupiono'])
        problem_areas = st.multiselect('Gdzie był problem?', ['bust', 'waist', 'hips', 'shoulders', 'length'])
        returned = st.radio('Czy został zwrócony?', ['nie', 'tak']) == 'tak'
        comment = st.text_area('Dodatkowy komentarz', placeholder='Np. w talii było idealnie, ale biodra za ciasne.')
        if st.button('Zapisz feedback', use_container_width=True):
            feedback_id = save_feedback(label_map[selected_label], purchased, chosen_size or None, overall_fit_label, problem_areas, returned, comment)
            st.success(f'Feedback zapisany. ID: {feedback_id}')

with history_tab:
    st.header('Ostatnie zapisane rekomendacje')
    rows = recent_recommendations(50)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info('Na razie brak zapisanych analiz.')

    st.markdown('### Podsumowanie calibration loop')
    cal_summary = calibration_summary()
    o1, o2, o3, o4 = st.columns(4)
    o1.metric('Próbek kalibracyjnych', cal_summary['overall']['sample_count'])
    o2.metric('Waga łączna', cal_summary['overall']['total_weight'])
    o3.metric('Śr. offset biust/klatka', f"{cal_summary['overall']['offsets']['bust']:+.1f} cm")
    o4.metric('Śr. offset biodra', f"{cal_summary['overall']['offsets']['hips']:+.1f} cm")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown('**Kalibracja wg płci**')
        st.json(cal_summary['by_gender'])
    with g2:
        st.markdown('**Kalibracja wg jakości zdjęcia**')
        st.json(cal_summary['by_photo_quality'])

    g3, g4 = st.columns(2)
    with g3:
        st.markdown('**Kalibracja wg typu produktu**')
        st.json(cal_summary['by_product_kind'])
    with g4:
        st.markdown('**Kalibracja wg dopasowania stroju do zdjęcia**')
        st.json(cal_summary['by_clothing_fit_bucket'])

    st.markdown('### Ostatnie próbki kalibracyjne')
    cal_rows = calibration_overview(30)
    if cal_rows:
        st.dataframe(pd.DataFrame(cal_rows), use_container_width=True, hide_index=True)
    else:
        st.info('Na razie brak próbek kalibracyjnych. Zapiszą się, gdy użyjesz trybu „AI scan + ręczna korekta”.')
