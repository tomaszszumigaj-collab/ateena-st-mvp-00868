from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from sanity_engine import analyze_measurement_sanity
from landmark_schema import FRONT_CENTRAL, FRONT_LEFT_RIGHT, PROFILE_CORE, BACK_QA

try:
    import mediapipe as mp  # type: ignore
    HAS_MEDIAPIPE = True
except Exception:  # pragma: no cover
    mp = None
    HAS_MEDIAPIPE = False

BODY_TYPES = ["gruszka", "jabłko", "klepsydra", "prostokąt", "odwrócony trójkąt"]

POSE_NAMES = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

REQUIRED_FRONT = [
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
REQUIRED_PROFILE = [
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
HAND_GROUPS = {
    "left": ["left_wrist", "left_thumb", "left_index", "left_pinky"],
    "right": ["right_wrist", "right_thumb", "right_index", "right_pinky"],
}
FOOT_GROUPS = {
    "left": ["left_ankle", "left_heel", "left_foot_index"],
    "right": ["right_ankle", "right_heel", "right_foot_index"],
}


@dataclass
class CaptureQuality:
    orientation: str
    score: float
    status: str
    status_code: str
    messages: List[str]
    checks: Dict[str, bool]
    missing_points: List[str]
    detected_orientation: str
    orientation_confidence: float
    roll_deg: float
    sitting_detected: bool
    full_body_detected: bool
    camera_height_hint: str
    camera_pitch_hint: str
    clothing_fit_assessment: str
    vision_clothing_fit: str
    camera_angle_hint: str
    landmark_visibility_score: float
    hand_position_hint: str
    capture_method: str
    selfie_risk: bool
    distance_hint: str
    blockers: List[str]
    accept_ready: bool
    measurement_ready: bool
    posture_ready: bool
    background_cleanup_applied: bool
    background_cleanup_score: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SilhouetteResult:
    success: bool
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray
    confidence: float
    debug_image: np.ndarray
    image_shape: Tuple[int, int]
    landmarks: Dict[str, Tuple[float, float, float, float]]
    width_points: Dict[str, int]
    message: str = ""
    quality: Optional[CaptureQuality] = None
    cleaned_image: Optional[np.ndarray] = None
    background_cleanup_score: float = 0.0


@dataclass
class BodyAnalysisResult:
    body_type: str
    suggested_bust_cm: float
    suggested_waist_cm: float
    suggested_hips_cm: float
    raw_bust_cm: float
    raw_waist_cm: float
    raw_hips_cm: float
    build_type: str
    confidence: float
    notes: List[str]
    measurement_source: str
    front_capture: Dict
    profile_capture: Dict
    left_profile_capture: Dict
    right_profile_capture: Dict
    back_capture: Dict
    calibration_info: Dict
    measurement_confidence: Dict[str, float]
    extra_estimates: Dict[str, float]
    weak_points: List[str]
    sanity_report: Dict
    front_debug_image: Optional[np.ndarray] = None
    profile_debug_image: Optional[np.ndarray] = None
    right_profile_debug_image: Optional[np.ndarray] = None
    back_debug_image: Optional[np.ndarray] = None
    front_cleaned_image: Optional[np.ndarray] = None
    profile_cleaned_image: Optional[np.ndarray] = None
    right_profile_cleaned_image: Optional[np.ndarray] = None
    back_cleaned_image: Optional[np.ndarray] = None
    front_measure_overlay_image: Optional[np.ndarray] = None
    profile_measure_overlay_image: Optional[np.ndarray] = None
    right_profile_measure_overlay_image: Optional[np.ndarray] = None
    back_measure_overlay_image: Optional[np.ndarray] = None
    landmark_segment_report: Dict = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        data.pop("front_debug_image", None)
        data.pop("profile_debug_image", None)
        data.pop("right_profile_debug_image", None)
        data.pop("back_debug_image", None)
        data.pop("front_cleaned_image", None)
        data.pop("profile_cleaned_image", None)
        data.pop("right_profile_cleaned_image", None)
        data.pop("back_cleaned_image", None)
        data.pop("front_measure_overlay_image", None)
        data.pop("profile_measure_overlay_image", None)
        data.pop("right_profile_measure_overlay_image", None)
        data.pop("back_measure_overlay_image", None)
        return data




def _estimate_extra_measurements(
    bust_cm: float,
    waist_cm: float,
    hips_cm: float,
    height_cm: float,
    weight_kg: float,
    gender: str,
    measurement_confidence: Dict[str, float],
    front_quality: float,
    profile_quality: float,
    back_quality: float = 0.0,
) -> tuple[Dict[str, float], Dict[str, float]]:
    base_conf = float(max(0.18, min(0.92, (front_quality + profile_quality + max(0.0, back_quality)) / (2.0 if back_quality <= 0 else 3.0))))
    bmi = weight_kg / max((height_cm / 100) ** 2, 1e-6)
    hip_anchor = max(hips_cm, waist_cm + 8)
    chest_anchor = max(bust_cm, 72.0)

    if gender == 'mężczyzna':
        abdomen = waist_cm * 1.03 + max(0, bmi - 24) * 0.35
        thigh = hip_anchor * 0.59 + max(0, bmi - 24) * 0.35
        arm = chest_anchor * 0.33 + max(0, bmi - 24) * 0.18
        neck = chest_anchor * 0.39
        wrist = height_cm * 0.091
    else:
        abdomen = waist_cm * 1.04 + max(0, bmi - 24) * 0.30
        thigh = hip_anchor * 0.60 + max(0, bmi - 24) * 0.30
        arm = chest_anchor * 0.31 + max(0, bmi - 24) * 0.16
        neck = chest_anchor * 0.36
        wrist = height_cm * 0.088
    calf_max = thigh * 0.62
    calf_min = calf_max * 0.78

    extra = {
        'abdomen_cm': round(_clamp(abdomen, 58, 170), 1),
        'thigh_cm': round(_clamp(thigh, 42, 95), 1),
        'arm_biceps_cm': round(_clamp(arm, 20, 65), 1),
        'neck_cm': round(_clamp(neck, 26, 60), 1),
        'wrist_cm': round(_clamp(wrist, 11, 25), 1),
        'calf_max_cm': round(_clamp(calf_max, 24, 60), 1),
        'calf_min_cm': round(_clamp(calf_min, 18, 48), 1),
    }
    if extra['calf_min_cm'] >= extra['calf_max_cm']:
        extra['calf_min_cm'] = max(18.0, round(extra['calf_max_cm'] - 1.0, 1))

    conf = {
        'bust': measurement_confidence.get('bust', 0.2),
        'waist': measurement_confidence.get('waist', 0.2),
        'hips': measurement_confidence.get('hips', 0.2),
        'abdomen_cm': round(_clamp(measurement_confidence.get('waist', 0.2) * 0.88, 0.18, 0.9), 2),
        'thigh_cm': round(_clamp(measurement_confidence.get('hips', 0.2) * (0.78 + (0.08 if back_quality >= 0.75 else 0)), 0.18, 0.9), 2),
        'arm_biceps_cm': round(_clamp(measurement_confidence.get('bust', 0.2) * 0.68, 0.18, 0.85), 2),
        'neck_cm': round(_clamp(measurement_confidence.get('bust', 0.2) * 0.58, 0.18, 0.82), 2),
        'wrist_cm': round(_clamp(base_conf * 0.45, 0.18, 0.78), 2),
        'calf_max_cm': round(_clamp((measurement_confidence.get('hips', 0.2) + base_conf) / 2 * (0.62 + (0.08 if back_quality >= 0.75 else 0)), 0.18, 0.84), 2),
        'calf_min_cm': round(_clamp(base_conf * 0.48, 0.18, 0.76), 2),
    }
    return extra, conf


def _missing_capture_stub(label: str) -> Dict:
    return {
        'orientation': label,
        'score': 0.0,
        'status': 'nie dostarczono',
        'status_code': 'missing',
        'messages': [f'Nie dodano zdjęcia: {label}.'],
        'checks': {},
        'missing_points': [],
        'detected_orientation': 'unknown',
        'orientation_confidence': 0.0,
        'roll_deg': 0.0,
        'sitting_detected': False,
        'full_body_detected': False,
        'camera_height_hint': 'niepewne',
        'camera_pitch_hint': 'niepewne',
        'clothing_fit_assessment': 'unknown',
        'vision_clothing_fit': 'unknown',
        'camera_angle_hint': 'niepewne',
        'landmark_visibility_score': 0.0,
        'hand_position_hint': 'niepewne',
        'capture_method': 'unknown',
        'selfie_risk': False,
        'distance_hint': 'niepewne',
    }

@dataclass
class RatioPack:
    bust_to_hips: float
    waist_to_hips: float
    shoulders_to_hips: float
    volume_factor: float


def _decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Nie udało się odczytać obrazu.")
    return img


def _prepare_for_pose(img: np.ndarray) -> np.ndarray:
    # mild contrast normalization + sharpening improves landmark stability on phone images
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.merge([l2, a, b])
    out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    out = cv2.filter2D(out, -1, kernel)
    return out


def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _largest_contour_mask(mask: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None
    contour = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [contour], -1, color=1, thickness=cv2.FILLED)
    return clean, contour


def _grabcut_segment(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    rect = (max(8, int(w * 0.05)), max(8, int(h * 0.02)), max(12, int(w * 0.90)), max(12, int(h * 0.96)))
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    out = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    return out


def _threshold_segment(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() > 0.55:
        th = 1 - th
    return th.astype(np.uint8)



def _apply_background_cleanup(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    mask = _ensure_binary_mask(mask)
    h, w = image.shape[:2]
    if mask.sum() == 0:
        return image.copy(), 0.0
    # Feather edge for cleaner foreground on white background
    mask_u8 = (mask * 255).astype(np.uint8)
    feather = cv2.GaussianBlur(mask_u8, (11, 11), 0).astype(np.float32) / 255.0
    feather = np.clip(feather[..., None], 0.0, 1.0)
    white = np.full_like(image, 255)
    cleaned = (image.astype(np.float32) * feather + white.astype(np.float32) * (1.0 - feather)).astype(np.uint8)

    # background cleanup score: low border contamination + plausible foreground area
    border = np.concatenate([
        mask[: max(1, h // 30), :].ravel(),
        mask[-max(1, h // 30):, :].ravel(),
        mask[:, : max(1, w // 30)].ravel(),
        mask[:, -max(1, w // 30):].ravel(),
    ])
    border_fg = float(border.mean()) if border.size else 1.0
    area_ratio = float(mask.mean())
    area_score = 1.0 - min(1.0, abs(area_ratio - 0.24) / 0.24)
    score = _clamp((1.0 - border_fg) * 0.65 + area_score * 0.35, 0.0, 1.0)
    return cleaned, float(score)


def _pose_from_image(img: np.ndarray) -> tuple[Dict[str, Tuple[float, float, float, float]], np.ndarray, float]:
    # isolated internal helper that runs pose + segmentation once on provided image
    h, w = img.shape[:2]
    landmarks: Dict[str, Tuple[float, float, float, float]] = {}
    mask = np.zeros((h, w), dtype=np.uint8)
    pose_conf = 0.0

    if HAS_MEDIAPIPE:
        pose = _pose_estimator()
        prepared = _prepare_for_pose(img)
        rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            vis_vals = []
            for name, idx in POSE_NAMES.items():
                lm = results.pose_landmarks.landmark[idx]
                px = float(lm.x * w)
                py = float(lm.y * h)
                vis = float(getattr(lm, 'visibility', 0.0))
                z = float(getattr(lm, 'z', 0.0))
                if vis >= 0.30 and -0.15 * w <= px <= 1.15 * w and -0.15 * h <= py <= 1.15 * h:
                    landmarks[name] = (px, py, vis, z)
                vis_vals.append(vis)
            if vis_vals:
                pose_conf = float(np.mean(vis_vals))
        if getattr(results, 'segmentation_mask', None) is not None:
            mask = (results.segmentation_mask > 0.52).astype(np.uint8)

    if mask.sum() == 0:
        try:
            mask = _grabcut_segment(img)
        except Exception:
            mask = _threshold_segment(img)
    mask = _ensure_binary_mask(mask)
    return landmarks, mask, pose_conf

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _pt(landmarks: Dict[str, Tuple[float, float, float, float]], name: str) -> Optional[Tuple[float, float, float, float]]:
    return landmarks.get(name)


def _mean_pts(landmarks: Dict[str, Tuple[float, float, float, float]], names: List[str]) -> Optional[Tuple[float, float]]:
    pts = [landmarks[n] for n in names if n in landmarks]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (float(np.mean(xs)), float(np.mean(ys)))


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    n1 = np.linalg.norm(ba)
    n2 = np.linalg.norm(bc)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = float(np.dot(ba, bc) / (n1 * n2))
    cosang = _clamp(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


@lru_cache(maxsize=1)
def _pose_estimator():
    if not HAS_MEDIAPIPE:
        return None
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
    )



def _extract_pose_and_mask(img: np.ndarray) -> Tuple[Dict[str, Tuple[float, float, float, float]], np.ndarray, float, np.ndarray, float, bool]:
    h, w = img.shape[:2]
    landmarks, mask, pose_conf = _pose_from_image(img)
    cleaned, bg_score = _apply_background_cleanup(img, mask)
    cleanup_applied = bool(bg_score >= 0.45)

    # second pass on background-cleaned image if first pass was weak or background score is mediocre
    if HAS_MEDIAPIPE and cleanup_applied:
        landmarks2, mask2, pose_conf2 = _pose_from_image(cleaned)
        vis1 = _visibility_score(landmarks, list(landmarks.keys())) if landmarks else 0.0
        vis2 = _visibility_score(landmarks2, list(landmarks2.keys())) if landmarks2 else 0.0
        if (pose_conf2 + vis2 * 0.5) > (pose_conf + vis1 * 0.5):
            landmarks, pose_conf = landmarks2, pose_conf2
            if mask2.sum() > 0:
                mask = mask2
    mask = _ensure_binary_mask(mask)
    return landmarks, mask, pose_conf, cleaned, bg_score, cleanup_applied

def _bbox_from_mask_or_landmarks(mask: np.ndarray, landmarks: Dict[str, Tuple[float, float, float, float]], shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    h, w = shape
    clean_mask, contour = _largest_contour_mask(mask)
    if contour is not None:
        x, y, bw, bh = cv2.boundingRect(contour)
        return int(x), int(y), int(bw), int(bh)
    if landmarks:
        xs = [p[0] for p in landmarks.values()]
        ys = [p[1] for p in landmarks.values()]
        x0, y0, x1, y1 = int(max(0, min(xs))), int(max(0, min(ys))), int(min(w - 1, max(xs))), int(min(h - 1, max(ys)))
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)
    return 0, 0, 0, 0


def _width_from_mask(mask: np.ndarray, y: int, center_x: Optional[float] = None, band: int = 3) -> int:
    h, w = mask.shape[:2]
    if y < 0 or y >= h:
        return 0
    center_x = center_x if center_x is not None else w / 2
    ys = range(max(0, y - band), min(h, y + band + 1))
    widths: List[int] = []
    for yy in ys:
        xs = np.where(mask[yy] > 0)[0]
        if len(xs) <= 5:
            continue
        breaks = np.where(np.diff(xs) > 1)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(xs) - 1]
        best_seg = None
        best_dist = None
        for st, en in zip(starts, ends):
            left = int(xs[st])
            right = int(xs[en])
            seg_center = (left + right) / 2
            dist = abs(seg_center - center_x)
            if best_seg is None or dist < best_dist:
                best_seg = (left, right)
                best_dist = dist
        if best_seg:
            widths.append(int(best_seg[1] - best_seg[0] + 1))
    return int(np.median(widths)) if widths else 0


def _visibility_score(landmarks: Dict[str, Tuple[float, float, float, float]], names: List[str]) -> float:
    vals = [landmarks[n][2] for n in names if n in landmarks]
    return float(np.mean(vals)) if vals else 0.0


def _hand_visible(landmarks: Dict[str, Tuple[float, float, float, float]], side: str) -> bool:
    pts = HAND_GROUPS[side]
    visible_count = sum(1 for p in pts if p in landmarks and landmarks[p][2] >= 0.40)
    return visible_count >= 2 and f"{side}_wrist" in landmarks


def _foot_visible(landmarks: Dict[str, Tuple[float, float, float, float]], side: str) -> bool:
    pts = FOOT_GROUPS[side]
    return all(p in landmarks and landmarks[p][2] >= 0.40 for p in pts)


def _classify_orientation(landmarks: Dict[str, Tuple[float, float, float, float]]) -> Tuple[str, float]:
    left_sh = _pt(landmarks, "left_shoulder")
    right_sh = _pt(landmarks, "right_shoulder")
    left_hip = _pt(landmarks, "left_hip")
    right_hip = _pt(landmarks, "right_hip")
    nose = _pt(landmarks, "nose")
    ankles = _mean_pts(landmarks, ["left_ankle", "right_ankle", "left_foot_index", "right_foot_index"])
    if not (left_sh and right_sh and left_hip and right_hip and nose and ankles):
        return "unknown", 0.0

    body_height = max(1.0, ankles[1] - nose[1])
    shoulder_span = abs(left_sh[0] - right_sh[0]) / body_height
    hip_span = abs(left_hip[0] - right_hip[0]) / body_height
    avg_span = (shoulder_span + hip_span) / 2
    sh_z = abs(left_sh[3] - right_sh[3])
    hip_z = abs(left_hip[3] - right_hip[3])
    avg_z = (sh_z + hip_z) / 2

    if avg_span >= 0.23 and avg_z <= 0.12:
        conf = _clamp(0.65 + min(0.30, (avg_span - 0.23) * 2.5), 0.0, 0.98)
        return "front", conf
    if avg_span <= 0.14 and avg_z >= 0.12:
        conf = _clamp(0.65 + min(0.28, (0.14 - avg_span) * 2.8) + min(0.12, (avg_z - 0.12)), 0.0, 0.98)
        return "profile", conf
    return "half_profile", _clamp(0.45 + min(0.25, abs(avg_span - 0.18) * 1.8), 0.0, 0.8)


def _camera_roll_deg(landmarks: Dict[str, Tuple[float, float, float, float]]) -> float:
    lines = []
    for a, b in [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"), ("left_ankle", "right_ankle")]:
        pa = _pt(landmarks, a)
        pb = _pt(landmarks, b)
        if pa and pb:
            dy = pb[1] - pa[1]
            dx = pb[0] - pa[0]
            if abs(dx) > 1:
                lines.append(float(np.degrees(np.arctan2(dy, dx))))
    if not lines:
        return 0.0
    roll = float(np.mean(lines))
    if abs(roll) > 90:
        roll = abs(roll) - 180
    return roll


def _standing_or_sitting(landmarks: Dict[str, Tuple[float, float, float, float]]) -> Tuple[bool, bool, str]:
    left_hip = _pt(landmarks, "left_hip")
    right_hip = _pt(landmarks, "right_hip")
    left_knee = _pt(landmarks, "left_knee")
    right_knee = _pt(landmarks, "right_knee")
    left_ankle = _pt(landmarks, "left_ankle")
    right_ankle = _pt(landmarks, "right_ankle")
    if not all([left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]):
        return False, False, "Brak kompletu punktów nóg do oceny pozycji stojącej."

    left_angle = _angle((left_hip[0], left_hip[1]), (left_knee[0], left_knee[1]), (left_ankle[0], left_ankle[1]))
    right_angle = _angle((right_hip[0], right_hip[1]), (right_knee[0], right_knee[1]), (right_ankle[0], right_ankle[1]))
    knee_ok = left_angle >= 155 and right_angle >= 155
    y_ok = left_hip[1] < left_knee[1] < left_ankle[1] and right_hip[1] < right_knee[1] < right_ankle[1]
    sitting = (left_angle < 142 or right_angle < 142) or not y_ok
    if knee_ok and y_ok:
        return True, False, ""
    return False, sitting, "Użytkownik wygląda na siedzącego albo nogi są zbyt mocno zgięte — wymagane jest zdjęcie na stojąco."


def _camera_height_hint(landmarks: Dict[str, Tuple[float, float, float, float]], shape: Tuple[int, int]) -> str:
    h, _ = shape
    nose = _pt(landmarks, "nose")
    hips = _mean_pts(landmarks, ["left_hip", "right_hip"])
    ankles = _mean_pts(landmarks, ["left_ankle", "right_ankle", "left_foot_index", "right_foot_index"])
    if not (nose and hips and ankles):
        return "niepewne"
    body_h = max(1.0, ankles[1] - nose[1])
    hips_ratio = (hips[1] - nose[1]) / body_h
    hips_in_frame = hips[1] / max(h, 1)
    if 0.46 <= hips_ratio <= 0.63 and 0.45 <= hips_in_frame <= 0.68:
        return "około wysokości bioder"
    if hips_ratio < 0.46:
        return "kamera wygląda na zbyt nisko"
    return "kamera wygląda na zbyt wysoko"


def _camera_pitch_hint(landmarks: Dict[str, Tuple[float, float, float, float]], shape: Tuple[int, int]) -> str:
    h, _ = shape
    nose = _pt(landmarks, "nose")
    shoulders = _mean_pts(landmarks, ["left_shoulder", "right_shoulder"])
    hips = _mean_pts(landmarks, ["left_hip", "right_hip"])
    ankles = _mean_pts(landmarks, ["left_ankle", "right_ankle", "left_foot_index", "right_foot_index"])
    if not (nose and shoulders and hips and ankles):
        return "niepewne"
    head_to_sh = max(1.0, shoulders[1] - nose[1])
    sh_to_hip = max(1.0, hips[1] - shoulders[1])
    hip_to_ankle = max(1.0, ankles[1] - hips[1])
    ratio = (head_to_sh / sh_to_hip, sh_to_hip / hip_to_ankle)
    if ratio[1] < 0.42:
        return "kamera patrzy z góry"
    if ratio[1] > 0.78:
        return "kamera patrzy z dołu"
    return "perspektywa zbliżona do neutralnej"


def _camera_angle_hint(roll_deg: float) -> str:
    a = abs(roll_deg)
    if a <= 2.5:
        return "kamera prawie prosto"
    if a <= 6.5:
        return "kamera lekko przechylona"
    return "kamera wyraźnie przechylona"


def _distance_hint(bbox: Tuple[int, int, int, int], shape: Tuple[int, int]) -> str:
    _, _, _, bh = bbox
    h, _ = shape
    coverage = bh / max(h, 1)
    if coverage < 0.74:
        return "za daleko od kamery"
    if coverage > 0.96:
        return "za blisko kamery"
    return "dystans wygląda poprawnie"


def _vision_clothing_fit(mask: np.ndarray, landmarks: Dict[str, Tuple[float, float, float, float]], bbox: Tuple[int, int, int, int]) -> str:
    left_sh = _pt(landmarks, "left_shoulder")
    right_sh = _pt(landmarks, "right_shoulder")
    left_hip = _pt(landmarks, "left_hip")
    right_hip = _pt(landmarks, "right_hip")
    if not (left_sh and right_sh and left_hip and right_hip):
        return "unknown"
    shoulder_y = int((left_sh[1] + right_sh[1]) / 2)
    hip_y = int((left_hip[1] + right_hip[1]) / 2)
    shoulder_landmark_span = max(1.0, abs(left_sh[0] - right_sh[0]))
    hip_landmark_span = max(1.0, abs(left_hip[0] - right_hip[0]))
    shoulder_mask_width = _width_from_mask(mask, shoulder_y, center_x=(left_sh[0] + right_sh[0]) / 2)
    hip_mask_width = _width_from_mask(mask, hip_y, center_x=(left_hip[0] + right_hip[0]) / 2)
    ratio = np.mean([
        shoulder_mask_width / shoulder_landmark_span if shoulder_landmark_span else 1.0,
        hip_mask_width / hip_landmark_span if hip_landmark_span else 1.0,
    ])
    if ratio <= 1.18:
        return "fitted"
    if ratio <= 1.30:
        return "regular"
    if ratio <= 1.42:
        return "loose"
    return "very_loose"


def _hand_position_hint(landmarks: Dict[str, Tuple[float, float, float, float]], bbox: Tuple[int, int, int, int]) -> Tuple[bool, str]:
    ls = _pt(landmarks, "left_shoulder")
    rs = _pt(landmarks, "right_shoulder")
    lw = _pt(landmarks, "left_wrist")
    rw = _pt(landmarks, "right_wrist")
    lh = _pt(landmarks, "left_hip")
    rh = _pt(landmarks, "right_hip")
    if not all([ls, rs, lw, rw, lh, rh]):
        return False, "brak kompletu punktów rąk"
    shoulder_span = max(1.0, abs(ls[0] - rs[0]))
    left_gap = abs(lw[0] - lh[0]) / shoulder_span
    right_gap = abs(rw[0] - rh[0]) / shoulder_span
    gap = min(left_gap, right_gap)
    if 0.18 <= gap <= 0.65:
        return True, "ręce lekko odsunięte od tułowia"
    if gap < 0.18:
        return False, "ręce są zbyt blisko tułowia"
    return False, "ręce są zbyt szeroko odsunięte"


def _draw_quality_overlay(img: np.ndarray, sil: SilhouetteResult, quality: CaptureQuality) -> np.ndarray:
    debug = sil.debug_image.copy() if sil.debug_image is not None else img.copy()
    x, y, bw, bh = sil.bbox
    color = (20, 184, 166) if quality.status_code == "accept" else (245, 158, 11) if quality.status_code == "retry" else (239, 68, 68)
    cv2.rectangle(debug, (x, y), (x + bw, y + bh), color, 2)
    for name, pt in sil.landmarks.items():
        px, py = int(pt[0]), int(pt[1])
        cv2.circle(debug, (px, py), 3, (255, 0, 255), -1)
    cv2.putText(debug, f"Status: {quality.status}", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    cv2.putText(debug, f"Orientacja: {quality.detected_orientation} ({quality.orientation_confidence:.2f}) | Roll: {quality.roll_deg:.1f} deg", (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(debug, f"Kamera: {quality.camera_height_hint} | {quality.camera_pitch_hint} | {quality.distance_hint}", (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(debug, f"Strój: deklaracja={quality.clothing_fit_assessment} / wizja={quality.vision_clothing_fit} | dłonie={quality.hand_position_hint}", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(debug, f"Cleanup tła: {'tak' if quality.background_cleanup_applied else 'nie'} | score={quality.background_cleanup_score:.2f}", (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (30, 30, 30), 1, cv2.LINE_AA)
    yy = 146
    for msg in quality.messages[:5]:
        cv2.putText(debug, f"- {msg[:88]}", (20, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(debug, f"- {msg[:88]}", (20, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (40, 40, 40), 1, cv2.LINE_AA)
        yy += 20
    return debug


def _extract_silhouette(image_bytes: bytes, expected_orientation: str, clothing_fit_answer: str, capture_method: str = "inna osoba trzyma telefon") -> SilhouetteResult:
    img = _decode_image(image_bytes)
    h0, w0 = img.shape[:2]
    scale = min(1.0, 1500 / max(h0, w0))
    if scale != 1.0:
        img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))
    h, w = img.shape[:2]

    landmarks, mask, pose_conf, cleaned_img, bg_cleanup_score, bg_cleanup_applied = _extract_pose_and_mask(img)
    mask = _ensure_binary_mask(mask)
    x, y, bw, bh = _bbox_from_mask_or_landmarks(mask, landmarks, (h, w))

    if bw <= 0 or bh <= 0:
        return SilhouetteResult(False, (0, 0, 0, 0), mask, 0.0, img, (h, w), landmarks, {}, "Nie wykryto użytecznej sylwetki.", cleaned_image=cleaned_img, background_cleanup_score=bg_cleanup_score)

    bbox_area_ratio = (bw * bh) / max(1.0, h * w)
    landmark_vis = _visibility_score(landmarks, list(landmarks.keys())) if landmarks else 0.0
    conf = _clamp(0.20 + pose_conf * 0.40 + landmark_vis * 0.25 + min(0.15, bbox_area_ratio * 0.7), 0.0, 0.97)

    torso_center = _mean_pts(landmarks, ["left_hip", "right_hip"])
    center_x = torso_center[0] if torso_center else x + bw / 2

    sh = _mean_pts(landmarks, ["left_shoulder", "right_shoulder"])
    hp = _mean_pts(landmarks, ["left_hip", "right_hip"])
    head_y = int(_pt(landmarks, "nose")[1]) if _pt(landmarks, "nose") else y
    if sh and hp:
        torso_h = max(1.0, hp[1] - sh[1])
        bust_y = int(sh[1] + 0.18 * torso_h)
        waist_y = int(sh[1] + 0.46 * torso_h)
        hips_y = int(hp[1] + 0.04 * torso_h)
        shoulder_y = int(sh[1])
    else:
        shoulder_y = y + int(bh * 0.18)
        bust_y = y + int(bh * 0.27)
        waist_y = y + int(bh * 0.43)
        hips_y = y + int(bh * 0.58)

    width_points = {
        "shoulders": _width_from_mask(mask, shoulder_y, center_x=center_x, band=max(2, int(bh * 0.008))),
        "bust": _width_from_mask(mask, bust_y, center_x=center_x, band=max(2, int(bh * 0.01))),
        "waist": _width_from_mask(mask, waist_y, center_x=center_x, band=max(2, int(bh * 0.01))),
        "hips": _width_from_mask(mask, hips_y, center_x=center_x, band=max(2, int(bh * 0.01))),
        "neck": _width_from_mask(mask, int(head_y + 0.08 * bh), center_x=center_x, band=max(2, int(bh * 0.006))),
    }

    sil = SilhouetteResult(True, (x, y, bw, bh), mask, conf, img.copy(), (h, w), landmarks, width_points, cleaned_image=cleaned_img, background_cleanup_score=bg_cleanup_score)
    quality = _evaluate_capture(sil, expected_orientation, clothing_fit_answer, capture_method)
    sil.quality = quality
    sil.debug_image = _draw_quality_overlay(img, sil, quality)
    return sil


def _evaluate_capture(sil: SilhouetteResult, orientation_expected: str, clothing_fit_answer: str, capture_method: str = "inna osoba trzyma telefon") -> CaptureQuality:
    h, w = sil.image_shape
    x, y, bw, bh = sil.bbox
    messages: List[str] = []
    checks: Dict[str, bool] = {}
    missing_points: List[str] = []

    if not sil.success:
        return CaptureQuality(
            orientation=orientation_expected,
            score=0.0,
            status="odrzuć zdjęcie",
            status_code="reject",
            messages=[sil.message or "Nie udało się wiarygodnie wykryć sylwetki."],
            checks={"sylwetka_wykryta": False},
            missing_points=[],
            detected_orientation="unknown",
            orientation_confidence=0.0,
            roll_deg=0.0,
            sitting_detected=False,
            full_body_detected=False,
            camera_height_hint="niepewne",
            camera_pitch_hint="niepewne",
            clothing_fit_assessment=clothing_fit_answer,
            vision_clothing_fit="unknown",
            camera_angle_hint="niepewne",
            landmark_visibility_score=0.0,
            hand_position_hint="niepewne",
            capture_method=capture_method,
            selfie_risk=(capture_method == "zdjęcie wykonane samemu"),
            distance_hint="niepewne",
            blockers=["brak wiarygodnie wykrytej sylwetki"],
            accept_ready=False,
            measurement_ready=False,
            posture_ready=False,
            background_cleanup_applied=False,
            background_cleanup_score=0.0,
        )

    landmarks = sil.landmarks
    required = REQUIRED_FRONT if orientation_expected == "front" else REQUIRED_PROFILE
    for name in required:
        if name not in landmarks or landmarks[name][2] < 0.40:
            missing_points.append(name)

    landmark_visibility = _visibility_score(landmarks, required)
    checks["landmarki_kluczowe"] = len(missing_points) == 0 and landmark_visibility >= 0.68
    if not checks["landmarki_kluczowe"]:
        messages.append("Brakuje kluczowych punktów ciała do wiarygodnej analizy.")

    top_margin = y / max(h, 1)
    bottom_margin = (h - (y + bh)) / max(h, 1)
    centered = abs((x + bw / 2) - w / 2) / max(w / 2, 1)
    coverage = bh / max(h, 1)
    checks["tło_po_cleanup_ok"] = sil.background_cleanup_score >= 0.46
    if not checks["tło_po_cleanup_ok"]:
        messages.append("Tło nadal utrudnia wyraźne odcięcie sylwetki — spróbuj prostszego, bardziej kontrastowego tła.")
    checks["pełna_sylwetka"] = 0.80 <= coverage <= 0.96 and top_margin <= 0.11 and bottom_margin <= 0.05
    if not checks["pełna_sylwetka"]:
        messages.append("Wymagane jest pełne ujęcie od głowy do stóp — obecny kadr jest zbyt ciasny albo niepełny.")

    checks["w_środku_kadru"] = centered <= 0.14
    if not checks["w_środku_kadru"]:
        messages.append("Stań bliżej środka kadru.")

    detected_orientation, orientation_conf = _classify_orientation(landmarks)
    checks["poprawna_orientacja"] = detected_orientation == orientation_expected and orientation_conf >= 0.72
    if not checks["poprawna_orientacja"]:
        if detected_orientation == "half_profile":
            messages.append("To wygląda na półprofil — ustaw się wyraźnie frontem albo idealnie bokiem.")
        elif detected_orientation == "unknown":
            messages.append("Nie udało się jednoznacznie rozpoznać ustawienia ciała.")
        else:
            messages.append(f"Wykryto ustawienie: {detected_orientation}. Dla tego ujęcia wymagane jest: {orientation_expected}.")

    standing_ok, sitting_detected, standing_msg = _standing_or_sitting(landmarks)
    checks["pozycja_stojąca"] = standing_ok
    if not standing_ok:
        messages.append(standing_msg)

    roll_deg = _camera_roll_deg(landmarks)
    checks["kamera_nieprzechylona"] = abs(roll_deg) <= 5.5
    if not checks["kamera_nieprzechylona"]:
        messages.append("Kamera jest zbyt mocno przechylona — ustaw telefon prościej względem ciała.")

    cam_height_hint = _camera_height_hint(landmarks, sil.image_shape)
    checks["wysokość_kamery_ok"] = cam_height_hint == "około wysokości bioder"
    if not checks["wysokość_kamery_ok"]:
        messages.append("Telefon powinien być mniej więcej na wysokości bioder.")

    cam_pitch_hint = _camera_pitch_hint(landmarks, sil.image_shape)
    checks["perspektywa_ok"] = cam_pitch_hint == "perspektywa zbliżona do neutralnej"
    if not checks["perspektywa_ok"]:
        messages.append("Perspektywa kamery jest nieprawidłowa — unikaj zdjęcia z góry i z dołu.")

    distance_hint = _distance_hint(sil.bbox, sil.image_shape)
    checks["dystans_ok"] = distance_hint == "dystans wygląda poprawnie"
    if not checks["dystans_ok"]:
        messages.append("Odsuń się lub podejdź — obecny dystans od kamery utrudnia pomiar.")

    left_hand_ok = _hand_visible(landmarks, "left")
    right_hand_ok = _hand_visible(landmarks, "right")
    left_foot_ok = _foot_visible(landmarks, "left")
    right_foot_ok = _foot_visible(landmarks, "right")
    checks["dłonie_widoczne"] = left_hand_ok and right_hand_ok
    checks["stopy_widoczne"] = left_foot_ok and right_foot_ok
    if not checks["dłonie_widoczne"]:
        messages.append("Pokaż obie dłonie / nadgarstki — ręce są potrzebne do wiarygodnej oceny pozycji ciała.")
    if not checks["stopy_widoczne"]:
        messages.append("Pokaż całe stopy — bez nich aplikacja nie powinna estymować sylwetki.")

    hands_ok, hand_hint = _hand_position_hint(landmarks, sil.bbox)
    checks["ręce_poprawnie_ustawione"] = hands_ok
    if not hands_ok:
        messages.append("Odsuń ręce od ciała o około 10–15 cm, ale nie rozstawiaj ich szeroko.")

    vision_fit = _vision_clothing_fit(sil.mask, landmarks, sil.bbox)
    checks["ubranie_ok"] = not (vision_fit == "very_loose" or clothing_fit_answer == "bardzo luźne")
    if vision_fit in {"loose", "very_loose"} or clothing_fit_answer in {"luźne", "bardzo luźne", "nie wiem"}:
        messages.append("Ubranie może być zbyt luźne do dokładnego pomiaru — dla najlepszej estymacji potrzebny jest bardziej dopasowany strój.")
    if clothing_fit_answer in {"obcisłe", "raczej dopasowane"} and vision_fit in {"loose", "very_loose"}:
        messages.append("Deklaracja stroju i ocena z obrazu są niespójne — model widzi więcej luzu niż deklarujesz.")

    selfie_risk = capture_method == "zdjęcie wykonane samemu"
    checks["brak_selfie_risk"] = not selfie_risk
    if selfie_risk:
        messages.append("Zdjęcie wykonane samemu zwykle zniekształca proporcje całej sylwetki — traktuj wynik ostrożnie.")

    weighted = {
        "landmarki_kluczowe": 0.16,
        "pełna_sylwetka": 0.15,
        "tło_po_cleanup_ok": 0.07,
        "stopy_widoczne": 0.12,
        "dłonie_widoczne": 0.10,
        "pozycja_stojąca": 0.12,
        "poprawna_orientacja": 0.12,
        "kamera_nieprzechylona": 0.06,
        "wysokość_kamery_ok": 0.06,
        "perspektywa_ok": 0.04,
        "dystans_ok": 0.03,
        "ręce_poprawnie_ustawione": 0.02,
        "ubranie_ok": 0.01,
    }
    score = 0.0
    for k, wgt in weighted.items():
        score += wgt if checks.get(k, False) else 0.0
    score = _clamp(score * 0.88 + sil.confidence * 0.12, 0.0, 0.99)
    if selfie_risk:
        score *= 0.90

    hard_fail = (
        not checks["landmarki_kluczowe"]
        or not checks["pełna_sylwetka"]
        or not checks["stopy_widoczne"]
        or not checks["dłonie_widoczne"]
        or not checks["pozycja_stojąca"]
        or sil.background_cleanup_score < 0.22
        or detected_orientation == "half_profile"
        or not checks["poprawna_orientacja"]
        or abs(roll_deg) > 10.0
        or cam_pitch_hint != "perspektywa zbliżona do neutralnej"
        or clothing_fit_answer == "bardzo luźne"
        or vision_fit == "very_loose"
    )
    soft_fail = (
        not hard_fail and (
            not checks["wysokość_kamery_ok"]
            or not checks["dystans_ok"]
            or not checks["ręce_poprawnie_ustawione"]
            or vision_fit == "loose"
            or clothing_fit_answer in {"luźne", "nie wiem"}
            or selfie_risk
            or not checks["tło_po_cleanup_ok"]
            or score < 0.84
        )
    )

    blockers: List[str] = []
    if not checks["landmarki_kluczowe"]:
        blockers.append("brak kompletu kluczowych landmarków")
    if not checks["pełna_sylwetka"]:
        blockers.append("brak pełnej sylwetki od głowy do stóp")
    if not checks["tło_po_cleanup_ok"]:
        blockers.append("tło nadal utrudnia odcięcie sylwetki")
    if not checks["stopy_widoczne"]:
        blockers.append("brak pełnych stóp")
    if not checks["dłonie_widoczne"]:
        blockers.append("brak dłoni / nadgarstków")
    if not checks["pozycja_stojąca"]:
        blockers.append("pozycja nie jest stojąca")
    if not checks["poprawna_orientacja"]:
        blockers.append("niepoprawna orientacja ujęcia")
    if not checks["kamera_nieprzechylona"]:
        blockers.append("kamera jest przechylona")
    if not checks["perspektywa_ok"]:
        blockers.append("perspektywa kamery jest nieprawidłowa")
    if not checks["wysokość_kamery_ok"]:
        blockers.append("telefon nie jest na wysokości bioder")
    if not checks["ubranie_ok"]:
        blockers.append("ubranie jest zbyt luźne")
    if not checks["ręce_poprawnie_ustawione"]:
        blockers.append("ręce zasłaniają tułów lub są ustawione niepoprawnie")

    if hard_fail:
        status_code = "reject"
        status = "odrzuć zdjęcie"
    elif soft_fail:
        status_code = "retry"
        status = "popraw i spróbuj ponownie"
        messages.append("To zdjęcie może dać tylko orientacyjny wynik. Do analizy V8.6.7 potrzebne jest lepsze ujęcie.")
    else:
        status_code = "accept"
        status = "gotowe do analizy"

    accept_ready = status_code == "accept"
    measurement_ready = accept_ready and landmark_visibility >= 0.72 and len(missing_points) == 0
    posture_ready = measurement_ready and not selfie_risk and orientation_conf >= 0.78

    return CaptureQuality(
        orientation=orientation_expected,
        score=round(score, 2),
        status=status,
        status_code=status_code,
        messages=messages,
        checks=checks,
        missing_points=missing_points,
        detected_orientation=detected_orientation,
        orientation_confidence=round(orientation_conf, 2),
        roll_deg=round(roll_deg, 1),
        sitting_detected=sitting_detected,
        full_body_detected=checks["pełna_sylwetka"],
        camera_height_hint=cam_height_hint,
        camera_pitch_hint=cam_pitch_hint,
        clothing_fit_assessment=clothing_fit_answer,
        vision_clothing_fit=vision_fit,
        camera_angle_hint=_camera_angle_hint(roll_deg),
        landmark_visibility_score=round(landmark_visibility, 2),
        hand_position_hint=hand_hint,
        capture_method=capture_method,
        selfie_risk=selfie_risk,
        distance_hint=distance_hint,
        blockers=blockers,
        accept_ready=accept_ready,
        measurement_ready=measurement_ready,
        posture_ready=posture_ready,
        background_cleanup_applied=bool(sil.cleaned_image is not None),
        background_cleanup_score=float(sil.background_cleanup_score),
    )


def derive_build_type(height_cm: float, weight_kg: float) -> str:
    bmi = weight_kg / ((height_cm / 100) ** 2)
    if bmi < 20:
        return "szczupła"
    if bmi < 26:
        return "średnia"
    return "pełniejsza"


def anthropometric_prior(height_cm: float, weight_kg: float, age: Optional[int], gender: str = "kobieta") -> Dict[str, float]:
    bmi = weight_kg / ((height_cm / 100) ** 2)
    age_adj = max(0.0, float((age or 30) - 30))
    if gender == "mężczyzna":
        bust = 0.46 * height_cm + 0.19 * weight_kg + age_adj * 0.03
        waist = 0.36 * height_cm + 0.17 * weight_kg + max(0.0, bmi - 24.0) * 2.0 + age_adj * 0.05
        hips = 0.42 * height_cm + 0.17 * weight_kg + max(0.0, bmi - 24.0) * 1.0 + age_adj * 0.02
        bust = round(_clamp(bust, 82, 155), 1)
        waist = round(_clamp(waist, 64, 140), 1)
        hips = round(_clamp(hips, 82, 145), 1)
    else:
        bust = 0.405 * height_cm + 0.20 * weight_kg + age_adj * 0.03
        waist = 0.325 * height_cm + 0.15 * weight_kg + max(0.0, bmi - 24.0) * 2.0 + age_adj * 0.05
        hips = 0.435 * height_cm + 0.22 * weight_kg + max(0.0, bmi - 24.0) * 1.3 + age_adj * 0.03
        bust = round(_clamp(bust, 72, 145), 1)
        waist = round(_clamp(waist, 55, 135), 1)
        hips = round(_clamp(hips, 80, 155), 1)
    return {"bust": bust, "waist": waist, "hips": hips}


def classify_body_type(bust_cm: float, waist_cm: float, hips_cm: float) -> str:
    waist_to_hips = waist_cm / max(hips_cm, 1)
    waist_to_bust = waist_cm / max(bust_cm, 1)
    if abs(bust_cm - hips_cm) / max(bust_cm, hips_cm, 1) <= 0.05 and waist_cm <= min(bust_cm, hips_cm) * 0.80:
        return "klepsydra"
    if waist_to_hips >= 0.87 and waist_to_bust >= 0.87:
        return "jabłko"
    if hips_cm >= bust_cm * 1.05:
        return "gruszka"
    if bust_cm >= hips_cm * 1.05:
        return "odwrócony trójkąt"
    return "prostokąt"


def _compute_ratios(front: SilhouetteResult, profile: SilhouetteResult, height_cm: float, weight_kg: float) -> RatioPack:
    bmi = weight_kg / ((height_cm / 100) ** 2)
    bust_front = front.width_points.get("bust", 0) / max(front.width_points.get("hips", 1), 1)
    waist_front = front.width_points.get("waist", 0) / max(front.width_points.get("hips", 1), 1)
    shoulders_front = front.width_points.get("shoulders", 0) / max(front.width_points.get("hips", 1), 1)
    bust_depth = profile.width_points.get("bust", 0) / max(profile.width_points.get("hips", 1), 1)
    waist_depth = profile.width_points.get("waist", 0) / max(profile.width_points.get("hips", 1), 1)

    bust_to_hips = _clamp(0.76 * bust_front + 0.24 * bust_depth, 0.82, 1.16)
    waist_to_hips = _clamp(0.86 * waist_front + 0.14 * waist_depth, 0.65, 0.98)
    shoulders_to_hips = _clamp(shoulders_front, 0.80, 1.24)

    front_aspect = front.bbox[2] / max(front.bbox[3], 1)
    profile_aspect = profile.bbox[2] / max(profile.bbox[3], 1)
    expected_front = 0.24 + max(-0.03, min(0.06, (bmi - 21.0) * 0.006))
    expected_profile = 0.14 + max(-0.03, min(0.05, (bmi - 21.0) * 0.004))
    volume_factor = _clamp(1.0 + (front_aspect - expected_front) * 0.28 + (profile_aspect - expected_profile) * 0.55, 0.92, 1.10)

    return RatioPack(
        bust_to_hips=round(bust_to_hips, 3),
        waist_to_hips=round(waist_to_hips, 3),
        shoulders_to_hips=round(shoulders_to_hips, 3),
        volume_factor=round(volume_factor, 3),
    )


def _regularize_dimensions(bust: float, waist: float, hips: float, body_type: str) -> Tuple[float, float, float]:
    waist_caps = {
        "klepsydra": 0.80,
        "gruszka": 0.83,
        "prostokąt": 0.90,
        "odwrócony trójkąt": 0.85,
        "jabłko": 0.96,
    }
    waist_floor = {"klepsydra": 0.68, "gruszka": 0.69, "prostokąt": 0.78, "odwrócony trójkąt": 0.72, "jabłko": 0.84}
    hips = max(hips, waist + 8)
    bust = max(bust, 72.0)
    waist = _clamp(waist, hips * waist_floor[body_type], min(bust, hips) * waist_caps[body_type] if body_type != "jabłko" else hips * waist_caps[body_type])
    if body_type == "gruszka":
        hips = max(hips, bust + 4)
    elif body_type == "odwrócony trójkąt":
        bust = max(bust, hips + 4)
    elif body_type == "klepsydra":
        diff = abs(bust - hips)
        if diff > 4:
            avg = (bust + hips) / 2
            bust = avg + (bust - hips) * 0.25
            hips = avg - (bust - hips) * 0.25
    bust = _clamp(bust, 72, 160)
    waist = _clamp(waist, 55, 145)
    hips = _clamp(hips, 80, 160)
    return round(bust, 1), round(waist, 1), round(hips, 1)




def _safe_point_from_landmarks(landmarks: Dict[str, Tuple[float, float, float, float]], names: List[str], fallback: Tuple[int, int], shape: Tuple[int, int]) -> Tuple[int, int]:
    h, w = shape[:2]
    pts = []
    for name in names:
        if name in landmarks:
            x, y, _z, vis = landmarks[name]
            if vis >= 0.35:
                pts.append((int(x * w), int(y * h)))
    if not pts:
        return fallback
    return (int(np.mean([p[0] for p in pts])), int(np.mean([p[1] for p in pts])))


def _draw_measurement_line(canvas: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], label: str, value_cm: float, color: Tuple[int, int, int], to_right: bool = True):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    cv2.line(canvas, (x1, y1), (x2, y2), color, 2)
    cv2.circle(canvas, (x1, y1), 4, color, -1)
    cv2.circle(canvas, (x2, y2), 4, color, -1)
    mid_x = int((x1 + x2) / 2)
    mid_y = int((y1 + y2) / 2)
    elbow_x = min(canvas.shape[1] - 20, max(20, mid_x + (120 if to_right else -120)))
    cv2.line(canvas, (mid_x, mid_y), (elbow_x, mid_y), color, 2)
    txt = f"{label}: {float(value_cm):.1f} cm"
    tx = elbow_x + (8 if to_right else -220)
    cv2.putText(canvas, txt, (tx, max(24, mid_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(canvas, txt, (tx, max(24, mid_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 1, cv2.LINE_AA)


def _draw_marker_label(canvas: np.ndarray, p: Tuple[int, int], label: str, value_cm: float, color: Tuple[int, int, int], to_right: bool = True, dy: int = 0):
    x, y = int(p[0]), int(p[1] + dy)
    cv2.circle(canvas, (x, y), 4, color, -1)
    elbow_x = min(canvas.shape[1] - 20, max(20, x + (95 if to_right else -95)))
    cv2.line(canvas, (x, y), (elbow_x, y), color, 2)
    txt = f"{label}: {float(value_cm):.1f} cm"
    tx = elbow_x + (8 if to_right else -210)
    cv2.putText(canvas, txt, (tx, max(24, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(canvas, txt, (tx, max(24, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 1, cv2.LINE_AA)



def _mask_width_at_y(mask: np.ndarray, yy: int) -> tuple[int, int]:
    if yy < 0 or yy >= mask.shape[0]:
        return -1, -1
    xs = np.where(mask[yy] > 0)[0]
    if len(xs) < 5:
        return -1, -1
    return int(xs.min()), int(xs.max())


def _is_occluded_horizontal(mask: np.ndarray, p1: tuple[int, int], p2: tuple[int, int]) -> bool:
    y = int((p1[1] + p2[1]) / 2)
    x1, x2 = sorted([int(p1[0]), int(p2[0])])
    if y < 0 or y >= mask.shape[0] or x1 < 0 or x2 >= mask.shape[1]:
        return True
    segment = mask[y, x1:x2+1]
    if segment.size == 0:
        return True
    coverage = float(segment.mean())
    return coverage < 0.55


def _build_landmark_segment_report(
    front: Optional[SilhouetteResult],
    profile: Optional[SilhouetteResult],
    back: Optional[SilhouetteResult],
) -> Dict:
    report = {"front": [], "profile": [], "back": []}
    if front is not None and front.success:
        x, y, bw, bh = front.bbox
        mask = front.mask
        # central FRONT
        rels = {"F_NECK_W":0.17, "F_SHOULDERS_W":0.21, "F_CHEST_W":0.29, "F_WAIST_W":0.46, "F_HIPS_W":0.57}
        labels = {r["id"]:r["label"] for r in FRONT_CENTRAL}
        fields = {r["id"]:r.get("field","") for r in FRONT_CENTRAL}
        for sid, rel in rels.items():
            yy = y + int(bh * rel)
            x1, x2 = _mask_width_at_y(mask, yy)
            vis = x1 >= 0 and x2 >= 0
            report["front"].append({
                "id": sid, "label": labels[sid], "field": fields[sid],
                "visible": vis, "occluded": not vis, "confidence": "high" if vis else "low",
                "notes": "partia centralna"
            })
        # limb segments
        limb_rules = {
            "F_BICEPS_L_W": (["left_shoulder","left_elbow"], "arm_biceps_cm", "biceps lewy"),
            "F_BICEPS_R_W": (["right_shoulder","right_elbow"], "arm_biceps_cm", "biceps prawy"),
            "F_FOREARM_L_W": (["left_elbow","left_wrist"], "arm_biceps_cm", "przedramię lewe"),
            "F_FOREARM_R_W": (["right_elbow","right_wrist"], "arm_biceps_cm", "przedramię prawe"),
            "F_WRIST_L_W": (["left_wrist"], "wrist_cm", "nadgarstek lewy"),
            "F_WRIST_R_W": (["right_wrist"], "wrist_cm", "nadgarstek prawy"),
            "F_THIGH_L_MAX_W": (["left_hip","left_knee"], "thigh_cm", "udo lewe"),
            "F_THIGH_R_MAX_W": (["right_hip","right_knee"], "thigh_cm", "udo prawe"),
            "F_CALF_L_MAX_W": (["left_knee","left_ankle"], "calf_max_cm", "łydka lewa max"),
            "F_CALF_R_MAX_W": (["right_knee","right_ankle"], "calf_max_cm", "łydka prawa max"),
            "F_CALF_L_MIN_W": (["left_ankle"], "calf_min_cm", "łydka lewa min"),
            "F_CALF_R_MIN_W": (["right_ankle"], "calf_min_cm", "łydka prawa min"),
        }
        for sid, (names, field, label) in limb_rules.items():
            pts = []
            for n in names:
                if n in front.landmarks:
                    lx, ly, _lz, lv = front.landmarks[n]
                    if lv >= 0.35:
                        pts.append((int(lx*front.image_shape[1]), int(ly*front.image_shape[0])))
            vis = len(pts) >= 1
            occl = False
            if vis and len(pts) == 2:
                mid = ((pts[0][0]+pts[1][0])//2, (pts[0][1]+pts[1][1])//2)
                x1,x2 = _mask_width_at_y(mask, mid[1])
                occl = (x1 < 0 or x2 < 0)
            report["front"].append({
                "id": sid, "label": label, "field": field,
                "visible": vis, "occluded": occl, "confidence": "medium" if vis and not occl else "low",
                "notes": "kończyna L/R"
            })
    if profile is not None and profile.success:
        for row in PROFILE_CORE:
            vis = True
            if row["id"] in {"P_WRIST_D"} and "right_wrist" not in profile.landmarks and "left_wrist" not in profile.landmarks:
                vis = False
            report["profile"].append({
                "id": row["id"], "label": row["label"], "field": row.get("field",""),
                "visible": vis, "occluded": not vis, "confidence": "high" if vis else "low",
                "notes": "głębokość / profil"
            })
    if back is not None and back.success:
        for row in BACK_QA:
            report["back"].append({
                "id": row["id"], "label": row["label"], "field": row.get("field",""),
                "visible": True, "occluded": False, "confidence": "medium",
                "notes": "symetria / postura / QA"
            })
    return report



def _build_measurement_overlay(
    sil: Optional[SilhouetteResult],
    orientation: str,
    gender: str,
    suggested_bust_cm: float,
    suggested_waist_cm: float,
    suggested_hips_cm: float,
    extra_estimates: Dict[str, float],
) -> Optional[np.ndarray]:
    if sil is None or sil.cleaned_image is None or not sil.success:
        return None
    canvas = sil.cleaned_image.copy()
    h, w = canvas.shape[:2]
    x, y, bw, bh = sil.bbox
    lms = sil.landmarks or {}

    cv2.rectangle(canvas, (12, 12), (w - 12, h - 12), (20, 184, 166), 2)
    title = f"{orientation.upper()} — standard landmarków ATEENA v1"
    cv2.putText(canvas, title, (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (20, 184, 166), 2, cv2.LINE_AA)

    def central_line(rel_y: float):
        yy = y + int(bh * rel_y)
        xs = np.where(sil.mask[yy] > 0)[0] if 0 <= yy < sil.mask.shape[0] else np.array([])
        if len(xs) >= 10:
            return (int(xs.min()), yy), (int(xs.max()), yy)
        return (x, yy), (x + bw, yy)

    if orientation == 'front':
        front_rows = [
            ('Szyja', extra_estimates.get('neck_cm', 0), 0.17, (142, 68, 173), True),
            ('Ramiona', max(0.0, suggested_bust_cm * 0.95), 0.21, (26, 188, 156), False),
            ('Klatka piersiowa', suggested_bust_cm, 0.29, (52, 152, 219), True),
            ('Talia', suggested_waist_cm, 0.46, (243, 156, 18), False),
            ('Biodra', suggested_hips_cm, 0.57, (231, 76, 60), True),
            ('Brzuch', extra_estimates.get('abdomen_cm', 0), 0.515, (241, 196, 15), False),
        ]
        for label, val, rel, color, right in front_rows:
            p1, p2 = central_line(rel)
            _draw_measurement_line(canvas, p1, p2, label, val, color, to_right=right)

        # limb labels based on left/right standard
        left_sh = _safe_point_from_landmarks(lms, ['left_shoulder'], (x + int(bw*0.18), y + int(bh*0.22)), canvas.shape)
        right_sh = _safe_point_from_landmarks(lms, ['right_shoulder'], (x + int(bw*0.82), y + int(bh*0.22)), canvas.shape)
        left_el = _safe_point_from_landmarks(lms, ['left_elbow'], (x + int(bw*0.13), y + int(bh*0.36)), canvas.shape)
        right_el = _safe_point_from_landmarks(lms, ['right_elbow'], (x + int(bw*0.87), y + int(bh*0.36)), canvas.shape)
        left_wr = _safe_point_from_landmarks(lms, ['left_wrist'], (x + int(bw*0.11), y + int(bh*0.55)), canvas.shape)
        right_wr = _safe_point_from_landmarks(lms, ['right_wrist'], (x + int(bw*0.89), y + int(bh*0.55)), canvas.shape)
        left_hip = _safe_point_from_landmarks(lms, ['left_hip'], (x + int(bw*0.36), y + int(bh*0.59)), canvas.shape)
        right_hip = _safe_point_from_landmarks(lms, ['right_hip'], (x + int(bw*0.64), y + int(bh*0.59)), canvas.shape)
        left_knee = _safe_point_from_landmarks(lms, ['left_knee'], (x + int(bw*0.40), y + int(bh*0.79)), canvas.shape)
        right_knee = _safe_point_from_landmarks(lms, ['right_knee'], (x + int(bw*0.60), y + int(bh*0.79)), canvas.shape)
        left_ankle = _safe_point_from_landmarks(lms, ['left_ankle'], (x + int(bw*0.42), y + int(bh*0.94)), canvas.shape)
        right_ankle = _safe_point_from_landmarks(lms, ['right_ankle'], (x + int(bw*0.58), y + int(bh*0.94)), canvas.shape)

        # Upperarm under armpit
        left_upper = ((left_sh[0] + left_el[0]) // 2, int(left_sh[1]*0.65 + left_el[1]*0.35))
        right_upper = ((right_sh[0] + right_el[0]) // 2, int(right_sh[1]*0.65 + right_el[1]*0.35))
        _draw_marker_label(canvas, left_upper, 'Lewe ramię', extra_estimates.get('arm_biceps_cm', 0), (46, 204, 113), to_right=False)
        _draw_marker_label(canvas, right_upper, 'Prawe ramię', extra_estimates.get('arm_biceps_cm', 0), (46, 204, 113), to_right=True)

        # Biceps at thickest place
        left_biceps = ((left_sh[0] + left_el[0]) // 2, (left_sh[1] + left_el[1]) // 2)
        right_biceps = ((right_sh[0] + right_el[0]) // 2, (right_sh[1] + right_el[1]) // 2)
        _draw_marker_label(canvas, left_biceps, 'Biceps lewy', extra_estimates.get('arm_biceps_cm', 0), (39, 174, 96), to_right=False, dy=10)
        _draw_marker_label(canvas, right_biceps, 'Biceps prawy', extra_estimates.get('arm_biceps_cm', 0), (39, 174, 96), to_right=True, dy=10)

        left_forearm = ((left_el[0] + left_wr[0]) // 2, (left_el[1] + left_wr[1]) // 2)
        right_forearm = ((right_el[0] + right_wr[0]) // 2, (right_el[1] + right_wr[1]) // 2)
        _draw_marker_label(canvas, left_forearm, 'Przedramię lewe', extra_estimates.get('arm_biceps_cm', 0), (52, 152, 219), to_right=False)
        _draw_marker_label(canvas, right_forearm, 'Przedramię prawe', extra_estimates.get('arm_biceps_cm', 0), (52, 152, 219), to_right=True)

        _draw_marker_label(canvas, left_wr, 'Nadgarstek lewy', extra_estimates.get('wrist_cm', 0), (22, 160, 133), to_right=False, dy=-4)
        _draw_marker_label(canvas, right_wr, 'Nadgarstek prawy', extra_estimates.get('wrist_cm', 0), (22, 160, 133), to_right=True, dy=-4)

        left_thigh = ((left_hip[0] + left_knee[0]) // 2, int(left_hip[1]*0.55 + left_knee[1]*0.45))
        right_thigh = ((right_hip[0] + right_knee[0]) // 2, int(right_hip[1]*0.55 + right_knee[1]*0.45))
        _draw_marker_label(canvas, left_thigh, 'Udo lewe — max', extra_estimates.get('thigh_cm', 0), (230, 126, 34), to_right=False)
        _draw_marker_label(canvas, right_thigh, 'Udo prawe — max', extra_estimates.get('thigh_cm', 0), (230, 126, 34), to_right=True)

        left_calf_max = ((left_knee[0] + left_ankle[0]) // 2, int(left_knee[1]*0.62 + left_ankle[1]*0.38))
        right_calf_max = ((right_knee[0] + right_ankle[0]) // 2, int(right_knee[1]*0.62 + right_ankle[1]*0.38))
        left_calf_min = ((left_knee[0] + left_ankle[0]) // 2, int(left_knee[1]*0.32 + left_ankle[1]*0.68))
        right_calf_min = ((right_knee[0] + right_ankle[0]) // 2, int(right_knee[1]*0.32 + right_ankle[1]*0.68))
        _draw_marker_label(canvas, left_calf_max, 'Łydka lewa — max', extra_estimates.get('calf_max_cm', 0), (52, 73, 94), to_right=False)
        _draw_marker_label(canvas, right_calf_max, 'Łydka prawa — max', extra_estimates.get('calf_max_cm', 0), (52, 73, 94), to_right=True)
        _draw_marker_label(canvas, left_calf_min, 'Łydka lewa — min', extra_estimates.get('calf_min_cm', 0), (127, 140, 141), to_right=False)
        _draw_marker_label(canvas, right_calf_min, 'Łydka prawa — min', extra_estimates.get('calf_min_cm', 0), (127, 140, 141), to_right=True)

    elif orientation == 'profil':
        # profile depth from back to front
        def depth_line(rel_y: float):
            yy = y + int(bh * rel_y)
            xs = np.where(sil.mask[yy] > 0)[0] if 0 <= yy < sil.mask.shape[0] else np.array([])
            if len(xs) >= 10:
                return (int(xs.min()), yy), (int(xs.max()), yy)
            return (x, yy), (x + bw, yy)

        profile_rows = [
            ('Szyja — głębokość', extra_estimates.get('neck_cm', 0), 0.17, (142, 68, 173), True),
            ('Klatka — głębokość', suggested_bust_cm, 0.29, (52, 152, 219), True),
            ('Talia — głębokość', suggested_waist_cm, 0.46, (243, 156, 18), False),
            ('Brzuch — głębokość', extra_estimates.get('abdomen_cm', 0), 0.515, (241, 196, 15), False),
            ('Biodra — głębokość', suggested_hips_cm, 0.57, (231, 76, 60), True),
            ('Udo — głębokość', extra_estimates.get('thigh_cm', 0), 0.69, (230, 126, 34), True),
            ('Łydka max — głębokość', extra_estimates.get('calf_max_cm', 0), 0.83, (52, 73, 94), True),
            ('Łydka min — głębokość', extra_estimates.get('calf_min_cm', 0), 0.91, (127, 140, 141), True),
        ]
        for label, val, rel, color, right in profile_rows:
            p1, p2 = depth_line(rel)
            _draw_measurement_line(canvas, p1, p2, label, val, color, to_right=right)

        wrist = _safe_point_from_landmarks(lms, ['right_wrist','left_wrist'], (x + int(bw*0.75), y + int(bh*0.54)), canvas.shape)
        _draw_marker_label(canvas, wrist, 'Nadgarstek — głębokość', extra_estimates.get('wrist_cm', 0), (22, 160, 133), to_right=True, dy=-4)

    else:  # tył
        back_rows = [
            ('Barki / ramiona od tyłu', max(0.0, suggested_bust_cm * 0.95), 0.21, (26, 188, 156), True),
            ('Talia od tyłu', suggested_waist_cm, 0.46, (243, 156, 18), False),
            ('Biodra / pośladki od tyłu', suggested_hips_cm, 0.57, (231, 76, 60), True),
        ]
        for label, val, rel, color, right in back_rows:
            p1, p2 = central_line(rel)
            _draw_measurement_line(canvas, p1, p2, label, val, color, to_right=right)
        # QA axes
        left_sh = _safe_point_from_landmarks(lms, ['left_shoulder'], (x + int(bw*0.28), y + int(bh*0.21)), canvas.shape)
        right_sh = _safe_point_from_landmarks(lms, ['right_shoulder'], (x + int(bw*0.72), y + int(bh*0.21)), canvas.shape)
        left_hip = _safe_point_from_landmarks(lms, ['left_hip'], (x + int(bw*0.36), y + int(bh*0.57)), canvas.shape)
        right_hip = _safe_point_from_landmarks(lms, ['right_hip'], (x + int(bw*0.64), y + int(bh*0.57)), canvas.shape)
        spine_top = ((left_sh[0] + right_sh[0]) // 2, y + int(bh*0.14))
        spine_bottom = ((left_hip[0] + right_hip[0]) // 2, y + int(bh*0.66))
        cv2.line(canvas, spine_top, spine_bottom, (155, 89, 182), 2)
        cv2.putText(canvas, 'Oś ciała', (min(w-180, spine_top[0]+20), max(40, spine_top[1]+30)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (155,89,182), 1, cv2.LINE_AA)
        cv2.line(canvas, left_sh, right_sh, (52, 73, 94), 2)
        cv2.line(canvas, left_hip, right_hip, (52, 73, 94), 2)
        cv2.putText(canvas, 'Linia barków', (min(w-220, right_sh[0]+18), max(40, right_sh[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (52,73,94), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'Linia miednicy', (min(w-220, right_hip[0]+18), max(40, right_hip[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (52,73,94), 1, cv2.LINE_AA)

    footer = "ATEENA Landmark Schema v1 — miejsca odczytu / estymacji zapisane w cm"
    cv2.putText(canvas, footer, (22, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (52, 73, 94), 1, cv2.LINE_AA)
    return canvas


def analyze_body(
    front_image_bytes: bytes,
    profile_image_bytes: bytes,
    height_cm: float,
    weight_kg: float,
    age: Optional[int],
    gender: str = "kobieta",
    clothing_fit_answer: str = "średnie",
    capture_method: str = "inna osoba trzyma telefon",
    back_image_bytes: Optional[bytes] = None,
    right_profile_image_bytes: Optional[bytes] = None,
) -> BodyAnalysisResult:
    front = _extract_silhouette(front_image_bytes, "front", clothing_fit_answer, capture_method)
    left_profile = _extract_silhouette(profile_image_bytes, "profile", clothing_fit_answer, capture_method)
    right_profile = _extract_silhouette(right_profile_image_bytes, "profile", clothing_fit_answer, capture_method) if right_profile_image_bytes is not None else None
    profile = left_profile
    if right_profile is not None and right_profile.quality.score > left_profile.quality.score:
        profile = right_profile
    back = _extract_silhouette(back_image_bytes, "front", clothing_fit_answer, capture_method) if back_image_bytes is not None else None

    prior = anthropometric_prior(height_cm, weight_kg, age, gender=gender)
    notes: List[str] = []
    measurement_source = "anthropometric_prior_v8_6_7"
    raw_bust = prior["bust"]
    raw_waist = prior["waist"]
    raw_hips = prior["hips"]
    measurement_confidence = {"bust": 0.18, "waist": 0.18, "hips": 0.18}

    if not HAS_MEDIAPIPE:
        notes.append("Nie wykryto biblioteki landmarków ciała (MediaPipe). Vision pipeline jest wyłączony, więc aplikacja przechodzi w fallback prior only. Ten wynik nie powinien być traktowany jako estymacja ze zdjęć.")

    strict_fail = not front.quality.measurement_ready or not left_profile.quality.measurement_ready or (right_profile is not None and not right_profile.quality.measurement_ready) or not profile.quality.measurement_ready
    if strict_fail:
        confidence = 0.08
        notes.append("V8.6.7 blokuje mylący fallback. Jeśli front/profil nie mają measurement_ready albo MediaPipe nie działa, wynik nie powinien być traktowany jako estymacja ze zdjęć.")
        if not front.quality.measurement_ready:
            notes.append("Front wymaga poprawy lub został odrzucony. Blokery: " + ", ".join(front.quality.blockers[:4]))
        if not left_profile.quality.measurement_ready:
            notes.append("Lewy profil wymaga poprawy lub został odrzucony. Blokery: " + ", ".join(left_profile.quality.blockers[:4]))
        if right_profile is not None and not right_profile.quality.measurement_ready:
            notes.append("Prawy profil wymaga poprawy lub został odrzucony. Blokery: " + ", ".join(right_profile.quality.blockers[:4]))
    else:
        pair_quality = min(front.quality.score, profile.quality.score)
        if right_profile is not None:
            notes.append(f"Wersja 8.6.1 korzysta z czterech zdjęć: FRONT, PROFIL LEWY, PROFIL PRAWY i TYŁ. Do obliczeń użyto profilu o wyższej jakości ({'prawy' if profile is right_profile else 'lewy'}).")
        back_bonus = 0.0
        if back is not None and back.quality.status_code == 'accept':
            back_bonus = 0.06
            notes.append("Dodano trzecie ujęcie TYŁ — poprawia to pewność części ocen, zwłaszcza symetrii i stref dolnych.")
        elif back is not None and back.quality.status_code != 'accept':
            notes.append("Zdjęcie TYŁ nie przeszło quality gate i nie zostało użyte do poprawy estymacji.")

        ratios = _compute_ratios(front, profile, height_cm, weight_kg)
        photo_hips = prior["hips"] * ratios.volume_factor
        photo_bust = photo_hips * ratios.bust_to_hips
        photo_waist = photo_hips * ratios.waist_to_hips

        clothing_penalty = 1.0
        if clothing_fit_answer in {"luźne", "nie wiem"} or front.quality.vision_clothing_fit == "loose" or profile.quality.vision_clothing_fit == "loose":
            clothing_penalty = 0.72
            notes.append("Model wykrył część luzu stroju — waga obrazu w estymacji została zmniejszona.")
        if capture_method == "zdjęcie wykonane samemu":
            clothing_penalty *= 0.80
            notes.append("Tryb zdjęcia wykonywanego samemu obniża wiarygodność proporcji — waga obrazu została dodatkowo ograniczona.")

        hips_alpha = _clamp((0.18 + 0.34 * pair_quality + back_bonus) * clothing_penalty, 0.10, 0.50)
        bust_alpha = _clamp((0.16 + 0.30 * pair_quality) * clothing_penalty, 0.10, 0.42)
        waist_alpha = _clamp((0.18 + 0.32 * pair_quality + back_bonus * 0.5) * clothing_penalty, 0.10, 0.44)

        raw_hips = prior["hips"] * (1 - hips_alpha) + photo_hips * hips_alpha
        raw_bust = prior["bust"] * (1 - bust_alpha) + photo_bust * bust_alpha
        raw_waist = prior["waist"] * (1 - waist_alpha) + photo_waist * waist_alpha
        photo_body_type = classify_body_type(photo_bust, photo_waist, photo_hips)
        raw_bust, raw_waist, raw_hips = _regularize_dimensions(raw_bust, raw_waist, raw_hips, photo_body_type)
        confidence = round(_clamp(0.34 + 0.56 * pair_quality * clothing_penalty + back_bonus * 0.65, 0.0, 0.96), 2)
        measurement_confidence = {
            "bust": round(_clamp((bust_alpha + pair_quality * 0.22) * (0.88 if capture_method == "zdjęcie wykonane samemu" else 1.0), 0.18, 0.90), 2),
            "waist": round(_clamp((waist_alpha + pair_quality * 0.22 + back_bonus * 0.6) * (0.88 if capture_method == "zdjęcie wykonane samemu" else 1.0), 0.18, 0.92), 2),
            "hips": round(_clamp((hips_alpha + pair_quality * 0.22 + back_bonus) * (0.90 if clothing_penalty < 0.9 else 1.0), 0.18, 0.94), 2),
        }
        measurement_source = "pose_landmarks_plus_prior_v8_6_7"
        notes.append("V8.6.7 rozróżnia photo-based estimate od fallback prior only i blokuje mylące rekomendacje, gdy vision pipeline nie działa.")
        notes.append("Aby wynik był wiarygodny, telefon powinien być na wysokości bioder, a ręce odsunięte od tułowia o około 10–15 cm.")
        notes.append(f"Wskaźnik proporcji front/profil: biust:biodra {ratios.bust_to_hips:.2f}, talia:biodra {ratios.waist_to_hips:.2f}, korekta objętości {ratios.volume_factor:.2f}x.")

    body_type = classify_body_type(raw_bust, raw_waist, raw_hips)
    suggested_bust, suggested_waist, suggested_hips = _regularize_dimensions(raw_bust, raw_waist, raw_hips, body_type)
    build_type = derive_build_type(height_cm, weight_kg)

    extra_estimates, measurement_confidence = _estimate_extra_measurements(
        suggested_bust, suggested_waist, suggested_hips, height_cm, weight_kg, gender, measurement_confidence,
        float(front.quality.score), float(profile.quality.score), float(back.quality.score) if back is not None else 0.0
    )
    if capture_method == "zdjęcie wykonane samemu":
        for k in list(measurement_confidence.keys()):
            measurement_confidence[k] = round(_clamp(float(measurement_confidence[k]) * 0.86, 0.18, 0.94), 2)
    if back is None:
        for key in ["thigh_cm", "calf_max_cm", "calf_min_cm"]:
            if key in measurement_confidence:
                measurement_confidence[key] = round(_clamp(float(measurement_confidence[key]) * 0.90, 0.18, 0.90), 2)

    sanity_payload = {'bust': suggested_bust, 'waist': suggested_waist, 'hips': suggested_hips, **extra_estimates}
    sanity_report = analyze_measurement_sanity(sanity_payload, gender=gender, body_type=body_type, height_cm=height_cm)
    weak_points = list(sanity_report.get('weak_points', []))
    for key, conf in measurement_confidence.items():
        if conf < 0.62:
            weak_points.append(key)
    weak_points = sorted(set(weak_points), key=lambda x: ['bust','waist','hips','abdomen_cm','thigh_cm','arm_biceps_cm','neck_cm','wrist_cm','calf_max_cm','calf_min_cm'].index(x) if x in ['bust','waist','hips','abdomen_cm','thigh_cm','arm_biceps_cm','neck_cm','wrist_cm','calf_max_cm','calf_min_cm'] else 99)
    if weak_points:
        notes.append("Aplikacja wskazała słabsze strefy pomiaru — warto potwierdzić ręcznie tylko te obszary.")
    if sanity_report.get('flags'):
        notes.append("Sanity engine wykrył niespójności pomiarowe i oznaczył strefy do ręcznego potwierdzenia.")

    back_capture = back.quality.to_dict() if back is not None else _missing_capture_stub('tył')
    if back is None:
        notes.append("Nie dodano trzeciego ujęcia TYŁ — wynik nadal działa, ale pewność części ocen postawy i nóg jest niższa.")

    bg_scores = [front.quality.background_cleanup_score, left_profile.quality.background_cleanup_score]
    if right_profile is not None:
        bg_scores.append(right_profile.quality.background_cleanup_score)
    if back is not None:
        bg_scores.append(back.quality.background_cleanup_score)
    notes.append(f"Oczyszczenie tła — średni score: {round(float(np.mean(bg_scores))*100)}%.")
    if any(s < 0.46 for s in bg_scores):
        notes.append("Co najmniej jedno ujęcie ma słabsze odcięcie postaci od tła — confidence został odpowiednio obniżony.")

    landmark_segment_report = _build_landmark_segment_report(front, profile, back)
    front_measure_overlay = _build_measurement_overlay(front, 'front', gender, suggested_bust, suggested_waist, suggested_hips, extra_estimates)
    left_profile_measure_overlay = _build_measurement_overlay(left_profile, 'profil lewy', gender, suggested_bust, suggested_waist, suggested_hips, extra_estimates)
    right_profile_measure_overlay = _build_measurement_overlay(right_profile, 'profil prawy', gender, suggested_bust, suggested_waist, suggested_hips, extra_estimates) if right_profile is not None else None
    profile_measure_overlay = left_profile_measure_overlay if profile is left_profile else right_profile_measure_overlay
    back_measure_overlay = _build_measurement_overlay(back, 'tył', gender, suggested_bust, suggested_waist, suggested_hips, extra_estimates) if back is not None else None

    return BodyAnalysisResult(
        body_type=body_type,
        suggested_bust_cm=suggested_bust,
        suggested_waist_cm=suggested_waist,
        suggested_hips_cm=suggested_hips,
        raw_bust_cm=round(raw_bust, 1),
        raw_waist_cm=round(raw_waist, 1),
        raw_hips_cm=round(raw_hips, 1),
        build_type=build_type,
        confidence=confidence,
        notes=notes,
        measurement_source=measurement_source,
        front_capture=front.quality.to_dict(),
        profile_capture=profile.quality.to_dict(),
        left_profile_capture=left_profile.quality.to_dict(),
        right_profile_capture=right_profile.quality.to_dict() if right_profile is not None else _missing_capture_stub('profil prawy'),
        back_capture=back_capture,
        calibration_info={'applied': False, 'used_scopes': [], 'offsets_cm': {'bust': 0.0, 'waist': 0.0, 'hips': 0.0}, 'sample_count': 0.0},
        measurement_confidence=measurement_confidence,
        extra_estimates=extra_estimates,
        weak_points=weak_points,
        sanity_report=sanity_report,
        front_debug_image=front.debug_image,
        profile_debug_image=left_profile.debug_image if left_profile is not None else None,
        right_profile_debug_image=right_profile.debug_image if right_profile is not None else None,
        back_debug_image=back.debug_image if back is not None else None,
        front_cleaned_image=front.cleaned_image if front is not None else None,
        profile_cleaned_image=left_profile.cleaned_image if left_profile is not None else None,
        right_profile_cleaned_image=right_profile.cleaned_image if right_profile is not None else None,
        back_cleaned_image=back.cleaned_image if back is not None else None,
        front_measure_overlay_image=front_measure_overlay,
        profile_measure_overlay_image=left_profile_measure_overlay,
        right_profile_measure_overlay_image=right_profile_measure_overlay,
        back_measure_overlay_image=back_measure_overlay,
        landmark_segment_report=landmark_segment_report,
    )
