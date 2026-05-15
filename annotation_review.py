
from __future__ import annotations
import io
from typing import Any, Dict, List
import pandas as pd
from landmark_schema import schema_rows_for_ui

ANNOTATION_TEMPLATE_COLUMNS = [
    "view",
    "landmark_id",
    "expected_visible",
    "expected_occluded",
    "expected_confidence",
    "metric_id",
    "manual_value_cm",
    "comment",
]

def template_annotation_dataframe() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in schema_rows_for_ui():
        rows.append({
            "view": row.get("view", ""),
            "landmark_id": row.get("id", ""),
            "expected_visible": 1,
            "expected_occluded": 0,
            "expected_confidence": "high" if row.get("status") == "CORE" else "medium",
            "metric_id": row.get("output_metric", ""),
            "manual_value_cm": "",
            "comment": "",
        })
    return pd.DataFrame(rows, columns=ANNOTATION_TEMPLATE_COLUMNS)

def template_annotation_csv_bytes() -> bytes:
    return template_annotation_dataframe().to_csv(index=False).encode("utf-8-sig")

def _normalize_bool(v: Any) -> bool:
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"1","true","tak","yes","y"}

def parse_annotation_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    missing = [c for c in ANNOTATION_TEMPLATE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Brak wymaganych kolumn: {', '.join(missing)}")
    df = df.copy()
    df["view"] = df["view"].astype(str).str.strip().str.lower()
    df["landmark_id"] = df["landmark_id"].astype(str).str.strip()
    df["expected_visible"] = df["expected_visible"].apply(_normalize_bool)
    df["expected_occluded"] = df["expected_occluded"].apply(_normalize_bool)
    df["expected_confidence"] = df["expected_confidence"].astype(str).str.strip().str.lower()
    df["metric_id"] = df["metric_id"].astype(str).str.strip()
    df["manual_value_cm"] = pd.to_numeric(df["manual_value_cm"], errors="coerce")
    df["comment"] = df["comment"].fillna("").astype(str)
    return df

def body_result_value_map(body_result: Any) -> Dict[str, float]:
    values = {
        "chest_cm": float(getattr(body_result, "suggested_bust_cm", 0.0)),
        "bust_cm": float(getattr(body_result, "suggested_bust_cm", 0.0)),
        "waist_cm": float(getattr(body_result, "suggested_waist_cm", 0.0)),
        "hips_cm": float(getattr(body_result, "suggested_hips_cm", 0.0)),
    }
    extra = getattr(body_result, "extra_estimates", {}) or {}
    for k, v in extra.items():
        try:
            values[k] = float(v)
        except Exception:
            pass
    return values

def compare_annotations(body_result: Any, ann_df: pd.DataFrame) -> Dict[str, Any]:
    report = getattr(body_result, "landmark_segment_report", {}) or {}
    auto_rows = []
    for view_key in ["front","profile","back"]:
        for row in report.get(view_key, []) or []:
            auto_rows.append({
                "view": view_key,
                "landmark_id": row.get("id",""),
                "visible_auto": bool(row.get("visible")),
                "occluded_auto": bool(row.get("occluded")),
                "confidence_auto": str(row.get("confidence","")).lower(),
                "field": row.get("field",""),
                "label_auto": row.get("label",""),
                "notes_auto": row.get("notes",""),
            })
    auto_df = pd.DataFrame(auto_rows)
    if auto_df.empty:
        raise ValueError("Body result nie zawiera raportu segmentów landmarków.")
    merged = ann_df.merge(auto_df, how="left", on=["view","landmark_id"])
    merged["auto_found"] = ~merged["label_auto"].isna()
    merged["visible_match"] = merged["auto_found"] & (merged["expected_visible"] == merged["visible_auto"])
    merged["occluded_match"] = merged["auto_found"] & (merged["expected_occluded"] == merged["occluded_auto"])
    merged["confidence_match"] = merged["auto_found"] & (merged["expected_confidence"] == merged["confidence_auto"])

    landmark_rows = merged[merged["landmark_id"].astype(str) != ""].copy()
    value_map = body_result_value_map(body_result)
    metric_rows = merged[merged["metric_id"].astype(str) != ""].copy()
    metric_rows["auto_value_cm"] = metric_rows["metric_id"].map(value_map)
    metric_rows["abs_error_cm"] = (metric_rows["manual_value_cm"] - metric_rows["auto_value_cm"]).abs()

    summary = {
        "landmark_total": int(len(landmark_rows)),
        "landmark_found": int(landmark_rows["auto_found"].sum()) if not landmark_rows.empty else 0,
        "visibility_accuracy": round(float(landmark_rows["visible_match"].mean()) * 100, 1) if not landmark_rows.empty else 0.0,
        "occlusion_accuracy": round(float(landmark_rows["occluded_match"].mean()) * 100, 1) if not landmark_rows.empty else 0.0,
        "confidence_accuracy": round(float(landmark_rows["confidence_match"].mean()) * 100, 1) if not landmark_rows.empty else 0.0,
        "metrics_compared": int(metric_rows["manual_value_cm"].notna().sum()) if not metric_rows.empty else 0,
        "mae_cm": round(float(metric_rows["abs_error_cm"].dropna().mean()), 2) if not metric_rows.empty and metric_rows["abs_error_cm"].notna().any() else None,
    }
    return {"summary": summary, "landmark_comparison": landmark_rows, "metric_comparison": metric_rows}
