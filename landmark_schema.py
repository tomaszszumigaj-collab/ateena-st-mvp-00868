
from __future__ import annotations

from typing import Dict, List

LANDMARK_SCHEMA_VERSION = "ATEENA Body Landmark Schema v1.1"

FRONT_CENTRAL: List[Dict] = [
    {"id":"F_HEAD_W","label":"Głowa — szerokość","field":"head_width_cm","kind":"central"},
    {"id":"F_NECK_W","label":"Szyja","field":"neck_cm","kind":"central"},
    {"id":"F_SHOULDERS_W","label":"Ramiona","field":"shoulders_width_cm","kind":"central"},
    {"id":"F_CHEST_W","label":"Klatka piersiowa","field":"chest_cm","kind":"central"},
    {"id":"F_WAIST_W","label":"Talia","field":"waist_cm","kind":"central"},
    {"id":"F_HIPS_W","label":"Biodra","field":"hips_cm","kind":"central"},
]

FRONT_LEFT_RIGHT: List[Dict] = [
    {"id":"F_BICEPS_L_W","label":"Biceps lewy","field":"arm_biceps_cm","side":"left"},
    {"id":"F_BICEPS_R_W","label":"Biceps prawy","field":"arm_biceps_cm","side":"right"},
    {"id":"F_FOREARM_L_W","label":"Przedramię lewe","field":"arm_biceps_cm","side":"left"},
    {"id":"F_FOREARM_R_W","label":"Przedramię prawe","field":"arm_biceps_cm","side":"right"},
    {"id":"F_WRIST_L_W","label":"Nadgarstek lewy","field":"wrist_cm","side":"left"},
    {"id":"F_WRIST_R_W","label":"Nadgarstek prawy","field":"wrist_cm","side":"right"},
    {"id":"F_THIGH_L_MAX_W","label":"Udo lewe — najszersze miejsce","field":"thigh_cm","side":"left"},
    {"id":"F_THIGH_R_MAX_W","label":"Udo prawe — najszersze miejsce","field":"thigh_cm","side":"right"},
    {"id":"F_CALF_L_MAX_W","label":"Łydka lewa — najszersze miejsce","field":"calf_max_cm","side":"left"},
    {"id":"F_CALF_R_MAX_W","label":"Łydka prawa — najszersze miejsce","field":"calf_max_cm","side":"right"},
    {"id":"F_CALF_L_MIN_W","label":"Łydka lewa — najszczuplejsze miejsce","field":"calf_min_cm","side":"left"},
    {"id":"F_CALF_R_MIN_W","label":"Łydka prawa — najszczuplejsze miejsce","field":"calf_min_cm","side":"right"},
    {"id":"F_FOOT_L_W","label":"Stopa lewa — szerokość","field":"foot_width_cm","side":"left"},
    {"id":"F_FOOT_R_W","label":"Stopa prawa — szerokość","field":"foot_width_cm","side":"right"},
]

PROFILE_CORE: List[Dict] = [
    {"id":"P_HEAD_D","label":"Głowa — głębokość","field":"head_depth_cm","kind":"profile"},
    {"id":"P_NECK_D","label":"Szyja — głębokość","field":"neck_cm","kind":"profile"},
    {"id":"P_CHEST_D","label":"Klatka — głębokość","field":"chest_cm","kind":"profile"},
    {"id":"P_WAIST_D","label":"Talia — głębokość","field":"waist_cm","kind":"profile"},
    {"id":"P_ABDOMEN_D","label":"Brzuch — głębokość","field":"abdomen_cm","kind":"profile"},
    {"id":"P_HIPS_D","label":"Biodra — głębokość","field":"hips_cm","kind":"profile"},
    {"id":"P_THIGH_D","label":"Udo — głębokość","field":"thigh_cm","kind":"profile"},
    {"id":"P_CALF_MAX_D","label":"Łydka max — głębokość","field":"calf_max_cm","kind":"profile"},
    {"id":"P_CALF_MIN_D","label":"Łydka min — głębokość","field":"calf_min_cm","kind":"profile"},
    {"id":"P_WRIST_D","label":"Nadgarstek — głębokość","field":"wrist_cm","kind":"profile"},
    {"id":"P_FOOT_L","label":"Stopa — długość","field":"foot_length_cm","kind":"profile"},
]

BACK_QA: List[Dict] = [
    {"id":"B_HEAD_W","label":"Głowa od tyłu — szerokość","field":"head_width_cm","kind":"back"},
    {"id":"B_SHOULDERS_W","label":"Barki / ramiona od tyłu","kind":"back"},
    {"id":"B_WAIST_W","label":"Talia od tyłu","kind":"back"},
    {"id":"B_HIPS_W","label":"Biodra / pośladki od tyłu","kind":"back"},
    {"id":"B_SPINE_AXIS","label":"Oś ciała / kręgosłup wizualnie","kind":"back"},
    {"id":"B_SHOULDER_LINE","label":"Linia barków","kind":"back"},
    {"id":"B_PELVIS_LINE","label":"Linia miednicy","kind":"back"},
]

def schema_rows_for_ui() -> List[Dict]:
    rows: List[Dict] = []
    for row in FRONT_CENTRAL:
        rows.append({"view":"FRONT","id":row["id"],"label":row["label"],"field":row.get("field",""),"notes":"partia centralna"})
    for row in FRONT_LEFT_RIGHT:
        rows.append({"view":"FRONT","id":row["id"],"label":row["label"],"field":row.get("field",""),"notes":"kończyny osobno L/R"})
    for row in PROFILE_CORE:
        rows.append({"view":"PROFIL","id":row["id"],"label":row["label"],"field":row.get("field",""),"notes":"głębokość / profil"})
    for row in BACK_QA:
        rows.append({"view":"TYŁ","id":row["id"],"label":row["label"],"field":row.get("field",""),"notes":"symetria / postura / QA"})
    return rows
