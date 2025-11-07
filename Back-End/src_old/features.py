import numpy as np

FEATURE_ORDER = ["age","sex_M","height_cm","weight_kg","waist_cm",
                 "sleep_hours","smokes_cig_day","days_mvpa_week","fruit_veg_portions_day"]

def to_features(payload: dict) -> np.ndarray:
    sex_M = 1 if str(payload.get("sex","F")).upper() == "M" else 0
    x = [
        float(payload["age"]),
        float(sex_M),
        float(payload["height_cm"]),
        float(payload["weight_kg"]),
        float(payload["waist_cm"]),
        float(payload.get("sleep_hours", 7.0)),
        float(payload.get("smokes_cig_day", 0)),
        float(payload.get("days_mvpa_week", 0)),
        float(payload.get("fruit_veg_portions_day", 0.0)),
    ]
    return np.array(x, dtype=float)

