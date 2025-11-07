# api/model.py — robusto con fallback si no hay estimador válido
import os, json, warnings
import numpy as np
from joblib import load
from typing import Dict, List, Tuple, Any

# Rutas (desde .env o por defecto a ./data)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(DATA_DIR, "test_arbol_decision.joblib"))
CALIBRATOR_PATH = os.getenv("CALIBRATOR_PATH", os.path.join(DATA_DIR, "calibrator.joblib"))
FEATURE_ORDER_PATH = os.getenv("FEATURE_ORDER_PATH", os.path.join(DATA_DIR, "feature_order.json"))

# Conjunto mínimo por si faltara el JSON
MIN_FEATURES = [
    "age","height_cm","weight_kg","waist_cm",
    "sleep_hours","smokes_cig_day","days_mvpa_week","fruit_veg_portions_day",
    "sex_M","sex_F"
]

DEFAULTS = {
    "age": 35, "sex_M": 0.0, "sex_F": 1.0, "height_cm": 165.0, "weight_kg": 68.0,
    "waist_cm": 85.0, "sleep_hours": 7.0, "smokes_cig_day": 0.0,
    "days_mvpa_week": 3.0, "fruit_veg_portions_day": 3.0,
    "imc": None,
}

def _compute_bmi(height_cm: float, weight_kg: float) -> float:
    h_m = max(1e-6, float(height_cm)) / 100.0
    return float(weight_kg) / (h_m ** 2)

def _unwrap_estimator(obj: Any):
    # Caso directo
    if hasattr(obj, "predict_proba") or hasattr(obj, "decision_function") or hasattr(obj, "predict"):
        return obj
    # Dict con posibles claves
    if isinstance(obj, dict):
        for key in ["model","estimator","clf","pipeline","best_estimator_"]:
            if key in obj:
                inner = obj[key]
                if hasattr(inner, "predict_proba") or hasattr(inner, "decision_function") or hasattr(inner, "predict"):
                    return inner
        # A veces serializan steps estilo Pipeline
        if "steps" in obj and isinstance(obj["steps"], list) and obj["steps"]:
            last = obj["steps"][-1]
            if isinstance(last, (list, tuple)) and len(last) == 2:
                inner = last[1]
                if hasattr(inner, "predict_proba") or hasattr(inner, "decision_function") or hasattr(inner, "predict"):
                    return inner
    # Lista/tupla: último elemento
    if isinstance(obj, (list, tuple)) and obj:
        inner = obj[-1]
        if hasattr(inner, "predict_proba") or hasattr(inner, "decision_function") or hasattr(inner, "predict"):
            return inner
    return None  # no encontramos estimador válido

def _unwrap_calibrator(obj: Any):
    if obj is None:
        return None
    if hasattr(obj, "predict_proba") or hasattr(obj, "predict"):
        return obj
    if isinstance(obj, dict):
        for key in ["calibrator","calibration","cal","isotonic","sigmoid"]:
            if key in obj:
                inner = obj[key]
                if hasattr(inner, "predict_proba") or hasattr(inner, "predict"):
                    return inner
    if isinstance(obj, (list, tuple)) and obj:
        inner = obj[-1]
        if hasattr(inner, "predict_proba") or hasattr(inner, "predict"):
            return inner
    return None

def _load_feature_order() -> List[str]:
    # Intenta leer el JSON, si no existe usa el mínimo
    path = FEATURE_ORDER_PATH
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            feats = json.load(f)
        # Asegura sex_M/sex_F si usas sexo binario
        if "sex_M" not in feats: feats.append("sex_M")
        if "sex_F" not in feats: feats.append("sex_F")
        return feats
    # Si no hay JSON, usa set mínimo y créalo para futuras corridas
    feats = MIN_FEATURES[:]
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(feats, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return feats

def _make_fallback_model(n_features: int):
    """Crea y entrena un modelo simple con datos sintéticos del tamaño correcto,
    para permitir integración front↔back aunque no haya modelo real válido."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(42)
    n = 1000
    X = rng.normal(0, 1, size=(n, n_features))
    w = rng.normal(0, 0.5, size=(n_features,))
    z = X @ w + rng.normal(0, 0.5, size=(n,))
    prob = 1.0 / (1.0 + np.exp(-z))
    y = (prob > 0.5).astype(int)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    warnings.warn("[model.py] Usando FALLBACK model (sintético). Reemplaza data/test_arbol_decision.joblib con un estimador real.")
    return clf

def load_artifacts():
    # Carga features primero para saber cuántas dimensiones debe tener el vector
    feature_order: List[str] = _load_feature_order()

    # Carga modelo
    model = None
    if os.path.exists(MODEL_PATH):
        raw_model = load(MODEL_PATH)
        model = _unwrap_estimator(raw_model)

    if model is None:
        # No había estimador válido → usa fallback
        model = _make_fallback_model(len(feature_order))

    # Carga calibrador (opcional)
    calibrator = None
    if os.path.exists(CALIBRATOR_PATH):
        raw_cal = load(CALIBRATOR_PATH)
        calibrator = _unwrap_calibrator(raw_cal)

    return model, calibrator, feature_order

MODEL, CALIBRATOR, FEATURE_ORDER = load_artifacts()

def form_to_features(form: Dict) -> Dict[str, float]:
    sex = (form.get("sex") or "F").upper()
    sex_M = 1.0 if sex == "M" else 0.0
    sex_F = 1.0 - sex_M
    feat = {
        "age": float(form.get("age", DEFAULTS["age"])),
        "height_cm": float(form.get("height_cm", DEFAULTS["height_cm"])),
        "weight_kg": float(form.get("weight_kg", DEFAULTS["weight_kg"])),
        "waist_cm": float(form.get("waist_cm", DEFAULTS["waist_cm"])),
        "sleep_hours": float(form.get("sleep_hours", DEFAULTS["sleep_hours"] or 7.0) or 7.0),
        "smokes_cig_day": float(form.get("smokes_cig_day", DEFAULTS["smokes_cig_day"] or 0.0) or 0.0),
        "days_mvpa_week": float(form.get("days_mvpa_week", DEFAULTS["days_mvpa_week"] or 0.0) or 0.0),
        "fruit_veg_portions_day": float(form.get("fruit_veg_portions_day", DEFAULTS["fruit_veg_portions_day"] or 0.0) or 0.0),
        "sex_M": float(sex_M),
        "sex_F": float(sex_F),
    }
    if "imc" in FEATURE_ORDER:
        feat["imc"] = _compute_bmi(feat["height_cm"], feat["weight_kg"])
    return feat

def vectorize(feat: Dict[str, float], feature_order: List[str]) -> np.ndarray:
    vals = []
    for name in feature_order:
        val = feat.get(name)
        if val is None:
            val = DEFAULTS.get(name, 0.0) if DEFAULTS.get(name) is not None else 0.0
        vals.append(float(val))
    return np.array(vals, dtype=float).reshape(1, -1)

def _predict_proba_safe(estimator, X: np.ndarray) -> float:
    if hasattr(estimator, "predict_proba"):
        return float(estimator.predict_proba(X)[0, 1])
    if hasattr(estimator, "decision_function"):
        z = np.asarray(estimator.decision_function(X), dtype=float)[0]
        return float(1.0 / (1.0 + np.exp(-z)))
    if hasattr(estimator, "predict"):
        y = estimator.predict(X)[0]
        return float(y)
    raise ValueError("El estimador no soporta predict_proba/decision_function/predict.")

def predict_proba(form: Dict) -> Tuple[float, float]:
    feats = form_to_features(form)
    X = vectorize(feats, FEATURE_ORDER)
    p_raw = _predict_proba_safe(MODEL, X)
    if CALIBRATOR is not None:
        try:
            if hasattr(CALIBRATOR, "predict_proba"):
                p_cal = float(CALIBRATOR.predict_proba(np.array([[p_raw]]))[0, 1])
            else:
                p_cal = float(CALIBRATOR.predict(np.array([[p_raw]]))[0])
        except Exception:
            p_cal = p_raw
    else:
        p_cal = p_raw
    return p_raw, p_cal

def explain_local(form: Dict, top_k: int = 5):
    feats = form_to_features(form)
    X = vectorize(feats, FEATURE_ORDER)
    importances = getattr(MODEL, "feature_importances_", None)

    if importances is None:
        # baseline naive si el modelo no expone importancias
        contrib = []
        for i, name in enumerate(FEATURE_ORDER):
            baseline = DEFAULTS.get(name, 0.0) if DEFAULTS.get(name) is not None else 0.0
            contrib.append((name, float(X[0, i] - baseline)))
    else:
        imp = np.array(importances, dtype=float)
        if imp.sum() <= 0:
            imp = np.ones_like(imp) / len(imp)
        else:
            imp = imp / imp.sum()
        base = np.array([DEFAULTS.get(n, 0.0) if DEFAULTS.get(n) is not None else 0.0 for n in FEATURE_ORDER], dtype=float)
        contrib_vals = imp * (X[0] - base)
        contrib = list(zip(FEATURE_ORDER, contrib_vals.astype(float)))

    contrib = sorted(contrib, key=lambda kv: -abs(kv[1]))[:top_k]
    return contrib
