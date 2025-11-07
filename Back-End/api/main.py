# api/main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model import predict_proba, explain_local, FEATURE_ORDER
from .schemas import PredictIn, PredictOut, CoachIn, CoachOut

# .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# LLM opcional (OpenAI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_LLM = True if OPENAI_API_KEY else False
llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

client = None
if USE_LLM:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None
        USE_LLM = False

app = FastAPI(title="Salud Preventiva API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod: pon tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

THRESHOLD_REFERRAL = float(os.getenv("THRESHOLD_REFERRAL", "0.70"))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/features")
def features():
    return {"feature_order": FEATURE_ORDER}

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    try:
        form = payload.model_dump()
        # trazas básicas
        print("[/predict] form:", form)

        # 1) prob
        p_raw, p_cal = predict_proba(form)
        print("[/predict] p_raw, p_cal:", p_raw, p_cal)

        # 2) drivers
        drivers = explain_local(form, top_k=5)
        print("[/predict] drivers:", drivers)

        out = {
            "probability": float(p_raw),
            "calibrated_probability": float(p_cal),
            "drivers": [{"feature": k, "contribution": float(v)} for k, v in drivers],
            "referral_flag": bool(p_cal >= THRESHOLD_REFERRAL),
        }
        print("[/predict] response:", out)
        return out

    except Exception as e:
        # imprime stack en servidor y devuelve detalle corto
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Predict error: {e}")

@app.post("/coach", response_model=CoachOut)
def coach(payload: CoachIn):
    try:
        score = float(payload.score)
        user = payload.model_dump()
        base_prompt = f"""
Eres un coach de salud preventiva. Con el perfil y riesgo del usuario,
crea un plan de 2 semanas con pasos concretos (sueño, actividad física,
alimentación y manejo de estrés), lenguaje simple y accionable. No des diagnósticos.

Perfil:
- Edad: {user['age']}
- Sexo: {user['sex']}
- Estatura (cm): {user['height_cm']}
- Peso (kg): {user['weight_kg']}
- Cintura (cm): {user['waist_cm']}
- Horas de sueño: {user.get('sleep_hours')}
- Cigarrillos/día: {user.get('smokes_cig_day')}
- Días actividad/semana: {user.get('days_mvpa_week')}
- Porciones fruta/verdura/día: {user.get('fruit_veg_portions_day')}
- Riesgo (0-1, calibrado): {score:.2f}

Devuelve:
- Un texto resumen (3-5 líneas)
- 8-12 pasos en viñetas (cortos, medibles)
""".strip()

        if USE_LLM and client is not None:
            resp = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "Eres un profesional en salud preventiva. Sé claro y conciso."},
                    {"role": "user", "content": base_prompt},
                ],
                temperature=0.4,
                max_tokens=600,
            )
            txt = resp.choices[0].message.content.strip()
            lines = [l.strip(" •-") for l in txt.splitlines() if l.strip()]
            summary = lines[0] if lines else ""
            steps = [l for l in lines[1:] if len(l) > 2][:12]
        else:
            summary = "Plan base de 2 semanas: prioriza buen sueño, actividad moderada diaria, más frutas/verduras e higiene del estrés."
            steps = [
                "Acuéstate y levántate a la misma hora (+/- 30 min).",
                "Camina 30 min diarios a ritmo cómodo.",
                "Haz 2 sesiones de fuerza de 15–20 min/semana (peso corporal).",
                "Agrega 2 porciones de frutas/verduras al día.",
                "Toma 6–8 vasos de agua/día.",
                "Respiración 4-7-8 por 3 minutos antes de dormir.",
                "Reduce pantallas 60 min antes de dormir.",
                "Si fumas, fija una ventana libre de cigarrillos de 4 horas.",
                "Registra hábitos 5/7 días para medir progreso.",
            ]

        return {
            "plan_text": summary,
            "steps": steps,
            "citations": [],
            "disclaimer": "Este asistente no entrega diagnósticos. Si tu riesgo es alto o presentas síntomas, consulta con un profesional de salud.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coach error: {e}")
@app.get("/_debug")
def _debug():
    try:
        from .model import MODEL, CALIBRATOR, FEATURE_ORDER
        return {
            "model_type": str(type(MODEL)),
            "has_predict_proba": hasattr(MODEL, "predict_proba"),
            "has_decision_function": hasattr(MODEL, "decision_function"),
            "has_predict": hasattr(MODEL, "predict"),
            "calibrator_type": str(type(CALIBRATOR)) if CALIBRATOR is not None else None,
            "cal_has_predict_proba": hasattr(CALIBRATOR, "predict_proba") if CALIBRATOR else None,
            "n_features": len(FEATURE_ORDER),
            "first_10_features": FEATURE_ORDER[:10],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"_debug error: {e}")
    