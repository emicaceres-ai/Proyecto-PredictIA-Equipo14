import os, json, requests, io
import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

API = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Salud Preventiva Demo", page_icon="💙", layout="centered")
st.title("Salud Preventiva Demo 💙")
st.caption("Calcula un riesgo estimado y genera un plan de 2 semanas (demo).")

# --- Estado inicial ---
if "predict" not in st.session_state:
    st.session_state.predict = None
if "payload" not in st.session_state:
    st.session_state.payload = None
if "plan" not in st.session_state:
    st.session_state.plan = None

# --- Formulario ---
with st.form("perfil"):
    c1, c2 = st.columns(2)
    age = c1.number_input("Edad", min_value=18, max_value=85, value=35)
    sex = c2.selectbox("Sexo", ["F", "M"], index=0)

    c3, c4 = st.columns(2)
    height_cm = c3.number_input("Estatura (cm)", min_value=120, max_value=220, value=165)
    weight_kg = c4.number_input("Peso (kg)", min_value=30, max_value=220, value=68)

    waist_cm = st.number_input("Cintura (cm)", min_value=40, max_value=170, value=85)

    c5, c6, c7 = st.columns(3)
    sleep_hours = c5.number_input("Horas de sueño", min_value=3.0, max_value=14.0, value=7.0, step=0.5)
    smokes_cig_day = c6.number_input("Cigarrillos/día", min_value=0, max_value=60, value=0)
    days_mvpa_week = c7.number_input("Días actividad/semana", min_value=0, max_value=7, value=3)

    fruit_veg_portions_day = st.number_input(
        "Porciones fruta/verdura/día", min_value=0.0, max_value=12.0, value=3.0, step=0.5
    )

    submitted = st.form_submit_button("Calcular riesgo")

# --- Acción: Calcular riesgo ---
if submitted:
    payload = {
        "age": age,
        "sex": sex,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "waist_cm": waist_cm,
        "sleep_hours": sleep_hours,
        "smokes_cig_day": smokes_cig_day,
        "days_mvpa_week": days_mvpa_week,
        "fruit_veg_portions_day": fruit_veg_portions_day,
    }
    st.session_state.payload = payload  # ✅ guardamos el payload

    try:
        with st.spinner("Calculando riesgo..."):
            r = requests.post(f"{API}/predict", json=payload, timeout=30)
        r.raise_for_status()
        st.session_state.predict = r.json()
        st.session_state.plan = None
    except requests.RequestException as e:
        st.session_state.predict = None
        st.error(f"Error en /predict: {e}")

# --- Mostrar resultado ---
pred = st.session_state.predict
if pred:
    st.subheader("Resultado")

    score = pred.get("score", pred.get("probability", 0.0))
    calibrated = pred.get("calibrated", pred.get("calibrated_probability", score))

    st.metric("Score (0-1)", f"{score:.2f}")

    if "drivers" in pred:
        st.write("**Factores principales:**")
        for d in pred["drivers"]:
            st.write(f"- {d['feature']}: {d['contribution']:+.2f}")

    if pred.get("referral_flag", False) or calibrated >= 0.70:
        st.warning("Recomendación: considerar consulta profesional.")

    # --- Generar plan ---
    if st.button("Generar plan de 2 semanas"):
        # ✅ recuperamos el payload guardado para evitar NameError
        base_payload = st.session_state.get("payload", {})
        coach_payload = dict(**base_payload, score=calibrated)

        try:
            with st.spinner("Generando plan..."):
                rc = requests.post(f"{API}/coach", json=coach_payload, timeout=60)
            rc.raise_for_status()
            st.session_state.plan = rc.json()
        except requests.RequestException as e:
            st.error(f"Error en /coach: {e}")

# --- Mostrar plan y exportar PDF ---
plan = st.session_state.plan
if plan:
    st.subheader("Plan")
    st.write(plan.get("plan_text") or "")

    steps = plan.get("steps") or []
    if steps:
        st.write("**Pasos clave**")
        for s in steps:
            st.write(f"- {s}")

    if plan.get("disclaimer"):
        st.caption(plan["disclaimer"])

    # Exportar PDF
    if st.button("Exportar PDF"):
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        w, h = A4
        y = h - 50

        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, y, "Informe Salud Preventiva")
        y -= 20

        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"Score: {score:.2f}")
        y -= 20

        if "drivers" in pred:
            c.drawString(40, y, "Factores:")
            y -= 15
            for d in pred["drivers"]:
                c.drawString(50, y, f"- {d['feature']}: {d['contribution']:+.2f}")
                y -= 15
                if y < 80:
                    c.showPage()
                    y = h - 80

        y -= 10
        c.drawString(40, y, "Plan (resumen):")
        y -= 15
        for s in steps[:200]:
            c.drawString(50, y, f"- {s[:90]}")
            y -= 15
            if y < 80:
                c.showPage()
                y = h - 80

        c.showPage()
        c.save()

        st.download_button(
            "Descargar PDF",
            data=buf.getvalue(),
            file_name="informe_salud.pdf",
            mime="application/pdf",
        )
