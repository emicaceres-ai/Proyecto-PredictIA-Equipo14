# Salud NHANES Starter (Backend + Frontend + LLM + RAG)

Proyecto de ejemplo **para principiantes** que integra:
- **FastAPI** (backend) con endpoints `/predict` y `/coach`.
- **Streamlit** (frontend) que consume la API.
- **LLM Wrapper** (OpenAI por defecto, opcional) para extracción NL→JSON y plan de coaching.
- **RAG** simple con BM25 contra `/kb`.
- **Plantillas** para entrenar un modelo tabular (scikit-learn/XGBoost).

## Requisitos
- Python 3.11
- (Opcional) Cuenta en OpenAI y variable `OPENAI_API_KEY` en `.env`

## Instalación rápida
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Ejecutar backend (API)
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Ejecutar frontend (Streamlit)
```bash
streamlit run app/streamlit_app.py
```

## Flujo de uso
1. Abre Streamlit → rellena el formulario → “Calcular riesgo” (llama `/predict`).
2. “Generar plan” → (llama `/coach`) con LLM+RAG si `OPENAI_API_KEY` está definido. Si no, usa un plan básico de ejemplo.
3. “Exportar PDF” → descarga un informe con score, drivers y plan.

## Archivos importantes
- `api/main.py`: endpoints.
- `src/llm_client.py`: integración LLM (modo seguro si no hay API key).
- `src/rag.py`: búsqueda en `/kb`.
- `src/features.py`: transformación de inputs a features del modelo.
- `src/model.py`: entrenamiento y carga del modelo (placeholder listo para extender).

> Sube tus datasets a `data/` (por ejemplo `DEMO_B.xpt`) y edita `src/load.py` + `src/features.py` para construir `X/y`.
