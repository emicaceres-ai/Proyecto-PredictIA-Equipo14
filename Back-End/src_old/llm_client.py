import os
import json
from typing import Any, Dict, Tuple, List

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _client():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)

def extract_json_from_text(user_text: str) -> Dict[str, Any]:
    """NL -> JSON. If LLM not configured, return clarification request."""
    prompt = _read(os.path.join(PROMPTS_DIR, "extractor.txt"))
    client = _client()
    if client is None:
        return {"need_clarification": "LLM no configurado. Por favor ingresa los datos manualmente."}
    # Simple call with JSON objective
    sys = "Responde SOLO con JSON válido (UTF-8)."
    msg = [
        { "role": "system", "content": sys + "\n\n" + prompt },
        { "role": "user", "content": user_text }
    ]
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=msg, temperature=0.0)
    txt = resp.choices[0].message.content
    try:
        return json.loads(txt)
    except Exception:
        return {"need_clarification": "No pude generar un JSON válido. Intenta ingresar los datos manualmente."}

def coach_with_rag(score: float, profile: Dict[str, Any], kb_texts: List[Tuple[str,str]]) -> Tuple[str, List[str], List[Dict[str,str]]]:
    """Generate a plan using LLM + small RAG. If LLM not set, returns a simple template."""
    citations = [{"kb_doc": name, "section": "general"} for name, _ in kb_texts]
    client = _client()
    if client is None:
        # Fallback simple plan
        steps = [
            "Dormir 7-8 h cada noche, horario fijo.",
            "Caminar 30 min, 5 días/semana.",
            "Agregar 2 porciones de verduras y 2 de frutas al día.",
            "Respiración 4-7-8 por 5 min antes de dormir.",
            "Registrar hábitos cada noche (checklist)."
        ]
        plan = "Plan de 2 semanas (versión básica sin LLM)."
        return plan, steps, citations
    prompt = _read(os.path.join(PROMPTS_DIR, "coach.txt"))
    context = "\n\n".join([f"[{name}]\n{txt}" for name, txt in kb_texts[:4]])
    sys = "No des diagnósticos. Usa pasos SMART y citas a la kb."
    user = f"Score={score:.2f}. Perfil={json.dumps(profile, ensure_ascii=False)}. Contexto KB:\n{context}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content": sys + "\n\n" + prompt},
            {"role":"user","content": user}
        ],
        temperature=0.2
    )
    txt = resp.choices[0].message.content or ""
    # Construir salida simple
    steps = [s.strip("- ").strip() for s in txt.split("\n") if s.strip().startswith("-")]
    if not steps:
        steps = ["Plan generado. Revisa recomendaciones por secciones en el texto."]
    return txt, steps[:8], citations
