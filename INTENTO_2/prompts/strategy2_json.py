"""
strategy2_json.py
-----------------
Strategy 2: Structured JSON output format.

Instructs the model to return a well-defined JSON object so that downstream
code can parse and display results programmatically (citations, confidence, etc.).
"""
from __future__ import annotations

SYSTEM_PROMPT = """\
Eres un asistente de investigación académica especializado en crimen organizado,
extorsión y gobernanza criminal en América Latina.

IMPORTANTE: Debes responder ÚNICAMENTE con un objeto JSON válido siguiendo el
esquema especificado. No incluyas texto antes o después del JSON.\
"""

USER_TEMPLATE = """\
Usando los fragmentos académicos a continuación, responde la pregunta de investigación.

FRAGMENTOS DE CONTEXTO:
{context}

PREGUNTA: {question}

Responde con el siguiente objeto JSON (sin texto adicional fuera del JSON):

{{
  "respuesta_principal": "Párrafo conciso de 3-5 oraciones con la respuesta directa",
  "hallazgos_clave": [
    {{
      "hallazgo": "Descripción del hallazgo",
      "fuente": "Apellido(s) (año)",
      "relevancia": "alta|media|baja"
    }}
  ],
  "paises_mencionados": ["lista", "de", "países"],
  "conceptos_clave": ["lista", "de", "conceptos", "centrales"],
  "limitaciones": "Qué no cubre la evidencia disponible o qué es incierto",
  "nivel_confianza": "alto|medio|bajo",
  "justificacion_confianza": "Por qué ese nivel de confianza en la respuesta"
}}\
"""


def build_prompt(question: str, context: str) -> list[dict]:
    """
    Build the OpenAI messages list for Strategy 2 (JSON output).

    Returns:
        List of {'role': ..., 'content': ...} dicts for the API.
    """
    user_content = USER_TEMPLATE.format(
        context=context,
        question=question,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
