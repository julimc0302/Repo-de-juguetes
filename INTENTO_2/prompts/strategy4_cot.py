"""
strategy4_cot.py
----------------
Strategy 4: Chain-of-Thought (CoT) reasoning.

Explicitly instructs the model to reason step by step before producing its final
answer. This improves accuracy on complex analytical questions by making
the reasoning process visible and structured.
"""
from __future__ import annotations

SYSTEM_PROMPT = """\
Eres un asistente de investigación académica especializado en crimen organizado,
extorsión y gobernanza criminal en América Latina. Piensas de manera sistemática
y rigurosa, mostrando tu razonamiento antes de llegar a conclusiones.\
"""

USER_TEMPLATE = """\
Responde la pregunta de investigación usando los fragmentos académicos proporcionados.
Debes pensar paso a paso siguiendo el proceso de razonamiento indicado.

FRAGMENTOS DE CONTEXTO:
{context}

PREGUNTA: {question}

---
Sigue EXACTAMENTE estos pasos de razonamiento:

PASO 1 — COMPRENSIÓN DE LA PREGUNTA
¿Qué pregunta exactamente? ¿Qué conceptos clave están involucrados?
¿Cuál es el alcance geográfico y temporal implícito?

PASO 2 — INVENTARIO DE EVIDENCIA
¿Qué fragmentos del contexto son directamente relevantes?
Para cada fragmento relevante: ¿qué afirma? ¿quiénes son los autores? ¿en qué contexto?

PASO 3 — ANÁLISIS DE PATRONES
¿Hay acuerdo entre las fuentes? ¿Contradicciones? ¿Complementariedades?
¿Qué generalizaciones se pueden hacer con confianza?

PASO 4 — IDENTIFICACIÓN DE GAPS
¿Qué aspecto de la pregunta NO cubre la evidencia disponible?
¿Qué habría que investigar adicionalmente?

PASO 5 — SÍNTESIS Y RESPUESTA FINAL
Con base en los pasos anteriores, proporciona:
- Respuesta académica clara y bien argumentada (3-5 párrafos)
- Citaciones en formato (Autor, año) integradas en el texto
- Conclusión sobre las implicaciones para la investigación en la región

Comienza con PASO 1:\
"""


def build_prompt(question: str, context: str) -> list[dict]:
    """
    Build the OpenAI messages list for Strategy 4 (Chain-of-Thought).

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
