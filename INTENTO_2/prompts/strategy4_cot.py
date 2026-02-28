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
Eres un asistente de investigación especializado en inteligencia artificial,
machine learning y estrategias de inversión en mercados bursátiles. Piensas
de manera sistemática y rigurosa, mostrando tu razonamiento antes de llegar
a conclusiones, con base en evidencia empírica y modelos cuantitativos.\
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
¿A qué tipo de mercado, activo financiero o horizonte temporal aplica?
¿Se trata de predicción de precios, gestión de riesgo, trading algorítmico u otro enfoque?

PASO 2 — INVENTARIO DE EVIDENCIA
¿Qué fragmentos del contexto son directamente relevantes?
Para cada fragmento relevante: ¿qué modelo o algoritmo propone?
¿Cuáles son sus métricas de rendimiento (MAE, RMSE, accuracy, Sharpe Ratio)?
¿Quiénes son los autores y en qué dataset o mercado fue validado?

PASO 3 — ANÁLISIS DE PATRONES
¿Hay acuerdo entre las fuentes sobre qué modelos funcionan mejor?
¿Existen contradicciones en los resultados según el tipo de mercado o datos?
¿Qué ventajas y limitaciones comparten los enfoques mencionados?

PASO 4 — IDENTIFICACIÓN DE GAPS
¿Qué aspecto de la pregunta NO cubre la evidencia disponible?
¿Qué mercados, activos o condiciones de mercado quedan sin estudiar?
¿Qué líneas de investigación futura se sugieren?

PASO 5 — SÍNTESIS Y RESPUESTA FINAL
Con base en los pasos anteriores, proporciona:
- Respuesta académica clara y bien argumentada (3-5 párrafos)
- Modelos y algoritmos clave citados en formato (Autor, año) integrados en el texto
- Conclusión sobre las implicaciones prácticas para inversores o gestores de portafolio

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