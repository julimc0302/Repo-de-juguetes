"""
strategy3_fewshot.py
--------------------
Strategy 3: Few-shot examples.

Provides two worked examples of Q&A pairs so the model learns the expected
style, depth, citation format, and structure before answering the real question.
"""
from __future__ import annotations

SYSTEM_PROMPT = """\
Eres un asistente de investigación especializado en inteligencia artificial,
machine learning y estrategias de inversión en mercados bursátiles. Respondes
como un investigador con experiencia en finanzas cuantitativas y ciencia de datos.\
"""

# ---- Few-shot examples ----

EXAMPLE_1_Q = "¿Qué son los modelos híbridos en predicción bursátil y por qué superan a los modelos individuales?"
EXAMPLE_1_A = """\
## Modelos Híbridos en Predicción Bursátil: Definición y Ventajas

Los modelos híbridos en predicción bursátil combinan múltiples algoritmos para aprovechar
las fortalezas de cada uno y mitigar sus limitaciones individuales. A diferencia de los modelos
tradicionales como ARIMA o LSTM en solitario, los enfoques híbridos integran métodos
estadísticos clásicos con técnicas de deep learning para capturar tanto dependencias lineales
como no lineales en los datos financieros (Vishwas et al., 2025).

**Evidencia empírica:** Polamuri et al. (2022) demostraron que el modelo MMGAN-HPA,
que combina Redes Generativas Adversariales (GANs) con ARIMA y LSTM, supera a los
modelos independientes al reducir el Error Absoluto Medio (MAE) y el Error Cuadrático
Medio (RMSE). Por su parte, Zhang y Chen (2023) introdujeron un enfoque en dos etapas
que combina Descomposición de Modo Variacional (VMD) con modelos ensemble como
Gradient Boosting y Random Forest, logrando mayor estabilidad predictiva frente a datos
no estacionarios y ruidosos.

**Limitaciones:** La mayoría de los modelos híbridos implican altos costos computacionales
y una complejidad considerable en el ajuste de hiperparámetros, lo que puede dificultar
su implementación en entornos de trading en tiempo real.\
"""

EXAMPLE_2_Q = "¿Cómo puede el análisis de sentimiento en redes sociales mejorar la predicción de precios en bolsa?"
EXAMPLE_2_A = """\
## Análisis de Sentimiento y Predicción Bursátil

El análisis de sentimiento en redes sociales ha emergido como una fuente de datos alternativa
valiosa para mejorar la predicción de precios en mercados financieros. Los modelos que
integran datos textuales de plataformas como Twitter con datos históricos de precios logran
capturar el estado emocional del mercado, un factor que los modelos puramente cuantitativos
no pueden detectar (Albahli et al., 2022).

**Evidencia empírica:** Gandhudi et al. (2024) implementaron una Red Neuronal Cuántica
Híbrida (HQNN) que integra análisis de sentimiento mediante FinBERT para clasificar
tweets como positivos, negativos o neutros. Sus resultados muestran que la incorporación
del sentimiento de Twitter mejora la precisión de predicción hasta en un 18% para acciones
volátiles. Para la interpretabilidad del modelo, utilizaron valores SHAP que identifican
los factores clave que impulsan cada predicción. Albahli et al. (2022) corroboran estos
hallazgos con el modelo Stock Senti WordNet (SSWN), combinando Random Forest y SVM
con análisis de sentimiento en tiempo real.

**Limitaciones:** Los datos de redes sociales son inherentemente ruidosos y no estructurados,
sensibles al contexto y difíciles de estandarizar. Además, la naturaleza dinámica de las
plataformas sociales introduce imprevisibilidad que puede reducir la robustez del modelo
en períodos de alta volatilidad.\
"""

# ---- Main template ----

USER_TEMPLATE = """\
A continuación te muestro dos ejemplos del tipo de respuesta esperada, seguidos de
los fragmentos de contexto y la pregunta que debes responder.

---
EJEMPLO 1:
Pregunta: {example_1_q}
Respuesta: {example_1_a}

---
EJEMPLO 2:
Pregunta: {example_2_q}
Respuesta: {example_2_a}

---
FRAGMENTOS DE CONTEXTO PARA LA PREGUNTA REAL:
{context}

---
PREGUNTA REAL: {question}

Responde siguiendo el mismo estilo, estructura y nivel de detalle de los ejemplos.
Cita los autores y años de los fragmentos de contexto cuando sean relevantes.\
"""


def build_prompt(question: str, context: str) -> list[dict]:
    """
    Build the OpenAI messages list for Strategy 3 (few-shot).

    Returns:
        List of {'role': ..., 'content': ...} dicts for the API.
    """
    user_content = USER_TEMPLATE.format(
        example_1_q=EXAMPLE_1_Q,
        example_1_a=EXAMPLE_1_A,
        example_2_q=EXAMPLE_2_Q,
        example_2_a=EXAMPLE_2_A,
        context=context,
        question=question,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]