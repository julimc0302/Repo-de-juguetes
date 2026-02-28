"""
strategy3_fewshot.py
--------------------
Strategy 3: Few-shot examples.

Provides two worked examples of Q&A pairs so the model learns the expected
style, depth, citation format, and structure before answering the real question.
"""
from __future__ import annotations

SYSTEM_PROMPT = """\
Eres un asistente de investigación académica especializado en crimen organizado,
extorsión y gobernanza criminal en América Latina. Respondes como un investigador
con experiencia en ciencias sociales comparadas.\
"""

# ---- Few-shot examples ----

EXAMPLE_1_Q = "¿Qué es la gobernanza criminal y cómo se diferencia del crimen ordinario?"
EXAMPLE_1_A = """\
## Gobernanza Criminal: Definición y Distinción

La gobernanza criminal se refiere a la provisión sistemática de orden, reglas y servicios
por parte de organizaciones criminales a poblaciones territorialmente delimitadas, usualmente
en ausencia o sustitución del Estado (Lessing, 2021). A diferencia del crimen ordinario,
la gobernanza criminal implica una relación continuada de autoridad: los actores criminales
no solo cometen delitos, sino que regulan comportamientos, resuelven disputas y proveen
seguridad a cambio de legitimidad y recursos.

**Evidencia empírica:** Arias y Barnes (2017) documentan en Rio de Janeiro cómo facciones
criminales coexisten con el Estado y actores comunitarios en "órdenes plurales", donde
distintas formas de autoridad se superponen y negocian. Feldmann y Luna (2022) amplían
este concepto al nivel latinoamericano, argumentando que la crisis del Estado en la región
ha creado vacíos que las organizaciones criminales llenan con sus propias estructuras
de gobernanza.

**Limitaciones:** La evidencia empírica se concentra en contextos urbanos (favelas, barrios
marginales), por lo que la aplicabilidad en áreas rurales o semi-rurales requiere cautela.\
"""

EXAMPLE_2_Q = "¿Cuáles son los factores que explican por qué algunas empresas resisten la extorsión?"
EXAMPLE_2_A = """\
## Factores de Resistencia a la Extorsión Empresarial

La literatura identifica múltiples factores que aumentan la probabilidad de que una
empresa resista a la extorsión, operando tanto a nivel individual como colectivo.

**A nivel individual:** Battisti et al. (2018) encuentran que las empresas con mayor
capital social —medido por participación en redes y asociaciones— son más propensas a
resistir en el contexto de la Mafia siciliana. La capacidad financiera también importa:
empresas con mayor liquidez pueden absorber represalias a corto plazo.

**A nivel colectivo:** Bull et al. (2024), estudiando El Salvador, muestran que las
micro y pequeñas empresas desarrollan estrategias adaptativas como relocalización,
diversificación de actividades e invisibilización, más que resistencia activa. La
acción colectiva organizada —como Addiopizzo en Italia— reduce los costos individuales
de resistir pero requiere coordinación sostenida.

**Factor estatal:** Moncada (2019) argumenta que la forma de resistencia adoptada depende
críticamente de la postura del Estado: cuando el Estado provee protección efectiva, la
resistencia activa es viable; cuando no, predominan estrategias de adaptación pasiva.

**Limitaciones:** La mayoría de estudios son de caso o regionales, lo que limita
generalizaciones causales robustas entre contextos.\
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
