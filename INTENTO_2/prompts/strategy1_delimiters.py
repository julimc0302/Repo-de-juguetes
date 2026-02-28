"""
strategy1_delimiters.py
-----------------------
Strategy 1: Structured sections with XML-style delimiters.

Uses clearly delimited sections (<<<CONTEXT>>>, <<<QUESTION>>>, <<<INSTRUCTIONS>>>)
to separate the retrieved context from the question and instructions.
This makes the prompt highly readable and helps the model parse each section.
"""
from __future__ import annotations

SYSTEM_PROMPT = """\
Eres un asistente de investigación académica especializado en crimen organizado,
extorsión y gobernanza criminal en América Latina. Tu función es responder preguntas
de investigación basándote exclusivamente en los artículos académicos proporcionados.

Principios de respuesta:
- Cita explícitamente los autores y años cuando uses información específica.
- Distingue entre hallazgos empíricos y afirmaciones teóricas.
- Si la evidencia es limitada o contradictoria, indícalo claramente.
- Responde en el mismo idioma en que se formula la pregunta.
- No inventes información que no esté en los textos proporcionados.\
"""

USER_TEMPLATE = """\
<<<CONTEXTO ACADÉMICO>>>
Los siguientes fragmentos provienen de artículos académicos revisados por pares.
Úsalos como única fuente de evidencia para tu respuesta.

{context}
<<<FIN DEL CONTEXTO>>>

<<<PREGUNTA DE INVESTIGACIÓN>>>
{question}
<<<FIN DE LA PREGUNTA>>>

<<<INSTRUCCIONES>>>
1. ANÁLISIS: Lee cuidadosamente todos los fragmentos del contexto.
2. SÍNTESIS: Identifica los hallazgos principales relevantes a la pregunta.
3. RESPUESTA: Proporciona una respuesta académica estructurada con:
   - Respuesta principal (2-3 párrafos)
   - Evidencia clave citando autores (Apellido, año)
   - Limitaciones o gaps en la evidencia disponible
4. FORMATO: Usa encabezados claros. Sé preciso y académico.
<<<FIN DE INSTRUCCIONES>>>

Proporciona tu respuesta a continuación:\
"""


def build_prompt(question: str, context: str) -> list[dict]:
    """
    Build the OpenAI messages list for Strategy 1 (delimiters).

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
