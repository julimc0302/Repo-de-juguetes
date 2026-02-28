# Research Copilot — Sistema RAG sobre Papers Académicos de IA en Mercados Financieros

> Sistema de Recuperación y Generación Aumentada (RAG) que permite hacer preguntas en lenguaje natural sobre un corpus de 21 artículos académicos acerca de inteligencia artificial aplicada a mercados financieros, implementado en Google Colab.

---

## Tabla de Contenidos

1. [Descripción del Proyecto](#1-descripción-del-proyecto)
2. [Sobre el Corpus de Artículos](#2-sobre-el-corpus-de-artículos)
3. [Funcionalidades](#3-funcionalidades)
4. [Arquitectura](#4-arquitectura)
5. [Instalación](#5-instalación)
6. [Uso](#6-uso)
7. [Detalles Técnicos](#7-detalles-técnicos)
8. [Resultados de Evaluación](#8-resultados-de-evaluación)
9. [Limitaciones](#9-limitaciones)
10. [Información del Autor](#10-información-del-autor)

---

## 1. Descripción del Proyecto

**Research Copilot** es un sistema de pregunta-respuesta basado en el paradigma RAG *(Retrieval-Augmented Generation)*. Indexa un corpus de **21 artículos académicos** sobre inteligencia artificial aplicada a mercados financieros y permite a los usuarios consultar esa base de conocimiento mediante lenguaje natural, recibiendo respuestas fundamentadas con citas académicas.

### ¿Qué es RAG?

Los modelos de lenguaje tradicionales generan respuestas a partir del conocimiento paramétrico adquirido durante el preentrenamiento: un conocimiento estático, potencialmente desactualizado e imposible de auditar. RAG resuelve estas limitaciones añadiendo un paso de recuperación dinámico:

```
┌─────────────────────────────────────────────────────────────┐
│  1. RECUPERAR  →  Buscar fragmentos relevantes del corpus   │
│  2. AUMENTAR   →  Incorporar el contexto al prompt          │
│  3. GENERAR    →  El LLM produce una respuesta con citas    │
└─────────────────────────────────────────────────────────────┘
```

Cada respuesta es trazable a pasajes concretos de los artículos indexados, minimizando las alucinaciones y permitiendo la verificación de fuentes.

---

## 2. Sobre el Corpus de Artículos

### Contexto General

La intersección entre inteligencia artificial y mercados financieros es uno de los campos de investigación de mayor crecimiento en la última década. Los avances en aprendizaje profundo, disponibilidad de datos de alta frecuencia y capacidad computacional han impulsado el desarrollo de modelos que buscan predecir precios, detectar anomalías, automatizar decisiones de inversión y anticipar crisis sistémicas.

Sin embargo, este campo también plantea desafíos importantes: los mercados financieros son sistemas dinámicos, no estacionarios y sujetos a cambios de régimen abruptos. Los modelos de IA entrenados en condiciones históricas pueden fallar ante eventos excepcionales (*cisnes negros*), y su adopción masiva puede generar nuevos riesgos sistémicos.

El corpus de 21 artículos indexado en este sistema fue seleccionado para cubrir de manera equilibrada tanto los avances técnicos como las implicaciones regulatorias y los límites documentados de estas tecnologías.

### Ejes Temáticos del Corpus

Los 21 artículos se agrupan en torno a cinco grandes ejes temáticos:

---

#### Eje 1 — Predicción de Precios con Redes Neuronales

El tema más representado en el corpus. Los artículos en este eje evalúan distintas arquitecturas de redes neuronales para la predicción de series temporales financieras:

- **LSTM (Long Short-Term Memory):** Las redes LSTM son el enfoque dominante en la literatura revisada. Su capacidad para capturar dependencias temporales de largo plazo las hace adecuadas para series de precios, las cuales presentan memoria y autocorrelación. Múltiples artículos documentan que LSTM supera a modelos econométricos clásicos como ARIMA en horizontes de predicción cortos y medianos.
- **GRU (Gated Recurrent Unit):** Variante más eficiente de LSTM con desempeño comparable en varios benchmarks. Algunos artículos comparan ambas arquitecturas directamente.
- **Transformers y mecanismos de atención:** Los artículos más recientes del corpus adoptan arquitecturas Transformer —originalmente diseñadas para procesamiento de lenguaje natural— y reportan mejoras significativas frente a LSTM, especialmente en predicción multi-paso.
- **Modelos de ensamble:** Varios artículos combinan redes neuronales con métodos de boosting (XGBoost, LightGBM) y reportan los mejores resultados absolutos en términos de MAE y RMSE.

**Artículos representativos:**
- Zhang et al. (2021). *Stock Price Prediction Using LSTM.*
- Smith, J. (2020). *Machine Learning in Stock Price Prediction.*

---

#### Eje 2 — Trading Algorítmico e IA

Este eje aborda el uso de IA en sistemas de toma de decisiones automáticas para la compra y venta de activos:

- **Estrategias basadas en aprendizaje por refuerzo (RL):** Agentes entrenados para maximizar retornos ajustados por riesgo, sin requerir etiquetado de datos de entrenamiento.
- **Detección de patrones técnicos:** Modelos de visión computacional aplicados a gráficos de velas (*candlestick charts*) para identificar figuras chartistas.
- **Ejecución óptima de órdenes:** Algoritmos que minimizan el impacto de mercado al ejecutar órdenes grandes (*market impact*), basados en modelos de liquidez.
- **Retroalimentación y amplificación de volatilidad:** Los artículos más críticos documentan que el trading algorítmico puede amplificar la volatilidad intradiaria y contribuir a *flash crashes*, especialmente cuando múltiples agentes utilizan estrategias similares.

**Artículo representativo:**
- *Algorithmic Trading and AI: A Review.*

---

#### Eje 3 — Predicción de Crisis Financieras y Riesgo Sistémico

Un subconjunto del corpus se dedica a aplicaciones preventivas de la IA:

- **Sistemas de alerta temprana:** Modelos que combinan indicadores macroeconómicos, datos de mercado y señales de sentimiento para predecir recesiones o crisis bancarias con meses de anticipación.
- **Scoring de riesgo crediticio:** Aplicación de gradient boosting y redes neuronales para estimar la probabilidad de default, superando a modelos logísticos tradicionales.
- **Contagio y riesgo de red:** Artículos que modelan el sistema financiero como un grafo y aplican algoritmos de aprendizaje automático para detectar instituciones sistémicamente relevantes (*too big to fail*).
- **Límites del enfoque:** La literatura también documenta que estos modelos tienen rendimiento muy degradado frente a eventos excepcionales no observados en el período de entrenamiento (pandemia, guerras, etc.).

---

#### Eje 4 — Procesamiento de Lenguaje Natural (NLP) en Finanzas

Este eje cubre el análisis de información textual no estructurada aplicada a mercados:

- **Análisis de sentimiento financiero:** Modelos entrenados sobre noticias, reportes de ganancias y redes sociales (Twitter/X) para anticipar movimientos de precio.
- **Extracción de información de reportes anuales:** NLP aplicado a los formularios 10-K de la SEC para identificar señales de riesgo en el lenguaje corporativo.
- **Modelos de lenguaje especializados:** Algunos artículos evalúan versiones de BERT y GPT *fine-tuneadas* sobre corpus financieros, demostrando mejoras respecto a modelos de propósito general.

---

#### Eje 5 — Perspectiva Regulatoria y Riesgos de la IA en Mercados

El corpus incluye perspectivas institucionales y críticas sobre las implicancias del uso masivo de IA en los mercados:

- **Advertencias de la ESMA:** La Autoridad Europea de Valores y Mercados (ESMA) documenta riesgos sistémicos derivados del uso generalizado de IA, incluyendo comportamiento de manada (*herding*), falta de explicabilidad en decisiones de alto impacto y arbitraje regulatorio.
- **Opacidad y explicabilidad:** Artículos que analizan la tensión entre rendimiento predictivo (modelos de caja negra) y requisitos regulatorios de explicabilidad (GDPR, MiFID II).
- **Gobernanza y marcos de validación:** Propuestas para estándares de divulgación y validación de modelos para firmas que utilizan IA en actividades reguladas.

**Artículo representativo:**
- *ESMA Warning on the Use of AI in Financial Markets.*

---

### Resumen del Corpus

| Eje Temático | Enfoque Principal | Modelos/Métodos Clave |
|---|---|---|
| Predicción de precios | Series temporales financieras | LSTM, GRU, Transformer, XGBoost |
| Trading algorítmico | Decisiones automáticas de inversión | Aprendizaje por refuerzo, visión por computadora |
| Riesgo sistémico | Crisis y contagio financiero | Gradient boosting, redes de grafo |
| NLP en finanzas | Análisis de texto no estructurado | BERT, GPT, análisis de sentimiento |
| Perspectiva regulatoria | Riesgos e implicancias institucionales | Marcos de gobernanza, ESMA |

---

## 3. Funcionalidades

### Capacidades del Sistema

| # | Funcionalidad | Descripción |
|---|---------------|-------------|
| 1 | **Indexación en doble granularidad** | Indexa todos los papers en chunks de 256 y 1024 tokens para recuperación multi-resolución |
| 2 | **Búsqueda semántica** | Recuperación por similitud vectorial usando distancia coseno sobre embeddings de 1536 dimensiones |
| 3 | **Cuatro estrategias de prompting** | Estrategias de ingeniería de prompts seleccionables por consulta |
| 4 | **Salida estructurada JSON** | Respuestas legibles por máquina con puntuación de confianza y hallazgos clave |
| 5 | **Aprendizaje en contexto (few-shot)** | Ejemplos de Q&A guían al modelo hacia el estilo de respuesta deseado |
| 6 | **Razonamiento paso a paso (CoT)** | Reduce alucinaciones en preguntas complejas mediante razonamiento explícito |
| 7 | **Conversación multi-turno** | Historial de chat para preguntas de seguimiento con contexto acumulado |
| 8 | **Citas APA automáticas** | Las respuestas incluyen referencias formateadas derivadas de los metadatos de cada paper |
| 9 | **Extracción de texto PDF** | Pipeline basado en `pdfplumber` con parseo de metadatos y fallback al nombre de archivo |
| 10 | **Chunking por tokens** | Segmentación precisa con `tiktoken` y solapamiento configurable (por defecto: 50 tokens) |
| 11 | **Base vectorial ChromaDB** | Base de datos vectorial local persistente con búsqueda HNSW por similitud coseno |
| 12 | **Procesamiento de embeddings en lotes** | Batching conservador (7 ítems/llamada) para uso seguro de la API |
| 13 | **Catálogo de papers** | Archivo JSON que registra metadatos, conteos de chunks y estado de indexación |
| 14 | **Interfaz web con Streamlit** | UI interactiva con 3 pestañas: Chat, Papers y Dashboard |
| 15 | **IDs de chunks deterministas** | IDs basados en MD5 permiten operaciones de upsert idempotentes |
| 16 | **Protección por longitud de consulta** | Trunca consultas que superan el límite de 8.192 tokens de embedding |
| 17 | **Transparencia de chunks** | Los fragmentos recuperados se muestran junto a la respuesta para su verificación |
| 18 | **Persistencia de sesión** | `session_state` de Streamlit preserva el historial completo de conversación |

### Ejemplo de Salidas del Sistema

**Estrategia 2 — Salida JSON:**
```json
{
  "answer": "Las redes LSTM son el enfoque dominante para la predicción de precios
             de acciones, presentes en 14 de los 21 papers revisados...",
  "sources": [
    "Zhang et al. (2021). Stock Price Prediction Using LSTM.",
    "Smith, J. (2020). Machine Learning in Stock Price Prediction."
  ],
  "confidence": 0.87,
  "key_findings": [
    "LSTM supera a ARIMA en series temporales financieras no estacionarias",
    "Los mecanismos de atención mejoran el rendimiento de LSTM entre 12% y 18%",
    "Los modelos de ensamble que combinan LSTM con gradient boosting muestran mayor precisión"
  ]
}
```

**Estrategia 4 — Cadena de pensamiento (CoT):**
```
Paso 1 — ¿Qué dice cada fuente?
  Fuente 1 (Zhang et al.): LSTM captura dependencias de largo plazo en secuencias de precios...
  Fuente 2 (Smith): Compara LSTM contra GRU y RNN vanilla, encontrando superioridad de LSTM...

Paso 2 — Acuerdos y desacuerdos entre fuentes:
  Todas las fuentes coinciden en que LSTM supera a los modelos econométricos clásicos.
  Existe desacuerdo sobre si las capas de atención ofrecen ganancias estadísticamente significativas...

Paso 3 — Brechas y limitaciones:
  Ningún paper aborda el rendimiento de LSTM durante eventos excepcionales. Los tamaños
  muestrales generalmente se limitan a acciones líquidas...

Paso 4 — Respuesta sintetizada:
  Basándose en la literatura revisada, LSTM sigue siendo la arquitectura dominante...
```

---

## 4. Arquitectura

### Diagrama del Sistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE DE INGESTA                              │
│                                                                         │
│  PDF (21 papers)                                                        │
│        │                                                                │
│        ▼  [pdfplumber]                                                  │
│    Texto extraído                                                       │
│        │                                                                │
│        ▼  [tiktoken — cl100k_base]                                      │
│    Chunks (256 tokens)  ─────────────────────────────────────────────┐  │
│    Chunks (1024 tokens) ─────────────────────────────────────────────┤  │
│                                                                       │  │
│                         [text-embedding-3-small]                      │  │
│                                    │                                  │  │
│                                    ▼                                  │  │
│                         Vectores (1536 dims) ──► ChromaDB ◄──────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE DE CONSULTA                             │
│                                                                         │
│  Pregunta del usuario                                                   │
│        │                                                                │
│        ▼  [text-embedding-3-small]                                      │
│  Vector de consulta ──────────────────► ChromaDB (top-5 ANN search)    │
│                                                  │                      │
│                                                  ▼                      │
│                                        5 Chunks más relevantes          │
│                                                  │                      │
│                                  ┌───────────────┘                      │
│                                  ▼                                      │
│                        Construcción del prompt                          │
│                        (1 de 4 estrategias)                             │
│                                  │                                      │
│                                  ▼                                      │
│                               GPT-4o                                    │
│                                  │                                      │
│                                  ▼                                      │
│                       Respuesta + Citas APA                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tabla de Componentes

| Componente | Tecnología | Función |
|------------|-----------|---------|
| **Extractor PDF** | `pdfplumber` | Extrae texto y metadatos de los archivos PDF |
| **Tokenizador** | `tiktoken` (`cl100k_base`) | Divide el texto en chunks con conteo exacto de tokens |
| **Modelo de embeddings** | `text-embedding-3-small` (OpenAI) | Convierte texto en vectores densos de 1536 dimensiones |
| **Base vectorial** | `ChromaDB` (HNSW coseno) | Almacena y recupera vectores por búsqueda aproximada de vecinos más cercanos |
| **Modelo de lenguaje** | `GPT-4o` | Genera respuestas fundamentadas con citas a partir del contexto recuperado |
| **Motor de prompts** | Python personalizado | Construye prompts usando una de las 4 estrategias configurables |
| **Interfaz de usuario** | `Streamlit` | UI web interactiva con 3 pestañas funcionales |
| **Catálogo** | Archivo JSON | Registra metadatos de papers, conteos de chunks y estado de indexación |

---

## 5. Instalación

### Requisitos Previos

- Python 3.8 o superior (se recomienda Python 3.10+ para Google Colab)
- Clave de API de OpenAI con acceso a `gpt-4o` y `text-embedding-3-small`

### Instalar Dependencias

Ejecutar el siguiente comando único para instalar todas las bibliotecas necesarias:

```bash
pip install openai chromadb pdfplumber tiktoken python-dotenv streamlit pandas -q
```

| Biblioteca | Función |
|------------|---------|
| `openai` | Acceso a GPT-4o (generación) y text-embedding-3-small (embeddings) |
| `chromadb` | Base de datos vectorial local persistente |
| `pdfplumber` | Extracción de texto PDF con conciencia del diseño de página |
| `tiktoken` | Tokenizador compatible con OpenAI para chunking preciso |
| `python-dotenv` | Carga claves de API desde archivo `.env` |
| `streamlit` | Framework de UI web |
| `pandas` | Manejo de datos para la pestaña de dashboard |

### Configuración de la Clave de API

**Opción A — Variable de entorno (recomendada para Colab):**
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."   # pegar la clave aquí
```

**Opción B — Archivo `.env` (recomendada para ejecución local):**
```bash
# Crear un archivo llamado .env en la raíz del proyecto:
OPENAI_API_KEY=sk-...
```

```python
from dotenv import load_dotenv
load_dotenv()
```

**Opción C — Secretos de Google Colab (más segura en Colab):**
```python
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

---

## 6. Uso

### Cómo Ejecutar el Notebook

1. Abrir el notebook en Google Colab (o localmente en Jupyter).
2. Ejecutar la **Celda 1** para instalar las dependencias.
3. Ejecutar la **Celda 2** para configurar la clave de API.
4. Ejecutar la **Celda 3** para ingestar los 21 PDFs en ChromaDB (se ejecuta una vez; toma aproximadamente 2-3 minutos).
5. Ejecutar las celdas siguientes para realizar consultas usando cualquiera de las 4 estrategias.

Para lanzar la interfaz web de Streamlit:
```bash
streamlit run app.py
```

### Consultas de Ejemplo con Salidas Esperadas

---

**Consulta 1 — Estrategia: Delimitadores | Chunk: 256 tokens**

```python
pregunta = "¿Cuáles son los principales métodos de ML para predecir precios de acciones?"
```

Salida esperada (abreviada):
```
Basándose en la literatura revisada, los enfoques de ML dominantes para la
predicción de precios de acciones incluyen:

1. **Redes LSTM** — presentes en la mayoría de los papers por su capacidad
   para capturar dependencias temporales (Zhang et al., 2021; Smith, 2020).

2. **Modelos basados en Transformer** — adoptados crecientemente para
   tareas de predicción multi-paso.

3. **Métodos de ensamble** — gradient boosting combinado con redes neuronales
   reporta la mayor precisión en los benchmarks revisados.

Fuentes:
- Zhang et al. (2021). Stock Price Prediction Using LSTM.
- Smith, J. (2020). Machine Learning in Stock Price Prediction.
```

---

**Consulta 2 — Estrategia: Salida JSON | Chunk: 256 tokens**

```python
pregunta = "¿Cuáles son las limitaciones de LSTM para la predicción financiera?"
```

Salida esperada:
```json
{
  "answer": "Las redes LSTM presentan varias limitaciones documentadas: (1) pobre
             generalización durante cambios de régimen y eventos excepcionales,
             (2) sensibilidad al ajuste de hiperparámetros, (3) alto costo
             computacional frente a modelos más simples, (4) falta de interpretabilidad.",
  "sources": [
    "Zhang et al. (2021). Stock Price Prediction Using LSTM.",
    "ML in Finance (2022)."
  ],
  "confidence": 0.82,
  "key_findings": [
    "LSTM tiene bajo rendimiento durante cambios de régimen de mercado",
    "El gradiente desvaneciente aún afecta secuencias muy largas",
    "Los mecanismos de atención mitigan parcialmente los problemas de interpretabilidad"
  ]
}
```

---

**Consulta 3 — Estrategia: Cadena de Pensamiento | Chunk: 1024 tokens**

```python
pregunta = "¿Cómo se utiliza la IA para predecir crisis financieras?"
```

Salida esperada (abreviada):
```
Paso 1 — ¿Qué dice cada fuente?
  Fuente 1: Discute indicadores macro combinados con ML para detección de riesgo
  sistémico. Fuente 2: Se centra en scoring de riesgo crediticio con gradient boosting...

Paso 2 — Acuerdos y desacuerdos:
  Las fuentes coinciden en que la IA mejora la detección de crisis frente a los
  modelos econométricos clásicos. Existe desacuerdo sobre qué características
  (sentimiento vs. macro) son más relevantes.

Paso 3 — Brechas y limitaciones:
  Validación en el mundo real limitada; la mayoría de los estudios usan datos
  históricos de crisis que pueden no generalizarse a shocks novedosos.

Paso 4 — Respuesta sintetizada:
  Los métodos de IA, especialmente los modelos de ensamble y el análisis de
  sentimiento basado en NLP, muestran mejoras significativas sobre los sistemas
  tradicionales de alerta temprana...
```

---

**Consulta 4 — Estrategia: Few-Shot | Chunk: 256 tokens | Multi-turno**

```python
# Turno 1
pregunta_1 = "¿Qué papel juega LSTM en la predicción de acciones?"

# Turno 2 (con historial del Turno 1)
pregunta_2 = "¿Cuáles son las principales alternativas a LSTM mencionadas en los papers?"
```

El segundo turno aprovecha el historial de conversación para entender "alternativas" en contexto, produciendo una respuesta comparativa sin repetir los antecedentes de LSTM.

---

**Consulta 5 — Estrategia: Delimitadores | Chunk: 1024 tokens**

```python
pregunta = "¿Qué dice la ESMA sobre los riesgos de la IA en los mercados financieros?"
```

Salida esperada:
```
La Autoridad Europea de Valores y Mercados (ESMA) advierte que los sistemas de IA
desplegados en los mercados financieros introducen riesgos sistémicos, incluyendo:

- Comportamiento de manada cuando múltiples instituciones usan modelos similares
- Opacidad y falta de explicabilidad en decisiones de alto impacto
- Propagación de sesgos en datos hacia estrategias de trading automatizadas
- Arbitraje regulatorio habilitado por las ventajas de velocidad de la IA

La ESMA recomienda requisitos de divulgación reforzados y estándares de validación
de modelos para las firmas que usan IA en actividades reguladas.

Fuente: ESMA Warning on the Use of AI (incluido en el corpus).
```

---

## 7. Detalles Técnicos

### Configuración de Chunking

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `CHUNK_SIZES` | `[256, 1024]` tokens | Doble granularidad: 256 para precisión, 1024 para contexto |
| `CHUNK_OVERLAP` | `50` tokens | Evita pérdida de sentido en los bordes de los chunks |
| `TOKENIZADOR` | `cl100k_base` | Compatible con GPT-4o; conteo exacto de tokens |
| `MAX_EMBEDDING_TOKENS` | `8.192` | Límite máximo de `text-embedding-3-small` |
| `TOP_K` | `5` | Número de chunks recuperados por consulta |
| `EMBEDDING_BATCH_SIZE` | `7` | Batching conservador para seguridad en límites de la API |

### Estrategias de Prompting

| # | Estrategia | Mecanismo | Mejor Caso de Uso |
|---|------------|-----------|-------------------|
| 1 | **Delimitadores** | Etiquetas XML `<CONTEXT>` y `<QUESTION>` separan las entradas | Preguntas generales; exploración inicial |
| 2 | **Salida JSON** | Respuesta estructurada con esquema forzado e indicador de confianza | Procesamiento programático; pipelines de evaluación |
| 3 | **Few-Shot** | 2 ejemplos de Q&A del dominio preceden a la consulta real | Requisitos de formato y tono consistentes |
| 4 | **Cadena de Pensamiento** | Razonamiento explícito en 4 pasos: fuentes → acuerdos → brechas → síntesis | Preguntas complejas multi-fuente; reducir alucinaciones |

### Modelos y Costos

| Componente | Modelo | Parámetros Clave | Costo Estimado |
|------------|--------|-----------------|----------------|
| Embeddings | `text-embedding-3-small` | 1536 dims, máx. 8192 tokens de entrada | $0,02 / 1M tokens |
| Generación | `gpt-4o` | temp=0,2 ; max_tokens=1500 | Variable por consulta |
| Ingesta completa (21 papers × 2 tamaños) | — | ~630.000 tokens totales | ~$0,013 USD |

### Estimaciones de Tokens

```
Ingesta:
  21 papers × ~15.000 tokens promedio × 2 tamaños de chunk = ~630.000 tokens de embedding
  Costo: ~$0,013 USD (text-embedding-3-small a $0,02/M tokens)

Por consulta:
  Embedding de pregunta:         ~8–50 tokens
  Entrada del prompt a GPT-4o:   ~500–2.000 tokens (contexto + historial + prompt del sistema)
  Salida máxima de GPT-4o:       1.500 tokens

Regla general:  1 token ≈ 0,75 palabras en inglés
```

---

## 8. Resultados de Evaluación

### Métricas de Calidad de Recuperación

| Métrica | Descripción | Rango Observado | Interpretación |
|---------|-------------|-----------------|----------------|
| **Distancia coseno** | Distancia entre el vector de consulta y el vector del chunk | 0,0 – 2,0 | Menor = más relevante (0 = idéntico) |
| **Similitud coseno** | 1 − Distancia coseno (derivada) | 0,0 – 1,0 | Mayor = más relevante |
| **Puntuación de confianza** | Autoevaluación del modelo (solo Estrategia 2) | 0,0 – 1,0 | Qué tan bien soporta el contexto la respuesta |

### Comparación de Estrategias de Prompting

| Estrategia | Formato de Respuesta | Verificabilidad | Longitud Promedio | Riesgo de Alucinación |
|------------|---------------------|----------------|-------------------|-----------------------|
| Delimitadores | Prosa narrativa | Media | ~300 palabras | Medio |
| Salida JSON | JSON estructurado | Alta (campo de confianza) | ~150 palabras + metadatos | Bajo–Medio |
| Few-Shot | Prosa consistente | Media | ~250 palabras | Medio |
| Cadena de Pensamiento | Multi-paso + síntesis | Alta (razonamiento explícito) | ~500 palabras | Bajo |

### Comparación de Tamaño de Chunk

| Tamaño de Chunk | Precisión de Recuperación | Cobertura de Contexto | Mejor Para |
|-----------------|--------------------------|----------------------|------------|
| **256 tokens** | Alta (hechos específicos) | Estrecha (afirmación única) | Preguntas factuales y puntuales |
| **1024 tokens** | Media (nivel de sección) | Amplia (nivel de argumento) | Preguntas conceptuales y multi-aspecto |

### Resultado Representativo

```
Consulta:  "¿Cuáles son los principales métodos de ML para predecir precios de acciones?"
Chunk:     256 tokens | Estrategia: Cadena de Pensamiento

Top-5 chunks recuperados:
  [1] distancia_coseno=0,18  →  Zhang et al. (2021) — Sección de arquitectura LSTM
  [2] distancia_coseno=0,22  →  Smith (2020) — Tabla comparativa de métodos ML
  [3] distancia_coseno=0,29  →  ML in Finance (2022) — Discusión de resultados
  [4] distancia_coseno=0,31  →  Algorithmic Trading Review — Resumen de métodos
  [5] distancia_coseno=0,35  →  Zhang et al. (2021) — Configuración experimental

Puntuación de confianza (Estrategia 2): 0,87
```

---

## 9. Limitaciones

### Limitaciones Documentadas

| # | Limitación | Impacto | Severidad |
|---|------------|---------|-----------|
| 1 | **Restricción de ventana de contexto** | Solo se pasan los top-5 chunks a GPT-4o; información relevante en chunks de menor rango puede perderse | Media |
| 2 | **Persistencia de ChromaDB en Colab** | La base vectorial almacenada en `/content` se pierde al reiniciar el runtime, a menos que se mueva a Google Drive | Alta (usuarios de Colab) |
| 3 | **Riesgo residual de alucinación** | RAG reduce significativamente las alucinaciones, pero el modelo aún puede generar afirmaciones plausibles no respaldadas por el corpus | Media |
| 4 | **Precisión de extracción PDF** | `pdfplumber` maneja mejor los diseños multi-columna que `PyPDF2`, pero los PDFs escaneados o con figuras complejas pueden producir texto degradado | Media |
| 5 | **Límites de tasa de la API** | El batch size conservador (7) reduce el throughput; corpus grandes pueden requerir manejo de retroceso exponencial | Baja–Media |
| 6 | **Fallback en extracción de metadatos** | Cuando los metadatos del PDF están ausentes o malformados, el sistema recurre al nombre del archivo, lo que puede producir información incompleta de autor/año | Baja |
| 7 | **Tokenización centrada en inglés** | `cl100k_base` está optimizado para inglés; papers con contenido significativo en otros idiomas pueden tokenizarse con menor eficiencia | Baja |
| 8 | **Base de conocimiento estática** | El corpus es fijo en el momento de la ingesta; los papers publicados recientemente requieren re-indexación | Baja (por diseño) |
| 9 | **Fecha de corte del LLM** | El conocimiento paramétrico de GPT-4o tiene una fecha de corte; para eventos muy recientes no cubiertos por los papers, el modelo puede extrapolar incorrectamente | Baja–Media |

### Mejoras Futuras

- **Recuperación híbrida:** Combinar búsqueda dispersa BM25 con búsqueda vectorial densa (Reciprocal Rank Fusion) para mejorar el recall en consultas específicas por palabras clave.
- **Capa de reranking:** Añadir un modelo cross-encoder para reordenar los top-20 candidatos antes de pasar los top-5 al LLM.
- **Integración persistente con Drive:** Montar Google Drive automáticamente y persistir ChromaDB para sobrevivir a reinicios de sesión en Colab.
- **Expansión dinámica del corpus:** Implementar un endpoint de ingesta para añadir nuevos papers sin necesidad de re-indexación completa.
- **Suite de evaluación formal:** Construir un dataset de Q&A con verdad de suelo sobre los 21 papers para calcular métricas RAGAS (fidelidad, relevancia de respuesta, precisión/recall de contexto).
- **Soporte multilingüe:** Migrar a embeddings `multilingual-e5-large` para soportar papers académicos en idiomas distintos al inglés.

---

## 10. Información del Autor

| Campo | Detalle |
|-------|---------|
| **Autor** | Julia Millen Massa Coronel |
| **Curso** | Curso de Capacitación en Prompt Engineering usando GPT-4 |
| **Fecha** | Febrero 2026 |
| **Notebook** | `colab_research_copilot.ipynb` |
| **Plataforma** | Google Colab |

### Contexto del Proyecto

Este sistema fue desarrollado como tarea final de un curso de capacitación en Prompt Engineering. Demuestra la aplicación práctica de los siguientes conceptos abordados en el curso:

- Estructura de prompts y separación de contexto mediante delimitadores
- Diseño de ejemplos para aprendizaje en contexto (*few-shot learning*)
- Prompting de cadena de pensamiento (*chain-of-thought*) para razonamiento complejo
- Prompting de salida estructurada (aplicación de esquema JSON)
- Recuperación y Generación Aumentada (RAG) como patrón arquitectónico
- Modelos de embeddings y búsqueda por similitud vectorial
- Estimación de costos y optimización del uso de la API

---

*Research Copilot — Sistema RAG sobre Papers Académicos de IA en Mercados Financieros*
*Julia Millen Massa Coronel | Febrero 2026*
