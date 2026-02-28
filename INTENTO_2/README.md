# Research Copilot — Asistente RAG para Papers Académicos

Sistema conversacional RAG (_Retrieval-Augmented Generation_) para consultar 20 artículos académicos sobre **crimen organizado, extorsión y gobernanza criminal en América Latina**.

Desarrollado como parte de la **Tarea 1** del curso de Prompt Engineering — QLab.

---

## Características

- **Extracción de texto** de 20 PDFs académicos con PyMuPDF
- **Chunking inteligente** en 256 o 1024 tokens con overlap configurable
- **Embeddings** con `text-embedding-3-small` de OpenAI
- **Vector store** persistente con ChromaDB (búsqueda cosine)
- **4 estrategias de prompting** comparables en tiempo real
- **Interfaz web** completa con Streamlit (chat + explorador + comparador)
- **Suite de evaluación** con 20 preguntas de prueba y métricas automáticas

---

## Estructura del proyecto

```
Tarea_1/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── papers/                    ← 20 PDFs + papers.json
├── src/
│   ├── __init__.py
│   ├── ingestion.py           ← extracción de texto (PyMuPDF)
│   ├── chunking.py            ← chunking por tokens (256 / 1024)
│   ├── embedding.py           ← embeddings text-embedding-3-small
│   ├── vectorstore.py         ← ChromaDB setup y persistencia
│   ├── retrieval.py           ← búsqueda semántica top-k
│   └── generation.py         ← orquestador de las 4 estrategias
├── prompts/
│   ├── __init__.py
│   ├── strategy1_delimiters.py
│   ├── strategy2_json.py
│   ├── strategy3_fewshot.py
│   └── strategy4_cot.py
├── app/
│   └── streamlit_app.py       ← interfaz web
├── eval/
│   ├── test_questions.json    ← 20 preguntas de evaluación
│   └── evaluation_script.py  ← evaluación automática
├── notebooks/
│   ├── 01_exploracion_papers.ipynb
│   ├── 02_pipeline_rag.ipynb
│   └── 03_comparacion_prompts.ipynb
└── demo/
    └── video_link.txt
```

---

## Instalación rápida

### 1. Clonar / descargar el proyecto

```bash
cd Tarea_1
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar API Key

```bash
cp .env.example .env
```

Edita `.env` y añade tu clave:

```
OPENAI_API_KEY=sk-...tu-clave-aqui...
```

### 5. Indexar los papers (primera vez)

```bash
python -m src.vectorstore
```

Esto extrae texto de los 20 PDFs, genera embeddings y los almacena en ChromaDB (`./chroma_db/`). Toma aproximadamente 3–5 minutos.

### 6. Lanzar la aplicación web

```bash
streamlit run app/streamlit_app.py
```

Abre [http://localhost:8501](http://localhost:8501) en tu navegador.

---

## Las 4 Estrategias de Prompting

| # | Módulo | Descripción |
|---|--------|-------------|
| **1** | `strategy1_delimiters.py` | Delimitadores XML (`<<<CONTEXTO>>>`, `<<<PREGUNTA>>>`) que separan claramente las secciones del prompt |
| **2** | `strategy2_json.py` | Respuesta en JSON estructurado con campos: `respuesta_principal`, `hallazgos_clave`, `nivel_confianza`, etc. |
| **3** | `strategy3_fewshot.py` | Dos ejemplos trabajados (Q&A) que enseñan el estilo y profundidad esperados antes de la pregunta real |
| **4** | `strategy4_cot.py` | Chain-of-Thought con 5 pasos explícitos: comprensión → inventario → análisis → gaps → síntesis |

---

## Pipeline RAG

```
                    ┌─────────────┐
         PDF ──────►│ PyMuPDF     │──► texto raw
     papers.json    └─────────────┘
                           │
                    ┌─────────────┐
                    │  Chunking   │──► chunks (256 / 1024 tokens)
                    │  + overlap  │
                    └─────────────┘
                           │
                    ┌─────────────┐
                    │  Embeddings │──► vectores 1536-dim
                    │  OAI-3-sm   │
                    └─────────────┘
                           │
                    ┌─────────────┐
                    │  ChromaDB   │──► índice persistente
                    └─────────────┘
                           ▲
    Query ─► embed ────────┘
                │
         top-k chunks
                │
    ┌───────────────────────┐
    │  Prompt Strategy 1-4  │
    └───────────────────────┘
                │
           GPT-4o
                │
           Answer + Citations
```

---

## Uso como módulo Python

```python
# Recuperar chunks relevantes
from src.retrieval import search

chunks = search(
    "¿Qué es la gobernanza criminal?",
    top_k=5,
    strategy="small",   # chunks de 256 tokens
)
for chunk in chunks:
    print(f"[{chunk.score:.3f}] {chunk.title}")

# Generar respuesta con una estrategia específica
from src.generation import rag_query

result = rag_query(
    question="¿Cuáles son las estrategias de resistencia a la extorsión?",
    strategy="cot",     # chain-of-thought
    top_k=5,
)
print(result["answer"])
```

---

## Evaluación

```bash
# Evaluar con las 4 estrategias sobre las 20 preguntas de prueba
python eval/evaluation_script.py

# Solo primeras 5 preguntas con una estrategia
python eval/evaluation_script.py --questions 5 --strategy cot

# Guardar resultados en archivo personalizado
python eval/evaluation_script.py --output mi_evaluacion.json
```

**Métricas calculadas:**
- `precision@k` — fracción de papers recuperados que eran esperados
- `recall@k` — fracción de papers esperados que fueron recuperados
- `keyword_overlap` — fracción de conceptos clave presentes en la respuesta
- `total_tokens` — costo computacional
- `generation_time_s` — tiempo de respuesta

---

## Notebooks

Ejecutar en orden:

1. `01_exploracion_papers.ipynb` — Análisis exploratorio: distribución temporal, temas frecuentes, extracción de texto
2. `02_pipeline_rag.ipynb` — Demostración del pipeline: chunking, indexación, retrieval con ejemplos
3. `03_comparacion_prompts.ipynb` — Comparación de las 4 estrategias en una pregunta real

```bash
jupyter notebook notebooks/
```

---

## Papers Indexados

| ID | Título | Autor(es) | Año |
|----|--------|-----------|-----|
| paper_001 | Crime and plural orders in Rio de Janeiro | Arias & Barnes | 2017 |
| paper_002 | Resisting the extortion racket | Battisti et al. | 2018 |
| paper_003 | Against the Odds: Small Business Strategies (El Salvador) | Bull et al. | 2024 |
| paper_004 | Community capacity and the reporting of extortion | Dulin | 2023 |
| paper_005 | Are Repeatedly Extorted Businesses Different? | Estévez-Soto et al. | 2021 |
| paper_006 | Gobernanza Criminal y la Crisis de los Estados Latinoamericanos | Feldmann & Luna | 2022 |
| paper_007 | ¿Extorsión, un laberinto sin salida? (Trujillo) | Felipe & Polo | 2024 |
| paper_008 | Legalized Extortion: Gamarra Market, Lima | Ginocchio | 2022 |
| paper_009 | Fear of crime in Peru | Hernández et al. | 2020 |
| paper_010 | Estado de Emergencia en el Callao | Hernández et al. | 2024 |
| paper_011 | Conceptualizing Criminal Governance | Lessing | 2021 |
| paper_012 | Resisting Protection: Rackets, Resistance | Moncada | 2019 |
| paper_013 | Medición del delito en el Perú | Mujica | 2013 |
| paper_014 | Inseguridad y mecanismos barriales en Perú | Rojas & Castillo | 2016 |
| paper_015 | Model (my) neighbourhood (Portugal/Lithuania) | Saraiva et al. | 2016 |
| paper_016 | Drugs, Violence, State-Sponsored Protection Rackets | Snyder & Durán Martínez | 2009 |
| paper_017 | Why Did Drug Cartels Go to War in Mexico? | Trejo & Ley | 2018 |
| paper_018 | Factores asociados a la extorsión (La Libertad, Perú) | Yupari-Azabache et al. | 2020 |
| paper_019 | Inseguridad, estado y desigualdad en Perú y AL | Zárate et al. | 2013 |
| paper_020 | Defensa comunitaria y culturas del terror (Guerrero) | Delgado | 2022 |

---

## Tecnologías

| Componente | Tecnología |
|-----------|-----------|
| Extracción PDF | PyMuPDF (`fitz`) |
| Tokenización | tiktoken (`cl100k_base`) |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dim) |
| Vector store | ChromaDB (cosine similarity, HNSW) |
| LLM | OpenAI `gpt-4o` |
| Web interface | Streamlit |
| Notebooks | Jupyter |

---

## Notas

- La primera indexación requiere llamadas a la API de OpenAI (~20 papers × ~200 chunks = ~4,000 embeddings).
- El índice se persiste en `./chroma_db/` y no necesita regenerarse en ejecuciones posteriores.
- Para re-indexar desde cero: `python -m src.vectorstore --force-rebuild`
- El modelo GPT-4o puede reemplazarse con `gpt-4o-mini` en `.env` para reducir costos en evaluaciones extensas.
