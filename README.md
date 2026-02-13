# LLM Evaluation Framework

A framework for systematic testing of Large Language Models with RAG support and comprehensive metrics.

**Author:** Ali
**Institution:** University of Duisburg-Essen
**Thesis:** Systematic Testing of Diagnostic Capabilities of Local Multimodal Language Models Using RAG

---

## Overview

This research framework evaluates LLM diagnostic capabilities on appliance fault diagnosis tasks. It supports multiple LLM providers, context modes (baseline, manual full, RAG retrieval), and comprehensive metrics for RAG quality, answer correctness, and output consistency.

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys in .env file
cp .env.example .env
# Edit .env with your API keys
```

### Run Experiment

```bash
python test_diagnosis_experiment.py
```

---

## Project Structure

```
Framwork/
├── framework/                      # Core framework
│   ├── __init__.py                 # Package exports
│   ├── core.py                     # Base classes (LLMProvider, ContextMode, ExperimentConfig, Results)
│   ├── dataset.py                  # Dataset management (Dataset, DatasetBuilder)
│   ├── prompts.py                  # Prompt templates (PromptTemplate, DiagnosticTemplate)
│   ├── metrics.py                  # Evaluation metrics (RagEvaluator, QualityEvaluator)
│   ├── runner.py                   # Experiment execution (ExperimentRunner)
│   ├── analysis.py                 # Results analysis (ResultsAnalyzer)
│   ├── llm_service.py             # LLM provider integration (LLMService)
│   ├── api_adapters.py             # API parameter handling (APIAdapter)
│   └── retrieval.py                # RAG retrieval (BM25, TF-IDF, enhanced retrieval)
│
├── test_diagnosis_experiment.py     # Main experiment script
├── requirements.txt                # Dependencies
├── .env.example                    # API key template
└── README.md                       # This file
```

---

## Supported LLM Providers

| Provider | API Backend | Notes |
|----------|--------------|-------|
| Gemini | Direct, Replicate | Native PDF support via Files API |
| GPT-5.2 | OpenAI (Direct) | Reasoning model (uses max_completion_tokens) |
| GPT-4o | OpenAI (Direct) | Standard model |
| Gemma 3 (4B/12B/27B) | Replicate, HuggingFace | Instruction-tuned variants |
| DeepSeek R1 | Replicate | Reasoning model |

---

## Context Modes

| Mode | Description |
|------|-------------|
| `BASELINE` | No context provided - tests model's intrinsic knowledge |
| `MANUAL_FULL` | Full manual content (up to 40 chunks or 40K chars) |
| `RAG_RETRIEVAL` | Top-k retrieved chunks using BM25/TF-IDF |

---

## Metrics

### RAG Metrics
- **faithfulness** - Grounding score against context
- **citation_coverage** - Fraction of sentences with citations
- **citation_correctness** - Correctness of citations
- **context_precision** - Retrieval precision
- **context_recall** - Retrieval recall

### Quality Metrics
- **answer_correctness** - Semantic similarity with ground truth
- **hallucination_rate** - Detection of ungrounded claims

### Consistency Metrics
- **output_consistency** - Variance across multiple repetitions
- **pairwise_consistency** - Consistency between repetition pairs

### Domain-Specific Metrics (Diagnosis)
- **scenario_identification_rate** - Correct scenario classification
- **property_identification_rate** - Correct property identification
- **ml_problem_type_accuracy** - ML problem classification accuracy
- **algorithm_suitability** - Algorithm recommendation correctness
- **fkgl** - Flesch-Kincaid Grade Level (readability)
- **cer** - Character Error Rate
- **wer** - Word Error Rate

---

## Configuration

### Environment Variables (.env)

```env
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
REPLICATE_API_TOKEN=...
```

### Basic Config

```python
from framework import ExperimentConfig, LLMProvider, ContextMode

config = ExperimentConfig(
    task_type="diagnosis",
    models=[LLMProvider.GEMINI, LLMProvider.GPT5],
    context_modes=[ContextMode.BASELINE, ContextMode.RAG_RETRIEVAL],
    metrics=["answer_correctness", "faithfulness", "citation_coverage"],
    n_repetitions=5,
    temperature=0.4,
    max_tokens=2048,
)
```

### Enhanced Retrieval Config (v2.0)

```python
config = ExperimentConfig(
    task_type="diagnosis",
    models=[LLMProvider.GEMINI],
    context_modes=[ContextMode.RAG_RETRIEVAL],
    metrics=["answer_correctness", "faithfulness"],
    # Enhanced Retrieval (v2.0)
    enable_query_expansion=True,      # Expand "no steam" -> "no steam valve blocked pressure"
    retrieval_window_size=1,          # Include 1 chunk before/after each hit
    retrieval_min_score=0.1,          # Filter low-quality matches
    retrieval_remove_stopwords=True,  # Remove "the", "a", "is", etc.
    retrieval_apply_stemming=True,    # "heating" -> "heat"
)
```

---

## Enhanced Retrieval (v2.0)

The framework includes 4 major retrieval enhancements:

### 1. Query Expansion
Maps symptom descriptions to related component/cause keywords:
- "no steam" -> "no steam valve blocked pressure wand boiler"
- "not heating" -> "not heating element thermostat thermal fuse"

### 2. Context Windows
Includes surrounding chunks for better context continuity:
- `window_size=1`: Include 1 chunk before and after each retrieved hit
- Maintains document order, removes duplicates from overlapping windows

### 3. Enhanced Tokenization
Improves retrieval quality through text normalization:
- Stop word removal: "The heating element is blocked" -> ["heating", "element", "blocked"]
- Basic stemming: "heating pumps valves" -> ["heat", "pump", "valve"]

### 4. Quality Metrics
Filter and debug retrieval results:
- `min_score=0.1`: Filter out low-relevance chunks
- Results tracking: See which features were enabled in output CSV

---

## Reproducibility

The framework ensures reproducibility through:

- **Deterministic seeds** for each repetition
- **SHA-256 hashes** of prompts and contexts
- **Environment export** (library versions)
- **Complete result logging** with provenance metadata

Results are exported to `results/` with full provenance metadata.

---

## API Adapter Layer

The framework separates technical API formatting from prompt content through the `APIAdapter` class:

- **Fairness Guarantee**: Semantic prompt content is IDENTICAL across all providers
- **Provider Templates**: Gemma3 chat markers, DeepSeek User/Assistant format
- **Backend Support**: Replicate, Direct APIs, HuggingFace

This ensures models perform optimally while maintaining fair comparison.

---

## License

MIT License - Free for research and education.
