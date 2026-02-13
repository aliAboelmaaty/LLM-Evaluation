"""
LLM Evaluation Framework

A general-purpose framework for systematic testing of Large Language Models
across different domains, with built-in RAG support and comprehensive metrics.

Author: Ali [Your Last Name]
Institution: University of Duisburg-Essen
Thesis: Systematic Testing of Diagnostic Capabilities of Local Multimodal
        Language Models Using RAG
"""

__version__ = "0.1.0"
__author__ = "Ali [Your Last Name]"

# Auto-load .env file if it exists
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Look for .env file in the project root (parent of framework directory)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip auto-loading
    pass

# Core classes
from .core import (
    LLMProvider,
    APIBackend,
    ContextMode,
    TestCase,
    ExperimentConfig,
    ExperimentResults,
    validate_reproducibility,
)

# Dataset management
from .dataset import (
    Dataset,
    DatasetBuilder,
)

# Prompt templates
from .prompts import (
    PromptTemplate,
    PromptLibrary,
)

# Metrics
from .metrics import (
    MetricCalculator,
    MetricsConfig,
)

# Experiment runner
from .runner import (
    ExperimentRunner,
)

# Results analysis
from .analysis import (
    ResultsAnalyzer,
)

# LLM Service
from .llm_service import (
    LLMService,
)

# API Adapters
from .api_adapters import (
    APIAdapter,
)

__all__ = [
    # Core
    "LLMProvider",
    "APIBackend",
    "ContextMode",
    "TestCase",
    "ExperimentConfig",
    "ExperimentResults",
    "validate_reproducibility",

    # Dataset
    "Dataset",
    "DatasetBuilder",

    # Prompts
    "PromptTemplate",
    "PromptLibrary",

    # Metrics
    "MetricCalculator",
    "MetricsConfig",

    # Runner
    "ExperimentRunner",

    # Analysis
    "ResultsAnalyzer",

    # LLM Service
    "LLMService",

    # API Adapters
    "APIAdapter",
]
