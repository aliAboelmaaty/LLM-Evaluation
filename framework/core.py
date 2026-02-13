"""
Core data structures and base classes for the LLM Evaluation Framework.

This module defines the fundamental abstractions that make the framework
domain-agnostic and reusable across different evaluation tasks.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import pandas as pd


# ================= LLM Provider Enum =================

class LLMProvider(Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"
    GEMMA3_4B = "gemma3-4b"
    GEMMA3_12B = "gemma3-12b"
    GEMMA3_27B = "gemma3-27b"
    GEMMA3 = "gemma3-27b"  # Alias for backward compatibility
    DEEPSEEK = "deepseek"
    GPT5 = "gpt5"


# ================= API Backend Enum =================

class APIBackend(Enum):
    """
    API backend sources for LLM providers.

    This allows the same model to be accessed through different APIs:
    - DIRECT: Provider's official API (e.g., Gemini API, OpenAI API)
    - REPLICATE: Replicate.com API
    - HUGGINGFACE: Hugging Face Inference API

    Examples:
    - (GEMINI, DIRECT) - Use Gemini's official API
    - (GEMINI, REPLICATE) - Use Gemini via Replicate
    - (GEMMA3, REPLICATE) - Use Gemma3 via Replicate
    - (GEMMA3, HUGGINGFACE) - Use Gemma3 via Hugging Face
    """
    DIRECT = "direct"
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"


# ================= Context Mode Enum =================

class ContextMode(Enum):
    """Context modes for experiments"""
    BASELINE = "baseline"  # No manual
    MANUAL_FULL = "manual_full"  # Full manual text (no retrieval)
    RAG_RETRIEVAL = "rag_retrieval"  # Retrieve top-k chunks from manual


# ================= Test Case Structure =================

@dataclass
class TestCase:
    """
    Generic test case structure that works for ANY domain.

    Examples:
    - Diagnosis: fault_description → expected_diagnosis
    - Repurposing: component_name → expected_scenarios
    - ML Recommendation: problem_description → expected_algorithms

    Attributes:
        case_id: Unique identifier for this test case
        input_data: Flexible dict containing any input fields
        ground_truth: Flexible dict containing expected outputs
        metadata: Additional information (complexity, category, etc.)
        context_document: Optional path to PDF/document for RAG
    """
    case_id: str
    input_data: Dict[str, Any]
    ground_truth: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_document: Optional[str] = None

    def __post_init__(self):
        """Validate test case after initialization"""
        if not self.case_id:
            raise ValueError("case_id cannot be empty")
        if not isinstance(self.input_data, dict):
            raise TypeError("input_data must be a dictionary")
        if not isinstance(self.ground_truth, dict):
            raise TypeError("ground_truth must be a dictionary")


# ================= Experiment Configuration =================

@dataclass
class ExperimentConfig:
    """
    Configuration for any LLM evaluation experiment.

    This class defines all parameters needed to run a reproducible experiment
    following the methodology from published research papers.

    Attributes:
        task_type: Type of task ("diagnosis", "repurposing", "ml_recommendation", etc.)
        models: List of LLM providers to test
        metrics: List of metric names to calculate
        n_repetitions: Number of times to run each test (for consistency analysis)
        context_modes: List of context modes to test (default: [BASELINE, RAG_RETRIEVAL])
        temperature: LLM temperature parameter (default: 0.2)
        max_tokens: Maximum output tokens (default: 2048)
        top_p: Nucleus sampling parameter (default: 1.0, typical: 0.9)
        random_seed: Random seed for reproducibility
        custom_prompt: Optional custom PromptTemplate to use instead of default
        prompt_variant: Optional variant name (e.g., "v1", "v2", "concise") to use specific prompt version
        top_k_retrieval: Number of chunks to retrieve in RAG_RETRIEVAL mode
        hallucination_faithfulness_thresh: Faithfulness threshold for hallucination detection (default: 0.25)
        hallucination_citation_thresh: Citation correctness threshold for hallucination detection (default: 0.30)
        chunk_size: Size of text chunks in characters (default: 1000)
        max_context_chars: Maximum context characters to use (default: 40000)
        manual_full_max_chunks: Maximum chunks for MANUAL_FULL mode (default: 40)
        context_budget_overrides: Per-provider budget overrides (dict[provider_value, dict[budget_params]])
        include_baseline: DEPRECATED - use context_modes instead
        enforce_output_schema: Validate output structure (default: False). WARNING: Only enable with compatible prompts!

        # Enhanced Retrieval Parameters (v2.0) - ENABLED BY DEFAULT
        enable_query_expansion: Expand symptom queries with related keywords (default: True, recommended)
        retrieval_window_size: Include N chunks before/after each retrieved chunk (default: 1, recommended)
        retrieval_min_score: Minimum score threshold for retrieved chunks (default: 0.0 = no filtering)
        retrieval_remove_stopwords: Remove English stop words during tokenization (default: True)
        retrieval_apply_stemming: Apply basic stemming during tokenization (default: True)
        custom_expansion_dict: Custom symptom-to-keyword dictionary for query expansion (default: None = use built-in)
    """
    task_type: str
    models: List[LLMProvider]
    metrics: List[str]
    n_repetitions: int = 5
    context_modes: Optional[List['ContextMode']] = None
    temperature: float = 0.2
    max_tokens: int = 2048
    baseline_max_tokens: Optional[int] = None
    top_p: float = 1.0  # Nucleus sampling parameter (1.0 = disabled, 0.9 = typical)
    random_seed: int = 42
    custom_prompt: Optional[Any] = None  # PromptTemplate (avoiding circular import)
    prompt_variant: Optional[str] = None
    top_k_retrieval: int = 6
    hallucination_faithfulness_thresh: float = 0.5
    hallucination_citation_thresh: float = 0.6
    chunk_size: int = 1000
    max_context_chars: int = 40000
    manual_full_max_chunks: int = 40
    context_budget_overrides: Optional[Dict[str, Dict[str, int]]] = None
    include_baseline: Optional[bool] = None  # Backward compatibility
    enforce_output_schema: bool = False  # FAIRNESS: Validate output structure (disabled by default)

    # Enhanced Retrieval Parameters (v2.0) - ENABLED BY DEFAULT
    # Recommended settings for +20-30% improvement over v1.0
    enable_query_expansion: bool = True  # Recommended: maps symptoms to keywords
    retrieval_window_size: int = 1  # Recommended: includes surrounding context
    retrieval_min_score: float = 0.0  # 0.0 = no filtering (v1.0 behavior)
    retrieval_remove_stopwords: bool = True  # Enhanced tokenization (safe default)
    retrieval_apply_stemming: bool = True  # Enhanced tokenization (safe default)
    custom_expansion_dict: Optional[Dict[str, List[str]]] = None  # Use built-in if None

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.task_type:
            raise ValueError("task_type cannot be empty")
        if not self.models:
            raise ValueError("models list cannot be empty")
        if not self.metrics:
            raise ValueError("metrics list cannot be empty")
        if self.n_repetitions < 1:
            raise ValueError("n_repetitions must be >= 1")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

        # Validate enhanced retrieval parameters
        if self.retrieval_window_size < 0:
            raise ValueError("retrieval_window_size must be >= 0")
        if not (0.0 <= self.retrieval_min_score <= 1.0):
            raise ValueError("retrieval_min_score must be between 0.0 and 1.0")

        # Backward compatibility: map include_baseline to context_modes
        if self.context_modes is None:
            if self.include_baseline is not None:
                import warnings
                warnings.warn(
                    "DEPRECATED: 'include_baseline' parameter is deprecated. "
                    "Use 'context_modes' instead with explicit [ContextMode.BASELINE, ContextMode.RAG_RETRIEVAL].",
                    DeprecationWarning,
                    stacklevel=2
                )
                # Legacy behavior
                if self.include_baseline:
                    self.context_modes = [ContextMode.BASELINE, ContextMode.RAG_RETRIEVAL]
                else:
                    self.context_modes = [ContextMode.RAG_RETRIEVAL]
            else:
                # Default: compare baseline vs RAG
                self.context_modes = [ContextMode.BASELINE, ContextMode.RAG_RETRIEVAL]

    def validate_retrieval_config(self, context_modes: List['ContextMode']) -> None:
        """
        Warn if suboptimal retrieval configuration detected.

        This method checks if enhanced retrieval features are disabled when using
        RAG modes, which can significantly impact performance (-20-30%).

        Args:
            context_modes: List of context modes being used in the experiment
        """
        import warnings

        uses_rag = any(mode in [ContextMode.RAG_RETRIEVAL, ContextMode.MANUAL_FULL]
                       for mode in context_modes)

        if uses_rag and not self.enable_query_expansion and self.retrieval_window_size == 0:
            warnings.warn(
                "⚠️  Enhanced retrieval is disabled. For +20-30% improvement, set "
                "enable_query_expansion=True and retrieval_window_size=1",
                UserWarning,
                stacklevel=3
            )

    @classmethod
    def recommended_for_diagnosis(cls, models: List['LLMProvider']) -> 'ExperimentConfig':
        """
        Create config with recommended settings for diagnostic tasks.

        This factory method creates an ExperimentConfig pre-configured with
        optimal settings for appliance fault diagnosis experiments:
        - Enhanced retrieval enabled (query expansion + context windows)
        - Diagnosis-specific metrics (answer_correctness, faithfulness, citation_coverage)
        - Balanced parameters (temperature=0.2, top_k=6)

        Args:
            models: List of LLM providers to test

        Returns:
            ExperimentConfig with recommended diagnosis settings

        Example:
            >>> config = ExperimentConfig.recommended_for_diagnosis([LLMProvider.GEMINI, LLMProvider.GPT5])
            >>> runner = ExperimentRunner(config, llm_service)
            >>> results = runner.run(dataset)
        """
        from .metrics import MetricsConfig
        return cls(
            task_type="diagnosis",
            models=models,
            metrics=MetricsConfig.DIAGNOSIS_METRICS,
            enable_query_expansion=True,
            retrieval_window_size=1,
        )

    def get_context_budgets(self, provider: LLMProvider) -> Dict[str, int]:
        """
        Resolve context budgets for a specific provider.

        Returns global defaults unless overridden in context_budget_overrides.

        Args:
            provider: LLM provider to get budgets for

        Returns:
            Dictionary with keys: chunk_size, max_context_chars, manual_full_max_chunks
        """
        # Start with global defaults
        budgets = {
            "chunk_size": self.chunk_size,
            "max_context_chars": self.max_context_chars,
            "manual_full_max_chunks": self.manual_full_max_chunks,
        }

        # Apply provider-specific overrides if they exist
        if self.context_budget_overrides and provider.value in self.context_budget_overrides:
            overrides = self.context_budget_overrides[provider.value]
            budgets.update(overrides)

        return budgets

    def get_retrieval_kwargs(self) -> Dict[str, Any]:
        """
        Get keyword arguments for retrieve_chunks_enhanced().

        This method centralizes all enhanced retrieval parameters into a single
        dictionary that can be passed directly to retrieve_chunks_enhanced().

        Returns:
            Dictionary with all retrieval enhancement parameters:
            - enable_expansion: bool
            - expansion_dict: Optional[Dict]
            - window_size: int
            - min_score: float
            - remove_stopwords: bool
            - apply_stemming: bool

        Example:
            >>> config = ExperimentConfig(task_type="diagnosis", models=[GPT5], metrics=["answer_correctness"])
            >>> kwargs = config.get_retrieval_kwargs()
            >>> chunks = retrieve_chunks_enhanced(query, all_chunks, top_k=6, **kwargs)
        """
        return {
            "enable_expansion": self.enable_query_expansion,
            "expansion_dict": self.custom_expansion_dict,
            "window_size": self.retrieval_window_size,
            "min_score": self.retrieval_min_score,
            "remove_stopwords": self.retrieval_remove_stopwords,
            "apply_stemming": self.retrieval_apply_stemming,
        }

    def is_enhanced_retrieval_enabled(self) -> bool:
        """
        Check if any enhanced retrieval features are enabled.

        This is useful for logging and result tracking to know if v2.0
        features were active during an experiment.

        Returns:
            True if any enhanced feature is enabled (query expansion or context windows)
        """
        return self.enable_query_expansion or self.retrieval_window_size > 0


# ================= Experiment Results =================

def _sanitize_csv_value(value: Any) -> Any:
    """
    Sanitize CSV value to prevent Excel formula injection.

    Handles both single-line and multiline strings. For multiline content,
    checks each line and prefixes dangerous lines with a tab character.

    Dangerous patterns:
    - Lines starting with =, +, -, @ (formula injection)
    - This includes the first line and any subsequent lines in multiline text

    Args:
        value: Cell value to sanitize

    Returns:
        Sanitized value (dangerous lines prefixed with tab)
    """
    if not isinstance(value, str) or len(value) == 0:
        return value

    # Check if multiline
    if '\n' in value:
        # Sanitize each line independently
        lines = value.split('\n')
        sanitized_lines = []
        for line in lines:
            if len(line) > 0 and line[0] in ('=', '+', '-', '@'):
                sanitized_lines.append('\t' + line)
            else:
                sanitized_lines.append(line)
        return '\n'.join(sanitized_lines)
    else:
        # Single line: check first character
        if value[0] in ('=', '+', '-', '@'):
            return '\t' + value
        return value


class ExperimentResults:
    """
    Container for experiment results with analysis methods.

    This class stores raw results and provides methods to generate
    publication-ready tables following the format from research papers.

    Methods generate tables like:
    - Table 1: Overall performance (per model, averaged across all test cases)
    - Table 2: Performance by complexity cluster
    - Table 3: Output consistency analysis
    """

    def __init__(self, raw_results: List[Dict[str, Any]], metrics: Optional[List[str]] = None):
        """
        Initialize results container.

        Args:
            raw_results: List of dictionaries, each containing metrics for one test run
            metrics: REQUIRED list of metric names from ExperimentConfig.metrics (allowlist for aggregation)
        """
        self.raw_results = raw_results
        self.df = pd.DataFrame(raw_results) if raw_results else pd.DataFrame()

        # CRITICAL: Store metrics as ALLOWLIST for aggregation
        # Only columns in this list will be averaged in performance tables
        # This prevents accidental averaging of debug/operational fields
        self.metrics = metrics if metrics is not None else []

        # Analysis tables (generated lazily)
        self._overall_table: Optional[pd.DataFrame] = None
        self._complexity_table: Optional[pd.DataFrame] = None
        self._consistency_table: Optional[pd.DataFrame] = None

    @property
    def overall_table(self) -> pd.DataFrame:
        """
        Overall performance table (like Table 1 in both papers).

        Returns DataFrame with columns:
        - model: Model name
        - use_rag: Whether RAG was used
        - [metric columns]: Average scores for each metric
        """
        if self._overall_table is None:
            self._overall_table = self._create_overall_table()
        return self._overall_table

    @overall_table.setter
    def overall_table(self, value: pd.DataFrame):
        self._overall_table = value

    @property
    def complexity_table(self) -> pd.DataFrame:
        """
        Performance by complexity table (like Dörnbach's Table 2).

        Returns DataFrame with performance broken down by complexity cluster.
        """
        if self._complexity_table is None:
            self._complexity_table = self._create_complexity_table()
        return self._complexity_table

    @complexity_table.setter
    def complexity_table(self, value: pd.DataFrame):
        self._complexity_table = value

    @property
    def consistency_table(self) -> pd.DataFrame:
        """
        Output consistency analysis (like Sonntag's OCR metric).

        Returns DataFrame with consistency scores per model.
        """
        if self._consistency_table is None:
            self._consistency_table = self._create_consistency_table()
        return self._consistency_table

    @consistency_table.setter
    def consistency_table(self, value: pd.DataFrame):
        self._consistency_table = value

    def _create_overall_table(self) -> pd.DataFrame:
        """Create overall performance table"""
        if self.df.empty:
            return pd.DataFrame()

        # CHANGE #4: Filter out both runtime errors AND sanity check failures
        had_error = self.df.get("had_error", False).fillna(False).astype(bool)
        sanity_failed = self.df.get("sanity_check_failed", False).fillna(False).astype(bool)
        valid_df = self.df[~(had_error | sanity_failed)]

        if valid_df.empty:
            return pd.DataFrame()

        # Group by model and context mode
        group_cols = ["model"]
        if "context_mode" in valid_df.columns:
            group_cols.append("context_mode")
        elif "use_rag" in valid_df.columns:
            # Backward compatibility
            group_cols.append("use_rag")

        # STRICT ALLOWLIST: Only average columns explicitly requested in config.metrics
        # This prevents accidental averaging of debug/operational fields
        # If someone adds "llm_call_duration" or "prompt_tokens_count" without adding to metrics,
        # it will NOT be averaged (safe by default)

        # Audit trail fields (never metrics, always excluded)
        AUDIT_TRAIL_FIELDS = {"run_id"}

        metric_cols = [
            m for m in self.metrics
            if m in valid_df.columns and m not in AUDIT_TRAIL_FIELDS
        ]

        # Validate: warn if requested metrics are missing
        missing_metrics = [m for m in self.metrics if m not in valid_df.columns]
        if missing_metrics:
            import warnings
            warnings.warn(
                f"Requested metrics not found in results: {missing_metrics}. "
                f"These will be skipped in aggregation.",
                UserWarning
            )

        # PER-REPETITION FIX: Deduplicate by case_id first
        # Since metrics are duplicated across repetitions, we need to ensure
        # each test case contributes equally (not weighted by # of repetitions)
        if "case_id" in valid_df.columns and "repetition_index" in valid_df.columns:
            # New per-repetition format: take first repetition of each case
            # (all repetitions have identical aggregated metrics)
            dedup_df = valid_df[valid_df["repetition_index"] == 0].copy()
        else:
            # Old aggregated format: use as-is
            dedup_df = valid_df

        # THESIS REQUIREMENT: Add denominators (n_total, n_valid, n_excluded, excluded_rate)
        # This shows reviewers how many TEST CASES (not repetitions) contributed to each mean

        # Calculate both mean AND std for research analysis
        overall_means = dedup_df.groupby(group_cols)[metric_cols].mean()
        overall_stds = dedup_df.groupby(group_cols)[metric_cols].std()

        # Rename std columns to avoid confusion (add _std suffix)
        overall_stds = overall_stds.rename(columns={col: f"{col}_std" for col in metric_cols})

        # Merge mean and std
        overall_stats = overall_means.join(overall_stds)

        # Compute denominators per group (counting test cases, not repetitions)
        n_valid = dedup_df.groupby(group_cols).size()

        # For n_total, also deduplicate (count unique test cases)
        if "case_id" in self.df.columns and "repetition_index" in self.df.columns:
            total_dedup_df = self.df[self.df["repetition_index"] == 0].copy()
        else:
            total_dedup_df = self.df
        n_total = total_dedup_df.groupby(group_cols).size()

        n_excluded = n_total - n_valid
        excluded_rate = (n_excluded / n_total).fillna(0.0)

        # Add denominator columns to the stats table
        overall_stats['n_total'] = n_total
        overall_stats['n_valid'] = n_valid
        overall_stats['n_excluded'] = n_excluded
        overall_stats['excluded_rate'] = excluded_rate

        return overall_stats

    def _create_complexity_table(self) -> pd.DataFrame:
        """Create performance by complexity table"""
        if self.df.empty or "complexity" not in self.df.columns:
            return pd.DataFrame()

        # CHANGE #4: Filter out both runtime errors AND sanity check failures
        had_error = self.df.get("had_error", False).fillna(False).astype(bool)
        sanity_failed = self.df.get("sanity_check_failed", False).fillna(False).astype(bool)
        valid_df = self.df[~(had_error | sanity_failed)]

        if valid_df.empty:
            return pd.DataFrame()

        # Group by model, context mode, and complexity
        group_cols = ["model"]
        if "context_mode" in valid_df.columns:
            group_cols.append("context_mode")
        elif "use_rag" in valid_df.columns:
            # Backward compatibility
            group_cols.append("use_rag")
        group_cols.append("complexity")

        # STRICT ALLOWLIST: Only average metrics from config.metrics
        metric_cols = [m for m in self.metrics if m in valid_df.columns]

        # PER-REPETITION FIX: Deduplicate by case_id first
        if "case_id" in valid_df.columns and "repetition_index" in valid_df.columns:
            dedup_df = valid_df[valid_df["repetition_index"] == 0].copy()
        else:
            dedup_df = valid_df

        # THESIS REQUIREMENT: Add denominators (n_total, n_valid, n_excluded, excluded_rate)
        # Calculate both mean AND std for research analysis
        complexity_means = dedup_df.groupby(group_cols)[metric_cols].mean()
        complexity_stds = dedup_df.groupby(group_cols)[metric_cols].std()

        # Rename std columns
        complexity_stds = complexity_stds.rename(columns={col: f"{col}_std" for col in metric_cols})

        # Merge mean and std
        complexity_stats = complexity_means.join(complexity_stds)

        # Compute denominators per group (counting test cases, not repetitions)
        n_valid = dedup_df.groupby(group_cols).size()

        if "case_id" in self.df.columns and "repetition_index" in self.df.columns:
            total_dedup_df = self.df[self.df["repetition_index"] == 0].copy()
        else:
            total_dedup_df = self.df
        n_total = total_dedup_df.groupby(group_cols).size()

        n_excluded = n_total - n_valid
        excluded_rate = (n_excluded / n_total).fillna(0.0)

        # Add denominator columns
        complexity_stats['n_total'] = n_total
        complexity_stats['n_valid'] = n_valid
        complexity_stats['n_excluded'] = n_excluded
        complexity_stats['excluded_rate'] = excluded_rate

        return complexity_stats

    def _create_consistency_table(self) -> pd.DataFrame:
        """Create consistency analysis table"""
        if self.df.empty or "output_consistency" not in self.df.columns:
            return pd.DataFrame()

        # CHANGE #4: Filter out both runtime errors AND sanity check failures
        had_error = self.df.get("had_error", False).fillna(False).astype(bool)
        sanity_failed = self.df.get("sanity_check_failed", False).fillna(False).astype(bool)
        valid_df = self.df[~(had_error | sanity_failed)]

        if valid_df.empty:
            return pd.DataFrame()

        group_cols = ["model"]
        if "context_mode" in valid_df.columns:
            group_cols.append("context_mode")
        elif "use_rag" in valid_df.columns:
            # Backward compatibility
            group_cols.append("use_rag")

        # PER-REPETITION FIX: Deduplicate by case_id first
        if "case_id" in valid_df.columns and "repetition_index" in valid_df.columns:
            dedup_df = valid_df[valid_df["repetition_index"] == 0].copy()
        else:
            dedup_df = valid_df

        consistency = dedup_df.groupby(group_cols)["output_consistency"].mean()
        return consistency.to_frame()

    def export_manifest(
        self,
        output_dir: str,
        config: 'ExperimentConfig',
        dataset_path: Optional[str] = None,
        run_id: Optional[str] = None
    ):
        """
        Export run manifest for full reproducibility.

        Creates run_manifest.json with:
        - Run ID (for joining to raw_results.csv)
        - Full experiment configuration
        - Environment information (Python, packages, git)
        - Model identifiers
        - Dataset information

        Args:
            output_dir: Directory to save manifest
            config: Experiment configuration
            dataset_path: Optional path to dataset file
            run_id: Optional run ID (extracted from results if not provided)
        """
        import os
        import sys
        import subprocess
        import hashlib
        import json
        from datetime import datetime

        os.makedirs(output_dir, exist_ok=True)

        # Extract run_id from dataframe if not provided
        if run_id is None and not self.df.empty and "run_id" in self.df.columns:
            # All rows should have same run_id, take first
            run_id = self.df["run_id"].iloc[0]

        manifest = {
            "run_id": run_id,  # CRITICAL: Allows joining manifest to raw_results
            "timestamp": datetime.now().isoformat(),
            "experiment_config": {
                "task_type": config.task_type,
                "models": [m.value for m in config.models],
                "context_modes": [cm.value for cm in config.context_modes],
                "metrics": config.metrics,
                "n_repetitions": config.n_repetitions,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "baseline_max_tokens": config.baseline_max_tokens,
                "random_seed": config.random_seed,
                "top_k_retrieval": config.top_k_retrieval,
                "chunk_size": config.chunk_size,
                "max_context_chars": config.max_context_chars,
                "manual_full_max_chunks": config.manual_full_max_chunks,
                "hallucination_thresholds": {
                    "faithfulness": config.hallucination_faithfulness_thresh,
                    "citation": config.hallucination_citation_thresh
                },
                # Enhanced Retrieval Parameters (v2.0)
                "enhanced_retrieval": {
                    "enable_query_expansion": config.enable_query_expansion,
                    "retrieval_window_size": config.retrieval_window_size,
                    "retrieval_min_score": config.retrieval_min_score,
                    "retrieval_remove_stopwords": config.retrieval_remove_stopwords,
                    "retrieval_apply_stemming": config.retrieval_apply_stemming,
                    "has_custom_expansion_dict": config.custom_expansion_dict is not None,
                }
            },
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
            },
            "dataset": {
                "n_test_cases": len(self.df) if not self.df.empty else 0,
            }
        }

        # Add dataset hash if path provided
        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                dataset_bytes = f.read()
                manifest["dataset"]["path"] = dataset_path
                manifest["dataset"]["hash"] = hashlib.sha256(dataset_bytes).hexdigest()

        # Try to get key package versions
        try:
            import pkg_resources
            key_packages = [
                'sentence-transformers',
                'sklearn',
                'scikit-learn',
                'replicate',
                'google-generativeai',
                'pypdf',
                'PyPDF2',
                'pandas',
                'tqdm'
            ]
            versions = {}
            for pkg in key_packages:
                try:
                    ver = pkg_resources.get_distribution(pkg).version
                    versions[pkg] = ver
                except Exception:
                    pass
            if versions:
                manifest["environment"]["package_versions"] = versions
        except Exception:
            pass

        # Write manifest
        manifest_path = os.path.join(output_dir, "run_manifest.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"Manifest exported to: {manifest_path}")

    def export_environment_info(self, output_dir: str):
        """
        Export detailed environment information for cryptographic reproducibility.

        CRITICAL FOR REPRODUCIBILITY:
        Different library versions (especially rank-bm25, scikit-learn) can produce
        different retrieval results, breaking SHA-256 hash matching across machines.
        This method exports exact version information so others can reproduce results.

        Creates environment.json with:
        - Python version (full and short)
        - Platform information (OS, architecture)
        - Exact versions of critical libraries:
          * rank_bm25 (affects BM25 retrieval ranking)
          * scikit-learn (affects semantic similarity metrics)
          * sentence_transformers (affects embeddings)
          * numpy (affects numerical precision)
          * pandas (affects data processing)

        WHY THIS MATTERS:
        - Machine A with rank-bm25==0.2.1 might retrieve chunks [2,5,7]
        - Machine B with rank-bm25==0.2.3 might retrieve chunks [2,5,8]
        - Different chunks → different context → different SHA-256 hash
        - Result: Cryptographic provenance fails

        Args:
            output_dir: Directory to save environment.json

        Example output (environment.json):
        {
          "python_version": "3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0]",
          "python_version_short": "3.10.12",
          "platform": "Linux-5.15.0-91-generic-x86_64-with-glibc2.35",
          "architecture": "x86_64",
          "rank_bm25_version": "0.2.2",
          "sklearn_version": "1.3.0",
          "sentence_transformers_version": "2.2.2",
          "numpy_version": "1.24.3",
          "pandas_version": "2.0.3"
        }
        """
        import os
        import sys
        import platform
        import json

        os.makedirs(output_dir, exist_ok=True)

        # Collect Python version information
        python_version_full = sys.version
        python_version_short = platform.python_version()

        # Collect platform information
        platform_str = platform.platform()
        architecture = platform.machine()

        # Build environment info dict
        env_info = {
            "python_version": python_version_full,
            "python_version_short": python_version_short,
            "platform": platform_str,
            "architecture": architecture,
        }

        # CRITICAL LIBRARIES: These affect retrieval and must be tracked
        # Different versions can produce different results, breaking reproducibility
        critical_libraries = {
            "rank_bm25": "rank_bm25_version",          # BM25 retrieval (version-sensitive)
            "sklearn": "sklearn_version",              # Semantic similarity metrics
            "sentence_transformers": "sentence_transformers_version",  # Embeddings
            "numpy": "numpy_version",                  # Numerical operations
            "pandas": "pandas_version",                # Data processing
        }

        # Try to get version for each critical library
        for import_name, field_name in critical_libraries.items():
            try:
                # Try importing the module
                if import_name == "sklearn":
                    import sklearn
                    env_info[field_name] = sklearn.__version__
                elif import_name == "rank_bm25":
                    import rank_bm25
                    # rank_bm25 might not have __version__, try pkg_resources
                    try:
                        env_info[field_name] = rank_bm25.__version__
                    except AttributeError:
                        try:
                            import pkg_resources
                            env_info[field_name] = pkg_resources.get_distribution("rank-bm25").version
                        except Exception:
                            env_info[field_name] = "unknown"
                elif import_name == "sentence_transformers":
                    import sentence_transformers
                    env_info[field_name] = sentence_transformers.__version__
                elif import_name == "numpy":
                    import numpy
                    env_info[field_name] = numpy.__version__
                elif import_name == "pandas":
                    import pandas
                    env_info[field_name] = pandas.__version__
            except ImportError:
                # Library not installed - mark as not available
                env_info[field_name] = "not_installed"
            except Exception as e:
                # Other error - mark as unknown
                env_info[field_name] = f"error: {str(e)}"

        # Write environment info to JSON
        env_path = os.path.join(output_dir, "environment.json")
        with open(env_path, 'w', encoding='utf-8') as f:
            json.dump(env_info, f, indent=2, ensure_ascii=False)

        print(f"Environment info exported to: {env_path}")
        print("REPRODUCIBILITY: Compare this file with published results to ensure exact environment match.")

    def export_all(self, output_dir: str, config: Optional['ExperimentConfig'] = None, dataset_path: Optional[str] = None):
        """
        Export all tables and manifest to CSV/JSON files.

        CSV files use German/European format for direct opening in Excel:
        - Semicolon (;) as delimiter
        - Comma (,) as decimal separator
        - UTF-8 with BOM encoding

        Args:
            output_dir: Directory to save CSV files
            config: Optional experiment configuration (for manifest)
            dataset_path: Optional dataset path (for manifest)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Overall performance
        if not self.overall_table.empty:
            df_overall = self.overall_table.reset_index()
            # Sanitize string columns for CSV injection prevention
            for col in df_overall.columns:
                if df_overall[col].dtype == 'object':
                    df_overall[col] = df_overall[col].apply(_sanitize_csv_value)
            df_overall.to_csv(
                os.path.join(output_dir, "table1_overall_performance.csv"),
                index=False,
                sep=";",
                decimal=",",
                encoding="utf-8-sig"
            )

        # By complexity
        if not self.complexity_table.empty:
            df_complexity = self.complexity_table.reset_index()
            # Sanitize string columns for CSV injection prevention
            for col in df_complexity.columns:
                if df_complexity[col].dtype == 'object':
                    df_complexity[col] = df_complexity[col].apply(_sanitize_csv_value)
            df_complexity.to_csv(
                os.path.join(output_dir, "table2_complexity_analysis.csv"),
                index=False,
                sep=";",
                decimal=",",
                encoding="utf-8-sig"
            )

        # Consistency
        if not self.consistency_table.empty:
            df_consistency = self.consistency_table.reset_index()
            # Sanitize string columns for CSV injection prevention
            for col in df_consistency.columns:
                if df_consistency[col].dtype == 'object':
                    df_consistency[col] = df_consistency[col].apply(_sanitize_csv_value)
            df_consistency.to_csv(
                os.path.join(output_dir, "table3_output_consistency.csv"),
                index=False,
                sep=";",
                decimal=",",
                encoding="utf-8-sig"
            )

        # Raw results - split into two files for usability and reproducibility
        if not self.df.empty:
            # 1. raw_results_view.csv: Human-friendly view (drop long text fields)
            view_columns = [col for col in self.df.columns if col not in ('prompt_text', 'contexts_text')]
            df_view = self.df[view_columns].copy()

            # Sanitize all string columns to prevent CSV injection
            for col in df_view.columns:
                if df_view[col].dtype == 'object':  # String columns
                    df_view[col] = df_view[col].apply(_sanitize_csv_value)

            df_view.to_csv(
                os.path.join(output_dir, "raw_results_view.csv"),
                index=False,
                sep=";",
                decimal=",",
                encoding="utf-8-sig"
            )

            # 2. raw_results_audit.jsonl: Complete audit trail (full prompt/context text)
            # JSONL format: one JSON object per line, preserves all data for reproducibility
            import json
            audit_path = os.path.join(output_dir, "raw_results_audit.jsonl")
            with open(audit_path, 'w', encoding='utf-8') as f:
                for _, row in self.df.iterrows():
                    json.dump(row.to_dict(), f, ensure_ascii=False)
                    f.write('\n')

            print(f"Raw results exported:")
            print(f"  - View: raw_results_view.csv ({len(df_view)} rows, {len(view_columns)} columns)")
            print(f"  - Audit: raw_results_audit.jsonl (full reproducibility data)")

        # Export manifest if config provided
        if config:
            self.export_manifest(output_dir, config, dataset_path)

        # CRITICAL: Always export environment info for reproducibility
        # This allows others to match your exact library versions for hash matching
        self.export_environment_info(output_dir)

    def export_task_specific_results(
        self,
        output_dir: str,
        task_type: str,
        config: Optional['ExperimentConfig'] = None
    ):
        """
        Export task-specific results with 4 clean files.

        This is a simplified export that produces only the essential files
        for thesis analysis, avoiding the proliferation of table files.

        Files exported:
        1. results_clean.csv - Detailed per-case results (metrics, identifiers, quality flags)
        2. aggregated_results.csv - Overall performance summary by (model, context_mode)
        3. summary_stats.json - Aggregate statistics for quick analysis
        4. raw_results_audit.jsonl - Full audit trail for reproducibility

        Args:
            output_dir: Directory to save files
            task_type: Type of task ("diagnosis", "ml_recommendation", etc.)
            config: Optional experiment configuration (for manifest)
        """
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        # Define important columns based on task type
        if task_type == "ml_recommendation":
            important_cols = [
                # Identifiers
                'model', 'context_mode', 'case_id', 'ml_problem_type', 'repetition_index',
                # ML Recommendation Metrics (PRIMARY)
                'ml_problem_type_accuracy', 'algorithm_suitability',
                # RAG Quality Metrics
                'citation_coverage', 'citation_correctness', 'faithfulness',
                'context_precision', 'context_recall', 'hallucination_rate',
                # Overall Quality
                'answer_correctness', 'output_consistency',
                # Context Info
                'contexts_count', 'context_chars',
                # Quality Flags
                'had_error', 'sanity_check_failed',
                # Provenance
                'prompt_hash', 'context_hash', 'run_id', 'seed_used',
                # LLM Response (for analysis)
                'answer',
            ]
        elif task_type == "diagnosis":
            important_cols = [
                # Identifiers
                'model', 'context_mode', 'case_id', 'complexity', 'repetition_index',
                # Diagnosis Metrics
                'answer_correctness', 'faithfulness',
                'citation_coverage', 'citation_correctness',
                'context_precision', 'context_recall', 'hallucination_rate',
                # Overall Quality
                'output_consistency',
                # Context Info
                'contexts_count', 'context_chars',
                # Quality Flags
                'had_error', 'sanity_check_failed',
                # Provenance
                'prompt_hash', 'context_hash', 'run_id', 'seed_used',
                # LLM Response (for analysis)
                'answer',
            ]
        else:
            # Generic fallback - use all metric columns plus standard identifiers
            important_cols = [
                'model', 'context_mode', 'case_id', 'complexity', 'repetition_index',
            ] + self.metrics + [
                'contexts_count', 'context_chars',
                'had_error', 'sanity_check_failed',
                'prompt_hash', 'context_hash', 'run_id', 'seed_used',
                'answer',  # LLM Response
            ]

        # FILE 1: results_clean.csv
        existing_cols = [col for col in important_cols if col in self.df.columns]
        clean_df = self.df[existing_cols].copy()

        # Sanitize string columns for CSV injection prevention
        for col in clean_df.columns:
            if clean_df[col].dtype == 'object':
                clean_df[col] = clean_df[col].apply(_sanitize_csv_value)

        clean_path = os.path.join(output_dir, "results_clean.csv")
        clean_df.to_csv(
            clean_path,
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8-sig"
        )
        print(f"[1/4] Clean results: {clean_path}")
        print(f"      Columns: {len(existing_cols)} (reduced from {len(self.df.columns)})")
        print(f"      Rows: {len(clean_df)}")

        # FILE 2: aggregated_results.csv (TWO-LEVEL AGGREGATION)
        # Level 1: Overall performance per (model, context_mode) with ml_problem_type='ALL'
        # Level 2: Per problem type performance per (model, context_mode, ml_problem_type)
        metric_columns_to_aggregate = [
            # Primary metrics
            'ml_problem_type_accuracy',
            'algorithm_suitability',
            # RAG quality metrics
            'citation_coverage',
            'citation_correctness',
            'faithfulness',
            'context_precision',
            'context_recall',
            'hallucination_rate',
            # Overall quality
            'output_consistency',
        ]

        # Filter to only columns that exist in the dataframe
        existing_metric_columns = [
            col for col in metric_columns_to_aggregate
            if col in clean_df.columns
        ]

        # Create aggregation dictionary
        agg_dict = {
            'case_id': 'count',  # Count how many test results
            **{col: 'mean' for col in existing_metric_columns}
        }

        # ═══════════════════════════════════════════════════════════════
        # AGGREGATION LEVEL 1: Overall Performance (ALL problem types)
        # ═══════════════════════════════════════════════════════════════
        overall_agg = clean_df.groupby(['model', 'context_mode']).agg(agg_dict).reset_index()
        overall_agg = overall_agg.rename(columns={'case_id': 'n_results'})
        overall_agg['ml_problem_type'] = 'ALL'

        # ═══════════════════════════════════════════════════════════════
        # AGGREGATION LEVEL 2: Per Problem Type Performance
        # ═══════════════════════════════════════════════════════════════
        per_type_agg = pd.DataFrame()
        if 'ml_problem_type' in clean_df.columns:
            # Only aggregate if ml_problem_type column exists and has non-empty values
            clean_df_with_type = clean_df[clean_df['ml_problem_type'].notna() & (clean_df['ml_problem_type'] != '')]
            if not clean_df_with_type.empty:
                per_type_agg = clean_df_with_type.groupby(['model', 'context_mode', 'ml_problem_type']).agg(agg_dict).reset_index()
                per_type_agg = per_type_agg.rename(columns={'case_id': 'n_results'})

        # ═══════════════════════════════════════════════════════════════
        # COMBINE BOTH LEVELS
        # ═══════════════════════════════════════════════════════════════
        if not per_type_agg.empty:
            aggregated_df = pd.concat([overall_agg, per_type_agg], ignore_index=True)
        else:
            aggregated_df = overall_agg

        # Sort: model → context_mode → ml_problem_type (with 'ALL' first)
        def sort_key(x):
            if x == 'ALL':
                return (0, '')  # 'ALL' comes first
            else:
                return (1, x)  # Specific types come after, alphabetically

        aggregated_df['sort_order'] = aggregated_df['ml_problem_type'].apply(lambda x: sort_key(x)[0])
        aggregated_df['sort_alpha'] = aggregated_df['ml_problem_type'].apply(lambda x: sort_key(x)[1])
        aggregated_df = aggregated_df.sort_values(
            ['model', 'context_mode', 'sort_order', 'sort_alpha']
        ).drop(['sort_order', 'sort_alpha'], axis=1)

        # Round metric values to 4 decimal places for readability
        for col in existing_metric_columns:
            if col in aggregated_df.columns:
                aggregated_df[col] = aggregated_df[col].round(4)

        # Reorder columns to match specification
        final_columns = ['model', 'context_mode', 'ml_problem_type', 'n_results'] + existing_metric_columns
        final_columns = [c for c in final_columns if c in aggregated_df.columns]
        aggregated_df = aggregated_df[final_columns]

        # Save aggregated results
        aggregated_path = os.path.join(output_dir, "aggregated_results.csv")
        aggregated_df.to_csv(
            aggregated_path,
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8-sig"
        )
        print(f"[2/4] Aggregated results: {aggregated_path}")
        print(f"      Rows: {len(aggregated_df)} (overall + per problem type)")
        print(f"      Columns: {len(aggregated_df.columns)}")

        # Preview aggregated results
        print(f"\n      AGGREGATED RESULTS PREVIEW:")
        print(f"      {aggregated_df.to_string(index=False)}")

        # FILE 3: summary_stats.json
        stats = self.summary_stats()

        # Add task-specific summary
        if task_type == "ml_recommendation":
            # Calculate baseline vs RAG comparison
            baseline_df = self.df[self.df['context_mode'] == 'baseline'] if 'context_mode' in self.df.columns else pd.DataFrame()
            rag_df = self.df[self.df['context_mode'] == 'rag_retrieval'] if 'context_mode' in self.df.columns else pd.DataFrame()

            stats["task_specific"] = {
                "task_type": task_type,
                "baseline_tfr": float(baseline_df['algorithm_suitability'].mean() * 100) if not baseline_df.empty and 'algorithm_suitability' in baseline_df.columns else None,
                "rag_tfr": float(rag_df['algorithm_suitability'].mean() * 100) if not rag_df.empty and 'algorithm_suitability' in rag_df.columns else None,
                "baseline_pta": float(baseline_df['ml_problem_type_accuracy'].mean() * 100) if not baseline_df.empty and 'ml_problem_type_accuracy' in baseline_df.columns else None,
                "rag_pta": float(rag_df['ml_problem_type_accuracy'].mean() * 100) if not rag_df.empty and 'ml_problem_type_accuracy' in rag_df.columns else None,
            }

        stats_path = os.path.join(output_dir, "summary_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        print(f"[3/4] Summary stats: {stats_path}")

        # FILE 4: raw_results_audit.jsonl (full audit trail)
        audit_path = os.path.join(output_dir, "raw_results_audit.jsonl")
        with open(audit_path, 'w', encoding='utf-8') as f:
            for _, row in self.df.iterrows():
                json.dump(row.to_dict(), f, ensure_ascii=False, default=str)
                f.write('\n')
        print(f"[4/4] Audit trail: {audit_path}")

        # Export manifest and environment info if config provided
        if config:
            self.export_manifest(output_dir, config)
        self.export_environment_info(output_dir)

        print(f"\nExport complete: {output_dir}/")
        print(f"  Total files: 4 (+ manifest + environment)")

    def summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the experiment.

        Returns:
            Dictionary with summary statistics
        """
        if self.df.empty:
            return {}

        # STRICT ALLOWLIST: Only use metrics from config.metrics
        metric_cols = [m for m in self.metrics if m in self.df.columns]

        # CHANGE #4: Filter out both runtime errors AND sanity check failures for averaging
        had_error = self.df.get("had_error", False).fillna(False).astype(bool)
        sanity_failed = self.df.get("sanity_check_failed", False).fillna(False).astype(bool)
        valid_df = self.df[~(had_error | sanity_failed)]

        # PER-REPETITION FIX: Deduplicate by case_id first
        if "case_id" in valid_df.columns and "repetition_index" in valid_df.columns:
            dedup_df = valid_df[valid_df["repetition_index"] == 0].copy()
            # Count unique test cases (not repetitions)
            n_test_cases = self.df[self.df["repetition_index"] == 0].shape[0]
            n_valid_cases = dedup_df.shape[0]
        else:
            dedup_df = valid_df
            n_test_cases = len(self.df)
            n_valid_cases = len(valid_df)

        return {
            "n_test_cases": n_test_cases,
            "n_valid_cases": n_valid_cases,
            "n_models": self.df["model"].nunique() if "model" in self.df.columns else 0,
            "metrics": metric_cols,
            "avg_scores": dedup_df[metric_cols].mean().to_dict() if not dedup_df.empty else {},
        }

    def __repr__(self) -> str:
        """String representation of results"""
        n_results = len(self.raw_results)
        n_models = self.df["model"].nunique() if not self.df.empty and "model" in self.df.columns else 0
        return f"ExperimentResults(n_results={n_results}, n_models={n_models})"


# ================= Reproducibility Validation =================

def validate_reproducibility(
    results_file: str,
    expected_prompt: str,
    expected_contexts: List[str]
) -> bool:
    """
    Validate experiment reproducibility by checking cryptographic hashes.

    This function ensures that someone trying to reproduce your results
    is using the EXACT same prompt text and context chunks.

    Args:
        results_file: Path to raw_results_audit.jsonl file
        expected_prompt: The prompt text being used for reproduction
        expected_contexts: The context chunks being used for reproduction

    Returns:
        True if hashes match (reproducible), raises ValueError otherwise

    Raises:
        ValueError: If prompt or context hashes don't match (not reproducible)

    Example:
        >>> validate_reproducibility(
        ...     "results/raw_results_audit.jsonl",
        ...     my_prompt_text,
        ...     my_context_chunks
        ... )
        True
    """
    import hashlib
    import json

    # Compute expected hashes from provided inputs
    prompt_hash = hashlib.sha256(expected_prompt.encode('utf-8')).hexdigest()
    context_str = "\n\n".join(expected_contexts) if expected_contexts else ""
    context_hash = hashlib.sha256(context_str.encode('utf-8')).hexdigest()

    # Load first result from audit file to get stored hashes
    with open(results_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if not first_line:
            raise ValueError(f"Results file {results_file} is empty")
        result = json.loads(first_line)

    # Validate prompt hash
    stored_prompt_hash = result.get('prompt_hash')
    if not stored_prompt_hash:
        raise ValueError("Results file missing 'prompt_hash' field")

    if stored_prompt_hash != prompt_hash:
        raise ValueError(
            f"REPRODUCIBILITY FAILURE: Prompt hash mismatch!\n"
            f"Expected: {prompt_hash}\n"
            f"Got:      {stored_prompt_hash}\n"
            f"\n"
            f"This means you are NOT using the exact same prompt text.\n"
            f"Even a single character difference will cause mismatch.\n"
            f"Check: whitespace, newlines, punctuation, capitalization."
        )

    # Validate context hash
    stored_context_hash = result.get('context_hash')
    if not stored_context_hash:
        raise ValueError("Results file missing 'context_hash' field")

    if stored_context_hash != context_hash:
        raise ValueError(
            f"REPRODUCIBILITY FAILURE: Context hash mismatch!\n"
            f"Expected: {context_hash}\n"
            f"Got:      {stored_context_hash}\n"
            f"\n"
            f"This means you are NOT using the exact same context chunks.\n"
            f"Possible causes:\n"
            f"- Different retrieval results (BM25 scores changed)\n"
            f"- Different chunk splitting (chunk_size parameter)\n"
            f"- Different PDF text extraction (OCR differences)\n"
            f"- Different retrieval parameters (top_k, window_size)"
        )

    # Both hashes match - reproducible!
    return True


# ================= Helper Functions =================

def validate_test_case(test_case: TestCase, task_type: Optional[str] = None) -> bool:
    """
    Validate that a test case has required fields for a specific task type.

    Args:
        test_case: TestCase to validate
        task_type: Optional task type for task-specific validation
                   ("diagnosis", "repurposing", "ml_recommendation", etc.)

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Basic validation
    if not test_case.case_id:
        raise ValueError("Test case must have a case_id")

    if not test_case.input_data:
        raise ValueError(f"Test case {test_case.case_id} has empty input_data")

    if not test_case.ground_truth:
        raise ValueError(f"Test case {test_case.case_id} has empty ground_truth")

    # CRITICAL: Check for ground-truth leakage
    # Prevent ground-truth field names from appearing in input_data
    # This ensures the model cannot simply copy answers from the input
    for gt_key in test_case.ground_truth.keys():
        if gt_key in test_case.input_data:
            raise ValueError(
                f"GROUND-TRUTH LEAKAGE in test case {test_case.case_id}: "
                f"Field '{gt_key}' appears in BOTH input_data and ground_truth. "
                f"This allows the model to see the answer in the prompt. "
                f"Ground-truth fields must ONLY appear in ground_truth, never in input_data."
            )

    # Task-specific validation
    if task_type == "diagnosis":
        # Diagnosis tasks require specific input fields
        required_inputs = ["fault_description", "appliance"]
        for field in required_inputs:
            if field not in test_case.input_data or not test_case.input_data[field]:
                raise ValueError(
                    f"Diagnosis test case {test_case.case_id} missing required input field: '{field}'. "
                    f"Available fields: {list(test_case.input_data.keys())}"
                )

        # Diagnosis tasks require diagnosis in ground truth
        if "diagnosis" not in test_case.ground_truth or not test_case.ground_truth["diagnosis"]:
            raise ValueError(
                f"Diagnosis test case {test_case.case_id} missing required ground_truth field: 'diagnosis'. "
                f"Available fields: {list(test_case.ground_truth.keys())}"
            )

    elif task_type == "repurposing":
        # Repurposing tasks require component and scenarios
        if "component" not in test_case.input_data or not test_case.input_data["component"]:
            raise ValueError(
                f"Repurposing test case {test_case.case_id} missing required input field: 'component'"
            )

        if "scenarios" not in test_case.ground_truth or not test_case.ground_truth["scenarios"]:
            raise ValueError(
                f"Repurposing test case {test_case.case_id} missing required ground_truth field: 'scenarios'"
            )

    elif task_type == "ml_recommendation":
        # ML recommendation tasks require problem formulation
        required_inputs = ["problem_formulation"]
        for field in required_inputs:
            if field not in test_case.input_data or not test_case.input_data[field]:
                raise ValueError(
                    f"ML recommendation test case {test_case.case_id} missing required input field: '{field}'. "
                    f"Available fields: {list(test_case.input_data.keys())}"
                )

        # ML recommendation requires both problem type and algorithm in ground truth
        if "ml_problem_type" not in test_case.ground_truth or not test_case.ground_truth["ml_problem_type"]:
            raise ValueError(
                f"ML recommendation test case {test_case.case_id} missing required ground_truth field: 'ml_problem_type'. "
                f"Available fields: {list(test_case.ground_truth.keys())}"
            )

        if "ml_algorithm" not in test_case.ground_truth or not test_case.ground_truth["ml_algorithm"]:
            raise ValueError(
                f"ML recommendation test case {test_case.case_id} missing required ground_truth field: 'ml_algorithm'. "
                f"Available fields: {list(test_case.ground_truth.keys())}"
            )

    return True
