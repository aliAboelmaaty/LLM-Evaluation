"""
Experiment Runner Module

Orchestrates LLM evaluation experiments following research methodology.

Features:
- Multi-model testing
- Baseline vs RAG comparison
- Multiple repetitions for consistency analysis
- Comprehensive metrics calculation
- Progress tracking
- Resilient error handling

TRI-STATE OPERATIONAL FIELDS:
===============================
Operational/debug fields use tri-state logic (True/False/None) to distinguish:
- True: Operation executed and succeeded
- False: Operation executed but failed/not used
- None: Operation not applicable or result not trustworthy

Fields using tri-state:
- pdf_extraction_cache_hit: None when baseline, no manual, or runtime error
- cache_hit: (backward compat alias for pdf_extraction_cache_hit)
- retrieval_executed: None when baseline, no manual, or runtime error
- manual_full_loaded: None when baseline, no manual, or runtime error

When fields are None:
1. context_mode == BASELINE: Cache/retrieval operations don't apply
2. manual_available == False: No manual document to process
3. had_error == True: Runtime error makes operational data untrustworthy
4. pdf_extraction_failed == True: PDF extraction failed, cache state unknown

This prevents misleading False values when operations were never attempted.
"""

from typing import List, Dict, Any, Optional
import json
import re
from tqdm import tqdm

from .core import ExperimentConfig, TestCase, ExperimentResults, LLMProvider, ContextMode
from .dataset import Dataset
from .prompts import PromptLibrary
from .metrics import MetricCalculator, MetricsConfig, get_citation_instruction
from .llm_service import LLMService


# ================= Provenance Helper =================

def compute_provenance_fields(
    first_output: Dict[str, Any],
    contexts: List[str],
    run_id: str,
    model: LLMProvider,
    llm_service: 'LLMService',
    temperature_used: float,
    max_tokens_used: int,
    top_p_used: float,
    prompt_variant: Optional[str]
) -> Dict[str, Any]:
    """
    CENTRALIZED provenance computation to guarantee consistency.

    Computes all reproducibility and audit trail fields in ONE place.
    This ensures no competing logic can create inconsistent hashes.

    Args:
        first_output: First repetition output dict
        contexts: List of context strings
        run_id: Unique run identifier
        model: LLM provider
        llm_service: LLM service instance
        temperature_used: Temperature parameter used
        max_tokens_used: Max tokens parameter used
        prompt_variant: Prompt variant name

    Returns:
        Dict with all provenance fields (hashes, identifiers, audit trail)
    """
    import hashlib

    # Extract strings (use empty string as default, never None)
    prompt_str = first_output.get("prompt", "")
    retrieval_query_str = first_output.get("retrieval_query", "")
    context_str = "\n\n".join(contexts) if contexts else ""

    # CRITICAL: Always compute hashes, even for empty strings
    # Empty string -> deterministic hash (not None)
    # This ensures hash fields are ALWAYS present and NEVER None
    prompt_hash = hashlib.sha256(prompt_str.encode('utf-8')).hexdigest()
    context_hash = hashlib.sha256(context_str.encode('utf-8')).hexdigest()

    # Get model identifier (provider-specific)
    if model == LLMProvider.GEMINI:
        model_identifier = llm_service.gemini_model
        provider_backend = "gemini"
    elif model in (LLMProvider.GEMMA3_4B, LLMProvider.GEMMA3_12B, LLMProvider.GEMMA3_27B, LLMProvider.GEMMA3):
        if model == LLMProvider.GEMMA3_4B:
            model_identifier = llm_service.gemma3_4b_it
        elif model == LLMProvider.GEMMA3_12B:
            model_identifier = llm_service.gemma3_12b_it
        else:
            model_identifier = llm_service.gemma3_27b_it
        provider_backend = "replicate"
    elif model == LLMProvider.DEEPSEEK:
        model_identifier = llm_service.deepseek_model
        provider_backend = "replicate"
    elif model == LLMProvider.GPT5:
        model_identifier = llm_service.gpt5_model
        provider_backend = "openai"
    else:
        model_identifier = model.value
        provider_backend = "unknown"

    # Return all provenance fields in one dict
    return {
        # Hash-based reproducibility fields
        "prompt_hash": prompt_hash,
        "context_hash": context_hash,
        "retrieval_query": retrieval_query_str,

        # Model identification fields
        "model_identifier": model_identifier,
        "provider_backend": provider_backend,

        # Protocol parameters (FAIRNESS)
        "temperature_used": temperature_used,
        "max_tokens_used": max_tokens_used,
        "top_p_used": top_p_used,
        "prompt_variant": prompt_variant if prompt_variant else "default",
        "seed_used": first_output.get("seed_used", None),

        # Audit trail (REPRODUCIBILITY)
        "run_id": run_id,

        # Full text for exact replay (REPRODUCIBILITY)
        "prompt_text": prompt_str,
        "contexts_text": context_str
    }


# ================= Experiment Runner =================

class ExperimentRunner:
    """
    Runs LLM evaluation experiments following methodology from research papers.

    Implements:
    - Multiple repetitions (for consistency analysis - Sonntag's OCR)
    - Baseline vs RAG comparison
    - Complexity-based analysis (Dörnbach's clusters)
    - Comprehensive metrics calculation
    """

    def __init__(
        self,
        config: ExperimentConfig,
        llm_service: LLMService
    ):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
            llm_service: LLM service instance
        """
        self.config = config
        self.llm_service = llm_service
        self.results: List[Dict[str, Any]] = []
        # Pass random_seed to MetricCalculator for reproducible metric calculations
        self.metrics_calculator = MetricCalculator(random_seed=config.random_seed)

    def run(self, dataset: Dataset, verbose: bool = True) -> ExperimentResults:
        """
        Main experiment execution.

        Process:
        1. For each model in config.models
        2. For each context_mode in config.context_modes
        3. For each test case in dataset
        4. Run config.n_repetitions times
        5. Calculate all configured metrics
        6. Aggregate results

        Args:
            dataset: Dataset of test cases
            verbose: Whether to show progress bars

        Returns:
            ExperimentResults with all data and analysis tables
        """
        # Validate retrieval configuration (warn if suboptimal)
        self.config.validate_retrieval_config(self.config.context_modes)

        # REPRODUCIBILITY: Set global random seed for consistent behavior
        import random
        random.seed(self.config.random_seed)

        # Also seed numpy if available for full reproducibility
        try:
            import numpy as np
            np.random.seed(self.config.random_seed)
        except ImportError:
            pass

        # REPRODUCIBILITY: Generate unique run ID and capture git commit
        import uuid
        import subprocess
        from datetime import datetime

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Experiment: {self.config.task_type}")
            print(f"Run ID: {run_id}")
            print(f"Models: {[m.value for m in self.config.models]}")
            print(f"Context Modes: {[cm.value for cm in self.config.context_modes]}")
            print(f"Test Cases: {len(dataset)}")
            print(f"Repetitions: {self.config.n_repetitions}")
            print(f"Random Seed: {self.config.random_seed}")
            # Enhanced Retrieval Status (v2.0)
            if self.config.is_enhanced_retrieval_enabled():
                print(f"Enhanced Retrieval: ENABLED")
                print(f"  - Query Expansion: {self.config.enable_query_expansion}")
                print(f"  - Window Size: {self.config.retrieval_window_size}")
                print(f"  - Min Score: {self.config.retrieval_min_score}")
            else:
                print(f"Enhanced Retrieval: disabled (v1.0 compatibility)")
            print(f"{'='*60}\n")

        # Progress bar setup
        total_runs = (
            len(self.config.models) *
            len(self.config.context_modes) *
            len(dataset) *
            self.config.n_repetitions
        )

        # REPRODUCIBILITY: Sort test cases by case_id for deterministic ordering
        sorted_test_cases = sorted(dataset, key=lambda tc: tc.case_id)

        with tqdm(total=total_runs, desc="Running experiments", disable=not verbose) as pbar:
            for model in self.config.models:
                for context_mode in self.config.context_modes:
                    if verbose:
                        print(f"\nTesting {model.value} ({context_mode.value})...")

                    for test_case in sorted_test_cases:
                        # LEAVE-ONE-OUT KB BUILDING for RAG mode
                        # Prevents data leakage by excluding current test case from KB
                        original_context_doc = test_case.context_document
                        if (context_mode == ContextMode.RAG_RETRIEVAL and
                            hasattr(dataset, 'metadata') and
                            dataset.metadata.get('source_path')):
                            # Build KB excluding this test case
                            from build_ml_knowledge_base import build_kb_for_case
                            kb_path = build_kb_for_case(
                                dataset_path=dataset.metadata['source_path'],
                                case_id=test_case.case_id,
                                cache_dir="kb_cache"
                            )
                            # Override context_document for this run
                            test_case.context_document = kb_path
                            if verbose:
                                print(f"  [Leave-one-out] Using KB excluding {test_case.case_id}")

                        # Mixed dataset behavior: skip MANUAL_FULL/RAG_RETRIEVAL when no manual exists
                        if context_mode != ContextMode.BASELINE and not test_case.context_document:
                            # Skip this test case for non-baseline modes
                            if pbar:
                                pbar.update(self.config.n_repetitions)
                            # Restore original context_document
                            test_case.context_document = original_context_doc
                            continue

                        # Run with repetitions
                        outputs = self._run_with_repetitions(
                            model=model,
                            test_case=test_case,
                            context_mode=context_mode,
                            pbar=pbar
                        )

                        # CHANGE #1: Determine extraction failure BEFORE sanity check
                        first_output = outputs[0] if outputs else {}
                        contexts = first_output.get("contexts", [])
                        if isinstance(contexts, str):
                            try:
                                contexts = json.loads(contexts) if contexts else []
                            except json.JSONDecodeError:
                                contexts = []

                        # Check if there was a runtime error first
                        had_runtime_error = (
                            first_output.get("error") or
                            str(first_output.get("answer", "")).startswith("[ERROR:")
                        )

                        # Check if extraction failed (manual exists but contexts empty for non-baseline)
                        # FIX: Only flag extraction failure if there was NO runtime error
                        pdf_extraction_failed = (
                            context_mode != ContextMode.BASELINE and
                            test_case.context_document and
                            len(contexts) == 0 and
                            not had_runtime_error
                        )

                        # CHANGE #5: Lightweight guard - check context count consistency across reps
                        context_count_first = len(contexts)
                        context_inconsistent = False
                        for i, out in enumerate(outputs[1:], start=1):
                            out_contexts = out.get("contexts", [])
                            if isinstance(out_contexts, str):
                                try:
                                    out_contexts = json.loads(out_contexts) if out_contexts else []
                                except json.JSONDecodeError:
                                    out_contexts = []
                            if len(out_contexts) != context_count_first:
                                print(f"WARNING: Inconsistent context count across repetitions for {test_case.case_id}: "
                                      f"rep 0 had {context_count_first}, rep {i} had {len(out_contexts)}")
                                context_inconsistent = True
                                break

                        # Sanity check: validate contexts based on context_mode
                        sanity_check_passed, sanity_error_msg = self._sanity_check_contexts(
                            outputs, context_mode, self.config.top_k_retrieval
                        )

                        # CHANGE #5: Mark as sanity failure if contexts inconsistent
                        if context_inconsistent:
                            sanity_check_passed = False
                            sanity_error_msg = f"Inconsistent context counts across repetitions (first={context_count_first})"

                        # FAIRNESS: Schema validation (ONLY if explicitly enabled)
                        # WARNING: Disabled by default because default prompts don't match the hardcoded schema
                        # Only enable if using custom prompts that request: Diagnosis, Root Cause, Steps, Safety, Citations
                        if sanity_check_passed and self.config.enforce_output_schema and self.config.task_type == "diagnosis":
                            first_answer = first_output.get("answer", "")
                            schema_passed, schema_error = self._check_output_schema(first_answer)
                            if not schema_passed:
                                sanity_check_passed = False
                                sanity_error_msg = f"Schema violation: {schema_error}"

                        # CHANGE #2, #3, #7: Handle sanity failure properly
                        # Calculate aggregated metrics ONCE (metrics that need all repetitions)
                        if not sanity_check_passed:
                            # Sanity check failed
                            print(f"SANITY CHECK FAILED for {test_case.case_id}: {sanity_error_msg}")
                            aggregated_metrics_shared = {}
                            if "output_consistency" in self.config.metrics:
                                aggregated_metrics_shared["output_consistency"] = float('nan')
                            sanity_error_detail = f"SANITY: {sanity_error_msg}"
                        else:
                            # Calculate aggregated metrics (output_consistency)
                            aggregated_metrics_shared = self._calculate_aggregated_metrics(outputs)
                            sanity_error_detail = ""

                        # PROVENANCE: Determine protocol parameters
                        temperature_used = self.config.temperature
                        if context_mode == ContextMode.BASELINE:
                            max_tokens_used = self.config.baseline_max_tokens if self.config.baseline_max_tokens is not None else self.config.max_tokens
                        else:
                            max_tokens_used = self.config.max_tokens

                        # PER-REPETITION RESULTS: Create one row per repetition
                        # Each row gets its OWN per-repetition metrics (answer_correctness for THAT answer)
                        # Plus shared aggregated metrics (output_consistency duplicated across all reps)
                        for rep_index, output in enumerate(outputs):
                            # Extract contexts for this repetition
                            rep_contexts = output.get("contexts", [])
                            if isinstance(rep_contexts, str):
                                try:
                                    rep_contexts = json.loads(rep_contexts) if rep_contexts else []
                                except json.JSONDecodeError:
                                    rep_contexts = []

                            # TRI-STATE: Extract debug info with proper None handling
                            # Determine if operations are applicable/trustworthy
                            manual_available = bool(test_case.context_document)
                            rep_had_error = (
                                output.get("error") or
                                str(output.get("answer", "")).startswith("[ERROR:")
                            )
                            operations_not_applicable = (
                                context_mode == ContextMode.BASELINE or
                                not manual_available or
                                rep_had_error or
                                pdf_extraction_failed
                            )

                            # Cache fields: None if operations not applicable, else get from output
                            if operations_not_applicable:
                                pdf_cache_hit = None
                                retrieval_exec = None
                                manual_loaded = None
                            else:
                                pdf_cache_hit = output.get("pdf_extraction_cache_hit", output.get("cache_hit", None))
                                retrieval_exec = output.get("retrieval_executed", None)
                                manual_loaded = output.get("manual_full_loaded", None)

                            # PROVENANCE: Centralized computation for this repetition
                            provenance = compute_provenance_fields(
                                first_output=output,  # Use this specific repetition's output
                                contexts=rep_contexts,
                                run_id=run_id,
                                model=model,
                                llm_service=self.llm_service,
                                temperature_used=temperature_used,
                                max_tokens_used=max_tokens_used,
                                top_p_used=self.config.top_p,
                                prompt_variant=self.config.prompt_variant
                            )

                            # PER-REPETITION METRICS: Calculate metrics for THIS specific answer
                            if not sanity_check_passed:
                                # Sanity failed: all per-rep metrics = NaN
                                per_rep_metrics = {metric_name: float('nan') for metric_name in self.config.metrics if metric_name != "output_consistency"}
                                per_rep_metrics["sanity_check_failed"] = 1.0
                                per_rep_metrics["had_error"] = 0.0
                                per_rep_metrics["pdf_extraction_failed"] = 1.0 if pdf_extraction_failed else 0.0
                            else:
                                # Calculate per-rep metrics for THIS answer
                                per_rep_metrics = self._calculate_per_repetition_metrics(output, test_case, context_mode)
                                per_rep_metrics["sanity_check_failed"] = 0.0
                                # Add pdf_extraction_failed if not already set
                                if "pdf_extraction_failed" not in per_rep_metrics:
                                    per_rep_metrics["pdf_extraction_failed"] = 1.0 if pdf_extraction_failed else 0.0

                            # Extract retrieval metadata (v2.0)
                            enhanced_retrieval_used = output.get("enhanced_retrieval_used", False)
                            query_expansion_enabled = output.get("query_expansion_enabled", False)
                            retrieval_window_size = output.get("retrieval_window_size", 0)

                            debug_info = {
                                "repetition_index": rep_index,  # Which repetition (0-4)
                                "answer": output.get("answer", ""),  # Actual answer for THIS repetition
                                "contexts_count": len(rep_contexts),
                                "context_chars": output.get("context_chars", 0),
                                "prompt_chars": output.get("prompt_chars", 0),
                                "total_chunks": output.get("total_chunks", 0),
                                "chunks_used": output.get("chunks_used", 0),
                                "truncated": output.get("truncated", False),
                                # TRI-STATE: Operational fields (True/False/None)
                                "pdf_extraction_cache_hit": pdf_cache_hit,
                                "cache_hit": pdf_cache_hit,  # Backward compatibility
                                "retrieval_executed": retrieval_exec,
                                "manual_full_loaded": manual_loaded,
                                # Enhanced Retrieval Metadata (v2.0)
                                "enhanced_retrieval_used": enhanced_retrieval_used,
                                "query_expansion_enabled": query_expansion_enabled,
                                "retrieval_window_size": retrieval_window_size,
                                # CHANGE #7: Combine runtime error and sanity error
                                "error": sanity_error_detail or output.get("error", ""),
                                # GROUND TRUTH: Store for audit/debugging (verify metric inputs)
                                "gt_ml_problem_type": test_case.ground_truth.get('ml_problem_type', ''),
                                "gt_ml_algorithm": test_case.ground_truth.get('ml_algorithm', ''),
                                # PROVENANCE: All fields from centralized computation
                                **provenance
                            }

                            # Extract ml_problem_type from ground_truth (for ML recommendation tasks)
                            # Handle compound types (e.g., "Classification; Regression") by taking the first
                            ml_problem_type_raw = test_case.ground_truth.get('ml_problem_type', '')
                            if ';' in ml_problem_type_raw:
                                ml_problem_type = ml_problem_type_raw.split(';')[0].strip()
                            else:
                                ml_problem_type = ml_problem_type_raw.strip()

                            # Normalize capitalization
                            problem_type_map = {
                                'classification': 'Classification',
                                'regression': 'Regression',
                                'clustering': 'Clustering',
                                'association': 'Association',
                                'association rules': 'Association',
                            }
                            ml_problem_type = problem_type_map.get(ml_problem_type.lower(), ml_problem_type.capitalize() if ml_problem_type else '')

                            # Store ONE result row per repetition
                            self.results.append({
                                "model": model.value,
                                "context_mode": context_mode.value,
                                "uses_context": context_mode != ContextMode.BASELINE,
                                "use_rag_compat": context_mode != ContextMode.BASELINE,  # Backwards compatibility
                                "case_id": test_case.case_id,
                                "ml_problem_type": ml_problem_type,  # NEW: For ML recommendation breakdown
                                "complexity": test_case.metadata.get("complexity", 1),
                                "manual_available": bool(test_case.context_document),
                                **per_rep_metrics,  # PER-REPETITION metrics (DIFFERENT for each rep!)
                                **aggregated_metrics_shared,  # Aggregated metrics (SAME for all reps - only output_consistency)
                                **debug_info  # Per-repetition data (different for each rep)
                            })

                        # Restore original context_document after leave-one-out KB use
                        test_case.context_document = original_context_doc

        return self._aggregate_results()

    def _run_with_repetitions(
        self,
        model: LLMProvider,
        test_case: TestCase,
        context_mode: ContextMode,
        pbar: Optional[tqdm] = None
    ) -> List[Dict[str, Any]]:
        """
        Run same test case n times to measure consistency.

        Args:
            model: LLM provider
            test_case: Test case to run
            context_mode: Context mode (BASELINE, MANUAL_FULL, or RAG_RETRIEVAL)
            pbar: Progress bar to update

        Returns:
            List of LLM outputs (one per repetition)
        """
        outputs = []

        # Select template based on context_mode
        # CRITICAL FIX: BASELINE uses template WITHOUT {context}
        # MANUAL_FULL and RAG_RETRIEVAL use template WITH {context}
        if self.config.custom_prompt:
            template = self.config.custom_prompt
        else:
            use_rag = (context_mode != ContextMode.BASELINE)
            template = PromptLibrary.get_template(
                self.config.task_type,
                use_rag=use_rag,
                variant=self.config.prompt_variant
            )

        # Extract retrieval query from input_data (separate from full prompt)
        # This is used for BM25 retrieval, NOT the full rendered prompt
        retrieval_query = (
            test_case.input_data.get("fault_description") or
            test_case.input_data.get("question") or
            test_case.input_data.get("component") or
            test_case.input_data.get("problem_description") or
            str(list(test_case.input_data.values())[0] if test_case.input_data else "")[:500]
        )

        for i in range(self.config.n_repetitions):
            # REPRODUCIBILITY: Compute deterministic seed per repetition
            seed_used = self.config.random_seed + i

            try:
                if context_mode == ContextMode.BASELINE:
                    # BASELINE: No context, no citation requirements
                    # This mode must be "chemically pure" - no citation constraints
                    # since the model has no sources to cite from
                    prompt = template.render(**test_case.input_data)
                    baseline_max = self.config.baseline_max_tokens if self.config.baseline_max_tokens is not None else self.config.max_tokens
                    result = self.llm_service.ask(
                        question=prompt,
                        provider=model,
                        temperature=self.config.temperature,
                        max_tokens=baseline_max,
                        top_p=self.config.top_p,
                        seed=seed_used  # Pass seed to provider (if supported)
                    )
                    result["contexts"] = []
                    # TRI-STATE: BASELINE doesn't use retrieval/caching (not applicable = None)
                    result["retrieval_executed"] = None
                    result["manual_full_loaded"] = None
                    result["pdf_extraction_cache_hit"] = None
                    result["cache_hit"] = None
                    # PROVENANCE: Store prompt for hash computation
                    result["prompt"] = prompt
                    result["retrieval_query"] = ""  # No retrieval in BASELINE
                    result["seed_used"] = seed_used  # Track seed for reproducibility

                elif test_case.context_document:
                    # MANUAL_FULL or RAG_RETRIEVAL: Get contexts, then render prompt
                    from .llm_service import get_contexts

                    # Get provider-aware budgets
                    budgets = self.config.get_context_budgets(model)

                    # Get enhanced retrieval kwargs (v2.0) - only for RAG_RETRIEVAL
                    retrieval_kwargs = None
                    if context_mode == ContextMode.RAG_RETRIEVAL:
                        retrieval_kwargs = self.config.get_retrieval_kwargs()

                    contexts, metadata = get_contexts(
                        manual_path=test_case.context_document,
                        context_mode=context_mode,
                        retrieval_query=retrieval_query,  # Use short query, NOT full prompt
                        top_k=self.config.top_k_retrieval,
                        chunk_size=budgets["chunk_size"],
                        max_context_chars=budgets["max_context_chars"],
                        manual_full_max_chunks=budgets["manual_full_max_chunks"],
                        retrieval_kwargs=retrieval_kwargs  # Enhanced retrieval (v2.0)
                    )

                    # Render prompt WITH contexts
                    # CRITICAL: Prepend [1], [2], etc. to each chunk so the model
                    # can actually cite them using the format we request in the prompt
                    numbered_contexts = [f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)]
                    context_text = "\n\n".join(numbered_contexts)
                    prompt = template.render(context=context_text, **test_case.input_data)
                    # LAYER 2: Inject citation instruction to enforce [1], [2] format
                    citation_rule = get_citation_instruction()
                    prompt = f"{prompt}\n\n{citation_rule}"

                    # Pass fully-rendered prompt to LLM (NO double wrapping)
                    result = self.llm_service.ask(
                        question=prompt,
                        provider=model,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        top_p=self.config.top_p,
                        seed=seed_used  # Pass seed to provider (if supported)
                    )
                    result["contexts"] = contexts
                    # FIX #1: Rename for clarity - this is PDF extraction caching, not retrieval caching
                    result["pdf_extraction_cache_hit"] = metadata.get("cache_hit", False)
                    result["cache_hit"] = metadata.get("cache_hit", False)  # Backward compatibility
                    result["total_chunks"] = metadata.get("total_chunks", 0)
                    result["chunks_used"] = metadata.get("chunks_used", 0)
                    result["truncated"] = metadata.get("truncated", False)
                    result["context_chars"] = metadata.get("context_chars", 0)
                    result["prompt_chars"] = len(prompt)
                    # Enhanced Retrieval Metadata (v2.0)
                    result["enhanced_retrieval_used"] = metadata.get("enhanced_retrieval_used", False)
                    result["query_expansion_enabled"] = metadata.get("query_expansion_enabled", False)
                    result["retrieval_window_size"] = metadata.get("retrieval_window_size", 0)
                    # TRI-STATE: Only set to True when applicable, None when not applicable
                    # RAG_RETRIEVAL: retrieval_executed=True, manual_full_loaded=None
                    # MANUAL_FULL: retrieval_executed=None, manual_full_loaded=True
                    if context_mode == ContextMode.RAG_RETRIEVAL:
                        result["retrieval_executed"] = True
                        result["manual_full_loaded"] = None
                    else:  # MANUAL_FULL
                        result["retrieval_executed"] = None
                        result["manual_full_loaded"] = True
                    # PROVENANCE: Store prompt and retrieval_query for hash computation
                    result["prompt"] = prompt
                    result["retrieval_query"] = retrieval_query
                    result["seed_used"] = seed_used  # Track seed for reproducibility

                else:
                    # No manual available but not baseline mode
                    result = {
                        "answer": "[ERROR: No manual document provided]",
                        "error": "No context_document in test case",
                        "contexts": [],
                        # TRI-STATE: Manual missing = operations not executed (None)
                        "retrieval_executed": None,
                        "manual_full_loaded": None,
                        "pdf_extraction_cache_hit": None,
                        "cache_hit": None,
                        "prompt": "",
                        "retrieval_query": "",
                        "seed_used": seed_used
                    }

                outputs.append(result)

            except Exception as e:
                # Error handling: store error message
                # TRI-STATE: Runtime error = operations not trustworthy (None)
                outputs.append({
                    "answer": f"[ERROR: {str(e)}]",
                    "error": str(e),
                    "contexts": [],
                    "retrieval_executed": None,
                    "manual_full_loaded": None,
                    "pdf_extraction_cache_hit": None,
                    "cache_hit": None,
                    "prompt": "",
                    "retrieval_query": "",
                    "seed_used": seed_used
                })

            if pbar:
                pbar.update(1)

        return outputs

    def _calculate_per_repetition_metrics(
        self,
        output: Dict[str, Any],
        test_case: TestCase,
        context_mode: ContextMode
    ) -> Dict[str, float]:
        """
        Calculate metrics for a SINGLE repetition (per-repetition metrics).

        This calculates metrics that can be evaluated on one answer alone,
        such as answer_correctness, faithfulness, citation_coverage, etc.

        Args:
            output: Single LLM output from one repetition
            test_case: Original test case
            context_mode: Context mode used

        Returns:
            Dictionary of per-repetition metric scores
        """
        metrics: Dict[str, float] = {}

        answer = output.get("answer", "")

        # Check for error
        if output.get("error") or str(answer).startswith("[ERROR:"):
            # Error case: all metrics 0, had_error = 1
            for metric_name in self.config.metrics:
                if metric_name != "output_consistency":  # Skip aggregated metric
                    metrics[metric_name] = 0.0
            metrics["had_error"] = 1.0
            return metrics

        # Extract contexts
        contexts = output.get("contexts", [])
        if isinstance(contexts, str):
            try:
                contexts = json.loads(contexts) if contexts else []
            except json.JSONDecodeError:
                contexts = []

        # Get question
        question = test_case.input_data.get("question") or \
                   test_case.input_data.get("fault_description", "")

        # RAG-specific metrics
        RAG_METRICS = {
            "faithfulness", "citation_coverage", "citation_correctness",
            "context_precision", "context_recall"
        }

        # Calculate per-repetition metrics (exclude output_consistency)
        for metric_name in self.config.metrics:
            if metric_name == "output_consistency":
                # Skip - this is aggregated metric calculated separately
                continue

            # Skip RAG metrics for BASELINE
            if metric_name in RAG_METRICS and context_mode == ContextMode.BASELINE:
                metrics[metric_name] = float('nan')
                continue

            try:
                score = self._calculate_single_metric(
                    metric_name,
                    answer,
                    contexts,
                    question,
                    test_case,
                    [output]  # Pass as single-item list for compatibility
                )
                metrics[metric_name] = round(score, 3)
            except Exception as e:
                metrics[metric_name] = 0.0
                print(f"Warning: Failed to calculate {metric_name}: {e}")

        # Ensure had_error is present
        if "had_error" not in metrics:
            metrics["had_error"] = 0.0

        return metrics

    def _calculate_aggregated_metrics(
        self,
        outputs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate metrics that require ALL repetitions (aggregated metrics).

        Currently only output_consistency requires all repetitions.

        Args:
            outputs: List of all LLM outputs from all repetitions

        Returns:
            Dictionary of aggregated metric scores
        """
        metrics: Dict[str, float] = {}

        # Only calculate output_consistency if it's in configured metrics
        if "output_consistency" in self.config.metrics:
            # Filter out error outputs
            valid_answers = [
                out.get("answer", "")
                for out in outputs
                if not out.get("error") and not str(out.get("answer", "")).startswith("[ERROR:")
            ]

            if len(valid_answers) < 2:
                # Single valid answer or all errors
                metrics["output_consistency"] = 1.0 if valid_answers else 0.0
            else:
                try:
                    metrics["output_consistency"] = round(
                        self.metrics_calculator.calculate_metric(
                            "output_consistency",
                            outputs=valid_answers
                        ),
                        3
                    )
                except Exception as e:
                    metrics["output_consistency"] = 0.0
                    print(f"Warning: Failed to calculate output_consistency: {e}")

        return metrics

    def _calculate_metrics(
        self,
        outputs: List[Dict[str, Any]],
        test_case: TestCase,
        context_mode: ContextMode
    ) -> Dict[str, float]:
        """
        DEPRECATED: Legacy method for backward compatibility.

        This method is kept for compatibility but now just calls the new
        per-repetition method on the first output. Use the new methods instead:
        - _calculate_per_repetition_metrics() for per-rep metrics
        - _calculate_aggregated_metrics() for aggregated metrics

        Args:
            outputs: List of LLM outputs from repetitions
            test_case: Original test case
            context_mode: Context mode used

        Returns:
            Dictionary of metric scores
        """
        if not outputs:
            metrics = {}
            for metric_name in self.config.metrics:
                metrics[metric_name] = 0.0
            metrics["had_error"] = 1.0
            return metrics

        # Use first output for per-rep metrics
        per_rep_metrics = self._calculate_per_repetition_metrics(outputs[0], test_case, context_mode)

        # Add aggregated metrics
        aggregated_metrics = self._calculate_aggregated_metrics(outputs)

        # Combine
        metrics = {**per_rep_metrics, **aggregated_metrics}

        return metrics


    def _calculate_single_metric(
        self,
        metric_name: str,
        answer: str,
        contexts: List[str],
        question: str,
        test_case: TestCase,
        all_outputs: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate a single metric using dynamic metric execution.

        This method uses the MetricCalculator's calculate_metric() method,
        which automatically injects the random seed for reproducibility.
        """

        # Special handling for metrics that need preprocessing or special logic
        if metric_name == "output_consistency":
            # Calculate consistency across all repetitions
            # Filter out error outputs before calculating consistency
            valid_answers = [
                out.get("answer", "")
                for out in all_outputs
                if not out.get("error") and not str(out.get("answer", "")).startswith("[ERROR:")
            ]
            if len(valid_answers) < 2:
                return 1.0  # Single valid answer is perfectly consistent
            return self.metrics_calculator.calculate_metric("output_consistency", outputs=valid_answers)

        elif metric_name == "scenario_identification_rate":
            # Extract scenarios from answer and ground truth
            llm_scenarios = self._extract_scenarios_from_answer(answer)
            gt_scenarios = test_case.ground_truth.get("scenarios", [])
            return self.metrics_calculator.calculate_metric(
                "scenario_identification_rate",
                llm_scenarios=llm_scenarios,
                ground_truth_scenarios=gt_scenarios
            )

        elif metric_name == "property_identification_rate":
            # Extract properties from answer and ground truth
            llm_properties = self._extract_properties_from_answer(answer)
            gt_properties = test_case.ground_truth.get("properties", [])
            return self.metrics_calculator.calculate_metric(
                "property_identification_rate",
                llm_properties=llm_properties,
                ground_truth_properties=gt_properties
            )

        elif metric_name == "cer":
            ocr_text = test_case.ground_truth.get("ocr_text", "")
            gt_text = test_case.ground_truth.get("gt_text", "")
            if ocr_text and gt_text:
                return self.metrics_calculator.calculate_metric("cer", ocr_text=ocr_text, gt_text=gt_text)
            return 0.0

        elif metric_name == "wer":
            ocr_text = test_case.ground_truth.get("ocr_text", "")
            gt_text = test_case.ground_truth.get("gt_text", "")
            if ocr_text and gt_text:
                return self.metrics_calculator.calculate_metric("wer", ocr_text=ocr_text, gt_text=gt_text)
            return 0.0

        # Standard metrics with straightforward parameter mapping
        elif metric_name == "answer_correctness":
            return self.metrics_calculator.calculate_metric(
                "answer_correctness",
                llm_answer=answer,
                ground_truth=test_case.ground_truth
            )

        elif metric_name == "citation_coverage":
            return self.metrics_calculator.calculate_metric(
                "citation_coverage",
                answer=answer,
                contexts=contexts
            )

        elif metric_name == "citation_correctness":
            return self.metrics_calculator.calculate_metric(
                "citation_correctness",
                answer=answer,
                contexts=contexts
            )

        elif metric_name == "faithfulness":
            return self.metrics_calculator.calculate_metric(
                "faithfulness",
                answer=answer,
                contexts=contexts
            )

        elif metric_name == "context_precision":
            return self.metrics_calculator.calculate_metric(
                "context_precision",
                contexts=contexts,
                question=question,
                answer=answer
            )

        elif metric_name == "context_recall":
            return self.metrics_calculator.calculate_metric(
                "context_recall",
                contexts=contexts,
                question=question,
                answer=answer
            )

        elif metric_name == "fkgl":
            return self.metrics_calculator.calculate_metric("fkgl", text=answer)

        elif metric_name == "hallucination_rate":
            # Hallucination can only be computed when contexts exist
            if not contexts:
                return float("nan")

            # Use configurable thresholds from ExperimentConfig
            return float(self.metrics_calculator.calculate_metric(
                "hallucination_rate",
                answer=answer,
                contexts=contexts,
                f_thresh=self.config.hallucination_faithfulness_thresh,
                c_thresh=self.config.hallucination_citation_thresh
            ))

        elif metric_name == "ml_problem_type_accuracy":
            # ML recommendation: problem type identification accuracy
            return self.metrics_calculator.calculate_metric(
                "ml_problem_type_accuracy",
                answer=answer,
                ground_truth=test_case.ground_truth
            )

        elif metric_name == "algorithm_suitability":
            # ML recommendation: algorithm suitability (TFR)
            return self.metrics_calculator.calculate_metric(
                "algorithm_suitability",
                answer=answer,
                ground_truth=test_case.ground_truth
            )

        else:
            # Unknown metric - return 0.0
            print(f"Warning: Unknown metric '{metric_name}'. Returning 0.0")
            return 0.0

    def _extract_scenarios_from_answer(self, answer: str) -> List[str]:
        """Extract repurposing scenarios from LLM answer"""
        # Look for "Component | Target System" format
        lines = answer.split('\n')
        scenarios = []
        for line in lines:
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    target = parts[1].strip()
                    if target and not target.lower().startswith('target'):
                        scenarios.append(target)
        return scenarios

    def _extract_properties_from_answer(self, answer: str) -> List[str]:
        """Extract technical properties from LLM answer"""
        # Look for "Property: Value" format
        lines = answer.split('\n')
        properties = []
        for line in lines:
            if ':' in line:
                prop = line.split(':')[0].strip()
                if prop and len(prop.split()) <= 5:  # Reasonable property name
                    properties.append(prop)
        return properties

    def _check_output_schema(
        self,
        answer: str,
        required_sections: Optional[List[str]] = None
    ) -> tuple[bool, str]:
        """
        FAIRNESS: Validate that output follows required schema across all conditions.

        WARNING: This is DISABLED BY DEFAULT (config.enforce_output_schema=False) because
        the hardcoded schema does NOT match the default diagnosis prompts!

        Default required sections (if enabled):
        - Diagnosis
        - Root Cause
        - Steps
        - Safety
        - Citations

        Compatible prompts: NONE of the built-in prompts! You must use custom prompts.
        - DIAGNOSIS_BASELINE asks for: Diagnosis, Recommended Action, Tools & Parts, Safety Warnings
        - DIAGNOSIS_BASELINE_V1 asks for: Diagnosis, Root Cause, Repair Procedure, Required Materials, Safety
        - etc.

        To use schema validation:
        1. Create custom prompts that request the exact sections above
        2. Set config.enforce_output_schema = True
        3. OR modify required_sections to match your prompts

        Args:
            answer: LLM answer text
            required_sections: List of required section headers (uses default if None)

        Returns:
            Tuple of (passed: bool, error_message: str)
        """
        if required_sections is None:
            # Default schema for diagnosis task
            required_sections = ["Diagnosis", "Root Cause", "Steps", "Safety", "Citations"]

        # Skip schema check for error outputs
        if answer.startswith("[ERROR:"):
            return True, ""

        missing_sections = []
        for section in required_sections:
            # Case-insensitive search for section headers
            # Matches "## Diagnosis", "**Diagnosis:**", "Diagnosis:", etc.
            pattern = rf'(?:^|\n)(?:#+\s*)?(?:\*\*)?{re.escape(section)}(?:\*\*)?(?::|$)'
            if not re.search(pattern, answer, re.IGNORECASE):
                missing_sections.append(section)

        if missing_sections:
            return False, f"Missing required sections: {', '.join(missing_sections)}"

        return True, ""

    def _sanity_check_contexts(
            self,
            outputs: List[Dict[str, Any]],
            context_mode: ContextMode,
            top_k: int
        ) -> tuple[bool, str]:
            """
            Sanity check: validate that contexts match expected counts for context_mode.

            Rules:
            - BASELINE: len(contexts) == 0
            - MANUAL_FULL: len(contexts) > 0
            - RAG_RETRIEVAL: 0 < len(contexts) <= top_k * (2 * window_size + 1) * 1.2

            Args:
                outputs: List of LLM outputs from repetitions
                context_mode: Context mode used
                top_k: Top-k parameter for RAG retrieval

            Returns:
                Tuple of (passed: bool, error_message: str)
            """
            if not outputs:
                return False, "No outputs to check"

            # Check first output (representative)
            first_output = outputs[0]

            # Skip check if there was an error in the output
            if first_output.get("error") or str(first_output.get("answer", "")).startswith("[ERROR:"):
                return True, ""  # Don't fail sanity check for error cases

            contexts = first_output.get("contexts", [])

            # Handle string contexts (legacy)
            if isinstance(contexts, str):
                try:
                    import json
                    contexts = json.loads(contexts) if contexts else []
                except json.JSONDecodeError:
                    contexts = []

            num_contexts = len(contexts)

            if context_mode == ContextMode.BASELINE:
                if num_contexts != 0:
                    return False, f"BASELINE should have 0 contexts, got {num_contexts}"

            elif context_mode == ContextMode.MANUAL_FULL:
                if num_contexts == 0:
                    return False, f"MANUAL_FULL should have >0 contexts, got 0"

            elif context_mode == ContextMode.RAG_RETRIEVAL:
                if num_contexts == 0:
                    return False, f"RAG_RETRIEVAL should have >0 contexts, got 0"

                # Calculate bounds accounting for overlap removal
                window_size = getattr(self.config, 'retrieval_window_size', 0)

                # Upper bound: No overlap case (all windows are separate)
                upper_bound = top_k * (2 * window_size + 1)

                # Lower bound: At minimum, we always get top_k chunks
                lower_bound = top_k

                # Check upper bound
                if num_contexts > upper_bound:
                    return False, (
                        f"RAG_RETRIEVAL should have <={upper_bound} contexts "
                        f"(top_k={top_k}, window_size={window_size}, max case with no overlap), "
                        f"got {num_contexts}"
                    )

                # Check lower bound
                if num_contexts < lower_bound:
                    return False, (
                        f"RAG_RETRIEVAL should have at least {lower_bound} contexts "
                        f"(top_k={top_k}), got {num_contexts}"
                    )

            return True, ""

    def _aggregate_results(self) -> ExperimentResults:
        """
        Aggregate results and create analysis tables.

        Returns:
            ExperimentResults with analysis tables
        """
        # CRITICAL: Pass metrics list as ALLOWLIST for aggregation
        # Only columns in config.metrics will be averaged in performance tables
        return ExperimentResults(self.results, metrics=self.config.metrics)


# ================= Helper Functions =================

def run_quick_experiment(
    task_type: str,
    test_cases: List[TestCase],
    model: LLMProvider = LLMProvider.GEMINI,
    llm_service: Optional[LLMService] = None,
    metrics: Optional[List[str]] = None
) -> ExperimentResults:
    """
    Quick experiment runner for testing.

    Args:
        task_type: Type of task
        test_cases: List of test cases
        model: LLM provider to use
        llm_service: LLM service instance (creates default if None)
        metrics: List of metrics (uses defaults if None)

    Returns:
        ExperimentResults
    """
    # Create default LLM service if not provided
    if llm_service is None:
        llm_service = LLMService()

    # Use default metrics for task type if not provided
    if metrics is None:
        metrics = MetricsConfig.get_metrics_for_task(task_type)

    # Create config
    config = ExperimentConfig(
        task_type=task_type,
        models=[model],
        metrics=metrics,
        n_repetitions=1,
        include_baseline=False
    )

    # Create dataset
    from .dataset import Dataset
    dataset = Dataset(test_cases)

    # Run experiment
    runner = ExperimentRunner(config, llm_service)
    return runner.run(dataset, verbose=True)
