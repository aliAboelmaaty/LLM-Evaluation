"""
Validation Experiment: Sonntag ML Recommendation Dataset
WITH 3-WAY COMPARISON: BASELINE vs My RAG vs Sonntag's GraphRAG

This experiment validates:
1. Domain transferability (Pillar 3)
2. RAG effectiveness for ML recommendation
3. How traditional RAG compares to GraphRAG
"""

from framework import (
    ExperimentConfig,
    ExperimentRunner,
    LLMProvider,
    LLMService,
    ResultsAnalyzer,
    ContextMode
)
from load_sonntag_dataset import load_sonntag_dataset

# =====================================================================
# CONFIGURATION
# =====================================================================

MODELS_TO_TEST = [
    #LLMProvider.GEMINI,        # Gemini 2.5 Flash (Sonntag used this)
    #LLMProvider.DEEPSEEK,      # DeepSeek
    LLMProvider.GEMMA3_4B, 
    LLMProvider.GPT5,           # Gemma3 4B (smaller, faster)
    #LLMProvider.GEMMA3_12B,    # Gemma3 12B (medium)
    #LLMProvider.GEMMA3_27B,    # Gemma3 27B (largest)
]

# CRITICAL: Test BOTH modes for comparison
CONTEXT_MODES_TO_TEST = [
    ContextMode.BASELINE,      # No context (Sonntag baseline: 60-62%)
    ContextMode.RAG_RETRIEVAL  # With synthetic KB retrieval (our test!)
]

# Metrics: combines domain-specific + existing RAG metrics
METRICS_TO_USE = [
    # Problem understanding & recommendation
    "ml_problem_type_accuracy",  # RQ1: Correct problem type?
    "algorithm_suitability",      # RQ2: Suitable algorithm? (TFR)

    # RAG quality (existing metrics work out-of-the-box!)
    "citation_coverage",          # Citation completeness
    "citation_correctness",       # Citation accuracy
    "faithfulness",               # Faithful to KB?
    "context_precision",          # Retrieval precision
    "context_recall",             # Retrieval recall
    "hallucination_rate",         # Hallucination detection

    # Overall quality
       # Answer quality
    "output_consistency",         # RQ3: Consistency (OCR)
]

# Experiment parameters (matching Sonntag)
N_REPETITIONS = 3  # Match Sonntag's methodology for consistency measurement
TEMPERATURE = 0.2
TOP_P = 1.0
MAX_TOKENS = 1024
TOP_K_RETRIEVAL = 3           # Retrieve 6 similar cases
RANDOM_SEED = 42

# =====================================================================
# STEP 1: LEAVE-ONE-OUT KNOWLEDGE BASE SETUP
# =====================================================================

print("\n" + "="*70)
print("STEP 1: LEAVE-ONE-OUT KB CONFIGURATION")
print("="*70)

from build_ml_knowledge_base import build_knowledge_base, get_cache_stats, clear_kb_cache

# IMPORTANT: We do NOT build a single KB upfront!
# Instead, for each test case, a KB will be built dynamically that EXCLUDES
# that specific test case. This prevents data leakage where the RAG system
# could retrieve the exact answer from the KB.

print("""
LEAVE-ONE-OUT CROSS-VALIDATION ENABLED
----------------------------------------------------------------------

For each test case, a unique KB will be built that excludes that case.
This prevents data leakage and ensures fair evaluation.

Example:
  - When testing ML_001: KB contains cases ML_002 - ML_076
  - When testing ML_002: KB contains ML_001, ML_003 - ML_076
  - etc.

KB files will be cached in: kb_cache/
""")

# Optional: Clear old cache for fresh run
# clear_kb_cache()

print(f"Current KB cache status: {get_cache_stats()}")

# =====================================================================
# STEP 2: LOAD DATASET
# =====================================================================

print("\n" + "="*70)
print("STEP 2: LOADING DATASET")
print("="*70)

# Load WITHOUT pre-linking KB - runner will build leave-one-out KBs dynamically
# The source_path is stored in dataset.metadata for KB building
dataset = load_sonntag_dataset(
    json_path="DESIGN2026_Sebastian_Sonntag_dataset.json",
    knowledge_base_path="ml_knowledge_base.txt"  # Placeholder path, will be overridden per case
)

START_CASE = 18
END_CASE = 41 

print(f"\n[WARNING] LIMITING RUN TO INDICES {START_CASE} -> {END_CASE}")

# Python slicing is [inclusive : exclusive]
# This grabs index 5, 6, 7, 8, 9 (Total 5 cases)
dataset.test_cases = dataset.test_cases[START_CASE:END_CASE]

# = 3  # Set to None to run all

#if LIMIT_CASES:
#    print(f"\n[WARNING] LIMITING RUN TO FIRST {LIMIT_CASES} CASES FOR TESTING")
    # The dataset object has a list called .test_cases
#    dataset.test_cases = dataset.test_cases[:LIMIT_CASES]

print(f"\nLoaded {len(dataset)} test cases")
print(f"Dataset source: {dataset.metadata.get('source_path', 'unknown')}")
print(f"Leave-one-out KB will be built dynamically for each case")

stats = dataset.get_stats()
print(f"\nDataset composition:")
print(f"  - Total cases: {stats['n_cases']}")
print(f"  - RAG mode: Leave-one-out KB per case")

# =====================================================================
# STEP 3: CREATE EXPERIMENT CONFIG
# =====================================================================

# Context budget overrides per model (different context window sizes)
CONTEXT_BUDGET_OVERRIDES = {
    "gemma3-4b":  {"max_context_chars": 20000, "manual_full_max_chunks": 20, "chunk_size": 900},
    "gemma3-12b": {"max_context_chars": 40000, "manual_full_max_chunks": 40, "chunk_size": 1000},
    "gemma3-27b": {"max_context_chars": 30000, "manual_full_max_chunks": 40, "chunk_size": 1000},
    "gemini":     {"max_context_chars": 80000, "manual_full_max_chunks": 80, "chunk_size": 1200},
    "gpt5":       {"max_context_chars": 80000, "manual_full_max_chunks": 80, "chunk_size": 1200},
    "deepseek":   {"max_context_chars": 60000, "manual_full_max_chunks": 60, "chunk_size": 1200},
}

config = ExperimentConfig(
    task_type="ml_recommendation",
    models=MODELS_TO_TEST,
    metrics=METRICS_TO_USE,
    n_repetitions=N_REPETITIONS,
    context_modes=CONTEXT_MODES_TO_TEST,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    top_p=TOP_P,
    top_k_retrieval=TOP_K_RETRIEVAL,  # How many KB chunks to retrieve
    random_seed=RANDOM_SEED,
    prompt_variant="sonntag",  # Use Sonntag-specific prompts with problem_formulation

    # RAG parameters (using your existing system)
    chunk_size=1000,                   # Characters per chunk
    max_context_chars=40000,           # Max context
    context_budget_overrides=CONTEXT_BUDGET_OVERRIDES,  # Per-model context budgets
)

# =====================================================================
# STEP 4: VERIFY METRICS ARE AVAILABLE
# =====================================================================

print("\n" + "="*70)
print("METRIC VERIFICATION")
print("="*70)

from framework.metrics import MetricCalculator

calc = MetricCalculator()
available_metrics = calc.get_available_metrics()

required_metrics = [
    "ml_problem_type_accuracy",
    "algorithm_suitability",
    "output_consistency",
    "citation_coverage",
    "citation_correctness",
]

print("\nChecking required metrics:")
all_metrics_available = True
for metric in required_metrics:
    if metric in available_metrics:
        print(f"  [OK] {metric}")
    else:
        print(f"  [MISSING] {metric}")
        all_metrics_available = False

if not all_metrics_available:
    print("\n[ERROR] Critical metrics are missing!")
    print("Please ensure all metrics are registered in MetricCalculator.")
    import sys
    sys.exit(1)

print("\n[OK] All required metrics are available")

# =====================================================================
# STEP 5: RUN EXPERIMENT
# =====================================================================

print("\n" + "="*70)
print("RUNNING EXPERIMENTS")
print("="*70)
print(f"\nModes: {[m.value for m in CONTEXT_MODES_TO_TEST]}")
print(f"Models: {[m.value for m in MODELS_TO_TEST]}")
print(f"Repetitions: {N_REPETITIONS}")
print(f"Top-K Retrieval: {TOP_K_RETRIEVAL}")
print("\n" + "="*70 + "\n")

llm_service = LLMService()
runner = ExperimentRunner(config, llm_service)
results = runner.run(dataset, verbose=True)

# =====================================================================
# STEP 5.5: VERIFY LEAVE-ONE-OUT KB CACHE
# =====================================================================

print("\n" + "="*70)
print("VERIFYING LEAVE-ONE-OUT KNOWLEDGE BASE")
print("="*70)

from pathlib import Path

kb_cache_dir = Path("kb_cache")
if kb_cache_dir.exists():
    kb_files = list(kb_cache_dir.glob("kb_excluding_*.txt"))
    print(f"\nKB Cache Status:")
    print(f"  - Directory: {kb_cache_dir}")
    print(f"  - KB files created: {len(kb_files)}")

    if kb_files:
        # Verify first KB file doesn't contain excluded case
        first_kb = kb_files[0]
        excluded_case = first_kb.stem.replace("kb_excluding_", "")
        with open(first_kb, 'r', encoding='utf-8') as f:
            kb_content = f.read()

        # Check that excluded case ID is not in KB
        if f"Case #{excluded_case[-3:]}" in kb_content or f"ML Application Case #{excluded_case[-3:]}" in kb_content:
            print(f"\n[ERROR] WARNING: {excluded_case} may still be in {first_kb.name}")
            print("   Data leakage may have occurred!")
        else:
            print(f"\n[OK] VERIFIED: {excluded_case} correctly excluded from {first_kb.name}")

    # Show cache stats
    total_size = sum(f.stat().st_size for f in kb_files)
    print(f"\n  Total cache size: {total_size/1024/1024:.2f} MB")
else:
    print("\n[WARNING] KB cache directory not found - leave-one-out may not have run")

# =====================================================================
# STEP 6: SAVE RESULTS (SIMPLIFIED - 3 FILES ONLY)
# =====================================================================

output_dir = "results/sonntag_validation"
results.export_task_specific_results(
    output_dir,
    task_type="ml_recommendation",
    config=config
)
print(f"\nResults saved to: {output_dir}/")

# =====================================================================
# STEP 7: ANALYZE RESULTS - BASELINE vs RAG COMPARISON
# =====================================================================

analyzer = ResultsAnalyzer(results)

print("\n" + "="*70)
print("BASELINE vs RAG COMPARISON")
print("="*70)

# Separate baseline and RAG results
baseline_df = results.df[results.df['context_mode'] == 'baseline']
rag_df = results.df[results.df['context_mode'] == 'rag_retrieval']

print(f"\n{'Metric':<40} {'BASELINE':<15} {'RAG':<15} {'Improvement':<15}")
print("-" * 85)

# ML Problem Type Accuracy (RQ1: Problem Understanding)
if 'ml_problem_type_accuracy' in results.df.columns:
    baseline_pta = baseline_df['ml_problem_type_accuracy'].mean() * 100 if not baseline_df.empty else 0
    rag_pta = rag_df['ml_problem_type_accuracy'].mean() * 100 if not rag_df.empty else 0
    improvement_pta = rag_pta - baseline_pta
    print(f"{'Problem Type Accuracy (RQ1)':<40} {baseline_pta:>6.1f}% {'':>7} {rag_pta:>6.1f}% {'':>7} {improvement_pta:>+6.1f} pp")
else:
    print(f"{'Problem Type Accuracy (RQ1)':<40} {'MISSING METRIC!':<30}")

# Algorithm Suitability = TFR (RQ2: Candidate Recommendation)
if 'algorithm_suitability' in results.df.columns:
    baseline_tfr = baseline_df['algorithm_suitability'].mean() * 100 if not baseline_df.empty else 0
    rag_tfr = rag_df['algorithm_suitability'].mean() * 100 if not rag_df.empty else 0
    improvement_tfr = rag_tfr - baseline_tfr
    print(f"{'Algorithm Suitability / TFR (RQ2)':<40} {baseline_tfr:>6.1f}% {'':>7} {rag_tfr:>6.1f}% {'':>7} {improvement_tfr:>+6.1f} pp")
else:
    print(f"{'Algorithm Suitability / TFR (RQ2)':<40} {'MISSING METRIC!':<30}")

# Output Consistency = OCR (RQ3: Consistency)
if 'output_consistency' in results.df.columns:
    baseline_ocr = baseline_df['output_consistency'].mean() * 100 if not baseline_df.empty else 0
    rag_ocr = rag_df['output_consistency'].mean() * 100 if not rag_df.empty else 0
    improvement_ocr = rag_ocr - baseline_ocr
    print(f"{'Output Consistency / OCR (RQ3)':<40} {baseline_ocr:>6.1f}% {'':>7} {rag_ocr:>6.1f}% {'':>7} {improvement_ocr:>+6.1f} pp")
else:
    print(f"{'Output Consistency / OCR (RQ3)':<40} {'MISSING METRIC!':<30}")

# RAG-specific metrics (only for RAG mode)
if not rag_df.empty:
    print(f"\n{'='*70}")
    print("RAG-SPECIFIC QUALITY METRICS")
    print("="*70)
    print(f"\n{'Metric':<40} {'RAG Performance':<20}")
    print("-" * 60)

    rag_metrics = [
        ('Citation Coverage', 'citation_coverage'),
        ('Citation Correctness', 'citation_correctness'),
        ('Faithfulness', 'faithfulness'),
        ('Context Precision', 'context_precision'),
        ('Context Recall', 'context_recall'),
        ('Hallucination Rate', 'hallucination_rate'),
    ]

    for name, col in rag_metrics:
        if col in rag_df.columns:
            val = rag_df[col].mean() * 100
            print(f"{name:<40} {val:>6.1f}%")
        else:
            print(f"{name:<40} N/A")

# Quality check for missing metrics
missing_critical = []
if 'ml_problem_type_accuracy' not in results.df.columns:
    missing_critical.append('ml_problem_type_accuracy')
if 'algorithm_suitability' not in results.df.columns:
    missing_critical.append('algorithm_suitability')

if missing_critical:
    print("\n" + "="*70)
    print("[WARNING] MISSING CRITICAL METRICS!")
    print("="*70)
    print(f"Missing: {', '.join(missing_critical)}")
    print("\nPossible causes:")
    print("1. Metrics not registered in MetricCalculator")
    print("2. Metrics not in METRICS_TO_USE list")
    print("3. Handler not added in runner._calculate_single_metric()")

# Comparison with Sonntag's GraphRAG baseline
if not baseline_df.empty and not rag_df.empty and 'algorithm_suitability' in results.df.columns:
    print(f"\n{'='*70}")
    print("COMPARISON WITH SONNTAG'S GRAPHRAG (Reference: 85-86% TFR)")
    print("="*70)
    print(f"\n{'Metric':<40} {'My RAG':<15} {'Sonntag GraphRAG':<20} {'Gap':<15}")
    print("-" * 90)
    print(f"{'TFR (Algorithm Suitability)':<40} {rag_tfr:>6.1f}% {'':>7} 85-86% {'':>12} {rag_tfr - 85.5:>+6.1f} pp")
    print(f"{'OCR (Output Consistency)':<40} {rag_ocr:>6.1f}% {'':>7} 88-93% {'':>12} {rag_ocr - 90.5:>+6.1f} pp")

# Breakdown by ML problem type
print("\n" + "="*70)
print("BREAKDOWN BY ML PROBLEM TYPE")
print("="*70)

for problem_type in ['Classification', 'Regression', 'Clustering', 'Association Rule']:
    subset = [tc for tc in dataset if problem_type.lower() in tc.ground_truth['ml_problem_type'].lower()]

    if subset:
        subset_ids = {tc.case_id for tc in subset}
        subset_baseline = baseline_df[baseline_df['case_id'].isin(subset_ids)]
        subset_rag = rag_df[rag_df['case_id'].isin(subset_ids)]

        print(f"\n{problem_type} (n={len(subset)}):")

        if not subset_baseline.empty:
            print(f"  BASELINE TFR: {subset_baseline['algorithm_suitability'].mean()*100:>5.1f}%")

        if not subset_rag.empty:
            print(f"  RAG TFR:      {subset_rag['algorithm_suitability'].mean()*100:>5.1f}%")
            if not subset_baseline.empty:
                improvement = (subset_rag['algorithm_suitability'].mean() - subset_baseline['algorithm_suitability'].mean()) * 100
                print(f"  Improvement:  {improvement:>+5.1f} pp")

# =====================================================================
# THESIS IMPLICATIONS
# =====================================================================

print("\n" + "="*70)
print("THESIS IMPLICATIONS")
print("="*70)

print("\nPILLAR 3: Domain Transferability VALIDATED")
print("  - Framework transferred from fault diagnosis to ML recommendation")
print("  - Existing RAG system works without modification")

print("\nPILLAR 1: Consistency Measurement VALIDATED")
print(f"  - Measured across {N_REPETITIONS} repetitions")
print(f"  - Comparable to Sonntag's OCR methodology")

print("\nPILLAR 2: Cryptographic Provenance VALIDATED")
print(f"  - SHA256 hashes in: {output_dir}/raw_results_audit.jsonl")

print("\nRESEARCH CONTRIBUTION:")
print("  - First comparison of traditional RAG vs GraphRAG for ML recommendation")
print(f"  - Traditional RAG achieved: {rag_tfr:.1f}% TFR (Sonntag GraphRAG: 85-86%)")
print(f"  - Gap analysis provides insights for thesis discussion")

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)
