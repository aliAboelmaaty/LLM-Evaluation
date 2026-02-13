from framework import (
    TestCase,
    Dataset,
    ExperimentConfig,
    ExperimentRunner,
    LLMProvider,
    LLMService,
    ResultsAnalyzer,
    ContextMode
)


# =====================================================================
# QUICK START EXAMPLES
# =====================================================================

# =====================================================================
# STEP 1: CONFIGURE YOUR EXPERIMENT
# =====================================================================

# ========================= CHOOSE YOUR MODELS =========================
# Uncomment the models you want to test. You can test multiple models at once.
# Note: Make sure you have the required API keys in your .env file
MODELS_TO_TEST = [
    #LLMProvider.GEMINI,           # Google Gemini (requires GEMINI_API_KEY)
    LLMProvider.GPT5,             # GPT-5 (via OpenAI API, requires OPENAI_API_KEY)
    #LLMProvider.DEEPSEEK,         # DeepSeek model
    LLMProvider.GEMMA3_4B,          # Gemma 3 4B - Fast, good for quick tests
    #LLMProvider.GEMMA3_12B,       # Gemma 3 12B - Balanced performance
    ##LLMProvider.GEMMA3_27B,       # Gemma 3 27B - Best quality, slower

]

# ========================= CHOOSE YOUR CONTEXT MODES =========================
# This is NEW! You can now specify exactly which modes to test:
#
# Option 0: Test baseline 
#CONTEXT_MODES_TO_TEST = [ContextMode.BASELINE]

# Option 1: Test baseline vs RAG (most common for papers)
#CONTEXT_MODES_TO_TEST = [ContextMode.BASELINE, ContextMode.RAG_RETRIEVAL]

# Option 2: Test all three modes (comprehensive comparison)
CONTEXT_MODES_TO_TEST = [ContextMode.BASELINE, ContextMode.MANUAL_FULL, ContextMode.RAG_RETRIEVAL]
#
# Option 3: Only test with manual (skip baseline)
#CONTEXT_MODES_TO_TEST = [ContextMode.RAG_RETRIEVAL]
#
# Option 4: Compare full manual vs RAG
# CONTEXT_MODES_TO_TEST = [ContextMode.MANUAL_FULL, ContextMode.RAG_RETRIEVAL]

# ========================= CHOOSE YOUR METRICS =========================
# Select which metrics to calculate. RAG-specific metrics (context_precision,
# context_recall, faithfulness, etc.) will be NaN for BASELINE mode.
METRICS_TO_USE = [
    # Core metrics (work in all modes)
    "answer_correctness",        # How correct is the diagnosis? (0-1)

    # RAG-specific metrics (only for MANUAL_FULL and RAG_RETRIEVAL)
    "hallucination_rate",        # Rate of hallucinations (0-1, lower is better)
    "faithfulness",              # Faithfulness to manual (0-1)
    "citation_coverage",         # How well citations cover the answer (0-1)
    "citation_correctness",      # Are citations correct? (0-1)
    "context_precision",         # Precision of retrieved contexts (0-1)
    "context_recall",            # Recall of retrieved contexts (0-1)

    # Readability (works in all modes)
    "fkgl",                      # Flesch-Kincaid Grade Level (readability)

    # Consistency metric (requires n_repetitions > 1)
    "output_consistency"         # Consistency across repetitions (0-1, 1.0 = identical)
]

# ========================= PROMPT VARIANT =========================
# Different prompt styles for different needs
PROMPT_VARIANT = "concise"
# Options:
#   None or "default" - Standard diagnosis prompt
#   "v1" - Version 1 prompt
#   "concise" - Shorter, more focused prompt
#   "safety_focused" - Emphasizes safety considerations

# ========================= EXPERIMENT SETTINGS =========================
N_REPETITIONS = 3           # How many times to run each test (for consistency analysis)
                            # Set to 5-10 for production experiments to measure output_consistency

TEMPERATURE = 0.2           # 0.0 = deterministic, 1.0 = creative
                            # Recommended: 0.1-0.3 for factual tasks like diagnosis

TOP_P = .5                 # Nucleus sampling (1.0 = disabled, 0.9 = typical, 0.5 = very focused)
                            # Lower = more focused/deterministic output
                            # Try 0.9 for more consistent answers, 0.5 for very deterministic

MAX_TOKENS = 2048           # Maximum response length
                            # 2048 is usually enough for diagnosis answers
BASELINE_MAX_TOKENS = None   # Maximum response length for BASELINE
                            

TOP_K_RETRIEVAL = 10         # Number of chunks to retrieve in RAG_RETRIEVAL mode
                            # Try 3-10 depending on your manual size and query complexity

# ========================= CONTEXT BUDGET SETTINGS =========================
# Adaptive context budgeting per model/provider to prevent oversized prompts
CHUNK_SIZE = 1000                    # Default chunk size in characters
MAX_CONTEXT_CHARS = 40000            # Default maximum context characters
MANUAL_FULL_MAX_CHUNKS = 40          # Default maximum chunks for MANUAL_FULL mode

# Per-provider budget overrides (key must match provider.value)
CONTEXT_BUDGET_OVERRIDES = {
    "gemma3-4b":  {"max_context_chars": 20000, "manual_full_max_chunks": 20, "chunk_size": 900},
    "gemma3-12b": {"max_context_chars": 40000, "manual_full_max_chunks": 40, "chunk_size": 1000},
    "gemma3-27b": {"max_context_chars": 30000, "manual_full_max_chunks": 40, "chunk_size": 1000},
    "gemini":     {"max_context_chars": 80000, "manual_full_max_chunks": 80, "chunk_size": 1200},
    "gpt5":       {"max_context_chars": 80000, "manual_full_max_chunks": 80, "chunk_size": 1200},
    "deepseek":   {"max_context_chars": 60000, "manual_full_max_chunks": 60, "chunk_size": 1200},
}

# DEPRECATED (for backward compatibility only)
INCLUDE_BASELINE = None     # Use CONTEXT_MODES_TO_TEST instead


# =====================================================================
# STEP 2: CREATE YOUR TEST CASES
# =====================================================================

def create_test_cases():
    """
    Create your diagnosis test cases.

    IMPORTANT FEATURES:
    ==================
    1. Each test case can have its OWN PDF manual!
       - Different appliances can have different manuals
       - The framework automatically handles per-case PDFs

    2. Test Case Structure:
       - case_id: Unique identifier (e.g., "1", "COFFEE_001", etc.)
       - input_data: Dict with keys the LLM will receive
           * fault_description: The problem description
           * appliance: The appliance name
       - ground_truth: Dict with expected outputs for evaluation
           * diagnosis: Expected diagnosis text
           * page_reference: Optional page reference for validation
       - metadata: Additional info for analysis
           * complexity: 1=easy, 2=medium, 3=hard (for complexity analysis)
           * category: Optional category tag
       - context_document: Path to PDF manual (use raw strings: r"path\to\file.pdf")

    3. Slicing Test Cases:
       You can select a subset of test cases for quick testing:
       - test_cases[0:3]  -> First 3 cases
       - test_cases[3:6]  -> Cases 4-6 (0-indexed)
       - test_cases       -> All cases

    4. Ground Truth for Metrics:
       The ground_truth dict is used to calculate answer_correctness.
       The framework compares the LLM's diagnosis with your expected diagnosis.
    """

    test_cases = [
        # ===== EXAMPLE 1: Bosch Dishwasher =====
          TestCase(
            case_id="1",
            input_data={
                "fault_description": (
                    "Espresso machine stopped during use. All indicator lights are off and "
                    "the machine no longer responds to the front controls."
                ),
                "appliance": "Dualit Espresso Coffee Machine"
            },
            ground_truth={
                # CHANGED: "Machine not working" (Symptom) -> "Fuse/Cut-out" (Root Cause)
                "diagnosis": "The fuse in the plug has failed or the thermal safety cut-out has activated",
                "page_reference": "p. 29" 
            },
            metadata={
                "complexity": 1,
                "category": "no_power"
            },
            context_document="DataSet/coffee_machine/espresso_coffee_machine.pdf"
        ),

        # Case 2: Steam Wand Blocked
        TestCase(
            case_id="2",
            input_data={
                "fault_description": (
                    "No hot water or steam comes out of the steam wand, or steam output is very weak "
                    "after frothing milk."
                ),
                "appliance": "Dualit Espresso Coffee Machine"
            },
            ground_truth={
                # CHANGED: "No steam" (Symptom) -> "Blockage" (Root Cause)
                "diagnosis": "Steam wand is blocked by dried milk or limescale residue",
                "page_reference": "p. 29"
            },
            metadata={
                "complexity": 1,
                "category": "steam_system"
            },
            context_document="DataSet/coffee_machine/espresso_coffee_machine.pdf"
        ),

        # Case 3: Poor Extraction (No Crema)
        TestCase(
            case_id="3",
            input_data={
                "fault_description": (
                    "Espresso has little or no crema, tastes weak and flat even when good beans are used."
                ),
                "appliance": "Dualit Espresso Coffee Machine"
            },
            ground_truth={
                # CHANGED: "No crema" (Symptom) -> "Grind/Freshness" (Root Cause)
                "diagnosis": "Coffee grind is too coarse or coffee beans are not fresh",
                "page_reference": "p. 30"
            },
            metadata={
                "complexity": 1,
                "category": "extraction_quality"
            },
            context_document="DataSet/coffee_machine/espresso_coffee_machine.pdf"
        ),

        # ==============================================================================
        # MANUAL: Cecotec Coffee 66 Grind & Drop (coffee_66_grinddrop.pdf)
        # ==============================================================================

        # Case 4: Temperature Issues
        TestCase(
            case_id="4",
            input_data={
                "fault_description": (
                    "Brewed coffee is only lukewarm; even immediately after brewing it is not properly hot."
                ),
                "appliance": "Cecotec Coffee 66 Grind & Drop"
            },
            ground_truth={
                # CHANGED: Root cause is often scale or cold cup
                "diagnosis": "Limescale build-up in thermoblock or cup/jug was not pre-heated",
                "page_reference": "p. 50"
            },
            metadata={
                "complexity": 1,
                "category": "temperature"
            },
            context_document="DataSet/coffee_machine/coffee_66_grinddrop.pdf"
        ),

        # Case 5: Flow Rate / Descaling
        TestCase(
            case_id="5",
            input_data={
                "fault_description": (
                    "During brewing, coffee drips very slowly into the jug and sometimes stops briefly, "
                    "although the tank is full."
                ),
                "appliance": "Cecotec Coffee 66 Grind & Drop"
            },
            ground_truth={
                # CHANGED: Explicitly mention Descaling
                "diagnosis": "Appliance needs descaling due to limescale accumulation",
                "page_reference": "p. 50"
            },
            metadata={
                "complexity": 1,
                "category": "flow_rate"
            },
            context_document="DataSet/coffee_machine/coffee_66_grinddrop.pdf"
        ),

        # Case 6: Grinder Blockage
        TestCase(
            case_id="6",
            input_data={
                "fault_description": (
                    "When starting a cycle with beans, the grinder runs but no ground coffee falls into the filter."
                ),
                "appliance": "Cecotec Coffee 66 Grind & Drop"
            },
            ground_truth={
                # CHANGED: Specific grinder channel issue
                "diagnosis": "Ground coffee output channel is blocked and needs cleaning with brush",
                "page_reference": "p. 51"
            },
            metadata={
                "complexity": 2,
                "category": "grinder"
            },
            context_document="DataSet/coffee_machine/coffee_66_grinddrop.pdf"
        ),

        # ==============================================================================
        # MANUAL: Cecotec Coffee 66 Drop & Thermo Time (coffee_66_drop__thermo_time.pdf)
        # ==============================================================================

        # Case 7: Descaling Indicator
        TestCase(
            case_id="7",
            input_data={
                "fault_description": (
                    "Over time the brewing cycle has become much slower. The self-cleaning indicator "
                    "starts flashing and coffee takes a long time to drip."
                ),
                "appliance": "Cecotec Coffee 66 Drop & Thermo Time"
            },
            ground_truth={
                "diagnosis": "Internal pipes blocked by limescale; descaling cycle required",
                "page_reference": "p. 40-41"
            },
            metadata={
                "complexity": 2,
                "category": "limescale_blockage"
            },
            context_document="DataSet/coffee_machine/coffee_66_drop__thermo_time.pdf"
        ),

        # Case 8: Filter Overflow
        TestCase(
            case_id="8",
            input_data={
                "fault_description": (
                    "During brewing, coffee or hot water spills or overflows from the filter area "
                    "instead of cleanly dripping into the thermal carafe."
                ),
                "appliance": "Cecotec Coffee 66 Drop & Thermo Time"
            },
            ground_truth={
                # CHANGED: Mechanical seating issue
                "diagnosis": "Filter holder not correctly seated or carafe not pushing anti-drip valve open",
                "page_reference": "p. 39"
            },
            metadata={
                "complexity": 1,
                "category": "overflow"
            },
            context_document="DataSet/coffee_machine/coffee_66_drop__thermo_time.pdf"
        ),

        # Case 9: No Flow (Priming/Blockage)
        TestCase(
            case_id="9",
            input_data={
                "fault_description": (
                    "The machine heats and makes noise but no water comes out of the filter basket, "
                    "or only a few drops appear."
                ),
                "appliance": "Cecotec Coffee 66 Drop & Thermo Time"
            },
            ground_truth={
                # CHANGED: Limescale is the standard manual answer for this
                "diagnosis": "Severe limescale blockage or airlock in pump system",
                "page_reference": "p. 39"
            },
            metadata={
                "complexity": 2,
                "category": "no_flow"
            },
            context_document="DataSet/coffee_machine/coffee_66_drop__thermo_time.pdf"
        ),

        # ===================== 10–12: FF255 – Side-by-side refrigerator =====================
        # PDF: manuals/ff255.pdf

        # 10 – Appliance does not work
        TestCase(
            case_id="10",
            input_data={
                "fault_description": (
                    "Side-by-side fridge-freezer is completely off. "
                    "No cooling, no sound, no lights."
                ),
                "appliance": "Side by Side Refrigerator Freezer FF255"
            },
            ground_truth={
                "diagnosis": "Appliance is not receiving power from the mains",
                "page_reference": "p. 18"
            },
            metadata={
                "complexity": 1,
                "category": "no_power"
            },
            context_document=r"DataSet\refrigerator\ff255.pdf" 
        ),

        # 11 – Food not frozen / not cold enough
        TestCase(
            case_id="11",
            input_data={
                "fault_description": (
                    "Freezer and fridge feel too warm. Frozen food is soft and ice cream is not firm."
                ),
                "appliance": "Side by Side Refrigerator Freezer FF255"
            },
            ground_truth={
                "diagnosis": "Insufficient cooling due to temperature settings or load and usage conditions",
                "page_reference": "p. 18"
            },
            metadata={
                "complexity": 1,
                "category": "insufficient_cooling"
            },
            context_document=r"DataSet\refrigerator\ff255.pdf"
        ),

        # 12 – Heavy build up of frost on door seal
        TestCase(
            case_id="12",
            input_data={
                "fault_description": (
                    "Heavy ice and frost building up around the door seal. Door does not seem to close properly."
                ),
                "appliance": "Side by Side Refrigerator Freezer FF255"
            },
            ground_truth={
                "diagnosis": "Door gasket not sealing properly, causing air leakage and frost build-up",
                "page_reference": "p. 18"
            },
            metadata={
                "complexity": 2,
                "category": "door_seal_frost"
            },
            context_document=r"DataSet\refrigerator\ff255.pdf"
        ),

        # ===================== 13–15: SFES2520A – Refrigerator =====================
        # PDF: manuals/sfes2520a.pdf

        # 13 – Refrigerator does not operate
        TestCase(
            case_id="13",
            input_data={
                "fault_description": (
                    "Refrigerator completely stopped. No cooling and no internal light."
                ),
                "appliance": "SFES2520A Refrigerator"
            },
            ground_truth={
                "diagnosis": "Refrigerator not operating due to missing or interrupted power supply",
                "page_reference": "p. 12"
            },
            metadata={
                "complexity": 1,
                "category": "no_power"
            },
            context_document=r"DataSet\refrigerator\sfes2520a.pdf"
        ),

        # 14 – Refrigerator performs poorly
        TestCase(
            case_id="14",
            input_data={
                "fault_description": (
                    "Fridge feels warm and food is not cooling properly, especially after loading more items."
                ),
                "appliance": "SFES2520A Refrigerator"
            },
            ground_truth={
                "diagnosis": "Insufficient cooling due to usage or installation conditions",
                "page_reference": "p. 12"
            },
            metadata={
                "complexity": 1,
                "category": "insufficient_cooling"
            },
            context_document=r"DataSet\refrigerator\sfes2520a.pdf"
        ),

        # 15 – Bubbling / normal operating noise
        TestCase(
            case_id="15",
            input_data={
                "fault_description": (
                    "User hears bubbling or hissing noise from inside the refrigerator and "
                    "is worried something is broken."
                ),
                "appliance": "SFES2520A Refrigerator"
            },
            ground_truth={
                "diagnosis": "Normal operating noise from refrigerant flow in the cooling circuit",
                "page_reference": "p. 12"
            },
            metadata={
                "complexity": 1,
                "category": "normal_noise_explained"
            },
            context_document=r"DataSet\refrigerator\sfes2520a.pdf"
        ),

        # ===================== 16–18: KQD 1250 – Refrigerator =====================
        # PDF: manuals/kqd_1250.pdf

        # 16 – Fridge runs very often / long time
        TestCase(
            case_id="16",
            input_data={
                "fault_description": (
                    "Fridge seems to run almost all the time and the compressor rarely stops."
                ),
                "appliance": "KQD 1250 Refrigerator"
            },
            ground_truth={
                "diagnosis": "Long running time due to normal behaviour or operating conditions",
                "page_reference": "p. 21"
            },
            metadata={
                "complexity": 2,
                "category": "long_run_time"
            },
            context_document=r"DataSet\refrigerator\kqd_1250.pdf"
        ),

        # 17 – Food in fridge drawers freezing
        TestCase(
            case_id="17",
            input_data={
                "fault_description": (
                    "Vegetables and other items in the fridge compartment drawers are freezing."
                ),
                "appliance": "KQD 1250 Refrigerator"
            },
            ground_truth={
                "diagnosis": "Fridge compartment temperature set too low for fresh-food drawers",
                "page_reference": "p. 22"
            },
            metadata={
                "complexity": 1,
                "category": "too_cold_fresh_food"
            },
            context_document=r"DataSet\refrigerator\kqd_1250.pdf"
        ),

        # 18 – Vibrations or noise
        TestCase(
            case_id="18",
            input_data={
                "fault_description": (
                    "User hears vibration or rattling noise when the fridge is running."
                ),
                "appliance": "KQD 1250 Refrigerator"
            },
            ground_truth={
                "diagnosis": "Mechanical vibration due to installation or floor conditions or items touching the cabinet",
                "page_reference": "p. 22"
            },
            metadata={
                "complexity": 1,
                "category": "noise_vibration"
            },
            context_document=r"DataSet\refrigerator\kqd_1250.pdf"
        ),

        # ===== ADD YOUR TEST CASES HERE =====
        # You can use the SAME manual for multiple test cases,
        # or DIFFERENT manuals for each test case!
        #
        # TestCase(
        #     case_id="DIAG_005",
        #     input_data={
        #         "fault_description": "Your fault description",
        #         "appliance": "Your appliance"
        #     },
        #     ground_truth={
        #         "diagnosis": "Expected diagnosis"
        #     },
        #     metadata={"complexity": 1},
        #     context_document="manuals/your_manual.pdf"  # ← Specify per case!
        # ),

    ]
    # determine number of cases to use (cases 4-6)
    selected_cases = test_cases[3:13]

    # ENHANCED: Task-specific validation ensures all cases have required fields
    return Dataset(selected_cases, task_type="diagnosis")


# =====================================================================
# STEP 3: RUN EXPERIMENT
# =====================================================================

def run_experiment():
    """
    Run the diagnosis experiment.

    This function:
    1. Loads test cases
    2. Creates experiment configuration
    3. Initializes LLM service
    4. Runs the experiment (for each model × context_mode × test case × repetition)
    5. Returns results for analysis
    """

    print("="*70)
    print("DIAGNOSIS EXPERIMENT")
    print("="*70)
    print()

    # Create dataset
    print("Loading test cases...")
    dataset = create_test_cases()
    print(f"Loaded {len(dataset)} test cases")
    print()

    # Create configuration
    print("Configuring experiment...")
    config = ExperimentConfig(
        task_type="diagnosis",
        models=MODELS_TO_TEST,
        metrics=METRICS_TO_USE,
        n_repetitions=N_REPETITIONS,
        context_modes=CONTEXT_MODES_TO_TEST,      # NEW: Explicit context modes
        temperature=TEMPERATURE,
        top_p=TOP_P,                              # NEW: Nucleus sampling parameter
        max_tokens=MAX_TOKENS,
        baseline_max_tokens=BASELINE_MAX_TOKENS,  # Overwrite max tokens for baseline
        top_k_retrieval=TOP_K_RETRIEVAL,          # NEW: Retrieval depth
        prompt_variant=PROMPT_VARIANT,
        chunk_size=CHUNK_SIZE,                    # NEW: Adaptive context budgeting
        max_context_chars=MAX_CONTEXT_CHARS,      # NEW: Adaptive context budgeting
        manual_full_max_chunks=MANUAL_FULL_MAX_CHUNKS,  # NEW: Adaptive context budgeting
        context_budget_overrides=CONTEXT_BUDGET_OVERRIDES,  # NEW: Per-provider overrides
        include_baseline=INCLUDE_BASELINE         # DEPRECATED: For backward compatibility
    )
  

    print(f"Configuration:")
    print(f"  - Task: {config.task_type}")
    print(f"  - Models: {[m.value for m in config.models]}")
    print(f"  - Context modes: {[cm.value for cm in config.context_modes]}")  # NEW
    print(f"  - Metrics: {config.metrics}")
    print(f"  - Repetitions: {config.n_repetitions}")
    print(f"  - Temperature: {config.temperature}")
    print(f"  - Top-p: {config.top_p}")  # NEW
    print(f"  - Max tokens: {config.max_tokens}")
    print(f"  - Top-k retrieval: {config.top_k_retrieval}")  # NEW
    if config.prompt_variant:
        print(f"  - Prompt variant: {config.prompt_variant}")
    print()

    # Initialize LLM service
    print("Initializing LLM service...")
    try:
        llm_service = LLMService()
        print("LLM service ready")
    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Make sure you have:")
        print("1. Created .env file from .env.example")
        print("2. Added your API keys (GEMINI_API_KEY, etc.)")
        return None, None
    print()

    # Run experiment
    print("="*70)
    print("RUNNING EXPERIMENT")
    print("="*70)
    print()
    print("This may take several minutes...")
    print()

    try:
        runner = ExperimentRunner(config, llm_service)
        results = runner.run(dataset, verbose=True)
        print()
        print("Experiment completed successfully!")
        print()
        # ENHANCED: Return both results and config for manifest export
        return results, config

    except KeyboardInterrupt:
        print()
        print("Experiment interrupted by user")
        return None, None

    except Exception as e:
        print()
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# =====================================================================
# STEP 4: ANALYZE RESULTS
# =====================================================================

def analyze_results(results, config):
    """
    Analyze and save experiment results with full provenance.

    UNDERSTANDING YOUR RESULTS:
    ==========================
    1. Overall Performance (table1_overall_performance.csv):
       - Shows average metrics for each model × context_mode combination
       - Compare BASELINE vs RAG_RETRIEVAL to see RAG improvement
       - Look for NaN values in RAG metrics for BASELINE rows (expected!)

    2. Complexity Analysis (table2_complexity_analysis.csv):
       - Breaks down performance by complexity level (1=easy, 2=medium, 3=hard)
       - Shows which models/modes handle complex cases better

    3. Output Consistency (table3_output_consistency.csv):
       - Only meaningful if n_repetitions > 1
       - Higher = more consistent across runs

    4. Raw Results (raw_results.csv):
       - All individual test runs with full metrics
       - Use for detailed analysis and plotting
       - ENHANCED: Now includes prompt_hash, context_hash, retrieval_query, model_identifier

    5. Run Manifest (run_manifest.json):
       - NEW: Full experiment configuration for reproducibility
       - Includes git commit, Python version, package versions
       - Enables supervisor to reproduce your exact results

    KEY INSIGHTS TO LOOK FOR:
    ========================
    - How much does RAG_RETRIEVAL improve over BASELINE?
    - Does MANUAL_FULL outperform RAG_RETRIEVAL? (suggests retrieval quality issues)
    - Are citation_coverage and citation_correctness high? (good grounding)
    - Is hallucination_rate low? (< 0.3 is good)
    - How does context_precision/recall vary? (measures retrieval quality)
    """

    if results is None:
        return

    print("="*70)
    print("RESULTS ANALYSIS")
    print("="*70)
    print()

    # Save detailed results with manifest
    output_dir = "results/diagnosis_experiment"
    print(f"Saving results to: {output_dir}/")

    # ENHANCED: Export with full provenance manifest
    results.export_all(
        output_dir=output_dir,
        config=config,
        dataset_path=None  # Could specify dataset file path here if saved
    )

    print()
    print("Saved files:")
    print(f"  - {output_dir}/table1_overall_performance.csv")
    print(f"  - {output_dir}/table2_complexity_analysis.csv")
    print(f"  - {output_dir}/table3_output_consistency.csv")
    print(f"  - {output_dir}/raw_results.csv")
    print(f"  - {output_dir}/run_manifest.json  <- NEW: Full provenance!")
    print()

    # Create analyzer for summary display
    analyzer = ResultsAnalyzer(results)

    # Print summary to console
    print("Overall Performance:")
    print("-" * 70)
    analyzer.print_summary()
    print()

    # Show sample results
    if not results.df.empty:
        print("Sample Results (first 5 rows):")
        print("-" * 70)
        cols_to_show = ['case_id', 'model', 'context_mode'] + METRICS_TO_USE[:3]
        print(results.df[cols_to_show].head().to_string())
        print()


# =====================================================================
# MAIN
# =====================================================================

def main():
    """Main function"""

    print()
    print("=" * 70)
    print(" " * 20 + "DIAGNOSIS EXPERIMENT TEST")
    print("=" * 70)
    print()

    try:
        # Run experiment (ENHANCED: returns both results and config)
        results, config = run_experiment()

        # Analyze results
        if results:
            analyze_results(results, config)

            print("="*70)
            print("TEST COMPLETED SUCCESSFULLY!")
            print("="*70)
            print()
            print("Next steps:")
            print("1. Review results in results/diagnosis_experiment/")
            print("2. Check run_manifest.json for full experiment configuration")
            print("3. Analyze the CSV files for your thesis")
            print("4. Compare performance across context modes")
            print()
            print("ENHANCED FRAMEWORK FEATURES:")
            print("  - Task-specific validation (diagnosis requires: fault_description, appliance, diagnosis)")
            print("  - Full provenance tracking (prompt_hash, context_hash, model_identifier)")
            print("  - Reproducible experiments (consistent random_seed, temperature, max_tokens)")
            print("  - Run manifest (git commit, Python version, package versions)")
            print()
            print("QUICK TIPS:")
            print("  - NaN values in RAG metrics for BASELINE rows are normal")
            print("  - To test all 3 modes, uncomment the CONTEXT_MODES_TO_TEST option 2")
            print("  - To increase retrieval depth, change TOP_K_RETRIEVAL (default: 6)")
            print("  - For consistency analysis, set N_REPETITIONS to 5-10")
            print()

    except KeyboardInterrupt:
        print()
        print("Interrupted by user")

    except Exception as e:
        print()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
