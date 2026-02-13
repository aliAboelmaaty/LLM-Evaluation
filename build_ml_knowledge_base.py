"""
Build Synthetic Knowledge Base from Sonntag's Dataset

This creates a "manual" document where each case becomes a retrievable chunk.
Your existing RAG system (BM25) will retrieve relevant examples during inference.

Strategy:
- Each dataset entry becomes a structured knowledge chunk
- Chunks follow consistent format for better retrieval
- The resulting document is saved as a PDF-equivalent text file
- Your retrieval system treats it like a diagnosis manual

IMPORTANT: For proper evaluation, use leave-one-out cross-validation by
excluding the test case from the knowledge base using exclude_case_id parameter.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def build_knowledge_base(
    dataset_path: str,
    output_path: str = "ml_knowledge_base.txt",
    exclude_case_id: Optional[str] = None,
    verbose: bool = True
) -> None:
    """
    Build synthetic knowledge base from Sonntag's dataset.

    CRITICAL: Use exclude_case_id for leave-one-out cross-validation to prevent
    data leakage during evaluation. Each test case should be evaluated against
    a KB that does NOT contain its own answer.

    Each case becomes a structured knowledge chunk:
    ===================================================================
    ML Application Case #001
    ===================================================================
    Paper: "Title from dataset"
    Domain: [Inferred from title]

    Problem Context:
    [Background information from dataset]

    Problem Formulation:
    [Problem formulation from dataset]

    ML Solution:
    - Problem Type: Classification; Clustering
    - Algorithm: Random Forest, RSKC
    - Rationale: [Inferred from problem type and algorithm combination]

    ===================================================================

    Args:
        dataset_path: Path to Sonntag's JSON dataset
        output_path: Path to save the knowledge base text file
        exclude_case_id: Case ID to exclude (e.g., "ML_001") for leave-one-out CV
        verbose: Print progress messages
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Build knowledge base document
    kb_chunks = []
    excluded_count = 0

    for idx, item in enumerate(data, start=1):
        case_id = f"ML_{idx:03d}"

        # CRITICAL: Skip if this is the test case (leave-one-out)
        if exclude_case_id and case_id == exclude_case_id:
            excluded_count += 1
            continue  # Don't include in KB to prevent data leakage

        chunk = _format_knowledge_chunk(idx, item)
        kb_chunks.append(chunk)

    # Write to file
    full_kb = "\n\n".join(kb_chunks)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_kb)

    if verbose:
        print(f"Knowledge base created: {output_path}")
        print(f"  - Total cases in dataset: {len(data)}")
        print(f"  - Cases included in KB: {len(kb_chunks)}")
        if exclude_case_id:
            print(f"  - Excluded (leave-one-out): {exclude_case_id}")
        print(f"  - Total characters: {len(full_kb):,}")


def build_kb_for_case(
    dataset_path: str,
    case_id: str,
    cache_dir: str = "kb_cache"
) -> str:
    """
    Build or retrieve cached KB for a specific test case (leave-one-out).

    This ensures each test case is evaluated against a KB that excludes
    its own answer, preventing data leakage.

    Args:
        dataset_path: Path to the dataset JSON
        case_id: Case ID to exclude (e.g., "ML_001")
        cache_dir: Directory to cache KB files

    Returns:
        Path to the KB file (excluding the specified case)
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    kb_filename = f"kb_excluding_{case_id}.txt"
    kb_path = cache_path / kb_filename

    # Only build if not already cached
    if not kb_path.exists():
        build_knowledge_base(
            dataset_path=dataset_path,
            output_path=str(kb_path),
            exclude_case_id=case_id,
            verbose=False  # Quiet mode for batch processing
        )

    return str(kb_path)


def clear_kb_cache(cache_dir: str = "kb_cache") -> None:
    """Clear cached KB files."""
    import shutil
    cache_path = Path(cache_dir)
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"KB cache cleared: {cache_dir}")


def get_cache_stats(cache_dir: str = "kb_cache") -> str:
    """Get statistics about KB cache."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return "No cache directory"

    files = list(cache_path.glob("*.txt"))
    if not files:
        return "Cache empty"

    total_size = sum(f.stat().st_size for f in files)
    return f"{len(files)} KB files, {total_size/1024/1024:.2f} MB total"


def _format_knowledge_chunk(idx: int, item: Dict[str, Any]) -> str:
    """
    Format a single dataset entry as a knowledge chunk.

    This format is optimized for BM25 retrieval:
    - Clear section headers for better parsing
    - Keywords repeated for better matching
    - Structured format for consistent extraction
    """
    # Extract fields
    title = item.get("Title", "")
    background = item.get("Background Information", "")
    problem = item.get("Problem Formulation", "")
    ml_type = item.get("ML-Problem Type", "")
    ml_algo = item.get("ML Algorithm", "")

    # Infer domain from title (for context)
    domain = _infer_domain(title)

    # Generate rationale based on problem type and algorithm
    rationale = _generate_rationale(ml_type, ml_algo, problem)

    # Format chunk
    chunk = f"""{'='*75}
ML Application Case #{idx:03d}
{'='*75}
Paper: {title}
Domain: {domain}

Problem Context:
{background.strip() if background else 'No background information provided.'}

Problem Formulation:
{problem.strip()}

ML Solution:
- Problem Type: {ml_type}
- Algorithm: {ml_algo}
- Rationale: {rationale}

{'='*75}"""

    return chunk


def _infer_domain(title: str) -> str:
    """Infer application domain from paper title."""
    title_lower = title.lower()

    if 'customer' in title_lower or 'market' in title_lower or 'user' in title_lower:
        return 'Customer Analysis'
    elif 'product' in title_lower or 'design' in title_lower:
        return 'Product Development'
    elif 'requirement' in title_lower or 'standard' in title_lower:
        return 'Requirements Engineering'
    elif 'lead user' in title_lower or 'innovation' in title_lower:
        return 'Innovation Management'
    elif 'sale' in title_lower or 'forecast' in title_lower:
        return 'Sales Forecasting'
    elif 'performance' in title_lower or 'evaluation' in title_lower:
        return 'Performance Evaluation'
    else:
        return 'General Product Development'


def _generate_rationale(ml_type: str, ml_algo: str, problem: str) -> str:
    """
    Generate a rationale explaining why this ML solution fits the problem.

    This helps the LLM understand the reasoning when retrieving this case.
    """
    rationales = []

    ml_type_lower = ml_type.lower()

    # Type-specific rationales
    if 'classification' in ml_type_lower:
        rationales.append("Classification is appropriate because the task involves categorizing entities into predefined classes or groups.")

    if 'regression' in ml_type_lower:
        rationales.append("Regression is suitable because the task requires predicting continuous numerical values or quantitative outcomes.")

    if 'clustering' in ml_type_lower:
        rationales.append("Clustering is applicable because the task involves grouping similar entities based on shared characteristics without predefined labels.")

    if 'association' in ml_type_lower or 'rule' in ml_type_lower:
        rationales.append("Association rule mining is appropriate because the task seeks to discover relationships and patterns between variables or items.")

    # Algorithm-specific notes
    if any(algo in ml_algo.lower() for algo in ['random forest', 'rf', 'xgboost', 'gradient boosting']):
        rationales.append("The ensemble method provides robust performance and handles complex feature interactions.")

    if any(algo in ml_algo.lower() for algo in ['svm', 'support vector']):
        rationales.append("Support vector methods are effective for high-dimensional data and non-linear decision boundaries.")

    if any(algo in ml_algo.lower() for algo in ['neural', 'nn', 'ann', 'lstm', 'bert']):
        rationales.append("Neural network approaches can capture complex patterns and handle large-scale data effectively.")

    if any(algo in ml_algo.lower() for algo in ['kmeans', 'k-means', 'hierarchical', 'dbscan']):
        rationales.append("The clustering algorithm can identify natural groupings in the data without requiring labeled examples.")

    if any(algo in ml_algo.lower() for algo in ['apriori', 'fp-growth']):
        rationales.append("The association rule algorithm efficiently discovers frequent patterns and relationships in transactional data.")

    return " ".join(rationales) if rationales else "This ML approach is well-suited for the described product development problem."


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python build_ml_knowledge_base.py <dataset.json> [output.txt] [exclude_case_id]")
        print("")
        print("Examples:")
        print("  python build_ml_knowledge_base.py dataset.json")
        print("  python build_ml_knowledge_base.py dataset.json kb.txt")
        print("  python build_ml_knowledge_base.py dataset.json kb.txt ML_001  # Exclude case ML_001")
        print("")
        print("For leave-one-out evaluation, use build_kb_for_case() function:")
        print("  from build_ml_knowledge_base import build_kb_for_case")
        print("  kb_path = build_kb_for_case('dataset.json', 'ML_001')")
        sys.exit(1)

    dataset_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "ml_knowledge_base.txt"
    exclude_case_id = sys.argv[3] if len(sys.argv) > 3 else None

    build_knowledge_base(dataset_path, output_path, exclude_case_id=exclude_case_id)
