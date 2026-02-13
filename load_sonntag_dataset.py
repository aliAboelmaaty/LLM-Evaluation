"""
Load Sonntag's ML Recommendation Dataset

Converts JSON dataset to TestCase objects.
Includes support for linking to the synthetic knowledge base.
"""

import json
from typing import List, Dict, Any, Optional

from framework import TestCase, Dataset


def load_sonntag_dataset(
    json_path: str,
    knowledge_base_path: Optional[str] = None
) -> Dataset:
    """
    Load Sonntag's ML recommendation dataset.

    Args:
        json_path: Path to dataset JSON
        knowledge_base_path: Optional path to synthetic knowledge base (for RAG mode)

    Returns:
        Dataset object
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_cases = []

    for idx, item in enumerate(data, start=1):
        case_id = f"ML_{idx:03d}"

        # Build input_data
        input_data = {
            "problem_formulation": item["Problem Formulation"]
        }

        # Add background if substantial
        background = item.get("Background Information", "")
        if background and len(background) > 50:
            input_data["background_information"] = background

        # Build ground_truth
        ground_truth = {
            "ml_problem_type": item["ML-Problem Type"],
            "ml_algorithm": item["ML Algorithm"]
        }

        # Build metadata
        metadata = {
            "title": item["Title"],
            "source": "Sonntag et al. (2025)",
            "domain": _infer_domain(item["Title"]),
            "complexity": 2 if ';' in item["ML-Problem Type"] else 1
        }

        # Link to knowledge base (if provided)
        context_document = knowledge_base_path if knowledge_base_path else None

        tc = TestCase(
            case_id=case_id,
            input_data=input_data,
            ground_truth=ground_truth,
            metadata=metadata,
            context_document=context_document
        )

        test_cases.append(tc)

    # Create dataset with metadata for leave-one-out KB building
    dataset = Dataset(
        test_cases,
        task_type="ml_recommendation",
        metadata={
            'source_path': json_path,  # Track source for dynamic KB building
            'knowledge_base_path': knowledge_base_path,
        }
    )

    return dataset


def _infer_domain(title: str) -> str:
    """Infer domain from paper title."""
    title_lower = title.lower()

    if 'customer' in title_lower or 'market' in title_lower or 'user' in title_lower:
        return 'customer_analysis'
    elif 'product' in title_lower or 'design' in title_lower:
        return 'product_development'
    elif 'requirement' in title_lower or 'standard' in title_lower:
        return 'requirements_engineering'
    elif 'lead user' in title_lower or 'innovation' in title_lower:
        return 'innovation'
    elif 'sale' in title_lower or 'forecast' in title_lower:
        return 'sales_forecasting'
    elif 'performance' in title_lower or 'evaluation' in title_lower:
        return 'performance_evaluation'
    else:
        return 'general'


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python load_sonntag_dataset.py <dataset.json> [knowledge_base.txt]")
        sys.exit(1)

    json_path = sys.argv[1]
    kb_path = sys.argv[2] if len(sys.argv) > 2 else None

    dataset = load_sonntag_dataset(json_path, kb_path)

    print(f"Loaded {len(dataset)} test cases")
    if kb_path:
        print(f"Linked to knowledge base: {kb_path}")

    stats = dataset.get_stats()
    print(f"\nDataset Statistics:")
    print(f"  - Total cases: {stats['n_cases']}")
    print(f"  - Complexity distribution: {stats['complexity_distribution']}")
