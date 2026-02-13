"""
Dataset Management Module

Handles loading, creating, and managing test case datasets for experiments.

Supports:
- Loading from JSON, CSV
- Building datasets programmatically
- Filtering by complexity, category
- Train/test splits
- PRISMA-like dataset extraction from PDFs
"""

from typing import List, Dict, Any, Tuple, Optional
import json
import csv
from pathlib import Path
import random

from .core import TestCase, validate_test_case


# ================= Dataset Class =================

class Dataset:
    """
    Generic dataset container for test cases.

    Can be loaded from files or built programmatically.
    Provides utilities for filtering, splitting, and validation.
    """

    def __init__(self, test_cases: List[TestCase], task_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize dataset with test cases.

        Args:
            test_cases: List of TestCase objects
            task_type: Optional task type for validation ("diagnosis", "repurposing", etc.)
            metadata: Optional metadata dict (e.g., source_path for leave-one-out KB building)
        """
        self.test_cases = test_cases
        self.task_type = task_type
        self.metadata = metadata or {}  # Store dataset-level metadata
        self._validate_all()

    def _validate_all(self):
        """Validate all test cases with task-specific rules"""
        for tc in self.test_cases:
            validate_test_case(tc, task_type=self.task_type)

    def __len__(self) -> int:
        """Number of test cases in dataset"""
        return len(self.test_cases)

    def __getitem__(self, idx: int) -> TestCase:
        """Get test case by index"""
        return self.test_cases[idx]

    def __iter__(self):
        """Iterate over test cases"""
        return iter(self.test_cases)

    def __repr__(self) -> str:
        """String representation"""
        return f"Dataset(n_cases={len(self.test_cases)})"

    # ================= Loading from Files =================

    @classmethod
    def from_json(cls, filepath: str, task_type: Optional[str] = None) -> 'Dataset':
        """
        Load dataset from JSON file.

        JSON format:
        [
            {
                "case_id": "001",
                "input_data": {"question": "..."},
                "ground_truth": {"answer": "..."},
                "metadata": {"complexity": 1},
                "context_document": "path/to/manual.pdf"
            },
            ...
        ]

        Args:
            filepath: Path to JSON file
            task_type: Optional task type for validation ("diagnosis", "repurposing", etc.)

        Returns:
            Dataset instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_cases = []
        for item in data:
            tc = TestCase(
                case_id=item['case_id'],
                input_data=item['input_data'],
                ground_truth=item['ground_truth'],
                metadata=item.get('metadata', {}),
                context_document=item.get('context_document')
            )
            test_cases.append(tc)

        return cls(test_cases, task_type=task_type)

    @classmethod
    def from_csv(cls, filepath: str, mapping: Dict[str, str], task_type: Optional[str] = None) -> 'Dataset':
        """
        Load dataset from CSV with column mapping.

        Args:
            filepath: Path to CSV file
            mapping: Dictionary mapping field paths to CSV columns
                Example:
                {
                    "case_id": "id",
                    "input_data.question": "fault_description",
                    "ground_truth.diagnosis": "expected_diagnosis",
                    "metadata.complexity": "complexity_cluster",
                    "context_document": "manual_pdf_path"
                }
            task_type: Optional task type for validation ("diagnosis", "repurposing", etc.)

        Returns:
            Dataset instance
        """
        test_cases = []

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Build test case from mapping
                case_id = row.get(mapping.get('case_id', 'case_id'), '')

                # Extract input_data
                input_data = {}
                for field_path, col_name in mapping.items():
                    if field_path.startswith('input_data.'):
                        key = field_path.split('.', 1)[1]
                        input_data[key] = row.get(col_name, '')

                # Extract ground_truth
                ground_truth = {}
                for field_path, col_name in mapping.items():
                    if field_path.startswith('ground_truth.'):
                        key = field_path.split('.', 1)[1]
                        ground_truth[key] = row.get(col_name, '')

                # Extract metadata
                metadata = {}
                for field_path, col_name in mapping.items():
                    if field_path.startswith('metadata.'):
                        key = field_path.split('.', 1)[1]
                        value = row.get(col_name, '')
                        # Try to convert to int if it looks like a number
                        try:
                            metadata[key] = int(value)
                        except ValueError:
                            metadata[key] = value

                # Context document
                context_doc = row.get(mapping.get('context_document', 'context_document'))

                tc = TestCase(
                    case_id=case_id,
                    input_data=input_data,
                    ground_truth=ground_truth,
                    metadata=metadata,
                    context_document=context_doc
                )
                test_cases.append(tc)

        return cls(test_cases, task_type=task_type)

    # ================= Saving to Files =================

    def to_json(self, filepath: str):
        """Save dataset to JSON file"""
        data = []
        for tc in self.test_cases:
            data.append({
                'case_id': tc.case_id,
                'input_data': tc.input_data,
                'ground_truth': tc.ground_truth,
                'metadata': tc.metadata,
                'context_document': tc.context_document
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def to_csv(self, filepath: str, mapping: Dict[str, str]):
        """
        Save dataset to CSV file.

        Args:
            filepath: Path to output CSV file
            mapping: Same format as from_csv mapping
        """
        # Determine all column names
        columns = list(set(mapping.values()))

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for tc in self.test_cases:
                row = {}
                for field_path, col_name in mapping.items():
                    if field_path == 'case_id':
                        row[col_name] = tc.case_id
                    elif field_path == 'context_document':
                        row[col_name] = tc.context_document or ''
                    elif field_path.startswith('input_data.'):
                        key = field_path.split('.', 1)[1]
                        row[col_name] = tc.input_data.get(key, '')
                    elif field_path.startswith('ground_truth.'):
                        key = field_path.split('.', 1)[1]
                        row[col_name] = tc.ground_truth.get(key, '')
                    elif field_path.startswith('metadata.'):
                        key = field_path.split('.', 1)[1]
                        row[col_name] = tc.metadata.get(key, '')

                writer.writerow(row)

    # ================= Filtering & Splitting =================

    def filter_by_complexity(self, cluster: int) -> 'Dataset':
        """
        Filter to specific complexity cluster.

        Args:
            cluster: Complexity cluster number (e.g., 1 or 2)

        Returns:
            New Dataset with filtered test cases
        """
        filtered = [
            tc for tc in self.test_cases
            if tc.metadata.get('complexity') == cluster
        ]
        return Dataset(filtered, task_type=self.task_type)

    def filter_by_metadata(self, **kwargs) -> 'Dataset':
        """
        Filter by metadata fields.

        Args:
            **kwargs: Metadata key-value pairs to match

        Returns:
            New Dataset with filtered test cases

        Example:
            dataset.filter_by_metadata(appliance_type="washing_machine")
        """
        filtered = []
        for tc in self.test_cases:
            match = all(
                tc.metadata.get(key) == value
                for key, value in kwargs.items()
            )
            if match:
                filtered.append(tc)

        return Dataset(filtered, task_type=self.task_type)

    def split_train_test(
        self,
        test_size: float = 0.2,
        random_seed: int = 42,
        stratify_by: Optional[str] = None
    ) -> Tuple['Dataset', 'Dataset']:
        """
        Split dataset into train and test sets.

        Args:
            test_size: Fraction for test set (0.0 to 1.0)
            random_seed: Random seed for reproducibility
            stratify_by: Optional metadata key to stratify by

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if not (0.0 < test_size < 1.0):
            raise ValueError("test_size must be between 0.0 and 1.0")

        cases = self.test_cases.copy()
        random.Random(random_seed).shuffle(cases)

        # Simple split (stratification would require more complex logic)
        n_test = int(len(cases) * test_size)
        test_cases = cases[:n_test]
        train_cases = cases[n_test:]

        return Dataset(train_cases, task_type=self.task_type), Dataset(test_cases, task_type=self.task_type)

    # ================= Statistics =================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'n_cases': len(self.test_cases),
            'complexity_distribution': {},
            'metadata_keys': set(),
            'has_context_documents': 0
        }

        for tc in self.test_cases:
            # Complexity distribution
            complexity = tc.metadata.get('complexity')
            if complexity is not None:
                stats['complexity_distribution'][complexity] = \
                    stats['complexity_distribution'].get(complexity, 0) + 1

            # Metadata keys
            stats['metadata_keys'].update(tc.metadata.keys())

            # Context documents
            if tc.context_document:
                stats['has_context_documents'] += 1

        stats['metadata_keys'] = list(stats['metadata_keys'])

        return stats


# ================= Dataset Builder =================

class DatasetBuilder:
    """
    Helper for creating datasets from various sources.

    Useful for:
    - Extracting test cases from PDF manuals
    - Programmatic dataset construction
    - Complexity classification
    """

    def __init__(self, task_type: str):
        """
        Initialize builder.

        Args:
            task_type: Type of task ("diagnosis", "repurposing", etc.)
        """
        self.task_type = task_type
        self.test_cases: List[TestCase] = []

    def add_test_case(
        self,
        case_id: str,
        input_data: Dict[str, Any],
        ground_truth: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        context_document: Optional[str] = None
    ) -> 'DatasetBuilder':
        """
        Add a test case to the builder.

        Args:
            case_id: Unique identifier
            input_data: Input fields
            ground_truth: Expected outputs
            metadata: Additional metadata
            context_document: Optional PDF path

        Returns:
            Self for chaining
        """
        tc = TestCase(
            case_id=case_id,
            input_data=input_data,
            ground_truth=ground_truth,
            metadata=metadata or {},
            context_document=context_document
        )
        self.test_cases.append(tc)
        return self

    def classify_complexity(
        self,
        complexity_fn: callable
    ) -> 'DatasetBuilder':
        """
        Assign complexity scores to test cases.

        Args:
            complexity_fn: Function that takes TestCase and returns complexity (1, 2, etc.)

        Returns:
            Self for chaining

        Example:
            def complexity(tc):
                # Simple if fault code is mentioned
                if "error code" in tc.input_data.get("fault_description", "").lower():
                    return 1
                return 2

            builder.classify_complexity(complexity)
        """
        for tc in self.test_cases:
            tc.metadata['complexity'] = complexity_fn(tc)

        return self

    def build(self) -> Dataset:
        """
        Build the dataset.

        Returns:
            Dataset instance with task-specific validation
        """
        return Dataset(self.test_cases, task_type=self.task_type)

    def __repr__(self) -> str:
        """String representation"""
        return f"DatasetBuilder(task_type='{self.task_type}', n_cases={len(self.test_cases)})"
