"""
Results Analysis Module

Provides visualization and statistical analysis of experiment results.

Features:
- Publication-ready table export
- Performance visualizations
- Statistical comparisons
- Complexity impact analysis
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd

from .core import ExperimentResults


# ================= Results Analyzer =================

class ResultsAnalyzer:
    """
    Analyze and visualize experiment results.

    Generates publication-ready tables and plots following the format
    from research papers (Dörnbach, Sonntag).
    """

    def __init__(self, results: ExperimentResults):
        """
        Initialize analyzer.

        Args:
            results: ExperimentResults from experiment run
        """
        self.results = results

    def export_all_tables(self, output_dir: str):
        """
        Export all analysis tables to CSV files.

        Creates:
        - table1_overall_performance.csv
        - table2_complexity_analysis.csv
        - table3_output_consistency.csv
        - raw_results.csv

        Args:
            output_dir: Directory to save CSV files
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Overall performance (Table 1)
        if not self.results.overall_table.empty:
            self.results.overall_table.reset_index().to_csv(
                out_path / "table1_overall_performance.csv",
                index=False,
                sep=";",
                decimal=",",
                encoding="utf-8-sig"
            )
            print(f"Saved: {out_path / 'table1_overall_performance.csv'}")

        # By complexity (Table 2)
        if not self.results.complexity_table.empty:
            self.results.complexity_table.reset_index().to_csv(
                out_path / "table2_complexity_analysis.csv",
                index=False,
                sep=";",
                decimal=",",
                encoding="utf-8-sig"
            )
            print(f"Saved: {out_path / 'table2_complexity_analysis.csv'}")

        # Consistency (Table 3)
        if not self.results.consistency_table.empty:
            self.results.consistency_table.reset_index().to_csv(
                out_path / "table3_output_consistency.csv",
                index=False,
                sep=";",
                decimal=",",
                encoding="utf-8-sig"
            )
            print(f"Saved: {out_path / 'table3_output_consistency.csv'}")

        # Raw results
        if not self.results.df.empty:
            self.results.df.to_csv(
                out_path / "raw_results.csv",
                index=False,
                sep=";",
                decimal=",",
                encoding="utf-8-sig"
            )
            print(f"Saved: {out_path / 'raw_results.csv'}")

    def print_summary(self):
        """Print a text summary of results to console"""
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*60)

        stats = self.results.summary_stats()

        print(f"\nTest Cases: {stats.get('n_test_cases', 0)}")
        print(f"Models: {stats.get('n_models', 0)}")
        print(f"Metrics: {len(stats.get('metrics', []))}")

        print("\n" + "-"*60)
        print("OVERALL PERFORMANCE (averaged across all test cases)")
        print("-"*60)

        if not self.results.overall_table.empty:
            print(self.results.overall_table.to_string())
        else:
            print("No results available")

        if not self.results.complexity_table.empty:
            print("\n" + "-"*60)
            print("PERFORMANCE BY COMPLEXITY CLUSTER")
            print("-"*60)
            print(self.results.complexity_table.to_string())

        if not self.results.consistency_table.empty:
            print("\n" + "-"*60)
            print("OUTPUT CONSISTENCY (across repetitions)")
            print("-"*60)
            print(self.results.consistency_table.to_string())

        print("\n" + "="*60 + "\n")

    def plot_performance_comparison(
        self,
        save_path: Optional[str] = None,
        metric: str = "answer_correctness"
    ):
        """
        Create bar chart comparing across context modes.

        Args:
            save_path: Optional path to save plot
            metric: Metric to plot
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        except ImportError:
            print("matplotlib not installed. Skipping plot.")
            return

        if self.results.df.empty:
            print("No results to plot")
            return

        # CHANGE #4: Filter out both runtime errors AND sanity check failures
        had_error = self.results.df.get("had_error", False).fillna(False).astype(bool)
        sanity_failed = self.results.df.get("sanity_check_failed", False).fillna(False).astype(bool)
        valid_df = self.results.df[~(had_error | sanity_failed)]

        if valid_df.empty:
            print("No valid results to plot (all had errors or sanity failures)")
            return

        # Determine grouping column (context_mode or use_rag for backward compatibility)
        if "context_mode" in valid_df.columns:
            group_col = "context_mode"
        elif "use_rag" in valid_df.columns:
            group_col = "use_rag"
        else:
            print(f"Missing 'context_mode' or 'use_rag' column in results. Cannot plot.")
            return

        if metric not in valid_df.columns:
            print(f"Metric '{metric}' not found in results")
            return

        # Group by model and context mode
        grouped = valid_df.groupby(["model", group_col])[metric].mean().unstack()

        # Create plot with 3 bars per model (baseline, manual_full, rag_retrieval)
        ax = grouped.plot(kind='bar', rot=45, figsize=(10, 6))
        ax.set_xlabel("Model")
        ax.set_ylabel(f"{metric} (higher is better)")
        ax.set_title(f"Performance Comparison: {metric}")

        # Set legend with actual context modes present
        ax.legend(title="Context Mode")

        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {save_path}")
        else:
            plt.show()

    def plot_complexity_impact(
        self,
        save_path: Optional[str] = None,
        metric: str = "answer_correctness"
    ):
        """
        Plot performance vs complexity.

        Args:
            save_path: Optional path to save plot
            metric: Metric to plot
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        except ImportError:
            print("matplotlib not installed. Skipping plot.")
            return

        if self.results.df.empty:
            print("No results to plot")
            return

        if "complexity" not in self.results.df.columns or metric not in self.results.df.columns:
            print(f"Missing required columns for plotting")
            return

        # Group by complexity
        grouped = self.results.df.groupby("complexity")[metric].mean()

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        grouped.plot(kind='bar', ax=ax, rot=0, color=['#2ecc71', '#e74c3c'])
        ax.set_xlabel("Complexity Cluster")
        ax.set_ylabel(f"{metric} (higher is better)")
        ax.set_title(f"Performance vs Complexity: {metric}")
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {save_path}")
        else:
            plt.show()

    def compare_models(self, metric: str = "answer_correctness") -> pd.DataFrame:
        """
        Compare models on a specific metric.

        Args:
            metric: Metric to compare

        Returns:
            DataFrame with model comparison
        """
        if self.results.df.empty or metric not in self.results.df.columns:
            return pd.DataFrame()

        comparison = self.results.df.groupby("model")[metric].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max')
        ]).round(3)

        return comparison.sort_values('mean', ascending=False)

    def statistical_test(
        self,
        metric: str = "answer_correctness",
        mode_a: Optional[str] = None,
        mode_b: Optional[str] = None,
        test: str = "ttest"
    ) -> Dict[str, Any]:
        """
        Perform statistical test comparing two context modes.

        Args:
            metric: Metric to test
            mode_a: First context mode to compare (e.g., "baseline")
            mode_b: Second context mode to compare (e.g., "rag_retrieval")
            test: Type of test ("ttest" or "wilcoxon")

        Returns:
            Dictionary with test results, or two default comparisons if modes not specified
        """
        if self.results.df.empty:
            return {}

        # CHANGE #4: Filter out both runtime errors AND sanity check failures
        had_error = self.results.df.get("had_error", False).fillna(False).astype(bool)
        sanity_failed = self.results.df.get("sanity_check_failed", False).fillna(False).astype(bool)
        valid_df = self.results.df[~(had_error | sanity_failed)]

        if valid_df.empty:
            return {"error": "No valid results (all had errors or sanity failures)"}

        # Determine grouping column (context_mode or use_rag for backward compatibility)
        if "context_mode" in valid_df.columns:
            group_col = "context_mode"
        elif "use_rag" in valid_df.columns:
            group_col = "use_rag"
        else:
            return {"error": "Missing 'context_mode' or 'use_rag' column"}

        if metric not in valid_df.columns:
            return {"error": f"Metric '{metric}' not found in results"}

        # If no modes specified, provide default comparisons
        if mode_a is None or mode_b is None:
            if group_col == "context_mode":
                return self._default_context_mode_comparisons(valid_df, metric, test)
            else:
                # Legacy use_rag comparison
                mode_a, mode_b = False, True

        # CRITICAL FIX: Extract PAIRED data by case_id AND model
        # The experiment design is PAIRED (same case_id across modes)
        # MUST pair by BOTH case_id AND model to avoid cross-model pairing
        # (e.g., baseline from model A with RAG from model B would be invalid)

        # Check if required columns exist for pairing
        if "case_id" not in valid_df.columns:
            return {"error": "Missing 'case_id' column - cannot perform paired test"}
        if "model" not in valid_df.columns:
            return {"error": "Missing 'model' column - cannot perform paired test"}

        # Get data for each mode (include model for proper pairing)
        df_a = valid_df[valid_df[group_col] == mode_a][["case_id", "model", metric]].dropna()
        df_b = valid_df[valid_df[group_col] == mode_b][["case_id", "model", metric]].dropna()

        # Merge on BOTH case_id AND model to ensure valid pairing
        # This prevents pairing baseline from modelA with RAG from modelB
        paired_df = df_a.merge(df_b, on=["case_id", "model"], suffixes=("_a", "_b"))

        if len(paired_df) == 0:
            return {"error": f"No paired data for {mode_a} vs {mode_b} (no overlapping case_ids)"}

        data_a_paired = paired_df[f"{metric}_a"].values
        data_b_paired = paired_df[f"{metric}_b"].values

        try:
            from scipy import stats

            if test == "ttest":
                # Use PAIRED t-test (not independent)
                statistic, pvalue = stats.ttest_rel(data_a_paired, data_b_paired)
                test_name = "Paired t-test"
            elif test == "wilcoxon":
                # Wilcoxon is already for paired data, now properly aligned
                statistic, pvalue = stats.wilcoxon(data_a_paired, data_b_paired)
                test_name = "Wilcoxon signed-rank test"
            else:
                return {"error": f"Unknown test: {test}"}

            return {
                "test": test_name,
                "metric": metric,
                f"{mode_a}_mean": float(data_a_paired.mean()),
                f"{mode_b}_mean": float(data_b_paired.mean()),
                "n_pairs": len(paired_df),
                "statistic": float(statistic),
                "p_value": float(pvalue),
                "significant": pvalue < 0.05
            }

        except ImportError:
            return {"error": "scipy not installed"}
        except Exception as e:
            return {"error": str(e)}

    def _default_context_mode_comparisons(
        self,
        valid_df: pd.DataFrame,
        metric: str,
        test: str
    ) -> Dict[str, Any]:
        """Provide default comparisons for context_mode using PAIRED tests"""
        # Check if required columns exist for pairing
        if "case_id" not in valid_df.columns:
            return {"error": "Missing 'case_id' column - cannot perform paired tests"}
        if "model" not in valid_df.columns:
            return {"error": "Missing 'model' column - cannot perform paired tests"}

        results = {}

        try:
            from scipy import stats

            # Comparison A: baseline vs rag_retrieval (PAIRED by case_id AND model)
            df_baseline = valid_df[valid_df["context_mode"] == "baseline"][["case_id", "model", metric]].dropna()
            df_rag = valid_df[valid_df["context_mode"] == "rag_retrieval"][["case_id", "model", metric]].dropna()

            if len(df_baseline) > 0 and len(df_rag) > 0:
                # Merge on BOTH case_id AND model to prevent cross-model pairing
                paired_df = df_baseline.merge(df_rag, on=["case_id", "model"], suffixes=("_baseline", "_rag"))

                if len(paired_df) > 0:
                    baseline_paired = paired_df[f"{metric}_baseline"].values
                    rag_paired = paired_df[f"{metric}_rag"].values

                    if test == "ttest":
                        statistic, pvalue = stats.ttest_rel(baseline_paired, rag_paired)
                        test_name = "Paired t-test"
                    elif test == "wilcoxon":
                        statistic, pvalue = stats.wilcoxon(baseline_paired, rag_paired)
                        test_name = "Wilcoxon signed-rank test"
                    else:
                        results["baseline_vs_rag"] = {"error": f"Unknown test: {test}"}
                        statistic = pvalue = None

                    if statistic is not None:
                        results["baseline_vs_rag"] = {
                            "test": test_name,
                            "metric": metric,
                            "baseline_mean": float(baseline_paired.mean()),
                            "rag_mean": float(rag_paired.mean()),
                            "n_pairs": len(paired_df),
                            "statistic": float(statistic),
                            "p_value": float(pvalue),
                            "significant": pvalue < 0.05
                        }
                else:
                    results["baseline_vs_rag"] = {"error": "No overlapping case_ids"}
            else:
                results["baseline_vs_rag"] = {}

            # Comparison B: manual_full vs rag_retrieval (PAIRED by case_id AND model)
            df_manual = valid_df[valid_df["context_mode"] == "manual_full"][["case_id", "model", metric]].dropna()

            if len(df_manual) > 0 and len(df_rag) > 0:
                # Merge on BOTH case_id AND model to prevent cross-model pairing
                paired_df = df_manual.merge(df_rag, on=["case_id", "model"], suffixes=("_manual", "_rag"))

                if len(paired_df) > 0:
                    manual_paired = paired_df[f"{metric}_manual"].values
                    rag_paired = paired_df[f"{metric}_rag"].values

                    if test == "ttest":
                        statistic, pvalue = stats.ttest_rel(manual_paired, rag_paired)
                        test_name = "Paired t-test"
                    elif test == "wilcoxon":
                        statistic, pvalue = stats.wilcoxon(manual_paired, rag_paired)
                        test_name = "Wilcoxon signed-rank test"
                    else:
                        results["manualfull_vs_rag"] = {"error": f"Unknown test: {test}"}
                        statistic = pvalue = None

                    if statistic is not None:
                        results["manualfull_vs_rag"] = {
                            "test": test_name,
                            "metric": metric,
                            "manual_full_mean": float(manual_paired.mean()),
                            "rag_mean": float(rag_paired.mean()),
                            "n_pairs": len(paired_df),
                            "statistic": float(statistic),
                            "p_value": float(pvalue),
                            "significant": pvalue < 0.05
                        }
                else:
                    results["manualfull_vs_rag"] = {"error": "No overlapping case_ids"}
            else:
                results["manualfull_vs_rag"] = {}

            return results

        except ImportError:
            return {"error": "scipy not installed"}
        except Exception as e:
            return {"error": str(e)}


# ================= Helper Functions =================

def quick_analysis(results: ExperimentResults, output_dir: str = "results"):
    """
    Quick analysis and export.

    Args:
        results: ExperimentResults from experiment
        output_dir: Directory to save results
    """
    analyzer = ResultsAnalyzer(results)

    # Print summary
    analyzer.print_summary()

    # Export tables
    print(f"\nExporting results to: {output_dir}")
    analyzer.export_all_tables(output_dir)

    print("\nAnalysis complete!")
