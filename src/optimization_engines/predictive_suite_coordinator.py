import os
import pandas as pd
from typing import Dict, Any
from src.optimization_engines.virtual_metrology_optimizer import VirtualMetrologyOptimizer
from src.optimization_engines.causal_attribution_engine import CausalAttributionEngine
from src.optimization_engines.wafer_sort_early_exit_optimizer import WaferSortEarlyExitOptimizer
from src.optimization_engines.adaptive_routing_optimizer import AdaptiveRoutingOptimizer
from src.optimization_engines.packaging_kitting_optimizer import PackagingKittingOptimizer
from src.optimization_engines.manufacturing_insight_engine import ManufacturingInsightEngine
from src.optimization_engines.tester_velocity_optimizer import TesterVelocityOptimizer

class PredictiveSuiteCoordinator:
    def __init__(self, data_directory: str):
        """
        The orchestrator for the Wafer Process Analysis predictive suite.
        Coordinates and manages dependencies across the inline optimization layer.
        """
        self.data_dir = data_directory
        self.wafer_data_path = os.path.join(data_directory, "wafer_sort_data.csv")
        self.package_data_path = os.path.join(data_directory, "silicon_packaged_data_groups.csv")

    def run_stage_1_wafer_sort_pipeline(self) -> Dict[str, Any]:
        """Runs the inline wafer metrics and causal failure optimization loops."""
        if not os.path.exists(self.wafer_data_path):
            # Safe runtime fallback if operating entirely on a single consolidated database archive
            df = pd.read_csv(self.package_data_path)
        else:
            df = pd.read_csv(self.wafer_data_path)

        metrology_opt = VirtualMetrologyOptimizer(df)
        causal_eng = CausalAttributionEngine(df)
        early_exit_opt = WaferSortEarlyExitOptimizer(df)

        processed_wafer = metrology_opt.process_spatial_anomalies()
        pass_probs = metrology_opt.train_virtual_metrology()
        exit_flags = early_exit_opt.evaluate_exit_criteria()
        importances, labels = causal_eng.get_feature_importances()

        return {
            "dataframe": processed_wafer,
            "pass_probabilities": pass_probs,
            "early_exit_triggered": bool(exit_flags.any()),
            "root_cause_weights": dict(zip(labels, importances))
        }

    def run_stage_2_packaged_process_pipeline(self) -> Dict[str, Any]:
        """Runs the assembly corner kitting, adaptive routing, and cell velocity optimizations."""
        df = pd.read_csv(self.package_data_path)

        routing_opt = AdaptiveRoutingOptimizer(df)
        kitting_opt = PackagingKittingOptimizer(df)
        insight_eng = ManufacturingInsightEngine(df)
        velocity_opt = TesterVelocityOptimizer(df)

        # Execute unified machine learning and statistical runs
        routing_opt.train_models()
        kitting_bins = kitting_opt.compute_kitting_profiles()
        velocity_stats = velocity_opt.calculate_throughput_velocity()
        time_saved = routing_opt.calculate_saved_test_time()

        return {
            "dataframe": df,
            "kitting_bins": kitting_bins,
            "saved_time_percentage": time_saved,
            "velocity_telemetry": velocity_stats
        }
