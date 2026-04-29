from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import DatasetSummaryMetric
import pandas as pd
import numpy as np
from typing import Dict, Optional
import mlflow
import mlflow.sklearn

class ModelMonitor:
    """Monitor model performance and data drift.
    Covers: monitoring, data drift, iterative updates from job requirements."""

    def __init__(self, model_name: str = "ecas-model"):
        self.model_name = model_name
        self.mlflow_client = mlflow.tracking.MlflowClient()
        
    def detect_data_drift(self, reference_data: pd.DataFrame, 
                         current_data: pd.DataFrame) -> Dict:
        """Detect data drift using Evidently AI."""
        column_mapping = ColumnMapping()
        
        report = Report(metrics=[
            DataDriftPreset(),
            DatasetSummaryMetric(),
        ])
        
        report.run(reference_data=reference_data, 
                  current_data=current_data,
                  column_mapping=column_mapping)
        
        result = report.as_dict()
        
        # Extract drift summary
        drift_summary = {
            "dataset_drift": result['metrics'][0]['result']['dataset_drift'],
            "number_of_drifted_columns": result['metrics'][0]['result']['number_of_drifted_columns'],
            "share_of_drifted_columns": result['metrics'][0]['result']['share_of_drifted_columns']
        }
        
        return drift_summary

    def monitor_model_performance(self, y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 metrics: list = ["accuracy", "precision", "recall"]) -> Dict:
        """Monitor model performance metrics over time."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        results = {}
        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(y_true, y_pred)
        if "precision" in metrics:
            results["precision"] = precision_score(y_true, y_pred, average='weighted')
        if "recall" in metrics:
            results["recall"] = recall_score(y_true, y_pred, average='weighted')
        
        # Log to MLflow
        with mlflow.start_run(run_name="performance_monitoring"):
            for metric_name, value in results.items():
                mlflow.log_metric(metric_name, value)
        
        return results

    def log_model_version(self, model, model_type: str = "sklearn"):
        """Log model version to MLflow for iterative updates."""
        with mlflow.start_run(run_name=f"{self.model_name}_version"):
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, "model")
            else:
                mlflow.pyfunc.log_model(model, "model")
            
            # Log model version
            version = self.mlflow_client.create_model_version(
                name=self.model_name,
                source=mlflow.get_artifact_uri("model"),
                run_id=mlflow.active_run().info.run_id
            )
        
        return version

    def should_retrain(self, drift_metrics: Dict, 
                       performance_metrics: Dict,
                       thresholds: Optional[Dict] = None) -> bool:
        """Determine if model needs retraining based on drift and performance."""
        if thresholds is None:
            thresholds = {
                "share_of_drifted_columns": 0.3,
                "accuracy_drop": 0.05
            }
        
        # Check drift threshold
        if drift_metrics.get("share_of_drifted_columns", 0) > thresholds["share_of_drifted_columns"]:
            return True
        
        # Check performance drop (would need historical comparison)
        # This is simplified - in practice, compare with baseline
        
        return False

    def create_monitoring_report(self, reference_data: pd.DataFrame,
                                current_data: pd.DataFrame,
                                output_path: str = "reports/drift_report.html"):
        """Generate HTML report with drift analysis."""
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            DatasetSummaryMetric(),
        ])
        
        report.run(reference_data=reference_data, 
                  current_data=current_data)
        
        report.save_html(output_path)
        return output_path
