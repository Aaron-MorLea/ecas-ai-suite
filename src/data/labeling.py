import pandas as pd
from typing import List, Dict, Tuple
import numpy as np

class DataLabeler:
    """Handles data labeling for supervised learning tasks.
    Covers: data collection and labeling plan from job requirements."""

    def __init__(self, schema: Dict):
        self.schema = schema

    def auto_label_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-label transactions based on rules (for pre-labeling)."""
        df_labeled = df.copy()
        
        # Rule-based labeling (pre-labeling for human review)
        conditions = [
            (df['amount'] > 5000) & (df['payment_method'] == 'crypto'),
            (df['amount'] > 10000),
            (df['location'] == 'high_risk_country'),
            (df['num_previous_chargebacks'] > 2)
        ]
        choices = ['fraudulent', 'fraudulent', 'suspicious', 'suspicious']
        
        df_labeled['auto_label'] = np.select(conditions, choices, default='legitimate')
        return df_labeled

    def create_labeling_tasks(self, data: List[Dict], task_type: str) -> List[Dict]:
        """Create labeling tasks for human annotators."""
        tasks = []
        
        if task_type == "fraud":
            for item in data:
                tasks.append({
                    "data": item,
                    "labels": self.schema.get("fraud_labels", []),
                    "instruction": "Classify this transaction as legitimate, suspicious, or fraudulent."
                })
        elif task_type == "support":
            for item in data:
                tasks.append({
                    "data": item,
                    "labels": self.schema.get("support_labels", []),
                    "instruction": "Classify this customer support query."
                })
        
        return tasks

    def export_to_label_studio(self, tasks: List[Dict], output_path: str):
        """Export labeling tasks to Label Studio format."""
        import json
        
        label_studio_tasks = []
        for i, task in enumerate(tasks):
            label_studio_tasks.append({
                "id": i,
                "data": {
                    "text": str(task["data"]),
                    "instruction": task["instruction"]
                },
                "meta": {
                    "labels": task["labels"]
                }
            })
        
        with open(output_path, 'w') as f:
            json.dump(label_studio_tasks, f, indent=2)
