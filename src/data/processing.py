import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data cleaning, normalization, and preparation for ML models.
    Covers: data cleaning, normalization, labeling plan from job requirements."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path) if config_path else {}

    def clean_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize transaction data for fraud detection."""
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = df_clean.fillna({
            'amount': df_clean['amount'].median(),
            'user_id': 'unknown',
            'timestamp': pd.Timestamp.now()
        })
        
        # Normalize numerical features
        numerical_cols = ['amount', 'quantity', 'price']
        for col in numerical_cols:
            if col in df_clean.columns:
                df_clean[f'{col}_normalized'] = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
        
        # Encode categorical variables
        categorical_cols = ['payment_method', 'category', 'location']
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[f'{col}_encoded'] = pd.Categorical(df_clean[col]).codes
        
        logger.info(f"Cleaned data shape: {df_clean.shape}")
        return df_clean

    def prepare_llm_training_data(self, conversations: List[Dict]) -> List[Dict]:
        """Prepare conversation data for LLM fine-tuning (LoRA/P-tuning)."""
        formatted_data = []
        
        for conv in conversations:
            # Apply Alpaca template or custom format
            formatted = {
                "instruction": conv.get("instruction", ""),
                "input": conv.get("input", ""),
                "output": conv.get("output", "")
            }
            formatted_data.append(formatted)
        
        return formatted_data

    def create_labeling_schema(self) -> Dict:
        """Define data labeling plan for supervised learning."""
        return {
            "fraud_labels": ["legitimate", "suspicious", "fraudulent"],
            "support_labels": ["inquiry", "complaint", "refund_request", "technical_issue"],
            "priority_levels": ["low", "medium", "high", "critical"],
            "labeling_guidelines": "docs/labeling_guidelines.md"
        }

    def _load_config(self, config_path: str) -> Dict:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
