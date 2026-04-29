# Data Pipeline Documentation

## Overview

The data pipeline handles data collection, cleaning, labeling, and preparation for ML models.

## Data Sources

1. **Transaction Data**: E-commerce transactions (amount, user, timestamp, payment method)
2. **Customer Support Logs**: Chat logs, email inquiries, ticket data
3. **Product Catalog**: Product descriptions, images, categories
4. **Knowledge Base**: Policy documents, FAQs, manuals

## Data Processing Steps

### 1. Data Collection
```python
from src.data.processing import DataProcessor
processor = DataProcessor()
```

### 2. Data Cleaning
- Handle missing values
- Normalize numerical features
- Encode categorical variables
- Remove duplicates

### 3. Data Labeling
- **Fraud Labels**: Legitimate, Suspicious, Fraudulent
- **Support Labels**: Inquiry, Complaint, Refund Request, Technical Issue
- **Tools**: Label Studio integration, auto-labeling with rules

### 4. Dataset Preparation
- **LLM Training**: Format conversations with Alpaca template
- **CNN Training**: Convert to image-like sequences for fraud patterns
- **RAG**: Chunk documents, generate embeddings

## Data Versioning (DVC)

```bash
# Initialize DVC
dvc init

# Add data to DVC
dvc add data/raw/transactions.csv
dvc add data/processed/training_data.json

# Push to remote
dvc push
```

## Data Schema

### Transaction Data
```json
{
  "transaction_id": "string",
  "user_id": "string",
  "amount": "float",
  "timestamp": "datetime",
  "payment_method": "string",
  "location": "string",
  "items": ["item_id"],
  "label": "string"  // legitimate, suspicious, fraudulent
}
```

### Conversation Data (for LLM)
```json
{
  "instruction": "string",
  "input": "string",
  "output": "string"
}
```
