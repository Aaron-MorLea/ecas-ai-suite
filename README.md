# ECAS: E-Commerce AI Suite

> Open-source AI platform for e-commerce automation, fraud prevention, and intelligent customer support.

## Overview

ECAS provides a complete pipeline for deploying production-ready AI solutions in e-commerce environments. Built entirely with open-source tools, it integrates fine-tuned LLMs, intelligent agents, fraud detection models, and MLOps best practices.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ECAS Platform                           │
├─────────────────────────────────────────────────────────────┤
│  Data Layer: DVC + Chroma (Vector DB) + Neo4j (Graph)     │
├─────────────────────────────────────────────────────────────┤
│  Model Layer: LoRA/Llama-3.1 + P-tuning + CNN Fraud       │
├─────────────────────────────────────────────────────────────┤
│  Agent Layer: LangChain + LangGraph + RAG                 │
├─────────────────────────────────────────────────────────────┤
│  Serving Layer: FastAPI + vLLM + SGLang                   │
├─────────────────────────────────────────────────────────────┤
│  Ops Layer: Docker + K8s + MLflow + Evidently AI          │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Fine-tuned LLMs**: Llama 3.1 with LoRA (Unsloth) and P-tuning (TRL) for domain-specific tasks
- **Intelligent Agents**: Multi-agent system using LangGraph for customer support and business analysis
- **Fraud Detection**: CNN-based transaction monitoring with TensorFlow/PyTorch
- **RAG System**: Vector database (Chroma) + Knowledge Graph (Neo4j) for contextual responses
- **Scalable Inference**: vLLM and SGLang for high-concurrency API serving
- **Full MLOps**: Docker, Kubernetes, experiment tracking with MLflow, data versioning with DVC
- **Monitoring**: Data drift detection and model performance tracking with Evidently AI

## Project Structure

```
ecas-ai-suite/
├── configs/              # Training and deployment configurations
├── data/                 # Versioned datasets (DVC)
├── models/               # Trained models and LoRA adapters
├── pipelines/            # ETL, training, and inference workflows
├── src/
│   ├── data/            # Data processing and labeling
│   ├── models/          # LoRA, P-tuning, CNN implementations
│   ├── agents/          # LangChain/LangGraph agents and RAG
│   ├── api/             # FastAPI endpoints
│   └── monitoring/      # Drift detection and metrics
├── tests/               # Unit and integration tests
├── docs/                # API docs and architecture
└── .github/             # CI/CD workflows
```

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd ecas-ai-suite
cp .env.example .env  # Add your API keys

# Install dependencies
poetry install

# Run with Docker
docker-compose up -d

# Train a model
python pipelines/train_lora.py --config configs/lora_config.yaml

# Start API
uvicorn src.api.app:app --reload
```

## Technology Stack

| Component | Tools |
|-----------|-------|
| ML Frameworks | PyTorch, TensorFlow, Hugging Face, Unsloth, TRL |
| LLM Fine-tuning | LoRA, P-tuning, QLoRA |
| Agents & RAG | LangChain, LangGraph, Chroma, Neo4j |
| Serving | FastAPI, vLLM, SGLang |
| MLOps | Docker, Kubernetes, MLflow, DVC, Evidently AI |
| Image Gen | Stable Diffusion |

## Documentation

- [Architecture Design](docs/architecture.md)
- [Data Processing](docs/data_pipeline.md)
- [Model Training](docs/training.md)
- [Agent Framework](docs/agents.md)
- [API Reference](docs/api.md)
- [MLOps Guide](docs/mlops.md)
