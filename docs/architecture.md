# ECAS Architecture

## System Overview

ECAS (E-Commerce AI Suite) is a complete AI platform for e-commerce automation. This document describes the architecture and design decisions.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       User Interface                        │
│              (Web App, Mobile App, API Clients)            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                     │
│  /api/support  │  /api/fraud/analyze  │  /api/generate   │
└─────────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Agent Layer    │ │  Model Layer   │ │  Data Layer     │
│  LangGraph     │ │  LoRA/P-tuning │ │  Chroma (Vector)│
│  LangChain     │ │  CNN (Fraud)   │ │  Neo4j (Graph)  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Ops Layer (MLOps)                       │
│  Docker │ Kubernetes │ MLflow │ DVC │ Evidently AI        │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Layer
- **Chroma**: Vector database for RAG (document retrieval)
- **Neo4j**: Knowledge graph for relationships between products, users, transactions
- **DVC**: Data version control for reproducibility

### 2. Model Layer
- **LoRA (Unsloth)**: Efficient fine-tuning of Llama 3.1-8B
- **P-tuning**: Soft prompt tuning for task adaptation
- **CNN (TensorFlow)**: Fraud pattern detection in transactions
- **Stable Diffusion**: Product image generation

### 3. Agent Layer
- **LangChain**: Tool integration and prompt management
- **LangGraph**: Multi-agent orchestration with state management
- **RAG System**: Combines vector search and graph queries

### 4. API Layer
- **FastAPI**: RESTful API endpoints
- **vLLM/SGLang**: High-performance inference serving

### 5. Ops Layer
- **Docker**: Containerization
- **Kubernetes**: Orchestration and scaling
- **MLflow**: Experiment tracking and model registry
- **Evidently AI**: Data drift and model performance monitoring

## Design Decisions

1. **Open Source First**: All components are open-source to meet job requirements
2. **Modular Design**: Each component can be developed and tested independently
3. **Scalability**: vLLM and SGLang enable high-concurrency inference
4. **Monitoring**: Built-in drift detection and performance tracking for iterative updates
