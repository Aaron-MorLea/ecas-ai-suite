from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
import torch
from src.models.lora_finetuning import LoRAFineTuner
from src.agents.langgraph_agents import ECommerceAgents
import yaml

app = FastAPI(
    title="ECAS API",
    description="E-Commerce AI Suite API for intelligent automation",
    version="0.1.0"
)

# Load config
with open("configs/deployment.yaml", 'r') as f:
    deploy_config = yaml.safe_load(f)

# Initialize components
llm_tuner = None
agents = None

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    context: Optional[dict] = None

class FraudRequest(BaseModel):
    transaction_data: dict
    user_id: str

class Response(BaseModel):
    result: str
    confidence: float
    sources: Optional[List[str]] = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and agents on startup."""
    global llm_tuner, agents
    
    # Initialize LLM (using vLLM or SGLang for inference)
    if deploy_config['inference']['vllm']['enabled']:
        from vllm import LLM
        llm_tuner = LLM(model=deploy_config['inference']['vllm']['model_path'])
    
    # Initialize agents
    agents = ECommerceAgents()

@app.post("/api/support", response_model=Response)
async def customer_support(request: QueryRequest):
    """Customer support endpoint using LangGraph agents."""
    try:
        agent = agents.create_support_agent()
        result = agent.invoke({"input": request.query})
        
        return Response(
            result=result.get("output", ""),
            confidence=0.95,
            sources=["knowledge_base", "order_system"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/fraud/analyze", response_model=Response)
async def analyze_fraud(request: FraudRequest):
    """Fraud analysis endpoint using CNN and LLM models."""
    try:
        agent = agents.create_fraud_agent()
        result = agent.invoke({"input": str(request.transaction_data)})
        
        return Response(
            result=result.get("output", ""),
            confidence=0.88,
            sources=["fraud_patterns", "user_history"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate_text(request: QueryRequest):
    """Text generation using fine-tuned LLM with vLLM/SGLang."""
    try:
        if llm_tuner:
            outputs = llm_tuner.generate([request.query], max_tokens=256)
            generated_text = outputs[0].outputs[0].text
        else:
            generated_text = "Model not loaded"
        
        return {"text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "components": ["api", "agents", "models"]}

@app.get("/api/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {"name": "llama-3.1-8b-lora", "type": "llm", "status": "loaded"},
            {"name": "fraud-cnn-v1", "type": "cnn", "status": "loaded"}
        ]
    }
