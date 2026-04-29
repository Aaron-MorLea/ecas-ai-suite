#!/usr/bin/env python3
"""
Inference Pipeline with vLLM/SGLang
Covers: Inference acceleration, high concurrency support
"""
from fastapi import FastAPI
import uvicorn
import yaml

app = FastAPI(title="ECAS Inference Pipeline")

@app.post("/predict")
async def predict(query: str, model_type: str = "llm"):
    """Run inference with optimized serving."""
    if model_type == "llm":
        from vllm import LLM, SamplingParams
        llm = LLM(model="models/final_model")
        sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
        outputs = llm.generate([query], sampling_params)
        return {"result": outputs[0].outputs[0].text}
    else:
        return {"result": "Model type not supported"}

@app.post("/fraud/predict")
async def predict_fraud(transaction: dict):
    """Predict fraud using CNN model."""
    import tensorflow as tf
    model = tf.keras.models.load_model("models/fraud_cnn_final.h5")
    
    # Preprocess transaction
    import numpy as np
    features = np.array([[transaction.get('amount', 0), 
                         transaction.get('quantity', 0)]])
    
    prediction = model.predict(features)
    fraud_class = prediction.argmax(axis=1)[0]
    
    return {
        "fraud_class": int(fraud_class),
        "confidence": float(prediction[0][fraud_class])
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
