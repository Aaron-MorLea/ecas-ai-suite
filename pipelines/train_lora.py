#!/usr/bin/env python3
"""
LoRA Fine-tuning Pipeline
Covers: Fine-tuning with LoRA, parameter optimization, loss functions
"""
import argparse
import yaml
from src.models.lora_finetuning import LoRAFineTuner
from src.data.processing import DataProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train LLM with LoRA")
    parser.add_argument("--config", type=str, default="configs/lora_config.yaml",
                       help="Path to config file")
    parser.add_argument("--data-path", type=str, default="data/processed/training_data.json",
                       help="Path to training data")
    args = parser.parse_args()
    
    logger.info("Starting LoRA fine-tuning pipeline")
    
    # Initialize components
    tuner = LoRAFineTuner(config_path=args.config)
    processor = DataProcessor()
    
    # Load and process data
    logger.info("Loading and processing data")
    import json
    with open(args.data_path, 'r') as f:
        raw_data = json.load(f)
    
    processed_data = processor.prepare_llm_training_data(raw_data)
    
    # Train model
    logger.info("Starting training")
    trainer = tuner.train()
    
    logger.info("Training completed successfully")
    logger.info(f"Model saved to: models/final_model")

if __name__ == "__main__":
    main()
