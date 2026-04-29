#!/usr/bin/env python3
"""
CNN Fraud Detection Training Pipeline
Covers: TensorFlow, CNN, precision/recall evaluation
"""
import argparse
import numpy as np
from src.models.cnn_fraud import FraudCNN
from src.data.processing import DataProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train CNN for fraud detection")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    args = parser.parse_args()
    
    logger.info("Starting CNN training pipeline")
    
    # Initialize model
    fraud_cnn = FraudCNN(input_shape=(224, 224, 3), num_classes=3)
    fraud_cnn.compile_model(learning_rate=0.001)
    
    # Load and process data
    processor = DataProcessor()
    logger.info("Loading transaction data")
    
    # Simulated data (replace with actual data loading)
    X_train = np.random.rand(1000, 224, 224, 3)
    y_train = np.random.randint(0, 3, 1000)
    X_val = np.random.rand(200, 224, 224, 3)
    y_val = np.random.randint(0, 3, 200)
    
    # Train model
    logger.info("Training CNN model")
    history = fraud_cnn.train(X_train, y_train, X_val, y_val,
                             epochs=args.epochs, batch_size=args.batch_size)
    
    # Evaluate
    X_test = np.random.rand(300, 224, 224, 3)
    y_test = np.random.randint(0, 3, 300)
    metrics = fraud_cnn.evaluate(X_test, y_test)
    
    logger.info(f"Test Results - Accuracy: {metrics['accuracy']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}")
    
    # Save model
    fraud_cnn.model.save("models/fraud_cnn_final.h5")
    logger.info("Model saved successfully")

if __name__ == "__main__":
    main()
