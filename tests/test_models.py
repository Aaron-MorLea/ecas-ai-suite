import pytest
import numpy as np
from src.models.cnn_fraud import FraudCNN
from src.models.lora_finetuning import LoRAFineTuner

class TestFraudCNN:
    """Test CNN fraud detection model."""
    
    def setup_method(self):
        self.model = FraudCNN(input_shape=(224, 224, 3), num_classes=3)
        self.model.compile_model()
    
    def test_model_creation(self):
        """Test model is created successfully."""
        assert self.model.model is not None
    
    def test_prediction_shape(self):
        """Test prediction output shape."""
        test_input = np.random.rand(1, 224, 224, 3)
        prediction = self.model.model.predict(test_input)
        assert prediction.shape == (1, 3)
    
    def test_metrics_computation(self):
        """Test precision and recall metrics."""
        X_test = np.random.rand(100, 224, 224, 3)
        y_test = np.random.randint(0, 3, 100)
        metrics = self.model.evaluate(X_test, y_test)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1


class TestLoRAFineTuner:
    """Test LoRA fine-tuning (integration test - requires model files)."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        tuner = LoRAFineTuner(config_path="configs/lora_config.yaml")
        assert tuner.config is not None
        assert 'model' in tuner.config
        assert 'lora' in tuner.config
    
    def test_model_selection(self):
        """Test that config specifies correct frameworks."""
        with open("configs/model_config.yaml", 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        frameworks = [f['name'] for f in config['frameworks']]
        assert 'pytorch' in frameworks
        assert 'tensorflow' in frameworks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
