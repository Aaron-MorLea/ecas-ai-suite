import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class FraudCNN:
    """CNN-based fraud detection model using TensorFlow.
    Covers: TensorFlow, CNN, algorithm selection from job requirements."""

    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """Build CNN architecture for fraud pattern detection."""
        model = models.Sequential([
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model

    def compile_model(self, learning_rate=0.001):
        """Compile model with optimizer, loss, and metrics."""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return self.model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the CNN model."""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self._get_callbacks()
        )
        return history

    def _get_callbacks(self):
        """Setup training callbacks."""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/fraud_cnn_best.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3
            )
        ]

    def evaluate(self, X_test, y_test):
        """Evaluate model with precision and recall."""
        results = self.model.evaluate(X_test, y_test)
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }
        return metrics
