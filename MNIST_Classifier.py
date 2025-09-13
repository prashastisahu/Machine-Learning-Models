#!/usr/bin/env python3
"""
MNIST Digit Classifier
======================

A complete implementation of a neural network for handwritten digit recognition
using TensorFlow/Keras on the MNIST dataset.

Features:
- Data loading and preprocessing
- Neural network training with configurable architecture
- Model evaluation and visualization
- Image prediction capabilities
- Interactive drawing interface (optional)
- Model saving and loading
- Comprehensive training history plots

Author: Enhanced implementation
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
import base64
from io import BytesIO
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


class MNISTClassifier:
    """Complete MNIST digit classification system"""
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the classifier
        
        Args:
            use_gpu: Whether to use GPU for training (if available)
        """
        # Set device configuration
        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            
        # Initialize model and data
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.history = None
        self.original_shape = None
        
        # Configure TensorFlow
        tf.random.set_seed(42)
        np.random.seed(42)
        
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [-1, 1] range
        
        Args:
            data: Image data with pixel values 0-255
            
        Returns:
            Normalized data in range [-1, 1]
        """
        return (data / 255.0) * 2 - 1
    
    def load_data(self, validation_split: float = 0.0) -> None:
        """
        Load and preprocess MNIST dataset
        
        Args:
            validation_split: Fraction of training data to use for validation
        """
        print("Loading MNIST dataset...")
        
        try:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        except Exception as e:
            print(f"Error loading MNIST dataset: {e}")
            print("Downloading dataset...")
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Store original shape for later use
        self.original_shape = x_train.shape[1:]
        
        # Reshape and convert to float32
        x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
        x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)
        
        # Normalize data
        x_train = self.normalize_data(x_train)
        x_test = self.normalize_data(x_test)
        
        # Split validation data if requested
        if validation_split > 0:
            val_size = int(len(x_train) * validation_split)
            indices = np.random.permutation(len(x_train))
            
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            
            self.x_val = x_train[val_indices]
            self.y_val = y_train[val_indices]
            self.x_train = x_train[train_indices]
            self.y_train = y_train[train_indices]
        else:
            self.x_train = x_train
            self.y_train = y_train
            self.x_val = None
            self.y_val = None
            
        self.x_test = x_test
        self.y_test = y_test
        
        print(f"Training samples: {len(self.x_train)}")
        if self.x_val is not None:
            print(f"Validation samples: {len(self.x_val)}")
        print(f"Test samples: {len(self.x_test)}")
        print(f"Input shape: {self.x_train.shape[1:]}")
        
    def create_model(self, architecture: str = 'simple', dropout_rate: float = 0.3) -> None:
        """
        Create neural network model
        
        Args:
            architecture: Model architecture ('simple', 'deep', or 'batch_norm')
            dropout_rate: Dropout rate for regularization
        """
        input_dim = self.x_train.shape[1]
        
        self.model = Sequential()
        
        if architecture == 'simple':
            # Original simple architecture
            self.model.add(Dense(512, activation='relu', input_shape=(input_dim,), name='hidden1'))
            self.model.add(Dense(512, activation='relu', name='hidden2'))
            self.model.add(Dense(10, activation='softmax', name='output'))
            
        elif architecture == 'deep':
            # Deeper network with dropout
            self.model.add(Dense(512, activation='relu', input_shape=(input_dim,), name='hidden1'))
            self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(256, activation='relu', name='hidden2'))
            self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(128, activation='relu', name='hidden3'))
            self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(10, activation='softmax', name='output'))
            
        elif architecture == 'batch_norm':
            # Network with batch normalization
            self.model.add(Dense(512, input_shape=(input_dim,), name='hidden1'))
            self.model.add(BatchNormalization())
            self.model.add(tf.keras.layers.Activation('relu'))
            self.model.add(Dropout(dropout_rate))
            
            self.model.add(Dense(256, name='hidden2'))
            self.model.add(BatchNormalization())
            self.model.add(tf.keras.layers.Activation('relu'))
            self.model.add(Dropout(dropout_rate))
            
            self.model.add(Dense(10, activation='softmax', name='output'))
            
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
            
        print(f"\nModel Architecture: {architecture}")
        self.model.summary()
        
    def compile_model(self, optimizer_name: str = 'sgd', learning_rate: float = 0.01) -> None:
        """
        Compile the model with specified optimizer
        
        Args:
            optimizer_name: Optimizer to use ('sgd', 'adam', 'rmsprop')
            learning_rate: Learning rate for optimizer
        """
        if optimizer_name.lower() == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_name.lower() == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        print(f"Model compiled with {optimizer_name} optimizer (lr={learning_rate})")
        
    def train(self, epochs: int = 25, batch_size: int = 128, 
              use_callbacks: bool = True, validation_data: Optional[str] = 'test') -> None:
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            use_callbacks: Whether to use early stopping and learning rate reduction
            validation_data: Validation data to use ('test', 'validation', or None)
        """
        print(f"\nStarting training for {epochs} epochs...")
        
        # Determine validation data
        if validation_data == 'test':
            val_data = (self.x_test, self.y_test)
        elif validation_data == 'validation' and self.x_val is not None:
            val_data = (self.x_val, self.y_val)
        else:
            val_data = None
            
        # Setup callbacks
        callbacks = []
        if use_callbacks and val_data is not None:
            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
            lr_reducer = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
            callbacks = [early_stopping, lr_reducer]
            
        # Train model
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating model...")
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Get predictions for detailed analysis
        y_pred = self.model.predict(self.x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate per-class accuracy
        per_class_accuracy = {}
        for digit in range(10):
            mask = self.y_test == digit
            if np.sum(mask) > 0:
                digit_accuracy = np.mean(y_pred_classes[mask] == digit)
                per_class_accuracy[digit] = digit_accuracy
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'per_class_accuracy': per_class_accuracy
        }
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        print("\nPer-digit accuracy:")
        for digit, accuracy in per_class_accuracy.items():
            print(f"  Digit {digit}: {accuracy:.4f}")
            
        return results
        
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
            
        history_df = pd.DataFrame(self.history.history)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Loss plot
        axes[0, 0].plot(history_df['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history_df.columns:
            axes[0, 0].plot(history_df['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(history_df['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history_df.columns:
            axes[0, 1].plot(history_df['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        if 'lr' in history_df.columns:
            axes[1, 0].plot(history_df['lr'], linewidth=2, color='orange')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Learning Rate')
        
        # Training metrics summary
        final_train_acc = history_df['accuracy'].iloc[-1]
        final_val_acc = history_df['val_accuracy'].iloc[-1] if 'val_accuracy' in history_df.columns else None
        best_val_acc = history_df['val_accuracy'].max() if 'val_accuracy' in history_df.columns else None
        
        summary_text = f"Final Training Accuracy: {final_train_acc:.4f}\n"
        if final_val_acc is not None:
            summary_text += f"Final Validation Accuracy: {final_val_acc:.4f}\n"
        if best_val_acc is not None:
            summary_text += f"Best Validation Accuracy: {best_val_acc:.4f}\n"
        summary_text += f"Total Epochs: {len(history_df)}"
        
        axes[1, 1].text(0.1, 0.7, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
            
        plt.show()
        
    def predict_image(self, image_data, input_type: str = 'array') -> Tuple[int, np.ndarray]:
        """
        Predict digit from image data
        
        Args:
            image_data: Image data (array, PIL Image, or base64 string)
            input_type: Type of input ('array', 'pil', 'base64')
            
        Returns:
            Tuple of (predicted_digit, confidence_scores)
        """
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
            
        # Process input based on type
        if input_type == 'base64':
            # Decode base64 image
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            image = image.convert('L')  # Convert to grayscale
            image = image.resize((28, 28))
            image_array = np.array(image).astype(np.float32)
            
        elif input_type == 'pil':
            # PIL Image
            image = image_data.convert('L').resize((28, 28))
            image_array = np.array(image).astype(np.float32)
            
        elif input_type == 'array':
            # NumPy array
            image_array = image_data.astype(np.float32)
            if image_array.shape != (28, 28):
                raise ValueError(f"Array must be 28x28, got {image_array.shape}")
                
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
        
        # Preprocess image
        image_array = self.normalize_data(image_array)
        image_array = image_array.reshape(1, -1)  # Flatten and add batch dimension
        
        # Make prediction
        predictions = self.model.predict(image_array, verbose=0)[0]
        predicted_digit = np.argmax(predictions)
        
        return predicted_digit, predictions
        
    def create_drawing_interface(self) -> None:
        """
        Create a simple drawing interface for testing predictions
        """
        try:
            import tkinter as tk
            from tkinter import ttk, messagebox
        except ImportError:
            print("tkinter not available. Cannot create drawing interface.")
            return
            
        def paint(event):
            # Paint on canvas
            x1, y1 = (event.x - 5), (event.y - 5)
            x2, y2 = (event.x + 5), (event.y + 5)
            canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white', width=3)
            
            # Paint on PIL image
            draw.ellipse([x1, y1, x2, y2], fill='white')
            
        def predict_drawing():
            # Resize and predict
            resized = pil_image.resize((28, 28))
            image_array = np.array(resized)[:, :, 0]  # Take first channel
            
            try:
                predicted_digit, confidence = self.predict_image(image_array, 'array')
                
                # Show results
                result_text.set(f"Predicted Digit: {predicted_digit}")
                confidence_text.set(f"Confidence: {confidence[predicted_digit]:.3f}")
                
                # Show confidence for all digits
                conf_details = "\n".join([f"Digit {i}: {conf:.3f}" 
                                        for i, conf in enumerate(confidence)])
                confidence_details.set(conf_details)
                
            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {e}")
                
        def clear_canvas():
            canvas.delete("all")
            draw.rectangle([0, 0, 280, 280], fill='black')
            result_text.set("Predicted Digit: -")
            confidence_text.set("Confidence: -")
            confidence_details.set("")
        
        # Create GUI
        root = tk.Tk()
        root.title("MNIST Digit Classifier - Draw Interface")
        root.geometry("600x400")
        
        # Create canvas
        canvas_frame = ttk.Frame(root)
        canvas_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        canvas = tk.Canvas(canvas_frame, width=280, height=280, bg='black')
        canvas.pack()
        canvas.bind('<B1-Motion>', paint)
        canvas.bind('<Button-1>', paint)
        
        # Create PIL image for prediction
        pil_image = Image.new('RGB', (280, 280), 'black')
        draw = ImageDraw.Draw(pil_image)
        
        # Control panel
        control_frame = ttk.Frame(root)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        ttk.Label(control_frame, text="Draw a digit (0-9)", font=('Arial', 12, 'bold')).pack(pady=5)
        
        predict_btn = ttk.Button(control_frame, text="Predict", command=predict_drawing)
        predict_btn.pack(pady=5, fill=tk.X)
        
        clear_btn = ttk.Button(control_frame, text="Clear", command=clear_canvas)
        clear_btn.pack(pady=5, fill=tk.X)
        
        # Results
        result_text = tk.StringVar(value="Predicted Digit: -")
        ttk.Label(control_frame, textvariable=result_text, font=('Arial', 14, 'bold')).pack(pady=10)
        
        confidence_text = tk.StringVar(value="Confidence: -")
        ttk.Label(control_frame, textvariable=confidence_text, font=('Arial', 12)).pack()
        
        ttk.Label(control_frame, text="All Confidences:", font=('Arial', 10, 'bold')).pack(pady=(20, 5))
        confidence_details = tk.StringVar(value="")
        ttk.Label(control_frame, textvariable=confidence_details, font=('Courier', 9)).pack()
        
        print("Drawing interface opened. Draw a digit and click 'Predict'!")
        root.mainloop()
        
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str) -> None:
        """Load a trained model"""
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='MNIST Digit Classifier')
    
    parser.add_argument('--architecture', choices=['simple', 'deep', 'batch_norm'], 
                       default='simple', help='Model architecture')
    parser.add_argument('--optimizer', choices=['sgd', 'adam', 'rmsprop'], 
                       default='sgd', help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.01, 
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=25, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='Batch size for training')
    parser.add_argument('--dropout_rate', type=float, default=0.3, 
                       help='Dropout rate for regularization')
    parser.add_argument('--use_gpu', action='store_true', 
                       help='Use GPU for training')
    parser.add_argument('--save_model', type=str, default='mnist_model.h5',
                       help='Path to save trained model')
    parser.add_argument('--load_model', type=str, 
                       help='Path to load pre-trained model')
    parser.add_argument('--no_training', action='store_true',
                       help='Skip training (use with --load_model)')
    parser.add_argument('--drawing_interface', action='store_true',
                       help='Open drawing interface for testing')
    parser.add_argument('--validation_split', type=float, default=0.0,
                       help='Fraction of training data for validation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MNIST DIGIT CLASSIFIER")
    print("="*60)
    
    # Initialize classifier
    classifier = MNISTClassifier(use_gpu=args.use_gpu)
    
    # Load data
    classifier.load_data(validation_split=args.validation_split)
    
    if args.load_model and not args.no_training:
        # Load pre-trained model and continue training
        classifier.load_model(args.load_model)
    elif args.load_model and args.no_training:
        # Load pre-trained model without training
        classifier.load_model(args.load_model)
    else:
        # Create and train new model
        classifier.create_model(args.architecture, args.dropout_rate)
        classifier.compile_model(args.optimizer, args.learning_rate)
        
    if not args.no_training:
        # Train model
        validation_data = 'validation' if args.validation_split > 0 else 'test'
        classifier.train(args.epochs, args.batch_size, 
                        use_callbacks=True, validation_data=validation_data)
        
        # Save model
        classifier.save_model(args.save_model)
        
        # Plot training history
        classifier.plot_training_history()
    
    # Evaluate model
    results = classifier.evaluate()
    
    # Test prediction on random samples
    print("\n" + "="*40)
    print("SAMPLE PREDICTIONS")
    print("="*40)
    
    # Test on a few random samples
    for i in range(5):
        idx = np.random.randint(0, len(classifier.x_test))
        test_image = classifier.x_test[idx].reshape(28, 28)
        true_label = classifier.y_test[idx]
        
        predicted_digit, confidence = classifier.predict_image(test_image, 'array')
        
        print(f"Sample {i+1}: True={true_label}, Predicted={predicted_digit}, "
              f"Confidence={confidence[predicted_digit]:.3f}")
    
    # Open drawing interface if requested
    if args.drawing_interface:
        print("\nOpening drawing interface...")
        classifier.create_drawing_interface()
    
    print("\n" + "="*60)
    print("CLASSIFICATION COMPLETE")
    print("="*60)
    
    return classifier, results


if __name__ == "__main__":
    # Example usage when run as script
    classifier, results = main()
