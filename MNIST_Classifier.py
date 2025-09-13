#!/usr/bin/env python3
"""
MNIST Digit Classifier - PyTorch Implementation
===============================================

A complete implementation of a neural network for handwritten digit recognition
using PyTorch on the MNIST dataset.

Features:
- Data loading and preprocessing with PyTorch DataLoaders
- Neural network training with configurable architecture
- Model evaluation and visualization
- Image prediction capabilities
- Interactive drawing interface (optional)
- Model saving and loading
- Comprehensive training history plots

Author: Enhanced PyTorch implementation
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
from typing import Tuple, Optional, Dict, Any, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import torchvision


class MNISTNet(nn.Module):
    """Neural Network architectures for MNIST classification"""
    
    def __init__(self, architecture: str = 'simple', dropout_rate: float = 0.3):
        """
        Initialize the neural network
        
        Args:
            architecture: Model architecture ('simple', 'deep', or 'batch_norm')
            dropout_rate: Dropout rate for regularization
        """
        super(MNISTNet, self).__init__()
        
        self.architecture = architecture
        self.dropout_rate = dropout_rate
        
        if architecture == 'simple':
            # Original simple architecture
            self.fc1 = nn.Linear(28 * 28, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 10)
            
        elif architecture == 'deep':
            # Deeper network with dropout
            self.fc1 = nn.Linear(28 * 28, 512)
            self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(512, 256)
            self.dropout2 = nn.Dropout(dropout_rate)
            self.fc3 = nn.Linear(256, 128)
            self.dropout3 = nn.Dropout(dropout_rate)
            self.fc4 = nn.Linear(128, 10)
            
        elif architecture == 'batch_norm':
            # Network with batch normalization
            self.fc1 = nn.Linear(28 * 28, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.dropout2 = nn.Dropout(dropout_rate)
            self.fc3 = nn.Linear(256, 10)
            
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def forward(self, x):
        """Forward pass through the network"""
        x = x.view(-1, 28 * 28)  # Flatten the input
        
        if self.architecture == 'simple':
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.log_softmax(self.fc3(x), dim=1)
            
        elif self.architecture == 'deep':
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            x = F.relu(self.fc3(x))
            x = self.dropout3(x)
            x = F.log_softmax(self.fc4(x), dim=1)
            
        elif self.architecture == 'batch_norm':
            x = self.fc1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = F.log_softmax(self.fc3(x), dim=1)
        
        return x


class MNISTClassifier:
    """Complete MNIST digit classification system using PyTorch"""
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the classifier
        
        Args:
            use_gpu: Whether to use GPU for training (if available)
        """
        # Set device configuration
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
            
        # Initialize model and data
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.optimizer = None
        self.scheduler = None
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
    
    def normalize_transform(self) -> transforms.Compose:
        """
        Create normalization transform
        
        Returns:
            Transform that normalizes to [-1, 1] range
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
    
    def load_data(self, batch_size: int = 128, validation_split: float = 0.0) -> None:
        """
        Load and preprocess MNIST dataset
        
        Args:
            batch_size: Batch size for data loaders
            validation_split: Fraction of training data to use for validation
        """
        print("Loading MNIST dataset...")
        
        transform = self.normalize_transform()
        
        # Load training data
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        
        # Split into train and validation if requested
        if validation_split > 0:
            val_size = int(len(train_dataset) * validation_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            print(f"Validation samples: {len(val_dataset)}")
        else:
            self.val_loader = None
            
        # Load test data
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_dataset) if validation_split == 0 else train_size}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Batch size: {batch_size}")
    
    def create_model(self, architecture: str = 'simple', dropout_rate: float = 0.3) -> None:
        """
        Create neural network model
        
        Args:
            architecture: Model architecture ('simple', 'deep', or 'batch_norm')
            dropout_rate: Dropout rate for regularization
        """
        self.model = MNISTNet(architecture, dropout_rate).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\nModel Architecture: {architecture}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Print model structure
        print("\nModel Structure:")
        print(self.model)
    
    def compile_model(self, optimizer_name: str = 'sgd', learning_rate: float = 0.01,
                     use_scheduler: bool = True) -> None:
        """
        Setup optimizer and learning rate scheduler
        
        Args:
            optimizer_name: Optimizer to use ('sgd', 'adam', 'rmsprop')
            learning_rate: Learning rate for optimizer
            use_scheduler: Whether to use learning rate scheduler
        """
        if optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, 
                                     momentum=0.9, weight_decay=1e-4)
        elif optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate,
                                      weight_decay=1e-4)
        elif optimizer_name.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate,
                                         weight_decay=1e-4)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Setup learning rate scheduler
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        else:
            self.scheduler = None
            
        print(f"Model compiled with {optimizer_name} optimizer (lr={learning_rate})")
        if use_scheduler:
            print("Learning rate scheduler enabled")
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = F.nll_loss(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate on given data loader
        
        Args:
            data_loader: Data loader for validation
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                total_loss += F.nll_loss(output, target, reduction='sum').item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, epochs: int = 25, early_stopping_patience: int = 5) -> None:
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
        """
        print(f"\nStarting training for {epochs} epochs...")
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            if self.val_loader is not None:
                val_loss, val_acc = self.validate_epoch(self.val_loader)
            else:
                val_loss, val_acc = self.validate_epoch(self.test_loader)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:3d}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | '
                  f'LR: {current_lr:.6f}')
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience and epoch > 10:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        print("Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating model...")
        
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        per_class_correct = torch.zeros(10)
        per_class_total = torch.zeros(10)
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Per-class accuracy
                for i in range(10):
                    class_mask = target == i
                    per_class_correct[i] += pred[class_mask].eq(target[class_mask].view_as(pred[class_mask])).sum().item()
                    per_class_total[i] += class_mask.sum().item()
        
        # Calculate metrics
        test_loss /= total
        test_accuracy = 100.0 * correct / total
        
        per_class_accuracy = {}
        for digit in range(10):
            if per_class_total[digit] > 0:
                accuracy = 100.0 * per_class_correct[digit] / per_class_total[digit]
                per_class_accuracy[digit] = accuracy.item()
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'per_class_accuracy': per_class_accuracy
        }
        
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        
        print("\nPer-digit accuracy:")
        for digit, accuracy in per_class_accuracy.items():
            print(f"  Digit {digit}: {accuracy:.2f}%")
            
        return results
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.history['train_loss']:
            print("No training history available. Train the model first.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        if self.scheduler is not None:
            lrs = []
            temp_scheduler = optim.lr_scheduler.StepLR(
                optim.SGD([torch.tensor(0., requires_grad=True)], lr=self.optimizer.param_groups[0]['lr']), 
                step_size=7, gamma=0.1
            )
            for _ in epochs:
                lrs.append(temp_scheduler.get_last_lr()[0])
                temp_scheduler.step()
            
            axes[1, 0].plot(epochs, lrs, 'g-', linewidth=2, color='orange')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nScheduler Disabled', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Learning Rate')
        
        # Training metrics summary
        final_train_acc = self.history['train_acc'][-1]
        final_val_acc = self.history['val_acc'][-1]
        best_val_acc = max(self.history['val_acc'])
        
        summary_text = f"Final Training Accuracy: {final_train_acc:.2f}%\n"
        summary_text += f"Final Validation Accuracy: {final_val_acc:.2f}%\n"
        summary_text += f"Best Validation Accuracy: {best_val_acc:.2f}%\n"
        summary_text += f"Total Epochs: {len(epochs)}"
        
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
            
        self.model.eval()
        
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
        
        # Normalize to [-1, 1] range
        image_array = (image_array / 255.0 - 0.5) / 0.5
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.FloatTensor(image_array).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            predicted_digit = np.argmax(probabilities)
        
        return predicted_digit, probabilities
    
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'architecture': self.model.architecture,
            'dropout_rate': self.model.dropout_rate,
            'history': self.history
        }, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str, architecture: str = None, dropout_rate: float = 0.3) -> None:
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Get architecture from checkpoint or parameter
        model_architecture = checkpoint.get('architecture', architecture or 'simple')
        model_dropout = checkpoint.get('dropout_rate', dropout_rate)
        
        # Create model with correct architecture
        if self.model is None:
            self.model = MNISTNet(model_architecture, model_dropout).to(self.device)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load history if available
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            
        print(f"Model loaded from {filepath}")
        print(f"Architecture: {model_architecture}, Dropout: {model_dropout}")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='MNIST Digit Classifier - PyTorch')
    
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
    parser.add_argument('--save_model', type=str, default='mnist_model_pytorch.pth',
                       help='Path to save trained model')
    parser.add_argument('--load_model', type=str, 
                       help='Path to load pre-trained model')
    parser.add_argument('--no_training', action='store_true',
                       help='Skip training (use with --load_model)')
    parser.add_argument('--drawing_interface', action='store_true',
                       help='Open drawing interface for testing')
    parser.add_argument('--validation_split', type=float, default=0.0,
                       help='Fraction of training data for validation')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Patience for early stopping')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MNIST DIGIT CLASSIFIER - PYTORCH")
    print("="*60)
    
    # Initialize classifier
    classifier = MNISTClassifier(use_gpu=args.use_gpu)
    
    # Load data
    classifier.load_data(batch_size=args.batch_size, validation_split=args.validation_split)
    
    if args.load_model and not args.no_training:
        # Load pre-trained model and continue training
        classifier.load_model(args.load_model)
        classifier.compile_model(args.optimizer, args.learning_rate)
    elif args.load_model and args.no_training:
        # Load pre-trained model without training
        classifier.load_model(args.load_model)
    else:
        # Create and train new model
        classifier.create_model(args.architecture, args.dropout_rate)
        classifier.compile_model(args.optimizer, args.learning_rate)
        
    if not args.no_training:
        # Train model
        classifier.train(args.epochs, args.early_stopping_patience)
        
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
    
    # Get a few test samples
    test_iter = iter(classifier.test_loader)
    test_data, test_labels = next(test_iter)
    
    for i in range(min(5, len(test_data))):
        test_image = test_data[i].squeeze().numpy()
        true_label = test_labels[i].item()
        
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
