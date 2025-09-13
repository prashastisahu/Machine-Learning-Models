#!/usr/bin/env python3
"""
K-Nearest Neighbors Image Classifier
=====================================

A complete implementation of KNN classifier for image classification using
HOG (Histogram of Oriented Gradients) and color features.

Features:
- Image feature extraction (HOG + Color)
- KNN classification with configurable K
- Interactive similarity search visualization
- Confusion matrix and metrics evaluation
- Comprehensive error handling

Author: Enhanced implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from skimage.feature import hog
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image, ImageOps
from glob import glob
import warnings
from random import shuffle
from typing import List, Tuple, Optional
import argparse

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
sns.set_theme()


class ImageFeatureExtractor:
    """Handles feature extraction from images using HOG and color features"""
    
    def __init__(self, color_size: int = 4, hog_size: int = 128, hog_cell_size: int = 16):
        """
        Initialize feature extractor parameters
        
        Args:
            color_size: Size for color feature extraction (creates color_size x color_size grid)
            hog_size: Size to resize images for HOG feature extraction
            hog_cell_size: Pixels per cell for HOG feature extraction
        """
        self.color_size = color_size
        self.hog_size = hog_size
        self.hog_cell_size = hog_cell_size
    
    def extract_color_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract color features by downsampling image to small grid
        
        Args:
            image: PIL Image to extract features from
            
        Returns:
            Flattened array of normalized color values
        """
        # Resize to small grid and normalize
        small_image = image.resize((self.color_size, self.color_size), Image.NEAREST)
        return np.array(small_image).flatten() / 255.0
    
    def extract_hog_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract HOG (Histogram of Oriented Gradients) features
        
        Args:
            image: PIL Image to extract features from
            
        Returns:
            HOG feature vector
        """
        # Resize image and extract HOG features
        resized_image = image.resize((self.hog_size, self.hog_size))
        hog_features = hog(
            resized_image, 
            pixels_per_cell=(self.hog_cell_size, self.hog_cell_size),
            visualize=False
        )
        return hog_features
    
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract combined feature vector (HOG + Color)
        
        Args:
            image: PIL Image to extract features from
            
        Returns:
            Combined feature vector
        """
        color_features = self.extract_color_features(image)
        hog_features = self.extract_hog_features(image)
        return np.concatenate([hog_features, color_features])


class KNNClassifier:
    """K-Nearest Neighbors classifier implementation"""
    
    def __init__(self, k: int = 5):
        """
        Initialize KNN classifier
        
        Args:
            k: Number of nearest neighbors to consider
        """
        self.k = k
        self.features = None
        self.labels = None
    
    def fit(self, features: List[np.ndarray], labels: np.ndarray):
        """
        Fit the classifier with training data
        
        Args:
            features: List of feature vectors
            labels: Corresponding labels
        """
        self.features = features
        self.labels = labels
    
    def find_knn(self, query: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """
        Find k nearest neighbors for a query feature vector
        
        Args:
            query: Feature vector to find neighbors for
            k: Number of neighbors (uses self.k if None)
            
        Returns:
            Indices of k nearest neighbors sorted by distance
        """
        if k is None:
            k = self.k
        
        if self.features is None:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        # Calculate Euclidean distances
        distances = []
        for feature in self.features:
            distance = np.sqrt(np.sum((query - feature) ** 2))
            distances.append(distance)
        
        # Return indices of k nearest neighbors
        return np.argsort(distances)[:k]
    
    def predict_single(self, query: np.ndarray) -> int:
        """
        Predict label for a single query
        
        Args:
            query: Feature vector to classify
            
        Returns:
            Predicted label
        """
        # Find k+1 nearest neighbors (including self if query is from training set)
        neighbor_indices = self.find_knn(query, self.k + 1)
        
        # Get labels of neighbors (exclude first one in case it's the query itself)
        neighbor_labels = [self.labels[idx] for idx in neighbor_indices[1:self.k + 1]]
        
        # If we don't have enough neighbors, use what we have
        if len(neighbor_labels) == 0:
            neighbor_labels = [self.labels[neighbor_indices[0]]]
        
        # Vote for most common label
        label_counts = Counter(neighbor_labels)
        return label_counts.most_common(1)[0][0]
    
    def predict(self, features: List[np.ndarray]) -> np.ndarray:
        """
        Predict labels for multiple queries
        
        Args:
            features: List of feature vectors to classify
            
        Returns:
            Array of predicted labels
        """
        predictions = []
        for feature in features:
            predictions.append(self.predict_single(feature))
        return np.array(predictions)


class ImageClassificationAnalyzer:
    """Handles visualization and analysis of image classification results"""
    
    def __init__(self, images: List[Image.Image], labels: np.ndarray, label_encoder: LabelEncoder):
        """
        Initialize analyzer
        
        Args:
            images: List of PIL Images
            labels: True labels (encoded)
            label_encoder: LabelEncoder for label names
        """
        self.images = images
        self.labels = labels
        self.label_encoder = label_encoder
        self.similarity_plot = None
    
    def calculate_metrics(self, cm: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate accuracy, precision, and recall from confusion matrix
        
        Args:
            cm: Confusion matrix
            
        Returns:
            Tuple of (accuracy, precision, recall)
        """
        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            recall = np.diag(cm) / np.sum(cm, axis=1)
            precision = np.diag(cm) / np.sum(cm, axis=0)
            
            # Replace NaN with 0
            recall = np.nan_to_num(recall)
            precision = np.nan_to_num(precision)
        
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        
        return accuracy, precision.mean(), recall.mean()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            title: str = "Confusion Matrix") -> plt.Figure:
        """
        Plot confusion matrix with metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        label_names = self.label_encoder.inverse_transform(np.unique(y_true))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        df_cm = pd.DataFrame(cm, label_names, label_names)
        sns.heatmap(df_cm, annot=True, cmap='Blues', cbar=True, ax=ax, fmt='d')
        
        # Calculate and display metrics
        accuracy, precision, recall = self.calculate_metrics(cm)
        stats = f"\nAccuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}"
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label' + stats)
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def create_similarity_search(self, features: List[np.ndarray], knn_classifier: KNNClassifier):
        """
        Create interactive similarity search visualization
        
        Args:
            features: List of feature vectors
            knn_classifier: Fitted KNN classifier
        """
        def plot_similarity(query_idx: int = 0):
            """Plot similarity search results"""
            query_feature = features[query_idx]
            similar_indices = knn_classifier.find_knn(query_feature, 100)  # Get top 100
            
            fig = plt.figure(figsize=(15, 10))
            
            # Plot query image
            ax_query = plt.subplot(1, 2, 1)
            ax_query.imshow(self.images[query_idx])
            ax_query.set_title(f"Query Image (Index: {query_idx})")
            ax_query.axis('off')
            
            # Plot similar images in grid
            ax_grid = plt.subplot(1, 2, 2)
            ax_grid.set_title("Most Similar Images")
            ax_grid.axis('off')
            
            # Create image grid
            grid = ImageGrid(fig, 122, nrows_ncols=(10, 10), axes_pad=0.02)
            
            for ax, idx in zip(grid, similar_indices[:100]):
                ax.axis('off')
                resized_img = ImageOps.fit(self.images[idx], (64, 64), Image.LANCZOS)
                ax.imshow(resized_img)
            
            plt.tight_layout()
            return fig
        
        # Create initial plot
        return plot_similarity()


class ImageDataLoader:
    """Handles loading and preprocessing of image datasets"""
    
    @staticmethod
    def load_images_from_directory(data_path: str, max_images: Optional[int] = None,
                                 shuffle_data: bool = True) -> Tuple[List[Image.Image], np.ndarray, LabelEncoder]:
        """
        Load images from directory structure where subdirectories are class names
        
        Args:
            data_path: Path to directory containing class subdirectories
            max_images: Maximum number of images to load (None for all)
            shuffle_data: Whether to shuffle the dataset
            
        Returns:
            Tuple of (images, encoded_labels, label_encoder)
        """
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        files = []
        for ext in image_extensions:
            files.extend(glob(os.path.join(data_path, '**', ext), recursive=True))
        
        if shuffle_data:
            shuffle(files)
        
        if max_images:
            files = files[:max_images]
        
        if not files:
            raise ValueError(f"No image files found in {data_path}")
        
        print(f"Loading {len(files)} images...")
        
        # Load images
        images = []
        labels = []
        
        for file_path in files:
            try:
                # Load and convert to RGB
                img = Image.open(file_path).convert('RGB')
                images.append(img)
                
                # Extract label from directory name
                label = Path(file_path).parent.name
                labels.append(label)
                
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
        
        if not images:
            raise ValueError("No images could be loaded successfully")
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        print(f"Loaded {len(images)} images with {len(label_encoder.classes_)} classes:")
        for i, class_name in enumerate(label_encoder.classes_):
            count = np.sum(encoded_labels == i)
            print(f"  {class_name}: {count} images")
        
        return images, encoded_labels, label_encoder


def main():
    """Main function to run the complete KNN image classification pipeline"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='KNN Image Classifier')
    parser.add_argument('--data_path', default='resources/images', 
                       help='Path to image directory')
    parser.add_argument('--k', type=int, default=5, 
                       help='Number of neighbors for KNN')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process')
    parser.add_argument('--no_visualization', action='store_true',
                       help='Skip visualization steps')
    
    args = parser.parse_args()
    
    print("="*60)
    print("K-NEAREST NEIGHBORS IMAGE CLASSIFIER")
    print("="*60)
    
    # Step 1: Load images
    try:
        images, y_true, label_encoder = ImageDataLoader.load_images_from_directory(
            args.data_path, 
            max_images=args.max_images,
            shuffle_data=True
        )
    except Exception as e:
        print(f"Error loading images: {e}")
        print("Please ensure the data path exists and contains image subdirectories.")
        return
    
    # Step 2: Extract features
    print("\nExtracting features...")
    feature_extractor = ImageFeatureExtractor()
    features = []
    
    for i, image in enumerate(images):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(images)} images")
        try:
            feature_vector = feature_extractor.extract_features(image)
            features.append(feature_vector)
        except Exception as e:
            print(f"Warning: Could not extract features from image {i}: {e}")
            # Use zero vector as fallback
            features.append(np.zeros(100))  # Approximate feature size
    
    print(f"✓ Extracted features for {len(features)} images")
    print(f"  Feature vector size: {len(features[0])}")
    
    # Step 3: Train KNN classifier
    print(f"\nTraining KNN classifier with k={args.k}...")
    knn_classifier = KNNClassifier(k=args.k)
    knn_classifier.fit(features, y_true)
    
    # Step 4: Make predictions (leave-one-out style)
    print("Making predictions...")
    y_pred = knn_classifier.predict(features)
    
    # Step 5: Evaluate results
    print("\nEvaluation Results:")
    print("-" * 30)
    
    analyzer = ImageClassificationAnalyzer(images, y_true, label_encoder)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    
    # Per-class results
    print(f"\nPer-class results:")
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
            print(f"  {class_name}: {class_acc:.3f}")
    
    # Step 6: Visualizations
    if not args.no_visualization:
        print("\nGenerating visualizations...")
        
        # Confusion matrix
        cm_fig = analyzer.plot_confusion_matrix(y_true, y_pred)
        plt.show()
        
        # Feature visualization example
        if len(images) > 0:
            print("Showing feature extraction example...")
            test_image = images[0]
            
            # Show original and processed versions
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(test_image)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # Color features (downsampled)
            color_features = test_image.resize((4, 4), Image.NEAREST).resize((128, 128), Image.NEAREST)
            axes[1].imshow(color_features)
            axes[1].set_title("Color Features (4x4 grid)")
            axes[1].axis('off')
            
            # HOG visualization
            resized = test_image.resize((128, 128))
            hog_features, hog_img = hog(resized, pixels_per_cell=(16, 16), visualize=True)
            axes[2].imshow(hog_img, cmap='gray')
            axes[2].set_title("HOG Features")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # Similarity search
        if len(images) > 10:
            print("Generating similarity search visualization...")
            similarity_fig = analyzer.create_similarity_search(features, knn_classifier)
            plt.show()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'classifier': knn_classifier,
        'features': features,
        'images': images,
        'labels': y_true,
        'label_encoder': label_encoder
    }


if __name__ == "__main__":
    # Example usage when run as script
    results = main()
