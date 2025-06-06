import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from sklearn.preprocessing import normalize
import logging
import time 
import gc
import cv2
from skimage import feature as skimage_feature
import torch
from efficientnet_pytorch import EfficientNet
import random
import tkinter as tk
from tkinter import filedialog, messagebox
import glob

from config import (PATH_TO_SSD, PICKLE_PATH, SIMILARITY_CONFIGS, FEATURE_CONFIGS,
                   get_annoy_index_path)
from db_api import load_features_from_pickle
from similarity_search_pipeline import SmartSimilaritySearch

logging.basicConfig(level=logging.INFO)

class InteractiveSimilaritySearch:
    """Interactive similarity search using proven FAISS backend."""
    
    def __init__(self, use_cuda=True):
        # Use the proven FAISS implementation
        self.searcher = SmartSimilaritySearch(use_gpu=False)  # GPU issues in interactive
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        
        # Initialize feature extraction models
        self.efficientnet_model = None
        self.orb_detector = None
        
        logging.info("Initializing Interactive Similarity Search with FAISS backend...")
        self._init_models()
    
    def _init_models(self):
        """Initialize feature extraction models."""
        try:
            # Initialize EfficientNet
            config = FEATURE_CONFIGS['efficientnet']
            self.efficientnet_model = EfficientNet.from_pretrained(config['model_name']).to(self.device)
            self.efficientnet_model.eval()
            logging.info(f"✓ EfficientNet {config['model_name']} loaded on {self.device}")
            
            # Initialize ORB detector
            orb_config = FEATURE_CONFIGS['orb']
            self.orb_detector = cv2.ORB_create(
                nfeatures=orb_config['n_features'],
                scaleFactor=orb_config['scale_factor'],
                nlevels=orb_config['n_levels']
            )
            logging.info(f"✓ ORB detector initialized")
            
        except Exception as e:
            logging.error(f"Error initializing models: {e}")
    
    def load_database_features(self):
        """Load database features using FAISS backend."""
        return self.searcher.load_and_process_features()
    
    def build_search_indices(self):
        """Build search indices using FAISS backend."""
        self.searcher.build_indices()
    
    def extract_features_from_image(self, image_path):
        """Extract features using the same approach as comparison mode."""
        logging.info(f"Extracting features from: {os.path.basename(image_path)}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_resized = image.resize((224, 224))
            image_rgb = np.array(image_resized)
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            
            features = {}
            
            # 1. Extract HSV histogram (same as comparison mode - no PCA)
            hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            config = FEATURE_CONFIGS['hsv']
            hist = cv2.calcHist([hsv_image], [0, 1, 2], None, 
                              config['bins'], config['ranges'])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-8)
            features['hsv'] = hist.astype(np.float32)
            
            # 2. Extract LBP features
            lbp_config = FEATURE_CONFIGS['lbp']
            lbp = skimage_feature.local_binary_pattern(
                image_gray, lbp_config['n_points'], lbp_config['radius'], 
                method=lbp_config['method']
            )
            n_bins = lbp_config['n_points'] + 2
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, 
                                     range=(0, n_bins), density=True)
            features['lbp'] = lbp_hist.astype(np.float32)
            
            # 3. Extract ORB features
            keypoints, descriptors = self.orb_detector.detectAndCompute(image_gray, None)
            if descriptors is not None:
                # Use median pooling (same as comparison mode)
                if len(descriptors.shape) == 2:
                    mean_descriptor = np.median(descriptors, axis=0)
                    features['orb'] = mean_descriptor.astype(np.float32)
                else:
                    features['orb'] = descriptors.astype(np.float32)
            else:
                features['orb'] = np.zeros(32, dtype=np.float32)
            
            # 4. Extract EfficientNet features
            image_tensor = torch.tensor(image_rgb / 255.0).permute(2, 0, 1).float().unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            with torch.no_grad():
                eff_features = self.efficientnet_model(image_tensor).cpu().numpy().flatten()
            features['efficientnet'] = eff_features.astype(np.float32)
            
            logging.info("✓ Feature extraction completed")
            logging.info(f"  HSV: {features['hsv'].shape}")
            logging.info(f"  LBP: {features['lbp'].shape}")
            logging.info(f"  ORB: {features['orb'].shape}")
            logging.info(f"  EfficientNet: {features['efficientnet'].shape}")
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            return None
    
    def find_similar_images_for_new_image(self, target_features, top_n=5):
        """Find similar images for a new image using direct similarity computation."""
        logging.info(f"Starting similarity search for new image with {len(target_features)} feature types")
        
        all_similarities = {}
        
        # Process each feature type
        for feature_type, target_feature in target_features.items():
            if feature_type not in self.searcher.indices:
                logging.warning(f"Skipping {feature_type}: not in searcher indices")
                continue
                
            index_data = self.searcher.indices[feature_type]
            valid_ids = index_data['image_ids']
            database_features = index_data['features']
            metric = index_data['metric']
            
            logging.info(f"Processing {feature_type}: target shape {target_feature.shape}, db shape {database_features.shape}")
            
            # Ensure target feature is properly shaped and normalized
            target_flat = target_feature.flatten().astype(np.float32)
            
            # Normalize target feature based on metric
            if metric == 'cosine':
                target_norm = target_flat / (np.linalg.norm(target_flat) + 1e-8)
            else:
                target_norm = target_flat
            
            # Compute similarities with all database features
            if metric == 'cosine':
                # Direct cosine similarity computation
                similarities_scores = np.dot(database_features, target_norm)
            elif metric == 'chi_squared':
                # Chi-squared for histograms (HSV, LBP)
                similarities_scores = []
                for db_feature in database_features:
                    distance = self.searcher.compute_chi_squared_distance(target_norm, db_feature)
                    similarity = 1.0 / (1.0 + distance)
                    similarities_scores.append(similarity)
                similarities_scores = np.array(similarities_scores)
            else:  # L2 distance
                # Euclidean distance
                distances = np.linalg.norm(database_features - target_norm, axis=1)
                similarities_scores = 1.0 / (1.0 + distances)
            
            # Get top candidates for this feature type
            top_indices = np.argsort(similarities_scores)[::-1][:top_n * 3]
            
            weight = self.searcher.weights.get(feature_type, 0.25)
            logging.info(f"Found {len(top_indices)} candidates for {feature_type} with weight {weight}")
            
            # Add to aggregated similarities
            for idx in top_indices:
                if idx < len(valid_ids):
                    image_id = valid_ids[idx]
                    similarity = float(similarities_scores[idx])
                    
                    if image_id not in all_similarities:
                        all_similarities[image_id] = {
                            'total_score': 0.0, 
                            'count': 0, 
                            'details': {}
                        }
                    
                    all_similarities[image_id]['total_score'] += similarity * weight
                    all_similarities[image_id]['count'] += 1
                    all_similarities[image_id]['details'][feature_type] = similarity
        
        # Debug: Show aggregation results
        logging.info(f"Aggregated similarities for {len(all_similarities)} images")
        
        if not all_similarities:
            logging.error("No similarities computed - check feature extraction and normalization")
            return []
        
        # Sort results (accept any feature match)
        sorted_results = []
        for image_id, data in all_similarities.items():
            if data['count'] >= 1:  # Accept any matching feature
                sorted_results.append({
                    'image_id': image_id,
                    'combined_similarity': data['total_score'],
                    'feature_count': data['count'],
                    'feature_details': data['details']
                })
        
        logging.info(f"Results after filtering: {len(sorted_results)}")
        
        # Sort by combined similarity
        sorted_results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        final_results = sorted_results[:top_n]
        
        # Debug: Show final results
        if final_results:
            logging.info(f"Top {len(final_results)} results:")
            for i, result in enumerate(final_results):
                logging.info(f"  {i+1}. {result['image_id']}: {result['combined_similarity']:.3f}")
        else:
            logging.error("No final results - this should not happen!")
        
        return final_results
    
    def find_similar_images(self, target_features, top_n=5):
        """Main interface for finding similar images."""
        return self.find_similar_images_for_new_image(target_features, top_n)
    
    def find_image_path(self, image_id):
        """Find the full path of an image in the database."""
        for root, dirs, files in os.walk(PATH_TO_SSD):
            if image_id in files:
                return os.path.join(root, image_id)
        return None
    
    def display_results(self, target_image_path, similar_images):
        """Display the target image and top similar images."""
        n_images = len(similar_images) + 1  # Target + similar images
        fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 5))
        
        if n_images == 1:
            axes = [axes]
        
        plt.suptitle(f"Image Similarity Search Results", fontsize=16)
        
        # Display target image
        try:
            target_img = Image.open(target_image_path)
            axes[0].imshow(target_img)
            axes[0].set_title(f"Target Image\n{os.path.basename(target_image_path)}", fontsize=12)
            axes[0].axis('off')
        except Exception as e:
            axes[0].text(0.5, 0.5, "Target\nImage Error", ha='center', va='center')
            axes[0].axis('off')
        
        # Display similar images
        for i, result in enumerate(similar_images):
            if i + 1 >= len(axes):
                break
            
            image_id = result['image_id']
            similarity = result['combined_similarity']
            
            image_path = self.find_image_path(image_id)
            
            if image_path and os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    axes[i + 1].imshow(img)
                    
                    # Show detailed scores
                    details = result['feature_details']
                    detail_text = f"Rank {i + 1}\nOverall: {similarity:.3f}\n"
                    for feat_type, score in details.items():
                        detail_text += f"{feat_type[:3]}: {score:.2f}\n"
                    
                    axes[i + 1].set_title(detail_text, fontsize=10)
                    axes[i + 1].axis('off')
                except Exception as e:
                    axes[i + 1].text(0.5, 0.5, f"Rank {i + 1}\nImage Error", 
                                    ha='center', va='center')
                    axes[i + 1].axis('off')
            else:
                axes[i + 1].text(0.5, 0.5, f"Rank {i + 1}\nNot Found", 
                                ha='center', va='center')
                axes[i + 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        plt.close()
    
    def select_image_gui(self):
        """GUI for selecting an image file."""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # File selection dialog
        file_path = filedialog.askopenfilename(
            title="Select an image for similarity search",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        return file_path
    
    def select_random_image(self):
        """Select a random image from the image directory."""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(PATH_TO_SSD, '**', ext), recursive=True))
        
        if all_images:
            return random.choice(all_images)
        else:
            return None
    
    def interactive_search(self):
        """Interactive similarity search workflow."""
        print("\n" + "="*60)
        print("🔍 INTERACTIVE IMAGE SIMILARITY SEARCH")
        print("="*60)
        
        while True:
            print("\nOptions:")
            print("1. Select image from file browser")
            print("2. Use random image from database")
            print("3. Enter image path manually")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            target_image_path = None
            
            if choice == '1':
                target_image_path = self.select_image_gui()
                if not target_image_path:
                    print("No image selected.")
                    continue
            
            elif choice == '2':
                target_image_path = self.select_random_image()
                if not target_image_path:
                    print("No images found in database directory.")
                    continue
                print(f"Random image selected: {os.path.basename(target_image_path)}")
            
            elif choice == '3':
                target_image_path = input("Enter image path: ").strip()
                if not os.path.exists(target_image_path):
                    print("Image not found.")
                    continue
            
            elif choice == '4':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")
                continue
            
            # Process the selected image
            print(f"\n🔬 Processing: {os.path.basename(target_image_path)}")
            
            # Extract features
            features = self.extract_features_from_image(target_image_path)
            if features is None:
                print("❌ Feature extraction failed.")
                continue
            
            # Find similar images
            print("🔍 Searching for similar images...")
            similar_images = self.find_similar_images(features, top_n=5)
            
            if not similar_images:
                print("❌ No similar images found.")
                continue
            
            # Display results
            print(f"✅ Found {len(similar_images)} similar images!")
            
            # Print text results
            print(f"\n📊 Top {len(similar_images)} Most Similar Images:")
            print("-" * 50)
            for i, result in enumerate(similar_images):
                print(f"{i+1}. {result['image_id']}")
                print(f"   Overall Similarity: {result['combined_similarity']:.3f}")
                print(f"   Feature Details: {result['feature_details']}")
                print()
            
            # Show visual results
            try:
                self.display_results(target_image_path, similar_images)
            except Exception as e:
                print(f"⚠️ Could not display visual results: {e}")
            
            # Ask if user wants to continue
            continue_search = input("\n🔄 Search for another image? (y/n): ").strip().lower()
            if continue_search not in ['y', 'yes']:
                break
        
        print("\n🎉 Thank you for using Interactive Image Similarity Search!")

def main():
    """Main function to run interactive similarity search."""
    print("🚀 Initializing Interactive Image Similarity Search...")
    
    # Initialize the search system
    searcher = InteractiveSimilaritySearch(use_cuda=True)
    
    # Load database features
    if not searcher.load_database_features():
        print("❌ Failed to load database features. Please run learning mode first.")
        return
    
    # Build search indices
    searcher.build_search_indices()
    
    print("✅ System ready!")
    
    # Start int