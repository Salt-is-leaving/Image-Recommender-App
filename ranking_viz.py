import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
import cv2
from skimage import feature as skimage_feature
import torch
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity, chi2_kernel
import tkinter as tk
from tkinter import filedialog

from config import PATH_TO_SSD, FEATURE_CONFIGS, SIMILARITY_WEIGHTS, resolve_image_path, find_image_in_database, get_similarity_weights
from db_api import load_features_from_pickle
from convnext_extractor import ConvNeXtFeatureExtractor

logging.basicConfig(level=logging.INFO)

class RankingVisualizer:
    """Visualize and compare individual vs integrated feature rankings."""
    
    def __init__(self, use_cuda=True):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.weights = get_similarity_weights()
        logging.info(f"Ranking visualization using dynamic weights: {self.weights}")
        
        # Load all feature databases
        self.features_data = {}
        self.valid_image_paths = None
        
        # Initialize models for new image feature extraction
        self.convnext_extractor = None
        
        logging.info("Initializing Ranking Visualizer...")
        self.load_feature_databases()
        self.init_models()
    
    def load_feature_databases(self):
        """Load all feature databases and find common image paths."""
        logging.info("Loading feature databases...")
        
        feature_types = ['convnext', 'hsv']
        loaded_counts = {}
        
        for feature_type in feature_types:
            features = load_features_from_pickle(feature_type)
            if features:
                self.features_data[feature_type] = features
                loaded_counts[feature_type] = len(features)
                
                # Find common image paths across all feature types
                current_paths = set(features.keys())
                if self.valid_image_paths is None:
                    self.valid_image_paths = current_paths
                else:
                    self.valid_image_paths = self.valid_image_paths.intersection(current_paths)
        
        self.valid_image_paths = list(self.valid_image_paths) if self.valid_image_paths else []
        
        logging.info("Feature database status:")
        for ft, count in loaded_counts.items():
            logging.info(f"  {ft}: {count} features")
        logging.info(f"Common images across features: {len(self.valid_image_paths)}")
    
    def init_models(self):
        """Initialize models for new image feature extraction."""
        # ConvNext for new images
        try:
            self.convnext_extractor = ConvNeXtFeatureExtractor(
                use_cuda=(self.device.type == 'cuda'),
                use_fp16=True
            )
            logging.info(f"ConvNeXt model initialized on {self.device}")
        except Exception as e:
            logging.error(f"Failed to initialize ConvNeXt extractor: {e}")
            self.convnext_extractor = None

    def extract_features_from_image(self, image_path):
        """Extract all features from a new image."""
        # Resolve the image path
        resolved_path = resolve_image_path(image_path)
        if resolved_path is None:
            logging.error(f"Image not found: {image_path}")
            logging.info(f"Searched in: {PATH_TO_SSD}")
            return None
        
        logging.info(f"Extracting features from: {os.path.basename(resolved_path)}")
        
        try:
            # Load and preprocess
            image = Image.open(resolved_path).convert('RGB')
            image_resized = image.resize((224, 224))
            image_rgb = np.array(image_resized)
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            
            features = {}
            
            # HSV histogram
            hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            config = FEATURE_CONFIGS['hsv']
            hist = cv2.calcHist([hsv_image], [0, 1, 2], None, config['bins'], config['ranges'])
            hist = hist.flatten()
            hist = hist + 1e-10
            hist = hist / hist.sum()
            features['hsv'] = hist.astype(np.float32)
            
            # ConvNext features
            if self.convnext_extractor is not None:
                convnext_features = self.convnext_extractor.extract_features(image_rgb, normalize=True)
                if convnext_features is not None:
                    features['convnext'] = convnext_features  
                else:
                    # Fallback to zero vector
                    features['convnext'] = np.zeros(1024, dtype=np.float32)  
            else:
                logging.warning("ConvNeXt extractor not available, using zero vector")
                features['convnext'] = np.zeros(1024, dtype=np.float32)
            
            logging.info("Feature extraction completed")
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            return None
    
    def compute_bhattacharyya_similarity(self, hist1, hist2):
        """Bhattacharyya coefficient for histogram similarity - consistent with search pipeline."""
        if len(hist1) != len(hist2):
            min_len = min(len(hist1), len(hist2))
            hist1 = hist1[:min_len]
            hist2 = hist2[:min_len]
            logging.warning(f"Histogram dimension mismatch, truncated to {min_len}")
        
        # Ensure positive values and normalization
        hist1_safe = np.maximum(hist1, 1e-10)
        hist2_safe = np.maximum(hist2, 1e-10)
        
        hist1_norm = hist1_safe / (np.sum(hist1_safe) + 1e-8)
        hist2_norm = hist2_safe / (np.sum(hist2_safe) + 1e-8)
        
        return np.sum(np.sqrt(hist1_norm * hist2_norm))
    
    def search_by_convnext(self, target_features, top_k=10):
        """Search using only ConvNext features."""
        target_feature = target_features['convnext']
        target_norm = target_feature / (np.linalg.norm(target_feature) + 1e-8)
        
        similarities = []
        for image_path in self.valid_image_paths:
            if image_path in self.features_data['convnext']:
                db_feature = self.features_data['convnext'][image_path]
                if db_feature is not None:
                    db_norm = db_feature / (np.linalg.norm(db_feature) + 1e-8)
                    similarity = np.dot(target_norm, db_norm)
                    similarities.append((image_path, float(similarity)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def search_by_hsv(self, target_features, top_k=5):
        """Search using only HSV features."""
        target_feature = target_features['hsv']
        
        similarities = []
        for image_path in self.valid_image_paths:
            if image_path in self.features_data['hsv']:
                db_feature = self.features_data['hsv'][image_path]
                if db_feature is not None:
                    similarity = self.compute_bhattacharyya_similarity(target_feature, db_feature)
                    similarities.append((image_path, float(similarity)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def search_integrated(self, target_features, top_k=10):
        """Search using weighted combination of all features."""
        all_similarities = {}
        
        # ConvNext
        convnext_results = self.search_by_convnext(target_features, top_k=50)
        for image_path, similarity in convnext_results:
            if image_path not in all_similarities:
                all_similarities[image_path] = 0.0
            all_similarities[image_path] += similarity * self.weights['convnext']
        
        # HSV
        hsv_results = self.search_by_hsv(target_features, top_k=50)
        for image_path, similarity in hsv_results:
            if image_path not in all_similarities:
                all_similarities[image_path] = 0.0
            all_similarities[image_path] += similarity * self.weights['hsv']
        
        # Sort and return top results
        integrated_results = [(image_path, score) for image_path, score in all_similarities.items()]
        integrated_results.sort(key=lambda x: x[1], reverse=True)
        return integrated_results[:top_k]
    
    def extract_all_rankings(self, target_features, top_k=10):
        """Extract rankings for all feature types plus integrated."""
        rankings = {}
        
        rankings['convnext'] = self.search_by_convnext(target_features, top_k)
        rankings['hsv'] = self.search_by_hsv(target_features, top_k)
        rankings['integrated'] = self.search_integrated(target_features, top_k)
        
        return rankings
     
    def visualize_rankings(self, target_image_path, rankings, top_k=10):
        """Visualize ranking comparison in a grid."""
        feature_types = ['convnext', 'hsv', 'integrated']
        n_features = len(feature_types)
        
        fig, axes = plt.subplots(n_features, top_k + 1, figsize=(4 * (top_k + 1), 4 * n_features))
        fig.suptitle(f"Feature-wise vs Integrated Rankings\nTarget: {os.path.basename(target_image_path)}", 
                     fontsize=16, y=0.98)
        
        # Load target image
        try:
            target_img = Image.open(target_image_path)
        except:
            target_img = None
        
        for i, feature_type in enumerate(feature_types):
            # Display target image in first column
            axes[i, 0].set_title(f"{feature_type.upper()}\nTarget", fontsize=12, fontweight='bold')
            if target_img:
                axes[i, 0].imshow(target_img)
            else:
                axes[i, 0].text(0.5, 0.5, "Target\nImage\nError", ha='center', va='center')
            axes[i, 0].axis('off')
            
            # Display top-k results for this feature type
            results = rankings[feature_type]
            for j, (image_path, similarity) in enumerate(results):
                if j >= top_k:
                    break
                
                col_idx = j + 1
                image_path = self.find_image_in_database(image_path)
                
                if image_path and os.path.exists(image_path):
                    try:
                        img = Image.open(image_path)
                        axes[i, col_idx].imshow(img)
                        title_text = f"#{j+1}\n{similarity:.3f}\n{image_path[:8]}..."
                        axes[i, col_idx].set_title(title_text, fontsize=10)
                    except:
                        axes[i, col_idx].text(0.5, 0.5, f"#{j+1}\nError\n{image_path[:8]}...", 
                                            ha='center', va='center', fontsize=9)
                else:
                    axes[i, col_idx].text(0.5, 0.5, f"#{j+1}\nNot Found\n{image_path[:8]}...", 
                                        ha='center', va='center', fontsize=9)
                
                axes[i, col_idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_ranking_differences(self, rankings):
        """Analyze differences between individual and integrated rankings."""
        logging.info("\n=== RANKING ANALYSIS ===")
        
        # Get top-k image paths for each ranking method
        ranking_sets = {}
        for feature_type, results in rankings.items():
            ranking_sets[feature_type] = [image_path for image_path, _ in results[:10]]
        
        integrated_set = set(ranking_sets['integrated'])
        
        # Compare each individual ranking with integrated
        logging.info("Overlap with Integrated Ranking:")
        for feature_type in ['convnext', 'hsv']:
            individual_set = set(ranking_sets[feature_type])
            overlap = len(individual_set.intersection(integrated_set))
            logging.info(f"  {feature_type}: {overlap}/10 images overlap")
        
        # Find unique contributions
        logging.info("\nUnique Image Contributions:")
        all_individual = set()
        for feature_type in ['convnext', 'hsv']:
            all_individual.update(ranking_sets[feature_type])
        
        unique_to_integrated = integrated_set - all_individual
        if unique_to_integrated:
            logging.info(f"  Images only in integrated ranking: {len(unique_to_integrated)}")
        else:
            logging.info("  No images unique to integrated ranking")
        
        # Show detailed rankings
        logging.info("\nDetailed Rankings:")
        for feature_type, results in rankings.items():
            logging.info(f"\n{feature_type.upper()} Top-10:")
            for i, (image_path, score) in enumerate(results[:10]):
                logging.info(f"  {i+1}. {image_path} (score: {score:.4f})")
    
    def run_ranking_comparison(self, target_image_path=None):
        """Run complete ranking comparison analysis."""
        logging.info("Starting ranking comparison analysis...")
        
        # Select target image
        if target_image_path is None:
            root = tk.Tk()
            root.withdraw()
            
            target_image_path = filedialog.askopenfilename(
                title="Select target image for ranking comparison",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
            )
            
            if not target_image_path:
                logging.info("No image selected")
                return False
        
        # Extract features from target image
        target_features = self.extract_features_from_image(target_image_path)
        if target_features is None:
            logging.error("Failed to extract features from target image")
            return False
        
        # Get all rankings
        rankings = self.extract_all_rankings(target_features, top_k=10)
        
        # Analyze differences
        self.analyze_ranking_differences(rankings)
        
        # Visualize results
        self.visualize_rankings(target_image_path, rankings, top_k=10)
        
        return True

def run_ranking_visualization(image_path=None, use_cuda=True):
    """Wrapper function for main.py integration - matches the import in main.py."""
    try:
        # Initialize visualizer
        visualizer = RankingVisualizer(use_cuda=use_cuda)
        
        # Run comparison (this is the actual method name)
        success = visualizer.run_ranking_comparison(image_path)
        
        if success:
            logging.info(" Ranking visualization completed successfully!")
        else:
            logging.error(" Ranking visualization failed!")
        
        return success
        
    except Exception as e:
        logging.error(f"Ranking visualization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for ranking visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize and compare individual vs integrated feature rankings")
    parser.add_argument('--image-path', type=str, help='Path to target image')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA if available')
    
    args = parser.parse_args()
    
    # Use the wrapper function for consistency
    success = run_ranking_visualization(args.image_path, args.cuda)
    
    if success:
        logging.info(" Ranking comparison completed successfully!")
    else:
        logging.error(" Ranking comparison failed!")

if __name__ == "__main__":
    main()
