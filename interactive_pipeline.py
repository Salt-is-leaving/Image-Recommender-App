import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import logging
import time 
import cv2
from skimage import feature as skimage_feature
import torch
import tkinter as tk
from tkinter import filedialog, messagebox

from config import PATH_TO_SSD, PICKLE_PATH, FEATURE_CONFIGS, resolve_image_path, find_image_in_database, get_similarity_weights
from db_api import load_features_from_pickle
from similarity_search_pipeline import ClusteringFirstSearch
from clustering import ClusteringPipeline
from convnext_extractor import ConvNeXtFeatureExtractor

logging.basicConfig(level=logging.INFO)

class InteractiveSimilaritySearch:
    """Interactive similarity search with embedding storage."""
    
    def __init__(self, use_cuda=True):
        self.searcher = ClusteringFirstSearch(use_gpu=False)
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        
        self.weights = get_similarity_weights()
        logging.info(f"Interactive search using dynamic weights: {self.weights}")

        # Initialize models
        self.convnext_model = None
        self.clustering = ClusteringPipeline()
        
        # Embedding storage for new images
        self.embedding_cache_path = os.path.join(PICKLE_PATH, 'new_image_embeddings.pkl')
        self.embedding_cache = self.load_embedding_cache()
        
        logging.info("Initializing Interactive Similarity Search...")
        self.init_model()
        
        if not self.clustering.load_clustering_data():
            logging.warning("No clustering data - run clustering mode first")
    
    def init_model(self):
        try:
            self.convnext_model = ConvNeXtFeatureExtractor(
                use_cuda=(self.device.type == 'cuda'),
                use_fp16=False
            )
            logging.info(f"ConvNeXt model loaded on {self.device}")
        except Exception as e:
            logging.error(f"Failed to initialize ConvNeXt model: {e}")
            self.convnext_model = None
    
    def load_embedding_cache(self):
        """Load cached embeddings for new images."""
        if os.path.exists(self.embedding_cache_path):
            try:
                with open(self.embedding_cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {}
    
    def save_embedding_cache(self):
        """Save embedding cache to disk."""
        try:
            with open(self.embedding_cache_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logging.info(f"Saved {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            logging.error(f"Error saving embedding cache: {e}")
    
    def load_database_features(self):
        """Load database features."""
        logging.info("Loading database features...")
        success = self.searcher.load_and_process_features()
        
        if success:
            self.indices = self.searcher.indices
            logging.info("Database features loaded")
            
            for ft, assignments in self.searcher.clustering.cluster_assignments.items():
                n_clusters = len(set(assignments.values()))
                logging.info(f" {ft}: {len(assignments)} images in {n_clusters} clusters")
        
        return success

    def build_search_indices(self):
        """Build search indices."""
        logging.info("Building search indices...")
        success = self.searcher.build_indices()
        
        if success:
            logging.info("Search indices built")
        
        return success
    
    def extract_features_from_image(self, image_path):
        """Extract features from image with caching."""
        resolved_path = resolve_image_path(image_path)
        if resolved_path is None:
            logging.error(f"Image not found: {image_path}")
            logging.info(f"Searched in: {PATH_TO_SSD}")
            return None
        
        # Check cache first
        cache_key = f"{image_path}:{os.path.getmtime(image_path)}"
        if cache_key in self.embedding_cache:
            logging.info(f"Using cached features for {os.path.basename(image_path)}")
            return self.embedding_cache[cache_key]
        
        logging.info(f"Extracting features from: {os.path.basename(image_path)}")
        
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
            
            #ConvNext features
            if self.convnext_model is not None:
                convnext_features = self.convnext_model.extract_features(image_rgb, normalize=True)
                if convnext_features is not None:
                    features['convnext'] = convnext_features
                else:
                    # Fallback to zero vector
                    features['convnext'] = np.zeros(1024, dtype=np.float32)
            else:
                logging.warning("ConvNeXt model not available, using zero vector")
                features['convnext'] = np.zeros(1024, dtype=np.float32)
            
            # Cache the features
            self.embedding_cache[cache_key] = features
            self.save_embedding_cache()
            
            logging.info("Feature extraction completed")
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            return None
    
    def find_similar_images_for_new_image(self, target_features, top_n=5):
        """Find similar images using clustering-first search."""
        logging.info("Starting clustering-first search...")
        
        try:
            results = self.searcher.multi_feature_search(target_features, top_n=top_n)
            
            if results:
                logging.info(f"Found {len(results)} results")
                return results
            else:
                logging.warning("No results from clustering search, trying fallback...")
                return self.fallback_search(target_features, top_n)
                
        except Exception as e:
            logging.error(f"Search failed: {e}")
            return self.fallback_search(target_features, top_n)
    
    def fallback_search(self, target_features, top_n=5):
        logging.info("Using fallback search...")
        
        all_similarities = {}
        
        for feature_type, target_feature in target_features.items():
            if feature_type not in self.indices:
                continue
                
            index_data = self.indices[feature_type]
            valid_ids = index_data['image_paths']
            database_features = index_data['features']
            
            # Simple cosine similarity
            target_flat = target_feature.flatten().astype(np.float32)
            target_norm = target_flat / (np.linalg.norm(target_flat) + 1e-8)
            similarities = np.dot(database_features, target_norm)
            
            weight = 0.5  # Equal weights
            top_indices = np.argsort(similarities)[::-1][:top_n * 2]
            
            for idx in top_indices:
                if idx < len(valid_ids):
                    image_path = valid_ids[idx]
                    similarity = float(similarities[idx])
                    
                    if image_path not in all_similarities:
                        all_similarities[image_path] = {'total_score': 0.0, 'count': 0, 'details': {}}
                    
                    all_similarities[image_path]['total_score'] += similarity * weight
                    all_similarities[image_path]['count'] += 1
                    all_similarities[image_path]['details'][feature_type] = similarity
        
        # Convert to list and sort
        results = []
        for image_path, data in all_similarities.items():
            if data['count'] >= 1:
                results.append({
                    'image_path': image_path,
                    'combined_similarity': data['total_score'],
                    'feature_count': data['count'],
                    'feature_details': data['details']
                })
        
        results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        return results[:top_n]
    
    def print_individual_rankings(self, target_features, integrated_results):
        """Print individual feature rankings alongside integrated results."""
        print(f"\n{'='*70}")
        print("INDIVIDUAL vs INTEGRATED FEATURE RANKINGS")
        print(f"{'='*70}")
        
        # Get individual rankings
        individual_rankings = self.searcher.get_individual_rankings(target_features, top_k=8)
        
        # Print individual rankings
        for feature_type, rankings in individual_rankings.items():
            print(f"\n{feature_type.upper()} ONLY Rankings:")
            print("-" * 40)
            for i, result in enumerate(rankings[:5]):
                print(f"  {i+1}. {result['image_path']:<25} | {result['similarity']:.4f}")
        
        # Print integrated rankings
        print(f"\nINTEGRATED Rankings:")
        print("-" * 40)
        for i, result in enumerate(integrated_results[:5]):
            print(f"  {i+1}. {result['image_path']:<25} | {result['combined_similarity']:.4f}")
            details = result.get('feature_details', {})
            if details:
                detail_str = " | ".join([f"{k}:{v:.3f}" for k, v in details.items()])
                print(f"      ({detail_str})")
        
        # Quick overlap analysis
        if 'hsv' in individual_rankings and individual_rankings['hsv']:
            hsv_ids = {r['image_path'] for r in individual_rankings['hsv'][:5]}
            integrated_ids = {r['image_path'] for r in integrated_results[:5]}
            overlap = len(hsv_ids.intersection(integrated_ids))
            
            hsv_scores = [r['similarity'] for r in individual_rankings['hsv'][:5]]
            print(f"\nHSV Analysis:")
            print(f"  Score range: {min(hsv_scores):.3f} - {max(hsv_scores):.3f}")
            print(f"  Overlap with integrated: {overlap}/5 images")

    def display_results(self, target_image_path, similar_images):
        n_images = len(similar_images) + 1
        fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 5))
        
        if n_images == 1:
            axes = [axes]
        
        plt.suptitle("Interactive Image Similarity Search", fontsize=16)
        
        # Display target image
        try:
            target_img = Image.open(target_image_path)
            axes[0].imshow(target_img)
            axes[0].set_title(f"Query Image\n{os.path.basename(target_image_path)}", fontsize=12)
            axes[0].axis('off')
        except:
            axes[0].text(0.5, 0.5, "Query\nImage Error", ha='center', va='center')
            axes[0].axis('off')
        
        # Display similar images
        for i, result in enumerate(similar_images):
            if i + 1 >= len(axes):
                break
            
            image_path = result['image_path']
            similarity = result['combined_similarity']
            
            image_path = self.find_image_in_database(image_path)
            
            if image_path and os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    axes[i + 1].imshow(img)
                    title_text = f"Rank {i + 1}\nScore: {similarity:.3f}"
                    axes[i + 1].set_title(title_text, fontsize=10)
                    axes[i + 1].axis('off')
                except:
                    axes[i + 1].text(0.5, 0.5, f"Rank {i + 1}\nError", ha='center', va='center')
                    axes[i + 1].axis('off')
            else:
                axes[i + 1].text(0.5, 0.5, f"Rank {i + 1}\nNot Found", ha='center', va='center')
                axes[i + 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def run_interactive_session(self, image_path=None):
        logging.info("Starting interactive session...")
        
        if not self.load_database_features():
            logging.error("Failed to load database features")
            return False
        
        if not self.build_search_indices():
            logging.error("Failed to build search indices")
            return False
        
        # Select image
        if image_path is None:
            root = tk.Tk()
            root.withdraw()
            
            image_path = filedialog.askopenfilename(
                title="Select an image for similarity search",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
            )
            
            if not image_path:
                logging.info("No image selected")
                return False
        
        # Extract features
        target_features = self.extract_features_from_image(image_path)
        if target_features is None:
            logging.error("Failed to extract features")
            return False
        
        # Find similar images
        similar_images = self.find_similar_images_for_new_image(target_features, top_n=8)
        
        if similar_images:
            logging.info(f"Found {len(similar_images)} similar images")
            
            # Display results
            self.display_results(image_path, similar_images)

            self.print_individual_rankings(target_features, similar_images)
            
            # Print text results
            print(f"\nSimilarity Search Results for: {os.path.basename(image_path)}")
            print("=" * 60)
            for i, result in enumerate(similar_images):
                print(f"{i+1}. {result['image_path']}")
                print(f"   Score: {result['combined_similarity']:.4f}")
                print(f"   Features: {result['feature_count']}")
                if 'cluster_info' in result:
                    clusters = result['cluster_info']
                    cluster_str = ", ".join([f"{k}:C{v}" for k, v in clusters.items()])
                    print(f"   Clusters: {cluster_str}")
                print()
            
            return True
        else:
            logging.error("No similar images found")
            return False

def run_interactive_mode(image_path=None, use_cuda=True):
    searcher = InteractiveSimilaritySearch(use_cuda=use_cuda)
    return searcher.run_interactive_session(image_path)

if __name__ == "__main__":
    success = run_interactive_mode()
    if success:
        logging.info("Interactive search completed!")
    else:
        logging.error("Interactive search failed!")
