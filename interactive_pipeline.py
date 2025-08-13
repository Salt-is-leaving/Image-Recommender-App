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
from config import PATH_TO_SSD, PICKLE_PATH, FEATURE_CONFIGS, resolve_image_path, get_similarity_weights
from db_api import load_features_from_pickle
from similarity_search_pipeline import ClusteringFirstSearch
from convnext_extractor import ConvNeXtFeatureExtractor

# ONLY import clustering modules when actually needed
try:
    from similarity_search_pipeline import ClusteringFirstSearch
    from clustering import ClusteringPipeline
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    logging.warning("Clustering modules not available")


logging.basicConfig(level=logging.INFO)

class InteractiveSimilaritySearch:
    """Interactive similarity search with embedding storage."""
    
    def __init__(self, use_cuda=True):
        self.searcher = ClusteringFirstSearch(use_gpu=False)
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        
        # Check if clustering is available before initializing
        if not CLUSTERING_AVAILABLE:
            raise ImportError("Clustering modules not available. Run clustering mode first.")
        
        # Check if clustering data exists
        cluster_file = os.path.join(PICKLE_PATH, 'cluster_data.pkl')
        if not os.path.exists(cluster_file):
            raise FileNotFoundError(
                "Clustering data not found. Please run:\n"
                "1. python main.py --mode learning\n"
                "2. python main.py --mode clustering\n"
                "3. python main.py --mode interactive"
            )
        
        self.weights = get_similarity_weights()
        logging.info(f"Interactive search using dynamic weights: {self.weights}")
        
        # Initialize models
        self.convnext_model = None
        self.clustering = ClusteringPipeline()
        
        # Load clustering data
        if not self.clustering.load_clustering_data():
            logging.warning("No clustering data - run clustering mode first")
        
        # Embedding storage for new images
        self.embedding_cache_path = os.path.join(PICKLE_PATH, 'new_image_embeddings.pkl')
        self.embedding_cache = self.load_embedding_cache()
        
        logging.info("Initializing Interactive Similarity Search...")
        self.init_model()
        
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
            self.indices = self.searcher.faiss_indices
            logging.info("Database features loaded")
            
            for ft, assignments in self.searcher.clustering.cluster_assignments.items():
                n_clusters = len(set(assignments.values()))
                logging.info(f" {ft}: {len(assignments)} images in {n_clusters} clusters")
        
        return success

    def resolve_image_path(self, image_path):
        """Find image in database directories."""
        from config import resolve_image_path
        return resolve_image_path(image_path)

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
        cache_key = f"{image_path}:{os.path.getmtime(resolved_path)}"
        if cache_key in self.embedding_cache:
            logging.info(f"Using cached features for {os.path.basename(image_path)}")
            return self.embedding_cache[cache_key]
        
        logging.info(f"Extracting features from: {os.path.basename(image_path)}")
        
        try:
            # Load and preprocess
            image = Image.open(resolved_path).convert('RGB')
            image_resized = image.resize((224, 224))
            image_rgb = np.array(image_resized)
            
            features = {}
            
            # HSV histogram - FIXED VERSION WITH PROPER MASKING
            hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            config = FEATURE_CONFIGS['hsv']
            
            # Create proper mask for low saturation regions - CRITICAL FIX
            mask = None
            if config.get('preprocessing', {}).get('mask_low_saturation', False):
                sat_threshold = config['preprocessing'].get('saturation_threshold', 25)
                mask = np.where(hsv_image[:, :, 1] > sat_threshold, 255, 0).astype(np.uint8)
            
            hist = cv2.calcHist([hsv_image], [0, 1, 2], mask, config['bins'], config['ranges'])
            hist = hist.flatten().astype(np.float32)
            
            # Proper normalization
            hist_sum = hist.sum()
            if hist_sum > 1e-10:
                hist = hist / hist_sum
                print(f"DEBUG: HSV query norm: {np.linalg.norm(hist):.6f}")
                print(f"DEBUG: HSV query sum: {hist.sum():.6f}")
                print(f"DEBUG: HSV query range: [{hist.min():.6f}, {hist.max():.6f}]")

                # Only add epsilon if histogram is too sparse
                if np.sum(hist > 0) < len(hist) * 0.1:
                    hist = hist + 1e-10
                    hist = hist / hist.sum()
            else:
                hist = np.ones_like(hist) / len(hist)
            
            features['hsv'] = hist
            
            # ConvNext features - CRITICAL: Keep RAW features to match database
            if self.convnext_model is not None:
                convnext_features = self.convnext_model.extract_features(image_rgb, normalize=False)
                if convnext_features is not None:
                    # Log the norm to verify it matches database range (28-33)
                    norm = np.linalg.norm(convnext_features)
                    logging.debug(f"ConvNeXt query norm: {norm:.4f} (should be 28-33 range)")
                    features['convnext'] = convnext_features.astype(np.float32)  # Keep raw, NO normalization
                else:
                    logging.warning("ConvNeXt model returned None, using zero vector")
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
    
    def find_similar_images_for_new_image(self, target_features, top_n=5, target_image_path=None):
        """Find similar images using clustering-first search."""
        logging.info("Starting clustering-first search...")
        
        # Debug target features
        for feature_type, feature in target_features.items():
            feat_norm = np.linalg.norm(feature)
            feat_shape = feature.shape
            logging.info(f"Target {feature_type}: shape={feat_shape}, norm={feat_norm:.4f}")
        
        try:
            # Use the existing multi_feature_search method (without target_image_path parameter)
            results = self.searcher.multi_feature_search(target_features, top_n=top_n)
            
            if results and len(results) > 0:
                logging.info(f"Clustering search found {len(results)} results")
                # Log first few results for debugging
                for i, result in enumerate(results[:3]):
                    logging.info(f"  {i+1}. {os.path.basename(result['image_path'])}: {result['combined_similarity']:.4f}")
                return results
            else:
                logging.warning("Clustering search returned empty results, trying fallback...")
                return self.fallback_search(target_features, top_n)
                
        except Exception as e:
            logging.error(f"Clustering search failed: {e}")
            import traceback
            traceback.print_exc()
            return self.fallback_search(target_features, top_n)
    
    def fallback_search(self, target_features, top_n=5):
        """Improved fallback search using raw feature data with proper similarity computation."""
        logging.info("Using fallback search...")
        
        all_similarities = {}
        
        # Use the raw feature data from the searcher
        for feature_type, target_feature in target_features.items():
            if feature_type not in self.searcher.features_data:
                logging.warning(f"No feature data for {feature_type}")
                continue
            
            feature_data = self.searcher.features_data[feature_type]
            logging.info(f"Fallback search using {len(feature_data)} {feature_type} features")
            
            # Debug target feature
            target_flat = target_feature.flatten().astype(np.float32)
            target_norm_value = np.linalg.norm(target_flat)
            logging.info(f"Target {feature_type} feature: shape={target_flat.shape}, norm={target_norm_value:.4f}")
            
            if target_norm_value < 1e-8:
                logging.warning(f"Target {feature_type} feature has near-zero norm, skipping")
                continue
            
            similarities = []
            image_paths = []
            
            for image_path, db_feature in feature_data.items():
                if db_feature is not None:
                    try:
                        db_flat = db_feature.flatten().astype(np.float32)
                        
                        # Check feature dimensions match
                        if len(db_flat) != len(target_flat):
                            logging.warning(f"Feature dimension mismatch: target={len(target_flat)}, db={len(db_flat)}")
                            continue
                        
                        if feature_type == 'hsv':
                            # For HSV: Use Bhattacharyya coefficient (both should be normalized histograms)
                            target_hist = np.maximum(target_flat, 1e-10)
                            db_hist = np.maximum(db_flat, 1e-10)
                            
                            target_hist = target_hist / np.sum(target_hist)
                            db_hist = db_hist / np.sum(db_hist)
                            
                            similarity = np.sum(np.sqrt(target_hist * db_hist))
                            
                        else:  # ConvNeXt - CRITICAL FIX
                            # For ConvNeXt: Use dot product of RAW features (both have same scale)
                            # Since both query and database features are raw (norm ~31), 
                            # we can use direct dot product
                            similarity = np.dot(target_flat, db_flat)
                            
                            # Normalize by the product of norms for cosine similarity
                            target_norm = np.linalg.norm(target_flat)
                            db_norm = np.linalg.norm(db_flat)
                            
                            if target_norm > 1e-8 and db_norm > 1e-8:
                                similarity = similarity / (target_norm * db_norm)
                            else:
                                similarity = 0.0
                            
                            # Ensure similarity is in valid range
                            similarity = max(-1.0, min(1.0, similarity))
                        
                        similarities.append(similarity)
                        image_paths.append(image_path)
                        
                    except Exception as e:
                        logging.debug(f"Error computing similarity for {image_path}: {e}")
                        continue
            
            if not similarities:
                logging.warning(f"No valid similarities computed for {feature_type}")
                continue
            
            # Log similarity statistics
            similarities = np.array(similarities)
            logging.info(f"{feature_type} similarities: min={np.min(similarities):.4f}, max={np.max(similarities):.4f}, mean={np.mean(similarities):.4f}")
            
            # Get top results for this feature type
            top_indices = np.argsort(similarities)[::-1][:top_n * 3]  # Get more candidates
            
            weight = self.weights.get(feature_type, 0.5)
            logging.info(f"Using weight {weight} for {feature_type}")
            
            for idx in top_indices:
                if idx < len(image_paths):
                    image_path = image_paths[idx]
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
        final_results = results[:top_n]
        
        if final_results:
            logging.info(f"Fallback search found {len(final_results)} results")
            # Log top results for debugging
            for i, result in enumerate(final_results[:3]):
                logging.info(f"  {i+1}. {os.path.basename(result['image_path'])}: {result['combined_similarity']:.4f} (details: {result['feature_details']})")
        else:
            logging.warning("Fallback search found no results")
        return final_results
    
    def debug_hsv_pipeline(self, target_features):
        """Debug HSV processing pipeline."""
        print("=== HSV PIPELINE DEBUG ===")
        
        # Check target features
        if 'hsv' in target_features:
            hsv_feat = target_features['hsv']
            print(f"HSV query: shape={hsv_feat.shape}, sum={hsv_feat.sum():.6f}, norm={np.linalg.norm(hsv_feat):.6f}")
        
        # Check cluster assignments
        assignments = self.searcher.clustering.assign_query_to_clusters(target_features, top_k_clusters=5)
        if 'hsv' in assignments:
            print(f"HSV clusters found: {len(assignments['hsv'])}")
            for cluster in assignments['hsv'][:3]:
                print(f"  Cluster {cluster['cluster_id']}: distance={cluster['distance']:.4f}")
        else:
            print("ERROR: No HSV cluster assignments!")
        
        # Check FAISS indices
        if 'hsv' in self.searcher.faiss_indices:
            hsv_indices = self.searcher.faiss_indices['hsv']
            print(f"HSV FAISS indices: {len(hsv_indices)} clusters")
        else:
            print("ERROR: No HSV FAISS indices!")
            
    def print_individual_rankings(self, target_features, integrated_results, target_image_path=None):
        """Print individual feature rankings alongside integrated results."""
        print(f"\n{'='*70}")
        print("INDIVIDUAL vs INTEGRATED FEATURE RANKINGS")
        print(f"{'='*70}")
        
    def get_individual_rankings(self, target_features, top_k=8):
        """Get individual rankings for each feature type."""
        individual_rankings = {}
        
        for feature_type, target_feature in target_features.items():
            if feature_type not in self.searcher.features_data:
                logging.warning(f"No feature data for {feature_type}")
                continue
            
            feature_data = self.searcher.features_data[feature_type]
            logging.info(f"Computing individual {feature_type} rankings from {len(feature_data)} features")
            
            similarities = []
            image_paths = []
            
            target_flat = target_feature.flatten().astype(np.float32)
            
            for image_path, db_feature in feature_data.items():
                if db_feature is not None:
                    try:
                        db_flat = db_feature.flatten().astype(np.float32)
                        
                        if len(db_flat) != len(target_flat):
                            continue
                        
                        if feature_type == 'hsv':
                            # Bhattacharyya coefficient for HSV
                            target_hist = np.maximum(target_flat, 1e-10)
                            db_hist = np.maximum(db_flat, 1e-10)
                            
                            target_hist = target_hist / np.sum(target_hist)
                            db_hist = db_hist / np.sum(db_hist)
                            
                            similarity = np.sum(np.sqrt(target_hist * db_hist))
                        else:
                            # Cosine similarity for ConvNeXt
                            target_norm = target_flat / (np.linalg.norm(target_flat) + 1e-8)
                            db_norm = db_flat / (np.linalg.norm(db_flat) + 1e-8)
                            
                            similarity = np.dot(target_norm, db_norm)
                            similarity = max(-1.0, min(1.0, similarity))
                        
                        similarities.append(similarity)
                        image_paths.append(image_path)
                        
                    except Exception as e:
                        continue
            
            if similarities:
                # Sort by similarity and create ranking
                similarities = np.array(similarities)
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                rankings = []
                for idx in top_indices:
                    rankings.append({
                        'image_path': image_paths[idx],
                        'similarity': float(similarities[idx])
                    })
                
                individual_rankings[feature_type] = rankings
                logging.info(f"{feature_type} rankings: top similarity = {rankings[0]['similarity']:.4f}")
            else:
                individual_rankings[feature_type] = []
                logging.warning(f"No valid {feature_type} rankings computed")
        
        return individual_rankings
        
        # Print individual rankings
        for feature_type, rankings in individual_rankings.items():
            print(f"\n{feature_type.upper()} ONLY Rankings:")
            print("-" * 40)
            for i, result in enumerate(rankings[:5]):
                print(f"  {i+1}. {os.path.basename(result['image_path']):<25} | {result['similarity']:.4f}")
        
        # Print integrated rankings
        print(f"\nINTEGRATED Rankings:")
        print("-" * 40)
        for i, result in enumerate(integrated_results[:5]):
            print(f"  {i+1}. {os.path.basename(result['image_path']):<25} | {result['combined_similarity']:.4f}")
            details = result.get('feature_details', {})
            if details:
                detail_str = " | ".join([f"{k}:{v:.3f}" for k, v in details.items()])
                print(f"      ({detail_str})")
        
        # Quick overlap analysis
        if 'hsv' in individual_rankings and individual_rankings['hsv']:
            hsv_ids = {os.path.basename(r['image_path']) for r in individual_rankings['hsv'][:5]}
            integrated_ids = {os.path.basename(r['image_path']) for r in integrated_results[:5]}
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
            
            resolved_path = self.resolve_image_path(image_path)
            
            if resolved_path and os.path.exists(resolved_path):
                try:
                    img = Image.open(resolved_path)
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
        similar_images = self.find_similar_images_for_new_image(target_features, top_n=8, target_image_path=image_path)
        
        if similar_images:
            logging.info(f"Found {len(similar_images)} similar images")
            
            # Display results
            self.display_results(image_path, similar_images)

            self.print_individual_rankings(target_features, similar_images, target_image_path=image_path)
            
            # Print text results
            print(f"\nSimilarity Search Results for: {os.path.basename(image_path)}")
            print("=" * 60)
            for i, result in enumerate(similar_images):
                print(f"{i+1}. {os.path.basename(result['image_path'])}")
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
    try:
        searcher = InteractiveSimilaritySearch(use_cuda=use_cuda)
        return searcher.run_interactive_session(image_path)
    except Exception as e:
        logging.error(f"Error occurred in interactive mode: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_interactive_mode()
    if success:
        logging.info("Interactive search completed!")
    else:
        logging.error("Interactive search failed!")
