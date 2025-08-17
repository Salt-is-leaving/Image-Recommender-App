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
from convnext_extractor import ConvNeXtFeatureExtractor
from clustering import Clustering

logging.basicConfig(level=logging.INFO)

class InteractiveSimilaritySearch:
    """Interactive similarity search with embedding storage."""
    
    def __init__(self, use_cuda=True):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        
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
        self.clustering = Clustering()
        
        # Load clustering data
        if not self.clustering.load_clustering_data():
            raise RuntimeError("Failed to load clustering data")
        
        # Initialize features_data attribute in clustering
        self.clustering.features_data = {}
        if not self.clustering.load_features_for_search():
            logging.warning("Failed to load features for search")
        
        # Embedding storage for new images for which we dont have embeddings yet
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

            # DEBUG Test extraction to verify normalization behavior
            logging.info("Testing ConvNeXt extractor normalization...")
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            test_features = self.convnext_model.extract_features(test_image, normalize=False)
            
            if test_features is not None:
                test_norm = np.linalg.norm(test_features)
                logging.info(f"ConvNeXt test extraction: norm={test_norm:.4f}")
                
                if test_norm < 2.0:
                    logging.error("ConvNeXt extractor is normalizing features internally!")
                    logging.error("This will cause mismatch with database features")
                elif test_norm > 50.0:
                    logging.warning(f"ConvNeXt features have unusually high norm: {test_norm:.4f}")
                else:
                    logging.info("ConvNeXt extractor normalization looks correct")

            logging.info(f"ConvNeXt model loaded on {self.device}")
        except Exception as e:
            logging.error(f"Failed to initialize ConvNeXt model: {e}")
            self.convnext_model = None
    
    def load_embedding_cache(self):
        # Load cached embeddings for new images 
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
        success = self.clustering.load_features_for_search()
        
        if success:
            logging.info("Database features loaded")
            for ft, assignments in self.clustering.cluster_assignments.items():
                n_clusters = len(set(assignments.values()))
                logging.info(f" {ft}: {len(assignments)} images in {n_clusters} clusters")
        return success

    def resolve_image_path(self, image_path):
        """Find image in database directories."""
        return resolve_image_path(image_path)
    
    
    def build_search_indices(self):
        logging.info("Building search indices...")
    
        # Build FAISS indices if available
        success = self.clustering.build_indices()
        
        if success:
            if hasattr(self.clustering, 'faiss_indices') and self.clustering.faiss_indices:
                total_indices = sum(len(indices) for indices in self.clustering.faiss_indices.values())
                logging.info(f"FAISS indices built: {total_indices} cluster indices")
            else:
                logging.info("Direct computation ready (FAISS not available)")
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
            
            # HSV histogram extraction, proper masking, raw features before clustering
            hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            config = FEATURE_CONFIGS['hsv']
            
            # Create a mask for low saturation regions
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
            # Store raw HSV features (before clustering preprocessing)
            features['hsv'] = hist
            
            # Keep raw ConvNext features to match DB
            if self.convnext_model is not None:
                convnext_features = self.convnext_model.extract_features(image_rgb, normalize=False)
                if convnext_features is not None:
                    # Log the norm to verify it matches database range (28-33)
                    norm = np.linalg.norm(convnext_features)
                    logging.debug(f"ConvNeXt raw query norm: {norm:.4f} (should be 28-33 range)")
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
    
    def find_similar_images_for_new_image(self, target_features, top_n=8, target_image_path=None):
        # Find similar images using clustering-first search
        logging.info("Starting clustering-first search...")
        
        # Apply preprocessing once here, then pass to Searcher, otherwise the loop tries to reduce dim again
        processed_features = {}

        # Debug target features
        for feature_type, raw_feature in target_features.items():
            if feature_type not in self.clustering.cluster_assignments:
                logging.warning(f"No clustering data for {feature_type}")
                continue
            
            # Apply the SAME preprocessing as used in Clustering: for HSV - Standardization → PCA (288→48) → L1 normalization
            # for ConvNeXt - Skip standardization → PCA (1024→128) → No normalization
            processed_feature = self.clustering.preprocess_query_feature(raw_feature, feature_type)
            if processed_feature is not None:
                processed_features[feature_type] = processed_feature
                logging.info(f"Processed {feature_type}: {raw_feature.shape} → {processed_feature.shape}")
            else:
                logging.error(f"Failed to preprocess {feature_type} features")
        
        if not processed_features:
            logging.error("No valid processed features for search")
            return []
        
        try:
            # Use clustering's multi-feature search directly
            results = self.clustering.multi_feature_search(
                processed_features, 
                self.weights, 
                top_n=top_n *2 # Get more results to filter
            )
            
            # Filter out target image if it's a DB comparison
            if target_image_path:
                target_basename = os.path.basename(target_image_path)
                results = [
                    r for r in results 
                    if os.path.basename(r['image_path']) != target_basename
                ]
            
             # Limit to requested number
            results = results[:top_n]

            if results:
                logging.info(f"Found {len(results)} similar images")
                for i, result in enumerate(results[:3]):
                    logging.info(f"  {i+1}. {os.path.basename(result['image_path'])}: {result['combined_similarity']:.4f}")
                return results
            else:
                logging.warning("No similar images found")
                return []
            
        except Exception as e:
            logging.error(f"Search failed: {e}")
            import traceback
            traceback.print_exc()
            return []
                
    def get_individual_rankings(self, target_features, top_k=8):
        """Get individual rankings for each feature type using clustering pipeline."""
        individual_rankings = {}
        
        # Process raw features to get preprocessed versions first 
        processed_features = {}
        for feature_type, target_feature in target_features.items():
            if feature_type not in self.clustering.cluster_assignments:
                logging.warning(f"No clustering data for {feature_type}")
                continue
                
            # Apply clustering preprocessing once
            processed_feature = self.clustering.preprocess_query_feature(target_feature, feature_type)
            if processed_feature is None:
                logging.warning(f"Failed to preprocess {feature_type} for individual ranking")
                continue
            
            processed_features[feature_type] = processed_feature

        # Now use the preprocessed features for individual searches (separate, not nested loop)
        for feature_type, processed_feature in processed_features.items():
            logging.info(f"Computing individual {feature_type} rankings using clustering search")
        
            try:
                 # Use clustering search directly - search within clusters for single feature
                cluster_assignments = self.clustering.assign_query_to_clusters(
                    {feature_type: processed_feature}, top_k_clusters=3
                )
                
                if feature_type not in cluster_assignments:
                    individual_rankings[feature_type] = []
                    continue
                
                results = []
                
                # Search within assigned clusters
                for cluster_info in cluster_assignments[feature_type]:
                    cluster_id = cluster_info['cluster_id']
                    
                    # Get all images in this cluster
                    cluster_images = [
                        img_path for img_path, assigned_cluster 
                        in self.clustering.cluster_assignments[feature_type].items()
                        if assigned_cluster == cluster_id
                    ]
                    
                    # Compare with each image in cluster
                    for image_path in cluster_images:
                        if image_path in self.clustering.features_data[feature_type]:
                            db_feature = self.clustering.features_data[feature_type][image_path]
                            if db_feature is not None:
                                # Preprocess database feature
                                db_processed = self.clustering.preprocess_query_feature(db_feature, feature_type)
                                if db_processed is not None:
                                    # Calculate similarity
                                    if feature_type == 'hsv':
                                        similarity = self.clustering.compute_bhattacharyya_similarity(
                                            processed_feature, db_processed
                                        )
                                    else:
                                        # Cosine similarity for ConvNeXt
                                        similarity = np.dot(processed_feature, db_processed)
                                        query_norm = np.linalg.norm(processed_feature)
                                        db_norm = np.linalg.norm(db_processed)
                                        if query_norm > 1e-8 and db_norm > 1e-8:
                                            similarity = similarity / (query_norm * db_norm)
                                    
                                    results.append({
                                        'image_path': image_path,
                                        'similarity': max(0.0, min(1.0, similarity)),
                                        'cluster_id': cluster_id
                                    })
                
                # Sort and convert to individual ranking format
                results.sort(key=lambda x: x['similarity'], reverse=True)
                rankings = []
                for result in results[:top_k]:
                    rankings.append({
                        'image_path': result['image_path'],
                        'similarity': result['similarity']
                    })
            
                individual_rankings[feature_type] = rankings
                
                if rankings:
                    logging.info(f"{feature_type} rankings: top similarity = {rankings[0]['similarity']:.4f}")
                else:
                    logging.warning(f"No valid {feature_type} rankings computed")
                
            except Exception as e:
                logging.error(f"Error getting {feature_type} rankings: {e}")
                individual_rankings[feature_type] = []
        return individual_rankings
    
    def print_individual_rankings(self, target_features, integrated_results, target_image_path=None):
        """Print individual feature rankings alongside integrated results."""
        print(f"\n{'='*70}")
        print("INDIVIDUAL vs INTEGRATED FEATURE RANKINGS")
        print(f"{'='*70}")

        # Get individual rankings for each feature type
        individual_rankings = self.get_individual_rankings(target_features, top_k=8)
        
        # Print individual rankings
        for feature_type, rankings in individual_rankings.items():
            print(f"\n{feature_type.upper()} ONLY Rankings:")
            print("-" * 40)
            if rankings:
                for i, result in enumerate(rankings[:5]):
                    print(f"  {i+1}. {os.path.basename(result['image_path']):<25} | {result['similarity']:.4f}")
            else:
                print("No rankings available")
        
        # Print integrated rankings
        print(f"\nINTEGRATED Rankings:")
        print("-" * 40)
        for i, result in enumerate(integrated_results[:5]):
            print(f"  {i+1}. {os.path.basename(result['image_path']):<25} | {result['combined_similarity']:.4f}")
            details = result.get('feature_details', {})
            if details:
                detail_str = " | ".join([f"{k}:{v:.3f}" for k, v in details.items()])
                print(f"      ({detail_str})")
        
        # HSV-specific analysis
        if 'hsv' in individual_rankings and individual_rankings['hsv']:
            hsv_ids = {os.path.basename(r['image_path']) for r in individual_rankings['hsv'][:5]}
            integrated_ids = {os.path.basename(r['image_path']) for r in integrated_results[:5]}
            overlap = len(hsv_ids.intersection(integrated_ids))
            
            hsv_scores = [r['similarity'] for r in individual_rankings['hsv'][:5]]
            print(f"\nHSV Analysis:")
            print(f"  Score range: {min(hsv_scores):.3f} - {max(hsv_scores):.3f}")
            print(f"  Overlap with integrated: {overlap}/5 images")
            
            # Check for low HSV scores indicating conversion issue
            if max(hsv_scores) < 0.5:
                print(f"  WARNING: HSV scores seem low - check score conversion")
            else:
                print(f"  HSV scores look healthy")
            
            # DEBUG Advanced Multi-Feature Analysis
        if 'hsv' in individual_rankings and 'convnext' in individual_rankings:
            hsv_results = individual_rankings['hsv']
            convnext_results = individual_rankings['convnext']
            
            # Find potential multi-feature candidates
            hsv_images = {os.path.basename(r['image_path']): r['similarity'] for r in hsv_results[:10]}
            convnext_images = {os.path.basename(r['image_path']): r['similarity'] for r in convnext_results[:10]}
            
            # Check for any overlap in top results
            common_in_top10 = set(hsv_images.keys()).intersection(set(convnext_images.keys()))
            
            print(f"\nMULTI-FEATURE ANALYSIS:")
            print(f"  Common images in top-10: {len(common_in_top10)}")
            
            if common_in_top10:
                print(f"  Potential multi-feature candidates:")
                for img in list(common_in_top10)[:3]:
                    hsv_score = hsv_images[img]
                    convnext_score = convnext_images[img]
                    combined_weighted = hsv_score * 0.469 + convnext_score * 0.531
                    print(f"    {img}: HSV={hsv_score:.3f}, ConvNeXt={convnext_score:.3f}, Combined={combined_weighted:.3f}")
            else:
                print(f"  No overlap between HSV and ConvNeXt top-10 results")
                print(f"  But integrated search found multi-feature matches through clustering")
                
                
            # DELETE Show integrated vs individual comparison
            integrated_images = {os.path.basename(r['image_path']) for r in integrated_results[:5]}
            hsv_overlap = len(hsv_images.keys() & integrated_images)
            convnext_overlap = len(convnext_images.keys() & integrated_images)
            
            print(f"\nINTEGRATED vs INDIVIDUAL OVERLAP:")
            print(f"  HSV overlap with integrated: {hsv_overlap}/5")
            print(f"  ConvNeXt overlap with integrated: {convnext_overlap}/5")

    def display_results(self, target_image_path, similar_images, target_features):
    
        # Get individual rankings
        individual_rankings = self.get_individual_rankings(target_features, top_k=5)
        
        # Create subplots: 3 rows x 5 columns
        fig, axes = plt.subplots(3, 5, figsize=(16, 8))
        
        plt.suptitle("Image Similarity Search - Integrated vs Individual Features", fontsize=14)
        
        # Row 1: Query + Top 4 Integrated Results
        # Query image
        try:
            target_img = Image.open(target_image_path)
            axes[0, 0].imshow(target_img)
            axes[0, 0].set_title(f"Query Image\n{os.path.basename(target_image_path)}", fontsize=10)
            axes[0, 0].axis('off')
        except:
            axes[0, 0].text(0.5, 0.5, "Query\nImage Error", ha='center', va='center')
            axes[0, 0].axis('off')
        
        # Top 4 integrated results
        for i in range(4):
            col = i + 1
            if i < len(similar_images):
                result = similar_images[i]
                image_path = result['image_path']
                similarity = result['combined_similarity']
                
                resolved_path = self.resolve_image_path(image_path)
                
                if resolved_path and os.path.exists(resolved_path):
                    try:
                        img = Image.open(resolved_path)
                        axes[0, col].imshow(img)
                        axes[0, col].set_title(f"Integrated #{i+1}\nScore: {similarity:.3f}", fontsize=9)
                        axes[0, col].axis('off')
                    except:
                        axes[0, col].text(0.5, 0.5, f"Int #{i+1}\nError", ha='center', va='center')
                        axes[0, col].axis('off')
                else:
                    axes[0, col].text(0.5, 0.5, f"Int #{i+1}\nNot Found", ha='center', va='center')
                    axes[0, col].axis('off')
            else:
                axes[0, col].axis('off')
        
        # Row 2: HSV-only results
        axes[1, 0].text(0.5, 0.5, "HSV\nOnly", ha='center', va='center', fontsize=12, weight='bold')
        axes[1, 0].axis('off')
        
        if 'hsv' in individual_rankings and individual_rankings['hsv']:
            for i in range(4):
                col = i + 1
                if i < len(individual_rankings['hsv']):
                    result = individual_rankings['hsv'][i]
                    image_path = result['image_path']
                    similarity = result['similarity']
                    
                    resolved_path = self.resolve_image_path(image_path)
                    
                    if resolved_path and os.path.exists(resolved_path):
                        try:
                            img = Image.open(resolved_path)
                            axes[1, col].imshow(img)
                            axes[1, col].set_title(f"HSV #{i+1}\nScore: {similarity:.3f}", fontsize=9)
                            axes[1, col].axis('off')
                        except:
                            axes[1, col].text(0.5, 0.5, f"HSV #{i+1}\nError", ha='center', va='center')
                            axes[1, col].axis('off')
                    else:
                        axes[1, col].text(0.5, 0.5, f"HSV #{i+1}\nNot Found", ha='center', va='center')
                        axes[1, col].axis('off')
                else:
                    axes[1, col].axis('off')
        else:
            for i in range(1, 5):
                axes[1, i].text(0.5, 0.5, "No HSV\nResults", ha='center', va='center')
                axes[1, i].axis('off')
        
        # Row 3: ConvNeXt-only results  
        axes[2, 0].text(0.5, 0.5, "ConvNeXt\nOnly", ha='center', va='center', fontsize=12, weight='bold')
        axes[2, 0].axis('off')
        
        if 'convnext' in individual_rankings and individual_rankings['convnext']:
            for i in range(4):
                col = i + 1
                if i < len(individual_rankings['convnext']):
                    result = individual_rankings['convnext'][i]
                    image_path = result['image_path']
                    similarity = result['similarity']
                    
                    resolved_path = self.resolve_image_path(image_path)
                    
                    if resolved_path and os.path.exists(resolved_path):
                        try:
                            img = Image.open(resolved_path)
                            axes[2, col].imshow(img)
                            axes[2, col].set_title(f"ConvNeXt #{i+1}\nScore: {similarity:.3f}", fontsize=9)
                            axes[2, col].axis('off')
                        except:
                            axes[2, col].text(0.5, 0.5, f"CNX #{i+1}\nError", ha='center', va='center')
                            axes[2, col].axis('off')
                    else:
                        axes[2, col].text(0.5, 0.5, f"CNX #{i+1}\nNot Found", ha='center', va='center')
                        axes[2, col].axis('off')
                else:
                    axes[2, col].axis('off')
        else:
            for i in range(1, 5):
                axes[2, i].text(0.5, 0.5, "No ConvNeXt\nResults", ha='center', va='center')
                axes[2, i].axis('off')
        
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
        
        # Extract raw features
        target_features = self.extract_features_from_image(image_path)
        if target_features is None:
            logging.error("Failed to extract features")
            return False
        
        # Find similar images
        similar_images = self.find_similar_images_for_new_image(target_features, top_n=8, target_image_path=image_path)
        
        if similar_images:
            logging.info(f"Found {len(similar_images)} similar images")
            
            # Display results
            self.display_results(image_path, similar_images, target_features)
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
