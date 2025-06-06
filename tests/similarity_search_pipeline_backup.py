import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import logging
import gc
from difflib import SequenceMatcher
try:
    import faiss
except ImportError:
    print("FAISS not installed. Install with: pip install faiss-cpu  # or faiss-gpu")
    raise

from config import (PATH_TO_SSD, PICKLE_PATH, SIMILARITY_CONFIGS, CLUSTERING_CONFIGS, 
                   UMAP_CONFIGS, get_feature_path)
from db_api import (create_connection, load_features_from_pickle, get_images_with_complete_features)

logging.basicConfig(level=logging.INFO)

class SimilaritySearch:
    """FAISS-based similarity search with proper feature handling."""
    
    def __init__(self, use_gpu=False):
        self.features = {}
        self.faiss_indices = {}
        self.image_ids = []
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        # Optimized weights for your feature dimensions
        self.weights = {
            'efficientnet': 0.45,  # High-quality deep features
            'hsv': 0.25,           # Reduced due to over-dimensioning issue
            'lbp': 0.25,           # Reliable texture features  
            'orb': 0.05            # Supplementary keypoint features
        }
        
        # Feature-specific processing
        self.feature_configs = {
            'efficientnet': {'metric': faiss.METRIC_COSINE, 'normalize': True, 'pca_dims': None},
            'hsv': {'metric': faiss.METRIC_L2, 'normalize': True, 'pca_dims': 512},  # Reduce HSV
            'lbp': {'metric': faiss.METRIC_COSINE, 'normalize': True, 'pca_dims': None},
            'orb': {'metric': faiss.METRIC_COSINE, 'normalize': True, 'pca_dims': None}
        }
        
        logging.info(f"FAISS Similarity Search initialized (GPU: {self.use_gpu})")
    
    def load_and_process_features(self):
        """Load features with intelligent preprocessing."""
        logging.info("Loading and processing features for FAISS...")
        
        feature_types = ['efficientnet', 'hsv', 'lbp', 'orb']
        
        for feature_type in feature_types:
            try:
                features = load_features_from_pickle(feature_type)
                if features:
                    processed_features = self._preprocess_features(features, feature_type)
                    self.features[feature_type] = processed_features
                    logging.info(f"✓ Processed {len(processed_features)} {feature_type} features")
                else:
                    logging.warning(f"No {feature_type} features found")
            except Exception as e:
                logging.error(f"Error loading {feature_type}: {e}")
        
        # Get common image IDs
        if self.features:
            all_image_sets = [set(features.keys()) for features in self.features.values()]
            self.image_ids = list(set.intersection(*all_image_sets))
            logging.info(f"Found {len(self.image_ids)} images with complete features")
        
        return len(self.image_ids) > 0
    
    def _preprocess_features(self, features, feature_type):
        """Intelligent feature preprocessing based on type."""
        processed = {}
        config = self.feature_configs[feature_type]
        
        # Convert features to matrix for batch processing
        valid_ids = []
        feature_list = []
        
        for image_id, feature in features.items():
            if feature is None:
                continue
                
            # Handle different feature shapes
            if feature_type == 'orb':
                # ORB: Use median pooling for variable-length descriptors
                if len(feature.shape) > 1:
                    feature = np.median(feature, axis=0)
                feature = feature.astype(np.float32)
            else:
                # Other features: flatten and convert
                if len(feature.shape) > 1:
                    feature = feature.flatten()
                feature = feature.astype(np.float32)
            
            feature_list.append(feature)
            valid_ids.append(image_id)
        
        if not feature_list:
            return processed
        
        # Convert to matrix
        feature_matrix = np.array(feature_list)
        logging.info(f"{feature_type} matrix shape: {feature_matrix.shape}")
        
        # Apply PCA if configured (especially for HSV)
        if config['pca_dims'] and feature_matrix.shape[1] > config['pca_dims']:
            logging.info(f"Applying PCA to {feature_type}: {feature_matrix.shape[1]} -> {config['pca_dims']}")
            
            from sklearn.decomposition import PCA
            pca = PCA(n_components=config['pca_dims'])
            feature_matrix = pca.fit_transform(feature_matrix)
            logging.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Normalize if required
        if config['normalize']:
            if feature_type in ['hsv', 'lbp']:
                # L1 normalization for histograms
                norms = np.sum(np.abs(feature_matrix), axis=1, keepdims=True)
                feature_matrix = feature_matrix / (norms + 1e-8)
            else:
                # L2 normalization for deep features and descriptors
                norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
                feature_matrix = feature_matrix / (norms + 1e-8)
        
        # Convert back to dictionary
        for i, image_id in enumerate(valid_ids):
            processed[image_id] = feature_matrix[i]
        
        return processed
    
    def build_faiss_indices(self):
        """Build FAISS indices for all feature types."""
        logging.info("Building FAISS indices...")
        
        for feature_type, features_dict in self.features.items():
            if not features_dict:
                continue
            
            config = self.feature_configs[feature_type]
            
            # Prepare feature matrix
            valid_ids = []
            feature_matrix = []
            
            for image_id in self.image_ids:
                if image_id in features_dict:
                    feature = features_dict[image_id]
                    if feature is not None:
                        feature_matrix.append(feature)
                        valid_ids.append(image_id)
            
            if not feature_matrix:
                logging.warning(f"No valid features for {feature_type}")
                continue
            
            feature_matrix = np.array(feature_matrix).astype(np.float32)
            d = feature_matrix.shape[1]  # Feature dimension
            
            logging.info(f"Building FAISS index for {feature_type}: {len(feature_matrix)} vectors, {d} dims")
            
            # Choose appropriate FAISS index
            if config['metric'] == faiss.METRIC_COSINE:
                # For cosine similarity, use Inner Product after normalization
                if not config['normalize']:
                    # Normalize if not already done
                    faiss.normalize_L2(feature_matrix)
                index = faiss.IndexFlatIP(d)  # Inner Product for cosine similarity
            else:
                # For L2 distance
                index = faiss.IndexFlatL2(d)
            
            # Use GPU if available and requested
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logging.info(f"  Using GPU for {feature_type}")
            
            # Add vectors to index
            index.add(feature_matrix)
            
            self.faiss_indices[feature_type] = {
                'index': index,
                'image_ids': valid_ids,
                'features': feature_matrix,
                'metric': config['metric']
            }
            
            logging.info(f"✓ Built FAISS index for {feature_type}: {index.ntotal} vectors")
    
    def find_similar_images_faiss(self, target_image_id, top_n=10):
        """Find similar images using FAISS indices."""
        if target_image_id not in self.image_ids:
            logging.error(f"Target image {target_image_id} not found")
            return []
        
        all_similarities = {}
        
        for feature_type, index_data in self.faiss_indices.items():
            faiss_index = index_data['index']
            valid_ids = index_data['image_ids']
            metric = index_data['metric']
            
            if target_image_id not in valid_ids:
                continue
            
            # Get target index
            target_idx = valid_ids.index(target_image_id)
            target_vector = index_data['features'][target_idx:target_idx+1].astype(np.float32)
            
            # Search for similar images
            scores, indices = faiss_index.search(target_vector, min(top_n * 3, len(valid_ids)))
            
            # Convert FAISS results to similarities
            weight = self.weights[feature_type]
            
            for score, idx in zip(scores[0], indices[0]):
                if idx == target_idx:  # Skip self
                    continue
                
                if idx < len(valid_ids):
                    image_id = valid_ids[idx]
                    
                    # Convert score to similarity based on metric
                    if metric == faiss.METRIC_COSINE:
                        similarity = float(score)  # Inner product is already similarity
                    else:  # L2 distance
                        similarity = 1.0 / (1.0 + float(score))  # Convert distance to similarity
                    
                    similarity = max(0.0, min(1.0, similarity))  # Clamp to [0,1]
                    
                    if image_id not in all_similarities:
                        all_similarities[image_id] = {
                            'total_score': 0.0, 
                            'feature_count': 0, 
                            'details': {}
                        }
                    
                    all_similarities[image_id]['total_score'] += similarity * weight
                    all_similarities[image_id]['feature_count'] += 1
                    all_similarities[image_id]['details'][feature_type] = similarity
        
        # Sort results
        sorted_results = []
        for image_id, data in all_similarities.items():
            if data['feature_count'] >= 3:  # Require at least 3 feature types
                sorted_results.append({
                    'image_id': image_id,
                    'combined_similarity': data['total_score'],
                    'feature_count': data['feature_count'],
                    'feature_details': data['details']
                })
        
        sorted_results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        return sorted_results[:top_n]
    
    def test_individual_features_faiss(self, target_image_id, top_n=5):
        """Test each feature type individually using FAISS."""
        logging.info(f"Testing individual FAISS features for {target_image_id}")
        
        results = {}
        
        for feature_type, index_data in self.faiss_indices.items():
            faiss_index = index_data['index']
            valid_ids = index_data['image_ids']
            metric = index_data['metric']
            
            if target_image_id not in valid_ids:
                continue
            
            target_idx = valid_ids.index(target_image_id)
            target_vector = index_data['features'][target_idx:target_idx+1].astype(np.float32)
            
            # Search
            scores, indices = faiss_index.search(target_vector, top_n + 1)
            
            # Convert results
            feature_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == target_idx:  # Skip self
                    continue
                
                if idx < len(valid_ids):
                    image_id = valid_ids[idx]
                    
                    if metric == faiss.METRIC_COSINE:
                        similarity = float(score)
                    else:
                        similarity = 1.0 / (1.0 + float(score))
                    
                    similarity = max(0.0, min(1.0, similarity))
                    feature_results.append((image_id, similarity))
            
            results[feature_type] = feature_results[:top_n]
            
            # Print results
            logging.info(f"\n=== FAISS {feature_type.upper()} Results ===")
            for i, (img_id, sim) in enumerate(feature_results[:top_n]):
                logging.info(f"  {i+1}. {img_id}: {sim:.4f}")
        
        return results
    
    def find_image_path(self, image_id):
        """Find full path of an image."""
        for root, dirs, files in os.walk(PATH_TO_SSD):
            if image_id in files:
                return os.path.join(root, image_id)
        return None
    
    def plot_faiss_results(self, target_image_id, results):
        """Plot FAISS similarity results."""
        n_images = min(len(results), 5) + 1
        fig, axes = plt.subplots(2, n_images, figsize=(4 * n_images, 8))
        
        if n_images == 1:
            axes = axes.reshape(2, 1)
        
        plt.suptitle(f"FAISS Similarity Search Results for {target_image_id}", fontsize=16)
        
        # Plot target image
        target_path = self.find_image_path(target_image_id)
        if target_path and os.path.exists(target_path):
            try:
                target_img = Image.open(target_path)
                axes[0, 0].imshow(target_img)
                axes[0, 0].set_title("Target Image", fontsize=12)
                axes[0, 0].axis('off')
                
                # Show feature info
                feature_info = "Features Available:\n"
                for ft in self.features.keys():
                    if target_image_id in self.features[ft]:
                        feature_info += f"{ft}: ✓\n"
                    else:
                        feature_info += f"{ft}: ✗\n"
                
                axes[1, 0].text(0.1, 0.5, feature_info, fontsize=10, 
                               verticalalignment='center', transform=axes[1, 0].transAxes)
                axes[1, 0].axis('off')
                
            except Exception as e:
                axes[0, 0].text(0.5, 0.5, "Target\nImage Error", ha='center', va='center')
                axes[0, 0].axis('off')
                axes[1, 0].axis('off')
        
        # Plot similar images
        for i, result in enumerate(results[:5]):
            col = i + 1
            if col >= n_images:
                break
            
            image_id = result['image_id']
            similarity = result['combined_similarity']
            details = result['feature_details']
            
            image_path = self.find_image_path(image_id)
            
            if image_path and os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    axes[0, col].imshow(img)
                    axes[0, col].set_title(f"Rank {i + 1}\nFAISS Score: {similarity:.3f}", fontsize=12)
                    axes[0, col].axis('off')
                    
                    # Show detailed scores
                    detail_text = "Feature Scores:\n"
                    for feat_type, score in details.items():
                        detail_text += f"{feat_type}: {score:.3f}\n"
                    
                    axes[1, col].text(0.1, 0.5, detail_text, fontsize=10,
                                     verticalalignment='center', transform=axes[1, col].transAxes)
                    axes[1, col].axis('off')
                    
                except Exception as e:
                    axes[0, col].text(0.5, 0.5, f"Rank {i + 1}\nImage Error", 
                                     ha='center', va='center')
                    axes[0, col].axis('off')
                    axes[1, col].axis('off')
            else:
                axes[0, col].text(0.5, 0.5, f"Rank {i + 1}\nNot Found", 
                                 ha='center', va='center')
                axes[0, col].axis('off')
                axes[1, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        plt.close()


def comparison_mode(target_image_id=None, use_gpu=False):
    """FAISS-based comparison mode."""
    logging.info("=== STARTING FAISS COMPARISON MODE ===")
    
    # Initialize FAISS similarity searcher
    searcher = SimilaritySearch(use_gpu=use_gpu)
    
    # Load and process features
    if not searcher.load_and_process_features():
        logging.error("Could not load features. Run learning mode first!")
        return False
    
    # Build FAISS indices
    searcher.build_faiss_indices()
    
    # Use random target if none specified
    if target_image_id is None:
        target_image_id = searcher.image_ids[7] if len(searcher.image_ids) > 7 else searcher.image_ids[0]
    
    if target_image_id not in searcher.image_ids:
        logging.error(f"Target image {target_image_id} not found in database")
        return False
    
    logging.info(f"Using target image: {target_image_id}")
    
    try:
        # Test individual features
        logging.info("=== TESTING INDIVIDUAL FAISS FEATURES ===")
        individual_results = searcher.test_individual_features_faiss(target_image_id, top_n=5)
        
        # Find similar images using FAISS
        logging.info("=== FINDING SIMILAR IMAGES (FAISS) ===")
        similar_images = searcher.find_similar_images_faiss(target_image_id, top_n=10)
        
        if similar_images:
            logging.info(f"\n✓ Found {len(similar_images)} similar images with FAISS!")
            
            # Print top results
            logging.info(f"Top {min(10, len(similar_images))} most similar images:")
            for i, result in enumerate(similar_images[:10]):
                logging.info(f"{i+1}. {result['image_id']} - Score: {result['combined_similarity']:.4f}")
                details = result['feature_details']
                logging.info(f"    EfficientNet: {details.get('efficientnet', 0):.3f}, "
                           f"HSV: {details.get('hsv', 0):.3f}, "
                           f"LBP: {details.get('lbp', 0):.3f}, "
                           f"ORB: {details.get('orb', 0):.3f}")
            
            # Plot results
            searcher.plot_faiss_results(target_image_id, similar_images)
            
        else:
            logging.error("No similar images found with FAISS!")
        
        return True
        
    except Exception as e:
        logging.error(f"FAISS comparison mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Install FAISS: pip install faiss-cpu  # or faiss-gpu for GPU support
    
    # Test with your black jacket image
    target = "pixels-amanjakhar-1124468.jpg"
    
    success = comparison_mode(target_image_id=target, use_gpu=False)
    
    if success:
        logging.info("FAISS comparison mode completed successfully!")
    else:
        logging.error("FAISS comparison mode failed!")
