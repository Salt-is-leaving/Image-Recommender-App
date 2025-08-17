import os
import numpy as np
import pickle
import logging
import gc 
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

from config import PICKLE_PATH, PERFORMANCE_CONFIGS, get_cluster_config, get_global_cache
from db_api import load_features_from_pickle

# FAISS with error handling
try:
    import faiss
    FAISS_AVAILABLE = True
    # Test FAISS functionality
    test_index = faiss.IndexFlatL2(10)
    FAISS_FUNCTIONAL = True
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_FUNCTIONAL = False
    faiss = None
    logging.warning("FAISS not available - using direct computation fallback")
except Exception as e:
    FAISS_AVAILABLE = True
    FAISS_FUNCTIONAL = False
    logging.warning(f"FAISS available but not functional: {e}")

logging.basicConfig(level=logging.INFO)

class Clustering:
    """Simple clustering pipeline for multi-feature similarity search."""
    
    def __init__(self):
        self.feature_types = ['convnext', 'hsv']  
        self.cluster_models = {}
        self.pca_models = {}
        self.scalers = {}
        self.cluster_assignments = {} # image_path -> cluster_id
        self.cluster_centers = {}
        self.faiss_indices = {}  # feature_type -> {cluster_id: faiss.Index}
        self.cache = get_global_cache()
    
    def load_all_features(self):
        """Load features and find common image paths."""
        features_data = {}
        common_image_paths = None
        
        for feature_type in self.feature_types:
            features = load_features_from_pickle(feature_type)
            if features:
                features_data[feature_type] = features
                current_paths = set(features.keys())
                common_image_paths = current_paths if common_image_paths is None else common_image_paths.intersection(current_paths)
        
        common_image_paths = list(common_image_paths) if common_image_paths else []
        logging.info(f"Found {len(common_image_paths)} images with all feature types")
        return features_data, common_image_paths
    
    def load_features_for_search(self):
        if not hasattr(self, 'features_data'):
            self.features_data = {}

        if not self.features_data:  # Only load if not already loaded
            for feature_type in self.feature_types:
                features = load_features_from_pickle(feature_type)
                if features:
                    self.features_data[feature_type] = features
                    logging.info(f"Loaded {len(features)} {feature_type} features for search")
        return True
    
    def build_indices(self):
        """Build FAISS indices per cluster with caching."""
        if not FAISS_AVAILABLE:
            return True  # Direct computation fallback
        
        for feature_type in self.features_data:
            if feature_type not in self.cluster_assignments:
                continue

            # Try to load from cache first (index caching)
            cluster_config = get_cluster_config(feature_type)
            cached_indices = self.cache.get_cached_indices(feature_type, cluster_config)
            
            if cached_indices is not None:
                self.faiss_indices[feature_type] = cached_indices
                logging.info(f"Loaded cached FAISS indices for {feature_type}")
                continue
        
            # Build indices if not cached
            logging.info(f"Building FAISS indices for {feature_type}...")
            feature_indices = self.build_faiss_indices(feature_type)
            
            if feature_indices:
                self.faiss_indices[feature_type] = feature_indices
            
            # Save to cache
                self.cache.save_indices_to_cache(feature_type, feature_indices, cluster_config)
                logging.info(f"Built and cached FAISS indices for {feature_type}")
        return True
            
      
    def build_faiss_indices(self, feature_type):
        """Build FAISS indices for a feature type with memory management."""
        features = self.features_data[feature_type]
        assignments = self.cluster_assignments[feature_type]
        
        # Group features by cluster
        cluster_groups = defaultdict(list)
        cluster_image_paths = defaultdict(list)
        logging.info(f"Grouping features by clusters for {feature_type}...")
        
        # Process in chunks for memory efficiency 
        chunk_size = PERFORMANCE_CONFIGS.get('chunk_size', 1000)
        assignment_items = list(assignments.items())
        
        for chunk_start in range(0, len(assignment_items), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(assignment_items))
            chunk_items = assignment_items[chunk_start:chunk_end]
            
            for image_path, cluster_id in chunk_items:
                if image_path in features and features[image_path] is not None:
                    processed_feature = self.preprocess_query_feature(
                        features[image_path], feature_type
                    )
                    if processed_feature is not None:
                        cluster_groups[cluster_id].append(processed_feature)
                        cluster_image_paths[cluster_id].append(image_path)
            
            # Periodic cleanup
            if chunk_start > 0 and chunk_start % (chunk_size * 10) == 0:
                gc.collect()
        
        # Build indices with memory management
        feature_indices = {}
        total_clusters = len(cluster_groups)
        
        for i, (cluster_id, cluster_features) in enumerate(cluster_groups.items()):
            if len(cluster_features) < 2:
                continue
            
            if i % 10 == 0:
                logging.info(f"Building index for cluster {i+1}/{total_clusters}")
            
            try:
                feature_matrix = np.array(cluster_features).astype(np.float32)
                dimension = feature_matrix.shape[1]
                
                # Create appropriate index type
                if feature_type == 'hsv':
                    index = faiss.IndexFlatL2(dimension)
                    norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
                    feature_matrix = feature_matrix / (norms + 1e-8)
                else:
                    index = faiss.IndexFlatIP(dimension)
                
                index.add(feature_matrix)
                
                feature_indices[cluster_id] = {
                    'index': index,
                    'image_path': cluster_image_paths[cluster_id],
                    'features': feature_matrix
                }
                
                # Clear large matrices from memory immediately - MEMORY MANAGEMENT
                del feature_matrix
                
                # Periodic garbage collection
                if i % 20 == 0:
                    gc.collect()
                
            except Exception as e:
                logging.warning(f"Failed to build index for cluster {cluster_id}: {e}")
                continue
        
        # Final cleanup - MEMORY MANAGEMENT
        del cluster_groups, cluster_image_paths
        gc.collect()
        logging.info(f"Built {len(feature_indices)} FAISS indices for {feature_type}")
        return feature_indices         
    
    def preprocess_features(self, features_dict, feature_type, image_paths):
        """Preprocess features: standardize, PCA, normalize with PCA logic."""
        config = get_cluster_config(feature_type)
        
        # Convert to matrix
        feature_list = []
        valid_paths = []
        
        for image_path in image_paths:
            if image_path in features_dict and features_dict[image_path] is not None:
                feature = features_dict[image_path]
                feature_list.append(feature.flatten().astype(np.float32))
                valid_paths.append(image_path)
        
        if not feature_list:
            return None, None
        
        feature_matrix = np.array(feature_list)
        n_samples, n_features = feature_matrix.shape
        
        if feature_type == 'convnext':
            norms = np.linalg.norm(feature_matrix, axis=1)
            logging.info(f"ConvNeXt norms preserved")
            
            # Check if we now have proper variation in the original magnitudes
            norm_std = np.std(norms)
            if norm_std > 0.1:
                logging.info(f"Good norm variation in raw features: std={norm_std:.3f}")
            else:
                logging.warning(f"Still low norm variation: std={norm_std:.3f}")
        else:
            # Apply standardization for other features (HSV)
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)
            self.scalers[feature_type] = scaler
        
        # PCA: Check dimensions before applying
        pca_dims = config['pca_dimensions']
        max_possible_dims = min(n_samples - 1, n_features)  # Maximum PCA dimensions possible
        
        if feature_matrix.shape[1] > pca_dims and n_samples > pca_dims:
            # Only apply PCA if we have enough samples
            actual_pca_dims = min(pca_dims, max_possible_dims)
            
            pca = PCA(n_components=actual_pca_dims, random_state=42)
            try:
                feature_matrix = pca.fit_transform(feature_matrix)
                self.pca_models[feature_type] = pca
                explained_var = np.sum(pca.explained_variance_ratio_)
                logging.info(f"PCA applied for {feature_type}: {feature_matrix.shape[1]} dims, "
                        f"{explained_var:.3f} variance explained")
            except Exception as e:
                logging.error(f"PCA failed for {feature_type}: {e}")
                logging.info(f"Skipping PCA due to insufficient data (samples: {n_samples}, features: {n_features})")
                self.pca_models[feature_type] = None
        else:
            self.pca_models[feature_type] = None
            if n_samples <= pca_dims:
                logging.info(f"Skipping PCA for {feature_type}: insufficient samples ({n_samples} <= {pca_dims})")
            else:
                logging.info(f"No PCA needed for {feature_type}: {feature_matrix.shape[1]} <= {pca_dims}")
        
        # Apply final normalization L1 only to HSV (it will maintain hist. probability distribution)
        if feature_type == 'hsv':
            normalization = config.get('normalization', 'none')
            if normalization == 'l1':
                feature_matrix = feature_matrix / (np.sum(np.abs(feature_matrix), axis=1, keepdims=True) + 1e-8)
                logging.debug(f"L1 normalization applied for HSV")
                logging.debug(f"Skipping clustering normalization for ConvNeXt (already L2 normalized)")
        return feature_matrix, valid_paths

    def process_in_chunks(self, features_dict, feature_type, image_paths, config):
        """Process large datasets in chunks."""
        chunk_size = PERFORMANCE_CONFIGS.get('memory_management', {}).get('chunk_size', 1000)
        logging.info(f"Processing {len(image_paths)} images in chunks of {chunk_size}")
        
        all_features = []
        all_valid_paths = []
        
        for chunk_start in range(0, len(image_paths), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(image_paths))
            chunk_paths = image_paths[chunk_start:chunk_end]
            
            # Process chunk
            feature_list = []
            valid_paths = []
            
            for image_path in chunk_paths:
                if image_path in features_dict and features_dict[image_path] is not None:
                    feature = features_dict[image_path]
                    feature_list.append(feature.flatten().astype(np.float32))
                    valid_paths.append(image_path)
            
            if feature_list:
                chunk_matrix = np.array(feature_list)
                all_features.append(chunk_matrix)
                all_valid_paths.extend(valid_paths)
                
                # Clear chunk data
                del feature_list
                gc.collect()
        
        if not all_features:
            return None, None
        
        # Combine chunks
        feature_matrix = np.vstack(all_features) if len(all_features) > 1 else all_features[0]
        del all_features
        gc.collect()
        return self.apply_preprocessing(feature_matrix, feature_type, config), all_valid_paths
    
    def process_normally(self, features_dict, feature_type, image_paths, config):
        # This is basically our existing preprocess_features logic. We also use an additional step for larger datasets.
        feature_list = []
        valid_paths = []
        
        for image_path in image_paths:
            if image_path in features_dict and features_dict[image_path] is not None:
                feature = features_dict[image_path]
                feature_list.append(feature.flatten().astype(np.float32))
                valid_paths.append(image_path)
        
        if not feature_list:
            return None, None
        
        feature_matrix = np.array(feature_list)
        return self.apply_preprocessing(feature_matrix, feature_type, config), valid_paths
    
    def apply_preprocessing(self, feature_matrix, feature_type, config):
        """Apply preprocessing steps - extracted from your existing preprocess_features."""
        if feature_type == 'convnext':
            logging.info(f"Skipping standardization for ConvNeXt to preserve extractor normalization")
            norms = np.linalg.norm(feature_matrix, axis=1)
            logging.info(f"ConvNeXt norms range: {np.min(norms):.3f} - {np.max(norms):.3f}")
        
            if np.std(norms) < 0.01:
                logging.warning("ConvNeXt features still have low norm variance - check extractor")
            else:
                logging.info("ConvNeXt features have good norm diversity")     
        else:
            # Apply standardization for other features (HSV)
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)
            self.scalers[feature_type] = scaler
        
        # PCA if needed (especially for over-dimensioned HSV)
        pca_dims = config['pca_dimensions']
        if feature_matrix.shape[1] > pca_dims:
            pca = PCA(n_components=pca_dims, random_state=42)
            try:
                feature_matrix = pca.fit_transform(feature_matrix)
                self.pca_models[feature_type] = pca
                explained_var = np.sum(pca.explained_variance_ratio_)
                logging.info(f"PCA applied for {feature_type}: {feature_matrix.shape[1]} dims, "
                           f"{explained_var:.3f} variance explained")
            except Exception as e:
                logging.error(f"PCA failed for {feature_type}: {e}")
                return None
        else:
            self.pca_models[feature_type] = None
            logging.info(f"No PCA needed for {feature_type}: {feature_matrix.shape[1]} <= {pca_dims}")
        
        normalization = config.get('normalization', 'none')
        if feature_type == 'hsv':
            # L1 normalization for HSV histograms (maintains probability distribution)
            if normalization == 'l1':
                feature_matrix = feature_matrix / (np.sum(np.abs(feature_matrix), axis=1, keepdims=True) + 1e-8)
                logging.debug(f"L1 normalization applied for HSV")
        elif feature_type == 'convnext':
            # Skip normalization for ConvNeXt - already L2 normalized by extractor
            logging.debug(f"Skipping clustering normalization for ConvNeXt (already L2 normalized)")
        return feature_matrix
    
    def assign_query_to_clusters(self, preprocessed_features, top_k_clusters=3):
        """Assign ALREADY PREPROCESSED query features to clusters."""
        cluster_assignments = {}
        
        for feature_type, preprocessed_feature in preprocessed_features.items():
            if feature_type not in self.cluster_centers:
                continue
            
            # Use preprocessed feature directly - NO additional preprocessing
            centers = self.cluster_centers[feature_type]
            distances = np.linalg.norm(centers - preprocessed_feature, axis=1)
            
            logging.debug(f"DEBUG {feature_type}: Preprocessed query shape: {preprocessed_feature.shape}")
            logging.debug(f"DEBUG {feature_type}: Centers shape: {centers.shape}")
            logging.debug(f"DEBUG {feature_type}: Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
            
            nearest_indices = np.argsort(distances)[:top_k_clusters]
            
            cluster_candidates = []
            for idx in nearest_indices:
                cluster_candidates.append({
                    'cluster_id': int(idx),
                    'distance': float(distances[idx])
                })
            
            cluster_assignments[feature_type] = cluster_candidates
        return cluster_assignments

    def cluster_features(self, feature_matrix, feature_type, image_paths):
        """Perform K-means clustering."""
        config = get_cluster_config(feature_type)
        n_clusters = config['base_clusters']
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)
        
        self.cluster_models[feature_type] = kmeans
        self.cluster_centers[feature_type] = kmeans.cluster_centers_
        
        # Store assignments
        assignments = {}
        for i, image_path in enumerate(image_paths):
            assignments[image_path] = int(cluster_labels[i])
        self.cluster_assignments[feature_type] = assignments
        
        return cluster_labels, kmeans.cluster_centers_
    
    def save_clustering_data(self):
        """Save clustering data to pickle file."""
        cluster_file = os.path.join(PICKLE_PATH, 'cluster_data.pkl')
        
        clustering_data = {
            'cluster_models': self.cluster_models,
            'pca_models': self.pca_models,
            'scalers': self.scalers,
            'cluster_assignments': self.cluster_assignments,
            'cluster_centers': self.cluster_centers,
            'feature_types': self.feature_types
        }
        
        try:
            with open(cluster_file, 'wb') as f:
                pickle.dump(clustering_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Clustering data saved to {cluster_file}")
            return True
        except Exception as e:
            logging.error(f"Error saving clustering data: {e}")
            return False
    
    def load_clustering_data(self):
        """Load clustering data from pickle file."""
        cluster_file = os.path.join(PICKLE_PATH, 'cluster_data.pkl')
        
        if not os.path.exists(cluster_file):
            return False
        
        try:
            with open(cluster_file, 'rb') as f:
                clustering_data = pickle.load(f)
            # Load core data
            self.cluster_models = clustering_data.get('cluster_models', {})
            self.pca_models = clustering_data.get('pca_models', {})
            self.scalers = clustering_data.get('scalers', {})
            self.cluster_assignments = clustering_data.get('cluster_assignments', {})
            self.cluster_centers = clustering_data.get('cluster_centers', {})
            self.feature_types = clustering_data.get('feature_types', self.feature_types)
            
            return True
        except Exception as e:
            logging.error(f"Error loading clustering data: {e}")
            return False
    
    def preprocess_query_feature(self, query_feature, feature_type):
        """Preprocess query feature using same pipeline as training."""
        try:
            query_reshaped = query_feature.flatten().reshape(1, -1).astype(np.float32)
            
            # Apply same preprocessing but SKIP standardization for ConvNeXt
            if feature_type == 'convnext':
                # ConvNeXt features are already properly preprocessed by extractor
                query_processed = query_reshaped
                logging.debug("Skipping standardization for ConvNeXt query feature")
            else:
                # Apply standardization for other features
                if feature_type in self.scalers:
                    query_processed = self.scalers[feature_type].transform(query_reshaped)
                else:
                    query_processed = query_reshaped
            # then PCA if available
            if feature_type in self.pca_models and self.pca_models[feature_type] is not None:
                query_pca = self.pca_models[feature_type].transform(query_processed)
            else:
                query_pca = query_processed
            
            # Consistent query normalization
            config = get_cluster_config(feature_type)
            normalization = config.get('normalization', 'none')
            
            if feature_type == 'hsv':
                # L1 normalization for HSV queries
                if normalization == 'l1':
                    query_norm = query_pca / (np.sum(np.abs(query_pca)) + 1e-8)
                    logging.debug("L1 normalization applied for HSV query")
                else:
                    query_norm = query_pca
            elif feature_type == 'convnext':
                # Skip normalization for ConvNeXt - already L2 normalized
                query_norm = query_pca
                logging.debug("Skipping clustering normalization for ConvNeXt query (already L2 normalized)")
            else:
                query_norm = query_pca
            
            result = query_norm[0]
            
            # Final validation
            if not np.isfinite(result).all():
                logging.error(f"Preprocessed query feature contains invalid values for {feature_type}")
                return None
            return result
        except Exception as e:
            logging.error(f"Error preprocessing query feature {feature_type}: {e}")
            return None
    
    def compute_bhattacharyya_similarity(self, hist1, hist2):
        """Bhattacharyya coefficient for histogram similarity."""
        try:
            # Ensure positive values
            hist1_clean = np.maximum(hist1, 1e-12)
            hist2_clean = np.maximum(hist2, 1e-12)
            
            # Normalize histograms
            hist1_norm = hist1_clean / (np.sum(hist1_clean) + 1e-10)
            hist2_norm = hist2_clean / (np.sum(hist2_clean) + 1e-10)
            
            # Compute Bhattacharyya coefficient
            product = hist1_norm * hist2_norm
            product = np.maximum(product, 0.0)
            sqrt_product = np.sqrt(product)
            
            similarity = np.sum(sqrt_product)
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logging.error(f"Error in Bhattacharyya computation: {e}")
            return 0.0

    def multi_feature_search(self, preprocessed_features, weights, top_n=10):
        """Multi-feature search using preprocessed features and dynamic weights."""
        from collections import defaultdict
        
        # Ensure features are loaded for search
        if not self.features_data:
            self.load_features_for_search()
        
        all_similarities = defaultdict(lambda: {
            'total_score': 0.0,
            'count': 0,
            'details': {},
            'cluster_info': {}
        })

        # Search using each available feature type
        for feature_type, preprocessed_feature in preprocessed_features.items():
            if feature_type not in weights:
                logging.warning(f"Feature type {feature_type} not in weights config")
                continue
        
            if feature_type not in self.cluster_assignments:
                logging.warning(f"No clustering data for {feature_type}")
                continue  

            try:
                # Get cluster assignments for this feature
                cluster_assignments = self.assign_query_to_clusters(
                    {feature_type: preprocessed_feature}, top_k_clusters=3
                )
                
                if feature_type not in cluster_assignments:
                    continue
                
                # Search within assigned clusters
                results = []
                for cluster_info in cluster_assignments[feature_type]:
                    cluster_id = cluster_info['cluster_id']
                    
                    # Get all images in this cluster
                    cluster_images = [
                        img_path for img_path, assigned_cluster 
                        in self.cluster_assignments[feature_type].items()
                        if assigned_cluster == cluster_id
                    ]
                    
                    # Compare with each image in cluster
                    for image_path in cluster_images:
                        if feature_type in self.features_data and image_path in self.features_data[feature_type]:
                            db_feature = self.features_data[feature_type][image_path]
                            if db_feature is not None:
                                # Preprocess database feature
                                db_processed = self.preprocess_query_feature(db_feature, feature_type)
                                if db_processed is not None:
                                    # Calculate similarity
                                    if feature_type == 'hsv':
                                        # Bhattacharyya coefficient for histograms
                                        similarity = self.compute_bhattacharyya_similarity(
                                            preprocessed_feature, db_processed
                                        )
                                    else:
                                        # Cosine similarity for ConvNeXt
                                        similarity = np.dot(preprocessed_feature, db_processed)
                                        query_norm = np.linalg.norm(preprocessed_feature)
                                        db_norm = np.linalg.norm(db_processed)
                                        if query_norm > 1e-8 and db_norm > 1e-8:
                                            similarity = similarity / (query_norm * db_norm)
                                    
                                    results.append({
                                        'image_path': image_path,
                                        'similarity': max(0.0, min(1.0, similarity)),
                                        'cluster_id': cluster_id
                                    })
                
                # Apply weights and store results
                weight = weights[feature_type]
                logging.info(f"{feature_type}: Found {len(results)} candidates, weight={weight:.3f}")
                
                for result in results:
                    image_path = result['image_path']
                    similarity = result['similarity']
                    weighted_similarity = similarity * weight
                    
                    all_similarities[image_path]['total_score'] += weighted_similarity
                    all_similarities[image_path]['count'] += 1
                    all_similarities[image_path]['details'][feature_type] = similarity
                    all_similarities[image_path]['cluster_info'][feature_type] = result['cluster_id']

            except Exception as e:
                logging.error(f"Search failed for {feature_type}: {e}")
                continue
        
        # Convert to final results
        multi_feature_results = []
        single_feature_results = []

        for image_path, data in all_similarities.items():
            result = {
                'image_path': image_path,
                'combined_similarity': data['total_score'],
                'feature_count': data['count'],
                'feature_details': data['details'],
                'cluster_info': data['cluster_info']
            }

            if data['count'] >= 2:
                multi_feature_results.append(result)
            else:
                single_feature_results.append(result)
        
        # Sort and combine
        multi_feature_results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        single_feature_results.sort(key=lambda x: x['combined_similarity'], reverse=True)

        # Prioritize multi-feature matches
        final_results = multi_feature_results[:top_n]
        remaining_slots = top_n - len(final_results)
        if remaining_slots > 0:
            final_results.extend(single_feature_results[:remaining_slots])
        
        logging.info(f"Multi-feature search: {len(multi_feature_results)} multi-feature + "
                    f"{len(single_feature_results)} single-feature results")
        
        return final_results[:top_n]
      
    def run_clustering_pipeline(self):
        # Clustering pipeline with chunked processing
        logging.info("Starting clustering pipeline...")
        
        # Clear old cache
        self.cache.clear_old_cache()
        
        # Load features (existing logic)
        features_data, common_image_paths = self.load_all_features()
        if not features_data or len(common_image_paths) < 10:
            logging.error("Insufficient features for clustering")
            return False
        
        # Process each feature type with chunking
        for feature_type in self.feature_types:
            if feature_type not in features_data:
                continue
            
            logging.info(f"Processing {feature_type} with chunked processing...")
            
            # Use chunked preprocessing instead of regular preprocessing
            feature_matrix, valid_paths = self.preprocess_features(
                features_data[feature_type], feature_type, common_image_paths
            )
            
            if feature_matrix is not None:
                self.cluster_features(feature_matrix, feature_type, valid_paths)
                
                # Clear large objects
                del feature_matrix
                gc.collect()
           
        # Save results
        success = self.save_clustering_data()
        if success:
            logging.info("Clustering pipeline completed successfully")
        return success

def run_clustering():
    """Main function to run clustering."""
    pipeline = Clustering()
    return pipeline.run_clustering_pipeline()

if __name__ == "__main__":
    success = run_clustering()
    if success:
        logging.info("Clustering completed successfully!")
    else:
        logging.error("Clustering failed!")
