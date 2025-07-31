import os
import numpy as np
import pickle
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

from config import PICKLE_PATH, get_cluster_config
from db_api import load_features_from_pickle

logging.basicConfig(level=logging.INFO)

class ClusteringPipeline:
    """Simple clustering pipeline for multi-feature similarity search."""
    
    def __init__(self):
        self.feature_types = ['convnext', 'hsv']  
        self.cluster_models = {}
        self.pca_models = {}
        self.scalers = {}
        self.cluster_assignments = {} # image_path -> cluster_id
        self.cluster_centers = {}
    
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
    
    def preprocess_features(self, features_dict, feature_type, image_paths):
        """Preprocess features: standardize, PCA, normalize."""
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
        
        if feature_type == 'convnext':
            logging.info(f"Skipping standardization for ConvNeXt to preserve extractor normalization")
        # ConvNeXt features are already properly normalized by our extractor
        # Just validate they look reasonable
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
                return None, None
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
        else:
            # L2 normalization for other deep features
            if normalization == 'l2':
                feature_matrix = normalize(feature_matrix, norm='l2')
                logging.debug(f"L2 normalization applied for {feature_type}")
        return feature_matrix, valid_paths
    
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
            
            # FIXED: Consistent query normalization
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
                # L2 normalization for other features
                if normalization == 'l2':
                    query_norm = normalize(query_pca, norm='l2')
                    logging.debug("L2 normalization applied for query")
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
    
    def assign_query_to_clusters(self, query_features, top_k_clusters=3):
        """Assign query to clusters for each feature type."""
        cluster_assignments = {}
        
        for feature_type, query_feature in query_features.items():
            if feature_type not in self.cluster_centers:
                continue
            
            query_processed = self.preprocess_query_feature(query_feature, feature_type)
            if query_processed is None:
                continue
            
            centers = self.cluster_centers[feature_type] # Calculate distances to cluster centers
            distances = np.linalg.norm(centers - query_processed, axis=1)
            nearest_indices = np.argsort(distances)[:top_k_clusters]
            
            cluster_candidates = []
            for idx in nearest_indices:
                cluster_candidates.append({
                    'cluster_id': int(idx),
                    'distance': float(distances[idx])
                })
            
            cluster_assignments[feature_type] = cluster_candidates
        
        return cluster_assignments
    
    def run_clustering_pipeline(self):
        logging.info("Starting clustering pipeline...")
        
        # Load features
        features_data, common_image_paths = self.load_all_features()
        if not features_data or len(common_image_paths) < 10:
            logging.error("Insufficient features for clustering")
            return False
        
        # Process each feature type
        for feature_type in self.feature_types:
            if feature_type not in features_data:
                continue
            
            logging.info(f"Processing {feature_type}...")
            
            # Preprocess
            feature_matrix, valid_paths = self.preprocess_features(
                features_data[feature_type], feature_type, common_image_paths
            )
            
            if feature_matrix is not None:
                self.cluster_features(feature_matrix, feature_type, valid_paths)
        
        # Save results
        success = self.save_clustering_data()
        if success:
            logging.info("Clustering pipeline completed successfully")
            return success

def run_clustering():
    """Main function to run clustering."""
    pipeline = ClusteringPipeline()
    return pipeline.run_clustering_pipeline()

if __name__ == "__main__":
    success = run_clustering()
    if success:
        logging.info("Clustering completed successfully!")
    else:
        logging.error("Clustering failed!")
