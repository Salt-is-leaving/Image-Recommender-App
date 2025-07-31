import os
import numpy as np
import pickle
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

from config import PICKLE_PATH, get_enhanced_cluster_config
from db_api import load_features_from_pickle

logging.basicConfig(level=logging.INFO)

class EnhancedClusteringPipeline:
    """Simple clustering pipeline for multi-feature similarity search."""
    
    def __init__(self):
        self.feature_types = ['efficientnet', 'hsv']  
        self.cluster_models = {}
        self.pca_models = {}
        self.scalers = {}
        self.cluster_assignments = {}
        self.cluster_centers = {}
    
    def load_all_features(self):
        """Load features and find common image IDs."""
        features_data = {}
        common_image_ids = None
        
        for feature_type in self.feature_types:
            features = load_features_from_pickle(feature_type)
            if features:
                features_data[feature_type] = features
                current_ids = set(features.keys())
                common_image_ids = current_ids if common_image_ids is None else common_image_ids.intersection(current_ids)
        
        common_image_ids = list(common_image_ids) if common_image_ids else []
        logging.info(f"Found {len(common_image_ids)} images with all feature types")
        return features_data, common_image_ids
    
    def preprocess_features(self, features_dict, feature_type, image_ids):
        """Preprocess features: standardize, PCA, normalize."""
        config = get_enhanced_cluster_config(feature_type)
        
        # Convert to matrix
        feature_list = []
        valid_ids = []
        
        for image_id in image_ids:
            if image_id in features_dict and features_dict[image_id] is not None:
                feature = features_dict[image_id]
                feature_list.append(feature.flatten().astype(np.float32))
                valid_ids.append(image_id)
        
        if not feature_list:
            return None, None
        
        feature_matrix = np.array(feature_list)
        
        # Standardization
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)
        self.scalers[feature_type] = scaler
        
        # PCA if needed (especially for over-dimensioned HSV)
        if feature_matrix.shape[1] > config['pca_dimensions']:
            pca = PCA(n_components=config['pca_dimensions'], random_state=42)
            feature_matrix = pca.fit_transform(feature_matrix)
            self.pca_models[feature_type] = pca
        else:
            self.pca_models[feature_type] = None
        
        # Normalize
        feature_matrix = normalize(feature_matrix, norm='l2')
        
        return feature_matrix, valid_ids
    
    def cluster_features(self, feature_matrix, feature_type, image_ids):
        """Perform K-means clustering."""
        config = get_enhanced_cluster_config(feature_type)
        n_clusters = config['base_clusters']
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)
        
        self.cluster_models[feature_type] = kmeans
        self.cluster_centers[feature_type] = kmeans.cluster_centers_
        
        # Store assignments
        assignments = {}
        for i, image_id in enumerate(image_ids):
            assignments[image_id] = int(cluster_labels[i])
        self.cluster_assignments[feature_type] = assignments
        
        return cluster_labels, kmeans.cluster_centers_
    
    def save_clustering_data(self):
        """Save clustering data to pickle file."""
        cluster_file = os.path.join(PICKLE_PATH, 'enhanced_cluster_data.pkl')
        
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
        cluster_file = os.path.join(PICKLE_PATH, 'enhanced_cluster_data.pkl')
        
        if not os.path.exists(cluster_file):
            return False
        
        try:
            with open(cluster_file, 'rb') as f:
                clustering_data = pickle.load(f)
            
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
            
            # Apply same preprocessing
            if feature_type in self.scalers:
                query_scaled = self.scalers[feature_type].transform(query_reshaped)
            else:
                query_scaled = query_reshaped
            
            if feature_type in self.pca_models and self.pca_models[feature_type] is not None:
                query_pca = self.pca_models[feature_type].transform(query_scaled)
            else:
                query_pca = query_scaled
            
            query_norm = normalize(query_pca, norm='l2')
            return query_norm[0]
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
            
            centers = self.cluster_centers[feature_type]
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
        features_data, common_image_ids = self.load_all_features()
        if not features_data or len(common_image_ids) < 10:
            logging.error("Insufficient features for clustering")
            return False
        
        # Process only enabled high-performance feature type
        for feature_type in self.feature_types:
            if feature_type not in features_data:
                continue
            
            logging.info(f"Processing {feature_type}...")
            
            # Preprocess
            feature_matrix, valid_ids = self.preprocess_features(
                features_data[feature_type], feature_type, common_image_ids
            )
            
            if feature_matrix is not None:
                self.cluster_features(feature_matrix, feature_type, valid_ids)
        
        # Save results
        success = self.save_clustering_data()
        if success:
            logging.info("Clustering pipeline completed successfully")
            return success

def run_enhanced_clustering():
    """Main function to run enhanced clustering."""
    pipeline = EnhancedClusteringPipeline()
    return pipeline.run_clustering_pipeline()

if __name__ == "__main__":
    success = run_enhanced_clustering()
    if success:
        logging.info("Enhanced clustering completed successfully!")
    else:
        logging.error("Enhanced clustering failed!")
