import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
from collections import defaultdict

# FAISS with simple error handling
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from config import PATH_TO_SSD, ENHANCED_SIMILARITY_WEIGHTS
from db_api import load_features_from_pickle
from clustering import EnhancedClusteringPipeline

logging.basicConfig(level=logging.INFO)

class ClusteringFirstSearch:
   
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu and FAISS_AVAILABLE and faiss.get_num_gpus() > 0
        self.weights = ENHANCED_SIMILARITY_WEIGHTS['clustering_optimized']
        
        self.clustering = EnhancedClusteringPipeline()
        self.features_data = {}
        self.faiss_indices = {}
        self.indices = {}  # For compatibility
    
    def load_and_process_features(self):
        """Load features and clustering data."""
        if not self.clustering.load_clustering_data():
            logging.error("No clustering data found. Run clustering mode first!")
            return False
        
        # Load raw features
        for feature_type in ['efficientnet', 'hsv']: 
            features = load_features_from_pickle(feature_type)
            if features:
                self.features_data[feature_type] = features
        
        # Create compatibility indices
        self.create_compatibility_indices()
        return True
    
    def create_compatibility_indices(self):
        """Create indices for backward compatibility."""
        for feature_type, features in self.features_data.items():
            if feature_type not in self.clustering.cluster_assignments:
                continue
            
            cluster_assignments = self.clustering.cluster_assignments[feature_type]
            common_ids = list(set(features.keys()) & set(cluster_assignments.keys()))
            
            if not common_ids:
                continue
            
            # Create processed feature matrix
            processed_features = []
            valid_ids = []
            
            for image_id in common_ids:
                feature = features[image_id]
                if feature is not None:
                    processed_feature = self.clustering.preprocess_query_feature(feature, feature_type)
                    if processed_feature is not None:
                        processed_features.append(processed_feature)
                        valid_ids.append(image_id)
            
            if processed_features:
                feature_matrix = np.array(processed_features)
                metric = 'bhattacharyya' if feature_type == 'hsv' else 'cosine'
                
                self.indices[feature_type] = {
                    'features': feature_matrix,
                    'image_ids': valid_ids,
                    'metric': metric
                }

    def build_indices(self):
        """Build FAISS indices per cluster."""
        if not FAISS_AVAILABLE:
            return True  # Direct computation fallback
        
        for feature_type in self.features_data:
            if feature_type not in self.clustering.cluster_assignments:
                continue
            
            features = self.features_data[feature_type]
            assignments = self.clustering.cluster_assignments[feature_type]
            
            # Group features by cluster
            cluster_groups = defaultdict(list)
            cluster_image_ids = defaultdict(list)
            
            for image_id, cluster_id in assignments.items():
                if image_id in features and features[image_id] is not None:
                    processed_feature = self.clustering.preprocess_query_feature(
                        features[image_id], feature_type
                    )
                    if processed_feature is not None:
                        cluster_groups[cluster_id].append(processed_feature)
                        cluster_image_ids[cluster_id].append(image_id)
            
            # Build FAISS index for each cluster
            feature_indices = {}
            for cluster_id, cluster_features in cluster_groups.items():
                if len(cluster_features) < 2:
                    continue
                
                feature_matrix = np.array(cluster_features).astype(np.float32)
                dimension = feature_matrix.shape[1]
                
                try:
                    if feature_type == 'hsv':
                        index = faiss.IndexFlatL2(dimension)
                    else:
                        index = faiss.IndexFlatIP(dimension)
                    
                    index.add(feature_matrix)
                    
                    feature_indices[cluster_id] = {
                        'index': index,
                        'image_ids': cluster_image_ids[cluster_id],
                        'features': feature_matrix
                    }
                except Exception:
                    # Fallback to direct computation for this cluster
                    feature_indices[cluster_id] = {
                        'index': None,
                        'image_ids': cluster_image_ids[cluster_id],
                        'features': feature_matrix
                    }
            
            self.faiss_indices[feature_type] = feature_indices
        
        return True
    
    def compute_bhattacharyya_similarity(self, hist1, hist2):
        """Bhattacharyya coefficient for histogram similarity is much better than chi-squared."""
        # Ensure normalized histograms
        hist1_norm = hist1 / (np.sum(hist1) + 1e-8)
        hist2_norm = hist2 / (np.sum(hist2) + 1e-8)
        
        # Bhattacharyya coefficient (ranges 0-1, higher is more similar)
        return np.sum(np.sqrt(hist1_norm * hist2_norm))
    
    def search_within_clusters(self, query_features, feature_type, top_k=50):
        """Enhanced search within relevant clusters with better error handling."""
        try:
            # Get cluster assignments for query
            cluster_assignments = self.clustering.assign_query_to_clusters(
                {feature_type: query_features}, top_k_clusters=3
            )
            
            if feature_type not in cluster_assignments:
                logging.warning(f"No cluster assignments found for {feature_type}")
                return self.fallback_direct_search(query_features, feature_type, top_k)
            
            # Preprocess query feature
            query_processed = self.clustering.preprocess_query_feature(query_features, feature_type)
            if query_processed is None:
                logging.warning(f"Failed to preprocess query feature for {feature_type}")
                return self.fallback_direct_search(query_features, feature_type, top_k)
            
            query_vector = query_processed.reshape(1, -1).astype(np.float32)
            all_results = []
            
            # Search in candidate clusters
            clusters_searched = 0
            for cluster_info in cluster_assignments[feature_type]:
                cluster_id = cluster_info['cluster_id']
                
                # Check if cluster data exists
                if (feature_type not in self.faiss_indices or 
                    cluster_id not in self.faiss_indices[feature_type]):
                    logging.debug(f"No FAISS index for {feature_type} cluster {cluster_id}")
                    continue
                
                cluster_data = self.faiss_indices[feature_type][cluster_id]
                index = cluster_data['index']
                image_ids = cluster_data['image_ids']
                feature_matrix = cluster_data['features']
                
                clusters_searched += 1
                
                try:
                    if FAISS_AVAILABLE and index is not None:
                        # FAISS search
                        search_k = min(top_k, len(image_ids))
                        scores, indices = index.search(query_vector, search_k)
                        
                        for score, idx in zip(scores[0], indices[0]):
                            if idx >= 0 and idx < len(image_ids):
                                # Convert FAISS score to similarity
                                if feature_type == 'hsv':
                                    similarity = 1.0 / (1.0 + max(0.0, float(score)))
                                else:
                                    similarity = max(0.0, min(1.0, float(score)))
                                
                                all_results.append({
                                    'image_id': image_ids[idx],
                                    'similarity': similarity,
                                    'cluster_id': cluster_id
                                })
                    else:
                        # Direct computation fallback
                        all_results.extend(
                            self.direct_cluster_search(query_vector[0], feature_matrix, 
                                                      image_ids, feature_type, cluster_id)
                        )
                except Exception as e:
                    logging.warning(f"Error searching cluster {cluster_id} for {feature_type}: {e}")
                    continue
            
            logging.debug(f"Searched {clusters_searched} clusters for {feature_type}")
            
            if not all_results:
                logging.warning(f"No results from cluster search for {feature_type}, using fallback")
                return self.fallback_direct_search(query_features, feature_type, top_k)
            
            # Sort and return top results
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            return all_results[:top_k]
            
        except Exception as e:
            logging.error(f"Critical error in cluster search for {feature_type}: {e}")
            return self.fallback_direct_search(query_features, feature_type, top_k)

    def direct_cluster_search(self, query_flat, feature_matrix, image_ids, feature_type, cluster_id):
        """Direct similarity computation within a cluster."""
        results = []
        
        for i, candidate_feature in enumerate(feature_matrix):
            try:
                if feature_type == 'hsv':
                    similarity = self.compute_bhattacharyya_similarity(query_flat, candidate_feature)
                else:
                    # Cosine similarity for other features
                    query_norm = np.linalg.norm(query_flat)
                    candidate_norm = np.linalg.norm(candidate_feature)
                    if query_norm > 0 and candidate_norm > 0:
                        similarity = np.dot(query_flat, candidate_feature) / (query_norm * candidate_norm)
                    else:
                        similarity = 0.0
                
                similarity = max(0.0, min(1.0, similarity))
                results.append({
                    'image_id': image_ids[i],
                    'similarity': similarity,
                    'cluster_id': cluster_id
                })
            except Exception:
                continue
        
        return results
    
    def fallback_direct_search(self, query_features, feature_type, top_k):
        """Fallback direct search when cluster search fails."""
        logging.info(f"Using direct search fallback for {feature_type}")
        
        if feature_type not in self.features_data:
            return []
        
        # Get common image IDs
        if hasattr(self.clustering, 'cluster_assignments') and feature_type in self.clustering.cluster_assignments:
            valid_ids = list(self.clustering.cluster_assignments[feature_type].keys())
        else:
            valid_ids = list(self.features_data[feature_type].keys())
        
        # Preprocess query
        query_processed = self.clustering.preprocess_query_feature(query_features, feature_type)
        if query_processed is None:
            query_processed = query_features.flatten().astype(np.float32)
            query_processed = query_processed / (np.linalg.norm(query_processed) + 1e-8)
        
        results = []
        for image_id in valid_ids[:1000]:  # Limit for performance
            if image_id in self.features_data[feature_type]:
                db_feature = self.features_data[feature_type][image_id]
                if db_feature is not None:
                    db_processed = self.clustering.preprocess_query_feature(db_feature, feature_type)
                    if db_processed is None:
                        db_processed = db_feature.flatten().astype(np.float32)
                        db_processed = db_processed / (np.linalg.norm(db_processed) + 1e-8)
                    
                    if feature_type == 'hsv':
                        similarity = self.compute_bhattacharyya_similarity(query_processed, db_processed)
                    else:
                        similarity = np.dot(query_processed, db_processed)
                    
                    results.append({
                        'image_id': image_id,
                        'similarity': max(0.0, min(1.0, similarity)),
                        'cluster_id': -1  # Indicate direct search
                    })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def multi_feature_search(self, query_features, top_n=10):
        """Multi-feature clustering-first search."""
        all_similarities = defaultdict(lambda: {
            'total_score': 0.0,
            'count': 0,
            'details': {},
            'cluster_info': {}
        })

        feature_search_counts = {'efficientnet': 0, 'hsv': 0} 
        # Search using each available feature type
        for feature_type, features in query_features.items():
            if feature_type not in self.weights:
                logging.warning(f"Feature type {feature_type} not in weights config")
                continue
            
            if feature_type not in self.clustering.cluster_assignments:
                logging.warning(f"No clustering data for {feature_type}")
                continue
            
            weight = self.weights[feature_type]
            
            try:
                # Get cluster-based search results
                results = self.search_within_clusters(features, feature_type, top_k=100)   # Increased top_k for better quality/coverage
                feature_search_counts[feature_type] = len(results)
                
                logging.info(f"{feature_type}: Found {len(results)} candidates")
                
                # Process results with proper weighting
                for result in results:
                    image_id = result['image_id']
                    similarity = result['similarity']
                    
                    # Apply feature-specific weight
                    weighted_similarity = similarity * weight
                    
                    all_similarities[image_id]['total_score'] += weighted_similarity
                    all_similarities[image_id]['count'] += 1
                    all_similarities[image_id]['details'][feature_type] = similarity
                    all_similarities[image_id]['cluster_info'][feature_type] = result['cluster_id']
                
            except Exception as e:
                logging.error(f"Error searching with {feature_type}: {e}")
                continue
        
        # Log search statistics
        logging.info("Feature search results:")
        for ft, count in feature_search_counts.items():
            logging.info(f"  {ft}: {count} candidates")
        
        # Convert to list and apply intelligent filtering
        final_results = []

        # Prioritize multi-feature matches
        multi_feature_results = []
        single_feature_results = []

        for image_id, data in all_similarities.items():
            # Require at least 2 features to match for better quality
            result = {
                    'image_id': image_id,
                    'combined_similarity': data['total_score'],
                    'feature_count': data['count'],
                    'feature_details': data['details'],
                    'cluster_info': data['cluster_info']
            }
        
            if data['count'] >= 2:
                multi_feature_results.append(result)
            elif data['count'] == 1: #and data['total_score'] > 0.3:  # High-quality single feature
                single_feature_results.append(result)
        
        # Sort both lists by similarity
        multi_feature_results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        single_feature_results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        
        # Combine results: prioritize multi-feature matches, then high-quality single-feature matches
        final_results = []
        final_results.extend(multi_feature_results[:top_n//2])
        
        remaining_slots = top_n - len(final_results)
        if remaining_slots > 0:
            final_results.extend(single_feature_results[:remaining_slots])
        
        # If we still don't have enough, add more single-feature
        if len(final_results) < top_n and len(single_feature_results) > remaining_slots:
            additional_needed = top_n - len(final_results)
            final_results.extend(single_feature_results[remaining_slots:remaining_slots + additional_needed])

        # Final sort
        final_results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        
        logging.info(f"Final results: {len(multi_feature_results)} multi-feature + {len(final_results) - len(multi_feature_results)} single-feature")
        return final_results[:top_n]

    def get_individual_rankings(self, target_features, top_k=10):
        """Get individual rankings for each feature type."""
        individual_rankings = {}
        
        for feature_type, features in target_features.items():
            if feature_type not in self.clustering.cluster_assignments:
                continue
            
            try:
                results = self.search_within_clusters(features, feature_type, top_k=top_k*2)
                individual_rankings[feature_type] = results[:top_k]
            except Exception as e:
                logging.error(f"Error getting {feature_type} rankings: {e}")
                individual_rankings[feature_type] = []
        
        return individual_rankings  
    def find_similar_images(self, target_image_id, top_n=10):
        """Find similar images for a database image."""
        # Extract target features
        target_features = {}
        for feature_type, features in self.features_data.items():
            if target_image_id in features:
                target_features[feature_type] = features[target_image_id]
        
        if not target_features:
            return []
        
        return self.multi_feature_search(target_features, top_n)
    
    def find_image_path(self, image_id):
        """Find full path of an image."""
        for root, dirs, files in os.walk(PATH_TO_SSD):
            if image_id in files:
                return os.path.join(root, image_id)
        return None

def comparison_mode(target_image_id=None, use_gpu=False, compare_all_methods=True, enable_clustering=True):
    """Enhanced comparison mode showing individual HSV rankings."""
    logging.info("Starting enhanced comparison mode with HSV rankings...")
    
    searcher = ClusteringFirstSearch(use_gpu=use_gpu)
    
    if not searcher.load_and_process_features():
        logging.error("Failed to load features and clustering data")
        return False
    
    if not searcher.build_indices():
        logging.error("Failed to build cluster indices")
        return False
    
    # Get available images
    available_images = list(next(iter(searcher.clustering.cluster_assignments.values())).keys())
    
    if target_image_id is None:
        target_image_id = available_images[0]
    
    if target_image_id not in available_images:
        logging.error(f"Target image {target_image_id} not found")
        return False
    
    # Extract target features
    target_features = {}
    for feature_type, features in searcher.features_data.items():
        if target_image_id in features:
            target_features[feature_type] = features[target_image_id]
    
    if not target_features:
        logging.error("Failed to extract target features")
        return False
    
    logging.info(f"Target Image: {target_image_id}")
    logging.info(f"Available features: {list(target_features.keys())}")
    
    # Get individual rankings
    individual_rankings = searcher.get_individual_rankings(target_features, top_k=10)
    
    # Get integrated ranking
    integrated_results = searcher.multi_feature_search(target_features, top_n=10)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"ENHANCED SIMILARITY SEARCH RESULTS FOR: {target_image_id}")
    print(f"{'='*80}")
    
    # Show individual feature rankings
    for feature_type, rankings in individual_rankings.items():
        print(f"\n{feature_type.upper()} ONLY RANKINGS:")
        print("-" * 50)
        for i, result in enumerate(rankings[:5]):
            print(f"  {i+1:2d}. {result['image_id']:<30} | Score: {result['similarity']:.4f}")
    
    # Show integrated results
    print(f"\nINTEGRATED RANKINGS (EfficientNet + HSV):")
    print("-" * 50)
    for i, result in enumerate(integrated_results[:10]):
        print(f"  {i+1:2d}. {result['image_id']:<30} | Score: {result['combined_similarity']:.4f}")
        
        details = result.get('feature_details', {})
        if details:
            feature_breakdown = " | ".join([f"{k}: {v:.3f}" for k, v in details.items()])
            print(f"      Individual scores: {feature_breakdown}")
        print()
    
    # HSV ranking analysis
    if 'hsv' in individual_rankings:
        hsv_rankings = individual_rankings['hsv']
        if hsv_rankings:
            hsv_scores = [r['similarity'] for r in hsv_rankings[:5]]
            print(f"\nHSV RANKING ANALYSIS:")
            print(f"HSV Score Range: {min(hsv_scores):.4f} - {max(hsv_scores):.4f}")
            print(f"HSV Average: {np.mean(hsv_scores):.4f}")
            
            # Check overlap with integrated
            hsv_ids = {r['image_id'] for r in hsv_rankings[:5]}
            integrated_ids = {r['image_id'] for r in integrated_results[:5]}
            overlap = len(hsv_ids.intersection(integrated_ids))
            print(f"HSV overlap with integrated top-5: {overlap}/5")
    
    return True


if __name__ == "__main__":
    target = "pixels-amanjakhar-1124468.jpg"
    success = comparison_mode(target_image_id=target, use_gpu=False)
    
    if success:
        logging.info("Clustering-first comparison completed successfully!")
    else:
        logging.error("Clustering-first comparison failed!")
