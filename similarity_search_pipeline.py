import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
import gc
from collections import defaultdict

# FAISS with simple error handling
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

from config import (PATH_TO_SSD, resolve_image_path, get_similarity_weights,  get_global_cache,
                   PERFORMANCE_CONFIGS, get_cluster_config)
from db_api import load_features_from_pickle
from clustering import ClusteringPipeline

logging.basicConfig(level=logging.INFO)

class ClusteringFirstSearch:
   
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu and FAISS_AVAILABLE and faiss.get_num_gpus() > 0
        self.weights = get_similarity_weights() # dynamic weights from config
        logging.info(f"Using dynamic weights: {self.weights}")
        
        self.clustering = ClusteringPipeline()
        self.features_data = {}
        self.faiss_indices = {}
        self.cache = get_global_cache()
      
    def load_and_process_features(self):
        """Load features and clustering data."""
        if not self.clustering.load_clustering_data():
            logging.error("No clustering data found. Run clustering mode first!")
            return False
        
        # Load raw features
        for feature_type in ['convnext', 'hsv']: 
            logging.info(f"Loading {feature_type} features...")
            features = load_features_from_pickle(feature_type)
            if features:
                self.features_data[feature_type] = features
                logging.info(f"Loaded {len(features)} {feature_type} features")
            
            # Periodic cleanup for large datasets
            if len(features) > 50000:
                gc.collect() 
        return True
    
    def resolve_image_path(self, image_path):
        """Find image in database directories."""
        from config import resolve_image_path
        return resolve_image_path(image_path)

    def build_indices(self):
        """Build FAISS indices per cluster with caching."""
        if not FAISS_AVAILABLE:
            return True  # Direct computation fallback
        
        for feature_type in self.features_data:
            if feature_type not in self.clustering.cluster_assignments:
                continue

                # Try to load from cache first - INDEX CACHING
            cluster_config = get_cluster_config(feature_type)
            cached_indices = self.cache.get_cached_indices(feature_type, cluster_config)
            
            if cached_indices is not None:
                self.faiss_indices[feature_type] = cached_indices
                logging.info(f"Loaded cached FAISS indices for {feature_type}")
                continue
        
            # Build indices if not cached
            logging.info(f"Building FAISS indices for {feature_type}...")
            feature_indices = self.build_feature_indices(feature_type)
            
            if feature_indices:
                self.faiss_indices[feature_type] = feature_indices
            
            # Save to cache
                self.cache.save_indices_to_cache(feature_type, feature_indices, cluster_config)
                logging.info(f"Built and cached FAISS indices for {feature_type}")
        return True
            
      
    def build_feature_indices(self, feature_type):
        """Build FAISS indices for a feature type with memory management."""
        features = self.features_data[feature_type]
        assignments = self.clustering.cluster_assignments[feature_type]
        
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
                    processed_feature = self.clustering.preprocess_query_feature(
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
                
    def batch_search_clusters(self, query_batch, feature_type, top_k=50):
        """Optimized batch search across clusters."""
        if not query_batch:
            return []
        
        batch_size = len(query_batch)
        batch_config = PERFORMANCE_CONFIGS['batch_processing']
        
        # Check if batch processing is enabled
        if not batch_config.get('parallel_processing', True) or batch_size == 1:
            # Fall back to individual searches
            return [self.search_within_clusters(query, feature_type, top_k) for query in query_batch]
        
        logging.debug(f"Batch searching {batch_size} queries for {feature_type}")
        
        all_batch_results = [[] for _ in range(batch_size)]
        
        try:
            # Get cluster assignments for all queries in batch
            batch_cluster_assignments = []
            for query_features in query_batch:
                assignments = self.clustering.assign_query_to_clusters(
                    {feature_type: query_features}, top_k_clusters=3
                )
                batch_cluster_assignments.append(assignments.get(feature_type, []))
            
            # Group queries by clusters they need to search
            cluster_to_queries = defaultdict(list)
            for query_idx, cluster_assignments in enumerate(batch_cluster_assignments):
                for cluster_info in cluster_assignments:
                    cluster_id = cluster_info['cluster_id']
                    cluster_to_queries[cluster_id].append(query_idx)
            
            # Batch search within each cluster
            for cluster_id, query_indices in cluster_to_queries.items():
                if (feature_type not in self.faiss_indices or 
                    cluster_id not in self.faiss_indices[feature_type]):
                    continue
                
                cluster_data = self.faiss_indices[feature_type][cluster_id]
                index = cluster_data['index']
                image_path = cluster_data['image_path']
                
                if not FAISS_AVAILABLE or index is None:
                    continue
                
                try:
                    # Prepare batch queries for this cluster
                    cluster_queries = []
                    valid_query_indices = []
                    
                    for query_idx in query_indices:
                        query_processed = self.clustering.preprocess_query_feature(
                            query_batch[query_idx], feature_type
                        )
                        if query_processed is not None:
                            cluster_queries.append(query_processed)
                            valid_query_indices.append(query_idx)
                    
                    if not cluster_queries:
                        continue
                    
                    # Batch FAISS search
                    query_matrix = np.array(cluster_queries).astype(np.float32)
                    search_k = min(top_k, len(image_path))
                    scores, indices = index.search(query_matrix, search_k)
                    
                    # Distribute results back to individual queries
                    for i, query_idx in enumerate(valid_query_indices):
                        if i < len(scores):
                            for score, idx in zip(scores[i], indices[i]):
                                if idx >= 0 and idx < len(image_path):
                                    similarity = self.convert_faiss_score(score, feature_type)
                                    all_batch_results[query_idx].append({
                                        'image_path': image_path[idx],
                                        'similarity': similarity,
                                        'cluster_id': cluster_id
                                    })
                                    
                except Exception as e:
                    logging.warning(f"Batch search failed for cluster {cluster_id}: {e}")
                    continue
            
            # Sort results for each query
            for results in all_batch_results:
                results.sort(key=lambda x: x['similarity'], reverse=True)
            return all_batch_results
            
        except Exception as e:
            logging.error(f"Batch search error: {e}")
            # Fall back to individual searches
            return [self.search_within_clusters(query, feature_type, top_k) for query in query_batch]

    def convert_faiss_score(self, score, feature_type):
        """Convert FAISS score to similarity score."""
        if feature_type == 'hsv':
            max_l2_distance = 1.41  # theoretical max for normalized histograms
            normalized_distance = min(float(score) / max_l2_distance, 1.0)
            similarity = 1.0 - normalized_distance
            return max(0.0, min(1.0, similarity))
        else:
            # ConvNeXt with IndexFlatIP (inner product) - already a similarity
            return max(0.0, min(1.0, float(score)))
    
    def search_within_clusters(self, query_features, feature_type, top_k=50):
        """Search within relevant clusters with caching."""
        # Check cache first for repeated queries
        cache_key = f"search_{feature_type}_{hash(query_features.tobytes())}_{top_k}"
        cached_result = self.cache.get_from_memory_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
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
                image_path = cluster_data['image_path']
                feature_matrix = cluster_data['features']
                
                clusters_searched += 1
                
                try:
                    if FAISS_AVAILABLE and index is not None:
                        # FAISS search
                        search_k = min(top_k, len(image_path))
                        scores, indices = index.search(query_vector, search_k)
                        
                        for score, idx in zip(scores[0], indices[0]):
                            if idx >= 0 and idx < len(image_path):
                                similarity = self.convert_faiss_score(score, feature_type)
                                all_results.append({
                                    'image_path': image_path[idx],
                                    'similarity': similarity,
                                    'cluster_id': cluster_id
                                })
                    else:
                        # Direct computation fallback
                        all_results.extend(
                            self.direct_cluster_search(query_vector[0], feature_matrix, 
                                                      image_path, feature_type, cluster_id)
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
            result = all_results[:top_k]
            
            # Cache the result
            self.cache.store_in_memory_cache(cache_key, result)
            return result
            
        except Exception as e:
            logging.error(f"Critical error in cluster search for {feature_type}: {e}")
            return self.fallback_direct_search(query_features, feature_type, top_k)

    def direct_cluster_search(self, query_flat, feature_matrix, image_path, feature_type, cluster_id):
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
                    'image_path': image_path[i],
                    'similarity': similarity,
                    'cluster_id': cluster_id
                })
            except Exception:
                continue
        return results
    
    def compute_bhattacharyya_similarity(self, hist1, hist2):
        """Bhattacharyya coefficient with proper NaN/Inf handling."""
        try:
            # Ensure positive values first
            hist1_clean = np.maximum(hist1, 1e-12)  # Larger epsilon
            hist2_clean = np.maximum(hist2, 1e-12)
            
            # Normalize histograms
            hist1_norm = hist1_clean / (np.sum(hist1_clean) + 1e-10)
            hist2_norm = hist2_clean / (np.sum(hist2_clean) + 1e-10)
            
            #Check for valid values before sqrt
            product = hist1_norm * hist2_norm
            
            # Remove any NaN or negative values
            product = np.nan_to_num(product, nan=0.0, posinf=0.0, neginf=0.0)
            product = np.maximum(product, 0.0)
            
            # Safe sqrt computation
            sqrt_product = np.sqrt(product)
            
            # Final validation
            if not np.isfinite(sqrt_product).all():
                logging.warning("Non-finite values in Bhattacharyya computation, using fallback")
                return 0.0
            
            similarity = np.sum(sqrt_product)
            
            # Restrict to valid range
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logging.error(f"Error in Bhattacharyya computation: {e}")
        return 0.0
    
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
        chunk_size = 1000
        for i in range(0, min(len(valid_ids), 5000), chunk_size):  # Max 5000 for performance
            chunk_ids = valid_ids[i:i+chunk_size]
            
            for image_path in chunk_ids:
                if image_path in self.features_data[feature_type]:
                    db_feature = self.features_data[feature_type][image_path]
                    if db_feature is not None:
                        db_processed = self.clustering.preprocess_query_feature(db_feature, feature_type)
                        if db_processed is None:
                            db_processed = db_feature.flatten().astype(np.float32)
                            db_processed = db_processed / (np.linalg.norm(db_processed) + 1e-8)
                        
                        if feature_type == 'hsv':
                            similarity = self.compute_bhattacharyya_similarity(query_processed, db_processed)
                        else:
                            similarity = np.dot(query_processed, db_processed)
                            # Ensure it's normalized properly
                            query_norm = np.linalg.norm(query_processed)
                            db_norm = np.linalg.norm(db_processed)
                            if query_norm > 1e-8 and db_norm > 1e-8:
                                similarity = similarity / (query_norm * db_norm)

                        results.append({
                            'image_path': image_path,
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

        feature_search_counts = {'convnext': 0, 'hsv': 0} 
        available_features = []

        # Search using each available feature type
        for feature_type, features in query_features.items():
            if feature_type not in self.weights:
                logging.warning(f"Feature type {feature_type} not in weights config")
                continue
        
            if feature_type not in self.clustering.cluster_assignments:
                logging.warning(f"No clustering data for {feature_type}")
                continue  
            
            available_features.append(feature_type)

            try:
                # Get cluster-based search results
                results = self.search_within_clusters(features, feature_type, top_k=150)   # Increased top_k for better quality/coverage
                if results:
                        feature_search_counts[feature_type] = len(results)
                        weight = self.weights[feature_type]
                        logging.info(f"{feature_type}: Found {len(results)} candidates, weight={weight:.3f}")
            
                for result in results:
                    image_path = result['image_path']
                    similarity = result['similarity']
                    
                    # Debug HSV similarities
                    if feature_type == 'hsv':
                        logging.debug(f"HSV similarity for {os.path.basename(image_path)}: {similarity:.4f}")
                    
                    # Apply feature-specific weight
                    weighted_similarity = similarity * weight
                    
                    all_similarities[image_path]['total_score'] += weighted_similarity
                    all_similarities[image_path]['count'] += 1
                    all_similarities[image_path]['details'][feature_type] = similarity
                    all_similarities[image_path]['cluster_info'][feature_type] = result['cluster_id']
    
            except Exception as e:
                logging.error(f"Search failed for {feature_type}: {e}")
                continue
     
    # Log search statistics
        logging.info("Feature search results:")
        for ft, count in feature_search_counts.items():
            logging.info(f"  {ft}: {count} candidates")
        
        # Convert to list and apply intelligent filtering
        multi_feature_results = []
        single_feature_results = []
        expected_feature_count = len(available_features)


        for image_path, data in all_similarities.items():
            result = {
                'image_path': image_path,
                'combined_similarity': data['total_score'],
                'feature_count': data['count'],
                'feature_details': data['details'],
                'cluster_info': data['cluster_info']
            }

            #  Accept results with at least 2 features OR high single-feature score
            if data['count'] >= 2:
                multi_feature_results.append(result)
            elif data['count'] == 1:
                 # Boost high-scoring single feature results
                if data['total_score'] > 0.3:  # Configurable threshold
                    single_feature_results.append(result)
        
        # Sort both lists by similarity
        multi_feature_results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        single_feature_results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        
        # Combine results: prioritize multi-feature matches
        final_results = []

        # Prioritize multi-feature matches but don't exclude single-feature entirely
        multi_feature_slots = max(top_n // 2, len(multi_feature_results) if len(multi_feature_results) <= top_n else top_n)
        final_results.extend(multi_feature_results[:multi_feature_slots])

        remaining_slots = top_n - len(final_results)
        if remaining_slots > 0:
            final_results.extend(single_feature_results[:remaining_slots])

        # Final sort
        final_results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        
        logging.info(f"Final results: {len(multi_feature_results)} multi-feature + {len(final_results) - len(multi_feature_results)} single-feature")
        
        # Debug log for troubleshooting
        if len(multi_feature_results) == 0:
            logging.warning("No multi-feature results found - checking similarity score distributions")
            all_scores = [(path, data['total_score'], data['count'], data['details']) 
                        for path, data in list(all_similarities.items())[:10]]
            for path, score, count, details in all_scores:
                logging.debug(f"  {os.path.basename(path)}: score={score:.4f}, count={count}, details={details}")
        
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
     
    def find_similar_images(self, target_image_path, top_n=10):
        """Find similar images for a database image."""
        # Extract target features
        target_features = {}
        for feature_type, features in self.features_data.items():
            if target_image_path in features:
                target_features[feature_type] = features[target_image_path]
        
        if not target_features:
            # Try to find by filename
            target_filename = os.path.basename(target_image_path)
        for feature_type, features in self.features_data.items():
            for path, feature in features.items():
                if os.path.basename(path) == target_filename:
                    target_features[feature_type] = feature
                    break
        return self.multi_feature_search(target_features, top_n)
    
def comparison_mode(target_image_path=None, use_gpu=False, compare_all_methods=True, enable_clustering=True):
    """Comparison mode showing individual HSV rankings."""
    logging.info("Starting comparison mode with HSV rankings...")
    searcher = ClusteringFirstSearch(use_gpu=use_gpu)
    
    if not searcher.load_and_process_features():
        logging.error("Failed to load features and clustering data")
        return False
    
    if not searcher.build_indices():
        logging.error("Failed to build cluster indices")
        return False
    
    # Get available images
    available_images = list(next(iter(searcher.clustering.cluster_assignments.values())).keys())
    
    if target_image_path is None:
        target_image_path = available_images[0]
    
    # Check if target exists (try both full path and basename)
    target_found = False
    if target_image_path in available_images:
        target_found = True
    else:
        # Try to find by basename
        target_basename = os.path.basename(target_image_path)
        for img_path in available_images:
            if os.path.basename(img_path) == target_basename:
                target_image_path = img_path
                target_found = True
                break
    
    if not target_found:
        logging.error(f"Target image {target_image_path} not found in database")
        logging.info(f"Available images: {len(available_images)}")
        return False
    
    # Extract target features
    target_features = {}
    for feature_type, features in searcher.features_data.items():
        if target_image_path in features:
            target_features[feature_type] = features[target_image_path]
    
    if not target_features:
        logging.error("Failed to extract target features")
        return False
    
    logging.info(f"Target Image: {target_image_path}")
    logging.info(f"Available features: {list(target_features.keys())}")
    
    # Get individual rankings
    individual_rankings = searcher.get_individual_rankings(target_features, top_k=10)
    
    # Get integrated ranking
    integrated_results = searcher.multi_feature_search(target_features, top_n=10)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"SIMILARITY SEARCH RESULTS FOR: {target_image_path}")
    print(f"{'='*80}")
    
    # Show individual feature rankings
    for feature_type, rankings in individual_rankings.items():
        print(f"\n{feature_type.upper()} ONLY RANKINGS:")
        print("-" * 50)
        for i, result in enumerate(rankings[:5]):
            print(f"  {i+1:2d}. {os.path.basename(result['image_path']):<30} | Score: {result['similarity']:.4f}")
    
    # Show integrated results
    print(f"\nINTEGRATED RANKINGS (ConvNext + HSV):")
    print("-" * 50)
    for i, result in enumerate(integrated_results[:10]):
        print(f"  {i+1:2d}. {result['image_path']:<30} | Score: {result['combined_similarity']:.4f}")
        
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
            hsv_ids = {os.path.basename(r['image_path']) for r in hsv_rankings[:5]}
            integrated_ids = {os.path.basename(r['image_path']) for r in integrated_results[:5]}
            overlap = len(hsv_ids.intersection(integrated_ids))
            print(f"HSV overlap with integrated top-5: {overlap}/5")
    return True


if __name__ == "__main__":
    target = "pixels-amanjakhar-1124468.jpg"
    success = comparison_mode(target_image_path=target, use_gpu=False)
    
    if success:
        logging.info("Clustering-first comparison completed successfully!")
    else:
        logging.error("Clustering-first comparison failed!")
