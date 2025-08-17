import os
import logging
import time
import psutil
import hashlib
import json
import gc
from contextlib import contextmanager
import pickle

 
# Base paths
BASE_PATH = r"D:\Code_image_rec"
PATH_TO_SSD =   r"E:\data"  #r"D:\images"

# Database path (metadata.db is in the main directory)
DB_PATH = os.path.join(BASE_PATH, 'metadata.db')

# Data paths
COLLECTED_DATA_PATH = os.path.join(BASE_PATH, 'collected_data')

# Pickle storage paths
PICKLE_PATH = os.path.join(BASE_PATH, 'pickles')
CHECKPOINT_PATH = os.path.join(PICKLE_PATH, 'processing_checkpoint.pkl')

# Feature embeddings files (consistent naming)
FEATURE_FILES = {
    'convnext': 'convnext_features.pkl',
    'hsv': 'hsv_features.pkl',
    'combined': 'combined_features.pkl'
}

# Default paths
FINAL_EMBEDDINGS_PATH = os.path.join(PICKLE_PATH, FEATURE_FILES['convnext'])

# Ensure directories exist
os.makedirs(PICKLE_PATH, exist_ok=True)

# Processing parameters
COMPRESS_QUALITY = 75
CHUNK_SIZE = 100

# Image processing parameters
MAX_IMAGE_SIZE = (512, 512)
TARGET_IMAGE_SIZE = (224, 224)
ALLOWED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.svg')

# Checkpoint parameters
CHECKPOINT_INTERVAL = 50

# Feature extraction configurations
FEATURE_CONFIGS = {
    'hsv': {
        'bins': [8, 6, 6], #[18, 16, 16], last before current was [12, 8, 8]
        'ranges': [0, 180, 0, 256, 0, 256],
        'enabled': True,
        'preprocessing': {
        'gaussian_blur': True,  #Reduce noise
        'blur_kernel': (3, 3),
        'histogram_equalization': False,  #Usually not needed for HSV
        'mask_low_saturation': True,  #Ignore grayscale regions
        'saturation_threshold': 25,
        'adaptive_binning': False,    
        'edge_preservation': False,
        'validate_dimensions': True 
    },
    'similarity_metric': 'bhattacharyya',
    'normalization': 'l1_epsilon',
    'dimension_validation': True,
    'error_handling': {
            'fallback_uniform': True,       # Use uniform distribution on errors
            'log_warnings': True,           # Log dimension mismatches
            'skip_corrupted': True          # Skip corrupted images instead of crashing
        }
    },
    'convnext': {  
        'model_name': 'convnext_base.fb_in22k_ft_in1k',
        'input_size': (224, 224),
        'enabled': True,
        'embedding_size': 1024,           
        'batch_size': 64,                  
        'preprocessing': {
            'normalize_mean': [0.485, 0.456, 0.406],  # ImageNet standards
            'normalize_std': [0.229, 0.224, 0.225],
            'interpolation': 'bicubic',                # ConvNeXt uses bicubic
            'use_fp16': True,                          # 2x speed improvement
            'image_quality_check': True,               # Check for corrupted images
            'min_image_size': (32, 32),                # Minimum valid size
            'max_image_size': (4096, 4096)             # Maximum valid size
        },
        'optimization': {
            'use_fp16': True,
            'compile_model': True,   # Can enable with PyTorch 2.0+
            'memory_efficient': True,
            'batch_inference': True   # Process multiple images together
        },
        'error_handling': {
            'skip_corrupted': True,   # Skip corrupted images
            'fallback_zeros': False,  # Use zero vector on extraction failure
            'fallback_random': True,  # Use random fallback to preserve diversity
            'log_errors': True        # Log extraction errors
        }
    }
}

# Memory management parameters
MAX_FEATURES_IN_MEMORY = 100000
FEATURE_BATCH_SIZE = 5000

# Performance optimization parameters
USE_CUDA = True
NUM_WORKERS = 8
PREFETCH_FACTOR = 6

# Similarity search parameters
SIMILARITY_CONFIGS = {
    'weights': {
        'convnext': 0.65,
        'hsv': 0.35,
    },

    'distance_metrics': ['cosine', 'euclidean', 'manhattan']
}

# UMAP parameters - optimized for ConvNext features
UMAP_CONFIGS = {
    'deep_features': {
        'n_components': 3,
        'n_neighbors': 15,
        'min_dist': 0.1,
        'metric': 'cosine'
    },
    'traditional_features': {
        'n_components': 3,
        'n_neighbors': 20,
        'min_dist': 0.05,
        'metric': 'euclidean'
    },
    'combined_features': {
        'n_components': 3,
        'n_neighbors': 12,
        'min_dist': 0.08,
        'metric': 'cosine'
    }
}

#Clustering configurations for clustering-first search
CLUSTERING_CONFIGS = {
    'clustering_per_feature': {
        'convnext': {
            'base_clusters': 35,
            'pca_dimensions': 128,
            'normalization': 'none',  # Already L2 normalized by extractor, #instead of 'l2' to preserve feature space
            'use_minibatch': True,
            'silhouette_optimization': False # No need for silhouette optimization on ConvNeXt and with such a large dataset
        },
    'hsv': {
            'base_clusters': 45,
            'pca_dimensions': 48,  # Dramatically reduce over-dimensioned HSV
            'normalization': 'l1',  # L1 for histogram probability distribution,
            'use_minibatch': True,
            'silhouette_optimization': False,
            'similarity_metric': 'bhattacharyya' 
        }
    }, 
    'preprocessing_pipeline': {
        'standardization': False,
        'pca_enabled': True,
        'pca_variance_threshold': 0.95,
        'normalization_enabled': False,
        'handle_missing_features': True
    },
    'clustering_optimization': {
        'min_cluster_size': 3,
        'max_cluster_size_ratio': 0.2,
        'silhouette_sample_size': 2000,
        'cluster_range_expansion': 10,
        'random_state': 42
    }
}

#FAISS configurations for clustering-first search
FAISS_CONFIGS = {
    'index_types_per_feature': {
        'convnext': {
            'type': 'IndexFlatIP',
            'metric': 'cosine',
            'gpu_enabled': False
        },
        'hsv': {
            'type': 'IndexFlatL2',
            'metric': 'bhattacharyya',
            'gpu_enabled': False
        },
    },
    'search_parameters': {
        'clusters_per_query': 7,
        'candidates_per_cluster': 150,
        'min_cluster_size_for_faiss': 5,
        'use_gpu_when_available': False
    },
    'performance_optimization': {
        'batch_search_size': 100,
        'memory_map_indices': True,
        'precompute_cluster_centers': True
    }
}

# Fallback weights for when dynamic calculation isn't available
FALLBACK_WEIGHTS = {
    'convnext': 0.65,
    'hsv': 0.35
}

SIMILARITY_WEIGHTS = {
    'clustering_optimized': FALLBACK_WEIGHTS,
    'default': FALLBACK_WEIGHTS 
}

PERFORMANCE_CONFIGS = {
    'feature_extraction': {
        'batch_size_convnext': 64,
        'use_fp16': True,
        'prefetch_factor': 6,
        'num_workers': 8,
        'pin_memory': True,
        'use_cuda_streams': True # Use CUDA streams for parallel processing of large datasets 
    },
    'search_optimization': {
        'clustering_first': True,
        'max_clusters_search': 7,
        'candidates_per_cluster': 150,
        'early_stopping_threshold': 0.95,
        'parallel_search': True
    },
     'memory_management': {
        'feature_cache_size': 100000,    # Cache frequently accessed features
        'lazy_loading': True,
        'compression_level': 6,         # Balance between speed and storage
        'memory_mapping': True         # For large datasets
    },
     'caching': {
        'enable_index_cache': True,   # Cache FAISS indices
        'cache_max_age_days': 7,      # Auto-cleanup
        'memory_cache_size': 1000,    # Items in memory
    },
    'batch_processing': {
        'batch_search_size': 32,      # FAISS batch size
        'parallel_processing': True,   # Enable batching
    }
}

# Database optimization settings
DB_OPTIMIZATION = {
    'journal_mode': 'WAL',
    'synchronous': 'NORMAL',
    'cache_size': 10000,
    'temp_store': 'MEMORY'
}

# ========== DYNAMIC IMAGE COUNTING FUNCTIONS ==========
def get_total_images(path_to_ssd=None):
    """Dynamically count total images in the dataset."""
    if path_to_ssd is None:
        path_to_ssd = PATH_TO_SSD
    
    if not os.path.exists(path_to_ssd):
        return 0
    
    supported_extensions = ('.jpg', '.jpeg', '.png', '.svg')
    total_count = 0
    
    try:
        for root, dirs, files in os.walk(path_to_ssd):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    total_count += 1
        return total_count
    except Exception as e:
        print(f"Error counting images: {e}")
        return 0
    
def get_processed_image_count():
    """Get count of processed images from feature files."""
    counts = {}
    
    for feature_type in ['convnext', 'hsv']:
        try:
            filepath = get_feature_path(feature_type)
            if os.path.exists(filepath):
                import pickle
                with open(filepath, 'rb') as f:
                    features = pickle.load(f)
                counts[feature_type] = len(features)
            else:
                counts[feature_type] = 0
        except:
            counts[feature_type] = 0
    return counts

def get_dataset_statistics():
    stats = {
        'total_images_on_disk': get_total_images(),
        'processed_features': get_processed_image_count(),
        'dataset_path': PATH_TO_SSD,
        'pickle_path': PICKLE_PATH
    }
    
    # Calculate processing completeness
    total_images = stats['total_images_on_disk']
    processed_convnext = stats['processed_features'].get('convnext', 0)
    processed_hsv = stats['processed_features'].get('hsv', 0)
    
    if total_images > 0:
        stats['convnext_completion'] = (processed_convnext / total_images) * 100
        stats['hsv_completion'] = (processed_hsv / total_images) * 100
        stats['overall_completion'] = min(stats['convnext_completion'], stats['hsv_completion'])
    else:
        stats['convnext_completion'] = 0
        stats['hsv_completion'] = 0
        stats['overall_completion'] = 0
    return stats


# ========== CENTRALIZED PATH RESOLUTION FUNCTIONS ==========
def resolve_image_path(image_path, search_paths=None):
    """
    Centralized image path resolution function.
    Handles both full paths and filenames, searches in multiple locations.
    
    Args:
        image_path (str): Path or filename to resolve
        search_paths (list, optional): Additional paths to search in
        
    Returns:
        str or None: Resolved absolute path if found, None otherwise
    """
    if search_paths is None:
        search_paths = [PATH_TO_SSD]
    
    # 1. If already absolute and exists, return as-is
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    
    # 2. If just filename provided, search in specified directories
    if not os.path.isabs(image_path):
        # Try direct path in each search directory
        for search_dir in search_paths:
            full_path = os.path.join(search_dir, image_path)
            if os.path.exists(full_path):
                return full_path
        
        # Recursive search in each directory
        for search_dir in search_paths:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    if image_path in files:
                        return os.path.join(root, image_path)
        
        # Try current directory as last resort
        if os.path.exists(image_path):
            return os.path.abspath(image_path)
    return None

def find_image_in_database(image_path, search_paths=None):
    """
    Find image in database directories - optimized for database searches.
    
    Args:
        image_path (str): Full path or just basename to find
        search_paths (list, optional): Directories to search in
        
    Returns:
        str or None: Full path if found, None otherwise
    """
    if search_paths is None:
        search_paths = [PATH_TO_SSD]
    
    # If it's already a full path and exists, return it
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    
    # Extract filename if full path provided
    filename = os.path.basename(image_path)
    
    # Search in database directories
    for search_dir in search_paths:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                if filename in files:
                    return os.path.join(root, filename)
    return None

def validate_image_path(image_path):
    """
    Validate that an image path exists and is a supported image format.
    
    Args:
        image_path (str): Path to validate
        
    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    if not image_path:
        return False, "No path provided"
    
    if not os.path.exists(image_path):
        return False, f"Path does not exist: {image_path}"
    
    if not os.path.isfile(image_path):
        return False, f"Path is not a file: {image_path}"
    
    # Check file extension
    _, ext = os.path.splitext(image_path.lower())
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return False, f"Unsupported image format: {ext}"
    return True, "Valid image path"

# ========== Feutures, FAISS and Clustering MANAGEMENT ==========
def get_feature_path(feature_name):
    """Get the path for feature file based on feature name."""
    if feature_name in FEATURE_FILES:
        filename = FEATURE_FILES[feature_name]
    else:
        filename = f"{feature_name}_features.pkl"
    return os.path.join(PICKLE_PATH, filename)

def get_feature_config(feature_name):
    return FEATURE_CONFIGS.get(feature_name, {})

def get_enabled_features(): #returns a list of enabled features
    return [name for name, config in FEATURE_CONFIGS.items() if config.get('enabled', False)]

def optimize_database_connection(conn):
    """Apply optimization settings to database connection."""
    cursor = conn.cursor()
    for setting, value in DB_OPTIMIZATION.items():
        if isinstance(value, str):
            cursor.execute(f"PRAGMA {setting} = {value}")
        else:
            cursor.execute(f"PRAGMA {setting} = {value}")
    conn.commit()

def get_cluster_config(feature_type):
    """Get clustering configuration for specific feature type."""
    return CLUSTERING_CONFIGS['clustering_per_feature'].get(
        feature_type, 
        CLUSTERING_CONFIGS['clustering_per_feature']['convnext']
    )

def get_faiss_config(feature_type):
    """Get FAISS configuration for specific feature type."""
    return FAISS_CONFIGS['index_types_per_feature'].get(
        feature_type,
        FAISS_CONFIGS['index_types_per_feature']['convnext']
    )

def get_similarity_weights(force_recalculate=False):
    """Get optimal similarity weights based on actual feature quality."""
    try:
        from dynamic_weights_optimizer import get_optimal_similarity_weights
        return get_optimal_similarity_weights(force_recalculate)
    except ImportError:
        logging.warning("Dynamic weight optimizer not available, using fallback")
        return FALLBACK_WEIGHTS
    except Exception as e:
        logging.error(f"Weight optimization failed: {e}")
        return FALLBACK_WEIGHTS
    
# ========== PATH MANAGEMENT FUNCTIONS ==========
def get_clustering_file_path(file_type='cluster_data'):
    """Get path for clustering-related files."""
    file_map = {
        'cluster_data': 'cluster_data.pkl',
        'cluster_statistics': 'cluster_statistics.json',
        'cluster_performance': 'cluster_performance_log.txt'
    }
    
    filename = file_map.get(file_type, f"{file_type}.pkl")
    return os.path.join(PICKLE_PATH, filename)

def get_clustering_status():
    """Get comprehensive status of clustering system."""
    status = {
        'clustering_available': False,
        'faiss_available': False,
        'cluster_data_size_mb': 0,
        'feature_statistics': {},
        'performance_ready': False
    }
    
    # Check clustering data
    cluster_file = get_clustering_file_path('cluster_data')
    if os.path.exists(cluster_file):
        status['clustering_available'] = True
        status['cluster_data_size_mb'] = os.path.getsize(cluster_file) / 1024 / 1024
        
        # Try to load feature statistics
        try:
            import pickle
            with open(cluster_file, 'rb') as f:
                cluster_data = pickle.load(f)
            
            if 'feature_stats' in cluster_data:
                status['feature_statistics'] = cluster_data['feature_stats']
                
            if 'cluster_assignments' in cluster_data:
                available_features = list(cluster_data['cluster_assignments'].keys())
                status['performance_ready'] = len(available_features) >= 2
                
        except Exception:
            pass
    
    # Check FAISS availability
    try:
        import faiss
        status['faiss_available'] = True
    except ImportError:
        status['faiss_available'] = False
    return status

def validate_hsv_config():
    """Validate HSV configuration for common issues."""
    import numpy as np
    config = FEATURE_CONFIGS['hsv']
    issues = []
    
    # Check bins
    bins = config['bins']
    if not isinstance(bins, list) or len(bins) != 3:
        issues.append("HSV bins must be a list of 3 integers")
    
    total_dims = np.prod(bins) if isinstance(bins, list) and len(bins) == 3 else 0
    if total_dims > 1000:
        issues.append(f"HSV dimensions too large: {total_dims}")
    
    # Check ranges
    ranges = config['ranges']
    if not isinstance(ranges, list) or len(ranges) != 6:
        issues.append("HSV ranges must be a list of 6 values")
    return issues

def validate_convnext_config():
    """Validate ConvNeXt configuration."""
    config = FEATURE_CONFIGS['convnext']
    issues = []
    
    # Check embedding size
    if config.get('embedding_size') != 1024:
        issues.append(f"ConvNeXt embedding size should be 1024, got {config.get('embedding_size')}")
    
    # Check batch size
    batch_size = config.get('batch_size', 64)
    if batch_size < 32:
        issues.append(f"ConvNeXt batch size too small: {batch_size} (recommended: 32+)")
    return issues

class FeatureCache:
    """Simple caching system for features and indices."""
    
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or os.path.join(PICKLE_PATH, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.memory_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.max_memory_cache_size = PERFORMANCE_CONFIGS['caching']['memory_cache_size']
    
    def get_cache_key(self, data_type, identifier, params=None):
        """Generate cache key from data type, identifier, and parameters."""
        key_data = f"{data_type}_{identifier}"
        if params:
            params_str = json.dumps(params, sort_keys=True)
            key_data += f"_{params_str}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def get_cached_indices(self, feature_type, cluster_config):
        """Get cached FAISS indices if available."""
        if not PERFORMANCE_CONFIGS['caching']['enable_index_cache']:
            return None
            
        cache_key = self.get_cache_key('faiss_indices', feature_type, cluster_config)
        cache_path = os.path.join(self.cache_dir, f"indices_{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            try:
                # Check if cache is newer than source features
                feature_path = get_feature_file_path(feature_type)
                if (os.path.exists(feature_path) and 
                    os.path.getmtime(cache_path) > os.path.getmtime(feature_path)):
                    
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    self.cache_stats['hits'] += 1
                    logging.info(f"Loaded cached indices for {feature_type}")
                    return cached_data
                    
            except Exception as e:
                logging.warning(f"Failed to load cached indices: {e}")
        
        self.cache_stats['misses'] += 1
        return None
    
    def save_indices_to_cache(self, feature_type, indices_data, cluster_config):
        """Save FAISS indices to cache."""
        cache_key = self.get_cache_key('faiss_indices', feature_type, cluster_config)
        cache_path = os.path.join(self.cache_dir, f"indices_{cache_key}.pkl")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(indices_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Cached indices for {feature_type}")
            return True
        except Exception as e:
            logging.error(f"Failed to cache indices: {e}")
            return False
    
    def get_from_memory_cache(self, key):
        """Get item from memory cache."""
        if key in self.memory_cache:
            self.cache_stats['hits'] += 1
            return self.memory_cache[key]
        self.cache_stats['misses'] += 1
        return None
    
    def store_in_memory_cache(self, key, value):
        """Store item in memory cache with simple LRU."""
        if len(self.memory_cache) >= self.max_memory_cache_size:
            # Remove oldest item
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        self.memory_cache[key] = value
    
    def clear_old_cache(self, max_age_days=None):
        """Clear old cache files."""
        if max_age_days is None:
            max_age_days = PERFORMANCE_CONFIGS['caching']['cache_max_age_days']
        
        import time
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        cleared_count = 0
        
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                        cleared_count += 1
                    except Exception:
                        pass
        except FileNotFoundError:
            pass
        
        if cleared_count > 0:
            logging.info(f"Cleared {cleared_count} old cache files")

    @staticmethod
    def get_memory_usage():
        """Get current memory usage statistics."""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }

    @staticmethod
    def force_garbage_collection():
        """Force garbage collection and return memory freed."""
        import gc
        mem_before = FeatureCache.get_memory_usage()['rss_mb']
        gc.collect()
        mem_after = FeatureCache.get_memory_usage()['rss_mb']
        freed = mem_before - mem_after
        if freed > 1.0:
            logging.info(f"Garbage collection freed {freed:.1f} MB")
        return freed

# Global cache instance
global_cache = None

def get_global_cache():
    """Get or create global cache instance."""
    global global_cache
    if global_cache is None:
        global_cache = FeatureCache()
    return global_cache

# ========== CONSOLIDATED FUNCTIONS ==========
def resolve_image_path(image_path, search_paths=None, validate=True):
    """Unified image path resolution with validation."""
    if search_paths is None:
        search_paths = [PATH_TO_SSD]
    
    # If already absolute and exists
    if os.path.isabs(image_path) and os.path.exists(image_path):
        if validate and not is_valid_image_file(image_path):
            return None
        return image_path
    
    # Search in directories
    filename = os.path.basename(image_path)
    for search_dir in search_paths:
        if os.path.exists(search_dir):
            # Try direct path
            full_path = os.path.join(search_dir, image_path)
            if os.path.exists(full_path):
                if validate and not is_valid_image_file(full_path):
                    continue
                return full_path
            
            # Recursive search
            for root, dirs, files in os.walk(search_dir):
                if filename in files:
                    full_path = os.path.join(root, filename)
                    if validate and not is_valid_image_file(full_path):
                        continue
                    return full_path
    
    return None

def is_valid_image_file(image_path):
    """Check if file is a valid image."""
    if not os.path.isfile(image_path):
        return False
    _, ext = os.path.splitext(image_path.lower())
    return ext in ALLOWED_IMAGE_EXTENSIONS

def get_feature_file_path(feature_type):
    """Get file path for any feature type."""
    filename_map = {
        'convnext': 'convnext_features.pkl',
        'hsv': 'hsv_features.pkl',
        'combined': 'combined_features.pkl'
    }
    filename = filename_map.get(feature_type, f"{feature_type}_features.pkl")
    return os.path.join(PICKLE_PATH, filename)

def get_feature_config_unified(feature_type, config_category='general'):
    """Unified configuration getter."""
    if config_category == 'clustering':
        return CLUSTERING_CONFIGS['clustering_per_feature'].get(
            feature_type, 
            CLUSTERING_CONFIGS['clustering_per_feature']['convnext']
        )
    elif config_category == 'faiss':
        return FAISS_CONFIGS['index_types_per_feature'].get(
            feature_type,
            FAISS_CONFIGS['index_types_per_feature']['convnext']
        )
    else:
        return FEATURE_CONFIGS.get(feature_type, {})

# Legacy function aliases for compatibility
def find_image_in_database(image_path, search_paths=None):
    """Legacy alias - use resolve_image_path instead."""
    return resolve_image_path(image_path, search_paths, validate=True)

def get_cluster_config(feature_type):
    """Legacy alias - use get_feature_config_unified instead."""
    return get_feature_config_unified(feature_type, 'clustering')

def get_faiss_config(feature_type):
    """Legacy alias - use get_feature_config_unified instead."""
    return get_feature_config_unified(feature_type, 'faiss')

def get_feature_path(feature_name):
    """Legacy alias - use get_feature_file_path instead."""
    return get_feature_file_path(feature_name)

def validate_image_path(image_path):
    """Validate that an image path exists and is a supported image format."""
    if not image_path:
        return False, "No path provided"
    
    if not os.path.exists(image_path):
        return False, f"Path does not exist: {image_path}"
    
    if not os.path.isfile(image_path):
        return False, f"Path is not a file: {image_path}"
    
    # Check file extension
    _, ext = os.path.splitext(image_path.lower())
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return False, f"Unsupported image format: {ext}"
    return True, "Valid image path"

