import os

# Base paths
BASE_PATH = r"D:\Code_image_rec"
PATH_TO_SSD = r"D:\images"

# Database path (metadata.db is in the main directory)
DB_PATH = os.path.join(BASE_PATH, 'metadata.db')

# Data paths
COLLECTED_DATA_PATH = os.path.join(BASE_PATH, 'collected_data')

# Pickle storage paths
PICKLE_PATH = os.path.join(BASE_PATH, 'pickles')
CHECKPOINT_PATH = os.path.join(PICKLE_PATH, 'processing_checkpoint.pkl')

# Enhanced clustering and FAISS index paths
ENHANCED_CLUSTER_PATH = os.path.join(PICKLE_PATH, 'enhanced_clusters')
FAISS_CLUSTER_INDEX_PATH = os.path.join(PICKLE_PATH, 'faiss_cluster_indices')

# Feature embeddings files (consistent naming)
FEATURE_FILES = {
    'efficientnet': 'efficientnet_features.pkl',
    'hsv': 'hsv_features.pkl',
    'combined': 'combined_features.pkl'
}

# Default paths
FINAL_EMBEDDINGS_PATH = os.path.join(PICKLE_PATH, FEATURE_FILES['efficientnet'])

# Annoy index files for fast similarity search

# Ensure directories exist
os.makedirs(PICKLE_PATH, exist_ok=True)

os.makedirs(COLLECTED_DATA_PATH, exist_ok=True)
os.makedirs(ENHANCED_CLUSTER_PATH, exist_ok=True)
os.makedirs(FAISS_CLUSTER_INDEX_PATH, exist_ok=True)

# Processing parameters
COMPRESS_QUALITY = 75
CHUNK_SIZE = 100
TOTAL_IMAGES = 5117

# Image processing parameters
MAX_IMAGE_SIZE = (512, 512)
TARGET_IMAGE_SIZE = (224, 224)
ALLOWED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.svg')

# Checkpoint parameters
CHECKPOINT_INTERVAL = 100

# Feature extraction configurations
FEATURE_CONFIGS = {
    'hsv': {
        'bins': [18, 16, 16],
        'ranges': [0, 180, 0, 256, 0, 256],
        'enabled': True,
        'preprocessing': {
        'gaussian_blur': True,  # Reduce noise
        'blur_kernel': (3, 3),
        'histogram_equalization': False,  # Usually not needed for HSV
        'mask_low_saturation': True,  # Ignore grayscale regions
        'saturation_threshold': 30,
        'adaptive_binning': True,    # NEW
        'edge_preservation': True    # NEW
    },
    'similarity_metric': 'bhattacharyya',
    'normalization': 'l1_epsilon',
    'dimension_validation': True 
    },

    'efficientnet': {
        'model_name': 'efficientnet-b7',
        'input_size': (224, 224),
        'embedding_size': 2560,
        'batch_size': 16,
        'enabled': True
    }
}

# Memory management parameters
MAX_FEATURES_IN_MEMORY = 10000
FEATURE_BATCH_SIZE = 1000

# Performance optimization parameters
USE_CUDA = True
NUM_WORKERS = 4
PREFETCH_FACTOR = 2

# Similarity search parameters
SIMILARITY_CONFIGS = {
    'weights': {
        'efficientnet': 0.6,
        'hsv': 0.4,
    },

    'distance_metrics': ['cosine', 'euclidean', 'manhattan']
}

# Clustering parameters - updated for multiple features
CLUSTERING_CONFIGS = {
    'kmeans': {
        'n_clusters': 15,
        'random_state': 42,
        'n_init': 10
    },
    'hdbscan': {
        'min_cluster_size': 8,
        'metric': 'euclidean',
        'cluster_selection_method': 'eom'
    },
    'agglomerative': {
        'n_clusters': 15,
        'linkage': 'ward'
    }
}

# UMAP parameters - optimized for multiple feature types
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

# Enhanced clustering configurations for clustering-first search
ENHANCED_CLUSTERING_CONFIGS = {
    'clustering_per_feature': {
        'efficientnet': {
            'base_clusters': 25,
            'pca_dimensions': 128,
            'normalization': 'l2',
            'use_minibatch': True,
            'silhouette_optimization': True
        },
    'hsv': {
            'base_clusters': 30,
            'pca_dimensions': 48,  # Dramatically reduce over-dimensioned HSV
            'normalization': 'l1',
            'use_minibatch': False,
            'silhouette_optimization': True,
            'similarity_metric': 'bhattacharyya' 
        }
    }, 
    'preprocessing_pipeline': {
        'standardization': True,
        'pca_enabled': True,
        'pca_variance_threshold': 0.95,
        'normalization_enabled': True,
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

# Enhanced FAISS configurations for clustering-first search
ENHANCED_FAISS_CONFIGS = {
    'index_types_per_feature': {
        'efficientnet': {
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
        'clusters_per_query': 3,
        'candidates_per_cluster': 50,
        'min_cluster_size_for_faiss': 5,
        'use_gpu_when_available': False
    },
    'performance_optimization': {
        'batch_search_size': 100,
        'memory_map_indices': False,
        'precompute_cluster_centers': True
    }
}

# Enhanced similarity fusion weights
ENHANCED_SIMILARITY_WEIGHTS = {
    'clustering_optimized': {
        'efficientnet': 0.6,
        'hsv': 0.4,
    },
    'color_focused': {
        'efficientnet': 0.3,
        'hsv': 0.7,
    },
    'semantic_focused': {
        'efficientnet': 0.8,
        'hsv': 0.2,
    },
    'balanced': {
        'efficientnet': 0.5,
        'hsv': 0.5,
    }
}

PERFORMANCE_CONFIGS = {
    'feature_extraction': {
        'batch_size_efficientnet': 32,
        'use_fp16': True,
        'prefetch_factor': 4,
        'num_workers': 6,
    },
    'search_optimization': {
        'max_clusters_search': 5,
        'candidates_per_cluster': 100,
        'parallel_search': True
    }
}

# Database optimization settings
DB_OPTIMIZATION = {
    'journal_mode': 'WAL',
    'synchronous': 'NORMAL',
    'cache_size': 10000,
    'temp_store': 'MEMORY'
}

def get_feature_path(feature_name):
    """Get the path for feature file based on feature name."""
    if feature_name in FEATURE_FILES:
        filename = FEATURE_FILES[feature_name]
    else:
        filename = f"{feature_name}_features.pkl"
    
    return os.path.join(PICKLE_PATH, filename)

def get_feature_config(feature_name):
    """Get feature configuration."""
    return FEATURE_CONFIGS.get(feature_name, {})

def get_enabled_features():
    """Get list of enabled features."""
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

# Enhanced clustering utility functions
def get_enhanced_cluster_config(feature_type):
    """Get enhanced clustering configuration for specific feature type."""
    return ENHANCED_CLUSTERING_CONFIGS['clustering_per_feature'].get(
        feature_type, 
        ENHANCED_CLUSTERING_CONFIGS['clustering_per_feature']['efficientnet']
    )

def get_enhanced_faiss_config(feature_type):
    """Get enhanced FAISS configuration for specific feature type."""
    return ENHANCED_FAISS_CONFIGS['index_types_per_feature'].get(
        feature_type,
        ENHANCED_FAISS_CONFIGS['index_types_per_feature']['efficientnet']
    )

def get_enhanced_similarity_weights(weight_type='clustering_optimized'):
    """Get enhanced similarity weights optimized for clustering-first search."""
    return ENHANCED_SIMILARITY_WEIGHTS.get(
        weight_type, 
        ENHANCED_SIMILARITY_WEIGHTS['clustering_optimized']
    )

def get_clustering_file_path(file_type='enhanced_cluster_data'):
    """Get path for clustering-related files."""
    file_map = {
        'enhanced_cluster_data': 'enhanced_cluster_data.pkl',
        'cluster_statistics': 'cluster_statistics.json',
        'cluster_performance': 'cluster_performance_log.txt'
    }
    
    filename = file_map.get(file_type, f"{file_type}.pkl")
    return os.path.join(PICKLE_PATH, filename)

def get_faiss_cluster_index_path(feature_type, cluster_id):
    """Get path for FAISS cluster index files."""
    filename = f"{feature_type}_cluster_{cluster_id}.faiss"
    return os.path.join(FAISS_CLUSTER_INDEX_PATH, filename)

def get_enhanced_clustering_status():
    """Get comprehensive status of enhanced clustering system."""
    status = {
        'clustering_available': False,
        'faiss_available': False,
        'cluster_data_size_mb': 0,
        'feature_statistics': {},
        'performance_ready': False
    }
    
    # Check clustering data
    cluster_file = get_clustering_file_path('enhanced_cluster_data')
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
                expected_features = ['efficientnet', 'hsv']
                available_features = list(cluster_data['cluster_assignments'].keys())
                status['performance_ready'] = len(available_features) >= 3
                
        except Exception:
            pass
    
    # Check FAISS availability
    try:
        import faiss
        status['faiss_available'] = True
    except ImportError:
        status['faiss_available'] = False
    
    return status
