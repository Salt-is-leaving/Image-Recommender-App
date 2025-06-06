import os

# Base paths
BASE_PATH = r"D:\Code_image_rec"
PATH_TO_SSD = r"D:\images"

# Database path (metadata.db is in the main directory)
DB_PATH = os.path.join(BASE_PATH, 'metadata.db')

# Data paths
COLLECTED_DATA_PATH = os.path.join(BASE_PATH, 'collected_data')

# No CSV files - all features stored in pickle files for efficiency

# Pickle storage paths
PICKLE_PATH = os.path.join(BASE_PATH, 'pickles')
CHECKPOINT_PATH = os.path.join(PICKLE_PATH, 'processing_checkpoint.pkl')

# Feature embeddings files (consistent naming)
FEATURE_FILES = {
    'efficientnet': 'efficientnet_features.pkl',
    'hsv': 'hsv_features.pkl',
    'lbp': 'lbp_features.pkl',
    'orb': 'orb_features.pkl',
    'combined': 'combined_features.pkl'
}

# Default paths
FINAL_EMBEDDINGS_PATH = os.path.join(PICKLE_PATH, FEATURE_FILES['efficientnet'])

# Annoy index files for fast similarity search
ANNOY_INDEX_PATH = os.path.join(PICKLE_PATH, 'annoy_indices')
ANNOY_FILES = {
    'efficientnet': 'efficientnet.ann',
    'hsv': 'hsv_features.ann',
    'lbp': 'lbp_features.ann',
    'orb': 'orb_features.ann',
    'combined': 'combined_features.ann'
}

# Ensure directories exist
os.makedirs(PICKLE_PATH, exist_ok=True)
os.makedirs(ANNOY_INDEX_PATH, exist_ok=True)
os.makedirs(COLLECTED_DATA_PATH, exist_ok=True)

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
        'bins': [32, 32, 32],  # reduced from [64, 64, 64]
        'ranges': [0, 256, 0, 256, 0, 256],
        'enabled': True
    },
    'lbp': {
        'radius': 3,
        'n_points': 24,
        'method': 'uniform',
        'enabled': True
    },
    'orb': {
        'n_features': 500,
        'scale_factor': 1.2,
        'n_levels': 8,
        'enabled': True
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
        'efficientnet': 0.5,
        'hsv': 0.2,
        'lbp': 0.2,
        'orb': 0.1
    },
    'annoy': {
        'n_trees': 100,
        'metric': 'angular',
        'search_k': -1
    },
    'distance_metrics': ['cosine', 'euclidean', 'manhattan']
}

# Clustering parameters - updated for multiple features
CLUSTERING_CONFIGS = {
    'kmeans': {
        'n_clusters': 15,  # More clusters for richer features
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

def get_annoy_index_path(index_name):
    """Get the path for Annoy index file."""
    if index_name in ANNOY_FILES:
        filename = ANNOY_FILES[index_name]
    else:
        filename = f"{index_name}.ann"
    
    return os.path.join(ANNOY_INDEX_PATH, filename)

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
