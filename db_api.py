import sqlite3
import pickle
import os
import numpy as np
from config import BASE_PATH, PICKLE_PATH

# Correct database path
DB_PATH = os.path.join(BASE_PATH, 'metadata.db')

def create_connection(db_file=None):
    """Create a database connection to the SQLite database."""
    if db_file is None:
        db_file = DB_PATH
    
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
    return conn

def create_tables(conn):
    """Create optimized tables - ONLY metadata, NO feature BLOBs."""
    cursor = conn.cursor()

    # Image metadata table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT PRIMARY KEY,
            image_path TEXT NOT NULL,
            file_size INTEGER,
            width INTEGER,
            height INTEGER,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Feature metadata table - tracks which features exist (NO BLOBs)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_metadata (
            image_id TEXT PRIMARY KEY,
            has_hsv BOOLEAN DEFAULT 0,
            has_lbp BOOLEAN DEFAULT 0,
            has_orb BOOLEAN DEFAULT 0,
            has_efficientnet BOOLEAN DEFAULT 0,
            processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES images (image_id)
        )
    ''')

    conn.commit()

def insert_image_metadata(conn, image_id, image_path, file_size=None, width=None, height=None):
    """Insert image metadata into the images table."""
    try:
        sql = '''INSERT OR REPLACE INTO images (image_id, image_path, file_size, width, height)
                 VALUES (?, ?, ?, ?, ?)'''
        cursor = conn.cursor()
        cursor.execute(sql, (image_id, image_path, file_size, width, height))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error inserting image metadata: {e}")

def update_feature_metadata(conn, image_id, has_hsv=None, has_lbp=None, has_orb=None, has_efficientnet=None):
    """Update feature metadata for an image."""
    try:
        # First, insert or get existing record
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO feature_metadata (image_id) VALUES (?)", (image_id,))
        
        # Build update query based on provided parameters
        updates = []
        params = []
        
        if has_hsv is not None:
            updates.append("has_hsv = ?")
            params.append(has_hsv)
        if has_lbp is not None:
            updates.append("has_lbp = ?")
            params.append(has_lbp)
        if has_orb is not None:
            updates.append("has_orb = ?")
            params.append(has_orb)
        if has_efficientnet is not None:
            updates.append("has_efficientnet = ?")
            params.append(has_efficientnet)
        
        if updates:
            sql = f"UPDATE feature_metadata SET {', '.join(updates)} WHERE image_id = ?"
            params.append(image_id)
            cursor.execute(sql, params)
            conn.commit()
            
    except sqlite3.Error as e:
        print(f"Error updating feature metadata: {e}")

def get_image_metadata(conn, image_id=None):
    """Get image metadata."""
    try:
        cursor = conn.cursor()
        if image_id:
            cursor.execute("SELECT * FROM images WHERE image_id = ?", (image_id,))
            return cursor.fetchone()
        else:
            cursor.execute("SELECT * FROM images")
            return cursor.fetchall()
    except sqlite3.Error as e:
        print(e)
        return None

def get_image_ids(conn):
    """Retrieve all image_ids from the metadata database."""
    cursor = conn.cursor()
    cursor.execute("SELECT image_id FROM images")
    image_ids = cursor.fetchall()
    return [row[0] for row in image_ids]

def get_feature_completeness(conn):
    """Get feature completeness statistics."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_images,
                SUM(has_hsv) as has_hsv,
                SUM(has_lbp) as has_lbp,
                SUM(has_orb) as has_orb,
                SUM(has_efficientnet) as has_efficientnet,
                SUM(has_hsv AND has_lbp AND has_orb AND has_efficientnet) as complete_features
            FROM feature_metadata
        """)
        return cursor.fetchone()
    except sqlite3.Error as e:
        print(f"Error getting feature completeness: {e}")
        return None

def get_images_with_complete_features(conn):
    """Get images that have all features processed."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT i.image_id, i.image_path 
            FROM images i 
            JOIN feature_metadata f ON i.image_id = f.image_id
            WHERE f.has_hsv = 1 AND f.has_lbp = 1 AND f.has_orb = 1 AND f.has_efficientnet = 1
        """)
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Error getting complete features: {e}")
        return []

# ========== FEATURE FILE MANAGEMENT (CONSISTENT NAMING) ==========

def get_feature_pickle_path(feature_type):
    """Get consistent pickle file path for feature type."""
    filename_map = {
        'efficientnet': 'efficientnet_features.pkl',
        'hsv': 'hsv_features.pkl', 
        'lbp': 'lbp_features.pkl',
        'orb': 'orb_features.pkl'
    }
    filename = filename_map.get(feature_type, f"{feature_type}_features.pkl")
    return os.path.join(PICKLE_PATH, filename)

def save_features_to_pickle(features_dict, feature_type):
    """Save features dictionary to pickle file with consistent naming."""
    filepath = get_feature_pickle_path(feature_type)
    try:
        # Load existing features if file exists
        existing_features = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    existing_features = pickle.load(f)
            except:
                pass  # If loading fails, start fresh
        
        # Update with new features
        existing_features.update(features_dict)
        
        # Save updated features
        with open(filepath, 'wb') as f:
            pickle.dump(existing_features, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"{feature_type} features saved to {filepath} ({len(existing_features)} total features)")
        return True
    except Exception as e:
        print(f"Error saving {feature_type} features: {e}")
        return False

def load_features_from_pickle(feature_type):
    """Load features dictionary from pickle file with consistent naming."""
    filepath = get_feature_pickle_path(feature_type)
    try:
        if not os.path.exists(filepath):
            print(f"Feature file not found: {filepath}")
            return {}
            
        with open(filepath, 'rb') as f:
            features = pickle.load(f)
        print(f"{feature_type} features loaded from {filepath} ({len(features)} features)")
        return features
    except Exception as e:
        print(f"Error loading {feature_type} features: {e}")
        return {}

def combine_all_features(image_ids, normalize_features=True):
    """Combine all feature types into a single feature vector per image."""
    from sklearn.preprocessing import normalize
    
    # Load all feature types with consistent naming
    print("Loading features for combination...")
    efficientnet_features = load_features_from_pickle('efficientnet')
    hsv_features = load_features_from_pickle('hsv')
    lbp_features = load_features_from_pickle('lbp')
    orb_features = load_features_from_pickle('orb')
    
    combined_features = {}
    feature_stats = {'efficientnet': 0, 'hsv': 0, 'lbp': 0, 'orb': 0}
    
    for image_id in image_ids:
        feature_vector = []
        
        # EfficientNet features (high-dimensional)
        if image_id in efficientnet_features:
            eff_feat = efficientnet_features[image_id]
            if eff_feat is not None:
                feature_vector.extend(eff_feat.flatten())
                feature_stats['efficientnet'] += 1
        
        # HSV histogram
        if image_id in hsv_features:
            hsv_feat = hsv_features[image_id]
            if hsv_feat is not None:
                feature_vector.extend(hsv_feat.flatten())
                feature_stats['hsv'] += 1
        
        # LBP features
        if image_id in lbp_features:
            lbp_feat = lbp_features[image_id]
            if lbp_feat is not None:
                feature_vector.extend(lbp_feat.flatten())
                feature_stats['lbp'] += 1
        
        # ORB features (handle variable-length descriptors)
        if image_id in orb_features:
            orb_feat = orb_features[image_id]
            if orb_feat is not None:
                if len(orb_feat.shape) > 1:
                    # Multiple descriptors - use mean
                    feature_vector.extend(np.mean(orb_feat, axis=0))
                else:
                    # Single descriptor
                    feature_vector.extend(orb_feat)
                feature_stats['orb'] += 1
        
        # Only add if we have at least some features
        if len(feature_vector) > 0:
            combined_features[image_id] = np.array(feature_vector)
    
    print(f"Feature combination stats: {feature_stats}")
    
    # Normalize if requested
    if normalize_features and combined_features:
        feature_matrix = np.array(list(combined_features.values()))
        normalized_matrix = normalize(feature_matrix)
        
        combined_features = {
            image_id: normalized_matrix[i] 
            for i, image_id in enumerate(combined_features.keys())
        }
        print(f"Combined and normalized {len(combined_features)} feature vectors")
    
    return combined_features

def get_weighted_similarity(target_features, all_features, weights=None):
    """Compute weighted similarity using multiple feature types."""
    if weights is None:
        weights = {'efficientnet': 0.5, 'hsv': 0.2, 'lbp': 0.2, 'orb': 0.1}
    
    # Load individual feature types
    feature_types = ['efficientnet', 'hsv', 'lbp', 'orb']
    similarities = {}
    
    for feature_type in feature_types:
        features = load_features_from_pickle(feature_type)
        if target_features['image_id'] in features:
            target_feat = features[target_features['image_id']]
            
            # Compute similarities for this feature type
            type_similarities = []
            image_ids = []
            
            for img_id, feat in features.items():
                if img_id != target_features['image_id'] and feat is not None:
                    # Normalize features
                    target_norm = target_feat / (np.linalg.norm(target_feat) + 1e-8)
                    feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
                    
                    # Cosine similarity
                    sim = np.dot(target_norm.flatten(), feat_norm.flatten())
                    type_similarities.append(sim)
                    image_ids.append(img_id)
            
            similarities[feature_type] = {
                'similarities': np.array(type_similarities),
                'image_ids': image_ids
            }
    
    # Combine weighted similarities
    if not similarities:
        return []
    
    # Get common image IDs
    common_ids = set(similarities[list(similarities.keys())[0]]['image_ids'])
    for feat_type in similarities:
        common_ids = common_ids.intersection(set(similarities[feat_type]['image_ids']))
    
    combined_similarities = []
    for img_id in common_ids:
        weighted_sim = 0
        for feat_type, weight in weights.items():
            if feat_type in similarities:
                idx = similarities[feat_type]['image_ids'].index(img_id)
                weighted_sim += weight * similarities[feat_type]['similarities'][idx]
        
        combined_similarities.append((img_id, weighted_sim))
    
    # Sort by similarity (descending)
    combined_similarities.sort(key=lambda x: x[1], reverse=True)
    
    return combined_similarities

# ========== MIGRATION AND CLEANUP ==========

def cleanup_old_tables(conn):
    """Remove old tables that are no longer needed."""
    tables_to_remove = ['rgb_histograms', 'hsv_histograms', 'lbp_features', 'orb_descriptors']
    
    try:
        cursor = conn.cursor()
        for table in tables_to_remove:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
        print("Old feature BLOB tables removed successfully")
    except sqlite3.Error as e:
        print(f"Error removing old tables: {e}")

if __name__ == '__main__':
    conn = create_connection()
    if conn is not None:
        create_tables(conn)
        
        # Clean up old tables
        cleanup_old_tables(conn)
        
        # Show feature completeness
        stats = get_feature_completeness(conn)
        if stats:
            print(f"Feature Statistics:")
            print(f"  Total images: {stats[0]}")
            print(f"  HSV features: {stats[1]}")
            print(f"  LBP features: {stats[2]}")
            print(f"  ORB features: {stats[3]}")
            print(f"  EfficientNet features: {stats[4]}")
            print(f"  Complete features: {stats[5]}")
        
        # Test feature loading
        print("\nTesting feature file access:")
        for feature_type in ['efficientnet', 'hsv', 'lbp', 'orb']:
            features = load_features_from_pickle(feature_type)
            print(f"  {feature_type}: {len(features)} features available")
        
        conn.close()
