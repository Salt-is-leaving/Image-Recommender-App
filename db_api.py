import sqlite3
import pickle
import re
import os
import numpy as np
from config import BASE_PATH, PICKLE_PATH, PATH_TO_SSD

def get_feature_file_path(feature_type):
    """Get file path for any feature type."""
    filename_map = {
        'convnext': 'convnext_features.pkl',
        'hsv': 'hsv_features.pkl',
        'combined': 'combined_features.pkl'
    }
    filename = filename_map.get(feature_type, f"{feature_type}_features.pkl")
    return os.path.join(PICKLE_PATH, filename)



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
            image_path TEXT PRIMARY KEY,
            image_filename TEXT NOT NULL,
            folder_name TEXT NOT NULL,
            semantic_category TEXT,
            extracted_caption TEXT,
            file_size INTEGER,
            width INTEGER,
            height INTEGER,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Feature metadata table - tracks which features exist (NO BLOBs)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_metadata (
            image_path TEXT PRIMARY KEY,
            has_hsv BOOLEAN DEFAULT 0,
            has_convnext BOOLEAN DEFAULT 0,
            processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_path) REFERENCES images (image_path)
        )
    ''')
    # Semantic folders table for folder-level insights
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS semantic_folders (
            folder_name TEXT PRIMARY KEY,
            folder_path TEXT NOT NULL,
            semantic_category TEXT,
            semantic_score INTEGER DEFAULT 0,
            image_count INTEGER DEFAULT 0,
            has_metadata_files BOOLEAN DEFAULT 0,
            processing_priority INTEGER DEFAULT 1
        )
    ''')

    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_semantic ON images(semantic_category)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_folder ON images(folder_name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_features_complete ON feature_metadata(has_hsv, has_convnext)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_folders_priority ON semantic_folders(processing_priority, semantic_score)')

    conn.commit()


# ========== SEMANTIC FOLDER ASSESSMENT AND REGISTRATION ==========
def assess_folder_semantic_value(folder_name):
    """Assess semantic value of a folder name for intelligent processing."""
    folder_lower = folder_name.lower()
    
    # High semantic value keywords (your actual folders)
    high_value_keywords = [
        'beach', 'city', 'drinks', 'electronics', 'forest', 'insects', 
        'mountains', 'sky', 'landscape', 'nature', 'urban', 'food', 
        'animals', 'cars', 'buildings', 'water', 'sunset', 'portrait'
    ]
    
    # Low/No semantic value patterns
    low_value_patterns = [
        r'mixed', r'misc', r'other', r'temp', r'new', r'old',
        r'folder_?\d+', r'batch_?\d+', r'set_?\d+', r'data_?\d+'
    ]
    
    # Check high value
    for keyword in high_value_keywords:
        if keyword in folder_lower:
            return 5  # Highest priority for processing
            
    # Check low value patterns
    for pattern in low_value_patterns:
        if re.search(pattern, folder_lower):
            return 1  # Low priority
            
    return 3  # Medium priority

def determine_semantic_category(folder_name):
    """Map folder name to high-level semantic category."""
    folder_lower = folder_name.lower()
    
    category_mapping = {
        'nature': ['beach', 'forest', 'mountains', 'sky', 'landscape', 'water', 'sunset'],
        'urban': ['city', 'buildings', 'urban', 'street', 'architecture'],
        'objects': ['electronics', 'drinks', 'food', 'cars', 'technology'],
        'living': ['insects', 'animals', 'portrait', 'people'],
        'mixed': ['mixed', 'misc', 'other', 'temp']
    }
    
    for category, keywords in category_mapping.items():
        if any(keyword in folder_lower for keyword in keywords):
            return category
            
    return 'uncategorized'

def extract_caption_from_filename(filename):
    """Extract semantic information from filename."""
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Common patterns in image filenames
    caption_patterns = [
        r'([a-zA-Z_]+)_\d+',           # word_123 -> word
        r'(\w+)-(\w+)',                # word-word -> word word
        r'(\w+)_(\w+)_(\w+)',          # word_word_word -> word word word
    ]
    
    # Try to extract meaningful words
    words = []
    for pattern in caption_patterns:
        match = re.search(pattern, name_without_ext)
        if match:
            words.extend([group for group in match.groups() if group.isalpha() and len(group) > 2])
            
    # Clean and return
    if words:
        return ' '.join(words).replace('_', ' ').title()
    else:
        return None

def register_semantic_folder(conn, folder_path):
    """Register a folder with its semantic information."""
    folder_name = os.path.basename(folder_path)
    semantic_category = determine_semantic_category(folder_name)
    semantic_score = assess_folder_semantic_value(folder_name)
    
    # Count images in folder
    image_count = 0
    has_metadata_files = False
    
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        image_count = sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
        has_metadata_files = any(f.lower().startswith('metadata') for f in files)
    
    # Set processing priority (higher score = higher priority)
    processing_priority = semantic_score
    
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO semantic_folders 
            (folder_name, folder_path, semantic_category, semantic_score, 
             image_count, has_metadata_files, processing_priority)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (folder_name, folder_path, semantic_category, semantic_score, 
              image_count, has_metadata_files, processing_priority))
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Error registering semantic folder: {e}")
        return False

# ========== IMAGE AND FEATURE METADATA MANAGEMENT ==========
def insert_image_metadata(conn, image_path, file_size=None, width=None, height=None):
    """Insert image metadata with semantic insights into the IMAGES table."""
    try:
        # Extract components from path
        image_filename = os.path.basename(image_path)
        folder_path = os.path.dirname(image_path)
        folder_name = os.path.basename(folder_path)
        
        # Get semantic information
        semantic_category = determine_semantic_category(folder_name)
        extracted_caption = extract_caption_from_filename(image_filename)
        
        # Register the folder if not already registered
        register_semantic_folder(conn, folder_path)
        
        # Insert image with semantic context
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO images 
            (image_path, image_filename, folder_name, semantic_category, 
             extracted_caption, file_size, width, height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (image_path, image_filename, folder_name, semantic_category, 
              extracted_caption, file_size, width, height))
        conn.commit()
        return True
        
    except sqlite3.Error as e:
        print(f"Error inserting image metadata: {e}")
        return False


def update_feature_metadata(conn, image_path, has_hsv=None, has_convnext=None):
    """Update feature metadata using image_path as key"""
    try:
        # First, insert or get existing record
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO feature_metadata (image_path) VALUES (?)", (image_path,))
        
        # Build update query based on provided parameters
        updates = []
        params = []
        
        if has_hsv is not None:
            updates.append("has_hsv = ?")
            params.append(has_hsv)
        if has_convnext is not None:
            updates.append("has_convnext = ?")
            params.append(has_convnext)
        
        if updates:
            sql = f"UPDATE feature_metadata SET {', '.join(updates)} WHERE image_path = ?"
            params.append(image_path)
            cursor.execute(sql, params)
            conn.commit()
            
    except sqlite3.Error as e:
        print(f"Error updating feature metadata: {e}")

def get_image_metadata(conn, image_path=None):
    """Get image metadata."""
    try:
        cursor = conn.cursor()
        if image_path:
            cursor.execute("SELECT * FROM images WHERE image_path = ?", (image_path,))
            return cursor.fetchone()
        else:
            cursor.execute("SELECT * FROM images")
            return cursor.fetchall()
    except sqlite3.Error as e:
        print(e)
        return None

def get_image_paths_by_category(conn, semantic_category=None, folder_name=None):
    """Get image paths filtered by semantic category or folder."""
    try:
        cursor = conn.cursor()
        if semantic_category:
            cursor.execute("""
                SELECT image_path FROM images 
                WHERE semantic_category = ?
            """, (semantic_category,))
        elif folder_name:
            cursor.execute("""
                SELECT image_path FROM images 
                WHERE folder_name = ?
            """, (folder_name,))
        else:
            cursor.execute("SELECT image_path FROM images")
            
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"Error getting image paths: {e}")
        return []

def get_semantic_folders_by_priority(conn, min_priority=3):
    """Get folders ordered by processing priority (high semantic value first)."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT folder_name, folder_path, semantic_category, semantic_score, 
                   image_count, processing_priority
            FROM semantic_folders 
            WHERE processing_priority >= ?
            ORDER BY processing_priority DESC, semantic_score DESC, image_count DESC
        """, (min_priority,))
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Error getting semantic folders: {e}")
        return []

def get_feature_completeness(conn):
    """Get feature completeness statistics."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_images,
                SUM(has_hsv) as has_hsv,
                SUM(has_convnext) as has_convnext,
                SUM(has_hsv AND has_convnext) as complete_features
            FROM feature_metadata
        """)
        all_stats = cursor.fetchone()

        # Semantic category breakdown
        cursor.execute("""
            SELECT 
                i.semantic_category,
                COUNT(*) as total_images,
                SUM(fm.has_hsv) as has_hsv,
                SUM(fm.has_convnext) as has_convnext,
                SUM(fm.has_hsv AND fm.has_convnext) as complete_features
            FROM images i
            LEFT JOIN feature_metadata fm ON i.image_path = fm.image_path
            GROUP BY i.semantic_category
            ORDER BY complete_features DESC
        """)
        category_stats = cursor.fetchall()
        
        return {
            'all': all_stats,
            'by_category': category_stats
        }
        
    except sqlite3.Error as e:
        print(f"Error getting feature completeness: {e}")
        return None
    

def get_images_with_complete_features(conn, semantic_category=None):
    """Get images with complete features, optionally filtered by semantic category."""
    try:
        cursor = conn.cursor()
        if semantic_category:
            cursor.execute("""
                SELECT i.image_path, i.folder_name, i.extracted_caption
                FROM images i 
                JOIN feature_metadata fm ON i.image_path = fm.image_path
                WHERE fm.has_hsv = 1 AND fm.has_convnext = 1
                AND i.semantic_category = ?
                ORDER BY i.folder_name, i.image_filename
            """, (semantic_category,))
        else:
            cursor.execute("""
                SELECT i.image_path, i.folder_name, i.extracted_caption
                FROM images i 
                JOIN feature_metadata fm ON i.image_path = fm.image_path
                WHERE fm.has_hsv = 1 AND fm.has_convnext = 1
                ORDER BY i.semantic_category, i.folder_name, i.image_filename
            """)
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Error getting complete features: {e}")
        return []

def get_category_aware_similar_candidates(conn, target_image_path, same_category_only=False):
    """Get similarity search candidates with category awareness."""
    try:
        cursor = conn.cursor()
        
        # Get target image category
        cursor.execute("SELECT semantic_category FROM images WHERE image_path = ?", (target_image_path,))
        target_category_result = cursor.fetchone()
        
        if not target_category_result:
            return []
            
        target_category = target_category_result[0]
        
        if same_category_only:
            # Search only within same semantic category
            cursor.execute("""
                SELECT i.image_path, i.folder_name, i.extracted_caption
                FROM images i
                JOIN feature_metadata fm ON i.image_path = fm.image_path
                WHERE fm.has_hsv = 1 AND fm.has_convnext = 1
                AND i.semantic_category = ?
                AND i.image_path != ?
                ORDER BY i.folder_name
            """, (target_category, target_image_path))
        else:
            # Weighted search: prefer same category, but include others
            cursor.execute("""
                SELECT i.image_path, i.folder_name, i.extracted_caption,
                       CASE WHEN i.semantic_category = ? THEN 1 ELSE 0 END as same_category
                FROM images i
                JOIN feature_metadata fm ON i.image_path = fm.image_path
                WHERE fm.has_hsv = 1 AND fm.has_convnext = 1
                AND i.image_path != ?
                ORDER BY same_category DESC, i.semantic_category, i.folder_name
            """, (target_category, target_image_path))
            
        return cursor.fetchall()
        
    except sqlite3.Error as e:
        print(f"Error getting category-aware candidates: {e}")
        return []

# ========== FEATURE FILE MANAGEMENT (CONSISTENT NAMING) ==========
def get_feature_pickle_path(feature_type):
    """Get consistent pickle file path for feature type."""
    filename_map = {
        'convnext': 'convnext_features.pkl',
        'hsv': 'hsv_features.pkl', 
    }
    filename = filename_map.get(feature_type, f"{feature_type}_features.pkl")
    return os.path.join(PICKLE_PATH, filename)

def save_features_to_pickle(features_dict, feature_type):
    """Save features dictionary to pickle file path-based naming"""
    filepath = get_feature_file_path(feature_type)
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
    """Load features dictionary from pickle file."""
    filepath = get_feature_file_path(feature_type)
    try:
        if not os.path.exists(filepath):
            return {}
            
        with open(filepath, 'rb') as f:
            features = pickle.load(f)
        
        return features
        
    except Exception as e:
        print(f"Error loading {feature_type} features: {e}")
        return {}

# ========== FEATURE SIMILARITY COMPUTATION ==========
def get_weighted_similarity_with_categories(target_path, weights=None, same_category_boost=0.1):
    """Enhanced similarity search with semantic category boosting."""
    if weights is None:
        weights = {'convnext': 0.65, 'hsv': 0.35}
    
    # Get database connection for category info
    conn = create_connection()
    if not conn:
        return []
    
    # Get target image category
    cursor = conn.cursor()
    cursor.execute("SELECT semantic_category FROM images WHERE image_path = ?", (target_path,))
    target_category_result = cursor.fetchone()
    target_category = target_category_result[0] if target_category_result else None
    
    # Load feature types
    feature_types = ['convnext', 'hsv']
    similarities = {}
    
    for feature_type in feature_types:
        features = load_features_from_pickle(feature_type)
        if target_path in features:
            target_feat = features[target_path]
            
            # Compute similarities
            type_similarities = []
            image_paths = []
            
            for img_path, feat in features.items():
                if img_path != target_path and feat is not None:
                    # Normalize features
                    target_norm = target_feat / (np.linalg.norm(target_feat) + 1e-8)
                    feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
                    
                    # Calculate base similarity
                    if feature_type == 'hsv':
                        sim = np.sum(np.sqrt(target_norm * feat_norm))
                    else:
                        sim = np.dot(target_norm.flatten(), feat_norm.flatten())
                    
                    # Apply category boost
                    if target_category:
                        cursor.execute("SELECT semantic_category FROM images WHERE image_path = ?", (img_path,))
                        img_category_result = cursor.fetchone()
                        if img_category_result and img_category_result[0] == target_category:
                            sim += same_category_boost  # Boost same-category matches
                    
                    type_similarities.append(sim)
                    image_paths.append(img_path)
            
            similarities[feature_type] = {
                'similarities': np.array(type_similarities),
                'image_paths': image_paths
            }
    
    conn.close()
    
    # Combine weighted similarities
    if not similarities:
        return []
    
    # Get common image paths
    common_paths = set(similarities[list(similarities.keys())[0]]['image_paths'])
    for feat_type in similarities:
        common_paths = common_paths.intersection(set(similarities[feat_type]['image_paths']))
    
    combined_similarities = []
    for img_path in common_paths:
        weighted_sim = 0
        for feat_type, weight in weights.items():
            if feat_type in similarities:
                idx = similarities[feat_type]['image_paths'].index(img_path)
                weighted_sim += weight * similarities[feat_type]['similarities'][idx]
        
        combined_similarities.append((img_path, weighted_sim))
    
    # Sort by similarity (descending)
    combined_similarities.sort(key=lambda x: x[1], reverse=True)
    
    return combined_similarities

# ========== MIGRATION AND CLEANUP ==========
def migrate_to_semantic_schema(conn):
    """Migrate existing data to new semantic schema."""
    print("Migrating to semantic schema...")
    
    try:
        cursor = conn.cursor()
        
        # Check if old schema exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images'")
        if not cursor.fetchone():
            print("No existing schema to migrate")
            return True
            
        # Check if new columns exist
        cursor.execute("PRAGMA table_info(images)")
        columns = [row[1] for row in cursor.fetchall()]
        
        # Add new columns if they don't exist
        new_columns = [
            ('folder_name', 'TEXT'),
            ('semantic_category', 'TEXT'),
            ('extracted_caption', 'TEXT')
        ]
        
        for col_name, col_type in new_columns:
            if col_name not in columns:
                cursor.execute(f"ALTER TABLE images ADD COLUMN {col_name} {col_type}")
                print(f"Added column: {col_name}")
        
        # Create semantic_folders table if it doesn't exist
        create_tables(conn)
        
        # Update existing records with semantic information
        cursor.execute("SELECT image_path FROM images WHERE folder_name IS NULL")
        paths_to_update = [row[0] for row in cursor.fetchall()]
        
        for image_path in paths_to_update:
            if os.path.exists(image_path):
                folder_name = os.path.basename(os.path.dirname(image_path))
                semantic_category = determine_semantic_category(folder_name)
                extracted_caption = extract_caption_from_filename(os.path.basename(image_path))
                
                cursor.execute("""
                    UPDATE images 
                    SET folder_name = ?, semantic_category = ?, extracted_caption = ?
                    WHERE image_path = ?
                """, (folder_name, semantic_category, extracted_caption, image_path))
        
        conn.commit()
        print(f"Updated {len(paths_to_update)} existing records with semantic information")
        return True
        
    except sqlite3.Error as e:
        print(f"Migration failed: {e}")
        return False

def cleanup_old_tables(conn):
    """Remove old tables that are no longer needed."""
    tables_to_remove = ['rgb_histograms', 'hsv_histograms']
    
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
        # Migrate existing schema
        migrate_to_semantic_schema(conn)
        
        # Clean up old tables
        cleanup_old_tables(conn)
        
        # Show enhanced feature completeness
        stats = get_feature_completeness(conn)
        if stats and stats['overall']:
            overall = stats['overall']
            print(f"\nOverall Feature Statistics:")
            print(f"  Total images: {overall[0]:,}")
            print(f"  HSV features: {overall[1]:,}")
            print(f"  ConvNeXt features: {overall[2]:,}")
            print(f"  Complete features: {overall[3]:,}")
            
            if stats['by_category']:
                print(f"\nBy Semantic Category:")
                for category_stat in stats['by_category']:
                    category, total, hsv, convnext, complete = category_stat
                    print(f"  {category:<12}: {complete:>6,}/{total:>6,} complete")
        
        # Show semantic folders by priority
        print(f"\nHigh-Priority Semantic Folders:")
        folders = get_semantic_folders_by_priority(conn, min_priority=4)
        for folder in folders[:10]:
            name, path, category, score, count, priority = folder
            print(f"  {name:<15} | {category:<10} | {count:>6,} images | Priority: {priority}")
        
        conn.close()