#!/usr/bin/env python3
"""
Database structure checker and initializer for metadata.db
This script shows the intended structure and creates/fixes the database.
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import BASE_PATH, DB_PATH
import db_api

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class DatabaseStructureChecker:
    """Check and initialize database structure for image similarity search."""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.expected_tables = {
            'images': {
                'description': 'Main image metadata table - maps image paths to metadata',
                'columns': {
                    'image_id': 'TEXT PRIMARY KEY (filename only, e.g., "image001.jpg")',
                    'image_path': 'TEXT NOT NULL (full path, e.g., "E:\\data\\folder1\\image001.jpg")',
                    'file_size': 'INTEGER (bytes)',
                    'width': 'INTEGER (pixels)',
                    'height': 'INTEGER (pixels)',
                    'created_date': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'processed_date': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
                },
                'purpose': 'Quick lookup: image_id -> full path and basic metadata'
            },
            'feature_metadata': {
                'description': 'Feature availability tracker - NO actual features stored here',
                'columns': {
                    'image_id': 'TEXT PRIMARY KEY (links to images.image_id)',
                    'has_hsv': 'BOOLEAN DEFAULT 0 (1 if HSV features extracted)',
                    'has_convnext': 'BOOLEAN DEFAULT 0 (1 if ConvNeXt features extracted)', 
                    'processing_date': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
                },
                'purpose': 'Track which images have complete features for search'
            }
        }
        
    def check_database_exists(self):
        """Check if database file exists."""
        logging.info(f"=== DATABASE EXISTENCE CHECK ===")
        
        if os.path.exists(self.db_path):
            size_mb = os.path.getsize(self.db_path) / 1024 / 1024
            logging.info(f"✓ Database exists: {self.db_path}")
            logging.info(f"  Size: {size_mb:.2f} MB")
            return True
        else:
            logging.warning(f"✗ Database missing: {self.db_path}")
            logging.info(f"  Directory exists: {os.path.exists(os.path.dirname(self.db_path))}")
            return False
            
    def analyze_current_structure(self):
        """Analyze current database structure."""
        logging.info(f"=== CURRENT DATABASE STRUCTURE ===")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            if not tables:
                logging.warning("No tables found in database")
                return False
                
            for (table_name,) in tables:
                logging.info(f"\nTable: {table_name}")
                
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                for column in columns:
                    col_id, name, type_, not_null, default, pk = column
                    pk_str = " PRIMARY KEY" if pk else ""
                    not_null_str = " NOT NULL" if not_null else ""
                    default_str = f" DEFAULT {default}" if default else ""
                    logging.info(f"  {name}: {type_}{pk_str}{not_null_str}{default_str}")
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                logging.info(f"  Rows: {count:,}")
                
            conn.close()
            return True
            
        except sqlite3.Error as e:
            logging.error(f"Database analysis failed: {e}")
            return False
            
    def show_expected_structure(self):
        """Show the expected database structure."""
        logging.info(f"=== EXPECTED DATABASE STRUCTURE ===")
        
        for table_name, table_info in self.expected_tables.items():
            logging.info(f"\nTable: {table_name}")
            logging.info(f"Purpose: {table_info['purpose']}")
            logging.info(f"Description: {table_info['description']}")
            logging.info("Columns:")
            
            for col_name, col_desc in table_info['columns'].items():
                logging.info(f"  {col_name}: {col_desc}")
                
        logging.info(f"\n=== KEY DESIGN PRINCIPLES ===")
        logging.info("1. NO FEATURE BLOBS: Actual features stored in pickle files")
        logging.info("2. FAST LOOKUP: image_id (filename) -> full path mapping")
        logging.info("3. FEATURE TRACKING: Boolean flags for feature availability")
        logging.info("4. METADATA ONLY: File size, dimensions, timestamps")
        logging.info("5. FOREIGN KEYS: feature_metadata links to images table")
        
        logging.info(f"\n=== USAGE PATTERN ===")
        logging.info("1. Learning mode: Populates both tables with metadata")
        logging.info("2. Search mode: Uses images table for path lookup")
        logging.info("3. Feature files: Separate pickle files with actual features")
        logging.info("4. Clustering: Uses feature_metadata to find complete images")
        
    def create_or_fix_database(self):
        """Create or fix the database structure."""
        logging.info(f"=== CREATING/FIXING DATABASE ===")
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Connect and create tables
            conn = create_connection()
            if conn is None:
                logging.error("Could not create database connection")
                return False
                
            # Create tables using db_api
            create_tables(conn)
            logging.info("✓ Tables created successfully")
            
            # Apply optimizations
            optimize_database_connection(conn)
            logging.info("✓ Database optimizations applied")
            
            # Verify creation
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for expected_table in self.expected_tables.keys():
                if expected_table in tables:
                    logging.info(f"✓ Table '{expected_table}' created")
                else:
                    logging.error(f"✗ Table '{expected_table}' missing")
                    
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"Database creation failed: {e}")
            return False
            
    def test_database_operations(self):
        """Test basic database operations."""
        logging.info(f"=== TESTING DATABASE OPERATIONS ===")
        
        try:
            conn = create_connection()
            if conn is None:
                return False
                
            cursor = conn.cursor()
            
            # Test insert into images table
            test_image_id = "test_image.jpg"
            test_path = "E:\\data\\test_folder\\test_image.jpg"
            
            cursor.execute('''
                INSERT OR REPLACE INTO images 
                (image_id, image_path, file_size, width, height)
                VALUES (?, ?, ?, ?, ?)
            ''', (test_image_id, test_path, 1024, 800, 600))
            
            logging.info("✓ Image insert test passed")
            
            # Test insert into feature_metadata table
            cursor.execute('''
                INSERT OR REPLACE INTO feature_metadata 
                (image_id, has_hsv, has_convnext)
                VALUES (?, ?, ?)
            ''', (test_image_id, 1, 1))
            
            logging.info("✓ Feature metadata insert test passed")
            
            # Test join query (typical usage pattern)
            cursor.execute('''
                SELECT i.image_id, i.image_path, f.has_hsv, f.has_convnext
                FROM images i
                JOIN feature_metadata f ON i.image_id = f.image_id
                WHERE f.has_hsv = 1 AND f.has_convnext = 1
            ''')
            
            results = cursor.fetchall()
            logging.info(f"✓ Join query test passed: {len(results)} complete images")
            
            # Clean up test data
            cursor.execute("DELETE FROM images WHERE image_id = ?", (test_image_id,))
            cursor.execute("DELETE FROM feature_metadata WHERE image_id = ?", (test_image_id,))
            
            conn.commit()
            conn.close()
            
            logging.info("✓ All database operations working correctly")
            return True
            
        except Exception as e:
            logging.error(f"Database operation test failed: {e}")
            return False
            
    def generate_usage_examples(self):
        """Generate SQL usage examples."""
        logging.info(f"=== DATABASE USAGE EXAMPLES ===")
        
        examples = {
            "Insert new image": '''
                INSERT OR REPLACE INTO images 
                (image_id, image_path, file_size, width, height)
                VALUES ('image001.jpg', 'E:\\data\\folder1\\image001.jpg', 2048, 1920, 1080);
            ''',
            
            "Mark features as extracted": '''
                INSERT OR REPLACE INTO feature_metadata 
                (image_id, has_hsv, has_convnext)
                VALUES ('image001.jpg', 1, 1);
            ''',
            
            "Find images with complete features": '''
                SELECT i.image_id, i.image_path 
                FROM images i 
                JOIN feature_metadata f ON i.image_id = f.image_id
                WHERE f.has_hsv = 1 AND f.has_convnext = 1;
            ''',
            
            "Get image path from filename": '''
                SELECT image_path FROM images WHERE image_id = 'image001.jpg';
            ''',
            
            "Check processing completeness": '''
                SELECT 
                    COUNT(*) as total_images,
                    SUM(has_hsv) as has_hsv,
                    SUM(has_convnext) as has_convnext,
                    SUM(has_hsv AND has_convnext) as complete_features
                FROM feature_metadata;
            ''',
            
            "Find unprocessed images": '''
                SELECT i.image_id, i.image_path
                FROM images i
                LEFT JOIN feature_metadata f ON i.image_id = f.image_id
                WHERE f.image_id IS NULL OR (f.has_hsv = 0 OR f.has_convnext = 0);
            '''
        }
        
        for description, sql in examples.items():
            logging.info(f"\n{description}:")
            for line in sql.strip().split('\n'):
                logging.info(f"  {line.strip()}")
                
    def run_full_check(self):
        """Run complete database structure check and fix."""
        logging.info("Starting database structure analysis...")
        
        # Step 1: Check existence
        exists = self.check_database_exists()
        
        # Step 2: Show expected structure
        self.show_expected_structure()
        
        # Step 3: Analyze current structure (if exists)
        if exists:
            self.analyze_current_structure()
        
        # Step 4: Create or fix database
        success = self.create_or_fix_database()
        
        if success:
            # Step 5: Test operations
            self.test_database_operations()
            
            # Step 6: Show usage examples
            self.generate_usage_examples()
            
            logging.info("\n" + "="*60)
            logging.info("DATABASE STRUCTURE CHECK COMPLETED SUCCESSFULLY")
            logging.info("="*60)
            logging.info("Your database is now ready for feature extraction!")
            logging.info("Next step: python main.py --mode learning")
            
        else:
            logging.error("\n" + "="*60)
            logging.error("DATABASE STRUCTURE CHECK FAILED")
            logging.error("="*60)
            logging.error("Please check the errors above and fix before proceeding")
            
        return success

if __name__ == "__main__":
    checker = DatabaseStructureChecker()
    success = checker.run_full_check()
    sys.exit(0 if success else 1)