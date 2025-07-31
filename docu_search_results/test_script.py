#!/usr/bin/env python3
"""
Comprehensive test script to verify:
1. Feature extraction works correctly
2. Features are stored in pickle files
3. Database metadata is stored correctly  
4. Path handling works with 216 folders
"""

import os
import sys
import pickle
import sqlite3
import numpy as np
import logging
from pathlib import Path
import random

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (PATH_TO_SSD, PICKLE_PATH, DB_PATH, FEATURE_CONFIGS, 
                   get_total_images, get_processed_image_count)
from db_api import (create_connection, get_feature_completeness, 
                   load_features_from_pickle, get_image_metadata)
from feature_extraction_pipeline import MultiFeatureExtractor, get_image_paths
from convnext_extractor import ConvNeXtFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class FeatureTestSuite:
    """Comprehensive test suite for feature extraction and storage."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.test_results = {}
        
    def log_error(self, test_name, error):
        """Log an error for a specific test."""
        self.errors.append(f"{test_name}: {error}")
        logging.error(f"{test_name}: {error}")
        
    def log_warning(self, test_name, warning):
        """Log a warning for a specific test."""
        self.warnings.append(f"{test_name}: {warning}")
        logging.warning(f"{test_name}: {warning}")
        
    def test_path_structure(self):
        """Test 1: Verify path structure and folder access."""
        logging.info("=== TEST 1: PATH STRUCTURE ===")
        
        # Check if E:\data exists
        if not os.path.exists(PATH_TO_SSD):
            self.log_error("PATH_STRUCTURE", f"Main data directory missing: {PATH_TO_SSD}")
            return False
            
        # Count total folders
        folder_count = 0
        total_images = 0
        folder_sizes = []
        
        try:
            for root, dirs, files in os.walk(PATH_TO_SSD):
                if root != PATH_TO_SSD:  # Don't count root directory
                    folder_count += 1
                    
                # Count images in this folder
                images_in_folder = sum(1 for f in files 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png')))
                if images_in_folder > 0:
                    folder_sizes.append(images_in_folder)
                    total_images += images_in_folder
                    
        except Exception as e:
            self.log_error("PATH_STRUCTURE", f"Error walking directory: {e}")
            return False
            
        logging.info(f"Found {folder_count} folders")
        logging.info(f"Total images: {total_images}")
        logging.info(f"Average images per folder: {np.mean(folder_sizes):.1f}")
        logging.info(f"Min/Max images per folder: {min(folder_sizes)}/{max(folder_sizes)}")
        
        # Test random folder access
        try:
            image_paths = get_image_paths(PATH_TO_SSD)
            if len(image_paths) != total_images:
                self.log_warning("PATH_STRUCTURE", 
                                f"get_image_paths() found {len(image_paths)} but walk found {total_images}")
            else:
                logging.info("✓ get_image_paths() correctly found all images")
                
        except Exception as e:
            self.log_error("PATH_STRUCTURE", f"get_image_paths() failed: {e}")
            return False
            
        # Test a few random paths
        if image_paths:
            test_paths = random.sample(image_paths, min(5, len(image_paths)))
            for path in test_paths:
                if not os.path.exists(path):
                    self.log_error("PATH_STRUCTURE", f"Random path check failed: {path}")
                    return False
                    
        logging.info("✓ Path structure test passed")
        self.test_results["path_structure"] = {
            "folders": folder_count,
            "total_images": total_images,
            "avg_per_folder": np.mean(folder_sizes)
        }
        return True
        
    def test_feature_extraction(self):
        """Test 2: Feature extraction functionality."""
        logging.info("=== TEST 2: FEATURE EXTRACTION ===")
        
        # Get a few test images
        try:
            image_paths = get_image_paths(PATH_TO_SSD)
            if not image_paths:
                self.log_error("FEATURE_EXTRACTION", "No images found for testing")
                return False
                
            test_images = random.sample(image_paths, min(3, len(image_paths)))
            
        except Exception as e:
            self.log_error("FEATURE_EXTRACTION", f"Failed to get test images: {e}")
            return False
            
        # Test ConvNeXt extractor
        try:
            extractor = ConvNeXtFeatureExtractor(use_cuda=False, use_fp16=False)
            logging.info("✓ ConvNeXt extractor initialized")
            
        except Exception as e:
            self.log_error("FEATURE_EXTRACTION", f"ConvNeXt initialization failed: {e}")
            return False
            
        # Test feature extraction on sample images
        convnext_results = []
        hsv_results = []
        
        multi_extractor = MultiFeatureExtractor(use_cuda=False)
        
        for i, image_path in enumerate(test_images):
            try:
                logging.info(f"Testing image {i+1}: {os.path.basename(image_path)}")
                
                # Extract features
                features = multi_extractor.extract_all_features(
                    image_path, os.path.basename(image_path)
                )
                
                if features is None:
                    self.log_error("FEATURE_EXTRACTION", 
                                  f"Feature extraction failed for {os.path.basename(image_path)}")
                    continue
                    
                # Validate ConvNeXt features
                if 'convnext' in features and features['convnext'] is not None:
                    conv_feat = features['convnext']
                    if len(conv_feat) != 1024:
                        self.log_error("FEATURE_EXTRACTION", 
                                      f"ConvNeXt wrong size: {len(conv_feat)} != 1024")
                    else:
                        convnext_results.append({
                            'norm': np.linalg.norm(conv_feat),
                            'std': np.std(conv_feat),
                            'mean': np.mean(conv_feat)
                        })
                        
                # Validate HSV features  
                if 'hsv' in features and features['hsv'] is not None:
                    hsv_feat = features['hsv']
                    expected_size = np.prod(FEATURE_CONFIGS['hsv']['bins'])  # 12*8*8 = 768
                    if len(hsv_feat) != expected_size:
                        self.log_error("FEATURE_EXTRACTION", 
                                      f"HSV wrong size: {len(hsv_feat)} != {expected_size}")
                    else:
                        hsv_results.append({
                            'sum': np.sum(hsv_feat),
                            'entropy': -np.sum(hsv_feat * np.log(hsv_feat + 1e-12)),
                            'sparsity': np.sum(hsv_feat > 1e-6) / len(hsv_feat)
                        })
                        
            except Exception as e:
                self.log_error("FEATURE_EXTRACTION", 
                              f"Exception processing {os.path.basename(image_path)}: {e}")
                
        # Analyze results
        if convnext_results:
            norms = [r['norm'] for r in convnext_results]
            stds = [r['std'] for r in convnext_results]
            logging.info(f"ConvNeXt norms: {np.min(norms):.3f} - {np.max(norms):.3f}")
            logging.info(f"ConvNeXt std variation: {np.std(norms):.6f}")
            
            if np.std(norms) < 0.001:
                self.log_warning("FEATURE_EXTRACTION", "ConvNeXt norms too similar - normalization issue")
            else:
                logging.info("✓ ConvNeXt features have good diversity")
                
        if hsv_results:
            sums = [r['sum'] for r in hsv_results]
            entropies = [r['entropy'] for r in hsv_results]
            logging.info(f"HSV histogram sums: {np.min(sums):.3f} - {np.max(sums):.3f}")
            logging.info(f"HSV entropies: {np.min(entropies):.2f} - {np.max(entropies):.2f}")
            
            if all(abs(s - 1.0) < 0.01 for s in sums):
                logging.info("✓ HSV histograms properly normalized")
            else:
                self.log_warning("FEATURE_EXTRACTION", "HSV histograms not normalized")
                
        self.test_results["feature_extraction"] = {
            "convnext_samples": len(convnext_results),
            "hsv_samples": len(hsv_results),
            "test_images": len(test_images)
        }
        
        logging.info("✓ Feature extraction test completed")
        return True
        
    def test_pickle_storage(self):
        """Test 3: Pickle file storage and loading."""
        logging.info("=== TEST 3: PICKLE STORAGE ===")
        
        # Check if pickle files exist
        feature_files = {
            'convnext': 'convnext_features.pkl',
            'hsv': 'hsv_features.pkl'
        }
        
        file_stats = {}
        
        for feature_type, filename in feature_files.items():
            filepath = os.path.join(PICKLE_PATH, filename)
            
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / 1024 / 1024
                logging.info(f"Found {filename}: {size_mb:.2f} MB")
                
                # Try to load and validate
                try:
                    features = load_features_from_pickle(feature_type)
                    file_stats[feature_type] = {
                        'size_mb': size_mb,
                        'count': len(features),
                        'sample_feature_size': len(list(features.values())[0]) if features else 0
                    }
                    
                    if not features:
                        self.log_warning("PICKLE_STORAGE", f"{filename} is empty")
                    else:
                        logging.info(f"✓ {filename}: {len(features)} features loaded")
                        
                        # Validate a few random features
                        sample_keys = random.sample(list(features.keys()), min(3, len(features)))
                        for key in sample_keys:
                            feat = features[key]
                            if feat is None:
                                self.log_warning("PICKLE_STORAGE", f"Null feature found: {key}")
                            elif not isinstance(feat, np.ndarray):
                                self.log_error("PICKLE_STORAGE", f"Invalid feature type: {type(feat)}")
                            else:
                                expected_size = (1024 if feature_type == 'convnext' 
                                               else np.prod(FEATURE_CONFIGS['hsv']['bins']))
                                if len(feat) != expected_size:
                                    self.log_error("PICKLE_STORAGE", 
                                                  f"Wrong feature size for {key}: {len(feat)} != {expected_size}")
                                    
                except Exception as e:
                    self.log_error("PICKLE_STORAGE", f"Failed to load {filename}: {e}")
                    
            else:
                logging.info(f"Missing: {filename}")
                file_stats[feature_type] = {'size_mb': 0, 'count': 0}
                
        self.test_results["pickle_storage"] = file_stats
        logging.info("✓ Pickle storage test completed")
        return True
        
    def test_database_storage(self):
        """Test 4: Database metadata storage."""
        logging.info("=== TEST 4: DATABASE STORAGE ===")
        
        # Check database connection
        try:
            conn = create_connection()
            if conn is None:
                self.log_error("DATABASE_STORAGE", "Cannot connect to database")
                return False
                
        except Exception as e:
            self.log_error("DATABASE_STORAGE", f"Database connection failed: {e}")
            return False
            
        # Check table structure
        try:
            cursor = conn.cursor()
            
            # Check images table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images'")
            if not cursor.fetchone():
                self.log_error("DATABASE_STORAGE", "Images table missing")
                return False
                
            # Check feature_metadata table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feature_metadata'")
            if not cursor.fetchone():
                self.log_error("DATABASE_STORAGE", "Feature_metadata table missing")
                return False
                
            logging.info("✓ Database tables exist")
            
        except Exception as e:
            self.log_error("DATABASE_STORAGE", f"Table check failed: {e}")
            return False
            
        # Check data integrity
        try:
            stats = get_feature_completeness(conn)
            if stats:
                total_images, has_hsv, has_convnext, complete = stats
                logging.info(f"Database stats: {total_images} total, {complete} complete")
                
                self.test_results["database_storage"] = {
                    'total_images': total_images,
                    'has_hsv': has_hsv,
                    'has_convnext': has_convnext,
                    'complete_features': complete
                }
                
                if complete == 0:
                    self.log_warning("DATABASE_STORAGE", "No images with complete features")
                else:
                    logging.info(f"✓ {complete} images have complete features")
                    
            else:
                self.log_warning("DATABASE_STORAGE", "Could not get feature completeness stats")
                
        except Exception as e:
            self.log_error("DATABASE_STORAGE", f"Data integrity check failed: {e}")
            
        # Test a few random image metadata entries
        try:
            cursor.execute("SELECT image_id, image_path FROM images LIMIT 5")
            sample_images = cursor.fetchall()
            
            for image_id, image_path in sample_images:
                # Check if file actually exists
                if not os.path.exists(image_path):
                    self.log_warning("DATABASE_STORAGE", f"Missing file: {image_path}")
                    
                # Check feature metadata
                cursor.execute(
                    "SELECT has_hsv, has_convnext FROM feature_metadata WHERE image_id = ?", 
                    (image_id,)
                )
                feat_meta = cursor.fetchone()
                if not feat_meta:
                    self.log_warning("DATABASE_STORAGE", f"Missing feature metadata: {image_id}")
                    
        except Exception as e:
            self.log_warning("DATABASE_STORAGE", f"Sample check failed: {e}")
            
        conn.close()
        logging.info("✓ Database storage test completed")
        return True
        
    def test_consistency(self):
        """Test 5: Consistency between pickle files and database."""
        logging.info("=== TEST 5: CONSISTENCY CHECK ===")
        
        # Load features from pickle files
        convnext_features = load_features_from_pickle('convnext')
        hsv_features = load_features_from_pickle('hsv')
        
        # Get database stats
        conn = create_connection()
        if conn:
            stats = get_feature_completeness(conn)
            if stats:
                db_total, db_hsv, db_convnext, db_complete = stats
                
                # Compare counts
                pickle_convnext = len(convnext_features)
                pickle_hsv = len(hsv_features)
                
                logging.info(f"Pickle vs DB - ConvNeXt: {pickle_convnext} vs {db_convnext}")
                logging.info(f"Pickle vs DB - HSV: {pickle_hsv} vs {db_hsv}")
                
                if pickle_convnext != db_convnext:
                    self.log_warning("CONSISTENCY", 
                                    f"ConvNeXt count mismatch: pickle={pickle_convnext}, db={db_convnext}")
                    
                if pickle_hsv != db_hsv:
                    self.log_warning("CONSISTENCY", 
                                    f"HSV count mismatch: pickle={pickle_hsv}, db={db_hsv}")
                    
                # Check common image IDs
                if convnext_features and hsv_features:
                    common_pickle = set(convnext_features.keys()).intersection(set(hsv_features.keys()))
                    logging.info(f"Common images in pickle files: {len(common_pickle)}")
                    
                    if len(common_pickle) != db_complete:
                        self.log_warning("CONSISTENCY", 
                                        f"Complete features mismatch: pickle={len(common_pickle)}, db={db_complete}")
                    else:
                        logging.info("✓ Pickle and database counts match")
                        
            conn.close()
            
        self.test_results["consistency"] = {
            "pickle_convnext": len(convnext_features),
            "pickle_hsv": len(hsv_features),
            "pickle_common": len(set(convnext_features.keys()).intersection(set(hsv_features.keys())))
        }
        
        logging.info("✓ Consistency check completed")
        return True
        
    def run_all_tests(self):
        """Run all tests and generate report."""
        logging.info("Starting comprehensive feature and database test suite...")
        
        tests = [
            self.test_path_structure,
            self.test_feature_extraction,
            self.test_pickle_storage,
            self.test_database_storage,
            self.test_consistency
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                self.log_error(test.__name__, f"Test crashed: {e}")
                
        # Generate report
        self.generate_report(passed, total)
        
    def generate_report(self, passed, total):
        """Generate final test report."""
        print(f"\n{'='*60}")
        print("FEATURE EXTRACTION AND DATABASE TEST REPORT")
        print(f"{'='*60}")
        
        print(f"\nTests Passed: {passed}/{total}")
        
        if self.test_results.get("path_structure"):
            ps = self.test_results["path_structure"]
            print(f"\nDataset Info:")
            print(f"  Folders: {ps['folders']}")
            print(f"  Total Images: {ps['total_images']:,}")
            print(f"  Avg per folder: {ps['avg_per_folder']:.1f}")
            
        if self.test_results.get("pickle_storage"):
            ps = self.test_results["pickle_storage"]
            print(f"\nFeature Files:")
            for feat_type, stats in ps.items():
                print(f"  {feat_type}: {stats['count']:,} features ({stats['size_mb']:.1f} MB)")
                
        if self.test_results.get("database_storage"):
            ds = self.test_results["database_storage"]
            print(f"\nDatabase:")
            print(f"  Total images: {ds['total_images']:,}")
            print(f"  Complete features: {ds['complete_features']:,}")
            
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
                
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
                
        if not self.errors:
            print(f"\n✅ ALL TESTS PASSED!")
            print("Your feature extraction and database setup is working correctly!")
        else:
            print(f"\n❌ {len(self.errors)} ERRORS FOUND")
            print("Please fix the errors before proceeding with full processing.")
            
        print(f"{'='*60}")

if __name__ == "__main__":
    tester = FeatureTestSuite()
    tester.run_all_tests()