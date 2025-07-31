#!/usr/bin/env python3
"""
Feature Extraction Pipeline & Database Test Script

This script comprehensively tests:
1. Database functionality and schema
2. Feature extraction pipeline
3. Path handling and file operations
4. ConvNeXt and HSV feature quality
5. Pickle file operations
6. Error handling and edge cases

Usage:
    python test_feature_extraction.py
    python test_feature_extraction.py --create-test-images
    python test_feature_extraction.py --test-real-images --num-samples 10
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import pickle
import tempfile
import shutil
import logging
import argparse
import time
import sqlite3
import random
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    PATH_TO_SSD, PICKLE_PATH, CHECKPOINT_PATH, FEATURE_CONFIGS, 
    FEATURE_FILES, DB_PATH, validate_hsv_config, validate_convnext_config
)
from db_api import (
    create_connection, create_tables, insert_image_metadata, 
    update_feature_metadata, save_features_to_pickle, load_features_from_pickle,
    get_feature_completeness, get_feature_pickle_path
)
from feature_extraction_pipeline import (
    MultiFeatureExtractor, get_image_paths, process_images_batch,
    store_features_in_database, validate_extracted_features
)
from convnext_extractor import ConvNeXtFeatureExtractor, test_normalization_fix

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

class FeatureExtractionTester:
    """Comprehensive tester for feature extraction pipeline and database."""
    
    def __init__(self, use_temp_db=True, use_temp_pickle=True):
        self.use_temp_db = use_temp_db
        self.use_temp_pickle = use_temp_pickle
        self.test_dir = None
        self.original_db_path = None
        self.original_pickle_path = None
        self.test_results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
    def setup_test_environment(self):
        """Setup temporary test environment."""
        logging.info("Setting up test environment...")
        
        # Create temporary directory for test images
        self.test_dir = tempfile.mkdtemp(prefix='feature_test_')
        logging.info(f"Test directory: {self.test_dir}")
        
        # Setup temporary database if requested
        if self.use_temp_db:
            self.original_db_path = DB_PATH
            # Temporarily modify the DB_PATH in db_api module
            import db_api
            db_api.DB_PATH = os.path.join(self.test_dir, 'test_metadata.db')
            logging.info(f"Using temporary database: {db_api.DB_PATH}")
        
        # Setup temporary pickle directory if requested
        if self.use_temp_pickle:
            self.original_pickle_path = PICKLE_PATH
            # Create temporary pickle directory
            test_pickle_dir = os.path.join(self.test_dir, 'pickles')
            os.makedirs(test_pickle_dir, exist_ok=True)
            
            # Temporarily modify PICKLE_PATH in relevant modules
            import config
            config.PICKLE_PATH = test_pickle_dir
            logging.info(f"Using temporary pickle directory: {test_pickle_dir}")
        
        return True
    
    def cleanup_test_environment(self):
        """Cleanup test environment."""
        logging.info("Cleaning up test environment...")
        
        # Restore original paths
        if self.use_temp_db and self.original_db_path:
            import db_api
            db_api.DB_PATH = self.original_db_path
        
        if self.use_temp_pickle and self.original_pickle_path:
            import config
            config.PICKLE_PATH = self.original_pickle_path
        
        # Remove temporary directory
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            logging.info(f"Removed test directory: {self.test_dir}")
    
    def assert_test(self, condition, test_name, error_msg=""):
        """Helper method for test assertions."""
        self.test_results['total_tests'] += 1
        
        if condition:
            self.test_results['passed'] += 1
            logging.info(f"✅ PASS: {test_name}")
            return True
        else:
            self.test_results['failed'] += 1
            full_error = f"❌ FAIL: {test_name}"
            if error_msg:
                full_error += f" - {error_msg}"
            logging.error(full_error)
            self.test_results['errors'].append(full_error)
            return False
    
    def create_test_images(self, num_images=10):
        """Create diverse test images for feature extraction."""
        logging.info(f"Creating {num_images} test images...")
        
        test_images = []
        
        for i in range(num_images):
            # Create different types of test images
            if i % 4 == 0:
                # Solid color image
                color = [random.randint(0, 255) for _ in range(3)]
                image = np.full((224, 224, 3), color, dtype=np.uint8)
                filename = f"solid_color_{i}.jpg"
                
            elif i % 4 == 1:
                # Random noise image
                image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                filename = f"random_noise_{i}.jpg"
                
            elif i % 4 == 2:
                # Gradient image
                image = np.zeros((224, 224, 3), dtype=np.uint8)
                for x in range(224):
                    image[:, x, 0] = int((x / 224) * 255)  # Red gradient
                    image[:, x, 1] = int(((224 - x) / 224) * 255)  # Green gradient
                image[:, :, 2] = 128  # Constant blue
                filename = f"gradient_{i}.jpg"
                
            else:
                # Pattern image (checkerboard)
                image = np.zeros((224, 224, 3), dtype=np.uint8)
                square_size = 16
                for y in range(0, 224, square_size):
                    for x in range(0, 224, square_size):
                        if (x // square_size + y // square_size) % 2 == 0:
                            image[y:y+square_size, x:x+square_size] = [255, 255, 255]
                        else:
                            image[y:y+square_size, x:x+square_size] = [0, 0, 0]
                filename = f"checkerboard_{i}.jpg"
            
            # Save image
            image_path = os.path.join(self.test_dir, filename)
            pil_image = Image.fromarray(image)
            pil_image.save(image_path, 'JPEG', quality=95)
            test_images.append(image_path)
        
        logging.info(f"Created {len(test_images)} test images")
        return test_images
    
    def test_database_functionality(self):
        """Test database creation, connection, and operations."""
        logging.info("\n=== Testing Database Functionality ===")
        
        try:
            # Test database connection
            conn = create_connection()
            self.assert_test(conn is not None, "Database connection")
            
            if not conn:
                return False
            
            # Test table creation
            create_tables(conn)
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['images', 'feature_metadata', 'semantic_folders']
            for table in required_tables:
                self.assert_test(
                    table in tables, 
                    f"Table '{table}' exists",
                    f"Missing table: {table}"
                )
            
            # Test image metadata insertion
            test_image_path = "/test/path/test_image.jpg"
            success = insert_image_metadata(
                conn, test_image_path, 
                file_size=12345, width=224, height=224
            )
            self.assert_test(success, "Image metadata insertion")
            
            # Test feature metadata update
            update_feature_metadata(
                conn, test_image_path, 
                has_hsv=True, has_convnext=True
            )
            
            # Verify data was inserted
            cursor.execute("SELECT * FROM images WHERE image_path = ?", (test_image_path,))
            image_data = cursor.fetchone()
            self.assert_test(
                image_data is not None, 
                "Image data retrieval",
                "No image data found after insertion"
            )
            
            cursor.execute("SELECT * FROM feature_metadata WHERE image_path = ?", (test_image_path,))
            feature_data = cursor.fetchone()
            self.assert_test(
                feature_data is not None, 
                "Feature metadata retrieval",
                "No feature metadata found after insertion"
            )
            
            conn.close()
            return True
            
        except Exception as e:
            self.assert_test(False, "Database functionality", str(e))
            return False
    
    def test_config_validation(self):
        """Test configuration validation functions."""
        logging.info("\n=== Testing Configuration Validation ===")
        
        try:
            # Test HSV config validation
            hsv_issues = validate_hsv_config()
            self.assert_test(
                isinstance(hsv_issues, list), 
                "HSV config validation returns list"
            )
            
            # Test ConvNeXt config validation
            convnext_issues = validate_convnext_config()
            self.assert_test(
                isinstance(convnext_issues, list), 
                "ConvNeXt config validation returns list"
            )
            
            # Log any configuration issues
            if hsv_issues:
                logging.warning(f"HSV config issues: {hsv_issues}")
            if convnext_issues:
                logging.warning(f"ConvNeXt config issues: {convnext_issues}")
            
            return True
            
        except Exception as e:
            self.assert_test(False, "Config validation", str(e))
            return False
    
    def test_convnext_extractor(self):
        """Test ConvNeXt feature extractor."""
        logging.info("\n=== Testing ConvNeXt Feature Extractor ===")
        
        try:
            # Test ConvNeXt initialization
            extractor = ConvNeXtFeatureExtractor(use_cuda=False, use_fp16=False)
            self.assert_test(
                extractor.model is not None, 
                "ConvNeXt model initialization"
            )
            
            # Test feature extraction with synthetic image
            test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            features = extractor.extract_features(test_image, normalize=True)
            
            self.assert_test(
                features is not None, 
                "ConvNeXt feature extraction"
            )
            
            self.assert_test(
                isinstance(features, np.ndarray), 
                "ConvNeXt features are numpy array"
            )
            
            self.assert_test(
                features.shape[0] == 1024, 
                "ConvNeXt feature dimension",
                f"Expected 1024, got {features.shape[0]}"
            )
            
            self.assert_test(
                features.dtype == np.float32, 
                "ConvNeXt features are float32"
            )
            
            # Test feature normalization
            norm = np.linalg.norm(features)
            self.assert_test(
                0.5 < norm < 2.0, 
                "ConvNeXt feature normalization",
                f"Norm: {norm:.4f} (should be reasonable, not exactly 1.0)"
            )
            
            # Test feature diversity
            feature_std = np.std(features)
            self.assert_test(
                feature_std > 0.01, 
                "ConvNeXt feature diversity",
                f"Std: {feature_std:.6f} (should have variation)"
            )
            
            # Test validation function
            is_valid, msg = extractor.validate_features(features)
            self.assert_test(is_valid, "ConvNeXt feature validation", msg)
            
            return True
            
        except Exception as e:
            self.assert_test(False, "ConvNeXt extractor", str(e))
            return False
    
    def test_hsv_extraction(self):
        """Test HSV histogram extraction."""
        logging.info("\n=== Testing HSV Feature Extraction ===")
        
        try:
            extractor = MultiFeatureExtractor(use_cuda=False)
            
            # Test with different colored images
            test_cases = [
                ("red_image", [255, 0, 0]),
                ("green_image", [0, 255, 0]),
                ("blue_image", [0, 0, 255]),
                ("white_image", [255, 255, 255]),
                ("black_image", [0, 0, 0])
            ]
            
            for name, color in test_cases:
                # Create solid color image
                test_image = np.full((224, 224, 3), color, dtype=np.uint8)
                
                # Extract HSV histogram
                hsv_hist = extractor.extract_hsv_histogram(test_image)
                
                self.assert_test(
                    hsv_hist is not None, 
                    f"HSV extraction for {name}"
                )
                
                self.assert_test(
                    isinstance(hsv_hist, np.ndarray), 
                    f"HSV histogram is numpy array for {name}"
                )
                
                expected_size = np.prod(FEATURE_CONFIGS['hsv']['bins'])
                self.assert_test(
                    len(hsv_hist) == expected_size, 
                    f"HSV histogram size for {name}",
                    f"Expected {expected_size}, got {len(hsv_hist)}"
                )
                
                # Test histogram normalization
                hist_sum = np.sum(hsv_hist)
                self.assert_test(
                    abs(hist_sum - 1.0) < 1e-6, 
                    f"HSV histogram normalization for {name}",
                    f"Sum: {hist_sum} (should be ~1.0)"
                )
            
            return True
            
        except Exception as e:
            self.assert_test(False, "HSV extraction", str(e))
            return False
    
    def test_multifeature_extractor(self):
        """Test the complete multi-feature extraction pipeline."""
        logging.info("\n=== Testing Multi-Feature Extractor ===")
        
        try:
            # Create test images
            test_images = self.create_test_images(5)
            
            # Initialize extractor
            extractor = MultiFeatureExtractor(use_cuda=False)
            
            for image_path in test_images:
                # Extract all features
                features = extractor.extract_all_features(image_path)
                
                self.assert_test(
                    features is not None, 
                    f"Multi-feature extraction for {os.path.basename(image_path)}"
                )
                
                # Check required fields
                required_fields = ['image_path', 'hsv', 'convnext', 'file_size', 'width', 'height']
                for field in required_fields:
                    self.assert_test(
                        field in features, 
                        f"Feature field '{field}' present"
                    )
                
                # Validate feature quality
                validation = validate_extracted_features(features)
                self.assert_test(
                    validation['valid'], 
                    f"Feature validation for {os.path.basename(image_path)}",
                    f"Issues: {validation['issues']}"
                )
            
            return True
            
        except Exception as e:
            self.assert_test(False, "Multi-feature extractor", str(e))
            return False
    
    def test_pickle_operations(self):
        """Test pickle file save/load operations."""
        logging.info("\n=== Testing Pickle Operations ===")
        
        try:
            # Create test features
            test_features = {}
            for i in range(5):
                image_path = f"/test/image_{i}.jpg"
                if i % 2 == 0:
                    # ConvNeXt features
                    test_features[image_path] = np.random.randn(1024).astype(np.float32)
                else:
                    # HSV features
                    hsv_size = np.prod(FEATURE_CONFIGS['hsv']['bins'])
                    features = np.random.rand(hsv_size).astype(np.float32)
                    test_features[image_path] = features / np.sum(features)  # Normalize
            
            # Test ConvNeXt pickle operations
            convnext_features = {k: v for i, (k, v) in enumerate(test_features.items()) if i % 2 == 0}
            success = save_features_to_pickle(convnext_features, 'convnext')
            self.assert_test(success, "Save ConvNeXt features to pickle")
            
            loaded_convnext = load_features_from_pickle('convnext')
            self.assert_test(
                len(loaded_convnext) == len(convnext_features), 
                "Load ConvNeXt features from pickle",
                f"Expected {len(convnext_features)}, got {len(loaded_convnext)}"
            )
            
            # Test HSV pickle operations
            hsv_features = {k: v for i, (k, v) in enumerate(test_features.items()) if i % 2 == 1}
            success = save_features_to_pickle(hsv_features, 'hsv')
            self.assert_test(success, "Save HSV features to pickle")
            
            loaded_hsv = load_features_from_pickle('hsv')
            self.assert_test(
                len(loaded_hsv) == len(hsv_features), 
                "Load HSV features from pickle",
                f"Expected {len(hsv_features)}, got {len(loaded_hsv)}"
            )
            
            # Test feature file paths
            for feature_type in ['convnext', 'hsv']:
                pickle_path = get_feature_pickle_path(feature_type)
                self.assert_test(
                    os.path.exists(pickle_path), 
                    f"Feature pickle file exists for {feature_type}"
                )
            
            return True
            
        except Exception as e:
            self.assert_test(False, "Pickle operations", str(e))
            return False
    
    def test_batch_processing(self):
        """Test batch processing functionality."""
        logging.info("\n=== Testing Batch Processing ===")
        
        try:
            # Create test images
            test_images = self.create_test_images(8)
            
            # Initialize components
            extractor = MultiFeatureExtractor(use_cuda=False)
            conn = create_connection()
            
            if not conn:
                self.assert_test(False, "Database connection for batch processing")
                return False
            
            create_tables(conn)
            
            # Test batch processing
            checkpoint_path = os.path.join(self.test_dir, 'test_checkpoint.pkl')
            
            all_features = process_images_batch(
                extractor, test_images, conn, checkpoint_path, batch_size=3
            )
            
            self.assert_test(
                'convnext' in all_features, 
                "Batch processing returns ConvNeXt features"
            )
            
            self.assert_test(
                'hsv' in all_features, 
                "Batch processing returns HSV features"
            )
            
            # Check that features were extracted for all images
            total_extracted = len(all_features['convnext']) + len(all_features['hsv'])
            self.assert_test(
                total_extracted > 0, 
                "Batch processing extracted features",
                f"Total features extracted: {total_extracted}"
            )
            
            # Test database storage
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM images")
            image_count = cursor.fetchone()[0]
            self.assert_test(
                image_count > 0, 
                "Images stored in database during batch processing",
                f"Image count: {image_count}"
            )
            
            conn.close()
            return True
            
        except Exception as e:
            self.assert_test(False, "Batch processing", str(e))
            return False
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        logging.info("\n=== Testing Error Handling ===")
        
        try:
            extractor = MultiFeatureExtractor(use_cuda=False)
            
            # Test with non-existent file
            features = extractor.extract_all_features("/nonexistent/file.jpg")
            self.assert_test(
                features is None, 
                "Non-existent file handling"
            )
            
            # Test with corrupted image (create a text file with .jpg extension)
            fake_image_path = os.path.join(self.test_dir, "fake_image.jpg")
            with open(fake_image_path, 'w') as f:
                f.write("This is not an image file")
            
            features = extractor.extract_all_features(fake_image_path)
            self.assert_test(
                features is None, 
                "Corrupted image file handling"
            )
            
            # Test with empty image directory
            empty_dir = os.path.join(self.test_dir, "empty")
            os.makedirs(empty_dir, exist_ok=True)
            
            image_paths = get_image_paths(empty_dir)
            self.assert_test(
                len(image_paths) == 0, 
                "Empty directory handling"
            )
            
            # Test feature validation with invalid features
            invalid_features = {
                'image_path': '/test/image.jpg',
                'hsv': None,  # Invalid
                'convnext': np.array([1, 2, 3]),  # Wrong size
                'file_size': 1000,
                'width': 224,
                'height': 224
            }
            
            validation = validate_extracted_features(invalid_features)
            self.assert_test(
                not validation['valid'], 
                "Invalid feature validation",
                "Should detect invalid features"
            )
            
            return True
            
        except Exception as e:
            self.assert_test(False, "Error handling", str(e))
            return False
    
    def test_path_handling(self):
        """Test path handling and resolution."""
        logging.info("\n=== Testing Path Handling ===")
        
        try:
            # Test with various path formats
            test_paths = [
                os.path.join(self.test_dir, "test1.jpg"),
                os.path.join(self.test_dir, "subdir", "test2.jpg"),
                "relative_test.jpg"
            ]
            
            # Create test structure
            subdir = os.path.join(self.test_dir, "subdir")
            os.makedirs(subdir, exist_ok=True)
            
            # Create test images
            for path in test_paths[:2]:  # Skip the relative path
                test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                pil_image = Image.fromarray(test_image)
                pil_image.save(path, 'JPEG')
            
            # Test path discovery
            found_paths = get_image_paths(self.test_dir)
            self.assert_test(
                len(found_paths) >= 2, 
                "Path discovery finds images",
                f"Found {len(found_paths)} images"
            )
            
            # Test that paths are absolute
            for path in found_paths:
                self.assert_test(
                    os.path.isabs(path), 
                    f"Path is absolute: {os.path.basename(path)}"
                )
            
            return True
            
        except Exception as e:
            self.assert_test(False, "Path handling", str(e))
            return False
    
    def test_real_images(self, num_samples=5):
        """Test with real images from the dataset (if available)."""
        logging.info(f"\n=== Testing Real Images (max {num_samples}) ===")
        
        try:
            if not os.path.exists(PATH_TO_SSD):
                logging.warning(f"Real image directory not found: {PATH_TO_SSD}")
                self.assert_test(True, "Real images test skipped (no dataset)")
                return True
            
            # Get sample of real images
            real_images = get_image_paths(PATH_TO_SSD)[:num_samples]
            
            if not real_images:
                logging.warning("No real images found in dataset")
                self.assert_test(True, "Real images test skipped (no images)")
                return True
            
            logging.info(f"Testing with {len(real_images)} real images")
            
            extractor = MultiFeatureExtractor(use_cuda=False)
            successful_extractions = 0
            
            for image_path in real_images:
                try:
                    features = extractor.extract_all_features(image_path)
                    if features is not None:
                        successful_extractions += 1
                        
                        # Quick validation
                        validation = validate_extracted_features(features)
                        if not validation['valid']:
                            logging.warning(f"Invalid features for {os.path.basename(image_path)}: {validation['issues']}")
                        
                except Exception as e:
                    logging.warning(f"Failed to extract features from {os.path.basename(image_path)}: {e}")
            
            success_rate = successful_extractions / len(real_images)
            self.assert_test(
                success_rate >= 0.8, 
                f"Real image extraction success rate",
                f"Success rate: {success_rate:.2%} ({successful_extractions}/{len(real_images)})"
            )
            
            return True
            
        except Exception as e:
            self.assert_test(False, "Real images test", str(e))
            return False
    
    def run_all_tests(self, test_real_images=False, num_real_samples=5):
        """Run all tests in sequence."""
        logging.info("🚀 Starting comprehensive feature extraction tests...")
        
        start_time = time.time()
        
        try:
            # Setup test environment
            if not self.setup_test_environment():
                logging.error("Failed to setup test environment")
                return False
            
            # Run test suite
            test_methods = [
                self.test_config_validation,
                self.test_database_functionality,
                self.test_convnext_extractor,
                self.test_hsv_extraction,
                self.test_multifeature_extractor,
                self.test_pickle_operations,
                self.test_batch_processing,
                self.test_error_handling,
                self.test_path_handling,
            ]
            
            if test_real_images:
                test_methods.append(lambda: self.test_real_images(num_real_samples))
            
            for test_method in test_methods:
                try:
                    test_method()
                except Exception as e:
                    logging.error(f"Test method {test_method.__name__} failed: {e}")
                    self.test_results['failed'] += 1
                    self.test_results['errors'].append(f"Test method error: {e}")
            
        finally:
            # Always cleanup
            self.cleanup_test_environment()
        
        # Print results
        self.print_test_results(time.time() - start_time)
        
        return self.test_results['failed'] == 0
    
    def print_test_results(self, duration):
        """Print comprehensive test results."""
        print(f"\n{'='*60}")
        print("🧪 FEATURE EXTRACTION PIPELINE TEST RESULTS")
        print(f"{'='*60}")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"📊 Total Tests: {self.test_results['total_tests']}")
        print(f"✅ Passed: {self.test_results['passed']}")
        print(f"❌ Failed: {self.test_results['failed']}")
        
        if self.test_results['failed'] == 0:
            print(f"🎉 ALL TESTS PASSED!")
        else:
            print(f"\n❌ FAILED TESTS:")
            for error in self.test_results['errors']:
                print(f"   {error}")
        
        success_rate = (self.test_results['passed'] / self.test_results['total_tests']) * 100
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"{'='*60}")


def main():
    """Main function for running tests."""
    parser = argparse.ArgumentParser(description="Test feature extraction pipeline and database")
    parser.add_argument('--create-test-images', action='store_true',
                       help='Create and save test images for manual inspection')
    parser.add_argument('--test-real-images', action='store_true',
                       help='Test with real images from dataset')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of real images to test (default: 5)')
    parser.add_argument('--use-real-db', action='store_true',
                       help='Use real database instead of temporary one')
    parser.add_argument('--use-real-pickle', action='store_true',
                       help='Use real pickle directory instead of temporary one')
    parser.add_argument('--test-normalization', action='store_true',
                       help='Run ConvNeXt normalization fix test')
    
    args = parser.parse_args()
    
    # Run normalization test if requested
    if args.test_normalization:
        print("🔧 Running ConvNeXt normalization fix test...")
        success = test_normalization_fix()
        if success:
            print("✅ ConvNeXt normalization test passed!")
        else:
            print("❌ ConvNeXt normalization test failed!")
        return 0 if success else 1
    
    # Create test images for manual inspection if requested
    if args.create_test_images:
        print("🎨 Creating test images for manual inspection...")
        temp_dir = tempfile.mkdtemp(prefix='test_images_')
        
        # Import random here to avoid issues
        
        test_images = []
        for i in range(20):
            if i % 5 == 0:
                # Solid colors
                color = [random.randint(0, 255) for _ in range(3)]
                image = np.full((224, 224, 3), color, dtype=np.uint8)
                filename = f"solid_color_{i}_{color[0]}_{color[1]}_{color[2]}.jpg"
            elif i % 5 == 1:
                # Random noise
                image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                filename = f"random_noise_{i}.jpg"
            elif i % 5 == 2:
                # Gradients
                image = np.zeros((224, 224, 3), dtype=np.uint8)
                for x in range(224):
                    image[:, x, 0] = int((x / 224) * 255)
                    image[:, x, 1] = int(((224 - x) / 224) * 255)
                image[:, :, 2] = 128
                filename = f"gradient_{i}.jpg"
            elif i % 5 == 3:
                # Patterns
                image = np.zeros((224, 224, 3), dtype=np.uint8)
                size = 16
                for y in range(0, 224, size):
                    for x in range(0, 224, size):
                        if (x // size + y // size) % 2 == 0:
                            image[y:y+size, x:x+size] = [255, 255, 255]
                filename = f"checkerboard_{i}.jpg"
            else:
                # Complex patterns
                image = np.zeros((224, 224, 3), dtype=np.uint8)
                for y in range(224):
                    for x in range(224):
                        image[y, x, 0] = int(127 * (1 + np.sin(x * 0.1)))
                        image[y, x, 1] = int(127 * (1 + np.sin(y * 0.1)))
                        image[y, x, 2] = int(127 * (1 + np.sin((x + y) * 0.05)))
                filename = f"complex_pattern_{i}.jpg"
            
            image_path = os.path.join(temp_dir, filename)
            pil_image = Image.fromarray(image)
            pil_image.save(image_path, 'JPEG', quality=95)
            test_images.append(image_path)
        
        print(f"✅ Created {len(test_images)} test images in: {temp_dir}")
        print("📁 You can inspect these images manually to verify quality")
        return 0
    
    # Run main test suite
    tester = FeatureExtractionTester(
        use_temp_db=not args.use_real_db,
        use_temp_pickle=not args.use_real_pickle
    )
    
    success = tester.run_all_tests(
        test_real_images=args.test_real_images,
        num_real_samples=args.num_samples
    )
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("✅ Feature extraction pipeline is working correctly")
        print("✅ Database operations are functioning properly")
        print("✅ Path handling is robust")
        print("✅ Error handling is comprehensive")
        
        print("\n🚀 Ready for production use!")
        print("Next steps:")
        print("   1. Run: python main.py --mode learning")
        print("   2. Run: python main.py --mode clustering") 
        print("   3. Run: python main.py --mode interactive")
        
    else:
        print("\n❌ Some tests failed!")
        print("Please review the errors above and fix the issues")
        print("Check the critical bugs identified in the analysis")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())