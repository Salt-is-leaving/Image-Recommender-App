#!/usr/bin/env python3
"""
Comprehensive test script for ConvNeXt feature extractor.
Tests functionality, performance, and identifies issues.

Usage:
    python test_convnext_extractor.py
    python test_convnext_extractor.py --quick-test
    python test_convnext_extractor.py --performance-test
"""

import os
import sys
import numpy as np
import torch
import timm
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time
import argparse
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from convnext_extractor import ConvNeXtFeatureExtractor
from config import PATH_TO_SSD, FEATURE_CONFIGS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConvNeXtTester:
    """Comprehensive tester for ConvNeXt feature extractor."""
    
    def __init__(self):
        self.test_results = {}
        self.extractor = None
        self.test_images = []
        
    def setup_test_images(self, num_test_images=10):
        """Create diverse test images for comprehensive testing."""
        logging.info("Setting up test images...")
        
        # Generate synthetic test images with different characteristics
        test_cases = [
            # Case 1: Pure colors
            ("solid_red", np.full((224, 224, 3), [255, 0, 0], dtype=np.uint8)),
            ("solid_blue", np.full((224, 224, 3), [0, 0, 255], dtype=np.uint8)),
            ("solid_green", np.full((224, 224, 3), [0, 255, 0], dtype=np.uint8)),
            ("solid_white", np.full((224, 224, 3), [255, 255, 255], dtype=np.uint8)),
            ("solid_black", np.full((224, 224, 3), [0, 0, 0], dtype=np.uint8)),
            
            # Case 2: Gradients
            ("gradient_h", self._create_horizontal_gradient()),
            ("gradient_v", self._create_vertical_gradient()),
            
            # Case 3: Patterns
            ("checkerboard", self._create_checkerboard()),
            ("stripes", self._create_stripes()),
            ("noise", self._create_noise_image()),
            
            # Case 4: Complex patterns
            ("circles", self._create_circles()),
            ("mixed_pattern", self._create_mixed_pattern()),
        ]
        
        for name, image in test_cases[:num_test_images]:
            self.test_images.append({
                'name': name,
                'image': image,
                'type': 'synthetic'
            })
        
        # Add real images if available
        self._add_real_images(max_real=3)
        
        logging.info(f"Created {len(self.test_images)} test images")
    
    def _create_horizontal_gradient(self):
        """Create horizontal gradient image."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        for x in range(224):
            intensity = int((x / 223) * 255)
            image[:, x, :] = intensity
        return image
    
    def _create_vertical_gradient(self):
        """Create vertical gradient image."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        for y in range(224):
            intensity = int((y / 223) * 255)
            image[y, :, :] = intensity
        return image
    
    def _create_checkerboard(self):
        """Create checkerboard pattern."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        square_size = 16
        for y in range(0, 224, square_size):
            for x in range(0, 224, square_size):
                if (x // square_size + y // square_size) % 2 == 0:
                    image[y:y+square_size, x:x+square_size] = 255
        return image
    
    def _create_stripes(self):
        """Create striped pattern."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        stripe_width = 8
        for x in range(0, 224, stripe_width * 2):
            image[:, x:x+stripe_width] = 255
        return image
    
    def _create_noise_image(self):
        """Create random noise image."""
        return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    def _create_circles(self):
        """Create circular patterns."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        centers = [(56, 56), (168, 56), (56, 168), (168, 168)]
        for center in centers:
            cv2.circle(image, center, 30, (255, 255, 255), -1)
        return image
    
    def _create_mixed_pattern(self):
        """Create complex mixed pattern."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        # Add gradient background
        for x in range(224):
            intensity = int((x / 223) * 128)
            image[:, x, 0] = intensity
        # Add some geometric shapes
        cv2.rectangle(image, (50, 50), (100, 100), (0, 255, 0), -1)
        cv2.circle(image, (150, 150), 40, (0, 0, 255), -1)
        return image
    
    def _add_real_images(self, max_real=3):
        """Add real images from dataset if available."""
        if not os.path.exists(PATH_TO_SSD):
            logging.warning(f"Image directory not found: {PATH_TO_SSD}")
            return
        
        real_count = 0
        for root, dirs, files in os.walk(PATH_TO_SSD):
            if real_count >= max_real:
                break
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')) and real_count < max_real:
                    try:
                        image_path = os.path.join(root, file)
                        image = np.array(Image.open(image_path).convert('RGB').resize((224, 224)))
                        self.test_images.append({
                            'name': f"real_{file[:20]}",
                            'image': image,
                            'type': 'real',
                            'path': image_path
                        })
                        real_count += 1
                    except Exception as e:
                        logging.warning(f"Could not load {file}: {e}")
        
        logging.info(f"Added {real_count} real images")
    
    def test_initialization(self):
        """Test ConvNeXt extractor initialization."""
        logging.info("Testing ConvNeXt initialization...")
        
        test_results = {
            'passed': False,
            'errors': [],
            'warnings': [],
            'model_info': {}
        }
        
        try:
            # Test CUDA availability
            cuda_available = torch.cuda.is_available()
            logging.info(f"CUDA available: {cuda_available}")
            
            # Test with CUDA
            if cuda_available:
                try:
                    self.extractor = ConvNeXtFeatureExtractor(use_cuda=True, use_fp16=False)
                    test_results['model_info']['cuda_init'] = 'success'
                except Exception as e:
                    test_results['errors'].append(f"CUDA initialization failed: {e}")
                    test_results['model_info']['cuda_init'] = 'failed'
            
            # Test CPU fallback
            if self.extractor is None:
                try:
                    self.extractor = ConvNeXtFeatureExtractor(use_cuda=False, use_fp16=False)
                    test_results['model_info']['cpu_init'] = 'success'
                except Exception as e:
                    test_results['errors'].append(f"CPU initialization failed: {e}")
                    return test_results
            
            # Get model information
            if self.extractor:
                info = self.extractor.get_model_info()
                test_results['model_info'].update(info)
                test_results['passed'] = True
                
                # Check for potential issues
                if info.get('fp16_enabled') == False and cuda_available:
                    test_results['warnings'].append("FP16 disabled despite CUDA availability")
                
                if info.get('feature_dim') != 1024:
                    test_results['errors'].append(f"Unexpected feature dimension: {info.get('feature_dim')}")
            
        except Exception as e:
            test_results['errors'].append(f"Critical initialization error: {e}")
        
        self.test_results['initialization'] = test_results
        return test_results
    
    def test_feature_extraction_quality(self):
        """Test feature extraction quality and consistency."""
        logging.info("Testing feature extraction quality...")
        
        if not self.extractor:
            return {'passed': False, 'errors': ['Extractor not initialized']}
        
        test_results = {
            'passed': False,
            'errors': [],
            'warnings': [],
            'feature_stats': {},
            'diversity_analysis': {}
        }
        
        try:
            all_features = []
            extraction_times = []
            
            for test_img in self.test_images:
                start_time = time.time()
                
                # Extract features
                features = self.extractor.extract_features(test_img['image'], normalize=True)
                
                extraction_time = time.time() - start_time
                extraction_times.append(extraction_time)
                
                if features is None:
                    test_results['errors'].append(f"Failed to extract features from {test_img['name']}")
                    continue
                
                # Validate features
                is_valid, message = self.extractor.validate_features(features)
                if not is_valid:
                    test_results['errors'].append(f"{test_img['name']}: {message}")
                    continue
                
                all_features.append({
                    'name': test_img['name'],
                    'features': features,
                    'norm': np.linalg.norm(features),
                    'std': np.std(features),
                    'mean': np.mean(features),
                    'extraction_time': extraction_time
                })
            
            if not all_features:
                test_results['errors'].append("No valid features extracted")
                return test_results
            
            # Analyze feature quality
            norms = [f['norm'] for f in all_features]
            stds = [f['std'] for f in all_features]
            means = [f['mean'] for f in all_features]
            times = [f['extraction_time'] for f in all_features]
            
            test_results['feature_stats'] = {
                'num_successful': len(all_features),
                'norm_range': [min(norms), max(norms)],
                'norm_std': np.std(norms),
                'norm_mean': np.mean(norms),
                'feature_std_mean': np.mean(stds),
                'feature_mean_range': [min(means), max(means)],
                'avg_extraction_time': np.mean(times),
                'extraction_time_std': np.std(times)
            }
            
            # Diversity analysis
            if len(all_features) >= 2:
                feature_matrix = np.array([f['features'] for f in all_features])
                
                # Compute pairwise similarities
                similarities = []
                for i in range(len(feature_matrix)):
                    for j in range(i + 1, len(feature_matrix)):
                        f1 = feature_matrix[i] / (np.linalg.norm(feature_matrix[i]) + 1e-8)
                        f2 = feature_matrix[j] / (np.linalg.norm(feature_matrix[j]) + 1e-8)
                        sim = np.dot(f1, f2)
                        similarities.append(sim)
                
                test_results['diversity_analysis'] = {
                    'similarity_mean': np.mean(similarities),
                    'similarity_std': np.std(similarities),
                    'similarity_range': [min(similarities), max(similarities)],
                    'max_similarity': max(similarities),
                    'min_similarity': min(similarities)
                }
            
            # Quality checks
            stats = test_results['feature_stats']
            
            # Check 1: Norm variation (should have some variation)
            if stats['norm_std'] < 0.001:
                test_results['errors'].append("Features have identical norms (over-normalized)")
            elif stats['norm_std'] < 0.01:
                test_results['warnings'].append("Very low norm variation")
            
            # Check 2: Feature diversity
            if stats['feature_std_mean'] < 0.001:
                test_results['errors'].append("Features lack diversity")
            elif stats['feature_std_mean'] < 0.01:
                test_results['warnings'].append("Low feature diversity")
            
            # Check 3: Reasonable magnitude
            if stats['norm_mean'] < 0.1 or stats['norm_mean'] > 100:
                test_results['warnings'].append(f"Unusual norm magnitude: {stats['norm_mean']:.3f}")
            
            # Check 4: Similarity distribution
            if 'diversity_analysis' in test_results:
                div_stats = test_results['diversity_analysis']
                if div_stats['similarity_std'] < 0.01:
                    test_results['warnings'].append("Low similarity variation (poor discriminative power)")
                if div_stats['max_similarity'] > 0.99:
                    test_results['warnings'].append("Some features are nearly identical")
            
            # Overall assessment
            test_results['passed'] = len(test_results['errors']) == 0
            
        except Exception as e:
            test_results['errors'].append(f"Feature extraction test failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        self.test_results['feature_quality'] = test_results
        return test_results
    
    def test_preprocessing_consistency(self):
        """Test preprocessing consistency."""
        logging.info("Testing preprocessing consistency...")
        
        if not self.extractor:
            return {'passed': False, 'errors': ['Extractor not initialized']}
        
        test_results = {
            'passed': False,
            'errors': [],
            'warnings': [],
            'consistency_scores': {}
        }
        
        try:
            # Use first test image for preprocessing tests
            base_image = self.test_images[0]['image']
            
            # Test 1: Same image should give identical features
            features1 = self.extractor.extract_features(base_image.copy(), normalize=True)
            features2 = self.extractor.extract_features(base_image.copy(), normalize=True)
            
            if features1 is not None and features2 is not None:
                consistency_score = np.corrcoef(features1, features2)[0, 1]
                test_results['consistency_scores']['identical_images'] = consistency_score
                
                if consistency_score < 0.9999:
                    test_results['errors'].append(f"Identical images give different features: {consistency_score:.6f}")
            else:
                test_results['errors'].append("Failed to extract features for consistency test")
            
            # Test 2: Different image sizes (resize consistency)
            sizes = [(224, 224), (256, 256), (512, 512)]
            size_features = {}
            
            for size in sizes:
                resized_image = cv2.resize(base_image, size)
                features = self.extractor.extract_features(resized_image, normalize=True)
                if features is not None:
                    size_features[size] = features
            
            # Compare features from different input sizes
            if len(size_features) >= 2:
                size_pairs = list(size_features.keys())
                for i in range(len(size_pairs)):
                    for j in range(i + 1, len(size_pairs)):
                        size1, size2 = size_pairs[i], size_pairs[j]
                        feat1, feat2 = size_features[size1], size_features[size2]
                        consistency = np.corrcoef(feat1, feat2)[0, 1]
                        test_results['consistency_scores'][f'{size1}_vs_{size2}'] = consistency
                        
                        if consistency < 0.95:
                            test_results['warnings'].append(
                                f"Low consistency between {size1} and {size2}: {consistency:.4f}"
                            )
            
            # Test 3: Center crop vs direct resize
            features_crop = self.extractor.extract_features(base_image, use_center_crop=True)
            features_direct = self.extractor.extract_features(base_image, use_center_crop=False)
            
            if features_crop is not None and features_direct is not None:
                crop_consistency = np.corrcoef(features_crop, features_direct)[0, 1]
                test_results['consistency_scores']['crop_vs_direct'] = crop_consistency
                
                if crop_consistency < 0.8:
                    test_results['warnings'].append(
                        f"Inconsistent preprocessing methods: {crop_consistency:.4f}"
                    )
            
            # Overall assessment
            test_results['passed'] = len(test_results['errors']) == 0
            
        except Exception as e:
            test_results['errors'].append(f"Preprocessing consistency test failed: {e}")
        
        self.test_results['preprocessing'] = test_results
        return test_results
    
    def test_performance(self):
        """Test extraction performance."""
        logging.info("Testing extraction performance...")
        
        if not self.extractor:
            return {'passed': False, 'errors': ['Extractor not initialized']}
        
        test_results = {
            'passed': False,
            'errors': [],
            'warnings': [],
            'performance_stats': {}
        }
        
        try:
            # Single image extraction timing
            test_image = self.test_images[0]['image']
            single_times = []
            
            for _ in range(10):
                start_time = time.time()
                features = self.extractor.extract_features(test_image, normalize=True)
                single_times.append(time.time() - start_time)
                
                if features is None:
                    test_results['errors'].append("Feature extraction failed during performance test")
                    break
            
            if single_times:
                test_results['performance_stats']['single_extraction'] = {
                    'mean_time': np.mean(single_times),
                    'std_time': np.std(single_times),
                    'min_time': min(single_times),
                    'max_time': max(single_times)
                }
            
            # Batch extraction timing (if available)
            if hasattr(self.extractor, 'extract_batch_features') and len(self.test_images) >= 5:
                batch_images = [img['image'] for img in self.test_images[:5]]
                
                start_time = time.time()
                batch_features = self.extractor.extract_batch_features(
                    batch_images, batch_size=5, normalize=True
                )
                batch_time = time.time() - start_time
                
                if batch_features and len(batch_features) == 5:
                    test_results['performance_stats']['batch_extraction'] = {
                        'total_time': batch_time,
                        'per_image_time': batch_time / 5,
                        'speedup_vs_single': np.mean(single_times) * 5 / batch_time if single_times else None
                    }
            
            # Performance assessment
            single_stats = test_results['performance_stats'].get('single_extraction', {})
            mean_time = single_stats.get('mean_time', float('inf'))
            
            if mean_time > 1.0:
                test_results['warnings'].append(f"Slow extraction: {mean_time:.3f}s per image")
            elif mean_time < 0.01:
                test_results['warnings'].append("Suspiciously fast extraction - check if working correctly")
            
            test_results['passed'] = len(test_results['errors']) == 0
            
        except Exception as e:
            test_results['errors'].append(f"Performance test failed: {e}")
        
        self.test_results['performance'] = test_results
        return test_results
    
    def run_all_tests(self, quick=False):
        """Run all tests."""
        logging.info("Starting comprehensive ConvNeXt extractor tests...")
        
        # Setup
        num_images = 5 if quick else 12
        self.setup_test_images(num_images)
        
        # Run tests
        self.test_initialization()
        self.test_feature_extraction_quality()
        self.test_preprocessing_consistency()
        
        if not quick:
            self.test_performance()
        
        # Generate report
        self.generate_report()
        
        return self.test_results
    
    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print("CONVNEXT EXTRACTOR TEST REPORT")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('passed', False))
        
        print(f"\nOVERALL: {passed_tests}/{total_tests} tests passed")
        
        # Initialization results
        if 'initialization' in self.test_results:
            init_result = self.test_results['initialization']
            print(f"\n1. INITIALIZATION: {'✓ PASSED' if init_result['passed'] else '✗ FAILED'}")
            
            if 'model_info' in init_result:
                info = init_result['model_info']
                print(f"   Model: {info.get('model_name', 'Unknown')}")
                print(f"   Device: {info.get('device', 'Unknown')}")
                print(f"   Feature dim: {info.get('feature_dim', 'Unknown')}")
                print(f"   FP16: {info.get('fp16_enabled', 'Unknown')}")
            
            for error in init_result.get('errors', []):
                print(f"   ❌ {error}")
            for warning in init_result.get('warnings', []):
                print(f"   ⚠️ {warning}")
        
        # Feature quality results
        if 'feature_quality' in self.test_results:
            quality_result = self.test_results['feature_quality']
            print(f"\n2. FEATURE QUALITY: {'✓ PASSED' if quality_result['passed'] else '✗ FAILED'}")
            
            if 'feature_stats' in quality_result:
                stats = quality_result['feature_stats']
                print(f"   Successful extractions: {stats.get('num_successful', 0)}")
                print(f"   Norm range: {stats.get('norm_range', [0, 0])[0]:.3f} - {stats.get('norm_range', [0, 0])[1]:.3f}")
                print(f"   Norm std: {stats.get('norm_std', 0):.6f}")
                print(f"   Feature diversity: {stats.get('feature_std_mean', 0):.6f}")
                print(f"   Avg extraction time: {stats.get('avg_extraction_time', 0):.3f}s")
            
            if 'diversity_analysis' in quality_result:
                div = quality_result['diversity_analysis']
                print(f"   Similarity range: {div.get('similarity_range', [0, 0])[0]:.3f} - {div.get('similarity_range', [0, 0])[1]:.3f}")
                print(f"   Similarity std: {div.get('similarity_std', 0):.6f}")
            
            for error in quality_result.get('errors', []):
                print(f"   ❌ {error}")
            for warning in quality_result.get('warnings', []):
                print(f"   ⚠️ {warning}")
        
        # Preprocessing consistency results
        if 'preprocessing' in self.test_results:
            prep_result = self.test_results['preprocessing']
            print(f"\n3. PREPROCESSING: {'✓ PASSED' if prep_result['passed'] else '✗ FAILED'}")
            
            if 'consistency_scores' in prep_result:
                scores = prep_result['consistency_scores']
                for test_name, score in scores.items():
                    print(f"   {test_name}: {score:.6f}")
            
            for error in prep_result.get('errors', []):
                print(f"   ❌ {error}")
            for warning in prep_result.get('warnings', []):
                print(f"   ⚠️ {warning}")
        
        # Performance results
        if 'performance' in self.test_results:
            perf_result = self.test_results['performance']
            print(f"\n4. PERFORMANCE: {'✓ PASSED' if perf_result['passed'] else '✗ FAILED'}")
            
            if 'performance_stats' in perf_result:
                stats = perf_result['performance_stats']
                if 'single_extraction' in stats:
                    single = stats['single_extraction']
                    print(f"   Single image: {single.get('mean_time', 0):.3f}s ± {single.get('std_time', 0):.3f}s")
                
                if 'batch_extraction' in stats:
                    batch = stats['batch_extraction']
                    print(f"   Batch extraction: {batch.get('per_image_time', 0):.3f}s per image")
                    if batch.get('speedup_vs_single'):
                        print(f"   Batch speedup: {batch['speedup_vs_single']:.2f}x")
            
            for error in perf_result.get('errors', []):
                print(f"   ❌ {error}")
            for warning in perf_result.get('warnings', []):
                print(f"   ⚠️ {warning}")
        
        # Overall assessment and recommendations
        print(f"\n" + "="*80)
        print("ASSESSMENT AND RECOMMENDATIONS")
        print("="*80)
        
        if passed_tests == total_tests:
            print("✅ ALL TESTS PASSED - ConvNeXt extractor is working correctly")
        else:
            print(f"⚠️ {total_tests - passed_tests} TEST(S) FAILED - Issues need attention")
        
        # Generate specific recommendations
        recommendations = self.generate_recommendations()
        if recommendations:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        
        print("\n" + "="*80)
    
    def generate_recommendations(self):
        """Generate specific recommendations based on test results."""
        recommendations = []
        
        # Check initialization issues
        if 'initialization' in self.test_results:
            init_result = self.test_results['initialization']
            if not init_result['passed']:
                recommendations.append("Fix initialization issues before proceeding")
            elif init_result.get('warnings'):
                if any('FP16 disabled' in w for w in init_result['warnings']):
                    recommendations.append("Enable FP16 for better performance: use_fp16=True")
        
        # Check feature quality issues
        if 'feature_quality' in self.test_results:
            quality_result = self.test_results['feature_quality']
            
            if 'feature_stats' in quality_result:
                stats = quality_result['feature_stats']
                norm_std = stats.get('norm_std', 0)
                
                if norm_std < 0.001:
                    recommendations.append("CRITICAL: Features have identical norms - fix normalization in extract_features()")
                elif norm_std < 0.01:
                    recommendations.append("Low norm variation - consider reviewing normalization approach")
                
                feature_diversity = stats.get('feature_std_mean', 0)
                if feature_diversity < 0.01:
                    recommendations.append("Low feature diversity - may affect similarity search quality")
            
            if any('over-normalized' in e for e in quality_result.get('errors', [])):
                recommendations.append("Remove complex normalization logic - use simple L2 norm only")
        
        # Check preprocessing issues
        if 'preprocessing' in self.test_results:
            prep_result = self.test_results['preprocessing']
            
            if 'consistency_scores' in prep_result:
                scores = prep_result['consistency_scores']
                for test_name, score in scores.items():
                    if 'crop_vs_direct' in test_name and score < 0.8:
                        recommendations.append("Inconsistent preprocessing - standardize on single transform")
        
        # Check performance issues
        if 'performance' in self.test_results:
            perf_result = self.test_results['performance']
            
            if 'performance_stats' in perf_result:
                stats = perf_result['performance_stats']
                if 'single_extraction' in stats:
                    mean_time = stats['single_extraction'].get('mean_time', 0)
                    if mean_time > 0.5:
                        recommendations.append("Slow extraction - consider enabling FP16 and model compilation")
        
        # General recommendations if no major issues
        if not recommendations:
            recommendations.extend([
                "ConvNeXt extractor appears to be working well",
                "Consider enabling model compilation for additional speedup",
                "Monitor feature quality during actual dataset processing"
            ])
        
        return recommendations
    
    def save_test_results(self, filepath=None):
        """Save detailed test results to file."""
        if filepath is None:
            filepath = "convnext_test_results.json"
        
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for test_name, result in self.test_results.items():
            serializable_results[test_name] = self._make_json_serializable(result)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            logging.info(f"Test results saved to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save test results: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


def run_comprehensive_test(args):
    """Run comprehensive ConvNeXt extractor test."""
    tester = ConvNeXtTester()
    
    # Run tests based on arguments
    if args.quick_test:
        results = tester.run_all_tests(quick=True)
    else:
        results = tester.run_all_tests(quick=False)
    
    # Save results if requested
    if args.save_results:
        tester.save_test_results(args.output_file)
    
    # Return success status
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result.get('passed', False))
    
    return passed_tests == total_tests


def run_feature_normalization_fix_test():
    """Test the specific normalization fix needed for ConvNeXt."""
    logging.info("Testing ConvNeXt normalization fix...")
    
    try:
        from convnext_extractor import ConvNeXtFeatureExtractor
        
        # Initialize extractor
        extractor = ConvNeXtFeatureExtractor(use_cuda=False, use_fp16=False)
        
        # Create test images with different characteristics
        test_images = []
        
        # Solid colors
        for color_val in [50, 100, 150, 200, 250]:
            img = np.full((224, 224, 3), color_val, dtype=np.uint8)
            test_images.append(f"solid_{color_val}")
            
        # Patterns
        noise_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        test_images.append(noise_img)
        
        # Extract features and analyze norms
        norms = []
        feature_stds = []
        
        for i, img in enumerate(test_images):
            if isinstance(img, str):
                continue  # Skip string labels
                
            features = extractor.extract_features(img, normalize=True)
            if features is not None:
                norm = np.linalg.norm(features)
                std = np.std(features)
                norms.append(norm)
                feature_stds.append(std)
                print(f"Image {i+1}: norm={norm:.6f}, std={std:.6f}")
        
        # Analyze results
        norm_variation = np.std(norms)
        avg_feature_std = np.mean(feature_stds)
        
        print(f"\nNormalization Analysis:")
        print(f"Norm variation: {norm_variation:.6f}")
        print(f"Average feature std: {avg_feature_std:.6f}")
        
        # Assessment
        if norm_variation < 0.001:
            print("❌ FAILED: All features have identical norms (over-normalized)")
            print("   Fix needed in extract_features() method")
            return False
        elif norm_variation < 0.01:
            print("⚠️ WARNING: Very low norm variation")
            return False
        else:
            print("✅ PASSED: Good norm variation")
            return True
    
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


def main():
    """Main test execution."""
    parser = argparse.ArgumentParser(description="Test ConvNeXt feature extractor")
    parser.add_argument('--quick-test', action='store_true', help="Run quick test with fewer images")
    parser.add_argument('--performance-test', action='store_true', help="Run performance-focused test")
    parser.add_argument('--normalization-fix-test', action='store_true', help="Test normalization fix specifically")
    parser.add_argument('--save-results', action='store_true', help="Save test results to file")
    parser.add_argument('--output-file', default='convnext_test_results.json', help="Output file for results")
    
    args = parser.parse_args()
    
    print("ConvNeXt Feature Extractor Test Suite")
    print("=" * 50)
    
    success = True
    
    if args.normalization_fix_test:
        success &= run_feature_normalization_fix_test()
    else:
        success &= run_comprehensive_test(args)
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("ConvNeXt extractor is ready for use.")
    else:
        print("\n⚠️ SOME TESTS FAILED!")
        print("Please address the issues before using in production.")
        
        # Show next steps
        print("\nNext steps:")
        print("1. Fix the identified issues in convnext_extractor.py")
        print("2. Re-run the tests to verify fixes")
        print("3. Update configuration if needed")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())