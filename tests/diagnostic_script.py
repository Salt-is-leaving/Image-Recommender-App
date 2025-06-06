#!/usr/bin/env python3
"""
Feature Diagnostic Script
Identifies problems in the current similarity search system
"""

import os
import numpy as np
import pickle
import logging
from config import PICKLE_PATH, FEATURE_FILES
from db_api import load_features_from_pickle

logging.basicConfig(level=logging.INFO)

def diagnose_features():
    """Comprehensive feature diagnosis."""
    print("=" * 60)
    print("🔍 FEATURE DIAGNOSTIC REPORT")
    print("=" * 60)
    
    # Check 1: Feature File Existence and Sizes
    print("\n📁 1. FEATURE FILE STATUS")
    print("-" * 40)
    
    for feature_type, filename in FEATURE_FILES.items():
        filepath = os.path.join(PICKLE_PATH, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"✓ {feature_type}: {filename} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {feature_type}: {filename} - NOT FOUND")
    
    # Check 2: Load and Analyze Features
    print("\n🔬 2. FEATURE CONTENT ANALYSIS")
    print("-" * 40)
    
    feature_stats = {}
    
    for feature_type in ['efficientnet', 'hsv', 'lbp', 'orb']:
        try:
            features = load_features_from_pickle(feature_type)
            if features:
                # Sample a few features for analysis
                sample_ids = list(features.keys())[:5]
                
                print(f"\n{feature_type.upper()} Features:")
                print(f"  Total images: {len(features)}")
                
                # Analyze feature dimensions and properties
                valid_features = []
                invalid_count = 0
                
                for img_id in sample_ids:
                    feat = features[img_id]
                    if feat is not None:
                        valid_features.append(feat)
                        print(f"  {img_id}: shape={feat.shape}, dtype={feat.dtype}, "
                              f"min={feat.min():.4f}, max={feat.max():.4f}, mean={feat.mean():.4f}")
                    else:
                        invalid_count += 1
                        print(f"  {img_id}: None (invalid)")
                
                if valid_features:
                    # Check consistency
                    shapes = [f.shape for f in valid_features]
                    unique_shapes = set(shapes)
                    
                    if len(unique_shapes) == 1:
                        print(f"  ✓ Consistent shape: {list(unique_shapes)[0]}")
                    else:
                        print(f"  ⚠ Inconsistent shapes: {unique_shapes}")
                    
                    # Check for proper normalization
                    means = [f.mean() for f in valid_features]
                    stds = [f.std() for f in valid_features]
                    
                    print(f"  Feature statistics:")
                    print(f"    Mean range: {min(means):.4f} to {max(means):.4f}")
                    print(f"    Std range: {min(stds):.4f} to {max(stds):.4f}")
                    
                    # Specific checks per feature type
                    if feature_type == 'hsv':
                        # HSV should be positive histogram
                        neg_count = sum(1 for f in valid_features if np.any(f < 0))
                        if neg_count > 0:
                            print(f"  ⚠ {neg_count} HSV histograms have negative values!")
                        
                        # Check if normalized (should sum to ~1)
                        sums = [f.sum() for f in valid_features]
                        if all(0.9 < s < 1.1 for s in sums):
                            print(f"  ✓ HSV histograms properly normalized")
                        else:
                            print(f"  ⚠ HSV histograms not normalized (sums: {min(sums):.3f} to {max(sums):.3f})")
                    
                    elif feature_type == 'lbp':
                        # LBP should be positive histogram
                        neg_count = sum(1 for f in valid_features if np.any(f < 0))
                        if neg_count > 0:
                            print(f"  ⚠ {neg_count} LBP histograms have negative values!")
                
                feature_stats[feature_type] = {
                    'total': len(features),
                    'valid': len(valid_features),
                    'invalid': invalid_count,
                    'sample_shape': valid_features[0].shape if valid_features else None
                }
                
            else:
                print(f"{feature_type.upper()}: No features loaded")
                feature_stats[feature_type] = {'total': 0, 'valid': 0, 'invalid': 0}
                
        except Exception as e:
            print(f"{feature_type.upper()}: Error loading - {e}")
            feature_stats[feature_type] = {'error': str(e)}
    
    # Check 3: Cross-Feature Consistency
    print("\n🔗 3. CROSS-FEATURE CONSISTENCY")
    print("-" * 40)
    
    all_image_sets = []
    for feature_type in feature_stats:
        if 'error' not in feature_stats[feature_type]:
            features = load_features_from_pickle(feature_type)
            if features:
                all_image_sets.append(set(features.keys()))
                print(f"{feature_type}: {len(features)} images")
    
    if len(all_image_sets) > 1:
        common_images = set.intersection(*all_image_sets)
        print(f"\n✓ Common images across all features: {len(common_images)}")
        
        if len(common_images) < 100:
            print("⚠ Very few common images - this will hurt similarity search!")
        
        # Check for feature alignment
        if len(common_images) > 0:
            sample_id = list(common_images)[0]
            print(f"\nSample image analysis ({sample_id}):")
            
            for feature_type in feature_stats:
                if 'error' not in feature_stats[feature_type]:
                    features = load_features_from_pickle(feature_type)
                    if sample_id in features:
                        feat = features[sample_id]
                        if feat is not None:
                            print(f"  {feature_type}: {feat.shape} - OK")
                        else:
                            print(f"  {feature_type}: None - MISSING")
                    else:
                        print(f"  {feature_type}: Not found - MISSING")
    else:
        print("⚠ Cannot check consistency - too few feature types loaded")
    
    # Check 4: Similarity Computation Test
    print("\n🧮 4. SIMILARITY COMPUTATION TEST")
    print("-" * 40)
    
    if len(all_image_sets) >= 2 and len(common_images) >= 2:
        test_images = list(common_images)[:2]
        print(f"Testing similarity between {test_images[0]} and {test_images[1]}")
        
        for feature_type in ['hsv', 'lbp', 'efficientnet']:
            if feature_type in feature_stats and 'error' not in feature_stats[feature_type]:
                try:
                    features = load_features_from_pickle(feature_type)
                    feat1 = features[test_images[0]]
                    feat2 = features[test_images[1]]
                    
                    if feat1 is not None and feat2 is not None:
                        # Test different similarity metrics
                        
                        # Cosine similarity
                        cos_sim = np.dot(feat1.flatten(), feat2.flatten()) / (
                            np.linalg.norm(feat1.flatten()) * np.linalg.norm(feat2.flatten()) + 1e-8
                        )
                        
                        # Chi-squared for histograms
                        if feature_type in ['hsv', 'lbp']:
                            f1, f2 = feat1.flatten(), feat2.flatten()
                            chi_sq = 0.5 * np.sum(((f1 - f2) ** 2) / (f1 + f2 + 1e-8))
                            chi_sim = 1.0 / (1.0 + chi_sq)
                            print(f"  {feature_type}: cosine={cos_sim:.4f}, chi_squared={chi_sim:.4f}")
                        else:
                            print(f"  {feature_type}: cosine={cos_sim:.4f}")
                    else:
                        print(f"  {feature_type}: Cannot compute - features are None")
                        
                except Exception as e:
                    print(f"  {feature_type}: Error computing similarity - {e}")
    else:
        print("⚠ Cannot test similarity - insufficient common images")
    
    # Check 5: Weight Configuration
    print("\n⚖️ 5. CURRENT WEIGHT CONFIGURATION")
    print("-" * 40)
    
    from similarity_search_pipeline import SimilaritySearch
    searcher = SimilaritySearch()
    print("Current weights:")
    total_weight = 0
    for feature_type, weight in searcher.weights.items():
        print(f"  {feature_type}: {weight}")
        total_weight += weight
    
    print(f"Total weight: {total_weight}")
    if abs(total_weight - 1.0) > 0.01:
        print("⚠ Weights don't sum to 1.0!")
    else:
        print("✓ Weights properly normalized")
    
    # Summary and Recommendations
    print("\n🎯 6. DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("-" * 40)
    
    issues = []
    recommendations = []
    
    # Check for major issues
    total_features = sum(stats.get('total', 0) for stats in feature_stats.values() if 'error' not in stats)
    if total_features == 0:
        issues.append("No features loaded successfully")
        recommendations.append("Re-run learning mode")
    
    if len(common_images) < 1000:
        issues.append(f"Too few common images ({len(common_images)})")
        recommendations.append("Ensure all feature extraction completes successfully")
    
    # Check for HSV/LBP normalization issues
    for feature_type in ['hsv', 'lbp']:
        if feature_type in feature_stats:
            features = load_features_from_pickle(feature_type)
            if features:
                sample_feat = list(features.values())[0]
                if sample_feat is not None and sample_feat.sum() > 2.0:
                    issues.append(f"{feature_type} features not properly normalized")
                    recommendations.append(f"Re-normalize {feature_type} histograms")
    
    if issues:
        print("🚨 CRITICAL ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("✅ No critical issues detected")
        print("💡 Consider using the fixed similarity search pipeline for better results")
    
    return feature_stats, issues, recommendations


if __name__ == "__main__":
    stats, issues, recommendations = diagnose_features()
    
    print(f"\n{'='*60}")
    if issues:
        print("❌ DIAGNOSIS COMPLETE - ISSUES FOUND")
        print("Run the recommendations above and re-test")
    else:
        print("✅ DIAGNOSIS COMPLETE - SYSTEM APPEARS HEALTHY")
        print("Consider switching to the fixed similarity pipeline")
    print(f"{'='*60}")
