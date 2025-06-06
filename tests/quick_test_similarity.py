#!/usr/bin/env python3
"""
Quick Test: Similarity Search without FAISS dependency
Tests your current pipeline to identify the exact problems
"""

import os
import numpy as np
import logging
from config import PICKLE_PATH
from db_api import load_features_from_pickle

logging.basicConfig(level=logging.INFO)

def quick_similarity_test(target_image_id=None):
    """Quick test of similarity search without external dependencies."""
    print("=" * 60)
    print("🧪 QUICK SIMILARITY TEST")
    print("=" * 60)
    
    # Load features
    print("\n📂 Loading features...")
    features = {}
    feature_types = ['efficientnet', 'hsv', 'lbp', 'orb']
    
    for feature_type in feature_types:
        try:
            feat_dict = load_features_from_pickle(feature_type)
            if feat_dict:
                # Quick preprocessing
                processed = {}
                for img_id, feat in feat_dict.items():
                    if feat is not None:
                        if feature_type == 'orb':
                            # Handle ORB variable shapes
                            if len(feat.shape) > 1:
                                feat = np.median(feat, axis=0)  # Better than mean
                        elif len(feat.shape) > 1:
                            feat = feat.flatten()
                        
                        # Quick normalization
                        if feature_type in ['hsv', 'lbp']:
                            # L1 norm for histograms
                            feat = feat / (np.sum(np.abs(feat)) + 1e-8)
                        else:
                            # L2 norm for deep features
                            feat = feat / (np.linalg.norm(feat) + 1e-8)
                        
                        processed[img_id] = feat.astype(np.float32)
                
                features[feature_type] = processed
                print(f"✓ {feature_type}: {len(processed)} features")
            else:
                print(f"✗ {feature_type}: No features found")
        except Exception as e:
            print(f"✗ {feature_type}: Error - {e}")
    
    # Find common images
    if not features:
        print("❌ No features loaded!")
        return False
    
    all_image_sets = [set(f.keys()) for f in features.values()]
    common_images = list(set.intersection(*all_image_sets))
    print(f"\n🔗 Common images: {len(common_images)}")
    
    if len(common_images) < 10:
        print("⚠️  Too few common images for meaningful similarity search!")
        return False
    
    # Select target image
    if target_image_id is None or target_image_id not in common_images:
        target_image_id = common_images[7] if len(common_images) > 7 else common_images[0]
    
    print(f"\n🎯 Target image: {target_image_id}")
    
    # Test individual feature similarities
    print("\n🔍 Computing similarities...")
    
    weights = {
        'efficientnet': 0.45,
        'hsv': 0.25,  # Reduced due to over-dimensioning
        'lbp': 0.25,
        'orb': 0.05
    }
    
    all_similarities = {}
    
    for feature_type, feat_dict in features.items():
        if target_image_id not in feat_dict:
            continue
        
        target_feat = feat_dict[target_image_id]
        feature_sims = []
        
        print(f"\n  Testing {feature_type}...")
        print(f"    Target feature shape: {target_feat.shape}")
        print(f"    Target feature stats: min={target_feat.min():.4f}, max={target_feat.max():.4f}, mean={target_feat.mean():.4f}")
        
        for candidate_id in common_images:
            if candidate_id == target_image_id:
                continue
            
            candidate_feat = feat_dict[candidate_id]
            
            # Compute cosine similarity
            if feature_type == 'hsv':
                # Chi-squared for HSV histogram
                denominator = target_feat + candidate_feat + 1e-8
                chi_sq = 0.5 * np.sum(((target_feat - candidate_feat) ** 2) / denominator)
                similarity = 1.0 / (1.0 + chi_sq)
            else:
                # Cosine similarity
                similarity = np.dot(target_feat, candidate_feat) / (
                    np.linalg.norm(target_feat) * np.linalg.norm(candidate_feat) + 1e-8
                )
            
            similarity = max(0.0, min(1.0, similarity))
            feature_sims.append((candidate_id, similarity))
        
        # Sort and show top results for this feature
        feature_sims.sort(key=lambda x: x[1], reverse=True)
        
        print(f"    Top 3 similar images by {feature_type}:")
        for i, (img_id, sim) in enumerate(feature_sims[:3]):
            print(f"      {i+1}. {img_id}: {sim:.4f}")
        
        # Check if similarities are reasonable
        similarities_list = [sim for _, sim in feature_sims]
        avg_sim = np.mean(similarities_list)
        max_sim = np.max(similarities_list)
        min_sim = np.min(similarities_list)
        
        print(f"    Similarity range: {min_sim:.4f} to {max_sim:.4f} (avg: {avg_sim:.4f})")
        
        if max_sim < 0.1:
            print(f"    ⚠️  Very low similarities for {feature_type} - possible normalization issue!")
        elif avg_sim > 0.9:
            print(f"    ⚠️  Very high similarities for {feature_type} - features may be too similar!")
        else:
            print(f"    ✓ {feature_type} similarities look reasonable")
        
        # Add to combined similarities
        weight = weights[feature_type]
        for candidate_id, similarity in feature_sims:
            if candidate_id not in all_similarities:
                all_similarities[candidate_id] = {'total': 0, 'count': 0, 'details': {}}
            
            all_similarities[candidate_id]['total'] += similarity * weight
            all_similarities[candidate_id]['count'] += 1
            all_similarities[candidate_id]['details'][feature_type] = similarity
    
    # Combined results
    print(f"\n🏆 COMBINED SIMILARITY RESULTS")
    print("=" * 50)
    
    combined_results = []
    for img_id, data in all_similarities.items():
        if data['count'] >= 3:  # At least 3 feature types
            combined_results.append({
                'image_id': img_id,
                'score': data['total'],
                'details': data['details']
            })
    
    combined_results.sort(key=lambda x: x['score'], reverse=True)
    
    if combined_results:
        print(f"Top 10 most similar images:")
        for i, result in enumerate(combined_results[:10]):
            print(f"{i+1:2d}. {result['image_id']} - Score: {result['score']:.4f}")
            details = result['details']
            print(f"     EfficientNet: {details.get('efficientnet', 0):.3f}, "
                  f"HSV: {details.get('hsv', 0):.3f}, "
                  f"LBP: {details.get('lbp', 0):.3f}, "
                  f"ORB: {details.get('orb', 0):.3f}")
        
        # Analysis
        top_score = combined_results[0]['score']
        avg_score = np.mean([r['score'] for r in combined_results[:10]])
        
        print(f"\n📊 Analysis:")
        print(f"   Top score: {top_score:.4f}")
        print(f"   Avg top-10: {avg_score:.4f}")
        
        if top_score > 0.7:
            print("   ✅ High similarities found - system working well!")
        elif top_score > 0.4:
            print("   ⚠️  Moderate similarities - may need tuning")
        else:
            print("   ❌ Low similarities - significant issues detected")
            print("   Recommendations:")
            print("     1. Fix HSV configuration (reduce from 64³ to 32³ bins)")
            print("     2. Re-run learning mode")
            print("     3. Check feature normalization")
        
        return True
    else:
        print("❌ No combined results - features are not compatible!")
        return False


def diagnose_hsv_problem():
    """Specifically diagnose the HSV over-dimensioning issue."""
    print("\n🔬 HSV DIAGNOSTIC")
    print("=" * 30)
    
    try:
        hsv_features = load_features_from_pickle('hsv')
        if hsv_features:
            sample_feature = list(hsv_features.values())[0]
            dims = sample_feature.shape[0] if len(sample_feature.shape) == 1 else np.prod(sample_feature.shape)
            
            print(f"HSV feature dimensions: {dims}")
            
            if dims > 100000:
                print("❌ CRITICAL: HSV features are over-dimensioned!")
                print(f"   Current: {dims} dimensions")
                print(f"   Expected: ~32,000 dimensions (32³ bins)")
                print(f"   Problem: Using 64³ = 262,144 bins creates noise")
                print("")
                print("🔧 SOLUTION:")
                print("   1. Edit config.py and change HSV bins from [64,64,64] to [32,32,32]")
                print("   2. Re-run: python main.py --mode learning")
                print("   3. This will reduce HSV from 23GB to ~1GB")
                return False
            else:
                print("✅ HSV dimensions are reasonable")
                return True
        else:
            print("❌ No HSV features found")
            return False
    except Exception as e:
        print(f"❌ Error checking HSV: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Quick Similarity Test (No external dependencies)")
    
    # First diagnose HSV
    hsv_ok = diagnose_hsv_problem()
    
    if not hsv_ok:
        print("\n" + "="*60)
        print("⚠️  HSV CONFIGURATION MUST BE FIXED FIRST")
        print("   Follow the solution steps above, then re-run this test")
        print("="*60)
    else:
        # Run similarity test
        success = quick_similarity_test()
        
        if success:
            print("\n" + "="*60)
            print("✅ QUICK TEST COMPLETED")
            print("   Your similarity system is working!")
            print("   Install FAISS for better performance: pip install faiss-cpu")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ ISSUES DETECTED")
            print("   Check the recommendations above")
            print("="*60)
