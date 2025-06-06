#!/usr/bin/env python3
"""
Fix HSV Features Only - Keep all other good features
Saves 4-5 hours by only re-extracting the corrupted HSV features
"""

import os
import numpy as np
import cv2
from PIL import Image
import pickle
from tqdm import tqdm
import logging

from config import PATH_TO_SSD, PICKLE_PATH, CHECKPOINT_PATH
from db_api import (create_connection, get_image_ids, save_features_to_pickle, 
                   load_features_from_pickle)

logging.basicConfig(level=logging.INFO)

# FIXED HSV Configuration
FIXED_HSV_CONFIG = {
    'bins': [32, 32, 32],  # REDUCED from [64, 64, 64]
    'ranges': [0, 256, 0, 256, 0, 256]
}

def extract_hsv_histogram_fixed(image_path):
    """Extract HSV histogram with fixed configuration."""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((224, 224))
        image_rgb = np.array(image_resized)
        
        # Convert to HSV
        hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # Calculate 3D histogram with FIXED bins
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, 
                          FIXED_HSV_CONFIG['bins'], FIXED_HSV_CONFIG['ranges'])
        
        # Normalize histogram (L1 normalization for histograms)
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-8)  # Avoid division by zero
        
        return hist.astype(np.float32)
        
    except Exception as e:
        logging.error(f"Error extracting HSV from {image_path}: {e}")
        return None

def get_all_image_paths():
    """Get all image paths from the database directory."""
    supported_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = {}  # image_id -> full_path
    
    for root, dirs, files in os.walk(PATH_TO_SSD):
        for file in files:
            if file.lower().endswith(supported_extensions):
                full_path = os.path.join(root, file)
                image_id = file  # filename as image_id
                image_paths[image_id] = full_path
    
    return image_paths

def fix_hsv_features_only():
    """Fix only the HSV features, keep all others intact."""
    print("=" * 60)
    print("🔧 FIXING HSV FEATURES ONLY")
    print("=" * 60)
    
    # Check current feature status
    print("\n📊 Current Feature Status:")
    feature_types = ['efficientnet', 'hsv', 'lbp', 'orb']
    for ft in feature_types:
        try:
            features = load_features_from_pickle(ft)
            if features:
                sample_feat = list(features.values())[0]
                dims = sample_feat.shape[0] if len(sample_feat.shape) == 1 else np.prod(sample_feat.shape)
                print(f"  ✓ {ft}: {len(features)} images, {dims} dims")
            else:
                print(f"  ✗ {ft}: No features found")
        except Exception as e:
            print(f"  ✗ {ft}: Error - {e}")
    
    # Get image paths and existing image IDs
    print("\n📂 Loading image information...")
    all_image_paths = get_all_image_paths()
    
    # Get image IDs that have other features (to know which ones to process)
    conn = create_connection()
    if conn:
        existing_image_ids = get_image_ids(conn)
        conn.close()
    else:
        # Fallback: use EfficientNet features as reference
        efficientnet_features = load_features_from_pickle('efficientnet')
        existing_image_ids = list(efficientnet_features.keys()) if efficientnet_features else []
    
    print(f"Found {len(existing_image_ids)} images with existing features")
    print(f"Found {len(all_image_paths)} total images on disk")
    
    # Filter to images that exist in both
    images_to_process = []
    for image_id in existing_image_ids:
        if image_id in all_image_paths:
            images_to_process.append((image_id, all_image_paths[image_id]))
    
    print(f"Will re-extract HSV for {len(images_to_process)} images")
    
    if len(images_to_process) == 0:
        print("❌ No images to process!")
        return False
    
    # Show the fix
    old_dims = 64 * 64 * 64  # 262,144
    new_dims = 32 * 32 * 32  # 32,768
    reduction = (old_dims - new_dims) / old_dims * 100
    
    print(f"\n🎯 HSV Feature Fix:")
    print(f"  Old dimensions: {old_dims:,} (64³ bins)")
    print(f"  New dimensions: {new_dims:,} (32³ bins)")
    print(f"  Reduction: {reduction:.1f}%")
    print(f"  File size: ~23GB → ~1GB")
    
    # Confirm before proceeding
    response = input(f"\n💡 Proceed with fixing HSV features for {len(images_to_process)} images? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("Operation cancelled.")
        return False
    
    # Extract new HSV features
    print(f"\n🔄 Extracting fixed HSV features...")
    
    new_hsv_features = {}
    batch_size = 100
    
    for i in tqdm(range(0, len(images_to_process), batch_size), desc="Processing batches"):
        batch = images_to_process[i:i + batch_size]
        
        for image_id, image_path in tqdm(batch, desc=f"Batch {i//batch_size + 1}", leave=False):
            try:
                hsv_feature = extract_hsv_histogram_fixed(image_path)
                if hsv_feature is not None:
                    new_hsv_features[image_id] = hsv_feature
                else:
                    logging.warning(f"Failed to extract HSV for {image_id}")
            except Exception as e:
                logging.error(f"Error processing {image_id}: {e}")
                continue
        
        # Save progress periodically
        if len(new_hsv_features) % 1000 == 0:
            print(f"  Progress: {len(new_hsv_features)} features extracted...")
    
    print(f"\n✅ Extracted {len(new_hsv_features)} fixed HSV features")
    
    # Verify the fix
    if new_hsv_features:
        sample_feature = list(new_hsv_features.values())[0]
        actual_dims = sample_feature.shape[0]
        expected_dims = 32 * 32 * 32
        
        print(f"\n🔍 Verification:")
        print(f"  Expected dimensions: {expected_dims}")
        print(f"  Actual dimensions: {actual_dims}")
        
        if actual_dims == expected_dims:
            print("  ✅ HSV features have correct dimensions!")
        else:
            print(f"  ⚠️  Dimension mismatch! Check configuration.")
            return False
        
        # Check normalization
        sample_sum = sample_feature.sum()
        print(f"  Feature sum (should be ~1.0): {sample_sum:.6f}")
        
        if 0.99 <= sample_sum <= 1.01:
            print("  ✅ HSV features properly normalized!")
        else:
            print("  ⚠️  Normalization issue detected.")
    
    # Save the fixed HSV features
    print(f"\n💾 Saving fixed HSV features...")
    
    # Backup old HSV features first
    old_hsv_path = os.path.join(PICKLE_PATH, 'hsv_features.pkl')
    backup_path = os.path.join(PICKLE_PATH, 'hsv_features_backup_corrupted.pkl')
    
    if os.path.exists(old_hsv_path):
        import shutil
        shutil.move(old_hsv_path, backup_path)
        print(f"  📦 Backed up corrupted HSV features to: {backup_path}")
    
    # Save new features
    success = save_features_to_pickle(new_hsv_features, 'hsv')
    
    if success:
        print("  ✅ Fixed HSV features saved successfully!")
        
        # Show file size improvement
        new_size = os.path.getsize(os.path.join(PICKLE_PATH, 'hsv_features.pkl')) / 1024 / 1024
        backup_size = os.path.getsize(backup_path) / 1024 / 1024 if os.path.exists(backup_path) else 0
        
        print(f"\n📈 File Size Improvement:")
        print(f"  Old HSV file: {backup_size:.1f} MB")
        print(f"  New HSV file: {new_size:.1f} MB")
        if backup_size > 0:
            reduction_mb = backup_size - new_size
            reduction_pct = reduction_mb / backup_size * 100
            print(f"  Reduction: {reduction_mb:.1f} MB ({reduction_pct:.1f}%)")
        
        return True
    else:
        print("  ❌ Failed to save fixed HSV features!")
        return False

def test_fixed_features():
    """Test the fixed features quickly."""
    print(f"\n🧪 Testing Fixed Features...")
    
    # Load all features
    features = {}
    for ft in ['efficientnet', 'hsv', 'lbp', 'orb']:
        feat_dict = load_features_from_pickle(ft)
        if feat_dict:
            features[ft] = feat_dict
            sample_feat = list(feat_dict.values())[0]
            dims = sample_feat.shape[0] if len(sample_feat.shape) == 1 else np.prod(sample_feat.shape)
            print(f"  ✓ {ft}: {len(feat_dict)} features, {dims} dims")
    
    # Find common images
    if len(features) >= 4:
        all_image_sets = [set(f.keys()) for f in features.values()]
        common_images = list(set.intersection(*all_image_sets))
        print(f"  ✓ Common images across all features: {len(common_images)}")
        
        if len(common_images) > 100:
            print("  ✅ System ready for similarity search!")
            return True
        else:
            print("  ⚠️  Few common images - check feature alignment")
            return False
    else:
        print("  ❌ Missing feature types")
        return False

if __name__ == "__main__":
    print("🚀 Smart HSV Fix - Keep Good Features, Fix Bad Ones")
    
    success = fix_hsv_features_only()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ HSV FEATURES FIXED SUCCESSFULLY!")
        print("=" * 60)
        
        # Test the system
        if test_fixed_features():
            print("\n🎉 Your similarity search system is now ready!")
            print("Next steps:")
            print("  1. Install FAISS: pip install faiss-cpu")
            print("  2. Test similarity: python faiss_similarity_pipeline.py")
            print("  3. Try interactive mode: python faiss_interactive_pipeline.py")
        else:
            print("\n⚠️  System needs additional attention - check the logs above")
    else:
        print("\n" + "=" * 60)
        print("❌ HSV FIX FAILED")
        print("Check the error messages above")
        print("=" * 60)
