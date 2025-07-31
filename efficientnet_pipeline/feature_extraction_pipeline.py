import os
import numpy as np
import cv2
from PIL import Image
import pickle
from tqdm import tqdm
from skimage import feature as skimage_feature
from efficientnet_pytorch import EfficientNet
import torch
import logging

from config import (PATH_TO_SSD, PICKLE_PATH, CHECKPOINT_PATH, CHECKPOINT_INTERVAL, 
                   FEATURE_CONFIGS, FEATURE_FILES, get_enabled_features)
from db_api import (create_connection, insert_image_metadata, update_feature_metadata,
                   save_features_to_pickle, DB_PATH)

logging.basicConfig(level=logging.INFO)

class MultiFeatureExtractor:
    """Extract multiple types of features from images."""
    
    def __init__(self, use_cuda=True):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.efficientnet_model = None
        
        # Initialize models based on enabled features
        enabled_features = get_enabled_features()
        
        if 'efficientnet' in enabled_features:
            self.init_efficientnet()
        
    def init_efficientnet(self):
        """Initialize EfficientNet model."""
        config = FEATURE_CONFIGS['efficientnet']
        self.efficientnet_model = EfficientNet.from_pretrained(config['model_name']).to(self.device)
        self.efficientnet_model.eval()
        logging.info(f"EfficientNet {config['model_name']} loaded on {self.device}")
    

    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Load and preprocess image for feature extraction."""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Get original size for metadata
            original_size = image.size
            
            # Resize for deep learning models
            image_resized = image.resize(target_size)
            
            # Convert to numpy arrays
            image_rgb = np.array(image_resized)
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            
            return {
                'rgb': image_rgb,
                'gray': image_gray,
                'original_size': original_size,
                'success': True
            }
        except Exception as e:
            logging.error(f"Error preprocessing image {image_path}: {e}")
            return {'success': False}
    
    def extract_hsv_histogram(self, image_rgb):
        """Extract enhanced HSV histogram with better preprocessing."""
        try:
            config = FEATURE_CONFIGS['hsv']
            
            # Optional: slight blur to reduce noise
            if config.get('preprocessing', {}).get('gaussian_blur', False):
                image_rgb = cv2.GaussianBlur(image_rgb, (3, 3), 0)
            
            # Convert to HSV
            hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            
            # Create mask for low saturation regions (optional)
            mask = None
            if config.get('preprocessing', {}).get('mask_low_saturation', False):
                sat_threshold = config['preprocessing'].get('saturation_threshold', 30)
                mask = hsv_image[:, :, 1] > sat_threshold
            
            # Calculate histogram with proper ranges
            hist = cv2.calcHist([hsv_image], [0, 1, 2], mask, config['bins'], config['ranges'])
            
            # Better normalization
            hist = hist.flatten().astype(np.float32)
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist = hist / hist_sum
                # Add small epsilon to avoid zeros in Bhattacharyya calculation
                hist = hist + 1e-10
                hist = hist / hist.sum()  # Re-normalize
            else:
                # Fallback uniform distribution
                hist = np.ones_like(hist) / len(hist)
            
            # Validate dimensions
            expected_dims = np.prod(config['bins'])
            if len(hist) != expected_dims:
                logging.warning(f"HSV histogram dimension mismatch: got {len(hist)}, expected {expected_dims}")
                if len(hist) > expected_dims:
                    hist = hist[:expected_dims]
                else:
                    padded_hist = np.ones(expected_dims, dtype=np.float32) / expected_dims
                    padded_hist[:len(hist)] = hist
                    hist = padded_hist
            
            logging.debug(f"Enhanced HSV histogram: shape={hist.shape}, sum={hist.sum():.6f}")
            return hist
            
        except Exception as e:
            logging.error(f"Error extracting enhanced HSV histogram: {e}")
            expected_dims = np.prod(FEATURE_CONFIGS['hsv']['bins'])
            return np.ones(expected_dims, dtype=np.float32) / expected_dims
    
    def extract_efficientnet_features(self, image_rgb):
        """Extract EfficientNet deep features with format validation."""
        try:
            # Preprocess for EfficientNet
            image_tensor = torch.tensor(image_rgb / 255.0).permute(2, 0, 1).float().unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.efficientnet_model.extract_features(image_tensor)
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.flatten().cpu().numpy()
            
            # Ensure consistent format
            features = features.astype(np.float32)
            
            # Validate expected dimensions
            config = FEATURE_CONFIGS['efficientnet']
            expected_dims = config.get('embedding_size', 2560)
            if len(features) != expected_dims:
                logging.warning(f"EfficientNet feature dimension mismatch: got {len(features)}, expected {expected_dims}")
            
            logging.debug(f"EfficientNet features shape: {features.shape}")
            return features
            
        except Exception as e:
            logging.error(f"Error extracting EfficientNet features: {e}")
            # Return default feature vector
            expected_dims = FEATURE_CONFIGS['efficientnet'].get('embedding_size', 2560)
            return np.zeros(expected_dims, dtype=np.float32)
    
    def extract_all_features(self, image_path, image_id):
        """Extract all enabled features from an image."""
        processed = self.preprocess_image(image_path)
        if not processed['success']:
            return None
        
        features = {'image_id': image_id, 'image_path': image_path}
        enabled_features = get_enabled_features()
        
        # Extract features based on enabled features
        if 'hsv' in enabled_features:
            features['hsv'] = self.extract_hsv_histogram(processed['rgb'])
        
        if 'efficientnet' in enabled_features:
            features['efficientnet'] = self.extract_efficientnet_features(processed['rgb'])
        
        # Add metadata
        features['file_size'] = os.path.getsize(image_path)
        features['width'], features['height'] = processed['original_size']
        
        return features

def get_image_paths(path_to_ssd):
    """Get all supported image paths."""
    supported_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = []
    for root, dirs, files in os.walk(path_to_ssd):
        for file in files:
            if file.lower().endswith(supported_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def save_checkpoint(data, checkpoint_path):
    """Save processing checkpoint."""
    try:
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Checkpoint saved: {len(data.get('processed_images', []))} images processed")
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")

def load_checkpoint(checkpoint_path):
    """Load processing checkpoint."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
    return {'processed_images': set(), 'features': {}}

def process_images_batch(extractor, image_paths, conn, checkpoint_path, batch_size=50):
    """Process images in batches for memory efficiency."""
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    processed_images = checkpoint['processed_images']
    all_features = {
        'efficientnet': {},
        'hsv': {}
    }
    
    # Filter out already processed images
    remaining_paths = [path for path in image_paths if os.path.basename(path) not in processed_images]
    
    logging.info(f"Total images: {len(image_paths)}")
    logging.info(f"Already processed: {len(processed_images)}")
    logging.info(f"Remaining to process: {len(remaining_paths)}")
    
    if not remaining_paths:
        logging.info("All images already processed!")
        return all_features
    
    # Process in batches
    for i in tqdm(range(0, len(remaining_paths), batch_size), desc="Processing batches"):
        batch_paths = remaining_paths[i:i + batch_size]
        
        for image_path in tqdm(batch_paths, desc=f"Batch {i//batch_size + 1}", leave=False):
            try:
                image_id = os.path.basename(image_path)
                
                # Extract all features
                features = extractor.extract_all_features(image_path, image_id)
                
                if features:
                    # Store in database
                    store_features_in_database(conn, features)
                    
                    # Collect features for pickle files
                    if features.get('efficientnet') is not None:
                        all_features['efficientnet'][image_id] = features['efficientnet']
                    
                    if features.get('hsv') is not None:
                        all_features['hsv'][image_id] = features['hsv']
                    
                    processed_images.add(image_id)
                    
            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")
                continue
        
        # Save checkpoint after each batch
        checkpoint_data = {
            'processed_images': processed_images,
            'features': all_features
        }
        save_checkpoint(checkpoint_data, checkpoint_path)
        
        # Save features to pickle files periodically
        if len(all_features['efficientnet']) % 500 == 0:  # Every 500 images
            save_features_periodically(all_features)
    
    # Final save
    save_all_features_to_pickle(all_features)
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logging.info("Checkpoint cleaned up")
    
    return all_features

def store_features_in_database(conn, features):
    """Store extracted features with enhanced validation."""
    try:
        image_id = features['image_id']
        image_path = features['image_path']
        
        # Insert image metadata in database
        insert_image_metadata(
            conn, image_id, image_path,
            features.get('file_size'),
            features.get('width'),
            features.get('height')
        )
        
        # Validate features before marking as available
        has_hsv = features.get('hsv') is not None and len(features.get('hsv', [])) > 0
        has_efficientnet = features.get('efficientnet') is not None
        
        # Update feature metadata flags (no BLOBs stored in DB)
        update_feature_metadata(
            conn, image_id,
            has_hsv=has_hsv,
            has_efficientnet=has_efficientnet
        )
        
    except Exception as e:
        logging.error(f"Error storing features in database: {e}")

def save_features_periodically(all_features):
    """Save features to pickle files periodically."""
    for feature_type, features_dict in all_features.items():
        if features_dict:  # Only save if there are features
            save_features_to_pickle(features_dict, feature_type)

def save_all_features_to_pickle(all_features):
    """Save all features to their respective pickle files."""
    logging.info("Saving all features to pickle files...")
    
    for feature_type, features_dict in all_features.items():
        if features_dict:
            save_features_to_pickle(features_dict, feature_type)
    
def validate_extracted_features(features):
    """Validate extracted features for consistency with clustering pipeline."""
    validation_results = {
        'valid': True,
        'issues': [],
        'warnings': []
    }
    
    if not features:
        validation_results['valid'] = False
        validation_results['issues'].append("No features provided")
        return validation_results
    
    # Check HSV features
    if 'hsv' in features and features['hsv'] is not None:
        hsv_feat = features['hsv']
        expected_size = np.prod(FEATURE_CONFIGS['hsv']['bins'])
        if len(hsv_feat) != expected_size:
            validation_results['issues'].append(f"HSV size mismatch: {len(hsv_feat)} vs {expected_size}")
        if hsv_feat.dtype != np.float32:
            validation_results['warnings'].append("HSV not float32")
    
    # Check EfficientNet features
    if 'efficientnet' in features and features['efficientnet'] is not None:
        eff_feat = features['efficientnet']
        expected_size = FEATURE_CONFIGS['efficientnet'].get('embedding_size', 2560)
        if len(eff_feat) != expected_size:
            validation_results['issues'].append(f"EfficientNet size mismatch: {len(eff_feat)} vs {expected_size}")
    
    validation_results['valid'] = len(validation_results['issues']) == 0
    return validation_results

# ========== LEARNING MODE ==========
def learning_mode(image_directory=None, use_cuda=True, batch_size=50):
    """
    Learning Mode: Extract and store all features from images.
    This should be run once or when adding new images.
    """
    logging.info("=== STARTING LEARNING MODE ===")
    
    if image_directory is None:
        image_directory = PATH_TO_SSD
    
    # Initialize feature extractor
    extractor = MultiFeatureExtractor(use_cuda=use_cuda)
    
    # Get image paths
    image_paths = get_image_paths(image_directory)
    logging.info(f"Found {len(image_paths)} images to process")
    
    # Connect to database
    conn = create_connection()
    if conn is None:
        logging.error("Could not connect to database")
        return False
    
    # Process all images
    try:
        features = process_images_batch(
            extractor, image_paths, conn, CHECKPOINT_PATH, batch_size
        )
        
        logging.info("=== LEARNING MODE COMPLETED ===")
        logging.info(f"Features extracted:")
        for feature_type, feature_dict in features.items():
            logging.info(f"  {feature_type}: {len(feature_dict)} images")
        
        return True
        
    except Exception as e:
        logging.error(f"Learning mode failed: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    # Run learning mode
    success = learning_mode(use_cuda=True, batch_size=30)
    
    if success:
        logging.info("Learning mode completed successfully!")
        logging.info("You can now run similarity_search.py for comparison mode")
    else:
        logging.error("Learning mode failed!")
