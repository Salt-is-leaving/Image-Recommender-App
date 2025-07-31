import os
import numpy as np
import cv2
from PIL import Image
import pickle
from tqdm import tqdm
import logging

from config import (PATH_TO_SSD, PICKLE_PATH, CHECKPOINT_PATH, CHECKPOINT_INTERVAL, 
                   FEATURE_CONFIGS, FEATURE_FILES, get_enabled_features, get_similarity_weights)
from db_api import (create_connection, insert_image_metadata, update_feature_metadata,
                   save_features_to_pickle, DB_PATH)
from convnext_extractor import ConvNeXtFeatureExtractor

logging.basicConfig(level=logging.INFO)

class MultiFeatureExtractor:
    """Extract multiple types of features from images using ConvNeXt and HSV."""
    
    def __init__(self, use_cuda=True):
        self.device_available = use_cuda
        self.convnext_extractor = None
        
        # Initialize models based on enabled features
        enabled_features = get_enabled_features()
        
        if 'convnext' in enabled_features:
            self.init_convnext()
        
        logging.info(f"MultiFeatureExtractor initialized with features: {enabled_features}")
    
    def init_convnext(self):
        """Initialize ConvNeXt model."""
        try:
            config = FEATURE_CONFIGS['convnext']
            self.convnext_extractor = ConvNeXtFeatureExtractor(
                use_cuda=self.device_available, 
                use_fp16=True)
            logging.info(f"ConvNeXt {config['model_name']} loaded successfully")
        except Exception as e:
            logging.error(f"Failed to initialize ConvNeXt: {e}")
            self.convnext_extractor = None
    
    def get_default_convnext_features(self):
        """Get default ConvNeXt feature vector on failure."""
        expected_dims = FEATURE_CONFIGS['convnext'].get('embedding_size', 1024)
        random_features = np.random.normal(0, 0.3, expected_dims).astype(np.float32) # Use random features with proper magnitude
    
        return random_features
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Load and preprocess image for feature extraction with robust error handling."""
        try:
            # First, try to open and verify the image
            with Image.open(image_path) as img_test:
                img_test.verify()  # Check if image is corrupted
            
            # If verification passes, load the actual image
            image = Image.open(image_path).convert('RGB')
            
            # Additional validation
            if image.size[0] == 0 or image.size[1] == 0:
                raise ValueError("Image has zero dimensions")
                
        except (IOError, OSError, Image.DecompressionBombError, ValueError) as e:
            logging.warning(f"Skipping corrupted/invalid image {os.path.basename(image_path)}: {e}")
            return {'success': False, 'error': str(e)}
        except Exception as e:
            logging.error(f"Unexpected error loading image {os.path.basename(image_path)}: {e}")
            return {'success': False, 'error': str(e)}
            
        # Get original size for metadata
        original_size = image.size
            
        # Resize for deep learning models
        image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
            
        # Convert to numpy arrays
        image_rgb = np.array(image_resized)
        
        # Validate arrays
        if image_rgb.shape != (target_size[1], target_size[0], 3):
            raise ValueError(f"Invalid image shape: {image_rgb.shape}")
        
        return {
            'rgb': image_rgb,
            'original_size': original_size,
            'success': True
        }
    
    def extract_hsv_histogram(self, image_rgb):
        """Extract HSV histogram with FIXED mask handling and better preprocessing."""
        try:
            config = FEATURE_CONFIGS['hsv']
            
            # Optional: slight blur to reduce noise
            if config.get('preprocessing', {}).get('gaussian_blur', False):
                blur_kernel = config['preprocessing'].get('blur_kernel', (3, 3))
                image_rgb = cv2.GaussianBlur(image_rgb, blur_kernel, 0)
            
            # Convert to HSV
            hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            
            # CRITICAL FIX: Create mask for low saturation regions with correct data type
            mask = None
            if config.get('preprocessing', {}).get('mask_low_saturation', False):
                sat_threshold = config['preprocessing'].get('saturation_threshold', 30)
                # FIX: Convert boolean mask to uint8 for OpenCV compatibility
                mask_bool = hsv_image[:, :, 1] > sat_threshold
                mask = mask_bool.astype(np.uint8) * 255  # Convert to 0/255 values
                logging.debug(f"Applied saturation mask with threshold {sat_threshold}")
            
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
            
            # CRITICAL: Validate dimensions
            expected_dims = np.prod(config['bins'])
            if len(hist) != expected_dims:
                logging.warning(f"HSV histogram dimension mismatch: got {len(hist)}, expected {expected_dims}")
                if len(hist) > expected_dims:
                    hist = hist[:expected_dims]
                else:
                    padded_hist = np.ones(expected_dims, dtype=np.float32) / expected_dims
                    padded_hist[:len(hist)] = hist
                    hist = padded_hist
            
            logging.debug(f"HSV histogram: shape={hist.shape}, sum={hist.sum():.6f}")
            return hist
            
        except Exception as e:
            logging.error(f"Error extracting HSV histogram: {e}")
            expected_dims = np.prod(FEATURE_CONFIGS['hsv']['bins'])
            return np.ones(expected_dims, dtype=np.float32) / expected_dims
    
    def extract_convnext_features(self, image_rgb):
        """Extract ConvNeXt deep features with format validation."""
        try:
            if self.convnext_extractor is None:
                logging.error("ConvNeXt extractor not initialized")
                return self.get_default_convnext_features()
            
            # Extract features using ConvNeXt extractor
            features = self.convnext_extractor.extract_features(image_rgb, normalize=True)
            
            if features is None:
                logging.warning("ConvNeXt extraction returned None, using default")
                return self.get_default_convnext_features()
            
            if not isinstance(features, np.ndarray) or len(features) != 1024:
                logging.warning("ConvNeXt feature format issue, using fallback")
                return self.get_default_convnext_features()
            
            # Log basic stats without overriding
            feature_norm = np.linalg.norm(features)
            feature_std = np.std(features)
            logging.debug(f"ConvNeXt extracted - norm: {feature_norm:.4f}, std: {feature_std:.4f}")
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting ConvNeXt features: {e}")
            return self.get_default_convnext_features()
    

    def extract_all_features(self, image_path):
        """Extract all enabled features from an image."""
        processed = self.preprocess_image(image_path)
        if not processed['success']:
            return None
        
        features = {'image_path': image_path}
        enabled_features = get_enabled_features()
        
        # Extract features based on enabled features
        if 'hsv' in enabled_features:
            features['hsv'] = self.extract_hsv_histogram(processed['rgb'])
        
        if 'convnext' in enabled_features:
            features['convnext'] = self.extract_convnext_features(processed['rgb'])
        
        # Add metadata
        try:
            features['file_size'] = os.path.getsize(image_path)
            features['width'], features['height'] = processed['original_size']
        except Exception as e:
            logging.warning(f"Could not get metadata for {image_path}: {e}")
            features['file_size'] = 0
            features['width'], features['height'] = 0, 0
        
        return features

def get_image_paths(path_to_ssd):
    """Get all supported image paths."""
    supported_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = []
    
    if not os.path.exists(path_to_ssd):
        logging.error(f"Image directory does not exist: {path_to_ssd}")
        return []
    
    for root, dirs, files in os.walk(path_to_ssd): #used os.walk to traverse all 216 directories
        for file in files:
            if file.lower().endswith(supported_extensions):
                image_paths.append(os.path.join(root, file)) # this line of code creates full path for all folders
    
    logging.info(f"Found {len(image_paths)} supported images")
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
    """Process images in batches for memory efficiency with error handling."""
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    processed_images = checkpoint.get('processed_images', set())
    all_features = {
        'convnext': {},  
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
    
    # Track processing statistics
    processed_count = 0
    error_count = 0
    skipped_count = 0
    
    # Process in batches
    for i in tqdm(range(0, len(remaining_paths), batch_size), desc="Processing batches"):
        batch_paths = remaining_paths[i:i + batch_size]
        batch_errors = 0
        
        for image_path in tqdm(batch_paths, desc=f"Batch {i//batch_size + 1}", leave=False):
            try:
                full_image_path = image_path #to preserve the full image path
                image_filename = os.path.basename(image_path)
                processed_count += 1
                
                # Extract all features
                features = extractor.extract_all_features(full_image_path)
                
                if features:
                    # Store in database
                    store_features_in_database(conn, features)
                    
                    # Collect features for pickle files
                    if features.get('convnext') is not None:
                        all_features['convnext'][full_image_path] = features['convnext']
                    
                    if features.get('hsv') is not None:
                        all_features['hsv'][full_image_path] = features['hsv']
                    
                    processed_images.add(image_filename)
                else:
                    skipped_count += 1
                    batch_errors += 1
                    logging.warning(f"Failed to extract features from {image_filename}")
                    
            except Exception as e:
                error_count += 1
                batch_errors += 1
                logging.error(f"Error processing {os.path.basename(image_path)}: {e}")
                continue
        
        # Log batch progress
        batch_success_rate = ((len(batch_paths) - batch_errors) / len(batch_paths)) * 100
        logging.info(f"Batch {i//batch_size + 1} completed: {batch_success_rate:.1f}% success rate")
        
        # Save checkpoint after each batch
        checkpoint_data = {
            'processed_images': processed_images,
            'features': all_features
        }
        save_checkpoint(checkpoint_data, checkpoint_path)
        
        # Save features to pickle files periodically
        if len(all_features['convnext']) % 500 == 0 and all_features['convnext']:  # Every 500 images
            save_features_periodically(all_features)
    
    # Final statistics
    success_rate = ((processed_count - error_count - skipped_count) / processed_count) * 100 if processed_count > 0 else 0
    logging.info(f"\nProcessing Summary:")
    logging.info(f"  Total processed: {processed_count}")
    logging.info(f"  Successful: {processed_count - error_count - skipped_count}")
    logging.info(f"  Errors: {error_count}")
    logging.info(f"  Skipped: {skipped_count}")
    logging.info(f"  Success rate: {success_rate:.1f}%")
    
    # Final save
    save_all_features_to_pickle(all_features)
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logging.info("Checkpoint cleaned up")
    
    return all_features

def store_features_in_database(conn, features):
    """Store extracted features with validation."""
    try:
        image_path = features['image_path']
        
        # Insert image metadata in database
        insert_image_metadata(
            conn, 
            image_path,
            features.get('file_size'),
            features.get('width'),
            features.get('height')
        )
        
        # Validate features before marking as available
        has_hsv = features.get('hsv') is not None and len(features.get('hsv', [])) > 0
        has_convnext = features.get('convnext') is not None  
        
        # Update feature metadata flags (no BLOBs stored in DB)
        update_feature_metadata(
            conn, 
            image_path,
            has_hsv=has_hsv,
            has_convnext=has_convnext 
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
            success = save_features_to_pickle(features_dict, feature_type)
            if success:
                logging.info(f"Saved {len(features_dict)} {feature_type} features")
            else:
                logging.error(f"Failed to save {feature_type} features")

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
    
    # Check ConvNeXt features
    if 'convnext' in features and features['convnext'] is not None:
        conv_feat = features['convnext']
        expected_size = FEATURE_CONFIGS['convnext'].get('embedding_size', 1024)
        if len(conv_feat) != expected_size:
            validation_results['issues'].append(f"ConvNeXt size mismatch: {len(conv_feat)} vs {expected_size}")
        if conv_feat.dtype != np.float32:
            validation_results['warnings'].append("ConvNeXt not float32")
    
    validation_results['valid'] = len(validation_results['issues']) == 0
    return validation_results

# ========== LEARNING MODE ==========
def learning_mode(image_directory=None, use_cuda=True, batch_size=50):
    """
    Learning Mode: Extract and store all features from images using ConvNeXt.
    This should be run once or when adding new images.
    """
    logging.info("=== STARTING LEARNING MODE (ConvNeXt + HSV) ===")
    
    if image_directory is None:
        image_directory = PATH_TO_SSD
    
    # Initialize feature extractor
    try:
        extractor = MultiFeatureExtractor(use_cuda=use_cuda)
    except Exception as e:
        logging.error(f"Failed to initialize feature extractor: {e}")
        return False
    
    # Get image paths
    image_paths = get_image_paths(image_directory)
    if not image_paths:
        logging.error("No images found to process")
        return False
    
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
        
        # Show model information
        if extractor.convnext_extractor:
            info = extractor.convnext_extractor.get_model_info()
            logging.info(f"ConvNeXt model info: {info['model_name']}, {info['feature_dim']} dims")
        
        return True
        
    except Exception as e:
        logging.error(f"Learning mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    # Run learning mode
    success = learning_mode(use_cuda=True, batch_size=30)
    
    if success:
        logging.info("Learning mode completed successfully!")
        logging.info("You can now run clustering mode: python main.py --mode clustering")
    else:
        logging.error("Learning mode failed!")
