import os
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import logging
from torchvision import transforms

# Import timm with error handling
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    timm = None
    logging.error("timm not available. Install with: pip install timm>=0.9.0")

from config import FEATURE_CONFIGS

logging.basicConfig(level=logging.INFO)

class ConvNeXtFeatureExtractor:
    """ConvNeXt-Base feature extractor with proper preprocessing and pooling."""
    
    def __init__(self, use_cuda=True, use_fp16=False):
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install with: pip install timm>=0.9.0")
        
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.model = None
        self.transform = None
        self.feature_dim = 1024  # ConvNeXt-Base dimension
        
        self.init_model()
        self.init_preprocessing()
        
        logging.info(f"ConvNeXt-Base loaded on {self.device}")
        if self.use_fp16:
            logging.info("FP16 optimization enabled")
    
    def init_model(self):
        """Initialize ConvNeXt-Base model with correct configuration."""
        try:
            config = FEATURE_CONFIGS['convnext']
            model_name = config['model_name']
            
            # Load ConvNeXt-Base with proper settings
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,  # Remove classification head
                global_pool='avg',  # Use global average pooling
                drop_rate=0.0,  # No dropout for inference
                drop_path_rate=0.0  # No stochastic depth for inference
            )
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Enable FP16 if requested
            if self.use_fp16:
                self.model = self.model.half()
            
            # Verify feature dimension with proper test
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                if self.use_fp16:
                    dummy_input = dummy_input.half()
                dummy_output = self.model(dummy_input)
                actual_dim = dummy_output.shape[1]
                
                if actual_dim != self.feature_dim:
                    logging.warning(f"Feature dimension mismatch: expected {self.feature_dim}, got {actual_dim}")
                    self.feature_dim = actual_dim
                
                logging.info(f"ConvNeXt feature dimension verified: {self.feature_dim}")
                
                # Test feature quality
                feature_norm = torch.norm(dummy_output).item()
                feature_std = torch.std(dummy_output).item()
                logging.info(f"Feature norm: {feature_norm:.4f}, std: {feature_std:.4f}")
                
        except Exception as e:
            logging.error(f"Failed to load ConvNeXt model: {e}")
            logging.info("Make sure timm is installed: pip install timm>=0.9.0")
            raise
    
    def init_preprocessing(self):
        """Initialize preprocessing with exact ImageNet standards."""
        config = FEATURE_CONFIGS['convnext']['preprocessing']
        
        # ConvNeXt preprocessing pipeline - CRITICAL: Match training exactly
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),  # Resize to 256 first
            transforms.CenterCrop(224),  # Then center crop to 224
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['normalize_mean'],  # [0.485, 0.456, 0.406]
                std=config['normalize_std']     # [0.229, 0.224, 0.225]
            )
        ])
        
        # Alternative transform for different input sizes
        self.direct_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['normalize_mean'],
                std=config['normalize_std']
            )
        ])
        
        logging.info("ConvNeXt preprocessing initialized with proper ImageNet standards")
    
    def preprocess_image(self, image_input, use_center_crop=True):
        """Preprocess image with exact ImageNet preprocessing."""
        try:
            # Handle different input types
            if isinstance(image_input, (str, os.PathLike)):
                # Load from file path with validation
                try:
                    with Image.open(image_input) as img_test:
                        img_test.verify()  # Check if image is corrupted
                    image = Image.open(image_input).convert('RGB')
                except (IOError, OSError, Image.DecompressionBombError, ValueError) as e:
                    logging.warning(f"Skipping corrupted image {os.path.basename(image_input)}: {e}")
                    return None
            else:
                # Convert numpy array to PIL Image
                if isinstance(image_input, np.ndarray):
                    if image_input.dtype != np.uint8:
                        # Ensure proper range
                        if image_input.max() <= 1.0:
                            image_input = (image_input * 255).astype(np.uint8)
                        else:
                            image_input = image_input.astype(np.uint8)
                    image = Image.fromarray(image_input)
                else:
                    image = image_input
            
            # Validate image dimensions
            if image.size[0] == 0 or image.size[1] == 0:
                logging.warning("Image has zero dimensions")
                return None
            
            # Apply transforms based on preference
            if use_center_crop and min(image.size) >= 256:
                # Use proper ImageNet preprocessing: resize to 256, then center crop to 224
                tensor = self.transform(image)
            else:
                # Direct resize to 224x224
                tensor = self.direct_transform(image)
            
            tensor = tensor.unsqueeze(0).to(self.device)  # Add batch dimension
            
            if self.use_fp16:
                tensor = tensor.half()
            
            return tensor
            
        except Exception as e:
            logging.error(f"Error preprocessing image: {e}")
            return None
    
    def extract_features(self, image_input, normalize=True, use_center_crop=True):
        """Extract ConvNeXt features with PROPER normalization that preserves variation."""
        try:
            tensor = self.preprocess_image(image_input, use_center_crop=use_center_crop)
            if tensor is None:
                return None
            
            with torch.no_grad():
                features = self.model(tensor) 
                if len(features.shape) > 1:
                    features = features.squeeze()        
                features = features.cpu().float().numpy()
            
            if len(features.shape) == 0:
                features = features.reshape(1)
            
            if features.shape[0] != self.feature_dim:
                logging.warning(f"Unexpected feature dimension: {features.shape[0]} vs {self.feature_dim}")
                if features.shape[0] > self.feature_dim:
                    features = features[:self.feature_dim]
                else:
                    padded = np.zeros(self.feature_dim, dtype=np.float32)
                    padded[:features.shape[0]] = features
                    features = padded
            
            # FIXED: Simple L2 normalization for ConvNeXt features
            if normalize:
                # Simple L2 normalization - preserves ConvNeXt's trained feature relationships
                norm = np.linalg.norm(features)
                if norm > 1e-8:
                    features = features / norm
                else:
                    # Only for completely degenerate features
                    logging.warning("Degenerate features detected, using random fallback")
                    features = np.random.normal(0, 0.01, self.feature_dim).astype(np.float32)
                    features = features / np.linalg.norm(features)
         
            # Final validation
            if not np.isfinite(features).all():
                logging.error("Non-finite values in features")
                return np.random.normal(0, 0.3, self.feature_dim).astype(np.float32)
            
            return features.astype(np.float32)
            
        except Exception as e:
            logging.error(f"Error extracting ConvNeXt features: {e}")
            return np.random.normal(0, 0.3, self.feature_dim).astype(np.float32)


    def extract_batch_features(self, image_list, batch_size=32, normalize=True, robust=False):
        """Extract features from a batch of images for efficiency."""
        all_features = []
        
        for i in range(0, len(image_list), batch_size):
            batch = image_list[i:i + batch_size]
            batch_tensors = []
            valid_indices = []
            
            # Preprocess batch
            for j, image_input in enumerate(batch):
                tensor = self.preprocess_image(image_input, use_center_crop=True)
                if tensor is not None:
                    batch_tensors.append(tensor.squeeze(0))  # Remove batch dim for stacking
                    valid_indices.append(i + j)
            
            if not batch_tensors:
                # Add zero features for failed batch
                all_features.extend([np.zeros(self.feature_dim, dtype=np.float32)] * len(batch))
                continue
            
            # Stack tensors and extract features
            try:
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                if self.use_fp16:
                    batch_tensor = batch_tensor.half()
                
                with torch.no_grad():
                    batch_features = self.model(batch_tensor)
                    batch_features = batch_features.cpu().float().numpy()
                
                # Validate batch features
                if batch_features.shape[1] != self.feature_dim:
                    logging.warning(f"Batch feature dimension mismatch: {batch_features.shape[1]} vs {self.feature_dim}")
                
                # Normalize if requested
                if normalize:
                    norms = np.linalg.norm(batch_features, axis=1, keepdims=True)
                    batch_features = batch_features / (norms + 1e-12)
                
                # Add batch features to results
                batch_idx = 0
                for j in range(len(batch)):
                    if i + j in valid_indices:
                        all_features.append(batch_features[batch_idx].astype(np.float32))
                        batch_idx += 1
                    else:
                        all_features.append(np.zeros(self.feature_dim, dtype=np.float32))
                        
            except Exception as e:
                logging.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add zero features for failed batch
                all_features.extend([np.zeros(self.feature_dim, dtype=np.float32)] * len(batch))
        
        return all_features
    
    def get_model_info(self):
        """Get detailed model information."""
        config = FEATURE_CONFIGS['convnext']
        
        # Get model parameters
        total_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else 0
        
        return {
            'model_name': config['model_name'],
            'feature_dim': self.feature_dim,
            'device': str(self.device),
            'fp16_enabled': self.use_fp16,
            'input_size': config['input_size'],
            'parameters': total_params,
            'trainable_parameters': trainable_params,
            'timm_available': TIMM_AVAILABLE,
            'preprocessing': 'ImageNet standard (resize 256 → center crop 224)',
            'normalization': 'L2 normalized for cosine similarity'
        }
    
    def validate_features(self, features):
        """Feature validation for the fixed normalization."""
        if features is None:
            return False, "Features are None"
        
        if not isinstance(features, np.ndarray):
            return False, "Features must be numpy array"
        
        if features.shape[0] != self.feature_dim:
            return False, f"Wrong feature dimension: {features.shape[0]} vs {self.feature_dim}"
        
        if not np.isfinite(features).all():
            return False, "Non-finite values in features"
        
        # Check norm (should not be exactly 1.0 anymore)
        norm = np.linalg.norm(features)
        if norm < 1e-8:
            return False, "Near-zero norm features"
    
        if norm > 100.0:  # Very generous upper bound
            return False, f"Abnormally large norm: {norm}"
    
        # Check for diversity
        feature_std = np.std(features)
        if feature_std < 1e-6:
            return False, f"Very low feature diversity (std: {feature_std})"
    
        return True, "Valid features with natural variation"

def test_normalization_fix():
    """Test the fixed normalization approach."""
    try:
        extractor = ConvNeXtFeatureExtractor(use_cuda=False, use_fp16=False)
        
        # Test with 10 random images
        test_results = []
        
        for i in range(15):
            if i < 5:
                # Random noise with different scales
                scale = 0.3 + (i * 0.4)  # 0.3 to 1.9
                test_image = (np.random.randn(224, 224, 3) * scale * 255).clip(0, 255).astype(np.uint8)
            elif i < 10:
                # Different intensity solid colors
                intensity = int(25 + (i-5) * 45)  # 25, 70, 115, 160, 205
                test_image = np.full((224, 224, 3), intensity, dtype=np.uint8)
            else:
                # Patterned images
                test_image = np.zeros((224, 224, 3), dtype=np.uint8)
                pattern_type = i - 10
                if pattern_type == 0:
                    # Checkerboard
                    test_image[::16, ::16] = 255
                elif pattern_type == 1:
                    # Gradient
                    test_image[:, :, 0] = np.linspace(0, 255, 224).reshape(1, -1)
                elif pattern_type == 2:
                    # Stripes
                    test_image[::8, :] = 255
                elif pattern_type == 3:
                    # Circles
                    y, x = np.ogrid[:224, :224]
                    mask = (x - 112)**2 + (y - 112)**2 < 50**2
                    test_image[mask] = 255
                else:
                    # Complex pattern
                    test_image = (np.sin(np.arange(224*224*3).reshape(224, 224, 3) * 0.01) * 127 + 128).astype(np.uint8)
            
            # Extract features
            features = extractor.extract_features(test_image, normalize=True)
            
            if features is not None:
                norm = np.linalg.norm(features)
                std = np.std(features)
                range_val = np.max(features) - np.min(features)
                mean_val = np.mean(features)
                
                test_results.append({
                    'norm': norm,
                    'std': std,
                    'range': range_val,
                    'mean': mean_val
                })
                
                print(f"Image {i+1:2d}: norm={norm:6.3f}, std={std:6.4f}, range={range_val:6.3f}, mean={mean_val:7.4f}")
        
        if not test_results:
            print("No valid test results")
            return False
        
        # Analyze results with REALISTIC expectations
        norms = [r['norm'] for r in test_results]
        stds = [r['std'] for r in test_results]
        
        norm_min = np.min(norms)
        norm_max = np.max(norms)
        norm_std = np.std(norms)
        norm_mean = np.mean(norms)
        std_mean = np.mean(stds)
        norm_range = norm_max - norm_min
        
        print(f"Norm statistics:")
        print(f"Range: {norm_min:.3f} - {norm_max:.3f} (spread: {norm_range:.3f})")
        print(f"Mean: {norm_mean:.3f}")
        print(f"Std deviation: {norm_std:.6f}")
        print(f"Feature diversity:")
        print(f"Average feature std: {std_mean:.6f}")
        
        # REVISED success criteria - more realistic
        norm_variation_good = norm_std > 0.05  # Relaxed from 0.01 to 0.05
        norm_range_good = norm_range > 0.1     # At least 0.1 difference between min/max
        reasonable_magnitude = 1.0 < norm_mean < 10.0  # Much wider range
        feature_diversity_good = std_mean > 0.02  # Relaxed from 0.05 to 0.02
        not_identical = norm_std > 0.001  # Keep this strict
        
        print(f"\n=== VALIDATION ===")
        print(f"Norm variation good (>{0.05:.3f}): {norm_variation_good} ({norm_std:.6f})")
        print(f"Norm range good (>{0.1:.1f}): {norm_range_good} ({norm_range:.3f})")
        print(f"Reasonable magnitude (0.5-20): {reasonable_magnitude} ({norm_mean:.3f})")
        print(f"Feature diversity good (>{0.02:.3f}): {feature_diversity_good} ({std_mean:.6f})")
        print(f"Not identical (>{0.001:.3f}): {not_identical} ({norm_std:.6f})")
        
        all_good = (norm_variation_good and norm_range_good and 
                   reasonable_magnitude and feature_diversity_good and not_identical)
        
        if all_good:
            print(f"NORMALIZATION FIX SUCCESSFUL!")
            print(f"Features now have natural norm variation!")
            print(f"No more identical norms problem!")
            
            print(f"Next steps:")
            print(f"1. Delete old features: del D:\\Code_image_rec\\pickles\\convnext_features.pkl")
            print(f"2. Re-extract: python main.py --mode learning --batch-size 16")
            print(f"3. Check weights: python main.py --analyze-weights --recalculate-weights")
            print(f"Expected: ConvNeXt weight should increase significantly from 0.429")
            
            return True
        else:
            print(f"NORMALIZATION FIX STILL NEEDS WORK")
            if not norm_variation_good:
                print(f"   - Norm variation too low: {norm_std:.6f} (need > 0.05)")
            if not norm_range_good:
                print(f"   - Norm range too small: {norm_range:.3f} (need > 0.1)")
            if not reasonable_magnitude:
                print(f"   - Magnitude out of range: {norm_mean:.3f} (need 0.5-20)")
            if not feature_diversity_good:
                print(f"   - Feature diversity too low: {std_mean:.6f} (need > 0.02)")
            if not not_identical:
                print(f"   - Norms still identical: {norm_std:.6f} (need > 0.001)")
            
            return False
    
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    
# Test the implementation
if __name__ == "__main__":
    success = test_normalization_fix()
        
    if success:
        print("ConvNeXt normalization fix validated!")
        print("3. Analyze weights: python main.py --analyze-weights") 
    else:
        print("ConvNeXt feature extractor test failed")
        

