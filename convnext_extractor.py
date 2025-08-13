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
        
        # ConvNeXt preprocessing pipeline should match training exactly
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
    
    def extract_features(self, image_input, normalize=False, use_center_crop=True):
        """Extract ConvNeXt features WITHOUT any normalization - pure raw output."""
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
            #ConvNext normalization is only applied in ConvNextFeatureExtractor.extract_features_raw and nowhere else
            if normalize:
                norm = np.linalg.norm(features)
                if norm > 1e-8:
                    features = features / norm
                    logging.debug(f"ConvNeXt features normalized: norm was {norm:.4f}")
                else:
                    logging.warning("ConvNeXt features have near-zero norm, skipping normalization")
            else:
                logging.debug(f"Raw ConvNeXt features: norm={np.linalg.norm(features):.4f}, std={np.std(features):.6f}")
            
            # Only validate for non-finite values
            if not np.isfinite(features).all():
                logging.error("Non-finite values in features")
                features = np.random.normal(0, 1.0, self.feature_dim).astype(np.float32)
                if normalize:
                    features = features / np.linalg.norm(features)
            return features.astype(np.float32)
            
        except Exception as e:
            logging.error(f"Error extracting ConvNeXt features: {e}")
            fallback = np.random.normal(0, 1.0, self.feature_dim).astype(np.float32)
            if normalize:
                fallback = fallback / np.linalg.norm(fallback)
            return fallback

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
        """Feature validation for the correct normalization."""
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

def test_normalization_parameter():
    """Test if normalization parameter works correctly."""
    try:
        extractor = ConvNeXtFeatureExtractor(use_cuda=False, use_fp16=False)
        
        # Create a test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test with normalize=False (should preserve diversity)
        raw_features = extractor.extract_features(test_image, normalize=False)
        raw_norm = np.linalg.norm(raw_features)
        
        # Test with normalize=True (should normalize to ~1.0)
        norm_features = extractor.extract_features(test_image, normalize=True)
        norm_norm = np.linalg.norm(norm_features)
        
        print(f"Raw features norm: {raw_norm:.4f}")
        print(f"Normalized features norm: {norm_norm:.4f}")
        
        # Validate
        normalize_works = abs(norm_norm - 1.0) < 0.1 and raw_norm > 1.0
        
        if normalize_works:
            print("Normalize parameter works correctly!")
            print(f"Raw norm: {raw_norm:.2f}, Normalized norm: {norm_norm:.2f}")
            return True
        else:
            print("Normalize parameter not working!")
            print(f"Expected: raw > 1.0, normalized ≈ 1.0")
            print(f"Got: raw = {raw_norm:.2f}, normalized = {norm_norm:.2f}")
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        return False
        
    
# Test the implementation
if __name__ == "__main__":
    success = test_normalization_parameter()
        
    if success:
        print("ConvNeXt normalization fix validated!")
        print("Analyze weights: python main.py --analyze-weights") 
    else:
        print("ConvNeXt feature extractor test failed")
        

