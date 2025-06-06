#!/usr/bin/env python3
"""
Feature Validation Script
Prüft die extrahierten Features in den Pickle-Dateien auf Korrektheit und Konsistenz.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import logging

# Konfiguration
PICKLE_PATH = r"D:\Code_image_rec\pickles"
FEATURE_FILES = {
    'efficientnet': 'efficientnet_features.pkl',
    'hsv': 'hsv_features.pkl',
    'lbp': 'lbp_features.pkl',
    'orb': 'orb_features.pkl'
}

# Erwartete Dimensionen
EXPECTED_DIMENSIONS = {
    'efficientnet': 2560,      # EfficientNet-B7 output
    'hsv': 262144,             # 64^3 HSV histogram
    'lbp': 26,                 # 24 LBP points + 2
    'orb': 32                  # Standard ORB descriptor size
}

logging.basicConfig(level=logging.INFO)

def load_and_analyze_features(feature_type):
    """Lade und analysiere Features eines bestimmten Typs."""
    filepath = os.path.join(PICKLE_PATH, FEATURE_FILES[feature_type])
    
    if not os.path.exists(filepath):
        logging.error(f"Feature file not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            features = pickle.load(f)
        
        logging.info(f"\n=== {feature_type.upper()} FEATURES ANALYSIS ===")
        logging.info(f"File: {filepath}")
        logging.info(f"File size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
        logging.info(f"Number of features: {len(features)}")
        
        if not features:
            logging.warning("No features found in file!")
            return None
        
        # Analysiere Feature-Dimensionen
        sample_keys = list(features.keys())[:10]  # Erste 10 Samples
        dimensions = []
        feature_types = []
        valid_features = 0
        invalid_features = 0
        
        for i, key in enumerate(sample_keys):
            feature = features[key]
            if feature is not None:
                if isinstance(feature, np.ndarray):
                    if len(feature.shape) == 1:
                        dimensions.append(feature.shape[0])
                    else:
                        dimensions.append(np.prod(feature.shape))  # Flattened size
                    feature_types.append(str(feature.dtype))
                    valid_features += 1
                else:
                    invalid_features += 1
                    logging.warning(f"Sample {key}: Feature is None")
            else:
                invalid_features += 1
        
        if dimensions:
            unique_dims = list(set(dimensions))
            unique_types = list(set(feature_types))
            
            logging.info(f"Feature dimensions found: {unique_dims}")
            logging.info(f"Expected dimension: {EXPECTED_DIMENSIONS.get(feature_type, 'Unknown')}")
            logging.info(f"Data types: {unique_types}")
            logging.info(f"Valid features: {valid_features}")
            logging.info(f"Invalid features: {invalid_features}")
            
            # Prüfe ob Dimensionen konsistent sind
            if len(unique_dims) == 1:
                actual_dim = unique_dims[0]
                expected_dim = EXPECTED_DIMENSIONS.get(feature_type)
                
                if expected_dim and actual_dim == expected_dim:
                    logging.info("✓ Dimensions match expected values!")
                    status = "CORRECT"
                elif expected_dim:
                    logging.warning(f"⚠ Dimension mismatch! Expected {expected_dim}, got {actual_dim}")
                    status = "DIMENSION_MISMATCH"
                else:
                    logging.info("? Unknown expected dimension")
                    status = "UNKNOWN"
            else:
                logging.error(f"✗ Inconsistent dimensions found: {unique_dims}")
                status = "INCONSISTENT"
        else:
            logging.error("No valid feature dimensions found!")
            status = "NO_VALID_FEATURES"
        
        # Analysiere Feature-Werte
        if valid_features > 0:
            sample_feature = None
            for key in sample_keys:
                if features[key] is not None:
                    sample_feature = features[key]
                    break
            
            if sample_feature is not None:
                if feature_type == 'orb' and len(sample_feature.shape) > 1:
                    # ORB kann mehrere Deskriptoren haben
                    logging.info(f"ORB descriptors shape: {sample_feature.shape}")
                    feature_flat = sample_feature.flatten()
                else:
                    feature_flat = sample_feature.flatten()
                
                logging.info(f"Feature value range: [{np.min(feature_flat):.6f}, {np.max(feature_flat):.6f}]")
                logging.info(f"Feature mean: {np.mean(feature_flat):.6f}")
                logging.info(f"Feature std: {np.std(feature_flat):.6f}")
                
                # Prüfe auf verdächtige Werte
                if np.any(np.isnan(feature_flat)):
                    logging.warning("⚠ NaN values found in features!")
                if np.any(np.isinf(feature_flat)):
                    logging.warning("⚠ Infinite values found in features!")
                if np.all(feature_flat == 0):
                    logging.warning("⚠ All-zero features found!")
        
        return {
            'feature_type': feature_type,
            'total_features': len(features),
            'valid_features': valid_features,
            'invalid_features': invalid_features,
            'dimensions': unique_dims if 'unique_dims' in locals() else [],
            'expected_dimension': EXPECTED_DIMENSIONS.get(feature_type),
            'status': status,
            'file_size_mb': os.path.getsize(filepath) / 1024 / 1024
        }
        
    except Exception as e:
        logging.error(f"Error loading {feature_type} features: {e}")
        return None

def check_feature_consistency():
    """Prüfe Konsistenz zwischen verschiedenen Feature-Typen."""
    logging.info("\n=== CROSS-FEATURE CONSISTENCY CHECK ===")
    
    all_image_ids = {}
    
    for feature_type in FEATURE_FILES.keys():
        filepath = os.path.join(PICKLE_PATH, FEATURE_FILES[feature_type])
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    features = pickle.load(f)
                all_image_ids[feature_type] = set(features.keys())
                logging.info(f"{feature_type}: {len(features)} image IDs")
            except Exception as e:
                logging.error(f"Error loading {feature_type}: {e}")
    
    if len(all_image_ids) > 1:
        # Finde gemeinsame Image IDs
        common_ids = set.intersection(*all_image_ids.values())
        logging.info(f"Common image IDs across all features: {len(common_ids)}")
        
        # Finde fehlende IDs pro Feature-Typ
        for feature_type, ids in all_image_ids.items():
            missing = len(ids) - len(common_ids)
            if missing > 0:
                logging.warning(f"{feature_type}: {missing} images missing from common set")
    
    return all_image_ids

def create_validation_report():
    """Erstelle einen vollständigen Validierungsbericht."""
    logging.info("=== FEATURE VALIDATION REPORT ===")
    logging.info(f"Pickle directory: {PICKLE_PATH}")
    
    results = {}
    
    # Analysiere jede Feature-Datei
    for feature_type in FEATURE_FILES.keys():
        result = load_and_analyze_features(feature_type)
        if result:
            results[feature_type] = result
    
    # Prüfe Konsistenz
    consistency_info = check_feature_consistency()
    
    # Erstelle Zusammenfassung
    logging.info("\n=== SUMMARY ===")
    
    total_correct = 0
    total_files = len(FEATURE_FILES)
    
    for feature_type, result in results.items():
        status_symbol = "✓" if result['status'] == "CORRECT" else "✗"
        logging.info(f"{status_symbol} {feature_type}: {result['status']} "
                   f"({result['valid_features']} valid features, {result['file_size_mb']:.1f} MB)")
        
        if result['status'] == "CORRECT":
            total_correct += 1
    
    logging.info(f"\nOverall: {total_correct}/{total_files} feature types are correct")
    
    if total_correct == total_files:
        logging.info("🎉 All features are correctly extracted!")
    else:
        logging.warning("⚠ Some features need attention")
    
    return results

def plot_feature_distributions(sample_size=100):
    """Plotte Verteilungen der Features zur visuellen Inspektion."""
    logging.info("\n=== CREATING FEATURE DISTRIBUTION PLOTS ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature_type in enumerate(['efficientnet', 'hsv', 'lbp', 'orb']):
        filepath = os.path.join(PICKLE_PATH, FEATURE_FILES[feature_type])
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    features = pickle.load(f)
                
                # Sample features für Plotting
                sample_keys = list(features.keys())[:sample_size]
                sample_values = []
                
                for key in sample_keys:
                    feature = features[key]
                    if feature is not None:
                        if feature_type == 'orb' and len(feature.shape) > 1:
                            sample_values.extend(feature.flatten())
                        else:
                            sample_values.extend(feature.flatten())
                
                if sample_values:
                    axes[i].hist(sample_values, bins=50, alpha=0.7)
                    axes[i].set_title(f'{feature_type.title()} Distribution')
                    axes[i].set_xlabel('Feature Value')
                    axes[i].set_ylabel('Frequency')
                else:
                    axes[i].text(0.5, 0.5, 'No valid features', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{feature_type.title()} - No Data')
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {str(e)}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{feature_type.title()} - Error')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logging.info("Feature distribution plots saved as 'feature_distributions.png'")

def quick_feature_check():
    """Schnelle Überprüfung ohne detaillierte Analyse."""
    logging.info("=== QUICK FEATURE CHECK ===")
    
    for feature_type, filename in FEATURE_FILES.items():
        filepath = os.path.join(PICKLE_PATH, filename)
        
        if os.path.exists(filepath):
            try:
                file_size_mb = os.path.getsize(filepath) / 1024 / 1024
                
                with open(filepath, 'rb') as f:
                    features = pickle.load(f)
                
                # Quick checks
                num_features = len(features)
                has_valid_features = any(v is not None for v in list(features.values())[:10])
                
                status = "✓" if has_valid_features and num_features > 0 else "✗"
                
                logging.info(f"{status} {feature_type}: {num_features} features, {file_size_mb:.1f} MB")
                
            except Exception as e:
                logging.error(f"✗ {feature_type}: Error - {e}")
        else:
            logging.error(f"✗ {feature_type}: File not found")

if __name__ == "__main__":
    print("Feature Validation Tool")
    print("=" * 50)
    
    # Wähle Validierungsmodus
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_feature_check()
    else:
        # Vollständige Validierung
        results = create_validation_report()
        
        # Erstelle Plots (optional)
        try:
            plot_feature_distributions()
        except Exception as e:
            logging.warning(f"Could not create plots: {e}")
        
        print("\n" + "=" * 50)
        print("Validation completed! Check the logs above for details.")
