import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import os
import logging
from collections import defaultdict
import seaborn as sns

from config import PICKLE_PATH, get_cluster_config
from db_api import load_features_from_pickle
from clustering import Clustering

class FeatureSpaceVisualizer:
    """Visualize HSV and ConvNeXt feature spaces using PCA."""
    
    def __init__(self):
        self.clustering = Clustering()
        self.hsv_features = {}
        self.convnext_features = {}
        self.common_images = []
        self.cluster_data_loaded = False
        
    def load_data(self, max_samples=2000):
        """Load features and clustering data."""
        print("Loading feature data...")
        
        # Load features
        self.hsv_features = load_features_from_pickle('hsv')
        self.convnext_features = load_features_from_pickle('convnext')
        
        if not self.hsv_features or not self.convnext_features:
            raise ValueError("Failed to load features. Run learning mode first.")
        
        # Find common images
        hsv_images = set(self.hsv_features.keys())
        convnext_images = set(self.convnext_features.keys())
        self.common_images = list(hsv_images.intersection(convnext_images))
        
        # Limit samples for visualization performance
        if len(self.common_images) > max_samples:
            import random
            random.seed(42)
            self.common_images = random.sample(self.common_images, max_samples)
        
        print(f"Found {len(self.common_images)} common images")
        
        # Try to load clustering data
        try:
            self.cluster_data_loaded = self.clustering.load_clustering_data()
            if self.cluster_data_loaded:
                print("Clustering data loaded - will show cluster assignments")
            else:
                print("No clustering data - will show feature distributions only")
        except:
            print("No clustering data available")
            
    def prepare_features_for_pca(self, feature_type):
        """Prepare features for PCA visualization."""
        if feature_type == 'hsv':
            features_dict = self.hsv_features
        else:
            features_dict = self.convnext_features
            
        # Extract features for common images
        feature_list = []
        valid_images = []
        
        for img_path in self.common_images:
            if img_path in features_dict and features_dict[img_path] is not None:
                feature = features_dict[img_path].flatten().astype(np.float32)
                feature_list.append(feature)
                valid_images.append(img_path)
        
        if not feature_list:
            raise ValueError(f"No valid {feature_type} features found")
            
        feature_matrix = np.array(feature_list)
        print(f"{feature_type.upper()} features: {feature_matrix.shape}")
        
        # Apply preprocessing similar to clustering pipeline
        if feature_type == 'hsv':
            # Standardize HSV features
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)
        else:
            # ConvNeXt features - no standardization to preserve norms
            pass
            
        return feature_matrix, valid_images
    
    def apply_pca_2d(self, features, feature_type):
        """Apply PCA to reduce to 2D for visualization."""
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features)
        
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"{feature_type.upper()} PCA explained variance: {explained_var:.3f}")
        
        return features_2d, pca
    
    def get_cluster_assignments(self, valid_images, feature_type):
        """Get cluster assignments for images if available."""
        if not self.cluster_data_loaded or feature_type not in self.clustering.cluster_assignments:
            return None
            
        assignments = []
        for img_path in valid_images:
            if img_path in self.clustering.cluster_assignments[feature_type]:
                assignments.append(self.clustering.cluster_assignments[feature_type][img_path])
            else:
                assignments.append(-1)  # Unknown cluster
                
        return np.array(assignments)
    
    def create_side_by_side_comparison(self, save_path=None):
        """Create side-by-side comparison of HSV and ConvNeXt feature spaces."""
        # Prepare features
        hsv_matrix, hsv_images = self.prepare_features_for_pca('hsv')
        convnext_matrix, convnext_images = self.prepare_features_for_pca('convnext')
        
        # Apply PCA
        hsv_2d, hsv_pca = self.apply_pca_2d(hsv_matrix, 'hsv')
        convnext_2d, convnext_pca = self.apply_pca_2d(convnext_matrix, 'convnext')
        
        # Get cluster assignments
        hsv_clusters = self.get_cluster_assignments(hsv_images, 'hsv')
        convnext_clusters = self.get_cluster_assignments(convnext_images, 'convnext')
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # HSV plot
        if hsv_clusters is not None:
            scatter1 = ax1.scatter(hsv_2d[:, 0], hsv_2d[:, 1], 
                                 c=hsv_clusters, cmap='tab20', alpha=0.6, s=20)
            ax1.set_title(f'HSV Feature Space (PCA)\n{len(set(hsv_clusters))} clusters')
        else:
            scatter1 = ax1.scatter(hsv_2d[:, 0], hsv_2d[:, 1], 
                                 c='blue', alpha=0.6, s=20)
            ax1.set_title('HSV Feature Space (PCA)')
            
        ax1.set_xlabel('First Principal Component')
        ax1.set_ylabel('Second Principal Component')
        ax1.grid(True, alpha=0.3)
        
        # ConvNeXt plot
        if convnext_clusters is not None:
            scatter2 = ax2.scatter(convnext_2d[:, 0], convnext_2d[:, 1], 
                                 c=convnext_clusters, cmap='tab20', alpha=0.6, s=20)
            ax2.set_title(f'ConvNeXt Feature Space (PCA)\n{len(set(convnext_clusters))} clusters')
        else:
            scatter2 = ax2.scatter(convnext_2d[:, 0], convnext_2d[:, 1], 
                                 c='red', alpha=0.6, s=20)
            ax2.set_title('ConvNeXt Feature Space (PCA)')
            
        ax2.set_xlabel('First Principal Component')
        ax2.set_ylabel('Second Principal Component')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
        
        return {
            'hsv_2d': hsv_2d,
            'convnext_2d': convnext_2d,
            'hsv_images': hsv_images,
            'convnext_images': convnext_images,
            'hsv_clusters': hsv_clusters,
            'convnext_clusters': convnext_clusters
        }
    
    def create_overlay_comparison(self, save_path=None):
        """Create overlay comparison showing both feature types on same plot."""
        # Prepare features
        hsv_matrix, hsv_images = self.prepare_features_for_pca('hsv')
        convnext_matrix, convnext_images = self.prepare_features_for_pca('convnext')
        
        # Apply PCA
        hsv_2d, _ = self.apply_pca_2d(hsv_matrix, 'hsv')
        convnext_2d, _ = self.apply_pca_2d(convnext_matrix, 'convnext')
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot both feature types
        plt.scatter(hsv_2d[:, 0], hsv_2d[:, 1], 
                   c='blue', alpha=0.6, s=30, label='HSV Features', marker='o')
        plt.scatter(convnext_2d[:, 0], convnext_2d[:, 1], 
                   c='red', alpha=0.6, s=30, label='ConvNeXt Features', marker='^')
        
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('HSV vs ConvNeXt Feature Spaces (PCA Projection)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved overlay visualization to {save_path}")
        
        plt.show()
    
    def create_combined_feature_analysis(self, save_path=None):
        """Analyze combined feature space."""
        print("\nCreating combined feature analysis...")
        
        # Get common images for both features
        common_imgs = []
        hsv_feats = []
        convnext_feats = []
        
        for img_path in self.common_images:
            if (img_path in self.hsv_features and img_path in self.convnext_features and
                self.hsv_features[img_path] is not None and self.convnext_features[img_path] is not None):
                
                hsv_feat = self.hsv_features[img_path].flatten().astype(np.float32)
                convnext_feat = self.convnext_features[img_path].flatten().astype(np.float32)
                
                # Apply same preprocessing as clustering
                # HSV: standardize
                hsv_feat = (hsv_feat - np.mean(hsv_feat)) / (np.std(hsv_feat) + 1e-8)
                
                common_imgs.append(img_path)
                hsv_feats.append(hsv_feat)
                convnext_feats.append(convnext_feat)
        
        if len(common_imgs) < 10:
            print("Not enough common images for combined analysis")
            return
            
        print(f"Analyzing {len(common_imgs)} images with both feature types")
        
        # Combine features
        combined_features = []
        for hsv_feat, convnext_feat in zip(hsv_feats, convnext_feats):
            # Concatenate features
            combined_feat = np.concatenate([hsv_feat, convnext_feat])
            combined_features.append(combined_feat)
        
        combined_matrix = np.array(combined_features)
        print(f"Combined feature matrix: {combined_matrix.shape}")
        
        # Apply PCA to combined features
        pca_combined = PCA(n_components=2, random_state=42)
        combined_2d = pca_combined.fit_transform(combined_matrix)
        
        explained_var = np.sum(pca_combined.explained_variance_ratio_)
        print(f"Combined PCA explained variance: {explained_var:.3f}")
        
        # Get cluster assignments if available
        clusters = None
        if self.cluster_data_loaded:
            # Use HSV clusters as representative
            clusters = self.get_cluster_assignments(common_imgs, 'hsv')
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        if clusters is not None:
            scatter = plt.scatter(combined_2d[:, 0], combined_2d[:, 1], 
                                c=clusters, cmap='tab20', alpha=0.7, s=40)
            plt.colorbar(scatter, label='Cluster ID')
            plt.title(f'Combined HSV+ConvNeXt Feature Space (PCA)\n{len(set(clusters))} clusters')
        else:
            plt.scatter(combined_2d[:, 0], combined_2d[:, 1], 
                       c='purple', alpha=0.7, s=40)
            plt.title('Combined HSV+ConvNeXt Feature Space (PCA)')
        
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved combined analysis to {save_path}")
        
        plt.show()
        
        return combined_2d, common_imgs, clusters
    
    def analyze_feature_correlation(self):
        """Analyze correlation between HSV and ConvNeXt feature spaces."""
        print("\nAnalyzing feature space correlation...")
        
        # Get PCA projections for common images
        results = self.create_side_by_side_comparison()
        
        hsv_2d = results['hsv_2d']
        convnext_2d = results['convnext_2d']
        
        # Calculate correlation between the 2D projections
        if len(hsv_2d) == len(convnext_2d):
            # Correlation between first principal components
            corr_pc1 = np.corrcoef(hsv_2d[:, 0], convnext_2d[:, 0])[0, 1]
            corr_pc2 = np.corrcoef(hsv_2d[:, 1], convnext_2d[:, 1])[0, 1]
            
            print(f"Correlation between HSV and ConvNeXt PC1: {corr_pc1:.3f}")
            print(f"Correlation between HSV and ConvNeXt PC2: {corr_pc2:.3f}")
            
            # Create correlation plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1.scatter(hsv_2d[:, 0], convnext_2d[:, 0], alpha=0.6, s=20)
            ax1.set_xlabel('HSV PC1')
            ax1.set_ylabel('ConvNeXt PC1')
            ax1.set_title(f'PC1 Correlation: {corr_pc1:.3f}')
            ax1.grid(True, alpha=0.3)
            
            ax2.scatter(hsv_2d[:, 1], convnext_2d[:, 1], alpha=0.6, s=20)
            ax2.set_xlabel('HSV PC2')
            ax2.set_ylabel('ConvNeXt PC2')
            ax2.set_title(f'PC2 Correlation: {corr_pc2:.3f}')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            return corr_pc1, corr_pc2
        
    def run_full_analysis(self, output_dir='feature_visualizations'):
        """Run complete feature space analysis."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("=== FEATURE SPACE VISUALIZATION ANALYSIS ===")
        
        # 1. Side-by-side comparison
        print("\n1. Creating side-by-side comparison...")
        self.create_side_by_side_comparison(
            save_path=os.path.join(output_dir, 'hsv_vs_convnext_comparison.png')
        )
        
        # 2. Overlay comparison
        print("\n2. Creating overlay comparison...")
        self.create_overlay_comparison(
            save_path=os.path.join(output_dir, 'feature_overlay.png')
        )
        
        # 3. Combined feature analysis
        print("\n3. Creating combined feature analysis...")
        self.create_combined_feature_analysis(
            save_path=os.path.join(output_dir, 'combined_features.png')
        )
        
        # 4. Correlation analysis
        print("\n4. Analyzing feature correlation...")
        self.analyze_feature_correlation()
        
        print(f"\nAll visualizations saved to: {output_dir}/")
        print("Analysis complete!")

def run_feature_visualization(max_samples=2000):
    """Main function to run feature visualization."""
    try:
        visualizer = FeatureSpaceVisualizer()
        visualizer.load_data(max_samples=max_samples)
        visualizer.run_full_analysis()
        return True
    except Exception as e:
        print(f"Error in feature visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running Feature Space Visualization...")
    success = run_feature_visualization()
    if success:
        print("Feature visualization completed successfully!")
    else:
        print("Feature visualization failed!")
