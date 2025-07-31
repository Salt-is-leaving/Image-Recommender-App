import os
import numpy as np
import pickle
import logging
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time

from config import PICKLE_PATH
from db_api import load_features_from_pickle

logging.basicConfig(level=logging.INFO)

class FeatureQualityAnalyzer:
    """Analyze feature quality to determine optimal weights automatically."""
    
    def __init__(self):
        self.features_data = {}
        self.quality_metrics = {}
        self.optimal_weights = None
        self.analysis_cache_path = os.path.join(PICKLE_PATH, 'feature_quality_analysis.pkl')
        
    def load_features(self):
        """Load all available features."""
        logging.info("Loading features for quality analysis...")
        
        for feature_type in ['convnext', 'hsv']:
            features = load_features_from_pickle(feature_type)
            if features:
                self.features_data[feature_type] = features
                logging.info(f"Loaded {len(features)} {feature_type} features")
            else:
                logging.warning(f"No {feature_type} features found")
        
        return len(self.features_data) > 0
    
    def analyze_convnext_quality(self, sample_size=2000):
        """Analyze ConvNeXt feature quality comprehensively."""
        if 'convnext' not in self.features_data:
            return {'quality_score': 0.0, 'issues': ['No ConvNeXt features']}
        
        features = self.features_data['convnext']
        sample_ids = list(features.keys())[:sample_size]
        
        # Convert to matrix for analysis
        feature_matrix = []
        valid_ids = []
        
        for img_id in sample_ids:
            feat = features[img_id]
            if feat is not None and len(feat) > 0:
                feature_matrix.append(feat.flatten())
                valid_ids.append(img_id)
        
        if not feature_matrix:
            return {'quality_score': 0.0, 'issues': ['No valid ConvNeXt features']}
        
        feature_matrix = np.array(feature_matrix)
        
        metrics = {}
        issues = []
        
        # 1. Norm analysis
        norms = np.linalg.norm(feature_matrix, axis=1)
        norm_std = np.std(norms)
        norm_mean = np.mean(norms)
        
        if norm_std < 1e-6:
            issues.append("All features have identical norms (over-normalized)")
            norm_quality = 0.0
        elif norm_std < 0.01:
            issues.append("Very low norm variance")
            norm_quality = 0.3
        else:
            norm_quality = min(1.0, norm_std * 10)  # Scale appropriately
        
        metrics['norm_variance'] = norm_std
        metrics['norm_mean'] = norm_mean
        metrics['norm_quality'] = norm_quality
        
        # 2. Feature diversity analysis
        feature_stds = np.std(feature_matrix, axis=0)
        avg_feature_std = np.mean(feature_stds)
        
        if avg_feature_std < 1e-6:
            issues.append("Features lack diversity across dimensions")
            diversity_quality = 0.0
        elif avg_feature_std < 0.01:
            issues.append("Low feature diversity")
            diversity_quality = 0.4
        else:
            diversity_quality = min(1.0, avg_feature_std * 100)
        
        metrics['diversity'] = avg_feature_std
        metrics['diversity_quality'] = diversity_quality
        
        # 3. Similarity distribution analysis
        if len(feature_matrix) > 100:
            # Sample pairs for similarity analysis
            n_pairs = min(1000, len(feature_matrix) * (len(feature_matrix) - 1) // 2)
            similarities = []
            
            indices = np.random.choice(len(feature_matrix), size=min(100, len(feature_matrix)), replace=False)
            sample_matrix = feature_matrix[indices]
            
            sim_matrix = cosine_similarity(sample_matrix)
            
            # Get upper triangle (excluding diagonal)
            upper_tri = np.triu(sim_matrix, k=1)
            similarities = upper_tri[upper_tri != 0]
            
            sim_mean = np.mean(similarities)
            sim_std = np.std(similarities)
            
            # Good similarity distribution should have reasonable spread
            if sim_std < 0.05:
                issues.append("Similarities too concentrated (low discriminative power)")
                sim_quality = 0.2
            elif sim_std < 0.1:
                issues.append("Limited similarity variance")
                sim_quality = 0.6
            else:
                sim_quality = min(1.0, sim_std * 5)
            
            metrics['similarity_mean'] = sim_mean
            metrics['similarity_std'] = sim_std
            metrics['similarity_quality'] = sim_quality
        else:
            metrics['similarity_quality'] = 0.5  # Neutral for small samples
        
        # 4. Range analysis
        feature_ranges = np.max(feature_matrix, axis=0) - np.min(feature_matrix, axis=0)
        avg_range = np.mean(feature_ranges)
        
        if avg_range < 1e-6:
            issues.append("Features have very small value ranges")
            range_quality = 0.0
        elif avg_range < 0.1:
            issues.append("Limited feature value ranges")
            range_quality = 0.4
        else:
            range_quality = min(1.0, avg_range * 2)
        
        metrics['range'] = avg_range
        metrics['range_quality'] = range_quality
        
        # 5. Overall quality score
        quality_components = [
            metrics['norm_quality'],
            metrics['diversity_quality'],
            metrics.get('similarity_quality', 0.5),
            metrics['range_quality']
        ]
        
        overall_quality = np.mean(quality_components)
        
        return {
            'quality_score': overall_quality,
            'metrics': metrics,
            'issues': issues,
            'sample_size': len(valid_ids)
        }
    
    def analyze_hsv_quality(self, sample_size=2000):
        """Analyze HSV feature quality and discriminative power."""
        if 'hsv' not in self.features_data:
            return {'quality_score': 0.0, 'issues': ['No HSV features']}
        
        features = self.features_data['hsv']
        sample_ids = list(features.keys())[:sample_size]
        
        # Convert to matrix
        feature_matrix = []
        valid_ids = []
        
        for img_id in sample_ids:
            feat = features[img_id]
            if feat is not None and len(feat) > 0:
                feature_matrix.append(feat.flatten())
                valid_ids.append(img_id)
        
        if not feature_matrix:
            return {'quality_score': 0.0, 'issues': ['No valid HSV features']}
        
        feature_matrix = np.array(feature_matrix)
        
        metrics = {}
        issues = []
        
        # 1. Histogram properties analysis
        entropies = []
        sparsities = []
        
        for hist in feature_matrix:
            # Entropy (information content)
            entropy = -np.sum(hist * np.log(hist + 1e-12))
            entropies.append(entropy)
            
            # Sparsity (how many bins are effectively used)
            non_zero_bins = np.sum(hist > 1e-6)
            sparsity = non_zero_bins / len(hist)
            sparsities.append(sparsity)
        
        avg_entropy = np.mean(entropies)
        avg_sparsity = np.mean(sparsities)
        
        # Good histograms should have reasonable entropy and not be too sparse
        if avg_entropy < 2.0:
            issues.append("Low histogram entropy (limited color diversity)")
            entropy_quality = avg_entropy / 5.0  # Scale to 0-1
        else:
            entropy_quality = min(1.0, avg_entropy / 8.0)
        
        if avg_sparsity < 0.1:
            issues.append("Very sparse histograms (most bins empty)")
            sparsity_quality = avg_sparsity * 5
        elif avg_sparsity > 0.9:
            issues.append("Histograms too dense (may lack discriminative power)")
            sparsity_quality = 0.7
        else:
            sparsity_quality = 1.0
        
        metrics['entropy'] = avg_entropy
        metrics['sparsity'] = avg_sparsity
        metrics['entropy_quality'] = entropy_quality
        metrics['sparsity_quality'] = sparsity_quality
        
        # 2. Discriminative power analysis
        if len(feature_matrix) > 50:
            # Compute Bhattacharyya similarities
            similarities = []
            n_comparisons = min(500, len(feature_matrix) * (len(feature_matrix) - 1) // 2)
            
            indices = np.random.choice(len(feature_matrix), size=min(50, len(feature_matrix)), replace=False)
            
            for i in range(len(indices)):
                for j in range(i + 1, min(i + 20, len(indices))):  # Limit comparisons
                    hist1 = feature_matrix[indices[i]]
                    hist2 = feature_matrix[indices[j]]
                    
                    # Bhattacharyya coefficient
                    sim = np.sum(np.sqrt(hist1 * hist2))
                    similarities.append(sim)
            
            if similarities:
                sim_mean = np.mean(similarities)
                sim_std = np.std(similarities)
                
                # Good HSV features should have good separation
                if sim_std < 0.05:
                    issues.append("HSV similarities too concentrated")
                    discriminative_quality = 0.3
                elif sim_mean > 0.8:
                    issues.append("HSV features too similar across images")
                    discriminative_quality = 0.4
                else:
                    discriminative_quality = min(1.0, sim_std * 10)
                
                metrics['hsv_similarity_mean'] = sim_mean
                metrics['hsv_similarity_std'] = sim_std
                metrics['discriminative_quality'] = discriminative_quality
            else:
                metrics['discriminative_quality'] = 0.5
        else:
            metrics['discriminative_quality'] = 0.5
        
        # 3. Overall HSV quality
        quality_components = [
            metrics['entropy_quality'],
            metrics['sparsity_quality'],
            metrics['discriminative_quality']
        ]
        
        overall_quality = np.mean(quality_components)
        
        return {
            'quality_score': overall_quality,
            'metrics': metrics,
            'issues': issues,
            'sample_size': len(valid_ids)
        }
    
    def analyze_feature_correlation(self, sample_size=1000):
        """Analyze correlation between feature types to avoid redundancy."""
        if len(self.features_data) < 2:
            return {'correlation': 0.0, 'redundancy_issues': []}
        
        # Get common image IDs
        common_ids = set(self.features_data['convnext'].keys())
        for feature_type in self.features_data:
            common_ids = common_ids.intersection(set(self.features_data[feature_type].keys()))
        
        common_ids = list(common_ids)[:sample_size]
        
        if len(common_ids) < 10:
            return {'correlation': 0.0, 'redundancy_issues': ['Insufficient common images']}
        
        # Extract features for common images
        convnext_features = []
        hsv_features = []
        
        for img_id in common_ids:
            conv_feat = self.features_data['convnext'][img_id]
            hsv_feat = self.features_data['hsv'][img_id]
            
            if conv_feat is not None and hsv_feat is not None:
                convnext_features.append(conv_feat.flatten())
                hsv_features.append(hsv_feat.flatten())
        
        if len(convnext_features) < 10:
            return {'correlation': 0.0, 'redundancy_issues': ['Insufficient valid feature pairs']}
        
        # Compute cross-feature correlations (simplified)
        # Compare similarity rankings rather than raw features
        conv_matrix = np.array(convnext_features)
        hsv_matrix = np.array(hsv_features)
        
        # Sample pairs for ranking correlation
        n_samples = min(50, len(conv_matrix))
        indices = np.random.choice(len(conv_matrix), n_samples, replace=False)
        
        ranking_correlations = []
        
        for i in range(min(10, n_samples)):  # Test with 10 query images
            query_idx = indices[i]
            
            # Get ConvNeXt similarities
            conv_query = conv_matrix[query_idx]
            conv_sims = cosine_similarity([conv_query], conv_matrix)[0]
            conv_ranking = np.argsort(conv_sims)[::-1]
            
            # Get HSV similarities
            hsv_query = hsv_matrix[query_idx]
            hsv_sims = []
            for hsv_feat in hsv_matrix:
                # Bhattacharyya similarity
                sim = np.sum(np.sqrt(hsv_query * hsv_feat))
                hsv_sims.append(sim)
            hsv_sims = np.array(hsv_sims)
            hsv_ranking = np.argsort(hsv_sims)[::-1]
            
            # Compute ranking correlation (Spearman-like)
            rank_correlation = np.corrcoef(conv_ranking, hsv_ranking)[0, 1]
            if not np.isnan(rank_correlation):
                ranking_correlations.append(abs(rank_correlation))
        
        avg_correlation = np.mean(ranking_correlations) if ranking_correlations else 0.0
        
        redundancy_issues = []
        if avg_correlation > 0.8:
            redundancy_issues.append("High correlation between features (potential redundancy)")
        elif avg_correlation > 0.6:
            redundancy_issues.append("Moderate correlation between features")
        
        return {
            'correlation': avg_correlation,
            'redundancy_issues': redundancy_issues,
            'sample_correlations': ranking_correlations
        }
    
    def calculate_optimal_weights(self):
        """Calculate optimal weights based on feature quality analysis."""
        logging.info("Calculating optimal feature weights...")
        
        # Analyze each feature type
        convnext_analysis = self.analyze_convnext_quality()
        hsv_analysis = self.analyze_hsv_quality()
        correlation_analysis = self.analyze_feature_correlation()
        
        # Store quality metrics
        self.quality_metrics = {
            'convnext': convnext_analysis,
            'hsv': hsv_analysis,
            'correlation': correlation_analysis,
            'timestamp': time.time()
        }
        
        # Calculate weights based on quality scores
        conv_quality = convnext_analysis['quality_score']
        hsv_quality = hsv_analysis['quality_score']
        correlation = correlation_analysis['correlation']
        
        logging.info(f"ConvNeXt quality score: {conv_quality:.3f}")
        logging.info(f"HSV quality score: {hsv_quality:.3f}")
        logging.info(f"Feature correlation: {correlation:.3f}")
        
        # Base weights from quality scores
        total_quality = conv_quality + hsv_quality
        if total_quality > 0:
            base_conv_weight = conv_quality / total_quality
            base_hsv_weight = hsv_quality / total_quality
        else:
            # Fallback to default if no quality info
            base_conv_weight = 0.65
            base_hsv_weight = 0.35
        
        # Adjust for correlation (reduce weight of correlated features)
        if correlation > 0.6:
            # If features are correlated, give more weight to the higher quality one
            if conv_quality > hsv_quality:
                correlation_adjustment = 0.1 * correlation
                base_conv_weight += correlation_adjustment
                base_hsv_weight -= correlation_adjustment
            else:
                correlation_adjustment = 0.1 * correlation
                base_hsv_weight += correlation_adjustment
                base_conv_weight -= correlation_adjustment
        
        # Adjust for dataset size
        dataset_size = len(self.features_data.get('convnext', {}))
        if dataset_size > 100000:
            # For large datasets, semantic features (ConvNeXt) become more important
            base_conv_weight += 0.05
            base_hsv_weight -= 0.05
        
        # Ensure weights sum to 1 and are reasonable
        base_conv_weight = max(0.1, min(0.9, base_conv_weight))
        base_hsv_weight = 1.0 - base_conv_weight
        
        self.optimal_weights = {
            'convnext': base_conv_weight,
            'hsv': base_hsv_weight
        }
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        
        # Log results
        logging.info("=== OPTIMAL WEIGHT CALCULATION RESULTS ===")
        logging.info(f"Optimal weights: ConvNeXt={base_conv_weight:.3f}, HSV={base_hsv_weight:.3f}")
        
        if recommendations:
            logging.info("Recommendations:")
            for rec in recommendations:
                logging.info(f"  - {rec}")
        
        # Save analysis to cache
        self.save_analysis_cache()
        
        return self.optimal_weights
    
    def generate_recommendations(self):
        """Generate recommendations based on quality analysis."""
        recommendations = []
        
        conv_quality = self.quality_metrics['convnext']['quality_score']
        hsv_quality = self.quality_metrics['hsv']['quality_score']
        
        # ConvNeXt recommendations
        if conv_quality < 0.3:
            recommendations.append("ConvNeXt features have very low quality - consider re-extraction with improved preprocessing")
        elif conv_quality < 0.6:
            recommendations.append("ConvNeXt features have moderate quality - check normalization and preprocessing")
        
        # HSV recommendations
        if hsv_quality < 0.3:
            recommendations.append("HSV features have low discriminative power - consider adjusting bin sizes or preprocessing")
        elif hsv_quality < 0.6:
            recommendations.append("HSV features have moderate quality - consider fine-tuning histogram parameters")
        
        # Correlation recommendations
        correlation = self.quality_metrics['correlation']['correlation']
        if correlation > 0.7:
            recommendations.append("Features are highly correlated - consider using only the higher-quality feature type")
        
        # Weight recommendations
        conv_weight = self.optimal_weights['convnext']
        if conv_weight > 0.8:
            recommendations.append("ConvNeXt heavily favored - ensure HSV features are providing value")
        elif conv_weight < 0.4:
            recommendations.append("HSV heavily favored - check ConvNeXt feature quality")
        
        return recommendations
    
    def save_analysis_cache(self):
        """Save analysis results to cache for future use."""
        cache_data = {
            'quality_metrics': self.quality_metrics,
            'optimal_weights': self.optimal_weights,
            'timestamp': time.time()
        }
        
        try:
            with open(self.analysis_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logging.info(f"Analysis cached to {self.analysis_cache_path}")
        except Exception as e:
            logging.error(f"Failed to save analysis cache: {e}")
    
    def load_analysis_cache(self, max_age_hours=24):
        """Load analysis from cache if recent enough."""
        if not os.path.exists(self.analysis_cache_path):
            return False
        
        try:
            with open(self.analysis_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if cache is recent enough
            cache_age = (time.time() - cache_data['timestamp']) / 3600
            if cache_age > max_age_hours:
                logging.info(f"Analysis cache is {cache_age:.1f} hours old - regenerating")
                return False
            
            self.quality_metrics = cache_data['quality_metrics']
            self.optimal_weights = cache_data['optimal_weights']
            
            logging.info(f"Loaded analysis from cache (age: {cache_age:.1f} hours)")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load analysis cache: {e}")
            return False

def get_optimal_similarity_weights(force_recalculate=False):
    """
    Get optimal similarity weights based on feature quality analysis.
    This replaces all the static weight configurations.
    """
    analyzer = FeatureQualityAnalyzer()
    
    # Try to load from cache first
    if not force_recalculate and analyzer.load_analysis_cache():
        return analyzer.optimal_weights
    
    # Load features and calculate optimal weights
    if not analyzer.load_features():
        logging.error("Could not load features for weight optimization")
        # Return fallback weights
        return {'convnext': 0.65, 'hsv': 0.35}
    
    optimal_weights = analyzer.calculate_optimal_weights()
    return optimal_weights

def print_weight_analysis_report():
    """Print comprehensive weight analysis report."""
    analyzer = FeatureQualityAnalyzer()
    
    if not analyzer.load_features():
        print("❌ Could not load features for analysis")
        return
    
    # Calculate optimal weights
    optimal_weights = analyzer.calculate_optimal_weights()
    
    print(f"\n{'='*60}")
    print("DYNAMIC WEIGHT OPTIMIZATION REPORT")
    print(f"{'='*60}")
    
    # Quality scores
    conv_quality = analyzer.quality_metrics['convnext']['quality_score']
    hsv_quality = analyzer.quality_metrics['hsv']['quality_score']
    correlation = analyzer.quality_metrics['correlation']['correlation']
    
    print(f"\nFeature Quality Scores:")
    print(f"  ConvNeXt: {conv_quality:.3f}/1.000")
    print(f"  HSV:      {hsv_quality:.3f}/1.000")
    print(f"  Correlation: {correlation:.3f}")
    
    # Optimal weights
    print(f"\nOptimal Weights:")
    print(f"  ConvNeXt: {optimal_weights['convnext']:.3f}")
    print(f"  HSV:      {optimal_weights['hsv']:.3f}")
    
    # Issues
    conv_issues = analyzer.quality_metrics['convnext'].get('issues', [])
    hsv_issues = analyzer.quality_metrics['hsv'].get('issues', [])
    
    if conv_issues or hsv_issues:
        print(f"\nIssues Found:")
        if conv_issues:
            print(f"  ConvNeXt:")
            for issue in conv_issues:
                print(f"    - {issue}")
        if hsv_issues:
            print(f"  HSV:")
            for issue in hsv_issues:
                print(f"    - {issue}")
    
    # Recommendations
    recommendations = analyzer.generate_recommendations()
    if recommendations:
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    print("Running dynamic weight optimization analysis...")
    print_weight_analysis_report()
