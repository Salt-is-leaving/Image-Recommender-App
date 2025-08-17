import os
import sys
import logging
import time
import argparse
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dynamic_weights_optimizer import print_weight_analysis_report, get_optimal_similarity_weights
from config import (PATH_TO_SSD, PICKLE_PATH, DB_PATH, FEATURE_FILES, 
                    get_global_cache, get_clustering_status)
from db_api import create_connection, create_tables, get_feature_completeness
from feature_extraction_pipeline import learning_mode
from clustering import run_clustering
from interactive_pipeline import run_interactive_mode
from feature_visualization import run_feature_visualization


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def check_system_requirements():
    """Check system requirements and paths."""
    logging.info("Checking system requirements...")
    
    # Check image directory
    if not os.path.exists(PATH_TO_SSD):
        logging.error(f"Image directory not found: {PATH_TO_SSD}")
        logging.error("Please update PATH_TO_SSD in config.py")
        return False
    
    # Check database connection
    try:
        conn = create_connection(DB_PATH)
        if conn is None:
            logging.error("Cannot connect to database")
            return False
        
        create_tables(conn)
        conn.close()
        logging.info(" Database connection successful")
    except Exception as e:
        logging.error(f"Database error: {e}")
        return False
    
    # Check clustering prerequisites
    cluster_status = get_clustering_status()
    if cluster_status['clustering_available']:
        logging.info("Clustering data found")
    else:
        logging.info("Clustering not available - run clustering mode for better performance")
    
    logging.info(" System requirements check passed")
    return True

def show_system_status():
    """Show comprehensive system status."""
    logging.info("=== SYSTEM STATUS ===")
    
    # Check image directory
    if os.path.exists(PATH_TO_SSD):
        image_count = sum(1 for root, dirs, files in os.walk(PATH_TO_SSD) 
                         for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png')))
        logging.info(f"Image directory: {PATH_TO_SSD}")
        logging.info(f"Images found: {image_count}")
    else:
        logging.info(f"Image directory not found: {PATH_TO_SSD}")
    
    # Check database with corrected function call
    try:
        conn = create_connection()
        if conn:
            stats = get_feature_completeness(conn)
            if stats and stats['all']:  #  use 'all' key
                overall = stats['all']
                logging.info(f"Database status:")
                logging.info(f"  Total images: {overall[0]}")
                logging.info(f"  HSV features: {overall[1]}")
                logging.info(f"  ConvNeXt features: {overall[2]}")
                logging.info(f"  Complete features: {overall[3]}")
            conn.close()
    except Exception as e:
        logging.info(f"Database connection failed: {e}")
    
    # Check pickle files
    logging.info("Feature files:")
    for feature_type, filename in FEATURE_FILES.items():
        filepath = os.path.join(PICKLE_PATH, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            logging.info(f"  {filename}: {size_mb:.2f} MB")
        else:
            logging.info(f"  {filename}: Not found")
    
    # Check clustering
    cluster_file = os.path.join(PICKLE_PATH, 'cluster_data.pkl')
    if os.path.exists(cluster_file):
        size_mb = os.path.getsize(cluster_file) / 1024 / 1024
        logging.info(f"Clustering: {size_mb:.2f} MB")
        logging.info("Clustering-first search available!")
        
        # Try to load clustering stats
        try:
            import pickle
            with open(cluster_file, 'rb') as f:
                cluster_data = pickle.load(f)
                
            if 'cluster_assignments' in cluster_data:
                logging.info("Clustering statistics:")
                for ft, assignments in cluster_data['cluster_assignments'].items():
                    n_clusters = len(set(assignments.values()))
                    logging.info(f"     {ft}: {len(assignments)} images, {n_clusters} clusters")
        except:
            pass
    else:
        logging.info("Clustering: Not available")
        logging.info("Run: python main.py --mode clustering")
    
    # Check embedding cache
    cache_file = os.path.join(PICKLE_PATH, 'new_image_embeddings.pkl')
    if os.path.exists(cache_file):
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            logging.info(f"Embedding cache: {len(cache)} cached embeddings")
        except:
            logging.info("Embedding cache: Found but corrupted")
    else:
        logging.info("Embedding cache: Empty")

def run_weight_analysis_mode(args):
    """Run comprehensive weight analysis and optimization."""
    logging.info("=== STARTING WEIGHT ANALYSIS MODE ===")
    
    try:
        # Force recalculation if requested
        force_recalc = args.recalculate_weights
        
        if force_recalc:
            logging.info("Forcing weight recalculation...")
        
        # Run comprehensive analysis
        print_weight_analysis_report()
        
        # Get and display current optimal weights
        optimal_weights = get_optimal_similarity_weights(force_recalculate=force_recalc)
        
        print(f"\n{'='*50}")
        print("CURRENT OPTIMAL WEIGHTS")
        print(f"{'='*50}")
        print(f"ConvNeXt: {optimal_weights['convnext']:.3f}")
        print(f"HSV:      {optimal_weights['hsv']:.3f}")
        print(f"{'='*50}")
        
        # Provide guidance
        conv_weight = optimal_weights['convnext']
        if conv_weight > 0.8:
            print("Analysis: ConvNeXt heavily favored")
            print("Your ConvNeXt features are high quality")
            print("Consider using 'semantic_focused' search for best results")
        elif conv_weight < 0.4:
            print("Analysis: HSV heavily favored") 
            print("Your ConvNeXt features may need improvement")
            print("Consider re-running learning mode with enhanced preprocessing")
        else:
            print("Analysis: Balanced feature contribution")
            print("Both feature types provide value")
            print("Current weights should work well")
        
        return True
        
    except Exception as e:
        logging.error(f"Weight analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_learning_mode(args):
    """Run feature extraction learning mode."""
    logging.info("=== STARTING LEARNING MODE ===")
    
    # Check image directory
    if not os.path.exists(PATH_TO_SSD):
        logging.error(f"Image directory not found: {PATH_TO_SSD}")
        return False
    
    # Run learning
    try:
        success = learning_mode(
            image_directory=PATH_TO_SSD,
            use_cuda=args.cuda,
            batch_size=args.batch_size
        )
        
        if success:
            logging.info("Learning mode completed successfully!")
            
            # Show feature extraction results
            logging.info("Feature extraction results:")
            for feature_type, filename in FEATURE_FILES.items():
                filepath = os.path.join(PICKLE_PATH, filename)
                if os.path.exists(filepath):
                    size_mb = os.path.getsize(filepath) / 1024 / 1024
                    logging.info(f"   {feature_type}: {size_mb:.2f} MB")
            
            logging.info("Next step: python main.py --mode clustering")
            return True
        else:
            logging.error("Learning mode failed!")
            return False
            
    except Exception as e:
        logging.error(f"Learning mode error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_clustering_mode(args):
    """Run clustering mode to create optimized feature clusters."""
    logging.info("=== STARTING CLUSTERING MODE ===")
    
    # Check if features exist
    required_files = ['convnext_features.pkl', 'hsv_features.pkl']
    missing_files = []
    
    for filename in required_files:
        filepath = os.path.join(PICKLE_PATH, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if missing_files:
        logging.error(f"Missing required feature files: {missing_files}")
        logging.error("Please run learning mode first to extract features")
        logging.info("Usage: python main.py --mode learning")
        return False
    
    logging.info("All required feature files found")
    
    # Check feature file sizes and warn about potential issues
    for filename in required_files:
        filepath = os.path.join(PICKLE_PATH, filename)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        logging.info(f"  {filename}: {size_mb:.2f} MB")
        
        # Warn about oversized HSV features
        if filename == 'hsv_features.pkl' and size_mb > 100:
            logging.warning(f"HSV features are large ({size_mb:.2f} MB)")
            logging.warning("Over-dimensioning will be fixed by clustering")
    
    # Run clustering
    logging.info("Starting clustering with preprocessing...")
    
    try:
        success = run_clustering()
        
        if success:
            logging.info("Clustering completed successfully!")
            logging.info("Clustering-first similarity search is now available")
            
            # Show clustering results summary
            cluster_file = os.path.join(PICKLE_PATH, 'cluster_data.pkl')
            if os.path.exists(cluster_file):
                size_mb = os.path.getsize(cluster_file) / 1024 / 1024
                logging.info(f"Clustering data saved: {size_mb:.2f} MB")
                
                # Load and show cluster statistics
                try:
                    import pickle
                    with open(cluster_file, 'rb') as f:
                        cluster_data = pickle.load(f)
                    
                    if 'cluster_assignments' in cluster_data:
                        logging.info("Clustering statistics:")
                        for ft, assignments in cluster_data['cluster_assignments'].items():
                            n_clusters = len(set(assignments.values()))
                            logging.info(f"   {ft}: {len(assignments)} images in {n_clusters} clusters")
                except:
                    pass
                
                logging.info(" You can now use similarity search:")
                logging.info("- python main.py --mode interactive")
            
            return True
        else:
            logging.error("Clustering failed!")
            return False
            
    except Exception as e:
        logging.error(f"Clustering mode failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_interactive_mode_wrapper(args):
    """Run interactive similarity search mode."""
    logging.info("=== STARTING INTERACTIVE MODE ===")
    
    # Check clustering availability
    cluster_status = get_clustering_status()
    if cluster_status['clustering_available']:
        logging.info("Clustering-first search available")
    else:
        logging.warning("Clustering not available")
        return False
        
    try:
        success = run_interactive_mode(
            image_path=args.image_path,
            use_cuda=args.cuda and not args.no_gpu
        )
        
        if success:
            logging.info(" Interactive mode completed successfully!")
            return True
        else:
            logging.error(" Interactive mode failed!")
            return False
            
    except Exception as e:
        logging.error(f"Interactive mode error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_visualization_mode(args):
    """Run feature space visualization."""
    logging.info("=== STARTING FEATURE VISUALIZATION ===")
    
    try:
        success = run_feature_visualization(max_samples=2000)
        
        if success:
            logging.info("Feature visualization completed!")
            return True
        else:
            logging.error("Feature visualization failed!")
            return False
    except Exception as e:
        logging.error(f"Visualization error: {e}")
        return False
    
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Image Similarity Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  # Compute embeddings for images in dataset
  python main.py --mode learning
  
  # Create clusters for a quicker similarity search. Requires features from learning mode.
  python main.py --mode clustering
  
  # Interactive mode - GUI to select any random image for similarity search
  python main.py --mode interactive
  
  # Interactive mode - specific image
  python main.py --mode interactive --image-path "path/to/image.jpg"
  
  # Check system status and clustering availability
  python main.py --status
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['learning', 'interactive', 'clustering', 'weights', 'vis'],
        help='Operation mode to run'
    )
    
    parser.add_argument(
    '--analyze-weights',
    action='store_true',
    help='Analyze feature quality and show optimal weights'
    )

    parser.add_argument(
    '--recalculate-weights',
    action='store_true',
    help='Force recalculation of optimal weights (ignore cache)'
    )

    parser.add_argument(
        '--target-image',
        type=str,
        help='Target image ID for interactive mode'
    )
    
    parser.add_argument(
        '--image-path',
        type=str,
        help='Path to image for interactive mode'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for feature extraction (default: 30)'
    )
    
    parser.add_argument(
        '--cuda',
        action='store_true',
        default=True,
        help='Use CUDA for feature extraction (default: True)'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage for search (prevents GPU memory issues)'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status and exit'
    )
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    if args.analyze_weights or args.mode == 'weights':
        success = run_weight_analysis_mode(args)
        return 0 if success else 1
        
    # Show status and exit if requested
    if args.status:
        show_system_status()
        return 0
    
    # Require mode to be specified
    if args.mode is None:
        print("Error: --mode is required")
        print("\nQuick start:")
        print("1. python main.py --mode learning     # Extract features first")
        print("2. python main.py --mode clustering   # Create clusters")
        print("3. python main.py --mode interactive  # Search images")
        print("\nUse --help for more options")
        return 1
    
    # Check system requirements
    if not check_system_requirements():
        logging.error("System requirements check failed")
        return 1
    success = True
    
    # Learning phase
    if args.mode == 'learning':
        success &= run_learning_mode(args)
    
    # Clustering phase
    elif args.mode == 'clustering':
        success &= run_clustering_mode(args)
    
    # Interactive phase
    elif args.mode == 'interactive':
        success &= run_interactive_mode_wrapper(args)

    # Feture space visualization for feature analysis
    elif args.mode == 'vis':
        success &= run_visualization_mode(args)

    # Final status and guidance
    if success:
        logging.info("=== ALL OPERATIONS COMPLETED SUCCESSFULLY ===")
        
        # Provide specific guidance based on mode
        if args.mode == 'clustering':
            logging.info("Clustering completed successfully!")
            logging.info("Now you can use similarity search:")
            logging.info("python main.py --mode interactive")
                      
        elif args.mode == 'learning':
            logging.info("Feature extraction completed!")

        elif args.mode == 'interactive':
            logging.info("Interactive similarity search completed!")
            
        return 0
    else:
        logging.error("=== SOME OPERATIONS FAILED ===")
        logging.info("Please check the logs for details.")
        return 1      

if __name__ == "__main__":
    sys.exit(main())