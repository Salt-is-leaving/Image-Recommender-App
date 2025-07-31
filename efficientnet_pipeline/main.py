
import os
import sys
import logging
import argparse
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (PATH_TO_SSD, PICKLE_PATH, DB_PATH, FEATURE_FILES, 
                   get_enhanced_clustering_status)
from db_api import create_connection, create_tables, get_feature_completeness
from feature_extraction_pipeline import learning_mode
from clustering import run_enhanced_clustering  # NEW IMPORT
from similarity_search_pipeline import comparison_mode
from interactive_pipeline import run_interactive_mode
from ranking_viz import run_ranking_visualization

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
        logging.info("✓ Database connection successful")
    except Exception as e:
        logging.error(f"Database error: {e}")
        return False
    
    # Check clustering prerequisites
    cluster_status = get_enhanced_clustering_status()
    if cluster_status['clustering_available']:
        logging.info("✓ Enhanced clustering data found")
    else:
        logging.info("ℹ Enhanced clustering not available - run clustering mode for better performance")
    
    logging.info("✓ System requirements check passed")
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
    
    # Check database
    try:
        conn = create_connection()
        if conn:
            stats = get_feature_completeness(conn)
            if stats:
                logging.info(f"   Database status:")
                logging.info(f"   Total images: {stats[0]}")
                logging.info(f"   Complete features: {stats[5]}")
            conn.close()
    except:
        logging.info("Database connection failed")
    
    # Check pickle files
    logging.info("Feature files:")
    for feature_type, filename in FEATURE_FILES.items():
        filepath = os.path.join(PICKLE_PATH, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            logging.info(f"   ✓ {filename}: {size_mb:.2f} MB")
        else:
            logging.info(f"   ✗ {filename}: Not found")
    
    # Check enhanced clustering
    cluster_file = os.path.join(PICKLE_PATH, 'enhanced_cluster_data.pkl')
    if os.path.exists(cluster_file):
        size_mb = os.path.getsize(cluster_file) / 1024 / 1024
        logging.info(f"Enhanced clustering: {size_mb:.2f} MB")
        logging.info("Clustering-first search available!")
        
        # Try to load clustering stats
        try:
            import pickle
            with open(cluster_file, 'rb') as f:
                cluster_data = pickle.load(f)
                
            if 'cluster_assignments' in cluster_data:
                logging.info("   Clustering statistics:")
                for ft, assignments in cluster_data['cluster_assignments'].items():
                    n_clusters = len(set(assignments.values()))
                    logging.info(f"     {ft}: {len(assignments)} images, {n_clusters} clusters")
        except:
            pass
    else:
        logging.info(" Enhanced clustering: Not available")
        logging.info("   Run: python main.py --mode clustering")
    
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
            logging.info("✓ Learning mode completed successfully!")
            
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
            logging.error("✗ Learning mode failed!")
            return False
            
    except Exception as e:
        logging.error(f"Learning mode error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_clustering_mode(args):
    """Run enhanced clustering mode to create optimized feature clusters."""
    logging.info("=== STARTING CLUSTERING MODE ===")
    
    # Check if features exist
    required_files = ['efficientnet_features.pkl', 'hsv_features.pkl']
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
            logging.warning("   Over-dimensioning will be fixed by clustering")
    
    # Run enhanced clustering
    logging.info("Starting enhanced clustering with intelligent preprocessing...")
    
    try:
        success = run_enhanced_clustering()
        
        if success:
            logging.info("Enhanced clustering completed successfully!")
            logging.info("Clustering-first similarity search is now available")
            
            # Show clustering results summary
            cluster_file = os.path.join(PICKLE_PATH, 'enhanced_cluster_data.pkl')
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
                
                logging.info(" You can now use enhanced similarity search:")
                logging.info("   - python main.py --mode comparison")
                logging.info("   - python main.py --mode interactive")
            
            return True
        else:
            logging.error(" Enhanced clustering failed!")
            return False
            
    except Exception as e:
        logging.error(f"Clustering mode failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comparison_mode(args):
    """Run similarity comparison mode."""
    logging.info("=== STARTING COMPARISON MODE ===")
    
    # Check clustering availability
    cluster_status = get_enhanced_clustering_status()
    if cluster_status['clustering_available']:
        logging.info("Using enhanced clustering-first search")
        use_clustering = True
    else:
        logging.warning("Clustering not available, using basic search")
        logging.info("For better performance, run: python main.py --mode clustering")
        use_clustering = False
    
    try:
        success = comparison_mode(
            target_image_id=args.target_image,
            use_gpu=args.cuda and not args.no_gpu,
            compare_all_methods=True,
            enable_clustering=use_clustering
        )
        
        if success:
            logging.info("Comparison mode completed successfully!")
            return True
        else:
            logging.error("Comparison mode failed!")
            return False
            
    except Exception as e:
        logging.error(f"Comparison mode error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_interactive_mode_wrapper(args):
    """Run interactive similarity search mode."""
    logging.info("=== STARTING INTERACTIVE MODE ===")
    
    # Check clustering availability
    cluster_status = get_enhanced_clustering_status()
    if cluster_status['clustering_available']:
        logging.info("Enhanced clustering-first search available")
    else:
        logging.warning("Enhanced clustering not available")
        logging.info("For better performance, run: python main.py --mode clustering")
    
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

def run_vis_mode_wrapper(args):
    """Run ranking visualization mode."""
    logging.info("=== STARTING RANKING VISUALIZATION MODE ===")
    
    # Check if features are available
    cluster_status = get_enhanced_clustering_status()
    if not cluster_status['clustering_available']:
        logging.warning("Enhanced clustering not available - results may be limited")
        logging.info("For best results, run: python main.py --mode clustering")
    
    try:
        success = run_ranking_visualization(
            image_path=args.image_path,
            use_cuda=args.cuda and not args.no_gpu
        )
        
        if success:
            logging.info("Ranking visualization completed successfully!")
            return True
        else:
            logging.error("Ranking visualization failed!")
            return False
            
    except Exception as e:
        logging.error(f"Ranking visualization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced Image Similarity Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features from database images
  python main.py --mode learning
  
  # Create optimized clusters for enhanced search
  python main.py --mode clustering
  
  # Compare database images (enhanced with clustering)
  python main.py --mode comparison --target-image "image.jpg"
  
  # Interactive mode - GUI to select any image (enhanced)
  python main.py --mode interactive
  
  # Interactive mode - specific image (enhanced)
  python main.py --mode interactive --image-path "path/to/your/image.jpg"
  
# Ranking visualization - compare individual vs integrated feature rankings
  python main.py --mode vis
  
  # Ranking visualization - specific image
  python main.py --mode vis --image-path "image.jpg"

  # Check system status and clustering availability
  python main.py --status
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['learning', 'comparison', 'interactive', 'clustering', 'vis'],
        default='interactive',
        help='Operation mode to run'
    )
    
    parser.add_argument(
        '--target-image',
        type=str,
        help='Target image ID for comparison mode'
    )
    
    parser.add_argument(
        '--image-path',
        type=str,
        help='Path to image for interactive mode'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=30,
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
    
    # Show status and exit if requested
    if args.status:
        show_system_status()
        return 0
    
    # Check system requirements
    if not check_system_requirements():
        logging.error("System requirements check failed")
        return 1
    
    success = True
    
    # Learning phase
    if args.mode == 'learning':
        success &= run_learning_mode(args)
    
    # Clustering phase
    if args.mode == 'clustering':
        success &= run_clustering_mode(args)
    
    # Comparison phase
    if args.mode == 'comparison':
        success &= run_comparison_mode(args)
    
    # Interactive phase
    if args.mode == 'interactive':
        success &= run_interactive_mode_wrapper(args)

    # Visualisation phase
    if args.mode == 'vis':
        success &= run_vis_mode_wrapper(args)
    
    # Final status and guidance
    if success:
        logging.info("=== ALL OPERATIONS COMPLETED SUCCESSFULLY ===")
        
        # Provide specific guidance based on mode
       
        if args.mode == 'clustering':
            logging.info(" Enhanced clustering completed successfully!")
            logging.info(" Now you can use enhanced similarity search:")
            logging.info("   python main.py --mode comparison")
            logging.info("   python main.py --mode interactive")
                      
        elif args.mode == 'learning':
            logging.info("Feature extraction completed!")

            
        elif args.mode == 'interactive':
            logging.info("Interactive similarity search completed!")
            
        elif args.mode == 'comparison':
            logging.info("Enhanced comparison mode completed!")

        elif args.mode == 'vis':
            logging.info("Ranking visualization completed!")
        
        return 0
    else:
        logging.error("=== SOME OPERATIONS FAILED ===")
        logging.info("Please check the logs for details.")
        return 1      

if __name__ == "__main__":
    sys.exit(main())