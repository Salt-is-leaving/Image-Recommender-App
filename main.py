#!/usr/bin/env python3
"""
Enhanced Main Orchestrator for Multi-Feature Image Similarity System

This script provides multiple modes:
1. Learning Mode: Extract and store features from database images
2. Comparison Mode: Compare images from the database  
3. Interactive Mode: Select any image and find similar ones (NEW!)
4. Both Mode: Run learning + comparison

Usage:
    python main.py --mode learning              # Extract features from database
    python main.py --mode comparison            # Compare database images
    python main.py --mode interactive           # Interactive similarity search
    python main.py --mode both                  # Run learning + comparison
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import our modules - FIXED IMPORTS
from config import PATH_TO_SSD, PICKLE_PATH, FEATURE_FILES
from db_api import create_connection, create_tables, get_feature_completeness
from feature_extraction_pipeline import learning_mode
from similarity_search_pipeline import comparison_mode  
from interactive_pipeline import InteractiveSimilaritySearch  

def setup_logging():
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('image_similarity.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_system_requirements():
    """Check if system has required dependencies and data."""
    requirements_met = True
    
    # Check if image directory exists
    if not os.path.exists(PATH_TO_SSD):
        logging.error(f"Image directory not found: {PATH_TO_SSD}")
        requirements_met = False
    
    # Check if pickle directory exists (create if not)
    if not os.path.exists(PICKLE_PATH):
        os.makedirs(PICKLE_PATH, exist_ok=True)
        logging.info(f"Created pickle directory: {PICKLE_PATH}")
    
    # Check database connection
    conn = create_connection()
    if conn is None:
        logging.error("Cannot connect to database")
        requirements_met = False
    else:
        # Ensure tables exist
        create_tables(conn)
        conn.close()
    
    return requirements_met

def show_system_status():
    """Show current system status and feature completeness."""
    logging.info("=== SYSTEM STATUS ===")
    
    # Check database status
    conn = create_connection()
    if conn:
        stats = get_feature_completeness(conn)
        if stats:
            total, hsv, lbp, orb, efficientnet, complete = stats
            logging.info(f"Database Statistics:")
            logging.info(f"  Total images: {total}")
            logging.info(f"  HSV features: {hsv}")
            logging.info(f"  LBP features: {lbp}")
            logging.info(f"  ORB features: {orb}")
            logging.info(f"  EfficientNet features: {efficientnet}")
            logging.info(f"  Complete features: {complete}")
            
            if complete > 0:
                logging.info("✓ System ready for comparison and interactive modes")
            else:
                logging.info("⚠ No complete features found. Run learning mode first.")
        else:
            logging.info("No feature statistics available")
        
        conn.close()
    
    # Check pickle files
    logging.info("Pickle Files Status:")
    for feature_type, filename in FEATURE_FILES.items():
        filepath = os.path.join(PICKLE_PATH, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            logging.info(f"  ✓ {filename}: {size_mb:.2f} MB")
        else:
            logging.info(f"  ✗ {filename}: Not found")
    
    # Check if ready for interactive mode
    required_files = ['efficientnet_features.pkl', 'hsv_features.pkl', 'lbp_features.pkl', 'orb_features.pkl']
    ready_for_interactive = all(
        os.path.exists(os.path.join(PICKLE_PATH, filename)) 
        for filename in required_files
    )
    
    if ready_for_interactive:
        logging.info("🎯 Interactive mode ready: You can search any image!")
    else:
        logging.info("⚠ Interactive mode requires all feature files. Run learning mode first.")

def run_learning_mode(args):
    """Run the learning mode to extract features."""
    logging.info("=== STARTING LEARNING MODE ===")
    
    # Parameters from args or defaults
    use_cuda = not args.no_cuda if hasattr(args, 'no_cuda') else True
    batch_size = getattr(args, 'batch_size', 30)
    image_dir = getattr(args, 'image_dir', PATH_TO_SSD)
    
    logging.info(f"Parameters:")
    logging.info(f"  Image directory: {image_dir}")
    logging.info(f"  Use CUDA: {use_cuda}")
    logging.info(f"  Batch size: {batch_size}")
    
    # Run learning mode
    success = learning_mode(
        image_directory=image_dir,
        use_cuda=use_cuda,
        batch_size=batch_size
    )
    
    if success:
        logging.info("✓ Learning mode completed successfully!")
        show_system_status()
        return True
    else:
        logging.error("✗ Learning mode failed!")
        return False

def run_comparison_mode(args):
    """Run the compatible comparison mode for similarity search."""
    logging.info("=== STARTING COMPARISON MODE ===")
    
    # Parameters from args or defaults
    target_image = getattr(args, 'target_image', None)
    compare_all = not getattr(args, 'no_compare_all', False)
    enable_clustering = not getattr(args, 'no_clustering', False)
    
    logging.info(f"Parameters:")
    logging.info(f"  Target image: {target_image or 'Random'}")
    logging.info(f"  Compare all methods: {compare_all}")
    logging.info(f"  Enable clustering: {enable_clustering}")
    
    # Run comparison mode
    success = comparison_mode(
        target_image_id=target_image,
        compare_all_methods=compare_all,
        enable_clustering=enable_clustering
    )
    
    if success:
        logging.info("✓ Comparison mode completed successfully!")
        return True
    else:
        logging.error("✗ Comparison mode failed!")
        return False

def run_interactive_mode(args):
    """Run the interactive mode for similarity search with any image."""
    logging.info("=== STARTING INTERACTIVE MODE ===")
    
    # Parameters from args or defaults
    use_cuda = not args.no_cuda if hasattr(args, 'no_cuda') else True
    
    logging.info(f"Parameters:")
    logging.info(f"  Use CUDA: {use_cuda}")
    logging.info("  Interactive mode: Select any image for similarity search")
    
    try:
        # Initialize interactive search
        searcher = InteractiveSimilaritySearch(use_cuda=use_cuda)
        
        # Load database features
        if not searcher.load_database_features():
            logging.error("Could not load database features. Run learning mode first!")
            return False
        
        # Build search indices
        searcher.build_search_indices()
        
        logging.info("✓ Interactive system ready!")
        
        # Handle different interaction modes based on args
        if hasattr(args, 'image_path') and args.image_path:
            # Single image mode
            logging.info(f"Processing single image: {args.image_path}")
            
            if not os.path.exists(args.image_path):
                logging.error(f"Image not found: {args.image_path}")
                return False
            
            # Extract features
            features = searcher.extract_features_from_image(args.image_path)
            if features is None:
                logging.error("Feature extraction failed")
                return False
            
            # Find similar images
            similar_images = searcher.find_similar_images(features, top_n=5)
            
            if similar_images:
                logging.info(f"Found {len(similar_images)} similar images!")
                
                # Print results
                print(f"\n📊 Top {len(similar_images)} Most Similar Images:")
                print("-" * 50)
                for i, result in enumerate(similar_images):
                    print(f"{i+1}. {result['image_id']}")
                    print(f"   Overall Similarity: {result['combined_similarity']:.3f}")
                    details = result['feature_details']
                    print(f"   EfficientNet: {details.get('efficientnet', 0):.3f}, "
                          f"HSV: {details.get('hsv', 0):.3f}, "
                          f"LBP: {details.get('lbp', 0):.3f}, "
                          f"ORB: {details.get('orb', 0):.3f}")
                    print()
                
                # Display visual results
                try:
                    searcher.display_results(args.image_path, similar_images)
                except Exception as e:
                    logging.warning(f"Could not display visual results: {e}")
            else:
                logging.info("No similar images found")
        
        else:
            # Interactive GUI mode
            searcher.interactive_search()
        
        logging.info("✓ Interactive mode completed successfully!")
        return True
        
    except Exception as e:
        logging.error(f"Interactive mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Feature Image Similarity System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features from database images
  python main.py --mode learning
  
  # Compare database images
  python main.py --mode comparison --target-image "image.jpg"
  
  # Interactive mode - GUI to select any image
  python main.py --mode interactive
  
  # Interactive mode - specific image
  python main.py --mode interactive --image-path "path/to/your/image.jpg"
  
  # Run learning then comparison
  python main.py --mode both --batch-size 50
  
  # Check system status
  python main.py --status
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', 
        choices=['learning', 'comparison', 'interactive', 'both'],
        default='interactive',  # Default to interactive mode
        help='Mode to run (default: interactive)'
    )
    
    # Learning mode options
    parser.add_argument(
        '--image-dir',
        default=PATH_TO_SSD,
        help=f'Directory containing images for learning (default: {PATH_TO_SSD})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=30,
        help='Batch size for processing (default: 30)'
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )
    
    # Comparison mode options
    parser.add_argument(
        '--target-image',
        help='Specific target image for comparison mode'
    )
    parser.add_argument(
        '--no-compare-all',
        action='store_true',
        help='Skip comparing all feature methods in comparison mode'
    )
    parser.add_argument(
        '--no-clustering',
        action='store_true',
        help='Skip clustering analysis'
    )
    
    # Interactive mode options
    parser.add_argument(
        '--image-path',
        help='Specific image path for interactive mode (skips GUI)'
    )
    
    # Status and utilities
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status and exit'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logging.info(f"Starting Enhanced Multi-Feature Image Similarity System - {datetime.now()}")
    
    # Check system requirements
    if not check_system_requirements():
        logging.error("System requirements not met. Exiting.")
        return 1
    
    # Show status if requested
    if args.status:
        show_system_status()
        return 0
    
    # Run requested modes
    success = True
    
    if args.mode in ['learning', 'both']:
        success &= run_learning_mode(args)
        
        # If learning failed and we're supposed to run both, don't continue
        if not success and args.mode == 'both':
            logging.error("Learning mode failed. Skipping comparison mode.")
            return 1
    
    if args.mode in ['comparison', 'both']:
        success &= run_comparison_mode(args)
    
    if args.mode == 'interactive':
        success &= run_interactive_mode(args)
    
    # Final status
    if success:
        logging.info("=== ALL OPERATIONS COMPLETED SUCCESSFULLY ===")
        
        if args.mode == 'both':
            logging.info("Your image similarity system is now fully operational!")
            logging.info("You can now use interactive mode to search any image!")
        elif args.mode == 'interactive':
            logging.info("Interactive similarity search completed!")
        
        return 0
    else:
        logging.error("=== SOME OPERATIONS FAILED ===")
        logging.error("Check the logs above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
