#!/usr/bin/env python3
"""
E-commerce CSV Analyzer for Big Data E-commerce Dataset
Analyzes the CSV file in E:\data\big_data_e_commerce for similarity search enhancement
"""

import os
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class EcommerceCSVAnalyzer:
    """Specialized analyzer for e-commerce dataset CSV files."""
    
    def __init__(self, csv_directory):
        self.csv_directory = csv_directory
        self.csv_path = None
        self.df = None
        self.analysis_results = {}
        
    def find_csv_file(self):
        """Find the CSV file in the e-commerce directory."""
        logging.info("=== FINDING CSV FILE ===")
        
        if not os.path.exists(self.csv_directory):
            logging.error(f"Directory not found: {self.csv_directory}")
            return False
            
        # Look for CSV files
        csv_files = []
        for file in os.listdir(self.csv_directory):
            if file.lower().endswith('.csv'):
                full_path = os.path.join(self.csv_directory, file)
                size_mb = os.path.getsize(full_path) / 1024 / 1024
                csv_files.append((full_path, size_mb, file))
                
        if not csv_files:
            logging.error(f"No CSV files found in {self.csv_directory}")
            
            # Check for other file types
            other_files = []
            for file in os.listdir(self.csv_directory):
                if any(file.lower().endswith(ext) for ext in ['.txt', '.tsv', '.json', '.xlsx']):
                    other_files.append(file)
                    
            if other_files:
                logging.info(f"Found other potential metadata files: {other_files}")
                
            return False
            
        # Display found CSV files
        logging.info(f"Found {len(csv_files)} CSV file(s):")
        for i, (path, size, filename) in enumerate(csv_files, 1):
            logging.info(f"  {i}. {filename} ({size:.1f} MB)")
            
        # Use the largest CSV file (likely the main dataset)
        csv_files.sort(key=lambda x: x[1], reverse=True)  # Sort by size
        self.csv_path = csv_files[0][0]
        
        logging.info(f"Using largest CSV: {os.path.basename(self.csv_path)} ({csv_files[0][1]:.1f} MB)")
        return True
        
    def load_and_inspect_csv(self):
        """Load and inspect the e-commerce CSV file."""
        logging.info("=== E-COMMERCE CSV INSPECTION ===")
        
        try:
            # Try different encodings common in e-commerce data
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    # Load a sample first to check structure
                    self.df = pd.read_csv(self.csv_path, encoding=encoding, nrows=1000)
                    logging.info(f"✓ CSV loaded successfully with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logging.error("Could not load CSV with any common encoding")
                return False
                
            # Get full dataset size
            total_rows = sum(1 for line in open(self.csv_path, 'r', encoding=encoding, errors='ignore')) - 1
            
            # Load full dataset if not too large
            if total_rows < 100000:
                self.df = pd.read_csv(self.csv_path, encoding=encoding)
                logging.info(f"Loaded full dataset: {total_rows:,} rows")
            else:
                # Use sample for analysis
                sample_size = min(50000, total_rows // 10)
                self.df = pd.read_csv(self.csv_path, encoding=encoding, nrows=sample_size)
                logging.info(f"Using sample: {len(self.df):,} rows (full dataset: {total_rows:,})")
                
            logging.info(f"Dataset shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
            logging.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Show columns
            logging.info(f"\nColumn names ({len(self.df.columns)}):")
            for i, col in enumerate(self.df.columns):
                sample_value = str(self.df[col].dropna().iloc[0] if len(self.df[col].dropna()) > 0 else "N/A")[:30]
                logging.info(f"  {i+1:2d}. {col:<25} | Sample: {sample_value}")
                
            return True
            
        except Exception as e:
            logging.error(f"Error loading CSV: {e}")
            return False
            
    def analyze_ecommerce_features(self):
        """Analyze e-commerce specific features."""
        logging.info("\n=== E-COMMERCE FEATURE ANALYSIS ===")
        
        ecommerce_features = {
            'product_identifiers': [],
            'image_identifiers': [],
            'categories': [],
            'descriptions': [],
            'prices': [],
            'brands': [],
            'attributes': [],
            'urls': []
        }
        
        for col in self.df.columns:
            col_lower = col.lower()
            sample_values = self.df[col].dropna().astype(str).head(10)
            
            # Product identifiers
            if any(keyword in col_lower for keyword in ['product_id', 'item_id', 'sku', 'asin', 'id']):
                ecommerce_features['product_identifiers'].append(col)
                
            # Image identifiers  
            elif any(keyword in col_lower for keyword in ['image', 'photo', 'picture', 'img']):
                ecommerce_features['image_identifiers'].append(col)
                
            # Categories
            elif any(keyword in col_lower for keyword in ['category', 'class', 'type', 'genre', 'department']):
                ecommerce_features['categories'].append(col)
                
            # Descriptions
            elif any(keyword in col_lower for keyword in ['description', 'title', 'name', 'summary']):
                if self.df[col].astype(str).str.len().mean() > 20:  # Longer text
                    ecommerce_features['descriptions'].append(col)
                    
            # Prices
            elif any(keyword in col_lower for keyword in ['price', 'cost', 'amount', 'value']) and self.df[col].dtype in ['float64', 'int64']:
                ecommerce_features['prices'].append(col)
                
            # Brands
            elif any(keyword in col_lower for keyword in ['brand', 'manufacturer', 'maker', 'company']):
                ecommerce_features['brands'].append(col)
                
            # URLs
            elif any(keyword in col_lower for keyword in ['url', 'link', 'path']) or any('http' in str(val) for val in sample_values):
                ecommerce_features['urls'].append(col)
                
            # Other attributes
            elif self.df[col].dtype == 'object' and self.df[col].nunique() < len(self.df) * 0.5:
                ecommerce_features['attributes'].append(col)
                
        # Report findings
        for feature_type, columns in ecommerce_features.items():
            if columns:
                logging.info(f"\n{feature_type.upper().replace('_', ' ')} ({len(columns)}):")
                for col in columns:
                    unique_count = self.df[col].nunique()
                    missing_count = self.df[col].isnull().sum()
                    sample_vals = self.df[col].dropna().head(3).tolist()
                    logging.info(f"  {col}: {unique_count:,} unique, {missing_count:,} missing")
                    logging.info(f"    Samples: {sample_vals}")
                    
        self.analysis_results['ecommerce_features'] = ecommerce_features
        return any(columns for columns in ecommerce_features.values())
        
    def analyze_image_path_mapping(self):
        """Analyze how CSV images map to your actual image files."""
        logging.info("\n=== IMAGE PATH MAPPING ANALYSIS ===")
        
        image_columns = self.analysis_results['ecommerce_features']['image_identifiers']
        
        if not image_columns:
            logging.warning("No image identifier columns found")
            return False
            
        for col in image_columns:
            logging.info(f"\nAnalyzing image column: {col}")
            
            sample_values = self.df[col].dropna().head(10)
            logging.info(f"Sample values:")
            for val in sample_values:
                logging.info(f"  {val}")
                
            # Check if values look like filenames
            has_extensions = sum(1 for val in sample_values 
                               if any(ext in str(val).lower() for ext in ['.jpg', '.jpeg', '.png', '.bmp']))
            
            # Check if values look like URLs
            has_urls = sum(1 for val in sample_values if 'http' in str(val).lower())
            
            # Check if values look like paths
            has_paths = sum(1 for val in sample_values if '/' in str(val) or '\\' in str(val))
            
            logging.info(f"Analysis:")
            logging.info(f"  Has image extensions: {has_extensions}/{len(sample_values)}")
            logging.info(f"  Has URLs: {has_urls}/{len(sample_values)}")
            logging.info(f"  Has path separators: {has_paths}/{len(sample_values)}")
            
            # Try to match with actual image files in your dataset
            if has_extensions > 0:
                logging.info(f"Checking if images exist in your dataset...")
                
                # Sample a few image names from CSV
                csv_images = self.df[col].dropna().head(20).tolist()
                found_count = 0
                
                # Check if any match files in E:\data
                for csv_image in csv_images:
                    image_name = os.path.basename(str(csv_image))
                    # This would require walking through E:\data to find matches
                    # For now, we'll just report the pattern
                    
                logging.info(f"  CSV contains {len(csv_images)} image references")
                logging.info(f"  To verify matches, run: find E:\\data -name \"*.jpg\" | head -10")
                
        return True
        
    def analyze_category_hierarchy(self):
        """Analyze category structure for filtering potential."""
        logging.info("\n=== CATEGORY HIERARCHY ANALYSIS ===")
        
        category_columns = self.analysis_results['ecommerce_features']['categories']
        
        if not category_columns:
            logging.warning("No category columns found")
            return False
            
        category_analysis = {}
        
        for col in category_columns:
            logging.info(f"\nCategory column: {col}")
            
            value_counts = self.df[col].value_counts().head(15)
            unique_ratio = self.df[col].nunique() / len(self.df)
            
            category_analysis[col] = {
                'unique_count': self.df[col].nunique(),
                'unique_ratio': unique_ratio,
                'top_categories': value_counts.to_dict(),
                'missing_count': self.df[col].isnull().sum()
            }
            
            logging.info(f"  Categories: {self.df[col].nunique():,} unique ({unique_ratio:.1%})")
            logging.info(f"  Missing: {self.df[col].isnull().sum():,}")
            logging.info(f"  Top categories:")
            
            for category, count in value_counts.head(10).items():
                percentage = count / len(self.df) * 100
                logging.info(f"    {category}: {count:,} ({percentage:.1f}%)")
                
            # Assess filtering potential
            if unique_ratio < 0.1 and self.df[col].isnull().sum() < len(self.df) * 0.3:
                logging.info(f"  ✅ EXCELLENT for category filtering!")
            elif unique_ratio < 0.3:
                logging.info(f"  ✓ Good for category filtering")
            else:
                logging.info(f"  ⚠ Limited filtering value (too many categories)")
                
        self.analysis_results['category_analysis'] = category_analysis
        return True
        
    def assess_similarity_enhancement_potential(self):
        """Assess overall potential for enhancing similarity search."""
        logging.info("\n=== SIMILARITY ENHANCEMENT ASSESSMENT ===")
        
        enhancements = []
        scores = {}
        
        # 1. Image path mapping potential
        image_cols = self.analysis_results['ecommerce_features']['image_identifiers']
        if image_cols:
            scores['image_mapping'] = 40
            enhancements.append("🔗 Image path mapping for feature validation")
        else:
            scores['image_mapping'] = 0
            
        # 2. Category filtering potential
        category_cols = self.analysis_results['ecommerce_features']['categories']
        useful_categories = 0
        if 'category_analysis' in self.analysis_results:
            for col, analysis in self.analysis_results['category_analysis'].items():
                if analysis['unique_ratio'] < 0.2 and analysis['missing_count'] < len(self.df) * 0.3:
                    useful_categories += 1
                    
        if useful_categories > 0:
            scores['category_filtering'] = min(30, useful_categories * 15)
            enhancements.append(f"🏷️ Category-based filtering ({useful_categories} useful columns)")
        else:
            scores['category_filtering'] = 0
            
        # 3. Text description enhancement
        desc_cols = self.analysis_results['ecommerce_features']['descriptions']
        if desc_cols:
            scores['text_enhancement'] = 20
            enhancements.append(f"📝 Text-based semantic enhancement ({len(desc_cols)} text columns)")
        else:
            scores['text_enhancement'] = 0
            
        # 4. Brand/attribute filtering
        brand_cols = self.analysis_results['ecommerce_features']['brands']
        attr_cols = self.analysis_results['ecommerce_features']['attributes']
        if brand_cols or attr_cols:
            scores['attribute_filtering'] = 10
            enhancements.append(f"🏪 Brand/attribute filtering ({len(brand_cols + attr_cols)} columns)")
        else:
            scores['attribute_filtering'] = 0
            
        total_score = sum(scores.values())
        
        logging.info(f"Enhancement Potential Score: {total_score}/100")
        logging.info(f"\nPossible enhancements ({len(enhancements)}):")
        for enhancement in enhancements:
            logging.info(f"  {enhancement}")
            
        # Overall recommendation
        if total_score >= 70:
            recommendation = "🟢 HIGH VALUE - Strongly recommend full CSV integration"
        elif total_score >= 40:
            recommendation = "🟡 MEDIUM VALUE - Integrate key features"
        elif total_score >= 20:
            recommendation = "🟠 LOW VALUE - Consider selective integration"
        else:
            recommendation = "🔴 MINIMAL VALUE - Focus on visual features only"
            
        logging.info(f"\nRecommendation: {recommendation}")
        
        return total_score, recommendation
        
    def generate_implementation_plan(self):
        """Generate specific implementation plan for your e-commerce dataset."""
        logging.info("\n=== IMPLEMENTATION PLAN ===")
        
        ecommerce_features = self.analysis_results['ecommerce_features']
        
        # Phase 1: Image mapping
        if ecommerce_features['image_identifiers']:
            logging.info("📋 PHASE 1 (IMMEDIATE): Image Path Mapping")
            logging.info("  Goal: Verify extracted features match CSV entries")
            
            for col in ecommerce_features['image_identifiers']:
                logging.info(f"  1. Use column '{col}' to map CSV → image files")
                
            logging.info("  2. Add validation to feature extraction:")
            logging.info("     - Check if extracted image exists in CSV")
            logging.info("     - Log mismatches for quality control")
            
        # Phase 2: Category filtering
        if ecommerce_features['categories']:
            logging.info("\n📋 PHASE 2 (SHORT-TERM): Category Filtering")
            logging.info("  Goal: Add category-based search refinement")
            
            for col in ecommerce_features['categories'][:2]:  # Top 2 category columns
                logging.info(f"  1. Implement filtering by '{col}'")
                
            logging.info("  2. Add to similarity search:")
            logging.info("     - Pre-filter candidates by category")
            logging.info("     - Add category-aware similarity scoring")
            
        # Phase 3: Text enhancement
        if ecommerce_features['descriptions']:
            logging.info("\n📋 PHASE 3 (MEDIUM-TERM): Text Enhancement")
            logging.info("  Goal: Combine visual + text similarity")
            
            for col in ecommerce_features['descriptions'][:2]:
                logging.info(f"  1. Extract text embeddings from '{col}'")
                
            logging.info("  2. Implement hybrid similarity:")
            logging.info("     - 70% visual similarity (ConvNeXt + HSV)")
            logging.info("     - 30% text similarity (sentence embeddings)")
            
        # Phase 4: Advanced features
        if ecommerce_features['brands'] or ecommerce_features['prices']:
            logging.info("\n📋 PHASE 4 (LONG-TERM): Advanced Features")
            logging.info("  Goal: Multi-modal similarity search")
            
            if ecommerce_features['brands']:
                logging.info("  1. Brand-aware similarity scoring")
            if ecommerce_features['prices']:
                logging.info("  2. Price-range filtering")
                
        logging.info(f"\n{'='*50}")
        logging.info("QUICK START CODE SNIPPETS:")
        logging.info(f"{'='*50}")
        
        # Generate code snippets
        if ecommerce_features['image_identifiers']:
            img_col = ecommerce_features['image_identifiers'][0]
            logging.info(f"""
# Add to config.py
ECOMMERCE_CONFIG = {{
    'csv_path': r'{self.csv_path}',
    'image_column': '{img_col}',
    'enable_csv_validation': True
}}

# Add to feature_extraction_pipeline.py
def validate_image_in_csv(image_id, csv_df):
    return image_id in csv_df['{img_col}'].values
            """)
            
    def run_full_analysis(self):
        """Run complete e-commerce CSV analysis."""
        logging.info("Starting e-commerce CSV analysis...")
        
        # Step 1: Find CSV file
        if not self.find_csv_file():
            return False
            
        # Step 2: Load and inspect
        if not self.load_and_inspect_csv():
            return False
            
        # Step 3: Analyze e-commerce features
        self.analyze_ecommerce_features()
        
        # Step 4: Analyze image mapping
        self.analyze_image_path_mapping()
        
        # Step 5: Analyze categories
        self.analyze_category_hierarchy()
        
        # Step 6: Overall assessment
        score, recommendation = self.assess_similarity_enhancement_potential()
        
        # Step 7: Implementation plan
        self.generate_implementation_plan()
        
        # Final summary
        logging.info(f"\n{'='*60}")
        logging.info("E-COMMERCE CSV ANALYSIS COMPLETED")
        logging.info(f"{'='*60}")
        logging.info(f"Dataset: {len(self.df):,} products × {self.df.shape[1]} attributes")
        logging.info(f"Enhancement Score: {score}/100")
        logging.info(f"Recommendation: {recommendation}")
        
        if score >= 40:
            logging.info("\n✅ PROCEED WITH CSV INTEGRATION")
            logging.info("This e-commerce metadata will significantly improve your search!")
        else:
            logging.info("\n⚠️  CSV INTEGRATION OPTIONAL") 
            logging.info("Visual similarity search alone should work well.")
            
        return True

if __name__ == "__main__":
    # Analyze e-commerce CSV
    csv_directory = r"E:\data\big_data_e_commerce"
    
    analyzer = EcommerceCSVAnalyzer(csv_directory)
    success = analyzer.run_full_analysis()
    
    sys.exit(0 if success else 1)