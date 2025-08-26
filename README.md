#### Here is an image similarity search system using ConvNeXt deep learning features & HSV color histograms with KMEANS clustering-optimized search. Last edit and library reqierements are of the year 2025. 


### Things you need for a start:
- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- 4GB+ RAM
- Images dataset directory. The program can traverse through nested folders.

### Requirements:
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
scikit-image>=0.21.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.66.0
tkinter-utils>=0.1.0
faiss-cpu>=1.7.4 
timm>=0.9.0
umap-learn>=0.5.4

####  For GPU support, use faiss-gpu>=1.7.4 instead. In 2025 only works with Linux

### Installation

1. **Clone this repository**:
```bash
git clone git@github.com:Salt-is-leaving/Image-Recommender-App
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure paths**
Edit `config.py` and update:

PATH_TO_SSD = r"D:\your_images"  
BASE_PATH = r"D:\Code_image_rec"  # Project directory
```

### Directory Structure
``````bash
your-project/
├── images/                 # Image dataset (PATH_TO_SSD)
├── pickles/               # Generated feature files
├── metadata.db            # Image metadata database
├── config.py                   # Application files
├── main.py
├── convnext_extractor.py
├── feature_extraction_pipeline.py
├── weight_optimizer.py
├── clustering.py
├── interactive_pipeline.py
└── fearure_visualization.py

```
### Available Modes. It's important to run the modes exactly in this order for the first run. Depending on the volume of your dataset and whether you set CUDA=True, the time for the inference mode might vary significantly. I recommend to compute  hsv_features.pkl on CPU cause it takes significantly less time & use batch processing of 32 or 64 for ConvNext features. 

### 1. **Learning Mode** (`--mode learning`)
**Purpose**: Extract ConvNeXt and HSV features from all images in your dataset
```bash
python main.py --mode learning
```
**Options**:
- `--batch-size 32` - Processing batch size
- `--cuda` - Enable GPU acceleration (default: true)

**What it does**:
- Scans your image directory
- Extracts 1024-dimensional ConvNeXt features
- Extracts HSV color histograms with bin 8, 8, 16
- Stores features in pickle files
- Updates metadata database. You might need to make some adjustments to SQL-statements, if you dont use SQLite. 

---
### 2. **Clustering Mode** (`--mode clustering`)
**Purpose**: Create optimized clusters for fast similarity search
```bash
python main.py --mode clustering
```

**Prerequisites**: Must run learning mode first

**What it does**:
- Groups similar images into clusters
- Reduces feature dimensions with PCA
- Builds FAISS search indices
- Enables 10-100x faster similarity search

---

### 3. **Interactive Mode** (`--mode interactive`)
**Purpose**: GUI-based similarity search for any image
```bash
# Select image via GUI
python main.py --mode interactive

# Search specific image
python main.py --mode interactive --image-path "path/to/query.jpg"
```
**Prerequisites**: Must run learning and clustering modes first

**Features**:
- Visual comparison: Integrated vs Individual features
- Clustering-optimized search
- Dynamic weight optimization
- Real-time feature extraction for new images

---
### 4. **Weight Analysis Mode** (`--mode weights`)
**Purpose**: Analyze feature quality and optimize search weights
```bash
# Analyze current weights
python main.py --mode weights

# Force recalculation
python main.py --mode weights --recalculate-weights
```

**What it provides**:
- Feature quality metrics
- Optimal weight recommendations
- Performance analysis
- Feature correlation insights

---
### 5. **Visualization Mode** (`--mode vis`)
**Purpose**: Visualize feature spaces using PCA projections
```bash
python main.py --mode vis
```

**Generates**:
- Side-by-side feature space comparisons
- Combined feature analysis
- Correlation visualizations
- Cluster distribution plots

## System Commands

### Check System Status
```bash
python main.py --status
```
Shows:
- Database statistics
- Feature file sizes
- Clustering availability
- System health

### Weight Analysis
```bash
python main.py --analyze-weights
```
Quick weight analysis without full mode

## Typical Workflow

### First-time Setup
```bash
# 1. Extract features from your images (required)
python main.py --mode learning

# 2. Create search clusters (recommended)
python main.py --mode clustering

# 3. Start searching. A context window will appear so that you can choose the target image to look for top_k most similair images
python main.py --mode interactive
```

### Feature Analysis
```bash
# Check system status
python main.py --status

# Analyze feature quality
python main.py --mode weights

# Visualize feature spaces
python main.py --mode vis
```

## Configuration Options

### GPU Settings
```bash
# Use GPU (default)
python main.py --mode learning --cuda

# Force CPU only
python main.py --mode interactive --no-gpu
```

### Performance Tuning
```bash
# Larger batch size for faster processing (if you have more RAM)
python main.py --mode learning --batch-size 64

# Smaller batch size for limited memory
python main.py --mode learning --batch-size 16
```

## Troubleshooting

### Common Issues

**"No images found"**
- Check `PATH_TO_SSD` in `config.py`
- Ensure images are in supported formats (jpg, png, jpeg)

**"Clustering data not found"**
- Run learning mode first: `python main.py --mode learning`
- Then run clustering: `python main.py --mode clustering`

**GPU memory errors**
- Use `--no-gpu` flag for search
- Reduce `--batch-size` for learning

**Poor search results**
- Run weight analysis: `python main.py --mode weights`
- Check feature quality in visualization mode

### Performance Tips

1. **For large datasets (>10k images)**:
   - Run clustering mode for 10-100x faster search
   - Use GPU acceleration
   - Monitor RAM usage during learning

2. **For better accuracy**:
   - Ensure diverse, high-quality image dataset
   - Run weight analysis to optimize feature balance
   - Use clustering mode for multi-feature search

3. **For debugging**:
   - Check system status regularly
   - Use visualization mode to understand feature spaces
   - Monitor logs for feature quality warnings

### Example Output

After running all modes, you'll have:
- **Database**: Image metadata and feature tracking
- **Features**: ConvNeXt (semantic) + HSV (color) embeddings
- **Clusters**: Optimized search indices
- **Search**: Interactive similarity matching

The system provides both individual feature comparisons and intelligent multi-feature integration for the best similarity results.

### Architecture

- **Features**: ConvNeXt-Base (1024D) + HSV histograms (288D)
- **Clustering**: K-means with PCA dimensionality reduction
- **Search**: FAISS-accelerated cluster-first similarity
- **Weights**: Dynamic optimization based on feature quality
- **Storage**: Pickle files + SQLite metadata


