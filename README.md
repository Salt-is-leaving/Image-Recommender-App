# Image Similarity Search System

A powerful image similarity search system using ConvNeXt deep learning features and HSV color histograms with clustering-optimized search.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- 4GB+ RAM
- Images dataset directory

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd image-similarity-search
```

2. **Install dependencies**
```bash
pip install torch torchvision timm
pip install opencv-python pillow scikit-learn
pip install numpy matplotlib seaborn tqdm
pip install faiss-cpu  # or faiss-gpu for GPU acceleration
```

3. **Configure paths**
Edit `config.py` and update:
```python
PATH_TO_SSD = r"D:\your_images"  # Your images directory
BASE_PATH = r"D:\Code_image_rec"  # Project directory
```

### Directory Structure
```
your-project/
├── images/                 # Your image dataset (PATH_TO_SSD)
├── pickles/               # Generated feature files
├── metadata.db            # Image metadata database
└── *.py                   # Application files
```

## Available Modes

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
- Extracts HSV color histograms
- Stores features in pickle files
- Updates metadata database

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

## 📋 System Commands

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

## 📊 Example Output

After running all modes, you'll have:
- **Database**: Image metadata and feature tracking
- **Features**: ConvNeXt (semantic) + HSV (color) embeddings
- **Clusters**: Optimized search indices
- **Search**: Interactive similarity matching

The system provides both individual feature comparisons and intelligent multi-feature integration for the best similarity results.

## Architecture

- **Features**: ConvNeXt-Base (1024D) + HSV histograms (288D)
- **Clustering**: K-means with PCA dimensionality reduction
- **Search**: FAISS-accelerated cluster-first similarity
- **Weights**: Dynamic optimization based on feature quality
- **Storage**: Pickle files + SQLite metadata