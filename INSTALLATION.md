# Installation Guide

This guide provides detailed installation instructions for the Fractal Dimension Analysis Tool, designed to support researchers and reviewers of the accompanying publication.

## Quick Installation (Recommended)

### For Publication Reviewers and Basic Users
```bash
# Clone the repository
git clone https://github.com/rwdlnk/Fractal-Dimension-Analyzer.git
cd Fractal-Dimension-Analyzer

# Install minimal dependencies
pip install -r requirements_minimal.txt

# Verify installation
python fractal_analyzer.py --generate koch --level 5
```

### For Researchers and Extended Use
```bash
# Install full dependencies for all features
pip install -r requirements.txt

# Verify installation with comprehensive test
python fractal_analyzer.py --generate koch --level 5 --analyze_linear_region
```

## Detailed Installation Options

### Option 1: Minimal Installation (Core Functionality)
**Best for:** Publication reviewers, basic fractal analysis
```bash
pip install -r requirements_minimal.txt
```
**Includes:** numpy, scipy, matplotlib, numba
**Features:** All core fractal analysis, basic plotting

### Option 2: Full Installation (All Features)
**Best for:** Research use, advanced analysis
```bash
pip install -r requirements.txt
```
**Includes:** All minimal packages plus optional enhancements
**Features:** Advanced plotting, development tools, extended file format support

### Option 3: Development Installation
**Best for:** Contributors, method development
```bash
pip install -r requirements.txt
pip install pytest jupyter black flake8
```
**Features:** All features plus testing and development tools

### Option 4: Conda Installation (Alternative)
```bash
# Create conda environment
conda create -n fractal-analysis python=3.9
conda activate fractal-analysis

# Install dependencies
conda install numpy scipy matplotlib numba -c conda-forge
# For optional packages
conda install pandas seaborn h5py pytest jupyter -c conda-forge
```

### Option 5: Using Makefile (Convenient)
```bash
# For reviewers
make install-minimal

# For researchers  
make install

# For developers
make install-dev

# For conda users
make install-conda
```

## System Requirements

### Python Version
- **Required:** Python 3.8 or higher
- **Recommended:** Python 3.9-3.11
- **Tested on:** Python 3.8, 3.9, 3.10, 3.11

### Operating Systems
- **Linux:** Tested on Ubuntu 20.04+, CentOS 7+
- **macOS:** Tested on macOS 10.15+
- **Windows:** Tested on Windows 10+

### Hardware Requirements
- **RAM:** Minimum 4GB, recommended 8GB+ for large datasets
- **CPU:** Any modern processor (numba provides significant acceleration)
- **Storage:** ~100MB for software, additional space for data and results

## Dependency Details

### Core Dependencies (Always Required)
```python
numpy>=1.21.0      # Numerical computations, array operations
scipy>=1.7.0       # Statistical analysis, linear regression
matplotlib>=3.5.0  # Plotting, visualization
numba>=0.56.0      # JIT compilation for performance critical code
```

### Optional Dependencies
```python
pandas>=1.3.0      # Data handling (enhanced analysis)
seaborn>=0.11.0    # Enhanced plotting aesthetics
h5py>=3.0.0        # HDF5 file format support
pytest>=6.0.0      # Testing framework
jupyter>=1.0.0     # Interactive notebooks
```

## Installation Verification

### Basic Verification
```bash
# Test core functionality
python fractal_analyzer.py --generate koch --level 5

# Expected output: Koch curve generation and basic analysis
# Should create koch_segments_level_5.txt and plots
```

### Comprehensive Verification
```bash
# Test all major features
python fractal_analyzer.py --generate koch --level 5 --analyze_linear_region --eps_plots

# Expected: Multiple EPS plots, detailed analysis output
# Files: koch_*.eps plots and comprehensive console output
```

### Using Test Scripts
```bash
# Test Python compatibility
python test_python_versions.py

# Test all examples
python test_examples.py

# Or use Makefile
make test
make validate
```

## Troubleshooting

### Common Issues

#### 1. Numba Installation Problems
```bash
# Error: "ModuleNotFoundError: No module named 'numba'"
pip install --upgrade numba

# If still failing, try conda:
conda install numba -c conda-forge
```

#### 2. Matplotlib Backend Issues
```bash
# Error: "Backend Qt5Agg is interactive backend"
export MPLBACKEND=Agg  # For headless systems

# Or set in Python:
import matplotlib
matplotlib.use('Agg')
```

#### 3. Memory Issues with Large Datasets
```bash
# For datasets > 50k segments
python fractal_analyzer.py --file large_data.txt --no_box_plot --log_level WARNING
```

#### 4. Permission Errors (Windows)
```bash
# Run as administrator or use user installation
pip install --user -r requirements_minimal.txt
```

#### 5. Array Dimension Errors (Fixed in Latest Version)
```bash
# If you encounter matplotlib array dimension errors:
# Make sure you're using the latest fixed versions of example scripts
git pull origin main  # Get latest fixes
```

### Environment-Specific Solutions

#### HPC/Cluster Systems
```bash
# Load modules if available
module load python/3.9 numpy scipy matplotlib

# Or use conda
module load conda
conda create -n fractal python=3.9
conda activate fractal
conda install -c conda-forge numpy scipy matplotlib numba
```

#### Virtual Environments
```bash
# Using venv
python -m venv fractal_env
source fractal_env/bin/activate  # Linux/Mac
# fractal_env\Scripts\activate   # Windows
pip install -r requirements_minimal.txt
```

#### Docker Container
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_minimal.txt .
RUN pip install -r requirements_minimal.txt
COPY fractal_analyzer.py .
CMD ["python", "fractal_analyzer.py", "--help"]
```

## Performance Optimization

### For Large Datasets (>10k segments)
```bash
# Optimize memory usage
export NUMBA_CACHE_DIR=/tmp/numba_cache
export NUMBA_NUM_THREADS=4  # Adjust based on your CPU cores

# Use streaming mode
python fractal_analyzer.py --file large_data.txt --no_box_plot
```

### For High-Resolution Analysis
```bash
# Enable all optimizations
python fractal_analyzer.py --file data.txt \
    --use_grid_optimization \
    --box_size_factor 1.2 \
    --analyze_linear_region
```

## Publication Support

### For Reviewers
The minimal installation provides everything needed to verify publication results:
```bash
pip install -r requirements_minimal.txt
python fractal_analyzer.py --generate koch --level 5  # ~1.2619 expected
```

### For Reproduction
All publication figures can be reproduced with:
```bash
pip install -r requirements.txt
# Run specific examples as documented in paper
```

### Citation Software Versions
To cite specific software versions used:
```bash
pip list | grep -E "(numpy|scipy|matplotlib|numba)"
# Include these versions in your methods section
```

## Known Issues and Fixes

### Fixed in Current Version
- **Array dimension mismatch**: Fixed in Minkowski and Sierpinski examples
- **UnboundLocalError**: Fixed in boundary analysis scripts
- **Import path issues**: Resolved in all example scripts
- **Memory leaks**: Improved garbage collection and resource management

## Support

### Getting Help
1. **Documentation:** Check README.md and example scripts
2. **Issues:** Report problems on GitHub issues page
3. **Performance:** See performance optimization section above
4. **Academic support:** Contact corresponding author for research-specific questions

### Contributing
See development installation section above for setting up a development environment.

---

**Installation successful?** Try the quick verification command:
```bash
python fractal_analyzer.py --generate koch --level 5
```
Expected dimension: ~1.2619 ± 0.001
