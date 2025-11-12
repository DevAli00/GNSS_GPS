#!/bin/bash
# filepath: setup_environment.sh

# GPS Signal Processing - Environment Setup Script
# This script creates a virtual environment and installs all required dependencies

set -e  # Exit on any error

echo "=================================================="
echo "GPS Signal Processing - Environment Setup"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${BLUE}🔍 Checking Python version...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✅ Found: $PYTHON_VERSION"
else
    echo -e "${RED}❌ Error: Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo -e "\n${BLUE}📁 Project Directory: $PROJECT_DIR${NC}"

# Remove old venv if exists
if [ -d "$VENV_DIR" ]; then
    echo -e "\n${BLUE}🗑️  Removing old virtual environment...${NC}"
    rm -rf "$VENV_DIR"
fi

# Create virtual environment
echo -e "\n${BLUE}🔨 Creating virtual environment...${NC}"
python3 -m venv "$VENV_DIR"
echo "✅ Virtual environment created at: $VENV_DIR"

# Activate virtual environment
echo -e "\n${BLUE}⚡ Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo -e "\n${BLUE}⬆️  Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "\n${BLUE}📦 Installing dependencies...${NC}"
pip install numpy scipy matplotlib

# Verify installations
echo -e "\n${BLUE}✅ Verifying installations...${NC}"
python3 << EOF
import sys
import numpy as np
import scipy
import matplotlib

print(f"  ✓ Python: {sys.version.split()[0]}")
print(f"  ✓ NumPy: {np.__version__}")
print(f"  ✓ SciPy: {scipy.__version__}")
print(f"  ✓ Matplotlib: {matplotlib.__version__}")
EOF

# Create requirements.txt
echo -e "\n${BLUE}📝 Creating requirements.txt...${NC}"
cat > "$PROJECT_DIR/requirements.txt" << EOF
# GPS Signal Processing - Python Dependencies
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
EOF
echo "✅ requirements.txt created"

# Create project structure
echo -e "\n${BLUE}📂 Creating project structure...${NC}"
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/output"
mkdir -p "$PROJECT_DIR/plots"

# Create .gitignore
echo -e "\n${BLUE}📝 Creating .gitignore...${NC}"
cat > "$PROJECT_DIR/.gitignore" << EOF
# Virtual Environment
venv/
env/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Data files (too large for git)
*.wav
data/

# Output files
output/
plots/
*.png
*.pdf

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF
echo "✅ .gitignore created"

# Create activation helper script
echo -e "\n${BLUE}📝 Creating activation helper...${NC}"
cat > "$PROJECT_DIR/activate.sh" << 'EOF'
#!/bin/bash
# Quick activation script
source venv/bin/activate
echo "✅ Virtual environment activated!"
echo "💡 Run 'deactivate' to exit the virtual environment"
EOF
chmod +x "$PROJECT_DIR/activate.sh"
echo "✅ activate.sh created"

# Create README
echo -e "\n${BLUE}📝 Creating README.md...${NC}"
cat > "$PROJECT_DIR/README.md" << 'EOF'
# GPS Signal Processing Project

This project processes GPS L1 C/A signals from WAV files to detect satellites and extract navigation data.

## Setup

### Automatic Setup (Recommended)
```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

### Manual Setup
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### Activate Environment
```bash
source activate.sh  # On Windows: venv\Scripts\activate
```

### Run Phase 0 (Data Inspection)
```bash
python setup.py
```

## Project Phases

- **Phase 0**: Setup and Data Inspection ✅
- **Phase 1**: C/A Code Generator (Next)
- **Phase 2**: Satellite Acquisition
- **Phase 3**: Signal Tracking
- **Phase 4**: Data Decoding
- **Phase 5**: Position Calculation

## File Structure
```
GNSS_GPS/
├── setup_environment.sh    # Environment setup script
├── activate.sh             # Quick activation helper
├── requirements.txt        # Python dependencies
├── setup.py               # Phase 0: Data inspection
├── code_generator.py      # Phase 1: C/A code generation (TODO)
├── acquisition.py         # Phase 2: Satellite acquisition (TODO)
├── data/                  # Place your .wav files here
├── output/                # Processed data output
└── plots/                 # Generated plots
```

## Dependencies

- Python 3.8+
- NumPy: Signal processing
- SciPy: WAV file handling
- Matplotlib: Visualization

## Usage

1. Place your GPS WAV file in the `data/` directory
2. Update the filename in `setup.py`
3. Run: `python setup.py`

## Troubleshooting

### "File not found" error
Make sure your WAV file path is correct in `setup.py`:
```python
wav_file = "data/groupe_20M.wav"  # Update this path
```

### Virtual environment issues
```bash
deactivate  # Exit current venv
rm -rf venv  # Remove old venv
./setup_environment.sh  # Run setup again
```
EOF
echo "✅ README.md created"

# Create Windows batch file for setup
echo -e "\n${BLUE}📝 Creating Windows setup script...${NC}"
cat > "$PROJECT_DIR/setup_environment.bat" << 'EOF'
@echo off
REM GPS Signal Processing - Windows Environment Setup Script

echo ==================================================
echo GPS Signal Processing - Environment Setup
echo ==================================================

echo.
echo Checking Python version...
python --version
if errorlevel 1 (
    echo Error: Python is not installed
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing dependencies...
pip install numpy scipy matplotlib

echo.
echo Verifying installations...
python -c "import numpy as np; import scipy; import matplotlib; print(f'NumPy: {np.__version__}'); print(f'SciPy: {scipy.__version__}'); print(f'Matplotlib: {matplotlib.__version__}')"

echo.
echo ✅ Setup complete!
echo.
echo To activate the virtual environment in the future:
echo   venv\Scripts\activate
echo.
pause
EOF
echo "✅ setup_environment.bat created (for Windows)"

# Final summary
echo -e "\n${GREEN}=================================================="
echo "✨ Setup Complete!"
echo "==================================================${NC}"
echo ""
echo "📁 Project Structure:"
echo "   ├── venv/              Virtual environment"
echo "   ├── data/              Place your .wav files here"
echo "   ├── output/            Processed data"
echo "   ├── plots/             Generated plots"
echo "   ├── setup.py           Phase 0 script"
echo "   └── requirements.txt   Dependencies"
echo ""
echo "🚀 Next Steps:"
echo "   1. Place your GPS WAV file in the data/ directory"
echo "   2. Activate the virtual environment:"
echo "      ${BLUE}source activate.sh${NC}"
echo "   3. Run Phase 0:"
echo "      ${BLUE}python setup.py${NC}"
echo ""
echo "💡 Tips:"
echo "   • To activate later: source activate.sh"
echo "   • To deactivate: deactivate"
echo "   • To reinstall: ./setup_environment.sh"
echo ""
echo -e "${GREEN}Happy GPS signal processing! 📡${NC}"