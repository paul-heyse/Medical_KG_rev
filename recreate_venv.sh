#!/bin/bash
set -e  # Exit on any error

echo "=== Fresh Virtual Environment Recreation ==="
echo ""

# Step 1: Remove old venv
echo "Step 1: Removing old .venv..."
if [ -d ".venv" ]; then
    rm -rf .venv
    echo "✓ Old .venv removed"
else
    echo "✓ No old .venv to remove"
fi

# Step 2: Clear pip cache to ensure fresh downloads
echo ""
echo "Step 2: Clearing pip cache..."
/usr/bin/python3.12 -m pip cache purge || echo "Note: Cache purge may fail if cache is empty"
echo "✓ Pip cache cleared"

# Step 3: Create fresh venv with system Python
echo ""
echo "Step 3: Creating fresh virtual environment..."
/usr/bin/python3.12 -m venv .venv --clear
echo "✓ Fresh .venv created"

# Step 4: Activate and upgrade pip/setuptools/wheel
echo ""
echo "Step 4: Upgrading core tools..."
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
echo "✓ Core tools upgraded"

# Step 5: Install requirements with no-cache to force fresh downloads
echo ""
echo "Step 5: Installing requirements (this will take a while)..."
echo "Using --no-cache-dir to ensure fresh package downloads..."
pip install --no-cache-dir -r requirements.txt
echo "✓ Requirements installed"

# Step 6: Verify installation
echo ""
echo "Step 6: Verifying installation..."
python --version
pip --version
echo ""
echo "✓ Installation complete!"
echo ""
echo "To activate the new environment, run:"
echo "  source .venv/bin/activate"
