#!/bin/bash
set -e

echo "=== NUCLEAR OPTION: Complete Fresh Rebuild ==="
echo ""

# Step 1: Remove old venv
echo "Step 1: Removing old .venv..."
rm -rf .venv
echo "✓ Removed"

# Step 2: Clear ALL Python caches
echo ""
echo "Step 2: Clearing all caches..."
rm -rf ~/.cache/pip
rm -rf ~/.cache/pypoetry
rm -rf /tmp/pip-*
/usr/bin/python3.12 -m pip cache purge 2>/dev/null || true
echo "✓ All caches cleared"

# Step 3: Verify system Python is clean
echo ""
echo "Step 3: Verifying system Python..."
/usr/bin/python3.12 -c "import sys; print(f'Python: {sys.version}'); import re._constants; print(f'MAXGROUPS: {re._constants.MAXGROUPS}')"
echo "✓ System Python verified"

# Step 4: Create venv with isolated pip
echo ""
echo "Step 4: Creating fresh isolated venv..."
/usr/bin/python3.12 -m venv .venv --clear --upgrade-deps
echo "✓ Created with upgraded deps"

# Step 5: Activate and verify
echo ""
echo "Step 5: Activating and verifying base environment..."
source .venv/bin/activate
python -c "import sys; print(f'Venv Python: {sys.version}')"
which python
which pip

# Step 6: Install requirements in batches with verification
echo ""
echo "Step 6: Installing requirements..."
echo "This will take 10-20 minutes for 555 packages..."
pip install --no-cache-dir --upgrade pip setuptools wheel
pip install --no-cache-dir -r requirements.txt 2>&1 | tee install_log.txt

echo ""
echo "✓ Complete!"
echo ""
echo "Check install_log.txt for any warnings or errors"
echo "Activate with: source .venv/bin/activate"

# Step 7: Fix known package issues
echo ""
echo "Step 7: Fixing known package issues..."
OLEFILE2=".venv/lib/python3.12/site-packages/olefile/olefile2.py"
if [ -f "$OLEFILE2" ]; then
    mv "$OLEFILE2" "$OLEFILE2.disabled"
    echo "✓ Disabled olefile2.py (Python 2 fallback)"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To use pip-extra-reqs correctly:"
echo "  ./check_requirements.sh"
echo ""
echo "Or manually:"
echo "  pip-extra-reqs src/ tests/ scripts/"
echo ""
echo "Never run: pip-extra-reqs ."
echo "(It will scan .venv and fail on incompatible packages)"
