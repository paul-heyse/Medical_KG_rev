#!/bin/bash
# Check for extra/missing requirements in the project

source .venv/bin/activate

echo "=== Checking for extra requirements ==="
pip-extra-reqs src/ tests/ scripts/ \
    -f ".venv/*" \
    -f "site/*" \
    -f "*_output/*" \
    -f "htmlcov/*" \
    -f "eval/*" \
    2>&1 | grep -v "SyntaxWarning"

echo ""
echo "=== Checking for missing requirements ==="
pip-missing-reqs src/ \
    -f ".venv/*" \
    -f "site/*" \
    2>&1 | grep -v "SyntaxWarning"
