# Virtual Environment Troubleshooting Guide

## Issues Encountered and Solutions

### Issue 1: `NameError: name 'MAXGROUPS' is not defined`

**Cause**: The `re/_constants.py` file in the venv was missing the `MAXGROUPS` import.

**Solution**: Fixed by adding `MAXGROUPS` to the import:

```python
from _sre import MAXREPEAT, MAXGROUPS
```

### Issue 2: `olefile2.py` Python 2 Syntax Errors

**Cause**: The `olefile==0.47` package from PyPI includes a Python 2 fallback file (`olefile2.py`) with syntax that's invalid in Python 3:

- Long integer literals with `L` suffix (e.g., `0xFFFFFFFAL`)
- Old-style `raise` statements (e.g., `raise ValueError, 'message'`)

**Solution**: Disable the Python 2 fallback file (it's not needed on Python 3):

```bash
mv .venv/lib/python3.12/site-packages/olefile/olefile2.py \
   .venv/lib/python3.12/site-packages/olefile/olefile2.py.disabled
```

### Issue 3: `pip-extra-reqs` Scanning `.venv` Directory

**Cause**: Running `pip-extra-reqs .` scans the entire current directory recursively, including `.venv/lib/python3.12/site-packages/`, which contains packages with:

- Python 2 legacy code
- Binary files that can't be decoded as UTF-8
- Test files with various encodings

**Solution**: Only scan your source directories:

```bash
pip-extra-reqs src/ tests/ scripts/ -f ".venv/*" -f "site/*"
```

## How to Properly Check Requirements

### Correct Usage

Use the provided script:

```bash
./check_requirements.sh
```

Or manually:

```bash
# Check for extra requirements (packages in requirements.txt not used in code)
pip-extra-reqs src/ tests/ scripts/ -f ".venv/*" -f "site/*" -f "*_output/*"

# Check for missing requirements (packages used in code but not in requirements.txt)
pip-missing-reqs src/ -f ".venv/*" -f "site/*"
```

### ⚠️ NEVER Run

```bash
pip-extra-reqs .   # ❌ Will scan .venv and fail
```

## Recreating the Virtual Environment

If you need to recreate the venv from scratch:

```bash
./recreate_venv_nuclear.sh
```

This script:

1. Removes old `.venv`
2. Clears all pip caches
3. Verifies system Python
4. Creates fresh venv
5. Installs requirements with `--no-cache-dir`
6. Automatically fixes the `olefile2.py` issue

## Why These Issues Occur

1. **Upstream Package Issues**: Some PyPI packages (like `olefile==0.47`) still include Python 2 legacy code for backward compatibility, even though they support Python 3.

2. **Tool Behavior**: Tools like `pip-extra-reqs` that parse Python files don't distinguish between production code and legacy/compatibility fallback files.

3. **Dependency Chains**: You don't directly use `olefile`, but it's required by:
   - `msoffcrypto-tool`
   - `python-oxmsg`

## SyntaxWarnings in Other Packages

You may see warnings like:

```
SyntaxWarning: invalid escape sequence '\S'
```

These are from packages that should use raw strings for regex patterns. They're **warnings only** and don't cause failures. Affected packages include:

- `docopt`
- `pysbd`
- `jieba`
- `hdbscan`

These are upstream issues and can be safely ignored.

## Summary

Your virtual environment is now **correctly configured**. The errors you were seeing were not due to corruption, but due to:

1. Scanning the wrong directories with `pip-extra-reqs`
2. An upstream package including Python 2 legacy code

Both issues are now resolved.
