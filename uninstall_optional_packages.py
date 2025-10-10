#!/usr/bin/env python3
"""
Script to uninstall optional/specialized packages from Medical_KG_rev project.
This removes packages that are not essential for core functionality.
"""

import subprocess
import sys
from typing import List

# Optional/specialized packages to remove
OPTIONAL_PACKAGES = [
    # Jupyter ecosystem
    "jupyter", "jupyterlab", "notebook", "ipython", "ipykernel", "ipywidgets",
    "jupyter-client", "jupyter-server", "jupyterlab-server", "jupyterlab-widgets",
    "jupyter-server-terminals", "jupyter-console", "jupyter-core", "jupyter-events",
    "jupyter-lsp", "nbconvert", "nbformat", "nbclient", "notebook-shim",
    "terminado", "pyzmq", "tornado", "traitlets", "prompt-toolkit", "parso",
    "jedi", "pexpect", "ptyprocess", "decorator", "pickleshare", "pure-eval",
    "stack-data", "executing", "asttokens", "wcwidth", "ipython-pygments-lexers",

    # Visualization packages
    "matplotlib", "matplotlib-inline", "plotly", "dash", "dash-bootstrap-components",
    "contourpy", "cycler", "fonttools", "kiwisolver", "pyparsing",

    # Browser automation
    "selenium", "playwright", "chromedriver-autoinstaller",

    # Specialized ML packages
    "ultralytics", "ultralytics-thop", "thop", "timm", "effdet", "pycocotools",
    "easyocr", "pyclipper", "shapely", "scikit-image", "albucore", "albumentations",
    "opencv-python", "opencv-python-headless", "cupy-cuda12x", "fastrlock",

    # GPU-specific packages
    "nvidia-cublas-cu12", "nvidia-cuda-cupti-cu12", "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12", "nvidia-cufft-cu12",
    "nvidia-cufile-cu12", "nvidia-curand-cu12", "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12", "nvidia-cusparselt-cu12", "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12", "nvidia-nvtx-cu12",

    # Specialized data processing
    "polars", "polars-runtime-32", "narwhals", "duckdb", "xlrd", "xlsxwriter",
    "openpyxl", "et-xmlfile", "python-docx", "python-pptx",

    # Specialized parsing
    "pdfplumber", "pypdfium2", "pdfminer-six", "pdf2image", "pi-heif",
    "reportlab", "pikepdf", "lxml-html-clean", "beautifulsoup4", "soupsieve",
    "html5lib", "webencodings", "cssselect", "tinycss2", "bleach", "defusedxml",
    "sgmllib3k", "feedparser", "feedfinder2", "markdownify", "html2text",
    "striprtf", "boilerpy3", "python-magic", "filetype", "magika",
    "msoffcrypto-tool", "olefile", "python-oxmsg",

    # Specialized ML frameworks
    "docling-core", "docling-parse", "unstructured-client", "unstructured-pytesseract",
    "pytesseract", "voyager", "modelscope", "hf-xet", "huggingface-hub",

    # Specialized utilities
    "quantulum3", "jieba3k", "syntok", "pysbd", "langdetect", "ftfy",
    "webcolors", "tldextract", "url-normalize", "requests-cache", "requests-file",
    "requests-toolbelt", "robust-downloader", "sseclient-py", "firecrawl-py",
    "oxylabs", "spider-client", "pyalex", "pycountry", "pybreaker", "hvac",
    "minio", "redis", "neo4j", "qdrant-client", "opensearch-py", "faiss-cpu",
    "rank-bm25", "hdbscan", "scikit-learn", "xgboost", "numpy-financial",
    "nmslib-metabrainz", "fast-pytorch-kmeans", "colbert-ai", "sentence-transformers",
    "pyserini", "llama-index-core", "llama-index-instrumentation", "llama-index-workflows",
    "langchain", "langchain-core", "langchain-text-splitters", "langsmith",
    "prompthub-py", "qwen-vl-utils", "rapid-table", "semchunk", "mpire",
    "multiprocess", "dill", "pyarrow", "fsspec", "smart-open",

    # Specialized adapters
    "unstructured", "unstructured-inference", "docling-parse", "spacy",
    "nltk", "jieba3k", "syntok", "pysbd", "langdetect", "ftfy",

    # Specialized storage
    "asyncpg", "psycopg2-binary", "sqlalchemy", "alembic", "greenlet",

    # Specialized monitoring
    "prometheus-client", "opentelemetry-api", "opentelemetry-sdk",
    "opentelemetry-instrumentation", "opentelemetry-instrumentation-asgi",
    "opentelemetry-instrumentation-fastapi", "opentelemetry-instrumentation-grpc",
    "opentelemetry-instrumentation-httpx", "opentelemetry-semantic-conventions",
    "opentelemetry-util-http", "sentry-sdk",

    # Specialized auth
    "bcrypt", "passlib", "argon2-cffi", "argon2-cffi-bindings", "python-jose",
    "ecdsa", "rsa", "pyasn1", "pyasn1-modules",

    # Specialized protocols
    "grpcio", "grpcio-status", "grpcio-tools", "grpcio-health-checking",
    "grpc-stubs", "protobuf", "proto-plus", "googleapis-common-protos",
    "google-api-core", "google-auth", "google-cloud-vision",

    # Specialized web frameworks
    "fastapi", "fastapi-cli", "fastapi-cloud-cli", "starlette", "uvicorn",
    "httptools", "websockets", "sse-starlette", "mcp", "strawberry-graphql",
    "graphql-core", "lia-web",

    # Specialized async
    "aiokafka", "aiolimiter", "aiosignal", "aiosqlite", "aiofiles", "aiohttp",
    "aiohappyeyeballs", "frozenlist", "multidict", "yarl", "propcache",
    "async-lru", "nest-asyncio", "anyio", "sniffio", "trio", "trio-websocket",
    "outcome", "sortedcontainers", "h2", "hpack", "hyperframe", "wsproto",

    # Specialized utilities
    "appdirs", "platformdirs", "pathlib-abc", "universal-pathlib", "cloudpathlib",
    "watchdog", "watchfiles", "filelock", "portalocker", "psutil", "humanfriendly",
    "coloredlogs", "colorlog", "colorama", "rich", "rich-toolkit", "typer",
    "click", "shellingham", "tabulate", "prettytable", "emoji", "inflect",
    "num2words", "quantulum3", "jieba3k", "syntok", "pysbd", "langdetect",
    "ftfy", "webcolors", "tldextract", "url-normalize", "requests-cache",
    "requests-file", "requests-toolbelt", "robust-downloader", "sseclient-py",
    "firecrawl-py", "oxylabs", "spider-client", "pyalex", "pycountry",
    "pybreaker", "hvac", "minio", "redis", "neo4j", "qdrant-client",
    "opensearch-py", "faiss-cpu", "rank-bm25", "hdbscan", "scikit-learn",
    "xgboost", "numpy-financial", "nmslib-metabrainz", "fast-pytorch-kmeans",
    "colbert-ai", "sentence-transformers", "pyserini", "llama-index-core",
    "llama-index-instrumentation", "llama-index-workflows", "langchain",
    "langchain-core", "langchain-text-splitters", "langsmith", "prompthub-py",
    "qwen-vl-utils", "rapid-table", "semchunk", "mpire", "multiprocess",
    "dill", "pyarrow", "fsspec", "smart-open",
]

def get_installed_packages() -> List[str]:
    """Get list of currently installed packages."""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"],
                              capture_output=True, text=True, check=True)
        packages = []
        for line in result.stdout.strip().split('\n'):
            if line and '==' in line:
                package_name = line.split('==')[0].lower()
                packages.append(package_name)
        return packages
    except subprocess.CalledProcessError as e:
        print(f"Error getting installed packages: {e}")
        return []

def uninstall_packages(packages: List[str]) -> None:
    """Uninstall specified packages."""
    if not packages:
        print("No packages to uninstall.")
        return

    print(f"Uninstalling {len(packages)} packages...")

    for package in packages:
        try:
            print(f"Uninstalling {package}...")
            subprocess.run([sys.executable, "-m", "pip", "uninstall", package, "-y"],
                         check=True, capture_output=True)
            print(f"✓ {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to uninstall {package}: {e}")
        except Exception as e:
            print(f"✗ Error with {package}: {e}")

def main():
    """Main function."""
    print("Medical_KG_rev Optional Package Uninstaller")
    print("=" * 50)

    # Get currently installed packages
    installed = get_installed_packages()
    print(f"Found {len(installed)} installed packages")

    # Find packages to uninstall
    to_uninstall = []
    for package in OPTIONAL_PACKAGES:
        if package.lower() in installed:
            to_uninstall.append(package)

    print(f"\nFound {len(to_uninstall)} optional packages to uninstall:")
    for package in to_uninstall:
        print(f"  - {package}")

    if not to_uninstall:
        print("\nNo optional packages found to uninstall.")
        return

    # Confirm before proceeding
    response = input(f"\nProceed with uninstalling {len(to_uninstall)} packages? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Uninstall packages
    uninstall_packages(to_uninstall)

    print(f"\n✓ Uninstallation complete!")
    print("You can now reinstall only the packages you actually need.")

if __name__ == "__main__":
    main()
