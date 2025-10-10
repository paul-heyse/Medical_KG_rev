#!/usr/bin/env python3
"""Setup script for full MinerU PDF processing capability.

This script sets up the complete MinerU environment including:
1. Installing MinerU CLI
2. Downloading required model weights
3. Starting vLLM server
4. Testing PDF processing with downloaded papers

Usage:
    python setup_mineru_full.py

Requirements:
    - NVIDIA GPU with CUDA support
    - Sufficient disk space for model weights
    - Internet connection for downloading models
"""
import subprocess
import sys
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_command(cmd, description, check=True):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        if result.stderr and not check:
            print(f"   Warning: {result.stderr.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False


def check_gpu():
    """Check GPU availability."""
    print("🔍 Checking GPU availability...")
    try:
        result = subprocess.run("nvidia-smi", capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU available")
            return True
        else:
            print("❌ No GPU available")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        return False


def install_mineru():
    """Install MinerU CLI."""
    print("📦 Installing MinerU...")

    # Check if already installed
    try:
        result = subprocess.run("magic-pdf --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ MinerU already installed")
            return True
    except:
        pass

    # Install MinerU
    install_cmd = "pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com"
    return run_command(install_cmd, "Installing MinerU CLI")


def download_models():
    """Download required model weights."""
    print("📥 Downloading model weights...")

    # Create models directory
    models_dir = Path.home() / ".cache" / "mineru"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Download models using MinerU's built-in downloader
    download_cmd = "magic-pdf download-models"
    return run_command(download_cmd, "Downloading model weights", check=False)


def start_vllm_server():
    """Start vLLM server for MinerU."""
    print("🚀 Starting vLLM server...")

    # Check if vLLM server is already running
    try:
        result = subprocess.run(
            "curl -s http://localhost:8000/health", shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✅ vLLM server already running")
            return True
    except:
        pass

    # Start vLLM server in background
    vllm_cmd = """
    vllm serve qwen2.5-vl-7b-instruct \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.8 \
        --max-model-len 8192 \
        --trust-remote-code &
    """

    print("🔄 Starting vLLM server (this may take several minutes)...")
    try:
        subprocess.Popen(vllm_cmd, shell=True)

        # Wait for server to start
        for i in range(60):  # Wait up to 5 minutes
            time.sleep(5)
            try:
                result = subprocess.run(
                    "curl -s http://localhost:8000/health",
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    print("✅ vLLM server started successfully")
                    return True
            except:
                pass
            print(f"   Waiting for server... ({i+1}/60)")

        print("❌ vLLM server failed to start within timeout")
        return False

    except Exception as e:
        print(f"❌ Failed to start vLLM server: {e}")
        return False


def test_mineru_processing():
    """Test MinerU processing with downloaded PDFs."""
    print("🧪 Testing MinerU processing...")

    pdf_dir = Path("random_papers_output/pdfs")
    if not pdf_dir.exists():
        print("❌ No PDFs found for testing")
        return False

    # Create output directory
    output_dir = Path("mineru_test_output")
    output_dir.mkdir(exist_ok=True)

    # Process PDFs with MinerU
    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"🔄 Processing {pdf_file.name}...")

        # Use MinerU CLI to process PDF
        cmd = f"magic-pdf --path {pdf_file} --output-dir {output_dir} --method auto"

        if run_command(cmd, f"Processing {pdf_file.name}", check=False):
            print(f"✅ Successfully processed {pdf_file.name}")
        else:
            print(f"❌ Failed to process {pdf_file.name}")

    # Check results
    results = list(output_dir.glob("**/*.md")) + list(output_dir.glob("**/*.json"))
    if results:
        print(f"✅ Generated {len(results)} output files")
        for result in results:
            print(f"   {result}")
        return True
    else:
        print("❌ No output files generated")
        return False


def main():
    """Main setup workflow."""
    print("=" * 70)
    print("MINERU FULL SETUP")
    print("=" * 70)

    # Step 1: Check GPU
    if not check_gpu():
        print("❌ GPU required for MinerU processing")
        return False

    # Step 2: Install MinerU
    if not install_mineru():
        print("❌ Failed to install MinerU")
        return False

    # Step 3: Download models
    if not download_models():
        print("⚠️  Model download failed, but continuing...")

    # Step 4: Start vLLM server
    if not start_vllm_server():
        print("❌ Failed to start vLLM server")
        return False

    # Step 5: Test processing
    if not test_mineru_processing():
        print("❌ MinerU processing test failed")
        return False

    print("\n" + "=" * 70)
    print("✅ MINERU SETUP COMPLETE!")
    print("=" * 70)
    print("MinerU is now ready for full PDF processing.")
    print("You can now process PDFs using:")
    print("  magic-pdf --path <pdf_file> --output-dir <output_dir> --method auto")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
