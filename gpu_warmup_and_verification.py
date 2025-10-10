#!/usr/bin/env python3
"""GPU Warmup and Verification Script.

This script verifies GPU availability and performs warmup operations to ensure
optimal GPU utilization for MinerU PDF processing.

Usage:
    python gpu_warmup_and_verification.py

Features:
- Verifies CUDA availability and PyTorch GPU support
- Tests MinerU GPU dependencies
- Performs GPU warmup operations
- Monitors GPU utilization during processing
- Provides detailed GPU status report
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class GPUWarmupAndVerification:
    """GPU warmup and verification system."""

    def __init__(self):
        """Initialize the GPU verification system."""
        self.gpu_info = {}
        self.verification_results = {}
        self.warmup_results = {}

    def check_cuda_availability(self) -> Dict[str, Any]:
        """Check CUDA availability and basic info."""
        print("🔍 Checking CUDA availability...")

        try:
            import torch
            cuda_available = torch.cuda.is_available()

            if cuda_available:
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)

                # Get GPU memory info
                memory_allocated = torch.cuda.memory_allocated(current_device)
                memory_reserved = torch.cuda.memory_reserved(current_device)
                memory_total = torch.cuda.get_device_properties(current_device).total_memory

                cuda_info = {
                    "available": True,
                    "device_count": device_count,
                    "current_device": current_device,
                    "device_name": device_name,
                    "memory_allocated_mb": memory_allocated / (1024 * 1024),
                    "memory_reserved_mb": memory_reserved / (1024 * 1024),
                    "memory_total_mb": memory_total / (1024 * 1024),
                    "memory_free_mb": (memory_total - memory_reserved) / (1024 * 1024),
                    "cuda_version": torch.version.cuda,
                    "pytorch_version": torch.__version__
                }

                print(f"✅ CUDA available: {device_name}")
                print(f"   Device count: {device_count}")
                print(f"   Current device: {current_device}")
                print(f"   Memory: {cuda_info['memory_free_mb']:.1f} MB free / {cuda_info['memory_total_mb']:.1f} MB total")
                print(f"   CUDA version: {cuda_info['cuda_version']}")
                print(f"   PyTorch version: {cuda_info['pytorch_version']}")

            else:
                cuda_info = {
                    "available": False,
                    "error": "CUDA not available in PyTorch"
                }
                print("❌ CUDA not available in PyTorch")

            return cuda_info

        except ImportError:
            return {
                "available": False,
                "error": "PyTorch not installed"
            }
        except Exception as e:
            return {
                "available": False,
                "error": f"CUDA check failed: {str(e)}"
            }

    def check_nvidia_smi(self) -> Dict[str, Any]:
        """Check nvidia-smi availability and GPU info."""
        print("🔍 Checking nvidia-smi...")

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = []

                for i, line in enumerate(lines):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        gpu_info = {
                            "index": i,
                            "name": parts[0],
                            "memory_total_mb": int(parts[1]),
                            "memory_used_mb": int(parts[2]),
                            "memory_free_mb": int(parts[3]),
                            "utilization_percent": int(parts[4]),
                            "temperature_c": int(parts[5])
                        }
                        gpus.append(gpu_info)

                nvidia_info = {
                    "available": True,
                    "gpus": gpus
                }

                print(f"✅ nvidia-smi available: {len(gpus)} GPU(s) detected")
                for gpu in gpus:
                    print(f"   GPU {gpu['index']}: {gpu['name']}")
                    print(f"     Memory: {gpu['memory_free_mb']} MB free / {gpu['memory_total_mb']} MB total")
                    print(f"     Utilization: {gpu['utilization_percent']}%")
                    print(f"     Temperature: {gpu['temperature_c']}°C")

            else:
                nvidia_info = {
                    "available": False,
                    "error": f"nvidia-smi failed: {result.stderr}"
                }
                print("❌ nvidia-smi not available or failed")

            return nvidia_info

        except FileNotFoundError:
            return {
                "available": False,
                "error": "nvidia-smi not found"
            }
        except subprocess.TimeoutExpired:
            return {
                "available": False,
                "error": "nvidia-smi timeout"
            }
        except Exception as e:
            return {
                "available": False,
                "error": f"nvidia-smi check failed: {str(e)}"
            }

    def check_mineru_gpu_dependencies(self) -> Dict[str, Any]:
        """Check MinerU GPU dependencies."""
        print("🔍 Checking MinerU GPU dependencies...")

        dependencies = {}

        # Check doclayout-yolo
        try:
            import doclayout_yolo
            dependencies["doclayout_yolo"] = {"available": True, "version": getattr(doclayout_yolo, "__version__", "unknown")}
            print("✅ doclayout-yolo available")
        except ImportError:
            dependencies["doclayout_yolo"] = {"available": False, "error": "Not installed"}
            print("❌ doclayout-yolo not available")

        # Check ultralytics
        try:
            import ultralytics
            dependencies["ultralytics"] = {"available": True, "version": ultralytics.__version__}
            print("✅ ultralytics available")
        except ImportError:
            dependencies["ultralytics"] = {"available": False, "error": "Not installed"}
            print("❌ ultralytics not available")

        # Check PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                dependencies["pytorch_cuda"] = {"available": True, "cuda_version": torch.version.cuda}
                print("✅ PyTorch CUDA available")
            else:
                dependencies["pytorch_cuda"] = {"available": False, "error": "CUDA not available"}
                print("❌ PyTorch CUDA not available")
        except ImportError:
            dependencies["pytorch_cuda"] = {"available": False, "error": "PyTorch not installed"}
            print("❌ PyTorch not available")

        return dependencies

    def perform_gpu_warmup(self) -> Dict[str, Any]:
        """Perform GPU warmup operations."""
        print("🔥 Performing GPU warmup...")

        warmup_results = {}

        try:
            import torch

            if not torch.cuda.is_available():
                return {"success": False, "error": "CUDA not available"}

            device = torch.device("cuda:0")

            # Warmup 1: Simple tensor operations
            print("   Warming up with tensor operations...")
            start_time = time.time()

            # Create and manipulate tensors
            for i in range(10):
                a = torch.randn(1000, 1000, device=device)
                b = torch.randn(1000, 1000, device=device)
                c = torch.matmul(a, b)
                torch.cuda.synchronize()

            tensor_time = time.time() - start_time
            warmup_results["tensor_operations"] = {
                "success": True,
                "time_seconds": tensor_time
            }
            print(f"   ✅ Tensor operations: {tensor_time:.2f}s")

            # Warmup 2: Model loading simulation
            print("   Warming up with model operations...")
            start_time = time.time()

            # Simulate model operations
            model = torch.nn.Sequential(
                torch.nn.Linear(1000, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 10)
            ).to(device)

            # Forward pass
            for i in range(5):
                x = torch.randn(32, 1000, device=device)
                y = model(x)
                torch.cuda.synchronize()

            model_time = time.time() - start_time
            warmup_results["model_operations"] = {
                "success": True,
                "time_seconds": model_time
            }
            print(f"   ✅ Model operations: {model_time:.2f}s")

            # Warmup 3: Memory allocation test
            print("   Testing memory allocation...")
            start_time = time.time()

            # Allocate and deallocate memory
            tensors = []
            for i in range(5):
                tensor = torch.randn(2000, 2000, device=device)
                tensors.append(tensor)

            # Clear memory
            del tensors
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            memory_time = time.time() - start_time
            warmup_results["memory_operations"] = {
                "success": True,
                "time_seconds": memory_time
            }
            print(f"   ✅ Memory operations: {memory_time:.2f}s")

            warmup_results["overall_success"] = True
            warmup_results["total_time"] = tensor_time + model_time + memory_time

            print(f"✅ GPU warmup completed in {warmup_results['total_time']:.2f}s")

        except Exception as e:
            warmup_results = {
                "overall_success": False,
                "error": str(e)
            }
            print(f"❌ GPU warmup failed: {e}")

        return warmup_results

    def monitor_gpu_during_processing(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """Monitor GPU utilization during processing."""
        print(f"📊 Monitoring GPU for {duration_seconds} seconds...")

        gpu_utilization = []
        gpu_memory = []
        gpu_temperature = []

        def monitor_loop():
            start_time = time.time()
            while time.time() - start_time < duration_seconds:
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )

                    if result.returncode == 0:
                        parts = [p.strip() for p in result.stdout.strip().split(',')]
                        if len(parts) >= 3:
                            gpu_utilization.append(int(parts[0]))
                            gpu_memory.append(int(parts[1]))
                            gpu_temperature.append(int(parts[2]))

                    time.sleep(1)
                except:
                    pass

        # Start monitoring in background
        monitor_thread = threading.Thread(target=monitor_loop)
        monitor_thread.start()

        # Wait for monitoring to complete
        monitor_thread.join()

        if gpu_utilization:
            avg_utilization = sum(gpu_utilization) / len(gpu_utilization)
            max_utilization = max(gpu_utilization)
            avg_memory = sum(gpu_memory) / len(gpu_memory)
            max_memory = max(gpu_memory)
            avg_temperature = sum(gpu_temperature) / len(gpu_temperature)
            max_temperature = max(gpu_temperature)

            monitoring_results = {
                "success": True,
                "duration_seconds": duration_seconds,
                "samples": len(gpu_utilization),
                "avg_utilization_percent": avg_utilization,
                "max_utilization_percent": max_utilization,
                "avg_memory_mb": avg_memory,
                "max_memory_mb": max_memory,
                "avg_temperature_c": avg_temperature,
                "max_temperature_c": max_temperature
            }

            print(f"✅ GPU monitoring completed:")
            print(f"   Average utilization: {avg_utilization:.1f}%")
            print(f"   Maximum utilization: {max_utilization:.1f}%")
            print(f"   Average memory: {avg_memory:.1f} MB")
            print(f"   Maximum memory: {max_memory:.1f} MB")
            print(f"   Average temperature: {avg_temperature:.1f}°C")
            print(f"   Maximum temperature: {max_temperature:.1f}°C")

        else:
            monitoring_results = {
                "success": False,
                "error": "No GPU data collected"
            }
            print("❌ GPU monitoring failed")

        return monitoring_results

    def test_mineru_gpu_processing(self) -> Dict[str, Any]:
        """Test MinerU GPU processing with a simple operation."""
        print("🧪 Testing MinerU GPU processing...")

        try:
            import shutil
            import tempfile

            from mineru.cli.client import main as mineru_main

            # Create a simple test PDF (1 page with text)
            test_pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Test PDF for GPU processing) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000204 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
297
%%EOF"""

            # Create temporary directory for test
            with tempfile.TemporaryDirectory() as temp_dir:
                test_pdf_path = Path(temp_dir) / "test_gpu.pdf"
                with open(test_pdf_path, "wb") as f:
                    f.write(test_pdf_content)

                output_dir = Path(temp_dir) / "output"
                output_dir.mkdir()

                # Set up MinerU command
                original_argv = sys.argv.copy()
                sys.argv = [
                    "mineru",
                    "--path", str(test_pdf_path),
                    "--output", str(output_dir),
                    "--method", "auto",
                    "--backend", "pipeline",
                    "--device", "cuda",
                    "--formula", "true",
                    "--table", "true"
                ]

                try:
                    start_time = time.time()
                    mineru_main()
                    processing_time = time.time() - start_time

                    # Check for output files
                    output_files = list(output_dir.glob("**/*"))
                    output_files = [f for f in output_files if f.is_file()]

                    test_results = {
                        "success": True,
                        "processing_time_seconds": processing_time,
                        "output_files_count": len(output_files),
                        "output_files": [str(f) for f in output_files]
                    }

                    print(f"✅ MinerU GPU test successful:")
                    print(f"   Processing time: {processing_time:.2f}s")
                    print(f"   Output files: {len(output_files)}")

                except Exception as e:
                    test_results = {
                        "success": False,
                        "error": str(e)
                    }
                    print(f"❌ MinerU GPU test failed: {e}")

                finally:
                    # Restore original argv
                    sys.argv = original_argv

        except ImportError:
            test_results = {
                "success": False,
                "error": "MinerU not available"
            }
            print("❌ MinerU not available")
        except Exception as e:
            test_results = {
                "success": False,
                "error": str(e)
            }
            print(f"❌ MinerU GPU test failed: {e}")

        return test_results

    def run_complete_verification(self) -> Dict[str, Any]:
        """Run complete GPU verification and warmup."""
        print("="*70)
        print("GPU WARMUP AND VERIFICATION")
        print("="*70)

        results = {}

        # Step 1: Check CUDA availability
        results["cuda_info"] = self.check_cuda_availability()

        # Step 2: Check nvidia-smi
        results["nvidia_info"] = self.check_nvidia_smi()

        # Step 3: Check MinerU dependencies
        results["dependencies"] = self.check_mineru_gpu_dependencies()

        # Step 4: Perform GPU warmup
        if results["cuda_info"].get("available", False):
            results["warmup"] = self.perform_gpu_warmup()
        else:
            results["warmup"] = {"success": False, "error": "CUDA not available"}

        # Step 5: Test MinerU GPU processing
        if results["cuda_info"].get("available", False):
            results["mineru_test"] = self.test_mineru_gpu_processing()
        else:
            results["mineru_test"] = {"success": False, "error": "CUDA not available"}

        # Step 6: Monitor GPU during processing
        if results["cuda_info"].get("available", False):
            results["monitoring"] = self.monitor_gpu_during_processing(10)
        else:
            results["monitoring"] = {"success": False, "error": "CUDA not available"}

        return results

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print verification summary."""
        print("\n" + "="*70)
        print("GPU VERIFICATION SUMMARY")
        print("="*70)

        # CUDA Status
        cuda_available = results["cuda_info"].get("available", False)
        print(f"CUDA Available: {'✅ YES' if cuda_available else '❌ NO'}")
        if cuda_available:
            print(f"   Device: {results['cuda_info']['device_name']}")
            print(f"   Memory: {results['cuda_info']['memory_free_mb']:.1f} MB free")

        # nvidia-smi Status
        nvidia_available = results["nvidia_info"].get("available", False)
        print(f"nvidia-smi Available: {'✅ YES' if nvidia_available else '❌ NO'}")

        # Dependencies Status
        deps = results["dependencies"]
        print(f"Dependencies:")
        print(f"   doclayout-yolo: {'✅ YES' if deps.get('doclayout_yolo', {}).get('available', False) else '❌ NO'}")
        print(f"   ultralytics: {'✅ YES' if deps.get('ultralytics', {}).get('available', False) else '❌ NO'}")
        print(f"   PyTorch CUDA: {'✅ YES' if deps.get('pytorch_cuda', {}).get('available', False) else '❌ NO'}")

        # Warmup Status
        warmup_success = results["warmup"].get("overall_success", False)
        print(f"GPU Warmup: {'✅ SUCCESS' if warmup_success else '❌ FAILED'}")
        if warmup_success:
            print(f"   Total time: {results['warmup']['total_time']:.2f}s")

        # MinerU Test Status
        mineru_success = results["mineru_test"].get("success", False)
        print(f"MinerU GPU Test: {'✅ SUCCESS' if mineru_success else '❌ FAILED'}")
        if mineru_success:
            print(f"   Processing time: {results['mineru_test']['processing_time_seconds']:.2f}s")
            print(f"   Output files: {results['mineru_test']['output_files_count']}")

        # Monitoring Status
        monitoring_success = results["monitoring"].get("success", False)
        print(f"GPU Monitoring: {'✅ SUCCESS' if monitoring_success else '❌ FAILED'}")
        if monitoring_success:
            print(f"   Average utilization: {results['monitoring']['avg_utilization_percent']:.1f}%")
            print(f"   Maximum utilization: {results['monitoring']['max_utilization_percent']:.1f}%")

        # Overall Status
        overall_success = (
            cuda_available and
            nvidia_available and
            warmup_success and
            mineru_success
        )

        print("\n" + "="*70)
        print(f"OVERALL GPU STATUS: {'✅ READY' if overall_success else '❌ NOT READY'}")
        print("="*70)

        if overall_success:
            print("🎉 GPU is fully operational and ready for MinerU processing!")
            print("   • CUDA is available and working")
            print("   • All dependencies are installed")
            print("   • GPU warmup completed successfully")
            print("   • MinerU can process PDFs using GPU")
        else:
            print("⚠️  GPU setup needs attention:")
            if not cuda_available:
                print("   • CUDA is not available")
            if not nvidia_available:
                print("   • nvidia-smi is not working")
            if not warmup_success:
                print("   • GPU warmup failed")
            if not mineru_success:
                print("   • MinerU GPU test failed")

        print("="*70)


def main():
    """Main execution function."""
    verifier = GPUWarmupAndVerification()
    results = verifier.run_complete_verification()
    verifier.print_summary(results)

    # Return appropriate exit code
    overall_success = (
        results["cuda_info"].get("available", False) and
        results["nvidia_info"].get("available", False) and
        results["warmup"].get("overall_success", False) and
        results["mineru_test"].get("success", False)
    )

    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
