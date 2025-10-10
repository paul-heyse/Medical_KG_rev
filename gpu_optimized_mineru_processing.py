#!/usr/bin/env python3
"""GPU-Optimized MinerU Processing Script.

This script processes PDFs with MinerU using GPU optimization, warmup,
and real-time monitoring to ensure maximum GPU utilization.

Usage:
    python gpu_optimized_mineru_processing.py

Features:
- GPU warmup before processing
- Real-time GPU monitoring during processing
- Optimized batch processing
- Memory management
- Performance metrics
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class GPUOptimizedMineruProcessor:
    """GPU-optimized MinerU processor with monitoring."""

    def __init__(
        self, input_dir: str = "random_papers_output/pdfs", output_dir: str = "gpu_optimized_output"
    ):
        """Initialize the GPU-optimized processor."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.markdown_dir = self.output_dir / "markdown"
        self.json_dir = self.output_dir / "json"
        self.reports_dir = self.output_dir / "reports"

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.markdown_dir.mkdir(exist_ok=True)
        self.json_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

        # GPU monitoring
        self.gpu_metrics = []
        self.monitoring_active = False

        # Results tracking
        self.results = {
            "pdfs_found": 0,
            "pdfs_processed": 0,
            "processing_errors": 0,
            "processing_details": [],
            "gpu_utilization": [],
            "processing_times": [],
        }

    def gpu_warmup(self) -> bool:
        """Perform GPU warmup operations."""
        print("🔥 Performing GPU warmup...")

        try:
            import torch

            if not torch.cuda.is_available():
                print("❌ CUDA not available for warmup")
                return False

            device = torch.device("cuda:0")

            # Warmup operations
            print("   Warming up GPU with tensor operations...")
            for i in range(5):
                a = torch.randn(2000, 2000, device=device)
                b = torch.randn(2000, 2000, device=device)
                c = torch.matmul(a, b)
                torch.cuda.synchronize()

            # Memory allocation warmup
            print("   Warming up GPU memory...")
            tensors = []
            for i in range(3):
                tensor = torch.randn(3000, 3000, device=device)
                tensors.append(tensor)

            # Clear memory
            del tensors
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            print("✅ GPU warmup completed")
            return True

        except Exception as e:
            print(f"❌ GPU warmup failed: {e}")
            return False

    def start_gpu_monitoring(self) -> None:
        """Start GPU monitoring in background thread."""
        self.monitoring_active = True
        self.gpu_metrics = []

        def monitor_loop():
            while self.monitoring_active:
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )

                    if result.returncode == 0:
                        parts = [p.strip() for p in result.stdout.strip().split(",")]
                        if len(parts) >= 4:
                            metric = {
                                "timestamp": time.time(),
                                "utilization_percent": int(parts[0]),
                                "memory_used_mb": int(parts[1]),
                                "temperature_c": int(parts[2]),
                                "power_draw_w": float(parts[3]) if parts[3] != "N/A" else 0,
                            }
                            self.gpu_metrics.append(metric)

                    time.sleep(1)
                except:
                    pass

        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.start()
        print("📊 GPU monitoring started")

    def stop_gpu_monitoring(self) -> dict[str, Any]:
        """Stop GPU monitoring and return summary."""
        self.monitoring_active = False
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join()

        if not self.gpu_metrics:
            return {"error": "No GPU metrics collected"}

        # Calculate statistics
        utilizations = [m["utilization_percent"] for m in self.gpu_metrics]
        memories = [m["memory_used_mb"] for m in self.gpu_metrics]
        temperatures = [m["temperature_c"] for m in self.gpu_metrics]
        powers = [m["power_draw_w"] for m in self.gpu_metrics if m["power_draw_w"] > 0]

        summary = {
            "duration_seconds": len(self.gpu_metrics),
            "samples": len(self.gpu_metrics),
            "avg_utilization_percent": sum(utilizations) / len(utilizations),
            "max_utilization_percent": max(utilizations),
            "min_utilization_percent": min(utilizations),
            "avg_memory_mb": sum(memories) / len(memories),
            "max_memory_mb": max(memories),
            "min_memory_mb": min(memories),
            "avg_temperature_c": sum(temperatures) / len(temperatures),
            "max_temperature_c": max(temperatures),
            "min_temperature_c": min(temperatures),
            "avg_power_w": sum(powers) / len(powers) if powers else 0,
            "max_power_w": max(powers) if powers else 0,
            "min_power_w": min(powers) if powers else 0,
        }

        print("📊 GPU monitoring stopped")
        return summary

    def process_pdf_with_gpu_optimization(self, pdf_path: Path) -> dict[str, Any]:
        """Process a PDF with GPU optimization."""
        try:
            print(f"🔄 Processing PDF with GPU optimization: {pdf_path.name}")

            # Start GPU monitoring for this PDF
            self.start_gpu_monitoring()

            # Process with MinerU
            from mineru.cli.client import main as mineru_main

            # Set up command with GPU optimization
            original_argv = sys.argv.copy()
            sys.argv = [
                "mineru",
                "--path",
                str(pdf_path),
                "--output",
                str(self.output_dir),
                "--method",
                "auto",
                "--backend",
                "pipeline",
                "--device",
                "cuda",
                "--formula",
                "true",
                "--table",
                "true",
            ]

            start_time = time.time()
            mineru_main()
            processing_time = time.time() - start_time

            # Stop monitoring and get metrics
            gpu_summary = self.stop_gpu_monitoring()

            # Restore original argv
            sys.argv = original_argv

            # Find output files
            pdf_id = pdf_path.stem
            output_files = []
            for output_dir in self.output_dir.glob(f"**/{pdf_id}"):
                if output_dir.is_dir():
                    auto_dir = output_dir / "auto"
                    if auto_dir.exists():
                        for file_path in auto_dir.iterdir():
                            if file_path.is_file():
                                output_files.append(
                                    {
                                        "path": str(file_path),
                                        "size_bytes": file_path.stat().st_size,
                                        "type": file_path.suffix,
                                    }
                                )

            # Generate results
            result = {
                "pdf_file": str(pdf_path),
                "processing_time_seconds": processing_time,
                "output_files": output_files,
                "gpu_metrics": gpu_summary,
                "success": True,
            }

            print(f"✅ Processed: {pdf_path.name} ({processing_time:.2f}s)")
            if "error" not in gpu_summary:
                print(
                    f"   GPU utilization: {gpu_summary['avg_utilization_percent']:.1f}% avg, {gpu_summary['max_utilization_percent']:.1f}% max"
                )
                print(
                    f"   GPU memory: {gpu_summary['avg_memory_mb']:.1f} MB avg, {gpu_summary['max_memory_mb']:.1f} MB max"
                )
                print(
                    f"   GPU temperature: {gpu_summary['avg_temperature_c']:.1f}°C avg, {gpu_summary['max_temperature_c']:.1f}°C max"
                )
                if gpu_summary["avg_power_w"] > 0:
                    print(
                        f"   GPU power: {gpu_summary['avg_power_w']:.1f}W avg, {gpu_summary['max_power_w']:.1f}W max"
                    )
            print(f"   Output files: {len(output_files)}")

            return result

        except Exception as e:
            # Stop monitoring if it's running
            if self.monitoring_active:
                self.stop_gpu_monitoring()

            print(f"❌ Error processing {pdf_path.name}: {e}")
            return {"pdf_file": str(pdf_path), "error": str(e), "success": False}

    def process_all_pdfs(self) -> None:
        """Process all PDFs with GPU optimization."""
        print("=" * 70)
        print("GPU-OPTIMIZED MINERU PROCESSING")
        print("=" * 70)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print("-" * 70)

        # Step 1: GPU warmup
        if not self.gpu_warmup():
            print("❌ GPU warmup failed, continuing without optimization")

        # Step 2: Find PDFs
        pdfs = list(self.input_dir.glob("*.pdf"))
        self.results["pdfs_found"] = len(pdfs)

        if not pdfs:
            print("❌ No PDFs found")
            return

        print(f"📁 Found {len(pdfs)} PDF files")
        for pdf in pdfs:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            print(f"   {pdf.name} ({size_mb:.1f} MB)")

        # Step 3: Process each PDF
        print(f"\n🔄 Processing {len(pdfs)} PDFs with GPU optimization...")
        for pdf_path in pdfs:
            try:
                result = self.process_pdf_with_gpu_optimization(pdf_path)
                self.results["processing_details"].append(result)

                if result["success"]:
                    self.results["pdfs_processed"] += 1
                    self.results["processing_times"].append(result["processing_time_seconds"])

                    # Store GPU metrics
                    if "gpu_metrics" in result and "error" not in result["gpu_metrics"]:
                        self.results["gpu_utilization"].append(result["gpu_metrics"])
                else:
                    self.results["processing_errors"] += 1

            except Exception as e:
                print(f"❌ Unexpected error processing {pdf_path.name}: {e}")
                self.results["processing_errors"] += 1
                self.results["processing_details"].append(
                    {"pdf_file": str(pdf_path), "error": str(e), "success": False}
                )

        # Step 4: Generate summary report
        self.generate_summary_report()

    def generate_summary_report(self) -> None:
        """Generate a comprehensive summary report."""
        report_file = self.reports_dir / "gpu_optimized_summary.json"

        # Calculate overall statistics
        total_processing_time = (
            sum(self.results["processing_times"]) if self.results["processing_times"] else 0
        )
        avg_processing_time = (
            total_processing_time / len(self.results["processing_times"])
            if self.results["processing_times"]
            else 0
        )

        # Calculate GPU statistics
        gpu_stats = {}
        if self.results["gpu_utilization"]:
            all_utilizations = []
            all_memories = []
            all_temperatures = []
            all_powers = []

            for gpu_metric in self.results["gpu_utilization"]:
                all_utilizations.append(gpu_metric["avg_utilization_percent"])
                all_memories.append(gpu_metric["avg_memory_mb"])
                all_temperatures.append(gpu_metric["avg_temperature_c"])
                if gpu_metric["avg_power_w"] > 0:
                    all_powers.append(gpu_metric["avg_power_w"])

            gpu_stats = {
                "avg_utilization_percent": sum(all_utilizations) / len(all_utilizations),
                "max_utilization_percent": max(all_utilizations),
                "avg_memory_mb": sum(all_memories) / len(all_memories),
                "max_memory_mb": max(all_memories),
                "avg_temperature_c": sum(all_temperatures) / len(all_temperatures),
                "max_temperature_c": max(all_temperatures),
                "avg_power_w": sum(all_powers) / len(all_powers) if all_powers else 0,
                "max_power_w": max(all_powers) if all_powers else 0,
            }

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_type": "GPU-Optimized MinerU",
            "input_directory": str(self.input_dir),
            "output_directory": str(self.output_dir),
            "results": self.results,
            "statistics": {
                "pdfs_found": self.results["pdfs_found"],
                "pdfs_processed": self.results["pdfs_processed"],
                "processing_errors": self.results["processing_errors"],
                "success_rate": (
                    self.results["pdfs_processed"] / self.results["pdfs_found"]
                    if self.results["pdfs_found"] > 0
                    else 0
                ),
                "total_processing_time_seconds": total_processing_time,
                "avg_processing_time_seconds": avg_processing_time,
                "gpu_statistics": gpu_stats,
            },
        }

        # Save report
        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n" + "=" * 70)
        print("GPU-OPTIMIZED PROCESSING SUMMARY")
        print("=" * 70)
        print(f"PDFs found: {self.results['pdfs_found']}")
        print(f"PDFs processed: {self.results['pdfs_processed']}")
        print(f"Processing errors: {self.results['processing_errors']}")

        if self.results["pdfs_found"] > 0:
            success_rate = self.results["pdfs_processed"] / self.results["pdfs_found"]
            print(f"Success rate: {success_rate:.1%}")

        if self.results["processing_times"]:
            print(f"Total processing time: {total_processing_time:.2f}s")
            print(f"Average processing time: {avg_processing_time:.2f}s per PDF")

        if gpu_stats:
            print("\nGPU Performance:")
            print(f"   Average utilization: {gpu_stats['avg_utilization_percent']:.1f}%")
            print(f"   Maximum utilization: {gpu_stats['max_utilization_percent']:.1f}%")
            print(f"   Average memory usage: {gpu_stats['avg_memory_mb']:.1f} MB")
            print(f"   Maximum memory usage: {gpu_stats['max_memory_mb']:.1f} MB")
            print(f"   Average temperature: {gpu_stats['avg_temperature_c']:.1f}°C")
            print(f"   Maximum temperature: {gpu_stats['max_temperature_c']:.1f}°C")
            if gpu_stats["avg_power_w"] > 0:
                print(f"   Average power draw: {gpu_stats['avg_power_w']:.1f}W")
                print(f"   Maximum power draw: {gpu_stats['max_power_w']:.1f}W")

        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"📊 Summary report: {report_file}")
        print("=" * 70)


def main():
    """Main execution function."""
    # Initialize processor
    processor = GPUOptimizedMineruProcessor()

    # Process PDFs with GPU optimization
    processor.process_all_pdfs()


if __name__ == "__main__":
    main()
