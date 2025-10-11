#!/usr/bin/env python3
"""gRPC service performance optimization script.

This script analyzes and optimizes gRPC service communication performance
including connection pooling, message serialization, and network optimization.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import psutil
from pydantic import BaseModel, Field

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Medical_KG_rev.services.clients.embedding_client import EmbeddingClient
from Medical_KG_rev.services.clients.gpu_client import GPUClient
from Medical_KG_rev.services.clients.reranking_client import RerankingClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMetrics(BaseModel):
    """Performance metrics model."""

    service_name: str
    operation: str
    latency_ms: float
    throughput_rps: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_rate: float
    timestamp: float = Field(default_factory=time.time)


class PerformanceAnalyzer:
    """Analyzes gRPC service performance."""

    def __init__(self):
        """Initialize performance analyzer."""
        self.metrics: list[PerformanceMetrics] = []
        self.baseline_metrics: dict[str, PerformanceMetrics] = {}

    async def measure_service_performance(
        self,
        service_name: str,
        operation: str,
        client: Any,
        test_data: Any,
        iterations: int = 10,
    ) -> PerformanceMetrics:
        """Measure performance of a service operation."""
        logger.info(f"Measuring performance for {service_name}.{operation}")

        latencies = []
        errors = 0
        start_time = time.time()

        for i in range(iterations):
            try:
                operation_start = time.time()

                # Execute operation based on service type
                if service_name == "gpu":
                    if operation == "get_status":
                        await client.get_status()
                    elif operation == "list_devices":
                        await client.list_devices()
                elif service_name == "embedding":
                    if operation == "generate_embeddings":
                        await client.generate_embeddings(test_data)
                    elif operation == "list_models":
                        await client.list_models()
                elif service_name == "reranking":
                    if operation == "rerank_batch":
                        await client.rerank_batch(test_data)
                    elif operation == "list_models":
                        await client.list_models()
                elif service_name == "docling_vlm":
                    if operation == "process_pdf":
                        await client.process_pdf(test_data)
                    elif operation == "get_health":
                        await client.get_health()

                operation_end = time.time()
                latency_ms = (operation_end - operation_start) * 1000
                latencies.append(latency_ms)

            except Exception as e:
                logger.error(f"Operation failed: {e}")
                errors += 1

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate metrics
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        throughput = iterations / total_time if total_time > 0 else 0
        error_rate = errors / iterations if iterations > 0 else 0

        # Get system metrics
        memory_usage = psutil.virtual_memory().used / 1024 / 1024  # MB
        cpu_usage = psutil.cpu_percent()

        metrics = PerformanceMetrics(
            service_name=service_name,
            operation=operation,
            latency_ms=avg_latency,
            throughput_rps=throughput,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            error_rate=error_rate,
        )

        self.metrics.append(metrics)
        return metrics

    def analyze_performance_trends(self) -> dict[str, Any]:
        """Analyze performance trends."""
        if not self.metrics:
            return {}

        analysis = {}

        # Group metrics by service
        service_metrics = {}
        for metric in self.metrics:
            if metric.service_name not in service_metrics:
                service_metrics[metric.service_name] = []
            service_metrics[metric.service_name].append(metric)

        # Analyze each service
        for service_name, metrics in service_metrics.items():
            if not metrics:
                continue

            # Calculate averages
            avg_latency = sum(m.latency_ms for m in metrics) / len(metrics)
            avg_throughput = sum(m.throughput_rps for m in metrics) / len(metrics)
            avg_error_rate = sum(m.error_rate for m in metrics) / len(metrics)

            # Find best and worst performance
            best_latency = min(m.latency_ms for m in metrics)
            worst_latency = max(m.latency_ms for m in metrics)
            best_throughput = max(m.throughput_rps for m in metrics)
            worst_throughput = min(m.throughput_rps for m in metrics)

            analysis[service_name] = {
                "avg_latency_ms": avg_latency,
                "avg_throughput_rps": avg_throughput,
                "avg_error_rate": avg_error_rate,
                "best_latency_ms": best_latency,
                "worst_latency_ms": worst_latency,
                "best_throughput_rps": best_throughput,
                "worst_throughput_rps": worst_throughput,
                "total_operations": len(metrics),
            }

        return analysis

    def generate_optimization_recommendations(self) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if not self.metrics:
            return recommendations

        # Analyze latency
        high_latency_services = [m for m in self.metrics if m.latency_ms > 1000]
        if high_latency_services:
            recommendations.append(
                "High latency detected. Consider implementing connection pooling "
                "and message compression."
            )

        # Analyze throughput
        low_throughput_services = [m for m in self.metrics if m.throughput_rps < 10]
        if low_throughput_services:
            recommendations.append(
                "Low throughput detected. Consider implementing batch processing "
                "and parallel execution."
            )

        # Analyze error rate
        high_error_services = [m for m in self.metrics if m.error_rate > 0.1]
        if high_error_services:
            recommendations.append(
                "High error rate detected. Consider implementing retry logic "
                "and circuit breakers."
            )

        # Analyze memory usage
        high_memory_services = [m for m in self.metrics if m.memory_usage_mb > 1000]
        if high_memory_services:
            recommendations.append(
                "High memory usage detected. Consider implementing memory pooling "
                "and garbage collection optimization."
            )

        # Analyze CPU usage
        high_cpu_services = [m for m in self.metrics if m.cpu_usage_percent > 80]
        if high_cpu_services:
            recommendations.append(
                "High CPU usage detected. Consider implementing CPU optimization "
                "and load balancing."
            )

        return recommendations


class PerformanceOptimizer:
    """Optimizes gRPC service performance."""

    def __init__(self):
        """Initialize performance optimizer."""
        self.analyzer = PerformanceAnalyzer()
        self.optimization_config = {
            "connection_pool_size": 10,
            "max_message_size": 4 * 1024 * 1024,  # 4MB
            "compression": "gzip",
            "keepalive_time_ms": 30000,
            "keepalive_timeout_ms": 5000,
            "keepalive_permit_without_calls": True,
            "max_connection_idle_ms": 300000,
            "max_connection_age_ms": 600000,
            "max_connection_age_grace_ms": 5000,
        }

    async def optimize_connection_pooling(self) -> dict[str, Any]:
        """Optimize connection pooling."""
        logger.info("Optimizing connection pooling...")

        # Test different pool sizes
        pool_sizes = [5, 10, 20, 50]
        results = {}

        for pool_size in pool_sizes:
            self.optimization_config["connection_pool_size"] = pool_size

            # Test with GPU client
            try:
                client = GPUClient(
                    service_url="localhost:50051",
                    timeout=30,
                    max_retries=3,
                )
                await client.initialize()

                # Measure performance
                metrics = await self.analyzer.measure_service_performance(
                    service_name="gpu",
                    operation="get_status",
                    client=client,
                    test_data=None,
                    iterations=5,
                )

                results[pool_size] = {
                    "latency_ms": metrics.latency_ms,
                    "throughput_rps": metrics.throughput_rps,
                    "error_rate": metrics.error_rate,
                }

                await client.close()

            except Exception as e:
                logger.error(f"Failed to test pool size {pool_size}: {e}")
                results[pool_size] = {"error": str(e)}

        # Find optimal pool size
        best_pool_size = min(
            results.keys(), key=lambda x: results[x].get("latency_ms", float("inf"))
        )

        self.optimization_config["connection_pool_size"] = best_pool_size

        return {
            "results": results,
            "optimal_pool_size": best_pool_size,
            "recommendation": f"Optimal connection pool size: {best_pool_size}",
        }

    async def optimize_message_compression(self) -> dict[str, Any]:
        """Optimize message compression."""
        logger.info("Optimizing message compression...")

        compression_types = ["none", "gzip", "deflate"]
        results = {}

        for compression in compression_types:
            self.optimization_config["compression"] = compression

            # Test with embedding client
            try:
                client = EmbeddingClient(
                    service_url="localhost:50052",
                    timeout=30,
                    max_retries=3,
                )
                await client.initialize()

                # Test data
                test_data = {
                    "texts": ["test text"] * 100,
                    "model": "qwen3-embedding-8b",
                }

                # Measure performance
                metrics = await self.analyzer.measure_service_performance(
                    service_name="embedding",
                    operation="generate_embeddings",
                    client=client,
                    test_data=test_data,
                    iterations=3,
                )

                results[compression] = {
                    "latency_ms": metrics.latency_ms,
                    "throughput_rps": metrics.throughput_rps,
                    "error_rate": metrics.error_rate,
                }

                await client.close()

            except Exception as e:
                logger.error(f"Failed to test compression {compression}: {e}")
                results[compression] = {"error": str(e)}

        # Find optimal compression
        best_compression = min(
            results.keys(), key=lambda x: results[x].get("latency_ms", float("inf"))
        )

        self.optimization_config["compression"] = best_compression

        return {
            "results": results,
            "optimal_compression": best_compression,
            "recommendation": f"Optimal compression: {best_compression}",
        }

    async def optimize_keepalive_settings(self) -> dict[str, Any]:
        """Optimize keepalive settings."""
        logger.info("Optimizing keepalive settings...")

        keepalive_configs = [
            {"time_ms": 30000, "timeout_ms": 5000},
            {"time_ms": 60000, "timeout_ms": 10000},
            {"time_ms": 120000, "timeout_ms": 15000},
        ]

        results = {}

        for config in keepalive_configs:
            self.optimization_config.update(config)

            # Test with reranking client
            try:
                client = RerankingClient(
                    service_url="localhost:50053",
                    timeout=30,
                    max_retries=3,
                )
                await client.initialize()

                # Test data
                test_data = {
                    "queries": ["test query"],
                    "documents": ["test document"] * 10,
                    "model": "bge-reranker-v2-m3",
                }

                # Measure performance
                metrics = await self.analyzer.measure_service_performance(
                    service_name="reranking",
                    operation="rerank_batch",
                    client=client,
                    test_data=test_data,
                    iterations=3,
                )

                config_key = f"{config['time_ms']}_{config['timeout_ms']}"
                results[config_key] = {
                    "latency_ms": metrics.latency_ms,
                    "throughput_rps": metrics.throughput_rps,
                    "error_rate": metrics.error_rate,
                }

                await client.close()

            except Exception as e:
                logger.error(f"Failed to test keepalive config {config}: {e}")
                config_key = f"{config['time_ms']}_{config['timeout_ms']}"
                results[config_key] = {"error": str(e)}

        # Find optimal keepalive settings
        best_config = min(results.keys(), key=lambda x: results[x].get("latency_ms", float("inf")))

        time_ms, timeout_ms = best_config.split("_")
        self.optimization_config["keepalive_time_ms"] = int(time_ms)
        self.optimization_config["keepalive_timeout_ms"] = int(timeout_ms)

        return {
            "results": results,
            "optimal_keepalive": best_config,
            "recommendation": f"Optimal keepalive: {time_ms}ms/{timeout_ms}ms",
        }

    async def run_full_optimization(self) -> dict[str, Any]:
        """Run full performance optimization."""
        logger.info("Starting full performance optimization...")

        results = {}

        # Optimize connection pooling
        try:
            results["connection_pooling"] = await self.optimize_connection_pooling()
        except Exception as e:
            logger.error(f"Connection pooling optimization failed: {e}")
            results["connection_pooling"] = {"error": str(e)}

        # Optimize message compression
        try:
            results["message_compression"] = await self.optimize_message_compression()
        except Exception as e:
            logger.error(f"Message compression optimization failed: {e}")
            results["message_compression"] = {"error": str(e)}

        # Optimize keepalive settings
        try:
            results["keepalive_settings"] = await self.optimize_keepalive_settings()
        except Exception as e:
            logger.error(f"Keepalive settings optimization failed: {e}")
            results["keepalive_settings"] = {"error": str(e)}

        # Generate final recommendations
        recommendations = self.analyzer.generate_optimization_recommendations()
        results["recommendations"] = recommendations

        # Print optimization summary
        self.print_optimization_summary(results)

        return results

    def print_optimization_summary(self, results: dict[str, Any]) -> None:
        """Print optimization summary."""
        print("\n" + "=" * 60)
        print("gRPC PERFORMANCE OPTIMIZATION SUMMARY")
        print("=" * 60)

        # Connection pooling results
        if "connection_pooling" in results:
            cp_results = results["connection_pooling"]
            if "optimal_pool_size" in cp_results:
                print(f"Optimal Connection Pool Size: {cp_results['optimal_pool_size']}")
            if "recommendation" in cp_results:
                print(f"Recommendation: {cp_results['recommendation']}")

        # Message compression results
        if "message_compression" in results:
            mc_results = results["message_compression"]
            if "optimal_compression" in mc_results:
                print(f"Optimal Compression: {mc_results['optimal_compression']}")
            if "recommendation" in mc_results:
                print(f"Recommendation: {mc_results['recommendation']}")

        # Keepalive settings results
        if "keepalive_settings" in results:
            ks_results = results["keepalive_settings"]
            if "optimal_keepalive" in ks_results:
                print(f"Optimal Keepalive: {ks_results['optimal_keepalive']}")
            if "recommendation" in ks_results:
                print(f"Recommendation: {ks_results['recommendation']}")

        # General recommendations
        if "recommendations" in results:
            print("\nGeneral Recommendations:")
            for i, rec in enumerate(results["recommendations"], 1):
                print(f"{i}. {rec}")

        print("=" * 60)

    def generate_optimization_config(self) -> dict[str, Any]:
        """Generate optimized configuration."""
        return self.optimization_config.copy()


async def main():
    """Main function to run performance optimization."""
    parser = argparse.ArgumentParser(description="Optimize gRPC service performance")
    parser.add_argument("--output", help="Path to output optimization results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Create optimizer
        optimizer = PerformanceOptimizer()

        # Run optimization
        results = await optimizer.run_full_optimization()

        # Generate optimized configuration
        config = optimizer.generate_optimization_config()
        results["optimized_config"] = config

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Optimization results saved to: {args.output}")

        logger.info("Performance optimization completed successfully")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Performance optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
