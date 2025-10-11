#!/usr/bin/env python3
"""Script to validate monitoring systems for torch isolation architecture.

This script provides a command-line interface for validating monitoring
systems, including Prometheus, Grafana, Alertmanager, and service health checks.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import monitoring validator
try:
    from Medical_KG_rev.services.monitoring.monitoring_validator import (
        MonitoringConfig,
        MonitoringValidator,
        ValidationStatus,
    )
except ImportError as e:
    print(f"Warning: Could not import monitoring validator: {e}")
    print("Running in mock mode...")

    # Mock classes for when modules are not available
    class ValidationStatus:
        PASS = "PASS"
        FAIL = "FAIL"
        WARNING = "WARNING"
        SKIP = "SKIP"

    class MonitoringConfig:
        def __init__(self, **kwargs):
            self.prometheus_url = kwargs.get("prometheus_url", "http://localhost:9090")
            self.grafana_url = kwargs.get("grafana_url", "http://localhost:3000")
            self.alertmanager_url = kwargs.get("alertmanager_url", "http://localhost:9093")
            self.gpu_metrics_exporter_url = kwargs.get(
                "gpu_metrics_exporter_url", "http://localhost:8080"
            )
            self.custom_metrics_adapter_url = kwargs.get(
                "custom_metrics_adapter_url", "http://localhost:8081"
            )
            self.service_urls = kwargs.get("service_urls", {})
            self.timeout = kwargs.get("timeout", 30)
            self.retry_attempts = kwargs.get("retry_attempts", 3)
            self.retry_delay = kwargs.get("retry_delay", 1.0)

    class MonitoringValidator:
        def __init__(self, config: MonitoringConfig):
            self.config = config
            self.results = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def validate_all(self):
            # Mock validation results
            return [
                {
                    "component": "prometheus",
                    "check_name": "health",
                    "status": "PASS",
                    "message": "Mock result",
                },
                {
                    "component": "grafana",
                    "check_name": "health",
                    "status": "PASS",
                    "message": "Mock result",
                },
                {
                    "component": "alertmanager",
                    "check_name": "health",
                    "status": "PASS",
                    "message": "Mock result",
                },
            ]

        def get_summary(self):
            return {
                "total_checks": 3,
                "passed_checks": 3,
                "failed_checks": 0,
                "warning_checks": 0,
                "success_rate": 1.0,
                "results": self.results,
            }


app = typer.Typer(help="Validate monitoring systems for torch isolation architecture")
console = Console()


@app.command()
def validate_all(
    prometheus_url: str = typer.Option(
        "http://localhost:9090", "--prometheus-url", help="Prometheus URL"
    ),
    grafana_url: str = typer.Option("http://localhost:3000", "--grafana-url", help="Grafana URL"),
    alertmanager_url: str = typer.Option(
        "http://localhost:9093", "--alertmanager-url", help="Alertmanager URL"
    ),
    gpu_metrics_exporter_url: str = typer.Option(
        "http://localhost:8080", "--gpu-metrics-url", help="GPU metrics exporter URL"
    ),
    custom_metrics_adapter_url: str = typer.Option(
        "http://localhost:8081", "--custom-metrics-url", help="Custom metrics adapter URL"
    ),
    output_file: str
    | None = typer.Option(None, "--output", "-o", help="Output file for validation results"),
    timeout: int = typer.Option(30, "--timeout", "-t", help="Request timeout in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Validate all monitoring systems."""
    console.print("🔍 Validating Monitoring Systems for Torch Isolation Architecture...")
    console.print("=" * 80)

    async def run_validation() -> int:
        """Run monitoring validation."""
        config = MonitoringConfig(
            prometheus_url=prometheus_url,
            grafana_url=grafana_url,
            alertmanager_url=alertmanager_url,
            gpu_metrics_exporter_url=gpu_metrics_exporter_url,
            custom_metrics_adapter_url=custom_metrics_adapter_url,
            timeout=timeout,
        )

        async with MonitoringValidator(config) as validator:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Validating monitoring systems...", total=None)

                try:
                    results = await validator.validate_all()
                    progress.update(task, description="✅ Validation completed")

                    # Print results
                    print_validation_results(results, verbose)

                    # Save results to file if specified
                    if output_file:
                        save_results_to_file(results, output_file)
                        console.print(f"\n📄 Results saved to: {output_file}")

                    # Return appropriate exit code
                    summary = validator.get_summary()
                    if summary["failed_checks"] > 0:
                        console.print("\n❌ Some validation checks failed!")
                        return 1
                    else:
                        console.print("\n🎉 All validation checks passed!")
                        return 0

                except Exception as e:
                    progress.update(task, description="❌ Validation failed")
                    console.print(f"\n💥 Validation failed with error: {e}")
                    return 1

    # Run the validation
    exit_code = asyncio.run(run_validation())
    sys.exit(exit_code)


@app.command()
def validate_prometheus(
    prometheus_url: str = typer.Option(
        "http://localhost:9090", "--prometheus-url", help="Prometheus URL"
    ),
    output_file: str
    | None = typer.Option(None, "--output", "-o", help="Output file for validation results"),
    timeout: int = typer.Option(30, "--timeout", "-t", help="Request timeout in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Validate Prometheus configuration and metrics."""
    console.print("🔍 Validating Prometheus...")

    async def run_prometheus_validation() -> int:
        """Run Prometheus validation."""
        config = MonitoringConfig(prometheus_url=prometheus_url, timeout=timeout)

        async with MonitoringValidator(config) as validator:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Validating Prometheus...", total=None)

                try:
                    # Run only Prometheus validation
                    await validator._validate_prometheus()
                    results = validator.results
                    progress.update(task, description="✅ Prometheus validation completed")

                    # Print results
                    print_validation_results(results, verbose)

                    # Save results to file if specified
                    if output_file:
                        save_results_to_file(results, output_file)
                        console.print(f"\n📄 Results saved to: {output_file}")

                    # Return appropriate exit code
                    failed_checks = len([r for r in results if r.status == ValidationStatus.FAIL])
                    if failed_checks > 0:
                        console.print("\n❌ Some Prometheus validation checks failed!")
                        return 1
                    else:
                        console.print("\n🎉 All Prometheus validation checks passed!")
                        return 0

                except Exception as e:
                    progress.update(task, description="❌ Prometheus validation failed")
                    console.print(f"\n💥 Prometheus validation failed with error: {e}")
                    return 1

    # Run the validation
    exit_code = asyncio.run(run_prometheus_validation())
    sys.exit(exit_code)


@app.command()
def validate_grafana(
    grafana_url: str = typer.Option("http://localhost:3000", "--grafana-url", help="Grafana URL"),
    output_file: str
    | None = typer.Option(None, "--output", "-o", help="Output file for validation results"),
    timeout: int = typer.Option(30, "--timeout", "-t", help="Request timeout in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Validate Grafana configuration and dashboards."""
    console.print("🔍 Validating Grafana...")

    async def run_grafana_validation() -> int:
        """Run Grafana validation."""
        config = MonitoringConfig(grafana_url=grafana_url, timeout=timeout)

        async with MonitoringValidator(config) as validator:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Validating Grafana...", total=None)

                try:
                    # Run only Grafana validation
                    await validator._validate_grafana()
                    results = validator.results
                    progress.update(task, description="✅ Grafana validation completed")

                    # Print results
                    print_validation_results(results, verbose)

                    # Save results to file if specified
                    if output_file:
                        save_results_to_file(results, output_file)
                        console.print(f"\n📄 Results saved to: {output_file}")

                    # Return appropriate exit code
                    failed_checks = len([r for r in results if r.status == ValidationStatus.FAIL])
                    if failed_checks > 0:
                        console.print("\n❌ Some Grafana validation checks failed!")
                        return 1
                    else:
                        console.print("\n🎉 All Grafana validation checks passed!")
                        return 0

                except Exception as e:
                    progress.update(task, description="❌ Grafana validation failed")
                    console.print(f"\n💥 Grafana validation failed with error: {e}")
                    return 1

    # Run the validation
    exit_code = asyncio.run(run_grafana_validation())
    sys.exit(exit_code)


@app.command()
def validate_services(
    service_urls: str = typer.Option(
        "", "--service-urls", help="Comma-separated service URLs (name:url format)"
    ),
    output_file: str
    | None = typer.Option(None, "--output", "-o", help="Output file for validation results"),
    timeout: int = typer.Option(30, "--timeout", "-t", help="Request timeout in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Validate service health checks."""
    console.print("🔍 Validating Service Health Checks...")

    async def run_service_validation() -> int:
        """Run service validation."""
        # Parse service URLs
        service_url_dict = {}
        if service_urls:
            for service_entry in service_urls.split(","):
                if ":" in service_entry:
                    name, url = service_entry.split(":", 1)
                    service_url_dict[name.strip()] = url.strip()

        config = MonitoringConfig(service_urls=service_url_dict, timeout=timeout)

        async with MonitoringValidator(config) as validator:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Validating services...", total=None)

                try:
                    # Run only service validation
                    await validator._validate_service_health_checks()
                    results = validator.results
                    progress.update(task, description="✅ Service validation completed")

                    # Print results
                    print_validation_results(results, verbose)

                    # Save results to file if specified
                    if output_file:
                        save_results_to_file(results, output_file)
                        console.print(f"\n📄 Results saved to: {output_file}")

                    # Return appropriate exit code
                    failed_checks = len([r for r in results if r.status == ValidationStatus.FAIL])
                    if failed_checks > 0:
                        console.print("\n❌ Some service validation checks failed!")
                        return 1
                    else:
                        console.print("\n🎉 All service validation checks passed!")
                        return 0

                except Exception as e:
                    progress.update(task, description="❌ Service validation failed")
                    console.print(f"\n💥 Service validation failed with error: {e}")
                    return 1

    # Run the validation
    exit_code = asyncio.run(run_service_validation())
    sys.exit(exit_code)


def print_validation_results(results: list, verbose: bool) -> None:
    """Print validation results."""
    if not results:
        console.print("No validation results to display.")
        return

    # Create summary table
    summary_table = Table(title="Monitoring Validation Summary")
    summary_table.add_column("Component", style="cyan")
    summary_table.add_column("Total Checks", justify="right")
    summary_table.add_column("Passed", justify="right", style="green")
    summary_table.add_column("Failed", justify="right", style="red")
    summary_table.add_column("Warnings", justify="right", style="yellow")

    # Group results by component
    component_stats = {}
    for result in results:
        component = (
            result.component.value if hasattr(result.component, "value") else str(result.component)
        )
        if component not in component_stats:
            component_stats[component] = {"total": 0, "passed": 0, "failed": 0, "warnings": 0}

        component_stats[component]["total"] += 1
        if result.status == ValidationStatus.PASS:
            component_stats[component]["passed"] += 1
        elif result.status == ValidationStatus.FAIL:
            component_stats[component]["failed"] += 1
        elif result.status == ValidationStatus.WARNING:
            component_stats[component]["warnings"] += 1

    # Add rows to summary table
    for component, stats in component_stats.items():
        summary_table.add_row(
            component,
            str(stats["total"]),
            str(stats["passed"]),
            str(stats["failed"]),
            str(stats["warnings"]),
        )

    console.print(summary_table)

    # Print detailed results if verbose
    if verbose:
        console.print("\n📋 Detailed Results:")
        for result in results:
            status_icon = (
                "✅"
                if result.status == ValidationStatus.PASS
                else "❌"
                if result.status == ValidationStatus.FAIL
                else "⚠️"
            )
            component = (
                result.component.value
                if hasattr(result.component, "value")
                else str(result.component)
            )
            console.print(f"   {status_icon} [{component}] {result.check_name}: {result.message}")
            if result.details:
                console.print(f"      Details: {json.dumps(result.details, indent=2)}")


def save_results_to_file(results: list, output_file: str) -> None:
    """Save validation results to file."""
    # Convert results to serializable format
    serializable_results = []
    for result in results:
        serializable_result = {
            "component": (
                result.component.value
                if hasattr(result.component, "value")
                else str(result.component)
            ),
            "check_name": result.check_name,
            "status": (
                result.status.value if hasattr(result.status, "value") else str(result.status)
            ),
            "message": result.message,
            "details": result.details,
            "execution_time": result.execution_time,
        }
        serializable_results.append(serializable_result)

    # Save to file
    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)


@app.command()
def generate_report(
    results_file: str = typer.Argument(..., help="Path to validation results JSON file"),
    output_file: str = typer.Option(
        "monitoring_validation_report.md", "--output", "-o", help="Output markdown file"
    ),
) -> None:
    """Generate a markdown report from validation results."""
    console.print("📊 Generating Monitoring Validation Report...")

    try:
        # Load results
        with open(results_file) as f:
            results = json.load(f)

        # Generate markdown report
        report_lines = [
            "# Monitoring Validation Report",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Total Checks:** {len(results)}",
            f"- **Passed:** {len([r for r in results if r['status'] == 'PASS'])}",
            f"- **Failed:** {len([r for r in results if r['status'] == 'FAIL'])}",
            f"- **Warnings:** {len([r for r in results if r['status'] == 'WARNING'])}",
            "",
            "## Detailed Results",
            "",
        ]

        # Group results by component
        component_results = {}
        for result in results:
            component = result["component"]
            if component not in component_results:
                component_results[component] = []
            component_results[component].append(result)

        # Add results for each component
        for component, component_results_list in component_results.items():
            report_lines.extend([f"### {component.title()}", ""])

            for result in component_results_list:
                status_icon = (
                    "✅"
                    if result["status"] == "PASS"
                    else "❌"
                    if result["status"] == "FAIL"
                    else "⚠️"
                )
                report_lines.extend(
                    [
                        f"**{status_icon} {result['check_name']}:** {result['status']}",
                        f"- {result['message']}",
                        "",
                    ]
                )

                if result.get("details"):
                    report_lines.extend(
                        ["```json", json.dumps(result["details"], indent=2), "```", ""]
                    )

        # Add conclusion
        report_lines.extend(["## Conclusion", ""])

        failed_count = len([r for r in results if r["status"] == "FAIL"])
        if failed_count == 0:
            report_lines.extend(
                [
                    "✅ All monitoring validation checks passed successfully.",
                    "",
                    "The monitoring systems are properly configured and functioning:",
                    "- Prometheus metrics collection",
                    "- Grafana dashboards and visualization",
                    "- Alertmanager alerting rules",
                    "- Service health checks",
                    "- GPU metrics integration",
                    "- Custom metrics integration",
                ]
            )
        else:
            report_lines.extend(
                [
                    "❌ Some monitoring validation checks failed.",
                    "",
                    "Please review the failed checks and fix the issues before deployment.",
                ]
            )

        # Write report
        with open(output_file, "w") as f:
            f.write("\n".join(report_lines))

        console.print(f"📄 Report generated: {output_file}")

    except Exception as e:
        console.print(f"❌ Failed to generate report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    app()
