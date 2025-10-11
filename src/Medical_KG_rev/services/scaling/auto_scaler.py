"""Auto-scaling service for GPU services.

This module provides auto-scaling capabilities for GPU services including
load-based scaling, performance-based scaling, and resource utilization monitoring.
"""

from datetime import datetime, timedelta
from typing import Any
import asyncio
import logging

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class ScalingMetrics(BaseModel):
    """Scaling metrics model."""

    service_name: str
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_usage_percent: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    request_rate_rps: float
    response_time_p95_ms: float
    error_rate: float
    active_connections: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ScalingDecision(BaseModel):
    """Scaling decision model."""

    service_name: str
    action: str  # "scale_up", "scale_down", "no_action"
    current_replicas: int
    target_replicas: int
    reason: str
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ScalingPolicy(BaseModel):
    """Scaling policy configuration."""

    service_name: str
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    target_gpu_percent: float = 80.0
    target_response_time_ms: float = 500.0
    target_error_rate: float = 0.05
    scale_up_cooldown_seconds: int = 300
    scale_down_cooldown_seconds: int = 600
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3


class AutoScaler:
    """Auto-scaler for GPU services."""

    def __init__(self, policies: dict[str, ScalingPolicy]):
        """Initialize auto-scaler."""
        self.policies = policies
        self.metrics_history: dict[str, list[ScalingMetrics]] = {}
        self.last_scaling_action: dict[str, datetime] = {}
        self.scaling_decisions: list[ScalingDecision] = []

        # Initialize metrics history
        for service_name in policies:
            self.metrics_history[service_name] = []

    async def collect_metrics(self, service_name: str) -> ScalingMetrics | None:
        """Collect metrics for a service."""
        try:
            if service_name == "gpu":
                return await self._collect_gpu_metrics()
            elif service_name == "embedding":
                return await self._collect_embedding_metrics()
            elif service_name == "reranking":
                return await self._collect_reranking_metrics()
            elif service_name == "docling_vlm":
                return await self._collect_docling_vlm_metrics()
            else:
                logger.warning(f"Unknown service: {service_name}")
                return None

        except Exception as e:
            logger.error(f"Failed to collect metrics for {service_name}: {e}")
            return None

    async def _collect_gpu_metrics(self) -> ScalingMetrics:
        """Collect GPU service metrics."""
        # Mock implementation - replace with actual GPU client
        return ScalingMetrics(
            service_name="gpu",
            cpu_usage_percent=45.0,
            memory_usage_mb=2048.0,
            gpu_usage_percent=60.0,
            gpu_memory_usage_mb=8192.0,
            request_rate_rps=10.5,
            response_time_p95_ms=150.0,
            error_rate=0.01,
            active_connections=5,
        )

    async def _collect_embedding_metrics(self) -> ScalingMetrics:
        """Collect embedding service metrics."""
        # Mock implementation - replace with actual embedding client
        return ScalingMetrics(
            service_name="embedding",
            cpu_usage_percent=55.0,
            memory_usage_mb=4096.0,
            gpu_usage_percent=75.0,
            gpu_memory_usage_mb=12288.0,
            request_rate_rps=25.0,
            response_time_p95_ms=200.0,
            error_rate=0.02,
            active_connections=12,
        )

    async def _collect_reranking_metrics(self) -> ScalingMetrics:
        """Collect reranking service metrics."""
        # Mock implementation - replace with actual reranking client
        return ScalingMetrics(
            service_name="reranking",
            cpu_usage_percent=40.0,
            memory_usage_mb=3072.0,
            gpu_usage_percent=50.0,
            gpu_memory_usage_mb=6144.0,
            request_rate_rps=8.0,
            response_time_p95_ms=300.0,
            error_rate=0.015,
            active_connections=3,
        )

    async def _collect_docling_vlm_metrics(self) -> ScalingMetrics:
        """Collect Docling VLM service metrics."""
        # Mock implementation - replace with actual Docling VLM client
        return ScalingMetrics(
            service_name="docling_vlm",
            cpu_usage_percent=65.0,
            memory_usage_mb=8192.0,
            gpu_usage_percent=85.0,
            gpu_memory_usage_mb=18432.0,
            request_rate_rps=5.0,
            response_time_p95_ms=800.0,
            error_rate=0.03,
            active_connections=2,
        )

    def analyze_metrics(self, service_name: str, metrics: ScalingMetrics) -> ScalingDecision:
        """Analyze metrics and make scaling decision."""
        policy = self.policies.get(service_name)
        if not policy:
            return ScalingDecision(
                service_name=service_name,
                action="no_action",
                current_replicas=1,
                target_replicas=1,
                reason="No scaling policy configured",
                confidence=0.0,
            )

        # Add metrics to history
        self.metrics_history[service_name].append(metrics)

        # Keep only recent metrics (last hour)
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        self.metrics_history[service_name] = [
            m for m in self.metrics_history[service_name] if m.timestamp > cutoff_time
        ]

        # Check cooldown periods
        last_action = self.last_scaling_action.get(service_name)
        if last_action:
            cooldown_seconds = (
                policy.scale_up_cooldown_seconds
                if len(self.metrics_history[service_name]) > 1
                else policy.scale_down_cooldown_seconds
            )
            if datetime.utcnow() - last_action < timedelta(seconds=cooldown_seconds):
                return ScalingDecision(
                    service_name=service_name,
                    action="no_action",
                    current_replicas=1,
                    target_replicas=1,
                    reason="Scaling cooldown period active",
                    confidence=0.0,
                )

        # Analyze current metrics
        current_replicas = 1  # Mock - replace with actual replica count

        # Calculate scaling score
        cpu_score = metrics.cpu_usage_percent / policy.target_cpu_percent
        memory_score = metrics.memory_usage_mb / (policy.target_memory_percent * 100)
        gpu_score = metrics.gpu_usage_percent / policy.target_gpu_percent
        response_time_score = metrics.response_time_p95_ms / policy.target_response_time_ms
        error_score = metrics.error_rate / policy.target_error_rate

        # Weighted average score
        scaling_score = (
            cpu_score * 0.3
            + memory_score * 0.2
            + gpu_score * 0.3
            + response_time_score * 0.15
            + error_score * 0.05
        )

        # Make scaling decision
        if scaling_score > policy.scale_up_threshold:
            target_replicas = min(current_replicas + 1, policy.max_replicas)
            action = "scale_up"
            reason = f"High resource utilization (score: {scaling_score:.2f})"
            confidence = min(scaling_score, 1.0)
        elif scaling_score < policy.scale_down_threshold:
            target_replicas = max(current_replicas - 1, policy.min_replicas)
            action = "scale_down"
            reason = f"Low resource utilization (score: {scaling_score:.2f})"
            confidence = 1.0 - scaling_score
        else:
            target_replicas = current_replicas
            action = "no_action"
            reason = f"Resource utilization within target range (score: {scaling_score:.2f})"
            confidence = 0.5

        decision = ScalingDecision(
            service_name=service_name,
            action=action,
            current_replicas=current_replicas,
            target_replicas=target_replicas,
            reason=reason,
            confidence=confidence,
        )

        self.scaling_decisions.append(decision)
        return decision

    async def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute scaling decision."""
        try:
            if decision.action == "no_action":
                return True

            logger.info(f"Executing scaling decision: {decision}")

            # Mock implementation - replace with actual scaling logic
            if decision.action == "scale_up":
                await self._scale_up_service(decision.service_name, decision.target_replicas)
            elif decision.action == "scale_down":
                await self._scale_down_service(decision.service_name, decision.target_replicas)

            # Update last scaling action time
            self.last_scaling_action[decision.service_name] = datetime.utcnow()

            logger.info(f"Successfully executed scaling decision for {decision.service_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to execute scaling decision: {e}")
            return False

    async def _scale_up_service(self, service_name: str, target_replicas: int) -> None:
        """Scale up a service."""
        logger.info(f"Scaling up {service_name} to {target_replicas} replicas")
        # Mock implementation - replace with actual scaling logic
        # This would typically involve:
        # 1. Updating Kubernetes deployment
        # 2. Waiting for new pods to be ready
        # 3. Verifying service health
        await asyncio.sleep(1)  # Mock delay

    async def _scale_down_service(self, service_name: str, target_replicas: int) -> None:
        """Scale down a service."""
        logger.info(f"Scaling down {service_name} to {target_replicas} replicas")
        # Mock implementation - replace with actual scaling logic
        # This would typically involve:
        # 1. Updating Kubernetes deployment
        # 2. Gracefully terminating excess pods
        # 3. Verifying service health
        await asyncio.sleep(1)  # Mock delay

    async def run_scaling_loop(self, interval_seconds: int = 60) -> None:
        """Run continuous scaling loop."""
        logger.info("Starting auto-scaling loop")

        while True:
            try:
                for service_name in self.policies.keys():
                    # Collect metrics
                    metrics = await self.collect_metrics(service_name)
                    if not metrics:
                        continue

                    # Analyze metrics and make decision
                    decision = self.analyze_metrics(service_name, metrics)

                    # Execute scaling decision
                    if decision.action != "no_action":
                        await self.execute_scaling_decision(decision)

                    # Log decision
                    logger.info(f"Scaling decision for {service_name}: {decision}")

                # Wait for next iteration
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(interval_seconds)

    def get_scaling_history(self, service_name: str | None = None) -> list[ScalingDecision]:
        """Get scaling decision history."""
        if service_name:
            return [d for d in self.scaling_decisions if d.service_name == service_name]
        return self.scaling_decisions

    def get_metrics_history(self, service_name: str) -> list[ScalingMetrics]:
        """Get metrics history for a service."""
        return self.metrics_history.get(service_name, [])

    def get_scaling_policy(self, service_name: str) -> ScalingPolicy | None:
        """Get scaling policy for a service."""
        return self.policies.get(service_name)

    def update_scaling_policy(self, service_name: str, policy: ScalingPolicy) -> None:
        """Update scaling policy for a service."""
        self.policies[service_name] = policy
        logger.info(f"Updated scaling policy for {service_name}")


class ScalingManager:
    """Manages auto-scaling for multiple services."""

    def __init__(self, config: dict[str, Any]):
        """Initialize scaling manager."""
        self.config = config
        self.auto_scalers: dict[str, AutoScaler] = {}
        self.scaling_tasks: dict[str, asyncio.Task] = {}

    def create_auto_scaler(self, service_name: str, policy: ScalingPolicy) -> AutoScaler:
        """Create auto-scaler for a service."""
        auto_scaler = AutoScaler({service_name: policy})
        self.auto_scalers[service_name] = auto_scaler
        return auto_scaler

    async def start_scaling(self, service_name: str, interval_seconds: int = 60) -> None:
        """Start auto-scaling for a service."""
        if service_name in self.scaling_tasks:
            logger.warning(f"Auto-scaling already running for {service_name}")
            return

        auto_scaler = self.auto_scalers.get(service_name)
        if not auto_scaler:
            logger.error(f"No auto-scaler configured for {service_name}")
            return

        task = asyncio.create_task(auto_scaler.run_scaling_loop(interval_seconds))
        self.scaling_tasks[service_name] = task

        logger.info(f"Started auto-scaling for {service_name}")

    async def stop_scaling(self, service_name: str) -> None:
        """Stop auto-scaling for a service."""
        task = self.scaling_tasks.get(service_name)
        if not task:
            logger.warning(f"No auto-scaling task found for {service_name}")
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        del self.scaling_tasks[service_name]
        logger.info(f"Stopped auto-scaling for {service_name}")

    async def start_all_scaling(self, interval_seconds: int = 60) -> None:
        """Start auto-scaling for all services."""
        for service_name in self.auto_scalers.keys():
            await self.start_scaling(service_name, interval_seconds)

    async def stop_all_scaling(self) -> None:
        """Stop auto-scaling for all services."""
        for service_name in list(self.scaling_tasks.keys()):
            await self.stop_scaling(service_name)

    def get_scaling_status(self) -> dict[str, Any]:
        """Get scaling status for all services."""
        status = {}
        for service_name, auto_scaler in self.auto_scalers.items():
            status[service_name] = {
                "is_running": service_name in self.scaling_tasks,
                "policy": auto_scaler.get_scaling_policy(service_name).model_dump(),
                "recent_decisions": auto_scaler.get_scaling_history(service_name)[-5:],
                "metrics_count": len(auto_scaler.get_metrics_history(service_name)),
            }
        return status


# Default scaling policies
DEFAULT_SCALING_POLICIES = {
    "gpu": ScalingPolicy(
        service_name="gpu",
        min_replicas=1,
        max_replicas=5,
        target_cpu_percent=70.0,
        target_memory_percent=80.0,
        target_gpu_percent=80.0,
        target_response_time_ms=200.0,
        target_error_rate=0.05,
        scale_up_cooldown_seconds=300,
        scale_down_cooldown_seconds=600,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
    ),
    "embedding": ScalingPolicy(
        service_name="embedding",
        min_replicas=2,
        max_replicas=10,
        target_cpu_percent=70.0,
        target_memory_percent=80.0,
        target_gpu_percent=80.0,
        target_response_time_ms=300.0,
        target_error_rate=0.05,
        scale_up_cooldown_seconds=300,
        scale_down_cooldown_seconds=600,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
    ),
    "reranking": ScalingPolicy(
        service_name="reranking",
        min_replicas=1,
        max_replicas=8,
        target_cpu_percent=70.0,
        target_memory_percent=80.0,
        target_gpu_percent=80.0,
        target_response_time_ms=400.0,
        target_error_rate=0.05,
        scale_up_cooldown_seconds=300,
        scale_down_cooldown_seconds=600,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
    ),
    "docling_vlm": ScalingPolicy(
        service_name="docling_vlm",
        min_replicas=1,
        max_replicas=3,
        target_cpu_percent=70.0,
        target_memory_percent=80.0,
        target_gpu_percent=80.0,
        target_response_time_ms=1000.0,
        target_error_rate=0.05,
        scale_up_cooldown_seconds=600,
        scale_down_cooldown_seconds=1200,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
    ),
}
