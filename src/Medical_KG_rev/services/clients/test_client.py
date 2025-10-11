"""Test client for gRPC services.

Provides mock implementations and testing utilities for service clients.
"""

import logging
import time
from typing import Any

from .error_handler import ServiceErrorHandler
from .errors import ServiceError, ServiceTimeoutError, ServiceUnavailableError

logger = logging.getLogger(__name__)


class MockServiceResponse:
    """Mock service response for testing."""

    def __init__(self, data: dict[str, Any], status: str = "success"):
        """Initialize mock response.

        Args:
            data: Response data
            status: Response status

        """
        self.data = data
        self.status = status
        self.timestamp = time.time()


class MockServiceClient:
    """Mock gRPC service client for testing."""

    def __init__(
        self,
        service_name: str,
        mock_responses: dict[str, Any] | None = None,
        error_scenarios: list[dict[str, Any]] | None = None,
    ):
        """Initialize mock service client.

        Args:
            service_name: Name of the service
            mock_responses: Mock responses for different methods
            error_scenarios: Error scenarios to simulate

        """
        self.service_name = service_name
        self.mock_responses = mock_responses or {}
        self.error_scenarios = error_scenarios or []
        self.call_count = 0
        self.error_handler = ServiceErrorHandler(service_name)

    async def mock_call(self, method_name: str, *args, **kwargs) -> MockServiceResponse:
        """Mock service call.

        Args:
            method_name: Name of the method being called
            *args: Method arguments
            **kwargs: Method keyword arguments

        Returns:
            Mock service response

        Raises:
            ServiceError: If error scenario is triggered

        """
        self.call_count += 1

        # Check for error scenarios
        for scenario in self.error_scenarios:
            if self._should_trigger_error(scenario):
                error = self._create_error(scenario)
                raise error

        # Return mock response
        response_data = self.mock_responses.get(method_name, {"result": "success"})
        return MockServiceResponse(response_data)

    def _should_trigger_error(self, scenario: dict[str, Any]) -> bool:
        """Check if error scenario should be triggered."""
        trigger_type = scenario.get("trigger_type", "call_count")

        if trigger_type == "call_count":
            return self.call_count == scenario.get("trigger_value", 1)
        elif trigger_type == "method_name":
            return scenario.get("method_name") == scenario.get("trigger_value")
        elif trigger_type == "random":
            return scenario.get("probability", 0.1) > 0.5
        else:
            return False

    def _create_error(self, scenario: dict[str, Any]) -> Exception:
        """Create error based on scenario."""
        error_type = scenario.get("error_type", "ServiceError")
        error_message = scenario.get("error_message", "Mock error")

        if error_type == "ServiceTimeoutError":
            return ServiceTimeoutError(error_message)
        elif error_type == "ServiceUnavailableError":
            return ServiceUnavailableError(error_message)
        else:
            return ServiceError(error_message)

    def reset(self) -> None:
        """Reset mock client state."""
        self.call_count = 0


class ServiceClientTester:
    """Test utilities for service clients."""

    def __init__(self, service_name: str):
        """Initialize service client tester.

        Args:
            service_name: Name of the service being tested

        """
        self.service_name = service_name
        self.mock_client = MockServiceClient(service_name)
        self.test_results = []

    async def test_successful_call(
        self,
        method_name: str,
        client_method: callable,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """Test successful service call.

        Args:
            method_name: Name of the method being tested
            client_method: Client method to test
            *args: Method arguments
            **kwargs: Method keyword arguments

        Returns:
            Test result

        """
        start_time = time.time()

        try:
            # Set up mock response
            self.mock_client.mock_responses[method_name] = {"result": "success"}

            # Execute test
            result = await client_method(*args, **kwargs)

            # Record success
            test_result = {
                "test_name": f"successful_{method_name}",
                "status": "passed",
                "duration": time.time() - start_time,
                "result": result,
                "error": None,
            }

            self.test_results.append(test_result)
            return test_result

        except Exception as e:
            # Record failure
            test_result = {
                "test_name": f"successful_{method_name}",
                "status": "failed",
                "duration": time.time() - start_time,
                "result": None,
                "error": str(e),
            }

            self.test_results.append(test_result)
            return test_result

    async def test_error_handling(
        self,
        method_name: str,
        client_method: callable,
        error_scenario: dict[str, Any],
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """Test error handling.

        Args:
            method_name: Name of the method being tested
            client_method: Client method to test
            error_scenario: Error scenario to test
            *args: Method arguments
            **kwargs: Method keyword arguments

        Returns:
            Test result

        """
        start_time = time.time()

        try:
            # Set up error scenario
            self.mock_client.error_scenarios = [error_scenario]

            # Execute test
            result = await client_method(*args, **kwargs)

            # Record unexpected success
            test_result = {
                "test_name": f"error_handling_{method_name}_{error_scenario['error_type']}",
                "status": "failed",
                "duration": time.time() - start_time,
                "result": result,
                "error": "Expected error but got success",
            }

            self.test_results.append(test_result)
            return test_result

        except Exception as e:
            # Check if error is expected
            expected_error_type = error_scenario.get("error_type", "ServiceError")
            is_expected_error = self._is_expected_error(e, expected_error_type)

            test_result = {
                "test_name": f"error_handling_{method_name}_{error_scenario['error_type']}",
                "status": "passed" if is_expected_error else "failed",
                "duration": time.time() - start_time,
                "result": None,
                "error": str(e),
                "expected_error": expected_error_type,
            }

            self.test_results.append(test_result)
            return test_result

    def _is_expected_error(self, error: Exception, expected_type: str) -> bool:
        """Check if error is of expected type."""
        error_type_map = {
            "ServiceError": ServiceError,
            "ServiceTimeoutError": ServiceTimeoutError,
            "ServiceUnavailableError": ServiceUnavailableError,
        }

        expected_exception = error_type_map.get(expected_type, ServiceError)
        return isinstance(error, expected_exception)

    async def test_retry_logic(
        self,
        method_name: str,
        client_method: callable,
        retry_scenario: dict[str, Any],
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """Test retry logic.

        Args:
            method_name: Name of the method being tested
            client_method: Client method to test
            retry_scenario: Retry scenario to test
            *args: Method arguments
            **kwargs: Method keyword arguments

        Returns:
            Test result

        """
        start_time = time.time()

        try:
            # Set up retry scenario
            self.mock_client.error_scenarios = retry_scenario.get("error_scenarios", [])
            self.mock_client.mock_responses[method_name] = retry_scenario.get(
                "success_response", {"result": "success"}
            )

            # Execute test
            result = await client_method(*args, **kwargs)

            # Check retry count
            expected_retries = retry_scenario.get("expected_retries", 0)
            actual_retries = self.mock_client.call_count - 1  # Subtract 1 for successful call

            test_result = {
                "test_name": f"retry_logic_{method_name}",
                "status": "passed" if actual_retries == expected_retries else "failed",
                "duration": time.time() - start_time,
                "result": result,
                "error": None,
                "expected_retries": expected_retries,
                "actual_retries": actual_retries,
            }

            self.test_results.append(test_result)
            return test_result

        except Exception as e:
            test_result = {
                "test_name": f"retry_logic_{method_name}",
                "status": "failed",
                "duration": time.time() - start_time,
                "result": None,
                "error": str(e),
            }

            self.test_results.append(test_result)
            return test_result

    async def test_performance(
        self,
        method_name: str,
        client_method: callable,
        performance_config: dict[str, Any],
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """Test performance characteristics.

        Args:
            method_name: Name of the method being tested
            client_method: Client method to test
            performance_config: Performance test configuration
            *args: Method arguments
            **kwargs: Method keyword arguments

        Returns:
            Test result

        """
        num_calls = performance_config.get("num_calls", 10)
        max_duration = performance_config.get("max_duration", 1.0)

        start_time = time.time()
        durations = []

        try:
            # Execute multiple calls
            for i in range(num_calls):
                call_start = time.time()
                await client_method(*args, **kwargs)
                call_duration = time.time() - call_start
                durations.append(call_duration)

            total_duration = time.time() - start_time
            avg_duration = sum(durations) / len(durations)
            max_call_duration = max(durations)

            # Check performance criteria
            performance_passed = (
                avg_duration <= max_duration and max_call_duration <= max_duration * 2
            )

            test_result = {
                "test_name": f"performance_{method_name}",
                "status": "passed" if performance_passed else "failed",
                "duration": total_duration,
                "result": {
                    "num_calls": num_calls,
                    "avg_duration": avg_duration,
                    "max_duration": max_call_duration,
                    "total_duration": total_duration,
                },
                "error": None,
                "performance_criteria": {
                    "max_duration": max_duration,
                    "max_call_duration": max_duration * 2,
                },
            }

            self.test_results.append(test_result)
            return test_result

        except Exception as e:
            test_result = {
                "test_name": f"performance_{method_name}",
                "status": "failed",
                "duration": time.time() - start_time,
                "result": None,
                "error": str(e),
            }

            self.test_results.append(test_result)
            return test_result

    def get_test_summary(self) -> dict[str, Any]:
        """Get test summary."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["status"] == "passed")
        failed_tests = total_tests - passed_tests

        return {
            "service_name": self.service_name,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "test_results": self.test_results,
        }

    def reset(self) -> None:
        """Reset test state."""
        self.mock_client.reset()
        self.test_results = []


class ServiceClientTestSuite:
    """Comprehensive test suite for service clients."""

    def __init__(self, service_name: str):
        """Initialize test suite.

        Args:
            service_name: Name of the service being tested

        """
        self.service_name = service_name
        self.tester = ServiceClientTester(service_name)
        self.test_suites = []

    async def run_basic_tests(self, client_methods: dict[str, callable]) -> dict[str, Any]:
        """Run basic functionality tests.

        Args:
            client_methods: Dictionary of method names to client methods

        Returns:
            Test results

        """
        results = {}

        for method_name, client_method in client_methods.items():
            result = await self.tester.test_successful_call(method_name, client_method)
            results[method_name] = result

        return results

    async def run_error_handling_tests(
        self,
        client_methods: dict[str, callable],
        error_scenarios: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run error handling tests.

        Args:
            client_methods: Dictionary of method names to client methods
            error_scenarios: List of error scenarios to test

        Returns:
            Test results

        """
        results = {}

        for method_name, client_method in client_methods.items():
            method_results = []

            for scenario in error_scenarios:
                result = await self.tester.test_error_handling(method_name, client_method, scenario)
                method_results.append(result)

            results[method_name] = method_results

        return results

    async def run_performance_tests(
        self,
        client_methods: dict[str, callable],
        performance_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Run performance tests.

        Args:
            client_methods: Dictionary of method names to client methods
            performance_config: Performance test configuration

        Returns:
            Test results

        """
        results = {}

        for method_name, client_method in client_methods.items():
            result = await self.tester.test_performance(
                method_name, client_method, performance_config
            )
            results[method_name] = result

        return results

    async def run_comprehensive_tests(
        self,
        client_methods: dict[str, callable],
        test_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Run comprehensive test suite.

        Args:
            client_methods: Dictionary of method names to client methods
            test_config: Test configuration

        Returns:
            Comprehensive test results

        """
        results = {
            "basic_tests": await self.run_basic_tests(client_methods),
            "error_handling_tests": await self.run_error_handling_tests(
                client_methods, test_config.get("error_scenarios", [])
            ),
            "performance_tests": await self.run_performance_tests(
                client_methods, test_config.get("performance_config", {})
            ),
        }

        # Add summary
        results["summary"] = self.tester.get_test_summary()

        return results


def create_service_client_tester(service_name: str) -> ServiceClientTester:
    """Create service client tester instance.

    Args:
        service_name: Name of the service

    Returns:
        ServiceClientTester instance

    """
    return ServiceClientTester(service_name)


def create_service_client_test_suite(service_name: str) -> ServiceClientTestSuite:
    """Create service client test suite instance.

    Args:
        service_name: Name of the service

    Returns:
        ServiceClientTestSuite instance

    """
    return ServiceClientTestSuite(service_name)
