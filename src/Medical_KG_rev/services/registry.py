"""Service registry for managing service instances."""

from typing import Any, Dict, Optional, Type, TypeVar

T = TypeVar('T')


class ServiceRegistry:
    """Registry for managing service instances."""

    def __init__(self):
        """Initialize the service registry."""
        self._services: Dict[str, Any] = {}

    def register(self, name: str, service: Any) -> None:
        """Register a service instance."""
        self._services[name] = service

    def get(self, name: str, service_type: Optional[Type[T]] = None) -> Optional[T]:
        """Get a service instance by name."""
        service = self._services.get(name)
        if service_type and service and not isinstance(service, service_type):
            return None
        return service

    def unregister(self, name: str) -> None:
        """Unregister a service instance."""
        self._services.pop(name, None)

    def list_services(self) -> Dict[str, Any]:
        """List all registered services."""
        return self._services.copy()

    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()


# Global service registry instance
service_registry = ServiceRegistry()
