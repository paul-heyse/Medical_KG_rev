#!/usr/bin/env python3
"""Test script to verify MinerU can connect to vLLM server.

This script tests the integration between MinerU worker and vLLM server
by checking:
1. vLLM server is reachable
2. vLLM server health check passes
3. MinerU settings are correctly configured
4. HTTP client can establish connection

Usage:
    python scripts/test_mineru_vllm_integration.py
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

try:
    import httpx

    from Medical_KG_rev.config.settings import get_settings  # type: ignore[import-untyped]
    from Medical_KG_rev.services.mineru.vllm_client import (
        VLLMClient,  # type: ignore[import-untyped]
    )
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print("Please ensure the package is installed: pip install -e .")
    sys.exit(1)


async def test_vllm_connectivity() -> dict[str, Any]:
    """Test vLLM server connectivity and health.

    Returns:
        Dictionary with test results
    """
    print("=" * 70)
    print("Testing MinerU → vLLM Integration")
    print("=" * 70)

    results: dict[str, Any] = {
        "settings_loaded": False,
        "vllm_url": None,
        "vllm_reachable": False,
        "vllm_healthy": False,
        "client_initialized": False,
        "errors": [],
    }

    # Test 1: Load settings
    print("\n1. Loading MinerU settings...")
    try:
        settings = get_settings()
        mineru_settings = settings.mineru
        results["settings_loaded"] = True
        results["vllm_url"] = str(mineru_settings.vllm_server.base_url)
        print(f"   ✓ Settings loaded successfully")
        print(f"   ✓ vLLM URL: {results['vllm_url']}")
        print(f"   ✓ Backend: {mineru_settings.workers.backend}")
        print(f"   ✓ Workers: {mineru_settings.workers.count}")
    except Exception as e:
        error_msg = f"Failed to load settings: {e}"
        print(f"   ✗ {error_msg}")
        results["errors"].append(error_msg)
        return results

    # Test 2: Check basic HTTP connectivity
    print("\n2. Testing basic HTTP connectivity to vLLM server...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.get(f"{results['vllm_url']}/health")
            results["vllm_reachable"] = True
            print(f"   ✓ vLLM server is reachable")
            print(f"   ✓ HTTP status: {response.status_code}")

            if response.status_code == 200:
                results["vllm_healthy"] = True
                print(f"   ✓ vLLM server reports healthy status")
            else:
                print(f"   ⚠ vLLM server responded but not healthy: {response.status_code}")
    except httpx.ConnectError as e:
        error_msg = f"Cannot connect to vLLM server: {e}"
        print(f"   ✗ {error_msg}")
        print(f"   ℹ Make sure vLLM server is running: docker-compose up vllm-server")
        results["errors"].append(error_msg)
    except Exception as e:
        error_msg = f"HTTP request failed: {e}"
        print(f"   ✗ {error_msg}")
        results["errors"].append(error_msg)

    # Test 3: Initialize VLLMClient
    print("\n3. Initializing VLLMClient...")
    try:
        client = VLLMClient(
            base_url=str(mineru_settings.vllm_server.base_url),
            timeout=mineru_settings.http_client.timeout_seconds,
            max_connections=mineru_settings.http_client.connection_pool_size,
            max_keepalive_connections=mineru_settings.http_client.keepalive_connections,
            retry_attempts=mineru_settings.http_client.retry_attempts,
            retry_backoff_multiplier=mineru_settings.http_client.retry_backoff_multiplier,
        )
        results["client_initialized"] = True
        print(f"   ✓ VLLMClient initialized successfully")

        # Test health check through client
        async with client:
            healthy = await client.health_check()
            if healthy:
                print(f"   ✓ Health check through VLLMClient passed")
            else:
                print(f"   ⚠ Health check through VLLMClient failed")
                results["errors"].append("VLLMClient health check failed")
    except Exception as e:
        error_msg = f"Failed to initialize VLLMClient: {e}"
        print(f"   ✗ {error_msg}")
        results["errors"].append(error_msg)

    # Test 4: Test chat completion (if server is healthy)
    if results["vllm_healthy"] and results["client_initialized"]:
        print("\n4. Testing chat completion...")
        try:
            vllm_client = VLLMClient(
                base_url=str(mineru_settings.vllm_server.base_url),
                timeout=30.0,  # Shorter timeout for test
            )
            async with vllm_client:
                chat_response: dict[str, Any] = await vllm_client.chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": "Say 'Hello, MinerU!' if you can read this.",
                        }
                    ],
                    max_tokens=50,
                    temperature=0.0,
                )
                if chat_response and "choices" in chat_response:
                    message_content = chat_response["choices"][0]["message"]["content"]
                    content = str(message_content) if message_content else ""
                    print(f"   ✓ Chat completion successful")
                    print(f"   ✓ Response: {content[:100]}")
                    results["chat_completion"] = True
                else:
                    print(f"   ⚠ Unexpected response format")
                    results["chat_completion"] = False
        except Exception as e:
            error_msg = f"Chat completion failed: {e}"
            print(f"   ⚠ {error_msg}")
            print(f"   ℹ This may be expected if the model is still loading")
            results["errors"].append(error_msg)
            results["chat_completion"] = False
    else:
        print("\n4. Skipping chat completion test (prerequisites not met)")
        results["chat_completion"] = False

    return results


async def main() -> int:
    """Run all tests and return exit code."""
    results = await test_vllm_connectivity()

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Settings loaded:       {'✓' if results['settings_loaded'] else '✗'}")
    print(f"vLLM reachable:        {'✓' if results['vllm_reachable'] else '✗'}")
    print(f"vLLM healthy:          {'✓' if results['vllm_healthy'] else '✗'}")
    print(f"Client initialized:    {'✓' if results['client_initialized'] else '✗'}")
    print(f"Chat completion:       {'✓' if results.get('chat_completion') else '✗'}")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for i, error in enumerate(results["errors"], 1):
            print(f"  {i}. {error}")

    # Determine success
    critical_tests = [
        results["settings_loaded"],
        results["vllm_reachable"],
        results["vllm_healthy"],
        results["client_initialized"],
    ]

    if all(critical_tests):
        print("\n✓ All critical tests passed! MinerU can connect to vLLM.")
        return 0
    else:
        print("\n✗ Some critical tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

