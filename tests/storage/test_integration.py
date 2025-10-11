"""Integration tests for storage functionality with moto and redis-mock."""

from __future__ import annotations

import pytest

from Medical_KG_rev.config.settings import ObjectStorageSettings, RedisCacheSettings
from Medical_KG_rev.storage.clients import create_storage_clients


class TestStorageIntegration:
    """Integration tests using moto for S3 and redis-mock for Redis."""

    @pytest.fixture
    def s3_settings(self) -> ObjectStorageSettings:
        """S3 settings for integration tests."""
        return ObjectStorageSettings(
            bucket="test-medical-kg-pdf",
            region="us-east-1",
            key_prefix="pdf",
        )

    @pytest.fixture
    def redis_settings(self) -> RedisCacheSettings:
        """Redis settings for integration tests."""
        return RedisCacheSettings(
            url="redis://localhost:6379/1",  # Use different DB for tests
            key_prefix="test-medical-kg",
            default_ttl=300,  # 5 minutes for tests
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_s3_pdf_storage_integration(self, s3_settings: ObjectStorageSettings) -> None:
        """Test PDF storage integration with S3 using moto."""
        try:
            from moto import mock_s3
        except ImportError:
            pytest.skip("moto not available for S3 integration tests")

        with mock_s3():
            # Create S3 bucket
            import boto3

            s3_client = boto3.client("s3", region_name=s3_settings.region)
            s3_client.create_bucket(Bucket=s3_settings.bucket)

            # Create storage clients
            redis_settings = RedisCacheSettings()  # Use in-memory for Redis
            clients = create_storage_clients(s3_settings, redis_settings)

            # Test PDF upload
            pdf_data = b"fake pdf content for integration test"
            tenant_id = "integration-tenant"
            document_id = "integration-doc"

            asset = await clients.pdf_storage_client.store_pdf(
                tenant_id=tenant_id,
                document_id=document_id,
                pdf_data=pdf_data,
                content_type="application/pdf",
            )

            assert asset.uri.startswith("s3://test-medical-kg-pdf/pdf/")
            assert asset.checksum is not None
            assert asset.size == len(pdf_data)

            # Test PDF retrieval
            retrieved_data = await clients.pdf_storage_client.get_pdf_data(
                tenant_id, document_id, asset.checksum
            )

            assert retrieved_data == pdf_data

            # Test presigned URL generation
            presigned_url = clients.pdf_storage_client.get_presigned_url(
                tenant_id, document_id, asset.checksum, expires_in=3600
            )

            assert presigned_url.startswith("https://")
            assert asset.checksum in presigned_url

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_redis_cache_integration(self, redis_settings: RedisCacheSettings) -> None:
        """Test Redis cache integration."""
        try:
            pass
        except ImportError:
            pytest.skip("redis not available for integration tests")

        # Create in-memory object store for this test
        object_settings = ObjectStorageSettings(bucket="test-bucket")
        clients = create_storage_clients(object_settings, redis_settings)

        # Test cache operations
        test_key = "test:integration:key"
        test_value = {"test": "data", "number": 42}

        # Set value in cache
        await clients.cache_backend.set(test_key, test_value, ttl=60)

        # Get value from cache
        retrieved_value = await clients.cache_backend.get(test_key)
        assert retrieved_value == test_value

        # Test cache expiration
        await clients.cache_backend.set(test_key, test_value, ttl=1)
        import asyncio

        await asyncio.sleep(1.1)  # Wait for expiration

        expired_value = await clients.cache_backend.get(test_key)
        assert expired_value is None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_document_artifact_storage_integration(
        self, s3_settings: ObjectStorageSettings
    ) -> None:
        """Test document artifact storage integration."""
        try:
            from moto import mock_s3
        except ImportError:
            pytest.skip("moto not available for S3 integration tests")

        with mock_s3():
            # Create S3 bucket
            import boto3

            s3_client = boto3.client("s3", region_name=s3_settings.region)
            s3_client.create_bucket(Bucket=s3_settings.bucket)

            # Create storage clients
            redis_settings = RedisCacheSettings()
            clients = create_storage_clients(s3_settings, redis_settings)

            # Test artifact upload
            artifact_data = b'{"docling_output": "test data"}'
            tenant_id = "integration-tenant"
            document_id = "integration-doc"
            artifact_type = "docling_output"
            file_extension = "json"

            uri = await clients.document_storage_client.upload_document_artifact(
                tenant_id=tenant_id,
                document_id=document_id,
                artifact_type=artifact_type,
                data=artifact_data,
                file_extension=file_extension,
            )

            assert uri.startswith("s3://test-medical-kg-pdf/documents/")
            assert artifact_type in uri
            assert file_extension in uri

            # Extract checksum from URI for retrieval
            import hashlib

            checksum = hashlib.sha256(artifact_data).hexdigest()

            # Test artifact retrieval
            retrieved_data = await clients.document_storage_client.get_document_artifact(
                tenant_id, document_id, artifact_type, checksum, file_extension
            )

            assert retrieved_data == artifact_data

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_storage_error_handling(self, s3_settings: ObjectStorageSettings) -> None:
        """Test storage error handling scenarios."""
        try:
            from moto import mock_s3
        except ImportError:
            pytest.skip("moto not available for S3 integration tests")

        with mock_s3():
            # Don't create bucket - this should cause errors
            redis_settings = RedisCacheSettings()
            clients = create_storage_clients(s3_settings, redis_settings)

            pdf_data = b"test data"
            tenant_id = "test-tenant"
            document_id = "test-doc"

            # This should raise an exception due to missing bucket
            with pytest.raises(Exception):  # Specific exception type depends on implementation
                await clients.pdf_storage_client.store_pdf(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    pdf_data=pdf_data,
                )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_storage_operations(self, s3_settings: ObjectStorageSettings) -> None:
        """Test concurrent storage operations."""
        try:
            from moto import mock_s3
        except ImportError:
            pytest.skip("moto not available for S3 integration tests")

        with mock_s3():
            # Create S3 bucket
            import boto3

            s3_client = boto3.client("s3", region_name=s3_settings.region)
            s3_client.create_bucket(Bucket=s3_settings.bucket)

            redis_settings = RedisCacheSettings()
            clients = create_storage_clients(s3_settings, redis_settings)

            # Test concurrent uploads
            import asyncio

            async def upload_pdf(doc_id: str) -> None:
                pdf_data = f"fake pdf content for doc {doc_id}".encode()
                await clients.pdf_storage_client.store_pdf(
                    tenant_id="concurrent-tenant",
                    document_id=doc_id,
                    pdf_data=pdf_data,
                )

            # Upload 5 PDFs concurrently
            tasks = [upload_pdf(f"doc-{i}") for i in range(5)]
            await asyncio.gather(*tasks)

            # Verify all uploads succeeded by checking S3
            s3_client = boto3.client("s3", region_name=s3_settings.region)
            response = s3_client.list_objects_v2(Bucket=s3_settings.bucket)

            assert "Contents" in response
            assert len(response["Contents"]) == 5

            # Verify all objects have the correct prefix
            for obj in response["Contents"]:
                assert obj["Key"].startswith("pdf/concurrent-tenant/doc-")


class TestStoragePerformance:
    """Performance tests for storage operations."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_file_upload_performance(self) -> None:
        """Test performance with large file uploads."""
        try:
            from moto import mock_s3
        except ImportError:
            pytest.skip("moto not available for performance tests")

        with mock_s3():
            # Create S3 bucket
            import boto3

            s3_settings = ObjectStorageSettings(
                bucket="test-performance-bucket",
                region="us-east-1",
                max_file_size=50 * 1024 * 1024,  # 50MB max
            )

            s3_client = boto3.client("s3", region_name=s3_settings.region)
            s3_client.create_bucket(Bucket=s3_settings.bucket)

            redis_settings = RedisCacheSettings()
            clients = create_storage_clients(s3_settings, redis_settings)

            # Create large PDF data (1MB)
            large_pdf_data = b"x" * (1024 * 1024)  # 1MB

            import time

            start_time = time.time()

            asset = await clients.pdf_storage_client.store_pdf(
                tenant_id="performance-tenant",
                document_id="large-doc",
                pdf_data=large_pdf_data,
            )

            upload_time = time.time() - start_time

            # Verify upload completed within reasonable time (adjust threshold as needed)
            assert upload_time < 10.0  # 10 seconds max for 1MB upload
            assert asset.size == len(large_pdf_data)

            # Test retrieval performance
            start_time = time.time()

            retrieved_data = await clients.pdf_storage_client.get_pdf_data(
                "performance-tenant", "large-doc", asset.checksum
            )

            retrieval_time = time.time() - start_time

            # Verify retrieval completed within reasonable time
            assert retrieval_time < 5.0  # 5 seconds max for 1MB retrieval
            assert retrieved_data == large_pdf_data

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_performance(self) -> None:
        """Test cache performance with many operations."""
        redis_settings = RedisCacheSettings(
            key_prefix="perf-test",
            default_ttl=3600,
        )

        object_settings = ObjectStorageSettings(bucket="test-bucket")
        clients = create_storage_clients(object_settings, redis_settings)

        # Test many cache operations
        import time

        start_time = time.time()

        # Set many cache entries
        for i in range(100):
            await clients.cache_backend.set(f"perf-key-{i}", {"data": f"value-{i}"})

        # Get many cache entries
        for i in range(100):
            value = await clients.cache_backend.get(f"perf-key-{i}")
            assert value == {"data": f"value-{i}"}

        total_time = time.time() - start_time

        # Verify operations completed within reasonable time
        assert total_time < 5.0  # 5 seconds max for 200 operations
