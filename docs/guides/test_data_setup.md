# Test Data Setup and Management

This document provides comprehensive guidance on test data setup and management for the Medical_KG_rev system, including data generation, fixtures, mocking strategies, and test environment configuration.

## Overview

Effective test data management is crucial for reliable testing of the Medical_KG_rev system. This guide covers strategies for creating, managing, and maintaining test data across different testing scenarios and environments.

## Test Data Categories

### Unit Test Data

- **Mock Objects**: Simple mock data for isolated testing
- **Fixtures**: Reusable test data structures
- **Factories**: Dynamic test data generation

### Integration Test Data

- **Database Fixtures**: Realistic database state
- **API Responses**: Mocked external API responses
- **File Content**: Sample documents and files

### Performance Test Data

- **Large Datasets**: Scalable test data for performance testing
- **Load Patterns**: Realistic usage patterns
- **Stress Data**: Edge cases and boundary conditions

## Test Data Generation

### Factory Pattern Implementation

#### Base Factory Class

```python
# tests/factories/base.py
import factory
from factory import Faker, SubFactory, LazyAttribute
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import uuid

class BaseFactory(factory.Factory):
    """Base factory class for all test data."""

    class Meta:
        abstract = True

    @classmethod
    def create_batch(cls, size: int, **kwargs) -> List[Any]:
        """Create a batch of instances."""
        return [cls(**kwargs) for _ in range(size)]

    @classmethod
    def build_batch(cls, size: int, **kwargs) -> List[Any]:
        """Build a batch of instances without saving."""
        return [cls.build(**kwargs) for _ in range(size)]

class TimestampFactory(BaseFactory):
    """Factory for timestamp-related fields."""

    created_at = factory.LazyFunction(datetime.utcnow)
    updated_at = factory.LazyFunction(datetime.utcnow)

    @classmethod
    def with_timestamps(cls, days_ago: int = 0, **kwargs):
        """Create instance with specific timestamp."""
        base_time = datetime.utcnow() - timedelta(days=days_ago)
        return cls(
            created_at=base_time,
            updated_at=base_time,
            **kwargs
        )
```

#### User Factory

```python
# tests/factories/user.py
import factory
from factory import Faker, SubFactory, LazyAttribute
from Medical_KG_rev.models.user import User
from tests.factories.base import BaseFactory, TimestampFactory

class UserFactory(TimestampFactory):
    """Factory for User model."""

    class Meta:
        model = User

    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    email = Faker('email')
    username = Faker('user_name')
    password_hash = factory.LazyFunction(lambda: "hashed_password_123")
    first_name = Faker('first_name')
    last_name = Faker('last_name')
    is_active = True
    is_verified = True

    @classmethod
    def inactive(cls, **kwargs):
        """Create inactive user."""
        return cls(is_active=False, **kwargs)

    @classmethod
    def unverified(cls, **kwargs):
        """Create unverified user."""
        return cls(is_verified=False, **kwargs)

    @classmethod
    def with_email(cls, email: str, **kwargs):
        """Create user with specific email."""
        return cls(email=email, **kwargs)

    @classmethod
    def with_username(cls, username: str, **kwargs):
        """Create user with specific username."""
        return cls(username=username, **kwargs)
```

#### Document Factory

```python
# tests/factories/document.py
import factory
from factory import Faker, SubFactory, LazyAttribute
from Medical_KG_rev.models.document import Document
from tests.factories.base import BaseFactory, TimestampFactory
from tests.factories.user import UserFactory

class DocumentFactory(TimestampFactory):
    """Factory for Document model."""

    class Meta:
        model = Document

    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    title = Faker('sentence', nb_words=6)
    content = Faker('text', max_nb_chars=1000)
    source = Faker('random_element', elements=['pubmed', 'clinicaltrials', 'openalex'])
    metadata = factory.LazyFunction(lambda: {
        'doi': Faker('bothify', text='10.1000/####.####').generate(),
        'pmcid': Faker('bothify', text='PMC#######').generate(),
        'authors': [Faker('name').generate() for _ in range(3)],
        'journal': Faker('company').generate(),
        'year': Faker('year'),
        'keywords': [Faker('word').generate() for _ in range(5)]
    })
    created_by = SubFactory(UserFactory)

    @classmethod
    def with_source(cls, source: str, **kwargs):
        """Create document with specific source."""
        return cls(source=source, **kwargs)

    @classmethod
    def with_content(cls, content: str, **kwargs):
        """Create document with specific content."""
        return cls(content=content, **kwargs)

    @classmethod
    def with_metadata(cls, metadata: dict, **kwargs):
        """Create document with specific metadata."""
        return cls(metadata=metadata, **kwargs)

    @classmethod
    def pubmed_document(cls, **kwargs):
        """Create PubMed document."""
        return cls(
            source='pubmed',
            metadata={
                'pmid': Faker('bothify', text='#########').generate(),
                'doi': Faker('bothify', text='10.1000/####.####').generate(),
                'authors': [Faker('name').generate() for _ in range(3)],
                'journal': Faker('company').generate(),
                'year': Faker('year'),
                'abstract': Faker('text', max_nb_chars=500).generate()
            },
            **kwargs
        )

    @classmethod
    def clinical_trial_document(cls, **kwargs):
        """Create clinical trial document."""
        return cls(
            source='clinicaltrials',
            metadata={
                'nct_id': Faker('bothify', text='NCT########').generate(),
                'title': Faker('sentence', nb_words=8).generate(),
                'phase': Faker('random_element', elements=['Phase I', 'Phase II', 'Phase III', 'Phase IV']).generate(),
                'status': Faker('random_element', elements=['Recruiting', 'Active', 'Completed', 'Terminated']).generate(),
                'sponsor': Faker('company').generate(),
                'conditions': [Faker('word').generate() for _ in range(3)]
            },
            **kwargs
        )
```

#### Chunk Factory

```python
# tests/factories/chunk.py
import factory
from factory import Faker, SubFactory, LazyAttribute
from Medical_KG_rev.models.chunk import Chunk
from tests.factories.base import BaseFactory, TimestampFactory
from tests.factories.document import DocumentFactory

class ChunkFactory(TimestampFactory):
    """Factory for Chunk model."""

    class Meta:
        model = Chunk

    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    document = SubFactory(DocumentFactory)
    content = Faker('text', max_nb_chars=500)
    start_position = factory.LazyFunction(lambda: Faker('random_int', min=0, max=1000).generate())
    end_position = factory.LazyFunction(lambda: Faker('random_int', min=1000, max=2000).generate())
    chunk_index = factory.LazyFunction(lambda: Faker('random_int', min=0, max=10).generate())
    metadata = factory.LazyFunction(lambda: {
        'chunk_type': Faker('random_element', elements=['paragraph', 'section', 'table', 'figure']).generate(),
        'language': 'en',
        'word_count': Faker('random_int', min=50, max=200).generate()
    })

    @classmethod
    def with_document(cls, document, **kwargs):
        """Create chunk for specific document."""
        return cls(document=document, **kwargs)

    @classmethod
    def with_content(cls, content: str, **kwargs):
        """Create chunk with specific content."""
        return cls(content=content, **kwargs)

    @classmethod
    def with_positions(cls, start: int, end: int, **kwargs):
        """Create chunk with specific positions."""
        return cls(start_position=start, end_position=end, **kwargs)
```

### Fixture Management

#### Database Fixtures

```python
# tests/fixtures/database.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from Medical_KG_rev.storage.database import Base
from tests.factories.user import UserFactory
from tests.factories.document import DocumentFactory
from tests.factories.chunk import ChunkFactory

@pytest.fixture(scope="session")
def engine():
    """Create test database engine."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine

@pytest.fixture(scope="function")
def session(engine):
    """Create test database session."""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.close()

@pytest.fixture
def sample_user(session):
    """Create sample user."""
    user = UserFactory()
    session.add(user)
    session.commit()
    return user

@pytest.fixture
def sample_document(session, sample_user):
    """Create sample document."""
    document = DocumentFactory(created_by=sample_user)
    session.add(document)
    session.commit()
    return document

@pytest.fixture
def sample_chunks(session, sample_document):
    """Create sample chunks."""
    chunks = ChunkFactory.create_batch(3, document=sample_document)
    for chunk in chunks:
        session.add(chunk)
    session.commit()
    return chunks

@pytest.fixture
def populated_database(session):
    """Create populated test database."""
    # Create users
    users = UserFactory.create_batch(5)
    for user in users:
        session.add(user)

    # Create documents
    documents = []
    for user in users:
        user_docs = DocumentFactory.create_batch(3, created_by=user)
        documents.extend(user_docs)
        for doc in user_docs:
            session.add(doc)

    # Create chunks
    for document in documents:
        chunks = ChunkFactory.create_batch(2, document=document)
        for chunk in chunks:
            session.add(chunk)

    session.commit()
    return {
        'users': users,
        'documents': documents,
        'chunks': [chunk for doc in documents for chunk in doc.chunks]
    }
```

#### API Response Fixtures

```python
# tests/fixtures/api_responses.py
import pytest
import json
from pathlib import Path

@pytest.fixture
def pubmed_response():
    """Sample PubMed API response."""
    return {
        "esearchresult": {
            "count": "1",
            "retmax": "1",
            "retstart": "0",
            "idlist": ["12345678"],
            "translationset": [],
            "translationstack": [],
            "querytranslation": "test query"
        }
    }

@pytest.fixture
def pubmed_summary_response():
    """Sample PubMed summary response."""
    return {
        "result": {
            "12345678": {
                "uid": "12345678",
                "title": "Sample Research Article",
                "authors": [
                    {"name": "John Doe", "authtype": "Author"},
                    {"name": "Jane Smith", "authtype": "Author"}
                ],
                "source": "Journal of Medical Research",
                "pubdate": "2024-01-01",
                "doi": "10.1000/sample.2024.001",
                "abstract": "This is a sample abstract for testing purposes."
            }
        }
    }

@pytest.fixture
def clinicaltrials_response():
    """Sample ClinicalTrials.gov API response."""
    return {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT12345678",
                        "briefTitle": "Sample Clinical Trial",
                        "officialTitle": "A Phase III Randomized Clinical Trial of Sample Treatment"
                    },
                    "statusModule": {
                        "overallStatus": "Recruiting",
                        "phase": "PHASE3"
                    },
                    "sponsorCollaboratorsModule": {
                        "leadSponsor": {
                            "name": "Sample Pharmaceutical Company"
                        }
                    },
                    "conditionsModule": {
                        "conditions": ["Sample Condition", "Another Condition"]
                    }
                }
            }
        ]
    }

@pytest.fixture
def openalex_response():
    """Sample OpenAlex API response."""
    return {
        "id": "https://openalex.org/W1234567890",
        "title": "Sample Research Paper",
        "authors": [
            {
                "id": "https://openalex.org/A1234567890",
                "name": "John Doe",
                "orcid": "https://orcid.org/0000-0000-0000-0000"
            }
        ],
        "publication_date": "2024-01-01",
        "doi": "https://doi.org/10.1000/sample.2024.001",
        "abstract": "This is a sample abstract for testing purposes.",
        "keywords": ["keyword1", "keyword2", "keyword3"]
    }
```

### Mock Data Generation

#### Mock External APIs

```python
# tests/mocks/external_apis.py
import pytest
from unittest.mock import Mock, patch
from tests.fixtures.api_responses import pubmed_response, clinicaltrials_response, openalex_response

@pytest.fixture
def mock_pubmed_api():
    """Mock PubMed API."""
    with patch('Medical_KG_rev.adapters.pubmed.PubMedAdapter._make_request') as mock_request:
        mock_request.return_value = pubmed_response()
        yield mock_request

@pytest.fixture
def mock_clinicaltrials_api():
    """Mock ClinicalTrials.gov API."""
    with patch('Medical_KG_rev.adapters.clinicaltrials.ClinicalTrialsAdapter._make_request') as mock_request:
        mock_request.return_value = clinicaltrials_response()
        yield mock_request

@pytest.fixture
def mock_openalex_api():
    """Mock OpenAlex API."""
    with patch('Medical_KG_rev.adapters.openalex.OpenAlexAdapter._make_request') as mock_request:
        mock_request.return_value = openalex_response()
        yield mock_request

@pytest.fixture
def mock_all_external_apis(mock_pubmed_api, mock_clinicaltrials_api, mock_openalex_api):
    """Mock all external APIs."""
    return {
        'pubmed': mock_pubmed_api,
        'clinicaltrials': mock_clinicaltrials_api,
        'openalex': mock_openalex_api
    }
```

#### Mock File System

```python
# tests/mocks/filesystem.py
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

@pytest.fixture
def mock_file_system():
    """Mock file system operations."""
    with patch('pathlib.Path.exists') as mock_exists, \
         patch('pathlib.Path.read_text') as mock_read_text, \
         patch('pathlib.Path.write_text') as mock_write_text:

        mock_exists.return_value = True
        mock_read_text.return_value = "Sample file content"
        mock_write_text.return_value = None

        yield {
            'exists': mock_exists,
            'read_text': mock_read_text,
            'write_text': mock_write_text
        }
```

## Test Environment Configuration

### Environment Setup

#### Test Configuration

```python
# tests/conftest.py
import pytest
import os
import tempfile
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from Medical_KG_rev.storage.database import Base
from Medical_KG_rev.storage.neo4j import Neo4jManager
from Medical_KG_rev.storage.redis import RedisManager

@pytest.fixture(scope="session")
def test_config():
    """Test configuration."""
    return {
        'DATABASE_URL': 'sqlite:///:memory:',
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USER': 'neo4j',
        'NEO4J_PASSWORD': 'test',
        'REDIS_URL': 'redis://localhost:6379/1',
        'GATEWAY_HOST': '0.0.0.0',
        'GATEWAY_PORT': '8000',
        'GATEWAY_LOG_LEVEL': 'DEBUG'
    }

@pytest.fixture(scope="session")
def test_environment(test_config):
    """Set up test environment."""
    # Set environment variables
    for key, value in test_config.items():
        os.environ[key] = value

    yield test_config

    # Cleanup
    for key in test_config.keys():
        if key in os.environ:
            del os.environ[key]

@pytest.fixture(scope="session")
def test_database(test_config):
    """Create test database."""
    engine = create_engine(test_config['DATABASE_URL'])
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)

@pytest.fixture(scope="function")
def test_session(test_database):
    """Create test session."""
    Session = sessionmaker(bind=test_database)
    session = Session()
    yield session
    session.rollback()
    session.close()

@pytest.fixture(scope="session")
def test_neo4j(test_config):
    """Create test Neo4j connection."""
    manager = Neo4jManager(
        uri=test_config['NEO4J_URI'],
        user=test_config['NEO4J_USER'],
        password=test_config['NEO4J_PASSWORD']
    )
    yield manager
    manager.close()

@pytest.fixture(scope="session")
def test_redis(test_config):
    """Create test Redis connection."""
    manager = RedisManager(url=test_config['REDIS_URL'])
    yield manager
    manager.close()

@pytest.fixture(scope="function")
def clean_neo4j(test_neo4j):
    """Clean Neo4j database."""
    with test_neo4j.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    yield
    with test_neo4j.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

@pytest.fixture(scope="function")
def clean_redis(test_redis):
    """Clean Redis database."""
    test_redis.flushdb()
    yield
    test_redis.flushdb()
```

### Docker Test Environment

#### Docker Compose for Testing

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  postgres-test:
    image: postgres:14
    environment:
      POSTGRES_DB: medical_kg_test
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
    ports:
      - "5433:5432"
    volumes:
      - postgres_test_data:/var/lib/postgresql/data
    networks:
      - test-network

  neo4j-test:
    image: neo4j:5
    environment:
      NEO4J_AUTH: neo4j/test
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - "7688:7687"
      - "7475:7474"
    volumes:
      - neo4j_test_data:/data
    networks:
      - test-network

  redis-test:
    image: redis:7
    ports:
      - "6380:6379"
    volumes:
      - redis_test_data:/data
    networks:
      - test-network

  qdrant-test:
    image: qdrant/qdrant:latest
    ports:
      - "6334:6333"
    volumes:
      - qdrant_test_data:/qdrant/storage
    networks:
      - test-network

volumes:
  postgres_test_data:
  neo4j_test_data:
  redis_test_data:
  qdrant_test_data:

networks:
  test-network:
    driver: bridge
```

#### Test Environment Script

```python
# scripts/setup_test_environment.py
import subprocess
import time
import requests
from pathlib import Path

def wait_for_service(url: str, timeout: int = 60):
    """Wait for service to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False

def setup_test_environment():
    """Set up test environment."""
    print("Setting up test environment...")

    # Start test services
    subprocess.run([
        "docker-compose", "-f", "docker-compose.test.yml", "up", "-d"
    ], check=True)

    # Wait for services to be ready
    services = [
        ("http://localhost:7475", "Neo4j"),
        ("http://localhost:6334/health", "Qdrant"),
        ("http://localhost:6380", "Redis")
    ]

    for url, name in services:
        print(f"Waiting for {name}...")
        if wait_for_service(url):
            print(f"✅ {name} is ready")
        else:
            print(f"❌ {name} failed to start")
            return False

    print("✅ Test environment is ready")
    return True

def teardown_test_environment():
    """Tear down test environment."""
    print("Tearing down test environment...")

    subprocess.run([
        "docker-compose", "-f", "docker-compose.test.yml", "down", "-v"
    ], check=True)

    print("✅ Test environment torn down")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "teardown":
        teardown_test_environment()
    else:
        setup_test_environment()
```

## Performance Test Data

### Large Dataset Generation

#### Scalable Data Generation

```python
# tests/performance/data_generator.py
import asyncio
import aiohttp
from typing import List, Dict, Any
from tests.factories.document import DocumentFactory
from tests.factories.user import UserFactory
from tests.factories.chunk import ChunkFactory

class PerformanceDataGenerator:
    """Generate large datasets for performance testing."""

    def __init__(self, session):
        self.session = session

    def generate_users(self, count: int) -> List[Any]:
        """Generate users for performance testing."""
        users = UserFactory.create_batch(count)
        for user in users:
            self.session.add(user)
        self.session.commit()
        return users

    def generate_documents(self, count: int, users: List[Any]) -> List[Any]:
        """Generate documents for performance testing."""
        documents = []
        for i in range(count):
            user = users[i % len(users)]
            document = DocumentFactory(created_by=user)
            documents.append(document)
            self.session.add(document)
        self.session.commit()
        return documents

    def generate_chunks(self, count: int, documents: List[Any]) -> List[Any]:
        """Generate chunks for performance testing."""
        chunks = []
        for i in range(count):
            document = documents[i % len(documents)]
            chunk = ChunkFactory(document=document)
            chunks.append(chunk)
            self.session.add(chunk)
        self.session.commit()
        return chunks

    def generate_full_dataset(self, user_count: int, doc_count: int, chunk_count: int):
        """Generate full dataset for performance testing."""
        print(f"Generating {user_count} users...")
        users = self.generate_users(user_count)

        print(f"Generating {doc_count} documents...")
        documents = self.generate_documents(doc_count, users)

        print(f"Generating {chunk_count} chunks...")
        chunks = self.generate_chunks(chunk_count, documents)

        return {
            'users': users,
            'documents': documents,
            'chunks': chunks
        }

# Performance test data sizes
PERFORMANCE_DATA_SIZES = {
    'small': {'users': 100, 'documents': 500, 'chunks': 1000},
    'medium': {'users': 1000, 'documents': 5000, 'chunks': 10000},
    'large': {'users': 10000, 'documents': 50000, 'chunks': 100000},
    'xlarge': {'users': 100000, 'documents': 500000, 'chunks': 1000000}
}
```

### Load Test Data

#### Realistic Usage Patterns

```python
# tests/performance/load_patterns.py
import random
from typing import List, Dict, Any
from datetime import datetime, timedelta

class LoadPatternGenerator:
    """Generate realistic load patterns for testing."""

    def __init__(self):
        self.patterns = {
            'normal': self.normal_pattern,
            'peak': self.peak_pattern,
            'spike': self.spike_pattern,
            'gradual': self.gradual_pattern
        }

    def normal_pattern(self, duration_minutes: int) -> List[Dict[str, Any]]:
        """Generate normal load pattern."""
        requests = []
        base_time = datetime.utcnow()

        for minute in range(duration_minutes):
            # Normal distribution around 100 requests per minute
            request_count = max(0, int(random.normalvariate(100, 20)))

            for _ in range(request_count):
                requests.append({
                    'timestamp': base_time + timedelta(minutes=minute, seconds=random.randint(0, 59)),
                    'endpoint': random.choice(['/api/v1/documents', '/api/v1/search', '/api/v1/users']),
                    'method': random.choice(['GET', 'POST']),
                    'response_time': random.normalvariate(200, 50)
                })

        return sorted(requests, key=lambda x: x['timestamp'])

    def peak_pattern(self, duration_minutes: int) -> List[Dict[str, Any]]:
        """Generate peak load pattern."""
        requests = []
        base_time = datetime.utcnow()

        for minute in range(duration_minutes):
            # Peak during middle 20% of duration
            if duration_minutes * 0.4 <= minute <= duration_minutes * 0.6:
                request_count = random.randint(500, 1000)
            else:
                request_count = random.randint(50, 150)

            for _ in range(request_count):
                requests.append({
                    'timestamp': base_time + timedelta(minutes=minute, seconds=random.randint(0, 59)),
                    'endpoint': random.choice(['/api/v1/documents', '/api/v1/search', '/api/v1/users']),
                    'method': random.choice(['GET', 'POST']),
                    'response_time': random.normalvariate(200, 50)
                })

        return sorted(requests, key=lambda x: x['timestamp'])

    def spike_pattern(self, duration_minutes: int) -> List[Dict[str, Any]]:
        """Generate spike load pattern."""
        requests = []
        base_time = datetime.utcnow()

        for minute in range(duration_minutes):
            # Random spikes
            if random.random() < 0.1:  # 10% chance of spike
                request_count = random.randint(1000, 2000)
            else:
                request_count = random.randint(50, 100)

            for _ in range(request_count):
                requests.append({
                    'timestamp': base_time + timedelta(minutes=minute, seconds=random.randint(0, 59)),
                    'endpoint': random.choice(['/api/v1/documents', '/api/v1/search', '/api/v1/users']),
                    'method': random.choice(['GET', 'POST']),
                    'response_time': random.normalvariate(200, 50)
                })

        return sorted(requests, key=lambda x: x['timestamp'])

    def gradual_pattern(self, duration_minutes: int) -> List[Dict[str, Any]]:
        """Generate gradual load pattern."""
        requests = []
        base_time = datetime.utcnow()

        for minute in range(duration_minutes):
            # Gradual increase from 50 to 500 requests per minute
            progress = minute / duration_minutes
            request_count = int(50 + (450 * progress))

            for _ in range(request_count):
                requests.append({
                    'timestamp': base_time + timedelta(minutes=minute, seconds=random.randint(0, 59)),
                    'endpoint': random.choice(['/api/v1/documents', '/api/v1/search', '/api/v1/users']),
                    'method': random.choice(['GET', 'POST']),
                    'response_time': random.normalvariate(200, 50)
                })

        return sorted(requests, key=lambda x: x['timestamp'])

    def generate_pattern(self, pattern_name: str, duration_minutes: int) -> List[Dict[str, Any]]:
        """Generate load pattern by name."""
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")

        return self.patterns[pattern_name](duration_minutes)
```

## Test Data Management

### Data Cleanup

#### Automatic Cleanup

```python
# tests/cleanup.py
import pytest
from sqlalchemy import text
from Medical_KG_rev.storage.database import get_engine
from Medical_KG_rev.storage.neo4j import Neo4jManager
from Medical_KG_rev.storage.redis import RedisManager

class TestDataCleanup:
    """Test data cleanup utilities."""

    def __init__(self, database_url: str, neo4j_uri: str, redis_url: str):
        self.engine = get_engine(database_url)
        self.neo4j = Neo4jManager(uri=neo4j_uri, user='neo4j', password='test')
        self.redis = RedisManager(url=redis_url)

    def cleanup_database(self):
        """Clean up database."""
        with self.engine.connect() as conn:
            # Disable foreign key checks
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))

            # Truncate tables
            tables = ['chunks', 'documents', 'users']
            for table in tables:
                conn.execute(text(f"TRUNCATE TABLE {table}"))

            # Re-enable foreign key checks
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
            conn.commit()

    def cleanup_neo4j(self):
        """Clean up Neo4j."""
        with self.neo4j.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def cleanup_redis(self):
        """Clean up Redis."""
        self.redis.flushdb()

    def cleanup_all(self):
        """Clean up all data."""
        self.cleanup_database()
        self.cleanup_neo4j()
        self.cleanup_redis()

@pytest.fixture(scope="function")
def cleanup_test_data():
    """Clean up test data after each test."""
    cleanup = TestDataCleanup(
        database_url="sqlite:///:memory:",
        neo4j_uri="bolt://localhost:7687",
        redis_url="redis://localhost:6379/1"
    )

    yield cleanup

    # Cleanup after test
    cleanup.cleanup_all()
```

### Data Validation

#### Test Data Validation

```python
# tests/validation.py
import pytest
from typing import List, Dict, Any
from Medical_KG_rev.models.user import User
from Medical_KG_rev.models.document import Document
from Medical_KG_rev.models.chunk import Chunk

class TestDataValidator:
    """Validate test data integrity."""

    @staticmethod
    def validate_user(user: User) -> bool:
        """Validate user data."""
        if not user.email or '@' not in user.email:
            return False
        if not user.username or len(user.username) < 3:
            return False
        if not user.password_hash:
            return False
        return True

    @staticmethod
    def validate_document(document: Document) -> bool:
        """Validate document data."""
        if not document.title or len(document.title) < 5:
            return False
        if not document.content or len(document.content) < 10:
            return False
        if not document.source:
            return False
        return True

    @staticmethod
    def validate_chunk(chunk: Chunk) -> bool:
        """Validate chunk data."""
        if not chunk.content or len(chunk.content) < 5:
            return False
        if chunk.start_position < 0:
            return False
        if chunk.end_position <= chunk.start_position:
            return False
        if not chunk.document:
            return False
        return True

    @staticmethod
    def validate_dataset(users: List[User], documents: List[Document], chunks: List[Chunk]) -> Dict[str, bool]:
        """Validate entire dataset."""
        return {
            'users_valid': all(TestDataValidator.validate_user(user) for user in users),
            'documents_valid': all(TestDataValidator.validate_document(doc) for doc in documents),
            'chunks_valid': all(TestDataValidator.validate_chunk(chunk) for chunk in chunks),
            'relationships_valid': all(chunk.document in documents for chunk in chunks),
            'overall_valid': True
        }

@pytest.fixture
def data_validator():
    """Data validator fixture."""
    return TestDataValidator()
```

## Best Practices

### Test Data Design

1. **Realistic Data**: Use realistic test data that matches production
2. **Varied Scenarios**: Cover edge cases and boundary conditions
3. **Performance Considerations**: Generate data efficiently
4. **Maintainability**: Keep test data easy to update and maintain
5. **Isolation**: Ensure test data doesn't interfere between tests

### Data Generation

1. **Factory Pattern**: Use factory pattern for consistent data generation
2. **Fixtures**: Use pytest fixtures for reusable test data
3. **Mocking**: Mock external dependencies appropriately
4. **Performance**: Generate large datasets efficiently
5. **Validation**: Validate generated data integrity

### Environment Management

1. **Isolation**: Isolate test environments from production
2. **Cleanup**: Clean up test data after tests
3. **Configuration**: Use separate configuration for testing
4. **Dependencies**: Manage test dependencies properly
5. **Monitoring**: Monitor test environment health

## Troubleshooting

### Common Issues

#### 1. Test Data Conflicts

```python
# Handle test data conflicts
@pytest.fixture(autouse=True)
def isolate_test_data():
    """Isolate test data between tests."""
    # Clean up before test
    cleanup_test_data()
    yield
    # Clean up after test
    cleanup_test_data()
```

#### 2. Performance Issues

```python
# Optimize test data generation
def generate_optimized_data(count: int):
    """Generate test data efficiently."""
    # Use bulk operations
    users = UserFactory.build_batch(count)
    session.bulk_save_objects(users)
    session.commit()
    return users
```

#### 3. Memory Issues

```python
# Manage memory usage
def generate_large_dataset(count: int, batch_size: int = 1000):
    """Generate large dataset in batches."""
    for i in range(0, count, batch_size):
        batch = UserFactory.create_batch(min(batch_size, count - i))
        yield batch
        # Clean up batch
        del batch
```

### Debug Commands

```bash
# Check test data generation
python -c "from tests.factories.user import UserFactory; print(UserFactory.build())"

# Validate test data
python -c "from tests.validation import TestDataValidator; print('Validator loaded')"

# Check test environment
python scripts/setup_test_environment.py

# Clean test environment
python scripts/setup_test_environment.py teardown

# Run tests with specific data
pytest tests/ -k "test_with_sample_data" -v

# Generate performance test data
python -c "from tests.performance.data_generator import PerformanceDataGenerator; print('Generator loaded')"
```

## Related Documentation

- [Testing Strategy](testing_strategy.md)
- [Development Workflow](development_workflow.md)
- [Environment Setup](environment_setup.md)
- [Troubleshooting Guide](troubleshooting.md)
