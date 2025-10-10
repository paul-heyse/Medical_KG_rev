# Knowledge Graph API Reference

The Knowledge Graph layer provides graph database operations, schema management, and query templates for the medical knowledge graph.

## Core Knowledge Graph Components

### Neo4j Client

::: Medical_KG_rev.kg.neo4j_client
    options:
      show_root_heading: true
      members:
        - Neo4jClient
        - Neo4jConfig
        - Neo4jError
        - Neo4jTransaction

### Cypher Templates

::: Medical_KG_rev.kg.cypher_templates
    options:
      show_root_heading: true
      members:
        - CypherTemplate
        - TemplateManager
        - TemplateError
        - create_node_template
        - create_relationship_template
        - create_query_template

### Graph Schema

::: Medical_KG_rev.kg.schema
    options:
      show_root_heading: true
      members:
        - GraphSchema
        - NodeSchema
        - RelationshipSchema
        - PropertySchema
        - IndexSchema
        - ConstraintSchema

### SHACL Validation

::: Medical_KG_rev.kg.shacl
    options:
      show_root_heading: true
      members:
        - ShaclValidator
        - ShaclShape
        - ShaclError
        - validate_graph
        - validate_node
        - validate_relationship

## Usage Examples

### Basic Neo4j Operations

```python
from Medical_KG_rev.kg.neo4j_client import Neo4jClient

# Initialize Neo4j client
client = Neo4jClient(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password",
    database="medical-kg"
)

# Create a document node
document_data = {
    "id": "doc-123",
    "title": "Medical Research Paper",
    "content": "Research content...",
    "doi": "10.1371/journal.pone.0123456",
    "created_at": "2024-01-01T00:00:00Z"
}

result = await client.create_node(
    label="Document",
    properties=document_data
)

print(f"Created document: {result['id']}")
```

### Cypher Template Usage

```python
from Medical_KG_rev.kg.cypher_templates import CypherTemplate

# Create a template for finding documents by DOI
template = CypherTemplate(
    name="find_document_by_doi",
    cypher="""
    MATCH (d:Document {doi: $doi})
    RETURN d.id as id, d.title as title, d.content as content
    """,
    parameters=["doi"]
)

# Execute template
result = await client.execute_template(
    template,
    parameters={"doi": "10.1371/journal.pone.0123456"}
)

if result:
    document = result[0]
    print(f"Found document: {document['title']}")
```

### Schema Management

```python
from Medical_KG_rev.kg.schema import GraphSchema, NodeSchema, RelationshipSchema

# Define document schema
document_schema = NodeSchema(
    label="Document",
    properties={
        "id": {"type": "string", "required": True, "unique": True},
        "title": {"type": "string", "required": True},
        "content": {"type": "string", "required": True},
        "doi": {"type": "string", "required": False},
        "created_at": {"type": "datetime", "required": True}
    },
    indexes=["id", "doi"],
    constraints=["id"]
)

# Define entity schema
entity_schema = NodeSchema(
    label="Entity",
    properties={
        "id": {"type": "string", "required": True, "unique": True},
        "name": {"type": "string", "required": True},
        "type": {"type": "string", "required": True},
        "confidence": {"type": "float", "required": False}
    },
    indexes=["id", "name", "type"],
    constraints=["id"]
)

# Define relationship schema
mentions_schema = RelationshipSchema(
    type="MENTIONS",
    properties={
        "start_offset": {"type": "int", "required": True},
        "end_offset": {"type": "int", "required": True},
        "confidence": {"type": "float", "required": False}
    },
    indexes=["confidence"]
)

# Create complete schema
schema = GraphSchema(
    nodes=[document_schema, entity_schema],
    relationships=[mentions_schema]
)

# Apply schema to database
await client.apply_schema(schema)
```

### SHACL Validation

```python
from Medical_KG_rev.kg.shacl import ShaclValidator

# Initialize validator
validator = ShaclValidator(
    shapes_file="config/kg/shapes.ttl",
    base_uri="https://medical-kg.example.com/"
)

# Validate a document node
document_node = {
    "id": "doc-123",
    "title": "Medical Research Paper",
    "content": "Research content...",
    "doi": "10.1371/journal.pone.0123456",
    "created_at": "2024-01-01T00:00:00Z"
}

validation_result = await validator.validate_node(
    node=document_node,
    shape="DocumentShape"
)

if validation_result.valid:
    print("Document node is valid")
else:
    print(f"Validation errors: {validation_result.errors}")
```

### Complex Graph Queries

```python
# Find documents mentioning specific entities
query = """
MATCH (d:Document)-[m:MENTIONS]->(e:Entity)
WHERE e.name CONTAINS $entity_name
  AND m.confidence > $min_confidence
RETURN d.id as document_id,
       d.title as document_title,
       e.name as entity_name,
       m.confidence as mention_confidence
ORDER BY m.confidence DESC
LIMIT $limit
"""

result = await client.execute_query(
    query,
    parameters={
        "entity_name": "diabetes",
        "min_confidence": 0.8,
        "limit": 10
    }
)

for record in result:
    print(f"Document: {record['document_title']}")
    print(f"Entity: {record['entity_name']}")
    print(f"Confidence: {record['mention_confidence']}")
```

### Batch Operations

```python
# Batch create multiple entities
entities = [
    {"id": "entity-1", "name": "Diabetes", "type": "Disease"},
    {"id": "entity-2", "name": "Insulin", "type": "Drug"},
    {"id": "entity-3", "name": "Glucose", "type": "Biomarker"}
]

result = await client.batch_create_nodes(
    label="Entity",
    nodes=entities
)

print(f"Created {len(result)} entities")

# Batch create relationships
relationships = [
    {
        "from": "entity-1",
        "to": "entity-2",
        "type": "TREATED_BY",
        "properties": {"confidence": 0.9}
    },
    {
        "from": "entity-1",
        "to": "entity-3",
        "type": "MEASURED_BY",
        "properties": {"confidence": 0.8}
    }
]

result = await client.batch_create_relationships(relationships)
print(f"Created {len(result)} relationships")
```

## Configuration

### Neo4j Configuration

```python
# Neo4j client configuration
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "password",
    "database": "medical-kg",
    "max_connection_lifetime": 3600,
    "max_connection_pool_size": 50,
    "connection_acquisition_timeout": 60,
    "encrypted": False,
    "trust": "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"
}

# Cypher template configuration
TEMPLATE_CONFIG = {
    "template_dir": "config/kg/templates",
    "cache_size": 1000,
    "cache_ttl": 3600
}

# SHACL validation configuration
SHACL_CONFIG = {
    "shapes_file": "config/kg/shapes.ttl",
    "base_uri": "https://medical-kg.example.com/",
    "validation_mode": "strict",
    "cache_shapes": True
}
```

### Environment Variables

- `NEO4J_URI`: Neo4j database URI
- `NEO4J_USERNAME`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
- `NEO4J_DATABASE`: Neo4j database name
- `KG_TEMPLATE_DIR`: Directory containing Cypher templates
- `KG_SHAPES_FILE`: Path to SHACL shapes file
- `KG_BASE_URI`: Base URI for SHACL validation

## Error Handling

### Knowledge Graph Error Hierarchy

```python
# Base knowledge graph error
class KnowledgeGraphError(Exception):
    """Base exception for knowledge graph errors."""
    pass

# Neo4j errors
class Neo4jError(KnowledgeGraphError):
    """Neo4j-specific errors."""
    pass

# Schema errors
class SchemaError(KnowledgeGraphError):
    """Schema-related errors."""
    pass

# Validation errors
class ValidationError(KnowledgeGraphError):
    """Validation errors."""
    pass

# Template errors
class TemplateError(KnowledgeGraphError):
    """Template-related errors."""
    pass
```

### Error Handling Patterns

```python
from Medical_KG_rev.kg.neo4j_client import Neo4jError

try:
    # Execute graph operation
    result = await client.execute_query(query, parameters)

    if not result:
        # Handle empty result
        logger.warning("Query returned no results")
        return []

except Neo4jError as e:
    # Handle Neo4j errors
    if "Connection" in str(e):
        # Handle connection errors
        logger.error("Neo4j connection failed")
        await client.reconnect()
        # Retry operation
        result = await client.execute_query(query, parameters)
    elif "Timeout" in str(e):
        # Handle timeout errors
        logger.error("Neo4j query timeout")
        raise
    else:
        # Handle other Neo4j errors
        logger.error(f"Neo4j error: {e}")
        raise
```

## Performance Considerations

- **Connection Pooling**: Neo4j client uses connection pooling for efficiency
- **Query Optimization**: Cypher queries are optimized for performance
- **Batch Operations**: Support for batch operations to reduce overhead
- **Indexing**: Proper indexing strategy for fast queries
- **Caching**: Template and schema caching to improve performance

## Monitoring and Observability

- **Query Performance**: Track query execution time and resource usage
- **Connection Health**: Monitor Neo4j connection health and pool status
- **Schema Validation**: Track validation errors and performance
- **Distributed Tracing**: OpenTelemetry spans for graph operations
- **Structured Logging**: Comprehensive logging with correlation IDs
- **Health Checks**: Knowledge graph health check endpoints

## Testing

### Mock Neo4j Client

```python
from Medical_KG_rev.kg.neo4j_client import Neo4jClient

class MockNeo4jClient(Neo4jClient):
    """Mock Neo4j client for testing."""

    def __init__(self):
        self.nodes = {}
        self.relationships = []
        self.queries = []

    async def execute_query(self, query: str, parameters: dict = None) -> list:
        """Mock query execution."""
        self.queries.append({"query": query, "parameters": parameters})

        # Mock query results based on query type
        if "MATCH" in query and "Document" in query:
            return [{"id": "doc-123", "title": "Test Document"}]
        elif "MATCH" in query and "Entity" in query:
            return [{"id": "entity-123", "name": "Test Entity"}]
        else:
            return []

    async def create_node(self, label: str, properties: dict) -> dict:
        """Mock node creation."""
        node_id = properties.get("id", f"node-{len(self.nodes)}")
        self.nodes[node_id] = {"label": label, "properties": properties}
        return {"id": node_id}
```

### Integration Tests

```python
import pytest
from Medical_KG_rev.kg.neo4j_client import Neo4jClient

@pytest.mark.asyncio
async def test_neo4j_client():
    """Test Neo4j client functionality."""
    client = Neo4jClient(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )

    # Test node creation
    result = await client.create_node(
        label="Document",
        properties={"id": "test-doc", "title": "Test Document"}
    )

    assert result["id"] == "test-doc"

    # Test query execution
    query = "MATCH (d:Document {id: $id}) RETURN d.title as title"
    result = await client.execute_query(query, {"id": "test-doc"})

    assert len(result) == 1
    assert result[0]["title"] == "Test Document"

    # Cleanup
    await client.delete_node("test-doc")
```
