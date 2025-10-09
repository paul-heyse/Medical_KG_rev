# Code Review Guidelines

This document provides comprehensive guidelines for conducting effective code reviews in the Medical_KG_rev system, including standards for code quality, checklists for reviewers, common pitfalls to avoid, and examples of constructive feedback.

## Overview

Code reviews are a critical part of our development process, ensuring code quality, knowledge sharing, and system reliability. This guide establishes standards and practices for conducting effective code reviews.

## Code Review Principles

### Core Principles

1. **Quality First**: Ensure code meets quality standards
2. **Knowledge Sharing**: Share knowledge and best practices
3. **Collaborative**: Work together to improve the code
4. **Constructive**: Provide helpful, actionable feedback
5. **Timely**: Review code promptly to avoid blocking development

### Review Goals

- **Correctness**: Code works as intended
- **Maintainability**: Code is easy to understand and modify
- **Performance**: Code performs efficiently
- **Security**: Code follows security best practices
- **Consistency**: Code follows established patterns

## Code Quality Standards

### Python Code Standards

#### Style Guidelines

```python
# Good: Clear, readable code
def process_document(document: Document) -> ProcessedDocument:
    """Process a document and extract metadata.

    Args:
        document: The document to process

    Returns:
        ProcessedDocument with extracted metadata

    Raises:
        ProcessingError: If document cannot be processed
    """
    try:
        metadata = extract_metadata(document)
        chunks = chunk_document(document)
        embeddings = generate_embeddings(chunks)

        return ProcessedDocument(
            document_id=document.id,
            metadata=metadata,
            chunks=chunks,
            embeddings=embeddings
        )
    except Exception as e:
        logger.error(f"Failed to process document {document.id}: {e}")
        raise ProcessingError(f"Document processing failed: {e}") from e

# Bad: Unclear, poorly structured code
def proc_doc(doc):
    md = ext_md(doc)
    chks = chk_doc(doc)
    emb = gen_emb(chks)
    return ProcDoc(doc.id, md, chks, emb)
```

#### Type Hints

```python
# Good: Comprehensive type hints
from typing import List, Dict, Optional, Union, Callable
from datetime import datetime

def search_documents(
    query: str,
    filters: Optional[Dict[str, Union[str, int, float]]] = None,
    limit: int = 100,
    offset: int = 0,
    callback: Optional[Callable[[str], None]] = None
) -> List[Document]:
    """Search documents with optional filters and pagination."""
    pass

# Bad: Missing type hints
def search_documents(query, filters=None, limit=100, offset=0, callback=None):
    """Search documents with optional filters and pagination."""
    pass
```

#### Error Handling

```python
# Good: Proper error handling
class DocumentProcessor:
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def process(self, document: Document) -> ProcessedDocument:
        """Process a document with proper error handling."""
        try:
            # Validate input
            if not document or not document.content:
                raise ValueError("Document and content are required")

            # Process document
            result = self._do_processing(document)

            # Log success
            self.logger.info(f"Successfully processed document {document.id}")
            return result

        except ValidationError as e:
            self.logger.error(f"Validation error for document {document.id}: {e}")
            raise
        except ProcessingError as e:
            self.logger.error(f"Processing error for document {document.id}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error processing document {document.id}: {e}")
            raise ProcessingError(f"Unexpected processing error: {e}") from e

# Bad: Poor error handling
def process(document):
    result = do_processing(document)
    return result
```

### API Design Standards

#### REST API Guidelines

```python
# Good: RESTful API design
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

class DocumentCreate(BaseModel):
    title: str
    content: str
    source: str
    metadata: Optional[Dict[str, Any]] = None

class DocumentResponse(BaseModel):
    id: str
    title: str
    content: str
    source: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

@router.post("/", response_model=DocumentResponse, status_code=201)
async def create_document(
    document: DocumentCreate,
    current_user: User = Depends(get_current_user)
) -> DocumentResponse:
    """Create a new document.

    Args:
        document: Document data
        current_user: Authenticated user

    Returns:
        Created document

    Raises:
        HTTPException: If document creation fails
    """
    try:
        created_doc = await document_service.create_document(
            document=document,
            user_id=current_user.id
        )
        return DocumentResponse.from_orm(created_doc)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DuplicateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Bad: Poor API design
@router.post("/create_document")
async def create_doc(data):
    doc = create_doc(data)
    return doc
```

#### GraphQL Guidelines

```python
# Good: GraphQL schema design
from strawberry import type, field, mutation
from typing import List, Optional

@type
class Document:
    id: str
    title: str
    content: str
    source: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    @field
    async def chunks(self) -> List[Chunk]:
        """Get document chunks."""
        return await chunk_service.get_chunks_by_document_id(self.id)

    @field
    async def embeddings(self) -> List[Embedding]:
        """Get document embeddings."""
        return await embedding_service.get_embeddings_by_document_id(self.id)

@type
class Query:
    @field
    async def document(self, id: str) -> Optional[Document]:
        """Get document by ID."""
        return await document_service.get_document_by_id(id)

    @field
    async def documents(
        self,
        query: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Document]:
        """Search documents with filters and pagination."""
        return await document_service.search_documents(
            query=query,
            source=source,
            limit=limit,
            offset=offset
        )

@type
class Mutation:
    @mutation
    async def create_document(
        self,
        title: str,
        content: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Create a new document."""
        return await document_service.create_document(
            title=title,
            content=content,
            source=source,
            metadata=metadata or {}
        )
```

### Database Standards

#### Query Optimization

```python
# Good: Optimized database queries
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy import select, func

class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_document_with_chunks(self, document_id: str) -> Optional[Document]:
        """Get document with chunks using eager loading."""
        query = (
            select(Document)
            .options(selectinload(Document.chunks))
            .where(Document.id == document_id)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def search_documents_optimized(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Document]:
        """Search documents with optimized query."""
        # Use full-text search if available
        if self._supports_full_text_search():
            search_query = (
                select(Document)
                .where(Document.content.match(query))
                .order_by(func.ts_rank(Document.content, query).desc())
                .limit(limit)
                .offset(offset)
            )
        else:
            # Fallback to ILIKE search
            search_query = (
                select(Document)
                .where(Document.content.ilike(f"%{query}%"))
                .limit(limit)
                .offset(offset)
            )

        result = await self.session.execute(search_query)
        return result.scalars().all()

# Bad: Inefficient queries
def get_document_with_chunks(document_id):
    doc = session.query(Document).filter(Document.id == document_id).first()
    chunks = session.query(Chunk).filter(Chunk.document_id == document_id).all()
    doc.chunks = chunks
    return doc
```

#### Transaction Management

```python
# Good: Proper transaction management
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager

class DocumentService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_document_with_chunks(
        self,
        document_data: DocumentCreate,
        chunks_data: List[ChunkCreate]
    ) -> Document:
        """Create document with chunks in a single transaction."""
        async with self.session.begin():
            try:
                # Create document
                document = Document(**document_data.dict())
                self.session.add(document)
                await self.session.flush()  # Get document ID

                # Create chunks
                for chunk_data in chunks_data:
                    chunk = Chunk(
                        document_id=document.id,
                        **chunk_data.dict()
                    )
                    self.session.add(chunk)

                # Commit transaction
                await self.session.commit()
                return document

            except Exception as e:
                await self.session.rollback()
                logger.error(f"Failed to create document with chunks: {e}")
                raise

# Bad: Poor transaction management
def create_document_with_chunks(document_data, chunks_data):
    doc = Document(**document_data)
    session.add(doc)
    session.commit()

    for chunk_data in chunks_data:
        chunk = Chunk(document_id=doc.id, **chunk_data)
        session.add(chunk)
    session.commit()
```

## Review Checklists

### For Authors

#### Pre-Submission Checklist

- [ ] **Code Quality**
  - [ ] Code follows PEP 8 style guidelines
  - [ ] Functions and classes have proper docstrings
  - [ ] Type hints are used consistently
  - [ ] No unused imports or variables
  - [ ] Error handling is implemented

- [ ] **Testing**
  - [ ] Unit tests cover new functionality
  - [ ] Integration tests verify component interactions
  - [ ] Contract tests ensure API compatibility
  - [ ] Performance tests validate non-functional requirements
  - [ ] All tests pass locally

- [ ] **Documentation**
  - [ ] Code is self-documenting
  - [ ] Docstrings follow Google style
  - [ ] README updated if needed
  - [ ] API documentation updated
  - [ ] Configuration documentation updated

- [ ] **Security**
  - [ ] No hardcoded secrets
  - [ ] Input validation implemented
  - [ ] SQL injection prevention
  - [ ] XSS prevention
  - [ ] CSRF protection

#### Self-Review Checklist

- [ ] **Functionality**
  - [ ] Code solves the intended problem
  - [ ] Edge cases are handled
  - [ ] Error scenarios are covered
  - [ ] Performance implications considered

- [ ] **Architecture**
  - [ ] Code follows established patterns
  - [ ] Dependencies are appropriate
  - [ ] Separation of concerns maintained
  - [ ] No circular dependencies

- [ ] **Maintainability**
  - [ ] Code is readable and understandable
  - [ ] Complex logic is commented
  - [ ] Configuration is externalized
  - [ ] Logging is appropriate

### For Reviewers

#### Code Review Checklist

- [ ] **Functionality Review**
  - [ ] Code implements the required functionality
  - [ ] Edge cases are properly handled
  - [ ] Error scenarios are covered
  - [ ] Performance implications are considered
  - [ ] Security implications are addressed

- [ ] **Code Quality Review**
  - [ ] Code follows style guidelines
  - [ ] Functions are focused and single-purpose
  - [ ] Variable names are descriptive
  - [ ] Code is well-commented
  - [ ] No code duplication

- [ ] **Architecture Review**
  - [ ] Code follows established patterns
  - [ ] Dependencies are appropriate
  - [ ] Separation of concerns is maintained
  - [ ] No circular dependencies
  - [ ] Design patterns are used correctly

- [ ] **Testing Review**
  - [ ] Tests cover new functionality
  - [ ] Tests are well-written and maintainable
  - [ ] Test data is appropriate
  - [ ] Mocking is used correctly
  - [ ] Test coverage is adequate

- [ ] **Documentation Review**
  - [ ] Code is self-documenting
  - [ ] Docstrings are comprehensive
  - [ ] README is updated if needed
  - [ ] API documentation is updated
  - [ ] Configuration documentation is updated

## Review Process

### Review Assignment

#### Automatic Assignment

```yaml
# .github/CODEOWNERS
# Global owners
* @team-lead @senior-dev

# Gateway module
src/Medical_KG_rev/gateway/ @gateway-team

# Services module
src/Medical_KG_rev/services/ @services-team

# Storage module
src/Medical_KG_rev/storage/ @storage-team

# Tests
tests/ @qa-team

# Documentation
docs/ @docs-team
```

#### Manual Assignment

- **Domain Experts**: Assign for complex changes
- **Security Team**: Include for security-related changes
- **Performance Team**: Include for performance-critical changes
- **Architecture Team**: Include for architectural changes

### Review Timeline

#### Response Times

- **Small PRs** (< 200 lines): Review within 24 hours
- **Medium PRs** (200-500 lines): Review within 48 hours
- **Large PRs** (500-1000 lines): Review within 72 hours
- **Critical PRs**: Review within 4 hours

#### Escalation Process

1. **Initial Review**: 24 hours
2. **Reminder**: 48 hours
3. **Escalation**: 72 hours
4. **Final Notice**: 96 hours

### Review Actions

#### Review Decisions

- **Approve**: Code is ready to merge
- **Request Changes**: Code needs modifications
- **Comment**: General feedback or questions
- **Block**: Code cannot be merged

#### Review Comments

```markdown
# Good: Constructive feedback
This looks good overall! I have a few suggestions:

1. **Performance**: Consider using `selectinload` instead of `joinedload` for the chunks relationship to avoid N+1 queries
2. **Error Handling**: Add specific exception handling for database connection errors
3. **Documentation**: The docstring could be more specific about the expected input format

Here's an example of how to implement the performance improvement:
```python
query = select(Document).options(selectinload(Document.chunks))
```

# Bad: Unhelpful feedback

This is wrong. Fix it.

```

## Common Pitfalls

### Code Quality Issues

#### 1. Missing Error Handling

```python
# Bad: No error handling
def process_document(document_id: str) -> Document:
    document = get_document(document_id)
    processed = process(document)
    return processed

# Good: Proper error handling
def process_document(document_id: str) -> Document:
    try:
        document = get_document(document_id)
        if not document:
            raise DocumentNotFoundError(f"Document {document_id} not found")

        processed = process(document)
        return processed
    except DocumentNotFoundError:
        raise
    except ProcessingError as e:
        logger.error(f"Failed to process document {document_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing document {document_id}: {e}")
        raise ProcessingError(f"Unexpected processing error: {e}") from e
```

#### 2. Inefficient Database Queries

```python
# Bad: N+1 query problem
def get_documents_with_chunks(document_ids: List[str]) -> List[Document]:
    documents = []
    for doc_id in document_ids:
        doc = get_document(doc_id)
        doc.chunks = get_chunks(doc_id)  # N+1 queries
        documents.append(doc)
    return documents

# Good: Optimized query
def get_documents_with_chunks(document_ids: List[str]) -> List[Document]:
    query = (
        select(Document)
        .options(selectinload(Document.chunks))
        .where(Document.id.in_(document_ids))
    )
    result = session.execute(query)
    return result.scalars().all()
```

#### 3. Missing Type Hints

```python
# Bad: No type hints
def search_documents(query, filters=None, limit=100):
    results = []
    for doc in documents:
        if matches(doc, query, filters):
            results.append(doc)
    return results[:limit]

# Good: Comprehensive type hints
from typing import List, Dict, Optional, Union

def search_documents(
    query: str,
    filters: Optional[Dict[str, Union[str, int, float]]] = None,
    limit: int = 100
) -> List[Document]:
    results: List[Document] = []
    for doc in documents:
        if matches(doc, query, filters):
            results.append(doc)
    return results[:limit]
```

### Security Issues

#### 1. SQL Injection

```python
# Bad: SQL injection vulnerability
def search_documents(query: str) -> List[Document]:
    sql = f"SELECT * FROM documents WHERE content LIKE '%{query}%'"
    return execute_query(sql)

# Good: Parameterized queries
def search_documents(query: str) -> List[Document]:
    sql = "SELECT * FROM documents WHERE content LIKE :query"
    return execute_query(sql, query=f"%{query}%")
```

#### 2. Hardcoded Secrets

```python
# Bad: Hardcoded secret
API_KEY = "sk-1234567890abcdef"

# Good: Environment variable
import os
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")
```

#### 3. Missing Input Validation

```python
# Bad: No input validation
def create_user(username: str, email: str) -> User:
    user = User(username=username, email=email)
    return save_user(user)

# Good: Input validation
from pydantic import BaseModel, validator, EmailStr

class UserCreate(BaseModel):
    username: str
    email: EmailStr

    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v

def create_user(user_data: UserCreate) -> User:
    user = User(username=user_data.username, email=user_data.email)
    return save_user(user)
```

## Feedback Examples

### Constructive Feedback

#### Positive Feedback

```markdown
Great work! This implementation looks solid. I particularly like:

1. **Clear Structure**: The code is well-organized and easy to follow
2. **Good Error Handling**: You've covered all the edge cases
3. **Comprehensive Tests**: The test coverage is excellent
4. **Documentation**: The docstrings are clear and helpful

One small suggestion: Consider adding a type hint for the return value of `process_chunks()` to make the API even clearer.
```

#### Improvement Suggestions

```markdown
This is a good start! Here are some suggestions to make it even better:

1. **Performance**: The current implementation processes documents sequentially. Consider using `asyncio.gather()` to process multiple documents concurrently:
   ```python
   results = await asyncio.gather(*[process_document(doc) for doc in documents])
   ```

2. **Error Handling**: Add specific exception handling for different types of processing errors:

   ```python
   except ValidationError as e:
       logger.warning(f"Validation error: {e}")
       raise
   except ProcessingError as e:
       logger.error(f"Processing error: {e}")
       raise
   ```

3. **Configuration**: Move the batch size to configuration instead of hardcoding it.

```

#### Critical Issues

```markdown
I found a critical security issue that needs to be addressed before this can be merged:

**Security Issue**: The API endpoint is vulnerable to SQL injection attacks. The query string is being concatenated directly into the SQL query without proper sanitization.

**Current Code**:
```python
query = f"SELECT * FROM documents WHERE content LIKE '%{search_term}%'"
```

**Recommended Fix**:

```python
query = "SELECT * FROM documents WHERE content LIKE :search_term"
params = {"search_term": f"%{search_term}%"}
```

This is a blocking issue that must be fixed before the PR can be approved.

```

### Unhelpful Feedback

#### Vague Feedback

```markdown
# Bad: Too vague
This doesn't look right.

# Good: Specific feedback
The error handling in the `process_document` function doesn't handle the case where the document content is empty. Consider adding a check for empty content before processing.
```

#### Personal Attacks

```markdown
# Bad: Personal attack
You clearly don't understand how this should work.

# Good: Professional feedback
I think there might be a misunderstanding about the expected behavior. Let me clarify the requirements and suggest an alternative approach.
```

#### Nitpicking

```markdown
# Bad: Nitpicking
You used `i` instead of `index` for the loop variable.

# Good: Focus on important issues
The main issue I see is the potential for a memory leak in the document processing loop. The current implementation keeps all processed documents in memory, which could cause issues with large datasets.
```

## Best Practices

### For Authors

1. **Prepare Your PR**
   - Write clear commit messages
   - Provide comprehensive PR description
   - Include relevant tests
   - Update documentation

2. **Respond to Feedback**
   - Address all comments
   - Ask questions if unclear
   - Provide context when needed
   - Be open to suggestions

3. **Follow Up**
   - Thank reviewers
   - Learn from feedback
   - Apply lessons to future PRs
   - Share knowledge with others

### For Reviewers

1. **Be Constructive**
   - Focus on the code, not the person
   - Provide specific, actionable feedback
   - Explain the reasoning behind suggestions
   - Offer alternatives when appropriate

2. **Be Timely**
   - Review within agreed timeframes
   - Communicate if delays are expected
   - Prioritize blocking issues
   - Follow up on requested changes

3. **Be Thorough**
   - Check all aspects of the code
   - Consider edge cases and error scenarios
   - Verify tests and documentation
   - Look for security and performance issues

### For Teams

1. **Establish Standards**
   - Define coding standards
   - Create review checklists
   - Set response time expectations
   - Document review processes

2. **Provide Training**
   - Train new team members
   - Share best practices
   - Conduct review workshops
   - Provide feedback on reviews

3. **Monitor and Improve**
   - Track review metrics
   - Gather feedback on the process
   - Continuously improve standards
   - Celebrate good reviews

## Tools and Automation

### Automated Checks

#### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.270
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

#### CI/CD Integration

```yaml
# .github/workflows/code-review.yml
name: Code Review Checks

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt

      - name: Run code quality checks
        run: |
          ruff check .
          mypy src/
          pytest --cov=src/ --cov-fail-under=80

      - name: Comment PR
        uses: actions/github-script@v6
        if: failure()
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '❌ Code quality checks failed. Please review the errors above.'
            })
```

### Review Tools

#### GitHub Integration

```yaml
# .github/pull_request_template.md
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Related Issues
Closes #123
```

#### Review Automation

```python
# scripts/auto-review.py
import os
import subprocess
import sys
from pathlib import Path

def run_code_quality_checks():
    """Run automated code quality checks."""
    checks = [
        ("ruff check .", "Linting"),
        ("mypy src/", "Type checking"),
        ("pytest --cov=src/", "Testing"),
        ("bandit -r src/", "Security"),
    ]

    results = {}
    for command, name in checks:
        try:
            result = subprocess.run(command.split(), capture_output=True, text=True)
            results[name] = {
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "output": result.stdout + result.stderr
            }
        except Exception as e:
            results[name] = {
                "status": "ERROR",
                "output": str(e)
            }

    return results

def generate_review_report(results):
    """Generate review report."""
    report = ["# Automated Code Review Report\n"]

    for name, result in results.items():
        status_emoji = "✅" if result["status"] == "PASS" else "❌"
        report.append(f"## {status_emoji} {name}: {result['status']}")

        if result["output"]:
            report.append("```")
            report.append(result["output"])
            report.append("```")
        report.append("")

    return "\n".join(report)

if __name__ == "__main__":
    results = run_code_quality_checks()
    report = generate_review_report(results)
    print(report)
```

## Troubleshooting

### Common Issues

1. **Review Delays**
   - Check if reviewers are available
   - Escalate to team lead if needed
   - Consider breaking large PRs into smaller ones

2. **Conflicting Feedback**
   - Discuss differences with reviewers
   - Seek consensus on approach
   - Involve team lead if needed

3. **Blocking Issues**
   - Address critical issues immediately
   - Communicate timeline for fixes
   - Consider alternative approaches

### Debug Commands

```bash
# Check code quality
ruff check src/
mypy src/
pytest --cov=src/

# Run security checks
bandit -r src/
safety check

# Check for common issues
python scripts/check_code_quality.py

# Generate review report
python scripts/auto-review.py
```

## Related Documentation

- [Development Workflow](development_workflow.md)
- [CI/CD Pipeline](ci_cd_pipeline.md)
- [Testing Strategy](testing_strategy.md)
- [Security Guidelines](security.md)
- [Troubleshooting Guide](troubleshooting.md)
