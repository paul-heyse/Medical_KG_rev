# Database Migration Procedures

This document provides comprehensive guidance on database migration procedures for the Medical_KG_rev system, including schema changes, data migrations, rollback procedures, and best practices.

## Overview

Database migrations are essential for managing schema changes and data transformations in the Medical_KG_rev system. This guide covers migration procedures for PostgreSQL, Neo4j, and Redis, ensuring data integrity and system reliability.

## Migration Architecture

### Migration Components

1. **Schema Migrations**: Database structure changes
2. **Data Migrations**: Data transformation and cleanup
3. **Index Migrations**: Performance optimization
4. **Constraint Migrations**: Data integrity enforcement
5. **Rollback Migrations**: Reversal procedures

### Migration Tools

- **Alembic**: PostgreSQL schema migrations
- **Neo4j Migrations**: Graph database migrations
- **Redis Migrations**: Cache and session migrations
- **Custom Scripts**: Complex data transformations

## PostgreSQL Migrations

### Alembic Configuration

#### Basic Setup

```python
# alembic.ini
[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = postgresql://user:pass@localhost:5432/medical_kg

[post_write_hooks]
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = -l 88 REVISION_SCRIPT_FILENAME

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

#### Environment Configuration

```python
# alembic/env.py
from alembic import context
from sqlalchemy import engine_from_config, pool
from Medical_KG_rev.storage.database import Base
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import all models to ensure they're registered
from Medical_KG_rev.models import *

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the target metadata
target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Migration Creation

#### Schema Migration Example

```python
# alembic/versions/001_add_user_table.py
"""Add user table

Revision ID: 001
Revises:
Create Date: 2024-01-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """Add user table."""
    # Create user table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('username', sa.String(100), nullable=False, unique=True),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('first_name', sa.String(100), nullable=True),
        sa.Column('last_name', sa.String(100), nullable=True),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('is_verified', sa.Boolean, nullable=False, default=False),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now()),
    )

    # Create indexes
    op.create_index('idx_users_email', 'users', ['email'])
    op.create_index('idx_users_username', 'users', ['username'])
    op.create_index('idx_users_created_at', 'users', ['created_at'])

def downgrade():
    """Remove user table."""
    op.drop_index('idx_users_created_at', 'users')
    op.drop_index('idx_users_username', 'users')
    op.drop_index('idx_users_email', 'users')
    op.drop_table('users')
```

#### Data Migration Example

```python
# alembic/versions/002_migrate_user_data.py
"""Migrate user data

Revision ID: 002
Revises: 001
Create Date: 2024-01-02 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade():
    """Migrate user data."""
    connection = op.get_bind()

    # Add new column
    op.add_column('users', sa.Column('full_name', sa.String(200), nullable=True))

    # Migrate data
    connection.execute(text("""
        UPDATE users
        SET full_name = CONCAT(first_name, ' ', last_name)
        WHERE first_name IS NOT NULL AND last_name IS NOT NULL
    """))

    # Set default for users without names
    connection.execute(text("""
        UPDATE users
        SET full_name = username
        WHERE full_name IS NULL
    """))

def downgrade():
    """Rollback user data migration."""
    op.drop_column('users', 'full_name')
```

### Migration Management

#### Migration Scripts

```python
# scripts/migration_manager.py
import os
import sys
import subprocess
from pathlib import Path
from alembic.config import Config
from alembic import command
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine

class MigrationManager:
    """Database migration manager."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.alembic_cfg = Config("alembic.ini")
        self.alembic_cfg.set_main_option("sqlalchemy.url", database_url)
        self.engine = create_engine(database_url)

    def get_current_revision(self) -> str:
        """Get current database revision."""
        with self.engine.connect() as connection:
            context = MigrationContext.configure(connection)
            return context.get_current_revision()

    def get_head_revision(self) -> str:
        """Get head revision."""
        return self.alembic_cfg.get_main_option("head")

    def is_up_to_date(self) -> bool:
        """Check if database is up to date."""
        current = self.get_current_revision()
        head = self.get_head_revision()
        return current == head

    def upgrade(self, revision: str = "head") -> bool:
        """Upgrade database to specified revision."""
        try:
            command.upgrade(self.alembic_cfg, revision)
            return True
        except Exception as e:
            print(f"Migration failed: {e}")
            return False

    def downgrade(self, revision: str) -> bool:
        """Downgrade database to specified revision."""
        try:
            command.downgrade(self.alembic_cfg, revision)
            return True
        except Exception as e:
            print(f"Downgrade failed: {e}")
            return False

    def create_migration(self, message: str) -> str:
        """Create new migration."""
        try:
            command.revision(self.alembic_cfg, message=message, autogenerate=True)
            return "Migration created successfully"
        except Exception as e:
            return f"Migration creation failed: {e}"

    def get_migration_history(self) -> list:
        """Get migration history."""
        try:
            command.history(self.alembic_cfg)
            return "Migration history displayed"
        except Exception as e:
            return f"Failed to get history: {e}"

def main():
    """Main migration management function."""
    database_url = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/medical_kg")

    manager = MigrationManager(database_url)

    if len(sys.argv) < 2:
        print("Usage: python migration_manager.py <command> [args]")
        print("Commands: status, upgrade, downgrade, create, history")
        return

    command = sys.argv[1]

    if command == "status":
        current = manager.get_current_revision()
        head = manager.get_head_revision()
        up_to_date = manager.is_up_to_date()

        print(f"Current revision: {current}")
        print(f"Head revision: {head}")
        print(f"Up to date: {up_to_date}")

    elif command == "upgrade":
        revision = sys.argv[2] if len(sys.argv) > 2 else "head"
        success = manager.upgrade(revision)
        print(f"Upgrade {'successful' if success else 'failed'}")

    elif command == "downgrade":
        if len(sys.argv) < 3:
            print("Usage: python migration_manager.py downgrade <revision>")
            return

        revision = sys.argv[2]
        success = manager.downgrade(revision)
        print(f"Downgrade {'successful' if success else 'failed'}")

    elif command == "create":
        if len(sys.argv) < 3:
            print("Usage: python migration_manager.py create <message>")
            return

        message = sys.argv[2]
        result = manager.create_migration(message)
        print(result)

    elif command == "history":
        result = manager.get_migration_history()
        print(result)

    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
```

## Neo4j Migrations

### Neo4j Migration Framework

#### Migration Structure

```python
# migrations/neo4j/001_create_user_nodes.py
"""Create user nodes in Neo4j.

Migration ID: 001
Description: Create user nodes with basic properties
"""

class CreateUserNodes:
    """Create user nodes migration."""

    def __init__(self, driver):
        self.driver = driver

    def up(self):
        """Apply migration."""
        with self.driver.session() as session:
            # Create user nodes
            session.run("""
                CREATE CONSTRAINT user_id_unique IF NOT EXISTS
                FOR (u:User) REQUIRE u.id IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT user_email_unique IF NOT EXISTS
                FOR (u:User) REQUIRE u.email IS UNIQUE
            """)

            session.run("""
                CREATE INDEX user_username_index IF NOT EXISTS
                FOR (u:User) ON (u.username)
            """)

    def down(self):
        """Rollback migration."""
        with self.driver.session() as session:
            session.run("DROP CONSTRAINT user_id_unique IF EXISTS")
            session.run("DROP CONSTRAINT user_email_unique IF EXISTS")
            session.run("DROP INDEX user_username_index IF EXISTS")

# migrations/neo4j/002_create_document_nodes.py
"""Create document nodes in Neo4j.

Migration ID: 002
Description: Create document nodes with relationships
"""

class CreateDocumentNodes:
    """Create document nodes migration."""

    def __init__(self, driver):
        self.driver = driver

    def up(self):
        """Apply migration."""
        with self.driver.session() as session:
            # Create document nodes
            session.run("""
                CREATE CONSTRAINT document_id_unique IF NOT EXISTS
                FOR (d:Document) REQUIRE d.id IS UNIQUE
            """)

            session.run("""
                CREATE INDEX document_title_index IF NOT EXISTS
                FOR (d:Document) ON (d.title)
            """)

            session.run("""
                CREATE INDEX document_source_index IF NOT EXISTS
                FOR (d:Document) ON (d.source)
            """)

            # Create relationships
            session.run("""
                CREATE CONSTRAINT created_by_relationship IF NOT EXISTS
                FOR ()-[r:CREATED_BY]-() REQUIRE r.timestamp IS NOT NULL
            """)

    def down(self):
        """Rollback migration."""
        with self.driver.session() as session:
            session.run("DROP CONSTRAINT document_id_unique IF EXISTS")
            session.run("DROP INDEX document_title_index IF EXISTS")
            session.run("DROP INDEX document_source_index IF EXISTS")
            session.run("DROP CONSTRAINT created_by_relationship IF EXISTS")
```

#### Neo4j Migration Manager

```python
# scripts/neo4j_migration_manager.py
import os
import sys
from pathlib import Path
from neo4j import GraphDatabase
from typing import List, Dict, Any

class Neo4jMigrationManager:
    """Neo4j migration manager."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.migrations_dir = Path("migrations/neo4j")

    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migrations."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Migration)
                RETURN m.id as id
                ORDER BY m.id
            """)
            return [record["id"] for record in result]

    def get_available_migrations(self) -> List[str]:
        """Get list of available migrations."""
        if not self.migrations_dir.exists():
            return []

        migration_files = list(self.migrations_dir.glob("*.py"))
        return [f.stem for f in sorted(migration_files)]

    def apply_migration(self, migration_id: str) -> bool:
        """Apply a specific migration."""
        try:
            # Import migration module
            migration_module = __import__(f"migrations.neo4j.{migration_id}", fromlist=[migration_id])
            migration_class = getattr(migration_module, migration_id.title().replace("_", ""))

            # Create migration instance
            migration = migration_class(self.driver)

            # Apply migration
            migration.up()

            # Record migration
            with self.driver.session() as session:
                session.run("""
                    CREATE (m:Migration {id: $id, applied_at: datetime()})
                """, id=migration_id)

            return True

        except Exception as e:
            print(f"Migration {migration_id} failed: {e}")
            return False

    def rollback_migration(self, migration_id: str) -> bool:
        """Rollback a specific migration."""
        try:
            # Import migration module
            migration_module = __import__(f"migrations.neo4j.{migration_id}", fromlist=[migration_id])
            migration_class = getattr(migration_module, migration_id.title().replace("_", ""))

            # Create migration instance
            migration = migration_class(self.driver)

            # Rollback migration
            migration.down()

            # Remove migration record
            with self.driver.session() as session:
                session.run("""
                    MATCH (m:Migration {id: $id})
                    DELETE m
                """, id=migration_id)

            return True

        except Exception as e:
            print(f"Rollback {migration_id} failed: {e}")
            return False

    def migrate_to_latest(self) -> bool:
        """Migrate to latest version."""
        applied = self.get_applied_migrations()
        available = self.get_available_migrations()

        pending = [m for m in available if m not in applied]

        for migration_id in pending:
            print(f"Applying migration: {migration_id}")
            if not self.apply_migration(migration_id):
                return False

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get migration status."""
        applied = self.get_applied_migrations()
        available = self.get_available_migrations()
        pending = [m for m in available if m not in applied]

        return {
            "applied": applied,
            "available": available,
            "pending": pending,
            "up_to_date": len(pending) == 0
        }

    def close(self):
        """Close database connection."""
        self.driver.close()

def main():
    """Main Neo4j migration management function."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    manager = Neo4jMigrationManager(uri, user, password)

    if len(sys.argv) < 2:
        print("Usage: python neo4j_migration_manager.py <command> [args]")
        print("Commands: status, migrate, rollback")
        return

    command = sys.argv[1]

    if command == "status":
        status = manager.get_status()
        print(f"Applied migrations: {status['applied']}")
        print(f"Available migrations: {status['available']}")
        print(f"Pending migrations: {status['pending']}")
        print(f"Up to date: {status['up_to_date']}")

    elif command == "migrate":
        if len(sys.argv) > 2:
            migration_id = sys.argv[2]
            success = manager.apply_migration(migration_id)
            print(f"Migration {'successful' if success else 'failed'}")
        else:
            success = manager.migrate_to_latest()
            print(f"Migration to latest {'successful' if success else 'failed'}")

    elif command == "rollback":
        if len(sys.argv) < 3:
            print("Usage: python neo4j_migration_manager.py rollback <migration_id>")
            return

        migration_id = sys.argv[2]
        success = manager.rollback_migration(migration_id)
        print(f"Rollback {'successful' if success else 'failed'}")

    else:
        print(f"Unknown command: {command}")

    manager.close()

if __name__ == "__main__":
    main()
```

## Redis Migrations

### Redis Migration Framework

#### Migration Structure

```python
# migrations/redis/001_clear_old_sessions.py
"""Clear old sessions from Redis.

Migration ID: 001
Description: Remove expired sessions and clean up Redis
"""

import redis
from datetime import datetime, timedelta

class ClearOldSessions:
    """Clear old sessions migration."""

    def __init__(self, redis_client):
        self.redis_client = redis_client

    def up(self):
        """Apply migration."""
        # Clear expired sessions
        expired_keys = self.redis_client.keys("session:*")
        for key in expired_keys:
            ttl = self.redis_client.ttl(key)
            if ttl == -1:  # No expiration set
                self.redis_client.delete(key)
            elif ttl == -2:  # Key doesn't exist
                continue

        # Clear old cache entries
        cache_keys = self.redis_client.keys("cache:*")
        for key in cache_keys:
            ttl = self.redis_client.ttl(key)
            if ttl == -1:  # No expiration set
                self.redis_client.delete(key)

    def down(self):
        """Rollback migration."""
        # Cannot rollback deletion
        pass

# migrations/redis/002_update_cache_format.py
"""Update cache format in Redis.

Migration ID: 002
Description: Update cache key format and structure
"""

class UpdateCacheFormat:
    """Update cache format migration."""

    def __init__(self, redis_client):
        self.redis_client = redis_client

    def up(self):
        """Apply migration."""
        # Get all old format cache keys
        old_keys = self.redis_client.keys("old_cache:*")

        for old_key in old_keys:
            # Get value and TTL
            value = self.redis_client.get(old_key)
            ttl = self.redis_client.ttl(old_key)

            if value:
                # Create new format key
                new_key = old_key.replace("old_cache:", "cache:")

                # Set new key with value
                self.redis_client.set(new_key, value)

                # Set TTL if it exists
                if ttl > 0:
                    self.redis_client.expire(new_key, ttl)

                # Delete old key
                self.redis_client.delete(old_key)

    def down(self):
        """Rollback migration."""
        # Get all new format cache keys
        new_keys = self.redis_client.keys("cache:*")

        for new_key in new_keys:
            # Get value and TTL
            value = self.redis_client.get(new_key)
            ttl = self.redis_client.ttl(new_key)

            if value:
                # Create old format key
                old_key = new_key.replace("cache:", "old_cache:")

                # Set old key with value
                self.redis_client.set(old_key, value)

                # Set TTL if it exists
                if ttl > 0:
                    self.redis_client.expire(old_key, ttl)

                # Delete new key
                self.redis_client.delete(new_key)
```

#### Redis Migration Manager

```python
# scripts/redis_migration_manager.py
import os
import sys
import json
from pathlib import Path
import redis
from typing import List, Dict, Any

class RedisMigrationManager:
    """Redis migration manager."""

    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.migrations_dir = Path("migrations/redis")
        self.migration_key = "redis_migrations"

    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migrations."""
        applied = self.redis_client.get(self.migration_key)
        if applied:
            return json.loads(applied)
        return []

    def get_available_migrations(self) -> List[str]:
        """Get list of available migrations."""
        if not self.migrations_dir.exists():
            return []

        migration_files = list(self.migrations_dir.glob("*.py"))
        return [f.stem for f in sorted(migration_files)]

    def apply_migration(self, migration_id: str) -> bool:
        """Apply a specific migration."""
        try:
            # Import migration module
            migration_module = __import__(f"migrations.redis.{migration_id}", fromlist=[migration_id])
            migration_class = getattr(migration_module, migration_id.title().replace("_", ""))

            # Create migration instance
            migration = migration_class(self.redis_client)

            # Apply migration
            migration.up()

            # Record migration
            applied = self.get_applied_migrations()
            applied.append(migration_id)
            self.redis_client.set(self.migration_key, json.dumps(applied))

            return True

        except Exception as e:
            print(f"Migration {migration_id} failed: {e}")
            return False

    def rollback_migration(self, migration_id: str) -> bool:
        """Rollback a specific migration."""
        try:
            # Import migration module
            migration_module = __import__(f"migrations.redis.{migration_id}", fromlist=[migration_id])
            migration_class = getattr(migration_module, migration_id.title().replace("_", ""))

            # Create migration instance
            migration = migration_class(self.redis_client)

            # Rollback migration
            migration.down()

            # Remove migration record
            applied = self.get_applied_migrations()
            if migration_id in applied:
                applied.remove(migration_id)
                self.redis_client.set(self.migration_key, json.dumps(applied))

            return True

        except Exception as e:
            print(f"Rollback {migration_id} failed: {e}")
            return False

    def migrate_to_latest(self) -> bool:
        """Migrate to latest version."""
        applied = self.get_applied_migrations()
        available = self.get_available_migrations()

        pending = [m for m in available if m not in applied]

        for migration_id in pending:
            print(f"Applying migration: {migration_id}")
            if not self.apply_migration(migration_id):
                return False

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get migration status."""
        applied = self.get_applied_migrations()
        available = self.get_available_migrations()
        pending = [m for m in available if m not in applied]

        return {
            "applied": applied,
            "available": available,
            "pending": pending,
            "up_to_date": len(pending) == 0
        }

    def close(self):
        """Close Redis connection."""
        self.redis_client.close()

def main():
    """Main Redis migration management function."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    manager = RedisMigrationManager(redis_url)

    if len(sys.argv) < 2:
        print("Usage: python redis_migration_manager.py <command> [args]")
        print("Commands: status, migrate, rollback")
        return

    command = sys.argv[1]

    if command == "status":
        status = manager.get_status()
        print(f"Applied migrations: {status['applied']}")
        print(f"Available migrations: {status['available']}")
        print(f"Pending migrations: {status['pending']}")
        print(f"Up to date: {status['up_to_date']}")

    elif command == "migrate":
        if len(sys.argv) > 2:
            migration_id = sys.argv[2]
            success = manager.apply_migration(migration_id)
            print(f"Migration {'successful' if success else 'failed'}")
        else:
            success = manager.migrate_to_latest()
            print(f"Migration to latest {'successful' if success else 'failed'}")

    elif command == "rollback":
        if len(sys.argv) < 3:
            print("Usage: python redis_migration_manager.py rollback <migration_id>")
            return

        migration_id = sys.argv[2]
        success = manager.rollback_migration(migration_id)
        print(f"Rollback {'successful' if success else 'failed'}")

    else:
        print(f"Unknown command: {command}")

    manager.close()

if __name__ == "__main__":
    main()
```

## Migration Best Practices

### Schema Design

1. **Backward Compatibility**: Design migrations to be backward compatible
2. **Gradual Changes**: Make changes gradually to minimize risk
3. **Data Validation**: Validate data before and after migrations
4. **Performance Impact**: Consider performance impact of migrations
5. **Rollback Planning**: Plan for rollback scenarios

### Data Migration

1. **Data Backup**: Always backup data before migrations
2. **Incremental Migration**: Use incremental migration for large datasets
3. **Data Validation**: Validate data integrity after migrations
4. **Performance Monitoring**: Monitor performance during migrations
5. **Error Handling**: Implement proper error handling

### Testing

1. **Test Migrations**: Test migrations in development environment
2. **Rollback Testing**: Test rollback procedures
3. **Performance Testing**: Test migration performance
4. **Data Integrity**: Verify data integrity after migrations
5. **Integration Testing**: Test migrations with application code

## Troubleshooting

### Common Issues

#### 1. Migration Conflicts

```bash
# Check for migration conflicts
alembic heads

# Resolve conflicts
alembic merge -m "merge conflict resolution" head1 head2
```

#### 2. Data Loss Prevention

```python
# Backup before migration
def backup_before_migration():
    """Backup database before migration."""
    import subprocess
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"backup_{timestamp}.sql"

    subprocess.run([
        "pg_dump",
        "-h", "localhost",
        "-U", "user",
        "-d", "medical_kg",
        "-f", backup_file
    ])

    return backup_file
```

#### 3. Performance Issues

```python
# Monitor migration performance
def monitor_migration_performance():
    """Monitor migration performance."""
    import time
    import psutil

    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss

    # Run migration
    # ... migration code ...

    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss

    print(f"Migration took: {end_time - start_time:.2f} seconds")
    print(f"Memory used: {(end_memory - start_memory) / 1024 / 1024:.2f} MB")
```

### Debug Commands

```bash
# Check migration status
python scripts/migration_manager.py status

# Check Neo4j migration status
python scripts/neo4j_migration_manager.py status

# Check Redis migration status
python scripts/redis_migration_manager.py status

# Backup database
pg_dump -h localhost -U user -d medical_kg > backup.sql

# Restore database
psql -h localhost -U user -d medical_kg < backup.sql

# Check database size
psql -h localhost -U user -d medical_kg -c "SELECT pg_size_pretty(pg_database_size('medical_kg'));"
```

## Related Documentation

- [Development Workflow](development_workflow.md)
- [Database Configuration](database_configuration.md)
- [Backup and Recovery](backup_recovery.md)
- [Troubleshooting Guide](troubleshooting.md)
