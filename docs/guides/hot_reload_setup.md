# Hot Reload and Development Server Setup

This document provides comprehensive guidance on setting up hot reload functionality and development servers for the Medical_KG_rev system, enabling rapid development cycles and improved productivity.

## Overview

Hot reload allows developers to see code changes immediately without restarting the entire application. This guide covers setup for various components of the Medical_KG_rev system, including the API gateway, services, and frontend components.

## FastAPI Hot Reload

### Development Server Configuration

#### Basic Hot Reload Setup

```python
# scripts/dev_server.py
import uvicorn
import os
from pathlib import Path

def start_dev_server():
    """Start development server with hot reload."""
    # Get project root directory
    project_root = Path(__file__).parent.parent

    # Set environment variables for development
    os.environ.setdefault("GATEWAY_LOG_LEVEL", "DEBUG")
    os.environ.setdefault("GATEWAY_HOST", "0.0.0.0")
    os.environ.setdefault("GATEWAY_PORT", "8000")

    # Start server with hot reload
    uvicorn.run(
        "Medical_KG_rev.gateway.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root / "src")],
        log_level="debug",
        access_log=True
    )

if __name__ == "__main__":
    start_dev_server()
```

#### Advanced Hot Reload Configuration

```python
# scripts/advanced_dev_server.py
import uvicorn
import os
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CodeReloadHandler(FileSystemEventHandler):
    """Custom file system event handler for code reloading."""

    def __init__(self, app_path):
        self.app_path = app_path
        self.last_reload = 0

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        # Only reload Python files
        if not event.src_path.endswith('.py'):
            return

        # Prevent multiple reloads for the same file
        import time
        current_time = time.time()
        if current_time - self.last_reload < 1:
            return

        self.last_reload = current_time

        # Reload the application
        print(f"🔄 Reloading due to change in: {event.src_path}")
        # The uvicorn reload mechanism will handle the actual reload

def start_advanced_dev_server():
    """Start development server with advanced hot reload."""
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"

    # Set up file watcher
    event_handler = CodeReloadHandler(src_path)
    observer = Observer()
    observer.schedule(event_handler, str(src_path), recursive=True)
    observer.start()

    try:
        # Start server with hot reload
        uvicorn.run(
            "Medical_KG_rev.gateway.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=[str(src_path)],
            reload_excludes=["*.pyc", "__pycache__", "*.log"],
            log_level="debug",
            access_log=True,
            use_colors=True
        )
    except KeyboardInterrupt:
        observer.stop()
    finally:
        observer.join()

if __name__ == "__main__":
    start_advanced_dev_server()
```

### Docker Development Setup

#### Docker Compose for Development

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  gateway:
    build:
      context: .
      dockerfile: docker/gateway/Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src:ro
      - ./config:/app/config:ro
      - ./tests:/app/tests:ro
    environment:
      - GATEWAY_HOST=0.0.0.0
      - GATEWAY_PORT=8000
      - GATEWAY_LOG_LEVEL=DEBUG
      - PYTHONPATH=/app/src
    depends_on:
      - postgres
      - neo4j
      - redis
    command: ["python", "-m", "uvicorn", "Medical_KG_rev.gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/src"]
    networks:
      - medical-kg-rev-dev

  services:
    build:
      context: .
      dockerfile: docker/services/Dockerfile.dev
    volumes:
      - ./src:/app/src:ro
      - ./config:/app/config:ro
    environment:
      - SERVICES_LOG_LEVEL=DEBUG
      - PYTHONPATH=/app/src
    depends_on:
      - postgres
      - neo4j
      - redis
    command: ["python", "-m", "Medical_KG_rev.services.main"]
    networks:
      - medical-kg-rev-dev

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: medical_kg_dev
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
    networks:
      - medical-kg-rev-dev

  neo4j:
    image: neo4j:5
    environment:
      NEO4J_AUTH: neo4j/dev
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - "7687:7687"
      - "7474:7474"
    volumes:
      - neo4j_dev_data:/data
    networks:
      - medical-kg-rev-dev

  redis:
    image: redis:7
    ports:
      - "6379:6379"
    volumes:
      - redis_dev_data:/data
    networks:
      - medical-kg-rev-dev

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_dev_data:/qdrant/storage
    networks:
      - medical-kg-rev-dev

volumes:
  postgres_dev_data:
  neo4j_dev_data:
  redis_dev_data:
  qdrant_dev_data:

networks:
  medical-kg-rev-dev:
    driver: bridge
```

#### Development Dockerfile

```dockerfile
# docker/gateway/Dockerfile.dev
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Install development tools
RUN pip install watchdog uvicorn[standard]

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "Medical_KG_rev.gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/src"]
```

### VS Code Development Configuration

#### Launch Configuration for Hot Reload

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Gateway (Hot Reload)",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/scripts/dev_server.py",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src",
        "GATEWAY_HOST": "0.0.0.0",
        "GATEWAY_PORT": "8000",
        "GATEWAY_LOG_LEVEL": "DEBUG"
      },
      "args": [],
      "justMyCode": false,
      "cwd": "${workspaceFolder}"
    },
    {
      "name": "Services (Hot Reload)",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/scripts/dev_services.py",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src",
        "SERVICES_LOG_LEVEL": "DEBUG"
      },
      "args": [],
      "justMyCode": false,
      "cwd": "${workspaceFolder}"
    }
  ]
}
```

#### Tasks for Development

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Start Dev Server",
      "type": "shell",
      "command": "python",
      "args": ["scripts/dev_server.py"],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      },
      "options": {
        "env": {
          "PYTHONPATH": "${workspaceFolder}/src"
        }
      }
    },
    {
      "label": "Start Dev Services",
      "type": "shell",
      "command": "python",
      "args": ["scripts/dev_services.py"],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      },
      "options": {
        "env": {
          "PYTHONPATH": "${workspaceFolder}/src"
        }
      }
    },
    {
      "label": "Start Docker Dev",
      "type": "shell",
      "command": "docker-compose",
      "args": ["-f", "docker-compose.dev.yml", "up", "--build"],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Stop Docker Dev",
      "type": "shell",
      "command": "docker-compose",
      "args": ["-f", "docker-compose.dev.yml", "down"],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    }
  ]
}
```

## Frontend Hot Reload

### React Development Setup

#### Webpack Configuration

```javascript
// webpack.config.js
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const webpack = require('webpack');

module.exports = {
  mode: 'development',
  entry: './src/index.js',
  devServer: {
    contentBase: path.join(__dirname, 'dist'),
    port: 3000,
    hot: true,
    open: true,
    historyApiFallback: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false
      }
    }
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: './src/index.html'
    }),
    new webpack.HotModuleReplacementPlugin()
  ],
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env', '@babel/preset-react']
          }
        }
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader']
      }
    ]
  },
  resolve: {
    extensions: ['.js', '.jsx']
  }
};
```

#### Package.json Scripts

```json
{
  "scripts": {
    "start": "webpack serve --mode development",
    "build": "webpack --mode production",
    "dev": "webpack serve --mode development --hot",
    "test": "jest",
    "lint": "eslint src/",
    "format": "prettier --write src/"
  }
}
```

### Vue.js Development Setup

#### Vite Configuration

```javascript
// vite.config.js
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 3000,
    hot: true,
    open: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
});
```

## Database Hot Reload

### Database Migration Hot Reload

#### Alembic Configuration

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

#### Migration Hot Reload Script

```python
# scripts/migration_hot_reload.py
import os
import sys
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from alembic import command
from alembic.config import Config

class MigrationReloadHandler(FileSystemEventHandler):
    """Handle migration file changes."""

    def __init__(self, alembic_cfg):
        self.alembic_cfg = alembic_cfg
        self.last_reload = 0

    def on_modified(self, event):
        """Handle migration file modifications."""
        if event.is_directory:
            return

        # Only handle migration files
        if not event.src_path.endswith('.py') or 'migrations' not in event.src_path:
            return

        # Prevent multiple reloads
        current_time = time.time()
        if current_time - self.last_reload < 2:
            return

        self.last_reload = current_time

        print(f"🔄 Migration file changed: {event.src_path}")

        try:
            # Run migration
            command.upgrade(self.alembic_cfg, "head")
            print("✅ Migration applied successfully")
        except Exception as e:
            print(f"❌ Migration failed: {e}")

def start_migration_hot_reload():
    """Start migration hot reload."""
    # Set up Alembic
    alembic_cfg = Config("alembic.ini")

    # Set up file watcher
    event_handler = MigrationReloadHandler(alembic_cfg)
    observer = Observer()

    # Watch migration directory
    migration_dir = Path("alembic/versions")
    if migration_dir.exists():
        observer.schedule(event_handler, str(migration_dir), recursive=True)

    observer.start()

    try:
        print("🔄 Migration hot reload started")
        print("Watching for changes in migration files...")

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("🛑 Migration hot reload stopped")
    finally:
        observer.join()

if __name__ == "__main__":
    start_migration_hot_reload()
```

## Configuration Hot Reload

### Configuration File Watching

#### Configuration Reload Handler

```python
# scripts/config_hot_reload.py
import os
import sys
import time
import yaml
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from Medical_KG_rev.config import Config

class ConfigReloadHandler(FileSystemEventHandler):
    """Handle configuration file changes."""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.last_reload = 0

    def on_modified(self, event):
        """Handle configuration file modifications."""
        if event.is_directory:
            return

        # Only handle YAML files
        if not event.src_path.endswith(('.yaml', '.yml')):
            return

        # Prevent multiple reloads
        current_time = time.time()
        if current_time - self.last_reload < 1:
            return

        self.last_reload = current_time

        print(f"🔄 Configuration file changed: {event.src_path}")

        try:
            # Reload configuration
            self.config_manager.reload_config()
            print("✅ Configuration reloaded successfully")
        except Exception as e:
            print(f"❌ Configuration reload failed: {e}")

def start_config_hot_reload():
    """Start configuration hot reload."""
    # Set up configuration manager
    config_manager = Config()

    # Set up file watcher
    event_handler = ConfigReloadHandler(config_manager)
    observer = Observer()

    # Watch config directory
    config_dir = Path("config")
    if config_dir.exists():
        observer.schedule(event_handler, str(config_dir), recursive=True)

    observer.start()

    try:
        print("🔄 Configuration hot reload started")
        print("Watching for changes in configuration files...")

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("🛑 Configuration hot reload stopped")
    finally:
        observer.join()

if __name__ == "__main__":
    start_config_hot_reload()
```

## Performance Optimization

### Hot Reload Performance Tips

#### 1. Exclude Unnecessary Files

```python
# Exclude patterns for better performance
EXCLUDE_PATTERNS = [
    "*.pyc",
    "__pycache__",
    "*.log",
    "*.tmp",
    ".git",
    "node_modules",
    "venv",
    ".pytest_cache",
    "htmlcov",
    ".coverage"
]
```

#### 2. Optimize File Watching

```python
# Use inotify for better performance on Linux
import platform

if platform.system() == "Linux":
    from watchdog.observers import Observer
    from watchdog.observers.inotify import InotifyObserver

    # Use inotify observer for better performance
    observer = InotifyObserver()
else:
    observer = Observer()
```

#### 3. Debounce File Changes

```python
import asyncio
from functools import wraps

def debounce(wait_time):
    """Debounce function calls."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await asyncio.sleep(wait_time)
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@debounce(0.5)
async def handle_file_change(file_path):
    """Handle file change with debouncing."""
    print(f"File changed: {file_path}")
    # Process the change
```

## Troubleshooting

### Common Issues

#### 1. Hot Reload Not Working

```bash
# Check if file watching is working
python -c "
import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class TestHandler(FileSystemEventHandler):
    def on_modified(self, event):
        print(f'File changed: {event.src_path}')

observer = Observer()
handler = TestHandler()
observer.schedule(handler, '.', recursive=True)
observer.start()

print('File watcher started. Modify a file to test...')
try:
    while True:
        pass
except KeyboardInterrupt:
    observer.stop()
"
```

#### 2. Port Already in Use

```bash
# Check what's using the port
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
export GATEWAY_PORT=8001
```

#### 3. File Permission Issues

```bash
# Check file permissions
ls -la src/

# Fix permissions if needed
chmod -R 755 src/
```

### Debug Commands

```bash
# Check if hot reload is working
python scripts/dev_server.py --reload

# Check file watcher status
ps aux | grep watchdog

# Monitor file changes
inotifywait -m -r src/

# Check Python path
python -c "import sys; print(sys.path)"

# Verify module imports
python -c "from Medical_KG_rev.gateway.main import app; print('OK')"
```

## Best Practices

### Development Workflow

1. **Use Hot Reload**: Enable hot reload for all development servers
2. **Monitor Performance**: Watch for performance impact of file watching
3. **Exclude Unnecessary Files**: Configure proper exclude patterns
4. **Use Debouncing**: Implement debouncing for frequent file changes
5. **Test Hot Reload**: Regularly test hot reload functionality

### Configuration Management

1. **Environment Variables**: Use environment variables for configuration
2. **Configuration Validation**: Validate configuration on reload
3. **Error Handling**: Handle configuration errors gracefully
4. **Backup Configuration**: Keep backup of working configurations
5. **Documentation**: Document configuration changes

### Performance Optimization

1. **File Watching**: Use efficient file watching mechanisms
2. **Memory Management**: Monitor memory usage during development
3. **CPU Usage**: Watch CPU usage of file watchers
4. **Network Optimization**: Optimize network requests during development
5. **Caching**: Implement appropriate caching strategies

## Related Documentation

- [Development Workflow](development_workflow.md)
- [IDE Configuration](ide_configuration.md)
- [Environment Setup](environment_setup.md)
- [Troubleshooting Guide](troubleshooting.md)
