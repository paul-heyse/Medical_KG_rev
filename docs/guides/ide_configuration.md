# IDE Configuration Guide

This document provides comprehensive configuration guidance for Integrated Development Environments (IDEs) used with the Medical_KG_rev system, including VS Code, PyCharm, and debugging setup.

## Overview

Proper IDE configuration is essential for efficient development of the Medical_KG_rev system. This guide covers setup for popular IDEs, debugging configurations, and productivity tools.

## VS Code Configuration

### Workspace Settings

#### `.vscode/settings.json`

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "ruff",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests"
  ],
  "python.testing.unittestEnabled": false,
  "python.testing.nosetestsEnabled": false,
  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.autoImportCompletions": true,
  "python.analysis.autoSearchPaths": true,
  "python.analysis.extraPaths": [
    "./src"
  ],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/htmlcov": true,
    "**/.coverage": true,
    "**/node_modules": true,
    "**/.git": true,
    "**/.DS_Store": true
  },
  "search.exclude": {
    "**/node_modules": true,
    "**/bower_components": true,
    "**/*.code-search": true,
    "**/htmlcov": true,
    "**/.pytest_cache": true
  },
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true,
    "source.fixAll": true
  },
  "editor.rulers": [88, 120],
  "editor.tabSize": 4,
  "editor.insertSpaces": true,
  "editor.detectIndentation": false,
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "files.trimFinalNewlines": true,
  "git.enableSmartCommit": true,
  "git.confirmSync": false,
  "git.autofetch": true,
  "terminal.integrated.defaultProfile.linux": "bash",
  "terminal.integrated.profiles.linux": {
    "bash": {
      "path": "/bin/bash",
      "args": ["-l"]
    }
  }
}
```

#### `.vscode/launch.json`

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Gateway",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/src/Medical_KG_rev/gateway/main.py",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src",
        "GATEWAY_HOST": "0.0.0.0",
        "GATEWAY_PORT": "8000",
        "GATEWAY_LOG_LEVEL": "DEBUG"
      },
      "args": [],
      "justMyCode": false
    },
    {
      "name": "Python: Services",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/src/Medical_KG_rev/services/main.py",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src",
        "SERVICES_LOG_LEVEL": "DEBUG"
      },
      "args": [],
      "justMyCode": false
    },
    {
      "name": "Python: Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": [
        "${workspaceFolder}/tests",
        "-v",
        "--tb=short"
      ],
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      },
      "justMyCode": false
    },
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      },
      "justMyCode": false
    },
    {
      "name": "Docker: Compose Up",
      "type": "node",
      "request": "launch",
      "program": "${workspaceFolder}/scripts/docker-compose-debug.js",
      "console": "integratedTerminal",
      "env": {
        "COMPOSE_FILE": "docker-compose.yml"
      }
    }
  ]
}
```

#### `.vscode/tasks.json`

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Install Dependencies",
      "type": "shell",
      "command": "pip",
      "args": [
        "install",
        "-r",
        "requirements.txt",
        "-r",
        "requirements-dev.txt"
      ],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Run Tests",
      "type": "shell",
      "command": "pytest",
      "args": [
        "tests/",
        "-v",
        "--cov=src/",
        "--cov-report=html"
      ],
      "group": "test",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Run Linting",
      "type": "shell",
      "command": "ruff",
      "args": [
        "check",
        "src/",
        "tests/"
      ],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Format Code",
      "type": "shell",
      "command": "ruff",
      "args": [
        "format",
        "src/",
        "tests/"
      ],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Type Check",
      "type": "shell",
      "command": "mypy",
      "args": [
        "src/"
      ],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Start Services",
      "type": "shell",
      "command": "docker-compose",
      "args": [
        "up",
        "-d"
      ],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Stop Services",
      "type": "shell",
      "command": "docker-compose",
      "args": [
        "down"
      ],
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

### Recommended Extensions

#### `.vscode/extensions.json`

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.flake8",
    "ms-python.mypy-type-checker",
    "charliermarsh.ruff",
    "ms-python.black-formatter",
    "ms-python.isort",
    "ms-python.pytest",
    "ms-vscode.vscode-json",
    "redhat.vscode-yaml",
    "ms-kubernetes-tools.vscode-kubernetes-tools",
    "ms-azuretools.vscode-docker",
    "GitHub.vscode-pull-request-github",
    "GitHub.copilot",
    "GitHub.copilot-chat",
    "ms-vscode.vscode-typescript-next",
    "bradlc.vscode-tailwindcss",
    "esbenp.prettier-vscode",
    "ms-vscode.vscode-eslint",
    "formulahendry.auto-rename-tag",
    "christian-kohler.path-intellisense",
    "ms-vscode.vscode-github-issue-notebooks",
    "ms-vscode.vscode-markdown-notebook",
    "ms-vscode.vscode-markdown-math",
    "yzhang.markdown-all-in-one",
    "shd101wyy.markdown-preview-enhanced",
    "ms-vscode.vscode-markdown",
    "ms-vscode.vscode-markdown-preview",
    "ms-vscode.vscode-markdown-preview-enhanced",
    "ms-vscode.vscode-markdown-math",
    "ms-vscode.vscode-markdown-notebook",
    "ms-vscode.vscode-github-issue-notebooks"
  ]
}
```

## PyCharm Configuration

### Project Settings

#### Python Interpreter

1. **File** → **Settings** → **Project** → **Python Interpreter**
2. Select the virtual environment: `./venv/bin/python`
3. Ensure all project dependencies are installed

#### Code Style

1. **File** → **Settings** → **Editor** → **Code Style** → **Python**
2. Set line length to 88 characters
3. Configure import optimization
4. Set up code formatting rules

#### Testing

1. **File** → **Settings** → **Tools** → **Python Integrated Tools**
2. Set default test runner to pytest
3. Configure test discovery patterns
4. Set up coverage reporting

### Run Configurations

#### Gateway Service

```json
{
  "name": "Gateway Service",
  "type": "python",
  "request": "launch",
  "program": "src/Medical_KG_rev/gateway/main.py",
  "console": "integratedTerminal",
  "env": {
    "PYTHONPATH": "src",
    "GATEWAY_HOST": "0.0.0.0",
    "GATEWAY_PORT": "8000",
    "GATEWAY_LOG_LEVEL": "DEBUG"
  },
  "args": [],
  "justMyCode": false
}
```

#### Test Configuration

```json
{
  "name": "Tests",
  "type": "python",
  "request": "launch",
  "module": "pytest",
  "args": [
    "tests/",
    "-v",
    "--tb=short"
  ],
  "console": "integratedTerminal",
  "env": {
    "PYTHONPATH": "src"
  },
  "justMyCode": false
}
```

## Debugging Configuration

### Breakpoint Strategies

#### Strategic Breakpoints

1. **Entry Points**: API endpoints, main functions
2. **Error Handling**: Exception blocks, error paths
3. **Data Flow**: Input validation, data transformation
4. **External Calls**: Database queries, API calls

#### Conditional Breakpoints

```python
# Example: Break only on specific conditions
def process_document(document_id: str):
    # Breakpoint: document_id == "specific-id"
    if document_id == "specific-id":
        breakpoint()  # Will only break for this specific ID

    # Breakpoint: When processing fails
    try:
        result = process_document_content(document_id)
    except Exception as e:
        breakpoint()  # Break on any exception
        raise
```

### Debugging Tools

#### Python Debugger (pdb)

```python
import pdb

def complex_function(data):
    pdb.set_trace()  # Set breakpoint
    # Debugging commands:
    # n (next line)
    # s (step into)
    # c (continue)
    # l (list code)
    # p variable_name (print variable)
    # pp variable_name (pretty print)
    # h (help)
    # q (quit)

    processed_data = transform_data(data)
    return processed_data
```

#### IPython Debugger (ipdb)

```python
import ipdb

def debug_function():
    ipdb.set_trace()  # Enhanced pdb with IPython features
    # Additional commands:
    # %debug (debug last exception)
    # %pdb (automatic debugging on exceptions)
    # %run (run file in debugger)
```

#### Remote Debugging

```python
# For debugging remote services
import debugpy

def start_remote_debugging():
    debugpy.listen(("0.0.0.0", 5678))
    debugpy.wait_for_client()  # Wait for debugger to attach
    print("Debugger attached")
```

### Debugging Workflows

#### Local Development

1. Set breakpoints in IDE
2. Start debugging session
3. Step through code execution
4. Inspect variables and state
5. Modify variables if needed

#### Remote Development

1. Configure remote debugging
2. Attach debugger to remote process
3. Set breakpoints remotely
4. Debug as if local

#### Production Debugging

1. Enable debug logging
2. Use structured logging
3. Add debug endpoints
4. Monitor application state

## Productivity Tools

### Code Snippets

#### VS Code Snippets

```json
{
  "Python Function": {
    "prefix": "def",
    "body": [
      "def ${1:function_name}(${2:parameters}):",
      "    \"\"\"${3:Function description}.",
      "",
      "    Args:",
      "        ${2:parameters}: ${4:Parameter description}",
      "",
      "    Returns:",
      "        ${5:Return description}",
      "    \"\"\"",
      "    ${6:pass}"
    ],
    "description": "Python function with docstring"
  },
  "Python Class": {
    "prefix": "class",
    "body": [
      "class ${1:ClassName}:",
      "    \"\"\"${2:Class description}.",
      "",
      "    ${3:Additional class documentation}",
      "    \"\"\"",
      "",
      "    def __init__(self${4:, parameters}):",
      "        \"\"\"Initialize ${1:ClassName}.",
      "",
      "        Args:",
      "            ${4:parameters}: ${5:Parameter description}",
      "        \"\"\"",
      "        ${6:pass}"
    ],
    "description": "Python class with docstring"
  },
  "Python Test": {
    "prefix": "test",
    "body": [
      "def test_${1:test_name}():",
      "    \"\"\"Test ${2:test description}.",
      "",
      "    ${3:Test documentation}",
      "    \"\"\"",
      "    # Arrange",
      "    ${4:setup_code}",
      "",
      "    # Act",
      "    ${5:action_code}",
      "",
      "    # Assert",
      "    ${6:assertion_code}"
    ],
    "description": "Python test function"
  }
}
```

### Git Integration

#### Git Hooks

```bash
#!/bin/sh
# .git/hooks/pre-commit

# Run linting
echo "Running pre-commit checks..."
ruff check src/ tests/
if [ $? -ne 0 ]; then
    echo "❌ Linting failed"
    exit 1
fi

# Run type checking
mypy src/
if [ $? -ne 0 ]; then
    echo "❌ Type checking failed"
    exit 1
fi

# Run tests
pytest tests/unit/ -q
if [ $? -ne 0 ]; then
    echo "❌ Tests failed"
    exit 1
fi

echo "✅ Pre-commit checks passed"
```

### Keyboard Shortcuts

#### VS Code Shortcuts

- `Ctrl+Shift+P`: Command Palette
- `Ctrl+P`: Quick Open
- `Ctrl+Shift+F`: Search in Files
- `Ctrl+Shift+H`: Replace in Files
- `F5`: Start Debugging
- `Shift+F5`: Stop Debugging
- `F9`: Toggle Breakpoint
- `F10`: Step Over
- `F11`: Step Into
- `Shift+F11`: Step Out
- `Ctrl+Shift+E`: Explorer
- `Ctrl+Shift+G`: Source Control
- `Ctrl+Shift+D`: Debug
- `Ctrl+Shift+X`: Extensions

#### PyCharm Shortcuts

- `Ctrl+Shift+A`: Find Action
- `Ctrl+N`: Go to Class
- `Ctrl+Shift+N`: Go to File
- `Ctrl+Alt+Shift+N`: Go to Symbol
- `F5`: Start Debugging
- `Shift+F5`: Stop Debugging
- `F9`: Toggle Breakpoint
- `F8`: Step Over
- `F7`: Step Into
- `Shift+F8`: Step Out
- `Ctrl+Shift+F`: Find in Path
- `Ctrl+Shift+R`: Replace in Path
- `Alt+F7`: Find Usages
- `Ctrl+B`: Go to Declaration

## Troubleshooting

### Common Issues

#### Python Path Issues

```bash
# Check Python path
echo $PYTHONPATH

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Verify Python can find modules
python -c "import Medical_KG_rev; print('OK')"
```

#### Virtual Environment Issues

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Verify installation
pip list
```

#### Debugging Issues

```bash
# Check if debugger is running
ps aux | grep debugpy

# Kill debugger process
pkill -f debugpy

# Check port availability
netstat -tulpn | grep 5678
```

### Debug Commands

```bash
# Check IDE configuration
code --list-extensions
code --show-versions

# Check Python environment
python --version
pip --version
which python
which pip

# Check project structure
tree -I '__pycache__|*.pyc|.git|node_modules'

# Run tests with debugging
pytest tests/ -v -s --pdb

# Run specific test with debugging
pytest tests/unit/test_specific.py -v -s --pdb
```

## Best Practices

### IDE Configuration

1. **Consistent Settings**: Use shared workspace settings
2. **Extension Management**: Recommend essential extensions
3. **Code Formatting**: Configure automatic formatting
4. **Linting Integration**: Enable real-time linting
5. **Testing Integration**: Set up test discovery and execution

### Debugging Practices

1. **Strategic Breakpoints**: Place breakpoints thoughtfully
2. **Conditional Debugging**: Use conditions for complex scenarios
3. **Logging Integration**: Combine debugging with logging
4. **Remote Debugging**: Set up remote debugging for distributed systems
5. **Performance Debugging**: Use profiling tools for performance issues

### Productivity Tips

1. **Keyboard Shortcuts**: Learn and use keyboard shortcuts
2. **Code Snippets**: Create and use code snippets
3. **Git Integration**: Use IDE Git features effectively
4. **Extension Management**: Keep extensions updated
5. **Workspace Organization**: Organize workspace efficiently

## Related Documentation

- [Development Workflow](development_workflow.md)
- [Environment Setup](environment_setup.md)
- [Testing Strategy](testing_strategy.md)
- [Troubleshooting Guide](troubleshooting.md)
