# Git Workflow and Branching Strategies

This document provides comprehensive guidance on Git workflow and branching strategies for the Medical_KG_rev system, including branching models, commit conventions, pull request procedures, and collaboration best practices.

## Overview

The Medical_KG_rev project uses GitFlow branching strategy with feature branches, pull requests, and automated testing. This ensures code quality, collaboration efficiency, and system reliability.

## Branching Strategy

### Branch Types

#### Main Branches

- **`main`**: Production-ready code
- **`develop`**: Integration branch for features
- **`release/*`**: Release preparation branches
- **`hotfix/*`**: Critical production fixes

#### Feature Branches

- **`feature/*`**: New features and enhancements
- **`bugfix/*`**: Bug fixes
- **`refactor/*`**: Code refactoring
- **`docs/*`**: Documentation updates

### Branch Naming Convention

```bash
# Feature branches
feature/add-user-authentication
feature/implement-graphql-api
feature/add-biomedical-adapters

# Bug fix branches
bugfix/fix-database-connection-pool
bugfix/resolve-memory-leak-in-embeddings

# Refactoring branches
refactor/simplify-gateway-routing
refactor/optimize-vector-search

# Documentation branches
docs/update-api-documentation
docs/add-deployment-guide

# Release branches
release/v1.2.0
release/v2.0.0

# Hotfix branches
hotfix/critical-security-patch
hotfix/fix-production-crash
```

## Git Workflow

### Feature Development Workflow

#### 1. Create Feature Branch

```bash
# Start from develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/add-new-adapter

# Push branch to remote
git push -u origin feature/add-new-adapter
```

#### 2. Development Process

```bash
# Make changes and commit
git add .
git commit -m "feat: add OpenAlex adapter implementation"

# Push changes
git push origin feature/add-new-adapter

# Keep branch updated
git fetch origin
git rebase origin/develop

# Resolve conflicts if any
git add .
git rebase --continue
```

#### 3. Complete Feature

```bash
# Create pull request
# - Open PR from feature branch to develop
# - Add description, tests, and documentation
# - Request code review

# Address feedback
git add .
git commit -m "fix: address code review feedback"
git push origin feature/add-new-adapter

# After PR is approved and merged
git checkout develop
git pull origin develop
git branch -d feature/add-new-adapter
git push origin --delete feature/add-new-adapter
```

### Bug Fix Workflow

#### 1. Create Bug Fix Branch

```bash
# Start from develop
git checkout develop
git pull origin develop

# Create bug fix branch
git checkout -b bugfix/fix-database-connection-pool

# Push branch to remote
git push -u origin bugfix/fix-database-connection-pool
```

#### 2. Fix and Test

```bash
# Make bug fix
git add .
git commit -m "fix: resolve database connection pool exhaustion"

# Add tests
git add .
git commit -m "test: add tests for connection pool fix"

# Push changes
git push origin bugfix/fix-database-connection-pool
```

#### 3. Complete Bug Fix

```bash
# Create pull request
# - Open PR from bug fix branch to develop
# - Add description and tests
# - Request code review

# After PR is approved and merged
git checkout develop
git pull origin develop
git branch -d bugfix/fix-database-connection-pool
git push origin --delete bugfix/fix-database-connection-pool
```

### Release Workflow

#### 1. Create Release Branch

```bash
# Start from develop
git checkout develop
git pull origin develop

# Create release branch
git checkout -b release/v1.2.0

# Push branch to remote
git push -u origin release/v1.2.0
```

#### 2. Release Preparation

```bash
# Update version numbers
git add .
git commit -m "chore: bump version to 1.2.0"

# Update changelog
git add .
git commit -m "docs: update changelog for v1.2.0"

# Push changes
git push origin release/v1.2.0
```

#### 3. Complete Release

```bash
# Merge to main
git checkout main
git pull origin main
git merge release/v1.2.0
git tag v1.2.0
git push origin main
git push origin v1.2.0

# Merge back to develop
git checkout develop
git pull origin develop
git merge release/v1.2.0
git push origin develop

# Delete release branch
git branch -d release/v1.2.0
git push origin --delete release/v1.2.0
```

### Hotfix Workflow

#### 1. Create Hotfix Branch

```bash
# Start from main
git checkout main
git pull origin main

# Create hotfix branch
git checkout -b hotfix/critical-security-patch

# Push branch to remote
git push -u origin hotfix/critical-security-patch
```

#### 2. Apply Hotfix

```bash
# Make hotfix
git add .
git commit -m "fix: patch critical security vulnerability"

# Push changes
git push origin hotfix/critical-security-patch
```

#### 3. Complete Hotfix

```bash
# Merge to main
git checkout main
git pull origin main
git merge hotfix/critical-security-patch
git tag v1.2.1
git push origin main
git push origin v1.2.1

# Merge back to develop
git checkout develop
git pull origin develop
git merge hotfix/critical-security-patch
git push origin develop

# Delete hotfix branch
git branch -d hotfix/critical-security-patch
git push origin --delete hotfix/critical-security-patch
```

## Commit Conventions

### Commit Message Format

```
<type>(<scope>): <description>

<body>

<footer>
```

#### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks
- **perf**: Performance improvements
- **ci**: CI/CD changes
- **build**: Build system changes

#### Scopes

- **gateway**: API gateway changes
- **services**: Service layer changes
- **adapters**: Adapter changes
- **storage**: Storage layer changes
- **models**: Data model changes
- **config**: Configuration changes
- **docs**: Documentation changes
- **tests**: Test changes

#### Examples

```bash
# Feature commits
git commit -m "feat(gateway): add GraphQL endpoint for document search"
git commit -m "feat(adapters): implement OpenAlex adapter"

# Bug fix commits
git commit -m "fix(storage): resolve database connection pool exhaustion"
git commit -m "fix(gateway): handle malformed request bodies"

# Documentation commits
git commit -m "docs(api): update OpenAPI specification"
git commit -m "docs(deployment): add Kubernetes deployment guide"

# Refactoring commits
git commit -m "refactor(services): simplify document processing logic"
git commit -m "refactor(models): improve data validation"

# Test commits
git commit -m "test(unit): add tests for document service"
git commit -m "test(integration): add end-to-end tests"

# Chore commits
git commit -m "chore(deps): update dependencies to latest versions"
git commit -m "chore(ci): update GitHub Actions workflow"
```

### Commit Body Guidelines

```bash
# Good commit message
git commit -m "feat(gateway): add rate limiting middleware

- Implement token bucket algorithm for rate limiting
- Add configuration options for rate limits
- Include metrics for monitoring rate limit usage
- Add tests for rate limiting functionality

Closes #123"

# Bad commit message
git commit -m "fix stuff"
```

## Pull Request Guidelines

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Contract tests added/updated
- [ ] Performance tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Tests pass locally
- [ ] CI/CD pipeline passes

## Related Issues
Closes #123
Fixes #456
```

### PR Size Guidelines

- **Small PRs**: < 200 lines changed
- **Medium PRs**: 200-500 lines changed
- **Large PRs**: 500-1000 lines changed
- **Extra Large PRs**: > 1000 lines changed (require special approval)

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

#### Review Timeline

- **Small PRs**: Review within 24 hours
- **Medium PRs**: Review within 48 hours
- **Large PRs**: Review within 72 hours
- **Critical PRs**: Review within 4 hours

## Git Hooks

### Pre-commit Hook

```bash
#!/bin/sh
# .git/hooks/pre-commit

echo "Running pre-commit checks..."

# Run linting
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

### Commit-msg Hook

```bash
#!/bin/sh
# .git/hooks/commit-msg

commit_regex='^(feat|fix|docs|style|refactor|test|chore|perf|ci|build)(\(.+\))?: .{1,50}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "❌ Invalid commit message format"
    echo "Format: <type>(<scope>): <description>"
    echo "Types: feat, fix, docs, style, refactor, test, chore, perf, ci, build"
    exit 1
fi

echo "✅ Commit message format is valid"
```

### Pre-push Hook

```bash
#!/bin/sh
# .git/hooks/pre-push

echo "Running pre-push checks..."

# Run full test suite
pytest tests/ --cov=src/ --cov-fail-under=80
if [ $? -ne 0 ]; then
    echo "❌ Tests failed"
    exit 1
fi

# Run security checks
bandit -r src/
if [ $? -ne 0 ]; then
    echo "❌ Security checks failed"
    exit 1
fi

echo "✅ Pre-push checks passed"
```

## Branch Protection Rules

### Main Branch Protection

```yaml
# .github/branch-protection/main.yml
name: Main Branch Protection
on:
  push:
    branches: [main]

jobs:
  protection:
    runs-on: ubuntu-latest
    steps:
      - name: Check branch protection
        run: |
          echo "Main branch is protected"
          echo "Required: Pull request reviews"
          echo "Required: Status checks"
          echo "Required: Up-to-date branches"
```

### Develop Branch Protection

```yaml
# .github/branch-protection/develop.yml
name: Develop Branch Protection
on:
  push:
    branches: [develop]

jobs:
  protection:
    runs-on: ubuntu-latest
    steps:
      - name: Check branch protection
        run: |
          echo "Develop branch is protected"
          echo "Required: Pull request reviews"
          echo "Required: Status checks"
```

## Collaboration Best Practices

### Code Review Guidelines

#### For Authors

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

#### For Reviewers

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

### Conflict Resolution

#### Merge Conflicts

```bash
# Resolve merge conflicts
git checkout feature/branch
git rebase develop

# Resolve conflicts in files
# Edit conflicted files
git add .
git rebase --continue

# Push resolved changes
git push origin feature/branch --force-with-lease
```

#### Rebase vs Merge

```bash
# Use rebase for clean history
git checkout feature/branch
git rebase develop

# Use merge for feature branches
git checkout develop
git merge feature/branch
```

## Git Tools and Extensions

### Useful Git Aliases

```bash
# Add to ~/.gitconfig
[alias]
    st = status
    co = checkout
    br = branch
    ci = commit
    unstage = reset HEAD --
    last = log -1 HEAD
    visual = !gitk
    lg = log --oneline --decorate --all --graph
    amend = commit --amend --no-edit
    wip = commit -am "WIP"
    unwip = reset HEAD~1
    cleanup = "!git branch --merged | grep -v '\\*\\|main\\|develop' | xargs -n 1 git branch -d"
```

### Git Hooks Management

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Update hooks
pre-commit autoupdate

# Run hooks manually
pre-commit run --all-files
```

### Git Workflow Tools

```bash
# Install git-flow
# Ubuntu/Debian
sudo apt-get install git-flow

# macOS
brew install git-flow

# Initialize git-flow
git flow init

# Start feature
git flow feature start add-new-adapter

# Finish feature
git flow feature finish add-new-adapter
```

## Troubleshooting

### Common Issues

#### 1. Merge Conflicts

```bash
# Check for conflicts
git status

# Resolve conflicts
git mergetool

# Complete merge
git add .
git commit
```

#### 2. Rebase Conflicts

```bash
# Check rebase status
git status

# Resolve conflicts
git add .
git rebase --continue

# Abort rebase if needed
git rebase --abort
```

#### 3. Lost Commits

```bash
# Find lost commits
git reflog

# Recover lost commit
git checkout <commit-hash>
git checkout -b recovered-branch
```

#### 4. Accidental Commits

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Undo multiple commits
git reset --hard HEAD~3
```

### Debug Commands

```bash
# Check repository status
git status
git log --oneline -10
git branch -a

# Check remote configuration
git remote -v
git config --list

# Check file differences
git diff
git diff --cached
git diff HEAD~1

# Check commit history
git log --graph --oneline --all
git log --stat
git log --grep="feat"

# Check branch information
git branch -vv
git show-branch
git merge-base main develop
```

## Best Practices

### Branch Management

1. **Keep Branches Small**: Create focused branches for specific changes
2. **Regular Updates**: Keep feature branches updated with develop
3. **Clean History**: Use rebase to maintain clean commit history
4. **Delete Branches**: Delete merged branches to keep repository clean
5. **Protect Branches**: Use branch protection rules for main branches

### Commit Practices

1. **Atomic Commits**: Make small, focused commits
2. **Clear Messages**: Write descriptive commit messages
3. **Consistent Format**: Follow commit message conventions
4. **Review Commits**: Review commits before pushing
5. **Squash Commits**: Squash related commits in PRs

### Collaboration

1. **Communication**: Communicate changes and intentions
2. **Code Reviews**: Participate actively in code reviews
3. **Documentation**: Keep documentation updated
4. **Testing**: Ensure tests pass before merging
5. **Continuous Integration**: Use CI/CD for automated checks

## Related Documentation

- [Development Workflow](development_workflow.md)
- [Code Review Guidelines](code_review_guidelines.md)
- [CI/CD Pipeline](ci_cd_pipeline.md)
- [Troubleshooting Guide](troubleshooting.md)
