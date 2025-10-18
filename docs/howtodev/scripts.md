# Developing Standalone Scripts

## Overview
When developing scripts in the LAMP project, we prioritize **reproducibility** and **scalability**. All scripts should be designed as Typer applications with self-documenting interfaces and clear execution instructions and paired with configs used to run the scripts in the project.

## 🎯 Core Principles

### 1. Reproducibility
- **Self-contained**: Scripts should work independently with minimal dependencies
- **Versioned**: All scripts should be version-controlled and tagged
- **Documented**: Clear instructions on how to run with specific inputs
- **Deterministic**: Same inputs should always produce same outputs

### 2. Scalability
- **Modular**: Scripts should be composable and reusable
- **Configurable**: Use configuration files for different scenarios
- **Parallelizable**: Support batch processing when applicable
- **Resource-aware**: Handle memory and compute constraints

### 3. Code Quality
- **Formatting**: Use `ruff format` to maintain consistent code style
- **Linting**: Run `ruff check` to catch common issues
- **Type Checking**: Use `pyright` for static type checking
- **Pre-commit**: Set up pre-commit hooks for automated checks