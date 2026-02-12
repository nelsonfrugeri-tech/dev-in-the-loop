# Python Packaging - Python 3.10+

Referência técnica completa de packaging e dependency management em Python. Para decisões de qual ferramenta usar (poetry vs pip-tools vs uv), consulte a skill principal (`/developer`).

## Fundamentos

Python packaging permite distribuir código como pacotes instaláveis. Componentes principais:
- **pyproject.toml**: Arquivo de configuração moderno (PEP 621)
- **Virtual environments**: Isolamento de dependências
- **Dependency management**: Controle de versões
- **Build system**: Geração de wheels/sdists
- **Distribution**: Publicação no PyPI

**Quando criar package:**
- Biblioteca compartilhada entre projetos
- CLI tool para distribuição
- Publicação no PyPI
- Internal package registry

**Quando NÃO criar package:**
- Script único simples
- Aplicação web deployada como container
- Protótipo descartável

---

## Estrutura de Projeto Moderna

### Layout Recomendado
```
my-project/
├── pyproject.toml          # Config moderna (PEP 621)
├── README.md               # Documentação
├── LICENSE                 # Licença
├── .gitignore             # Git ignore
├── src/                   # Source layout (recomendado)
│   └── myproject/
│       ├── __init__.py
│       ├── core.py
│       ├── utils.py
│       └── py.typed       # Type hints marker
├── tests/                 # Tests fora do package
│   ├── __init__.py
│   ├── test_core.py
│   └── test_utils.py
├── docs/                  # Documentação
│   └── index.md
└── scripts/               # Scripts auxiliares
    └── setup_dev.sh
```

### src/ Layout
```python
# src/ layout é recomendado (previne imports acidentais)

# ❌ EVITE - flat layout
myproject/
├── myproject/
│   ├── __init__.py
│   └── core.py
└── tests/

# ✅ RECOMENDADO - src layout
myproject/
├── src/
│   └── myproject/
│       ├── __init__.py
│       └── core.py
└── tests/

# src/ layout força install antes de testar
# Previne import do source sem install
```

---

## pyproject.toml - Configuração Moderna

### Estrutura Básica
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "myproject"
version = "0.1.0"
description = "A short description"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"}
]
keywords = ["api", "web", "framework"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "fastapi>=0.100.0",
    "pydantic>=2.0.0",
    "structlog>=23.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.0.0",
]

[project.urls]
Homepage = "https://github.com/username/myproject"
Documentation = "https://myproject.readthedocs.io"
Repository = "https://github.com/username/myproject"
Issues = "https://github.com/username/myproject/issues"

[project.scripts]
myproject = "myproject.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/myproject"]
```

### Exemplo do Mundo Real

**FastAPI Project:**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-api"
version = "1.2.3"
description = "Production FastAPI application"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "API Team", email = "team@company.com"}
]
classifiers = [
    "Framework :: FastAPI",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    # Web framework
    "fastapi>=0.100.0,<1.0.0",
    "uvicorn[standard]>=0.23.0",
    
    # Database
    "sqlalchemy[asyncio]>=2.0.0",
    "asyncpg>=0.28.0",
    
    # Validation
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    
    # Observability
    "structlog>=23.0.0",
    
    # HTTP client
    "httpx>=0.24.0",
]

[project.optional-dependencies]
dev = [
    # Testing
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "httpx>=0.24.0",
    
    # Code quality
    "ruff>=0.1.0",
    "mypy>=1.5.0",
    "pre-commit>=3.3.0",
    
    # Type stubs
    "types-requests",
]

test = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
]

[project.urls]
Homepage = "https://api.company.com"
Repository = "https://github.com/company/my-api"

[project.scripts]
my-api = "my_api.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/my_api"]

[tool.hatch.build.targets.sdist]
include = [
    "/src",
    "/tests",
    "/README.md",
    "/LICENSE",
]
```

---

## Dependency Management

### Versioning Constraints
```toml
[project]
dependencies = [
    # Exact version (evite)
    "package==1.2.3",
    
    # Minimum version (permissivo)
    "package>=1.2.0",
    
    # Compatible release (recomendado)
    "package~=1.2.0",  # >=1.2.0, <1.3.0
    
    # Range (específico)
    "package>=1.2.0,<2.0.0",
    
    # Extras
    "package[extra]>=1.0.0",
    
    # Git dependency (evite em produção)
    "package @ git+https://github.com/user/package.git@main",
]
```

### Optional Dependencies
```toml
[project.optional-dependencies]
# Development tools
dev = [
    "pytest>=7.0.0",
    "ruff>=0.1.0",
]

# Production dependencies
postgres = [
    "asyncpg>=0.28.0",
]

mysql = [
    "aiomysql>=0.2.0",
]

# All extras
all = [
    "my-api[postgres]",
    "my-api[mysql]",
]
```

### Lock Files
```toml
# pyproject.toml - abstract dependencies
[project]
dependencies = [
    "fastapi>=0.100.0",
]

# requirements.txt - concrete versions (lock file)
fastapi==0.104.1
starlette==0.27.0
pydantic==2.5.0
# ... todas as transitive dependencies
```

---

## Build Backends

### Hatchling (Recomendado)
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/myproject"]

[tool.hatch.build.targets.sdist]
include = [
    "/src",
    "/tests",
    "/README.md",
]
exclude = [
    "/.github",
    "/docs",
]
```

### Setuptools (Legacy)
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

### Poetry
```toml
[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "myproject"
version = "0.1.0"
description = ""
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"
fastapi = "^0.100.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"
```

---

## Virtual Environments

### venv (Built-in)
```bash
# Criar virtual environment
python -m venv .venv

# Ativar
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Instalar dependências
pip install -e ".[dev]"

# Desativar
deactivate
```

### virtualenv (Third-party)
```bash
# Instalar
pip install virtualenv

# Criar
virtualenv .venv

# Mesmo activation que venv
```

### Exemplo do Mundo Real

**Makefile para Desenvolvimento:**
```makefile
.PHONY: install dev test lint clean

# Setup virtual environment
venv:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip

# Install production dependencies
install: venv
	.venv/bin/pip install -e .

# Install development dependencies
dev: venv
	.venv/bin/pip install -e ".[dev]"

# Run tests
test:
	.venv/bin/pytest tests/ -v --cov=src

# Run linters
lint:
	.venv/bin/ruff check src/ tests/
	.venv/bin/mypy src/

# Format code
format:
	.venv/bin/ruff format src/ tests/

# Clean build artifacts
clean:
	rm -rf .venv/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
```

---

## Installing Packages

### Editable Install
```bash
# Editable install (development)
pip install -e .

# Com optional dependencies
pip install -e ".[dev]"

# Múltiplos extras
pip install -e ".[dev,test,docs]"
```

### Regular Install
```bash
# From PyPI
pip install myproject

# From local directory
pip install .

# From git
pip install git+https://github.com/user/project.git

# From git branch
pip install git+https://github.com/user/project.git@develop

# From local wheel
pip install dist/myproject-1.0.0-py3-none-any.whl
```

### Requirements Files
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Install from multiple files
pip install -r requirements/base.txt -r requirements/dev.txt

# Generate requirements.txt (simple projects)
pip freeze > requirements.txt

# Better: use pip-compile (pip-tools)
pip-compile pyproject.toml -o requirements.txt
```

---

## Building Distributions

### Build Package
```bash
# Install build tool
pip install build

# Build both wheel and sdist
python -m build

# Output:
# dist/
# ├── myproject-0.1.0-py3-none-any.whl
# └── myproject-0.1.0.tar.gz
```

### Wheel vs Source Distribution
```bash
# Wheel (.whl) - binary distribution
# - Faster to install
# - No build step needed
# - Platform-specific (or pure Python)
myproject-1.0.0-py3-none-any.whl
#            │   │    │
#            │   │    └─ platform (any)
#            │   └────── ABI tag (none = pure Python)
#            └────────── Python version (py3 = 3.x)

# Source distribution (.tar.gz)
# - Requires build step
# - Platform-independent
# - Includes source code
myproject-1.0.0.tar.gz
```

### Exemplo do Mundo Real

**Build Script:**
```bash
#!/bin/bash
# build.sh - Build and verify package

set -e

echo "Cleaning old builds..."
rm -rf dist/ build/ *.egg-info/

echo "Building package..."
python -m build

echo "Verifying wheel..."
python -m twine check dist/*

echo "Contents of wheel:"
unzip -l dist/*.whl

echo "Build complete!"
ls -lh dist/
```

---

## Publishing to PyPI

### Setup
```bash
# Install twine
pip install twine

# Configure PyPI credentials
# ~/.pypirc
[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...

[testpypi]
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9yZwI...
```

### Upload to TestPyPI
```bash
# Build
python -m build

# Check distributions
twine check dist/*

# Upload to TestPyPI (test first!)
twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ myproject
```

### Upload to PyPI
```bash
# Upload to PyPI (production)
twine upload dist/*

# Install from PyPI
pip install myproject
```

### Exemplo do Mundo Real

**Release Script:**
```bash
#!/bin/bash
# release.sh - Automated release process

set -e

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: ./release.sh <version>"
    exit 1
fi

echo "Releasing version $VERSION..."

# Update version in pyproject.toml
sed -i "s/^version = .*/version = \"$VERSION\"/" pyproject.toml

# Run tests
pytest tests/

# Build
rm -rf dist/
python -m build

# Verify
twine check dist/*

# Create git tag
git add pyproject.toml
git commit -m "Release v$VERSION"
git tag -a "v$VERSION" -m "Version $VERSION"

# Upload to PyPI
twine upload dist/*

# Push to git
git push origin main
git push origin "v$VERSION"

echo "Released v$VERSION successfully!"
```

---

## Entry Points e CLI

### Console Scripts
```toml
[project.scripts]
myproject = "myproject.cli:main"
my-admin = "myproject.admin:admin_main"
```
```python
# src/myproject/cli.py
import structlog

logger = structlog.get_logger()

def main() -> int:
    """CLI entry point."""
    logger.info("cli_started")
    
    # CLI logic here
    
    logger.info("cli_completed")
    return 0

# Instalado como:
# $ myproject
# $ my-admin
```

### Exemplo do Mundo Real

**CLI with Click:**
```toml
[project]
dependencies = [
    "click>=8.0.0",
    "structlog>=23.0.0",
]

[project.scripts]
myapp = "myapp.cli:cli"
```
```python
# src/myapp/cli.py
import click
import structlog
from pathlib import Path

logger = structlog.get_logger()

@click.group()
@click.version_option()
def cli():
    """My application CLI."""
    pass

@cli.command()
@click.option("--config", type=click.Path(exists=True), help="Config file")
@click.option("--verbose", is_flag=True, help="Verbose output")
def run(config: str | None, verbose: bool):
    """Run the application."""
    logger.info("app_starting", config=config, verbose=verbose)
    
    # Application logic
    
    logger.info("app_completed")

@cli.command()
@click.argument("path", type=click.Path())
def process(path: str):
    """Process files."""
    logger.info("processing_started", path=path)
    
    # Process logic
    
    logger.info("processing_completed")

if __name__ == "__main__":
    cli()
```

---

## Versioning

### Semantic Versioning
```toml
# pyproject.toml
[project]
version = "1.2.3"
#         │ │ │
#         │ │ └─ PATCH (bug fixes)
#         │ └─── MINOR (new features, backwards compatible)
#         └───── MAJOR (breaking changes)

# Examples:
# 1.0.0 - Initial release
# 1.1.0 - Add new feature
# 1.1.1 - Fix bug
# 2.0.0 - Breaking change
```

### Dynamic Versioning
```toml
[project]
dynamic = ["version"]

[tool.hatch.version]
path = "src/myproject/__init__.py"
```
```python
# src/myproject/__init__.py
__version__ = "1.2.3"
```

### Version from Git Tags
```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]

[tool.hatch.version]
source = "vcs"

# Version from git tag: v1.2.3 → 1.2.3
```

---

## Type Hints Support

### py.typed Marker
```
src/
└── myproject/
    ├── __init__.py
    ├── core.py
    └── py.typed  # Empty file - signals type hints
```
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/myproject"]
include = ["src/myproject/py.typed"]
```

### Stub Files
```python
# src/myproject/core.pyi - stub file
def process_data(data: str) -> dict[str, int]: ...

class Processor:
    def __init__(self, config: dict) -> None: ...
    def run(self) -> None: ...
```

---

## Dependency Management Tools

### pip-tools
```bash
# Install
pip install pip-tools

# Create requirements.in
cat > requirements.in <<EOF
fastapi>=0.100.0
pydantic>=2.0.0
EOF

# Compile to requirements.txt (lock file)
pip-compile requirements.in

# Install
pip-sync requirements.txt

# Update all dependencies
pip-compile --upgrade requirements.in
```

### Poetry
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Create project
poetry new myproject

# Add dependency
poetry add fastapi

# Add dev dependency
poetry add --group dev pytest

# Install dependencies
poetry install

# Update dependencies
poetry update

# Lock dependencies
poetry lock
```

### uv (Modern, Fast)
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt

# Install package
uv pip install -e ".[dev]"

# Extremely fast (~10-100x faster than pip)
```

---

## Advanced Patterns

### Namespace Packages
```
project1/
└── src/
    └── company/
        └── project1/
            └── __init__.py

project2/
└── src/
    └── company/
        └── project2/
            └── __init__.py

# Both install to: company.project1, company.project2
```
```python
# Usage
from company.project1 import feature1
from company.project2 import feature2
```

### Plugin Systems
```toml
[project.entry-points."myapp.plugins"]
builtin = "myapp.plugins.builtin:BuiltinPlugin"
```
```python
# Discover plugins
from importlib.metadata import entry_points

plugins = entry_points(group="myapp.plugins")
for plugin in plugins:
    plugin_class = plugin.load()
    instance = plugin_class()
```

---

## Best Practices

✅ **Use src/ layout**
```
project/
└── src/
    └── package/
```

✅ **Use pyproject.toml moderno**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

✅ **Pin versions em produção**
```bash
# Lock dependencies
pip-compile pyproject.toml -o requirements.txt
```

✅ **Use semantic versioning**
```
1.2.3
MAJOR.MINOR.PATCH
```

✅ **Test no TestPyPI primeiro**
```bash
twine upload --repository testpypi dist/*
```

❌ **Não commite virtual environments**
```gitignore
# .gitignore
.venv/
venv/
```

❌ **Não use setup.py para projetos novos**
```toml
# Use pyproject.toml ao invés de setup.py
```

❌ **Não faça hard pin sem motivo**
```toml
# EVITE
dependencies = ["package==1.2.3"]

# PREFIRA
dependencies = ["package>=1.2.0,<2.0.0"]
```

---

## Casos de Uso Estabelecidos

### Library Package
```toml
[project]
name = "mylib"
dependencies = ["requests>=2.28.0"]
```

### CLI Application
```toml
[project.scripts]
mytool = "mytool.cli:main"
```

### Internal Package
```bash
# Install from git
pip install git+https://github.com/company/internal-lib.git
```

### Monorepo
```
monorepo/
├── packages/
│   ├── core/
│   │   └── pyproject.toml
│   ├── api/
│   │   └── pyproject.toml
│   └── cli/
│       └── pyproject.toml
```

---

## Referências

- [Packaging User Guide](https://packaging.python.org/)
- [PEP 517](https://peps.python.org/pep-0517/) - Build System Interface
- [PEP 518](https://peps.python.org/pep-0518/) - pyproject.toml
- [PEP 621](https://peps.python.org/pep-0621/) - Project Metadata
- [Semantic Versioning](https://semver.org/)
- [PyPI](https://pypi.org/)