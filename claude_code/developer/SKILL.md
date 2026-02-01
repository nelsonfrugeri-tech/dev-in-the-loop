---
name: developer
description: |
  Skill para desenvolvimento Python moderno e universal - serve para scripts, CLIs, APIs web, pipelines de dados, qualquer projeto Python.
  Cobre: type system avançado, async/await, Pydantic, testing, error handling, logging estruturado, e melhores práticas estado da arte.
  Use quando: (1) Desenvolver código Python, (2) Analisar/revisar código, (3) Aplicar padrões e arquitetura.
  Triggers: /developer, /dev, desenvolvimento Python, code quality, best practices.
---

# Developer Skill - Python Modern Best Practices

## Padrão de Conversa

### Princípios de Comunicação

**Verificabilidade e Transparência:**
- Nunca apresente conteúdo gerado, inferido, especulado ou deduzido como fato.
- Se você não pode verificar algo diretamente, diga claramente:
  - "Não posso verificar isso."
  - "Não tenho acesso a essa informação."
  - "Minha base de conhecimento não contém isso."

**Rotulação de Conteúdo Não Verificado:**
- Rotule conteúdo não verificado no início da sentença usando:
  - `[Inference]` - Para inferências baseadas em padrões
  - `[Speculation]` - Para especulação ou hipóteses
  - `[Unverified]` - Para informações que não podem ser confirmadas
- Se qualquer parte da resposta for não verificada, rotule a resposta inteira.

**Esclarecimentos:**
- Peça esclarecimentos se houver informação faltando.
- Não adivinhe ou preencha lacunas por conta própria.
- Não parafraseie nem reinterprete o input do usuário a menos que solicitado.

**Afirmações sobre LLMs:**
- Para afirmações sobre comportamento de LLM (incluindo o próprio), inclua `[Inference]` ou `[Unverified]`.
- Adicione nota indicando que é baseado em padrões observados.

**Correções:**
- Se quebrar esta diretiva, reconheça imediatamente:
  - "Correção: Eu anteriormente fiz uma afirmação não verificada. Isso estava incorreto e deveria ter sido rotulado."

**Preservação de Input:**
- Nunca altere ou modifique o input do usuário a menos que explicitamente solicitado.

---

## Princípios Fundamentais

**Qualidade de Código:**
- Use código estado da arte com padrões de alto nível.
- Pense profundamente sobre como resolver os problemas.
- Adote uma abordagem cética e questionadora.

**Decomposição de Problemas:**
- Entenda o problema como um todo.
- Avalie se faz sentido quebrar em partes menores.
- Proponha essa quebra quando necessário, explicando o raciocínio.

**Idioma:**
- Escreva código e comentários sempre em inglês.
- Documentação técnica e nomes de variáveis em inglês.
- Discussões e explicações podem ser em português quando solicitado.

---

## Padrões Python Básicos

### Tipagem Explícita

Utilize sempre tipagem explícita em todas as variáveis e funções:
```python
from typing import Optional

def process_items(items: list[str], limit: int = 10) -> dict[str, int]:
    return {item: len(item) for item in items[:limit]}
```

### Formatação Black

Formate todo código no estilo da biblioteca `black`:
- Linha máxima de 88 caracteres (padrão black)
- Use aspas duplas para strings
- Trailing commas em estruturas multi-linha
```python
def long_function_name(
    parameter_one: str,
    parameter_two: int,
    parameter_three: list[str] | None = None,
) -> dict[str, Any]:
    """Function with properly formatted parameters."""
    pass
```

---

## Conceitos Python Modernos - Overview

Visão geral de cada conceito Python estado da arte. Para detalhes e exemplos avançados, consulte os arquivos de referência indicados.

### 1. Type System Avançado
**Quando usar:** Contratos claros, duck typing estrutural, tipos genéricos reutilizáveis.

Python 3.10+ oferece: `Protocol` (duck typing estrutural), `TypeVar`/`Generic` (código genérico), `Literal` (valores específicos), `TypedDict` (dicts tipados), `X | Y` (union moderna).
```python
from typing import Protocol, TypeVar, Generic, Literal

class Readable(Protocol):
    def read(self) -> str: ...

T = TypeVar("T")

class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

Status = Literal["pending", "done", "failed"]

def process(status: Status, data: str | None = None) -> str:
    return f"{status}: {data or 'empty'}"
```

**Referência:** [references/python/type-system.md](references/python/type-system.md)

---

### 2. Async/Await
**Quando usar:** I/O-bound operations (APIs, databases, network, files).

Asyncio permite I/O concorrente sem bloquear: `async def` (coroutines), `await` (aguarda operação), `asyncio.gather()` (paraleliza).
```python
import asyncio
import httpx

async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

async def fetch_all(urls: list[str]) -> list[dict]:
    return await asyncio.gather(*[fetch_data(url) for url in urls])
```

**Referência:** [references/python/async-patterns.md](references/python/async-patterns.md)

---

### 3. Data Classes
**Quando usar:** Estruturas de dados com representação clara, comparação, imutabilidade opcional.

Elimina boilerplate: `@dataclass` gera `__init__`, `__repr__`, `__eq__`; `frozen=True` (imutável); `slots=True` (menos memória).
```python
from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float
    label: str = ""
    tags: list[str] = field(default_factory=list)

    def distance_from_origin(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5
```

**Referência:** [references/python/dataclasses.md](references/python/dataclasses.md)

---

### 4. Context Managers
**Quando usar:** Gerenciamento de recursos (arquivos, conexões, locks), setup/teardown, transações.

Garantem cleanup mesmo com exceções: `with` (uso), `@contextmanager` (criação simples), `__enter__/__exit__` (manual).
```python
from contextlib import contextmanager
from typing import Iterator

@contextmanager
def managed_connection(host: str) -> Iterator[Connection]:
    conn = Connection(host)
    try:
        conn.open()
        yield conn
    finally:
        conn.close()

with managed_connection("localhost") as conn:
    conn.execute("SELECT 1")
```

**Referência:** [references/python/context-managers.md](references/python/context-managers.md)

---

### 5. Decorators
**Quando usar:** Cross-cutting concerns (logging, caching, auth), modificar comportamento sem alterar código.

Funções que envolvem outras: `@functools.cache` (memoização), `@property` (atributos computados), custom decorators.
```python
from functools import wraps, cache
from typing import Callable, TypeVar

T = TypeVar("T")

def retry(times: int = 3) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == times - 1:
                        raise
            raise RuntimeError("Unreachable")
        return wrapper
    return decorator
```

**Referência:** [references/python/decorators.md](references/python/decorators.md)

---

### 6. Pydantic v2
**Quando usar:** Validação de dados externos (APIs, configs, arquivos), serialização, schemas documentados.

Padrão para validação: `@field_validator` (validação custom), `@computed_field` (campos derivados), integração FastAPI.
```python
from pydantic import BaseModel, field_validator, computed_field

class User(BaseModel):
    name: str
    email: str
    age: int

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email")
        return v.lower()

    @computed_field
    @property
    def is_adult(self) -> bool:
        return self.age >= 18
```

**Referência:** [references/python/pydantic.md](references/python/pydantic.md)

---

### 7. Error Handling
**Quando usar:** Sempre; crie hierarquias para erros de domínio específicos.

Exceções bem estruturadas: hierarquia customizada, mensagens claras, use para casos excepcionais (não fluxo normal).
```python
class AppError(Exception):
    """Base exception for application errors."""

class ValidationError(AppError):
    """Raised when data validation fails."""

class NotFoundError(AppError):
    """Raised when requested resource is not found."""
    
    def __init__(self, resource: str, id: str) -> None:
        super().__init__(f"{resource} with id '{id}' not found")
        self.resource = resource
        self.id = id
```

**Referência:** [references/python/error-handling.md](references/python/error-handling.md)

---

### 8. Testing com Pytest
**Quando usar:** Sempre. Testes são parte integral do desenvolvimento.

Framework padrão: fixtures (setup reutilizável), `@pytest.mark.parametrize` (múltiplos casos), assertions simples.
```python
import pytest

@pytest.fixture
def premium_user() -> User:
    return User(name="Test", tier="premium")

@pytest.mark.parametrize("price,expected", [(100.0, 90.0), (50.0, 45.0), (0.0, 0.0)])
def test_discount(premium_user: User, price: float, expected: float) -> None:
    result = calculate_discount(premium_user, price)
    assert result == pytest.approx(expected)
```

**Referência:** [references/testing/pytest.md](references/testing/pytest.md)

---

### 9. Logging Estruturado
**Quando usar:** Produção e debugging; essencial para observabilidade.

Structlog produz logs JSON: contexto automático, processadores customizáveis, integração com logging padrão.
```python
import structlog

logger = structlog.get_logger()

def process_order(order_id: str, user_id: str) -> None:
    log = logger.bind(order_id=order_id, user_id=user_id)
    log.info("processing_started")
    try:
        # ... process ...
        log.info("processing_completed", status="success")
    except Exception as e:
        log.error("processing_failed", error=str(e))
        raise
```

**Referência:** [references/python/logging.md](references/python/logging.md)

---

### 10. Configuration Management
**Quando usar:** Qualquer app que precise de configuração externa (envvars, arquivos).

Pydantic-settings: carrega de env vars, validação integrada, suporte a .env e secrets.
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env")
    
    database_url: str
    debug: bool = False
    max_connections: int = 10

settings = Settings()  # Loads from APP_DATABASE_URL, etc.
```

**Referência:** [references/python/configuration.md](references/python/configuration.md)

---

### 11. Generators e Lazy Evaluation
**Quando usar:** Grandes volumes de dados, pipelines de transformação, economia de memória.

Valores sob demanda: `yield` (produz incrementalmente), generator expressions, `itertools`.
```python
from typing import Iterator
from pathlib import Path

def read_chunks(path: Path, size: int = 8192) -> Iterator[str]:
    with open(path) as f:
        while chunk := f.read(size):
            yield chunk

def process_lines(path: Path) -> Iterator[str]:
    for chunk in read_chunks(path):
        for line in chunk.splitlines():
            if line.strip():
                yield line.upper()
```

**Referência:** [references/python/generators.md](references/python/generators.md)

---

### 12. Concurrency
**Quando usar:** Escolha baseada no tipo de workload - I/O-bound vs CPU-bound.

Três modelos: `asyncio` (I/O, single-thread, cooperativo), `threading` (I/O, multi-thread, GIL), `multiprocessing` (CPU, bypassa GIL).
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# asyncio for I/O-bound (preferred)
results = await asyncio.gather(*tasks)

# threading for legacy I/O-bound
with ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(io_func, items))

# multiprocessing for CPU-bound
with ProcessPoolExecutor() as ex:
    results = list(ex.map(cpu_func, items))
```

**Referência:** [references/python/concurrency.md](references/python/concurrency.md)

---

### 13. Packaging Moderno
**Quando usar:** Projetos distribuídos ou com dependências gerenciadas.

`pyproject.toml` (PEP 621): substitui setup.py/setup.cfg, config centralizada, build systems modernos.
```toml
[project]
name = "mypackage"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["pydantic>=2.0", "httpx>=0.24"]

[project.optional-dependencies]
dev = ["pytest>=7.0", "ruff>=0.1"]

[tool.ruff]
line-length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Referência:** [references/python/packaging.md](references/python/packaging.md)

---

## Ferramentas Essenciais

| Categoria | Ferramenta | Propósito | Comando |
|-----------|------------|-----------|---------|
| Lint | **ruff** | Linter ultrarrápido | `ruff check . --fix` |
| Format | **black** | Formatador opinado | `black .` |
| Types | **mypy** | Type checker | `mypy src/` |
| Test | **pytest** | Framework de testes | `pytest` |
| Coverage | **pytest-cov** | Cobertura | `pytest --cov=src` |
| Hooks | **pre-commit** | Git hooks | `pre-commit install` |

**Referência:** [references/tooling/setup.md](references/tooling/setup.md)

---

## Workflow Recomendado
```
PLANEJAR → TIPAR → IMPLEMENTAR → TESTAR → VALIDAR → REVISAR
```

1. **Planejar**: Entenda problema, defina interfaces
2. **Tipar**: Type hints antes da implementação
3. **Implementar**: Código seguindo os tipos
4. **Testar**: pytest + fixtures
5. **Validar**: ruff/black/mypy (via pre-commit)
6. **Revisar**: Code review focado em clareza

---

## Referências por Domínio

### Python Core
- [references/python/type-system.md](references/python/type-system.md) - Protocol, TypeVar, Generic, Literal
- [references/python/async-patterns.md](references/python/async-patterns.md) - Async/await avançado
- [references/python/dataclasses.md](references/python/dataclasses.md) - Dataclasses em profundidade
- [references/python/context-managers.md](references/python/context-managers.md) - Context managers
- [references/python/decorators.md](references/python/decorators.md) - Decorators avançados
- [references/python/generators.md](references/python/generators.md) - Generators e iteradores
- [references/python/concurrency.md](references/python/concurrency.md) - Threading, multiprocessing, asyncio
- [references/python/error-handling.md](references/python/error-handling.md) - Exceções e error handling
- [references/python/packaging.md](references/python/packaging.md) - pyproject.toml e packaging

### Frameworks e Bibliotecas
- [references/python/pydantic.md](references/python/pydantic.md) - Pydantic v2 completo
- [references/python/configuration.md](references/python/configuration.md) - Pydantic-settings
- [references/python/logging.md](references/python/logging.md) - Structlog
- [references/fastapi/best-practices.md](references/fastapi/best-practices.md) - FastAPI patterns

### Testing
- [references/testing/pytest.md](references/testing/pytest.md) - Pytest completo
- [references/testing/fixtures.md](references/testing/fixtures.md) - Fixtures avançadas
- [references/testing/mocking.md](references/testing/mocking.md) - Mocking e patching

### Arquitetura
- [references/architecture/clean-architecture.md](references/architecture/clean-architecture.md) - Clean Architecture
- [references/architecture/dependency-injection.md](references/architecture/dependency-injection.md) - DI patterns
- [references/architecture/repository-pattern.md](references/architecture/repository-pattern.md) - Repository pattern

### Tooling
- [references/tooling/setup.md](references/tooling/setup.md) - Setup completo de ferramentas
- [references/tooling/pre-commit.md](references/tooling/pre-commit.md) - Pre-commit hooks
- [references/tooling/makefile.md](references/tooling/makefile.md) - Makefile para Python