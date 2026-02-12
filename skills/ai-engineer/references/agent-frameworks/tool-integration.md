# Tool Integration - Tool Calling & API Integration

Referência completa para integrar tools em agents: design, implementation, patterns de produção.

---

## O que são Tools para Agents?

### Conceito

**Tools** são funções que um LLM pode invocar para interagir com o mundo externo. O LLM não executa a tool — ele gera um JSON estruturado descrevendo qual tool chamar e com quais parâmetros. O runtime executa e retorna o resultado.

```
User message → LLM decide usar tool → Gera tool call JSON → Runtime executa → Resultado volta ao LLM → LLM gera resposta
```

### Por que Agents Precisam de Tools

Sem tools, LLMs são **read-only** — geram texto, mas não agem. Tools habilitam:

- **Acesso a dados em tempo real** — APIs, databases, file systems
- **Ações no mundo** — enviar email, criar PR, deploy
- **Cálculos precisos** — math, date operations, hashing
- **Integração com sistemas** — CRM, ERP, monitoring

### Tool Use vs RAG vs Fine-tuning

| Critério | Tool Use | RAG | Fine-tuning |
|----------|----------|-----|-------------|
| **Quando usar** | Ações e dados em tempo real | Conhecimento estático grande | Mudar comportamento/estilo |
| **Latência** | Alta (API calls) | Média (vector search) | Baixa (no retrieval) |
| **Dados atualizados** | ✅ Sempre atualizados | ⚠️ Depende do index | ❌ Snapshot do training |
| **Ações no mundo** | ✅ Sim | ❌ Não | ❌ Não |
| **Custo** | Por API call | Embedding + storage | Training compute |
| **Complexidade** | Média | Média | Alta |
| **Melhor para** | Integração com APIs, CRUD | Q&A sobre documentos | Estilo, formato, domínio |

**Regra prática:** Use tools para **ações** e **dados dinâmicos**. RAG para **conhecimento estático**. Fine-tuning para **comportamento**.

---

## Tool Schema Design

### JSON Schema Format (Anthropic)

```python
# Formato Anthropic nativo
weather_tool = {
    "name": "get_weather",
    "description": (
        "Get current weather for a location. Returns temperature, "
        "humidity, and conditions. Use when user asks about weather, "
        "temperature, or outdoor conditions for a specific place."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": (
                    "City name with optional country code "
                    "(e.g., 'São Paulo, BR', 'Tokyo, JP')"
                ),
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units. Default: celsius",
            },
        },
        "required": ["location"],
    },
}
```

### Pydantic Models para Schemas

```python
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class WeatherInput(BaseModel):
    """Get current weather for a location."""

    location: str = Field(
        description=(
            "City name with optional country code "
            "(e.g., 'São Paulo, BR', 'Tokyo, JP')"
        )
    )
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature units",
    )

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Location cannot be empty")
        return v.strip()


class SearchInput(BaseModel):
    """Search documentation for a specific topic."""

    query: str = Field(
        description="Specific topic to search (e.g., 'asyncio.gather', 'context managers')"
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    source: Literal["web", "docs", "code"] = Field(
        default="web",
        description="Source to search in",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 500:
            raise ValueError("Query too long (max 500 chars)")
        return v.strip()


def pydantic_to_anthropic_tool(
    name: str,
    model: type[BaseModel],
    description: str | None = None,
) -> dict[str, Any]:
    """Convert Pydantic model to Anthropic tool format."""
    schema = model.model_json_schema()
    # Remove title e $defs que Pydantic adiciona
    schema.pop("title", None)
    schema.pop("$defs", None)

    return {
        "name": name,
        "description": description or model.__doc__ or "",
        "input_schema": schema,
    }


# Uso
weather_tool = pydantic_to_anthropic_tool(
    "get_weather",
    WeatherInput,
    description="Get current weather. Use for temperature, conditions, forecasts.",
)
```

### Description Quality

Descriptions são **o fator mais crítico** para o agent usar a tool corretamente.

```python
# ❌ Description vaga — agent não sabe quando usar
{
    "name": "search",
    "description": "Search for stuff",
    "input_schema": {"properties": {"q": {"type": "string"}}},
}

# ✅ Description específica — agent entende exatamente quando e como usar
{
    "name": "search_python_docs",
    "description": (
        "Search Python official documentation for specific topics. "
        "Use when user asks about Python built-in functions, standard library, "
        "or language features. Returns relevant doc excerpts with examples. "
        "NOT for third-party packages — use search_pypi for those."
    ),
    "input_schema": {
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Specific Python topic (e.g., 'asyncio.gather', "
                    "'list comprehension syntax', 'dataclass frozen')"
                ),
            },
            "section": {
                "type": "string",
                "enum": ["builtin", "stdlib", "reference", "tutorial"],
                "description": "Documentation section to search",
            },
        },
        "required": ["query"],
    },
}
```

**Regras para boas descriptions:**
1. Diga **o que** a tool faz em uma frase
2. Diga **quando** usar (e quando NÃO usar)
3. Descreva **o que retorna**
4. Parâmetros com **exemplos concretos**

---

## Tool Calling Patterns

### Native Tool Use (Anthropic)

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic()


async def run_with_tools(
    user_message: str,
    tools: list[dict],
    tool_handlers: dict[str, callable],
) -> str:
    """Run conversation with tool use loop."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = await client.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )

        # Se não usou tool, retorna texto
        if response.stop_reason == "end_turn":
            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        # Processa tool calls
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            handler = tool_handlers.get(block.name)
            if not handler:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Unknown tool: {block.name}",
                    "is_error": True,
                })
                continue

            try:
                result = await handler(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })
            except Exception as e:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Error: {e}",
                    "is_error": True,
                })

        messages.append({"role": "user", "content": tool_results})
```

### Parallel vs Sequential Tool Calls

```python
# Anthropic pode retornar múltiplos tool_use blocks em uma resposta
# = parallel tool calls

# Forçar uso de tool específica
response = await client.messages.create(
    model="claude-sonnet-4-5-20250514",
    max_tokens=4096,
    tools=tools,
    tool_choice={"type": "tool", "name": "get_weather"},  # Força esta tool
    messages=messages,
)

# Deixar o model decidir (default)
response = await client.messages.create(
    model="claude-sonnet-4-5-20250514",
    max_tokens=4096,
    tools=tools,
    tool_choice={"type": "auto"},  # Agent decide
    messages=messages,
)

# Forçar que USE alguma tool (qualquer uma)
response = await client.messages.create(
    model="claude-sonnet-4-5-20250514",
    max_tokens=4096,
    tools=tools,
    tool_choice={"type": "any"},  # Deve usar pelo menos uma
    messages=messages,
)
```

**Quando forçar tool use:**
- `tool_choice: auto` — Default, agent decide (maioria dos casos)
- `tool_choice: any` — Primeiro turno quando sabe que precisa de tool
- `tool_choice: {name}` — Workflow determinístico, sabe exatamente qual tool

---

## API Integration as Tools

### REST API Wrapper

```python
from typing import Any
from dataclasses import dataclass
import aiohttp
import asyncio
from pydantic import BaseModel


@dataclass
class APIConfig:
    """API connection configuration."""

    base_url: str
    api_key: str | None = None
    timeout: float = 30.0
    max_retries: int = 3
    headers: dict[str, str] | None = None


class APITool:
    """Base class for API-backed tools."""

    def __init__(self, config: APIConfig) -> None:
        self.config = config
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "APITool":
        headers = dict(self.config.headers or {})
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        self._session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._session:
            await self._session.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request with timeout and error handling."""
        if not self._session:
            raise RuntimeError("Use as context manager: async with APITool(config)")

        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        try:
            async with asyncio.timeout(self.config.timeout):
                async with self._session.request(
                    method, url, json=data, params=params
                ) as resp:
                    body = await resp.text()
                    if resp.status >= 400:
                        raise APIError(resp.status, body)
                    return await resp.json()

        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Request to {method} {url} timed out after {self.config.timeout}s"
            )


class APIError(Exception):
    """API returned an error status."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body[:200]}")
```

### Rate Limiting

```python
import asyncio
import time
from collections import deque


class RateLimiter:
    """Token bucket rate limiter para API calls."""

    def __init__(self, requests_per_second: float = 10.0) -> None:
        self._rate = requests_per_second
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until rate limit allows another request."""
        async with self._lock:
            now = time.monotonic()
            # Remove timestamps mais velhos que 1 segundo
            while self._timestamps and now - self._timestamps[0] > 1.0:
                self._timestamps.popleft()

            if len(self._timestamps) >= self._rate:
                wait = 1.0 - (now - self._timestamps[0])
                if wait > 0:
                    await asyncio.sleep(wait)

            self._timestamps.append(time.monotonic())


class RateLimitedAPITool(APITool):
    """API tool com rate limiting."""

    def __init__(
        self,
        config: APIConfig,
        requests_per_second: float = 10.0,
    ) -> None:
        super().__init__(config)
        self._limiter = RateLimiter(requests_per_second)

    async def _request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        await self._limiter.acquire()
        return await super()._request(*args, **kwargs)
```

### Authentication Patterns

```python
import os
from abc import ABC, abstractmethod


class AuthStrategy(ABC):
    """Base authentication strategy."""

    @abstractmethod
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        """Apply auth to request headers."""


class BearerAuth(AuthStrategy):
    def __init__(self, token: str) -> None:
        self._token = token

    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        headers["Authorization"] = f"Bearer {self._token}"
        return headers


class APIKeyAuth(AuthStrategy):
    def __init__(self, key: str, header_name: str = "X-API-Key") -> None:
        self._key = key
        self._header_name = header_name

    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        headers[self._header_name] = self._key
        return headers


# Uso
auth = BearerAuth(os.environ["API_TOKEN"])
# ou
auth = APIKeyAuth(os.environ["API_KEY"], header_name="X-Api-Key")
```

---

## Error Handling em Tools

### Estratégia Completa

```python
import asyncio
import logging
from typing import Any, TypeVar, Callable, Awaitable
from pydantic import ValidationError

logger = logging.getLogger(__name__)
T = TypeVar("T")


class ToolError(Exception):
    """Base error for tool execution."""

    def __init__(self, message: str, retryable: bool = False) -> None:
        self.retryable = retryable
        super().__init__(message)


class ToolValidationError(ToolError):
    """Input validation failed — do NOT retry."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class ToolAPIError(ToolError):
    """External API error — may be retryable."""

    def __init__(self, status: int, message: str) -> None:
        retryable = status in {429, 500, 502, 503, 504}
        self.status = status
        super().__init__(f"HTTP {status}: {message}", retryable=retryable)


class ToolTimeoutError(ToolError):
    """Timeout — retryable."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


async def execute_with_retry(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> T:
    """Execute async function with exponential backoff retry.

    Retries only on retryable errors. Validation errors fail immediately.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            return await fn(*args, **kwargs)

        except ToolError as e:
            if not e.retryable:
                raise
            last_error = e
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Tool call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, max_retries, delay, e,
            )
            await asyncio.sleep(delay)

        except asyncio.TimeoutError:
            last_error = ToolTimeoutError("Operation timed out")
            delay = base_delay * (2 ** attempt)
            await asyncio.sleep(delay)

        except ValidationError as e:
            raise ToolValidationError(str(e))

    raise last_error or RuntimeError("All retries exhausted")
```

### Fallback Tools

```python
async def execute_with_fallback(
    primary: Callable[..., Awaitable[T]],
    fallback: Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Try primary tool, fall back if it fails."""
    try:
        return await primary(*args, **kwargs)
    except Exception as e:
        logger.warning("Primary tool failed (%s), using fallback", e)
        return await fallback(*args, **kwargs)


# Exemplo: search com fallback
async def search_primary(query: str) -> list[dict]:
    """Search via premium API."""
    ...

async def search_fallback(query: str) -> list[dict]:
    """Search via free API (lower quality)."""
    ...

results = await execute_with_fallback(search_primary, search_fallback, query="python asyncio")
```

---

## Tool Selection Strategies

### Tool Description Best Practices

```python
# ❌ Agent não sabe quando usar cada tool
tools_bad = [
    {"name": "search", "description": "Search things"},
    {"name": "lookup", "description": "Look up information"},
]

# ✅ Agent sabe exatamente quando usar cada tool
tools_good = [
    {
        "name": "search_web",
        "description": (
            "Search the web for current information. Use for: recent events, "
            "current prices, live data. Do NOT use for: historical facts, "
            "documentation (use search_docs instead)."
        ),
    },
    {
        "name": "search_docs",
        "description": (
            "Search internal documentation and knowledge base. Use for: "
            "company policies, product specs, API docs. Returns exact matches "
            "from verified sources."
        ),
    },
]
```

### Tool Categorization via System Prompt

```python
SYSTEM_PROMPT = """You have access to the following tool categories:

**Data Retrieval:**
- search_web: Current web information
- search_docs: Internal documentation
- query_database: SQL queries on product database

**Actions:**
- send_email: Send email to a recipient
- create_ticket: Create support ticket in Jira

**Calculations:**
- calculate: Math expressions
- convert_units: Unit conversion

Rules:
- ALWAYS search before answering factual questions
- NEVER send_email without user confirmation
- Use query_database for anything about products, orders, users
"""
```

---

## Caching Tool Results

### Quando Cachear

| Tool Type | Cachear? | Motivo |
|-----------|----------|--------|
| Weather (current) | ✅ TTL curto (5min) | Muda devagar, API cara |
| Search results | ✅ TTL médio (1h) | Resultados estáveis |
| Database query | ⚠️ Depende | TTL curto se dados mudam |
| Send email | ❌ Nunca | Side effect, não idempotente |
| Calculator | ✅ TTL longo | Determinístico |

### Implementation com Redis

```python
import hashlib
import json
from typing import Any, Callable, Awaitable
from redis.asyncio import Redis


class ToolCache:
    """Cache for tool results using Redis."""

    def __init__(self, redis: Redis, default_ttl: int = 300) -> None:
        self._redis = redis
        self._default_ttl = default_ttl

    def _cache_key(self, tool_name: str, inputs: dict) -> str:
        """Generate deterministic cache key."""
        input_str = json.dumps(inputs, sort_keys=True)
        hash_val = hashlib.sha256(input_str.encode()).hexdigest()[:16]
        return f"tool:{tool_name}:{hash_val}"

    async def get_or_execute(
        self,
        tool_name: str,
        inputs: dict[str, Any],
        executor: Callable[..., Awaitable[Any]],
        ttl: int | None = None,
    ) -> Any:
        """Get cached result or execute tool and cache."""
        key = self._cache_key(tool_name, inputs)

        # Check cache
        cached = await self._redis.get(key)
        if cached is not None:
            return json.loads(cached)

        # Execute and cache
        result = await executor(**inputs)
        await self._redis.set(
            key,
            json.dumps(result),
            ex=ttl or self._default_ttl,
        )
        return result


# Uso
cache = ToolCache(Redis.from_url("redis://localhost:6379"))

result = await cache.get_or_execute(
    tool_name="get_weather",
    inputs={"location": "São Paulo"},
    executor=weather_tool.get_weather,
    ttl=300,  # 5 minutos
)
```

---

## Advanced Patterns

### Composite Tools

```python
class CompositeSearchTool:
    """Tool that combines multiple search sources."""

    def __init__(
        self,
        web_search: Callable,
        doc_search: Callable,
        db_search: Callable,
    ) -> None:
        self._web = web_search
        self._docs = doc_search
        self._db = db_search

    async def search_all(self, query: str) -> dict[str, Any]:
        """Search all sources in parallel and merge results."""
        web_task = asyncio.create_task(self._safe_call(self._web, query))
        doc_task = asyncio.create_task(self._safe_call(self._docs, query))
        db_task = asyncio.create_task(self._safe_call(self._db, query))

        web, docs, db = await asyncio.gather(web_task, doc_task, db_task)

        return {
            "web_results": web or [],
            "doc_results": docs or [],
            "db_results": db or [],
        }

    async def _safe_call(
        self, fn: Callable, query: str
    ) -> list[dict] | None:
        try:
            return await fn(query)
        except Exception as e:
            logger.warning("Search source failed: %s", e)
            return None

    def to_anthropic_tool(self) -> dict:
        return {
            "name": "search_all",
            "description": (
                "Search across web, internal docs, and database simultaneously. "
                "Use when user needs comprehensive information from multiple sources."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    }
                },
                "required": ["query"],
            },
        }
```

### Tool Observability

```python
import time
import logging
import structlog
from typing import Any, Callable, Awaitable
from dataclasses import dataclass, field

logger = structlog.get_logger()


@dataclass
class ToolMetrics:
    """Metrics for a single tool call."""

    tool_name: str
    duration_ms: float
    success: bool
    error: str | None = None
    input_keys: list[str] = field(default_factory=list)


class ObservableToolExecutor:
    """Wraps tool execution with logging and metrics."""

    def __init__(self) -> None:
        self._metrics: list[ToolMetrics] = []

    async def execute(
        self,
        tool_name: str,
        handler: Callable[..., Awaitable[Any]],
        inputs: dict[str, Any],
    ) -> Any:
        """Execute tool with observability."""
        start = time.perf_counter()

        try:
            result = await handler(**inputs)
            duration = (time.perf_counter() - start) * 1000

            metric = ToolMetrics(
                tool_name=tool_name,
                duration_ms=duration,
                success=True,
                input_keys=list(inputs.keys()),
            )
            self._metrics.append(metric)

            logger.info(
                "tool_call_success",
                tool=tool_name,
                duration_ms=round(duration, 1),
            )
            return result

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000

            metric = ToolMetrics(
                tool_name=tool_name,
                duration_ms=duration,
                success=False,
                error=str(e),
                input_keys=list(inputs.keys()),
            )
            self._metrics.append(metric)

            logger.error(
                "tool_call_failed",
                tool=tool_name,
                duration_ms=round(duration, 1),
                error=str(e),
            )
            raise
```

### Tool Permissions

```python
from enum import StrEnum


class ToolPermission(StrEnum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


class ToolPermissionGuard:
    """Check permissions before tool execution."""

    def __init__(self, user_permissions: set[ToolPermission]) -> None:
        self._permissions = user_permissions

    def check(self, required: ToolPermission) -> bool:
        return required in self._permissions

    def guard(self, required: ToolPermission) -> None:
        if not self.check(required):
            raise PermissionError(
                f"Permission '{required}' required but user has: {self._permissions}"
            )


# Tool registry com permissions
TOOL_PERMISSIONS: dict[str, ToolPermission] = {
    "search_docs": ToolPermission.READ,
    "query_database": ToolPermission.READ,
    "send_email": ToolPermission.WRITE,
    "create_ticket": ToolPermission.WRITE,
    "delete_user": ToolPermission.DELETE,
    "modify_config": ToolPermission.ADMIN,
}
```

---

## Best Practices

### 1. Schema Claro e Descritivo

```python
# ❌ Errado — agent não entende quando usar
{"name": "search", "description": "Search for stuff"}

# ✅ Correto — agent sabe exatamente o que, quando, e como
{
    "name": "search_python_docs",
    "description": (
        "Search Python official documentation. Use for questions about "
        "built-in functions, standard library, or language features. "
        "NOT for third-party packages."
    ),
}
```

**Por quê:** Agent decide qual tool usar baseado na description. Vaga → usa errado.

### 2. Validation de Inputs com Pydantic

```python
# ❌ Errado — aceita qualquer input
async def search(query: str) -> list:
    return await api.search(query)

# ✅ Correto — valida antes de executar
class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=10, ge=1, le=100)

async def search(input: SearchInput) -> list:
    return await api.search(input.query, limit=input.max_results)
```

### 3. Error Messages Úteis para o Agent

```python
# ❌ Errado — agent não sabe como corrigir
raise Exception("Error")

# ✅ Correto — agent pode tentar de novo com input correto
raise ToolValidationError(
    "Invalid date format '2024-13-01'. Use ISO format: YYYY-MM-DD. "
    "Month must be 01-12."
)
```

### 4. Tools Idempotentes

```python
# ❌ Errado — duplica ação se agent chama 2x
async def add_item(name: str) -> dict:
    return await db.insert({"name": name})

# ✅ Correto — idempotente via upsert
async def add_item(name: str) -> dict:
    return await db.upsert({"name": name}, key="name")
```

### 5. Timeouts em Todo I/O

```python
# ❌ Errado — pode travar para sempre
result = await api.call(endpoint)

# ✅ Correto — timeout explícito
async with asyncio.timeout(30):
    result = await api.call(endpoint)
```

### 6. Retornar Dados Estruturados

```python
# ❌ Errado — LLM precisa parsear texto livre
return "The weather in Tokyo is 22 degrees celsius and sunny"

# ✅ Correto — estruturado, fácil para LLM processar
return {
    "location": "Tokyo",
    "temperature": 22,
    "units": "celsius",
    "condition": "sunny",
}
```

### 7. Limitar Tamanho do Resultado

```python
# ❌ Errado — pode retornar 10MB de dados
return await db.query("SELECT * FROM logs")

# ✅ Correto — limita resultado
results = await db.query("SELECT * FROM logs LIMIT 50")
return {
    "results": results,
    "total_count": await db.count("logs"),
    "truncated": True,
}
```

---

## Common Pitfalls

### 1. Tool descriptions vagas

**Problema:** Agent não usa a tool certa ou não usa nenhuma.
**Solução:** Descriptions com: o que faz, quando usar, quando NÃO usar, exemplos de input.

### 2. Sem error handling

**Problema:** Exceção não tratada mata o agent loop.
**Solução:** Sempre retornar `is_error: true` com mensagem útil no tool result.

### 3. Sem timeouts

**Problema:** API lenta trava o agent indefinidamente.
**Solução:** `asyncio.timeout()` em toda operação I/O.

### 4. Resultados muito grandes

**Problema:** Retornar documentos inteiros estoura o context window.
**Solução:** Limitar resultados, paginar, retornar summaries.

### 5. Tools com side effects sem confirmação

**Problema:** Agent envia email ou deleta dados sem pedir confirmação.
**Solução:** Para ações destrutivas, retornar preview e exigir confirmação explícita.

---

## Troubleshooting

### Agent não usa a tool

1. **Verifique a description** — está clara o suficiente?
2. **Verifique tool_choice** — está como `auto`?
3. **Verifique o system prompt** — instrui o agent a usar tools?
4. **Teste com `tool_choice: any`** — se funciona, o problema é na description

### Tool execution falha silenciosamente

1. **Retorne `is_error: true`** no tool result para o LLM saber que falhou
2. **Inclua mensagem de erro** específica para o LLM tentar corrigir
3. **Adicione logging** — `structlog` com tool name, inputs, duration, error

### Timeout errors frequentes

1. **Aumente timeout** se API é legitimamente lenta
2. **Adicione cache** para evitar chamadas repetidas
3. **Use fallback tool** com timeout menor
4. **Verifique rate limits** — pode estar sendo throttled

### Rate limit errors

1. **Implemente rate limiter** no client (veja seção acima)
2. **Exponential backoff** com jitter nos retries
3. **Cache agressivo** para reduzir chamadas
4. **Verifique quota** — pode precisar de plano maior

---

## References

### External
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Pydantic V2 Docs](https://docs.pydantic.dev/latest/)
- [aiohttp Documentation](https://docs.aiohttp.org/)
- [structlog Documentation](https://www.structlog.org/)

### Internal (ai-engineer skill)
- [LangChain](./langchain.md) — Framework-level tool integration
- [LangGraph](./langgraph.md) — Tools em state machines
- [Custom Agents](./custom-agents.md) — Custom agent loops com tools
- [Multi-Agent Systems](./multi-agent.md) — Tools compartilhadas entre agents
- [Anthropic SDK](../llm-integration/anthropic-sdk.md) — SDK nativo para tool use

### Internal (arch-py skill)
- [Async Patterns](../../arch-py/references/python/async-patterns.md)
- [Error Handling](../../arch-py/references/python/error-handling.md)
- [Type System](../../arch-py/references/python/type-system.md)
