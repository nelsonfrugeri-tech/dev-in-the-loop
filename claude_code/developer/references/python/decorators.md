# Decorators - Python 3.10+

Referência técnica completa de decorators em Python. Para decisões de quando criar custom decorators, consulte a skill principal (`/developer`).

## Fundamentos

Decorators são funções que modificam comportamento de outras funções ou classes. Implementam o padrão Wrapper, permitindo adicionar funcionalidade sem alterar código original.

**Quando usar:**
- Cross-cutting concerns (logging, caching, auth, metrics)
- Modificar comportamento sem alterar implementação
- DRY - evitar código repetitivo
- Separação de responsabilidades

**Sintaxe:**
```python
@decorator
def function():
    pass

# Equivalente a:
function = decorator(function)
```

---

## Function Decorators Básicos

### Definição
```python
from functools import wraps
from typing import Callable, TypeVar, Any

F = TypeVar("F", bound=Callable[..., Any])

def simple_decorator(func: F) -> F:
    """Decorator básico que preserva metadata."""
    @wraps(func)  # Preserva __name__, __doc__, etc
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Antes da função
        result = func(*args, **kwargs)
        # Depois da função
        return result
    return wrapper  # type: ignore

@simple_decorator
def greet(name: str) -> str:
    """Greets someone."""
    return f"Hello, {name}"

greet("Alice")  # wrapper executa
```

### @wraps Importance
```python
from functools import wraps

# Sem @wraps - perde metadata
def bad_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@bad_decorator
def my_function():
    """Important docstring."""
    pass

print(my_function.__name__)  # "wrapper" (ERRADO)
print(my_function.__doc__)   # None (ERRADO)

# Com @wraps - preserva metadata
def good_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@good_decorator
def my_function():
    """Important docstring."""
    pass

print(my_function.__name__)  # "my_function" (CORRETO)
print(my_function.__doc__)   # "Important docstring." (CORRETO)
```

---

## Decorators com Parâmetros

### Definição

Decorator que aceita argumentos precisa de função adicional:
```python
from functools import wraps
from typing import Callable, TypeVar, Any

T = TypeVar("T")

def repeat(times: int) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Executa função N vezes."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for _ in range(times - 1):
                func(*args, **kwargs)
            return func(*args, **kwargs)  # Retorna última execução
        return wrapper
    return decorator

@repeat(times=3)
def say_hello() -> None:
    print("Hello")

say_hello()  # Executa 3 vezes
```

### Exemplo do Mundo Real

**Retry com Backoff Exponencial:**
```python
from functools import wraps
from typing import Callable, TypeVar, Type
import time
import structlog

logger = structlog.get_logger()

T = TypeVar("T")

def retry(
    max_attempts: int = 3,
    backoff_factor: float = 1.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator com exponential backoff.
    
    Args:
        max_attempts: Número máximo de tentativas
        backoff_factor: Fator de backoff (segundos)
        exceptions: Tupla de exceções para retry
    
    Exemplo:
        @retry(max_attempts=3, backoff_factor=2.0, exceptions=(httpx.HTTPError,))
        async def fetch_data(url: str) -> dict:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts - 1:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempts=max_attempts,
                            error=str(exc)
                        )
                        raise
                    
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        wait_seconds=wait_time,
                        error=str(exc)
                    )
                    time.sleep(wait_time)
            
            raise RuntimeError("Unreachable")
        return wrapper
    return decorator

# Uso
import httpx

@retry(max_attempts=3, backoff_factor=2.0, exceptions=(httpx.HTTPError,))
def fetch_user(user_id: str) -> dict:
    response = httpx.get(f"https://api.example.com/users/{user_id}")
    response.raise_for_status()
    return response.json()
```

---

## Built-in Decorators

### @property

Converte método em atributo computado:
```python
from datetime import datetime

class User:
    def __init__(self, name: str, birth_year: int):
        self.name = name
        self.birth_year = birth_year
    
    @property
    def age(self) -> int:
        """Idade calculada dinamicamente."""
        return datetime.now().year - self.birth_year
    
    @age.setter
    def age(self, value: int) -> None:
        """Permite setar idade (atualiza birth_year)."""
        self.birth_year = datetime.now().year - value

user = User("Alice", 1990)
print(user.age)  # 36 (calculado)
user.age = 30    # Setter
print(user.birth_year)  # 1996 (atualizado)
```

### @cached_property

Property que calcula uma vez e cacheia:
```python
from functools import cached_property
import time

class DataProcessor:
    def __init__(self, data: list[int]):
        self.data = data
    
    @cached_property
    def expensive_calculation(self) -> int:
        """Calcula apenas uma vez."""
        time.sleep(2)  # Simulação
        return sum(x * x for x in self.data)

processor = DataProcessor([1, 2, 3, 4, 5])
print(processor.expensive_calculation)  # Calcula (2s)
print(processor.expensive_calculation)  # Cache (instantâneo)
```

### @staticmethod e @classmethod
```python
class MathUtils:
    pi = 3.14159
    
    @staticmethod
    def add(a: int, b: int) -> int:
        """Método estático - sem acesso a self ou cls."""
        return a + b
    
    @classmethod
    def from_radius(cls, radius: float) -> "Circle":
        """Factory method - acessa cls."""
        return Circle(radius * cls.pi)

# Uso
result = MathUtils.add(2, 3)  # 5
```

### @functools.cache e @lru_cache
```python
from functools import cache, lru_cache

@cache  # Cache ilimitado (Python 3.9+)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

@lru_cache(maxsize=128)  # Cache com limite
def expensive_function(x: int) -> int:
    return x ** 2

# Uso
print(fibonacci(100))  # Rápido devido ao cache
```

---

## Type-Safe Decorators com ParamSpec

### Definição

`ParamSpec` preserva assinatura de função decorada:
```python
from functools import wraps
from typing import Callable, ParamSpec, TypeVar
import structlog

logger = structlog.get_logger()

P = ParamSpec("P")
T = TypeVar("T")

def log_call(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator que preserva assinatura exata."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        logger.info(
            "function_called",
            function=func.__name__,
            args=args,
            kwargs=kwargs
        )
        result = func(*args, **kwargs)
        logger.info(
            "function_returned",
            function=func.__name__,
            result=result
        )
        return result
    return wrapper

@log_call
def divide(a: int, b: int) -> float:
    return a / b

# Type checker sabe assinatura exata
result: float = divide(10, 2)  # OK
# divide("10", "2")  # Type error
```

### Exemplo do Mundo Real

**Async Decorator Type-Safe:**
```python
from functools import wraps
from typing import Callable, ParamSpec, TypeVar, Awaitable
import asyncio
import structlog

logger = structlog.get_logger()

P = ParamSpec("P")
T = TypeVar("T")

def async_retry(
    max_attempts: int = 3
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Retry async function preservando assinatura."""
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(
                        "async_retry_attempt",
                        function=func.__name__,
                        attempt=attempt + 1
                    )
                    await asyncio.sleep(2 ** attempt)
            raise RuntimeError("Unreachable")
        return wrapper
    return decorator

# Uso
import httpx

@async_retry(max_attempts=3)
async def fetch_user(user_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"/users/{user_id}")
        return response.json()

# Type checker preserva assinatura
user: dict = await fetch_user("123")
```

---

## Class Decorators

### Definição

Decorators podem modificar classes inteiras:
```python
from typing import Type, TypeVar

T = TypeVar("T")

def singleton(cls: Type[T]) -> Type[T]:
    """Transforma classe em singleton."""
    instances: dict[Type, Any] = {}
    
    @wraps(cls, updated=())
    def get_instance(*args, **kwargs) -> T:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance  # type: ignore

@singleton
class DatabaseConnection:
    def __init__(self, host: str):
        self.host = host

# Sempre retorna mesma instância
db1 = DatabaseConnection("localhost")
db2 = DatabaseConnection("localhost")
assert db1 is db2  # True
```

### Exemplo do Mundo Real

**Auto-register Classes:**
```python
from typing import Type, Dict

# Registry global
_handlers: Dict[str, Type] = {}

def register_handler(name: str):
    """Registra handler automaticamente."""
    def decorator(cls: Type) -> Type:
        _handlers[name] = cls
        return cls
    return decorator

# Uso
@register_handler("user_created")
class UserCreatedHandler:
    def handle(self, event: dict) -> None:
        # Process event
        pass

@register_handler("order_placed")
class OrderPlacedHandler:
    def handle(self, event: dict) -> None:
        # Process event
        pass

# Dispatch dinâmico
def handle_event(event_type: str, event_data: dict) -> None:
    handler_class = _handlers.get(event_type)
    if handler_class:
        handler = handler_class()
        handler.handle(event_data)
```

**Add Methods to Class:**
```python
from typing import Type, TypeVar, Callable

T = TypeVar("T")

def add_str_method(cls: Type[T]) -> Type[T]:
    """Adiciona método __str__ automaticamente."""
    def __str__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{cls.__name__}({attrs})"
    
    cls.__str__ = __str__  # type: ignore
    return cls

@add_str_method
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

p = Point(1.0, 2.0)
print(p)  # Point(x=1.0, y=2.0)
```

---

## Observability Decorators - Padrão de Produção

### Estrutura de Logging

Use structured logging (structlog), nunca print:
```python
import structlog

logger = structlog.get_logger()
```

### Decorator para Métricas de LLM

Baseado em padrões estabelecidos (observerai, LangChain, LangSmith):
```python
from functools import wraps
from typing import Callable, TypeVar, Any, ParamSpec
from contextvars import ContextVar
import time
import uuid
from datetime import datetime, timezone
import structlog

# Context vars para tracing (thread-safe)
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
span_id_var: ContextVar[str] = ContextVar("span_id", default="")

logger = structlog.get_logger()

P = ParamSpec("P")
T = TypeVar("T")

def metric_chat_create(
    metadata: dict[str, Any] | None = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator para observabilidade de chamadas LLM.
    
    Captura:
    - Latency
    - Token usage
    - Conversation (question/answer)
    - Exceptions
    - Custom metadata
    
    Exemplo:
        @metric_chat_create(metadata={"user_id": "user-123", "session": "abc"})
        def generate_response(prompt: str) -> str:
            return client.chat.completions.create(...)
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Generate span ID
            span_id = str(uuid.uuid4())
            span_id_var.set(span_id)
            
            # Capture start time
            start_time = time.perf_counter()
            exception_info = None
            response = None
            
            try:
                # Execute original function
                response = func(*args, **kwargs)
                status_code = 200
                
            except Exception as exc:
                status_code = 500
                exception_info = {
                    "type": type(exc).__name__,
                    "message": str(exc)
                }
                raise
                
            finally:
                # Calculate latency
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                
                # Extract metrics from response
                token_metrics = None
                conversation = None
                
                if response and hasattr(response, 'usage'):
                    token_metrics = {
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens,
                        "total": response.usage.total_tokens
                    }
                
                if response and hasattr(response, 'choices'):
                    # Extract question from kwargs or args
                    question = None
                    if 'messages' in kwargs:
                        messages = kwargs['messages']
                        if messages and isinstance(messages[-1], dict):
                            question = messages[-1].get('content')
                    
                    # Extract answer
                    answer = None
                    if response.choices:
                        answer = response.choices[0].message.content
                    
                    conversation = {
                        "question": question,
                        "answer": answer
                    }
                
                # Log structured metric
                logger.info(
                    "observerai.openai.completion.chat_create",
                    trace_id=trace_id_var.get(),
                    span_id=span_id,
                    provider="openai",
                    endpoint="/chat/completions",
                    model=kwargs.get('model', 'unknown'),
                    response={
                        "status_code": status_code,
                        "latency": {
                            "time": latency_ms,
                            "unit": "ms"
                        }
                    },
                    token=token_metrics,
                    conversation=conversation,
                    exception=exception_info,
                    metadata=metadata or {},
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            
            return response
        
        return wrapper
    return decorator
```

### Uso em Produção
```python
import uuid
from openai import OpenAI
from contextvars import ContextVar

client = OpenAI()
trace_id_var: ContextVar[str] = ContextVar("trace_id")

# Set trace ID para toda request
trace_id_var.set(str(uuid.uuid4()))

@metric_chat_create(metadata={"user_id": "user-123", "feature": "chat"})
def generate_chat_response(prompt: str) -> str:
    """Generate LLM response com observabilidade completa."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Chamada - métricas logadas automaticamente
answer = generate_chat_response("Qual a capital da França?")
```

### Output (JSON Estruturado)
```json
{
  "event": "observerai.openai.completion.chat_create",
  "trace_id": "fadfd7d6-9150-4327-961f-dad5f048add1",
  "span_id": "c346b100-30d8-4eea-91e3-ddcc67d8d5e0",
  "provider": "openai",
  "endpoint": "/chat/completions",
  "model": "gpt-4",
  "response": {
    "status_code": 200,
    "latency": {"time": 481, "unit": "ms"}
  },
  "token": {
    "prompt": 14,
    "completion": 9,
    "total": 23
  },
  "conversation": {
    "question": "Qual a capital da França?",
    "answer": "A capital da França é Paris."
  },
  "exception": null,
  "metadata": {
    "user_id": "user-123",
    "feature": "chat"
  },
  "timestamp": "2025-02-02T19:21:08.115226Z",
  "level": "info"
}
```

### Tool Calls Support
```python
# Quando LLM usa function calling
conversation = {
    "question": {
        "content": "Como está o clima em São Paulo?",
        "role": "user",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Obtém clima de uma cidade",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        }
                    }
                }
            }
        ]
    },
    "answer": {
        "content": None,
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"city\": \"São Paulo\"}"
                }
            }
        ]
    }
}
```

---

## Performance Decorators

### Timing Decorator
```python
from functools import wraps
from typing import Callable, TypeVar, ParamSpec
import time
import structlog

logger = structlog.get_logger()

P = ParamSpec("P")
T = TypeVar("T")

def measure_time(func: Callable[P, T]) -> Callable[P, T]:
    """Mede tempo de execução de função."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        logger.info(
            "function_timing",
            function=func.__name__,
            duration_ms=round(elapsed * 1000, 2)
        )
        return result
    return wrapper

@measure_time
def expensive_operation() -> list[int]:
    return [i ** 2 for i in range(1_000_000)]
```

### Rate Limiter Decorator
```python
from functools import wraps
from typing import Callable, TypeVar, ParamSpec
import time
from collections import deque
import structlog

logger = structlog.get_logger()

P = ParamSpec("P")
T = TypeVar("T")

class RateLimiter:
    """Rate limiter baseado em sliding window."""
    
    def __init__(self, max_calls: int, window_seconds: float):
        self.max_calls = max_calls
        self.window = window_seconds
        self.calls: deque[float] = deque()
    
    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            now = time.time()
            
            # Remove calls antigas
            while self.calls and self.calls[0] < now - self.window:
                self.calls.popleft()
            
            # Check limit
            if len(self.calls) >= self.max_calls:
                wait_time = self.window - (now - self.calls[0])
                logger.warning(
                    "rate_limit_exceeded",
                    function=func.__name__,
                    wait_seconds=round(wait_time, 2)
                )
                time.sleep(wait_time)
                # Retry
                return wrapper(*args, **kwargs)
            
            # Record call
            self.calls.append(now)
            return func(*args, **kwargs)
        
        return wrapper

# Uso: máximo 10 chamadas por segundo
@RateLimiter(max_calls=10, window_seconds=1.0)
def api_call(endpoint: str) -> dict:
    # Make API request
    return {"status": "ok"}
```

---

## Validation Decorators

### Type Validation
```python
from functools import wraps
from typing import Callable, TypeVar, ParamSpec, get_type_hints

P = ParamSpec("P")
T = TypeVar("T")

def validate_types(func: Callable[P, T]) -> Callable[P, T]:
    """Valida tipos em runtime."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # Get type hints
        hints = get_type_hints(func)
        
        # Validate args
        for arg, (name, expected_type) in zip(args, hints.items()):
            if name != 'return' and not isinstance(arg, expected_type):
                raise TypeError(
                    f"Argument '{name}' must be {expected_type}, "
                    f"got {type(arg)}"
                )
        
        # Execute
        result = func(*args, **kwargs)
        
        # Validate return type
        if 'return' in hints:
            expected_return = hints['return']
            if not isinstance(result, expected_return):
                raise TypeError(
                    f"Return value must be {expected_return}, "
                    f"got {type(result)}"
                )
        
        return result
    return wrapper

@validate_types
def add(a: int, b: int) -> int:
    return a + b

add(1, 2)      # OK
# add("1", "2")  # TypeError em runtime
```

---

## Casos de Uso Estabelecidos

### Observability (observerai, LangSmith, W&B)
```python
@metric_chat_create(metadata={"user_id": "123"})
def llm_call():
    ...
```

### Caching (functools, Redis)
```python
@lru_cache(maxsize=128)
def expensive_computation(n: int) -> int:
    ...
```

### Authentication (FastAPI, Flask)
```python
@require_auth
def protected_endpoint():
    ...
```

### Retry Logic (tenacity, backoff)
```python
@retry(max_attempts=3, backoff_factor=2.0)
def flaky_api_call():
    ...
```

### Rate Limiting (slowapi, ratelimit)
```python
@RateLimiter(max_calls=100, window_seconds=60)
def api_endpoint():
    ...
```

### Timing/Profiling
```python
@measure_time
def performance_critical_function():
    ...
```

### Validation (Pydantic, marshmallow)
```python
@validate_types
def typed_function(x: int) -> str:
    ...
```

---

## Best Practices

✅ **Sempre use @wraps**
```python
from functools import wraps

def my_decorator(func):
    @wraps(func)  # Preserva metadata
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

✅ **Use ParamSpec para type safety**
```python
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

def decorator(func: Callable[P, T]) -> Callable[P, T]:
    ...
```

✅ **Structured logging, nunca print**
```python
import structlog

logger = structlog.get_logger()

def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info("function_called", function=func.__name__)
        return func(*args, **kwargs)
    return wrapper
```

✅ **Preservar exception stacktrace**
```python
def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.error("function_failed", exc_info=True)
            raise  # Re-raise preserva stacktrace
    return wrapper
```

❌ **Não use decorators para lógica de negócio**
```python
# EVITE
@calculate_discount  # Lógica de negócio deve estar explícita
def get_price():
    return 100

# PREFIRA
def get_price():
    price = 100
    return calculate_discount(price)  # Explícito
```

❌ **Não esconda side effects**
```python
# EVITE
@modify_database  # Side effect não óbvio
def calculate():
    return 42

# PREFIRA - seja explícito
def calculate():
    result = 42
    save_to_database(result)  # Óbvio
    return result
```

---

## Referências

- [PEP 318](https://peps.python.org/pep-0318/) - Decorators for Functions and Methods
- [PEP 3129](https://peps.python.org/pep-3129/) - Class Decorators
- [PEP 612](https://peps.python.org/pep-0612/) - Parameter Specification Variables
- [functools Documentation](https://docs.python.org/3/library/functools.html)
- [observerai](https://github.com/nelsonfrugeri-tech/observerai) - LLM Observability