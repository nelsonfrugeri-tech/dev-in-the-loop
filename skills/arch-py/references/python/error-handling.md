# Error Handling - Python 3.10+

Referência técnica completa de error handling em Python. Para decisões de quando criar exceções customizadas vs usar built-in, consulte a skill principal (`/developer`).

## Fundamentos

Error handling em Python usa exceções para sinalizar e tratar erros. Exceções bem estruturadas:
- Comunicam erros claramente
- Permitem recuperação apropriada
- Facilitam debugging
- Documentam casos de erro

**Quando criar exceções customizadas:**
- Erros de domínio específicos
- Hierarchia de erros relacionados
- Contexto adicional necessário
- Recuperação diferenciada por tipo

**Quando usar exceções built-in:**
- Erros genéricos (ValueError, TypeError)
- Erros de sistema (IOError, OSError)
- Não adiciona valor criar custom

---

## Exceções Built-in

### Exceções Comuns
```python
# ValueError - valor inválido
def set_age(age: int) -> None:
    if age < 0 or age > 150:
        raise ValueError(f"Invalid age: {age}")

# TypeError - tipo incorreto
def process_data(data: list) -> None:
    if not isinstance(data, list):
        raise TypeError(f"Expected list, got {type(data).__name__}")

# KeyError - chave não existe
user = {"name": "Alice"}
email = user["email"]  # KeyError: 'email'

# IndexError - índice fora do range
items = [1, 2, 3]
item = items[10]  # IndexError: list index out of range

# AttributeError - atributo não existe
user.missing_attr  # AttributeError

# FileNotFoundError - arquivo não existe
with open("missing.txt") as f:
    pass  # FileNotFoundError

# RuntimeError - erro genérico em runtime
raise RuntimeError("Something went wrong")
```

### Quando Usar Built-in
```python
from typing import Any

def validate_email(email: str) -> None:
    """Valida email - usa ValueError built-in."""
    if "@" not in email:
        raise ValueError(f"Invalid email format: {email}")

def divide(a: float, b: float) -> float:
    """Divide numbers - usa ZeroDivisionError built-in."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

def get_config(key: str, config: dict[str, Any]) -> Any:
    """Get config - usa KeyError built-in."""
    try:
        return config[key]
    except KeyError:
        raise KeyError(f"Configuration key '{key}' not found")
```

---

## Hierarquia de Exceções Customizadas

### Estrutura Básica
```python
class AppError(Exception):
    """Base exception para todas exceções da aplicação."""
    pass

class ValidationError(AppError):
    """Erro de validação de dados."""
    pass

class NotFoundError(AppError):
    """Recurso não encontrado."""
    pass

class AuthenticationError(AppError):
    """Erro de autenticação."""
    pass

class AuthorizationError(AppError):
    """Erro de autorização."""
    pass

# Uso
def get_user(user_id: str) -> dict:
    user = db.find_user(user_id)
    if not user:
        raise NotFoundError(f"User {user_id} not found")
    return user

# Captura específica
try:
    user = get_user("123")
except NotFoundError as e:
    # Handle not found
    pass
except AppError as e:
    # Handle other app errors
    pass
```

### Exemplo do Mundo Real

**E-commerce Domain Errors:**
```python
class EcommerceError(Exception):
    """Base exception para erros de e-commerce."""
    
    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code

class ProductError(EcommerceError):
    """Erros relacionados a produtos."""
    pass

class ProductNotFoundError(ProductError):
    """Produto não encontrado."""
    
    def __init__(self, product_id: str):
        super().__init__(
            f"Product {product_id} not found",
            error_code="PRODUCT_NOT_FOUND"
        )
        self.product_id = product_id

class ProductOutOfStockError(ProductError):
    """Produto sem estoque."""
    
    def __init__(self, product_id: str, requested: int, available: int):
        super().__init__(
            f"Product {product_id}: requested {requested}, only {available} available",
            error_code="PRODUCT_OUT_OF_STOCK"
        )
        self.product_id = product_id
        self.requested = requested
        self.available = available

class OrderError(EcommerceError):
    """Erros relacionados a pedidos."""
    pass

class InvalidOrderStateError(OrderError):
    """Estado do pedido inválido para operação."""
    
    def __init__(self, order_id: str, current_state: str, required_state: str):
        super().__init__(
            f"Order {order_id} is in state '{current_state}', "
            f"required '{required_state}'",
            error_code="INVALID_ORDER_STATE"
        )
        self.order_id = order_id
        self.current_state = current_state
        self.required_state = required_state

class PaymentError(EcommerceError):
    """Erros relacionados a pagamentos."""
    pass

class PaymentDeclinedError(PaymentError):
    """Pagamento recusado."""
    
    def __init__(self, reason: str, provider_code: str | None = None):
        super().__init__(
            f"Payment declined: {reason}",
            error_code="PAYMENT_DECLINED"
        )
        self.reason = reason
        self.provider_code = provider_code

# Uso
def place_order(cart: dict) -> dict:
    import structlog
    logger = structlog.get_logger()
    
    for item in cart["items"]:
        product = get_product(item["product_id"])
        
        if product["stock"] < item["quantity"]:
            logger.warning(
                "insufficient_stock",
                product_id=item["product_id"],
                requested=item["quantity"],
                available=product["stock"]
            )
            raise ProductOutOfStockError(
                item["product_id"],
                item["quantity"],
                product["stock"]
            )
    
    # Process order...
    return {"order_id": "ord-123"}

# Captura com handling específico
try:
    order = place_order(cart)
except ProductOutOfStockError as e:
    # Oferecer produtos alternativos
    suggest_alternatives(e.product_id)
except PaymentDeclinedError as e:
    # Sugerir método de pagamento alternativo
    suggest_payment_methods()
except EcommerceError as e:
    # Log erro genérico
    logger.error("order_failed", error_code=e.error_code, error=str(e))
```

---

## Exception Context

### Adding Context
```python
class APIError(Exception):
    """API error com contexto rico."""
    
    def __init__(
        self,
        message: str,
        status_code: int,
        endpoint: str,
        method: str,
        response_body: dict | None = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.endpoint = endpoint
        self.method = method
        self.response_body = response_body
    
    def to_dict(self) -> dict:
        """Convert to dict for logging."""
        return {
            "message": self.message,
            "status_code": self.status_code,
            "endpoint": self.endpoint,
            "method": self.method,
            "response_body": self.response_body
        }

# Uso
def call_api(endpoint: str) -> dict:
    import httpx
    
    try:
        response = httpx.get(f"https://api.example.com{endpoint}")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        raise APIError(
            message=str(exc),
            status_code=exc.response.status_code if hasattr(exc, 'response') else 500,
            endpoint=endpoint,
            method="GET",
            response_body=exc.response.json() if hasattr(exc, 'response') else None
        ) from exc

# Captura com contexto
import structlog
logger = structlog.get_logger()

try:
    data = call_api("/users/123")
except APIError as e:
    logger.error(
        "api_call_failed",
        **e.to_dict()
    )
```

---

## try/except Patterns

### Basic Pattern
```python
import structlog

logger = structlog.get_logger()

def process_file(filepath: str) -> dict:
    """Process file com error handling."""
    try:
        with open(filepath) as f:
            data = f.read()
        
        result = parse_data(data)
        logger.info("file_processed", filepath=filepath)
        return result
        
    except FileNotFoundError:
        logger.error("file_not_found", filepath=filepath)
        raise
    
    except ValueError as e:
        logger.error("invalid_data", filepath=filepath, error=str(e))
        raise
    
    except Exception as e:
        logger.error(
            "unexpected_error",
            filepath=filepath,
            error_type=type(e).__name__,
            error=str(e),
            exc_info=True
        )
        raise
```

### Multiple Exceptions
```python
def fetch_user_data(user_id: str) -> dict:
    """Fetch data with multiple exception handlers."""
    import structlog
    import httpx
    
    logger = structlog.get_logger()
    
    try:
        response = httpx.get(f"https://api.example.com/users/{user_id}")
        response.raise_for_status()
        return response.json()
        
    except httpx.TimeoutException:
        logger.error("api_timeout", user_id=user_id)
        raise APIError("API request timed out", status_code=504)
    
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning("user_not_found", user_id=user_id)
            raise NotFoundError(f"User {user_id} not found")
        else:
            logger.error("api_error", user_id=user_id, status_code=e.response.status_code)
            raise APIError(f"API error: {e.response.status_code}", status_code=e.response.status_code)
    
    except httpx.RequestError as e:
        logger.error("network_error", user_id=user_id, error=str(e))
        raise APIError("Network error", status_code=503)
```

### else and finally
```python
def process_transaction(transaction_id: str) -> None:
    """Process transaction com else/finally."""
    import structlog
    
    logger = structlog.get_logger()
    
    logger.info("transaction_started", transaction_id=transaction_id)
    
    try:
        # Try block
        validate_transaction(transaction_id)
        process_payment(transaction_id)
        
    except ValidationError as e:
        # Handle validation error
        logger.error("validation_failed", transaction_id=transaction_id, error=str(e))
        rollback_transaction(transaction_id)
        raise
    
    except PaymentError as e:
        # Handle payment error
        logger.error("payment_failed", transaction_id=transaction_id, error=str(e))
        rollback_transaction(transaction_id)
        raise
    
    else:
        # Success - only executes if no exception
        logger.info("transaction_completed", transaction_id=transaction_id)
        send_confirmation(transaction_id)
    
    finally:
        # Always executes (cleanup)
        release_lock(transaction_id)
        logger.info("transaction_finalized", transaction_id=transaction_id)
```

---

## Exception Chaining

### from keyword
```python
class DatabaseError(Exception):
    """Database error."""
    pass

def get_user(user_id: str) -> dict:
    """Get user with exception chaining."""
    import psycopg2
    
    try:
        conn = psycopg2.connect("...")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        
        if not result:
            raise NotFoundError(f"User {user_id} not found")
        
        return dict(result)
        
    except psycopg2.Error as e:
        # Chain exceptions - preserva contexto original
        raise DatabaseError(f"Failed to fetch user {user_id}") from e

# Captura - ambas exceções disponíveis
try:
    user = get_user("123")
except DatabaseError as e:
    # e.__cause__ contém psycopg2.Error original
    import structlog
    logger = structlog.get_logger()
    
    logger.error(
        "database_error",
        error=str(e),
        cause=str(e.__cause__),
        exc_info=True
    )
```

### suppress chain
```python
def process_data(data: str) -> dict:
    """Process data suprimindo chain."""
    try:
        result = parse_json(data)
        return result
    except ValueError:
        # Raise sem chain - esconde ValueError
        raise ValidationError("Invalid data format") from None
```

---

## FastAPI Error Handling

### Exception Handlers
```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import structlog

app = FastAPI()
logger = structlog.get_logger()

@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError):
    """Handle NotFoundError."""
    logger.warning(
        "not_found",
        path=request.url.path,
        error=str(exc)
    )
    
    return JSONResponse(
        status_code=404,
        content={
            "error": "not_found",
            "message": str(exc)
        }
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle ValidationError."""
    logger.warning(
        "validation_error",
        path=request.url.path,
        error=str(exc)
    )
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "validation_error",
            "message": str(exc)
        }
    )

@app.exception_handler(AuthenticationError)
async def authentication_error_handler(request: Request, exc: AuthenticationError):
    """Handle AuthenticationError."""
    logger.warning(
        "authentication_failed",
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=401,
        content={
            "error": "authentication_required",
            "message": "Authentication required"
        }
    )

@app.exception_handler(AuthorizationError)
async def authorization_error_handler(request: Request, exc: AuthorizationError):
    """Handle AuthorizationError."""
    logger.warning(
        "authorization_failed",
        path=request.url.path,
        error=str(exc)
    )
    
    return JSONResponse(
        status_code=403,
        content={
            "error": "forbidden",
            "message": "Insufficient permissions"
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "unexpected_error",
        path=request.url.path,
        method=request.method,
        error_type=type(exc).__name__,
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred"
        }
    )

# Endpoints
@app.get("/users/{user_id}")
async def get_user(user_id: str) -> dict:
    user = await db.get_user(user_id)
    
    if not user:
        raise NotFoundError(f"User {user_id} not found")
    
    return user

@app.post("/orders")
async def create_order(order: dict) -> dict:
    # Validação
    if not order.get("items"):
        raise ValidationError("Order must have at least one item")
    
    # Verificar estoque
    for item in order["items"]:
        product = await db.get_product(item["product_id"])
        if product["stock"] < item["quantity"]:
            raise ProductOutOfStockError(
                item["product_id"],
                item["quantity"],
                product["stock"]
            )
    
    # Criar pedido
    new_order = await db.create_order(order)
    return new_order
```

---

## Retry Patterns

### Simple Retry
```python
import time
import structlog
from typing import TypeVar, Callable

logger = structlog.get_logger()
T = TypeVar("T")

def retry(
    func: Callable[..., T],
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,)
) -> T:
    """Retry function on exception."""
    
    for attempt in range(max_attempts):
        try:
            return func()
        except exceptions as exc:
            if attempt == max_attempts - 1:
                logger.error(
                    "retry_exhausted",
                    function=func.__name__,
                    attempts=max_attempts,
                    error=str(exc)
                )
                raise
            
            wait_time = backoff_factor ** attempt
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

# Uso
def flaky_api_call() -> dict:
    import httpx
    response = httpx.get("https://flaky-api.example.com/data")
    response.raise_for_status()
    return response.json()

result = retry(
    flaky_api_call,
    max_attempts=3,
    backoff_factor=2.0,
    exceptions=(httpx.HTTPError,)
)
```

### Async Retry
```python
import asyncio
import structlog
from typing import TypeVar, Callable, Awaitable

logger = structlog.get_logger()
T = TypeVar("T")

async def async_retry(
    func: Callable[..., Awaitable[T]],
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,)
) -> T:
    """Retry async function on exception."""
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except exceptions as exc:
            if attempt == max_attempts - 1:
                logger.error(
                    "async_retry_exhausted",
                    function=func.__name__,
                    attempts=max_attempts,
                    error=str(exc)
                )
                raise
            
            wait_time = backoff_factor ** attempt
            logger.warning(
                "async_retry_attempt",
                function=func.__name__,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                wait_seconds=wait_time,
                error=str(exc)
            )
            await asyncio.sleep(wait_time)
    
    raise RuntimeError("Unreachable")

# Uso
async def flaky_async_call() -> dict:
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get("https://flaky-api.example.com/data")
        response.raise_for_status()
        return response.json()

result = await async_retry(
    flaky_async_call,
    max_attempts=3,
    exceptions=(httpx.HTTPError,)
)
```

---

## Graceful Degradation

### Fallback Pattern
```python
import structlog

logger = structlog.get_logger()

def get_user_recommendations(user_id: str) -> list[dict]:
    """Get recommendations com fallback."""
    
    try:
        # Tenta ML model
        recommendations = ml_model.predict(user_id)
        logger.info("recommendations_from_ml", user_id=user_id)
        return recommendations
        
    except Exception as exc:
        logger.warning(
            "ml_fallback",
            user_id=user_id,
            error=str(exc)
        )
        
        try:
            # Fallback: trending items
            recommendations = get_trending_items()
            logger.info("recommendations_from_trending", user_id=user_id)
            return recommendations
            
        except Exception as exc2:
            logger.error(
                "recommendations_failed",
                user_id=user_id,
                error=str(exc2)
            )
            # Fallback final: empty list
            return []

def get_user_profile(user_id: str) -> dict:
    """Get profile with partial degradation."""
    profile = {"user_id": user_id}
    
    # Essential data (must succeed)
    try:
        user_data = db.get_user(user_id)
        profile["name"] = user_data["name"]
        profile["email"] = user_data["email"]
    except Exception as exc:
        logger.error("essential_data_failed", user_id=user_id, exc_info=True)
        raise
    
    # Optional data (can fail)
    try:
        preferences = db.get_preferences(user_id)
        profile["preferences"] = preferences
    except Exception as exc:
        logger.warning("preferences_unavailable", user_id=user_id, error=str(exc))
        profile["preferences"] = {}
    
    # Optional data (can fail)
    try:
        orders = db.get_recent_orders(user_id)
        profile["recent_orders"] = orders
    except Exception as exc:
        logger.warning("orders_unavailable", user_id=user_id, error=str(exc))
        profile["recent_orders"] = []
    
    return profile
```

---

## Circuit Breaker Pattern

### Basic Implementation
```python
import time
from typing import Callable, TypeVar
from enum import Enum
import structlog

logger = structlog.get_logger()
T = TypeVar("T")

class CircuitState(Enum):
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """Circuit breaker para prevenir cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        name: str = "circuit"
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function through circuit breaker."""
        
        if self.state == CircuitState.OPEN:
            # Check if timeout passed
            if time.time() - self.last_failure_time >= self.timeout:
                logger.info("circuit_half_open", circuit=self.name)
                self.state = CircuitState.HALF_OPEN
            else:
                logger.warning("circuit_open", circuit=self.name)
                raise CircuitBreakerOpenError(f"Circuit {self.name} is open")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset if half-open
            if self.state == CircuitState.HALF_OPEN:
                logger.info("circuit_closed", circuit=self.name)
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            
            return result
            
        except Exception as exc:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                logger.error(
                    "circuit_opened",
                    circuit=self.name,
                    failures=self.failure_count
                )
                self.state = CircuitState.OPEN
            
            raise

class CircuitBreakerOpenError(Exception):
    """Circuit breaker is open."""
    pass

# Uso
api_circuit = CircuitBreaker(
    failure_threshold=5,
    timeout=60.0,
    name="external_api"
)

def call_external_api() -> dict:
    """Call API through circuit breaker."""
    import httpx
    
    def _call():
        response = httpx.get("https://api.example.com/data", timeout=5.0)
        response.raise_for_status()
        return response.json()
    
    try:
        return api_circuit.call(_call)
    except CircuitBreakerOpenError:
        logger.warning("api_circuit_open")
        # Return cached data or default
        return get_cached_data()
```

---

## Best Practices

✅ **Use hierarquia de exceções**
```python
class AppError(Exception): pass
class ValidationError(AppError): pass
class NotFoundError(AppError): pass
```

✅ **Include context em exceções**
```python
class OrderError(Exception):
    def __init__(self, order_id: str, message: str):
        super().__init__(message)
        self.order_id = order_id
```

✅ **Log antes de raise**
```python
try:
    result = operation()
except Exception as exc:
    logger.error("operation_failed", error=str(exc), exc_info=True)
    raise
```

✅ **Chain exceptions com from**
```python
try:
    db_operation()
except DatabaseError as e:
    raise AppError("Operation failed") from e
```

✅ **Use finally para cleanup**
```python
try:
    resource = acquire()
    process(resource)
finally:
    release(resource)  # Always executes
```

❌ **Não capture Exception genérica sem re-raise**
```python
# EVITE
try:
    operation()
except Exception:
    pass  # Silencia todos erros!

# CORRETO
try:
    operation()
except SpecificError:
    handle_specific()
except Exception as exc:
    logger.error("unexpected", exc_info=True)
    raise
```

❌ **Não use exceções para flow control**
```python
# EVITE
try:
    user = users[user_id]
except KeyError:
    user = None

# PREFIRA
user = users.get(user_id)
```

❌ **Não esconda exceptions sem logging**
```python
# EVITE
try:
    operation()
except Exception:
    return None  # Perde contexto do erro

# CORRETO
try:
    operation()
except Exception as exc:
    logger.error("operation_failed", exc_info=True)
    return None
```

---

## Casos de Uso Estabelecidos

### Domain Errors (DDD)
```python
class OrderCannotBeCancelledError(OrderError):
    """Order in state that cannot be cancelled."""
```

### API Errors (FastAPI, Flask)
```python
@app.exception_handler(NotFoundError)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"error": str(exc)})
```

### External Service Errors
```python
class ExternalAPIError(Exception):
    def __init__(self, service: str, status_code: int):
        self.service = service
        self.status_code = status_code
```

### Validation Errors (Pydantic)
```python
from pydantic import ValidationError

try:
    user = User(**data)
except ValidationError as e:
    logger.error("validation_failed", errors=e.errors())
```

### Database Errors
```python
try:
    db.execute(query)
except IntegrityError:
    raise DuplicateRecordError("Record already exists")
```

---

## Referências

- [Python Exception Handling](https://docs.python.org/3/tutorial/errors.html)
- [PEP 3151](https://peps.python.org/pep-3151/) - Reworking the OS and IO exception hierarchy
- [Exception Hierarchy](https://docs.python.org/3/library/exceptions.html#exception-hierarchy)
- [FastAPI Exception Handling](https://fastapi.tiangolo.com/tutorial/handling-errors/)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)