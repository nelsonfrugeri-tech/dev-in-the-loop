# Async/Await Patterns - Python 3.10+

Referência técnica completa de programação assíncrona em Python. Para decisões de quando usar async vs sync, consulte a skill principal (`/developer`).

## Fundamentos

Async/await permite I/O concorrente sem bloquear a thread principal. Ao invés de esperar operações (network, database, file I/O), a coroutine libera controle para outras tasks executarem.

**Quando usar:**
- I/O-bound: APIs HTTP, databases, file operations
- Múltiplas operações concorrentes (10+ requests simultâneos)
- WebSockets, streaming, long-polling

**Quando NÃO usar:**
- CPU-bound (use `multiprocessing`)
- Código predominantemente síncrono (overhead sem ganho)
- Bibliotecas sem suporte async

---

## Coroutines Básicas

### Definição

Funções `async def` retornam coroutines. Use `await` para aguardar operações assíncronas.
```python
import asyncio
import httpx

async def fetch_user(user_id: str) -> dict:
    """Coroutine que busca dados de usuário."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/users/{user_id}")
        return response.json()

# Executar coroutine
user = asyncio.run(fetch_user("123"))
```

### await vs await + assignment
```python
# await sem assignment - descarta resultado
await asyncio.sleep(1)

# await com assignment - captura resultado
result: dict = await fetch_user("123")

# Múltiplos awaits sequenciais (não concorrente!)
user1 = await fetch_user("1")  # Aguarda completar
user2 = await fetch_user("2")  # Só começa após user1
```

### Exemplo do Mundo Real

**FastAPI - Async Endpoint:**
```python
from fastapi import FastAPI, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: str, db: AsyncSession) -> dict:
    """Endpoint totalmente assíncrono."""
    # Não bloqueia thread durante query
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"id": user.id, "name": user.name}
```

---

## asyncio.gather() - Execução Paralela

### Definição

`asyncio.gather()` executa múltiplas coroutines concorrentemente e aguarda todas completarem.
```python
import asyncio

async def task1() -> str:
    await asyncio.sleep(1)
    return "Result 1"

async def task2() -> str:
    await asyncio.sleep(1)
    return "Result 2"

async def main() -> None:
    # Executa concorrentemente (total ~1s, não 2s)
    results = await asyncio.gather(task1(), task2())
    print(results)  # ["Result 1", "Result 2"]

asyncio.run(main())
```

### Return Exceptions

Por padrão, `gather()` propaga primeira exceção. Use `return_exceptions=True` para capturar erros:
```python
async def failing_task() -> str:
    raise ValueError("Error")

async def successful_task() -> str:
    return "Success"

# Sem return_exceptions - ValueError é propagada
results = await asyncio.gather(failing_task(), successful_task())

# Com return_exceptions - erros são retornados
results = await asyncio.gather(
    failing_task(),
    successful_task(),
    return_exceptions=True
)
# [ValueError("Error"), "Success"]
```

### Exemplos do Mundo Real

**httpx - Múltiplos Requests:**
```python
import httpx
import asyncio

async def fetch_all_users(user_ids: list[str]) -> list[dict]:
    """Busca múltiplos usuários concorrentemente."""
    async with httpx.AsyncClient() as client:
        tasks = [
            client.get(f"https://api.example.com/users/{uid}")
            for uid in user_ids
        ]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]

# 10 requests em paralelo (~latência de 1 request)
users = await fetch_all_users(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
```

**FastAPI - Aggregate Data:**
```python
from fastapi import FastAPI

@app.get("/dashboard/{user_id}")
async def get_dashboard(user_id: str) -> dict:
    """Busca dados de múltiplas fontes concorrentemente."""
    
    # Executa 3 queries em paralelo
    user, orders, notifications = await asyncio.gather(
        fetch_user(user_id),
        fetch_orders(user_id),
        fetch_notifications(user_id)
    )
    
    return {
        "user": user,
        "orders": orders,
        "notifications": notifications
    }
```

**Database - Batch Operations:**
```python
from sqlalchemy.ext.asyncio import AsyncSession

async def create_users_batch(
    db: AsyncSession,
    users: list[dict]
) -> list[User]:
    """Cria múltiplos usuários concorrentemente."""
    
    async def create_one(user_data: dict) -> User:
        user = User(**user_data)
        db.add(user)
        await db.flush()
        return user
    
    # Cria todos concorrentemente
    return await asyncio.gather(*[create_one(u) for u in users])
```

---

## asyncio.create_task() - Background Tasks

### Definição

`create_task()` agenda coroutine para execução em background sem aguardar.
```python
async def background_task() -> None:
    await asyncio.sleep(5)
    print("Background task completed")

async def main() -> None:
    # Agenda task mas não aguarda
    task = asyncio.create_task(background_task())
    
    print("Main continues immediately")
    
    # Aguarda quando necessário
    await task
```

### Task com Resultado
```python
async def compute() -> int:
    await asyncio.sleep(1)
    return 42

async def main() -> None:
    task = asyncio.create_task(compute())
    
    # Faz outras coisas enquanto task executa
    await other_work()
    
    # Obtém resultado
    result: int = await task
```

### Exemplos do Mundo Real

**FastAPI - Fire and Forget:**
```python
from fastapi import FastAPI, BackgroundTasks

@app.post("/orders")
async def create_order(order: OrderCreate) -> dict:
    """Cria pedido e envia email em background."""
    
    # Salva pedido
    new_order = await db_create_order(order)
    
    # Envia email sem aguardar
    asyncio.create_task(send_order_confirmation_email(new_order.id))
    
    # Retorna imediatamente
    return {"id": new_order.id, "status": "created"}
```

**Rate Limiting:**
```python
from collections import deque
import time

class RateLimiter:
    def __init__(self, max_requests: int, window: float):
        self.max_requests = max_requests
        self.window = window
        self.requests: deque[float] = deque()
        self._cleanup_task: asyncio.Task | None = None
    
    async def acquire(self) -> None:
        """Aguarda até ter slot disponível."""
        # Inicia cleanup em background
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        while len(self.requests) >= self.max_requests:
            await asyncio.sleep(0.1)
        
        self.requests.append(time.time())
    
    async def _cleanup_loop(self) -> None:
        """Remove requests antigas periodicamente."""
        while True:
            await asyncio.sleep(1)
            now = time.time()
            while self.requests and self.requests[0] < now - self.window:
                self.requests.popleft()
```

---

## Async Context Managers

### Definição

`async with` para recursos que precisam de setup/cleanup assíncrono.
```python
from typing import AsyncIterator

class AsyncResource:
    async def __aenter__(self) -> "AsyncResource":
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

# Uso
async with AsyncResource() as resource:
    await resource.operation()
```

### Exemplos do Mundo Real

**httpx - HTTP Client:**
```python
import httpx

async def fetch_data() -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get("https://api.example.com/data")
        return response.json()
    # Client fechado automaticamente
```

**asyncpg - Database Connection:**
```python
import asyncpg

async def query_users() -> list[dict]:
    async with asyncpg.create_pool(
        "postgresql://user:pass@localhost/db"
    ) as pool:
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM users")
            return [dict(row) for row in rows]
    # Pool e conexão fechados automaticamente
```

**Motor (MongoDB) - Transaction:**
```python
from motor.motor_asyncio import AsyncIOMotorClient

async def transfer_funds(from_id: str, to_id: str, amount: float) -> None:
    client = AsyncIOMotorClient()
    
    async with await client.start_session() as session:
        async with session.start_transaction():
            # Operações dentro da transação
            await accounts.update_one(
                {"_id": from_id},
                {"$inc": {"balance": -amount}},
                session=session
            )
            await accounts.update_one(
                {"_id": to_id},
                {"$inc": {"balance": amount}},
                session=session
            )
    # Commit automático ou rollback em caso de erro
```

**Custom - Database Session:**
```python
from contextlib import asynccontextmanager
from typing import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

@asynccontextmanager
async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Context manager para sessão de database."""
    session = async_sessionmaker(engine, class_=AsyncSession)()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()

# Uso
async with get_db_session() as db:
    user = User(name="Alice")
    db.add(user)
    # Commit automático no __aexit__
```

---

## Async Generators

### Definição

Generators assíncronos usam `async def` + `yield`, consumidos com `async for`.
```python
from typing import AsyncIterator

async def count_async(n: int) -> AsyncIterator[int]:
    """Generator que produz números com delay."""
    for i in range(n):
        await asyncio.sleep(0.1)
        yield i

# Consumir
async for num in count_async(5):
    print(num)
```

### Exemplos do Mundo Real

**Streaming Database Results:**
```python
from typing import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession

async def stream_users(
    db: AsyncSession,
    batch_size: int = 100
) -> AsyncIterator[list[dict]]:
    """Stream usuários em batches."""
    offset = 0
    while True:
        result = await db.execute(
            select(User).offset(offset).limit(batch_size)
        )
        users = result.scalars().all()
        
        if not users:
            break
        
        yield [{"id": u.id, "name": u.name} for u in users]
        offset += batch_size

# Processar em batches
async for batch in stream_users(db):
    await process_batch(batch)
```

**FastAPI - Server-Sent Events:**
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

@app.get("/events")
async def event_stream():
    """Stream de eventos via SSE."""
    
    async def generate() -> AsyncIterator[str]:
        for i in range(10):
            await asyncio.sleep(1)
            yield f"data: Event {i}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

**WebSocket Messages:**
```python
from typing import AsyncIterator
from fastapi import WebSocket

async def message_stream(
    websocket: WebSocket
) -> AsyncIterator[dict]:
    """Stream mensagens do websocket."""
    try:
        while True:
            data = await websocket.receive_json()
            yield data
    except WebSocketDisconnect:
        return

# Processar mensagens
async for message in message_stream(websocket):
    await handle_message(message)
```

---

## Async Comprehensions

### List Comprehensions
```python
# Async list comprehension
results = [
    await fetch_user(uid)
    async for uid in async_user_ids()
]

# Equivalente a:
results = []
async for uid in async_user_ids():
    result = await fetch_user(uid)
    results.append(result)
```

### Dict Comprehensions
```python
user_map = {
    user["id"]: user["name"]
    async for user in stream_users()
    if user["active"]
}
```

### Exemplo do Mundo Real

**FastAPI - Batch Processing:**
```python
@app.post("/users/batch")
async def create_users_batch(users: list[UserCreate]) -> list[dict]:
    """Cria múltiplos usuários e retorna resultados."""
    
    # Cria todos concorrentemente com comprehension
    created = await asyncio.gather(*[
        create_user(user)
        for user in users
    ])
    
    return [{"id": u.id, "name": u.name} for u in created]
```

---

## Error Handling

### Try/Except em Async
```python
async def safe_fetch(url: str) -> dict | None:
    """Fetch com error handling."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

### Exemplo do Mundo Real

**Retry Pattern:**
```python
async def fetch_with_retry(
    url: str,
    max_retries: int = 3,
    backoff: float = 1.0
) -> dict:
    """Fetch com retry exponencial."""
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            if attempt == max_retries - 1:
                raise
            
            wait_time = backoff * (2 ** attempt)
            print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
            await asyncio.sleep(wait_time)
    
    raise RuntimeError("Unreachable")
```

**Graceful Degradation:**
```python
@app.get("/dashboard/{user_id}")
async def get_dashboard(user_id: str) -> dict:
    """Dashboard com fallback para dados opcionais."""
    
    # Dados obrigatórios
    user = await fetch_user(user_id)
    
    # Dados opcionais com fallback
    async def safe_fetch_orders() -> list[dict]:
        try:
            return await fetch_orders(user_id)
        except Exception as e:
            print(f"Orders fetch failed: {e}")
            return []
    
    async def safe_fetch_notifications() -> list[dict]:
        try:
            return await fetch_notifications(user_id)
        except Exception as e:
            print(f"Notifications fetch failed: {e}")
            return []
    
    orders, notifications = await asyncio.gather(
        safe_fetch_orders(),
        safe_fetch_notifications()
    )
    
    return {
        "user": user,
        "orders": orders,
        "notifications": notifications
    }
```

---

## Timeouts

### asyncio.wait_for()
```python
async def fetch_with_timeout(url: str, timeout: float) -> dict:
    """Fetch com timeout."""
    try:
        result = await asyncio.wait_for(
            fetch_data(url),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        raise TimeoutError(f"Request to {url} timed out after {timeout}s")
```

### Exemplo do Mundo Real

**FastAPI - Request Timeout:**
```python
from fastapi import FastAPI, HTTPException

@app.get("/external-data")
async def get_external_data() -> dict:
    """Fetch de API externa com timeout de 5s."""
    try:
        data = await asyncio.wait_for(
            fetch_external_api(),
            timeout=5.0
        )
        return data
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="External API timeout"
        )
```

**Multiple Operations com Timeout:**
```python
async def fetch_all_with_timeout(
    urls: list[str],
    timeout: float
) -> list[dict | None]:
    """Fetch múltiplas URLs com timeout global."""
    
    async def safe_fetch(url: str) -> dict | None:
        try:
            return await fetch_data(url)
        except Exception:
            return None
    
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[safe_fetch(url) for url in urls]),
            timeout=timeout
        )
        return results
    except asyncio.TimeoutError:
        return [None] * len(urls)
```

---

## asyncio.wait() - Controle Fino

### Definição

`asyncio.wait()` oferece mais controle que `gather()`, permitindo especificar condições de conclusão.
```python
import asyncio

async def task1() -> str:
    await asyncio.sleep(1)
    return "Task 1"

async def task2() -> str:
    await asyncio.sleep(2)
    return "Task 2"

# Aguarda primeira completar
done, pending = await asyncio.wait(
    [task1(), task2()],
    return_when=asyncio.FIRST_COMPLETED
)

# Cancela tasks pendentes
for task in pending:
    task.cancel()
```

### Exemplo do Mundo Real

**Race Condition - Fastest Wins:**
```python
async def fetch_from_multiple_sources(data_id: str) -> dict:
    """Busca de múltiplas fontes, retorna a primeira."""
    
    tasks = [
        asyncio.create_task(fetch_from_primary(data_id)),
        asyncio.create_task(fetch_from_secondary(data_id)),
        asyncio.create_task(fetch_from_cache(data_id))
    ]
    
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )
    
    # Cancela tasks restantes
    for task in pending:
        task.cancel()
    
    # Retorna primeiro resultado
    return done.pop().result()
```

---

## asyncio.as_completed() - Processar Conforme Completa

### Definição

Itera sobre coroutines conforme completam, não na ordem original.
```python
async def process_as_completed(urls: list[str]) -> None:
    """Processa resultados conforme ficam prontos."""
    
    tasks = [fetch_data(url) for url in urls]
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(f"Got result: {result}")
        # Processa imediatamente, sem aguardar outras
```

### Exemplo do Mundo Real

**Progress Reporting:**
```python
async def download_files_with_progress(
    urls: list[str]
) -> list[bytes]:
    """Download com progress reporting."""
    
    total = len(urls)
    completed = 0
    results: list[bytes] = []
    
    tasks = [download_file(url) for url in urls]
    
    for coro in asyncio.as_completed(tasks):
        data = await coro
        results.append(data)
        completed += 1
        print(f"Progress: {completed}/{total}")
    
    return results
```

---

## Executando Código Síncrono

### run_in_executor()

Para código blocking (CPU-bound ou libs sem async):
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

def blocking_operation(n: int) -> int:
    """Função CPU-bound síncrona."""
    return sum(i * i for i in range(n))

async def async_wrapper(n: int) -> int:
    """Executa função síncrona sem bloquear event loop."""
    loop = asyncio.get_event_loop()
    
    # Executa em thread separada
    result = await loop.run_in_executor(
        None,  # Usa default executor
        blocking_operation,
        n
    )
    return result
```

### Exemplo do Mundo Real

**FastAPI - File Processing:**
```python
from fastapi import FastAPI, UploadFile
import asyncio

def process_image_sync(data: bytes) -> bytes:
    """Processamento de imagem (biblioteca síncrona)."""
    from PIL import Image
    import io
    
    img = Image.open(io.BytesIO(data))
    img.thumbnail((800, 800))
    
    output = io.BytesIO()
    img.save(output, format="JPEG")
    return output.getvalue()

@app.post("/upload-image")
async def upload_image(file: UploadFile) -> dict:
    """Upload e processa imagem sem bloquear."""
    data = await file.read()
    
    # Processa em executor
    loop = asyncio.get_event_loop()
    processed = await loop.run_in_executor(
        None,
        process_image_sync,
        data
    )
    
    return {"size": len(processed)}
```

---

## Connection Pooling

### Exemplo do Mundo Real

**httpx - Connection Pool:**
```python
import httpx
from typing import AsyncIterator

class APIClient:
    def __init__(self, base_url: str, max_connections: int = 100):
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=20
            ),
            timeout=30.0
        )
    
    async def get(self, path: str) -> dict:
        response = await self.client.get(path)
        response.raise_for_status()
        return response.json()
    
    async def close(self) -> None:
        await self.client.aclose()
    
    async def __aenter__(self) -> "APIClient":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

# Uso
async with APIClient("https://api.example.com") as client:
    users = await asyncio.gather(*[
        client.get(f"/users/{i}")
        for i in range(100)
    ])
    # Reutiliza conexões do pool
```

**asyncpg - Database Pool:**
```python
import asyncpg

class Database:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: asyncpg.Pool | None = None
    
    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=10,
            max_size=100,
            command_timeout=60
        )
    
    async def disconnect(self) -> None:
        if self.pool:
            await self.pool.close()
    
    async def fetch_users(self) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM users")
            return [dict(row) for row in rows]
```

---

## Lifespan Events (FastAPI)

### Startup/Shutdown
```python
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Gerencia lifecycle da aplicação."""
    
    # Startup
    print("Connecting to database...")
    app.state.db = await create_db_pool()
    app.state.redis = await create_redis_pool()
    
    yield
    
    # Shutdown
    print("Closing connections...")
    await app.state.db.close()
    await app.state.redis.close()

app = FastAPI(lifespan=lifespan)

@app.get("/users")
async def get_users():
    # Usa conexões do pool
    async with app.state.db.acquire() as conn:
        return await conn.fetch("SELECT * FROM users")
```

---

## Casos de Uso Estabelecidos

### Web APIs (FastAPI, aiohttp)
- Endpoints assíncronos
- Multiple database queries em paralelo
- Integração com APIs externas

### Database Operations (asyncpg, motor)
- Connection pooling
- Batch operations
- Streaming large result sets

### Message Queues (aio-pika, aiokafka)
- Consumidores assíncronos
- Processing pipeline

### WebSockets (FastAPI, websockets)
- Bidirectional communication
- Broadcasting

### File I/O (aiofiles)
- Large file processing
- Concurrent file operations

---

## Anti-Patterns

❌ **Não fazer**: Misturar sync e async sem cuidado
```python
# ERRADO - bloqueia event loop
async def bad():
    time.sleep(1)  # Bloqueia!
    
# CORRETO
async def good():
    await asyncio.sleep(1)
```

❌ **Não fazer**: await em loop sem gather
```python
# ERRADO - sequencial
for user_id in user_ids:
    user = await fetch_user(user_id)  # Um por vez

# CORRETO - concorrente
users = await asyncio.gather(*[
    fetch_user(uid) for uid in user_ids
])
```

❌ **Não fazer**: Create task sem await
```python
# ERRADO - task pode não completar
asyncio.create_task(important_operation())
return "Done"

# CORRETO - aguarda ou track
task = asyncio.create_task(important_operation())
# ... fazer outras coisas ...
await task
```

---

## Referências

- [asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [PEP 492](https://peps.python.org/pep-0492/) - Coroutines with async/await
- [FastAPI Async](https://fastapi.tiangolo.com/async/)
- [httpx Async](https://www.python-httpx.org/async/)
- [asyncpg Documentation](https://magicstack.github.io/asyncpg/)