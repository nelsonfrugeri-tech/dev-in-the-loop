# Context Managers - Python 3.10+

Referência técnica completa de context managers em Python. Para decisões de quando criar custom context managers, consulte a skill principal (`/developer`).

## Fundamentos

Context managers garantem setup e cleanup de recursos, mesmo quando exceções ocorrem. O protocolo `with` chama `__enter__` no início e `__exit__` no final.

**Quando usar:**
- Gerenciamento de recursos (files, connections, locks)
- Setup/teardown automático
- Transações (commit/rollback)
- Temporary state changes

**Benefícios:**
- Cleanup garantido (mesmo com exceções)
- Código mais limpo e legível
- Previne resource leaks

---

## with Statement - Uso Básico

### Definição
```python
# Sem context manager - manual cleanup
file = open("data.txt")
try:
    data = file.read()
finally:
    file.close()  # Sempre executa

# Com context manager - cleanup automático
with open("data.txt") as file:
    data = file.read()
# file.close() chamado automaticamente
```

### Múltiplos Context Managers
```python
# Forma antiga (nested)
with open("input.txt") as infile:
    with open("output.txt", "w") as outfile:
        outfile.write(infile.read())

# Forma moderna (Python 3.10+)
with (
    open("input.txt") as infile,
    open("output.txt", "w") as outfile,
):
    outfile.write(infile.read())
```

### Exemplo do Mundo Real

**Database Session (SQLAlchemy):**
```python
from sqlalchemy.orm import Session

# Sem context manager - manual
session = Session(engine)
try:
    user = session.query(User).filter_by(id=1).first()
    user.name = "Updated"
    session.commit()
except Exception:
    session.rollback()
    raise
finally:
    session.close()

# Com context manager - automático
with Session(engine) as session:
    user = session.query(User).filter_by(id=1).first()
    user.name = "Updated"
    session.commit()
    # rollback automático em caso de exceção
    # close automático no final
```

---

## Implementando __enter__ e __exit__

### Protocol Básico
```python
class ManagedResource:
    def __enter__(self):
        """Setup - executa no início do with."""
        print("Acquiring resource")
        return self  # Retorna objeto para 'as' clause
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup - executa no final do with."""
        print("Releasing resource")
        return False  # Propaga exceção (True = suprime)

# Uso
with ManagedResource() as resource:
    print("Using resource")
    # resource é o retorno de __enter__
```

### Parâmetros de __exit__
```python
class ResourceWithErrorHandling:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        exc_type: Tipo da exceção (ou None)
        exc_val: Instância da exceção (ou None)
        exc_tb: Traceback (ou None)
        """
        if exc_type is None:
            print("Success - no exception")
        else:
            print(f"Exception occurred: {exc_type.__name__}: {exc_val}")
        
        # return False: propaga exceção
        # return True: suprime exceção
        return False
```

### Exemplo do Mundo Real

**Database Connection Pool:**
```python
from typing import Any
import psycopg2
from psycopg2.pool import SimpleConnectionPool

class DatabaseConnection:
    """Context manager para conexão do pool."""
    
    def __init__(self, pool: SimpleConnectionPool):
        self.pool = pool
        self.conn = None
    
    def __enter__(self):
        """Adquire conexão do pool."""
        self.conn = self.pool.getconn()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Retorna conexão ao pool."""
        if exc_type is not None:
            # Rollback em caso de erro
            self.conn.rollback()
        else:
            # Commit em sucesso
            self.conn.commit()
        
        # Retorna ao pool
        self.pool.putconn(self.conn)
        return False  # Propaga exceção

# Uso
pool = SimpleConnectionPool(minconn=1, maxconn=10, dsn="...")

with DatabaseConnection(pool) as conn:
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users VALUES (%s, %s)", ("Alice", "alice@example.com"))
    # commit automático se sucesso
    # rollback automático se erro
```

**File Lock:**
```python
import fcntl
from pathlib import Path

class FileLock:
    """Context manager para lock exclusivo em arquivo."""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.file = None
    
    def __enter__(self):
        """Adquire lock."""
        self.file = open(self.filepath, "a")
        fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock."""
        if self.file:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            self.file.close()
        return False

# Uso - garante acesso exclusivo
with FileLock(Path("/tmp/myapp.lock")) as f:
    # Somente um processo por vez aqui
    f.write("Processing...\n")
```

---

## @contextmanager Decorator

### Definição

`@contextmanager` simplifica criação de context managers usando generators.
```python
from contextlib import contextmanager

@contextmanager
def managed_resource():
    """Context manager simplificado."""
    # Setup (antes do yield)
    print("Acquiring resource")
    resource = acquire_resource()
    
    try:
        yield resource  # Passa para 'with' block
    finally:
        # Cleanup (sempre executa)
        print("Releasing resource")
        release_resource(resource)

# Uso
with managed_resource() as resource:
    use(resource)
```

### Error Handling
```python
from contextlib import contextmanager

@contextmanager
def transaction(connection):
    """Context manager para transação database."""
    cursor = connection.cursor()
    try:
        yield cursor
        # Sucesso - commit
        connection.commit()
    except Exception:
        # Erro - rollback
        connection.rollback()
        raise
    finally:
        cursor.close()

# Uso
with transaction(conn) as cursor:
    cursor.execute("INSERT INTO users VALUES (%s)", ("Alice",))
    # commit automático se sucesso
    # rollback automático se erro
```

### Exemplo do Mundo Real

**Temporary Directory:**
```python
from contextlib import contextmanager
from pathlib import Path
import tempfile
import shutil

@contextmanager
def temp_directory():
    """Cria diretório temporário e remove ao final."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Uso
with temp_directory() as tmp:
    # Trabalhar com arquivos temporários
    (tmp / "data.txt").write_text("temporary data")
    process_files(tmp)
    # tmp é removido automaticamente
```

**Timing Context:**
```python
from contextlib import contextmanager
import time

@contextmanager
def timer(name: str):
    """Mede tempo de execução de bloco."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name} took {elapsed:.4f}s")

# Uso
with timer("Database query"):
    results = db.execute("SELECT * FROM large_table")
# Output: Database query took 2.3451s
```

**Database Session (FastAPI Pattern):**
```python
from contextlib import contextmanager
from typing import Iterator
from sqlalchemy.orm import Session, sessionmaker

SessionLocal = sessionmaker(bind=engine)

@contextmanager
def get_db() -> Iterator[Session]:
    """Context manager para database session."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# Uso em endpoint
def create_user(name: str, email: str) -> User:
    with get_db() as db:
        user = User(name=name, email=email)
        db.add(user)
        # commit automático se sucesso
        return user
```

**Changing Working Directory:**
```python
from contextlib import contextmanager
from pathlib import Path
import os

@contextmanager
def cd(path: Path):
    """Muda diretório temporariamente."""
    old_dir = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_dir)

# Uso
print(f"Before: {Path.cwd()}")
with cd(Path("/tmp")):
    print(f"Inside: {Path.cwd()}")
    # Trabalhar em /tmp
print(f"After: {Path.cwd()}")  # Volta ao original
```

---

## Async Context Managers

### Definição

Context managers assíncronos usam `__aenter__` e `__aexit__` com `async with`.
```python
class AsyncResource:
    async def __aenter__(self):
        """Async setup."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async cleanup."""
        await self.disconnect()
        return False

# Uso
async with AsyncResource() as resource:
    await resource.operation()
```

### Exemplo do Mundo Real

**httpx - HTTP Client:**
```python
import httpx

# httpx.AsyncClient é async context manager
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        return response.json()
    # client.aclose() chamado automaticamente
```

**asyncpg - Database Connection:**
```python
import asyncpg

async def query_users():
    # Connection pool como async context manager
    async with asyncpg.create_pool(
        "postgresql://user:pass@localhost/db",
        min_size=10,
        max_size=100
    ) as pool:
        # Connection do pool
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM users")
            return [dict(row) for row in rows]
        # Conexão retornada ao pool
    # Pool fechado
```

**FastAPI - Lifespan:**
```python
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle."""
    # Startup
    print("Starting up...")
    app.state.db_pool = await create_db_pool()
    app.state.redis = await create_redis_client()
    
    yield
    
    # Shutdown
    print("Shutting down...")
    await app.state.db_pool.close()
    await app.state.redis.close()

app = FastAPI(lifespan=lifespan)
```

---

## @asynccontextmanager

### Definição

Versão assíncrona de `@contextmanager`.
```python
from contextlib import asynccontextmanager
import asyncio

@asynccontextmanager
async def async_resource():
    """Async context manager simplificado."""
    # Async setup
    resource = await acquire_async_resource()
    try:
        yield resource
    finally:
        # Async cleanup
        await release_async_resource(resource)

# Uso
async with async_resource() as r:
    await r.operation()
```

### Exemplo do Mundo Real

**Database Session:**
```python
from contextlib import asynccontextmanager
from typing import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

@asynccontextmanager
async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Async database session."""
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
async def create_user(name: str) -> User:
    async with get_db_session() as db:
        user = User(name=name)
        db.add(user)
        return user
```

**Rate Limiter:**
```python
from contextlib import asynccontextmanager
import asyncio
from typing import AsyncIterator

class AsyncRateLimiter:
    def __init__(self, max_concurrent: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """Acquire rate limit slot."""
        await self.semaphore.acquire()
        try:
            yield
        finally:
            self.semaphore.release()

# Uso
limiter = AsyncRateLimiter(max_concurrent=5)

async def fetch_with_limit(url: str) -> dict:
    async with limiter.acquire():
        return await fetch_data(url)
```

**Distributed Lock (Redis):**
```python
from contextlib import asynccontextmanager
from typing import AsyncIterator
import aioredis
from uuid import uuid4

@asynccontextmanager
async def redis_lock(
    redis: aioredis.Redis,
    key: str,
    timeout: int = 10
) -> AsyncIterator[bool]:
    """Distributed lock usando Redis."""
    lock_id = str(uuid4())
    
    # Tentar adquirir lock
    acquired = await redis.set(
        key,
        lock_id,
        ex=timeout,
        nx=True  # Só seta se não existe
    )
    
    try:
        yield acquired
    finally:
        if acquired:
            # Release lock (só se ainda somos donos)
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            await redis.eval(lua_script, 1, key, lock_id)

# Uso
async with redis_lock(redis, "order:123") as acquired:
    if acquired:
        # Processar pedido com lock exclusivo
        await process_order("123")
    else:
        # Lock não disponível
        raise LockNotAvailableError()
```

---

## contextlib Utilities

### suppress()

Suprime exceções específicas.
```python
from contextlib import suppress

# Sem suppress
try:
    os.remove("file.txt")
except FileNotFoundError:
    pass

# Com suppress - mais limpo
with suppress(FileNotFoundError):
    os.remove("file.txt")
```

### redirect_stdout() / redirect_stderr()
```python
from contextlib import redirect_stdout
import io

# Capturar stdout
buffer = io.StringIO()
with redirect_stdout(buffer):
    print("This goes to buffer")
    print("Not to console")

output = buffer.getvalue()
print(f"Captured: {output}")
```

### nullcontext()

Context manager que não faz nada - útil para condicionais.
```python
from contextlib import nullcontext

def process_file(filepath: str, use_lock: bool = True):
    """Processa arquivo, opcionalmente com lock."""
    lock = FileLock(filepath) if use_lock else nullcontext()
    
    with lock:
        # Código funciona com ou sem lock
        process(filepath)
```

### ExitStack

Gerencia múltiplos context managers dinamicamente.
```python
from contextlib import ExitStack

def process_files(filenames: list[str]):
    """Processa múltiplos arquivos simultaneamente."""
    with ExitStack() as stack:
        # Abre todos os arquivos
        files = [
            stack.enter_context(open(fname))
            for fname in filenames
        ]
        
        # Processa todos
        for f in files:
            process(f)
        # Todos fechados automaticamente
```

### Exemplo do Mundo Real

**Dynamic Resource Management:**
```python
from contextlib import ExitStack
from typing import Iterator

def process_batch(
    input_files: list[str],
    output_file: str,
    use_compression: bool = False
) -> None:
    """Processa múltiplos inputs com recursos dinâmicos."""
    with ExitStack() as stack:
        # Open all inputs
        inputs = [
            stack.enter_context(open(f))
            for f in input_files
        ]
        
        # Open output (possivelmente comprimido)
        if use_compression:
            import gzip
            output = stack.enter_context(gzip.open(output_file, "wt"))
        else:
            output = stack.enter_context(open(output_file, "w"))
        
        # Process
        for infile in inputs:
            output.write(infile.read())
        # Todos fechados automaticamente
```

---

## Threading Locks

### thread.Lock
```python
import threading

lock = threading.Lock()

# Sem context manager - manual
lock.acquire()
try:
    # Critical section
    shared_resource.modify()
finally:
    lock.release()

# Com context manager - automático
with lock:
    # Critical section
    shared_resource.modify()
    # lock.release() automático
```

### RLock (Reentrant Lock)
```python
import threading

class Counter:
    def __init__(self):
        self._lock = threading.RLock()
        self._count = 0
    
    def increment(self):
        with self._lock:
            self._count += 1
    
    def increment_by(self, n: int):
        # RLock permite acquire múltiplas vezes
        with self._lock:
            for _ in range(n):
                self.increment()  # Adquire lock novamente
```

### Exemplo do Mundo Real

**Thread-Safe Cache:**
```python
import threading
from typing import Dict, Any

class ThreadSafeCache:
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = value
    
    def get_or_compute(self, key: str, compute_fn) -> Any:
        # Lock para verificação e computação
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            
            value = compute_fn()
            self._cache[key] = value
            return value

# Uso multi-threaded
cache = ThreadSafeCache()

def worker(item_id: str):
    result = cache.get_or_compute(
        item_id,
        lambda: expensive_computation(item_id)
    )
    process(result)
```

---

## Testing Context Managers

### pytest
```python
import pytest
from contextlib import contextmanager

@contextmanager
def db_transaction():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def test_transaction_success():
    """Testa commit em caso de sucesso."""
    with db_transaction() as conn:
        conn.execute("INSERT INTO users VALUES ('Alice')")
    
    # Verificar que commit foi chamado
    assert user_exists("Alice")

def test_transaction_rollback():
    """Testa rollback em caso de erro."""
    with pytest.raises(ValueError):
        with db_transaction() as conn:
            conn.execute("INSERT INTO users VALUES ('Bob')")
            raise ValueError("Force rollback")
    
    # Verificar que rollback foi feito
    assert not user_exists("Bob")
```

---

## Casos de Uso Estabelecidos

### File Operations
```python
with open("file.txt") as f:
    data = f.read()
```

### Database Sessions (SQLAlchemy, asyncpg)
```python
with Session(engine) as session:
    user = session.query(User).first()
```

### HTTP Clients (httpx, aiohttp)
```python
async with httpx.AsyncClient() as client:
    response = await client.get(url)
```

### Locks (threading, asyncio)
```python
with lock:
    # Critical section
    modify_shared_resource()
```

### Temporary Resources (tempfile)
```python
with tempfile.TemporaryDirectory() as tmpdir:
    # Work with temp files
    process(tmpdir)
```

### Transactions (database, file systems)
```python
with transaction(conn):
    conn.execute(query)
    # Auto commit/rollback
```

### Timing and Profiling
```python
with timer("operation"):
    expensive_operation()
```

### Context Changes (cd, environment)
```python
with cd("/tmp"):
    # Work in /tmp
    process_files()
```

---

## Best Practices

✅ **Sempre use context managers para resources**
```python
# CORRETO
with open("file.txt") as f:
    data = f.read()

# EVITE (pode vazar resource)
f = open("file.txt")
data = f.read()
f.close()
```

✅ **Use @contextmanager para simplicidade**
```python
from contextlib import contextmanager

@contextmanager
def simple_manager():
    setup()
    try:
        yield
    finally:
        cleanup()
```

✅ **Cleanup em finally**
```python
def __exit__(self, exc_type, exc_val, exc_tb):
    # Cleanup SEMPRE executa
    self.resource.close()
    return False
```

✅ **Propague exceções (return False)**
```python
def __exit__(self, exc_type, exc_val, exc_tb):
    cleanup()
    return False  # Não suprime exceção
```

❌ **Não suprima exceções sem motivo**
```python
# EVITE
def __exit__(self, exc_type, exc_val, exc_tb):
    cleanup()
    return True  # Suprime TODAS exceções!
```

❌ **Não faça I/O pesado em __enter__**
```python
# EVITE
def __enter__(self):
    self.data = load_huge_file()  # Pode travar
    return self

# PREFIRA - lazy loading
def __enter__(self):
    return self

def get_data(self):
    if not hasattr(self, '_data'):
        self._data = load_huge_file()
    return self._data
```

---

## Referências

- [PEP 343](https://peps.python.org/pep-0343/) - The "with" Statement
- [contextlib Documentation](https://docs.python.org/3/library/contextlib.html)
- [PEP 492](https://peps.python.org/pep-0492/) - Async Context Managers
- [Real Python - Context Managers](https://realpython.com/python-with-statement/)