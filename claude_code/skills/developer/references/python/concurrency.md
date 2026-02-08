# Concurrency - Python 3.10+

Referência técnica completa de concorrência em Python. Para decisões de qual modelo usar (asyncio vs threading vs multiprocessing), consulte a skill principal (`/developer`).

## Fundamentos

Python oferece três modelos de concorrência, cada um otimizado para diferentes workloads:

| Modelo | Best For | Parallelism | GIL Impact |
|--------|----------|-------------|------------|
| **asyncio** | I/O-bound (network, disk) | Cooperativo (single-thread) | Não afetado |
| **threading** | I/O-bound legado, blocking libs | Concurrent (multi-thread) | Limitado pelo GIL |
| **multiprocessing** | CPU-bound (cálculos pesados) | Paralelo (multi-process) | Bypassa GIL |

**GIL (Global Interpreter Lock):**
- Lock que impede múltiplas threads Python executarem bytecode simultaneamente
- Threads em operações I/O liberam GIL automaticamente
- CPU-bound em threads não ganha speedup (precisa multiprocessing)

---

## asyncio - I/O Cooperativo

### Quando Usar

**✅ Use asyncio para:**
- Múltiplas requisições HTTP simultâneas
- WebSockets, streaming, long-polling
- Database queries concorrentes (com drivers async)
- File I/O assíncrono (aiofiles)
- Qualquer I/O-bound com latência

**❌ Não use asyncio para:**
- CPU-bound (cálculos pesados) → use multiprocessing
- Bibliotecas sem suporte async → use threading + run_in_executor
- Scripts simples sem I/O concorrente → overhead sem ganho

### Sintaxe Básica
```python
import asyncio
import httpx

async def fetch_user(user_id: str) -> dict:
    """Busca usuário da API."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/users/{user_id}")
        return response.json()

async def main() -> None:
    # Executa 10 requests concorrentemente
    user_ids = [f"user-{i}" for i in range(10)]
    users = await asyncio.gather(*[fetch_user(uid) for uid in user_ids])

# Run
asyncio.run(main())
```

### Exemplo do Mundo Real

**FastAPI - Async Endpoints:**
```python
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession
import httpx
import structlog

logger = structlog.get_logger()

app = FastAPI()

@app.get("/dashboard/{user_id}")
async def get_dashboard(user_id: str, db: AsyncSession) -> dict:
    """Dashboard com múltiplas fontes de dados."""
    
    # Buscar dados em paralelo
    user_data, orders_data, notifications_data = await asyncio.gather(
        fetch_user_from_db(db, user_id),
        fetch_orders_from_api(user_id),
        fetch_notifications_from_redis(user_id),
        return_exceptions=True  # Não falha tudo se um der erro
    )
    
    # Handle errors gracefully
    if isinstance(user_data, Exception):
        logger.error("user_fetch_failed", error=str(user_data))
        raise HTTPException(status_code=500)
    
    return {
        "user": user_data,
        "orders": orders_data if not isinstance(orders_data, Exception) else [],
        "notifications": notifications_data if not isinstance(notifications_data, Exception) else []
    }

async def fetch_user_from_db(db: AsyncSession, user_id: str) -> dict:
    """Busca usuário do database."""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise ValueError("User not found")
    return {"id": user.id, "name": user.name}

async def fetch_orders_from_api(user_id: str) -> list[dict]:
    """Busca pedidos de API externa."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"https://orders-api.example.com/users/{user_id}/orders")
        return response.json()

async def fetch_notifications_from_redis(user_id: str) -> list[dict]:
    """Busca notificações do Redis."""
    import aioredis
    redis = await aioredis.from_url("redis://localhost")
    notifications = await redis.lrange(f"notifications:{user_id}", 0, -1)
    await redis.close()
    return [json.loads(n) for n in notifications]
```

**Batch Processing - asyncpg:**
```python
import asyncpg
import asyncio
from typing import AsyncIterator

async def stream_users_batch(
    pool: asyncpg.Pool,
    batch_size: int = 1000
) -> AsyncIterator[list[dict]]:
    """Stream usuários em batches."""
    offset = 0
    
    while True:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM users ORDER BY id LIMIT $1 OFFSET $2",
                batch_size,
                offset
            )
            
            if not rows:
                break
            
            yield [dict(row) for row in rows]
            offset += batch_size

async def process_users_concurrently(pool: asyncpg.Pool) -> None:
    """Processa usuários em batches concorrentes."""
    
    async def process_batch(batch: list[dict]) -> None:
        """Processa um batch de usuários."""
        # Simula processamento pesado (API calls, etc)
        await asyncio.gather(*[
            send_notification(user["id"], user["email"])
            for user in batch
        ])
    
    # Processa 3 batches concorrentemente
    semaphore = asyncio.Semaphore(3)
    
    async def limited_process(batch: list[dict]) -> None:
        async with semaphore:
            await process_batch(batch)
    
    tasks = []
    async for batch in stream_users_batch(pool):
        task = asyncio.create_task(limited_process(batch))
        tasks.append(task)
    
    await asyncio.gather(*tasks)
```

### asyncio Patterns

**Rate Limiting:**
```python
import asyncio
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")

class AsyncRateLimiter:
    """Rate limiter usando asyncio.Semaphore."""
    
    def __init__(self, max_concurrent: int, calls_per_second: float):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
    
    async def __aenter__(self):
        await self.semaphore.acquire()
        
        # Espera intervalo mínimo desde última chamada
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        
        self.last_call = asyncio.get_event_loop().time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()

# Uso
limiter = AsyncRateLimiter(max_concurrent=10, calls_per_second=100)

async def fetch_with_limit(url: str) -> dict:
    async with limiter:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            return response.json()
```

**Retry with Exponential Backoff:**
```python
import asyncio
from typing import TypeVar, Callable
import structlog

logger = structlog.get_logger()

T = TypeVar("T")

async def async_retry(
    func: Callable[..., T],
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,)
) -> T:
    """Retry async function com exponential backoff."""
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except exceptions as exc:
            if attempt == max_attempts - 1:
                logger.error(
                    "async_retry_exhausted",
                    attempts=max_attempts,
                    error=str(exc)
                )
                raise
            
            wait_time = backoff_factor ** attempt
            logger.warning(
                "async_retry_attempt",
                attempt=attempt + 1,
                max_attempts=max_attempts,
                wait_seconds=wait_time
            )
            await asyncio.sleep(wait_time)
    
    raise RuntimeError("Unreachable")

# Uso
async def flaky_api_call() -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get("https://flaky-api.example.com/data")
        response.raise_for_status()
        return response.json()

result = await async_retry(flaky_api_call, max_attempts=3)
```

### Casos de Uso Estabelecidos

**Web Frameworks** (FastAPI, aiohttp, Sanic):
- Endpoints assíncronos para alta concorrência
- WebSocket handlers

**Database Operations** (asyncpg, motor, tortoise-orm):
- Connection pooling
- Batch queries

**HTTP Clients** (httpx, aiohttp):
- Multiple API calls
- Web scraping

**Message Queues** (aio-pika, aiokafka):
- Async consumers
- Event processing

---

## threading - I/O com Bibliotecas Síncronas

### Quando Usar

**✅ Use threading para:**
- Bibliotecas sem suporte async (requests, Pillow, etc)
- I/O-bound com blocking calls
- Compatibilidade com código legado
- Background tasks leves

**❌ Não use threading para:**
- CPU-bound → use multiprocessing (GIL impede speedup)
- Quando async está disponível → asyncio é mais eficiente
- Coordenação complexa → asyncio é mais simples

### Sintaxe Básica
```python
import threading
import requests
from queue import Queue

def fetch_url(url: str, result_queue: Queue) -> None:
    """Worker thread que busca URL."""
    response = requests.get(url)
    result_queue.put((url, response.json()))

def fetch_multiple_urls(urls: list[str]) -> list[tuple[str, dict]]:
    """Busca múltiplas URLs com threads."""
    result_queue: Queue = Queue()
    threads = []
    
    # Criar threads
    for url in urls:
        thread = threading.Thread(target=fetch_url, args=(url, result_queue))
        thread.start()
        threads.append(thread)
    
    # Aguardar todas
    for thread in threads:
        thread.join()
    
    # Coletar resultados
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    return results

# Uso
urls = ["https://api.example.com/1", "https://api.example.com/2"]
results = fetch_multiple_urls(urls)
```

### ThreadPoolExecutor

Forma moderna e recomendada:
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from typing import Iterator
import structlog

logger = structlog.get_logger()

def fetch_url(url: str) -> tuple[str, dict | None]:
    """Busca URL com error handling."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return (url, response.json())
    except Exception as exc:
        logger.error("fetch_failed", url=url, error=str(exc))
        return (url, None)

def fetch_urls_parallel(urls: list[str], max_workers: int = 10) -> list[tuple[str, dict | None]]:
    """Busca URLs em paralelo com thread pool."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submete todas as tasks
        futures = [executor.submit(fetch_url, url) for url in urls]
        
        # Coleta resultados conforme completam
        results = []
        for future in as_completed(futures):
            results.append(future.result())
        
        return results

# Uso
urls = [f"https://api.example.com/items/{i}" for i in range(100)]
results = fetch_urls_parallel(urls, max_workers=20)
```

### Exemplo do Mundo Real

**Image Processing com Pillow:**
```python
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from typing import Iterator
import structlog

logger = structlog.get_logger()

def resize_image(input_path: Path, output_dir: Path, size: tuple[int, int]) -> Path:
    """Redimensiona imagem (I/O-bound)."""
    try:
        img = Image.open(input_path)
        img.thumbnail(size)
        
        output_path = output_dir / f"{input_path.stem}_thumb{input_path.suffix}"
        img.save(output_path)
        
        logger.info("image_resized", input=str(input_path), output=str(output_path))
        return output_path
        
    except Exception as exc:
        logger.error("resize_failed", path=str(input_path), error=str(exc))
        raise

def batch_resize_images(
    input_dir: Path,
    output_dir: Path,
    size: tuple[int, int] = (800, 600),
    max_workers: int = 4
) -> list[Path]:
    """Redimensiona múltiplas imagens em paralelo."""
    
    output_dir.mkdir(exist_ok=True)
    image_files = list(input_dir.glob("*.{jpg,jpeg,png}"))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(resize_image, img_path, output_dir, size)
            for img_path in image_files
        ]
        
        return [future.result() for future in futures]

# Uso
resized = batch_resize_images(
    Path("/input/images"),
    Path("/output/thumbnails"),
    size=(400, 300),
    max_workers=8
)
```

**Background Tasks (FastAPI):**
```python
from fastapi import FastAPI, BackgroundTasks
from concurrent.futures import ThreadPoolExecutor
import structlog

logger = structlog.get_logger()
executor = ThreadPoolExecutor(max_workers=10)

app = FastAPI()

def send_email_blocking(to: str, subject: str, body: str) -> None:
    """Envia email usando biblioteca síncrona."""
    import smtplib
    from email.message import EmailMessage
    
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["To"] = to
    
    with smtplib.SMTP("smtp.example.com", 587) as server:
        server.send_message(msg)
    
    logger.info("email_sent", to=to, subject=subject)

@app.post("/orders")
async def create_order(order: OrderCreate, background_tasks: BackgroundTasks):
    """Cria pedido e envia email em background."""
    
    # Salva pedido (async)
    new_order = await db.create_order(order)
    
    # Agenda email (thread pool para blocking call)
    background_tasks.add_task(
        lambda: executor.submit(
            send_email_blocking,
            order.customer_email,
            "Order Confirmed",
            f"Your order {new_order.id} has been confirmed"
        )
    )
    
    return {"id": new_order.id, "status": "created"}
```

### Thread Safety - Locks
```python
import threading
from typing import Dict, Any
import structlog

logger = structlog.get_logger()

class ThreadSafeCache:
    """Cache thread-safe usando Lock."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()  # Reentrant lock
    
    def get(self, key: str) -> Any | None:
        """Thread-safe get."""
        with self._lock:
            return self._cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Thread-safe set."""
        with self._lock:
            self._cache[key] = value
            logger.debug("cache_set", key=key)
    
    def get_or_compute(self, key: str, compute_fn) -> Any:
        """Get from cache ou computa (thread-safe)."""
        with self._lock:
            # Check cache
            if key in self._cache:
                logger.debug("cache_hit", key=key)
                return self._cache[key]
            
            # Compute
            logger.debug("cache_miss", key=key)
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

threads = [threading.Thread(target=worker, args=(f"item-{i}",)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### Casos de Uso Estabelecidos

**Legacy Code Integration:**
- Wrapping bibliotecas síncronas em async contexts
- Migração gradual para async

**I/O-bound com Blocking Libraries:**
- requests, ftplib, smtplib
- Pillow, OpenCV (I/O operations)

**Background Tasks:**
- Email sending
- Report generation
- File processing

---

## multiprocessing - CPU-bound

### Quando Usar

**✅ Use multiprocessing para:**
- Cálculos pesados (CPU-bound)
- Data processing paralelo
- Machine learning training/inference
- Image/video processing (compute-intensive)

**❌ Não use multiprocessing para:**
- I/O-bound → asyncio ou threading são mais leves
- Shared state complexo → overhead de serialização
- Tasks muito pequenas → overhead de process creation

### Sintaxe Básica
```python
from multiprocessing import Pool
from typing import List

def compute_heavy(n: int) -> int:
    """Cálculo CPU-intensive."""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

def parallel_compute(numbers: List[int]) -> List[int]:
    """Processa em paralelo usando múltiplos cores."""
    with Pool() as pool:
        results = pool.map(compute_heavy, numbers)
    return results

# Uso - utiliza todos os CPU cores
numbers = [10_000_000] * 8
results = parallel_compute(numbers)
```

### ProcessPoolExecutor

Forma moderna (interface similar a ThreadPoolExecutor):
```python
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from typing import List
import structlog

logger = structlog.get_logger()

def process_chunk(data: np.ndarray) -> float:
    """Processa chunk de dados (CPU-bound)."""
    # Operações pesadas (matrix multiplication, etc)
    result = np.sum(data ** 2)
    logger.info("chunk_processed", size=len(data), result=result)
    return result

def parallel_data_processing(
    data: np.ndarray,
    num_chunks: int = 4
) -> float:
    """Divide dados e processa em paralelo."""
    
    # Divide dados em chunks
    chunks = np.array_split(data, num_chunks)
    
    # Processa em paralelo
    with ProcessPoolExecutor(max_workers=num_chunks) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        results = [future.result() for future in futures]
    
    # Agrega resultados
    return sum(results)

# Uso
large_array = np.random.rand(10_000_000)
total = parallel_data_processing(large_array, num_chunks=8)
```

### Exemplo do Mundo Real

**ML Inference Paralelo:**
```python
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List
import numpy as np
import structlog

logger = structlog.get_logger()

def load_model():
    """Carrega modelo ML (executado em cada processo)."""
    # Cada processo carrega sua própria cópia do modelo
    import tensorflow as tf
    model = tf.keras.models.load_model("/path/to/model.h5")
    return model

# Global model (um por processo)
_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

def predict_batch(image_paths: List[Path]) -> List[dict]:
    """Processa batch de imagens."""
    model = get_model()
    
    # Load e preprocess images
    images = [preprocess_image(path) for path in image_paths]
    
    # Batch prediction
    predictions = model.predict(np.array(images))
    
    return [
        {"path": str(path), "prediction": pred.tolist()}
        for path, pred in zip(image_paths, predictions)
    ]

def parallel_inference(
    image_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4
) -> List[dict]:
    """Inference paralelo em múltiplos processos."""
    
    # Listar imagens
    image_paths = list(image_dir.glob("*.jpg"))
    
    # Dividir em batches
    batches = [
        image_paths[i:i + batch_size]
        for i in range(0, len(image_paths), batch_size)
    ]
    
    logger.info(
        "starting_inference",
        total_images=len(image_paths),
        num_batches=len(batches),
        workers=num_workers
    )
    
    # Processar batches em paralelo
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(predict_batch, batch) for batch in batches]
        results = [item for future in futures for item in future.result()]
    
    logger.info("inference_complete", total_predictions=len(results))
    return results

# Uso
predictions = parallel_inference(
    Path("/data/images"),
    batch_size=32,
    num_workers=8
)
```

**Data Processing Pipeline:**
```python
from multiprocessing import Pool, cpu_count
from pathlib import Path
import pandas as pd
from typing import List
import structlog

logger = structlog.get_logger()

def process_csv_file(filepath: Path) -> pd.DataFrame:
    """Processa arquivo CSV (CPU-intensive transformations)."""
    df = pd.read_csv(filepath)
    
    # Operações pesadas
    df["processed"] = df["value"].apply(lambda x: expensive_calculation(x))
    df["normalized"] = (df["value"] - df["value"].mean()) / df["value"].std()
    
    logger.info("csv_processed", file=str(filepath), rows=len(df))
    return df

def expensive_calculation(value: float) -> float:
    """Cálculo CPU-intensive."""
    result = 0.0
    for i in range(10000):
        result += (value ** 0.5) * (i % 10)
    return result

def parallel_csv_processing(input_dir: Path, output_file: Path) -> None:
    """Processa múltiplos CSVs em paralelo."""
    
    csv_files = list(input_dir.glob("*.csv"))
    num_workers = cpu_count()
    
    logger.info(
        "starting_processing",
        files=len(csv_files),
        workers=num_workers
    )
    
    # Processar em paralelo
    with Pool(processes=num_workers) as pool:
        dataframes = pool.map(process_csv_file, csv_files)
    
    # Concatenar resultados
    final_df = pd.concat(dataframes, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    
    logger.info(
        "processing_complete",
        total_rows=len(final_df),
        output=str(output_file)
    )

# Uso
parallel_csv_processing(
    Path("/data/raw"),
    Path("/data/processed/combined.csv")
)
```

### Shared Memory (Python 3.8+)

Para compartilhar dados grandes entre processos sem cópia:
```python
from multiprocessing import Process, shared_memory
import numpy as np

def worker_process(shm_name: str, shape: tuple, dtype: np.dtype) -> None:
    """Processo worker que acessa shared memory."""
    # Attach to existing shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    
    # Create numpy array from shared memory
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    
    # Modify array in-place
    array[:] = array ** 2
    
    shm.close()

# Main process
data = np.arange(1000000, dtype=np.float64)

# Create shared memory
shm = shared_memory.SharedMemory(create=True, size=data.nbytes)

# Copy data to shared memory
shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
shared_array[:] = data

# Start worker processes
processes = [
    Process(target=worker_process, args=(shm.name, data.shape, data.dtype))
    for _ in range(4)
]

for p in processes:
    p.start()
for p in processes:
    p.join()

# Read results
result = shared_array.copy()

# Cleanup
shm.close()
shm.unlink()
```

### Casos de Uso Estabelecidos

**Data Science** (pandas, numpy):
- Large dataset processing
- Feature engineering pipelines

**Machine Learning** (scikit-learn, PyTorch):
- Hyperparameter tuning
- Cross-validation
- Batch inference

**Image Processing** (OpenCV, Pillow):
- Video frame processing
- Batch transformations

**Scientific Computing** (scipy, sympy):
- Simulations
- Monte Carlo methods

---

## Comparação de Performance

### Benchmark: I/O-bound (HTTP Requests)
```python
import time
import requests
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def sync_fetch(url: str) -> int:
    """Fetch síncrono."""
    response = requests.get(url)
    return response.status_code

async def async_fetch(url: str) -> int:
    """Fetch assíncrono."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.status_code

# Test URLs
urls = ["https://httpbin.org/delay/1"] * 10

# 1. Sequential (baseline)
start = time.perf_counter()
for url in urls:
    sync_fetch(url)
sequential_time = time.perf_counter() - start
# ~10 segundos (1s * 10)

# 2. Threading
start = time.perf_counter()
with ThreadPoolExecutor(max_workers=10) as executor:
    list(executor.map(sync_fetch, urls))
threading_time = time.perf_counter() - start
# ~1 segundo (paralelo)

# 3. Asyncio
start = time.perf_counter()
asyncio.run(asyncio.gather(*[async_fetch(url) for url in urls]))
asyncio_time = time.perf_counter() - start
# ~1 segundo (paralelo)

# 4. Multiprocessing (overhead, não recomendado para I/O)
start = time.perf_counter()
with ProcessPoolExecutor(max_workers=10) as executor:
    list(executor.map(sync_fetch, urls))
multiprocessing_time = time.perf_counter() - start
# ~2-3 segundos (overhead de processos)
```

**Resultado:**
- **Sequential:** ~10s (baseline)
- **Threading:** ~1s (10x speedup)
- **Asyncio:** ~1s (10x speedup, menos overhead)
- **Multiprocessing:** ~2-3s (overhead sem ganho)

### Benchmark: CPU-bound (Cálculos)
```python
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def cpu_intensive(n: int) -> int:
    """Cálculo CPU-bound."""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

numbers = [10_000_000] * 8

# 1. Sequential
start = time.perf_counter()
for n in numbers:
    cpu_intensive(n)
sequential_time = time.perf_counter() - start
# ~8 segundos (baseline)

# 2. Threading (limitado pelo GIL)
start = time.perf_counter()
with ThreadPoolExecutor(max_workers=8) as executor:
    list(executor.map(cpu_intensive, numbers))
threading_time = time.perf_counter() - start
# ~8 segundos (SEM speedup devido ao GIL)

# 3. Multiprocessing
start = time.perf_counter()
with ProcessPoolExecutor(max_workers=8) as executor:
    list(executor.map(cpu_intensive, numbers))
multiprocessing_time = time.perf_counter() - start
# ~1 segundo (8x speedup em 8 cores)
```

**Resultado:**
- **Sequential:** ~8s (baseline)
- **Threading:** ~8s (GIL impede paralelismo)
- **Multiprocessing:** ~1s (8x speedup real)

---

## Escolhendo o Modelo Certo

### Decision Tree
```
Workload é CPU-bound (cálculos pesados)?
├─ SIM → multiprocessing
└─ NÃO (I/O-bound)
   └─ Biblioteca tem suporte async?
      ├─ SIM → asyncio
      └─ NÃO → threading
```

### Tabela de Decisão

| Cenário | Modelo | Motivo |
|---------|--------|--------|
| 100+ HTTP requests | asyncio | I/O concorrente eficiente |
| Image resize (Pillow) | threading | Biblioteca síncrona, I/O-bound |
| ML training | multiprocessing | CPU-bound pesado |
| Database queries (asyncpg) | asyncio | Driver async disponível |
| Legacy requests library | threading | Biblioteca síncrona |
| Matrix operations | multiprocessing | CPU-bound, bypassa GIL |
| WebSocket connections | asyncio | I/O assíncrono nativo |
| File compression | multiprocessing | CPU-bound |

---

## Best Practices

✅ **Use asyncio como primeira escolha para I/O**
```python
# Prefira asyncio quando possível
async def fetch_data():
    async with httpx.AsyncClient() as client:
        return await client.get(url)
```

✅ **Use executors para mixing sync/async**
```python
# Executar código síncrono em async context
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(None, blocking_function, arg)
```

✅ **Limit worker count adequadamente**
```python
# Threading: I/O-bound pode ter muitos workers
ThreadPoolExecutor(max_workers=100)

# Multiprocessing: CPU-bound = número de cores
ProcessPoolExecutor(max_workers=cpu_count())
```

✅ **Use context managers**
```python
# CORRETO - cleanup garantido
with ThreadPoolExecutor() as executor:
    results = executor.map(func, items)

# EVITE - pode vazar resources
executor = ThreadPoolExecutor()
results = executor.map(func, items)
# Esqueceu executor.shutdown()
```

❌ **Não use multiprocessing para I/O-bound**
```python
# EVITE - overhead sem ganho
with ProcessPoolExecutor() as executor:
    executor.map(requests.get, urls)

# PREFIRA - asyncio ou threading
async with httpx.AsyncClient() as client:
    await asyncio.gather(*[client.get(url) for url in urls])
```

❌ **Não confie em threading para CPU-bound**
```python
# EVITE - GIL impede speedup
with ThreadPoolExecutor() as executor:
    executor.map(cpu_heavy_function, items)

# USE - multiprocessing bypassa GIL
with ProcessPoolExecutor() as executor:
    executor.map(cpu_heavy_function, items)
```

---

## Referências

- [asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [threading Documentation](https://docs.python.org/3/library/threading.html)
- [multiprocessing Documentation](https://docs.python.org/3/library/multiprocessing.html)
- [concurrent.futures Documentation](https://docs.python.org/3/library/concurrent.futures.html)
- [Understanding the GIL](https://realpython.com/python-gil/)
- [PEP 3148](https://peps.python.org/pep-3148/) - futures