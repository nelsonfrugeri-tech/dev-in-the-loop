# Generators e Lazy Evaluation - Python 3.10+

Referência técnica completa de generators e iteradores em Python. Para decisões de quando usar generators vs lists, consulte a skill principal (`/developer`).

## Fundamentos

Generators produzem valores sob demanda (lazy evaluation) ao invés de criar tudo em memória. Benefícios:
- **Memory efficiency**: O(1) vs O(n) em memória
- **Performance**: Começa imediatamente (não aguarda tudo computar)
- **Composability**: Pipeline de transformações
- **Infinite sequences**: Pode representar séries infinitas

**Quando usar:**
- Large datasets (não cabem na memória)
- Streaming/pipeline processing
- Infinite sequences
- File processing linha por linha
- One-pass iteration suficiente

**Quando NÃO usar:**
- Precisa indexing/slicing (`data[5]`)
- Múltiplas iterações sobre mesmos dados
- Precisa `len()` ou `reversed()`
- Dataset pequeno (overhead sem ganho)

---

## Generator Functions

### Definição

Função com `yield` retorna generator, não valor único:
```python
def count_up_to(n: int):
    """Generator que conta até n."""
    i = 0
    while i < n:
        yield i
        i += 1

# Uso
counter = count_up_to(5)
print(type(counter))  # <class 'generator'>

# Iterar
for num in counter:
    print(num)  # 0, 1, 2, 3, 4

# Generator exhausted após primeira iteração
for num in counter:
    print(num)  # Não imprime nada (generator já consumido)
```

### yield vs return
```python
def normal_function():
    """Função normal - retorna uma vez."""
    return [1, 2, 3]

def generator_function():
    """Generator - yields múltiplas vezes."""
    yield 1
    yield 2
    yield 3

# normal_function retorna lista completa
result = normal_function()
print(result)  # [1, 2, 3]

# generator_function retorna generator object
gen = generator_function()
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3
# print(next(gen))  # StopIteration
```

### Exemplo do Mundo Real

**File Processing (Memory Efficient):**
```python
from pathlib import Path
from typing import Iterator

def read_large_file(filepath: Path) -> Iterator[str]:
    """
    Lê arquivo linha por linha (memory efficient).
    
    Vantagem: arquivo de 10GB usa ~1KB de memória.
    """
    with open(filepath) as f:
        for line in f:
            yield line.strip()

def process_logs(log_file: Path) -> Iterator[dict]:
    """
    Processa logs sem carregar tudo na memória.
    """
    for line in read_large_file(log_file):
        if line.startswith("ERROR"):
            parts = line.split("|")
            yield {
                "timestamp": parts[0],
                "level": parts[1],
                "message": parts[2]
            }

# Uso - memory efficient mesmo com arquivo gigante
import structlog
logger = structlog.get_logger()

for error in process_logs(Path("/var/log/app.log")):
    logger.error("log_entry", **error)

# Alternativa SEM generator (ruim):
def read_all_file_bad(filepath: Path) -> list[str]:
    """Carrega tudo na memória (ruim para arquivos grandes)."""
    with open(filepath) as f:
        return [line.strip() for line in f]  # 10GB arquivo = 10GB RAM!
```

---

## Generator Expressions

### Sintaxe

Similar a list comprehension, mas com `()`:
```python
# List comprehension - cria lista completa
squares_list = [x**2 for x in range(1000000)]  # ~8MB de memória

# Generator expression - lazy evaluation
squares_gen = (x**2 for x in range(1000000))   # ~100 bytes de memória

# Uso
print(next(squares_gen))  # 0
print(next(squares_gen))  # 1
print(next(squares_gen))  # 4
```

### Memory Comparison
```python
import sys

# List comprehension
numbers_list = [x for x in range(1000000)]
print(sys.getsizeof(numbers_list))  # ~8MB

# Generator expression
numbers_gen = (x for x in range(1000000))
print(sys.getsizeof(numbers_gen))   # ~120 bytes

# Generator é ~66,000x mais memory efficient!
```

### Exemplo do Mundo Real

**Data Pipeline:**
```python
from typing import Iterator

def load_raw_data(filename: str) -> Iterator[str]:
    """Load raw CSV lines."""
    with open(filename) as f:
        next(f)  # Skip header
        for line in f:
            yield line.strip()

def parse_line(line: str) -> dict:
    """Parse CSV line to dict."""
    parts = line.split(",")
    return {
        "user_id": parts[0],
        "amount": float(parts[1]),
        "timestamp": parts[2]
    }

def filter_large_amounts(records: Iterator[dict]) -> Iterator[dict]:
    """Filter records with amount > 1000."""
    for record in records:
        if record["amount"] > 1000:
            yield record

def transform_currency(records: Iterator[dict], rate: float) -> Iterator[dict]:
    """Convert amounts to different currency."""
    for record in records:
        record["amount"] = record["amount"] * rate
        yield record

# Pipeline composition - memory efficient
raw_lines = load_raw_data("transactions.csv")
records = (parse_line(line) for line in raw_lines)
large_transactions = filter_large_amounts(records)
converted = transform_currency(large_transactions, rate=5.5)

# Process (lazy - só processa quando necessário)
for transaction in converted:
    process_transaction(transaction)

# Arquivo de 10GB processado com memória constante (~1MB)
```

---

## yield from

### Delegação

`yield from` delega para outro generator:
```python
def generator1():
    yield 1
    yield 2

def generator2():
    yield 3
    yield 4

def combined():
    """Combina generators."""
    yield from generator1()
    yield from generator2()

# Uso
for num in combined():
    print(num)  # 1, 2, 3, 4
```

### Exemplo do Mundo Real

**Tree Traversal:**
```python
from typing import Iterator
from dataclasses import dataclass

@dataclass
class TreeNode:
    value: int
    children: list["TreeNode"]

def traverse_tree(node: TreeNode) -> Iterator[int]:
    """Traverse tree depth-first (generator)."""
    yield node.value
    
    for child in node.children:
        yield from traverse_tree(child)

# Uso
root = TreeNode(1, [
    TreeNode(2, [
        TreeNode(4, []),
        TreeNode(5, [])
    ]),
    TreeNode(3, [
        TreeNode(6, [])
    ])
])

for value in traverse_tree(root):
    print(value)  # 1, 2, 4, 5, 3, 6
```

**Flatten Nested Lists:**
```python
from typing import Iterator, Any

def flatten(nested: list) -> Iterator[Any]:
    """Flatten arbitrarily nested list."""
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

# Uso
nested_list = [1, [2, [3, 4], 5], [6, 7]]
flat = list(flatten(nested_list))
print(flat)  # [1, 2, 3, 4, 5, 6, 7]
```

---

## Iterators vs Generators

### Diferença
```python
from typing import Iterator

# Generator (usando yield)
def generator_counter(n: int) -> Iterator[int]:
    """Generator function."""
    i = 0
    while i < n:
        yield i
        i += 1

# Iterator (implementando protocolo)
class IteratorCounter:
    """Iterator class (mais verboso)."""
    
    def __init__(self, n: int):
        self.n = n
        self.i = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> int:
        if self.i >= self.n:
            raise StopIteration
        value = self.i
        self.i += 1
        return value

# Ambos funcionam igual
for i in generator_counter(3):
    print(i)  # 0, 1, 2

for i in IteratorCounter(3):
    print(i)  # 0, 1, 2

# Generator é mais conciso (3 linhas vs 15)
```

### Custom Iterator
```python
from typing import Iterator

class RangeIterator:
    """Custom iterator similar a range()."""
    
    def __init__(self, start: int, end: int, step: int = 1):
        self.current = start
        self.end = end
        self.step = step
    
    def __iter__(self):
        return self
    
    def __next__(self) -> int:
        if self.current >= self.end:
            raise StopIteration
        
        value = self.current
        self.current += self.step
        return value

# Uso
for num in RangeIterator(0, 10, 2):
    print(num)  # 0, 2, 4, 6, 8
```

---

## itertools - Ferramentas Poderosas

### count, cycle, repeat
```python
import itertools

# count - contagem infinita
counter = itertools.count(start=10, step=2)
print(next(counter))  # 10
print(next(counter))  # 12
print(next(counter))  # 14

# cycle - repete sequência infinitamente
colors = itertools.cycle(["red", "green", "blue"])
print(next(colors))  # red
print(next(colors))  # green
print(next(colors))  # blue
print(next(colors))  # red (reinicia)

# repeat - repete valor N vezes
threes = itertools.repeat(3, times=5)
print(list(threes))  # [3, 3, 3, 3, 3]
```

### chain, islice, takewhile
```python
import itertools

# chain - concatena iterables
combined = itertools.chain([1, 2], [3, 4], [5, 6])
print(list(combined))  # [1, 2, 3, 4, 5, 6]

# islice - slice de iterator
numbers = itertools.count()
first_ten = itertools.islice(numbers, 10)
print(list(first_ten))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# takewhile - pega enquanto condição
numbers = itertools.count()
less_than_five = itertools.takewhile(lambda x: x < 5, numbers)
print(list(less_than_five))  # [0, 1, 2, 3, 4]

# dropwhile - pula enquanto condição
numbers = [1, 2, 3, 4, 5, 4, 3, 2, 1]
after_peak = itertools.dropwhile(lambda x: x < 5, numbers)
print(list(after_peak))  # [5, 4, 3, 2, 1]
```

### groupby
```python
import itertools
from typing import Iterator

data = [
    {"name": "Alice", "city": "NYC"},
    {"name": "Bob", "city": "NYC"},
    {"name": "Charlie", "city": "LA"},
    {"name": "David", "city": "LA"},
]

# Agrupar por cidade (precisa estar ordenado)
data.sort(key=lambda x: x["city"])

for city, group in itertools.groupby(data, key=lambda x: x["city"]):
    people = [person["name"] for person in group]
    print(f"{city}: {people}")
# LA: ['Charlie', 'David']
# NYC: ['Alice', 'Bob']
```

### Exemplo do Mundo Real

**Batch Processing:**
```python
import itertools
from typing import Iterator, TypeVar

T = TypeVar("T")

def batch_iterator(iterable: Iterator[T], batch_size: int) -> Iterator[list[T]]:
    """Divide iterator em batches."""
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch

# Uso
def process_users_in_batches(user_ids: Iterator[str]) -> None:
    """Processa usuários em batches de 100."""
    import structlog
    logger = structlog.get_logger()
    
    for batch in batch_iterator(user_ids, batch_size=100):
        logger.info("processing_batch", batch_size=len(batch))
        
        # Process batch
        results = db.bulk_update_users(batch)
        
        logger.info("batch_completed", updated=len(results))

# Memory efficient - processa 1 milhão de users com memória constante
all_user_ids = (user["id"] for user in db.stream_all_users())
process_users_in_batches(all_user_ids)
```

---

## Async Generators

### Definição

Generator assíncrono usa `async def` + `yield`:
```python
import asyncio
from typing import AsyncIterator

async def async_count(n: int) -> AsyncIterator[int]:
    """Async generator."""
    for i in range(n):
        await asyncio.sleep(0.1)  # Simulate async operation
        yield i

# Uso
async def main():
    async for num in async_count(5):
        print(num)

asyncio.run(main())
```

### Exemplo do Mundo Real

**Stream Database Results:**
```python
import asyncio
from typing import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

logger = structlog.get_logger()

async def stream_users(
    db: AsyncSession,
    batch_size: int = 1000
) -> AsyncIterator[dict]:
    """Stream users from database (memory efficient)."""
    offset = 0
    
    while True:
        # Fetch batch
        result = await db.execute(
            select(User).offset(offset).limit(batch_size)
        )
        users = result.scalars().all()
        
        if not users:
            break
        
        logger.info("batch_fetched", count=len(users), offset=offset)
        
        # Yield individual users
        for user in users:
            yield {
                "id": user.id,
                "name": user.name,
                "email": user.email
            }
        
        offset += batch_size

# Uso
async def process_all_users(db: AsyncSession):
    """Process all users without loading all in memory."""
    count = 0
    
    async for user in stream_users(db):
        await send_notification(user)
        count += 1
    
    logger.info("processing_completed", total_users=count)
```

**Stream API Responses:**
```python
import asyncio
import httpx
from typing import AsyncIterator
import structlog

logger = structlog.get_logger()

async def stream_api_pages(
    base_url: str,
    endpoint: str
) -> AsyncIterator[dict]:
    """Stream paginated API results."""
    async with httpx.AsyncClient() as client:
        page = 1
        
        while True:
            logger.info("fetching_page", page=page)
            
            response = await client.get(
                f"{base_url}{endpoint}",
                params={"page": page, "per_page": 100}
            )
            response.raise_for_status()
            
            data = response.json()
            items = data["items"]
            
            if not items:
                break
            
            for item in items:
                yield item
            
            page += 1

# Uso
async def sync_external_data():
    """Sync data from external API."""
    count = 0
    
    async for item in stream_api_pages("https://api.example.com", "/products"):
        await db.upsert_product(item)
        count += 1
    
    logger.info("sync_completed", synced_items=count)
```

---

## Streaming Patterns

### Pipeline Pattern
```python
from typing import Iterator
import structlog

logger = structlog.get_logger()

def load_data(filename: str) -> Iterator[str]:
    """Stage 1: Load raw lines."""
    logger.info("loading_data", filename=filename)
    with open(filename) as f:
        for line in f:
            yield line.strip()

def parse_data(lines: Iterator[str]) -> Iterator[dict]:
    """Stage 2: Parse lines to dicts."""
    for line in lines:
        parts = line.split(",")
        yield {
            "id": parts[0],
            "value": float(parts[1]),
            "timestamp": parts[2]
        }

def filter_data(records: Iterator[dict]) -> Iterator[dict]:
    """Stage 3: Filter invalid records."""
    for record in records:
        if record["value"] > 0:
            yield record

def transform_data(records: Iterator[dict]) -> Iterator[dict]:
    """Stage 4: Transform records."""
    for record in records:
        record["value"] = record["value"] * 1.1
        yield record

def save_data(records: Iterator[dict]) -> int:
    """Stage 5: Save to database."""
    count = 0
    for record in records:
        db.insert(record)
        count += 1
    return count

# Pipeline execution (lazy - memory efficient)
pipeline = (
    load_data("data.csv")
    | parse_data
    | filter_data
    | transform_data
)

# Better syntax with explicit chaining
raw_lines = load_data("data.csv")
parsed = parse_data(raw_lines)
filtered = filter_data(parsed)
transformed = transform_data(filtered)
saved_count = save_data(transformed)

logger.info("pipeline_completed", records_saved=saved_count)
```

### Producer-Consumer Pattern
```python
import asyncio
from typing import AsyncIterator
import structlog

logger = structlog.get_logger()

async def producer(queue: asyncio.Queue, n: int) -> None:
    """Produce items asynchronously."""
    for i in range(n):
        item = f"item-{i}"
        await queue.put(item)
        logger.info("produced", item=item)
        await asyncio.sleep(0.1)
    
    # Signal completion
    await queue.put(None)

async def consumer(queue: asyncio.Queue) -> None:
    """Consume items asynchronously."""
    while True:
        item = await queue.get()
        
        if item is None:
            break
        
        logger.info("consuming", item=item)
        await process_item(item)
        await asyncio.sleep(0.2)

async def main():
    queue = asyncio.Queue(maxsize=10)
    
    await asyncio.gather(
        producer(queue, 20),
        consumer(queue)
    )

asyncio.run(main())
```

---

## Performance Comparisons

### List vs Generator
```python
import time
import sys

# List - eager evaluation
def process_with_list(n: int) -> list[int]:
    return [x ** 2 for x in range(n)]

# Generator - lazy evaluation
def process_with_generator(n: int):
    return (x ** 2 for x in range(n))

n = 10_000_000

# List
start = time.perf_counter()
result_list = process_with_list(n)
list_time = time.perf_counter() - start
list_memory = sys.getsizeof(result_list)

# Generator
start = time.perf_counter()
result_gen = process_with_generator(n)
gen_time = time.perf_counter() - start
gen_memory = sys.getsizeof(result_gen)

print(f"List: {list_time:.3f}s, {list_memory / 1_000_000:.1f}MB")
# List: 0.450s, 80.0MB

print(f"Generator: {gen_time:.6f}s, {gen_memory}bytes")
# Generator: 0.000001s, 112bytes

# Generator é ~450,000x mais rápido (inicialização)
# Generator é ~714,000x mais memory efficient
```

### Real Processing Time
```python
import time

def process_first_10_list(n: int) -> list[int]:
    """Process with list - cria tudo antes."""
    data = [x ** 2 for x in range(n)]  # Aguarda processar tudo
    return data[:10]

def process_first_10_gen(n: int) -> list[int]:
    """Process with generator - lazy."""
    data = (x ** 2 for x in range(n))  # Instantâneo
    return list(itertools.islice(data, 10))  # Só processa 10

n = 10_000_000

# List - processa 10M antes de retornar 10
start = time.perf_counter()
result = process_first_10_list(n)
print(f"List: {time.perf_counter() - start:.3f}s")
# List: 0.450s

# Generator - processa apenas 10
start = time.perf_counter()
result = process_first_10_gen(n)
print(f"Generator: {time.perf_counter() - start:.6f}s")
# Generator: 0.000010s

# Generator é ~45,000x mais rápido (só processa o necessário)
```

---

## Infinite Sequences

### Definição

Generators podem representar sequências infinitas:
```python
import itertools

def fibonacci() -> Iterator[int]:
    """Generate infinite fibonacci sequence."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Uso - pega quantos precisar
fib = fibonacci()
first_10 = list(itertools.islice(fib, 10))
print(first_10)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

### Exemplo do Mundo Real

**ID Generator:**
```python
import itertools
from typing import Iterator
from datetime import datetime

def generate_ids(prefix: str) -> Iterator[str]:
    """Generate infinite unique IDs."""
    counter = itertools.count(1)
    
    while True:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        count = next(counter)
        yield f"{prefix}-{timestamp}-{count:06d}"

# Uso
id_gen = generate_ids("order")

order_id_1 = next(id_gen)  # "order-20260211103045-000001"
order_id_2 = next(id_gen)  # "order-20260211103045-000002"
order_id_3 = next(id_gen)  # "order-20260211103045-000003"
```

---

## Casos de Uso Estabelecidos

### Data Processing
```python
for record in read_large_csv("data.csv"):
    process_record(record)
```

### Log Analysis
```python
for error_line in parse_error_logs("/var/log/app.log"):
    alert_team(error_line)
```

### Streaming APIs
```python
async for event in stream_api_events():
    handle_event(event)
```

### Database Cursors
```python
for row in db.stream_query("SELECT * FROM large_table"):
    transform_row(row)
```

### ETL Pipelines
```python
pipeline = extract() | transform() | load()
```

---

## Best Practices

✅ **Use generators para large datasets**
```python
# CORRETO - memory efficient
def read_file(path):
    with open(path) as f:
        for line in f:
            yield line.strip()

# EVITE - carrega tudo
def read_file_bad(path):
    with open(path) as f:
        return [line.strip() for line in f]
```

✅ **Prefira generator expressions**
```python
# CORRETO
sum(x**2 for x in range(1000000))

# EVITE - desnecessário
sum([x**2 for x in range(1000000)])
```

✅ **Use itertools para operações complexas**
```python
import itertools

# Batching
for batch in itertools.batched(items, 100):
    process_batch(batch)
```

✅ **Documente se generator é one-shot**
```python
def stream_data():
    """
    Stream data from API.
    
    Note: Generator can only be consumed once.
    """
    yield from fetch_data()
```

❌ **Não tente len() em generator**
```python
# ERRO
gen = (x for x in range(10))
# len(gen)  # TypeError

# Se precisa len, não use generator
data = list(gen)
print(len(data))
```

❌ **Não use generator se precisa múltiplas iterações**
```python
# EVITE
gen = (x**2 for x in range(100))
sum1 = sum(gen)
sum2 = sum(gen)  # 0 (exhausted!)

# USE lista
data = [x**2 for x in range(100)]
sum1 = sum(data)
sum2 = sum(data)  # OK
```

---

## Referências

- [Generators Documentation](https://docs.python.org/3/howto/functional.html#generators)
- [PEP 255](https://peps.python.org/pep-0255/) - Simple Generators
- [PEP 342](https://peps.python.org/pep-0342/) - Coroutines via Enhanced Generators
- [itertools Documentation](https://docs.python.org/3/library/itertools.html)
- [Generator Tricks for Systems Programmers](http://www.dabeaz.com/generators/)