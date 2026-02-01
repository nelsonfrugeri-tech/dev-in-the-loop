# Type System Avançado - Python 3.10+

Referência técnica completa do sistema de tipos moderno em Python. Para decisões de quando aplicar cada padrão, consulte a skill principal (`/developer`).

## Fundamentos

Python 3.10+ introduz sintaxe moderna para tipos:
- **Union com `|`**: `str | None` ao invés de `Union[str, None]`
- **Generics built-in**: `list[str]` ao invés de `List[str]`
- **Pattern matching**: `match`/`case` com type narrowing

Este documento cobre recursos avançados para type safety em produção.

---

## Protocol - Structural Subtyping

### Definição

Protocol define interface baseada em estrutura (duck typing), não herança. Classes que implementam os métodos satisfazem o protocolo implicitamente.
```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str: ...

class Circle:
    def draw(self) -> str:
        return "○"

class Square:
    def draw(self) -> str:
        return "□"

def render(shape: Drawable) -> None:
    print(shape.draw())

# Type checker aceita ambos sem herança
render(Circle())  # OK
render(Square())  # OK
```

### Runtime Checking

Por padrão, Protocol é apenas para type checking estático. Use `@runtime_checkable` para validação em runtime:
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Closeable(Protocol):
    def close(self) -> None: ...

class File:
    def close(self) -> None:
        pass

f = File()
isinstance(f, Closeable)  # True (com @runtime_checkable)
```

### Exemplos do Mundo Real

**FastAPI - Dependency Protocol:**
```python
# fastapi/dependencies/utils.py
from typing import Protocol

class Dependable(Protocol):
    async def __call__(self) -> Any: ...

# Qualquer callable async satisfaz o protocolo
async def get_db() -> Database:
    return Database()

# FastAPI aceita porque implementa __call__
@app.get("/users")
async def users(db: Database = Depends(get_db)):
    ...
```

**httpx - Transport Protocol:**
```python
# httpx/_transports/base.py
from typing import Protocol

class AsyncBaseTransport(Protocol):
    async def handle_async_request(self, request: Request) -> Response: ...

# Implementações concretas
class AsyncHTTPTransport:
    async def handle_async_request(self, request: Request) -> Response:
        # HTTP/1.1 implementation
        ...

class AsyncHTTP2Transport:
    async def handle_async_request(self, request: Request) -> Response:
        # HTTP/2 implementation
        ...

# httpx aceita qualquer transport que implemente o protocolo
client = httpx.AsyncClient(transport=AsyncHTTPTransport())
```

**Pydantic - Validator Protocol:**
```python
# pydantic/functional_validators.py
from typing import Protocol, Any

class FieldValidator(Protocol):
    def __call__(self, __value: Any) -> Any: ...

# Qualquer callable satisfaz
def validate_email(value: str) -> str:
    if "@" not in value:
        raise ValueError("Invalid email")
    return value.lower()

# Pydantic aceita como validator
class User(BaseModel):
    email: str
    
    _validate_email = field_validator("email")(validate_email)
```

### Comparação: Protocol vs ABC

| Aspecto | Protocol | ABC |
|---------|----------|-----|
| Acoplamento | Baixo (estrutural) | Alto (herança) |
| Compatibilidade | Código existente | Requer modificação |
| Validação | Estática (mypy) | Runtime (`isinstance`) |
| Quando usar | Bibliotecas, plugins | Hierarquias internas |

### Casos de Uso Estabelecidos

**Bibliotecas públicas** (FastAPI, httpx, Pydantic):
- Aceitar objetos de terceiros sem forçar herança
- Definir contratos de plugins/extensões

**Dependency Injection**:
- Definir interfaces de serviços
- Permitir múltiplas implementações

**Testing**:
- Criar mocks que satisfazem protocolos
- Sem necessidade de herdar classes de teste

---

## TypeVar e Generic - Tipos Parametrizados

### Definição

`TypeVar` cria variáveis de tipo para funções e classes genéricas. `Generic[T]` define classes que aceitam parâmetros de tipo.
```python
from typing import TypeVar, Generic

T = TypeVar("T")

def first(items: list[T]) -> T:
    return items[0]

# Type checker infere o tipo
x: int = first([1, 2, 3])      # T = int
y: str = first(["a", "b"])     # T = str
```

### Generic Classes
```python
from typing import TypeVar, Generic

T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()

# Type-safe usage
int_stack: Stack[int] = Stack()
int_stack.push(1)      # OK
int_stack.push("a")    # Type error
```

### Bounded TypeVar

Restringe tipos aceitos:
```python
from typing import TypeVar

# Aceita apenas int ou float
Numeric = TypeVar("Numeric", int, float)

def add(a: Numeric, b: Numeric) -> Numeric:
    return a + b

add(1, 2)      # OK, retorna int
add(1.0, 2.0)  # OK, retorna float
add("a", "b")  # Type error
```

### Exemplos do Mundo Real

**SQLAlchemy - Generic Query:**
```python
# sqlalchemy/orm/query.py
from typing import TypeVar, Generic

_T = TypeVar("_T")

class Query(Generic[_T]):
    def filter(self, *criterion) -> Query[_T]:
        ...
    
    def first(self) -> _T | None:
        ...
    
    def all(self) -> list[_T]:
        ...

# Type-safe queries
users: Query[User] = session.query(User)
user: User | None = users.filter(User.id == 1).first()
all_users: list[User] = users.all()
```

**FastAPI - Generic Response:**
```python
# fastapi/responses.py
from typing import TypeVar, Generic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class JSONResponse(Generic[T]):
    def __init__(self, content: T) -> None:
        self.content = content
    
    def render(self) -> bytes:
        return self.content.model_dump_json().encode()

# Type-safe responses
class UserResponse(BaseModel):
    id: int
    name: str

response: JSONResponse[UserResponse] = JSONResponse(
    UserResponse(id=1, name="Alice")
)
```

**Repository Pattern:**
```python
from typing import TypeVar, Generic, Protocol

class Entity(Protocol):
    id: str

T = TypeVar("T", bound=Entity)

class Repository(Generic[T]):
    def find_by_id(self, id: str) -> T | None:
        ...
    
    def save(self, entity: T) -> T:
        ...
    
    def delete(self, entity: T) -> None:
        ...

# Type-safe repositories
class User:
    id: str
    name: str

user_repo: Repository[User] = Repository()
user: User | None = user_repo.find_by_id("123")
```

### Casos de Uso Estabelecidos

**ORMs e Query Builders** (SQLAlchemy, Tortoise ORM):
- Queries type-safe que retornam tipos corretos

**Containers genéricos** (Repository, Service layers):
- Reutilizar lógica para diferentes entidades

**API clients** (httpx, aiohttp wrappers):
- Response types baseados no endpoint

---

## Literal - Valores Literais Específicos

### Definição

`Literal` restringe valores a literais específicos, criando tipos mais precisos que strings ou ints genéricos.
```python
from typing import Literal

def set_status(status: Literal["pending", "done", "failed"]) -> None:
    print(f"Status: {status}")

set_status("pending")  # OK
set_status("done")     # OK
set_status("invalid")  # Type error
```

### Union de Literals
```python
from typing import Literal

HttpMethod = Literal["GET", "POST", "PUT", "DELETE"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

def make_request(method: HttpMethod, url: str) -> None:
    ...

def log(level: LogLevel, message: str) -> None:
    ...
```

### Type Narrowing

Type checkers refinam tipos em branches:
```python
from typing import Literal

def process(mode: Literal["sync", "async"]) -> None:
    if mode == "sync":
        # mypy sabe que mode é Literal["sync"]
        run_sync()
    else:
        # mypy sabe que mode é Literal["async"]
        run_async()
```

### Exemplos do Mundo Real

**Typer - Command Arguments:**
```python
# typer/models.py
from typing import Literal

Environment = Literal["dev", "staging", "production"]

@app.command()
def deploy(env: Environment) -> None:
    if env == "production":
        confirm = typer.confirm("Deploy to production?")
        if not confirm:
            raise typer.Abort()
    deploy_to(env)
```

**Pydantic - Discriminated Unions:**
```python
from pydantic import BaseModel, Field
from typing import Literal

class Cat(BaseModel):
    type: Literal["cat"]
    meow: str

class Dog(BaseModel):
    type: Literal["dog"]
    bark: str

Animal = Cat | Dog

def handle_animal(animal: Animal) -> None:
    if animal.type == "cat":
        # Type narrowed to Cat
        print(animal.meow)
    else:
        # Type narrowed to Dog
        print(animal.bark)
```

**FastAPI - Response Status:**
```python
from typing import Literal
from fastapi import HTTPException

def get_user(user_id: str) -> User:
    user = db.get(user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    return user

# Com Literal para status codes
StatusCode = Literal[200, 201, 400, 404, 500]

class Response(BaseModel):
    status: StatusCode
    data: dict
```

### Casos de Uso Estabelecidos

**Enums como strings** (Typer, Click):
- Argumentos de CLI com valores fixos

**Discriminated unions** (Pydantic, dataclasses):
- Diferentes tipos baseados em campo discriminador

**State machines**:
- Estados válidos como literais

**API status codes, HTTP methods**:
- Type safety para constantes

---

## TypedDict - Dicts com Schema

### Definição

`TypedDict` define schema para dicts, permitindo type checking em dicionários.
```python
from typing import TypedDict

class User(TypedDict):
    id: int
    name: str
    email: str

user: User = {"id": 1, "name": "Alice", "email": "alice@example.com"}

# Type checking
print(user["name"])    # OK
print(user["age"])     # Type error: 'age' not in User
```

### Required vs Optional
```python
from typing import TypedDict, NotRequired

class User(TypedDict):
    id: int
    name: str
    email: NotRequired[str]  # Optional field

user1: User = {"id": 1, "name": "Alice"}              # OK
user2: User = {"id": 1, "name": "Bob", "email": "b"}  # OK
```

### Total=False
```python
from typing import TypedDict

class PartialUser(TypedDict, total=False):
    id: int
    name: str
    email: str

# Todos os campos são opcionais
user: PartialUser = {"id": 1}  # OK
```

### Herança
```python
from typing import TypedDict

class BaseEntity(TypedDict):
    id: str
    created_at: str

class User(BaseEntity):
    name: str
    email: str

user: User = {
    "id": "123",
    "created_at": "2024-01-01",
    "name": "Alice",
    "email": "alice@example.com"
}
```

### Exemplos do Mundo Real

**FastAPI - JSON Responses:**
```python
from typing import TypedDict
from fastapi import FastAPI

class UserResponse(TypedDict):
    id: int
    name: str
    email: str

class ErrorResponse(TypedDict):
    error: str
    detail: str

@app.get("/users/{user_id}")
async def get_user(user_id: int) -> UserResponse | ErrorResponse:
    user = db.get(user_id)
    if not user:
        return {"error": "NotFound", "detail": "User not found"}
    return {"id": user.id, "name": user.name, "email": user.email}
```

**Pydantic - Config Dicts:**
```python
from typing import TypedDict
from pydantic import BaseModel

class DatabaseConfig(TypedDict):
    host: str
    port: int
    database: str

class Settings(BaseModel):
    db: DatabaseConfig
    debug: bool = False

config: DatabaseConfig = {
    "host": "localhost",
    "port": 5432,
    "database": "myapp"
}
settings = Settings(db=config)
```

**JSON Schema Validation:**
```python
from typing import TypedDict

class ApiRequest(TypedDict):
    method: str
    url: str
    headers: dict[str, str]
    body: str | None

def validate_request(request: dict) -> ApiRequest:
    # Runtime validation would go here
    return request  # type: ignore
```

### Casos de Uso Estabelecidos

**JSON APIs sem Pydantic**:
- Type hints para payloads JSON
- Alternativa leve a BaseModel

**Config files** (YAML, TOML):
- Schema para configurações carregadas

**Database rows**:
- Type hints para resultados de queries raw

**kwargs estruturados**:
- Type checking em `**kwargs` com schema fixo

---

## Union Moderna (|) - Python 3.10+

### Sintaxe Nova

Python 3.10+ permite `X | Y` ao invés de `Union[X, Y]`:
```python
# Antiga (Python < 3.10)
from typing import Union, Optional
def process(value: Union[str, int]) -> Optional[str]:
    ...

# Moderna (Python 3.10+)
def process(value: str | int) -> str | None:
    ...
```

### Type Narrowing

Type checkers refinam tipos automaticamente:
```python
def process(value: str | int) -> str:
    if isinstance(value, str):
        # mypy sabe que value é str aqui
        return value.upper()
    else:
        # mypy sabe que value é int aqui
        return str(value)
```

### Exemplos do Mundo Real

**FastAPI - Flexible Parameters:**
```python
from fastapi import FastAPI, Query

@app.get("/items")
async def get_items(
    skip: int = 0,
    limit: int | None = None,  # Optional limit
    filter: str | list[str] | None = Query(None)  # String or list
) -> list[Item]:
    ...
```

**Pydantic - Flexible Fields:**
```python
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    content: str
    tags: str | list[str]  # Accept single tag or list
    metadata: dict | None = None

# Works with both
article1 = Article(title="...", content="...", tags="python")
article2 = Article(title="...", content="...", tags=["python", "fastapi"])
```

### Casos de Uso Estabelecidos

**Optional values**: `T | None` mais conciso que `Optional[T]`

**Multiple return types**: Funções que retornam diferentes tipos

**Flexible inputs**: APIs que aceitam formatos variados

---

## NewType - Tipos Distintos

### Definição

`NewType` cria tipos distintos baseados em tipos existentes, prevenindo mistura acidental.
```python
from typing import NewType

UserId = NewType("UserId", int)
OrderId = NewType("OrderId", int)

def get_user(user_id: UserId) -> User:
    ...

def get_order(order_id: OrderId) -> Order:
    ...

user_id = UserId(123)
order_id = OrderId(456)

get_user(user_id)    # OK
get_user(order_id)   # Type error: OrderId não é UserId
get_user(123)        # Type error: int não é UserId
```

### Exemplos do Mundo Real

**Database IDs:**
```python
from typing import NewType

UserId = NewType("UserId", str)
ProductId = NewType("ProductId", str)
OrderId = NewType("OrderId", str)

def link_order_to_user(user_id: UserId, order_id: OrderId) -> None:
    db.execute(
        "INSERT INTO user_orders VALUES (?, ?)",
        user_id, order_id
    )

# Prevents bugs
user = UserId("user_123")
product = ProductId("prod_456")
link_order_to_user(user, product)  # Type error!
```

### Casos de Uso Estabelecidos

**Strongly-typed IDs**: Prevenir mistura de diferentes tipos de IDs

**Units**: Diferenciar valores com mesma representação mas significados diferentes

---

## Final - Valores Imutáveis

### Definição

`Final` indica que valor não deve ser reatribuído:
```python
from typing import Final

MAX_CONNECTIONS: Final = 100
API_VERSION: Final[str] = "v1"

# Type error
MAX_CONNECTIONS = 200
```

### Classes e Métodos Final
```python
from typing import final

@final
class SealedClass:
    """Cannot be subclassed."""
    pass

class Base:
    @final
    def process(self) -> None:
        """Cannot be overridden."""
        pass
```

### Casos de Uso Estabelecidos

**Constants**: Valores que não devem mudar

**Sealed classes**: Prevenir herança indesejada

**Template method pattern**: Métodos que não devem ser sobrescritos

---

## Annotated - Metadados em Tipos

### Definição

`Annotated` adiciona metadados a tipos sem afetar type checking:
```python
from typing import Annotated

# Adiciona metadata
PositiveInt = Annotated[int, "must be positive"]
Username = Annotated[str, "alphanumeric only", "max 20 chars"]

def create_user(age: PositiveInt, name: Username) -> User:
    ...
```

### Exemplos do Mundo Real

**FastAPI - Parameter Validation:**
```python
from typing import Annotated
from fastapi import FastAPI, Query

@app.get("/items")
async def get_items(
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    offset: Annotated[int, Query(ge=0)] = 0
) -> list[Item]:
    ...
```

**Pydantic - Field Constraints:**
```python
from typing import Annotated
from pydantic import BaseModel, Field

class User(BaseModel):
    name: Annotated[str, Field(min_length=3, max_length=50)]
    age: Annotated[int, Field(ge=0, le=150)]
    email: Annotated[str, Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")]
```

### Casos de Uso Estabelecidos

**Validation metadata** (FastAPI, Pydantic):
- Constraints em parâmetros

**Documentation**:
- Metadados para geração de docs

**Custom type checking**:
- Informações adicionais para linters customizados

---

## ParamSpec - Decorators Type-Safe

### Definição

`ParamSpec` preserva assinaturas em decorators:
```python
from typing import ParamSpec, TypeVar, Callable

P = ParamSpec("P")
T = TypeVar("T")

def log_calls(func: Callable[P, T]) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log_calls
def add(a: int, b: int) -> int:
    return a + b

# Type checker preserva assinatura
result: int = add(1, 2)  # OK
add("1", "2")  # Type error: espera int
```

### Exemplos do Mundo Real

**Retry Decorator:**
```python
from typing import ParamSpec, TypeVar, Callable
import functools

P = ParamSpec("P")
T = TypeVar("T")

def retry(times: int) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == times - 1:
                        raise
            raise RuntimeError("Unreachable")
        return wrapper
    return decorator

@retry(times=3)
async def fetch_data(url: str, timeout: int) -> dict:
    ...

# Signature preserved
data: dict = await fetch_data("https://...", timeout=30)
```

### Casos de Uso Estabelecidos

**Decorators genéricos**: Preservar assinaturas complexas

**Wrapper functions**: Type-safe wrappers

---

## Referências

- [PEP 544](https://peps.python.org/pep-0544/) - Protocols
- [PEP 585](https://peps.python.org/pep-0585/) - Generics built-in
- [PEP 604](https://peps.python.org/pep-0604/) - Union com |
- [PEP 612](https://peps.python.org/pep-0612/) - ParamSpec
- [mypy documentation](https://mypy.readthedocs.io/)