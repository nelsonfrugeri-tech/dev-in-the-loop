# Data Classes - Python 3.10+

Referência técnica completa de dataclasses em Python. Para decisões de quando usar dataclass vs Pydantic vs TypedDict, consulte a skill principal (`/developer`).

## Fundamentos

Dataclasses eliminam boilerplate de classes que são principalmente dados. O decorator `@dataclass` gera automaticamente `__init__`, `__repr__`, `__eq__` e outros métodos.

**Quando usar:**
- Classes de domínio (entities, value objects)
- DTOs internos (sem validação externa)
- Configurações estruturadas
- Alternativa leve a Pydantic

**Quando NÃO usar:**
- Validação de dados externos → Use Pydantic
- Schema para JSON/dict → Use TypedDict
- Apenas tupla nomeada → Use NamedTuple

---

## Dataclass Básico

### Definição
```python
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    email: str
    active: bool = True

# Uso
user = User(id=1, name="Alice", email="alice@example.com")

print(user)  # User(id=1, name='Alice', email='alice@example.com', active=True)
print(user.name)  # Alice

# Comparação automática
user2 = User(id=1, name="Alice", email="alice@example.com")
print(user == user2)  # True
```

### Métodos Gerados
```python
@dataclass
class Point:
    x: float
    y: float

# __init__ gerado
p = Point(1.0, 2.0)

# __repr__ gerado
print(p)  # Point(x=1.0, y=2.0)

# __eq__ gerado
p2 = Point(1.0, 2.0)
print(p == p2)  # True

# __hash__ não gerado por padrão (se mutável)
# Para tornar hashable, use frozen=True
```

### Exemplo do Mundo Real

**Domain Models - E-commerce:**
```python
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

@dataclass
class Product:
    id: str
    name: str
    price: Decimal
    stock: int
    created_at: datetime

@dataclass
class OrderItem:
    product_id: str
    quantity: int
    unit_price: Decimal
    
    @property
    def total(self) -> Decimal:
        return self.unit_price * self.quantity

@dataclass
class Order:
    id: str
    customer_id: str
    items: list[OrderItem]
    created_at: datetime
    
    @property
    def total(self) -> Decimal:
        return sum(item.total for item in self.items)

# Uso
item1 = OrderItem("prod-1", quantity=2, unit_price=Decimal("10.99"))
item2 = OrderItem("prod-2", quantity=1, unit_price=Decimal("5.50"))

order = Order(
    id="order-123",
    customer_id="user-456",
    items=[item1, item2],
    created_at=datetime.now()
)

print(f"Order total: ${order.total}")  # Order total: $27.48
```

---

## frozen=True - Imutabilidade

### Definição

`frozen=True` torna instância imutável e hashable.
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x: float
    y: float

p = Point(1.0, 2.0)
# p.x = 3.0  # FrozenInstanceError

# Agora é hashable
points_set = {Point(1.0, 2.0), Point(3.0, 4.0)}
```

### Exemplo do Mundo Real

**Value Objects (DDD):**
```python
from dataclasses import dataclass
from decimal import Decimal

@dataclass(frozen=True)
class Money:
    """Value object para dinheiro - sempre imutável."""
    amount: Decimal
    currency: str
    
    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("Amount cannot be negative")
        if self.currency not in ("USD", "EUR", "BRL"):
            raise ValueError(f"Invalid currency: {self.currency}")
    
    def add(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)
    
    def multiply(self, factor: int | float) -> "Money":
        return Money(self.amount * Decimal(str(factor)), self.currency)

# Uso
price = Money(Decimal("10.99"), "USD")
total = price.multiply(3)  # Cria nova instância
print(total)  # Money(amount=Decimal('32.97'), currency='USD')

# Pode usar em dicts/sets
prices = {
    Money(Decimal("10.00"), "USD"): "Basic",
    Money(Decimal("20.00"), "USD"): "Premium"
}
```

**Coordinates e Geometry:**
```python
from dataclasses import dataclass
from math import sqrt

@dataclass(frozen=True)
class Point2D:
    x: float
    y: float
    
    def distance_to(self, other: "Point2D") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return sqrt(dx**2 + dy**2)
    
    def translate(self, dx: float, dy: float) -> "Point2D":
        return Point2D(self.x + dx, self.y + dy)

@dataclass(frozen=True)
class Rectangle:
    top_left: Point2D
    bottom_right: Point2D
    
    @property
    def width(self) -> float:
        return self.bottom_right.x - self.top_left.x
    
    @property
    def height(self) -> float:
        return self.bottom_right.y - self.top_left.y
    
    @property
    def area(self) -> float:
        return self.width * self.height

# Uso imutável
p1 = Point2D(0.0, 0.0)
p2 = p1.translate(10.0, 5.0)  # Novo objeto
print(p2)  # Point2D(x=10.0, y=5.0)
```

### Casos de Uso Estabelecidos

**Value Objects** (DDD):
- Money, Email, PhoneNumber
- Coordinates, Dimensions
- Sempre imutáveis por design

**Keys em Dicts/Sets**:
- Precisa ser hashable
- Identidade por valor

**Thread-safe Data**:
- Compartilhar entre threads sem locks
- Imutabilidade garante segurança

---

## slots=True - Otimização de Memória

### Definição

`slots=True` usa `__slots__` para reduzir uso de memória e aumentar velocidade de acesso.
```python
from dataclasses import dataclass

@dataclass(slots=True)
class Point:
    x: float
    y: float

# Sem slots: ~152 bytes por instância
# Com slots: ~64 bytes por instância (economia ~58%)
```

### Comparação de Memória
```python
import sys
from dataclasses import dataclass

@dataclass
class PointNoSlots:
    x: float
    y: float

@dataclass(slots=True)
class PointWithSlots:
    x: float
    y: float

p1 = PointNoSlots(1.0, 2.0)
p2 = PointWithSlots(1.0, 2.0)

print(sys.getsizeof(p1.__dict__))  # ~240 bytes (dict overhead)
print(sys.getsizeof(p2))           # ~64 bytes (sem dict)
```

### Exemplo do Mundo Real

**High-Volume Data Processing:**
```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class LogEntry:
    """Entrada de log - milhões em memória."""
    timestamp: float
    level: str
    message: str
    source: str

# Processar milhões de logs
logs: list[LogEntry] = []
for line in read_large_log_file():
    logs.append(parse_log_line(line))

# Com slots: ~40% menos memória
# Importante para datasets grandes
```

**Particle Simulation:**
```python
from dataclasses import dataclass

@dataclass(slots=True)
class Particle:
    """Partícula em simulação física."""
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    mass: float

# Simular 1 milhão de partículas
particles = [
    Particle(x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, mass=1.0)
    for _ in range(1_000_000)
]
# slots=True economiza ~300MB neste caso
```

### Casos de Uso Estabelecidos

**Large Collections**:
- Milhões de instâncias em memória
- Data processing, ML datasets

**Performance-Critical Code**:
- Game engines
- Simulations
- Real-time systems

**Restrição: Não pode adicionar atributos dinamicamente**
```python
@dataclass(slots=True)
class Point:
    x: float
    y: float

p = Point(1.0, 2.0)
# p.z = 3.0  # AttributeError: 'Point' object has no attribute 'z'
```

---

## field() - Controle Fino

### default_factory

Para valores mutáveis default:
```python
from dataclasses import dataclass, field

@dataclass
class User:
    name: str
    tags: list[str] = field(default_factory=list)  # Correto
    # tags: list[str] = []  # ERRADO - compartilhado entre instâncias!

# Cada instância tem sua própria lista
user1 = User("Alice")
user2 = User("Bob")

user1.tags.append("admin")
print(user1.tags)  # ["admin"]
print(user2.tags)  # [] - lista separada
```

### init=False

Campo calculado, não aceito no `__init__`:
```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Order:
    items: list[str]
    created_at: datetime = field(default_factory=datetime.now, init=False)

# created_at gerado automaticamente
order = Order(items=["item1", "item2"])
print(order.created_at)  # Timestamp de criação
```

### repr=False, compare=False

Controla geração de métodos:
```python
from dataclasses import dataclass, field

@dataclass
class User:
    id: int
    name: str
    password_hash: str = field(repr=False)  # Não aparece no repr
    internal_state: dict = field(default_factory=dict, compare=False)

user = User(id=1, name="Alice", password_hash="secret123")
print(user)  # User(id=1, name='Alice') - sem password_hash
```

### Exemplo do Mundo Real

**Audit Trail:**
```python
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

@dataclass
class AuditedEntity:
    """Entidade com metadados de auditoria."""
    name: str
    data: dict
    
    # Metadados gerados automaticamente
    id: str = field(default_factory=lambda: str(uuid4()), init=False)
    created_at: datetime = field(default_factory=datetime.now, init=False)
    updated_at: datetime = field(default_factory=datetime.now, init=False)
    version: int = field(default=1, init=False)
    
    # Não comparar metadados de auditoria
    _metadata: dict = field(
        default_factory=dict,
        compare=False,
        repr=False
    )

# Criar entidade
entity = AuditedEntity(name="Product", data={"price": 10.99})
print(entity.id)  # UUID gerado
print(entity.created_at)  # Timestamp atual
```

**Configuration with Validation:**
```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ServerConfig:
    host: str
    port: int
    workers: int = 4
    
    # Campos derivados
    base_url: str = field(init=False)
    log_dir: Path = field(default_factory=lambda: Path("/var/log"), init=False)
    
    def __post_init__(self) -> None:
        # Validação
        if not 1024 <= self.port <= 65535:
            raise ValueError(f"Invalid port: {self.port}")
        if self.workers < 1:
            raise ValueError("Workers must be >= 1")
        
        # Calcular campos derivados
        self.base_url = f"http://{self.host}:{self.port}"

config = ServerConfig(host="localhost", port=8080)
print(config.base_url)  # http://localhost:8080
```

---

## __post_init__ - Validação e Processamento

### Definição

`__post_init__` executa após `__init__`, permitindo validação e processamento.
```python
from dataclasses import dataclass

@dataclass
class Rectangle:
    width: float
    height: float
    
    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("Width must be positive")
        if self.height <= 0:
            raise ValueError("Height must be positive")

# Validação automática
rect = Rectangle(10.0, 5.0)  # OK
# rect = Rectangle(-10.0, 5.0)  # ValueError
```

### Exemplo do Mundo Real

**Email Value Object:**
```python
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Email:
    """Email validado e normalizado."""
    address: str
    
    def __post_init__(self) -> None:
        # Validação
        if "@" not in self.address:
            raise ValueError(f"Invalid email: {self.address}")
        
        pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        if not re.match(pattern, self.address):
            raise ValueError(f"Invalid email format: {self.address}")
        
        # Normalização (frozen requer object.__setattr__)
        normalized = self.address.lower().strip()
        object.__setattr__(self, "address", normalized)
    
    @property
    def domain(self) -> str:
        return self.address.split("@")[1]

# Uso
email = Email("  Alice@EXAMPLE.com  ")
print(email.address)  # alice@example.com (normalizado)
print(email.domain)   # example.com
```

**Computed Fields:**
```python
from dataclasses import dataclass, field

@dataclass
class Product:
    name: str
    price_cents: int
    
    # Campo calculado
    price_formatted: str = field(init=False)
    
    def __post_init__(self) -> None:
        dollars = self.price_cents / 100
        self.price_formatted = f"${dollars:.2f}"

product = Product(name="Widget", price_cents=1299)
print(product.price_formatted)  # $12.99
```

**Normalization and Defaults:**
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SearchQuery:
    term: str
    page: int = 1
    limit: int = 10
    
    def __post_init__(self) -> None:
        # Normalização
        self.term = self.term.strip().lower()
        
        # Validação
        if not self.term:
            raise ValueError("Search term cannot be empty")
        
        # Ajustes
        if self.page < 1:
            self.page = 1
        if self.limit < 1:
            self.limit = 10
        if self.limit > 100:
            self.limit = 100

query = SearchQuery("  Python  ", page=0, limit=500)
print(query.term)   # python
print(query.page)   # 1
print(query.limit)  # 100
```

---

## Herança

### Definição

Dataclasses suportam herança, com campos da base classe primeiro.
```python
from dataclasses import dataclass

@dataclass
class Animal:
    name: str
    age: int

@dataclass
class Dog(Animal):
    breed: str

# Ordem: name, age, breed
dog = Dog(name="Rex", age=3, breed="Labrador")
print(dog)  # Dog(name='Rex', age=3, breed='Labrador')
```

### Exemplo do Mundo Real

**Domain Entities:**
```python
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

@dataclass
class BaseEntity:
    """Entidade base com campos comuns."""
    id: str
    created_at: datetime
    updated_at: datetime

@dataclass
class User(BaseEntity):
    """Usuário do sistema."""
    email: str
    name: str
    is_active: bool = True

@dataclass
class Product(BaseEntity):
    """Produto do catálogo."""
    name: str
    price: float
    stock: int

# Criar entidades com campos base
user = User(
    id=str(uuid4()),
    created_at=datetime.now(),
    updated_at=datetime.now(),
    email="alice@example.com",
    name="Alice"
)
```

**Event Hierarchy:**
```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class EventType(Enum):
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    ORDER_PLACED = "order.placed"

@dataclass
class BaseEvent:
    """Evento base."""
    event_id: str
    event_type: EventType
    timestamp: datetime

@dataclass
class UserCreatedEvent(BaseEvent):
    """Evento de criação de usuário."""
    user_id: str
    email: str
    name: str

@dataclass
class OrderPlacedEvent(BaseEvent):
    """Evento de pedido criado."""
    order_id: str
    user_id: str
    total_amount: float
    items_count: int

# Polimorfismo
def handle_event(event: BaseEvent) -> None:
    if isinstance(event, UserCreatedEvent):
        send_welcome_email(event.user_id, event.email)
    elif isinstance(event, OrderPlacedEvent):
        process_order(event.order_id)
```

---

## Comparação com Alternativas

### Dataclass vs NamedTuple
```python
from dataclasses import dataclass
from typing import NamedTuple

# NamedTuple - imutável, herda de tuple
class PointTuple(NamedTuple):
    x: float
    y: float

# Dataclass - mutável por padrão, mais flexível
@dataclass
class PointClass:
    x: float
    y: float

# Dataclass frozen - similar a NamedTuple
@dataclass(frozen=True)
class PointFrozen:
    x: float
    y: float
```

**Quando usar cada:**

| Aspecto | NamedTuple | Dataclass | Dataclass(frozen=True) |
|---------|------------|-----------|------------------------|
| Mutabilidade | Imutável | Mutável | Imutável |
| Métodos | Pode adicionar | Pode adicionar | Pode adicionar |
| Herança | Limitada | Completa | Completa |
| Performance | Mais rápido | Normal | Normal |
| Memory | Menor | Maior | Maior |
| Use quando | Tuplas simples | Objetos complexos | Value objects |

### Dataclass vs TypedDict
```python
from dataclasses import dataclass
from typing import TypedDict

# TypedDict - apenas type hint para dict
class UserDict(TypedDict):
    id: int
    name: str

# Dataclass - classe real
@dataclass
class UserClass:
    id: int
    name: str

# TypedDict é apenas dict em runtime
user_dict: UserDict = {"id": 1, "name": "Alice"}
print(type(user_dict))  # <class 'dict'>

# Dataclass é classe própria
user_class = UserClass(id=1, name="Alice")
print(type(user_class))  # <class '__main__.UserClass'>
```

**Quando usar cada:**

| Use Case | TypedDict | Dataclass |
|----------|-----------|-----------|
| JSON/API payloads | ✅ Ideal | ❌ Overhead |
| Objetos de domínio | ❌ Sem métodos | ✅ Ideal |
| Database rows | ✅ Leve | Depende |
| Config files | ✅ Simples | ❌ Complexo |

### Dataclass vs Pydantic
```python
from dataclasses import dataclass
from pydantic import BaseModel, field_validator

# Dataclass - sem validação runtime
@dataclass
class UserDataclass:
    id: int
    email: str

# Pydantic - validação automática
class UserPydantic(BaseModel):
    id: int
    email: str
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email")
        return v

# Dataclass aceita qualquer coisa (type checker reclama)
user_dc = UserDataclass(id=1, email="invalid")  # OK em runtime

# Pydantic valida
# user_p = UserPydantic(id=1, email="invalid")  # ValidationError
```

**Quando usar cada:**

| Use Case | Dataclass | Pydantic |
|----------|-----------|----------|
| Dados internos | ✅ Leve | ❌ Overhead |
| API externa | ❌ Sem validação | ✅ Ideal |
| Config de produção | ❌ Sem validação | ✅ Ideal |
| Domain models | ✅ Simples | Depende |
| Performance crítica | ✅ Mais rápido | ❌ Mais lento |

---

## Serialization

### asdict() e astuple()
```python
from dataclasses import dataclass, asdict, astuple

@dataclass
class User:
    id: int
    name: str
    email: str

user = User(id=1, name="Alice", email="alice@example.com")

# Converter para dict
user_dict = asdict(user)
print(user_dict)  # {'id': 1, 'name': 'Alice', 'email': 'alice@example.com'}

# Converter para tuple
user_tuple = astuple(user)
print(user_tuple)  # (1, 'Alice', 'alice@example.com')
```

### Nested Dataclasses
```python
from dataclasses import dataclass, asdict

@dataclass
class Address:
    street: str
    city: str
    country: str

@dataclass
class Person:
    name: str
    address: Address

person = Person(
    name="Alice",
    address=Address(street="123 Main St", city="NYC", country="USA")
)

# asdict recursivo
person_dict = asdict(person)
print(person_dict)
# {
#     'name': 'Alice',
#     'address': {'street': '123 Main St', 'city': 'NYC', 'country': 'USA'}
# }
```

### Exemplo do Mundo Real

**JSON API Response:**
```python
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class Product:
    id: str
    name: str
    price: float
    created_at: datetime

def serialize_product(product: Product) -> str:
    """Serializa produto para JSON."""
    data = asdict(product)
    # Converter datetime para ISO string
    data["created_at"] = product.created_at.isoformat()
    return json.dumps(data)

product = Product(
    id="prod-123",
    name="Widget",
    price=19.99,
    created_at=datetime.now()
)

json_str = serialize_product(product)
print(json_str)  # {"id": "prod-123", "name": "Widget", ...}
```

---

## Casos de Uso Estabelecidos

### Domain Models (DDD)
```python
@dataclass
class Order:
    """Aggregate root."""
    id: str
    customer_id: str
    items: list[OrderItem]
    status: OrderStatus
```

### Value Objects
```python
@dataclass(frozen=True)
class Money:
    """Immutable value object."""
    amount: Decimal
    currency: str
```

### DTOs (Data Transfer Objects)
```python
@dataclass
class UserDTO:
    """Transfer data between layers."""
    id: int
    name: str
    email: str
```

### Configuration
```python
@dataclass
class DatabaseConfig:
    host: str
    port: int
    database: str
    max_connections: int = 10
```

### Events / Messages
```python
@dataclass
class UserCreatedEvent:
    user_id: str
    email: str
    timestamp: datetime
```

---

## Best Practices

✅ **Use frozen=True para value objects**
```python
@dataclass(frozen=True)
class Email:
    address: str
```

✅ **Use slots=True para grandes coleções**
```python
@dataclass(slots=True)
class Particle:
    x: float
    y: float
    z: float
```

✅ **Use default_factory para mutáveis**
```python
@dataclass
class User:
    tags: list[str] = field(default_factory=list)
```

✅ **Valide em __post_init__**
```python
def __post_init__(self) -> None:
    if self.age < 0:
        raise ValueError("Age cannot be negative")
```

❌ **Não use [] como default**
```python
# ERRADO
@dataclass
class User:
    tags: list[str] = []  # Compartilhado!

# CORRETO
@dataclass
class User:
    tags: list[str] = field(default_factory=list)
```

---

## Referências

- [PEP 557](https://peps.python.org/pep-0557/) - Data Classes
- [dataclasses Documentation](https://docs.python.org/3/library/dataclasses.html)
- [attrs](https://www.attrs.org/) - Biblioteca que inspirou dataclasses
- [Pydantic](https://docs.pydantic.dev/) - Quando precisa validação