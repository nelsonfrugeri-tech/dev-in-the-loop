# Multi-Agent Systems - Agent Collaboration Patterns

Referência completa para sistemas multi-agent: comunicação, orchestration, patterns de colaboração.

---

## Por Que Multi-Agent?

### Vantagens

1. **Especialização** - Cada agent é expert em uma área (research, coding, review)
2. **Paralelização** - Agents executam tarefas concorrentemente via `asyncio`
3. **Separation of concerns** - Responsabilidades claras e isoladas
4. **Scalability** - Adicionar agents conforme necessidade sem reescrever
5. **Robustez** - Fallback agents quando um falha

### Desvantagens (complexidade)

1. **Debugging difícil** - Rastrear mensagens entre múltiplos agents
2. **Overhead de comunicação** - Latência adicional em cada hop
3. **State consistency** - Manter estado sincronizado é complexo
4. **Custo** - Mais LLM calls = mais tokens = mais custo
5. **Failure modes** - Mais pontos de falha possíveis

### Single Agent Limitations

- Context window finita (não cabe tudo)
- Sem especialização (jack of all trades, master of none)
- Sem paralelização (sequential por natureza)
- Difícil escalar complexidade além de certo ponto

### Decision Tree

```
Precisa de múltiplos agents?
  ├─ Não, single agent suficiente → Use single agent
  └─ Sim:
      ├─ Sequential workflow simples? → CrewAI
      ├─ Complex state/routing? → LangGraph
      ├─ Debate/discussion pattern? → Autogen
      ├─ Parallel execution? → Custom com asyncio.gather
      └─ Máximo controle? → Custom implementation
```

---

## Communication Patterns

### Sequential (Chain)

Agent A finaliza, passa resultado para B, que passa para C.

```python
from typing import Any
from dataclasses import dataclass
from anthropic import AsyncAnthropic
import structlog

logger = structlog.get_logger()


@dataclass
class AgentResult:
    """Resultado de um agent."""
    agent_id: str
    output: str
    metadata: dict[str, Any]


async def sequential_pipeline(
    task: str,
    agents: list["BaseAgent"],
) -> AgentResult:
    """Executa agents em sequência, passando output como input.

    Args:
        task: Tarefa inicial
        agents: Lista de agents na ordem de execução

    Returns:
        Resultado do último agent
    """
    current_input = task

    for agent in agents:
        logger.info("agent_executing", agent_id=agent.agent_id, step=agent.agent_id)

        result = await agent.run(current_input)
        current_input = result.output

        logger.info("agent_completed", agent_id=agent.agent_id, output_length=len(result.output))

    return result
```

### Parallel (Concurrent)

Múltiplos agents executam ao mesmo tempo via `asyncio.gather`.

```python
import asyncio
from anthropic import AsyncAnthropic


async def parallel_agents(
    queries: list[str],
    llm: AsyncAnthropic,
    *,
    timeout: float = 30.0,
) -> list[str]:
    """Executa múltiplos agents em paralelo.

    Args:
        queries: Lista de queries para agents independentes
        llm: Cliente Anthropic
        timeout: Timeout por agent em segundos

    Returns:
        Lista de resultados na mesma ordem das queries
    """
    async def single_agent(query: str) -> str:
        response = await asyncio.wait_for(
            llm.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                messages=[{"role": "user", "content": query}],
            ),
            timeout=timeout,
        )
        return response.content[0].text

    results = await asyncio.gather(
        *[single_agent(q) for q in queries],
        return_exceptions=True,
    )

    processed: list[str] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("parallel_agent_failed", query=queries[i], error=str(result))
            processed.append(f"Error: {result}")
        else:
            processed.append(result)

    return processed
```

### Hierarchical (Manager + Workers)

Manager delega tarefas para workers especializados.

```python
from pydantic import BaseModel, Field
from typing import Literal
import uuid


class TaskAssignment(BaseModel):
    """Tarefa delegada pelo manager."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    worker_id: str
    instruction: str
    priority: Literal["low", "medium", "high"] = "medium"


class ManagerAgent:
    """Manager que delega tarefas para workers."""

    def __init__(self, agent_id: str, llm: AsyncAnthropic):
        self.agent_id = agent_id
        self.llm = llm
        self.workers: dict[str, "WorkerAgent"] = {}

    def register_worker(self, worker: "WorkerAgent") -> None:
        self.workers[worker.agent_id] = worker

    async def delegate(self, task: str) -> dict[str, str]:
        """Analisa task, delega para workers, coleta resultados.

        Args:
            task: Descrição da tarefa

        Returns:
            Dict de worker_id → resultado
        """
        # Manager decide quais workers usar
        plan_response = await self.llm.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"Workers disponíveis: {list(self.workers.keys())}\n"
                    f"Tarefa: {task}\n"
                    f"Retorne JSON com assignments: "
                    f'[{{"worker_id": "...", "instruction": "..."}}]'
                ),
            }],
        )

        # Parse assignments e delega
        # (Simplificado - em produção use structured output)
        assignments = self._parse_assignments(plan_response.content[0].text)

        # Executa workers em paralelo
        tasks_coro = []
        for assignment in assignments:
            worker = self.workers.get(assignment.worker_id)
            if worker:
                tasks_coro.append(worker.run(assignment.instruction))

        results = await asyncio.gather(*tasks_coro, return_exceptions=True)

        return {
            a.worker_id: r.output if not isinstance(r, Exception) else f"Error: {r}"
            for a, r in zip(assignments, results)
        }

    def _parse_assignments(self, text: str) -> list[TaskAssignment]:
        """Parse LLM response into assignments."""
        import json

        try:
            data = json.loads(text)
            return [TaskAssignment(**item) for item in data]
        except (json.JSONDecodeError, ValueError):
            return []
```

### Peer-to-Peer

Agents se comunicam diretamente sem coordenador central.

```python
class PeerAgent:
    """Agent que se comunica diretamente com peers."""

    def __init__(self, agent_id: str, llm: AsyncAnthropic):
        self.agent_id = agent_id
        self.llm = llm
        self.inbox: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.peers: dict[str, "PeerAgent"] = {}

    def connect(self, peer: "PeerAgent") -> None:
        """Conecta com outro agent."""
        self.peers[peer.agent_id] = peer
        peer.peers[self.agent_id] = self

    async def send(self, recipient_id: str, content: dict[str, Any]) -> None:
        """Envia mensagem para peer."""
        peer = self.peers.get(recipient_id)
        if not peer:
            raise ValueError(f"Peer {recipient_id} not connected")

        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient_id,
            message_type="request",
            content=content,
        )
        await peer.inbox.put(message)

    async def receive(self, timeout: float = 10.0) -> AgentMessage | None:
        """Recebe próxima mensagem da inbox."""
        try:
            return await asyncio.wait_for(self.inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
```

---

## Message Passing Protocols

### Message Format

```python
from pydantic import BaseModel, Field
from typing import Literal, Any
from datetime import datetime
import uuid


class AgentMessage(BaseModel):
    """Protocolo de mensagem entre agents."""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = Field(description="Agent ID remetente")
    recipient: str = Field(description="Agent ID destinatário")
    message_type: Literal["request", "response", "notification", "error"]
    content: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = Field(
        default=None,
        description="Para parear request/response",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ttl: int = Field(default=300, description="Time-to-live em segundos")


class AgentError(BaseModel):
    """Mensagem de erro entre agents."""

    error_type: str
    message: str
    agent_id: str
    original_message_id: str | None = None
```

### Async Message Queues

```python
import asyncio
from collections import defaultdict


class MessageBroker:
    """Broker centralizado para routing de mensagens."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[AgentMessage]] = defaultdict(asyncio.Queue)
        self._subscribers: dict[str, list[str]] = defaultdict(list)

    async def publish(self, message: AgentMessage) -> None:
        """Publica mensagem na queue do destinatário."""
        await self._queues[message.recipient].put(message)

        logger.info(
            "message_published",
            sender=message.sender,
            recipient=message.recipient,
            message_type=message.message_type,
        )

    async def subscribe(self, agent_id: str, topic: str) -> None:
        """Inscreve agent em um tópico."""
        self._subscribers[topic].append(agent_id)

    async def broadcast_topic(self, topic: str, message: AgentMessage) -> None:
        """Broadcast para todos inscritos no tópico."""
        for agent_id in self._subscribers.get(topic, []):
            msg = message.model_copy(update={"recipient": agent_id})
            await self._queues[agent_id].put(msg)

    async def consume(
        self,
        agent_id: str,
        timeout: float = 30.0,
    ) -> AgentMessage | None:
        """Consome próxima mensagem da queue do agent."""
        try:
            return await asyncio.wait_for(
                self._queues[agent_id].get(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return None
```

### Request/Response Pattern

```python
class RequestResponseAgent:
    """Agent com pattern request/response usando futures."""

    def __init__(self, agent_id: str, broker: MessageBroker):
        self.agent_id = agent_id
        self.broker = broker
        self._pending: dict[str, asyncio.Future[AgentMessage]] = {}

    async def request(
        self,
        recipient: str,
        content: dict[str, Any],
        *,
        timeout: float = 30.0,
    ) -> AgentMessage:
        """Envia request e aguarda response.

        Args:
            recipient: Agent ID destinatário
            content: Conteúdo da mensagem
            timeout: Timeout em segundos

        Returns:
            Mensagem de resposta

        Raises:
            asyncio.TimeoutError: Se timeout expirar
        """
        correlation_id = str(uuid.uuid4())
        future: asyncio.Future[AgentMessage] = asyncio.get_event_loop().create_future()
        self._pending[correlation_id] = future

        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type="request",
            content=content,
            correlation_id=correlation_id,
        )
        await self.broker.publish(message)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending.pop(correlation_id, None)

    async def handle_response(self, message: AgentMessage) -> None:
        """Processa response recebida."""
        if message.correlation_id and message.correlation_id in self._pending:
            self._pending[message.correlation_id].set_result(message)
```

---

## Shared State vs Isolated State

### When to Share State

| Cenário | State | Justificativa |
|---------|-------|---------------|
| Agents trabalham no mesmo documento | **Shared** | Precisam ver edições um do outro |
| Agents fazem pesquisas independentes | **Isolated** | Resultados independentes |
| Pipeline sequencial | **Shared** | Output vira input do próximo |
| Agents fazem debate/review | **Shared** | Precisam ver argumentos |
| Workers paralelos independentes | **Isolated** | Sem dependência entre eles |

### State Synchronization

```python
import asyncio
from typing import Any
from copy import deepcopy


class SharedState:
    """Thread-safe shared state para multi-agent."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._version = 0

    async def get(self, key: str, default: Any = None) -> Any:
        """Lê valor do state."""
        async with self._lock:
            return deepcopy(self._state.get(key, default))

    async def set(self, key: str, value: Any) -> int:
        """Escreve valor no state. Retorna versão."""
        async with self._lock:
            self._state[key] = deepcopy(value)
            self._version += 1
            return self._version

    async def update(self, key: str, updater: callable) -> Any:
        """Atomic read-modify-write."""
        async with self._lock:
            current = self._state.get(key)
            new_value = updater(deepcopy(current))
            self._state[key] = new_value
            self._version += 1
            return deepcopy(new_value)

    async def snapshot(self) -> dict[str, Any]:
        """Retorna cópia imutável do state completo."""
        async with self._lock:
            return deepcopy(self._state)
```

### Immutable State Pattern

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ImmutableState:
    """State imutável — cada modificação cria nova instância."""

    messages: tuple[str, ...] = ()
    results: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: str) -> "ImmutableState":
        return ImmutableState(
            messages=(*self.messages, message),
            results=self.results,
            metadata=self.metadata,
        )

    def add_result(self, result: dict[str, Any]) -> "ImmutableState":
        return ImmutableState(
            messages=self.messages,
            results=(*self.results, result),
            metadata=self.metadata,
        )
```

---

## Orchestration Patterns

### Centralized Orchestrator

```python
import asyncio
from enum import Enum
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()


class AgentStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"


class Orchestrator:
    """Orquestrador centralizado para sistema multi-agent."""

    def __init__(self) -> None:
        self.agents: dict[str, "BaseAgent"] = {}
        self.status: dict[str, AgentStatus] = {}
        self.message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()

    def register(self, agent: "BaseAgent") -> None:
        self.agents[agent.agent_id] = agent
        self.status[agent.agent_id] = AgentStatus.IDLE

    async def dispatch(self, message: AgentMessage) -> AgentMessage | None:
        """Despacha mensagem para agent, gerenciando status."""
        agent = self.agents.get(message.recipient)
        if not agent:
            logger.error("agent_not_found", agent_id=message.recipient)
            return None

        if self.status[message.recipient] == AgentStatus.BUSY:
            logger.warning("agent_busy", agent_id=message.recipient)
            await self.message_queue.put(message)  # Re-queue
            return None

        self.status[message.recipient] = AgentStatus.BUSY

        try:
            result = await asyncio.wait_for(
                agent.process(message),
                timeout=60.0,
            )
            self.status[message.recipient] = AgentStatus.IDLE
            return result

        except asyncio.TimeoutError:
            logger.error("agent_timeout", agent_id=message.recipient)
            self.status[message.recipient] = AgentStatus.ERROR
            return None

        except Exception as e:
            logger.error("agent_error", agent_id=message.recipient, error=str(e))
            self.status[message.recipient] = AgentStatus.ERROR
            return None

    async def run_loop(self) -> None:
        """Loop principal do orquestrador."""
        while True:
            message = await self.message_queue.get()
            response = await self.dispatch(message)

            if response:
                await self.message_queue.put(response)
```

### State Machine Orchestration (LangGraph)

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


class MultiAgentState(TypedDict):
    """State compartilhado entre agents no graph."""
    messages: Annotated[list, add_messages]
    research: str
    code: str
    review: str
    status: str


def research_node(state: MultiAgentState) -> dict:
    """Node: agent de pesquisa."""
    # Chamada LLM para pesquisa
    return {"research": "Research results...", "status": "researched"}


def code_node(state: MultiAgentState) -> dict:
    """Node: agent de código."""
    research = state["research"]
    return {"code": f"Code based on: {research}", "status": "coded"}


def review_node(state: MultiAgentState) -> dict:
    """Node: agent de review."""
    code = state["code"]
    return {"review": f"Review of: {code}", "status": "reviewed"}


def should_revise(state: MultiAgentState) -> str:
    """Routing: precisa revisar?"""
    if "LGTM" in state["review"]:
        return "done"
    return "revise"


# Build graph
graph = StateGraph(MultiAgentState)

graph.add_node("researcher", research_node)
graph.add_node("coder", code_node)
graph.add_node("reviewer", review_node)

graph.set_entry_point("researcher")
graph.add_edge("researcher", "coder")
graph.add_edge("coder", "reviewer")
graph.add_conditional_edges("reviewer", should_revise, {
    "done": END,
    "revise": "coder",
})

app = graph.compile()
```

---

## Frameworks Comparison

| Feature | **LangGraph** | **CrewAI** | **Autogen** | **Custom** |
|---------|:---:|:---:|:---:|:---:|
| **Orchestration** | State machine | Sequential/Hierarchical | Conversation-based | Any |
| **State** | Shared TypedDict | Isolated per agent | Shared conversation | Any |
| **Learning curve** | Medium | Low | Medium | High |
| **Cycles/loops** | Native | Limited | Via conversation | Manual |
| **Parallelism** | Via branches | Built-in | Limited | Full control |
| **Debugging** | Graph visualization | Task logs | Chat history | Custom logging |
| **Production-ready** | Yes | Growing | Research-oriented | Depends |
| **Best for** | Complex routing | Business workflows | Research/debate | Max control |

### LangGraph Multi-Agent

**Prós:** State machine explícito, graph visualization, cycles nativos, checkpoint/resume.
**Contras:** Verboso, tied to LangChain ecosystem, curva de aprendizado.

```python
# Veja langgraph.md para exemplos completos
# Multi-agent em LangGraph = multiple nodes no graph, cada um um "agent"
```

### CrewAI

**Prós:** API simples, roles/tasks/tools claros, bom para workflows de negócio.
**Contras:** Menos flexível, orchestration limitada, menos controle sobre routing.

```python
# pip install crewai crewai-tools

from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Researcher",
    goal="Research the topic thoroughly",
    backstory="Expert researcher with attention to detail",
    verbose=True,
)

writer = Agent(
    role="Writer",
    goal="Write clear, concise content",
    backstory="Technical writer with 10 years experience",
    verbose=True,
)

research_task = Task(
    description="Research Python async patterns",
    expected_output="Comprehensive research summary",
    agent=researcher,
)

write_task = Task(
    description="Write article based on research",
    expected_output="Well-structured article",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()
```

### Autogen

**Prós:** Conversation-driven, debate pattern nativo, group chat built-in.
**Contras:** Menos production-ready, conversation overhead, difícil controlar flow.

```python
# pip install autogen-agentchat

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

coder = AssistantAgent(
    name="Coder",
    system_message="You are a Python expert. Write clean code.",
    llm_config={"model": "claude-sonnet-4-5-20250929"},
)

reviewer = AssistantAgent(
    name="Reviewer",
    system_message="You review code for bugs and improvements.",
    llm_config={"model": "claude-sonnet-4-5-20250929"},
)

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "output"},
)

group_chat = GroupChat(
    agents=[user_proxy, coder, reviewer],
    messages=[],
    max_round=10,
)

manager = GroupChatManager(groupchat=group_chat)
user_proxy.initiate_chat(manager, message="Write a CSV parser in Python")
```

### Custom Implementation

**Prós:** Controle total, sem dependências, otimizado para seu caso.
**Contras:** Mais código, mais manutenção, reinventa a roda.

Veja seção [Implementation Patterns](#implementation-patterns) para exemplos completos.

---

## Implementation Patterns

### Sequential Agents (Research → Code → Review)

```python
from anthropic import AsyncAnthropic
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()


class PipelineResult(BaseModel):
    """Resultado do pipeline."""
    research: str
    code: str
    review: str
    approved: bool


async def research_code_review_pipeline(
    task: str,
    llm: AsyncAnthropic,
) -> PipelineResult:
    """Pipeline: research → code → review com loop de revisão.

    Args:
        task: Descrição da tarefa

    Returns:
        Resultado completo do pipeline
    """
    # Step 1: Research
    research_resp = await llm.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"Research best practices for: {task}",
        }],
    )
    research = research_resp.content[0].text
    logger.info("research_done", length=len(research))

    # Step 2: Code
    code_resp = await llm.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": (
                f"Based on this research:\n{research}\n\n"
                f"Write production-quality Python code for: {task}"
            ),
        }],
    )
    code = code_resp.content[0].text
    logger.info("code_done", length=len(code))

    # Step 3: Review (com retry loop)
    max_revisions = 3
    approved = False

    for attempt in range(max_revisions):
        review_resp = await llm.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": (
                    f"Review this code. Reply APPROVED if good, "
                    f"or list issues:\n\n{code}"
                ),
            }],
        )
        review = review_resp.content[0].text

        if "APPROVED" in review.upper():
            approved = True
            break

        # Revise code based on review
        logger.info("revision_needed", attempt=attempt + 1)
        code_resp = await llm.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": f"Fix these issues:\n{review}\n\nOriginal code:\n{code}",
            }],
        )
        code = code_resp.content[0].text

    return PipelineResult(
        research=research,
        code=code,
        review=review,
        approved=approved,
    )
```

### Parallel Agents (Multiple Search)

```python
import asyncio
from anthropic import AsyncAnthropic
import structlog

logger = structlog.get_logger()


async def multi_perspective_search(
    query: str,
    perspectives: list[str],
    llm: AsyncAnthropic,
    *,
    timeout: float = 30.0,
) -> dict[str, str]:
    """Múltiplos agents pesquisam a mesma query de perspectivas diferentes.

    Args:
        query: Query de pesquisa
        perspectives: Lista de perspectivas (ex: ["technical", "business", "user"])
        llm: Cliente Anthropic
        timeout: Timeout por agent

    Returns:
        Dict perspective → resultado
    """
    async def search_with_perspective(perspective: str) -> tuple[str, str]:
        response = await asyncio.wait_for(
            llm.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": (
                        f"From a {perspective} perspective, "
                        f"analyze: {query}"
                    ),
                }],
            ),
            timeout=timeout,
        )
        return perspective, response.content[0].text

    tasks = [search_with_perspective(p) for p in perspectives]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output: dict[str, str] = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error("perspective_failed", error=str(result))
        else:
            perspective, text = result
            output[perspective] = text

    return output


# Uso
async def main():
    llm = AsyncAnthropic()
    results = await multi_perspective_search(
        query="Should we migrate to microservices?",
        perspectives=["technical", "business", "operations"],
        llm=llm,
    )
    for perspective, analysis in results.items():
        print(f"\n--- {perspective.upper()} ---\n{analysis}")
```

### Hierarchical (Manager + Specialist Workers)

```python
from anthropic import AsyncAnthropic
from pydantic import BaseModel
from typing import Protocol, runtime_checkable
import asyncio
import json
import structlog

logger = structlog.get_logger()


@runtime_checkable
class Worker(Protocol):
    """Protocolo para worker agents."""
    agent_id: str
    specialty: str

    async def execute(self, instruction: str) -> str: ...


class SpecialistWorker:
    """Worker especializado em uma área."""

    def __init__(self, agent_id: str, specialty: str, llm: AsyncAnthropic):
        self.agent_id = agent_id
        self.specialty = specialty
        self.llm = llm

    async def execute(self, instruction: str) -> str:
        response = await self.llm.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            system=f"You are a {self.specialty} specialist.",
            messages=[{"role": "user", "content": instruction}],
        )
        return response.content[0].text


class HierarchicalManager:
    """Manager que coordena workers especializados."""

    def __init__(self, llm: AsyncAnthropic):
        self.llm = llm
        self.workers: dict[str, Worker] = {}

    def add_worker(self, worker: Worker) -> None:
        self.workers[worker.agent_id] = worker

    async def solve(self, problem: str) -> dict[str, str]:
        """Decompõe problema e delega para workers.

        Args:
            problem: Problema a resolver

        Returns:
            Dict worker_id → resultado
        """
        # Manager planeja decomposição
        available = [
            {"id": w.agent_id, "specialty": w.specialty}
            for w in self.workers.values()
        ]

        plan_resp = await self.llm.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"Problem: {problem}\n"
                    f"Available workers: {json.dumps(available)}\n"
                    f"Return JSON array of assignments: "
                    f'[{{"worker_id": "...", "task": "..."}}]'
                ),
            }],
        )

        try:
            assignments = json.loads(plan_resp.content[0].text)
        except json.JSONDecodeError:
            logger.error("plan_parse_failed")
            return {}

        # Executa workers em paralelo
        async def run_worker(assignment: dict) -> tuple[str, str]:
            worker = self.workers[assignment["worker_id"]]
            result = await worker.execute(assignment["task"])
            return assignment["worker_id"], result

        tasks = [run_worker(a) for a in assignments if a["worker_id"] in self.workers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, str] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error("worker_failed", error=str(result))
            else:
                worker_id, text = result
                output[worker_id] = text

        return output


# Uso
async def main():
    llm = AsyncAnthropic()

    manager = HierarchicalManager(llm)
    manager.add_worker(SpecialistWorker("backend", "Python backend development", llm))
    manager.add_worker(SpecialistWorker("frontend", "React frontend development", llm))
    manager.add_worker(SpecialistWorker("devops", "DevOps and infrastructure", llm))

    results = await manager.solve("Build a user authentication system")

    for worker_id, result in results.items():
        print(f"\n=== {worker_id.upper()} ===\n{result}")
```

### Debate Pattern

```python
from anthropic import AsyncAnthropic
import structlog

logger = structlog.get_logger()


async def debate(
    topic: str,
    positions: list[str],
    llm: AsyncAnthropic,
    *,
    rounds: int = 3,
) -> str:
    """Múltiplos agents debatem um tópico e chegam a consenso.

    Args:
        topic: Tópico do debate
        positions: Posições iniciais dos debatedores
        llm: Cliente Anthropic
        rounds: Número de rounds de debate

    Returns:
        Síntese final do debate
    """
    history: list[dict[str, str]] = []

    for round_num in range(rounds):
        logger.info("debate_round", round=round_num + 1)

        round_arguments: list[str] = []

        for i, position in enumerate(positions):
            context = "\n".join(
                f"[{h['position']}]: {h['argument']}" for h in history
            )

            response = await llm.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\n"
                        f"Your position: {position}\n"
                        f"Previous arguments:\n{context}\n\n"
                        f"Present your argument for round {round_num + 1}. "
                        f"Address counterpoints if any."
                    ),
                }],
            )

            argument = response.content[0].text
            history.append({"position": position, "argument": argument})
            round_arguments.append(argument)

    # Synthesis
    all_arguments = "\n\n".join(
        f"[{h['position']}]: {h['argument']}" for h in history
    )

    synthesis = await llm.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                f"Synthesize this debate into a balanced conclusion:\n\n"
                f"{all_arguments}"
            ),
        }],
    )

    return synthesis.content[0].text
```

---

## Error Handling & Resilience

### Agent Failure Handling

```python
import asyncio
from typing import TypeVar, Callable, Awaitable
import structlog

logger = structlog.get_logger()

T = TypeVar("T")


async def with_fallback(
    primary: Callable[[], Awaitable[T]],
    fallback: Callable[[], Awaitable[T]],
    *,
    timeout: float = 30.0,
) -> T:
    """Executa primary agent, fallback se falhar.

    Args:
        primary: Função do agent primário
        fallback: Função do agent fallback
        timeout: Timeout para cada tentativa

    Returns:
        Resultado do agent que sucedeu
    """
    try:
        return await asyncio.wait_for(primary(), timeout=timeout)
    except Exception as e:
        logger.warning("primary_failed", error=str(e))
        return await asyncio.wait_for(fallback(), timeout=timeout)
```

### Retry com Exponential Backoff

```python
async def retry_agent(
    func: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> T:
    """Retry com exponential backoff.

    Args:
        func: Função async para retry
        max_retries: Máximo de tentativas
        base_delay: Delay base em segundos
        max_delay: Delay máximo em segundos

    Returns:
        Resultado da função

    Raises:
        Exception: Se todas tentativas falharem
    """
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_error = e
            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(
                "agent_retry",
                attempt=attempt + 1,
                delay=delay,
                error=str(e),
            )
            await asyncio.sleep(delay)

    raise last_error  # type: ignore[misc]
```

### Circuit Breaker

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"       # Normal: requests passam
    OPEN = "open"           # Falhou: requests bloqueados
    HALF_OPEN = "half_open" # Testando: 1 request passa


@dataclass
class CircuitBreaker:
    """Circuit breaker para agents não confiáveis."""

    failure_threshold: int = 3
    recovery_timeout: float = 60.0
    _state: CircuitState = CircuitState.CLOSED
    _failures: int = 0
    _last_failure: datetime | None = None

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and self._last_failure:
            elapsed = (datetime.utcnow() - self._last_failure).total_seconds()
            if elapsed >= self.recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Executa função com circuit breaker."""
        if self.state == CircuitState.OPEN:
            raise RuntimeError("Circuit breaker is OPEN")

        try:
            result = await func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        self._failures = 0
        self._state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        self._failures += 1
        self._last_failure = datetime.utcnow()
        if self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning("circuit_breaker_opened", failures=self._failures)
```

---

## Best Practices

### 1. Clear Agent Responsibilities

```python
# ❌ Errado: Agent faz tudo
class DoEverythingAgent:
    async def process(self, task: str) -> str:
        # Research + code + test + deploy
        ...

# ✅ Correto: Cada agent tem responsabilidade clara
class ResearchAgent:
    """Apenas pesquisa."""
    async def research(self, topic: str) -> str: ...

class CodingAgent:
    """Apenas codifica."""
    async def code(self, spec: str) -> str: ...

class ReviewAgent:
    """Apenas revisa."""
    async def review(self, code: str) -> str: ...
```

### 2. Message Schema Validation

```python
# ❌ Errado: Dicts sem validação
message = {"from": "a", "data": "something"}

# ✅ Correto: Pydantic models validados
class AgentMessage(BaseModel):
    sender: str
    recipient: str
    message_type: Literal["request", "response", "error"]
    content: dict[str, Any]
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
```

### 3. Timeout em Toda Comunicação

```python
# ❌ Errado: Sem timeout
result = await agent.process(message)

# ✅ Correto: Timeout explícito
result = await asyncio.wait_for(agent.process(message), timeout=30.0)
```

### 4. Isolated State Quando Possível

```python
# ❌ Errado: Todos agents compartilham dict mutável
shared = {"data": []}
agent_a.state = shared
agent_b.state = shared  # Race condition!

# ✅ Correto: State isolado, comunicação via mensagens
result_a = await agent_a.process(input_a)
result_b = await agent_b.process(input_b)
combined = merge(result_a, result_b)
```

### 5. Logging Estruturado de Comunicação

```python
# ❌ Errado: print debugging
print(f"Agent {agent_id} received message")

# ✅ Correto: Logging estruturado com correlation_id
logger.info(
    "message_received",
    agent_id=agent_id,
    sender=message.sender,
    message_type=message.message_type,
    correlation_id=message.correlation_id,
)
```

### 6. Graceful Degradation

```python
# ❌ Errado: Falha de um agent para tudo
results = await asyncio.gather(*tasks)  # Uma exceção cancela tudo

# ✅ Correto: return_exceptions para continuar
results = await asyncio.gather(*tasks, return_exceptions=True)
valid = [r for r in results if not isinstance(r, Exception)]
```

### 7. Idempotent Message Handling

```python
# ❌ Errado: Processar mensagem duplicada
async def process(self, message: AgentMessage) -> None:
    await self.execute(message)  # Duplicatas causam efeitos colaterais

# ✅ Correto: Deduplicação por message_id
async def process(self, message: AgentMessage) -> None:
    if message.message_id in self._processed:
        return
    self._processed.add(message.message_id)
    await self.execute(message)
```

---

## Common Pitfalls

### 1. Shared state sem sincronização

**Problema:** Múltiplos agents escrevem no mesmo dict sem lock.
**Solução:** Use `asyncio.Lock` ou `SharedState` class com atomic operations.

### 2. No timeout em comunicação

**Problema:** Agent trava esperando resposta que nunca chega.
**Solução:** `asyncio.wait_for(coro, timeout=30.0)` em toda comunicação.

### 3. Agent responsibilities não claras

**Problema:** Overlap entre agents, duplicação de trabalho.
**Solução:** Defina interfaces claras com Protocol, docstrings descrevendo escopo.

### 4. No error handling em gather

**Problema:** `asyncio.gather` sem `return_exceptions=True` - uma falha cancela tudo.
**Solução:** Sempre use `return_exceptions=True` para parallel agents.

### 5. Debugging multi-agent difícil

**Problema:** Não sabe qual agent falhou ou onde.
**Solução:** `correlation_id` em todas mensagens + logging estruturado com `structlog`.

### 6. Overhead de comunicação ignorado

**Problema:** Muitos hops entre agents = latência alta.
**Solução:** Minimize hops. Combine agents se comunicação exceder processamento.

---

## Troubleshooting

### Agent não responde

1. Verifique timeout configurado
2. Cheque logs com `correlation_id`
3. Verifique se agent está registrado no orchestrator
4. Verifique se message queue não está cheia

### Deadlock entre agents

1. Agent A espera B, B espera A
2. Use timeouts em toda comunicação
3. Evite dependências circulares
4. Implemente deadlock detection (track waiting chains)

### State inconsistency

1. Verifique se `asyncio.Lock` está sendo usado
2. Use `SharedState.snapshot()` para debug
3. Considere immutable state pattern
4. Verifique ordering de updates (versioning)

### Performance issues

1. Profile com `time.perf_counter()` por agent
2. Verifique se agents que podem ser paralelos estão em `asyncio.gather`
3. Reduza hops de comunicação desnecessários
4. Considere batching de mensagens
5. Cache resultados de agents que não mudam

---

## References

### External
- [LangGraph Multi-Agent Tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [Microsoft Autogen](https://microsoft.github.io/autogen/)
- [Anthropic Multi-Agent Patterns](https://docs.anthropic.com/en/docs/build-with-claude/agentic-systems)
- [asyncio Documentation](https://docs.python.org/3/library/asyncio.html)

### Internal (ai-engineer skill)
- [LangChain](./langchain.md) - Framework para LLM applications
- [LangGraph](./langgraph.md) - State machines para agents
- [Custom Agents](./custom-agents.md) - Build your own agent loop
- [Tool Integration](./tool-integration.md) - Tool calling, API integration

### Internal (arch-py skill)
- [Async Patterns](../../arch-py/references/python/async-patterns.md)
- [Concurrency](../../arch-py/references/python/concurrency.md)
- [Error Handling](../../arch-py/references/python/error-handling.md)
