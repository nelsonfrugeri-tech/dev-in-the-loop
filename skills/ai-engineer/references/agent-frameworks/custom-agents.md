# Custom Agents - Build Your Own Agent Loop

Referência completa para construir custom agents from scratch, sem frameworks.

---

## Por Que Build Custom?

### ✅ Vantagens

1. **Máximo controle** - Você decide cada detalhe do loop, parsing, retry, fallback
2. **Debugging fácil** - Stack traces claros, sem abstrações escondidas
3. **Performance** - Zero overhead de framework, chamadas diretas ao SDK
4. **Estabilidade** - Sem breaking changes de dependencies externas
5. **Learning** - Entende exatamente como agents funcionam por dentro

### ❌ Desvantagens

1. **Mais código** - Precisa implementar o que frameworks dão de graça
2. **Sem ecosystem** - Sem LangSmith, sem templates prontos
3. **Manutenção** - Você mantém tudo (retry, parsing, state management)
4. **Tempo inicial** - Setup demora mais que `pip install langchain`

### Quando Usar Custom?

**Use custom quando:**
- Performance é requisito (latência < 500ms por step)
- Produção long-term (>6 meses de manutenção)
- Debugging é crítico (precisa entender cada falha)
- Caso de uso específico (não encaixa em patterns prontos)
- Team tem experiência com SDKs diretos

**NÃO use custom quando:**
- Prototipagem rápida (use LangChain)
- Complex state/routing com checkpointing (use LangGraph)
- Team não tem experiência com async Python

### Decision Tree

```
Precisa de um agent?
  ├─ Protótipo rápido? → LangChain
  ├─ Complex state/routing com checkpointing? → LangGraph
  └─ Produção com máximo controle? → Custom ✅
      ├─ Performance crítica? → Custom ✅
      ├─ Debugging importante? → Custom ✅
      ├─ Long-term maintenance? → Custom ✅
      └─ Código simples/claro? → Custom ✅
```

---

## Agent Loop Básico

### Thought → Action → Observation (ReAct)

O pattern ReAct (Reasoning + Acting) é o core de qualquer agent. O LLM pensa, decide uma ação, observa o resultado, e repete até ter a resposta final.

```python
from typing import Literal, Any
from dataclasses import dataclass, field
from anthropic import AsyncAnthropic
import structlog

logger = structlog.get_logger()


@dataclass
class AgentStep:
    """Um passo do agent loop."""
    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None


@dataclass
class AgentState:
    """Estado completo do agent."""
    query: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None
    status: Literal["running", "completed", "error", "max_iterations"] = "running"
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def add_step(self, step: AgentStep) -> None:
        """Adiciona step ao histórico."""
        self.steps.append(step)

    @property
    def iteration_count(self) -> int:
        """Número de iterações executadas."""
        return len(self.steps)
```

**Fluxo:**
1. LLM recebe query + histórico de observações
2. LLM decide: usar tool ou responder diretamente
3. Se tool: executa tool, adiciona observation, volta ao passo 1
4. Se resposta: retorna e encerra

---

## Tool Architecture

### Tool Registry com Pydantic

O registry centraliza tools disponíveis, seus schemas, e execução.

```python
import inspect
from typing import Any, Callable, get_type_hints
from pydantic import BaseModel, Field, create_model


class ToolDefinition(BaseModel):
    """Schema de uma tool para enviar ao LLM."""
    name: str
    description: str
    input_schema: dict[str, Any]


class Tool:
    """Wrapper de uma tool com schema e execução."""

    def __init__(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or ""
        self._input_model = self._build_input_model()

    def _build_input_model(self) -> type[BaseModel]:
        """Gera Pydantic model a partir dos type hints da função."""
        hints = get_type_hints(self.func)
        hints.pop("return", None)

        fields: dict[str, Any] = {}
        sig = inspect.signature(self.func)

        for param_name, param in sig.parameters.items():
            param_type = hints.get(param_name, str)
            default = param.default if param.default is not inspect.Parameter.empty else ...
            fields[param_name] = (param_type, default)

        return create_model(f"{self.name}_input", **fields)

    @property
    def definition(self) -> ToolDefinition:
        """Retorna definição para enviar ao LLM."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self._input_model.model_json_schema(),
        )

    def validate_input(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Valida input com Pydantic."""
        validated = self._input_model.model_validate(kwargs)
        return validated.model_dump()

    async def execute(self, **kwargs: Any) -> str:
        """Executa tool com validação."""
        validated = self.validate_input(kwargs)

        if inspect.iscoroutinefunction(self.func):
            result = await self.func(**validated)
        else:
            result = self.func(**validated)

        return str(result)


class ToolRegistry:
    """Registry de tools disponíveis."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[..., Any]:
        """Registra uma tool. Pode ser usado como decorator.

        Usage:
            @registry.register
            async def search(query: str) -> str:
                ...

            @registry.register(name="web_search", description="Search the web")
            async def search(query: str) -> str:
                ...
        """
        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            tool = Tool(f, name=name, description=description)
            self._tools[tool.name] = tool
            return f

        if func is not None:
            return decorator(func)
        return decorator

    def get(self, name: str) -> Tool | None:
        """Retorna tool por nome."""
        return self._tools.get(name)

    @property
    def definitions(self) -> list[dict[str, Any]]:
        """Retorna definições de todas tools no formato Anthropic."""
        return [
            {
                "name": tool.definition.name,
                "description": tool.definition.description,
                "input_schema": tool.definition.input_schema,
            }
            for tool in self._tools.values()
        ]

    @property
    def tool_names(self) -> list[str]:
        """Lista nomes das tools registradas."""
        return list(self._tools.keys())
```

**Uso:**

```python
registry = ToolRegistry()

@registry.register
async def search(query: str) -> str:
    """Search for information on the web."""
    # Implementação real aqui
    return f"Results for: {query}"

@registry.register(name="calculator", description="Evaluate math expressions safely")
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    # Usar ast.literal_eval ou library segura, NUNCA eval()
    import ast
    result = ast.literal_eval(expression)
    return str(result)

# Listar tools disponíveis
print(registry.tool_names)  # ["search", "calculator"]

# Obter schemas para o LLM
print(registry.definitions)
```

---

## LLM Client Wrapper

Wrapper fino sobre o Anthropic SDK com retry, logging, e cost tracking.

```python
from anthropic import AsyncAnthropic
from anthropic.types import Message, ContentBlock, ToolUseBlock, TextBlock
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import anthropic
import structlog
import time

logger = structlog.get_logger()


@dataclass
class LLMResponse:
    """Resposta normalizada do LLM."""
    text: str | None
    tool_calls: list[dict[str, Any]]
    input_tokens: int
    output_tokens: int
    stop_reason: str
    latency_ms: float


class LLMClient:
    """Wrapper sobre Anthropic SDK com retry e observability."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = AsyncAnthropic(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(anthropic.RateLimitError),
    )
    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Gera resposta do LLM.

        Args:
            messages: Lista de mensagens no formato Anthropic.
            system: System prompt opcional.
            tools: Lista de tool definitions opcional.

        Returns:
            LLMResponse com texto, tool calls, e métricas.

        Raises:
            anthropic.APIError: Se API falhar após retries.
        """
        log = logger.bind(model=self.model)
        start = time.monotonic()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        try:
            response: Message = await self._client.messages.create(**kwargs)
        except anthropic.APIError as e:
            log.error("llm_error", error=str(e))
            raise

        latency_ms = (time.monotonic() - start) * 1000

        # Parse response
        text = None
        tool_calls = []

        for block in response.content:
            if isinstance(block, TextBlock):
                text = block.text
            elif isinstance(block, ToolUseBlock):
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        result = LLMResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            latency_ms=latency_ms,
        )

        log.info(
            "llm_call",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            stop_reason=result.stop_reason,
            latency_ms=round(result.latency_ms, 1),
            tool_calls=len(tool_calls),
        )

        return result
```

---

## Complete Custom Agent

Implementação completa de um agent production-ready.

```python
import structlog

logger = structlog.get_logger()


class MaxIterationsError(Exception):
    """Agent excedeu máximo de iterações."""

    def __init__(self, iterations: int, state: AgentState):
        self.iterations = iterations
        self.state = state
        super().__init__(f"Agent exceeded {iterations} max iterations")


class ToolExecutionError(Exception):
    """Erro na execução de uma tool."""

    def __init__(self, tool_name: str, error: str):
        self.tool_name = tool_name
        self.error = error
        super().__init__(f"Tool '{tool_name}' failed: {error}")


AGENT_SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
Use tools when you need external information or to perform actions.
Always think step by step before acting.
When you have enough information to answer, provide a clear final answer."""


class CustomAgent:
    """Custom agent com tool use via Anthropic SDK.

    Implementa o loop ReAct (Thought → Action → Observation) sem frameworks.

    Usage:
        llm = LLMClient()
        tools = ToolRegistry()

        @tools.register
        async def search(query: str) -> str:
            ...

        agent = CustomAgent(llm, tools, max_iterations=10)
        state = await agent.run("What is the capital of France?")
        print(state.final_answer)
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        max_iterations: int = 10,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
    ):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt

    async def run(self, query: str) -> AgentState:
        """Executa agent loop completo.

        Args:
            query: Pergunta ou tarefa do usuário.

        Returns:
            AgentState com resposta final e histórico de steps.

        Raises:
            MaxIterationsError: Se exceder max_iterations sem resposta.
        """
        state = AgentState(query=query)
        log = logger.bind(query=query[:100])
        log.info("agent_start")

        # Mensagens acumuladas para o LLM (mantém contexto entre iterações)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": query}
        ]

        for i in range(self.max_iterations):
            log.debug("agent_iteration", iteration=i + 1)

            # 1. Chama LLM
            response = await self.llm.generate(
                messages=messages,
                system=self.system_prompt,
                tools=self.tools.definitions if self.tools.definitions else None,
            )

            state.total_input_tokens += response.input_tokens
            state.total_output_tokens += response.output_tokens

            # 2. Sem tool calls → resposta final
            if not response.tool_calls:
                state.final_answer = response.text
                state.status = "completed"
                state.add_step(AgentStep(thought=response.text or ""))
                log.info(
                    "agent_complete",
                    iterations=i + 1,
                    total_input_tokens=state.total_input_tokens,
                    total_output_tokens=state.total_output_tokens,
                )
                return state

            # 3. Processa tool calls
            # Adiciona resposta do assistant (com tool_use blocks) ao histórico
            assistant_content: list[dict[str, Any]] = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})

            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                })

            messages.append({"role": "assistant", "content": assistant_content})

            # Executa cada tool call e coleta resultados
            tool_results: list[dict[str, Any]] = []

            for tc in response.tool_calls:
                observation = await self._execute_tool(tc["name"], tc["input"], log)

                step = AgentStep(
                    thought=response.text or "",
                    action=tc["name"],
                    action_input=tc["input"],
                    observation=observation,
                )
                state.add_step(step)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": observation,
                })

            # Adiciona tool results ao histórico
            messages.append({"role": "user", "content": tool_results})

        # Max iterations atingido
        state.status = "max_iterations"
        log.warning("agent_max_iterations", max=self.max_iterations)
        raise MaxIterationsError(self.max_iterations, state)

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        log: Any,
    ) -> str:
        """Executa uma tool com error handling.

        Args:
            tool_name: Nome da tool a executar.
            tool_input: Argumentos da tool.
            log: Logger com contexto.

        Returns:
            String com resultado ou mensagem de erro.
        """
        tool = self.tools.get(tool_name)

        if tool is None:
            error_msg = f"Tool '{tool_name}' not found. Available: {self.tools.tool_names}"
            log.warning("tool_not_found", tool=tool_name)
            return error_msg

        try:
            log.info("tool_execute", tool=tool_name, input=tool_input)
            result = await tool.execute(**tool_input)
            log.info("tool_success", tool=tool_name, result_length=len(result))
            return result

        except Exception as e:
            error_msg = f"Error executing '{tool_name}': {type(e).__name__}: {e}"
            log.error("tool_error", tool=tool_name, error=str(e))
            return error_msg
```

### Usage Example

```python
import asyncio
from anthropic import AsyncAnthropic


async def main():
    # Setup
    llm = LLMClient(model="claude-sonnet-4-5-20250929")
    tools = ToolRegistry()

    # Registra tools
    @tools.register
    async def search(query: str) -> str:
        """Search for information on the web."""
        # Aqui você conectaria a uma API de busca real
        return f"Search results for '{query}': Python is a programming language..."

    @tools.register
    async def get_weather(location: str, unit: str = "celsius") -> str:
        """Get current weather for a location."""
        return f"Weather in {location}: 22°{unit[0].upper()}, sunny"

    # Cria e executa agent
    agent = CustomAgent(llm, tools, max_iterations=5)

    state = await agent.run("What's the weather in São Paulo?")

    print(f"Answer: {state.final_answer}")
    print(f"Steps: {state.iteration_count}")
    print(f"Tokens: {state.total_input_tokens} in / {state.total_output_tokens} out")


asyncio.run(main())
```

---

## Error Handling

### Retry Logic com Tenacity

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import anthropic
import structlog
import logging

logger = structlog.get_logger()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    )),
    before_sleep=before_sleep_log(logging.getLogger("retry"), logging.WARNING),
)
async def resilient_llm_call(client: AsyncAnthropic, **kwargs) -> Message:
    """LLM call com retry automático para erros transientes."""
    return await client.messages.create(**kwargs)
```

### Circuit Breaker

```python
from datetime import datetime, timedelta
from typing import Literal


class CircuitBreaker:
    """Circuit breaker para proteger contra cascading failures.

    States:
        closed: Normal, requests passam
        open: Falhas demais, requests bloqueados
        half_open: Testando se serviço voltou

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        if breaker.can_execute():
            try:
                result = await api_call()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
        else:
            # Fallback ou raise
            ...
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._last_failure: datetime | None = None
        self._state: Literal["closed", "open", "half_open"] = "closed"

    def can_execute(self) -> bool:
        """Verifica se request pode ser executado."""
        if self._state == "closed":
            return True

        if self._state == "open":
            if (
                self._last_failure
                and datetime.now() - self._last_failure
                > timedelta(seconds=self.recovery_timeout)
            ):
                self._state = "half_open"
                return True
            return False

        # half_open: permite um request para testar
        return True

    def record_success(self) -> None:
        """Registra sucesso. Reseta circuit."""
        self._failures = 0
        self._state = "closed"

    def record_failure(self) -> None:
        """Registra falha. Pode abrir circuit."""
        self._failures += 1
        self._last_failure = datetime.now()

        if self._failures >= self.failure_threshold:
            self._state = "open"
```

### Fallback Strategies

```python
from typing import Any


class LLMClientWithFallback:
    """Client com fallback para modelo alternativo.

    Se modelo principal falha, tenta modelo fallback.

    Usage:
        client = LLMClientWithFallback(
            primary=LLMClient(model="claude-sonnet-4-5-20250929"),
            fallback=LLMClient(model="claude-haiku-4-5-20251001"),
        )
        response = await client.generate(messages=[...])
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient):
        self.primary = primary
        self.fallback = fallback
        self._breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

    async def generate(self, **kwargs: Any) -> LLMResponse:
        """Tenta primary, fallback se falhar."""
        if self._breaker.can_execute():
            try:
                result = await self.primary.generate(**kwargs)
                self._breaker.record_success()
                return result
            except Exception as e:
                self._breaker.record_failure()
                logger.warning("primary_failed_using_fallback", error=str(e))

        # Fallback
        return await self.fallback.generate(**kwargs)
```

---

## Logging & Observability

### Structured Logging com structlog

```python
import structlog
import time
from typing import Any
from contextvars import ContextVar

# Request ID para correlacionar logs
request_id_var: ContextVar[str] = ContextVar("request_id", default="unknown")


def configure_logging() -> None:
    """Configura structlog para produção."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
    )


# Usage em agent
logger = structlog.get_logger()

async def run_agent_with_logging(agent: CustomAgent, query: str) -> AgentState:
    """Executa agent com logging completo."""
    import uuid
    request_id = str(uuid.uuid4())[:8]

    structlog.contextvars.bind_contextvars(request_id=request_id)

    logger.info("request_start", query=query[:100])
    start = time.monotonic()

    try:
        state = await agent.run(query)
        elapsed = time.monotonic() - start

        logger.info(
            "request_complete",
            status=state.status,
            iterations=state.iteration_count,
            total_tokens=state.total_input_tokens + state.total_output_tokens,
            elapsed_s=round(elapsed, 2),
        )

        return state

    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error("request_error", error=str(e), elapsed_s=round(elapsed, 2))
        raise

    finally:
        structlog.contextvars.unbind_contextvars("request_id")
```

### Cost Tracking

```python
from dataclasses import dataclass

# Preços por 1M tokens (Jan 2025)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
}


@dataclass
class CostTracker:
    """Rastreia custo de tokens por sessão.

    Usage:
        tracker = CostTracker(model="claude-sonnet-4-5-20250929")
        tracker.add(input_tokens=500, output_tokens=200)
        print(tracker.total_cost_usd)  # 0.0045
    """
    model: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        """Adiciona tokens ao tracker."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    @property
    def total_cost_usd(self) -> float:
        """Custo total em USD."""
        pricing = MODEL_PRICING.get(self.model, {"input": 0, "output": 0})
        input_cost = (self.total_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)

    def summary(self) -> dict[str, Any]:
        """Retorna resumo para logging."""
        return {
            "model": self.model,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
        }
```

### Metrics

```python
import time
from dataclasses import dataclass, field


@dataclass
class AgentMetrics:
    """Métricas agregadas de execuções do agent.

    Usage:
        metrics = AgentMetrics()

        # Após cada execução
        metrics.record(state)

        print(metrics.summary())
    """
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_iterations: int = 0
    total_latency_ms: float = 0
    _latencies: list[float] = field(default_factory=list)

    def record(self, state: AgentState, latency_ms: float) -> None:
        """Registra resultado de uma execução."""
        self.total_runs += 1
        self.total_iterations += state.iteration_count
        self.total_latency_ms += latency_ms
        self._latencies.append(latency_ms)

        if state.status == "completed":
            self.successful_runs += 1
        else:
            self.failed_runs += 1

    @property
    def success_rate(self) -> float:
        """Taxa de sucesso (0.0 a 1.0)."""
        if self.total_runs == 0:
            return 0.0
        return self.successful_runs / self.total_runs

    @property
    def avg_latency_ms(self) -> float:
        """Latência média por execução."""
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def avg_iterations(self) -> float:
        """Iterações médias por execução."""
        if self.total_runs == 0:
            return 0.0
        return self.total_iterations / self.total_runs

    def summary(self) -> dict[str, Any]:
        """Resumo para logging/monitoring."""
        return {
            "total_runs": self.total_runs,
            "success_rate": round(self.success_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_iterations": round(self.avg_iterations, 1),
            "failed_runs": self.failed_runs,
        }
```

---

## Best Practices

### 1. Make Agent Loop Explicit

```python
# ❌ Errado: loop escondido em abstração
class Agent:
    def run(self, query):
        return self._internal_magic(query)

# ✅ Correto: loop explícito e visível
class Agent:
    async def run(self, query: str) -> AgentState:
        state = AgentState(query=query)
        for i in range(self.max_iterations):
            response = await self.llm.generate(...)
            if not response.tool_calls:
                state.final_answer = response.text
                return state
            # ... execute tools
        raise MaxIterationsError(...)
```

**Por quê:** Debugging, auditoria, e entendimento do fluxo são triviais quando o loop é explícito.

### 2. Use Type Hints Everywhere

```python
# ❌ Errado: sem tipos
def execute_tool(name, input):
    tool = self.tools[name]
    return tool(input)

# ✅ Correto: tipos completos
async def execute_tool(
    self,
    tool_name: str,
    tool_input: dict[str, Any],
) -> str:
    tool = self.tools.get(tool_name)
    if tool is None:
        return f"Tool '{tool_name}' not found"
    return await tool.execute(**tool_input)
```

**Por quê:** IDE autocomplete, erros caught em dev, self-documenting code.

### 3. Log Everything in Production

```python
# ❌ Errado: sem logs
async def run(self, query):
    response = await self.llm.generate(...)
    return response

# ✅ Correto: logs estruturados em cada etapa
async def run(self, query: str) -> AgentState:
    log = logger.bind(query=query[:100])
    log.info("agent_start")

    for i in range(self.max_iterations):
        response = await self.llm.generate(...)
        log.info("llm_response", iteration=i, tokens=response.input_tokens)

        if response.tool_calls:
            for tc in response.tool_calls:
                log.info("tool_call", tool=tc["name"])

    log.info("agent_complete", iterations=i)
```

**Por quê:** Em produção, logs são a única forma de entender o que aconteceu.

### 4. Always Set Max Iterations

```python
# ❌ Errado: loop infinito possível
while not done:
    response = await llm.generate(...)

# ✅ Correto: limite explícito
for i in range(self.max_iterations):
    response = await llm.generate(...)
    if is_done(response):
        return state

raise MaxIterationsError(self.max_iterations, state)
```

**Por quê:** LLMs podem entrar em loops. Sem limite = custo infinito e requests travados.

### 5. Return Error Messages to LLM (Don't Crash)

```python
# ❌ Errado: crash no agent se tool falha
result = await tool.execute(**input)  # Exception = agent morre

# ✅ Correto: retorna erro como observation
try:
    result = await tool.execute(**input)
except Exception as e:
    result = f"Error: {type(e).__name__}: {e}"
# LLM recebe o erro e pode tentar abordagem diferente
```

**Por quê:** O LLM é capaz de se adaptar a erros. Deixe ele tentar de novo com outra abordagem.

### 6. Separate State from Logic

```python
# ❌ Errado: state misturado com lógica
class Agent:
    def __init__(self):
        self.messages = []  # state no agent
        self.answer = None

# ✅ Correto: state isolado em dataclass
@dataclass
class AgentState:
    query: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None

class Agent:
    async def run(self, query: str) -> AgentState:
        state = AgentState(query=query)  # state criado por execução
        ...
```

**Por quê:** Testabilidade, concorrência segura, state não vaza entre execuções.

### 7. Validate Tool Inputs with Pydantic

```python
# ❌ Errado: aceita qualquer input do LLM
result = tool_func(**llm_output)  # Input pode ser inválido

# ✅ Correto: valida com Pydantic antes de executar
validated = tool.input_model.model_validate(llm_output)
result = await tool_func(**validated.model_dump())
```

**Por quê:** LLMs podem gerar inputs inválidos. Validação previne erros silenciosos.

---

## Common Pitfalls

### 1. No Max Iterations

**Problema:** Agent entra em loop infinito, gastando tokens sem parar.

**Solução:** Sempre use `for i in range(max_iterations)` ao invés de `while True`.

### 2. Poor Error Messages to LLM

**Problema:** Tool falha e retorna "Error" sem contexto. LLM não sabe o que deu errado.

**Solução:**
```python
# Retorne erros descritivos
return f"Error calling '{tool_name}': {type(e).__name__}: {e}. Try a different approach."
```

### 3. Not Accumulating Messages

**Problema:** Cada iteração envia só a última mensagem. LLM perde contexto.

**Solução:** Mantenha lista completa de messages (user → assistant → tool_result → assistant → ...).

### 4. No State Persistence

**Problema:** Agent crash = todo progresso perdido.

**Solução:** Para agents long-running, persista state a cada iteração:
```python
for i in range(max_iterations):
    response = await self.llm.generate(...)
    state.add_step(step)
    await self._persist_state(state)  # Salva em DB/arquivo
```

### 5. Using eval() in Tools

**Problema:** `eval(expression)` é vulnerável a code injection.

**Solução:** Use `ast.literal_eval()` para expressões simples, ou uma library de math parsing como `simpleeval`.

### 6. Ignoring Stop Reason

**Problema:** Não checar `stop_reason` do LLM. Se for `max_tokens`, a resposta foi truncada.

**Solução:**
```python
if response.stop_reason == "max_tokens":
    logger.warning("response_truncated")
    # Pode precisar continuar a geração
```

---

## Troubleshooting

### Agent não termina

**Causa:** Loop infinito ou LLM sempre escolhe usar tool.

**Debug:**
```python
# Adicione logging por iteração
for i in range(max_iterations):
    logger.info("iteration", i=i, tool_calls=len(response.tool_calls))
```

**Soluções:**
- Verifique `max_iterations`
- Melhore system prompt (instrua a responder quando tiver info suficiente)
- Verifique se tools retornam resultados úteis

### Tool não é chamada

**Causa:** LLM não entende quando usar a tool.

**Soluções:**
- Melhore `description` da tool (seja específico sobre quando usar)
- Adicione exemplos no system prompt
- Verifique se `tools` está sendo passado na chamada ao LLM

### Parse errors no tool input

**Causa:** LLM gera input inválido para tool.

**Soluções:**
- Use Pydantic validation (retorna erro claro ao LLM)
- Simplifique schema da tool (menos parâmetros = menos erros)
- Use `temperature=0` para respostas mais determinísticas

### Custo alto

**Causa:** Muitas iterações ou context window grande.

**Soluções:**
- Reduza `max_iterations`
- Use modelo mais barato para tasks simples (Haiku)
- Implemente cost tracking e alertas
- Limite tamanho do histórico de messages

---

## Testing Custom Agents

```python
import pytest
from unittest.mock import AsyncMock


@pytest.fixture
def mock_llm() -> LLMClient:
    """LLM mock que retorna respostas controladas."""
    llm = AsyncMock(spec=LLMClient)
    return llm


@pytest.fixture
def tools() -> ToolRegistry:
    """Registry com tools de teste."""
    registry = ToolRegistry()

    @registry.register
    async def search(query: str) -> str:
        return f"Mock result for: {query}"

    return registry


@pytest.mark.asyncio
async def test_agent_simple_answer(mock_llm: LLMClient, tools: ToolRegistry):
    """Agent responde diretamente sem usar tools."""
    mock_llm.generate.return_value = LLMResponse(
        text="Paris is the capital of France",
        tool_calls=[],
        input_tokens=50,
        output_tokens=10,
        stop_reason="end_turn",
        latency_ms=200,
    )

    agent = CustomAgent(mock_llm, tools, max_iterations=5)
    state = await agent.run("What is the capital of France?")

    assert state.status == "completed"
    assert state.final_answer == "Paris is the capital of France"
    assert state.iteration_count == 1


@pytest.mark.asyncio
async def test_agent_uses_tool(mock_llm: LLMClient, tools: ToolRegistry):
    """Agent usa tool e depois responde."""
    # Primeira chamada: LLM quer usar tool
    # Segunda chamada: LLM responde com base na observation
    mock_llm.generate.side_effect = [
        LLMResponse(
            text="I need to search for this.",
            tool_calls=[{"id": "tc_1", "name": "search", "input": {"query": "RAG"}}],
            input_tokens=50, output_tokens=30,
            stop_reason="tool_use", latency_ms=300,
        ),
        LLMResponse(
            text="RAG is Retrieval-Augmented Generation.",
            tool_calls=[],
            input_tokens=100, output_tokens=20,
            stop_reason="end_turn", latency_ms=200,
        ),
    ]

    agent = CustomAgent(mock_llm, tools, max_iterations=5)
    state = await agent.run("What is RAG?")

    assert state.status == "completed"
    assert state.iteration_count == 2
    assert state.steps[0].action == "search"


@pytest.mark.asyncio
async def test_agent_max_iterations(mock_llm: LLMClient, tools: ToolRegistry):
    """Agent para quando atinge max iterations."""
    # LLM sempre quer usar tool (loop infinito)
    mock_llm.generate.return_value = LLMResponse(
        text="Searching...",
        tool_calls=[{"id": "tc_1", "name": "search", "input": {"query": "test"}}],
        input_tokens=50, output_tokens=30,
        stop_reason="tool_use", latency_ms=300,
    )

    agent = CustomAgent(mock_llm, tools, max_iterations=3)

    with pytest.raises(MaxIterationsError) as exc_info:
        await agent.run("Infinite loop query")

    assert exc_info.value.iterations == 3
```

---

## References

### External

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Anthropic Tool Use Documentation](https://docs.anthropic.com/claude/docs/tool-use)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Tenacity - Python retrying library](https://tenacity.readthedocs.io/)
- [structlog - Structured Logging](https://www.structlog.org/)

### Internal (ai-engineer skill)

- [LangChain](./langchain.md) - Quando usar framework ao invés de custom
- [LangGraph](./langgraph.md) - State machines para agents complexos
- [Tool Integration](./tool-integration.md) - Patterns avançados de tools
- [Multi-Agent Systems](./multi-agent.md) - Múltiplos agents colaborando
- [Anthropic SDK](../llm-integration/anthropic-sdk.md) - SDK patterns detalhados

### Internal (arch-py skill)

- [Async Patterns](../../arch-py/references/python/async-patterns.md)
- [Error Handling](../../arch-py/references/python/error-handling.md)
- [Testing](../../arch-py/references/python/testing.md)
- [Type System](../../arch-py/references/python/type-system.md)
