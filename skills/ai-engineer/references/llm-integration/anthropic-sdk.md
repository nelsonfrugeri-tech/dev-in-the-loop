# Anthropic SDK - Claude API Patterns

Referência técnica completa para integração com Claude via Anthropic SDK Python.

---

## Installation

```bash
pip install anthropic
```

---

## Basic Setup

```python
import anthropic
from typing import AsyncIterator

# Sync client
client = anthropic.Anthropic(
    api_key="sk-ant-...",  # usar env var: ANTHROPIC_API_KEY
)

# Async client (preferido para produção)
async_client = anthropic.AsyncAnthropic(
    api_key="sk-ant-...",
)
```

**Best practice:** SEMPRE use async client em produção

```python
import os
from anthropic import AsyncAnthropic

client = AsyncAnthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    timeout=60.0,  # timeout em segundos
    max_retries=3,  # retries automáticos
)
```

---

## Models Available

| Model | Context Window | Best For | Cost |
|-------|----------------|----------|------|
| `claude-3-5-sonnet-20241022` | 200k tokens | Production, coding, analysis | $$$ |
| `claude-3-5-haiku-20241022` | 200k tokens | Fast, simple tasks | $ |
| `claude-3-opus-20240229` | 200k tokens | Most capable, complex tasks | $$$$ |

**Recomendação:** Comece com Sonnet 3.5 (melhor custo-benefício)

---

## Basic Messages API

### Sync (não recomendado para produção)

```python
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

print(message.content[0].text)
```

### Async (recomendado)

```python
async def generate_response(prompt: str) -> str:
    """Generate response from Claude."""
    message = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

# Usage
response = await generate_response("Explain RAG systems")
```

---

## Streaming Responses

**Use streaming para:**
- UIs interativas (mostrar resposta progressivamente)
- Respostas longas (reduz perceived latency)
- Real-time feedback

```python
async def stream_response(prompt: str) -> AsyncIterator[str]:
    """Stream response from Claude."""
    async with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        async for text in stream.text_stream:
            yield text

# Usage
async for chunk in stream_response("Write a long essay"):
    print(chunk, end="", flush=True)
```

**Com context manager (cleanup automático):**

```python
async def stream_with_metadata(prompt: str):
    """Stream com acesso a metadata."""
    async with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        # Stream text chunks
        async for text in stream.text_stream:
            print(text, end="", flush=True)

    # Após stream completo, acesse metadata
    final_message = await stream.get_final_message()
    print(f"\nTokens used: {final_message.usage.input_tokens + final_message.usage.output_tokens}")
```

---

## System Prompts

**Use system prompts para:**
- Definir comportamento do assistant
- Contexto que não muda entre requests
- Instruções de formato

```python
message = await client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="You are a helpful AI assistant specialized in Python programming. Always provide code examples with type hints and docstrings.",
    messages=[
        {"role": "user", "content": "How do I read a CSV file?"}
    ]
)
```

**Pattern: System prompt parametrizado**

```python
from typing import Literal

async def generate_with_persona(
    prompt: str,
    persona: Literal["developer", "teacher", "analyst"]
) -> str:
    """Generate response com persona específica."""
    system_prompts = {
        "developer": "You are a senior Python developer. Focus on code quality, testing, and best practices.",
        "teacher": "You are a patient teacher. Explain concepts clearly with examples and analogies.",
        "analyst": "You are a data analyst. Focus on insights, patterns, and actionable recommendations."
    }

    message = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompts[persona],
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text
```

---

## Multi-turn Conversations

```python
from typing import TypedDict

class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str

async def chat(messages: list[Message], new_message: str) -> tuple[str, list[Message]]:
    """Add message to conversation and get response."""
    # Add user message
    messages.append({"role": "user", "content": new_message})

    # Get assistant response
    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=messages
    )

    assistant_message = response.content[0].text

    # Add assistant response to history
    messages.append({"role": "assistant", "content": assistant_message})

    return assistant_message, messages

# Usage
conversation: list[Message] = []
response1, conversation = await chat(conversation, "What is RAG?")
response2, conversation = await chat(conversation, "Can you give an example?")
```

---

## Tool Use (Function Calling)

**Claude 3.5 tem excelente tool use. Use para:**
- Chamar APIs externas
- Query databases
- Execute código
- Acessar informações real-time

```python
from typing import Any
from pydantic import BaseModel

class WeatherInput(BaseModel):
    """Input schema for weather tool."""
    location: str
    unit: Literal["celsius", "fahrenheit"] = "celsius"

# Define tool
weather_tool = {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "input_schema": WeatherInput.model_json_schema()
}

async def call_with_tools(prompt: str) -> str:
    """Generate response with tool use."""
    message = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[weather_tool],
        messages=[{"role": "user", "content": prompt}]
    )

    # Check if Claude wants to use a tool
    if message.stop_reason == "tool_use":
        tool_use = next(block for block in message.content if block.type == "tool_use")

        # Execute tool
        if tool_use.name == "get_weather":
            # Call your actual weather API
            weather_result = await get_weather_api(tool_use.input["location"])

            # Send tool result back to Claude
            response = await client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                tools=[weather_tool],
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": message.content},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": str(weather_result)
                            }
                        ]
                    }
                ]
            )

            return response.content[0].text

    # No tool use, return text response
    return message.content[0].text
```

**Pattern: Tool executor genérico**

Ver `examples/agent-with-tools/` para implementação completa.

---

## Structured Outputs (JSON mode)

**Force JSON output com Pydantic schemas:**

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    """Person extracted from text."""
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Job title")

async def extract_person(text: str) -> Person:
    """Extract person info as structured JSON."""
    prompt = f"""Extract person information from this text and return as JSON following this schema:

{Person.model_json_schema()}

Text:
{text}

Return ONLY the JSON, no explanation."""

    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse response into Pydantic model
    json_str = response.content[0].text
    return Person.model_validate_json(json_str)

# Usage
person = await extract_person("John Smith is a 35 year old software engineer")
# person.name = "John Smith"
# person.age = 35
# person.occupation = "software engineer"
```

**Com Instructor library (recomendado):**

```python
import instructor
from anthropic import AsyncAnthropic

# Patch client
client = instructor.from_anthropic(AsyncAnthropic())

async def extract_person_instructor(text: str) -> Person:
    """Extract using Instructor (auto-parsing)."""
    return await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        response_model=Person,
        messages=[
            {"role": "user", "content": f"Extract person from: {text}"}
        ]
    )
```

Ver: `frameworks/instructor.md` para detalhes

---

## Error Handling

### Rate Limits

```python
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
async def generate_with_retry(prompt: str) -> str:
    """Generate with automatic retry on rate limits."""
    try:
        message = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    except anthropic.RateLimitError as e:
        # Log and retry
        print(f"Rate limit hit, retrying... {e}")
        raise  # tenacity will retry

    except anthropic.APIError as e:
        # Log API errors
        print(f"API error: {e}")
        raise
```

### Timeout Handling

```python
import asyncio

async def generate_with_timeout(prompt: str, timeout: float = 30.0) -> str:
    """Generate with timeout."""
    try:
        return await asyncio.wait_for(
            client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            ),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Claude request timed out after {timeout}s")
```

### Circuit Breaker Pattern

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    """Simple circuit breaker for API calls."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time: datetime | None = None
        self.state: Literal["closed", "open", "half_open"] = "closed"

    def is_open(self) -> bool:
        """Check if circuit is open."""
        if self.state == "open":
            # Check if timeout has elapsed
            if self.last_failure_time and \
               datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "half_open"
                return False
            return True
        return False

    def record_success(self):
        """Record successful call."""
        self.failures = 0
        self.state = "closed"

    def record_failure(self):
        """Record failed call."""
        self.failures += 1
        self.last_failure_time = datetime.now()

        if self.failures >= self.failure_threshold:
            self.state = "open"

# Usage
circuit_breaker = CircuitBreaker()

async def generate_with_circuit_breaker(prompt: str) -> str:
    """Generate with circuit breaker."""
    if circuit_breaker.is_open():
        raise Exception("Circuit breaker is open, service unavailable")

    try:
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        circuit_breaker.record_success()
        return response.content[0].text

    except Exception as e:
        circuit_breaker.record_failure()
        raise
```

---

## Token Management

### Counting Tokens

```python
async def count_tokens(text: str) -> int:
    """Count tokens in text."""
    response = await client.messages.count_tokens(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": text}]
    )
    return response.input_tokens

# Usage
tokens = await count_tokens("Long text here...")
print(f"This text uses {tokens} tokens")
```

### Truncating to Fit Context

```python
async def truncate_to_context(text: str, max_tokens: int = 100000) -> str:
    """Truncate text to fit within context window."""
    current_tokens = await count_tokens(text)

    if current_tokens <= max_tokens:
        return text

    # Simple truncation (character-based estimate)
    # 1 token ≈ 4 characters para inglês
    target_chars = max_tokens * 4
    return text[:target_chars]
```

---

## Best Practices

### 1. Use Async Client

```python
# ❌ Sync (bloqueia event loop)
response = client.messages.create(...)

# ✅ Async (não bloqueia)
response = await async_client.messages.create(...)
```

### 2. Set Reasonable Timeouts

```python
client = AsyncAnthropic(
    timeout=60.0,  # 60s timeout
    max_retries=3,
)
```

### 3. Use Environment Variables para API Keys

```python
import os

# ✅ From env var
client = AsyncAnthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

# ❌ Hardcoded
client = AsyncAnthropic(api_key="sk-ant-...")
```

### 4. Handle Rate Limits Gracefully

```python
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential())
async def generate(...):
    # Auto-retry com backoff exponencial
    ...
```

### 5. Log All LLM Calls (produção)

```python
import structlog

logger = structlog.get_logger()

async def generate_logged(prompt: str) -> str:
    """Generate com logging completo."""
    log = logger.bind(model="claude-3-5-sonnet-20241022")

    log.info("llm_call_start", prompt_length=len(prompt))

    try:
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        log.info(
            "llm_call_success",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )

        return response.content[0].text

    except Exception as e:
        log.error("llm_call_error", error=str(e))
        raise
```

---

## References

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Python SDK Reference](https://github.com/anthropics/anthropic-sdk-python)
- [Tool Use Guide](https://docs.anthropic.com/claude/docs/tool-use)
- [Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/prompt-engineering)
