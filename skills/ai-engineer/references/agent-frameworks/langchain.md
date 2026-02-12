# LangChain - Framework para LLM Applications

Referência técnica completa de LangChain, quando usar, como usar, e quando evitar.

---

## O que é LangChain?

**LangChain** é um framework Python/TypeScript para construir aplicações com LLMs.

**Core value proposition:**
- Componentes reutilizáveis (chains, agents, tools, memory)
- Abstrações para múltiplos providers (OpenAI, Anthropic, etc.)
- Patterns prontos (RAG, agents, chat, summarization)
- Ecosystem rico (LangSmith, LangServe, templates)

**Trade-offs principais:**
- ✅ **Prós:** Rápido para prototipar, muitos componentes prontos, comunidade grande
- ❌ **Contras:** Abstração pesada, debugging difícil, breaking changes frequentes, overhead de performance

---

## Quando Usar LangChain

### ✅ Use LangChain quando:

1. **Prototipagem rápida** - Precisa validar ideia em horas/dias
2. **Patterns comuns** - Seu caso de uso é RAG, chatbot, summarization (casos cobertos por templates)
3. **Múltiplos providers** - Quer abstrair diferenças entre OpenAI/Anthropic/etc
4. **Ecosystem** - Quer usar LangSmith (tracing), LangServe (deploy)
5. **Team já conhece** - Time tem experiência com LangChain

### ❌ NÃO use LangChain quando:

1. **Performance crítica** - Overhead de abstrações é significativo
2. **Debugging importante** - Stack traces são complexos, hard to debug
3. **Máximo controle** - Quer entender/controlar cada etapa
4. **Produção long-term** - Breaking changes frequentes são problemáticos
5. **Casos específicos** - Seu caso de uso não encaixa nos patterns prontos

---

## Installation

```bash
# Core
pip install langchain langchain-core

# Provider-specific (escolha o que precisa)
pip install langchain-anthropic  # Anthropic/Claude
pip install langchain-openai     # OpenAI
pip install langchain-community  # Community integrations

# Para RAG
pip install langchain-chroma     # Vector DB
pip install langchain-qdrant     # Qdrant

# Observability
pip install langsmith
```

---

## Core Concepts

### 1. LLMs e Chat Models

**LLM** = Input string → Output string (completion)
**ChatModel** = Input messages → Output message (chat)

```python
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import asyncio

# Claude
claude = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0,
    max_tokens=1024,
    api_key="..."  # ou ANTHROPIC_API_KEY env var
)

# OpenAI
gpt = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key="..."  # ou OPENAI_API_KEY env var
)

# Invoke (sync - não recomendado para produção)
response = claude.invoke("Explain RAG in one sentence")
print(response.content)

# Async (recomendado)
async def generate():
    response = await claude.ainvoke("Explain RAG")
    return response.content

# Usage
result = asyncio.run(generate())
```

**Com mensagens estruturadas:**

```python
from langchain.schema import HumanMessage, SystemMessage, AIMessage

messages = [
    SystemMessage(content="You are a helpful Python expert."),
    HumanMessage(content="How do I read a CSV file?")
]

response = await claude.ainvoke(messages)
print(response.content)
```

### 2. Prompts e Templates

**PromptTemplate** = String template com variáveis

```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# Simple template
template = PromptTemplate.from_template(
    "Explain {concept} to a {audience} in {style} style."
)

prompt = template.format(
    concept="quantum computing",
    audience="5 year old",
    style="simple"
)

# Chat template (preferido)
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "Explain {topic}")
])

messages = chat_template.format_messages(
    role="Python expert",
    topic="async/await"
)

response = await claude.ainvoke(messages)
```

**Com Pydantic para type safety:**

```python
from pydantic import BaseModel, Field

class PromptInput(BaseModel):
    """Input para prompt template."""
    concept: str = Field(description="Concept to explain")
    audience: str = Field(description="Target audience")

# Template com input model
template = PromptTemplate.from_template(
    "Explain {concept} to {audience}.",
    input_variables=["concept", "audience"]
)

# Type-safe input
input_data = PromptInput(concept="RAG", audience="developers")
prompt = template.format(**input_data.model_dump())
```

### 3. Chains

**Chain** = Sequência de chamadas (LLM → processing → LLM → ...)

```python
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

# Define chain
prompt = ChatPromptTemplate.from_template("Translate '{text}' to {language}")

chain = prompt | claude

# Invoke
result = await chain.ainvoke({
    "text": "Hello world",
    "language": "Portuguese"
})

print(result.content)
```

**LCEL (LangChain Expression Language) - Preferred:**

```python
from langchain_core.output_parsers import StrOutputParser

# Pipeline: prompt → llm → parse
chain = (
    ChatPromptTemplate.from_template("Explain {topic} in one sentence")
    | claude
    | StrOutputParser()  # Extrai .content automaticamente
)

# Invoke
result = await chain.ainvoke({"topic": "RAG"})
print(result)  # String, não Message object
```

**Sequential chains:**

```python
from langchain.chains import SequentialChain

# Chain 1: Generate code
code_prompt = ChatPromptTemplate.from_template("Write Python code for: {task}")
code_chain = code_prompt | claude | StrOutputParser()

# Chain 2: Review code
review_prompt = ChatPromptTemplate.from_template("Review this code:\n{code}")
review_chain = review_prompt | claude | StrOutputParser()

# Combine
async def code_and_review(task: str) -> dict[str, str]:
    code = await code_chain.ainvoke({"task": task})
    review = await review_chain.ainvoke({"code": code})
    return {"code": code, "review": review}

result = await code_and_review("read CSV file")
```

### 4. Agents

**Agent** = LLM que decide quais tools usar

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from typing import Annotated

# Define tools
@tool
def calculate(
    expression: Annotated[str, "Mathematical expression to evaluate"]
) -> str:
    """Calculate mathematical expression."""
    try:
        # WARNING: eval is unsafe, use only for demos
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
async def search_web(
    query: Annotated[str, "Search query"]
) -> str:
    """Search the web for information."""
    # Implement actual search
    return f"Search results for: {query}"

# Create agent
tools = [calculate, search_web]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # Tool usage history
])

agent = create_tool_calling_agent(claude, tools, prompt)

# Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Log tool calls
    max_iterations=5,  # Prevent infinite loops
)

# Run
result = await agent_executor.ainvoke({
    "input": "What is 25 * 17 + 100?"
})

print(result["output"])
```

**Agent types:**

| Type | Use Case | Provider Support |
|------|----------|------------------|
| `create_tool_calling_agent` | Modern, uses native tool calling | Claude 3+, GPT-4+ |
| `create_react_agent` | ReAct pattern (thought → action → observation) | All models |
| `create_openai_functions_agent` | OpenAI function calling (deprecated) | OpenAI only |

**Recomendação:** Use `create_tool_calling_agent` (mais moderno, melhor performance)

### 5. Memory

**Memory** = Mantém contexto entre chamadas

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain

# Buffer memory (guarda todas mensagens)
memory = ConversationBufferMemory()

chain = ConversationChain(
    llm=claude,
    memory=memory,
    verbose=True
)

# Conversation
await chain.ainvoke({"input": "My name is Alice"})
response = await chain.ainvoke({"input": "What's my name?"})  # "Your name is Alice"

# Acessar histórico
print(memory.load_memory_variables({}))
```

**Summary memory (para conversas longas):**

```python
from langchain.memory import ConversationSummaryMemory

# Summariza automaticamente quando fica longo
summary_memory = ConversationSummaryMemory(
    llm=claude,
    max_token_limit=2000  # Summariza se exceder
)

chain = ConversationChain(
    llm=claude,
    memory=summary_memory
)
```

**Custom memory backend (Redis, MongoDB):**

```python
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory
from typing import List
from langchain.schema.messages import BaseMessage
import redis

class RedisMessageHistory(BaseChatMessageHistory):
    """Chat history stored in Redis."""

    def __init__(self, session_id: str, redis_client: redis.Redis):
        self.session_id = session_id
        self.redis = redis_client
        self.key = f"chat_history:{session_id}"

    @property
    def messages(self) -> List[BaseMessage]:
        """Get messages from Redis."""
        messages_data = self.redis.lrange(self.key, 0, -1)
        # Deserialize messages
        return [deserialize_message(m) for m in messages_data]

    def add_message(self, message: BaseMessage):
        """Add message to Redis."""
        self.redis.rpush(self.key, serialize_message(message))
        self.redis.expire(self.key, 3600)  # 1 hour TTL

    def clear(self):
        """Clear history."""
        self.redis.delete(self.key)

# Usage
redis_client = redis.Redis(host='localhost', port=6379)
history = RedisMessageHistory("user_123", redis_client)

memory = ConversationBufferMemory(
    chat_memory=history,
    return_messages=True
)
```

### 6. Tools

**Tool** = Função que agent pode chamar

```python
from langchain.tools import tool, StructuredTool
from pydantic import BaseModel, Field

# Decorator style (simples)
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

# Pydantic schema (type-safe)
class CalculatorInput(BaseModel):
    """Input for calculator."""
    operation: str = Field(description="Operation: add, subtract, multiply, divide")
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

@tool(args_schema=CalculatorInput)
def calculator(operation: str, a: float, b: float) -> float:
    """Perform mathematical operations."""
    ops = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float('inf')
    }
    return ops[operation](a, b)

# Async tool
@tool
async def fetch_url(url: str) -> str:
    """Fetch content from URL."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

### 7. Callbacks

**Callbacks** = Hooks para logging, tracing, debugging

```python
from langchain.callbacks import AsyncCallbackHandler
from langchain.schema import LLMResult
from typing import Any
import structlog

logger = structlog.get_logger()

class LoggingCallback(AsyncCallbackHandler):
    """Log all LLM calls."""

    async def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ):
        """Log when LLM starts."""
        logger.info("llm_start", prompt_count=len(prompts))

    async def on_llm_end(self, response: LLMResult, **kwargs: Any):
        """Log when LLM ends."""
        logger.info(
            "llm_end",
            generations=len(response.generations),
            llm_output=response.llm_output
        )

    async def on_llm_error(self, error: Exception, **kwargs: Any):
        """Log LLM errors."""
        logger.error("llm_error", error=str(error))

# Usage
callback = LoggingCallback()

response = await claude.ainvoke(
    "Explain RAG",
    config={"callbacks": [callback]}
)
```

**LangSmith (observability):**

```python
import os

# Set env vars
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "..."

# Todas chamadas são automaticamente traced no LangSmith
response = await claude.ainvoke("Explain RAG")

# View traces: https://smith.langchain.com
```

---

## RAG com LangChain

### Basic RAG

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. Load documents
loader = TextLoader("docs.txt")
documents = loader.load()

# 2. Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3. Embed and store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Top 3 docs
)

# 5. Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=claude,
    retriever=retriever,
    return_source_documents=True  # Include sources
)

# Query
result = await qa_chain.ainvoke({"query": "What is RAG?"})
print(result["result"])
print(result["source_documents"])
```

### Advanced RAG com LCEL

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# Prompt template
template = """Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Format docs helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain com LCEL
rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,  # Retrieve e formata docs
        question=RunnablePassthrough()    # Passa query através
    )
    | prompt      # Formata prompt
    | claude      # Gera resposta
    | StrOutputParser()
)

# Query
answer = await rag_chain.ainvoke("What is RAG?")
```

### RAG com Reranking

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# Cohere reranker
compressor = CohereRerank(
    model="rerank-english-v2.0",
    top_n=3
)

# Compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# Use in chain
rag_chain = (
    RunnableParallel(
        context=compression_retriever | format_docs,
        question=RunnablePassthrough()
    )
    | prompt
    | claude
    | StrOutputParser()
)
```

---

## Best Practices

### 1. Use Async

```python
# ❌ Sync (bloqueia)
result = chain.invoke({"input": "..."})

# ✅ Async (não bloqueia)
result = await chain.ainvoke({"input": "..."})
```

**Por quê:** Async permite I/O concorrente, melhor performance

### 2. Use LCEL (não legacy chains)

```python
# ❌ Legacy LLMChain
chain = LLMChain(llm=claude, prompt=prompt)

# ✅ LCEL
chain = prompt | claude | StrOutputParser()
```

**Por quê:** LCEL é mais moderno, melhor performance, streaming nativo

### 3. Set Timeouts

```python
from langchain_anthropic import ChatAnthropic

# ✅ Com timeout
claude = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    timeout=60.0,
    max_retries=3
)
```

**Por quê:** Previne hangs em produção

### 4. Use Pydantic para Tool Schemas

```python
from pydantic import BaseModel, Field

# ✅ Type-safe
class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=10, description="Max results")

@tool(args_schema=SearchInput)
def search(query: str, max_results: int = 10) -> str:
    ...
```

**Por quê:** Type safety, better IDE support, validation automática

### 5. Log com Callbacks ou LangSmith

```python
# ✅ LangSmith (produção)
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Ou custom callback
response = await claude.ainvoke(
    "...",
    config={"callbacks": [LoggingCallback()]}
)
```

**Por quê:** Debugging, monitoring, cost tracking

---

## Common Pitfalls

### 1. Não usar Async

**Problem:** Sync calls bloqueiam event loop
**Solution:** Use `ainvoke()`, `astream()`, etc.

### 2. Memory leak em loops

**Problem:** Memory acumula infinitamente
```python
# ❌ Memory cresce sem limite
for query in queries:
    await chain.ainvoke({"input": query})
```

**Solution:** Clear memory periodicamente
```python
# ✅ Clear memory
for i, query in enumerate(queries):
    await chain.ainvoke({"input": query})
    if i % 100 == 0:
        memory.clear()
```

### 3. Não tratar rate limits

**Problem:** Falha em API calls sem retry
**Solution:** Configure `max_retries`
```python
claude = ChatAnthropic(
    max_retries=3,
    timeout=60.0
)
```

### 4. Tool descriptions ruins

**Problem:** Agent não sabe quando usar tool
```python
# ❌ Descrição vaga
@tool
def search(query: str) -> str:
    """Search."""
    ...

# ✅ Descrição clara
@tool
def search(query: str) -> str:
    """Search Wikipedia for information about a topic.

    Use this when you need factual information about:
    - Historical events
    - Famous people
    - Scientific concepts
    - Geographic locations

    Do NOT use for:
    - Current events (data is outdated)
    - Opinions or subjective topics
    """
    ...
```

### 5. Chains muito complexas

**Problem:** Hard to debug, slow
**Solution:** Break into smaller chains
```python
# ❌ Monolithic chain
mega_chain = step1 | step2 | step3 | step4 | step5

# ✅ Modular
chain_a = step1 | step2
chain_b = step3 | step4
result = await chain_b.ainvoke(await chain_a.ainvoke(input))
```

---

## Troubleshooting

### Error: "Too many tokens"

**Causa:** Context window exceeded

**Solution:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Truncate ou summarize
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
chunks = splitter.split_text(long_text)
```

### Error: "Rate limit exceeded"

**Causa:** Too many requests

**Solution:**
```python
# Add retry com backoff
claude = ChatAnthropic(
    max_retries=5,
    timeout=120.0
)
```

### Agent não usa tools

**Causa:** Tool description não é clara

**Solution:**
- Melhore description da tool
- Adicione examples no system prompt
- Use `verbose=True` para debug
```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True  # Ver reasoning
)
```

### Debugging chains

**Use `verbose=True` e callbacks:**
```python
from langchain.globals import set_debug

# Global debug
set_debug(True)

# Ou per-chain
chain = prompt | claude
result = await chain.ainvoke(
    {"input": "..."},
    config={"callbacks": [LoggingCallback()]}
)
```

---

## Migration Paths

### De LangChain para Custom

**Quando migrar:**
- Debugging é muito difícil
- Performance é crítica
- Precisa de controle total

**Como:**
1. Identifique componentes LangChain que usa
2. Reimplemente com providers diretos (Anthropic SDK, OpenAI SDK)
3. Mantenha testes (behavior deve ser igual)

**Exemplo:**
```python
# ❌ LangChain
chain = prompt | claude | StrOutputParser()
result = await chain.ainvoke({"input": "..."})

# ✅ Direct SDK
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

async def generate(prompt: str) -> str:
    message = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

result = await generate("...")
```

### De Legacy Chains para LCEL

```python
# ❌ Legacy
from langchain.chains import LLMChain
chain = LLMChain(llm=claude, prompt=prompt)

# ✅ LCEL
chain = prompt | claude | StrOutputParser()
```

---

## Decision Tree

```
Precisa usar framework para LLMs?
  ├─ Não, código simples suficiente → Use SDK direto (Anthropic, OpenAI)
  └─ Sim, framework ajuda:
      ├─ Prototipagem rápida? → LangChain ✅
      ├─ Produção enterprise?
      │   ├─ Team experiente em LangChain? → LangChain ✅
      │   ├─ Precisa de LangSmith? → LangChain ✅
      │   └─ Team novo em LangChain? → Custom ou LangGraph
      ├─ Máximo controle? → Custom agent ✅
      ├─ Complex state/routing? → LangGraph ✅
      └─ Multi-agent? → LangGraph ou Custom ✅
```

---

## References

### External

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
- [LangSmith Observability](https://docs.smith.langchain.com/)
- [LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates)
- [LangChain API Reference](https://api.python.langchain.com/)

### Internal (ai-engineer skill)

- [LangGraph](./langgraph.md)
- [Custom Agents](./custom-agents.md)
- [Tool Integration](./tool-integration.md)
- [Multi-Agent Systems](./multi-agent.md)
- [Anthropic SDK](../llm-integration/anthropic-sdk.md)
- [RAG Architecture](../rag/architecture.md)

### Internal (arch-py skill)

- [Async Patterns](../../arch-py/references/python/async-patterns.md)
- [Error Handling](../../arch-py/references/python/error-handling.md)
- [Type System](../../arch-py/references/python/type-system.md)
- [Testing](../../arch-py/references/python/testing.md)
