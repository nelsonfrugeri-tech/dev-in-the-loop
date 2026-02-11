---
name: ai-engineer
description: |
  Baseline de conhecimento para AI/ML engineering moderno em Python. Foco em LLM engineering,
  RAG systems, agent frameworks (LangChain/LangGraph), multiple LLM providers (Anthropic, OpenAI,
  Bedrock, Gemini, Meta), vector databases (Qdrant), semantic caching (MongoDB, Redis), testing,
  observability, security, e production patterns. Complementa arch-py skill com patterns AI-specific.
  Use quando: (1) Desenvolver sistemas LLM/RAG/Agents, (2) Integrar múltiplos providers, (3) Implementar
  caching semântico, (4) Testar e monitorar AI systems, (5) Deploy de AI em produção.
  Triggers: /ai-engineer, AI engineering, LLM development, RAG systems, agent frameworks.
---

# AI-Engineer Skill - Modern AI/ML Engineering Best Practices

## Propósito

Esta skill é a **biblioteca de conhecimento** para AI/ML engineering moderno em Python (2026).
Ela complementa a `arch-py` skill com patterns específicos de AI systems.

**Quem usa esta skill:**
- Agent `dev-py` → ao desenvolver features com LLMs, RAG, agents
- Agent `review-py` → ao revisar código que usa AI systems
- Você diretamente → quando precisar de referência de patterns AI

**O que esta skill contém:**
- LLM integration patterns (Anthropic, OpenAI, Bedrock, Gemini, Meta)
- RAG architecture (retrieval, embeddings, chunking, evaluation)
- Agent frameworks (LangChain, LangGraph)
- Vector databases (Qdrant)
- Semantic caching (MongoDB, Redis)
- Testing AI systems (mocking, evaluation, regression)
- Observability (logging, tracing, metrics)
- Security (prompt injection, PII, content filtering)
- Production patterns (deployment, scaling, cost optimization)

**O que esta skill NÃO contém:**
- Classical ML (scikit-learn, feature engineering) — foco em LLMs
- Deep Learning frameworks (PyTorch, TensorFlow) — foco em LLM APIs
- Workflow de execução — isso está nos agents

---

## Filosofia

### AI Engineering ≠ Prompt Engineering

**AI Engineering é Software Engineering aplicado a sistemas de AI:**
- Requer TUDO de `arch-py` (types, testing, async, architecture)
- PLUS patterns AI-specific (prompts, RAG, agents, observability)
- Foco em **systems thinking**, não só prompts bonitos

### Princípios Fundamentais

**1. Determinismo onde possível**
- Minimize non-determinism desnecessário
- Use temperature=0 quando determinismo importa
- Fixe seeds quando reproduzibilidade é crítica
- Cache aggressivamente resultados LLM

**2. Testabilidade**
- AI systems PODEM e DEVEM ser testados
- Mock LLM calls em unit tests
- Use golden datasets para regression
- LLM-as-judge para evaluation
- Assertions específicas para structured outputs

**3. Observabilidade**
- Você não pode melhorar o que não mede
- Log TODOS prompts e responses (com PII sanitization)
- Trace chains e agents end-to-end
- Métricas: latency, cost, tokens, quality
- Dashboards e alerting

**4. Cost-Awareness**
- LLMs são caros, otimize
- Cache semantic para queries similares
- Model selection baseada em task complexity
- Batch requests quando possível
- Prompt compression

**5. Safety & Security**
- Prompt injection é real, previna
- PII handling rigoroso
- Content filtering (input e output)
- Rate limiting e abuse prevention
- API key rotation

**6. Multi-Provider Strategy**
- Não dependa de um único provider
- Fallbacks entre providers (Anthropic → OpenAI → Bedrock)
- Cost optimization (route por preço/performance)
- A/B testing de providers

---

## Composição com Arch-Py Skill

**arch-py** (fundação) + **ai-engineer** (AI layer) = AI System completo

| Aspecto | Arch-Py Skill | AI-Engineer Skill |
|---------|---------------|-------------------|
| **Type System** | Protocol, TypeVar, Generic | LLM response schemas, Pydantic models |
| **Async** | asyncio patterns gerais | Async LLM calls, parallel embeddings |
| **Error Handling** | Exception hierarchy | Rate limits, API failures, retries |
| **Testing** | pytest, fixtures, mocking | LLM mocking, golden datasets, evaluation |
| **Logging** | structlog patterns | LLM call logging, token tracking |
| **Architecture** | Clean Architecture, DI | RAG architecture, agent loops |
| **Specific** | — | Prompts, RAG, Agents, Vector DBs, Caching |

**Sempre use ambas:**
- `arch-py` para fundação Python sólida
- `ai-engineer` para layer AI-specific

---

## Estrutura da Skill

### References (Documentação Técnica)

#### LLM Integration
| Arquivo | Conteúdo |
|---------|----------|
| `llm-integration/anthropic-sdk.md` | Claude API patterns, streaming, tool use |
| `llm-integration/openai-sdk.md` | GPT API patterns, function calling |
| `llm-integration/bedrock.md` | AWS Bedrock multi-model patterns |
| `llm-integration/gemini.md` | Google Gemini API patterns |
| `llm-integration/meta.md` | Llama via APIs (Replicate, Together, etc.) |
| `llm-integration/prompt-engineering.md` | Few-shot, CoT, structured prompts |
| `llm-integration/structured-outputs.md` | JSON mode, schema enforcement, Pydantic |
| `llm-integration/streaming.md` | Streaming responses, SSE |
| `llm-integration/tool-use.md` | Function calling, tool schemas |
| `llm-integration/error-handling.md` | Rate limits, retries, fallbacks, circuit breakers |
| `llm-integration/multi-provider.md` | Strategy pattern, routing, fallbacks |

#### RAG (Retrieval-Augmented Generation)
| Arquivo | Conteúdo |
|---------|----------|
| `rag/architecture.md` | RAG patterns modernos (naive, advanced, agentic) |
| `rag/chunking-strategies.md` | Fixed-size, semantic, recursive, document-aware |
| `rag/embeddings.md` | Modelos (OpenAI, Cohere, sentence-transformers) |
| `rag/retrieval.md` | Semantic search, hybrid search, re-ranking |
| `rag/evaluation.md` | Métricas (ragas, faithfulness, relevance) |
| `rag/query-transformation.md` | Rewrite, decomposition, multi-query |
| `rag/context-compression.md` | Reranking, filtering, summarization |

#### Vector Databases
| Arquivo | Conteúdo |
|---------|----------|
| `vector-db/qdrant.md` | Qdrant setup, collections, search, filters |
| `vector-db/selection-guide.md` | Quando usar qual vector DB |
| `vector-db/indexing-strategies.md` | HNSW, IVF, performance tuning |

#### Semantic Caching
| Arquivo | Conteúdo |
|---------|----------|
| `caching/mongodb.md` | MongoDB semantic cache patterns |
| `caching/redis.md` | Redis semantic cache patterns |
| `caching/strategies.md` | Similarity threshold, TTL, invalidation |
| `caching/cost-optimization.md` | ROI de caching, métricas |

#### Agent Frameworks
| Arquivo | Conteúdo |
|---------|----------|
| `agent-frameworks/langchain.md` | LangChain patterns, quando usar/evitar |
| `agent-frameworks/langgraph.md` | State machines, graph-based agents |
| `agent-frameworks/custom-agents.md` | Build your own agent loop |
| `agent-frameworks/multi-agent.md` | Comunicação entre agents |
| `agent-frameworks/tool-integration.md` | Tool calling, API integration |

#### Frameworks Modernos
| Arquivo | Conteúdo |
|---------|----------|
| `frameworks/pydantic-ai.md` | Pydantic AI (Anthropic-recommended) |
| `frameworks/instructor.md` | Structured outputs library |
| `frameworks/dspy.md` | Programmatic prompting |
| `frameworks/marvin.md` | AI-powered functions |

#### Testing
| Arquivo | Conteúdo |
|---------|----------|
| `testing/llm-testing.md` | Strategies gerais, test pyramid |
| `testing/mocking.md` | Mock LLM calls, fixtures |
| `testing/evaluation.md` | LLM-as-judge, ragas, custom metrics |
| `testing/assertions.md` | Libraries (guardrails, etc.) |
| `testing/regression.md` | Golden datasets, snapshot testing |
| `testing/rag-evaluation.md` | RAG-specific metrics |

#### Observability
| Arquivo | Conteúdo |
|---------|----------|
| `observability/logging.md` | Structured logging para AI (prompts, responses, tokens) |
| `observability/tracing.md` | LangSmith, Phoenix, OpenTelemetry |
| `observability/metrics.md` | Latency, cost, tokens, quality |
| `observability/monitoring.md` | Alerting, dashboards, SLOs |
| `observability/debugging.md` | Debug LLM chains, agent loops |

#### Security
| Arquivo | Conteúdo |
|---------|----------|
| `security/prompt-injection.md` | Prevention patterns, input validation |
| `security/pii-handling.md` | Detection, sanitization, redaction |
| `security/content-filtering.md` | Safety layers, moderation APIs |
| `security/api-key-management.md` | Secrets, rotation, least privilege |
| `security/adversarial.md` | Jailbreaks, model exploits |

#### Production
| Arquivo | Conteúdo |
|---------|----------|
| `production/deployment.md` | API, serverless, containers |
| `production/scaling.md` | Queueing, batching, load balancing |
| `production/cost-optimization.md` | Model selection, caching, batching |
| `production/reliability.md` | Fallbacks, circuit breakers, retries |
| `production/gradual-rollout.md` | Feature flags, A/B testing |

#### Data Engineering
| Arquivo | Conteúdo |
|---------|----------|
| `data-engineering/dataset-preparation.md` | Cleaning, formatting, quality |
| `data-engineering/fine-tuning-data.md` | Patterns para fine-tuning |
| `data-engineering/synthetic-data.md` | Geração com LLMs |
| `data-engineering/versioning.md` | DVC, W&B, MLflow |

### Examples (Código Completo)

| Diretório | Conteúdo |
|-----------|----------|
| `examples/rag-system/` | RAG completo (chunking, embeddings, retrieval, generation) |
| `examples/agent-with-tools/` | Agent com tool calling |
| `examples/multi-provider-fallback/` | Fallback entre providers |
| `examples/semantic-cache/` | MongoDB/Redis semantic cache |

---

## Padrões de Uso

### Dev-Py usando AI-Engineer

**Cenário:** Usuário pede "cria um sistema RAG que responde perguntas sobre documentação"

**Dev-py consulta:**
1. `ai-engineer/references/rag/architecture.md` → escolhe pattern RAG
2. `ai-engineer/references/rag/chunking-strategies.md` → decide chunking
3. `ai-engineer/references/vector-db/qdrant.md` → setup Qdrant
4. `ai-engineer/references/llm-integration/anthropic-sdk.md` → client Claude
5. `ai-engineer/references/testing/rag-evaluation.md` → como testar
6. `ai-engineer/references/caching/mongodb.md` → semantic cache
7. `arch-py/references/python/async-patterns.md` → async embeddings
8. `arch-py/references/python/error-handling.md` → error handling

**Workflow:**
```
1. QUESTIONAR
   - Tipo de documentos? (markdown, pdf, html)
   - Volume de dados? (define chunking strategy)
   - Latência esperada? (define caching need)
   - Budget? (define provider selection)

2. PESQUISAR
   - Consulta ai-engineer skill para RAG patterns
   - Consulta arch-py skill para Python patterns

3. PROJETAR
   Opção A: Naive RAG
   - ✅ Simples, rápido para prototipar
   - ❌ Menor qualidade, menos controle

   Opção B: Advanced RAG (reranking, query transformation)
   - ✅ Maior qualidade
   - ❌ Mais complexidade, latência

   Recomendação: Opção A para MVP, migrar para B se necessário

4. TESTAR (test-first)
   - Golden dataset de perguntas/respostas
   - Métricas: faithfulness, relevance (ragas)
   - Mock LLM calls para unit tests

5. IMPLEMENTAR
   - Chunking → Embeddings → Qdrant → Retrieval → LLM → Cache
   - Seguindo arch-py (types, async, error handling)
   - Seguindo ai-engineer (RAG patterns, observability)

6. VALIDAR
   - mypy, ruff (arch-py)
   - ragas evaluation (ai-engineer)
   - Cost tracking

7. REVISAR
   - Auto-review contra ambas skills
```

### Review-Py usando AI-Engineer

**Checklist expandido para AI code:**

```markdown
## 🔒 Security (AI-specific)
- [ ] API keys não hardcoded (usar env vars)
- [ ] Prompt injection prevention (input validation)
- [ ] PII sanitization (antes de log e LLM)
- [ ] Content filtering (moderation API)
- [ ] Rate limiting configurado

Referência: ai-engineer/references/security/

## ⚡ Performance (AI-specific)
- [ ] Semantic caching configurado (MongoDB ou Redis)
- [ ] Batching de embeddings (paralelo com asyncio)
- [ ] Streaming usado para respostas longas
- [ ] Model selection justificada (não usar GPT-4 para tudo)

Referência: ai-engineer/references/production/cost-optimization.md

## 🧪 Testing (AI-specific)
- [ ] LLM calls mockados em unit tests
- [ ] Golden dataset para regression
- [ ] Evaluation metrics definidas (ragas, custom)
- [ ] Assertions para structured outputs (schema validation)

Referência: ai-engineer/references/testing/

## 💰 Cost Management
- [ ] Token counting e limits
- [ ] Cache hit rate tracking
- [ ] Model selection baseada em complexity
- [ ] Prompt optimization (compression)

Referência: ai-engineer/references/production/cost-optimization.md

## 📊 Observability
- [ ] Logging de prompts/responses (com PII redaction)
- [ ] Tracing configurado (LangSmith, Phoenix)
- [ ] Cost tracking por endpoint/user
- [ ] Quality metrics (latency, token usage)

Referência: ai-engineer/references/observability/

## 🏗️ Architecture (AI-specific)
- [ ] Fallback entre providers configurado
- [ ] RAG architecture justificada
- [ ] Agent loop com max iterations
- [ ] Separação de concerns (retrieval vs generation)

Referência: ai-engineer/references/rag/architecture.md
```

---

## Providers Suportados

### LLM Providers

| Provider | Models | Strengths | Use Cases |
|----------|--------|-----------|-----------|
| **Anthropic** | Claude 3.5 Sonnet, Opus | Longo contexto, segurança, tool use | RAG, agents, análise de docs |
| **OpenAI** | GPT-4, GPT-3.5 | Ecossistema maduro, function calling | Aplicações gerais, protótipos |
| **AWS Bedrock** | Claude, Llama, Titan | Multi-model, integração AWS | Enterprise, compliance |
| **Google Gemini** | Gemini Pro, Ultra | Multimodal, integração Google | Visão + texto, apps Google |
| **Meta (APIs)** | Llama 2, Llama 3 | Open source, custo baixo | Self-hosting, fine-tuning |

**Referências:**
- `llm-integration/anthropic-sdk.md`
- `llm-integration/openai-sdk.md`
- `llm-integration/bedrock.md`
- `llm-integration/gemini.md`
- `llm-integration/meta.md`
- `llm-integration/multi-provider.md`

### Vector Databases

| Database | Strengths | Use Cases |
|----------|-----------|-----------|
| **Qdrant** | Rust-based, rápido, filtros avançados | RAG, semantic search, produção |

**Referência:** `vector-db/qdrant.md`

### Semantic Caching

| Storage | Strengths | Use Cases |
|---------|-----------|-----------|
| **MongoDB** | Flexível, queries complexas | Cache com metadados, analytics |
| **Redis** | Ultra-rápido, TTL built-in | Cache high-throughput, low-latency |

**Referências:**
- `caching/mongodb.md`
- `caching/redis.md`

---

## Quick Start por Use Case

### Use Case 1: RAG System

**Goal:** Responder perguntas sobre documentação usando RAG

**Consulte:**
1. `rag/architecture.md` → escolha pattern (naive vs advanced)
2. `rag/chunking-strategies.md` → como quebrar docs
3. `vector-db/qdrant.md` → setup vector DB
4. `llm-integration/anthropic-sdk.md` → Claude para generation
5. `caching/redis.md` → semantic cache
6. `testing/rag-evaluation.md` → metrics (ragas)

**Stack recomendado:**
- LangChain para orchestration
- Qdrant para vectors
- Claude 3.5 Sonnet para generation
- Redis para semantic cache
- Ragas para evaluation

### Use Case 2: Agent com Tools

**Goal:** Agent que usa tools/APIs para resolver tasks

**Consulte:**
1. `agent-frameworks/langgraph.md` → state machine design
2. `llm-integration/tool-use.md` → tool calling patterns
3. `agent-frameworks/tool-integration.md` → API integration
4. `testing/llm-testing.md` → test agent loops
5. `observability/tracing.md` → debug agent behavior

**Stack recomendado:**
- LangGraph para agent loop
- Claude 3.5 Sonnet (excelente tool use)
- LangSmith para tracing
- pytest para testing

### Use Case 3: Multi-Provider System

**Goal:** Sistema com fallback entre providers para reliability

**Consulte:**
1. `llm-integration/multi-provider.md` → routing strategy
2. `production/reliability.md` → fallback patterns
3. `production/cost-optimization.md` → cost-based routing
4. `observability/metrics.md` → track provider performance

**Stack recomendado:**
- Anthropic (primary)
- OpenAI (fallback)
- Bedrock (enterprise fallback)
- Circuit breaker pattern
- Cost + latency tracking

### Use Case 4: Fine-Tuning Pipeline

**Goal:** Fine-tune modelo para task específico

**Consulte:**
1. `data-engineering/dataset-preparation.md` → data quality
2. `data-engineering/fine-tuning-data.md` → format patterns
3. `data-engineering/synthetic-data.md` → augmentation
4. `testing/evaluation.md` → eval metrics
5. `data-engineering/versioning.md` → data + model versioning

**Stack recomendado:**
- OpenAI fine-tuning API ou Bedrock
- W&B para experiment tracking
- DVC para data versioning
- Golden eval set

---

## Integração com Arch-Py

**SEMPRE use arch-py como fundação:**

### Type System
```python
# arch-py: Protocol, TypeVar
from typing import Protocol

class LLMProvider(Protocol):
    async def generate(self, prompt: str) -> str: ...

# ai-engineer: LLM response schemas
from pydantic import BaseModel

class RAGResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
```

### Async Patterns
```python
# arch-py: async/await, gather
import asyncio

# ai-engineer: parallel embeddings
embeddings = await asyncio.gather(
    *[embed_text(chunk) for chunk in chunks]
)
```

### Error Handling
```python
# arch-py: exception hierarchy
class AIServiceError(Exception): pass

# ai-engineer: rate limit handling
class RateLimitError(AIServiceError):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after
```

### Testing
```python
# arch-py: pytest, fixtures
import pytest

@pytest.fixture
def mock_llm():
    # ai-engineer: mock LLM responses
    return MockLLMClient(responses={...})
```

---

## Referências

### Arquivos desta Skill (por categoria)

**LLM Integration:**
- [llm-integration/anthropic-sdk.md](references/llm-integration/anthropic-sdk.md)
- [llm-integration/openai-sdk.md](references/llm-integration/openai-sdk.md)
- [llm-integration/bedrock.md](references/llm-integration/bedrock.md)
- [llm-integration/gemini.md](references/llm-integration/gemini.md)
- [llm-integration/meta.md](references/llm-integration/meta.md)
- [llm-integration/multi-provider.md](references/llm-integration/multi-provider.md)

**RAG:**
- [rag/architecture.md](references/rag/architecture.md)
- [rag/chunking-strategies.md](references/rag/chunking-strategies.md)
- [rag/evaluation.md](references/rag/evaluation.md)

**Caching:**
- [caching/mongodb.md](references/caching/mongodb.md)
- [caching/redis.md](references/caching/redis.md)

**Testing:**
- [testing/rag-evaluation.md](references/testing/rag-evaluation.md)
- [testing/mocking.md](references/testing/mocking.md)

**Observability:**
- [observability/tracing.md](references/observability/tracing.md)
- [observability/metrics.md](references/observability/metrics.md)

**Production:**
- [production/cost-optimization.md](references/production/cost-optimization.md)
- [production/reliability.md](references/production/reliability.md)

### Arch-Py Skill (Fundação Python)
- [../arch-py/SKILL.md](../arch-py/SKILL.md)
- [../arch-py/references/python/type-system.md](../arch-py/references/python/type-system.md)
- [../arch-py/references/python/async-patterns.md](../arch-py/references/python/async-patterns.md)
- [../arch-py/references/testing/pytest.md](../arch-py/references/testing/pytest.md)

### External Resources
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Ragas Documentation](https://docs.ragas.io/)
