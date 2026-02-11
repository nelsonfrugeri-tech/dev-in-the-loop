# RAG Architecture - Retrieval-Augmented Generation Patterns

Referência completa de arquiteturas RAG, do mais simples ao mais avançado.

---

## O que é RAG?

**Retrieval-Augmented Generation** = Retrieval (buscar docs relevantes) + Generation (LLM gera resposta usando docs)

**Por que usar RAG:**
- Reduz hallucination (LLM se baseia em fatos)
- Informação atualizada (não limitado ao training data)
- Transparência (pode citar fontes)
- Cost-effective (não precisa fine-tuning)

---

## RAG Architecture Evolution

### 1. Naive RAG (Básico)

**Pipeline:**
```
User Query → Embeddings → Vector Search → Top-K Docs → LLM → Response
```

**Prós:**
- ✅ Simples de implementar
- ✅ Rápido para prototipar
- ✅ Funciona para casos simples

**Contras:**
- ❌ Qualidade limitada (sem reranking)
- ❌ Sem query optimization
- ❌ Retrieval pode falhar em queries complexas

**Quando usar:**
- MVPs e protótipos
- Conhecimento simples e direto
- Budget limitado

**Exemplo:**
```python
from typing import Protocol

class VectorDB(Protocol):
    async def search(self, query_embedding: list[float], k: int) -> list[str]: ...

class LLMClient(Protocol):
    async def generate(self, prompt: str) -> str: ...

async def naive_rag(
    query: str,
    vector_db: VectorDB,
    llm: LLMClient,
    embedding_fn: Callable[[str], list[float]]
) -> str:
    """Naive RAG implementation."""
    # 1. Embed query
    query_embedding = await embedding_fn(query)

    # 2. Search vector DB
    docs = await vector_db.search(query_embedding, k=3)

    # 3. Create prompt with docs
    context = "\n\n".join(docs)
    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer:"""

    # 4. Generate response
    return await llm.generate(prompt)
```

---

### 2. Advanced RAG (Produção)

**Pipeline:**
```
User Query
  → Query Transformation (rewrite, decompose)
  → Multi-Query Generation
  → Embeddings
  → Hybrid Search (vector + keyword)
  → Reranking (reorder by relevance)
  → Context Compression (filter irrelevant)
  → LLM → Response
  → Citation Extraction
```

**Melhorias sobre Naive:**
- ✅ Query transformation melhora retrieval
- ✅ Hybrid search (vector + BM25) mais robusto
- ✅ Reranking melhora relevância
- ✅ Compression reduz noise no context
- ✅ Citations aumentam transparência

**Quando usar:**
- Produção com qualidade alta
- Queries complexas
- Domínios especializados

**Componentes:**

#### 2.1 Query Transformation

```python
async def transform_query(query: str, llm: LLMClient) -> list[str]:
    """Transform query into multiple optimized versions."""
    prompt = f"""You are a query optimizer. Given a user query, generate 3 variations
that would help retrieve relevant documents:
1. A more specific version
2. A broader version
3. A rephrased version focusing on key concepts

Query: {query}

Return as JSON array of strings."""

    response = await llm.generate(prompt)
    return json.loads(response)

# Example:
# Input: "How do I deploy FastAPI?"
# Output: [
#   "FastAPI production deployment best practices",
#   "Python API deployment",
#   "Deploy asynchronous Python web applications"
# ]
```

#### 2.2 Hybrid Search

```python
from typing import TypedDict

class SearchResult(TypedDict):
    doc_id: str
    score: float
    text: str

async def hybrid_search(
    query: str,
    query_embedding: list[float],
    vector_db: VectorDB,
    keyword_search_fn: Callable[[str], list[SearchResult]],
    k: int = 10
) -> list[SearchResult]:
    """Combine vector and keyword search."""
    # Vector search
    vector_results = await vector_db.search(query_embedding, k=k)

    # Keyword search (BM25, Elasticsearch, etc.)
    keyword_results = await keyword_search_fn(query)

    # Merge with RRF (Reciprocal Rank Fusion)
    return reciprocal_rank_fusion(vector_results, keyword_results, k=k)

def reciprocal_rank_fusion(
    results1: list[SearchResult],
    results2: list[SearchResult],
    k: int = 60
) -> list[SearchResult]:
    """Merge results using RRF algorithm."""
    scores: dict[str, float] = {}

    for rank, result in enumerate(results1):
        scores[result["doc_id"]] = scores.get(result["doc_id"], 0) + 1 / (k + rank + 1)

    for rank, result in enumerate(results2):
        scores[result["doc_id"]] = scores.get(result["doc_id"], 0) + 1 / (k + rank + 1)

    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Return top-k merged results
    all_results = {r["doc_id"]: r for r in results1 + results2}
    return [all_results[doc_id] for doc_id in sorted_ids[:10]]
```

#### 2.3 Reranking

```python
from sentence_transformers import CrossEncoder

class Reranker:
    """Rerank results using cross-encoder."""

    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    async def rerank(
        self,
        query: str,
        docs: list[str],
        top_k: int = 3
    ) -> list[tuple[str, float]]:
        """Rerank docs by relevance to query."""
        # Score all query-doc pairs
        pairs = [(query, doc) for doc in docs]
        scores = self.model.predict(pairs)

        # Sort by score
        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]
```

#### 2.4 Context Compression

```python
async def compress_context(
    query: str,
    docs: list[str],
    llm: LLMClient
) -> list[str]:
    """Remove irrelevant parts from retrieved docs."""
    compressed = []

    for doc in docs:
        prompt = f"""Extract only the parts of this document that are relevant to answering the query.
Remove any irrelevant information.

Query: {query}

Document:
{doc}

Relevant excerpt (return empty if nothing relevant):"""

        excerpt = await llm.generate(prompt)
        if excerpt.strip():
            compressed.append(excerpt)

    return compressed
```

#### 2.5 Full Advanced RAG

```python
async def advanced_rag(
    query: str,
    vector_db: VectorDB,
    llm: LLMClient,
    embedding_fn: Callable[[str], list[float]],
    keyword_search_fn: Callable[[str], list[SearchResult]],
    reranker: Reranker
) -> tuple[str, list[str]]:
    """Advanced RAG with all optimizations."""
    # 1. Transform query
    query_variations = await transform_query(query, llm)
    all_queries = [query] + query_variations

    # 2. Embed all queries
    embeddings = [await embedding_fn(q) for q in all_queries]

    # 3. Hybrid search for each query variation
    all_results = []
    for q, emb in zip(all_queries, embeddings):
        results = await hybrid_search(q, emb, vector_db, keyword_search_fn)
        all_results.extend(results)

    # Deduplicate
    unique_docs = {r["doc_id"]: r["text"] for r in all_results}
    docs = list(unique_docs.values())

    # 4. Rerank
    reranked_docs = await reranker.rerank(query, docs, top_k=5)
    top_docs = [doc for doc, score in reranked_docs]

    # 5. Compress context
    compressed_docs = await compress_context(query, top_docs, llm)

    # 6. Generate response
    context = "\n\n---\n\n".join(compressed_docs)
    prompt = f"""Answer the question based on the context below. Include citations.

Context:
{context}

Question: {query}

Answer (include [1], [2] citations):"""

    response = await llm.generate(prompt)

    return response, compressed_docs
```

---

### 3. Agentic RAG (Mais Avançado)

**Pipeline:**
```
User Query
  → Agent Loop:
    → Agent decide: retrieval necessário?
    → Se sim: query transformation + retrieval + rerank
    → Agent avalia: informação suficiente?
    → Se não: refine query, retrieve novamente
    → Se sim: generate response
  → Response + Citations
```

**Características:**
- ✅ Agent decide quando/como fazer retrieval
- ✅ Iterative refinement (busca múltiplas vezes se necessário)
- ✅ Self-correction (identifica respostas ruins)
- ✅ Multi-step reasoning

**Quando usar:**
- Queries que requerem múltiplos passos
- Domínios onde primeira busca pode não ser suficiente
- Quando qualidade > latência/custo

**Exemplo com LangGraph:**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

class RAGState(TypedDict):
    """State for agentic RAG."""
    query: str
    docs_retrieved: list[str]
    answer: str | None
    iterations: int
    max_iterations: int

def should_retrieve(state: RAGState) -> bool:
    """Decide if more retrieval is needed."""
    if state["iterations"] >= state["max_iterations"]:
        return False

    if not state["docs_retrieved"]:
        return True  # No docs yet, must retrieve

    # Ask LLM if current docs are sufficient
    # ... (LLM call to evaluate)
    return False  # Simplified

async def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve documents."""
    # Advanced RAG retrieval
    docs = await advanced_rag_retrieval(state["query"])

    return {
        **state,
        "docs_retrieved": docs,
        "iterations": state["iterations"] + 1
    }

async def generate_node(state: RAGState) -> RAGState:
    """Generate answer."""
    context = "\n\n".join(state["docs_retrieved"])
    prompt = f"""Answer based on context:

{context}

Question: {state["query"]}"""

    answer = await llm.generate(prompt)

    return {**state, "answer": answer}

# Build graph
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

graph.add_conditional_edges(
    "retrieve",
    should_retrieve,
    {
        True: "retrieve",  # Retrieve again
        False: "generate"  # Enough docs, generate
    }
)

graph.add_edge("generate", END)
graph.set_entry_point("retrieve")

app = graph.compile()

# Run
result = await app.ainvoke({
    "query": "Complex multi-step question",
    "docs_retrieved": [],
    "answer": None,
    "iterations": 0,
    "max_iterations": 3
})
```

---

## Component Deep Dives

### Chunking Strategies

Ver: `rag/chunking-strategies.md`

**Resumo:**
- Fixed-size (simples, pode quebrar contexto)
- Semantic (mantém contexto, mais complexo)
- Recursive (hierárquico, bom para code/docs)
- Document-aware (respeita estrutura do doc)

### Embeddings

Ver: `rag/embeddings.md`

**Modelos recomendados:**
- OpenAI `text-embedding-3-small` (custo-benefício)
- OpenAI `text-embedding-3-large` (melhor qualidade)
- Cohere `embed-multilingual-v3.0` (multilingual)
- `all-MiniLM-L6-v2` (local, rápido)

### Vector DBs

Ver: `vector-db/qdrant.md`

**Qdrant setup:**
- Collections
- Indexing (HNSW)
- Filters (metadata)
- Hybrid search

---

## Evaluation

Ver: `rag/evaluation.md`

**Métricas chave:**
1. **Retrieval Metrics:**
   - Precision@K
   - Recall@K
   - MRR (Mean Reciprocal Rank)

2. **Generation Metrics:**
   - Faithfulness (resposta baseada no context?)
   - Relevance (responde a query?)
   - Coherence (faz sentido?)

3. **End-to-End:**
   - Correctness (resposta correta?)
   - Completeness (resposta completa?)
   - User satisfaction

**Tools:** ragas, trulens

---

## Production Considerations

### Caching

**Semantic caching para queries similares:**

```python
from redis import Redis
import numpy as np

class SemanticCache:
    """Cache using embedding similarity."""

    def __init__(self, redis_client: Redis, threshold: float = 0.95):
        self.redis = redis_client
        self.threshold = threshold

    async def get(self, query: str, query_embedding: list[float]) -> str | None:
        """Get cached response if query is similar enough."""
        # Get all cached queries
        cached = self.redis.hgetall("rag:cache")

        for cached_query, cached_response in cached.items():
            cached_embedding = json.loads(self.redis.hget("rag:embeddings", cached_query))

            # Compute cosine similarity
            similarity = cosine_similarity(query_embedding, cached_embedding)

            if similarity >= self.threshold:
                return cached_response.decode()

        return None

    async def set(
        self,
        query: str,
        query_embedding: list[float],
        response: str,
        ttl: int = 3600
    ):
        """Cache query-response pair."""
        self.redis.hset("rag:cache", query, response)
        self.redis.hset("rag:embeddings", query, json.dumps(query_embedding))
        self.redis.expire("rag:cache", ttl)
```

Ver: `caching/redis.md`, `caching/mongodb.md`

### Cost Optimization

**Model selection por query complexity:**

```python
async def classify_query_complexity(query: str) -> Literal["simple", "medium", "complex"]:
    """Classify query complexity to select appropriate model."""
    # Use cheap model to classify
    prompt = f"""Classify this query complexity as: simple, medium, or complex.

Simple: factual, direct answer
Medium: requires some reasoning
Complex: multi-step, requires deep analysis

Query: {query}

Classification (one word):"""

    response = await cheap_llm.generate(prompt)
    return response.strip().lower()

async def rag_with_model_selection(query: str) -> str:
    """Select model based on query complexity."""
    complexity = await classify_query_complexity(query)

    model_map = {
        "simple": "claude-3-5-haiku-20241022",     # Fastest, cheapest
        "medium": "claude-3-5-sonnet-20241022",    # Balanced
        "complex": "claude-3-opus-20240229"        # Most capable
    }

    selected_model = model_map[complexity]

    # Run RAG with selected model
    return await rag(query, model=selected_model)
```

---

## Common Pitfalls

### 1. Chunk Size Too Large
- **Problem:** LLM context overflow, slow retrieval
- **Solution:** Use 512-1024 tokens per chunk

### 2. No Reranking
- **Problem:** Retrieved docs may not be most relevant
- **Solution:** Add reranker (cross-encoder)

### 3. No Query Optimization
- **Problem:** Poor retrieval for complex queries
- **Solution:** Query transformation, multi-query

### 4. No Evaluation
- **Problem:** Don't know if RAG is working well
- **Solution:** ragas metrics, human eval

### 5. No Caching
- **Problem:** High cost, high latency for repeated queries
- **Solution:** Semantic cache (Redis, MongoDB)

---

## Architecture Decision Tree

```
Need RAG?
  ├─ No: Just use LLM
  └─ Yes:
      ├─ Simple use case (FAQ, docs lookup)?
      │   └─ Naive RAG
      │
      ├─ Production, high quality required?
      │   └─ Advanced RAG (hybrid search + reranking)
      │
      └─ Complex multi-step queries?
          └─ Agentic RAG (LangGraph)
```

---

## References

- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [Advanced RAG Techniques (Pinecone)](https://www.pinecone.io/learn/advanced-rag-techniques/)
- [RAG Evaluation (ragas)](https://docs.ragas.io/)
- [Chunking Strategies](rag/chunking-strategies.md)
- [Embeddings Guide](rag/embeddings.md)
- [Vector DB: Qdrant](../vector-db/qdrant.md)
