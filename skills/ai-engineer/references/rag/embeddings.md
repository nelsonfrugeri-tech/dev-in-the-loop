# Embeddings - Vector Representations para RAG

Referência completa de embeddings para RAG systems.

---

## O que são Embeddings?

**Embeddings** = Representações vetoriais de texto que capturam significado semântico

**Propriedade chave:** Textos semanticamente similares têm embeddings similares (próximos no espaço vetorial)

**Exemplo:**
```python
embedding("cachorro") ≈ embedding("cão")  # Similar
embedding("cachorro") ≠ embedding("computador")  # Diferente
```

**No RAG:**
1. Documents → Embeddings → Vector DB
2. Query → Embedding → Busca vetores similares → Documentos relevantes

---

## Modelos de Embedding (2026)

### 1. OpenAI Embeddings (Recomendado para maioria)

**Modelos disponíveis:**

| Model | Dimensions | Cost (per 1M tokens) | Best For |
|-------|------------|---------------------|----------|
| `text-embedding-3-small` | 1536 | $0.02 | Custo-benefício, produção |
| `text-embedding-3-large` | 3072 | $0.13 | Máxima qualidade |
| `text-embedding-ada-002` | 1536 | $0.10 | Legacy (usar small em vez) |

**Recomendação:** `text-embedding-3-small` para 95% dos casos

**Prós:**
- ✅ Excelente qualidade
- ✅ API simples
- ✅ Multilingual
- ✅ Suporta dimensionality reduction

**Contras:**
- ❌ Custo (pago por token)
- ❌ Depende de API externa
- ❌ Rate limits

**Implementação:**

```python
from openai import AsyncOpenAI
import os

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

async def embed_text(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Generate embedding usando OpenAI."""
    response = await client.embeddings.create(
        model=model,
        input=text,
        encoding_format="float"  # ou "base64" para economizar bandwidth
    )

    return response.data[0].embedding

# Batch embeddings (mais eficiente)
async def embed_batch(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100
) -> list[list[float]]:
    """Batch embed multiple texts."""
    all_embeddings = []

    # Process em batches (API limit: 2048 texts por request)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = await client.embeddings.create(
            model=model,
            input=batch,
            encoding_format="float"
        )

        embeddings = [data.embedding for data in response.data]
        all_embeddings.extend(embeddings)

    return all_embeddings

# Usage
embedding = await embed_text("What is RAG?")
print(f"Embedding dimension: {len(embedding)}")  # 1536 for small
```

**Dimensionality Reduction (economizar storage):**

```python
async def embed_with_reduced_dimensions(
    text: str,
    dimensions: int = 512  # Reduzir de 1536 para 512
) -> list[float]:
    """Embed com dimensões reduzidas (mantém ~98% da qualidade)."""
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        dimensions=dimensions  # Suporta qualquer valor até 1536
    )

    return response.data[0].embedding

# 3x menos storage, ~98% da qualidade
```

---

### 2. Cohere Embeddings

**Modelos disponíveis:**

| Model | Dimensions | Best For |
|-------|------------|----------|
| `embed-english-v3.0` | 1024 | Inglês apenas, alta qualidade |
| `embed-multilingual-v3.0` | 1024 | 100+ idiomas |
| `embed-english-light-v3.0` | 384 | Rápido, menor |

**Prós:**
- ✅ Multilingual excelente
- ✅ Suporta input types (search_document, search_query)
- ✅ Compression built-in

**Contras:**
- ❌ Pago (similar a OpenAI)
- ❌ Menos usado que OpenAI

**Implementação:**

```python
import cohere
import os

co = cohere.AsyncClient(api_key=os.environ.get("COHERE_API_KEY"))

async def embed_cohere(
    texts: list[str],
    input_type: Literal["search_document", "search_query", "classification"] = "search_document",
    model: str = "embed-multilingual-v3.0"
) -> list[list[float]]:
    """Embed usando Cohere.

    Args:
        texts: Textos para embed
        input_type:
            - "search_document": para documentos sendo indexed
            - "search_query": para queries de busca
            - "classification": para tarefas de classificação
    """
    response = await co.embed(
        texts=texts,
        model=model,
        input_type=input_type
    )

    return response.embeddings

# Usage - Index documents
doc_embeddings = await embed_cohere(
    ["Document 1", "Document 2"],
    input_type="search_document"
)

# Usage - Query
query_embedding = await embed_cohere(
    ["What is RAG?"],
    input_type="search_query"
)
```

---

### 3. Sentence Transformers (Self-Hosted, Free)

**Modelos recomendados:**

| Model | Dimensions | Best For | Size |
|-------|------------|----------|------|
| `all-MiniLM-L6-v2` | 384 | Rápido, leve, geral | 80 MB |
| `all-mpnet-base-v2` | 768 | Melhor qualidade | 420 MB |
| `multi-qa-mpnet-base-dot-v1` | 768 | QA/RAG específico | 420 MB |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | Multilingual | 970 MB |

**Prós:**
- ✅ Totalmente gratuito
- ✅ Self-hosted (sem rate limits)
- ✅ Rápido (GPU)
- ✅ Privacy (dados não saem)

**Contras:**
- ❌ Qualidade menor que OpenAI/Cohere
- ❌ Requer infraestrutura (GPU para produção)
- ❌ Precisa gerenciar modelos

**Implementação:**

```python
from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    """Wrapper para Sentence Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        """Initialize model.

        Args:
            model_name: Nome do modelo no HuggingFace
            device: "cuda" para GPU, "cpu" para CPU, None = auto-detect
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(model_name, device=device)
        self.device = device

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts (sync)."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings.tolist()

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """Embed texts (async - roda em thread pool)."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed, texts)

# Usage
model = EmbeddingModel("all-MiniLM-L6-v2")

# Sync
embeddings = model.embed(["Text 1", "Text 2"])

# Async
embeddings = await model.embed_async(["Text 1", "Text 2"])
```

**Batch processing para produção:**

```python
def embed_batch_optimized(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32
) -> list[list[float]]:
    """Batch embed com controle de memória."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True)
        all_embeddings.extend(embeddings.tolist())

    return all_embeddings
```

---

### 4. AWS Bedrock Embeddings

**Modelos disponíveis:**

| Provider | Model | Dimensions |
|----------|-------|------------|
| Amazon | `amazon.titan-embed-text-v1` | 1536 |
| Amazon | `amazon.titan-embed-text-v2` | 1024 |
| Cohere | `cohere.embed-english-v3` | 1024 |
| Cohere | `cohere.embed-multilingual-v3` | 1024 |

**Prós:**
- ✅ Integrado com AWS ecosystem
- ✅ Compliance (dados ficam na AWS)
- ✅ Múltiplos providers

**Contras:**
- ❌ Setup mais complexo (IAM, regions)
- ❌ Menor comunidade

**Implementação:**

```python
import boto3
import json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

async def embed_bedrock(
    text: str,
    model_id: str = "amazon.titan-embed-text-v2"
) -> list[float]:
    """Embed usando AWS Bedrock."""
    body = json.dumps({"inputText": text})

    response = bedrock.invoke_model(
        modelId=model_id,
        body=body
    )

    response_body = json.loads(response["body"].read())
    return response_body["embedding"]
```

---

## Escolhendo Modelo de Embedding

### Decision Tree

```
Qual embedding usar?

Precisa ser self-hosted / privacy?
  └─ Yes → Sentence Transformers (all-MiniLM-L6-v2)

Precisa de multilingual excelente?
  └─ Yes → Cohere (embed-multilingual-v3.0)

Precisa de integração AWS?
  └─ Yes → Bedrock (Titan v2)

Precisa de melhor qualidade?
  └─ Yes → OpenAI (text-embedding-3-large)

Default (custo-benefício)
  └─ OpenAI (text-embedding-3-small)
```

### Comparação de Qualidade

**Benchmark (MTEB - Massive Text Embedding Benchmark):**

| Model | Avg Score | Cost | Latency |
|-------|-----------|------|---------|
| OpenAI text-embedding-3-large | 64.6 | $$$ | Baixa (API) |
| Cohere embed-multilingual-v3 | 64.5 | $$$ | Baixa (API) |
| OpenAI text-embedding-3-small | 62.3 | $$ | Baixa (API) |
| all-mpnet-base-v2 | 57.8 | Free | Alta (CPU) / Baixa (GPU) |
| all-MiniLM-L6-v2 | 56.3 | Free | Baixa (CPU/GPU) |

**Recomendação por caso:**

- **Produção (custo OK):** OpenAI text-embedding-3-small
- **Produção (máxima qualidade):** OpenAI text-embedding-3-large
- **Self-hosted (qualidade):** all-mpnet-base-v2
- **Self-hosted (speed):** all-MiniLM-L6-v2
- **Multilingual:** Cohere embed-multilingual-v3
- **Enterprise/AWS:** Bedrock Titan v2

---

## Best Practices

### 1. Normalize Embeddings

```python
import numpy as np

def normalize_embedding(embedding: list[float]) -> list[float]:
    """Normalize embedding to unit length.

    Importante para cosine similarity (se vector DB não normaliza automaticamente).
    """
    vec = np.array(embedding)
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist()

# Usage
embedding = await embed_text("Query here")
normalized = normalize_embedding(embedding)
```

### 2. Cache Embeddings

**Embeddings são caros e não mudam. SEMPRE cache!**

```python
from functools import lru_cache
import hashlib
import json

# In-memory cache (para development)
@lru_cache(maxsize=1000)
def embed_cached(text: str) -> list[float]:
    """Cached embedding (sync only)."""
    # Call real embedding API
    return embed_text_sync(text)

# Redis cache (produção)
from redis import Redis

class EmbeddingCache:
    """Cache embeddings em Redis."""

    def __init__(self, redis_client: Redis, ttl: int = 86400 * 30):  # 30 days
        self.redis = redis_client
        self.ttl = ttl

    def _hash_text(self, text: str) -> str:
        """Hash text para key."""
        return hashlib.sha256(text.encode()).hexdigest()

    async def get_or_embed(
        self,
        text: str,
        embed_fn: Callable[[str], Awaitable[list[float]]]
    ) -> list[float]:
        """Get cached embedding ou gera novo."""
        key = f"embedding:{self._hash_text(text)}"

        # Check cache
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)

        # Generate embedding
        embedding = await embed_fn(text)

        # Cache it
        self.redis.setex(key, self.ttl, json.dumps(embedding))

        return embedding

# Usage
cache = EmbeddingCache(redis_client)
embedding = await cache.get_or_embed("Query", embed_text)
```

### 3. Batch Sempre que Possível

```python
# ❌ Um por vez (lento, caro)
embeddings = []
for text in texts:
    emb = await embed_text(text)
    embeddings.append(emb)

# ✅ Batch (rápido, econômico)
embeddings = await embed_batch(texts, batch_size=100)
```

**Savings:**
- Latência: ~10-50x mais rápido
- Custo: Mesmo preço, mas menos overhead

### 4. Async Embeddings para Produção

```python
import asyncio

async def embed_documents_async(
    documents: list[str],
    batch_size: int = 100
) -> list[list[float]]:
    """Embed documents com parallelism."""
    # Split em batches
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

    # Process batches em paralelo (cuidado com rate limits!)
    tasks = [embed_batch(batch) for batch in batches]
    batch_results = await asyncio.gather(*tasks)

    # Flatten results
    return [emb for batch in batch_results for emb in batch]

# Usage
embeddings = await embed_documents_async(documents, batch_size=100)
```

### 5. Handle Rate Limits

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import anthropic

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
async def embed_with_retry(text: str) -> list[float]:
    """Embed com auto-retry em rate limits."""
    try:
        return await embed_text(text)
    except Exception as e:
        if "rate_limit" in str(e).lower():
            print(f"Rate limit hit, retrying...")
            raise  # tenacity will retry
        raise
```

---

## Similarity Metrics

### 1. Cosine Similarity (Mais Comum)

**Quando usar:** Default para embeddings normalized

```python
import numpy as np

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity (-1 to 1).

    1 = identical
    0 = orthogonal
    -1 = opposite
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    return dot / (norm1 * norm2)

# Usage
sim = cosine_similarity(embedding1, embedding2)
print(f"Similarity: {sim:.4f}")  # 0.0 to 1.0
```

### 2. Euclidean Distance

**Quando usar:** Embeddings não normalized

```python
def euclidean_distance(vec1: list[float], vec2: list[float]) -> float:
    """Compute Euclidean distance.

    0 = identical
    Maior = mais diferente
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.linalg.norm(v1 - v2)
```

### 3. Dot Product

**Quando usar:** Embeddings JÁ normalized, mais rápido que cosine

```python
def dot_product(vec1: list[float], vec2: list[float]) -> float:
    """Compute dot product (embeddings normalized).

    Faster than cosine if embeddings are already normalized.
    """
    return np.dot(vec1, vec2)
```

**Equivalência:**
```python
# Se embeddings são normalized (norm = 1):
cosine_similarity(v1, v2) == dot_product(v1, v2)
```

---

## Advanced Techniques

### 1. Query Prefix para Embedding

**Alguns modelos (Cohere, Instructor) performam melhor com prefixes:**

```python
async def embed_with_instruction(
    text: str,
    instruction: str = "Represent this sentence for retrieval:"
) -> list[float]:
    """Embed com instruction prefix."""
    prefixed_text = f"{instruction} {text}"
    return await embed_text(prefixed_text)

# Query
query_emb = await embed_with_instruction(
    "What is RAG?",
    instruction="Represent this query for searching relevant documents:"
)

# Document
doc_emb = await embed_with_instruction(
    "RAG is Retrieval-Augmented Generation",
    instruction="Represent this document for retrieval:"
)
```

### 2. Embedding de Chunks com Context

**Adicione contexto ao chunk antes de embedding:**

```python
async def embed_chunk_with_context(
    chunk_text: str,
    doc_title: str,
    section_headers: list[str]
) -> list[float]:
    """Embed chunk com contexto do documento.

    Ajuda retrieval a ter mais contexto semântico.
    """
    # Build context
    headers = " > ".join(section_headers)
    contextual_text = f"{doc_title}\n{headers}\n\n{chunk_text}"

    return await embed_text(contextual_text)

# Exemplo
embedding = await embed_chunk_with_context(
    chunk_text="This explains RAG systems...",
    doc_title="RAG Tutorial",
    section_headers=["Introduction", "What is RAG"]
)
```

### 3. Hybrid Embeddings (Multi-Vector)

**Combine diferentes embeddings:**

```python
async def hybrid_embedding(text: str) -> dict[str, list[float]]:
    """Generate multiple embeddings para hybrid search."""
    # Embedding semântico
    semantic = await embed_text(text, model="text-embedding-3-small")

    # Embedding de keywords (BM25-like, usando modelo específico)
    # keyword = await embed_sparse(text)

    return {
        "semantic": semantic,
        # "keyword": keyword
    }
```

---

## Troubleshooting

### Problema: Low Retrieval Quality

**Diagnóstico:**
1. Embedding model inadequado para domínio
2. Chunk size errado
3. Query muito diferente de documents

**Soluções:**
```python
# 1. Teste modelos diferentes
models = ["text-embedding-3-small", "text-embedding-3-large"]
for model in models:
    evaluate_retrieval(model)

# 2. Experimente chunk sizes
for size in [256, 512, 1024]:
    chunks = chunking(text, size)
    evaluate_retrieval(chunks)

# 3. Query transformation
transformed_queries = await transform_query(query)
```

### Problema: High Latency

**Diagnóstico:**
- Embedding API calls síncronos
- Sem batching
- Sem caching

**Soluções:**
```python
# 1. Async + batch
embeddings = await embed_batch_async(texts)

# 2. Cache
embeddings = await cache.get_or_embed(text, embed_fn)

# 3. Pre-compute embeddings (offline)
# Embed todos documents antecipadamente
```

### Problema: High Cost

**Diagnóstico:**
- Re-embedding documentos
- Embedding queries múltiplas vezes
- Usando modelo muito caro

**Soluções:**
```python
# 1. Cache TUDO
embedding_cache = EmbeddingCache(redis)

# 2. Dimensionality reduction
embedding = await embed_with_reduced_dimensions(text, dimensions=512)

# 3. Use modelo mais barato
# text-embedding-3-large → text-embedding-3-small (6.5x cheaper)

# 4. Self-host
model = SentenceTransformer("all-MiniLM-L6-v2")  # Free
```

---

## References

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Sentence Transformers](https://www.sbert.net/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Cohere Embed](https://docs.cohere.com/docs/embeddings)
- [RAG Architecture](rag/architecture.md)
- [Chunking Strategies](rag/chunking-strategies.md)
- [Vector DB: Qdrant](../vector-db/qdrant.md)
