# RAG Evaluation - Como Medir Qualidade de RAG Systems

Referência completa de metrics e frameworks para avaliar RAG systems.

---

## Por que Avaliar RAG?

**Problema:** RAG systems são não-determinísticos
- Retrieval pode falhar
- LLM pode hallucinar
- Sem avaliação, impossível melhorar

**Solução:** Métricas objetivas + avaliação sistemática

**O que medir:**
1. **Retrieval Quality** → Documentos certos foram recuperados?
2. **Generation Quality** → Resposta é correta e baseada nos docs?
3. **End-to-End Quality** → Sistema funciona bem como um todo?

---

## Métricas de Retrieval

### 1. Precision@K

**Definição:** % de documentos recuperados que são relevantes

```
Precision@K = (Documentos relevantes nos top-K) / K
```

**Exemplo:**
- Top-5 documentos: [relevante, irrelevante, relevante, relevante, irrelevante]
- Precision@5 = 3/5 = 0.6 (60%)

**Implementação:**

```python
def precision_at_k(
    retrieved_docs: list[str],
    relevant_docs: set[str],
    k: int
) -> float:
    """Calculate Precision@K.

    Args:
        retrieved_docs: Documentos recuperados (em ordem)
        relevant_docs: Set de doc IDs relevantes (ground truth)
        k: Número de top docs a considerar

    Returns:
        Precision score (0-1)
    """
    top_k = retrieved_docs[:k]
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
    return relevant_in_top_k / k if k > 0 else 0.0

# Exemplo
retrieved = ["doc1", "doc5", "doc2", "doc8", "doc3"]
relevant = {"doc1", "doc2", "doc3", "doc4"}  # Ground truth

precision_5 = precision_at_k(retrieved, relevant, k=5)
print(f"Precision@5: {precision_5:.2%}")  # 60%
```

**Quando usar:** Quando precisão importa mais que recall (ex: busca onde usuário vê poucos resultados)

---

### 2. Recall@K

**Definição:** % de documentos relevantes que foram recuperados

```
Recall@K = (Documentos relevantes nos top-K) / (Total de docs relevantes)
```

**Exemplo:**
- Top-5: [doc1, doc5, doc2, doc8, doc3]
- Relevantes: {doc1, doc2, doc3, doc4} (total: 4)
- Recuperados relevantes: {doc1, doc2, doc3} (total: 3)
- Recall@5 = 3/4 = 0.75 (75%)

**Implementação:**

```python
def recall_at_k(
    retrieved_docs: list[str],
    relevant_docs: set[str],
    k: int
) -> float:
    """Calculate Recall@K."""
    top_k = retrieved_docs[:k]
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
    total_relevant = len(relevant_docs)

    return relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0

# Exemplo
recall_5 = recall_at_k(retrieved, relevant, k=5)
print(f"Recall@5: {recall_5:.2%}")  # 75%
```

**Quando usar:** Quando importante não perder documentos relevantes

---

### 3. MRR (Mean Reciprocal Rank)

**Definição:** Média da posição do PRIMEIRO documento relevante

```
RR = 1 / (posição do primeiro relevante)
MRR = média de RR para todas queries
```

**Exemplo:**
- Query 1: primeiro relevante na posição 2 → RR = 1/2 = 0.5
- Query 2: primeiro relevante na posição 1 → RR = 1/1 = 1.0
- Query 3: primeiro relevante na posição 5 → RR = 1/5 = 0.2
- MRR = (0.5 + 1.0 + 0.2) / 3 = 0.57

**Implementação:**

```python
def reciprocal_rank(retrieved_docs: list[str], relevant_docs: set[str]) -> float:
    """Calculate Reciprocal Rank para uma query."""
    for i, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant_docs:
            return 1.0 / i
    return 0.0  # Nenhum relevante encontrado

def mean_reciprocal_rank(
    queries_results: list[tuple[list[str], set[str]]]
) -> float:
    """Calculate MRR para múltiplas queries.

    Args:
        queries_results: Lista de (retrieved_docs, relevant_docs) para cada query

    Returns:
        MRR score (0-1)
    """
    rr_scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in queries_results
    ]

    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0

# Exemplo
queries = [
    (["doc2", "doc1", "doc3"], {"doc1", "doc3"}),  # RR = 1/2
    (["doc5", "doc2", "doc1"], {"doc1"}),           # RR = 1/3
    (["doc1", "doc4", "doc5"], {"doc1", "doc4"}),  # RR = 1/1
]

mrr = mean_reciprocal_rank(queries)
print(f"MRR: {mrr:.2%}")
```

**Quando usar:** Quando posição do primeiro resultado importa (search ranking)

---

### 4. NDCG (Normalized Discounted Cumulative Gain)

**Definição:** Considera relevância graduada e posição

**Mais sofisticado:** Documentos podem ter relevância 0-3 (não só 0/1)

**Implementação:**

```python
import numpy as np

def dcg_at_k(relevances: list[float], k: int) -> float:
    """Calculate DCG@K.

    Args:
        relevances: Lista de relevance scores (0-3) na ordem retrieved
        k: Top-k docs

    Returns:
        DCG score
    """
    relevances = np.array(relevances[:k])
    gains = 2 ** relevances - 1
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(gains / discounts)

def ndcg_at_k(
    retrieved_relevances: list[float],
    ideal_relevances: list[float],
    k: int
) -> float:
    """Calculate NDCG@K.

    Args:
        retrieved_relevances: Relevances dos docs recuperados (em ordem)
        ideal_relevances: Relevances ideais (sorted desc)
        k: Top-k docs

    Returns:
        NDCG score (0-1)
    """
    dcg = dcg_at_k(retrieved_relevances, k)
    idcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)

    return dcg / idcg if idcg > 0 else 0.0

# Exemplo
# Relevances: 0 (irrelevante), 1 (algo relevante), 2 (relevante), 3 (muito relevante)
retrieved = [3, 0, 2, 1, 2]  # Ordem retrieved
ideal = [3, 2, 2, 1, 0]      # Ordem ideal

ndcg = ndcg_at_k(retrieved, ideal, k=5)
print(f"NDCG@5: {ndcg:.2%}")
```

**Quando usar:** Quando relevância é graduada (não binária)

---

## Métricas de Generation (ragas)

**ragas** = Framework para avaliar RAG systems

**Install:**
```bash
pip install ragas
```

### 1. Faithfulness (Fidelidade)

**Definição:** Resposta é factualmente consistente com o context?

**Mede:** Hallucination

**Range:** 0-1 (1 = perfeitamente faithful)

**Como funciona:**
1. Extrai "statements" da resposta
2. Para cada statement, verifica se está suportado pelo context
3. Faithfulness = statements suportados / total statements

**Implementação:**

```python
from ragas import evaluate
from ragas.metrics import faithfulness
from datasets import Dataset

# Prepare data
data = {
    "question": ["What is the capital of France?"],
    "answer": ["The capital of France is Paris."],
    "contexts": [["France is a country in Europe. Paris is the capital city of France."]],
    "ground_truth": ["Paris"]  # Opcional
}

dataset = Dataset.from_dict(data)

# Evaluate
result = evaluate(dataset, metrics=[faithfulness])

print(f"Faithfulness: {result['faithfulness']:.2%}")
```

**Manual implementation:**

```python
from typing import Literal
import anthropic

async def check_faithfulness(
    answer: str,
    contexts: list[str],
    llm_client: anthropic.AsyncAnthropic
) -> float:
    """Check se answer é faithful aos contexts.

    Uses LLM-as-judge pattern.
    """
    context_text = "\n\n".join(contexts)

    prompt = f"""Given this context and answer, extract all factual statements from the answer.
Then, for EACH statement, verify if it's supported by the context.

Context:
{context_text}

Answer:
{answer}

Return JSON:
{{
    "statements": [
        {{"statement": "...", "supported": true/false}}
    ]
}}"""

    response = await llm_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    import json
    result = json.loads(response.content[0].text)

    statements = result["statements"]
    supported = sum(1 for s in statements if s["supported"])
    total = len(statements)

    return supported / total if total > 0 else 1.0
```

---

### 2. Answer Relevance

**Definição:** Resposta realmente responde a query?

**Mede:** Se resposta está on-topic

**Range:** 0-1

**Como funciona:**
1. Gera queries sintéticas a partir da answer
2. Compara similaridade entre original query e synthetic queries
3. Alta similaridade = answer é relevante para query

**Implementação:**

```python
from ragas.metrics import answer_relevancy

# Evaluate
result = evaluate(
    dataset,
    metrics=[answer_relevancy],
    llm=your_llm,  # Precisa de LLM para gerar synthetic queries
    embeddings=your_embeddings  # Precisa de embeddings para comparar
)

print(f"Answer Relevancy: {result['answer_relevancy']:.2%}")
```

---

### 3. Context Precision

**Definição:** Contextos relevantes aparecem no top?

**Mede:** Qualidade do ranking de retrieval

**Range:** 0-1

**Como funciona:**
1. Verifica se cada context é relevante para ground truth answer
2. Penaliza contexts irrelevantes que aparecem antes de relevantes

**Implementação:**

```python
from ragas.metrics import context_precision

result = evaluate(
    dataset,
    metrics=[context_precision],
    llm=your_llm
)

print(f"Context Precision: {result['context_precision']:.2%}")
```

---

### 4. Context Recall

**Definição:** % do ground truth answer que pode ser atribuído ao context

**Mede:** Se contexto recuperado cobre informação necessária

**Range:** 0-1

**Implementação:**

```python
from ragas.metrics import context_recall

result = evaluate(
    dataset,
    metrics=[context_recall],
    llm=your_llm
)

print(f"Context Recall: {result['context_recall']:.2%}")
```

---

### 5. Answer Correctness

**Definição:** Resposta é factualmente correta comparada ao ground truth?

**Combina:**
- Semantic similarity
- Factual overlap

**Implementação:**

```python
from ragas.metrics import answer_correctness

result = evaluate(
    dataset,
    metrics=[answer_correctness],
    llm=your_llm,
    embeddings=your_embeddings
)

print(f"Answer Correctness: {result['answer_correctness']:.2%}")
```

---

## Evaluation Framework Completo

### 1. Golden Dataset

**Crie dataset de teste com:**
- Queries reais
- Ground truth answers
- Documentos relevantes (manual)

```python
from typing import TypedDict

class EvalExample(TypedDict):
    """Single eval example."""
    query: str
    ground_truth_answer: str
    relevant_doc_ids: list[str]
    domain: str  # Categoria

# Build golden dataset
golden_dataset: list[EvalExample] = [
    {
        "query": "What is RAG?",
        "ground_truth_answer": "RAG stands for Retrieval-Augmented Generation...",
        "relevant_doc_ids": ["doc_123", "doc_456"],
        "domain": "definitions"
    },
    {
        "query": "How to implement chunking?",
        "ground_truth_answer": "Chunking can be implemented using...",
        "relevant_doc_ids": ["doc_789"],
        "domain": "how-to"
    },
    # ... mais exemplos
]
```

**Recomendação:**
- Mínimo: 50 exemplos
- Ideal: 200+ exemplos
- Diversidade de domínios/tipos de query

---

### 2. End-to-End Evaluation

```python
from dataclasses import dataclass
from typing import Protocol

class RAGSystem(Protocol):
    """RAG system interface."""
    async def query(self, query: str) -> tuple[str, list[str]]:
        """Query RAG system.

        Returns:
            (answer, retrieved_doc_ids)
        """
        ...

@dataclass
class EvalMetrics:
    """Metrics result."""
    # Retrieval
    precision_at_5: float
    recall_at_5: float
    mrr: float

    # Generation (ragas)
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: float

    # Overall
    overall_score: float

async def evaluate_rag_system(
    rag_system: RAGSystem,
    golden_dataset: list[EvalExample],
    llm_client: anthropic.AsyncAnthropic
) -> EvalMetrics:
    """Evaluate RAG system end-to-end."""
    # Storage for metrics
    precisions = []
    recalls = []
    rr_scores = []

    faithfulness_scores = []
    # ... outras métricas ragas

    for example in golden_dataset:
        # Run RAG
        answer, retrieved_doc_ids = await rag_system.query(example["query"])

        # Retrieval metrics
        relevant_docs = set(example["relevant_doc_ids"])
        precision = precision_at_k(retrieved_doc_ids, relevant_docs, k=5)
        recall = recall_at_k(retrieved_doc_ids, relevant_docs, k=5)
        rr = reciprocal_rank(retrieved_doc_ids, relevant_docs)

        precisions.append(precision)
        recalls.append(recall)
        rr_scores.append(rr)

        # Generation metrics (ragas)
        # ... (usar ragas para faithfulness, etc.)

    return EvalMetrics(
        precision_at_5=sum(precisions) / len(precisions),
        recall_at_5=sum(recalls) / len(recalls),
        mrr=sum(rr_scores) / len(rr_scores),
        faithfulness=sum(faithfulness_scores) / len(faithfulness_scores),
        # ... outras métricas
        overall_score=...  # Weighted combination
    )

# Run evaluation
metrics = await evaluate_rag_system(rag_system, golden_dataset, llm_client)
print(f"Precision@5: {metrics.precision_at_5:.2%}")
print(f"Faithfulness: {metrics.faithfulness:.2%}")
```

---

### 3. A/B Testing

**Compare diferentes configurações:**

```python
from typing import Literal

async def ab_test_rag_configs(
    config_a: dict,
    config_b: dict,
    golden_dataset: list[EvalExample]
) -> dict:
    """A/B test two RAG configurations."""
    # Build RAG systems
    rag_a = build_rag_system(**config_a)
    rag_b = build_rag_system(**config_b)

    # Evaluate both
    metrics_a = await evaluate_rag_system(rag_a, golden_dataset, llm_client)
    metrics_b = await evaluate_rag_system(rag_b, golden_dataset, llm_client)

    # Compare
    comparison = {
        "config_a": config_a,
        "config_b": config_b,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "winner": "A" if metrics_a.overall_score > metrics_b.overall_score else "B",
        "improvement": abs(metrics_a.overall_score - metrics_b.overall_score)
    }

    return comparison

# Teste: chunk_size 512 vs 1024
result = await ab_test_rag_configs(
    config_a={"chunk_size": 512, "top_k": 3},
    config_b={"chunk_size": 1024, "top_k": 3},
    golden_dataset=golden_dataset
)

print(f"Winner: Config {result['winner']}")
print(f"Improvement: {result['improvement']:.2%}")
```

---

### 4. Regression Testing

**Detecte se mudanças pioram qualidade:**

```python
import json
from datetime import datetime

class RegressionTest:
    """Track metrics over time."""

    def __init__(self, baseline_file: str = "baseline_metrics.json"):
        self.baseline_file = baseline_file

    def save_baseline(self, metrics: EvalMetrics):
        """Save current metrics as baseline."""
        baseline = {
            "date": datetime.now().isoformat(),
            "metrics": vars(metrics)
        }

        with open(self.baseline_file, "w") as f:
            json.dump(baseline, f, indent=2)

    def check_regression(
        self,
        current_metrics: EvalMetrics,
        threshold: float = 0.05  # 5% degradation = regression
    ) -> dict:
        """Check if current metrics regressed vs baseline."""
        with open(self.baseline_file, "r") as f:
            baseline = json.load(f)

        baseline_metrics = EvalMetrics(**baseline["metrics"])

        # Compare each metric
        regressions = {}
        for metric_name in ["precision_at_5", "recall_at_5", "faithfulness", "answer_correctness"]:
            baseline_value = getattr(baseline_metrics, metric_name)
            current_value = getattr(current_metrics, metric_name)

            diff = current_value - baseline_value
            relative_diff = diff / baseline_value if baseline_value > 0 else 0

            if relative_diff < -threshold:  # Degradation > threshold
                regressions[metric_name] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "degradation": relative_diff
                }

        return {
            "has_regression": len(regressions) > 0,
            "regressions": regressions,
            "baseline_date": baseline["date"]
        }

# Usage
regression_test = RegressionTest()

# First run: set baseline
baseline_metrics = await evaluate_rag_system(...)
regression_test.save_baseline(baseline_metrics)

# After changes: check regression
new_metrics = await evaluate_rag_system(...)
result = regression_test.check_regression(new_metrics)

if result["has_regression"]:
    print("⚠️ Regression detected!")
    for metric, details in result["regressions"].items():
        print(f"  {metric}: {details['degradation']:.2%} worse")
```

---

## Human Evaluation

**Apesar de métricas automáticas, human eval é gold standard:**

### Setup Human Eval

```python
from typing import Literal
import random

class HumanEvalTask(TypedDict):
    """Single human eval task."""
    query: str
    answer: str
    contexts: list[str]
    eval_id: str

def create_human_eval_tasks(
    rag_system: RAGSystem,
    queries: list[str],
    sample_size: int = 50
) -> list[HumanEvalTask]:
    """Create tasks para human evaluation."""
    # Sample queries
    sampled = random.sample(queries, min(sample_size, len(queries)))

    tasks = []
    for i, query in enumerate(sampled):
        answer, context_ids = await rag_system.query(query)

        tasks.append({
            "query": query,
            "answer": answer,
            "contexts": context_ids,  # IDs ou textos
            "eval_id": f"eval_{i}"
        })

    return tasks

# Raters avaliam cada task:
# - Faithfulness: 1-5
# - Relevance: 1-5
# - Correctness: 1-5
# - Overall: 1-5
```

**Calcular inter-rater agreement:**

```python
from sklearn.metrics import cohen_kappa_score

def calculate_agreement(
    rater1_scores: list[int],
    rater2_scores: list[int]
) -> float:
    """Calculate Cohen's Kappa (inter-rater agreement)."""
    return cohen_kappa_score(rater1_scores, rater2_scores)

# Agreement > 0.6 = good
# Agreement > 0.8 = excellent
```

---

## Production Monitoring

**Em produção, monitore continuamente:**

```python
import structlog
from datetime import datetime

logger = structlog.get_logger()

class RAGMonitor:
    """Monitor RAG quality em produção."""

    async def log_rag_call(
        self,
        query: str,
        answer: str,
        retrieved_docs: list[str],
        latency_ms: float,
        cost_usd: float
    ):
        """Log RAG call para análise posterior."""
        logger.info(
            "rag_call",
            query=query[:100],  # Truncate
            answer=answer[:100],
            num_docs_retrieved=len(retrieved_docs),
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            timestamp=datetime.now().isoformat()
        )

    async def compute_daily_metrics(self) -> dict:
        """Aggregate metrics do dia."""
        # Query logs
        # Calcular:
        # - Avg latency
        # - Avg cost
        # - Queries per hour
        # - Error rate
        # - (Se tiver feedback) satisfaction rate

        return {
            "avg_latency_ms": ...,
            "avg_cost_usd": ...,
            "total_queries": ...,
            "error_rate": ...,
        }

# Usage
monitor = RAGMonitor()

# Em cada RAG call
await monitor.log_rag_call(
    query=query,
    answer=answer,
    retrieved_docs=docs,
    latency_ms=123.4,
    cost_usd=0.002
)
```

**Dashboards (Grafana, DataDog):**
- Latency P50, P95, P99
- Cost por query
- Error rate
- Retrieval quality (se tiver ground truth sample)
- User satisfaction (thumbs up/down)

---

## Best Practices

### 1. Start Simple

```python
# ✅ Start com métricas simples
metrics = {
    "precision_at_5": ...,
    "faithfulness": ...,
}

# ❌ Não comece com 20 métricas
```

### 2. Golden Dataset é Crucial

```python
# ✅ Invest tempo em golden dataset de qualidade
# - 200+ exemplos
# - Diversidade de queries
# - Ground truth de alta qualidade

# ❌ Não use dataset sintético gerado por LLM
```

### 3. Track Metrics Over Time

```python
# ✅ Version metrics junto com código
# ✅ Regression tests em CI/CD
# ✅ Dashboard de metrics

# ❌ Não avalie uma vez e esqueça
```

### 4. Combine Auto + Human Eval

```python
# ✅ Auto eval (ragas) para iteração rápida
# ✅ Human eval sample para validação

# ❌ Não confie 100% em métricas automáticas
```

---

## References

- [ragas Documentation](https://docs.ragas.io/)
- [RAG Evaluation Guide (LangChain)](https://python.langchain.com/docs/guides/evaluation)
- [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard)
- [RAG Architecture](rag/architecture.md)
- [Chunking Strategies](rag/chunking-strategies.md)
- [Embeddings](rag/embeddings.md)
