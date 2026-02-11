# Chunking Strategies - Como Quebrar Documentos para RAG

Referência completa de estratégias de chunking para RAG systems.

---

## Por que Chunking é Crítico?

**Problema:** Documentos são grandes, mas:
- Vector DBs armazenam chunks individuais
- LLMs têm context window limitado
- Retrieval precisa encontrar pedaços relevantes

**Solução:** Quebrar documentos em chunks menores, semanticamente coerentes.

**Trade-offs:**
- Chunks pequenos → mais precisão, mas perde contexto
- Chunks grandes → mais contexto, mas menos precisão na busca
- Chunk ideal → **512-1024 tokens** (sweet spot para maioria dos casos)

---

## Estratégias de Chunking

### 1. Fixed-Size Chunking (Mais Simples)

**Como funciona:**
- Quebra texto em pedaços de tamanho fixo
- Overlap opcional entre chunks (recomendado)

**Prós:**
- ✅ Simples de implementar
- ✅ Rápido
- ✅ Previsível

**Contras:**
- ❌ Pode quebrar no meio de frases/parágrafos
- ❌ Perde contexto semântico
- ❌ Não respeita estrutura do documento

**Quando usar:**
- Protótipos rápidos
- Texto sem estrutura clara
- Quando simplicidade > qualidade

**Implementação:**

```python
from typing import Iterator

def fixed_size_chunking(
    text: str,
    chunk_size: int = 512,
    overlap: int = 128
) -> list[str]:
    """Chunk text em pedaços de tamanho fixo com overlap.

    Args:
        text: Texto para chunking
        chunk_size: Tamanho de cada chunk (em tokens aproximados)
        overlap: Quantidade de overlap entre chunks

    Returns:
        Lista de chunks
    """
    # Aproximação: 1 token ≈ 4 caracteres
    chunk_chars = chunk_size * 4
    overlap_chars = overlap * 4

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_chars
        chunk = text[start:end]

        # Evita chunks muito pequenos no final
        if len(chunk) > chunk_chars * 0.1:
            chunks.append(chunk)

        start = end - overlap_chars  # Overlap

    return chunks

# Exemplo
text = "Long document text here..."
chunks = fixed_size_chunking(text, chunk_size=512, overlap=128)
print(f"Created {len(chunks)} chunks")
```

**Com tiktoken (contagem precisa de tokens):**

```python
import tiktoken

def fixed_size_chunking_tokens(
    text: str,
    chunk_size: int = 512,
    overlap: int = 128,
    encoding_name: str = "cl100k_base"  # GPT-4, GPT-3.5
) -> list[str]:
    """Chunk text com contagem EXATA de tokens."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        start = end - overlap

    return chunks
```

---

### 2. Sentence-Based Chunking

**Como funciona:**
- Quebra em sentenças completas
- Agrupa sentenças até atingir tamanho target
- Nunca quebra no meio de sentença

**Prós:**
- ✅ Preserva semântica (sentenças completas)
- ✅ Mais natural que fixed-size
- ✅ Funciona bem para texto narrativo

**Contras:**
- ❌ Chunks podem ter tamanhos muito variados
- ❌ Sentença longa pode ser chunk sozinha
- ❌ Precisa de sentence splitter

**Quando usar:**
- Artigos, documentação, narrativas
- Quando preservar semântica > uniformidade
- Texto bem escrito (sentenças claras)

**Implementação:**

```python
import re
from typing import List

def split_sentences(text: str) -> list[str]:
    """Split text em sentenças.

    Usa regex simples. Para produção, use spaCy ou nltk.
    """
    # Regex básico para sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def sentence_based_chunking(
    text: str,
    target_chunk_size: int = 512,
    encoding_name: str = "cl100k_base"
) -> list[str]:
    """Chunk agrupando sentenças até target size."""
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    sentences = split_sentences(text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence))

        # Se adicionar esta sentença ultrapassa target
        if current_tokens + sentence_tokens > target_chunk_size and current_chunk:
            # Salva chunk atual
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            # Adiciona sentença ao chunk atual
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    # Adiciona último chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Exemplo
text = """First sentence here. Second sentence follows. Third one too.
Fourth sentence starts a new line. Fifth concludes."""

chunks = sentence_based_chunking(text, target_chunk_size=512)
```

**Com spaCy (produção):**

```python
import spacy
from spacy.lang.en import English

def sentence_based_chunking_spacy(
    text: str,
    target_chunk_size: int = 512
) -> list[str]:
    """Sentence chunking usando spaCy (mais robusto)."""
    # Load modelo
    nlp = English()
    nlp.add_pipe("sentencizer")

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    # Mesmo algoritmo de agrupamento
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence))

        if current_tokens + sentence_tokens > target_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

---

### 3. Recursive Chunking (Hierárquico)

**Como funciona:**
- Tenta quebrar por separadores em ordem (parágrafo → sentença → palavra)
- Se chunk ainda grande, quebra recursivamente
- Mantém hierarquia semântica

**Prós:**
- ✅ Respeita estrutura do documento
- ✅ Preserva hierarquia (parágrafo > sentença > palavra)
- ✅ Flexível para diferentes tipos de texto

**Contras:**
- ❌ Mais complexo
- ❌ Pode ser mais lento
- ❌ Precisa definir separadores corretos

**Quando usar:**
- Documentação técnica (markdown, code)
- Texto com estrutura hierárquica clara
- Quando qualidade > simplicidade

**Implementação:**

```python
from typing import Literal

def recursive_chunking(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    separators: list[str] | None = None,
    encoding_name: str = "cl100k_base"
) -> list[str]:
    """Recursive text splitting mantendo hierarquia.

    Args:
        text: Texto para chunking
        chunk_size: Tamanho target (tokens)
        chunk_overlap: Overlap entre chunks (tokens)
        separators: Lista de separadores por ordem de preferência

    Returns:
        Lista de chunks
    """
    import tiktoken

    if separators is None:
        # Ordem de preferência: parágrafo → linha → sentença → espaço
        separators = ["\n\n", "\n", ". ", " ", ""]

    encoding = tiktoken.get_encoding(encoding_name)

    def split_text(text: str, separator: str) -> list[str]:
        """Split text by separator."""
        if separator == "":
            return list(text)  # Character-level
        return text.split(separator)

    def recursive_split(text: str, separators: list[str]) -> list[str]:
        """Recursively split text."""
        # Base case: text fits in chunk_size
        if len(encoding.encode(text)) <= chunk_size:
            return [text]

        # Try splitting with current separator
        separator = separators[0]
        splits = split_text(text, separator)

        # Group splits into chunks
        chunks = []
        current_chunk = []
        current_tokens = 0

        for split in splits:
            split_tokens = len(encoding.encode(split))

            # If single split is too large, use next separator
            if split_tokens > chunk_size:
                if len(separators) > 1:
                    # Recursively split this piece
                    sub_chunks = recursive_split(split, separators[1:])
                    chunks.extend(sub_chunks)
                else:
                    # No more separators, force include
                    chunks.append(split)
                continue

            # Check if adding this split exceeds chunk_size
            if current_tokens + split_tokens > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = separator.join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                overlap_tokens = 0
                overlap_splits = []
                for s in reversed(current_chunk):
                    s_tokens = len(encoding.encode(s))
                    if overlap_tokens + s_tokens <= chunk_overlap:
                        overlap_splits.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break

                current_chunk = overlap_splits + [split]
                current_tokens = sum(len(encoding.encode(s)) for s in current_chunk)
            else:
                current_chunk.append(split)
                current_tokens += split_tokens

        # Add last chunk
        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    return recursive_split(text, separators)

# Exemplo
markdown_text = """# Title

## Section 1

Paragraph 1 here. With multiple sentences.

Paragraph 2 follows.

## Section 2

More content here."""

chunks = recursive_chunking(markdown_text, chunk_size=512, chunk_overlap=128)
```

**LangChain RecursiveCharacterTextSplitter:**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def recursive_chunking_langchain(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128
) -> list[str]:
    """Recursive chunking usando LangChain (produção ready)."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,  # Aproximação caracteres
        chunk_overlap=chunk_overlap * 4,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return text_splitter.split_text(text)
```

---

### 4. Semantic Chunking (Mais Avançado)

**Como funciona:**
- Usa embeddings para detectar mudanças semânticas
- Quebra onde similaridade entre sentenças cai
- Mantém chunks semanticamente coerentes

**Prós:**
- ✅ Chunks semanticamente coerentes
- ✅ Respeita mudanças de tópico
- ✅ Melhor qualidade de retrieval

**Contras:**
- ❌ Mais lento (precisa embeddings)
- ❌ Mais complexo
- ❌ Requer modelo de embeddings

**Quando usar:**
- Produção com foco em qualidade
- Documentos com múltiplos tópicos
- Quando latência < qualidade

**Implementação:**

```python
import numpy as np
from typing import Callable

async def semantic_chunking(
    text: str,
    embed_fn: Callable[[str], list[float]],
    similarity_threshold: float = 0.5,
    min_chunk_size: int = 200,
    max_chunk_size: int = 1000
) -> list[str]:
    """Chunk baseado em similaridade semântica.

    Args:
        text: Texto para chunking
        embed_fn: Função async que gera embeddings
        similarity_threshold: Threshold de similaridade (0-1)
        min_chunk_size: Tamanho mínimo de chunk (caracteres)
        max_chunk_size: Tamanho máximo de chunk (caracteres)

    Returns:
        Lista de chunks semanticamente coerentes
    """
    # Split em sentenças
    sentences = split_sentences(text)

    # Gera embeddings para todas sentenças
    embeddings = [await embed_fn(sent) for sent in sentences]

    # Calcula similaridade entre sentenças adjacentes
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    # Identifica breakpoints (onde similaridade cai)
    breakpoints = [0]  # Sempre começa no 0

    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            breakpoints.append(i + 1)

    breakpoints.append(len(sentences))  # Sempre termina no final

    # Cria chunks baseado nos breakpoints
    chunks = []
    for i in range(len(breakpoints) - 1):
        start = breakpoints[i]
        end = breakpoints[i + 1]
        chunk_sentences = sentences[start:end]
        chunk_text = " ".join(chunk_sentences)

        # Força chunks dentro de limites de tamanho
        if len(chunk_text) < min_chunk_size and chunks:
            # Merge com chunk anterior se muito pequeno
            chunks[-1] += " " + chunk_text
        elif len(chunk_text) > max_chunk_size:
            # Split chunk grande recursivamente
            sub_chunks = recursive_chunking(chunk_text, chunk_size=max_chunk_size // 4)
            chunks.extend(sub_chunks)
        else:
            chunks.append(chunk_text)

    return chunks

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity entre dois vetores."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Exemplo de uso
async def embed_text(text: str) -> list[float]:
    """Função exemplo de embedding."""
    # Usar OpenAI, Cohere, etc.
    # return await openai_client.embeddings.create(...)
    pass

chunks = await semantic_chunking(
    text="Long document...",
    embed_fn=embed_text,
    similarity_threshold=0.5
)
```

**LangChain SemanticChunker:**

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

def semantic_chunking_langchain(text: str) -> list[str]:
    """Semantic chunking usando LangChain."""
    embeddings = OpenAIEmbeddings()

    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",  # ou "standard_deviation", "interquartile"
        breakpoint_threshold_amount=50  # percentile 50
    )

    return text_splitter.split_text(text)
```

---

### 5. Document-Aware Chunking (Para Formatos Específicos)

**Como funciona:**
- Respeita estrutura do documento (headers, sections, code blocks)
- Usa parsers específicos por formato
- Mantém metadata (título da seção, etc.)

**Formatos suportados:**
- Markdown (headers, code blocks)
- HTML (tags, structure)
- PDF (pages, sections)
- Code (functions, classes)

**Quando usar:**
- Documentação técnica (markdown)
- Codebases (Python, JavaScript)
- PDFs estruturados
- HTML/web content

#### 5.1 Markdown Chunking

```python
from typing import TypedDict

class MarkdownChunk(TypedDict):
    """Chunk com metadata."""
    text: str
    headers: list[str]  # Header hierarchy
    level: int
    type: Literal["text", "code"]

def markdown_chunking(
    markdown_text: str,
    max_chunk_size: int = 512
) -> list[MarkdownChunk]:
    """Chunk markdown respeitando estrutura.

    Mantém:
    - Headers hierarchy
    - Code blocks intactos
    - Metadata de seção
    """
    import re

    chunks: list[MarkdownChunk] = []
    current_headers: list[str] = []
    current_level = 0

    lines = markdown_text.split('\n')
    current_chunk = []
    current_tokens = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect header
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2)

            # Save previous chunk if exists
            if current_chunk:
                chunks.append({
                    "text": '\n'.join(current_chunk),
                    "headers": current_headers.copy(),
                    "level": current_level,
                    "type": "text"
                })
                current_chunk = []
                current_tokens = 0

            # Update header hierarchy
            current_headers = current_headers[:level-1] + [title]
            current_level = level

            current_chunk.append(line)
            i += 1
            continue

        # Detect code block
        if line.strip().startswith('```'):
            # Find end of code block
            code_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                code_lines.append(lines[i])  # Closing ```

            # Save code block as separate chunk
            chunks.append({
                "text": '\n'.join(code_lines),
                "headers": current_headers.copy(),
                "level": current_level,
                "type": "code"
            })
            i += 1
            continue

        # Regular line
        current_chunk.append(line)
        current_tokens += len(line.split())

        # Check if chunk is getting too large
        if current_tokens > max_chunk_size:
            chunks.append({
                "text": '\n'.join(current_chunk),
                "headers": current_headers.copy(),
                "level": current_level,
                "type": "text"
            })
            current_chunk = []
            current_tokens = 0

        i += 1

    # Save last chunk
    if current_chunk:
        chunks.append({
            "text": '\n'.join(current_chunk),
            "headers": current_headers.copy(),
            "level": current_level,
            "type": "text"
        })

    return chunks

# Exemplo
markdown = """# Main Title

## Section 1

Some text here.

```python
def example():
    pass
```

## Section 2

More text."""

chunks = markdown_chunking(markdown)
# chunks[0] = {"text": "# Main Title", "headers": ["Main Title"], "level": 1, "type": "text"}
# chunks[1] = {"text": "## Section 1\n\nSome text here.", "headers": ["Main Title", "Section 1"], ...}
# chunks[2] = {"text": "```python...", "headers": ["Main Title", "Section 1"], "type": "code"}
```

**LangChain MarkdownHeaderTextSplitter:**

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

def markdown_chunking_langchain(markdown_text: str) -> list[dict]:
    """Markdown chunking usando LangChain."""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    return markdown_splitter.split_text(markdown_text)
```

#### 5.2 Code Chunking

```python
import ast

def python_code_chunking(code: str) -> list[dict]:
    """Chunk Python code por functions/classes.

    Mantém:
    - Function/class completa em um chunk
    - Docstrings
    - Imports no contexto
    """
    tree = ast.parse(code)

    chunks = []
    imports = []

    # Extract imports
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))

    # Extract functions and classes
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            chunk_code = ast.unparse(node)

            # Add imports context
            full_chunk = '\n'.join(imports) + '\n\n' + chunk_code

            chunks.append({
                "text": full_chunk,
                "type": "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class",
                "name": node.name,
                "line": node.lineno
            })

    return chunks
```

---

## Decisão: Qual Estratégia Usar?

```
Caso de Uso → Estratégia Recomendada

Prototipagem rápida
  → Fixed-Size (simples, rápido)

Documentação técnica (markdown)
  → Recursive + Markdown-Aware (respeita estrutura)

Artigos/narrativas
  → Sentence-Based (preserva semântica)

Produção, alta qualidade
  → Semantic Chunking (melhor retrieval)

Codebase
  → Code-Aware (por function/class)

Texto sem estrutura clara
  → Fixed-Size ou Recursive

Múltiplos tópicos no doc
  → Semantic Chunking (detecta mudanças)
```

---

## Best Practices

### 1. Sempre Use Overlap

```python
# ❌ Sem overlap
chunks = fixed_size_chunking(text, chunk_size=512, overlap=0)

# ✅ Com overlap (128 tokens = 25%)
chunks = fixed_size_chunking(text, chunk_size=512, overlap=128)
```

**Por quê:** Overlap garante que contexto importante não se perca na fronteira entre chunks.

### 2. Adicione Metadata aos Chunks

```python
class ChunkWithMetadata(TypedDict):
    """Chunk enriquecido com metadata."""
    text: str
    doc_id: str
    chunk_index: int
    headers: list[str]  # Hierarquia de headers
    source: str  # URL, filepath, etc.
    created_at: str

def create_chunks_with_metadata(
    text: str,
    doc_id: str,
    source: str
) -> list[ChunkWithMetadata]:
    """Cria chunks com metadata completo."""
    chunks = recursive_chunking(text)

    return [
        {
            "text": chunk,
            "doc_id": doc_id,
            "chunk_index": i,
            "headers": [],  # Extrair se markdown
            "source": source,
            "created_at": datetime.now().isoformat()
        }
        for i, chunk in enumerate(chunks)
    ]
```

**Por quê:** Metadata ajuda em filtering, debugging, e citation.

### 3. Teste Diferentes Chunk Sizes

```python
async def evaluate_chunk_sizes(
    text: str,
    test_queries: list[str],
    chunk_sizes: list[int] = [256, 512, 1024, 2048]
) -> dict[int, float]:
    """Testa qual chunk size tem melhor retrieval."""
    results = {}

    for chunk_size in chunk_sizes:
        chunks = recursive_chunking(text, chunk_size=chunk_size)

        # Embed chunks e test queries
        # Run retrieval evaluation
        # Calculate metrics (precision, recall)

        score = await evaluate_retrieval(chunks, test_queries)
        results[chunk_size] = score

    return results
```

**Recomendação geral:**
- Documentação: 512-1024 tokens
- Code: Por function/class (variável)
- Narrativas: 1024-2048 tokens
- FAQ: Menor (256-512 tokens)

### 4. Preserve Context com Parent/Child Chunks

```python
class ParentChildChunks(TypedDict):
    """Chunks hierárquicos."""
    parent_id: str
    parent_text: str  # Chunk maior (contexto)
    child_chunks: list[str]  # Chunks menores (retrieval)

def parent_child_chunking(text: str) -> ParentChildChunks:
    """Cria parent chunks (contexto) e child chunks (retrieval).

    Strategy:
    - Parents: 2048 tokens (contexto amplo)
    - Children: 512 tokens (retrieval preciso)
    - Children são sub-chunks dos parents
    """
    # Parents
    parents = recursive_chunking(text, chunk_size=2048, chunk_overlap=256)

    all_chunks = []

    for parent_id, parent_text in enumerate(parents):
        # Children deste parent
        children = recursive_chunking(parent_text, chunk_size=512, chunk_overlap=128)

        all_chunks.append({
            "parent_id": f"parent_{parent_id}",
            "parent_text": parent_text,
            "child_chunks": children
        })

    return all_chunks

# Retrieval strategy:
# 1. Search nos children (precisão)
# 2. Retorna parent correspondente (contexto)
# 3. LLM vê parent completo (mais contexto)
```

---

## Tools e Libraries

### LangChain Text Splitters

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    Language
)

# Recursive (recomendado geral)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512 * 4,
    chunk_overlap=128 * 4
)

# Markdown-aware
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2")]
)

# Code-aware (Python)
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=512 * 4
)
```

### LlamaIndex Node Parser

```python
from llama_index.core.node_parser import (
    SimpleNodeParser,
    SentenceSplitter,
    SemanticSplitterNodeParser
)

# Sentence-based
parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=128
)

# Semantic
semantic_parser = SemanticSplitterNodeParser(
    embed_model=embed_model,
    breakpoint_percentile_threshold=95
)
```

---

## References

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [LlamaIndex Node Parser](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)
- [Chunking Strategies Guide (Pinecone)](https://www.pinecone.io/learn/chunking-strategies/)
- [RAG Architecture](rag/architecture.md)
- [Embeddings Guide](rag/embeddings.md)
