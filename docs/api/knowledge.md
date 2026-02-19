# sktk.knowledge

RAG pipeline: chunking, indexing, retrieval, and grounding.

## Table of Contents

- [KnowledgeBase](#knowledgebase)
- [TextSource](#textsource)
- [Protocols](#protocols)
  - [Embedder](#embedder)
  - [Source](#source)
  - [Queryable](#queryable)
  - [Reranker](#reranker)
- [Chunking](#chunking)
  - [Chunk](#chunk)
  - [Chunker (type alias)](#chunker)
  - [fixed_size_chunker](#fixed_size_chunker)
  - [sentence_chunker](#sentence_chunker)
- [Retrieval](#retrieval)
  - [RetrievalMode](#retrievalmode)
  - [RetrievalConfig](#retrievalconfig)
  - [ScoredChunk](#scoredchunk)
  - [BM25Index](#bm25index)
  - [reciprocal_rank_fusion](#reciprocal_rank_fusion)
- [Grounding](#grounding)
  - [GroundingConfig](#groundingconfig)
  - [GroundingFilter](#groundingfilter)
- [Backends](#backends)
  - [InMemoryKnowledgeBackend](#inmemorknowledgebackend)

---

## KnowledgeBase

```python
class KnowledgeBase
```

Orchestrates source ingestion, chunking, indexing, and retrieval.

### Constructor

```python
KnowledgeBase(
    sources: list[Any],
    embedder: Embedder,
    chunker: Chunker,
    retrieval: RetrievalConfig,
    backend: InMemoryKnowledgeBackend | None = None,
)
```

| Param | Type | Description |
|---|---|---|
| `sources` | `list[Any]` | Raw strings, `TextSource` instances, or any object implementing the `Source` protocol. |
| `embedder` | `Embedder` | An object satisfying the `Embedder` protocol (provides `embed` and `embed_query`). |
| `chunker` | `Chunker` | A callable `(str, str) -> list[Chunk]` -- use one of the chunker factories below. |
| `retrieval` | `RetrievalConfig` | Retrieval pipeline configuration (mode, top_k, reranker, weights). |
| `backend` | `InMemoryKnowledgeBackend \| None` | Vector store backend. Defaults to `InMemoryKnowledgeBackend()`. |

### Methods

#### build

```python
async def build() -> None
```

Ingest all sources, chunk them, compute embeddings, and build indices. Must be called before `query`.

#### query

```python
async def query(query: str) -> list[ScoredChunk]
```

Retrieve the most relevant chunks for a query using the configured retrieval mode.

| Param | Type | Description |
|---|---|---|
| `query` | `str` | The user query to search against. |

**Returns:** `list[ScoredChunk]` -- ranked results, highest relevance first.

#### chunk_count

```python
async def chunk_count() -> int
```

Return the total number of indexed chunks.

**Returns:** `int`

---

## TextSource

```python
@dataclass
class TextSource
```

Inline text source that wraps a plain string.

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `content` | `str` | *(required)* | The raw text content. |
| `name` | `str` | `"inline"` | Source identifier for provenance tracking. |

### Methods

#### load

```python
async def load() -> str
```

Return `self.content`. Satisfies the `Source` protocol.

**Returns:** `str`

---

## Protocols

### Embedder

```python
@runtime_checkable
class Embedder(Protocol)
```

Protocol for embedding providers.

#### embed

```python
async def embed(texts: list[str]) -> list[list[float]]
```

Embed a batch of texts into vector representations.

#### embed_query

```python
async def embed_query(text: str) -> list[float]
```

Embed a single query string.

---

### Source

```python
@runtime_checkable
class Source(Protocol)
```

Protocol for loadable document sources.

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Human-readable source identifier. |

#### load

```python
async def load() -> str
```

Load and return the full text content.

---

### Queryable

```python
@runtime_checkable
class Queryable(Protocol)
```

Protocol for anything that can be queried for context (used by `GroundingFilter`).

#### query

```python
async def query(query: str) -> list[Any]
```

Return a list of results relevant to the query.

---

### Reranker

```python
@runtime_checkable
class Reranker(Protocol)
```

Protocol for reranking retrieved chunks.

#### rerank

```python
async def rerank(query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]
```

Reorder and rescore the given chunks with respect to the query.

---

## Chunking

### Chunk

```python
@dataclass(frozen=True)
class Chunk
```

A chunk of text with source provenance.

#### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | *(required)* | The chunk text content. |
| `source` | `str` | *(required)* | Name of the source this chunk originates from. |
| `index` | `int` | *(required)* | Zero-based position of this chunk within its source. |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata. |

---

### Chunker

```python
Chunker = Callable[[str, str], list[Chunk]]
```

Type alias for chunker callables. A chunker takes `(text, source_name)` and returns a list of `Chunk` objects.

`ChunkingStrategy` is a backward-compatible alias for `Chunker`.

---

### fixed_size_chunker

```python
def fixed_size_chunker(max_words: int, overlap_words: int = 0) -> Chunker
```

Create a chunker that splits text into fixed-size word windows.

| Param | Type | Default | Description |
|---|---|---|---|
| `max_words` | `int` | *(required)* | Maximum number of words per chunk. |
| `overlap_words` | `int` | `0` | Number of overlapping words between consecutive chunks. Must be less than `max_words`. |

**Returns:** `Chunker` -- a callable `(text, source) -> list[Chunk]`.

**Raises:** `ValueError` if `overlap_words >= max_words`.

---

### sentence_chunker

```python
def sentence_chunker(max_sentences: int) -> Chunker
```

Create a chunker that groups consecutive sentences together.

| Param | Type | Default | Description |
|---|---|---|---|
| `max_sentences` | `int` | *(required)* | Maximum number of sentences per chunk. |

**Returns:** `Chunker` -- a callable `(text, source) -> list[Chunk]`.

---

## Retrieval

### RetrievalMode

```python
class RetrievalMode(Enum)
```

Retrieval strategy selector.

| Value | Description |
|---|---|
| `DENSE` | Vector similarity search only. |
| `SPARSE` | BM25 keyword search only. |
| `HYBRID` | Combines dense and sparse via reciprocal rank fusion. |

---

### RetrievalConfig

```python
@dataclass
class RetrievalConfig
```

Configuration for the retrieval pipeline.

#### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `RetrievalMode` | `RetrievalMode.DENSE` | Which retrieval strategy to use. |
| `top_k` | `int` | `5` | Number of results to return. |
| `reranker` | `Reranker \| None` | `None` | Optional reranker applied after initial retrieval. |
| `sparse_weight` | `float` | `0.3` | Weight for sparse results (used in hybrid mode). |
| `dense_weight` | `float` | `0.7` | Weight for dense results (used in hybrid mode). |

---

### ScoredChunk

```python
@dataclass
class ScoredChunk
```

A chunk with a relevance score.

#### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `chunk` | `Chunk` | *(required)* | The underlying chunk. |
| `score` | `float` | *(required)* | Relevance score (higher is more relevant). |
| `retrieval_method` | `str` | `"dense"` | Which retrieval method produced this result (`"dense"`, `"sparse"`, or `"hybrid"`). |

---

### BM25Index

```python
class BM25Index
```

Pure Python BM25 implementation for sparse retrieval.

#### Constructor

```python
BM25Index(k1: float = 1.5, b: float = 0.75)
```

| Param | Type | Default | Description |
|---|---|---|---|
| `k1` | `float` | `1.5` | Term frequency saturation parameter. |
| `b` | `float` | `0.75` | Document length normalization parameter (0 = no normalization, 1 = full normalization). |

#### Methods

##### index

```python
def index(chunks: list[Chunk]) -> None
```

Build the BM25 index from chunks.

| Param | Type | Description |
|---|---|---|
| `chunks` | `list[Chunk]` | Chunks to index. |

##### search

```python
def search(query: str, top_k: int = 5) -> list[ScoredChunk]
```

Search the index and return top-k scored chunks.

| Param | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | *(required)* | Search query string. |
| `top_k` | `int` | `5` | Maximum number of results to return. |

**Returns:** `list[ScoredChunk]` -- results with `retrieval_method="sparse"`, filtered to positive scores only.

---

### reciprocal_rank_fusion

```python
def reciprocal_rank_fusion(
    result_lists: list[list[ScoredChunk]],
    k: int = 60,
    top_k: int = 5,
) -> list[ScoredChunk]
```

Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

Each chunk receives an RRF score of `1 / (k + rank + 1)` from each list it appears in, summed across all lists.

| Param | Type | Default | Description |
|---|---|---|---|
| `result_lists` | `list[list[ScoredChunk]]` | *(required)* | Two or more ranked result lists to merge. |
| `k` | `int` | `60` | RRF smoothing constant. |
| `top_k` | `int` | `5` | Maximum number of merged results to return. |

**Returns:** `list[ScoredChunk]` -- merged results with `retrieval_method="hybrid"`.

---

## Grounding

### GroundingConfig

```python
@dataclass
class GroundingConfig
```

Configuration for automatic grounding.

#### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `max_results` | `int` | `3` | Maximum number of retrieved results to inject. |
| `min_score` | `float` | `0.0` | Minimum relevance score threshold. Results below this are discarded. |
| `context_prefix` | `str` | `"\n\n[Relevant context from knowledge base]:\n"` | String prepended to the injected context block. |
| `context_suffix` | `str` | `"\n[End of context]\n\n"` | String appended to the injected context block. |

---

### GroundingFilter

```python
class GroundingFilter
```

Filter that auto-grounds prompts with knowledge base context. Works with any `Queryable` source (KnowledgeBase, custom retriever). Results can be `ScoredChunk` objects or dicts with `"text"`/`"content"` keys.

#### Constructor

```python
GroundingFilter(
    source: Queryable,
    config: GroundingConfig | None = None,
)
```

| Param | Type | Default | Description |
|---|---|---|---|
| `source` | `Queryable` | *(required)* | The queryable knowledge source to retrieve context from. |
| `config` | `GroundingConfig \| None` | `None` | Grounding settings. Defaults to `GroundingConfig()`. |

#### Methods

##### on_input

```python
async def on_input(context: FilterContext) -> FilterResult
```

Query the source and inject retrieved context into the prompt. Returns `Modify` with prepended context, or `Allow` if no relevant results are found.

| Param | Type | Description |
|---|---|---|
| `context` | `FilterContext` | The incoming filter context containing the user input. |

**Returns:** `FilterResult` (`Allow` or `Modify`)

##### on_output

```python
async def on_output(context: FilterContext) -> FilterResult
```

Pass-through (returns `Allow`).

##### on_function_call

```python
async def on_function_call(context: FilterContext) -> FilterResult
```

Pass-through (returns `Allow`).

---

## Backends

### InMemoryKnowledgeBackend

```python
class InMemoryKnowledgeBackend
```

Simple in-memory vector store using cosine similarity.

#### Constructor

```python
InMemoryKnowledgeBackend()
```

No parameters.

#### Methods

##### store

```python
async def store(chunks: list[Chunk], embeddings: list[list[float]]) -> None
```

Store chunks and their corresponding embeddings.

| Param | Type | Description |
|---|---|---|
| `chunks` | `list[Chunk]` | Chunks to store. |
| `embeddings` | `list[list[float]]` | Embedding vectors, one per chunk. |

##### search

```python
async def search(query_embedding: list[float], top_k: int = 5) -> list[ScoredChunk]
```

Return the top-k most similar chunks by cosine similarity.

| Param | Type | Default | Description |
|---|---|---|---|
| `query_embedding` | `list[float]` | *(required)* | The query vector. |
| `top_k` | `int` | `5` | Maximum number of results. |

**Returns:** `list[ScoredChunk]` -- results with `retrieval_method="dense"`.

##### count

```python
async def count() -> int
```

Return the number of stored chunks.

**Returns:** `int`

##### clear

```python
async def clear() -> None
```

Remove all stored chunks and embeddings.
