# RAG Search Engine

Small movie-search project that combines:

- BM25 keyword retrieval
- Sentence-transformer semantic retrieval
- Hybrid fusion over both ranking strategies
- Optional Gemma-based query enhancement for hybrid search

The codebase is organized as standalone Python CLIs rather than a packaged library. The current primary workflow is `cli/hybrid_search_cli.py`.

## Current Capabilities

- Build and reuse a cached BM25 index from `data/movies.json`
- Build and reuse full-document semantic embeddings
- Build and reuse chunked semantic embeddings
- Run BM25-only search
- Run semantic-only search
- Run hybrid search with either weighted score fusion or Reciprocal Rank Fusion (RRF)
- Optionally fix spelling or rewrite vague queries before hybrid search

## Repository Layout

```text
.
├── cli/
│   ├── hybrid_search.py
│   ├── hybrid_search_cli.py
│   ├── keyword_search_cli.py
│   ├── semantic_search_cli.py
│   ├── Inverted_Index.py
│   └── lib/
│       ├── semantic_search.py
│       └── chunked_semantic_search.py
├── data/
│   ├── movies.json
│   └── stopwords.txt
├── cache/
├── pyproject.toml
└── README.md
```

Legacy files such as `cli/Old_keyword_search_cli.py` and `cli/Old_Inverted_Index.py` are kept in the repo but are not the current entry points.

## Requirements

- Python `>=3.13`
- `uv` recommended for dependency management

Dependencies are declared in [pyproject.toml](/home/krish/Krish/RAG/rag-search-engine/pyproject.toml).

## Setup

Install dependencies:

```bash
uv sync
```

Optional environment variables:

- `GEMINI_API_KEY`: enables query enhancement in hybrid RRF search
- `GEMMA_MODEL`: overrides the Gemma model used for query enhancement in `cli/hybrid_search_cli.py`
- `GEMINI_MODEL`: used by `test_gemini.py`

## Quick Start

Build the BM25 cache:

```bash
uv run python cli/keyword_search_cli.py build
```

Run BM25 search:

```bash
uv run python cli/keyword_search_cli.py bm25search "dream within a dream" 5
```

Run chunked semantic search:

```bash
uv run python cli/semantic_search_cli.py search_chunked "dream within a dream" --limit 5
```

Run hybrid RRF search:

```bash
uv run python cli/hybrid_search_cli.py rrf-search "dream within a dream" --limit 5
```

Run hybrid weighted search:

```bash
uv run python cli/hybrid_search_cli.py weighted-search "dream within a dream" --alpha 0.5 --limit 5
```

Run hybrid RRF search with optional spell correction:

```bash
uv run python cli/hybrid_search_cli.py rrf-search "interstller space travel" --enhance spell --limit 5
```

Run hybrid RRF search with optional query rewrite:

```bash
uv run python cli/hybrid_search_cli.py rrf-search "that bear movie where leo gets attacked" --enhance rewrite --limit 5
```

## CLI Reference

### `cli/hybrid_search_cli.py`

This is the main CLI for combined retrieval.

Commands:

- `normalize <scores...>`
  - Applies min-max normalization to a list of scores.
- `weighted-search <query> [--alpha FLOAT] [--limit INT]`
  - Combines normalized BM25 and semantic scores.
  - `alpha=1.0` means keyword-heavy, `alpha=0.0` means semantic-heavy.
- `rrf-search <query> [-k INT] [--limit INT] [--enhance {spell,rewrite}]`
  - Combines BM25 and semantic rankings using Reciprocal Rank Fusion.
  - `--enhance spell` attempts typo correction when `GEMINI_API_KEY` is available.
  - `--enhance rewrite` asks Gemma to rewrite vague queries into search-friendly movie phrases.

Example:

```bash
uv run python cli/hybrid_search_cli.py rrf-search "briish bear" --enhance spell
```

```bash
uv run python cli/hybrid_search_cli.py rrf-search "movie about bear in london with marmalade" --enhance rewrite
```

### `cli/keyword_search_cli.py`

Commands:

- `build`
  - Builds and saves the BM25/index cache.
- `tf <docid> <term>`
  - Prints raw term frequency for a normalized term in a document.
- `idf <term>`
  - Prints inverse document frequency.
- `tfidf <docid> <term>`
  - Prints TF-IDF for a term/document pair.
- `bm25idf <term>`
  - Calls the BM25 IDF calculation.
- `bm25tf <doc_id> <term> [k1] [b]`
  - Prints the BM25 term-frequency component.
- `bm25search <query> [limit]`
  - Runs BM25 retrieval and prints ranked results.

Notes:

- `search <query>` exists but is not implemented beyond loading the index.
- The keyword pipeline indexes `title + description`.

### `cli/semantic_search_cli.py`

Commands:

- `verify`
  - Loads the sentence-transformer model and prints model details.
- `embed_text <text>`
  - Embeds arbitrary text and prints a short vector preview.
- `verify_embeddings`
  - Loads or rebuilds full-document embeddings and prints their shape.
- `embedquery <query>`
  - Embeds a query and prints a short vector preview.
- `search <query> --limit <k>`
  - Full-document semantic search.
- `search_chunked <query> --limit <k>`
  - Chunk-based semantic search.
- `chunk <query> [--chunk-size INT] [--overlap INT]`
  - Simple word-count chunking helper.
- `embed_chunks`
  - Builds or reloads cached chunk embeddings.
- `semantic_chunk <query> [--max-chunk-size INT] [--overlap INT]`
  - Sentence-aware chunking helper.

## Retrieval Pipelines

### 1. Keyword Search

`cli/Inverted_Index.py` builds an inverted index over:

```text
title + " " + description
```

Preprocessing:

1. Lowercase
2. Remove punctuation
3. Remove stopwords from `data/stopwords.txt`
4. Apply Porter stemming

Artifacts written to `cache/`:

- `index.pkl`
- `docmap.pkl`
- `term_frequencies.pkl`
- `doc_lenghts.pkl`

BM25 scoring is implemented in `InvertedIndex.bm25_search(...)`.

### 2. Semantic Search

Full-document semantic search lives in [cli/lib/semantic_search.py](/home/krish/Krish/RAG/rag-search-engine/cli/lib/semantic_search.py).

- Model: `all-MiniLM-L6-v2`
- Input per movie: `"title: description"`
- Similarity: cosine similarity
- Cache artifact: `cache/movie_embeddings.npy`

### 3. Chunked Semantic Search

Chunked semantic search lives in [cli/lib/chunked_semantic_search.py](/home/krish/Krish/RAG/rag-search-engine/cli/lib/chunked_semantic_search.py).

- Splits descriptions into sentence-based chunks
- Default chunking in current code uses `chunk_size=4` and `overlap=1`
- Stores:
  - `cache/chunk_embeddings.npy`
  - `cache/chunk_metadata.json`
- Scores chunks against the query, then keeps the best chunk per movie

Hybrid search currently uses this chunked semantic path.

### 4. Hybrid Search

`cli/hybrid_search.py` combines:

- BM25 results from `InvertedIndex`
- Chunked semantic results from `ChunkedSemanticSearch`

Available fusion methods:

- Weighted score fusion
- Reciprocal Rank Fusion

RRF score:

```text
1 / (k + rank)
```

## Cache Behavior

The project writes reusable artifacts into `cache/`.

- BM25 cache is created manually with `keyword_search_cli.py build`
- Semantic caches are created on demand when the relevant search CLI runs
- Chunk caches are automatically rebuilt if the cached chunk count no longer matches the current dataset

## Testing

Current automated coverage in the repo is focused on hybrid CLI behavior:

```bash
uv run pytest cli/test_hybrid_search_cli.py
```

`test_gemini.py` is a manual connectivity check for the Gemini API and requires `GEMINI_API_KEY`.

## Known Limitations

- Several paths are hardcoded to `~/Krish/RAG/rag-search-engine/...` instead of being resolved relative to the repo root.
- `cli/keyword_search_cli.py search` is not implemented as an actual search command.
- `bm25idf` currently calls into the BM25 IDF logic but does not print a formatted result.
- The repo is CLI-oriented and does not yet expose a reusable application package or service interface.

## Recommended Workflow

For the current codebase, the cleanest path is:

1. `uv sync`
2. `uv run python cli/keyword_search_cli.py build`
3. `uv run python cli/hybrid_search_cli.py rrf-search "<query>" --limit 5`
4. Add `--enhance spell` or `--enhance rewrite` only if `GEMINI_API_KEY` is configured
