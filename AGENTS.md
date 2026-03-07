# AGENTS.md -- llm-rag

## Purpose
Single-header C++ retrieval-augmented generation pipeline. Chunks text,
embeds via OpenAI, stores in a flat binary index, retrieves top-k chunks by
cosine similarity, builds an augmented prompt, and generates an answer.

## Architecture
```
llm-rag/
  include/llm_rag.hpp   <- THE ENTIRE LIBRARY. Do not split.
  examples/
    basic_rag.cpp
    chunker.cpp
  CMakeLists.txt
```

## Build
```bash
cmake -B build && cmake --build build
```

## Rules
- Single header only.
- Only libcurl as external dep.
- All public API in namespace llm.
- Implementation inside #ifdef LLM_RAG_IMPLEMENTATION guard.

## API Surface
- chunk_text(text, chunk_size, overlap) -> vector<string>
- RagPipeline: ingest/ingest_file, retrieve, query, save_index/load_index
