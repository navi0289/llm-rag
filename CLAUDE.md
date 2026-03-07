# CLAUDE.md -- llm-rag

## Build
```bash
cmake -B build && cmake --build build
```

## THE ONE RULE: SINGLE HEADER
`include/llm_rag.hpp` is the entire library. Never split it.

## Common Notes
- Index stored as binary file (not JSONL)
- Chunking tries to break at sentence boundaries (. ! ?)
- Augmented prompt format: "Context:\n[1] ...\n\nQuestion: ...\nAnswer:"
- embed_text and complete are private helpers using libcurl
