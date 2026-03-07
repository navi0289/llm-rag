# llm-rag

Retrieval-augmented generation for C++. One header, libcurl dep.

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Single Header](https://img.shields.io/badge/single-header-orange.svg)
![Requires libcurl](https://img.shields.io/badge/deps-libcurl-yellow.svg)

## Quickstart

```cpp
#define LLM_RAG_IMPLEMENTATION
#include "llm_rag.hpp"

llm::RagConfig cfg;
cfg.embed_api_key = "sk-...";
cfg.llm_api_key   = "sk-...";
cfg.top_k         = 3;

llm::RagPipeline rag(cfg);
rag.ingest("Mitochondria produce ATP through cellular respiration...", "biology");
rag.save_index();

auto result = rag.query("What produces ATP?");
std::cout << result.answer << "\n";
```

## Pipeline

```
text -> chunk_text() -> embed each chunk -> store in binary index
query -> embed query -> cosine search -> top-k chunks -> augmented prompt -> LLM -> answer
```

## API Reference

```cpp
// Free function
std::vector<std::string> chunk_text(text, chunk_size, overlap);

// Pipeline
llm::RagPipeline rag(config);
rag.ingest(text, source_name);
rag.ingest_file(filepath);
rag.save_index();
rag.load_index();
rag.chunk_count();

auto chunks = rag.retrieve(question);
auto result = rag.query(question);
// result.answer, result.retrieved_chunks, result.augmented_prompt
```

## Examples

| File | What it shows |
|------|--------------|
| [`examples/basic_rag.cpp`](examples/basic_rag.cpp) | Ingest two docs, query for an answer |
| [`examples/chunker.cpp`](examples/chunker.cpp) | chunk_text() standalone demo |

## Building

```bash
cmake -B build && cmake --build build
export OPENAI_API_KEY=sk-...
./build/basic_rag
./build/chunker
```

## Requirements

C++17. Requires libcurl.

## See Also

| Repo | What it does |
|------|-------------|
| [llm-stream](https://github.com/Mattbusel/llm-stream) | Stream OpenAI and Anthropic responses via SSE |
| [llm-cache](https://github.com/Mattbusel/llm-cache) | LRU response cache |
| [llm-cost](https://github.com/Mattbusel/llm-cost) | Token counting and cost estimation |
| [llm-retry](https://github.com/Mattbusel/llm-retry) | Retry and circuit breaker |
| [llm-format](https://github.com/Mattbusel/llm-format) | Structured output / JSON schema |
| [llm-embed](https://github.com/Mattbusel/llm-embed) | Embeddings and vector search |
| [llm-pool](https://github.com/Mattbusel/llm-pool) | Concurrent request pool |
| [llm-log](https://github.com/Mattbusel/llm-log) | Structured JSONL logging |
| [llm-template](https://github.com/Mattbusel/llm-template) | Prompt templating |
| [llm-agent](https://github.com/Mattbusel/llm-agent) | Tool-calling agent loop |
| [llm-rag](https://github.com/Mattbusel/llm-rag) | RAG pipeline |
| [llm-eval](https://github.com/Mattbusel/llm-eval) | Evaluation and consistency scoring |
| [llm-chat](https://github.com/Mattbusel/llm-chat) | Conversation memory manager |
| [llm-vision](https://github.com/Mattbusel/llm-vision) | Multimodal image+text |
| [llm-mock](https://github.com/Mattbusel/llm-mock) | Mock LLM for testing |
| [llm-router](https://github.com/Mattbusel/llm-router) | Model routing by complexity |
| [llm-guard](https://github.com/Mattbusel/llm-guard) | PII detection and injection guard |
| [llm-compress](https://github.com/Mattbusel/llm-compress) | Context compression |
| [llm-batch](https://github.com/Mattbusel/llm-batch) | Batch processing and checkpointing |

## License

MIT -- see [LICENSE](LICENSE).
