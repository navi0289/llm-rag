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

| Repo | Purpose |
|------|---------|
| [llm-stream](https://github.com/Mattbusel/llm-stream) | SSE streaming |
| [llm-cache](https://github.com/Mattbusel/llm-cache) | Response caching |
| [llm-cost](https://github.com/Mattbusel/llm-cost) | Token cost estimation |
| [llm-retry](https://github.com/Mattbusel/llm-retry) | Retry + circuit breaker |
| [llm-format](https://github.com/Mattbusel/llm-format) | Markdown/code formatting |
| [llm-embed](https://github.com/Mattbusel/llm-embed) | Embeddings + cosine similarity |
| [llm-pool](https://github.com/Mattbusel/llm-pool) | Connection pooling |
| [llm-log](https://github.com/Mattbusel/llm-log) | Structured logging |
| [llm-template](https://github.com/Mattbusel/llm-template) | Prompt templates |
| [llm-agent](https://github.com/Mattbusel/llm-agent) | Tool-use agent loop |
| [llm-rag](https://github.com/Mattbusel/llm-rag) | Retrieval-augmented generation |
| [llm-eval](https://github.com/Mattbusel/llm-eval) | Output evaluation |
| [llm-chat](https://github.com/Mattbusel/llm-chat) | Multi-turn chat |
| [llm-vision](https://github.com/Mattbusel/llm-vision) | Vision/image inputs |
| [llm-mock](https://github.com/Mattbusel/llm-mock) | Mock LLM for testing |
| [llm-router](https://github.com/Mattbusel/llm-router) | Model routing |
| [llm-guard](https://github.com/Mattbusel/llm-guard) | Content moderation |
| [llm-compress](https://github.com/Mattbusel/llm-compress) | Prompt compression |
| [llm-batch](https://github.com/Mattbusel/llm-batch) | Batch processing |
| [llm-audio](https://github.com/Mattbusel/llm-audio) | Audio transcription/TTS |
| [llm-finetune](https://github.com/Mattbusel/llm-finetune) | Fine-tuning jobs |
| [llm-rank](https://github.com/Mattbusel/llm-rank) | Passage reranking |
| [llm-parse](https://github.com/Mattbusel/llm-parse) | HTML/markdown parsing |
| [llm-trace](https://github.com/Mattbusel/llm-trace) | Distributed tracing |
| [llm-ab](https://github.com/Mattbusel/llm-ab) | A/B testing |
| [llm-json](https://github.com/Mattbusel/llm-json) | JSON parsing/building |

## License

MIT -- see [LICENSE](LICENSE).
