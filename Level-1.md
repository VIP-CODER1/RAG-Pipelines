# Level 1 — The Standard RAG Pipeline, Deep Dive

## Stage 1 — Ingestion & parsing

### How PDF parsing actually works

A PDF is a *display* format. The spec (ISO 32000) describes a page as drawing operators — `BT … Tj … ET` (text), `re/m/l` (rectangles/lines), image XObjects, and font dictionaries that map glyph IDs to Unicode via a `ToUnicode` CMap. **There is no native notion of "paragraph," "table," or "reading order."** Three regimes:

1. **Text-layer extraction** — walk the content stream, collect `Tj`/`TJ` operators, use `Tm` transformation matrices to recover (x, y, w, h) per glyph. PyMuPDF (~50–500 pages/sec), pdfplumber, pdfminer.six.
2. **OCR** — no text layer or broken `ToUnicode` (you get `□□□□`). Rasterize at 200–300 DPI, run Tesseract (~1–3 s/page), PaddleOCR, or vision LLM (~3–10 s, $0.005–0.02/page).
3. **Layout analysis** — a CV model (LayoutLMv3, DiT, YOLO-doc) detects bounding boxes of `{Title, Text, Table, Figure, List, Header, Footer}`. Unstructured uses detectron2+yolox; Docling uses DocLayNet-trained models; Marker uses Surya. 0.5–5 s/page on GPU.

### Why naive extractors break

- **Multi-column** — drawing order ≠ reading order. Two-column papers often zigzag (left line, right line, …). Fix: cluster by x then sort by y.
- **Tables** — just lines + positioned text. No `<table>` tag. pdfplumber detects rulings; Camelot has `lattice`/`stream` modes — both fragile. Modern: Microsoft Table Transformer (TATR) outputs cell bboxes, OCR fills cells.
- **Headers/footers** — repeated text at fixed y; cluster and drop.
- **Equations** — Computer Modern glyphs with broken Unicode maps. Use `pix2tex`/Nougat to OCR to LaTeX.
- **Code blocks** — most parsers collapse whitespace; indentation lost.
- **Ligatures** — "fi" is one glyph (U+FB01).

### Tool comparison

| Tool | Strength | Weakness | Cost |
|---|---|---|---|
| **PyMuPDF (fitz)** | Fastest text; span/block API | No layout intelligence; AGPL | Free |
| **pdfplumber** | Best low-level table geometry | Slow; manual layout | Free |
| **Unstructured.io** | Element typing (Title/NarrativeText/Table); `chunk_by_title` | Heavy deps; OSS lags hosted | OSS + paid API |
| **LlamaParse** | Excellent tables; markdown out; multimodal | API only | ~$3/1k pages |
| **Docling** (IBM) | OSS, fast on CPU, strong tables, JSON export | Newer ecosystem | Free |
| **Marker** | High-quality markdown; equations → LaTeX | GPU recommended; license tiers | Free/commercial |
| **Azure Document Intelligence** | Production tables/forms; SLA | Expensive | ~$10/1k pages |
| **AWS Textract** | Forms + queries | Pricey; weak on scientific PDFs | $1.50–$15/1k |

Community ranking on scientific PDFs (Docling tech report arXiv:2408.09869 and Marker's benchmark): **Marker ≈ Docling > LlamaParse > Unstructured > pdfplumber > PyMuPDF** on fidelity; PyMuPDF wins on speed.

### Non-PDF sources

- **HTML** — 90% chrome (nav/ads). Use **trafilatura** (best F1 per Barbaresi 2021) or **readability-lxml** (Arc90 heuristics: link-density-inverse × text-length × semantic tags).
- **Markdown** — `markdown-it-py` to AST, walk by heading depth.
- **Code** — split by AST via **tree-sitter** (100+ grammars). LangChain's `RecursiveCharacterTextSplitter.from_language(...)` is the lazy version; tree-sitter split at function/class boundaries is the right version. See `aider`'s repo-map and Sourcegraph Cody.

### Element typing matters

If parser emits `[{type: Title, text}, {type: Table, html}, {type: NarrativeText, text}]`, you can:

- Chunk by title boundaries (`chunk_by_title`)
- Embed table HTML separately or run an LLM caption ("Summarize this table for retrieval")
- Skip page chrome
- Use titles as parent nodes for hierarchical retrieval

### Metadata to store at ingest

```python
{
  "doc_id": "uuid",
  "source_uri": "s3://...",
  "source_type": "pdf|html|md|code",
  "title": "...",
  "section_path": ["Chapter 3", "Section 3.2"],
  "page": 17,
  "bbox": [x0, y0, x1, y1],               # for highlight-in-PDF UX
  "element_type": "NarrativeText",
  "language": "en",
  "modified_at": "...",                    # incremental re-index
  "hash": "sha256(text)",                  # dedup + change detection
  "acl": ["group:engineering"],            # filtered retrieval
}
```

The `hash` is what makes ingestion idempotent and incremental.

---

## Stage 2 — Chunking

### The fundamental tradeoff

Small chunks → precise retrieval, thin generation context. Large chunks → rich context, embedding becomes a "topic average" and recall drops. Empirically (Pinecone, LlamaIndex benchmarks), nDCG@10 peaks around **256–512 tokens** for prose with OpenAI/Cohere embeddings.

### Fixed-size

- **Token-based** is correct (matches embedder's tokenizer).
- **Character-based** is approximate (~4 chars/token English, ~2 code, less CJK).
- **Overlap** prevents straddled answers — typical **10–20%** (50–100 tokens on 512-token chunks).

### Recursive character splitting (LangChain algorithm)

1. Try splitting on the first separator (`"\n\n"`).
2. For pieces still over the limit, recurse with the next (`"\n"`, then `" "`, then `""`).
3. Merge adjacent pieces up to size limit, applying overlap.

Default separators: `["\n\n", "\n", " ", ""]`. The principle is *graceful degradation* — split on the most semantic boundary that fits.

### Markdown / structural splitting

Heading-aware splitters emit chunks + `headers` metadata (`{"h1": "Intro", "h2": "Motivation"}`). Critical for citation and hierarchical retrieval. Code-block-aware splitters refuse to cut inside a fence — half a code block destroys retrieval AND generation.

### Semantic chunking (Kamradt's "5 levels")

```python
# 1. Split into sentences
# 2. Embed each sentence
# 3. Cosine distance between adjacent pairs
# 4. Threshold = 95th percentile distance
# 5. Cut wherever distance > threshold
```

Helps on narratives that shift topic mid-paragraph; hurts on dense technical text (every sentence is "different"). 50–100× more expensive at ingest. Default to recursive; reach for semantic only when measured.

### Sentence-window retrieval

Index single sentences; return ±k neighbors at retrieval (k = 1–3). LlamaIndex `SentenceWindowNodeParser`. Precise match + generous generation context.

### Parent-document / hierarchical / auto-merging

Index children (small, ~128 tokens); return parents (~1024 tokens or full doc) when a child hits. LangChain `ParentDocumentRetriever`, LlamaIndex `AutoMergingRetriever`. Auto-merging twist: if N children of one parent are all in top-k, swap them for the parent.

### Late chunking (Jina, Günther et al. arXiv:2409.04701, 2024)

Inverts the standard order:

1. Run the **entire doc** through a long-context encoder (Jina v2/v3, 8192 tokens).
2. You get token-level hidden states *contextualized over the whole doc*.
3. *Then* mean-pool inside each chunk's token range.

Each chunk's embedding "knows" surrounding context (resolves pronouns, inherits topic). Jina reports +5–10% nDCG on some BEIR tasks. Needs a long-context encoder.

### Sizing by content type

| Content | Chunk size | Overlap | Splitter |
|---|---|---|---|
| Prose / docs | 256–512 tok | 10–20% | Recursive |
| Code | 200–800 tok | 0 (split at function) | tree-sitter |
| Tables | 1 table = 1 chunk | n/a | Element-aware + LLM-summary |
| Chat logs | 1 turn or thread | n/a | Conversation-aware |
| Legal / scientific | 512–1024 + parent-doc | 50–100 | Hierarchical |
| Tweets / SMS | 1 message | 0 | Identity |

### Common bugs

- **Orphaned sentences** ("It does this." with no referent) → overlap, sentence-window, late chunking.
- **Split tables** → element-aware chunking; never split tables.
- **Mid-code-block cuts** → language-aware separators or tree-sitter.
- **Lost section headers** → prepend the heading path to chunk text AND keep in metadata.

---

## Stage 3 — Embedding models

### Bi-encoders: why dual-tower matters

Query and passage go through the *same* (or twin) transformer independently → vectors `q, p ∈ R^d` → score = `cos(q, p)`. Passages encode **once at ingest**; query time is one forward pass + ANN lookup. Cross-encoders (Stage 6) take `[q; p]` as a single input — accurate but O(N) per query. Dual-tower is what makes ANN possible.

### Asymmetric embeddings (the #1 silent bug)

Many modern embedders are trained with prefixes:

- **E5** — `"query: "` vs `"passage: "`
- **BGE** — `"Represent this sentence for searching relevant passages: "` on query
- **Nomic** — `search_query:` / `search_document:` / `classification:` / `clustering:`

Forget the prefix → 5–15% recall drop. Most common bug in self-hosted pipelines.

### Matryoshka Representation Learning (Kusupati et al., NeurIPS 2022)

Trained so *any prefix* `v[:k]` is a valid lower-fidelity embedding. OpenAI text-embedding-3-large (3072 native) supports `dimensions=256/512/1024/...`; same for Nomic, mxbai, Jina v3, Cohere v3-light.

Uses:

- Two-stage: search 256-dim (10× faster), rerank top-100 with full 3072.
- Storage caps (pgvector historically 2000-dim cap).

Quality loss 3072 → 256 on text-embedding-3-large: ~1–3% MTEB.

### Concrete comparison (late 2024 / early 2025)

| Model | Dim | Max tok | MTEB avg | Notes | Price |
|---|---|---|---|---|---|
| OpenAI 3-small | 1536 (matryoshka) | 8191 | ~62.3 | Default workhorse | $0.02/1M |
| OpenAI 3-large | 3072 (matryoshka) | 8191 | ~64.6 | Best proprietary general | $0.13/1M |
| Cohere embed-v3-english | 1024 | 512 | ~64.5 | Compression-aware | $0.10/1M |
| Voyage-3-large | 1024 (matryoshka) | 32k | ~65+ | Long docs + code | $0.18/1M |
| BGE-m3 | 1024 | 8192 | ~60 dense (+ sparse + ColBERT modes) | OSS, three views in one model | Self-host |
| bge-large-en-v1.5 | 1024 | 512 | ~64.2 | OSS classic | Self-host |
| E5-mistral-7b-instruct | 4096 | 32k | ~66 | 7B, instruction-tuned, slow | Self-host (GPU) |
| Nomic-embed-text-v1.5 | 768 (matryoshka) | 8192 | ~62.3 | Apache-2, reproducible | Self-host |
| Jina embeddings v3 | 1024 (matryoshka) | 8192 | ~65.5 | Late-chunking, task LoRAs | API/self-host |
| GTE-large-en-v1.5 | 1024 | 8192 | ~65 | Alibaba, strong OSS | Self-host |

Always run a domain-specific eval; MTEB rankings are ordinals not gospel.

### Fine-tuning rule of thumb

- **Don't** fine-tune for general English with <10k pairs.
- **Do** fine-tune for legal/medical/code, non-English, or thousands of labeled pairs.
- **GPL** (Wang 2022): synthetic queries → hard-negative mining → contrastive fine-tune. The recipe for zero-label domain adaptation.

### Practical knobs

- **Normalize** vectors L2 at ingest → cosine ≡ dot product. Most ANN libs are tuned for dot.
- **Truncation** — embedders silently truncate at max length. Log truncation rate.
- **Batching** — sentence-transformers `encode(..., batch_size=64)`. GPU peaks 32–128 batch.
- **Throughput** — single A10G: ~2–5k 512-token passages/sec with bge-large. OpenAI API: ~3k embeddings/sec batched.

---

## Stage 4 — Vector stores

### ANN under the hood

- **HNSW** (Malkov & Yashunin, arXiv:1603.09320) — multi-layer graph, upper layers sparse "highways." Search: greedy descent + best-first beam (`efSearch` = beam width). RAM-resident, O(log N), recall@10 ≥ 0.95 routine.
- **IVF** — k-means cluster into `nlist` Voronoi cells (typically √N). Query: find closest `nprobe` cells, brute-force inside.
- **IVF-PQ** — after IVF, residual vectors quantized via Product Quantization (split into m sub-vectors, k-means each → 1 byte). 768-dim f32 (3072 B) → ~96 B. **30× compression, 2–5% recall loss.**
- **DiskANN / Vamana** (Subramanya 2019) — SSD-resident graph; compressed vectors in RAM, full on SSD. Used by Pinecone serverless and pgvectorscale.

### Knobs

| Knob | Index | Higher = | Tradeoff |
|---|---|---|---|
| `efSearch` | HNSW | more recall | linear latency; defaults 40–200 |
| `M` | HNSW | denser graph | RAM; better ceiling |
| `nprobe` | IVF | more clusters | linear latency |
| `nlist` | IVF | finer clusters | build time |

### Filtering breaks naive ANN

"Top-10 WHERE `tenant_id = 42`":

- **Pre-filter** (filter then exact) — correct; small filter sets fall back to brute force.
- **Post-filter** (ANN then filter) — fast but under-returns; over-fetch (`k × 10`) and pray.
- **Single-stage filtered ANN** — Qdrant builds payload indexes consulted during HNSW traversal; Weaviate and Milvus similar. pgvectorscale's StreamingDiskANN is filter-aware.

This is the **single biggest real-world differentiator** between vector DBs.

### Comparison

| Store | Index | Filter design | Strength | Weakness |
|---|---|---|---|---|
| **pgvector + pgvectorscale** | HNSW / StreamingDiskANN | Native SQL + filter-aware ANN | One DB; transactions; mature ops | Tuning fiddly |
| **Pinecone serverless** | DiskANN-like, S3-tiered | Metadata index | Zero ops; namespaces | Vendor lock; cold-start tax |
| **Weaviate** | HNSW + inverted | Filter-aware HNSW | GraphQL, hybrid built-in | Heavier ops |
| **Qdrant** | HNSW | Payload index, filter-aware HNSW | Best filtering; Rust; gRPC | Newer |
| **Milvus** | HNSW/IVF/DiskANN/GPU | Bitmap + scalar | Billion-scale, GPU | Complex deploy |
| **Chroma** | HNSW | Metadata | Dead-simple DX | Not for scale |
| **Vespa** | HNSW + tensor + WAND | Lucene queries | Full hybrid + ML ranking; first-class ColBERT/ColPali | Steep curve |
| **LanceDB** | IVF-PQ on Lance format | Pushdown | Embedded; zero-copy | Younger |
| **Elastic / OpenSearch** | HNSW + Lucene | Lucene | Existing Elastic shops; trivial hybrid | Vector perf trails |

### Quantization

| Scheme | Memory | Recall hit |
|---|---|---|
| float32 | 4 B/dim | baseline |
| float16/bf16 | 2 B/dim | <1% |
| int8 scalar | 1 B/dim | 1–3% |
| PQ (m=64, 8-bit) | ~m B/vec | 2–5% |
| Binary | 1 bit/dim | 5–15% (recoverable with rerank-on-floats) |

**Binary + rerank-on-floats** is the modern trick (Cohere, mxbai): 1-bit index → top-1000 → exact-score top-100 with floats → top-10. ~32× memory cut for ~1% loss.

### Operational

- **Pinecone serverless** (2024): ~$0.33/M reads, $4/M writes, $0.33/GB-month. Cold namespaces sleep → warm-up tax.
- **Self-hosted rule**: 1.5–2× raw vector size in RAM for HNSW + payload. 10M × 768-dim f32 ≈ **30 GB vectors + ~20 GB graph**.

---

## Stage 5 — Retrieval strategies

### Dense ≠ relevance

Dense encodes *topical similarity*. "Python pickle vulnerabilities" matches "Python serialization security" — even without the word "pickle." That's the win. The loss: "error code E1042" matches "error code E1043" with cosine 0.99.

### BM25 formula (Robertson & Zaragoza 2009)

`score(q, d) = Σ_t∈q IDF(t) · (f(t,d)·(k1+1)) / (f(t,d) + k1·(1 − b + b·|d|/avgdl))`

`k1 ≈ 1.2`, `b ≈ 0.75`. `f(t,d)` = term frequency (saturated by k1), `|d|/avgdl` = length normalization. IDF rewards rare terms. **BM25 wins on exact tokens** — error codes, identifiers, names, SKUs.

### SPLADE (Formal et al., SIGIR 2021)

BERT outputs a sparse vector in *vocabulary space* — each dim is a term, value is learned importance (with expansion: synonyms get non-zero weight even if absent). Runs on inverted index like BM25 but understands semantics. SPLADE-v3 current; bge-m3 has a sparse view too.

### Hybrid: normalization vs RRF

- **Score normalization** (min-max, z-score) + weighted sum `α·s_dense + (1−α)·s_bm25` — brittle; per-query distributions vary.
- **RRF** (Cormack 2009): `RRF(d) = Σ 1/(k + rank_i(d))` with k=60. **No calibration needed; nearly always beats tuned weighted-sum.** Elastic, Weaviate, Vespa, Qdrant, OpenSearch ship it natively.

### MMR (Carbonell & Goldstein 1998)

`MMR = arg max_{d∈R\S} [λ·sim(d,q) − (1−λ)·max_{d'∈S} sim(d,d')]`

λ = 0.5–0.7 typical. Use when top-k is dominated by near-duplicates.

### Multi-index routing

- **Fan-out + RRF** — query all indexes, fuse. Simple, robust.
- **Router** — small LLM/classifier picks index. Lower cost; fragile on ambiguity.
- **Tool-use retrieval** — agent calls `search_code()`/`search_docs()` as tools. Emerging norm.

---

## Stage 6 — Reranking

### Bi vs cross architecturally

- **Bi-encoder** — two forward passes, cheap vector compare, enables ANN.
- **Cross-encoder** — input `[CLS] query [SEP] passage [SEP]`, full self-attention between query and passage tokens, output one relevance scalar. **10–100× more accurate**, O(N) per query → only on a reduced candidate set.

### Top-k → top-n funnel

Retrieve top-50 to 200 with hybrid → rerank to top-3 to 10 with cross-encoder. Cross-encoder *demotes* lexically-matched-but-wrong (from BM25) and *promotes* semantically-right-but-lexically-light (from dense).

### Models

| Model | Type | Latency | Notes |
|---|---|---|---|
| Cohere Rerank 3 | API cross | ~100ms/100 docs | Proprietary best-in-class; 4k tok docs |
| Cohere Rerank 3.5 | API | similar | Newer, better multilingual |
| bge-reranker-v2-m3 | OSS XLM-Roberta | ~10ms/pair A10G | Strong multilingual default |
| bge-reranker-v2-gemma | OSS Gemma-2B | slower | Higher accuracy |
| Jina Reranker v2 | API/OSS | fast | Code + multilingual |
| Voyage Rerank-2 | API | fast | Pairs with voyage embeds |
| mxbai-rerank-large-v1 | OSS | medium | Strong English |

### LLM-as-reranker

**RankGPT** (Sun et al., EMNLP 2023) — sliding-window listwise: "Rank these 20 passages by relevance to Q." Permutation distillation → RankLLM/RankVicuna/RankZephyr. Use when zero-shot domain or you need explanations. Costly — one frontier LLM call per query.

### Latency

Cross-encoder ~3–10 ms/pair on A10G (512-tok pair). 50 candidates ≈ 150–500 ms. Cohere ~100–200 ms for 100 docs hosted. **Batch all pairs in one forward pass.**

### Calibration

Cross-encoder scores aren't calibrated probabilities. Either Platt-scale on a labeled set, or use relative threshold (top-N) + a floor below which you route to "no good results."

---

## Stage 7 — Prompt assembly & generation

### Context budgeting

```
total = sys + few_shots + question + history + retrieved + reserved_output
# e.g. 16k window: 500 + 1000 + 200 + 1500 + 11000 + 1800
```

Reserve `max_tokens` carefully — APIs error if under-reserved vs what the model wants to produce.

### Lost in the middle (Liu et al., TACL 2024, arXiv:2307.03172)

Models attend disproportionately to start and end; middle systematically under-used. **20+ point accuracy drop** between "answer at position 1" vs "position 10 of 20." Practical:

- Keep retrieved chunks ≤ 5–10 when possible.
- Highest-rank-last or -first, not -middle.
- Aggressively rerank to *shrink*, not expand.

### Ordering

- **Relevance-desc** — simplest; middle decay if many chunks.
- **U-shape** — best at top AND bottom (which the model attends to most).
- **Interleaved by source** — avoid stylistic clumping when mixing sources.

### Citation patterns

```text
# Numbered (model-agnostic)
[1] {chunk_1}
[2] {chunk_2}
Answer using bracketed citations like [1].

# XML (Anthropic recommends for Claude)
<document index="1" source="...">...</document>
<document index="2" source="...">...</document>

# JSON output (tool use / structured output)
{ "answer": "...", "citations": [{"doc_id": "...", "quote": "..."}] }
```

### Anti-hallucination patterns

- "If the answer is not in the context, respond exactly: 'I don't know based on the provided documents.'"
- Require verbatim quotes per claim.
- Two-step **extract-then-answer** (Anthropic pattern): (1) extract relevant quotes, (2) answer using only those quotes.
- Self-check: verify each claim is supported by a cited chunk.

### Structured output

JSON-Schema-constrained decoding (OpenAI `response_format`, Anthropic tool use, vLLM/Outlines/llama.cpp grammars). Output parseable by construction. Small latency tax; never `try/except json.JSONDecodeError` again.

### "No good results"

- **Refuse** ("I don't know") — high-stakes (legal/medical/customer-facing).
- **Guess with caveat** — low-stakes brainstorming.
- **Ask for clarification** — chat UX, genuinely ambiguous query.

Trigger: top-1 reranker score < threshold OR top-N avg < threshold.

### Model choice

| Use case | Tier |
|---|---|
| FAQ, classification | Small: gpt-4o-mini, Claude Haiku, Llama-3.1-8B ($0.15–0.50/M) |
| General Q&A, summarization | Mid: gpt-4o, Claude Sonnet, Llama-3.1-70B |
| Multi-doc synthesis, code, reasoning | Frontier: Claude Opus, GPT-4.1/5, Gemini 2.5 Pro |

**Rule of thumb: spend savings on better retrieval before paying for a frontier generator.** A small model on perfect context beats a frontier model on noisy context, almost always.

---

## How the stages compose

```text
Query: str
  │
  ├─► (optional) rewrite / expand / HyDE
  │       Query → Query'  (or [Query, hypothetical_doc])
  ▼
[Retrievers]                          ── Stage 5
  ├─ DenseANN(embed(Query'))   ─► [ScoredChunk{id, score_dense}]
  └─ BM25(Query')              ─► [ScoredChunk{id, score_bm25}]
        ▼ RRF fuse                ─► [ScoredChunk] (top-100)
  ▼
[Reranker — cross-encoder]            ── Stage 6
   score(Q, chunk) per pair       ─► [RerankedChunk] (top-5)
  ▼
[Prompt assembler]                    ── Stage 7
   sys + question + ordered(chunks) + citation scaffold
  ▼
[LLM]
   Answer: str (+ citations)


Offline / ingest path (Stages 1–4):

source_uri
  ▼ parse                Stage 1
[Document{title, elements:[{type, text, bbox, page, ...}]}]
  ▼ chunk                Stage 2
[Chunk{text, parent_id?, metadata:{section_path, page, doc_id, ...}}]
  ▼ embed (asymmetric)   Stage 3
[Chunk & {vec: float32[d]}]
  ▼ upsert               Stage 4
VectorStore (HNSW/IVF + payload) + InvertedIndex (BM25/SPLADE)
```

Type signatures:

```
parse    : URI                  → Document
chunk    : Document             → List[Chunk]
embed    : List[Chunk]          → List[Chunk & {vec: Vec[d]}]
upsert   : List[Chunk & {vec}]  → VectorStore × InvertedIndex
retrieve : Query × Filters      → List[ScoredChunk]
rerank   : Query × List[Chunk]  → List[Chunk]                 (k → n, n ≪ k)
assemble : Query × List[Chunk]  → Prompt
generate : Prompt               → Answer × List[Citation]
```
