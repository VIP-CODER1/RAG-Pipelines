# RAG Pipeline — Zero to Production

A four-level deep dive into Retrieval-Augmented Generation, from the seven core pipeline stages through advanced techniques, production concerns, and a six-week build plan.

Roughly **18,000 words** of study material — about 2–3 focused hours of reading per level if you pause to follow citations.

---

## Contents

| # | File | What's inside | Read when |
|---|---|---|---|
| 1 | **[Level-1.md](./Level-1.md)** | The standard pipeline, stage by stage — ingestion, chunking, embeddings, vector stores, retrieval, reranking, prompt assembly. Mechanics, tool comparisons, concrete numbers, common bugs. | You want to understand *how RAG actually works* under the hood. |
| 2 | **[Level-2.md](./Level-2.md)** | Advanced techniques — HyDE, multi-query, step-back, RRF, hierarchical retrieval, RAPTOR, self-querying, ColBERT, ColPali, GraphRAG, Self-RAG / CRAG / Adaptive-RAG, agentic RAG, long-context vs RAG, **Anthropic Contextual Retrieval**, and a when-to-reach-for-what decision matrix. | Your L1 pipeline works but you need to push retrieval quality further. |
| 3 | **[Level-3.md](./Level-3.md)** | Production concerns — evaluation (RAGAS, TruLens, ARES, LLM-as-judge biases), observability (OTEL, Langfuse, Phoenix), caching at four layers, security (prompt injection, PII, ACLs, embedding inversion), cost & latency optimization with worked numbers, freshness & index lifecycle, multi-tenancy, reliability, reference architecture, **30-item production readiness checklist**. | You're moving from prototype to something users depend on. |
| 4 | **[Level-4.md](./Level-4.md)** | A six-week opinionated build path with weekly stack, success criteria, reading list, and mini-deliverables. War stories of what kills RAG projects. Deep recommended reading list grouped by topic. "How to know you're done." | You want to actually build the thing, not just read about it. |

---

## How to use this

### Default reading order

Read **L1 → L2 → L3 → L4** in order. Each level assumes the previous. L1 lays the vocabulary; L2 extends it; L3 is about running L2 in production; L4 is the project plan that ties them together.

### Faster paths if you're short on time

| You want… | Read |
|---|---|
| **Just enough to ship a prototype this week** | L1 §1–§3 (ingest, chunk, embed) + L4 Week 1 |
| **Make a working prototype noticeably better** | L1 §5–§7 (retrieval, rerank, prompt) + L2 §2 (hybrid + RRF) + L2 §10 (Contextual Retrieval) |
| **Decide between RAG and long-context** | L2 §9 (Long-context vs RAG) |
| **Pass a security review** | L3 §4 (Security) + L3 §10 (checklist) |
| **Defend a stack choice to your team** | L1 (mechanics) + L3 §1 (evaluation) so you can back claims with measurements |
| **Plan an actual project** | L4 end-to-end, then L1–L3 as references |

### Who this is for

- Engineers who've used a RAG framework (LlamaIndex, LangChain, etc.) and want to know what the abstractions hide.
- Tech leads scoping a RAG project and trying to plan it honestly.
- Anyone preparing for a system-design interview on retrieval-augmented systems.

### Prerequisites

- Comfortable Python.
- Rough mental model of transformers and embeddings — you don't need to derive attention, but you should know what "cosine similarity between embeddings" means.
- Familiarity with at least one LLM API (OpenAI, Anthropic, etc.).

### Conventions in the notes

- **Bold** marks paper titles, tool names, and key concepts.
- arXiv numbers are inline so you can `arxiv.org/abs/<n>` directly.
- Tables compare tools head-to-head with strengths, weaknesses, and rough cost.
- Code snippets are pseudo-code that illuminates the mechanism, not copy-paste implementations.
- Concrete numbers (latency, recall lift, cost) are sourced from the cited paper or vendor benchmark — treat them as ordinals, not absolutes; always measure on your own corpus.

### A note on currency

Written against the **late-2024 / early-2026** state of the field. RAG moves fast. Models, MTEB rankings, and prices will drift. The *pipeline shape*, the *failure modes*, and the *evaluation discipline* will not. Re-check the MTEB leaderboard and the Anthropic / OpenAI pricing pages before betting infrastructure on specific numbers.

### The single most important takeaway

If you only adopt one habit from these notes, it is this: **build a golden set of 100 (question, ideal_answer, ideal_chunks) triples before you tune anything else.** Every section of every level eventually reduces to "and now measure it against your eval set." Teams with evals ship; teams without them argue.

---

## File map

```
RAG-Pipeline/
├── README.md          (you are here)
├── Level-1.md         standard pipeline, seven stages
├── Level-2.md         advanced / customized techniques
├── Level-3.md         production concerns
└── Level-4.md         six-week build path + reading list
```
