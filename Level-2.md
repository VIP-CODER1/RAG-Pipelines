# Level 2 of a RAG Pipeline — Advanced / Customized RAG Techniques

> Prerequisite: you already know the L1 pipeline — load → chunk → embed → index → retrieve (dense + maybe BM25) → rerank → stuff into prompt → generate. Everything below sits *on top of* that and assumes you can measure recall@k and nDCG on a held-out set. If you can't, build that first; otherwise these techniques will feel like cargo cult.

A working mental model: L1 RAG fails for four reasons — **the query is wrong shape** (sections 1, 10), **a single retriever is biased** (sections 2, 5), **chunks are the wrong granularity** (sections 3, 10), **the question requires reasoning that retrieval-once cannot satisfy** (sections 6, 7, 8). Each section below maps to one (or more) of those failure modes.

---

## 1. Query transformation (deep)

The bridge between user intent and corpus vocabulary is the single biggest lever in a RAG system. Users ask short, context-poor questions; documents are long and full of jargon. Cosine similarity does not care that your user said "how do I lower my AWS bill" and the doc says "Reserved Instance pricing optimization." Query transformation rewrites the query into shapes that match the corpus.

### 1.1 HyDE — Hypothetical Document Embeddings

Paper: **Gao, Ma, Lin, Callan, "Precise Zero-Shot Dense Retrieval without Relevance Labels," ACL 2023, arXiv:2212.10496.**

**Mechanic.** Instead of embedding the user query, prompt an LLM to *hallucinate* a plausible answer document, then embed *that*. The embedding now lives in document-shaped space, so cosine similarity against the corpus stops comparing apples (a 12-token question) with oranges (an 800-token passage).

```python
HYDE_PROMPT = """Please write a passage to answer the question.
Question: {q}
Passage:"""

hypo = llm.complete(HYDE_PROMPT.format(q=user_q), max_tokens=256)
hypo_vec = embed(hypo)
hits = vector_store.search(hypo_vec, k=20)
```

In the paper, on TREC-DL-19 passage, zero-shot HyDE with InstructGPT + Contriever beat Contriever-only by **+11 nDCG@10** (45.1 → 56.1) and matched a fine-tuned BM25→DPR cascade. On BEIR-FiQA the lift was ~+8 nDCG@10. Critically, the gains *shrink* as the dense retriever itself becomes stronger and as in-domain fine-tuning data appears.

**When it wins.** (i) Strong query-document length/style asymmetry — short factoid questions over technical prose. (ii) Out-of-domain or zero-shot retrieval. (iii) Domains where the LLM has reasonable surface knowledge of the *form* of an answer even if not the *fact*.

**When it loses.** (i) Highly specialized corpora (medical codes, internal product names) where the LLM hallucinates entities that pull *wrong-but-confidently-related* documents — the failure mode is silent and ugly. (ii) When user queries are already long and document-shaped. (iii) Latency-tight paths: HyDE adds one full generation (typically 200–400 ms with a small model, more with GPT-4-class).

**Gotchas.** Cache by query hash — repeated queries are common. Use a *small fast* generator (Haiku, gpt-4o-mini, Llama-3-8B) — the paper shows the cheap models recover most of the gain. Some teams average several hypothetical embeddings (n=4–8) — modest extra lift, n× latency.

### 1.2 Multi-query retrieval

Prompt the LLM for k paraphrases of the query, retrieve top-n for each, then **union with score aggregation** (almost always Reciprocal Rank Fusion — see §2.1). Standard recipe popularized by LangChain's `MultiQueryRetriever`.

```python
MQ_PROMPT = """Generate 4 alternative phrasings of this question for a search engine.
Return one per line, no numbering.
Question: {q}"""
```

Cost: 1 LLM call + k× retrievals (cheap; retrieval is sub-50ms typically). Latency: ~300-600 ms added. Recall@k lift: empirically +3–8% over single-query on diverse internal eval sets, more on ambiguous queries, near-zero on already-explicit queries.

Dedup: hash-and-drop exact dupes; for near-dupes either embed-and-cluster (overkill) or just let RRF naturally down-weight redundancy.

### 1.3 Step-back prompting

Paper: **Zheng, Mishra, Chen, Cheng, Chi, Le, Zhou, "Take a Step Back: Evoking Reasoning via Abstraction in LLMs," ICLR 2024, arXiv:2310.06117 (Google DeepMind).**

**Mechanic.** Before retrieving for the literal question, prompt the LLM for a *more abstract* version, retrieve for both, fuse contexts.

```
Original: "Was Estella Leopold employed at Stanford between Aug 1954 and Nov 1954?"
Step-back: "What is Estella Leopold's employment history?"
```

The step-back query pulls broader background facts; the original pulls specifics. The model then answers using both. On MMLU-Physics/Chemistry the paper reports **PaLM-2L +7–11 points**; on TimeQA **+27 points**, on MuSiQue **+7 points**. The technique is essentially free of fine-tuning.

**When to use.** Multi-hop QA, temporal/comparative questions, anything where the literal query is too specific to retrieve the relevant *background*.

**Gotcha.** The abstraction prompt itself is brittle. Few-shot it with 3–5 domain-tuned exemplars or accept ~30% useless step-back queries.

### 1.4 Query decomposition / sub-question generation

For compound questions ("How do GraphRAG and RAPTOR differ in their handling of global queries?") split into sub-questions, retrieve and answer each, then synthesize. LlamaIndex's `SubQuestionQueryEngine` is the canonical implementation; it ties each sub-question to a specific index (`tool`) so you also get routing for free.

Pseudo:

```
plan = llm.decompose(q)               # list of (sub_q, tool_name)
parts = [tools[t].query(sq) for sq, t in plan]
final = llm.synthesize(q, parts)
```

Latency cost is linear in number of sub-questions; do them in parallel. Recurse only if a sub-question is itself compound — flat decomposition handles 90% of cases.

### 1.5 Standalone-rewrite for chat RAG

The single most-omitted-and-most-broken pattern in chat RAG. The user says "and the v2 version?" — embedded alone this matches nothing. Rewrite using last N turns:

```
CONDENSE_PROMPT = """Given chat history and a follow-up question,
rewrite the follow-up to be a standalone question.

History:
{history}
Follow-up: {q}
Standalone:"""
```

This is the LangChain `ConversationalRetrievalChain` pattern and the default in essentially every production chat system that does RAG well. Skip it and 30%+ of follow-up turns will retrieve nothing useful.

### 1.6 Query routing

Two patterns: (a) **embedding-based classifier** (embed query, kNN against centroids of each index's training queries) — fast (<5 ms), no LLM cost, brittle when you add a new index. (b) **LLM router** — prompt with index descriptions, return the index name(s). LlamaIndex `RouterQueryEngine`, LangChain `MultiRetrievalQAChain`. Slower (~200 ms) but maintainable: adding an index is a one-line description change.

Hybrid: cheap classifier, fall back to LLM when confidence < threshold.

### 1.7 RAG-Fusion

Adrian Raudaschl, 2023 (Towards Data Science / open source). **Recipe:** multi-query (4 paraphrases) → retrieve top-k for each → fuse with RRF (k=60). That's it. The reason this gets a name is that it became *the* "what should I try first beyond vanilla" baseline; it's the cheapest +5% you'll find. Use it as a control against which to measure anything else in this document.

---

## 2. Hybrid search & rank fusion (deep)

Dense retrieval handles semantics, BM25 handles rare-token recall (codes, names, identifiers). Neither dominates. Hybrid is now table-stakes.

### 2.1 Reciprocal Rank Fusion

Paper: **Cormack, Clarke, Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," SIGIR 2009.**

Formula:

$$\text{RRF}(d) = \sum_{i \in \text{systems}} \frac{1}{k + \text{rank}_i(d)}$$

Default **k = 60**. Why 60? Empirical — Cormack swept k ∈ {10, 30, 60, 100} on TREC; 60 was robust across runs. The intuition: with k=60 a document ranked 1 contributes 1/61 ≈ 0.0164; ranked 10 contributes 1/70 ≈ 0.0143; ranked 100 contributes 1/160 ≈ 0.00625. The k=60 prior keeps top-1 and top-10 close (so a single system can't dominate via one strong hit) while still meaningfully demoting tail ranks.

**Why it beats CombSUM/CombMNZ.** Those operate on *scores*, which are not comparable across retrievers — BM25 scores are unbounded positive, cosine sim is [-1, 1], a cross-encoder logit is something else again. Normalize them and you introduce arbitrary scaling decisions. RRF operates only on *ranks*, which are scale-free, so you don't have to commit to a normalization scheme. In the original paper RRF beat Condorcet, CombSUM, and CombMNZ on every TREC run tested.

Pseudo:

```python
def rrf(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:                 # ranking: list[doc_id]
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

### 2.2 Score normalization alternatives and their failure modes

Min-max: collapses to noise when the top score is an outlier. Z-score: assumes scores are roughly normal; BM25 is *not*. Sigmoid: needs a temperature you have no principled way to set. CombSUM after any of the above is workable on a single fixed pair of retrievers tuned on validation, but breaks the moment you swap one. RRF is the default for a reason.

### 2.3 Convex combination (α · dense + (1−α) · sparse)

Tunable α is worth the effort only when (a) you have a labeled eval set ≥500 queries, (b) one retriever consistently dominates so the optimum α is far from 0.5, and (c) you'll commit to retuning after any retriever upgrade. Otherwise: RRF and move on.

Pinecone, Weaviate, and Qdrant all expose hybrid-with-α; Elastic 8.x supports RRF natively (`text_expansion` + `match` + `rrf` retriever).

### 2.4 Cross-system fusion: dense + BM25 + SPLADE + ColBERT

SPLADE (Formal et al., SIGIR 2021/2022) is learned sparse — it produces BM25-shaped postings via an MLM head; recall@1k is ~5 points above BM25 on MS MARCO. ColBERT is late-interaction dense (§5). Combining 3–4 retrievers with RRF gives diminishing returns past 3; in practice **dense + BM25 + reranker** beats **dense + BM25 + SPLADE + ColBERT + reranker** on most internal evals because the reranker absorbs most of the diversity benefit.

### 2.5 Weighted RRF and learned fusion

Weighted RRF: $\sum_i w_i / (k + \text{rank}_i)$. Set w_i from validation nDCG. Learned-to-rank tail (LightGBM ranker fed features = score per system, query length, domain match) — squeezes another 1–3 points if you have the data. Most teams skip this.

---

## 3. Hierarchical & multi-granular retrieval (deep)

The chunking dilemma: small chunks embed precisely but lack context for the LLM; large chunks have context but embed mushily. Multi-granular retrieval breaks the tradeoff by **decoupling what you embed/search from what you return/feed**.

### 3.1 Parent-document retrieval

Index small (200–400 token) child chunks. On hit, return the *parent* (1500–3000 tokens, or a whole section). LangChain `ParentDocumentRetriever`, LlamaIndex `HierarchicalNodeParser`.

```
chunks_index:  child_id, child_vec, parent_id
parents_store: parent_id -> full parent text
```

Picking sizes: child ≈ "one fact" (1–3 sentences); parent ≈ "enough context to answer" (whatever fits N×child + LLM budget). Common: 256 / 1024 split.

### 3.2 Sentence-window retrieval

Index every sentence. On hit, return ±k sentences. Cheaper than parent-doc and beats it when answers are *highly local* (definitions, single-fact lookups). LlamaIndex `SentenceWindowNodeParser` with `window_size=3` is a good default. Worse when answers require multi-paragraph context (procedures, comparisons).

### 3.3 Auto-merging

**Rule:** if ≥ N children of the same parent appear in the top-k, replace them with the parent. Implemented in LlamaIndex `AutoMergingRetriever`. Avoids feeding the LLM five disjoint snippets when one full section would serve. Threshold typically N = ceil(0.5 × children_per_parent).

### 3.4 RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval

Paper: **Sarthi, Abdullah, Tuli, Khanna, Goldie, Manning, "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval," ICLR 2024, arXiv:2401.18059 (Stanford).**

**Mechanic.** Bottom-up: embed leaf chunks, cluster (Gaussian Mixture in the paper, with UMAP dim reduction), summarize each cluster with an LLM, embed those summaries, cluster again, recurse until one root. At query time, search the *entire tree* (or descend; the paper finds "collapsed tree" — flat search over all levels — works best). Retrieval returns a mix of granular leaves and high-level summaries.

Reported gains: on QuALITY (long-document QA) **+ 20 points** over BM25 baseline; SOTA on NarrativeQA when paired with GPT-4. Cost: O(N) extra LLM calls at ingest (one per cluster across all levels) — typically 1.1–1.3× the leaf count.

**When the tree pays off.** Long, narrative documents where answers require *abstraction across many chunks* (literary QA, multi-chapter reports). Pays poorly on short flat corpora (FAQs, support docs) — overkill.

### 3.5 Summary-and-detail dual-index

The poor-man's RAPTOR: per-document, LLM-generate a 200-token summary at ingest. Maintain two indices. Retrieve k_s summaries and k_d detail chunks; fuse. Cheap (one summary per doc) and works well for "what is doc X about" vs "what does doc X say about Y" mixed workloads. LangChain `MultiVectorRetriever`.

---

## 4. Metadata filtering & self-querying retrievers (deep)

Filtering is *the* lowest-cost recall lever and the most-neglected. A user query "Q3 2024 revenue from EMEA" against an unfiltered store competes with the entire corpus; with `quarter=Q3-2024 AND region=EMEA` it competes with ~0.1% of it.

### 4.1 Filterable payload design at ingest

Decide schema before you ingest 10M chunks. Categorical: `doc_type`, `team`, `product`. Numeric: `version_major`, `word_count`. Date: `created_at`, `valid_until` — store as Unix epoch for range queries. Hierarchical: encode paths as `/eng/platform/rag` and use prefix filters. ACL: per-chunk `allowed_groups: ["eng","leadership"]` — filter applied *server-side*, not post-hoc, or you'll leak via timing.

All major vector DBs (Pinecone, Qdrant, Weaviate, Milvus, pgvector w/ HNSW + WHERE) support payload filtering during HNSW traversal; performance drops gracefully unless you're filtering to <1% of the corpus, at which case dedicated partitioning beats filtered ANN.

### 4.2 Self-querying retrievers

LangChain `SelfQueryRetriever`. The LLM is given the metadata schema and parses NL into `(semantic_query, structured_filter)`:

```
User: "papers about diffusion models published after 2022 by researchers at Stanford"

Parsed:
  semantic_query: "diffusion models"
  filter: AND(gt(year, 2022), eq(affiliation, "Stanford"))
```

Prompt structure: schema (with field names, types, allowed values), few-shot examples, output format (a small DSL the library compiles to the vector DB's filter language). Failure modes: hallucinated field names (always validate), wrong operator on numeric vs categorical (constrain with types), missed filter ("recent" — don't extract; let semantic handle).

### 4.3 LLM-extracted metadata at ingest

For each doc, prompt: "Extract title, authors, date, topic-tags (from this list of 50), summary." Cache aggressively. Anthropic's Contextual Retrieval (§10) is a specialized case of this. Latency at ingest is unimportant (it's a one-time cost); recall at query time is everything.

### 4.4 Time-decay scoring

Blend recency into similarity:

$$\text{score}(d) = \text{sim}(q,d) \cdot \exp(-\lambda \cdot \Delta t)$$

Pick λ so half-life matches domain freshness — for news, ~30 days; for engineering docs, ~365 days; for legal, ~3650 days. LlamaIndex `TimeWeightedPostprocessor`. Crucial when you have multiple versions of the same fact and the old one is wrong.

---

## 5. Late interaction / multi-vector retrieval (deep)

Single-vector dense retrieval projects an entire passage to one point — information bottleneck. Late interaction keeps **one vector per token** and matches at query time.

### 5.1 ColBERT and ColBERTv2

Papers: **Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT," SIGIR 2020.** **Santhanam, Khattab, Saad-Falcon, Potts, Zaharia, "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction," NAACL 2022.**

**Mechanic — MaxSim.** Each query token q_i has an embedding; each doc token d_j has one. The similarity of doc to query is

$$\text{score}(q,d) = \sum_{i \in q} \max_{j \in d} \langle q_i, d_j \rangle$$

Each query token "picks its best matching doc token." Captures phrase-level alignment that single-vector cosine cannot. Trained with in-batch negatives + denoised supervision (v2).

On BEIR (out-of-domain), ColBERTv2 averages **nDCG@10 ≈ 0.50** vs DPR-style single-vector **≈ 0.42** — roughly +8 points zero-shot. On MS MARCO dev MRR@10 ColBERTv2 hits ~39.7.

**Storage problem.** Naive ColBERT v1 stored a 128-d vector per token — 100 GB for MS MARCO 8.8M passages. ColBERTv2 introduces:
- **Centroid + residual compression:** k-means cluster all token vectors; per token, store cluster ID + a small (2-bit) residual. ~10× storage shrink with negligible recall loss.
- **PLAID** (Santhanam 2022) is the optimized engine: candidate generation via centroid scoring, then refined MaxSim only on candidates. Brings latency to ~50–100 ms per query at MS MARCO scale.

**Practical entry point.** RAGatouille (Ben Clavié, AnswerAI). One line to index, one line to retrieve. If you've been on bi-encoders + cross-encoder rerank and want to *try* late interaction without a research project, this is the path.

### 5.2 bge-m3 multi-vector mode

BAAI's bge-m3 (Chen et al., 2024, arXiv:2402.03216) exposes three retrieval signals from one model: dense (single vec), sparse (BM25-like), and multi-vector (ColBERT-style). Hybrid all three with RRF. State of the art on MIRACL multilingual benchmarks. The one-model-three-signals story makes infra dramatically simpler.

### 5.3 ColPali / ColQwen — late interaction on visual documents

Paper: **Faysse, Sibille, Wu, Omrani, Viaud, Hudelot, Colombo, "ColPali: Efficient Document Retrieval with Vision Language Models," ICLR 2025, arXiv:2407.01449.**

**Mechanic.** Skip text extraction entirely. Render each PDF page as an image, feed through a vision-language model (PaliGemma, then Qwen2-VL for ColQwen2), late-interact with query tokens via MaxSim on patch embeddings.

On the ViDoRe benchmark, ColPali at **nDCG@5 = 81.3** beats Unstructured+BM25 (~65) and Unstructured+BGE (~67) by 15–18 points, and is **dramatically simpler** — no OCR, no layout parsing, no table extraction.

**When to reach for it.** Corpora that are fundamentally visual: financial reports with charts, scientific PDFs with figures, slide decks, forms. **When not.** Plain prose where text extraction is already lossless. Latency: ingestion is GPU-bound and 5–10× slower than text-pipeline; query latency similar to ColBERT.

---

## 6. Knowledge-graph RAG (deep)

Vector RAG fails at **global** questions — "what are the recurring themes," "summarize this customer's history," "what entities connect these reports." No single chunk contains the answer; the answer is a structural property of the corpus.

### 6.1 Microsoft GraphRAG

Paper: **Edge, Trinh, Cheng, Bradley, Chao, Mody, Truitt, Larson, "From Local to Global: A Graph RAG Approach to Query-Focused Summarization," arXiv:2404.16130 (Microsoft Research, 2024).**

**Pipeline.**
1. **Chunk** the corpus.
2. **Extract entities and relations** per chunk with an LLM (typed nodes/edges + claims).
3. **Build a graph**, dedup/merge entities across chunks.
4. **Detect communities** via **Leiden** (hierarchical, multi-resolution).
5. **Summarize each community** with the LLM (one LLM call per community per level).
6. At query time:
   - **Local mode:** entity-anchored — find entities matching the query, traverse N-hop neighborhood, feed entity descriptions + relevant claims.
   - **Global mode:** map-reduce over **community summaries** at chosen level — answer per community, then reduce into a final answer.

Reported wins on dataset-spanning queries (podcast transcripts, news): GraphRAG-global produces answers humans prefer 70–80% of the time vs vector-RAG, especially on "what are the themes" and "compare across" prompts.

**Cost.** Ingestion is *expensive* — one LLM call per chunk for extraction, plus per-community summaries. The 1M-token podcast corpus in the paper cost on the order of single-digit dollars to GraphRAG-ize on a small model and tens on a large one. Vector RAG over the same corpus costs cents.

### 6.2 Property-graph RAG (Neo4j + LLM)

Less ambitious: extract entities/relations into Neo4j, retrieve subgraphs via Cypher generated by the LLM (or via embedding the entity nodes). LlamaIndex `PropertyGraphIndex`. Good fit when you already have a graph (knowledge management, biomedical, fraud).

### 6.3 When KG-RAG genuinely wins

- "What are the themes across all these reports?" (global sensemaking)
- "Summarize everything about Customer X" (entity-centric across many docs)
- Compliance / audit ("show me every mention of vendor Y and the surrounding context")
- Multi-hop where edges are explicit ("which authors collaborated with collaborators of X")

### 6.4 When it's overkill

- FAQ / point lookup
- Single-doc QA
- High doc churn (graphs get stale; you pay re-extraction continuously)

Ongoing maintenance cost (entity resolution drift, schema evolution) is the unsung killer — budget for it.

---

## 7. Self-correcting / adaptive RAG (deep)

Static RAG retrieves once and prays. These techniques inject *evaluation* and *correction* loops.

### 7.1 Self-RAG

Paper: **Asai, Wu, Wang, Sil, Hajishirzi, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," ICLR 2024, arXiv:2310.11511 (UW + AI2).**

**Mechanic.** Fine-tune the generator to emit **reflection tokens** inline:
- `[Retrieve]` / `[No Retrieve]` — should I retrieve now?
- `[IsRel]` / `[IsNotRel]` — is the retrieved passage relevant?
- `[IsSup]` / `[PartialSup]` / `[NoSup]` — does the passage support my answer?
- `[IsUse]` 1–5 — is the answer useful?

Training data: distilled from GPT-4 critic labels over Llama-2 outputs (~150k instances). The fine-tuned model self-decides when to retrieve, branches on relevance, and self-grades the output. Beat ChatGPT and Llama2-chat on PopQA, TriviaQA, ALCE-ASQA.

**Cost to adopt.** **Requires fine-tuning.** This is the chief blocker. If you can't fine-tune your generator, skip to CRAG.

### 7.2 CRAG — Corrective RAG

Paper: **Yan, Gu, Zhu, Lapata, "Corrective Retrieval Augmented Generation," arXiv:2401.15884 (2024).**

**Mechanic.** A lightweight **retrieval evaluator** (small classifier — they use a T5-large) scores each retrieved passage as **Correct / Ambiguous / Incorrect**. Then:
- **Correct** → use as-is (with knowledge refinement: decompose into strips, re-filter).
- **Incorrect** → discard, do **web search fallback** to fetch external evidence.
- **Ambiguous** → both.

No generator fine-tuning required. The evaluator is small and cheap. On PopQA, Bio, PubHealth, CRAG over Self-RAG and standard RAG by 3–10 points.

**This is the practical choice** when you control the pipeline but not the generator weights. Pair the "incorrect" branch with whichever fallback fits — web search, a stronger retriever, query rewriting.

### 7.3 Adaptive-RAG

Paper: **Jeong, Baek, Cho, Hwang, Park, "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity," NAACL 2024, arXiv:2403.14403.**

A small **complexity classifier** (T5-based) routes each query to one of three pipelines:
- **A: No retrieval** (model knows it)
- **B: Single-step RAG** (one retrieve + generate)
- **C: Multi-hop iterative RAG** (interleaved retrieve + reason)

Trained on silver labels from running each pipeline against benchmarks and seeing which succeeds. Matches multi-hop-RAG accuracy at a fraction of the latency on easy queries — which are usually the majority.

### 7.4 Iterative retrieval — FLARE, ITER-RETGEN

**FLARE** (Jiang et al., EMNLP 2023, arXiv:2305.06983): generate a sentence; if the next sentence has low-confidence tokens, pause, retrieve using the predicted-but-unsure sentence as the query, then continue. Forward-Looking Active Retrieval.

**ITER-RETGEN** (Shao et al., EMNLP 2023, arXiv:2305.15294): generate a draft → use it as a richer query → retrieve → generate again → repeat 2–3 times. Effectively HyDE-in-a-loop.

Both turn one-shot RAG into closed-loop retrieval; both add 2–5× latency. Use when single-shot consistently misses and a reranker can't save you.

### 7.5 LongRAG

Lyu et al., 2024 (arXiv:2406.15319). Chunk into ~4k-token "long retrieval units" instead of 256-token chunks, retrieve fewer of them, feed to a long-context reader. Beats classic RAG on NQ and HotpotQA when paired with GPT-4 / Claude. Anticipates the RAG-vs-long-context blur (§9).

---

## 8. Agentic RAG (deep)

Static RAG fixes the *shape* of retrieval at design time. Agentic RAG lets the LLM **decide** when to retrieve, what to retrieve, and when to stop. Retrieval becomes a tool, not a stage.

### 8.1 Retrieval as a tool

OpenAI function calling, Anthropic tool use, LangGraph nodes, LlamaIndex `FunctionTool`, etc. The generator sees `search_docs(query: str) -> list[Passage]` and chooses to call it. The single most important consequence: the LLM can issue multiple, refined searches per turn.

### 8.2 ReAct vs plan-execute vs reflection

- **ReAct** (Yao et al., ICLR 2023, arXiv:2210.03629): interleaved Thought → Action → Observation. Conversational, recoverable, slow.
- **Plan-Execute** (Wang et al., 2023, arXiv:2305.04091): plan all steps up front, then execute. Lower latency; brittle when plans are wrong.
- **Reflection / Reflexion** (Shinn et al., NeurIPS 2023, arXiv:2303.11366): execute → self-critique → retry. Adds another pass; pays off on tasks with verifiable failure signals.

For RAG, ReAct is the practical default; plan-execute wins for known decompositions ("compare X and Y" → known plan); reflection adds value when you have a cheap verifier (test passing, citation present, schema check).

### 8.3 Multi-tool retrieval

Expose multiple retrievers as separate tools — `search_code`, `search_docs`, `search_tickets`, `search_slack`. The agent routes by reading tool descriptions. This subsumes §1.6 (query routing): the routing decision is made by the same LLM that synthesizes, with the full conversation in scope.

### 8.4 Sub-agent decomposition

For "compare A and B," spawn one sub-agent per entity, each with its own context, run in parallel, then a parent agent synthesizes. Anthropic's "How we built our multi-agent research system" (anthropic.com, 2025) reports ~90% reduction in wall-clock latency on research-style queries vs serial agent. Cost goes up linearly with sub-agents.

### 8.5 Cost & latency control

- `max_turns` cap (5–8 typical)
- Tool-call budget per turn
- Per-tool result truncation
- Token budget across the loop, enforced as a hard cutoff

Agentic RAG quietly burns 10–100× the tokens of static RAG. Cache aggressively (Anthropic prompt caching gives 90% input discount on cached prefixes — critical here).

### 8.6 When agentic beats static

- Multi-hop with unknown hop count
- Tools that require argument refinement (SQL gen, code search)
- Mixed sources, where the right one isn't known upfront

### 8.7 When it doesn't

- Latency-tight chat (you'll feel every extra second)
- Highly predictable queries (FAQ)
- Cost-sensitive paths (each turn is another full prompt)

---

## 9. Long-context vs RAG (deep)

The "throw it all in the context" alternative has gotten serious. Gemini 1.5 / 2.0 ships 1M–2M tokens; Claude offers 200k and a 1M tier; GPT-4.1 is 1M; open models (Llama 3.1, Qwen 2.5) hit 128k+.

### 9.1 The actual numbers

- **Single-needle retrieval** (e.g., needle-in-haystack, NIAH): frontier models score >95% at 128k, often into 1M. This was the early hype.
- **Multi-needle** (RULER, Hsieh et al., 2024, arXiv:2404.06654): degrades sharply. At 128k, even top models often drop into the 60–80% range; performance on multi-key/multi-value retrieval is significantly worse than single needle.
- **Reasoning over haystack** (BABILong, Kuratov et al., 2024, arXiv:2406.10149): 5-fact reasoning at 64k crushes most models to <50%.

So: long context **retrieves** decently but **reasons over** poorly. RAG retrieves + a smaller working context for reasoning often wins on quality and *always* on cost.

### 9.2 Prompt caching economics

- **Anthropic prompt caching**: cached input tokens cost ~10% of normal input (effectively 90% discount); 5-minute default TTL (1-hour available). Write cost is ~1.25× a normal input token. Crossover at ~2 reads.
- **OpenAI**: automatic prompt caching gives 50% discount on cached input tokens for eligible models.
- **Gemini context caching**: explicit cached content priced separately; cost per cached token + per-hour storage.

When "load once, query many" beats RAG: stable corpus, repeated reads per cached load, query patterns that span the whole corpus. Classic example: codebase Q&A in a single session — load the repo, ask 20 questions, never re-retrieve.

### 9.3 Hybrid pattern

RAG into 50–200k tokens, then let long-context reason. You get RAG's freshness and ACL, and long-context's reasoning. Most production high-stakes RAG systems are now this shape (e.g., Anthropic's own internal demos).

### 9.4 When RAG still wins clearly

- **Freshness** — updating a cache or re-prompting on every change is wasteful
- **Citations** — RAG knows *which chunk* answered; long-context guesses
- **ACL / per-tenant isolation** — you cannot share a cache across tenants; you can per-tenant-filter a vector store
- **Deterministic cost** — context size scales answer cost; RAG retrieval cost is roughly constant
- **"Lost in the middle"** (Liu et al., TACL 2024, arXiv:2307.03172) — even at 1M, mid-context recall degrades; the U-shape stays real

---

## 10. Contextual Retrieval — Anthropic, September 2024 (deep)

Source: **Anthropic, "Introducing Contextual Retrieval," anthropic.com/news/contextual-retrieval, Sept 2024.**

**The insight.** Standard chunking strips a chunk of its document context. The chunk "The company's revenue grew 3% over the previous quarter." is uninterpretable in isolation — *which* company, *which* quarter? Embeddings of such chunks match poorly because the matching signal isn't *in* the text.

**The fix.** Before embedding/indexing, prepend a 50–100-token **chunk-specific context** generated by an LLM describing what this chunk is and where it sits. Now embed the *contextualized* chunk.

### 10.1 The prompt

```
<document>
{whole_document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within
the overall document for the purposes of improving search retrieval
of the chunk. Answer only with the succinct context and nothing else.
```

Output:

```
This chunk is from ACME Corp's Q2 2023 SEC filing, in the section
on financial performance. It compares Q2 2023 revenue to Q1 2023.
```

Prepended to the chunk, both contextual chunk *and* its BM25 form are indexed.

### 10.2 Why it's affordable: prompt caching

The whole document gets re-sent for every chunk in that document. Naïvely that's quadratic. With prompt caching, the document is cached on the first chunk and reused for the rest at ~10% cost. Anthropic reports the full pipeline costs **~$1.02 per million document tokens** processed.

### 10.3 Reported gains

Anthropic's benchmark (retrieval-failure rate on a curated eval set):

- Baseline (embedding only): **5.7% failure rate**
- + Contextual Embeddings: **3.7%** — **−35% retrieval failures**
- + Contextual BM25 (hybrid): **2.9%** — **−49% vs baseline**
- + Reranker (Cohere or similar) on top: **1.9%** — **−67% vs baseline**

Numbers are aggregated across nine datasets the team used. The marginal gain from each stage is roughly multiplicative — each addresses a different failure mode.

### 10.4 Why this is the new baseline

It's nearly free at inference (the context is precomputed at ingest), composes with everything else (hybrid, rerank, parent-doc), and the gains are large and consistent. If you're on Claude (or any LLM-as-ingest-helper), there's almost no reason not to do this. Contextual Retrieval is now the assumed L1.5 baseline above which advanced techniques should be measured.

### 10.5 Variants

- **Contextual-only-for-embed:** prepend context only to the embedded text; serve the original chunk to the LLM. Less LLM token cost at serve time.
- **Contextual-in-BM25-but-not-embed:** BM25 benefits hugely from entity strings ("ACME Corp Q2 2023") even when embeddings already capture semantics.
- **Don't show context to the LLM:** the context is a *retrieval aid*; showing it can mildly hurt generation quality by repeating content. Common pattern: store both, return clean chunk to generator.

### 10.6 Pipeline shape

```
Ingest:
  doc -> chunks
  for chunk in chunks:
      ctx = llm(prompt with cached doc, chunk)        # ~$1/M tokens
      embedded_text = ctx + "\n" + chunk
      vector_index.add(embed(embedded_text), payload={"raw": chunk, "ctx": ctx})
      bm25_index.add(embedded_text, doc_id)

Query:
  dense_hits  = vector_index.search(embed(q), k=150)
  sparse_hits = bm25_index.search(q, k=150)
  fused = rrf(dense_hits, sparse_hits, k=60)
  top_n = reranker.rerank(q, fused, top_k=20)
  return [hit.payload["raw"] for hit in top_n]
```

This is the recipe. It's roughly the new state of the art for "RAG you can build in two weeks."

---

## 11. Other emerging techniques worth knowing

### 11.1 HyPE — Hypothetical Question Embeddings

Mirror image of HyDE. At *ingest*, generate 3–8 hypothetical questions a chunk answers; embed *those*, not the chunk. At retrieval, the user query matches question embeddings (question-to-question). Bridges the same asymmetry HyDE addresses but pays the LLM cost at ingest, not query — better latency, worse adaptability. Recent paper: Vatsal et al. 2025, arXiv:2502.18452.

### 11.2 LangChain MultiVectorRetriever patterns

Same chunk, multiple representations: raw, summary, hypothetical questions, extracted facts. Embed all; retrieve any; return raw. Conceptually unifies §3.5 + §11.1 + chunked-and-summary patterns.

### 11.3 Ensemble retrieval + LTR

Beyond RRF: train a LightGBM or XGBoost ranker on (query, doc) feature vectors (per-retriever scores, BM25 stats, query length, embedding similarity, click-through if available). Squeezes 1–3 nDCG points if you have ≥10k labeled query-doc pairs. The Anthropic Contextual Retrieval blog implicitly uses something similar at the rerank stage.

### 11.4 Text-to-SQL and table-aware retrieval

For structured data, embedding rows is the wrong abstraction. Patterns:
- **Text-to-SQL** with schema-in-prompt (NSText2SQL, DAIL-SQL, Spider-trained models).
- **Table retrieval**: embed table schemas + column descriptions + sample rows; retrieve the *table*, then text-to-SQL within it.
- **Hybrid**: vector RAG over docs + SQL over tables, agent routes (see §8).

OpenAI's "Code Interpreter / Advanced Data Analysis" pattern is the productized form.

### 11.5 Multimodal RAG

- **ColPali / ColQwen** (§5.3) — late interaction over page images.
- **CLIP / SigLIP** — text-image joint embedding for image retrieval.
- **Whisper-embed / CLAP** — audio retrieval.
- **Video**: frame sampling + CLIP-style embeddings, or video-native (VideoCoCa, InternVideo).

The trend is unmistakable: text-only RAG is becoming a subset of a richer multimodal retrieval surface. Expect ColPali-style late interaction to spread to audio/video within 18 months.

---

## When to reach for what — decision matrix

| Symptom or context | Reach for |
|---|---|
| Short queries, long-form documents, weak retrieval baseline | **HyDE** (§1.1) |
| Ambiguous or under-specified queries | **Multi-query + RRF** = RAG-Fusion (§1.2/§1.7) |
| Multi-hop or comparison questions | **Step-back** (§1.3) or **decomposition** (§1.4) |
| Chat with follow-ups | **Standalone rewrite** (§1.5) — non-negotiable |
| Multiple indices / data sources | **Routing** (§1.6) or **multi-tool agent** (§8.3) |
| Rare tokens, codes, identifiers in queries | **BM25 + dense, RRF** (§2.1) |
| Chunks lose context once isolated | **Parent-document** (§3.1) or **Contextual Retrieval** (§10) |
| Definitions / single-fact lookups | **Sentence-window** (§3.2) |
| Long narrative docs, "themes" questions over a doc | **RAPTOR** (§3.4) |
| Per-tenant / time-bound / typed filters | **Metadata filters + self-querying** (§4) |
| Out-of-domain dense retrieval / phrase-sensitive | **ColBERTv2 via RAGatouille** (§5.1) |
| PDFs, slide decks, figures, forms | **ColPali / ColQwen** (§5.3) |
| "What are the themes across N reports" / global sensemaking | **GraphRAG** (§6.1) |
| Entity-centric corpora | **Property-graph RAG** (§6.2) |
| You can fine-tune the generator and want self-grading | **Self-RAG** (§7.1) |
| You cannot fine-tune but want correction | **CRAG** (§7.2) |
| Mixed easy/hard queries, want latency control | **Adaptive-RAG** (§7.3) |
| Generation needs more facts mid-answer | **FLARE** or **ITER-RETGEN** (§7.4) |
| Multi-hop with unknown hop count | **Agentic RAG / ReAct** (§8) |
| "Compare A and B" style | **Sub-agent decomposition** (§8.4) |
| Stable corpus, repeated queries per session | **Long-context + prompt caching** (§9.2) |
| Per-tenant ACL, freshness, citations required | **Stay on RAG** (§9.4) |
| You're on Claude and want one big win | **Contextual Retrieval** (§10) |
| Ingest-side latency is fine, query-side latency tight | **HyPE** (§11.1) |
| Mostly structured / tabular data | **Text-to-SQL + table retrieval** (§11.4) |
| Multimodal corpus | **CLIP/SigLIP, ColPali, Whisper-embed** (§11.5) |

A defensible 2026 baseline stack, in order of marginal lift: hybrid dense + BM25 → Contextual Retrieval at ingest → reranker (Cohere Rerank 3 / bge-reranker-v2-m3 / Voyage Rerank) → parent-document return → metadata filters where applicable → multi-query (RAG-Fusion) on ambiguous-query paths → CRAG correction loop for high-stakes answers. That's the spine. Everything else is targeted, measured against an eval set, and adopted only when symptoms match. Build the eval set first; the rest is mechanical.
