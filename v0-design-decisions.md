# AI Engineering Tutor - V0 design

## Problem

Aspiring AI Engineers preparing for interviews need answers grounded in a specific, trustworthy corpus, with sources they can go read.

A general chatbot answers from memory, cannot cite, and cannot say "that is not in here".

**Input:** a natural language question.
**Output:** an answer grounded in the corpus, with numbered citations, or an explicit refusal.

## Scope

**In V0:** ingestion pipeline, dense retrieval, generation with citations, abstention, FastAPI service, minimal web UI, two-tier eval suite, tracing, Docker Compose.

**Deliberately out of V0:** reranking, hybrid search, query rewriting, agents, streaming, caching, auth, chat history, web fetching at query time.

Reranking and hybrid search are V1, and each ships with a measured before/after against the V0 numbers.

## Architecture

The system is two programs sharing one datastore.

```
OFFLINE (ingest.py)
  sources.yml -> fetch -> data/raw/ -> parse -> chunk -> embed -> [ CHROMA ]
                                                                      |
ONLINE (api.py -> answer.py)                                          v
  query -> embed -> top-k search -> assemble -> prompt -> LLM -> cite -> answer
```

The embedding model is the only code shared by both paths, and it must be identical on both sides or the vectors are not comparable.

The index is the only state shared by both paths.

Everything upstream of the index is expensive to change because it forces a full rebuild.
Everything downstream is free to change.

## Chunk record

Every field exists because a named consumer requires it.

| Field | Required by |
|---|---|
| `id` = `f"{doc_id}:{i}"` | determinism, debugging |
| `text` | context assembly |
| `doc_id` | citations, eval labels, re-ingestion |
| `title` | context assembly, citations |
| `url` | citations |
| `section` | context assembly, citations, eval labels |
| `doc_hash` | re-ingestion |

Ids are deterministic so the pipeline is a pure function of its inputs.

Nothing query-dependent is ever stored on a chunk.

## Modules

```
src/config.py      settings: model names, k, thresholds
src/fetch.py       sources.yml -> data/raw/
src/parse.py       raw -> clean text + document metadata
src/chunking.py    text -> chunk records
src/embedding.py   text -> vectors (one embedder, used by both paths)
src/store.py       thin vector store wrapper
src/retrieval.py   query -> ranked chunks
src/assemble.py    chunks -> numbered context block
src/prompts.py     system prompt, abstention wording
src/llm.py         provider client, timeouts, retries
src/answer.py      orchestration of the online path
src/api.py         FastAPI, HTTP only, no RAG logic
ingest.py          CLI, orchestration of the offline path
data/sources.yml   the corpus manifest
```

Boundaries were chosen so that each likely change opens exactly one file.

`api.py` contains no RAG logic, so the eval suite runs the same pipeline without touching HTTP.

## Pipeline settings

- Chunking: recursive, 400-500 tokens, 20% overlap.
- Embedding: `BAAI/bge-small-en-v1.5`, 384 dims, 512 max sequence length, query prefix applied at retrieval time only.
- Store: Chroma, persistent local mode. Model name and version written to collection metadata at ingest and asserted at query time.
- Retrieval: top k = 5.
- Generation: mid-tier open model via OpenRouter, model name in config.

Context is assembled as `[n] {title} > {section} ({url})` followed by the chunk text.

## Failure handling

| Failure | Detect | Handle |
|---|---|---|
| Question outside the corpus | top result below similarity threshold | skip the LLM, return an explicit refusal |
| Correct chunk ranked below k | not detectable at runtime | only visible in eval, this is why the harness exists |
| Provider 429 / 5xx / timeout | HTTP status, explicit client timeout | bounded retries with exponential backoff and jitter, then a fast error |
| Answer not grounded in context | citation indices validated against the assembled context | drop invalid citations, measure faithfulness offline |
| Stale index after a model swap | collection metadata mismatch | fail loudly at startup |

Prompt instructions are mitigation, not enforcement.
Enforcement is the similarity threshold and the citation validation.
Score calibration: In-corpus tops at 0.81-0.83, out-of-corpus at 0.48-0.51. -> Similarity threshold = 0.6

## Evaluation

**Golden set:** about 20 questions, hand written from the asker's point of view, never copied from chunk text.
Three types: conceptual, document specific, out of corpus (3-4).
Ground truth is labelled as `(doc_id, section)`, never as a chunk id, so labels survive rechunking.

**Tier 1, every commit, in CI.** Deterministic, no LLM.
`hit-rate@5` and MRR, written to a committed per-question snapshot file that CI diffs.
A coarse absolute floor guards against catastrophic regressions.
CI rebuilds the index from `data/raw/` on every run so it can never drift from the code.

**Tier 2, on demand and before each version.** LLM judged, so slow and non-deterministic.
Faithfulness, answer relevancy, abstention rate on the out-of-corpus questions.
Judge model is different from the generator model, at temperature 0.

**Baseline:** the same generator with no retrieved context.
Reported per question type, never as a single aggregate, because on out-of-corpus questions the baseline scores well by answering when it should refuse.

## Stack and trade-offs

| Choice | Why | Cost accepted |
|---|---|---|
| LangChain components, not chains | keeps retrieve / assemble / generate visible and separately testable | more glue code than a prebuilt chain |
| `bge-small-en-v1.5` over MiniLM | 512 token sequence length fits the chunk size, better retrieval, same dims | needs the query prefix, easy to get wrong |
| Chroma, persistent local | no server to run at this scale | swap needed if the corpus grows a lot |
| Snapshotted corpus | eval numbers are only comparable against a frozen corpus | manual refetch when sources change |
| Mid-tier generator | makes the RAG vs baseline delta visible rather than hidden by strong priors | lower raw answer quality |
| Langfuse | self-hostable, unlike LangSmith | more services to run if self-hosted |

Honest limitation: the corpus is public writing the generator was likely trained on.
RAG's value here is attribution, verifiability and currency, not novel knowledge.

Known gap: top-k retrieval samples, it does not scan, so aggregation questions ("what is the most common", "list all") cannot be answered correctly.
V0 does not detect them.

## Done when

- `python ingest.py` builds the index from `sources.yml` reproducibly.
- `POST /chat` returns a grounded answer with citations, or a refusal.
- The UI talks to the API over HTTP, not by importing the pipeline.
- Tier 1 evals run in GitHub Actions on every push.
- Tier 2 and the baseline comparison have been run once and the numbers are written down.
- `docker compose up` starts the whole thing on a clean machine.

## Known gaps left for later

- A document that shrinks leaves orphaned chunks in the index. -> Design robust re-ingestion
