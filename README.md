# AI Engineering Tutor

A RAG chatbot that answers AI engineering and AI system design questions from a fixed corpus of reference material, cites its sources, and says so when the corpus does not contain the answer.

It exists because I am preparing for AI engineering interviews and wanted answers I could trace back to a specific paragraph of a specific document, rather than a plausible paragraph from a general model.

## What it does

```
Q: What is query rewriting and why does it help retrieval?
top_distance=0.1904  abstained=False  invalid_citations=[]

Query rewriting is the process of rephrasing a user's query to increase the
likelihood of fetching the right information [1]. It helps retrieval by making
the query more specific and less ambiguous [1]. For example, "How about Emily
Doe?" can be rewritten to "When was the last time Emily Doe bought something
from us?" to reflect what the user is actually asking [1].

  [1] Building A Generative AI Platform > Step 1. Enhance Context > Query rewriting
  [2] Building A Generative AI Platform > Step 1. Enhance Context > RAGs
```

```
Q: What is the best pizza dough hydration?
top_distance=0.5151  abstained=True

I cannot answer this question from the available sources.
```

The second case is the interesting one.
Vector search always returns k nearest neighbours, and nearest is not near: with no pizza content in the corpus, the five closest chunks are still five confident-looking chunks about something else.
The system compares the top distance against a threshold and refuses before calling the model at all.

## Architecture

The system is two programs sharing one datastore.

```
OFFLINE  (ingest.py)
  sources.yml -> fetch -> data/raw/ -> parse -> chunk -> embed -> [ CHROMA ]
                                                                      |
ONLINE   (answer.py)                                                  v
  query -> embed -> top-k -> guardrail -> assemble -> prompt -> LLM -> cite
```

The embedding model is the only code shared by both paths, and it must be identical on both sides or the vectors are not comparable.
The index is the only state shared by both paths.

Everything upstream of the index is expensive to change, because it forces a full rebuild.
Everything downstream is free to change.

| Module | Responsibility |
|---|---|
| `src/fetch_and_parse.py` | `sources.yml` to raw snapshots to clean markdown with metadata |
| `src/chunk.py` | heading-aware split, then token-aware split |
| `src/embed.py` | the one embedder, used by both paths |
| `src/index.py` | Chroma collection, deterministic ids |
| `src/retrieval.py` | query to ranked chunks, index/model mismatch check |
| `src/assemble.py` | chunks to a numbered context block |
| `src/prompt.py` | system prompt, citation rules, refusal string |
| `src/llm.py` | OpenRouter client |
| `src/answer.py` | the online path, end to end |

Boundaries were chosen so that each likely change opens exactly one file.

## Corpus

Two sources, pinned in `data/sources.lock.yml`: the [AI Engineering Field Guide](https://github.com/alexeygrigorev/ai-engineering-field-guide) at a fixed commit, and [Building A Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html) by content hash.
58 documents, 1023 chunks.

The corpus is snapshotted rather than fetched live, because evaluation numbers are only comparable against a frozen corpus.
If a source changes underneath you, a hit-rate movement no longer tells you whether your retriever got better or the corpus moved.
The lockfile only stamps a new `fetched_at` when the content hash or commit actually changes, so a diff in that file always means the corpus moved.

## Running it

```bash
uv sync
cp .env.example .env          # add an OpenRouter key
uv run python ingest.py       # fetch, parse, chunk, embed, index
uv run python -m src.answer   # ask the two sample questions
uv run pytest                 # 68 tests
```

Ingestion is reproducible: raw sources are committed because they are not reproducible, and everything downstream is regenerated.
`data/processed/` and the Chroma index are derived and gitignored.

## Design decisions

| Choice | Why | Cost accepted |
|---|---|---|
| `bge-small-en-v1.5` over MiniLM | 512-token sequence length fits the chunk size; MiniLM truncates at 256 silently | needs a query-side instruction prefix, easy to omit |
| LangChain components, not chains | keeps retrieve / assemble / generate separately testable | more glue than a prebuilt chain |
| Chroma, persistent local mode | no server to run at this scale | swap needed if the corpus grows |
| Snapshotted corpus with a lockfile | eval numbers are only comparable against a frozen corpus | manual refetch when sources change |
| Ground truth labelled `(doc_id, section)` | survives a change of chunk size, unlike chunk ids | slightly looser matching |
| Deterministic chunk ids | re-indexing upserts instead of duplicating | ids shift when chunking changes, which is why nothing references them |
| Mid-tier generator | makes the RAG vs baseline difference visible rather than hidden by strong priors | lower raw answer quality |
| Non-reasoning generator | a reasoning model took 18.9s and 171 reasoning tokens on a trivial prompt | less capable on hard questions |

## Guardrails

| Failure | Detection | Handling |
|---|---|---|
| Question outside the corpus | top distance above threshold | refuse before calling the model |
| Answer cites a source that was not supplied | citation indices validated against the context | reported on the result object |
| Index built with a different embedding model | model name stamped in collection metadata | raises at query time |
| Correct chunk ranked below k | not detectable at runtime | only visible in evaluation |

The threshold was calibrated against the real corpus rather than guessed.
In-corpus questions score around 0.19 distance, out-of-corpus questions around 0.51.
That gap is smaller than intuition suggests: unrelated text does not score near zero, because embedding vectors cluster rather than spread over the whole space.

## Tests

68 tests, no mocks.
A real local git repository, a real HTTP server on localhost, the real tokenizer, real Chroma.
`monkeypatch.chdir` into a temporary directory is enough to redirect the pipeline, because every path constant is relative.

The suite covers the metadata contract on both source types, lockfile stability across runs, chunk sizes measured in tokens rather than characters, chunk ids being deterministic and unique, optional metadata being absent rather than `None`, embeddings being unit length on both the query and document paths, the query prefix being applied to queries only, and re-indexing upserting rather than duplicating.

Three of those exist because they caught real bugs.

## Known limitations

**Top-k retrieval samples, it does not scan.**
Questions that need completeness ("what is the most common approach to X", "compare every method in the corpus") get five chunks and a confident synthesis over them.
Answering those properly needs a map-reduce path or hierarchical summaries, neither of which is built.

**Re-ingestion is full-rebuild only.**
Indexing upserts, so a changed document overwrites its chunks, but a document that shrinks leaves orphaned chunks behind.
Delete `data/chroma/` before re-ingesting.

**The corpus is public writing the generator was probably trained on.**
Retrieval's value here is attribution, verifiability and currency, not novel knowledge.
The planned baseline comparison is expected to show a smaller gap on conceptual questions than on document-specific ones, and that is the honest finding rather than a flaw to hide.

## Status

The pipeline runs end to end. The evaluation harness is designed and not yet built.

- [x] Ingestion: fetch, parse, chunk, embed, index, with a corpus lockfile
- [x] Retrieval with an abstention guardrail and an index/model mismatch check
- [x] Generation with numbered citations and citation validation
- [x] 68 tests covering the pipeline and its known failure modes
- [ ] Golden dataset of ~20 labelled questions
- [ ] Deterministic retrieval metrics (`hit-rate@k`, MRR) in CI
- [ ] LLM-judged generation metrics and a no-context baseline
- [ ] Web interface
- [ ] Tracing

The evaluation design, including why the deterministic and LLM-judged metrics need different tooling, is written up in [`v0-design-decisions.md`](v0-design-decisions.md).
