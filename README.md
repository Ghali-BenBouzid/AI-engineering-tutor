# AI Engineering Tutor

Answers AI engineering and system design questions from a fixed corpus of reference material, with a citation for every claim, and refuses when the corpus does not cover the question.
Built for people preparing for AI engineering interviews who need an answer they can trace to a paragraph in a named document, not a plausible paragraph from a general model.

## Sample output

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

## Run it

```bash
uv sync
cp .env.example .env             # add an OpenRouter API key

uv run python ingest.py          # build the index: fetch, parse, chunk, embed
uv run python -m src.answer      # ask the two sample questions above
uv run pytest                    # 68 tests, ~60s
uv run pytest -m "not slow"      # 54 tests, no model load, ~12s
```

There is no eval command yet.
The harness is designed and not built, and the section below says what it will measure.

## Problem, user, input, output

**Problem.** Interview preparation material for AI engineering is scattered across long documents and blog posts.
General chatbots answer these questions fluently from memory, but cannot tell you which source says it, cannot link to the paragraph, and will not tell you when they are guessing.

**User.** Someone preparing for AI engineering interviews, working through a specific reading list.

**Input.** A natural language question, plus a corpus of 58 documents pinned at fixed versions.

**Output.** An answer grounded in retrieved chunks with numbered citations back to source URLs, or an explicit refusal.
The API returns the answer text, the source list, an abstention flag, any invalid citations, and the top retrieval distance.

## Architecture

Two programs sharing one datastore.

```
OFFLINE  (ingest.py)
  sources.yml -> fetch -> data/raw/ -> parse -> chunk -> embed -> [ CHROMA ]
                                                                      |
ONLINE   (src/answer.py)                                              v
  query -> embed -> top-k -> guardrail -> assemble -> prompt -> LLM -> cite
```

The embedding model is the only code on both paths, and it must be identical on both sides or the vectors are not comparable.
The index is the only shared state.
Everything upstream of the index forces a full rebuild when it changes; everything downstream is free to change.

| Setting | Value |
|---|---|
| Chunking | markdown headings, then recursive split, 450 tokens, 90 overlap |
| Embedding | `BAAI/bge-small-en-v1.5`, 384 dims, 512 max sequence length |
| Vector store | Chroma, persistent local, cosine |
| Retrieval | top k = 5, abstain above 0.4 cosine distance |
| Generation | mid-tier open model via OpenRouter, temperature 0 |

## Important files

| File | What it holds |
|---|---|
| [`v0-spec.md`](v0-spec.md) | scope, users, what is deliberately excluded from V0 |
| [`v0-design-decisions.md`](v0-design-decisions.md) | architecture, chunk record, evaluation design, trade-offs |
| [`data/sources.yml`](data/sources.yml) | the corpus manifest |
| [`data/sources.lock.yml`](data/sources.lock.yml) | pinned commit and content hash per source |
| [`src/answer.py`](src/answer.py) | the online path, end to end |
| [`src/fetch_and_parse.py`](src/fetch_and_parse.py) | fetch, parse, and the metadata contract |
| [`src/chunk.py`](src/chunk.py) | heading-aware then token-aware splitting |
| [`tests/test_ingest.py`](tests/test_ingest.py) | end-to-end ingestion against a real git repo and a local HTTP server |

## Corpus

Two sources, pinned in `data/sources.lock.yml`: the [AI Engineering Field Guide](https://github.com/alexeygrigorev/ai-engineering-field-guide) at commit `7eadc13`, and [Building A Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html) by content hash.
58 documents, 1023 chunks.

The corpus is snapshotted rather than fetched live, because evaluation numbers are only comparable against a frozen corpus.
If a source changes underneath you, a hit-rate movement no longer tells you whether the retriever improved or the corpus moved.
The lockfile stamps a new `fetched_at` only when the hash or commit changes, so a diff in that file always means the corpus actually moved.

## Tests

68 tests, no mocks.
A real local git repository, a real HTTP server on localhost, the real tokenizer, real Chroma.
Every path constant is a relative `Path`, so `monkeypatch.chdir` into a temporary directory redirects the whole pipeline without patching internals.

Covered: the metadata contract on both source types, lockfile stability across runs, chunk sizes measured in tokens rather than characters, chunk ids deterministic and unique, optional metadata absent rather than `None`, embeddings unit length on both paths, the query prefix applied to queries only, re-indexing upserting rather than duplicating, and excludes surviving the whole pipeline.

Three of these were written after the bug they now catch.

## Evaluation

Not built. This is the next milestone, and the design is in [`v0-design-decisions.md`](v0-design-decisions.md).

The plan splits by determinism rather than by metric family:

**Deterministic, every commit, in CI.** `hit-rate@5` and MRR against a golden set of ~20 hand-written questions labelled `(doc_id, section)`, plus abstention on out-of-corpus questions, which needs no model because the guardrail fires before the LLM is called.
These are set arithmetic on a frozen corpus, so they are free, instant, and identical run to run.

**LLM-judged, on demand.** Faithfulness and answer relevancy through a judge model that is stronger than and different from the generator.
These are slow, paid, and non-deterministic, so they produce a scored report compared against the previous run rather than a pass/fail gate.

**Baseline.** The same generator with no retrieved context, reported per question type.
On out-of-corpus questions the baseline is expected to beat the system, because the system correctly refuses while the baseline answers from memory.
Aggregating the question types into one number would make correct behaviour look like a loss.

## What failed or changed

Five bugs that shaped the design. All of them produced no error.

**MiniLM truncates at 256 tokens.** The original plan paired `all-MiniLM-L6-v2` with 400-500 token chunks. Anything past 256 would have been silently dropped from the embedding while still being fed to the model as context. Switched to `bge-small-en-v1.5` for its 512-token sequence length.

**Chunk size was counted in characters.** `RecursiveCharacterTextSplitter` counts characters by default, so `chunk_size=450` meant roughly a quarter of the intended size. Fixed by constructing it `from_huggingface_tokenizer`. A test now asserts the unit.

**A source started serving zstd.** `trafilatura.fetch_url` advertised zstd support and returned the compressed bytes undecoded, which overwrote a good snapshot with binary garbage and produced an empty extraction. Fetching moved to `httpx`; fetching is HTTP's job, extraction is trafilatura's.

**A timeout in the wrong unit.** `request_timeout` in `langchain-openrouter` is milliseconds, not seconds. `request_timeout=30` meant 30ms, so every request died instantly and two stacked retry layers backed off, which looked like an indefinite hang on every model.

**The generator was a reasoning model.** `deepseek-v4-flash` spent 18.9 seconds and 171 reasoning tokens on "introduce yourself". Wrong for a system whose evaluation runs the same 20 questions repeatedly, and it would have compressed the baseline comparison by reasoning its way to good answers without the corpus.

Four of the five are the same bug shape: a number that is correct in the wrong unit, or a value that is correct for a different component. None of them raised.

## Limitations

**Top-k retrieval samples, it does not scan.**
Questions needing completeness ("what is the most common approach to X", "compare every method in the corpus") get five chunks and a confident synthesis over them.
Answering those properly needs a map-reduce path or hierarchical summaries. Neither is built, and the system does not currently detect such questions.

**Re-ingestion is full-rebuild only.**
Indexing upserts, so a changed document overwrites its chunks, but a document that shrinks leaves orphaned chunks in the index.
Delete `data/chroma/` before re-ingesting.

**The corpus is public writing the generator was probably trained on.**
Retrieval's value here is attribution, verifiability and currency, not novel knowledge.
The baseline comparison is expected to show a smaller gap on conceptual questions than on document-specific ones.

**The abstention threshold rests on four measurements.**
In-corpus questions score around 0.19 cosine distance, out-of-corpus around 0.51, and 0.4 splits them.
That is calibration, not validation. The golden set is what will settle it.

## Status

- [x] Ingestion: fetch, parse, chunk, embed, index, with a corpus lockfile
- [x] Retrieval with an abstention guardrail and an index/model mismatch check
- [x] Generation with numbered citations and citation validation
- [x] 68 tests covering the pipeline and its known failure modes
- [ ] Golden dataset of ~20 labelled questions
- [ ] Deterministic retrieval metrics in CI
- [ ] LLM-judged metrics and a no-context baseline
- [ ] Web interface
- [ ] Tracing
