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

The second case is the guardrail working, and it is not representative.
The evaluation shows it fires on questions far outside the corpus and misses ones that are merely in a different domain.
See "What failed or changed".

## Run it

```bash
uv sync
cp .env.example .env             # add an OpenRouter API key

uv run python ingest.py          # build the index: fetch, parse, chunk, embed
uv run python -m src.answer      # ask the two sample questions above
uv run pytest                    # 103 tests, ~60s
uv run pytest -m "not slow"      # 76 tests, no model load, ~13s

uv run python -m evals.tier1     # retrieval metrics, no API key needed
uv run python -m evals.pool      # rebuild the judgment pool
```

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
| [`evals/golden.yml`](evals/golden.yml) | the 22 evaluation questions |
| [`evals/pool.yml`](evals/pool.yml) | the judged candidates, which are the ground truth |
| [`evals/pool.py`](evals/pool.py) | pooling: dense plus BM25, fused, judgments carried forward |
| [`evals/tier1.py`](evals/tier1.py) | deterministic scoring and the snapshot gate |

## Corpus

Two sources, pinned in `data/sources.lock.yml`: the [AI Engineering Field Guide](https://github.com/alexeygrigorev/ai-engineering-field-guide) at commit `7eadc13`, and [Building A Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html) by content hash.
58 documents, 1023 chunks.

The corpus is snapshotted rather than fetched live, because evaluation numbers are only comparable against a frozen corpus.
If a source changes underneath you, a hit-rate movement no longer tells you whether the retriever improved or the corpus moved.
The lockfile stamps a new `fetched_at` only when the hash or commit changes, so a diff in that file always means the corpus actually moved.
It is also read back: a recorded commit is checked out rather than whatever HEAD is that day, so a hit-rate movement cannot be the corpus moving underneath the retriever.
Delete the lockfile entry to move the corpus forward deliberately.

## Tests

103 tests, no mocks.
A real local git repository, a real HTTP server on localhost, the real tokenizer, real Chroma.
Every path constant is a relative `Path`, so `monkeypatch.chdir` into a temporary directory redirects the whole pipeline without patching internals.

Covered: the metadata contract on both source types, lockfile stability across runs, a new upstream commit not moving a pinned corpus, chunk sizes measured in tokens rather than characters, chunk ids deterministic and unique, optional metadata absent rather than `None`, embeddings unit length on both paths, the query prefix applied to queries only, re-indexing upserting rather than duplicating, excludes surviving the whole pipeline, every eval label still pointing at a section that exists, and the committed snapshot agreeing with live settings.

Four of these were written after the bug they now catch.

## Evaluation

Tier 1 is built and runs on every commit in CI.
Tier 2 is not built.

```
      hit_rate_at_5 = 0.7647      13 of 17 in-corpus questions
                mrr = 0.5216
    abstention_rate = 0.0          0 of 5 out-of-corpus questions
```

`evals/tier1.py` scores retrieval against a golden set of 22 questions with no LLM, no judge and no API key, because whether a labelled section appeared in the top k is set arithmetic on a frozen corpus.
Results go to a committed snapshot, so a commit that changes retrieval changes that file and the effect arrives as a reviewable diff (`q07 rank 3 -> rank 1`) rather than as a metric moving for reasons nobody can see.
Coarse absolute floors sit below the baseline to catch a collapse that nobody should be able to accept by rerunning with `--update`.

The gap between hit-rate and MRR is the finding.
Seven of the thirteen hits are at rank 1, but three only scrape in at rank 4 or 5, so they would be misses at `k=3`.
That fragility is invisible in hit-rate alone and is the case a reranker is meant to fix, which is now a V1 change with a number to beat.

### How the labels were made

Labelling every relevant chunk in a corpus is infeasible, so `evals/pool.py` pools: it takes the top 20 from dense retrieval and the top 20 from BM25, fuses them by reciprocal rank, and only those candidates get judged.
Anything unjudged counts as not relevant, which can understate a score but never inflate one.
Two retrieval methods rather than one, because a pool built by a single retriever never surfaces what a different method would have found, and then scores that method as a miss.

Pooling only ever adds.
A judged candidate is kept forever, even once it drops out of the pool, so labels get more complete with every retrieval change instead of freezing around whatever today's retriever likes.

**20 of the 71 labels are for sections that neither method surfaced at all.**
The misses cluster: `Observability > Metrics` was retrieved while `Observability`, `> Logs` and `> Traces` were not, and for a question about latency the entire `Step 4. Reduce Latency with Cache` family was absent.
Chunking flattened a heading hierarchy that was carrying meaning, and top-k has no way to know a retrieved chunk has relatives.
That is the argument for parent-document retrieval, with a number attached instead of an intuition.

Labels are LLM-assisted and human-reviewed, not hand-written.
Stating that matters, because ground truth produced by a model and then scored by a model is circular unless the provenance is visible.

Two questions were reclassified out-of-corpus during labelling because nothing answered them: one asks how tokenization and attention work, which the corpus names only inside lists of interview questions, and one asks the system to generate a debugging exercise, which is not a retrieval question at any corpus size.

### Not built

**LLM-judged, on demand.** Faithfulness and answer relevancy through a judge model that is stronger than and different from the generator.
These are slow, paid and non-deterministic, so they produce a scored report compared against the previous run rather than a pass/fail gate.

**Baseline.** The same generator with no retrieved context, reported per question type.
On out-of-corpus questions the baseline is expected to beat the system, because the system should refuse while the baseline answers from memory.
Aggregating the question types into one number would make correct behaviour look like a loss.

## What failed or changed

Six findings that shaped the design. None of them produced an error.

**MiniLM truncates at 256 tokens.** The original plan paired `all-MiniLM-L6-v2` with 400-500 token chunks. Anything past 256 would have been silently dropped from the embedding while still being fed to the model as context. Switched to `bge-small-en-v1.5` for its 512-token sequence length.

**Chunk size was counted in characters.** `RecursiveCharacterTextSplitter` counts characters by default, so `chunk_size=450` meant roughly a quarter of the intended size. Fixed by constructing it `from_huggingface_tokenizer`. A test now asserts the unit.

**A source started serving zstd.** `trafilatura.fetch_url` advertised zstd support and returned the compressed bytes undecoded, which overwrote a good snapshot with binary garbage and produced an empty extraction. Fetching moved to `httpx`; fetching is HTTP's job, extraction is trafilatura's.

**A timeout in the wrong unit.** `request_timeout` in `langchain-openrouter` is milliseconds, not seconds. `request_timeout=30` meant 30ms, so every request died instantly and two stacked retry layers backed off, which looked like an indefinite hang on every model.

**The generator was a reasoning model.** `deepseek-v4-flash` spent 18.9 seconds and 171 reasoning tokens on "introduce yourself". Wrong for a system whose evaluation runs the same 20 questions repeatedly, and it would have compressed the baseline comparison by reasoning its way to good answers without the corpus.

**The abstention threshold was calibrated against the wrong kind of question.**
The 0.4 cosine-distance cutoff was set from in-corpus questions at ~0.19 against "what is the best pizza dough hydration?" at 0.51.
That question differs from the corpus in three ways at once: domain, register and length, seven words against forty.

The golden set includes three out-of-corpus questions that change **only the domain** - pastry-shop interview questions written in the same first-person interview-prep voice as the real ones.
They score 0.290, 0.335 and 0.398, all under the threshold.
Worse, a real question with the correct section retrieved scores 0.324, inside that range, so no cutoff on this signal separates the two classes.
Abstention is 0 of 5 and there is no threshold that fixes it.

The cause is dilution.
The query embedding averages over the whole string, and in a forty-word question the framing (`I want to come across as someone with good judgment`) outweighs the few tokens that carry the subject.
The corpus is entirely interview-prep material, so that framing matches it no matter what the question is about.

The general lesson: a calibration set that varies more than one thing at a time measures the easiest difference rather than the one you care about.
The fix is left unmade on purpose so the repair is a measured change with a before and after.

The first five findings share one shape: a number that is correct in the wrong unit, or a value that is correct for a different component.
The sixth is different in kind. It is not a bug in the code, it is a measurement that was never valid, and only building the eval exposed it.

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

**Abstention does not work.**
The guardrail fires on none of the five out-of-corpus questions, and the in-corpus and out-of-corpus distance ranges overlap, so retuning the threshold cannot fix it.
See "What failed or changed" for the measurements. The candidate repairs are stripping the framing before embedding (query rewriting) or replacing the distance test with a groundedness check after retrieval.

**Ground truth is LLM-assisted.**
The 71 labels were produced by a model against a pooled candidate set and reviewed rather than written from scratch.
Sixteen of the calls are adjacent rather than squarely on the question and are the ones worth challenging first.

## Status

- [x] Ingestion: fetch, parse, chunk, embed, index, with a corpus lockfile
- [x] Retrieval with an abstention guardrail and an index/model mismatch check
- [x] Generation with numbered citations and citation validation
- [x] 103 tests covering the pipeline and its known failure modes
- [x] Golden dataset of 22 questions, labelled by pooling two retrieval methods
- [x] Deterministic retrieval metrics in CI, gated by a committed snapshot
- [ ] LLM-judged metrics and a no-context baseline
- [ ] Web interface
- [ ] Tracing
