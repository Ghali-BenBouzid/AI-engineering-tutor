import re
from dataclasses import dataclass, field

from src.assemble import format_context, format_sources
from src.config import settings
from src.llm import get_llm
from src.prompt import NO_ANSWER, RAG_PROMPT
from src.retrieval import top_k_search

CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)
    abstained: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    top_distance: float | None = None


def cited(text: str) -> set[int]:
    return {int(n) for n in CITATION.findall(text)}


def answer(question: str) -> Answer:
    hits = top_k_search(question)

    # Chroma returns cosine distance here, smaller is closer.
    if not hits or hits[0][1] > settings.max_distance:
        return Answer(
            text=NO_ANSWER,
            abstained=True,
            top_distance=hits[0][1] if hits else None,
        )

    docs = [doc for doc, _ in hits]
    messages = RAG_PROMPT.format_messages(
        context=format_context(docs), question=question
    )
    text = get_llm().invoke(messages).content

    valid = set(range(1, len(docs) + 1))

    return Answer(
        text=text,
        sources=format_sources(docs),
        abstained=text.strip() == NO_ANSWER,
        invalid_citations=sorted(cited(text) - valid),
        top_distance=hits[0][1],
    )


def main() -> None:
    for question in [
        "What is query rewriting and why does it help retrieval?",
        "What is the best pizza dough hydration?",
        """
        Cite all RAG components, then explain to me how to build a \
        simple RAG pipeline that retrieves data from a reference \
        corpus and answers only from that corpus.
        """,
        """
        Sketch a roadmap to prepare for a Junior AI Engineer job, \
        make it specific to someone who already has a solid foundation \
        in data science and Python.
        """
    ]:
        result = answer(question)
        print(f"\n{'=' * 70}\nQ: {question}")
        print(f"top_distance={result.top_distance:.4f}  abstained={result.abstained}"
              f"  invalid_citations={result.invalid_citations}")
        print(f"\n{result.text}\n")
        for source in result.sources:
            print(f"  [{source['n']}] {source['title']} > {source['section']}")


if __name__ == "__main__":
    main()
