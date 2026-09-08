from langchain_core.documents import Document

SEPARATOR = "\n\n---\n\n"


def format_context(docs: list[Document]) -> str:
    """
    Number the chunks so the model has something short to cite.
    """

    blocks = []
    for n, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = f"[{n}] {meta['title']}"
        if meta.get("section"):
            header += f" > {meta['section']}"
        blocks.append(f"{header}\n{meta['url']}\n\n{doc.page_content}")

    return SEPARATOR.join(blocks)


def format_sources(docs: list[Document]) -> list[dict]:
    """
    What the UI renders under the answer, in citation order.
    """
    
    return [
        {
            "n": n,
            "title": doc.metadata["title"],
            "section": doc.metadata.get("section", ""),
            "url": doc.metadata["url"],
            "doc_id": doc.metadata["doc_id"],
        }
        for n, doc in enumerate(docs, start=1)
    ]
