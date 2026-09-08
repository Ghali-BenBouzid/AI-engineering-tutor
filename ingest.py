from src.fetch_and_parse import fetch_and_parse
from src.chunk import chunk_all
from src.index import index_chunks


def ingest() -> None:
    fetch_and_parse()
    chunks = chunk_all()
    index_chunks(chunks)

if __name__=="__main__":
    ingest()
