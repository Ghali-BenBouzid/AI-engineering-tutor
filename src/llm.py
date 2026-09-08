from langchain_openrouter import ChatOpenRouter

from functools import lru_cache

from src.config import settings


def get_llm() -> ChatOpenRouter:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    return ChatOpenRouter(
        model=settings.llm_model,
        openrouter_api_key=settings.openrouter_api_key,
        temperature=0,
        request_timeout=60000,
    )

def main() -> None:
    model = get_llm()
    print("Calling OpenRouter...")
    response = model.invoke("Say OK")
    print("Model response: \n", response.content)

if __name__=="__main__":
    main()