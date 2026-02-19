from langchain_core.documents import Document

from langchain_community.retrievers import BM25Retriever


def check_source(source: dict, text_selection: str):
    docs = []
    for item in source["final_sources"]:
        docs.append(
            Document(
                page_content=item["content"],
                metadata={"title": item["title"], "url": item["url"]},
            )
        )

    retriever = BM25Retriever.from_documents(docs)
    selections = retriever.invoke(text_selection)
    result = selections[0]
    return {
        "title": result.metadata["title"],
        "url": result.metadata["url"],
        "content": result.page_content,
    }
