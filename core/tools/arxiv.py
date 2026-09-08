"""arXiv paper search.

Not currently wired into the agent — kept as the provider a future
`paper_search` capability would sit on. See core/tools/adapters.py for how
capabilities are registered.
"""

import arxiv


def arxiv_search(query: str, k: int = 5) -> list[dict]:
    """
    Perform an arxiv search using Arxiv API.

    Args:
        query (str): The search query.
        k (int): The number of results to return. Default is 3. Max is 5.

    Returns:
        list[dict]: A list of search result dictionaries.
    """
    k = min(k, 5)
    search = arxiv.Search(
        query=query, max_results=k, sort_by=arxiv.SortCriterion.SubmittedDate
    )
    results = []
    for result in search.results():
        results.append(
            {
                "title": result.title,
                "url": result.pdf_url,
                "content": result.summary,
            }
        )
    if not results:
        return [
            {
                "title": "No results found, please change your query",
                "url": "",
                "content": "",
            }
        ]
    return results
