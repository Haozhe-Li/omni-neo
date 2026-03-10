from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
from core.database.db_user_files import get_thread_files
from typing import Optional

import logging

logger = logging.getLogger(__name__)

try:
    from upstash_search import Search

    upstash_client = Search.from_env()
    upstash_index = upstash_client.index("omni-documents")
except Exception as e:
    logger.error(f"Failed to initialize Upstash Search in tools: {e}")
    upstash_index = None


@tool
def read_user_document(
    query: Optional[str] = None, *, config: RunnableConfig = None
) -> str:
    """
    Read or search user uploaded documents.
    When user upload documents, you must use this tool to read or search the documents.
    """
    if config is None:
        config = {}

    current_thread_id = config.get("configurable", {}).get("thread_id")
    if not current_thread_id:
        return "Unable to get the current conversation context ID. Cannot limit the search scope."

    logger.debug(f"[read_user_document] thread_id={current_thread_id}, query='{query}'")

    if not query:
        try:
            files = get_thread_files(current_thread_id)
            if not files:
                return "No documents uploaded in the current conversation were found."

            context_parts = []
            for f in files:
                filename = f.get("original_filename", "Unknown Document")
                content = f.get("extracted_text", "")
                if not content:
                    content = "[No text content]"
                context_parts.append(f"[Full document text of {filename}]:\n{content}")

            full_text = (
                "Here is the full text of the user's uploaded documents:\n\n"
                + "\n\n".join(context_parts)
            )
            if len(full_text) > 30000:
                warning_msg = "\n\n[WARNING]: The document content exceeds 30,000 characters. Only the first 30,000 characters are loaded. If you want to get more information, please use the 'query' parameter to search."
                return full_text[:30000] + warning_msg

            return full_text
        except Exception as e:
            logger.error(f"Failed to fetch document full text: {e}")
            return f"Failed to read the full text of the document. Please try again later: {e}"

    if not upstash_index:
        return "Search functionality is currently disabled or unreachable."

    search_filter = f"@metadata.thread_id = '{current_thread_id}'"

    try:
        logger.debug(f"[read_user_document] Upstash search filter: '{search_filter}'")
        results = upstash_index.search(query=query, limit=5, filter=search_filter)

        if not results:
            return "Could not retrieve relevant snippets from the provided documents."

        context_parts = []
        for r in results:
            metadata = getattr(r, "metadata", {})
            if isinstance(metadata, property) or not isinstance(metadata, dict):
                metadata = r.metadata if hasattr(r, "metadata") else {}

            filename = (
                metadata.get("filename", "Unknown Document")
                if isinstance(metadata, dict)
                else "Unknown Document"
            )

            content_obj = (
                r.data if hasattr(r, "data") else getattr(r, "content", str(r))
            )

            if isinstance(r, dict):
                filename = r.get("metadata", {}).get("filename", "Unknown Document")
                content_obj = r.get("data") or r.get("content")

            if isinstance(content_obj, dict):
                content = content_obj.get("text", str(content_obj))
            else:
                content = content_obj

            context_parts.append(f"[Snippet extracted from {filename}]:\n{content}")

        return (
            "The following relevant document snippets were retrieved:\n\n"
            + "\n\n".join(context_parts)
        )
    except Exception as e:
        logger.error(f"Upstash Search failed during agent tool execution: {e}")
        return f"Search failed, please try again later: {e}"
