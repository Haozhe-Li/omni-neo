from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig

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
def search_in_document(query: str, config: RunnableConfig) -> str:
    """
    搜索当前对话上下文中用户上传的文档内容。
    当你被要求查阅文档，或者询问之前上传文件的具体细节时，务必优先调用此工具。
    参数:
    - query: 用来检索的问题陈述或者关键词组合（例如 "Q3的利润增长率"）。不可以是空字符串。
    """
    current_thread_id = config.get("configurable", {}).get("thread_id")
    if not current_thread_id:
        return "无法获取当前对话的上下文ID，无法限制搜索范围。"
    print("thread_id", current_thread_id)

    print(f"[search_in_document] thread_id={current_thread_id}, query='{query}'")

    if not upstash_index:
        return "Search functionality is currently disabled or unreachable."

    search_filter = f"@metadata.thread_id = '{current_thread_id}'"

    try:
        print(f"[search_in_document] Upstash search filter: '{search_filter}'")
        results = upstash_index.search(query=query, limit=5, filter=search_filter)

        if not results:
            return "未能在已提供的文档中检索到相关片段。"

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

            context_parts.append(f"[片段摘录自 {filename}]:\n{content}")

        return "检索到了以下相关文档片段：\n\n" + "\n\n".join(context_parts)
    except Exception as e:
        logger.error(f"Upstash Search failed during agent tool execution: {e}")
        return f"搜索失败，请稍后重试: {e}"
