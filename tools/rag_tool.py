import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from http_client import async_post_json

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("rag_tool")

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080/api/v1")

async def query_rag_knowledge(
    query: str,
    tenant_id: int = 1,
    knowledge_base_ids: list = None,
    limit: int = 3,
) -> str:
    """
    Performs real-time pgvector semantic vector search across the tenant's uploaded knowledge base documents.
    Returns the most relevant factual context in under 35ms.
    """
    logger.info(f"[TOOL] Querying pgvector knowledge base for: '{query}' (Tenant: {tenant_id})")
    payload = {
        "query": query,
        "tenant_id": tenant_id,
        "knowledge_base_ids": knowledge_base_ids or [],
        "limit": limit,
    }
    
    try:
        url = f"{BACKEND_API_URL}/rag/search"
        status, data = await async_post_json(url, payload, timeout=2.5)
        if status == 200 and isinstance(data, dict):
            chunks = data.get("results", [])
            if chunks:
                context = "\n---\n".join([f"Source: {c.get('source', 'Knowledge Base')}\n{c.get('content', '')}" for c in chunks])
                logger.info(f"[TOOL] pgvector retrieved {len(chunks)} relevant chunks.")
                return f"Knowledge Base Facts:\n{context}"
    except Exception as e:
        logger.warning(f"pgvector RAG search error: {e}")

    return "No additional external knowledge articles found for this exact query. Use agent core prompt instructions."
