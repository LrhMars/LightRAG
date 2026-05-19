"""
mcp_server.py  —  HyperLogRAG MCP Server
════════════════════════════════════════════════════════════════════════════════
基于 FastMCP SDK 实现，将知识库工具标准化暴露为 MCP 协议接口。

外部工具（Cursor / Claude Desktop 等支持 MCP 的客户端）可直接接入，
无需了解 LightRAG 内部细节。

暴露工具（4 个）：
  search            —— 语义搜索（向量检索 + KG 推理，适合大多数问题）
  get_regulation    —— 按章节路径获取规范原文
  query_entity      —— 查询实体详细属性和邻居关系
  list_knowledge_bases  —— 列出当前可用的知识库

启动方式（独立进程，stdio 传输）：
  python mcp_server.py

Cursor 接入配置（~/.cursor/mcp.json）：
  {
    "mcpServers": {
      "hyperlograg": {
        "command": "python",
        "args": ["D:/path/to/HyperLogRAG/mcp_server.py"],
        "env": {}
      }
    }
  }

环境变量（与主服务共用 .env）：
  LIGHTRAG_WORKING_DIR      —— 知识库数据目录（默认 ./data/rag_storage）
  LLM_API_BASE              —— LLM API 地址
  LLM_API_KEY               —— LLM API 密钥
  LLM_MODEL                 —— 模型名称
  EMBEDDING_API_BASE        —— Embedding API 地址
  EMBEDDING_API_KEY         —— Embedding API 密钥
  EMBEDDING_MODEL           —— Embedding 模型名称
════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# 加载项目 .env
_project_root = Path(__file__).parent
_dotenv_path = _project_root / ".env"
if _dotenv_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_dotenv_path)
    except ImportError:
        pass

# ── 环境变量配置 ──────────────────────────────────────────────────────────────

WORKING_DIR = os.getenv("LIGHTRAG_WORKING_DIR", str(_project_root / "data" / "rag_storage"))
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:8080/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "dummy")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-32b")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", LLM_API_BASE)
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", LLM_API_KEY)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("mcp_server")

# ── FastMCP 初始化 ────────────────────────────────────────────────────────────

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="HyperLogRAG 工程知识库",
    instructions=(
        "这是一个面向工程文档（港口、航运、施工规范等）的知识库服务。"
        "你可以通过 search 工具查询大多数问题，"
        "通过 get_regulation 工具获取具体规范条文原文，"
        "通过 query_entity 工具查询特定实体的详细属性。"
    ),
)

# ── 懒加载 LightRAG 实例（避免启动时立即连接数据库）────────────────────────────

_rag_instance = None
_rag_lock = asyncio.Lock()


async def _get_rag():
    """懒加载 LightRAG 实例（首次调用时初始化）。"""
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance

    async with _rag_lock:
        if _rag_instance is not None:
            return _rag_instance

        try:
            # 动态导入，避免 MCP 服务启动时加载所有 LightRAG 依赖
            sys.path.insert(0, str(_project_root))
            from lightrag import LightRAG, QueryParam
            from lightrag.utils import EmbeddingFunc
            from lightrag.llm.openai import openai_complete_if_cache, openai_embed

            async def llm_func(prompt, **kwargs):
                kwargs.pop("hashing_kv", None)
                return await openai_complete_if_cache(
                    LLM_MODEL,
                    prompt,
                    api_key=LLM_API_KEY,
                    base_url=LLM_API_BASE,
                    **kwargs,
                )

            embed_func = EmbeddingFunc(
                embedding_dim=1536,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model=EMBEDDING_MODEL,
                    api_key=EMBEDDING_API_KEY,
                    base_url=EMBEDDING_API_BASE,
                ),
            )

            _rag_instance = LightRAG(
                working_dir=WORKING_DIR,
                llm_model_func=llm_func,
                embedding_func=embed_func,
            )
            await _rag_instance.initialize_storages()
            logger.info("LightRAG 实例初始化完成，working_dir=%s", WORKING_DIR)

        except Exception as e:
            logger.error("LightRAG 初始化失败: %s", e)
            raise RuntimeError(f"知识库服务初始化失败: {e}") from e

    return _rag_instance


# ── MCP 工具定义 ───────────────────────────────────────────────────────────────


@mcp.tool()
async def search(question: str, mode: str = "hybrid") -> str:
    """
    在工程知识库中搜索答案，适合大多数问题。

    支持语义理解和知识图谱多跳推理，能够回答需要关联多个概念的问题。

    Args:
        question: 你的问题（中文或英文均可）
        mode: 检索模式，可选 hybrid（默认，综合）/ local（实体中心）/ global（关系中心）

    Returns:
        基于知识库的综合回答
    """
    try:
        from lightrag.base import QueryParam
        rag = await _get_rag()
        valid_modes = {"local", "global", "hybrid", "mix", "naive"}
        if mode not in valid_modes:
            mode = "hybrid"
        answer = await rag.aquery(question, QueryParam(mode=mode))
        if not isinstance(answer, str):
            chunks = []
            async for token in answer:
                chunks.append(str(token))
            answer = "".join(chunks)
        return answer or "知识库中未找到相关信息。"
    except Exception as e:
        logger.warning("search 工具执行失败: %s", e)
        return f"[搜索失败] {e}"


@mcp.tool()
async def get_regulation(section_path: str) -> str:
    """
    按章节路径获取规范条文原文，用于精确溯源。

    Args:
        section_path: 章节路径，如 "第三章 -> 安全规范 -> 水上作业条件"
                      也可以只提供部分路径，如 "安全规范" 或 "水上作业"

    Returns:
        匹配该路径的规范条文文本（若有多条，返回最相关的几条）
    """
    try:
        rag = await _get_rag()
        from lightrag.agent.tools import ToolExecutor
        executor = ToolExecutor(rag)
        result = await executor.execute("get_section", {"path": section_path, "top_k": 3})

        if result.get("error"):
            return f"[查询失败] {result['error']}"

        chunks = result.get("result", [])
        if not chunks:
            return f"未找到与路径 '{section_path}' 匹配的条文。"

        parts = []
        for i, chunk in enumerate(chunks, 1):
            path = chunk.get("section_path", "未知路径")
            content = chunk.get("content", "")
            match_type = chunk.get("match_type", "")
            parts.append(
                f"【条文{i}】路径: {path}（{match_type}匹配）\n{content}"
            )
        return "\n\n".join(parts)

    except Exception as e:
        logger.warning("get_regulation 工具执行失败: %s", e)
        return f"[查询失败] {e}"


@mcp.tool()
async def query_entity(entity_name: str) -> str:
    """
    查询某个实体的详细属性和邻居关系，适合了解具体实体的特性。

    Args:
        entity_name: 实体名称，如 "波浪高度"、"高桩码头"、"停工条件"

    Returns:
        实体的类型、描述、以及与其相关联的实体列表
    """
    try:
        rag = await _get_rag()
        from lightrag.agent.tools import ToolExecutor
        executor = ToolExecutor(rag)
        result = await executor.execute("entity_lookup", {"name": entity_name, "include_edges": True})

        if result.get("error"):
            return f"[查询失败] {result['error']}"

        data = result.get("result", {})
        if not data.get("found"):
            return f"知识库中未找到实体 '{entity_name}'。"

        props = data.get("properties", {})
        edges = data.get("edges", [])

        lines = [
            f"实体名称: {entity_name}",
            f"类型: {props.get('entity_type', '未知')}",
            f"描述: {props.get('description', '无描述')}",
            f"关联实体数: {data.get('degree', 0)}",
        ]
        if edges:
            edge_strs = [f"  · {e['src']} → {e['tgt']}" for e in edges[:10]]
            lines.append("关联关系:")
            lines.extend(edge_strs)
        return "\n".join(lines)

    except Exception as e:
        logger.warning("query_entity 工具执行失败: %s", e)
        return f"[查询失败] {e}"


@mcp.tool()
async def list_knowledge_bases() -> str:
    """
    列出当前系统中所有可用的知识库（数据集）。

    Returns:
        知识库列表，包含名称、文档数量等信息
    """
    try:
        rag = await _get_rag()
        inner = {}
        for attr in ("_data", "_kv_data", "_store"):
            inner = getattr(rag.datasets, attr, None) or {}
            if inner:
                break

        if not inner:
            return "当前系统中没有可用的知识库。"

        lines = [f"共有 {len(inner)} 个知识库："]
        for kb_id, kb_data in inner.items():
            if not isinstance(kb_data, dict):
                continue
            name = kb_data.get("name", kb_id)
            doc_count = len(kb_data.get("doc_ids", []))
            lines.append(f"  · {name}（ID: {kb_id[:8]}...，文档数: {doc_count}）")
        return "\n".join(lines)

    except Exception as e:
        logger.warning("list_knowledge_bases 执行失败: %s", e)
        return f"[查询失败] {e}"


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("HyperLogRAG MCP Server 启动中...", file=sys.stderr)
    print(f"  知识库目录: {WORKING_DIR}", file=sys.stderr)
    print(f"  LLM: {LLM_MODEL} @ {LLM_API_BASE}", file=sys.stderr)
    print(f"  工具: search / get_regulation / query_entity / list_knowledge_bases", file=sys.stderr)
    mcp.run(transport="stdio")
