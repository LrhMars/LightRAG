"""
从 raw_json 构图并生成三类问答：
- 单跳事实
- 多跳推理
- 全局总结
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 raw_json 目录生成三类问答数据集")
    parser.add_argument("--input_dir", required=True, help="raw_json 目录路径")
    parser.add_argument("--output_path", default="qa_dataset.jsonl", help="输出 jsonl 路径")
    parser.add_argument("--target_size", type=int, default=6000, help="目标问答条数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--ratio_1hop", type=float, default=0.3, help="单跳占比")
    parser.add_argument("--ratio_multi", type=float, default=0.5, help="多跳占比")
    parser.add_argument("--ratio_star", type=float, default=0.2, help="全局总结占比")
    return parser.parse_args()


def load_raw_chunks(input_dir: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for file_path in sorted(input_dir.glob("*_raw.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunks.append(json.load(f))
        except Exception:
            continue
    return chunks


def build_graph(chunks: list[dict[str, Any]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for chunk in chunks:
        entities = chunk.get("entities") or []
        relations = chunk.get("relations") or []
        chunk_id = chunk.get("chunk_id", "")

        for ent in entities:
            name = (ent.get("entity_name") or "").strip()
            if not name:
                continue
            node_data = {
                "name": name,
                "etype": ent.get("entity_type", ""),
                "desc": ent.get("entity_description", ""),
                "section": ent.get("source_section", ""),
                "source_chunk": ent.get("source_chunk", chunk_id),
            }
            if not graph.has_node(name):
                graph.add_node(name, **node_data)
            else:
                # 补充已有节点缺失字段
                old = graph.nodes[name]
                for k, v in node_data.items():
                    if not old.get(k) and v:
                        old[k] = v

        for rel in relations:
            src = (rel.get("source") or "").strip()
            tgt = (rel.get("target") or "").strip()
            if not (src and tgt):
                continue
            if not graph.has_node(src):
                graph.add_node(src, name=src, etype="", desc="", section=rel.get("source_section", ""), source_chunk=chunk_id)
            if not graph.has_node(tgt):
                graph.add_node(tgt, name=tgt, etype="", desc="", section=rel.get("source_section", ""), source_chunk=chunk_id)
            graph.add_edge(
                src,
                tgt,
                rel_name=rel.get("relation_name") or rel.get("predicate") or "",
                rel_desc=rel.get("relationship_description", ""),
                section=rel.get("source_section", ""),
                source_chunk=rel.get("source_chunk", chunk_id),
            )
    return graph


def node_text(graph: nx.DiGraph, nid: str) -> str:
    n = graph.nodes[nid]
    return f"{n.get('name', nid)}（{n.get('etype', '未标注')}）"


def pick_one(rng: random.Random, items: list[str]) -> str:
    return items[rng.randrange(len(items))]


def gen_1hop_qa(graph: nx.DiGraph, u: str, v: str, ed: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    rel_name = ed.get("rel_name") or "关联"
    rel_desc = ed.get("rel_desc") or ""
    q_templates = [
        f"在实际施工里，{u} 和 {v} 分别承担什么角色，它们之间是什么联系？",
        f"围绕“{u}”开展工作时，为什么经常要同时考虑“{v}”？",
        f"如果要说明 {u} 与 {v} 的配合逻辑，最关键的一点是什么？",
        f"在这一章节语境下，{u} 对 {v} 的影响主要体现在哪里？",
    ]
    question = pick_one(rng, q_templates)
    answer = (
        f"{node_text(graph, u)} 与 {node_text(graph, v)} 的核心联系是“{rel_name}”。"
        f"{rel_desc if rel_desc else '这说明两者在同一业务流程中存在直接配合关系。'}"
    )
    return {
        "question": question,
        "answer": answer,
        "path_type": "单跳事实",
        "source_section": ed.get("section", ""),
        "source_chunk": ed.get("source_chunk", ""),
    }


def gen_2hop_qa(
    graph: nx.DiGraph,
    u: str,
    v: str,
    w: str,
    uv: dict[str, Any],
    vw: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    rel1 = uv.get("rel_name") or "关联"
    rel2 = vw.get("rel_name") or "关联"
    q_templates = [
        f"如果现场先处理“{u}”，再处理“{v}”，最终会怎样影响“{w}”？",
        f"从执行顺序看，{u} 通过 {v} 对 {w} 产生了什么间接作用？",
        f"为什么说在这个问题里，{v} 是连接 {u} 和 {w} 的关键环节？",
        f"把 {u}、{v}、{w} 放在同一流程里，前后因果关系该怎么理解？",
    ]
    question = pick_one(rng, q_templates)
    answer = (
        f"可以归纳为：{u} 先通过“{rel1}”作用到 {v}，"
        f"再由 {v} 通过“{rel2}”影响 {w}。"
        f"也就是说，{u} 对 {w} 的影响主要是经由中间环节逐步传导，而不是一步到位。"
    )
    return {
        "question": question,
        "answer": answer,
        "path_type": "多跳推理",
        "source_section": uv.get("section", "") or vw.get("section", ""),
        "source_chunk": uv.get("source_chunk", "") or vw.get("source_chunk", ""),
    }


def gen_star_qa(
    graph: nx.DiGraph,
    center: str,
    neighbors: list[str],
    rels: list[str],
    section: str,
    chunk: str,
    rng: random.Random,
) -> dict[str, Any]:
    neighbor_str = "、".join(neighbors[:5])
    rel_str = "；".join(rels[:5])
    q_templates = [
        f"如果把“{center}”当作这部分内容的主线，通常会牵涉哪些重点对象？",
        f"围绕“{center}”开展管理或施工时，最需要同时关注哪些方面？",
        f"从章节整体看，“{center}”为什么是一个高频出现的关键点？",
        f"在这部分知识里，以“{center}”为中心能串起哪些核心内容？",
    ]
    question = pick_one(rng, q_templates)
    answer = (
        f"围绕“{center}”，常见的重点对象包括：{neighbor_str}。"
        f"从内容联系看，典型关联可概括为：{rel_str}。"
        f"因此它在本章节里更像一个组织性节点，能够把不同知识点串联起来。"
    )
    return {
        "question": question,
        "answer": answer,
        "path_type": "全局总结",
        "source_section": section,
        "source_chunk": chunk,
    }


def section_balanced_pick(items: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        buckets[item.get("source_section") or "未标注章节"].append(item)
    for b in buckets.values():
        rng.shuffle(b)

    picked: list[dict[str, Any]] = []
    sections = sorted(buckets.keys())
    for sec in sections:
        if buckets[sec]:
            picked.append(buckets[sec].pop())
            if len(picked) >= target:
                return picked

    active = [s for s in sections if buckets[s]]
    while active and len(picked) < target:
        next_active: list[str] = []
        for sec in active:
            if buckets[sec]:
                picked.append(buckets[sec].pop())
                if len(picked) >= target:
                    return picked
            if buckets[sec]:
                next_active.append(sec)
        active = next_active
    return picked


def dedup(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        key = (it.get("question", "").strip(), it.get("answer", "").strip(), it.get("path_type", "").strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def build_candidates(graph: nx.DiGraph, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)

    one_hop: list[dict[str, Any]] = []
    for u, v, ed in graph.edges(data=True):
        one_hop.append(gen_1hop_qa(graph, u, v, ed, rng))

    multi_hop: list[dict[str, Any]] = []
    for u in graph.nodes():
        succ = list(graph.successors(u))
        rng.shuffle(succ)
        for v in succ[:20]:
            succ2 = list(graph.successors(v))
            rng.shuffle(succ2)
            for w in succ2[:20]:
                if w == u:
                    continue
                multi_hop.append(gen_2hop_qa(graph, u, v, w, graph[u][v], graph[v][w], rng))

    star: list[dict[str, Any]] = []
    ug = graph.to_undirected()
    for c in ug.nodes():
        nbrs = list(ug.neighbors(c))
        if len(nbrs) < 3:
            continue
        rng.shuffle(nbrs)
        sampled = nbrs[: min(6, len(nbrs))]
        rels: list[str] = []
        section = graph.nodes[c].get("section", "")
        chunk = graph.nodes[c].get("source_chunk", "")
        for nb in sampled:
            ed = graph.get_edge_data(c, nb) or graph.get_edge_data(nb, c) or {}
            rel_name = ed.get("rel_name") or "关联"
            rels.append(f"{c}-{rel_name}-{nb}")
            if not section:
                section = ed.get("section", "")
            if not chunk:
                chunk = ed.get("source_chunk", "")
        star.append(gen_star_qa(graph, c, sampled, rels, section, chunk, rng))
    return one_hop, multi_hop, star


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    chunks = load_raw_chunks(input_dir)
    print(f"[加载] chunk 文件数：{len(chunks)}")
    graph = build_graph(chunks)
    print(f"[构图] 节点数={graph.number_of_nodes()} 边数={graph.number_of_edges()}")

    one_hop, multi_hop, star = build_candidates(graph, args.seed)
    one_hop = dedup(one_hop)
    multi_hop = dedup(multi_hop)
    star = dedup(star)
    print(f"[候选] 单跳={len(one_hop)} 多跳={len(multi_hop)} 全局={len(star)}")

    total_ratio = args.ratio_1hop + args.ratio_multi + args.ratio_star
    target_1 = int(args.target_size * args.ratio_1hop / total_ratio)
    target_2 = int(args.target_size * args.ratio_multi / total_ratio)
    target_3 = args.target_size - target_1 - target_2

    picked_1 = section_balanced_pick(one_hop, min(target_1, len(one_hop)), args.seed)
    picked_2 = section_balanced_pick(multi_hop, min(target_2, len(multi_hop)), args.seed + 1)
    picked_3 = section_balanced_pick(star, min(target_3, len(star)), args.seed + 2)

    final = picked_1 + picked_2 + picked_3
    rng.shuffle(final)
    final = dedup(final)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in final:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[完成] 输出条数={len(final)}")
    print(f"[完成] 类型分布：单跳={len(picked_1)} 多跳={len(picked_2)} 全局={len(picked_3)}")
    print(f"[完成] 输出文件={output_path}")


if __name__ == "__main__":
    main()
