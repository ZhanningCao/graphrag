# -*- coding: utf-8 -*-
"""
GraphRAG Local Search 批量问答脚本 (适配 v3.0.5)

读取 QA.jsonl 中的问题，逐条调用 Local Search，输出回答与标准答案对比。
支持 PCST 子图检索（通过 settings.yaml 中 use_pcst 控制）。

用法:
  python batch_local_search.py --root <graphrag_root> --queries_file <QA.jsonl> --out_dir <output_dir> [选项]

示例:
  python batch_local_search.py ^
    --root D:\SEU\SRTP项目\graphrag\graph_database ^
    --queries_file D:\SEU\SRTP项目\graphrag\QA.jsonl ^
    --out_dir D:\SEU\SRTP项目\graphrag\trace_batch ^
    --query_col input ^
    --max_questions 10
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import json
import time
from pathlib import Path
from typing import Any

# Windows 上 nest_asyncio2 与 ProactorEventLoop (IOCP) 冲突，强制使用 SelectorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pandas as pd

# GraphRAG v3.0.5 imports
from graphrag.config.load_config import load_config
from graphrag.config.embeddings import entity_description_embedding
from graphrag.utils.api import get_embedding_store
from graphrag.query.factory import get_local_search_engine
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_text_units,
    read_indexer_reports,
)
from graphrag.cli.query import _resolve_output_files
from graphrag.query.structured_search.local_search.search import SearchResult


def load_queries(path: Path, query_col: str = "input") -> list[dict[str, Any]]:
    """从 .jsonl / .txt / .csv 加载问题列表。"""
    suffix = path.suffix.lower()
    items: list[dict[str, Any]] = []

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q = str(obj.get(query_col, "")).strip()
                if not q:
                    continue
                items.append({
                    "id": f"q{i:04d}",
                    "query": q,
                    "ground_truth": str(obj.get("output", "")),
                    "raw": obj,
                })
        return items

    if suffix == ".txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, start=1):
            q = line.strip()
            if q:
                items.append({"id": f"q{i:04d}", "query": q, "ground_truth": "", "raw": {}})
        return items

    if suffix == ".csv":
        df = pd.read_csv(path)
        for i, row in df.iterrows():
            q = str(row.get(query_col, "")).strip()
            if q and q.lower() != "nan":
                items.append({
                    "id": f"q{int(i)+1:04d}",
                    "query": q,
                    "ground_truth": str(row.get("output", "")),
                    "raw": row.to_dict(),
                })
        return items

    raise ValueError(f"不支持的文件格式: {suffix}，仅支持 .jsonl / .txt / .csv")


def load_existing_item_ids(output_file: Path) -> set[str]:
    """从单文件输出中读取已完成的 item_id，用于断点续跑。"""
    existing: set[str] = set()
    if not output_file.exists():
        return existing

    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                item_id = str(row.get("item_id", "")).strip()
                if item_id:
                    existing.add(item_id)
            except Exception:
                continue
    return existing


def main():
    ap = argparse.ArgumentParser(description="GraphRAG Local Search 批量问答")
    ap.add_argument("--root", required=True, help="GraphRAG 项目根目录 (含 settings.yaml 和 output/)")
    ap.add_argument("--queries_file", required=True, help="问题文件: .jsonl / .txt / .csv")
    ap.add_argument("--query_col", default="input", help="问题字段名 (默认 input)")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    ap.add_argument("--max_questions", type=int, default=None, help="只跑前 N 个问题")
    ap.add_argument("--community_level", type=int, default=0, help="社区层级（用于报告读取，默认 0）")
    ap.add_argument("--response_type", default="Multiple Paragraphs", help="回答类型")
    ap.add_argument("--skip_existing", action="store_true", help="若 output_file 中已有该题结果则跳过")
    ap.add_argument("--output_file", default="all_answers.jsonl", help="单文件输出名（写在 out_dir 下）")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    queries_path = Path(args.queries_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / args.output_file

    # ── 1. 加载配置 & 数据 ──
    print("[1/4] 加载配置和数据...", flush=True)
    t0 = time.perf_counter()

    config = load_config(root_dir=root)
    dataframe_dict = _resolve_output_files(
        config=config,
        output_list=["communities", "community_reports", "text_units", "relationships", "entities"],
        optional_list=["covariates"],
    )

    communities = dataframe_dict["communities"]
    community_reports = dataframe_dict["community_reports"]
    text_units = dataframe_dict["text_units"]
    relationships = dataframe_dict["relationships"]
    entities = dataframe_dict["entities"]
    covariates = dataframe_dict["covariates"]

    t1 = time.perf_counter()
    print(f"    数据加载完成 ({t1 - t0:.1f}s)", flush=True)

    # ── 2. 初始化搜索引擎 ──
    print("[2/4] 初始化 Local Search 引擎...", flush=True)

    description_embedding_store = get_embedding_store(
        config=config.vector_store,
        embedding_name=entity_description_embedding,
    )

    from graphrag.query.indexer_adapters import read_indexer_covariates
    # 对当前数据集，community_level=0 会导致实体为空，因此这里固定读取全部实体
    entities_ = read_indexer_entities(entities, communities, None)
    covariates_ = read_indexer_covariates(covariates) if covariates is not None else []

    # 加载 prompt
    from graphrag.prompts.query.local_search_system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT
    prompt = LOCAL_SEARCH_SYSTEM_PROMPT
    if config.local_search.prompt:
        prompt_path = root / config.local_search.prompt
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8")

    search_engine = get_local_search_engine(
        config=config,
        reports=read_indexer_reports(community_reports, communities, args.community_level),
        text_units=read_indexer_text_units(text_units),
        entities=entities_,
        relationships=read_indexer_relationships(relationships),
        covariates={"claims": covariates_},
        description_embedding_store=description_embedding_store,
        response_type=args.response_type,
        system_prompt=prompt,
    )

    ls_config = config.local_search
    print(f"    PCST: use_pcst={ls_config.use_pcst}, "
          f"top_k_nodes={ls_config.pcst_top_k_nodes}, "
          f"top_k_edges={ls_config.pcst_top_k_edges}, "
          f"cost={ls_config.pcst_cost_per_edge}", flush=True)

    t2 = time.perf_counter()
    print(f"    引擎初始化完成 ({t2 - t1:.1f}s)", flush=True)

    # ── 3. 加载问题 ──
    print("[3/4] 加载问题...", flush=True)
    items = load_queries(queries_path, query_col=args.query_col)
    if args.max_questions is not None:
        items = items[:args.max_questions]
    print(f"    共 {len(items)} 个问题", flush=True)

    existing_ids: set[str] = set()
    if args.skip_existing:
        existing_ids = load_existing_item_ids(output_file)
        print(f"    断点续跑: output_file 中已存在 {len(existing_ids)} 条", flush=True)
    else:
        if output_file.exists():
            output_file.unlink()
        output_file.touch()
        print(f"    输出文件已初始化: {output_file}", flush=True)

    # ── 4. 逐题搜索 ──
    print(f"[4/4] 开始批量搜索 ({len(items)} 题)...", flush=True)

    results: list[dict[str, Any]] = []
    success_count = 0
    error_count = 0
    skipped_count = 0

    for idx, item in enumerate(items, start=1):
        item_id = item["id"]
        query = item["query"]
        ground_truth = item["ground_truth"]

        # 检查是否跳过
        if args.skip_existing and item_id in existing_ids:
            print(f"  [{idx}/{len(items)}] SKIP {item_id}", flush=True)
            skipped_count += 1
            continue

        print(f"  [{idx}/{len(items)}] {item_id}: {query[:60]}...", flush=True)
        q_t0 = time.perf_counter()

        try:
            search_result: SearchResult = asyncio.run(
                search_engine.search(query=query)
            )
            answer = str(search_result.response)
            q_t1 = time.perf_counter()
            context_text = str(search_result.context_text or "")
            prompt_cats = search_result.prompt_tokens_categories or {}

            meta = {
                "item_id": item_id,
                "query": query,
                "ground_truth": ground_truth,
                "answer": answer,
                "status": "ok",
                "time_sec": round(q_t1 - q_t0, 2),
                "completion_time": round(search_result.completion_time, 2),
                "llm_calls": search_result.llm_calls,
                "prompt_tokens": search_result.prompt_tokens,
                "output_tokens": search_result.output_tokens,
                "prompt_tokens_response": int(prompt_cats.get("response", 0)),
                "prompt_tokens_build_context": int(prompt_cats.get("build_context", 0)),
                "context_chars": len(context_text),
                "context_text": context_text,
            }

            # 实时追加到单文件，避免中断时丢失进度
            with output_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            results.append(meta)
            success_count += 1

            print(f"    ✓ {q_t1 - q_t0:.1f}s | 回答: {answer[:80]}...", flush=True)

        except Exception as e:
            q_t1 = time.perf_counter()
            err_text = repr(e)
            print(f"    ✗ ERROR: {err_text[:120]}", flush=True)

            err_row = {
                "item_id": item_id,
                "query": query,
                "ground_truth": ground_truth,
                "answer": "",
                "status": "error",
                "error": err_text,
                "time_sec": round(q_t1 - q_t0, 2),
            }

            with output_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(err_row, ensure_ascii=False) + "\n")

            results.append(err_row)
            error_count += 1

    # ── 5. 写入汇总 ──
    total_time = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"批量搜索完成!")
    print(f"  成功: {success_count}, 失败: {error_count}, 跳过: {skipped_count}, 总计: {len(items)}")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  单文件输出: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
