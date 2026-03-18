# -*- coding: utf-8 -*-
"""
GraphRAG local-search 批量问答脚本（先批量 embedding / context，再统一 chat）

相对 local_search_qa_ollama_batch.py 的主要变化：
1. 启动后只加载一次 settings / parquet / GraphRAG 对象 / embedder / LanceDB；
2. 第一阶段先对所有问题做 query embedding（尽量走批量接口；不支持则降级循环）；
3. 仍在第一阶段完成 retrieval + build_context，并写出 retrieval/context 产物；
4. 第二阶段再统一调用 Ollama /api/chat 逐题生成答案；
5. 避免“每题 embedding -> chat -> embedding -> chat”的模型频繁切换。

注意：
- 上下文构建逻辑保持不变，仍调用 LocalSearchMixedContext.build_context()；
- 由于 build_context() 内部可能仍会再次调用 text_embedder，这一部分不会被本脚本改写；
  但至少整个批处理会先完成全部 context，再开始 chat，不再与 chat 交替执行。
- “query_embed_only” 在单题 meta 中为批量分摊估算值（estimated），真实总批量 embedding 时间写在 batch.meta.json 中。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests
import yaml

from graphrag.config.load_config import load_config
from graphrag.language_model.manager import ModelManager
from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.vector_stores.lancedb import LanceDBVectorStore

from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_text_units,
)

from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.input.retrieval.entities import get_entity_by_id, get_entity_by_key


# ---------- utils ----------

def _now() -> float:
    return time.perf_counter()


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _pick_latest_by_mtime(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return max(paths, key=lambda x: x.stat().st_mtime)


def _read_text_if_exists(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _to_plain_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj

    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            data = model_dump(mode="python")
            if isinstance(data, dict):
                return data
        except TypeError:
            try:
                data = model_dump()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        except Exception:
            pass

    try:
        raw = vars(obj)
        if isinstance(raw, dict):
            data = {k: v for k, v in raw.items() if not k.startswith("_") and not callable(v)}
            if data:
                return data
    except Exception:
        pass

    data: Dict[str, Any] = {}
    skip_keys = {"model_fields", "model_computed_fields"}
    for key in dir(obj):
        if key.startswith("_") or key in skip_keys:
            continue
        try:
            value = getattr(obj, key)
        except Exception:
            continue
        if callable(value):
            continue
        data[key] = value
    return data


def _normalize_native_ollama_base(api_base: Optional[str]) -> str:
    if not api_base:
        return "http://localhost:11434"
    s = str(api_base).strip().rstrip("/")
    parsed = urlparse(s)
    if not parsed.scheme or not parsed.netloc:
        if "://" not in s:
            s = f"http://{s}"
            parsed = urlparse(s)
        else:
            return s
    path = parsed.path or ""
    for suffix in ("/v1", "/api"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    if path in ("", "/"):
        path = ""
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", "")).rstrip("/")


def _parse_think_value(value: str | bool | None) -> bool | str:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off", "none", "disable", "disabled"}:
        return False
    if s in {"low", "medium", "high"}:
        return s
    raise ValueError("--think 仅支持: false/true/low/medium/high")


def _safe_slug(text: str, max_len: int = 80) -> str:
    text = (text or "").strip()
    if not text:
        return "empty"
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "item")[:max_len]


def find_latest_artifacts_dir(root: Path) -> Path:
    output_dir = root / "output"
    if not output_dir.exists():
        raise FileNotFoundError(f"Cannot find output dir: {output_dir}")
    candidates = list(output_dir.rglob("entities.parquet"))
    latest_entities = _pick_latest_by_mtime(candidates)
    if latest_entities is None:
        raise FileNotFoundError(f"Cannot find any entities.parquet under: {output_dir}")
    return latest_entities.parent


def locate_table_parquet(artifacts_dir: Path, name: str) -> Path:
    p = artifacts_dir / f"{name}.parquet"
    if p.exists():
        return p
    p2 = artifacts_dir.parent / f"{name}.parquet"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Cannot find {name}.parquet under {artifacts_dir} (or its parent).")


def infer_lancedb_dir(root: Path) -> Path:
    p = root / "lancedb"
    if p.exists():
        return p
    for cand in root.rglob("*.lance"):
        return cand.parent
    raise FileNotFoundError("Cannot infer LanceDB directory. Please pass --lancedb_dir explicitly.")


def dump_df(df: pd.DataFrame, path: Path, max_rows: int = 2000) -> None:
    _ensure_parent(path)
    if df is None:
        df = pd.DataFrame()
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).copy()
    df.to_csv(path, index=False, encoding="utf-8-sig")


def dump_text(text: str, path: Path) -> None:
    _ensure_parent(path)
    path.write_text(text or "", encoding="utf-8")


def dump_json(obj: Any, path: Path) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_query_for_cache(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def make_embedding_cache_key(model_id: str, normalized_query: str) -> str:
    payload = f"{model_id}\n{normalized_query}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _embedding_to_jsonable(vec: Any) -> List[float]:
    if vec is None:
        raise ValueError("embedding is None")
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    if isinstance(vec, tuple):
        vec = list(vec)
    if not isinstance(vec, list):
        vec = list(vec)
    return [float(x) for x in vec]


def _open_embedding_cache(cache_db: Path) -> sqlite3.Connection:
    _ensure_parent(cache_db)
    conn = sqlite3.connect(str(cache_db))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_cache (
            cache_key TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            normalized_query TEXT NOT NULL,
            original_query TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_embedding_cache_model_query ON embedding_cache(model_id, normalized_query)"
    )
    conn.commit()
    return conn


def load_embeddings_from_cache(conn: sqlite3.Connection, cache_keys: List[str]) -> Dict[str, List[float]]:
    if not cache_keys:
        return {}
    result: Dict[str, List[float]] = {}
    chunk_size = 500
    for start in range(0, len(cache_keys), chunk_size):
        chunk = cache_keys[start : start + chunk_size]
        placeholders = ",".join(["?"] * len(chunk))
        sql = f"SELECT cache_key, embedding_json FROM embedding_cache WHERE cache_key IN ({placeholders})"
        for key, emb_json in conn.execute(sql, chunk):
            try:
                result[str(key)] = [float(x) for x in json.loads(emb_json)]
            except Exception:
                continue
    return result


def save_embeddings_to_cache(
    conn: sqlite3.Connection,
    rows: List[Tuple[str, str, str, str, Any]],
) -> int:
    if not rows:
        return 0
    now_ts = time.time()
    payload = []
    for cache_key, model_id, normalized_query, original_query, embedding in rows:
        payload.append((
            str(cache_key),
            str(model_id),
            str(normalized_query),
            str(original_query),
            json.dumps(_embedding_to_jsonable(embedding), ensure_ascii=False),
            float(now_ts),
        ))
    conn.executemany(
        """
        INSERT OR REPLACE INTO embedding_cache
        (cache_key, model_id, normalized_query, original_query, embedding_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    conn.commit()
    return len(payload)


# ---------- retrieval / context ----------

def resolve_local_search_embedding_model_id(config) -> str:
    model_id = getattr(getattr(config, "local_search", None), "embedding_model_id", None)
    if model_id is None:
        model_id = "default_embedding_model"
    return str(model_id)


def build_embedder_from_config(config, embedding_model_id: Optional[str] = None):
    mm = ModelManager()
    model_id = embedding_model_id
    if model_id is None:
        model_id = getattr(getattr(config, "local_search", None), "embedding_model_id", None)
    if model_id is None:
        model_id = "default_embedding_model"

    emb_cfg = config.get_language_model_config(model_id)
    return mm.get_or_create_embedding_model(
        name="trace_embedding",
        model_type=emb_cfg.type,
        config=emb_cfg,
    )


def open_entity_desc_vectorstore(lancedb_dir: Path, index_name: str) -> LanceDBVectorStore:
    schema = VectorStoreSchemaConfig(index_name=index_name)
    store = LanceDBVectorStore(vector_store_schema_config=schema)
    store.connect(db_uri=str(lancedb_dir))
    if store.document_collection is None:
        raise RuntimeError(
            f"LanceDB table not found: {index_name}. "
            f"Available tables: {store.db_connection.table_names() if store.db_connection else 'UNKNOWN'}"
        )
    return store


def _try_embed_batch(embedder, texts: List[str]) -> Optional[List[Any]]:
    candidates = [
        "embed_batch",
        "embed_texts",
        "encode",
        "encode_queries",
        "get_embeddings",
    ]
    for name in candidates:
        fn = getattr(embedder, name, None)
        if not callable(fn):
            continue
        try:
            out = fn(texts)
            if out is not None:
                return list(out)
        except TypeError:
            try:
                out = fn(input=texts)
                if out is not None:
                    return list(out)
            except Exception:
                continue
        except Exception:
            continue
    return None


def precompute_query_embeddings(
    queries: List[str],
    embedder,
    batch_size: int = 32,
    cache_db: Optional[Path] = None,
    cache_model_id: str = "default_embedding_model",
    use_cache: bool = True,
) -> Tuple[List[Any], Dict[str, Any]]:
    if not queries:
        return [], {
            "mode": "empty",
            "batch_size": batch_size,
            "total_sec": 0.0,
            "count": 0,
            "unique_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_db": str(cache_db) if cache_db else None,
        }

    total = len(queries)
    batch_size = max(int(batch_size), 1)

    norm_queries = [normalize_query_for_cache(q) for q in queries]
    keys = [make_embedding_cache_key(cache_model_id, nq) for nq in norm_queries]

    key_to_query: Dict[str, str] = {}
    unique_keys_in_order: List[str] = []
    for q, nq, key in zip(queries, norm_queries, keys):
        if key not in key_to_query:
            key_to_query[key] = q
            unique_keys_in_order.append(key)

    unique_total = len(unique_keys_in_order)
    cache_t0 = _now()
    cached_map: Dict[str, List[float]] = {}
    conn: Optional[sqlite3.Connection] = None
    if use_cache and cache_db is not None:
        conn = _open_embedding_cache(cache_db)
        cached_map = load_embeddings_from_cache(conn, unique_keys_in_order)
    cache_t1 = _now()

    miss_keys = [k for k in unique_keys_in_order if k not in cached_map]
    hit_count = unique_total - len(miss_keys)
    miss_count = len(miss_keys)

    print(
        f"[CACHE] unique_queries={unique_total}, hits={hit_count}, misses={miss_count}, cache_db={str(cache_db) if cache_db else 'DISABLED'}",
        flush=True,
    )

    miss_embeddings: Dict[str, Any] = {}
    embed_t0 = _now()
    mode = "cache_only" if miss_count == 0 else "loop"
    if miss_count > 0:
        total_batches = (miss_count + batch_size - 1) // batch_size
        print(
            f"[EMBED] Start embedding missing queries {miss_count}/{unique_total} (batch_size={batch_size}, batches={total_batches})",
            flush=True,
        )
        for batch_idx, start in enumerate(range(0, miss_count, batch_size), start=1):
            end = min(start + batch_size, miss_count)
            chunk_keys = miss_keys[start:end]
            chunk_queries = [key_to_query[k] for k in chunk_keys]
            chunk_t0 = _now()
            print(
                f"[EMBED {batch_idx}/{total_batches}] miss {start + 1}-{end}/{miss_count} (resolved_total={hit_count + end}/{unique_total}) ...",
                flush=True,
            )

            chunk_emb = _try_embed_batch(embedder, chunk_queries)
            if chunk_emb is not None and len(chunk_emb) == len(chunk_queries):
                mode = "chunked_batch_with_cache" if use_cache else "chunked_batch"
                for ck, emb in zip(chunk_keys, chunk_emb):
                    miss_embeddings[ck] = _embedding_to_jsonable(emb)
                if conn is not None:
                    rows = [
                        (ck, cache_model_id, normalize_query_for_cache(key_to_query[ck]), key_to_query[ck], miss_embeddings[ck])
                        for ck in chunk_keys
                    ]
                    save_embeddings_to_cache(conn, rows)
                chunk_t1 = _now()
                print(
                    f"[EMBED {batch_idx}/{total_batches}] done miss {end}/{miss_count} (chunk_sec={chunk_t1 - chunk_t0:.3f}, embed_total_sec={chunk_t1 - embed_t0:.3f})",
                    flush=True,
                )
                continue

            if mode == "cache_only":
                mode = "loop_with_cache" if use_cache else "loop"
            print(f"[EMBED {batch_idx}/{total_batches}] batch api unavailable, fallback to per-query embedding", flush=True)
            rows_to_save = []
            for local_idx, (ck, q) in enumerate(zip(chunk_keys, chunk_queries), start=1):
                item_t0 = _now()
                emb = _embedding_to_jsonable(embedder.embed(q))
                miss_embeddings[ck] = emb
                item_t1 = _now()
                if conn is not None:
                    rows_to_save.append((ck, cache_model_id, normalize_query_for_cache(q), q, emb))
                print(
                    f"[EMBED {batch_idx}/{total_batches}] item {start + local_idx}/{miss_count} done (item_sec={item_t1 - item_t0:.3f}, embed_total_sec={item_t1 - embed_t0:.3f})",
                    flush=True,
                )
            if conn is not None and rows_to_save:
                save_embeddings_to_cache(conn, rows_to_save)
            chunk_t1 = _now()
            print(
                f"[EMBED {batch_idx}/{total_batches}] done miss {end}/{miss_count} (chunk_sec={chunk_t1 - chunk_t0:.3f}, embed_total_sec={chunk_t1 - embed_t0:.3f})",
                flush=True,
            )
    embed_t1 = _now()

    if conn is not None:
        cached_map.update(miss_embeddings)
        conn.close()
    else:
        cached_map.update(miss_embeddings)

    embeddings: List[Any] = []
    for key in keys:
        if key not in cached_map:
            raise RuntimeError(f"query embedding cache/build failed for key={key}")
        embeddings.append(cached_map[key])

    total_t1 = _now()
    total_sec = round(total_t1 - cache_t0, 4)
    avg_sec = round((total_t1 - cache_t0) / max(len(queries), 1), 6)
    compute_sec = round(embed_t1 - embed_t0, 4)
    cache_lookup_sec = round(cache_t1 - cache_t0, 4)
    print(
        f"[EMBED] Finished total_queries={len(embeddings)} unique_queries={unique_total} cache_hits={hit_count} cache_misses={miss_count} total_sec={total_t1 - cache_t0:.3f}s compute_sec={embed_t1 - embed_t0:.3f}s mode={mode}",
        flush=True,
    )
    return embeddings, {
        "mode": mode,
        "batch_size": int(batch_size),
        "count": int(len(queries)),
        "unique_queries": int(unique_total),
        "cache_hits": int(hit_count),
        "cache_misses": int(miss_count),
        "cache_db": str(cache_db) if cache_db else None,
        "cache_model_id": str(cache_model_id),
        "cache_lookup_sec": cache_lookup_sec,
        "embed_compute_sec": compute_sec,
        "total_sec": total_sec,
        "avg_sec_est": avg_sec,
    }


def retrieval_from_query_embedding_profiled(
    query_embedding: Any,
    store: LanceDBVectorStore,
    entities_by_id: Dict[str, Any],
    embedding_key: str,
    top_k: int,
    oversample: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    k = max(int(top_k * oversample), 1)

    search_t0 = _now()
    if hasattr(store, "similarity_search_by_vector"):
        results = store.similarity_search_by_vector(query_embedding, k=k)
    else:
        raise RuntimeError(
            "当前 LanceDBVectorStore 不支持 similarity_search_by_vector，"
            "该脚本要求先预计算 query embedding 再做检索。"
        )
    search_t1 = _now()

    map_t0 = _now()
    rows = []
    all_entities_list = list(entities_by_id.values())
    raw_results_count = 0
    mapped_success_count = 0
    for r in results:
        raw_results_count += 1
        doc_id = str(r.document.id)
        score = float(r.score)

        if embedding_key == "id":
            ent = get_entity_by_id(entities_by_id, doc_id)
        else:
            ent = get_entity_by_key(all_entities_list, "title", doc_id)

        if ent is not None:
            mapped_success_count += 1

        rows.append(
            {
                "score": score,
                "vectorstore_id": doc_id,
                "entity_id": getattr(ent, "id", None),
                "entity_title": getattr(ent, "title", None),
                "entity_type": getattr(ent, "type", None),
                "entity_rank": getattr(ent, "rank", None),
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["entity_title"]).head(top_k).reset_index(drop=True)
    map_t1 = _now()

    timings = {
        "vector_search_only": round(search_t1 - search_t0, 4),
        "entity_mapping_only": round(map_t1 - map_t0, 4),
        "retrieval_only_excluding_embed": round(map_t1 - search_t0, 4),
        "retrieval_raw_hits": int(raw_results_count),
        "retrieval_mapped_hits": int(mapped_success_count),
        "retrieval_kept_rows": int(len(df)),
    }
    return df, timings


# ---------- settings / ollama chat ----------

def load_raw_settings_dict(settings_path: Path) -> Dict[str, Any]:
    if not settings_path.exists():
        return {}
    with settings_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def resolve_local_search_prompt_path(root: Path, raw_settings: Dict[str, Any]) -> Optional[Path]:
    local_search_cfg = raw_settings.get("local_search") or {}
    prompt_value = local_search_cfg.get("prompt")
    if not prompt_value:
        return None
    p = Path(str(prompt_value))
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def resolve_local_search_chat_model_id(config, raw_settings: Dict[str, Any]) -> Optional[str]:
    local_cfg_obj = getattr(config, "local_search", None)
    for attr in ("chat_model_id", "completion_model_id"):
        value = getattr(local_cfg_obj, attr, None)
        if value:
            return str(value)

    local_cfg = raw_settings.get("local_search") or {}
    for key in ("chat_model_id", "completion_model_id"):
        value = local_cfg.get(key)
        if value:
            return str(value)

    for candidate in ("default_chat_model", "default_completion_model"):
        if candidate in (raw_settings.get("completion_models") or {}):
            return candidate
        if candidate in (raw_settings.get("models") or {}):
            return candidate
    return None


def resolve_chat_model_config(config, raw_settings: Dict[str, Any]) -> Dict[str, Any]:
    model_id = resolve_local_search_chat_model_id(config, raw_settings)
    result: Dict[str, Any] = {
        "model_id": model_id,
        "model": None,
        "api_base": None,
        "request_timeout": 600,
        "raw": {},
    }

    if model_id:
        try:
            cfg_obj = config.get_language_model_config(model_id)
            cfg = _to_plain_dict(cfg_obj)
            result["raw"] = cfg
            result["model"] = cfg.get("model") or cfg.get("deployment_name") or cfg.get("name")
            result["api_base"] = cfg.get("api_base") or cfg.get("base_url") or cfg.get("url") or cfg.get("host")
            result["request_timeout"] = int(cfg.get("request_timeout") or cfg.get("timeout") or 600)
        except Exception:
            pass

    if model_id:
        for block_name in ("completion_models", "models"):
            block = raw_settings.get(block_name) or {}
            if model_id in block and isinstance(block[model_id], dict):
                raw = block[model_id]
                if not result["raw"]:
                    result["raw"] = raw
                result["model"] = result["model"] or raw.get("model") or raw.get("deployment_name") or raw.get("name")
                result["api_base"] = result["api_base"] or raw.get("api_base") or raw.get("base_url") or raw.get("url") or raw.get("host")
                result["request_timeout"] = int(raw.get("request_timeout") or raw.get("timeout") or result["request_timeout"] or 600)
                break

    if not result["model"]:
        llm_cfg = raw_settings.get("llm") or {}
        if isinstance(llm_cfg, dict):
            result["raw"] = result["raw"] or llm_cfg
            result["model"] = llm_cfg.get("model") or llm_cfg.get("deployment_name") or llm_cfg.get("name")
            result["api_base"] = llm_cfg.get("api_base") or llm_cfg.get("base_url") or llm_cfg.get("url") or llm_cfg.get("host")
            result["request_timeout"] = int(llm_cfg.get("request_timeout") or llm_cfg.get("timeout") or result["request_timeout"] or 600)

    if not result["model"]:
        raise RuntimeError(
            "无法从 settings.yaml / GraphRAG config 中解析 local_search 对应的 chat model。"
            "请检查 local_search.chat_model_id 或 local_search.completion_model_id 是否正确。"
        )

    result["native_ollama_base"] = _normalize_native_ollama_base(result.get("api_base"))
    return result


def build_qa_messages(query: str, context_text: str, force_chinese: bool = True, extra_system: Optional[str] = None) -> List[Dict[str, str]]:
    system_parts = [
        "你是一个基于给定上下文回答问题的问答助手。",
        "只能依据提供的上下文回答，不要编造上下文中没有的信息。",
        "如果上下文不足以支持结论，请明确回答“根据当前上下文无法确定”或“上下文未提供”。",
        "回答尽量直接、简洁、准确。",
    ]
    if force_chinese:
        system_parts.append("请使用中文作答。")
    if extra_system:
        system_parts.append(extra_system.strip())

    system_prompt = "\n".join(system_parts)
    user_prompt = (
        "下面给出用于回答问题的上下文。\n\n"
        "[上下文开始]\n"
        f"{context_text or ''}\n"
        "[上下文结束]\n\n"
        f"问题：{query}\n\n"
        "请基于上述上下文直接回答。"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_ollama_native_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    think: bool | str = False,
    stream: bool = False,
    timeout: int = 600,
    keep_alive: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "think": think,
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    if options:
        payload["options"] = options

    url = f"{base_url.rstrip('/')}/api/chat"
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"Ollama /api/chat 请求失败: HTTP {resp.status_code}\nURL: {url}\n响应: {resp.text}") from e
    try:
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"Ollama 返回不是合法 JSON:\n{resp.text[:2000]}") from e


def extract_answer_and_thinking(resp_json: Dict[str, Any]) -> Tuple[str, str]:
    msg = resp_json.get("message") or {}
    return str(msg.get("content") or ""), str(msg.get("thinking") or "")


# ---------- batch input ----------

def load_queries(queries_path: Path, query_col: str = "query", id_col: Optional[str] = None) -> List[Dict[str, Any]]:
    suffix = queries_path.suffix.lower()
    items: List[Dict[str, Any]] = []

    if suffix == ".txt":
        lines = queries_path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, start=1):
            q = line.strip()
            if q:
                items.append({"id": f"q{i:04d}", "query": q})
        return items

    if suffix == ".csv":
        df = pd.read_csv(queries_path)
        if query_col not in df.columns:
            raise ValueError(f"CSV 中找不到问题列: {query_col}. 可用列: {list(df.columns)}")
        for i, row in df.iterrows():
            q = str(row[query_col]).strip()
            if not q or q.lower() == "nan":
                continue
            item_id = str(row[id_col]).strip() if (id_col and id_col in df.columns and pd.notna(row[id_col])) else f"q{i+1:04d}"
            items.append({"id": item_id, "query": q, "row_index": int(i)})
        return items

    if suffix == ".jsonl":
        with queries_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if query_col not in obj:
                    raise ValueError(f"JSONL 第 {i} 行缺少字段: {query_col}")
                q = str(obj[query_col]).strip()
                if not q:
                    continue
                item_id = str(obj.get(id_col)).strip() if (id_col and obj.get(id_col) is not None) else f"q{i:04d}"
                items.append({"id": item_id, "query": q, **obj})
        return items

    raise ValueError("--queries_file 仅支持 .txt / .csv / .jsonl")


# ---------- phase 1: retrieval / context ----------

def run_one_query_context_only(
    query: str,
    query_embedding: Any,
    out_prefix: Path,
    *,
    root: Path,
    settings_path: Path,
    artifacts_dir: Path,
    lancedb_dir: Path,
    entities: List[Any],
    relationships: List[Any],
    text_units: List[Any],
    entities_by_id: Dict[str, Any],
    store: LanceDBVectorStore,
    embedder,
    entity_desc_collection: str,
    embedding_key: str,
    max_dump_rows: int,
    max_context_tokens: int,
    community_prop: float,
    text_unit_prop: float,
    top_k_mapped_entities: int,
    top_k_relationships: int,
    include_entity_rank: bool,
    include_relationship_weight: bool,
    relationship_ranking_attribute: str,
    use_community_summary: bool,
    return_candidate_context: bool,
    batch_embed_avg_sec_est: float,
    batch_embed_mode: str,
    use_pcst: bool = False,
    pcst_top_k_nodes: int = 10,
    pcst_top_k_edges: int = 10,
    pcst_cost_per_edge: float = 0.5,
) -> Dict[str, Any]:
    t0 = _now()

    mapped_df, retrieval_profile = retrieval_from_query_embedding_profiled(
        query_embedding=query_embedding,
        store=store,
        entities_by_id=entities_by_id,
        embedding_key=embedding_key,
        top_k=top_k_mapped_entities,
        oversample=2,
    )

    embedding_key_enum = EntityVectorStoreKey.ID if embedding_key == "id" else EntityVectorStoreKey.TITLE
    ctx_builder = LocalSearchMixedContext(
        entities=entities,
        relationships=relationships,
        text_units=text_units,
        community_reports=[],
        covariates={},
        entity_text_embeddings=store,
        text_embedder=embedder,
        embedding_vectorstore_key=embedding_key_enum,
    )

    context_t0 = _now()
    context_result = ctx_builder.build_context(
        query=query,
        max_context_tokens=int(max_context_tokens),
        community_prop=float(community_prop),
        text_unit_prop=float(text_unit_prop),
        top_k_mapped_entities=int(top_k_mapped_entities),
        top_k_relationships=int(top_k_relationships),
        include_entity_rank=bool(include_entity_rank),
        include_relationship_weight=bool(include_relationship_weight),
        relationship_ranking_attribute=str(relationship_ranking_attribute),
        use_community_summary=bool(use_community_summary),
        return_candidate_context=bool(return_candidate_context),
        use_pcst=bool(use_pcst),
        pcst_top_k_nodes=int(pcst_top_k_nodes),
        pcst_top_k_edges=int(pcst_top_k_edges),
        pcst_cost_per_edge=float(pcst_cost_per_edge),
    )
    context_t1 = _now()

    dump_t0 = _now()
    dump_df(mapped_df, Path(str(out_prefix) + ".retrieval.mapped_entities.csv"), max_dump_rows)
    dump_text(context_result.context_chunks, Path(str(out_prefix) + ".context.txt"))
    if isinstance(context_result.context_records, dict):
        for k, df in context_result.context_records.items():
            dump_df(df, Path(str(out_prefix) + f".context.{k}.csv"), max_dump_rows)
    dump_t1 = _now()

    meta = {
        "root": str(root),
        "settings_path": str(settings_path),
        "artifacts_dir": str(artifacts_dir),
        "lancedb_dir": str(lancedb_dir),
        "entity_desc_collection": entity_desc_collection,
        "embedding_key": embedding_key,
        "query": query,
        "only_context": False,
        "phase": "context_ready",
        "timing_sec": {
            "query_embed_only": round(float(batch_embed_avg_sec_est), 6),
            "query_embed_only_is_estimated": True,
            "vector_search_only": retrieval_profile["vector_search_only"],
            "entity_mapping_only": retrieval_profile["entity_mapping_only"],
            "retrieval_only": round(float(batch_embed_avg_sec_est) + retrieval_profile["retrieval_only_excluding_embed"], 4),
            "retrieval_only_excluding_embed": retrieval_profile["retrieval_only_excluding_embed"],
            "context_only": round(context_t1 - context_t0, 4),
            "retrieval_and_context": round(float(batch_embed_avg_sec_est) + retrieval_profile["retrieval_only_excluding_embed"] + (context_t1 - context_t0), 4),
            "dump_outputs": round(dump_t1 - dump_t0, 4),
            "chat_answer": None,
            "total_phase1_local": round(dump_t1 - t0, 4),
        },
        "embedding_batch": {
            "mode": batch_embed_mode,
            "avg_sec_est": round(float(batch_embed_avg_sec_est), 6),
        },
        "counts": {
            "entities": len(entities),
            "relationships": len(relationships),
            "text_units": len(text_units),
            "mapped_entities_rows": int(len(mapped_df)),
            "retrieval_raw_hits": int(retrieval_profile.get("retrieval_raw_hits", 0)),
            "retrieval_mapped_hits": int(retrieval_profile.get("retrieval_mapped_hits", 0)),
            "context_chars": len(context_result.context_chunks or ""),
            "answer_chars": 0,
            "thinking_chars": 0,
        },
        "chat": {
            "model_id": None,
            "model": None,
            "api_base_from_settings": None,
            "native_ollama_base": None,
            "think": None,
            "keep_alive": None,
            "chat_timeout": None,
            "options": {},
        },
    }
    dump_json(meta, Path(str(out_prefix) + ".meta.json"))

    return {
        "context_text": context_result.context_chunks,
        "meta": meta,
    }


# ---------- phase 2: chat ----------

def run_one_query_chat_only(
    query: str,
    context_text: str,
    out_prefix: Path,
    *,
    config,
    raw_settings: Dict[str, Any],
    root: Path,
    think_arg: str,
    keep_alive: Optional[str],
    chat_timeout_override: Optional[int],
    temperature: Optional[float],
    num_ctx: Optional[int],
    top_p: Optional[float],
    repeat_penalty: Optional[float],
    force_chinese: bool,
) -> Dict[str, Any]:
    prompt_path = resolve_local_search_prompt_path(root, raw_settings)
    prompt_text = _read_text_if_exists(prompt_path)
    extra_system = None
    if prompt_text:
        extra_system = (
            "以下为项目中的 local_search 提示词文件内容，仅作为风格参考，不要求逐字遵循：\n"
            + prompt_text[:4000]
        )

    chat_cfg = resolve_chat_model_config(config, raw_settings)
    think_value = _parse_think_value(think_arg)

    messages = build_qa_messages(
        query=query,
        context_text=context_text,
        force_chinese=bool(force_chinese),
        extra_system=extra_system,
    )

    options: Dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if top_p is not None:
        options["top_p"] = top_p
    if repeat_penalty is not None:
        options["repeat_penalty"] = repeat_penalty
    if not options:
        options = None

    chat_timeout = int(chat_timeout_override or chat_cfg.get("request_timeout") or 600)
    t0 = _now()
    raw_answer_resp = call_ollama_native_chat(
        base_url=chat_cfg["native_ollama_base"],
        model=str(chat_cfg["model"]),
        messages=messages,
        think=think_value,
        stream=False,
        timeout=chat_timeout,
        keep_alive=keep_alive,
        options=options,
    )
    answer, thinking = extract_answer_and_thinking(raw_answer_resp)
    t1 = _now()

    dump_text(answer, Path(str(out_prefix) + ".answer.txt"))
    if thinking:
        dump_text(thinking, Path(str(out_prefix) + ".thinking.txt"))
    dump_json(raw_answer_resp, Path(str(out_prefix) + ".answer.raw.json"))

    meta_path = Path(str(out_prefix) + ".meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    timing_sec = dict(meta.get("timing_sec") or {})
    timing_sec["chat_answer"] = round(t1 - t0, 4)
    phase1_local = timing_sec.get("total_phase1_local")
    if isinstance(phase1_local, (int, float)):
        timing_sec["total"] = round(float(phase1_local) + (t1 - t0), 4)

    counts = dict(meta.get("counts") or {})
    counts["answer_chars"] = len(answer or "")
    counts["thinking_chars"] = len(thinking or "")

    meta["phase"] = "done"
    meta["timing_sec"] = timing_sec
    meta["counts"] = counts
    meta["chat"] = {
        "model_id": chat_cfg.get("model_id"),
        "model": chat_cfg.get("model"),
        "api_base_from_settings": chat_cfg.get("api_base"),
        "native_ollama_base": chat_cfg.get("native_ollama_base"),
        "think": think_value,
        "keep_alive": keep_alive,
        "chat_timeout": chat_timeout,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
        },
    }
    dump_json(meta, meta_path)

    return {
        "answer": answer,
        "thinking": thinking,
        "meta": meta,
        "chat_cfg": chat_cfg,
        "raw_answer_resp": raw_answer_resp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="GraphRAG project root (contains settings.yaml and output/)")
    ap.add_argument("--settings", default=None, help="Optional: explicit settings.yaml path")
    ap.add_argument("--data_dir", default=None, help="Optional: explicit artifacts dir containing *.parquet")
    ap.add_argument("--lancedb_dir", default=None, help="Optional: explicit LanceDB dir (contains *.lance folders)")

    ap.add_argument("--queries_file", required=True, help="批量问题文件: .txt / .csv / .jsonl")
    ap.add_argument("--query_col", default="query", help="CSV/JSONL 问题列名，默认 query")
    ap.add_argument("--id_col", default=None, help="CSV/JSONL ID 列名，可选")
    ap.add_argument("--out_dir", required=True, help=r"输出目录，例如 D:\trace_batch")
    ap.add_argument("--item_prefix_name", default="result", help="单题输出前缀名，默认 result")
    ap.add_argument("--max_questions", type=int, default=None, help="只跑前 N 个问题，便于调试")
    ap.add_argument("--skip_existing", action="store_true", help="若单题 meta.json 已存在则跳过")
    ap.add_argument("--max_dump_rows", type=int, default=2000)
    ap.add_argument("--embed_batch_size", type=int, default=32, help="预计算 query embedding 的分块大小")
    ap.add_argument("--embedding_cache_db", default=None, help="query embedding 磁盘缓存 SQLite 文件；默认 <root>/.cache/query_embedding_cache.sqlite")
    ap.add_argument("--disable_embedding_cache", action="store_true", help="关闭 query embedding 磁盘缓存")

    ap.add_argument("--entity_desc_collection", default="default-entity-description")
    ap.add_argument("--embedding_key", choices=["id", "title"], default="id")

    ap.add_argument("--max_context_tokens", type=int, default=6000)
    ap.add_argument("--community_prop", type=float, default=0.0)
    ap.add_argument("--text_unit_prop", type=float, default=0.7)
    ap.add_argument("--top_k_mapped_entities", type=int, default=10)
    ap.add_argument("--top_k_relationships", type=int, default=10)
    ap.add_argument("--include_entity_rank", action="store_true")
    ap.add_argument("--include_relationship_weight", action="store_true")
    ap.add_argument("--relationship_ranking_attribute", default="rank")
    ap.add_argument("--use_community_summary", action="store_true")
    ap.add_argument("--return_candidate_context", action="store_true")

    # PCST 子图检索参数（默认读取 settings.yaml 中的配置）
    ap.add_argument("--use_pcst", default=None, help="是否启用PCST子图检索，默认读settings.yaml")
    ap.add_argument("--pcst_top_k_nodes", type=int, default=None)
    ap.add_argument("--pcst_top_k_edges", type=int, default=None)
    ap.add_argument("--pcst_cost_per_edge", type=float, default=None)

    ap.add_argument("--only_context", action="store_true")
    ap.add_argument("--think", default="false")
    ap.add_argument("--keep_alive", default=None)
    ap.add_argument("--chat_timeout", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--num_ctx", type=int, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--repeat_penalty", type=float, default=None)
    ap.add_argument("--force_chinese", action="store_true", default=True)
    ap.add_argument("--no_force_chinese", dest="force_chinese", action="store_false")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    settings_path = Path(args.settings).resolve() if args.settings else (root / "settings.yaml")
    artifacts_dir = Path(args.data_dir).resolve() if args.data_dir else find_latest_artifacts_dir(root)
    lancedb_dir = Path(args.lancedb_dir).resolve() if args.lancedb_dir else infer_lancedb_dir(root)
    queries_path = Path(args.queries_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_t0 = _now()
    t0 = _now()
    config = load_config(root_dir=root, config_filepath=settings_path if settings_path.exists() else None)
    raw_settings = load_raw_settings_dict(settings_path)
    t1 = _now()

    entities_pq = locate_table_parquet(artifacts_dir, "entities")
    relationships_pq = locate_table_parquet(artifacts_dir, "relationships")
    text_units_pq = locate_table_parquet(artifacts_dir, "text_units")
    try:
        communities_pq = locate_table_parquet(artifacts_dir, "communities")
    except FileNotFoundError:
        communities_pq = None

    t2 = _now()
    entities_df = pd.read_parquet(entities_pq)
    relationships_df = pd.read_parquet(relationships_pq)
    text_units_df = pd.read_parquet(text_units_pq)
    if communities_pq and Path(communities_pq).exists():
        communities_df = pd.read_parquet(communities_pq)
    else:
        communities_df = pd.DataFrame(columns=["community", "level", "entity_ids"])
    t3 = _now()

    entities = read_indexer_entities(entities=entities_df, communities=communities_df, community_level=None)
    relationships = read_indexer_relationships(relationships_df)
    text_units = read_indexer_text_units(text_units_df)
    entities_by_id = {e.id: e for e in entities}
    t4 = _now()

    embedding_model_id = resolve_local_search_embedding_model_id(config)
    if args.embedding_cache_db:
        embedding_cache_db = Path(args.embedding_cache_db).resolve()
    else:
        embedding_cache_db = (root / ".cache" / "query_embedding_cache.sqlite").resolve()
    embedding_cache_db.parent.mkdir(parents=True, exist_ok=True)

    embedder = build_embedder_from_config(config, embedding_model_id=embedding_model_id)
    store = open_entity_desc_vectorstore(lancedb_dir=lancedb_dir, index_name=args.entity_desc_collection)
    t5 = _now()

    items = load_queries(queries_path, query_col=args.query_col, id_col=args.id_col)
    if args.max_questions is not None:
        items = items[: int(args.max_questions)]
    if not items:
        raise RuntimeError("问题列表为空。")

    # PCST 配置：CLI 优先，否则读 settings.yaml
    local_search_cfg = raw_settings.get("local_search") or {}
    use_pcst = local_search_cfg.get("use_pcst", False)
    if args.use_pcst is not None:
        use_pcst = str(args.use_pcst).lower() in ("1", "true", "yes")
    pcst_top_k_nodes = args.pcst_top_k_nodes or local_search_cfg.get("pcst_top_k_nodes", 10)
    pcst_top_k_edges = args.pcst_top_k_edges or local_search_cfg.get("pcst_top_k_edges", 10)
    pcst_cost_per_edge = args.pcst_cost_per_edge if args.pcst_cost_per_edge is not None else local_search_cfg.get("pcst_cost_per_edge", 0.5)

    print(f"[INFO] Loaded {len(items)} questions from: {queries_path}")
    print(f"[INFO] Artifacts: {artifacts_dir}")
    print(f"[INFO] LanceDB:   {lancedb_dir}")
    print(f"[INFO] PCST:      use_pcst={use_pcst}, top_k_nodes={pcst_top_k_nodes}, top_k_edges={pcst_top_k_edges}, cost={pcst_cost_per_edge}")

    # 保留待处理项，并提前处理 skip_existing
    active_items: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    success_count = 0
    error_count = 0

    for idx, item in enumerate(items, start=1):
        item_id = str(item.get("id") or f"q{idx:04d}")
        query = str(item["query"]).strip()
        slug = _safe_slug(item_id)
        item_dir = out_dir / slug
        item_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = item_dir / str(args.item_prefix_name)
        meta_path = Path(str(out_prefix) + ".meta.json")
        enriched = {**item, "item_id": item_id, "query": query, "item_dir": item_dir, "out_prefix": out_prefix, "meta_path": meta_path}

        if args.skip_existing and meta_path.exists():
            print(f"[{idx}/{len(items)}] SKIP {item_id}")
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                answer_path = Path(str(out_prefix) + ".answer.txt")
                answer_text = answer_path.read_text(encoding="utf-8") if answer_path.exists() else ""
                summary_rows.append({
                    "item_id": item_id,
                    "query": query,
                    "status": "skipped_existing",
                    "answer": answer_text,
                    "error": "",
                    "out_prefix": str(out_prefix),
                    **{f"timing_{k}": v for k, v in (meta.get("timing_sec") or {}).items()},
                    **{f"count_{k}": v for k, v in (meta.get("counts") or {}).items()},
                })
                success_count += 1
            except Exception as e:
                summary_rows.append({
                    "item_id": item_id,
                    "query": query,
                    "status": "skip_read_failed",
                    "answer": "",
                    "error": repr(e),
                    "out_prefix": str(out_prefix),
                })
                error_count += 1
            continue

        active_items.append(enriched)

    # Phase 1: pre-embed all remaining queries
    phase1_t0 = _now()
    query_texts = [str(it["query"]) for it in active_items]
    query_embeddings, batch_embed_profile = precompute_query_embeddings(
        queries=query_texts,
        embedder=embedder,
        batch_size=args.embed_batch_size,
        cache_db=None if args.disable_embedding_cache else embedding_cache_db,
        cache_model_id=embedding_model_id,
        use_cache=not args.disable_embedding_cache,
    ) if active_items else ([], {"mode": "empty", "batch_size": args.embed_batch_size, "count": 0, "unique_queries": 0, "cache_hits": 0, "cache_misses": 0, "cache_db": str(embedding_cache_db), "cache_model_id": embedding_model_id, "total_sec": 0.0, "avg_sec_est": 0.0})
    phase1_embed_t1 = _now()

    prepared_for_chat: List[Dict[str, Any]] = []

    for idx, (item, query_embedding) in enumerate(zip(active_items, query_embeddings), start=1):
        item_id = item["item_id"]
        query = item["query"]
        out_prefix: Path = item["out_prefix"]
        print(f"[PHASE1 {idx}/{len(active_items)}] CONTEXT {item_id}: {query[:80]}")
        try:
            result = run_one_query_context_only(
                query=query,
                query_embedding=query_embedding,
                out_prefix=out_prefix,
                root=root,
                settings_path=settings_path,
                artifacts_dir=artifacts_dir,
                lancedb_dir=lancedb_dir,
                entities=entities,
                relationships=relationships,
                text_units=text_units,
                entities_by_id=entities_by_id,
                store=store,
                embedder=embedder,
                entity_desc_collection=args.entity_desc_collection,
                embedding_key=args.embedding_key,
                max_dump_rows=args.max_dump_rows,
                max_context_tokens=args.max_context_tokens,
                community_prop=args.community_prop,
                text_unit_prop=args.text_unit_prop,
                top_k_mapped_entities=args.top_k_mapped_entities,
                top_k_relationships=args.top_k_relationships,
                include_entity_rank=args.include_entity_rank,
                include_relationship_weight=args.include_relationship_weight,
                relationship_ranking_attribute=args.relationship_ranking_attribute,
                use_community_summary=args.use_community_summary,
                return_candidate_context=args.return_candidate_context,
                batch_embed_avg_sec_est=float(batch_embed_profile.get("avg_sec_est", 0.0)),
                batch_embed_mode=str(batch_embed_profile.get("mode", "unknown")),
                use_pcst=use_pcst,
                pcst_top_k_nodes=pcst_top_k_nodes,
                pcst_top_k_edges=pcst_top_k_edges,
                pcst_cost_per_edge=pcst_cost_per_edge,
            )
            prepared_for_chat.append({
                "item_id": item_id,
                "query": query,
                "ground_truth": str(item.get("output", "")),
                "out_prefix": out_prefix,
                "context_text": result["context_text"],
                "meta": result["meta"],
            })
            if args.only_context:
                summary_rows.append({
                    "item_id": item_id,
                    "query": query,
                    "status": "context_only_ok",
                    "answer": "",
                    "error": "",
                    "out_prefix": str(out_prefix),
                    **{f"timing_{k}": v for k, v in (result["meta"].get("timing_sec") or {}).items()},
                    **{f"count_{k}": v for k, v in (result["meta"].get("counts") or {}).items()},
                })
                success_count += 1
        except Exception as e:
            err_text = repr(e)
            dump_text(err_text, Path(str(out_prefix) + ".error.txt"))
            dump_json({"item_id": item_id, "query": query, "status": "context_error", "error": err_text}, Path(str(out_prefix) + ".meta.json"))
            summary_rows.append({
                "item_id": item_id,
                "query": query,
                "status": "context_error",
                "answer": "",
                "error": err_text,
                "out_prefix": str(out_prefix),
            })
            error_count += 1
            print(f"[PHASE1 {idx}/{len(active_items)}] ERR {item_id}: {err_text}")

    phase1_t1 = _now()

    # Phase 2: chat after all contexts are ready
    phase2_t0 = _now()
    if not args.only_context:
        for idx, item in enumerate(prepared_for_chat, start=1):
            item_id = item["item_id"]
            query = item["query"]
            out_prefix: Path = item["out_prefix"]
            print(f"[PHASE2 {idx}/{len(prepared_for_chat)}] CHAT {item_id}: {query[:80]}")
            try:
                chat_result = run_one_query_chat_only(
                    query=query,
                    context_text=item["context_text"],
                    out_prefix=out_prefix,
                    config=config,
                    raw_settings=raw_settings,
                    root=root,
                    think_arg=args.think,
                    keep_alive=args.keep_alive,
                    chat_timeout_override=args.chat_timeout,
                    temperature=args.temperature,
                    num_ctx=args.num_ctx,
                    top_p=args.top_p,
                    repeat_penalty=args.repeat_penalty,
                    force_chinese=args.force_chinese,
                )
                meta = chat_result["meta"]
                summary_rows.append({
                    "item_id": item_id,
                    "query": query,
                    "ground_truth": str(item.get("output", "")),
                    "status": "ok",
                    "answer": chat_result.get("answer", ""),
                    "error": "",
                    "out_prefix": str(out_prefix),
                    **{f"timing_{k}": v for k, v in (meta.get("timing_sec") or {}).items()},
                    **{f"count_{k}": v for k, v in (meta.get("counts") or {}).items()},
                })
                success_count += 1
            except Exception as e:
                err_text = repr(e)
                dump_text(err_text, Path(str(out_prefix) + ".error.txt"))
                meta_path = Path(str(out_prefix) + ".meta.json")
                meta = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        meta = {}
                meta.update({"item_id": item_id, "query": query, "status": "chat_error", "error": err_text})
                dump_json(meta, meta_path)
                summary_rows.append({
                    "item_id": item_id,
                    "query": query,
                    "status": "chat_error",
                    "answer": "",
                    "error": err_text,
                    "out_prefix": str(out_prefix),
                })
                error_count += 1
                print(f"[PHASE2 {idx}/{len(prepared_for_chat)}] ERR {item_id}: {err_text}")
    phase2_t1 = _now()

    batch_t1 = _now()

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "batch.summary.csv"
    summary_jsonl = out_dir / "batch.summary.jsonl"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    append_jsonl(summary_rows, summary_jsonl)

    batch_meta = {
        "root": str(root),
        "settings_path": str(settings_path),
        "artifacts_dir": str(artifacts_dir),
        "lancedb_dir": str(lancedb_dir),
        "queries_file": str(queries_path),
        "out_dir": str(out_dir),
        "embedding_cache": {
            "enabled": not bool(args.disable_embedding_cache),
            "cache_db": None if args.disable_embedding_cache else str(embedding_cache_db),
            "embedding_model_id": embedding_model_id,
        },
        "items_total": len(items),
        "items_active": len(active_items),
        "items_prepared_for_chat": len(prepared_for_chat),
        "success_count": success_count,
        "error_count": error_count,
        "embedding_batch": batch_embed_profile,
        "timing_sec": {
            "load_config": round(t1 - t0, 4),
            "read_parquet": round(t3 - t2, 4),
            "adapt_objects": round(t4 - t3, 4),
            "embedder_and_store": round(t5 - t4, 4),
            "precompute_query_embeddings": round(phase1_embed_t1 - phase1_t0, 4),
            "phase1_total": round(phase1_t1 - phase1_t0, 4),
            "phase2_chat_total": round(phase2_t1 - phase2_t0, 4),
            "total": round(batch_t1 - batch_t0, 4),
        },
        "args": vars(args),
    }
    dump_json(batch_meta, out_dir / "batch.meta.json")

    print("\n[OK] Wrote:")
    print(f"  {summary_csv}")
    print(f"  {summary_jsonl}")
    print(f"  {out_dir / 'batch.meta.json'}")
    print(f"\n[DONE] success={success_count}, error={error_count}, total={len(items)}")


if __name__ == "__main__":
    main()
