"""测试 PCST 子图检索模块（不调用 LLM，仅验证子图构建逻辑）"""
import os, sys
os.chdir(r"d:\SEU\SRTP项目\graphrag\graph_database")

import pandas as pd
import numpy as np
from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.query.context_builder.pcst_subgraph import retrieve_pcst_subgraph

# 加载数据
entities_df = pd.read_parquet("output/entities.parquet")
relationships_df = pd.read_parquet("output/relationships.parquet")

print(f"总实体数: {len(entities_df)}")
print(f"总关系数: {len(relationships_df)}")

# 构建 Entity 对象列表
all_entities = []
for _, row in entities_df.iterrows():
    emb = row.get("description_embedding")
    if emb is not None and hasattr(emb, '__len__') and len(emb) > 0:
        emb = list(emb)
    else:
        emb = None
    all_entities.append(Entity(
        id=str(row["id"]),
        short_id=str(row.get("human_readable_id", row["id"])),
        title=str(row["title"]),
        type=row.get("type"),
        description=row.get("description"),
        description_embedding=emb,
        community_ids=row.get("community"),
        text_unit_ids=row.get("text_unit_ids"),
        rank=row.get("degree", 1),
    ))

# 构建 Relationship 对象列表
all_relationships = []
for _, row in relationships_df.iterrows():
    emb = row.get("description_embedding")
    if emb is not None and hasattr(emb, '__len__') and len(emb) > 0:
        emb = list(emb)
    else:
        emb = None
    all_relationships.append(Relationship(
        id=str(row["id"]),
        short_id=str(row.get("human_readable_id", row["id"])),
        source=str(row["source"]),
        target=str(row["target"]),
        weight=row.get("weight", 1.0),
        description=row.get("description"),
        description_embedding=emb,
        text_unit_ids=row.get("text_unit_ids"),
        rank=row.get("rank", 1),
    ))

print(f"有 embedding 的实体数: {sum(1 for e in all_entities if e.description_embedding)}")
print(f"有 embedding 的关系数: {sum(1 for r in all_relationships if r.description_embedding)}")

# 用 embedding 模型生成查询向量
from graphrag_llm.embedding import create_embedding
from graphrag.config.load_config import load_config

config = load_config(root_dir=r"d:\SEU\SRTP项目\graphrag\graph_database")
emb_settings = config.get_embedding_model_config(config.local_search.embedding_model_id)
text_embedder = create_embedding(emb_settings)

query = "MIG/MAG焊接作业有哪些安全注意事项？"
print(f"\n查询: {query}")

# 模拟已由向量检索选出的 top-k entities
# (实际中由 map_query_to_entities 向量检索返回，这里按 degree 取 top-5 模拟)
sorted_entities = sorted(all_entities, key=lambda e: e.rank or 0, reverse=True)
selected = sorted_entities[:5]
print(f"初始选中实体 (top-5): {[e.title for e in selected]}")

# 执行 PCST 子图检索
print("\n执行 PCST 子图检索...")
pcst_entities, pcst_relationships = retrieve_pcst_subgraph(
    selected_entities=selected,
    all_entities=all_entities,
    all_relationships=all_relationships,
    query=query,
    text_embedder=text_embedder,
    top_k_nodes=10,
    top_k_edges=10,
    cost_per_edge=0.5,
)

print(f"\nPCST 结果:")
print(f"  选中实体数: {len(pcst_entities)}")
print(f"  选中关系数: {len(pcst_relationships)}")
print(f"\n  实体列表:")
for e in pcst_entities:
    print(f"    - {e.title} (type={e.type})")
print(f"\n  关系列表:")
for r in pcst_relationships:
    print(f"    - {r.source} --[{r.description[:30] if r.description else '?'}]--> {r.target}")

print("\nPCST 子图检索测试完成！")
